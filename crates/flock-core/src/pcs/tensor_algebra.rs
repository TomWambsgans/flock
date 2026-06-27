// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Ported from binius64's `crates/math/src/tensor_algebra.rs`
// (https://github.com/binius-zk/binius64), specialized to `F = F_2`,
// `FE = F_{2^256}`.

//! Tensor algebra over `F_{2^256} ⊗_{F_2} F_{2^256}`.
//!
//! An element is a length-256 vector of `F256` (the "vertical-subring" elements
//! in DP24 nomenclature). Conceptually it's a 256×256 F_2 matrix, where row `i`
//! is `elems[i]` viewed via its bit-decomposition in the natural F256 basis
//! `{x^a·u^b : a∈[0,128), b∈[0,2)}` (`bit_j(elems[i])` = coefficient of
//! `δ_i ⊗ δ_j`, with `δ_k` the k-th F256 basis element).
//!
//! Bit layout of an `F256` element (matching `pack`'s convention and the
//! `(c0, c1)` memory order): bit `k` for `k∈[0,256)` is
//! - `k∈[0,64)`    → `c0.lo` bit `k`
//! - `k∈[64,128)`  → `c0.hi` bit `k−64`
//! - `k∈[128,192)` → `c1.lo` bit `k−128`
//! - `k∈[192,256)` → `c1.hi` bit `k−192`
//!
//! Used by the verifier's polylog `eval_rs_eq` (DP24 §1.3, Figure 3).

use crate::field::{F128, F256};
use core::ops::{Add, AddAssign};

/// The degree of `F_{2^256}` over `F_2`.
pub const DEGREE: usize = 256;

/// An element of `F_{2^256} ⊗_{F_2} F_{2^256}`, stored as 256 `F256` elements
/// (the vertical-subring decomposition).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorAlgebra {
    /// Length-256 vector. `elems[i]` is the coefficient of `δ_i` in the
    /// vertical basis decomposition.
    pub elems: Vec<F256>,
}

impl TensorAlgebra {
    /// All-zero element.
    pub fn zero() -> Self {
        Self {
            elems: vec![F256::ZERO; DEGREE],
        }
    }

    /// Multiplicative identity: `1 ⊗ 1`.
    pub fn one() -> Self {
        let mut elems = vec![F256::ZERO; DEGREE];
        elems[0] = F256::ONE;
        Self { elems }
    }

    /// Embed `x ∈ F_{2^256}` into the vertical subring: returns `1 ⊗ x`.
    pub fn from_vertical(x: F256) -> Self {
        let mut elems = vec![F256::ZERO; DEGREE];
        elems[0] = x;
        Self { elems }
    }

    /// Multiply by an element of the vertical subring: each `elems[i]` is
    /// scaled by `scalar` in `F_{2^256}`.
    pub fn scale_vertical(mut self, scalar: F256) -> Self {
        for e in self.elems.iter_mut() {
            *e *= scalar;
        }
        self
    }

    /// Multiply by an element of the horizontal subring. Implemented as
    /// `transpose ∘ scale_vertical ∘ transpose`.
    pub fn scale_horizontal(self, scalar: F256) -> Self {
        self.transpose().scale_vertical(scalar).transpose()
    }

    /// Transpose the tensor algebra element: swap vertical and horizontal
    /// subring roles. Concretely, after transpose, `bit_j(elems'[i]) =
    /// bit_i(elems[j])` for all `i, j ∈ [0, 256)`.
    pub fn transpose(mut self) -> Self {
        square_transpose(&mut self.elems);
        self
    }

    /// Fold the tensor algebra element to a single `F256` by scaling rows with
    /// `coeffs` (length 256) and summing.
    ///
    /// Computes `Σ_i coeffs[i] · transpose(self).elems[i]`.
    pub fn fold_vertical(self, coeffs: &[F256]) -> F256 {
        assert_eq!(
            coeffs.len(),
            DEGREE,
            "fold_vertical: coeffs.len() must be 256"
        );
        let transposed = self.transpose();
        let mut acc = F256::ZERO;
        for (e, c) in transposed.elems.iter().zip(coeffs.iter()) {
            acc += *e * *c;
        }
        acc
    }
}

impl Add<&TensorAlgebra> for TensorAlgebra {
    type Output = TensorAlgebra;
    fn add(mut self, rhs: &TensorAlgebra) -> TensorAlgebra {
        self += rhs;
        self
    }
}

impl AddAssign<&TensorAlgebra> for TensorAlgebra {
    fn add_assign(&mut self, rhs: &TensorAlgebra) {
        for (a, b) in self.elems.iter_mut().zip(rhs.elems.iter()) {
            *a = *a + *b;
        }
    }
}

/// Read bit `b ∈ [0, 256)` of an `F256` element (natural `(c0, c1)` layout).
#[inline]
fn f256_bit(x: F256, b: usize) -> u64 {
    let word = match b >> 6 {
        0 => x.c0.lo,
        1 => x.c0.hi,
        2 => x.c1.lo,
        _ => x.c1.hi,
    };
    (word >> (b & 63)) & 1
}

/// Build an `F256` from its 256 bits given as a per-word accumulator.
#[inline]
fn f256_from_words(w0: u64, w1: u64, w2: u64, w3: u64) -> F256 {
    F256 {
        c0: F128 { lo: w0, hi: w1 },
        c1: F128 { lo: w2, hi: w3 },
    }
}

/// In-place 256×256 F_2 matrix transpose of the F256 coefficient table.
///
/// On input: `elems[i]` viewed as a 256-bit row; bit `j` is the F_2 coefficient
/// at position `(i, j)`.
/// On output: bit `j` of `elems[i]` becomes the old bit `i` of `elems[j]`.
///
/// V1 implementation: naive O(D²) bit-scan. Each of 256² output bits is read
/// from exactly one input bit.
fn square_transpose(elems: &mut [F256]) {
    assert_eq!(
        elems.len(),
        DEGREE,
        "square_transpose: input must be length 256"
    );

    let mut out = vec![F256::ZERO; DEGREE];
    for (j, slot) in out.iter_mut().enumerate() {
        // out[j] gathers bit j of every input row: out[j] bit i = elems[i] bit j.
        let mut w = [0u64; 4];
        for i in 0..DEGREE {
            w[i >> 6] |= f256_bit(elems[i], j) << (i & 63);
        }
        *slot = f256_from_words(w[0], w[1], w[2], w[3]);
    }
    elems.copy_from_slice(&out);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn nx(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.nx(),
                hi: self.nx(),
            }
        }
        fn f256(&mut self) -> F256 {
            F256 {
                c0: self.f128(),
                c1: self.f128(),
            }
        }
        fn ta(&mut self) -> TensorAlgebra {
            let elems = (0..DEGREE).map(|_| self.f256()).collect();
            TensorAlgebra { elems }
        }
    }

    #[test]
    fn transpose_involution() {
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..10 {
            let t = rng.ta();
            assert_eq!(t.clone().transpose().transpose(), t);
        }
    }

    #[test]
    fn transpose_bit_semantics() {
        // bit_j(elems[i]) on input becomes bit_i(elems[j]) on output.
        let mut rng = Rng::new(42);
        let original = rng.ta();
        let transposed = original.clone().transpose();

        for i in 0..DEGREE {
            for j in 0..DEGREE {
                let orig_ij = f256_bit(original.elems[i], j);
                let trans_ji = f256_bit(transposed.elems[j], i);
                assert_eq!(orig_ij, trans_ji, "transpose mismatch at (i={i}, j={j})");
            }
        }
    }

    #[test]
    fn from_vertical_scale_vertical() {
        // from_vertical(x).scale_vertical(y) should equal from_vertical(x*y).
        let mut rng = Rng::new(123);
        for _ in 0..10 {
            let x = rng.f256();
            let y = rng.f256();
            let lhs = TensorAlgebra::from_vertical(x).scale_vertical(y);
            let rhs = TensorAlgebra::from_vertical(x * y);
            assert_eq!(lhs, rhs);
        }
    }

    #[test]
    fn scale_horizontal_via_transpose() {
        let mut rng = Rng::new(456);
        for _ in 0..10 {
            let t = rng.ta();
            let s = rng.f256();
            let via_api = t.clone().scale_horizontal(s);
            let manual = t.transpose().scale_vertical(s).transpose();
            assert_eq!(via_api, manual);
        }
    }

    #[test]
    fn add_is_xor_pairwise() {
        let mut rng = Rng::new(789);
        let a = rng.ta();
        let b = rng.ta();
        let sum = a.clone() + &b;
        for i in 0..DEGREE {
            assert_eq!(sum.elems[i], a.elems[i] + b.elems[i]);
        }
    }

    #[test]
    fn add_zero_is_identity() {
        let mut rng = Rng::new(1011);
        let t = rng.ta();
        let z = TensorAlgebra::zero();
        assert_eq!(t.clone() + &z, t);
    }

    #[test]
    fn one_from_vertical_one() {
        assert_eq!(
            TensorAlgebra::one(),
            TensorAlgebra::from_vertical(F256::ONE)
        );
    }

    #[test]
    fn scale_vertical_distributes_over_add() {
        let mut rng = Rng::new(1213);
        let a = rng.ta();
        let b = rng.ta();
        let s = rng.f256();
        let lhs = (a.clone() + &b).scale_vertical(s);
        let rhs = a.scale_vertical(s) + &b.scale_vertical(s);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn fold_vertical_with_zero_coeffs_is_zero() {
        let mut rng = Rng::new(1415);
        let t = rng.ta();
        let zeros = vec![F256::ZERO; DEGREE];
        assert_eq!(t.fold_vertical(&zeros), F256::ZERO);
    }
}
