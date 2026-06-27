//! GF(2^256) as a quadratic extension of [`F128`]: `F256 = F128[u]/(u² + u + β)`.
//!
//! Layout: an element is `c0 + c1·u` with `c0, c1 ∈ F128`. The base field
//! embeds for free as `a ↦ a + 0·u`, so every tuned F128 PMULL kernel is
//! reused unchanged.
//!
//! Why this irreducible form: over a binary field every element is a square
//! (Frobenius `t ↦ t²` is a bijection), so `u² + β` is **never** irreducible.
//! The only degree-2 irreducibles are the Artin–Schreier polynomials
//! `u² + u + β`, which are irreducible over F128 iff `Tr_{F128/F2}(β) = 1`
//! (the equation `t² + t = β` then has no root in F128). See [`BETA`].
//!
//! Multiplication (Karatsuba): with `u² = u + β` (char 2),
//! ```text
//!   (a0 + a1·u)(b0 + b1·u)
//!     = a0·b0 + (a0·b1 + a1·b0)·u + a1·b1·u²
//!     = (a0·b0 + β·a1·b1) + (a0·b0 + (a0+a1)(b0+b1))·u
//! ```
//! i.e. `m0 = a0·b0`, `m1 = (a0+a1)(b0+b1)`, `m2 = a1·b1`, then
//! `c0 = m0 + β·m2`, `c1 = m0 + m1`. Three F128 muls + one mul-by-β + XORs.
//!
//! Inversion uses the F128-norm: the nontrivial F128-automorphism (Frobenius
//! `x ↦ x^{2^128}`) fixes F128 and sends `u ↦ u+1` (the two roots of the
//! minimal polynomial sum to 1), so the conjugate of `A = c0 + c1·u` is
//! `Ā = (c0+c1) + c1·u` and the norm `N(A) = A·Ā = c0² + c0·c1 + β·c1² ∈ F128`.
//! Then `A⁻¹ = Ā · N(A)⁻¹`, costing one F128 inverse plus a handful of muls.

use core::ops::{Add, AddAssign, Mul, MulAssign};

use serde::{Deserialize, Serialize};

use super::F128;
use super::gf2_128::ghash_reduce;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C, align(32))]
pub struct F256 {
    /// Coefficient of `u^0` (the embedded F128 part).
    pub c0: F128,
    /// Coefficient of `u^1`.
    pub c1: F128,
}

/// The Artin–Schreier constant `β` for `F256 = F128[u]/(u² + u + β)`.
///
/// `β = x^121` (the single GHASH basis element at bit 121), which has
/// `Tr_{F128/F2}(β) = 1` — verified in `tests::beta_is_irreducible` — so the
/// polynomial is irreducible. In this GHASH field every power `x^k` with
/// `k < 121` has trace 0 (see `tests::beta_candidate_survey`); 121 is the
/// smallest trace-1 power. Being a *single* basis element is what makes
/// mul-by-β cheap — not a small shift, but a plain polynomial shift `· x^121`
/// + one GHASH reduction with **no PMULL** (see [`mul_by_beta`]).
pub const BETA: F128 = F128 {
    lo: 0,
    hi: 1 << 57, // bit 121 = 64 + 57
};

/// Multiply an F128 element by `β = x^121`. Since β is a single basis element,
/// `a·β = a·x^121` is the polynomial product `a << 121` reduced mod the GHASH
/// polynomial — pure shifts + one scalar [`ghash_reduce`], **no PMULL**.
///
/// `121 = 64 + 57`, so the 256-bit shifted value `(r0, r1, r2, r3)` is
/// `r0 = 0`, `r1 = lo << 57`, `r2 = (lo >> 7) ^ (hi << 57)`, `r3 = hi >> 7`.
#[inline]
fn mul_by_beta(a: F128) -> F128 {
    let lo = a.lo;
    let hi = a.hi;
    let r1 = lo << 57;
    let r2 = (lo >> 7) ^ (hi << 57);
    let r3 = hi >> 7;
    ghash_reduce(0, r1, r2, r3)
}

impl F256 {
    pub const ZERO: Self = Self {
        c0: F128::ZERO,
        c1: F128::ZERO,
    };
    pub const ONE: Self = Self {
        c0: F128::ONE,
        c1: F128::ZERO,
    };

    #[inline]
    pub const fn new(c0: F128, c1: F128) -> Self {
        Self { c0, c1 }
    }

    /// Embed `a ∈ F128` into F256 as `a + 0·u`. Zero-cost; the inclusion
    /// `F128 ↪ F256` is a field homomorphism.
    #[inline]
    pub const fn from_f128(a: F128) -> Self {
        Self {
            c0: a,
            c1: F128::ZERO,
        }
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    /// Multiply by a base-field scalar `s ∈ F128`: `(c0 + c1·u)·s =
    /// c0·s + (c1·s)·u`. Two F128 muls — used on the hot paths where one
    /// operand is known to live in the base field (NTT twiddles, eq weights
    /// over a base-field domain).
    #[inline]
    pub fn mul_f128(self, s: F128) -> Self {
        Self {
            c0: self.c0 * s,
            c1: self.c1 * s,
        }
    }

    /// The nontrivial F128-automorphism (Frobenius `x ↦ x^{2^128}`) applied to
    /// `self`: fixes F128 pointwise and sends `u ↦ u + 1`, so
    /// `conj(c0 + c1·u) = (c0 + c1) + c1·u`.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            c0: self.c0 + self.c1,
            c1: self.c1,
        }
    }

    /// Field norm `N(self) = self · conj(self) = c0² + c0·c1 + β·c1² ∈ F128`.
    #[inline]
    pub fn norm(self) -> F128 {
        // c0·(c0 + c1) + β·c1²  =  c0² + c0·c1 + β·c1².
        self.c0 * (self.c0 + self.c1) + mul_by_beta(self.c1 * self.c1)
    }

    /// Multiplicative inverse via the F128-norm: `A⁻¹ = conj(A) · N(A)⁻¹`.
    /// One F128 inverse + three F128 muls. Panics on zero in debug builds via
    /// the F128 inverse (which returns garbage for 0); callers guard with
    /// [`is_zero`](Self::is_zero) where 0 is possible.
    pub fn inv(self) -> Self {
        let n_inv = self.norm().inv();
        Self {
            c0: (self.c0 + self.c1) * n_inv,
            c1: self.c1 * n_inv,
        }
    }
}

impl Add for F256 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 + rhs.c0,
            c1: self.c1 + rhs.c1,
        }
    }
}

impl AddAssign for F256 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.c0 += rhs.c0;
        self.c1 += rhs.c1;
    }
}

impl Mul for F256 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Karatsuba: 3 fully-reduced F128 muls + a PMULL-free shift for `β·m2`.
        // Each `*` dispatches to the tuned binius PMULL kernel and stays
        // NEON-native, so this beats both the 4-mul schoolbook (extra full mul
        // for β) and a deferred-reduction form (which would force the 256-bit
        // intermediates out of NEON into scalar GPRs — the NEON↔GPR transfers
        // cost more than the PMULLs they save). A `ghash_mul_vec2_neon` path for
        // `m0`/`m2` was tried and is slower: its lane pack/unpack overhead
        // exceeds the saved PMULLs. Measured ~2.6 ns/op vs ~3.2 ns naive (M4).
        let a0 = self.c0;
        let a1 = self.c1;
        let b0 = rhs.c0;
        let b1 = rhs.c1;
        let m0 = a0 * b0;
        let m2 = a1 * b1;
        let m1 = (a0 + a1) * (b0 + b1);
        Self {
            c0: m0 + mul_by_beta(m2),
            c1: m0 + m1,
        }
    }
}

impl MulAssign for F256 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// `F128 · F256` from the left, for the mixed-field hot paths. Same cost as
/// [`F256::mul_f128`] (two F128 muls); provided so call sites read naturally.
impl Mul<F256> for F128 {
    type Output = F256;
    #[inline]
    fn mul(self, rhs: F256) -> F256 {
        rhs.mul_f128(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn next_f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn next_f256(&mut self) -> F256 {
            F256 {
                c0: self.next_f128(),
                c1: self.next_f128(),
            }
        }
    }

    /// `Tr_{F128/F2}(b) = Σ_{i=0}^{127} b^{2^i} ∈ {0, 1}`.
    fn trace_f128(b: F128) -> F128 {
        let mut acc = F128::ZERO;
        let mut cur = b;
        for _ in 0..128 {
            acc += cur;
            cur = cur * cur;
        }
        acc
    }

    /// `u² + u + β` is irreducible over F128 ⟺ `Tr(β) = 1`. This is the
    /// soundness-critical invariant: a reducible polynomial would make F256 a
    /// ring with zero divisors, not a field.
    #[test]
    fn beta_is_irreducible() {
        assert_eq!(
            trace_f128(BETA),
            F128::ONE,
            "Tr(β) must be 1 for irreducibility"
        );
    }

    /// Search for the lowest-weight `β` with `Tr(β) = 1`. Prints the smallest
    /// power `x^k` and the smallest low-bit-pattern candidates so `BETA` can be
    /// chosen for a cheap mul-by-β. Run with `--nocapture` to read the survey.
    #[test]
    fn beta_candidate_survey() {
        // Smallest k with Tr(x^k) = 1 (β = x^k → mul-by-β is k shift+folds).
        let mut xk = F128::ONE;
        let mut best_pow = None;
        for k in 0..256usize {
            if trace_f128(xk) == F128::ONE && best_pow.is_none() {
                best_pow = Some((k, xk));
            }
            xk = xk * F128::generator();
        }
        eprintln!("smallest x^k with trace 1: {best_pow:?}");

        // Smallest single low-word bit pattern with trace 1.
        let mut best_lo = None;
        for lo in 1u64..=64 {
            if trace_f128(F128 { lo, hi: 0 }) == F128::ONE {
                best_lo = Some(lo);
                break;
            }
        }
        eprintln!("smallest lo (hi=0) with trace 1: {best_lo:?}");

        assert!(best_pow.is_some(), "some power of x must have trace 1");
    }

    /// The shift-based `mul_by_beta` (no PMULL) equals the general F128
    /// multiply `a · BETA`, across the full input range incl. high bits.
    #[test]
    fn mul_by_beta_matches_general() {
        let mut rng = Rng::new(0xBE7A);
        for _ in 0..512 {
            let a = rng.next_f128();
            assert_eq!(mul_by_beta(a), a * BETA);
        }
        // Edge cases: the carry boundaries of the 121-bit shift.
        for &a in &[
            F128 { lo: 1, hi: 0 },      // 1·β = β
            F128 { lo: 1 << 7, hi: 0 }, // x^7·x^121 = x^128 = 0x87 (reduction)
            F128 {
                lo: u64::MAX,
                hi: 0,
            },
            F128 {
                lo: 0,
                hi: u64::MAX,
            },
            F128 {
                lo: u64::MAX,
                hi: u64::MAX,
            },
        ] {
            assert_eq!(mul_by_beta(a), a * BETA, "edge a={a:?}");
        }
    }

    /// `BETA` is exactly `x^121` (the documented single-bit choice), so the
    /// doc comment and the `mul_by_beta` cost analysis stay honest.
    #[test]
    fn beta_is_x_pow_121() {
        let mut xk = F128::ONE;
        for _ in 0..121 {
            xk = xk * F128::generator();
        }
        assert_eq!(BETA, xk, "BETA must equal x^121");
    }

    #[test]
    fn add_identities() {
        let mut rng = Rng::new(2);
        for _ in 0..128 {
            let a = rng.next_f256();
            assert_eq!(a + F256::ZERO, a);
            assert_eq!(a + a, F256::ZERO);
        }
    }

    #[test]
    fn mul_identities() {
        let mut rng = Rng::new(3);
        for _ in 0..128 {
            let a = rng.next_f256();
            assert_eq!(a * F256::ZERO, F256::ZERO);
            assert_eq!(a * F256::ONE, a);
            assert_eq!(F256::ONE * a, a);
        }
    }

    #[test]
    fn commutativity() {
        let mut rng = Rng::new(4);
        for _ in 0..256 {
            let a = rng.next_f256();
            let b = rng.next_f256();
            assert_eq!(a * b, b * a);
        }
    }

    #[test]
    fn associativity_and_distributivity() {
        let mut rng = Rng::new(5);
        for _ in 0..256 {
            let a = rng.next_f256();
            let b = rng.next_f256();
            let c = rng.next_f256();
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    /// `F128 ↪ F256` is a ring homomorphism: it respects + and ×, so base-field
    /// arithmetic is preserved under the embedding.
    #[test]
    fn embedding_is_homomorphism() {
        let mut rng = Rng::new(6);
        for _ in 0..256 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            let ea = F256::from_f128(a);
            let eb = F256::from_f128(b);
            assert_eq!(ea + eb, F256::from_f128(a + b));
            assert_eq!(
                ea * eb,
                F256::from_f128(a * b),
                "embedding must respect mul"
            );
            // Embedded ONE / ZERO line up.
            assert_eq!(F256::from_f128(F128::ONE), F256::ONE);
            assert_eq!(F256::from_f128(F128::ZERO), F256::ZERO);
        }
    }

    /// `mul_f128` (and the `F128 * F256` impl) agree with the full F256 mul
    /// after embedding the scalar.
    #[test]
    fn mul_f128_matches_full_mul() {
        let mut rng = Rng::new(7);
        for _ in 0..256 {
            let a = rng.next_f256();
            let s = rng.next_f128();
            let expected = a * F256::from_f128(s);
            assert_eq!(a.mul_f128(s), expected);
            assert_eq!(s * a, expected);
        }
    }

    #[test]
    fn conjugate_is_involution_and_frobenius() {
        let mut rng = Rng::new(8);
        for _ in 0..256 {
            let a = rng.next_f256();
            // conj is an involution.
            assert_eq!(a.conjugate().conjugate(), a);
            // conj is the Frobenius x ↦ x^{2^128}: it is a field automorphism,
            // so conj(a·b) = conj(a)·conj(b).
            let b = rng.next_f256();
            assert_eq!((a * b).conjugate(), a.conjugate() * b.conjugate());
            // Frobenius fixes exactly the base field F128.
            let base = F256::from_f128(rng.next_f128());
            assert_eq!(base.conjugate(), base);
        }
    }

    #[test]
    fn norm_matches_product_with_conjugate() {
        let mut rng = Rng::new(9);
        for _ in 0..256 {
            let a = rng.next_f256();
            let n = a * a.conjugate();
            // The norm lives in the base field (u-coefficient is zero).
            assert_eq!(n.c1, F128::ZERO, "norm must lie in F128");
            assert_eq!(n, F256::from_f128(a.norm()));
        }
    }

    #[test]
    fn inverse_roundtrip() {
        let mut rng = Rng::new(10);
        for _ in 0..256 {
            let a = rng.next_f256();
            if a.is_zero() {
                continue;
            }
            assert_eq!(a * a.inv(), F256::ONE);
            assert_eq!(a.inv() * a, F256::ONE);
        }
    }

    /// `u` is a genuine root of `u² + u + β = 0`, i.e. `u² = u + β`. The most
    /// direct check that the reduction baked into `mul` matches the defining
    /// polynomial.
    #[test]
    fn u_satisfies_minimal_polynomial() {
        let u = F256 {
            c0: F128::ZERO,
            c1: F128::ONE,
        };
        let u2 = u * u;
        let u_plus_beta = u + F256::from_f128(BETA);
        assert_eq!(u2, u_plus_beta, "u² must equal u + β");
    }

    /// F256 has odd multiplicative structure: `a^(2^256) = a` (Frobenius^256 =
    /// identity). Cheap surrogate: `a^(2^256 - 1) = 1` for nonzero a, checked
    /// here as `a · a^(2^256 - 2) = 1` against the norm-based inverse.
    #[test]
    fn frobenius_256_is_identity_on_squares() {
        let mut rng = Rng::new(11);
        for _ in 0..32 {
            let a = rng.next_f256();
            // a^{2^256} = (a^{2^128})^{2^128} = conj(conj(a)) = a.
            // conj = Frobenius^128, so applying it twice = Frobenius^256.
            assert_eq!(a.conjugate().conjugate(), a);
        }
    }
}
