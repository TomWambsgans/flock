// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// The verifier's polylog `eval_rs_eq` helper is ported from binius64's
// `crates/verifier/src/ring_switch.rs`
// (https://github.com/binius-zk/binius64). The rest of this module (the
// prover-side reduction adapted for the φ_8 LCH basis) is original to Flock.

//! Ring-switching reduction (DP24-style, adapted for the φ_8 LCH basis).
//!
//! Converts the zerocheck's claim `ẑ_skip(z_skip, x_outer) = v` into a BaseFold
//! sumcheck claim over the packed multilinear `f_packed` with a transparent
//! multilinear `rs_eq_ind`.
//!
//! ## Non-novelty basis: only affects the claim-check step
//!
//! Binius's DP24 ring-switching uses tensor-product (`eq_ind`) weights for the
//! verifier's claim check. That requires the prefix's LCH-Lagrange to factor
//! as `eq(x_skip, i_skip)`, which holds only for the *novelty basis* of the
//! subspace.
//!
//! Our zerocheck uses the φ_8 image of {1,2,4,…,32} as the 6-dim LCH basis.
//! That basis is **not** a novelty basis (verified at k=2: the ratio of
//! Lagrange values doesn't satisfy the tensor identity), so the 64 weights
//! `ν_φ8(i_skip)(z_skip)` are not tensor-factorizable.
//!
//! Resolution: replace the verifier's claim check with **direct** Lagrange
//! weights (computed via [`lagrange_weights_naive`]); every other component of
//! the reduction (`s_hat_v`, `s_hat_u`, BaseFold target `T`, `rs_eq_ind`) is
//! independent of the prefix and stays identical to Binius.
//!
//! ## Prover vs. verifier paths for `rs_eq_ind`
//!
//! - **Prover side** (used by [`prove`], [`prove_batched`]): materializes
//!   `rs_eq_ind` densely (or sparsely) via [`fold_b128_elems`] / [`RsEqInd`].
//!   The dense vector becomes the BaseFold target witness, so the prover does
//!   need the full `2^(m-8)` entries.
//! - **Verifier side** (used by [`verify_succinct`] + [`eval_rs_eq`]): never
//!   materializes `rs_eq_ind`. Instead, evaluates `MLE(rs_eq_ind)(c)` at the
//!   BaseFold final challenge point in `O((m-8) · 256²)` field ops via the
//!   DP24 tensor-algebra iterative algorithm ([DP24] §1.3, Figure 3). This is
//!   polylog in the witness size.
//!
//! [DP24]: <https://eprint.iacr.org/2024/504>
//!
//! ## Layout (for m-bit witness, F_{2^256} packing with LOG_PACKING = 8)
//!
//! Zerocheck output: `(z_skip ∈ F, x_outer ∈ F^{m−6})` with claim `v`.
//!
//! After translation:
//! - **prefix bits 0..6**: weighted by `ν_φ8(·)(z_skip)` (the 64 Lagrange weights).
//! - **prefix bits 6, 7**: weighted by `eq(x_outer[0], ·)` and `eq(x_outer[1], ·)`
//!   (the two F_{2^256} multilinear coords past the 6 univariate-skip coords).
//! - **suffix coords**: `x_outer[2..]`, length `m − 8`.
//!
//! The packed witness has `2^(m−8)` F_{2^256} elements indexed by the suffix.
//! `s_hat_v` has 256 entries indexed by the 8-bit prefix.
//!
//! ## PERF NOTE (F256 migration)
//!
//! The F128 fold kernels used the method-of-four-Russians algorithm + 8×8 bit
//! transposes and (on aarch64) NEON. None of that generalizes mechanically to
//! the 256-bit / 32-byte F256 layout, so every fold kernel here is a clean
//! portable bit-scan / byte-table over F256 (correctness over micro-opt).
//! PERF TODO(f256): re-derive the MFR / NEON kernels for F256.

use crate::challenger::Challenger;
use crate::field::{F128, F256};
use crate::zerocheck::PaddingSpec;
use crate::zerocheck::multilinear::lagrange_weights_naive;
use crate::zerocheck::univariate_skip::build_eq;
use serde::{Deserialize, Serialize};

use super::pack::LOG_PACKING;

// ---------------------------------------------------------------------------
// Portable F256 bit primitives.
// ---------------------------------------------------------------------------

/// The four 64-bit limbs of an `F256` in natural bit order
/// (`c0.lo, c0.hi, c1.lo, c1.hi`); bit `r ∈ [0,256)` lives in limb `r >> 6`,
/// bit `r & 63`. Matches `pack`'s convention and `tensor_algebra::f256_bit`.
#[inline(always)]
fn f256_limbs(x: F256) -> [u64; 4] {
    [x.c0.lo, x.c0.hi, x.c1.lo, x.c1.hi]
}

/// Read bit `b ∈ [0,256)` of an `F256` (natural limb layout).
#[inline(always)]
fn f256_bit(x: F256, b: usize) -> u64 {
    let word = match b >> 6 {
        0 => x.c0.lo,
        1 => x.c0.hi,
        2 => x.c1.lo,
        _ => x.c1.hi,
    };
    (word >> (b & 63)) & 1
}

/// `acc[r] += val` for every set bit `r ∈ [0,256)` of `elem`.
#[inline(always)]
fn scatter_set_bits(acc: &mut [F256], elem: F256, val: F256) {
    for (limb_idx, &word) in f256_limbs(elem).iter().enumerate() {
        let base = limb_idx * 64;
        let mut bits = word;
        while bits != 0 {
            let r = bits.trailing_zeros() as usize;
            acc[base + r] += val;
            bits &= bits - 1;
        }
    }
}

/// `Σ_{b set in elem} eq[b]`, `b ∈ [0,256)`. The DP24 `fold_b128` per-element
/// kernel (decompose `elem ∈ F256` into 256 F_2-bits, inner-product with `eq`).
#[inline(always)]
fn fold_bits_against(elem: F256, eq: &[F256]) -> F256 {
    let mut acc = F256::ZERO;
    for (limb_idx, &word) in f256_limbs(elem).iter().enumerate() {
        let base = limb_idx * 64;
        let mut bits = word;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            acc += eq[base + b];
            bits &= bits - 1;
        }
    }
    acc
}

/// Build the 256-entry weights vector for the verifier's ring-switching claim
/// check, given the zerocheck's `z_skip` (univariate-skip coord, absorbs 6
/// boolean coords via the φ_8 basis) and the two fresh F_{2^256} multilinear
/// coords `x_outer_0`, `x_outer_1` (the 7th and 8th prefix bits).
///
/// ```text
/// weights[i] = ν_φ8(i & 63)(z_skip) · eq(x_outer_0, (i >> 6) & 1)
///                                   · eq(x_outer_1, (i >> 7) & 1)
///            for i ∈ {0..256}
/// ```
///
/// `i & 63` selects the low 6 bits (LCH dimensions); `(i >> 6) & 1` is the 7th
/// bit and `(i >> 7) & 1` the 8th bit (two standard multilinear coords).
pub fn build_claim_weights(z_skip: F256, x_outer_0: F256, x_outer_1: F256) -> Vec<F256> {
    const K_SKIP: usize = 6;
    // The 8-bit index tiles as 6 (φ8/LCH) + 1 (x_outer_0) + 1 (x_outer_1), so
    // this routine is correct only when LOG_PACKING == K_SKIP + 2. Guard the
    // coupling: if LOG_PACKING changes, this prefix width must change too.
    debug_assert_eq!(LOG_PACKING, K_SKIP + 2);
    let lambda = lagrange_weights_naive(K_SKIP, z_skip); // length 64
    debug_assert_eq!(lambda.len(), 1 << K_SKIP);

    // eq(x_outer_j, {0, 1}).
    let eq0 = [F256::ONE + x_outer_0, x_outer_0];
    let eq1 = [F256::ONE + x_outer_1, x_outer_1];

    let n = 1 << LOG_PACKING; // 256
    let mut weights = Vec::with_capacity(n);
    // Layout: i ∈ {0..256}; low 6 bits → ν_φ8 node, bit 6 → x_outer_0 branch,
    // bit 7 → x_outer_1 branch.
    for i in 0..n {
        let i_lo = i & 63;
        let bit_6 = (i >> 6) & 1;
        let bit_7 = (i >> 7) & 1;
        weights.push(lambda[i_lo] * eq0[bit_6] * eq1[bit_7]);
    }
    weights
}

// ---------------------------------------------------------------------------
// `fold_1b_rows`: s_hat_v[r] = Σ_i bit_r(W[i]) · tensor[i], r ∈ 0..256.
//
// PERF TODO(f256): re-derive the method-of-four-Russians + transpose (+ NEON)
// kernel for the 256-bit layout. All the variants below are thin wrappers over
// the portable bit-scan core so the public API is preserved.
// ---------------------------------------------------------------------------

/// Portable F256 `fold_1b_rows`: `s_hat_v[r] = Σ_i bit_r(W[i]) · tensor[i]`.
/// O(2^L · 256) parallelized across packed-witness positions: each thread folds
/// a chunk into a private length-256 accumulator; the reduce XORs partials.
pub fn fold_1b_rows_naive(packed_witness: &[F256], suffix_tensor: &[F256]) -> Vec<F256> {
    use rayon::prelude::*;
    assert_eq!(packed_witness.len(), suffix_tensor.len());
    let n = 1 << LOG_PACKING; // 256
    let zero_acc = || vec![F256::ZERO; n];

    packed_witness
        .par_iter()
        .zip(suffix_tensor.par_iter())
        .fold(zero_acc, |mut acc, (&elem, &w)| {
            scatter_set_bits(&mut acc, elem, w);
            acc
        })
        .reduce(zero_acc, |mut a, b| {
            for (av, bv) in a.iter_mut().zip(b.iter()) {
                *av += *bv;
            }
            a
        })
}

/// Tensor-split sibling of [`fold_1b_rows_naive`]: takes the two factors
/// `(eq_lo, eq_hi)` from [`build_eq_split`] and reconstructs each suffix entry
/// `tensor[i_hi·B + i_lo] = eq_lo[i_lo] · eq_hi[i_hi]` on the fly. Block-parallel
/// over `eq_hi` (each block reads `eq_lo` against one `e_hi`). Output is
/// **byte-identical** to `fold_1b_rows_naive(W, build_eq(r))` for the split of
/// `r` (GF multiply is exact and distributes).
fn fold_1b_rows_split_portable(
    packed_witness: &[F256],
    eq_lo: &[F256],
    eq_hi: &[F256],
) -> Vec<F256> {
    use rayon::prelude::*;
    let n = 1 << LOG_PACKING; // 256
    let b = eq_lo.len();
    assert_eq!(packed_witness.len(), b * eq_hi.len());
    let zero_acc = || vec![F256::ZERO; n];

    packed_witness
        .par_chunks(b)
        .zip(eq_hi.par_iter())
        .fold(zero_acc, |mut acc, (w_block, &e_hi)| {
            for (i_lo, &elem) in w_block.iter().enumerate() {
                scatter_set_bits(&mut acc, elem, eq_lo[i_lo] * e_hi);
            }
            acc
        })
        .reduce(zero_acc, |mut a, b| {
            for r in 0..n {
                a[r] += b[r];
            }
            a
        })
}

/// Compute `s_hat_v_k` for each `suffix_tensors[k]`. All suffix tensors must
/// have the same length as `packed_witness`.
pub fn fold_1b_rows_multi(packed_witness: &[F256], suffix_tensors: &[&[F256]]) -> Vec<Vec<F256>> {
    let m = LOG_PACKING + (packed_witness.len().trailing_zeros() as usize);
    fold_1b_rows_multi_padded(packed_witness, suffix_tensors, &PaddingSpec::dense(m))
}

/// Padding-aware variant of [`fold_1b_rows_multi`]. The portable F256 bit-scan
/// is correct for honestly zero-padded witnesses regardless of `padding`
/// (skipped chunks are zero → contribute nothing), so `padding` is currently
/// unused. PERF TODO(f256): exploit `padding` to skip zero chunks.
pub fn fold_1b_rows_multi_padded(
    packed_witness: &[F256],
    suffix_tensors: &[&[F256]],
    padding: &PaddingSpec,
) -> Vec<Vec<F256>> {
    let _ = padding;
    assert!(
        suffix_tensors
            .iter()
            .all(|t| t.len() == packed_witness.len())
    );
    suffix_tensors
        .iter()
        .map(|t| fold_1b_rows_naive(packed_witness, t))
        .collect()
}

/// Two-claim fold. Portable F256 wrapper (was a fused MFR kernel).
pub fn fold_1b_rows_2way_mfr(
    packed_witness: &[F256],
    t0: &[F256],
    t1: &[F256],
) -> (Vec<F256>, Vec<F256>) {
    (
        fold_1b_rows_naive(packed_witness, t0),
        fold_1b_rows_naive(packed_witness, t1),
    )
}

/// Padding-aware variant of [`fold_1b_rows_2way_mfr`] (portable; `padding`
/// unused — see [`fold_1b_rows_multi_padded`]).
pub fn fold_1b_rows_2way_mfr_padded(
    packed_witness: &[F256],
    t0: &[F256],
    t1: &[F256],
    padding: &PaddingSpec,
) -> (Vec<F256>, Vec<F256>) {
    let _ = padding;
    fold_1b_rows_2way_mfr(packed_witness, t0, t1)
}

/// Portable F256 wrapper (was an 8-wide fused MFR kernel).
pub fn fold_1b_rows_2way_mfr_8wide(
    packed_witness: &[F256],
    t0: &[F256],
    t1: &[F256],
) -> (Vec<F256>, Vec<F256>) {
    fold_1b_rows_2way_mfr(packed_witness, t0, t1)
}

/// Padding-aware variant of [`fold_1b_rows_2way_mfr_8wide`] (portable).
pub fn fold_1b_rows_2way_mfr_8wide_padded(
    packed_witness: &[F256],
    t0: &[F256],
    t1: &[F256],
    padding: &PaddingSpec,
) -> (Vec<F256>, Vec<F256>) {
    let _ = padding;
    fold_1b_rows_2way_mfr(packed_witness, t0, t1)
}

/// Single-tensor fold. Portable F256 wrapper (was a 4-wide MFR kernel).
pub fn fold_1b_rows_1way_mfr(packed_witness: &[F256], t: &[F256]) -> Vec<F256> {
    fold_1b_rows_naive(packed_witness, t)
}

/// Portable F256 wrapper (was an 8-wide / two-k=4-table MFR kernel).
pub fn fold_1b_rows_1way_mfr_8wide_k4(packed_witness: &[F256], t: &[F256]) -> Vec<F256> {
    fold_1b_rows_naive(packed_witness, t)
}

/// Portable F256 wrapper (was a 16-wide MFR kernel); `padding` unused.
pub fn fold_1b_rows_1way_mfr_16wide_padded(
    packed_witness: &[F256],
    t: &[F256],
    padding: &PaddingSpec,
) -> Vec<F256> {
    let _ = padding;
    fold_1b_rows_naive(packed_witness, t)
}

/// Dense (no-skip) wrapper over [`fold_1b_rows_1way_mfr_16wide_padded`].
pub fn fold_1b_rows_1way_mfr_16wide_k4(packed_witness: &[F256], t: &[F256]) -> Vec<F256> {
    fold_1b_rows_naive(packed_witness, t)
}

/// Tensor-split fold from `(eq_lo, eq_hi)` (output of [`build_eq_split`]).
/// Portable F256; `padding` unused (correct for honest zero-padded witnesses).
/// Byte-identical to `fold_1b_rows_naive(W, build_eq_parallel(r))`.
pub fn fold_1b_rows_split(
    packed_witness: &[F256],
    eq_lo: &[F256],
    eq_hi: &[F256],
    padding: &PaddingSpec,
) -> Vec<F256> {
    let _ = padding;
    fold_1b_rows_split_portable(packed_witness, eq_lo, eq_hi)
}

/// Two-claim split fold. Portable F256; `padding` unused. Per-claim output is
/// byte-identical to two [`fold_1b_rows_split`] calls.
pub fn fold_1b_rows_split_2way(
    packed_witness: &[F256],
    eq_lo_0: &[F256],
    eq_hi_0: &[F256],
    eq_lo_1: &[F256],
    eq_hi_1: &[F256],
    padding: &PaddingSpec,
) -> (Vec<F256>, Vec<F256>) {
    let _ = padding;
    (
        fold_1b_rows_split_portable(packed_witness, eq_lo_0, eq_hi_0),
        fold_1b_rows_split_portable(packed_witness, eq_lo_1, eq_hi_1),
    )
}

// ---------------------------------------------------------------------------
// build_eq helpers.
// ---------------------------------------------------------------------------

/// Parallel `build_eq` for ring-switching's suffix tensors. Same output as
/// [`crate::zerocheck::univariate_skip::build_eq`] (byte-identical), but
/// parallelizes the inner doubling loop across rayon threads.
fn build_eq_parallel(r: &[F256]) -> Vec<F256> {
    use rayon::prelude::*;
    let n = r.len();
    // Uninit alloc — at iter `i`, the loop reads from t[..2^i] (always written
    // by an earlier iter or the t[0] = ONE seed) and writes to t[2^i..2^(i+1)]
    // (purely written, never read first). So every slot is written before any
    // read; uninit is safe.
    let mut t = crate::alloc_uninit_vec::<F256>(1usize << n);
    t[0] = F256::ONE;
    const PAR_THRESHOLD: usize = 1 << 12;
    for i in 0..n {
        let r_i = r[i];
        let one_minus_r = F256::ONE + r_i;
        let half = 1usize << i;
        let (lo, hi_rest) = t.split_at_mut(half);
        let hi = &mut hi_rest[..half];
        if half < PAR_THRESHOLD {
            for (lo_x, hi_x) in lo.iter_mut().zip(hi.iter_mut()) {
                let old = *lo_x;
                *hi_x = old * r_i;
                *lo_x = old * one_minus_r;
            }
        } else {
            lo.par_iter_mut()
                .zip(hi.par_iter_mut())
                .for_each(|(lo_x, hi_x)| {
                    let old = *lo_x;
                    *hi_x = old * r_i;
                    *lo_x = old * one_minus_r;
                });
        }
    }
    t
}

/// Tensor-factored `build_eq`: split the point `r` (length `n`) into a low
/// part `r[..n_lo]` and a high part `r[n_lo..]`, returning the two smaller
/// eq-tables `(eq_lo, eq_hi)` of lengths `2^n_lo` and `2^(n - n_lo)`.
///
/// The full tensor factors **exactly** (GF(2^256) is a field):
/// `build_eq_parallel(r)[i] == eq_lo[i & (2^n_lo - 1)] * eq_hi[i >> n_lo]`.
pub fn build_eq_split(r: &[F256], n_lo: usize) -> (Vec<F256>, Vec<F256>) {
    assert!(n_lo <= r.len());
    let eq_lo = build_eq_parallel(&r[..n_lo]);
    let eq_hi = build_eq_parallel(&r[n_lo..]);
    (eq_lo, eq_hi)
}

/// Pick the low-split width `n_lo` for a suffix tensor of length `2^n`.
/// Balanced near `n/2`, clamped to `[4, n]`.
pub fn split_n_lo(n: usize) -> usize {
    (n / 2).clamp(4, n)
}

/// Compute the verifier's claim check: `Σ_i weights[i] · s_hat_v[i]`.
pub fn claim_check(weights: &[F256], s_hat_v: &[F256]) -> F256 {
    inner_product(weights, s_hat_v)
}

/// Standard inner product `Σ_i a[i] · b[i]` over F_{2^256}.
pub fn inner_product(a: &[F256], b: &[F256]) -> F256 {
    assert_eq!(a.len(), b.len());
    let mut acc = F256::ZERO;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

/// **TensorAlgebra transpose** (a.k.a. "bit transpose" of `s_hat_v`).
///
/// View `s_hat_v` (length 256) as a 256×256 binary matrix with row `i_skip` =
/// the 256 natural-basis bits of `s_hat_v[i_skip]`. Output `s_hat_u`
/// (length 256) is the transposed matrix re-packed:
/// ```text
///     bit i_skip of s_hat_u[b]  ==  bit b of s_hat_v[i_skip]
/// ```
///
/// Naive O(256²) bit-gather implementation.
/// PERF TODO(f256): NEON / blocked bit-transpose.
pub fn tensor_algebra_transpose(s_hat_v: &[F256]) -> Vec<F256> {
    let n = 1 << LOG_PACKING; // 256
    assert_eq!(s_hat_v.len(), n);
    let mut s_hat_u = vec![F256::ZERO; n];
    for (b, slot) in s_hat_u.iter_mut().enumerate() {
        // s_hat_u[b] gathers bit b of every input row: bit i_skip = s_hat_v[i_skip] bit b.
        let mut w = [0u64; 4];
        for i_skip in 0..n {
            w[i_skip >> 6] |= f256_bit(s_hat_v[i_skip], b) << (i_skip & 63);
        }
        *slot = F256 {
            c0: F128 { lo: w[0], hi: w[1] },
            c1: F128 { lo: w[2], hi: w[3] },
        };
    }
    s_hat_u
}

// ---------------------------------------------------------------------------
// fold_b128_elems: rs_eq_ind[i] = Σ_b bit_b(suffix[i]) · eq_r_dprime[b], b ∈ 0..256.
// ---------------------------------------------------------------------------

/// Naive bit-scan `fold_b128_elems`. O(256 · 2^L) parallel over positions.
pub fn fold_b128_elems_naive(suffix_tensor: &[F256], eq_r_dprime: &[F256]) -> Vec<F256> {
    use rayon::prelude::*;
    assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);
    suffix_tensor
        .par_iter()
        .map(|&elem| fold_bits_against(elem, eq_r_dprime))
        .collect()
}

/// Number of bytes in an `F256` (= byte-lookup tables for the fold).
const FOLD_N_BYTES: usize = 32;
/// Entries per byte-lookup table.
const FOLD_TABLE_SIZE: usize = 256;

/// 32 little-endian bytes of an `F256` (limb order `c0.lo, c0.hi, c1.lo, c1.hi`).
#[inline(always)]
fn f256_to_le_bytes(x: F256) -> [u8; FOLD_N_BYTES] {
    let mut b = [0u8; FOLD_N_BYTES];
    b[0..8].copy_from_slice(&x.c0.lo.to_le_bytes());
    b[8..16].copy_from_slice(&x.c0.hi.to_le_bytes());
    b[16..24].copy_from_slice(&x.c1.lo.to_le_bytes());
    b[24..32].copy_from_slice(&x.c1.hi.to_le_bytes());
    b
}

/// Build the 32×256 byte-lookup table the fold indexes: `table[k·256 + v]` =
/// `Σ_{bit b set in v} eq_r_dprime[k·8 + b]`. For the ring-switch fold,
/// `eq_r_dprime` already has γ_k baked in, so the table carries γ too.
fn build_fold_byte_table(eq_r_dprime: &[F256]) -> Vec<F256> {
    assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);
    let mut tables = vec![F256::ZERO; FOLD_N_BYTES * FOLD_TABLE_SIZE];
    for byte_idx in 0..FOLD_N_BYTES {
        let bit_base = byte_idx * 8;
        for value in 0..FOLD_TABLE_SIZE {
            let mut acc = F256::ZERO;
            for bit_in_byte in 0..8 {
                if (value >> bit_in_byte) & 1 == 1 {
                    acc += eq_r_dprime[bit_base + bit_in_byte];
                }
            }
            tables[byte_idx * FOLD_TABLE_SIZE + value] = acc;
        }
    }
    tables
}

/// One folded output slot: `Σ_{k=0..32} tables[k·256 + byte_k(elem)]`, where
/// `byte_k` are the 32 little-endian bytes of `elem`. `tables` MUST be a
/// [`build_fold_byte_table`] output (length `32·256`).
/// PERF TODO(f256): re-derive the tree-reduced / NEON byte-table fold.
#[inline(always)]
pub(crate) fn fold_one_slot(elem: F256, tables: &[F256]) -> F256 {
    debug_assert_eq!(tables.len(), FOLD_N_BYTES * FOLD_TABLE_SIZE);
    let bytes = f256_to_le_bytes(elem);
    let mut acc = F256::ZERO;
    for (k, &byte) in bytes.iter().enumerate() {
        acc += tables[k * FOLD_TABLE_SIZE + byte as usize];
    }
    acc
}

/// Per-output-index value of a [`RsEqInd::DeferredDense`] fold (the value the
/// materialized `fold_b128_elems_split` would store at position `j`):
/// `fold_one_slot(eq_lo[j & (B−1)] · eq_hi[j >> log2 B], table)`, `B = eq_lo.len()`.
#[inline(always)]
pub(crate) fn deferred_dense_value(
    eq_lo: &[F256],
    eq_hi: &[F256],
    table: &[F256],
    log_b: usize,
    j: usize,
) -> F256 {
    let mask = (1usize << log_b) - 1;
    fold_one_slot(eq_lo[j & mask] * eq_hi[j >> log_b], table)
}

/// Bit-table accelerated `fold_b128_elems`. Builds the 32×256 byte-table once,
/// then one [`fold_one_slot`] per suffix position.
pub fn fold_b128_elems(suffix_tensor: &[F256], eq_r_dprime: &[F256]) -> Vec<F256> {
    use rayon::prelude::*;
    let tables = build_fold_byte_table(eq_r_dprime);
    suffix_tensor
        .par_iter()
        .map(|&elem| fold_one_slot(elem, &tables))
        .collect()
}

/// Tensor-split sibling of [`fold_b128_elems`]. Reconstructs each entry
/// `eq_lo[i_lo] * eq_hi[i_hi]` on the fly (one GF multiply) and folds it via the
/// byte-table. Output order matches the materialized tensor
/// (`out[i_hi·B + i_lo]`, `B = eq_lo.len()`); byte-identical to
/// `fold_b128_elems(build_eq_parallel(r), eq_r_dprime)`.
pub fn fold_b128_elems_split(eq_lo: &[F256], eq_hi: &[F256], eq_r_dprime: &[F256]) -> Vec<F256> {
    let tables = build_fold_byte_table(eq_r_dprime);
    fold_b128_from_table(eq_lo, eq_hi, &tables)
}

/// Materialize a split-tensor fold from a prebuilt byte `tables`
/// ([`build_fold_byte_table`] output). Block-parallel over `eq_hi`. Used to
/// un-defer a [`RsEqInd::DeferredDense`] in the pcs combine's general fallback.
pub(crate) fn fold_b128_from_table(eq_lo: &[F256], eq_hi: &[F256], tables: &[F256]) -> Vec<F256> {
    use rayon::prelude::*;
    let b = eq_lo.len();
    // Each slot is written exactly once (`*slot = acc`) before any read.
    let mut out = crate::scratch::take_f256(b * eq_hi.len());
    out.par_chunks_mut(b)
        .zip(eq_hi.par_iter())
        .for_each(|(out_block, &e_hi)| {
            for (i_lo, slot) in out_block.iter_mut().enumerate() {
                *slot = fold_one_slot(eq_lo[i_lo] * e_hi, tables);
            }
        });
    out
}

// ---------------------------------------------------------------------------
// Sparse-tensor fast path.
//
// When the suffix `x_outer[2..]` has `k` coords exactly equal to `F256::ZERO`
// (as is the case for the hash-chain ẑ-opening, whose `x_inner_rest` is padded
// with trailing zeros), `build_eq` zeros out half the table per zero coord — so
// `1 − 2^{-k}` of the suffix tensor is zero and contributes nothing to
// `s_hat_v` (in `fold_1b_rows`) or `rs_eq_ind` (in `fold_b128_elems`). The
// sparse kernels touch only the `2^{-k}` support and produce byte-identical
// outputs to the dense kernels.
// ---------------------------------------------------------------------------

/// Minimum number of exactly-zero suffix coords for a claim to be routed
/// through the sparse kernels instead of the dense fold.
const SPARSE_ZERO_THRESHOLD: usize = 3;

/// Sparse representation of `build_eq(coords)` when `coords` contains exact
/// `F256::ZERO` entries: stores values at the compact (live) tensor positions
/// and a `live_positions` table that maps compact bit `j` → original coord
/// position. The scattered idx is computed on-the-fly via [`Self::scatter_idx`].
#[derive(Clone, Debug)]
pub struct SparseEqTensor {
    /// `build_eq(live_coords)` — length `2^live_positions.len()`.
    pub live_tensor: Vec<F256>,
    /// Original-coord positions of each live coord, ascending.
    pub live_positions: Vec<usize>,
}

impl SparseEqTensor {
    /// Compact-to-scattered index translation: deposit the live bits of `c`
    /// into the original-coord positions.
    #[inline(always)]
    pub fn scatter_idx(&self, c: usize) -> usize {
        let mut full = 0usize;
        for (j, &pos) in self.live_positions.iter().enumerate() {
            full |= ((c >> j) & 1) << pos;
        }
        full
    }

    /// Materialize the scattered `(idx, val)` pairs (test-oracle / external use).
    pub fn materialize(&self) -> Vec<(usize, F256)> {
        self.live_tensor
            .iter()
            .enumerate()
            .map(|(c, &v)| (self.scatter_idx(c), v))
            .collect()
    }

    /// Number of scattered entries.
    pub fn len(&self) -> usize {
        self.live_tensor.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live_tensor.is_empty()
    }
}

/// Build the sparse `build_eq(coords)` representation, skipping the zero-coord
/// halvings. `O(2^live_count)` time and memory.
pub fn build_eq_sparse(coords: &[F256]) -> SparseEqTensor {
    let live_positions: Vec<usize> = coords
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| if c == F256::ZERO { None } else { Some(i) })
        .collect();
    let live_coords: Vec<F256> = live_positions.iter().map(|&i| coords[i]).collect();
    let live_tensor = build_eq(&live_coords);
    SparseEqTensor {
        live_tensor,
        live_positions,
    }
}

/// Sparse counterpart of one column of [`fold_1b_rows_multi`]: scans only the
/// nonzero entries of the suffix tensor. Produces the same 256-entry `s_hat_v`
/// as `fold_1b_rows_naive(packed_witness, build_eq(coords))`.
pub fn fold_1b_rows_sparse(packed_witness: &[F256], eq: &SparseEqTensor) -> Vec<F256> {
    fold_1b_rows_sparse_scalar(packed_witness, eq)
}

/// Scalar bit-scan implementation of [`fold_1b_rows_sparse`].
fn fold_1b_rows_sparse_scalar(packed_witness: &[F256], eq: &SparseEqTensor) -> Vec<F256> {
    use rayon::prelude::*;
    let n = 1 << LOG_PACKING;
    let zero_acc = || vec![F256::ZERO; n];

    eq.live_tensor
        .par_iter()
        .enumerate()
        .fold(zero_acc, |mut acc, (c, &val)| {
            let idx = eq.scatter_idx(c);
            scatter_set_bits(&mut acc, packed_witness[idx], val);
            acc
        })
        .reduce(zero_acc, |mut a, b| {
            for r in 0..n {
                a[r] += b[r];
            }
            a
        })
}

/// Sparse counterpart of [`fold_b128_elems`] returning **sparse pairs** instead
/// of a dense vector — skips the O(L) zero-init / scatter entirely. Each pair
/// `(idx, value)` has the same per-element bit-scan over `eq_r_dprime` as the
/// dense kernel computed at that index.
pub fn fold_b128_elems_sparse_pairs(
    eq: &SparseEqTensor,
    eq_r_dprime: &[F256],
) -> Vec<(usize, F256)> {
    use rayon::prelude::*;
    assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);
    eq.live_tensor
        .par_iter()
        .enumerate()
        .map(|(c, &tensor_val)| {
            (
                eq.scatter_idx(c),
                fold_bits_against(tensor_val, eq_r_dprime),
            )
        })
        .collect()
}

/// Dense-output sparse fold — kept for tests/oracles. Returns a length-`len`
/// `Vec<F256>` that is zero outside the support.
pub fn fold_b128_elems_sparse(len: usize, eq: &SparseEqTensor, eq_r_dprime: &[F256]) -> Vec<F256> {
    let pairs = fold_b128_elems_sparse_pairs(eq, eq_r_dprime);
    let mut out = vec![F256::ZERO; len];
    for (idx, val) in pairs {
        out[idx] = val;
    }
    out
}

/// AB-claim `s_hat_v` specialization that **skips `fold_1b_rows` entirely**
/// when the upstream layer has already produced
/// `z_vec[i_inner] = ẑ(i_inner, x_outer)` (length `2^k_log`) — the pre-sumcheck
/// partial fold lincheck builds via `partial_fold_packed_z`.
///
/// For a PCS opening at point `(r_inner_skip, r_inner_rest, x_outer)` where
/// `x_outer` matches lincheck's, the AB-suffix tensor in `fold_1b_rows` factors
/// over the same axis decomposition that `z_vec` was built along:
///
/// ```text
/// s_hat_v[b] = Σ_{j ∈ {0,1}^(m−8)} eq(suffix, j) · bit_b(packed_witness[j])
///            = Σ_{k} eq(x_inner_rest_tail, k) · z_vec[b + 2^LOG_PACKING · k]
/// ```
///
/// (With LOG_PACKING = 8 the prefix now spans `K_SKIP + 2` coords, so the
/// inner-rest tail passed here is `x_inner_rest[2..]`.)
///
/// Output is **byte-identical** to
/// `fold_1b_rows_naive(packed_witness, build_eq(suffix))` for the AB claim.
///
/// # Panics
/// - if `z_vec.len() != 2^(LOG_PACKING + tail.len())`.
pub fn s_hat_v_from_z_vec(z_vec: &[F256], x_inner_rest_tail: &[F256]) -> Vec<F256> {
    use rayon::prelude::*;
    let n_packed = 1usize << LOG_PACKING; // 256
    let n_tail = 1usize << x_inner_rest_tail.len();
    assert_eq!(
        z_vec.len(),
        n_packed * n_tail,
        "z_vec length {} mismatches 2^(LOG_PACKING + tail.len()) = {}",
        z_vec.len(),
        n_packed * n_tail,
    );

    if x_inner_rest_tail.is_empty() {
        // Degenerate case (k_log == LOG_PACKING): the LOG_PACKING boundary
        // ate the only inner-rest coord — z_vec IS the per-prefix-bit answer.
        return z_vec.to_vec();
    }

    let eq_tail = build_eq_parallel(x_inner_rest_tail);

    eq_tail
        .par_iter()
        .enumerate()
        .fold(
            || vec![F256::ZERO; n_packed],
            |mut acc, (k, &w)| {
                let block = &z_vec[k * n_packed..(k + 1) * n_packed];
                for b in 0..n_packed {
                    acc[b] += w * block[b];
                }
                acc
            },
        )
        .reduce(
            || vec![F256::ZERO; n_packed],
            |mut a, b| {
                for i in 0..n_packed {
                    a[i] += b[i];
                }
                a
            },
        )
}

// ---------------------------------------------------------------------------
// Prover / verifier of the ring-switching reduction.
// ---------------------------------------------------------------------------

/// The prover message: the 256 slice-MLEs at the suffix point.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingSwitchProof {
    pub s_hat_v: Vec<F256>,
}

/// What both prover and verifier compute as a result of the reduction:
/// the transparent multilinear and the BaseFold sumcheck target.
#[derive(Clone, Debug)]
pub struct RingSwitchOutput {
    pub rs_eq_ind: Vec<F256>,
    pub sumcheck_claim: F256,
}

/// Per-claim output of [`prove_batched`]. Mirrors [`RingSwitchOutput`] but lets
/// the prover skip the dense `2^(m-8)` `rs_eq_ind` allocation for sparse claims.
#[derive(Clone, Debug)]
pub struct RingSwitchBatchOutput {
    /// For dense claims this is `γ_k · B_k` — γ is baked into the byte table
    /// during the fold, so pcs's combine just adds it without per-slot γ-mul.
    pub rs_eq_ind: RsEqInd,
    pub sumcheck_claim: F256,
}

/// Sparse-or-dense representation of `rs_eq_ind`. All variants here have γ_k
/// pre-multiplied in (see [`RingSwitchBatchOutput`]).
#[derive(Clone, Debug)]
pub enum RsEqInd {
    Dense(Vec<F256>),
    /// Deferred dense: the `γ_k·B_k` buffer is **not** materialized. Instead the
    /// fold ingredients ([`build_eq_split`] factors + the γ-baked byte table)
    /// are carried so pcs's combine can fold each slot on the fly.
    /// `value(j) = deferred_dense_value(eq_lo, eq_hi, table, log2(B), j)`,
    /// `B = eq_lo.len()`; byte-identical to `Dense(fold_b128_elems_split(..))`.
    DeferredDense {
        eq_lo: Vec<F256>,
        eq_hi: Vec<F256>,
        table: Vec<F256>,
    },
    Sparse {
        len: usize,
        entries: Vec<(usize, F256)>,
    },
}

impl RsEqInd {
    /// Logical length of the underlying vector.
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::DeferredDense { eq_lo, eq_hi, .. } => eq_lo.len() * eq_hi.len(),
            Self::Sparse { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Accumulate `gamma * self[j]` into `out[j]` for all `j`.
    pub fn add_scaled_into(&self, gamma: F256, out: &mut [F256]) {
        debug_assert_eq!(out.len(), self.len());
        match self {
            Self::Dense(v) => {
                for (o, &x) in out.iter_mut().zip(v.iter()) {
                    *o += gamma * x;
                }
            }
            Self::DeferredDense {
                eq_lo,
                eq_hi,
                table,
            } => {
                let log_b = eq_lo.len().trailing_zeros() as usize;
                for (j, o) in out.iter_mut().enumerate() {
                    *o += gamma * deferred_dense_value(eq_lo, eq_hi, table, log_b, j);
                }
            }
            Self::Sparse { entries, .. } => {
                for &(idx, val) in entries {
                    out[idx] += gamma * val;
                }
            }
        }
    }

    /// Materialize the dense view. O(L) regardless of variant; use sparingly.
    pub fn to_dense(&self) -> Vec<F256> {
        match self {
            Self::Dense(v) => v.clone(),
            Self::DeferredDense {
                eq_lo,
                eq_hi,
                table,
            } => {
                let log_b = eq_lo.len().trailing_zeros() as usize;
                let l = eq_lo.len() * eq_hi.len();
                (0..l)
                    .map(|j| deferred_dense_value(eq_lo, eq_hi, table, log_b, j))
                    .collect()
            }
            Self::Sparse { len, entries } => {
                let mut out = vec![F256::ZERO; *len];
                for &(idx, val) in entries {
                    out[idx] = val;
                }
                out
            }
        }
    }

    /// Consume into a dense `Vec<F256>`. Returns the inner vector directly when
    /// already `Dense` (no copy).
    pub fn into_dense(self) -> Vec<F256> {
        match self {
            Self::Dense(v) => v,
            Self::DeferredDense { .. } => self.to_dense(),
            Self::Sparse { len, entries } => {
                let mut out = vec![F256::ZERO; len];
                for (idx, val) in entries {
                    out[idx] = val;
                }
                out
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    ClaimMismatch,
}

/// Prover side of the ring-switching reduction.
///
/// Inputs:
/// - `packed_witness` (length `2^L`, L = m − 8), the F_{2^256}-packed witness.
/// - `x_outer` (length m − 6), the multilinear coords from the zerocheck.
/// - `challenger` for sampling row-batching `r''`.
///
/// Output: the proof message `s_hat_v` (256 F_{2^256} values to send) plus the
/// BaseFold inputs `(rs_eq_ind, sumcheck_claim)`.
pub fn prove<Ch: Challenger>(
    packed_witness: &[F256],
    x_outer: &[F256],
    challenger: &mut Ch,
) -> (RingSwitchProof, RingSwitchOutput) {
    assert!(
        x_outer.len() >= 2,
        "x_outer must contain at least 2 coords (the 2 prefix-bit factors past the skip)"
    );
    let l = packed_witness.len();
    // packed_witness.len() = 2^L where L = m - 8, and x_outer.len() = m - 6, so
    // packed_witness.len() = 2^(x_outer.len() - 2).
    assert_eq!(l, 1 << (x_outer.len() - 2));

    let trace = std::env::var("PCS_TRACE").is_ok();

    challenger.observe_label(b"flock-ring-switch-v0");

    // Suffix is x_outer[2..] (length m-8); the first two coords become the
    // 7th/8th-bit prefix factors.
    let suffix = &x_outer[2..];
    let t = std::time::Instant::now();
    let suffix_tensor = build_eq_parallel(suffix);
    if trace {
        eprintln!(
            "    [rs::prove] build_eq(suffix L={}): {:6.2} ms",
            suffix.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    debug_assert_eq!(suffix_tensor.len(), l);

    // Compute and send s_hat_v.
    let t = std::time::Instant::now();
    let s_hat_v = fold_1b_rows_naive(packed_witness, &suffix_tensor);
    if trace {
        eprintln!(
            "    [rs::prove] fold_1b_rows:          {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    challenger.observe_f256_slice(&s_hat_v);

    // Sample row-batching r''.
    let r_dprime = challenger.sample_f256_vec(LOG_PACKING);
    let eq_r_dprime = build_eq(&r_dprime);

    // Compute BaseFold target: T = ⟨transpose(s_hat_v), eq(r'')⟩.
    let s_hat_u = tensor_algebra_transpose(&s_hat_v);
    let sumcheck_claim = inner_product(&s_hat_u, &eq_r_dprime);

    // Compute transparent multilinear rs_eq_ind.
    let t = std::time::Instant::now();
    let rs_eq_ind = fold_b128_elems(&suffix_tensor, &eq_r_dprime);
    if trace {
        eprintln!(
            "    [rs::prove] fold_b128_elems:       {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    (
        RingSwitchProof { s_hat_v },
        RingSwitchOutput {
            rs_eq_ind,
            sumcheck_claim,
        },
    )
}

/// Batched prover: produce ring-switching proofs for `x_outers.len()` opening
/// points in one pass. Challenger interaction is byte-identical to calling
/// [`prove`] sequentially for each `x_outer`.
pub fn prove_batched<Ch: Challenger>(
    packed_witness: &[F256],
    x_outers: &[&[F256]],
    challenger: &mut Ch,
) -> (Vec<(RingSwitchProof, RingSwitchBatchOutput)>, Vec<F256>) {
    let m = LOG_PACKING + (packed_witness.len().trailing_zeros() as usize);
    prove_batched_padded(packed_witness, x_outers, &PaddingSpec::dense(m), challenger)
}

/// Padding-aware variant of [`prove_batched`].
///
/// Returns `(results, gammas_rs)` — γ_rs is sampled internally after all claims
/// are observed (Schwartz-Zippel-sound), and is **baked into each
/// `RingSwitchBatchOutput::rs_eq_ind`** so the pcs combine doesn't need a
/// per-slot γ-mul.
pub fn prove_batched_padded<Ch: Challenger>(
    packed_witness: &[F256],
    x_outers: &[&[F256]],
    padding: &PaddingSpec,
    challenger: &mut Ch,
) -> (Vec<(RingSwitchProof, RingSwitchBatchOutput)>, Vec<F256>) {
    prove_batched_padded_with_precomputed(packed_witness, x_outers, &[], padding, challenger)
}

/// Variant of [`prove_batched_padded`] that accepts an optional precomputed
/// `s_hat_v` per claim. When `precomputed_s_hat_v[i] = Some(v)` for claim `i`,
/// the prover skips that claim's `fold_1b_rows` work and uses `v` directly.
///
/// `precomputed_s_hat_v` must be `&[]` or have length `x_outers.len()`. Each
/// precomputed slice must be length `2^LOG_PACKING`.
pub fn prove_batched_padded_with_precomputed<Ch: Challenger>(
    packed_witness: &[F256],
    x_outers: &[&[F256]],
    precomputed_s_hat_v: &[Option<&[F256]>],
    padding: &PaddingSpec,
    challenger: &mut Ch,
) -> (Vec<(RingSwitchProof, RingSwitchBatchOutput)>, Vec<F256>) {
    assert!(!x_outers.is_empty());
    let trace = std::env::var("PCS_TRACE").is_ok();
    let n = x_outers.len();
    let l = packed_witness.len();
    for x in x_outers {
        assert!(x.len() >= 2);
        assert_eq!(l, 1 << (x.len() - 2));
    }
    assert!(
        precomputed_s_hat_v.is_empty() || precomputed_s_hat_v.len() == n,
        "precomputed_s_hat_v: must be empty or length {n}, got {}",
        precomputed_s_hat_v.len(),
    );
    let n_packed = 1usize << LOG_PACKING;
    for p in precomputed_s_hat_v.iter().flatten() {
        assert_eq!(
            p.len(),
            n_packed,
            "precomputed_s_hat_v entry must have length 2^LOG_PACKING"
        );
    }

    let has_precomputed =
        |orig: usize| -> bool { precomputed_s_hat_v.get(orig).copied().flatten().is_some() };

    // 1. Classify each claim. Claims whose suffix `x_outer[2..]` has at least
    //    `SPARSE_ZERO_THRESHOLD` exactly-zero coords skip the dense kernels.
    #[derive(Clone, Copy)]
    enum Kind {
        Dense(usize),
        Sparse(usize),
    }
    let mut kinds: Vec<Kind> = Vec::with_capacity(n);
    let mut dense_suffixes: Vec<&[F256]> = Vec::new();
    let mut sparse_suffixes: Vec<&[F256]> = Vec::new();
    let mut dense_to_orig: Vec<usize> = Vec::new();
    let mut sparse_to_orig: Vec<usize> = Vec::new();
    for (orig, x) in x_outers.iter().enumerate() {
        let suffix = &x[2..];
        let n_zeros = suffix.iter().filter(|&&c| c == F256::ZERO).count();
        if n_zeros >= SPARSE_ZERO_THRESHOLD {
            kinds.push(Kind::Sparse(sparse_suffixes.len()));
            sparse_to_orig.push(orig);
            sparse_suffixes.push(suffix);
        } else {
            kinds.push(Kind::Dense(dense_suffixes.len()));
            dense_to_orig.push(orig);
            dense_suffixes.push(suffix);
        }
    }

    // 2. Build suffix representations. Dense claims use the tensor-split
    //    factorization whenever `len` is a power of two ≥ 16; tiny test sizes
    //    fall back to the materialized tensor.
    let use_split = l.is_multiple_of(16);
    let t = std::time::Instant::now();
    let dense_splits: Vec<(Vec<F256>, Vec<F256>)> = if use_split {
        dense_suffixes
            .iter()
            .map(|s| build_eq_split(s, split_n_lo(s.len())))
            .collect()
    } else {
        Vec::new()
    };
    let dense_tensors: Vec<Vec<F256>> = if use_split {
        Vec::new()
    } else {
        dense_suffixes
            .iter()
            .map(|s| build_eq_parallel(s))
            .collect()
    };
    let sparse_supports: Vec<SparseEqTensor> =
        sparse_suffixes.iter().map(|s| build_eq_sparse(s)).collect();
    if trace {
        eprintln!(
            "    [rs::prove_batched] build_eq dense×{} ({}) + sparse×{}: {:6.2} ms",
            dense_suffixes.len(),
            if use_split { "split" } else { "full" },
            sparse_supports.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 3. fold_1b_rows. Precomputed claims skip it.
    let dense_needs_fold: Vec<usize> = (0..dense_suffixes.len())
        .filter(|&d| !has_precomputed(dense_to_orig[d]))
        .collect();
    let sparse_needs_fold: Vec<usize> = (0..sparse_suffixes.len())
        .filter(|&s| !has_precomputed(sparse_to_orig[s]))
        .collect();
    let t = std::time::Instant::now();
    let mut dense_s_hat_v: Vec<Vec<F256>> = vec![Vec::new(); dense_suffixes.len()];
    let mut sparse_s_hat_v: Vec<Vec<F256>> = vec![Vec::new(); sparse_suffixes.len()];
    // Fill precomputed slots first.
    for d in 0..dense_suffixes.len() {
        if let Some(p) = precomputed_s_hat_v.get(dense_to_orig[d]).copied().flatten() {
            dense_s_hat_v[d] = p.to_vec();
        }
    }
    for s in 0..sparse_suffixes.len() {
        if let Some(p) = precomputed_s_hat_v
            .get(sparse_to_orig[s])
            .copied()
            .flatten()
        {
            sparse_s_hat_v[s] = p.to_vec();
        }
    }
    // Run the kernel only on claims that genuinely need fold_1b_rows.
    if use_split {
        match dense_needs_fold.len() {
            0 => {}
            2 => {
                let d0 = dense_needs_fold[0];
                let d1 = dense_needs_fold[1];
                let (lo0, hi0) = (dense_splits[d0].0.as_slice(), dense_splits[d0].1.as_slice());
                let (lo1, hi1) = (dense_splits[d1].0.as_slice(), dense_splits[d1].1.as_slice());
                let (a, b) = fold_1b_rows_split_2way(packed_witness, lo0, hi0, lo1, hi1, padding);
                dense_s_hat_v[d0] = a;
                dense_s_hat_v[d1] = b;
            }
            _ => {
                for &d in &dense_needs_fold {
                    let (eq_lo, eq_hi) = (&dense_splits[d].0, &dense_splits[d].1);
                    dense_s_hat_v[d] = fold_1b_rows_split(packed_witness, eq_lo, eq_hi, padding);
                }
            }
        }
    } else if !dense_needs_fold.is_empty() {
        let dense_refs: Vec<&[F256]> = dense_needs_fold
            .iter()
            .map(|&d| dense_tensors[d].as_slice())
            .collect();
        let out = fold_1b_rows_multi_padded(packed_witness, &dense_refs, padding);
        for (i, &d) in dense_needs_fold.iter().enumerate() {
            dense_s_hat_v[d] = out[i].clone();
        }
    }
    for &s in &sparse_needs_fold {
        sparse_s_hat_v[s] = fold_1b_rows_sparse(packed_witness, &sparse_supports[s]);
    }
    if trace {
        eprintln!(
            "    [rs::prove_batched] fold_1b_rows dense(k={})+sparse(k={}): {:6.2} ms",
            dense_s_hat_v.len(),
            sparse_s_hat_v.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 4. Per-opening tail.
    let t = std::time::Instant::now();

    struct ClaimWork {
        s_hat_v: Vec<F256>,
        sumcheck_claim: F256,
        eq_r_dprime: Vec<F256>,
    }
    let mut work: Vec<ClaimWork> = Vec::with_capacity(n);
    for i in 0..n {
        challenger.observe_label(b"flock-ring-switch-v0");
        let s_hat_v: Vec<F256> = match kinds[i] {
            Kind::Dense(d) => dense_s_hat_v[d].clone(),
            Kind::Sparse(s) => sparse_s_hat_v[s].clone(),
        };
        challenger.observe_f256_slice(&s_hat_v);
        let r_dprime = challenger.sample_f256_vec(LOG_PACKING);
        let eq_r_dprime = build_eq(&r_dprime);

        let s_hat_u = tensor_algebra_transpose(&s_hat_v);
        let sumcheck_claim = inner_product(&s_hat_u, &eq_r_dprime);

        work.push(ClaimWork {
            s_hat_v,
            sumcheck_claim,
            eq_r_dprime,
        });
    }

    // γ_rs sampled after all RS observations — sound. Each γ_rs[k] is baked into
    // eq_r_dprime[k] before building the Φ byte table, so the fold output is
    // γ_k · B_k directly.
    let gammas_rs: Vec<F256> = (0..n).map(|_| challenger.sample_f256()).collect();

    let results: Vec<(RingSwitchProof, RingSwitchBatchOutput)> = work
        .into_iter()
        .zip(gammas_rs.iter())
        .enumerate()
        .map(|(i, (w, &g))| {
            let scaled_eq_r_dprime: Vec<F256> = w.eq_r_dprime.iter().map(|x| g * *x).collect();
            let rs_eq_ind = match kinds[i] {
                Kind::Dense(d) => {
                    if use_split {
                        let (eq_lo, eq_hi) = &dense_splits[d];
                        RsEqInd::DeferredDense {
                            eq_lo: eq_lo.clone(),
                            eq_hi: eq_hi.clone(),
                            table: build_fold_byte_table(&scaled_eq_r_dprime),
                        }
                    } else {
                        RsEqInd::Dense(fold_b128_elems(&dense_tensors[d], &scaled_eq_r_dprime))
                    }
                }
                Kind::Sparse(s) => RsEqInd::Sparse {
                    len: l,
                    entries: fold_b128_elems_sparse_pairs(&sparse_supports[s], &scaled_eq_r_dprime),
                },
            };
            (
                RingSwitchProof { s_hat_v: w.s_hat_v },
                RingSwitchBatchOutput {
                    rs_eq_ind,
                    sumcheck_claim: w.sumcheck_claim,
                },
            )
        })
        .collect();

    if trace {
        eprintln!(
            "    [rs::prove_batched] per-opening tail ×{}: {:6.2} ms",
            n,
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    (results, gammas_rs)
}

/// Verifier side of the ring-switching reduction.
///
/// Output: the matching BaseFold inputs `(rs_eq_ind, sumcheck_claim)`, or a
/// `ClaimMismatch` error if `weights · s_hat_v ≠ claim`.
pub fn verify<Ch: Challenger>(
    claim: F256,
    z_skip: F256,
    x_outer: &[F256],
    proof: &RingSwitchProof,
    challenger: &mut Ch,
) -> Result<RingSwitchOutput, VerifyError> {
    assert!(x_outer.len() >= 2);
    let l = 1usize << (x_outer.len() - 2);
    assert_eq!(proof.s_hat_v.len(), 1 << LOG_PACKING);

    challenger.observe_label(b"flock-ring-switch-v0");

    // Verifier observes s_hat_v.
    challenger.observe_f256_slice(&proof.s_hat_v);

    // Check the claim against ν_φ8 ⊗ eq ⊗ eq weights.
    let weights = build_claim_weights(z_skip, x_outer[0], x_outer[1]);
    if claim_check(&weights, &proof.s_hat_v) != claim {
        return Err(VerifyError::ClaimMismatch);
    }

    // Sample r''.
    let r_dprime = challenger.sample_f256_vec(LOG_PACKING);
    let eq_r_dprime = build_eq(&r_dprime);

    // Compute BaseFold target.
    let s_hat_u = tensor_algebra_transpose(&proof.s_hat_v);
    let sumcheck_claim = inner_product(&s_hat_u, &eq_r_dprime);

    // Compute rs_eq_ind (verifier reconstructs it from x_outer[2..] and r'').
    let suffix = &x_outer[2..];
    let suffix_tensor = build_eq(suffix);
    debug_assert_eq!(suffix_tensor.len(), l);
    let rs_eq_ind = fold_b128_elems(&suffix_tensor, &eq_r_dprime);

    Ok(RingSwitchOutput {
        rs_eq_ind,
        sumcheck_claim,
    })
}

/// Verifier-side output of [`verify_succinct`]: everything needed to drive the
/// BaseFold consistency check, *without* materializing the dense `rs_eq_ind`.
#[derive(Clone, Debug)]
pub struct RingSwitchVerifierOutput {
    pub sumcheck_claim: F256,
    /// `eq` tensor of length `2^LOG_PACKING = 256` derived from the verifier's
    /// sampled `r''`. Used by [`eval_rs_eq`] at the BaseFold final point.
    pub eq_r_dprime: Vec<F256>,
}

/// Polylog-cost ring-switching verifier.
///
/// Same FS interface as [`verify`] but **does not** build the dense
/// `rs_eq_ind`. Pair with [`eval_rs_eq`] at the BaseFold final point to
/// evaluate `MLE(rs_eq_ind)(challenges)` in `O((m − 8) · 256²)` field ops.
pub fn verify_succinct<Ch: Challenger>(
    claim: F256,
    z_skip: F256,
    x_outer: &[F256],
    proof: &RingSwitchProof,
    challenger: &mut Ch,
) -> Result<RingSwitchVerifierOutput, VerifyError> {
    assert!(x_outer.len() >= 2);
    assert_eq!(proof.s_hat_v.len(), 1 << LOG_PACKING);

    challenger.observe_label(b"flock-ring-switch-v0");
    challenger.observe_f256_slice(&proof.s_hat_v);

    let weights = build_claim_weights(z_skip, x_outer[0], x_outer[1]);
    if claim_check(&weights, &proof.s_hat_v) != claim {
        return Err(VerifyError::ClaimMismatch);
    }

    let r_dprime = challenger.sample_f256_vec(LOG_PACKING);
    let eq_r_dprime = build_eq(&r_dprime);

    let s_hat_u = tensor_algebra_transpose(&proof.s_hat_v);
    let sumcheck_claim = inner_product(&s_hat_u, &eq_r_dprime);

    Ok(RingSwitchVerifierOutput {
        sumcheck_claim,
        eq_r_dprime,
    })
}

/// Polylog-cost evaluation of `MLE(rs_eq_ind)(query)` at the BaseFold final
/// challenge point, following [DP24] §1.3 Figure 3.
///
/// Costs `O(|z_vals| · 2^{2·LOG_PACKING}) = O(|z_vals| · 65536)` field
/// operations: a length-256 `TensorAlgebra` element is iteratively updated by
/// `scale_vertical` / `scale_horizontal` over `|z_vals|` iterations, then folded
/// against `eq_r_dprime` (length 256).
///
/// ## Arguments
///
/// * `z_vals` — the suffix-side coords, i.e. `x_outer[2..]` from
///   [`verify_succinct`]. Length `ℓ' = m − 8`.
/// * `query` — the BaseFold sumcheck final challenges, length `ℓ'`.
/// * `eq_r_dprime` — the `eq` tensor over the sampled `r''`, length 256.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
pub fn eval_rs_eq(z_vals: &[F256], query: &[F256], eq_r_dprime: &[F256]) -> F256 {
    use crate::pcs::tensor_algebra::TensorAlgebra;

    assert_eq!(
        z_vals.len(),
        query.len(),
        "eval_rs_eq: z_vals and query must have equal length"
    );
    assert_eq!(
        eq_r_dprime.len(),
        1 << LOG_PACKING,
        "eval_rs_eq: eq_r_dprime length must be 256"
    );

    let mut eval = TensorAlgebra::from_vertical(F256::ONE);
    for (&z_i, &q_i) in z_vals.iter().zip(query.iter()) {
        // In characteristic 2: eq(z, q) = 1 + z + q + 2·z·q = 1 + z + q.
        let vert_scaled = eval.clone().scale_vertical(z_i);
        let hztl_scaled = eval.clone().scale_horizontal(q_i);
        eval += &vert_scaled;
        eval += &hztl_scaled;
    }
    eval.fold_vertical(eq_r_dprime)
}

/// **Prefix-only** variant of [`eval_rs_eq`]: walks `prefix_len` of the
/// (z_vals, query) pairs and returns the partially-evolved `TensorAlgebra`.
pub fn eval_rs_eq_prefix(
    z_vals: &[F256],
    query_prefix: &[F256],
) -> crate::pcs::tensor_algebra::TensorAlgebra {
    use crate::pcs::tensor_algebra::TensorAlgebra;
    assert!(query_prefix.len() <= z_vals.len());
    let mut eval = TensorAlgebra::from_vertical(F256::ONE);
    for (&z_i, &q_i) in z_vals.iter().zip(query_prefix.iter()) {
        let vert_scaled = eval.clone().scale_vertical(z_i);
        let hztl_scaled = eval.clone().scale_horizontal(q_i);
        eval += &vert_scaled;
        eval += &hztl_scaled;
    }
    eval
}

/// Finish [`eval_rs_eq`] given a precomputed prefix tensor + the remaining
/// (z, query) suffix.
pub fn eval_rs_eq_finish_from_prefix(
    prefix: &crate::pcs::tensor_algebra::TensorAlgebra,
    z_vals_suffix: &[F256],
    query_suffix: &[F256],
    eq_r_dprime: &[F256],
) -> F256 {
    assert_eq!(z_vals_suffix.len(), query_suffix.len());
    assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);
    let mut eval = prefix.clone();
    for (&z_i, &q_i) in z_vals_suffix.iter().zip(query_suffix.iter()) {
        let vert_scaled = eval.clone().scale_vertical(z_i);
        let hztl_scaled = eval.clone().scale_horizontal(q_i);
        eval += &vert_scaled;
        eval += &hztl_scaled;
    }
    eval.fold_vertical(eq_r_dprime)
}

/// Specialized variant of [`eval_rs_eq_finish_from_prefix`] for the case where
/// `query_suffix` is known to be **binary** (each coord is `F256::ZERO` or
/// `F256::ONE`). When `q_i ∈ {0, 1}`, the recurrence collapses (in char 2) to a
/// single in-place `scale_vertical`:
/// - `q_i = 0`: `new_eval = (1 + z_i) · eval`
/// - `q_i = 1`: `new_eval = z_i · eval`
///
/// `y_bits` encodes the suffix as a bitmask: bit `j` is the j-th suffix coord.
pub fn eval_rs_eq_finish_from_prefix_binary_q(
    prefix: &crate::pcs::tensor_algebra::TensorAlgebra,
    z_vals_suffix: &[F256],
    y_bits: u32,
    eq_r_dprime: &[F256],
) -> F256 {
    assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);
    debug_assert!(
        z_vals_suffix.len() <= 32,
        "y_bits is u32; suffix > 32 not supported"
    );
    let mut eval = prefix.clone();
    for (j, &z_i) in z_vals_suffix.iter().enumerate() {
        let scalar = if (y_bits >> j) & 1 == 1 {
            z_i
        } else {
            F256::ONE + z_i
        };
        for e in eval.elems.iter_mut() {
            *e *= scalar;
        }
    }
    eval.fold_vertical(eq_r_dprime)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::pack::pack_witness;
    use crate::zerocheck::univariate_skip::build_eq;

    /// Binary-query specialization matches the general path bit-for-bit.
    #[test]
    fn eval_rs_eq_finish_binary_q_matches_general() {
        use crate::challenger::Challenger;
        let mut rng = crate::challenger::RandomChallenger::new(0x_B17_0BBE);
        let log_n = 20usize;
        let prefix_len = 15usize;
        let suffix_len = log_n - prefix_len; // 5
        let z_vals: Vec<F256> = (0..log_n).map(|_| rng.sample_f256()).collect();
        let query_prefix: Vec<F256> = (0..prefix_len).map(|_| rng.sample_f256()).collect();
        let eq_r_dprime: Vec<F256> = (0..(1 << LOG_PACKING)).map(|_| rng.sample_f256()).collect();
        let prefix = eval_rs_eq_prefix(&z_vals[..prefix_len], &query_prefix);

        for y in 0..(1usize << suffix_len) {
            let query_suffix: Vec<F256> = (0..suffix_len)
                .map(|j| {
                    if (y >> j) & 1 == 1 {
                        F256::ONE
                    } else {
                        F256::ZERO
                    }
                })
                .collect();
            let general = eval_rs_eq_finish_from_prefix(
                &prefix,
                &z_vals[prefix_len..],
                &query_suffix,
                &eq_r_dprime,
            );
            let binary = eval_rs_eq_finish_from_prefix_binary_q(
                &prefix,
                &z_vals[prefix_len..],
                y as u32,
                &eq_r_dprime,
            );
            assert_eq!(general, binary, "y={y} mismatch");
        }
    }

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
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn f256(&mut self) -> F256 {
            F256 {
                c0: self.f128(),
                c1: self.f128(),
            }
        }
    }

    /// Reference: directly compute ẑ_skip(z_skip, x_outer) for a Boolean witness `z`.
    fn zhat_skip_reference(z: &[bool], m: usize, z_skip: F256, x_outer: &[F256]) -> F256 {
        const K_SKIP: usize = 6;
        let ell = 1usize << K_SKIP;
        assert_eq!(z.len(), 1 << m);
        assert_eq!(x_outer.len(), m - K_SKIP);

        let lambda = lagrange_weights_naive(K_SKIP, z_skip); // 64 weights
        let eq_outer = build_eq(x_outer); // 2^(m-6) values

        let mut acc = F256::ZERO;
        for i_outer in 0..(1usize << (m - K_SKIP)) {
            let base = i_outer * ell;
            let mut inner = F256::ZERO;
            for i_skip in 0..ell {
                if z[base + i_skip] {
                    inner += lambda[i_skip];
                }
            }
            acc += eq_outer[i_outer] * inner;
        }
        acc
    }

    /// The key identity: with weights and s_hat_v constructed from the right
    /// places, the claim-check yields `ẑ_skip(z_skip, x_outer)`.
    #[test]
    fn claim_check_recovers_zhat_skip() {
        let mut rng = Rng::new(0xAA7);
        // m ≥ LOG_PACKING = 8 so x_outer has ≥ 2 coords (x_outer[0], x_outer[1]).
        for &m in &[8usize, 9, 10] {
            let z = rng.bits(1 << m);
            let z_skip = rng.f256();
            let x_outer: Vec<F256> = (0..(m - 6)).map(|_| rng.f256()).collect();

            let expected = zhat_skip_reference(&z, m, z_skip, &x_outer);

            let packed = pack_witness(&z, m);
            let suffix_tensor = build_eq(&x_outer[2..]); // length 2^(m-8)
            assert_eq!(packed.len(), suffix_tensor.len());
            let s_hat_v = fold_1b_rows_naive(&packed, &suffix_tensor);

            let weights = build_claim_weights(z_skip, x_outer[0], x_outer[1]);
            let got = claim_check(&weights, &s_hat_v);

            assert_eq!(got, expected, "claim-check mismatch at m={m}");
        }
    }

    #[test]
    fn weights_have_correct_length() {
        let w = build_claim_weights(
            F256::from_f128(F128 { lo: 1, hi: 0 }),
            F256::from_f128(F128 { lo: 2, hi: 0 }),
            F256::from_f128(F128 { lo: 3, hi: 0 }),
        );
        assert_eq!(w.len(), 256);
    }

    /// Round-trip: prove() and verify() with the same challenger seed must
    /// produce identical (rs_eq_ind, sumcheck_claim).
    #[test]
    fn prove_verify_roundtrip() {
        use crate::challenger::FsChallenger;
        let mut rng = Rng::new(0xBEEF);
        for &m in &[8usize, 9, 10, 11] {
            let z = rng.bits(1 << m);
            let z_skip = rng.f256();
            let x_outer: Vec<F256> = (0..(m - 6)).map(|_| rng.f256()).collect();

            let claim = zhat_skip_reference(&z, m, z_skip, &x_outer);

            let packed = pack_witness(&z, m);

            let mut ch_p = FsChallenger::new(b"flock-test-v0");
            let (proof, out_p) = prove(&packed, &x_outer, &mut ch_p);

            let mut ch_v = FsChallenger::new(b"flock-test-v0");
            let out_v = verify(claim, z_skip, &x_outer, &proof, &mut ch_v)
                .unwrap_or_else(|e| panic!("verify rejected honest at m={m}: {e:?}"));

            assert_eq!(
                out_p.sumcheck_claim, out_v.sumcheck_claim,
                "sumcheck_claim mismatch at m={m}"
            );
            assert_eq!(
                out_p.rs_eq_ind, out_v.rs_eq_ind,
                "rs_eq_ind mismatch at m={m}"
            );
        }
    }

    /// DP24 identity: `⟨packed_witness, rs_eq_ind⟩ = sumcheck_claim`.
    #[test]
    fn dp24_identity_holds() {
        use crate::challenger::FsChallenger;
        let mut rng = Rng::new(0xABCD);
        for &m in &[8usize, 9, 10, 11] {
            let z = rng.bits(1 << m);
            let x_outer: Vec<F256> = (0..(m - 6)).map(|_| rng.f256()).collect();

            let packed = pack_witness(&z, m);
            let mut ch = FsChallenger::new(b"flock-test-v0");
            let (_proof, out) = prove(&packed, &x_outer, &mut ch);

            let lhs = inner_product(&packed, &out.rs_eq_ind);
            assert_eq!(lhs, out.sumcheck_claim, "DP24 identity fails at m={m}");
        }
    }

    /// Mutation rejection: flipping one bit of the proof must cause verify to reject.
    #[test]
    fn verify_rejects_mutated_proof() {
        use crate::challenger::FsChallenger;
        let m = 10usize;
        let mut rng = Rng::new(0x99);
        let z = rng.bits(1 << m);
        let z_skip = rng.f256();
        let x_outer: Vec<F256> = (0..(m - 6)).map(|_| rng.f256()).collect();
        let claim = zhat_skip_reference(&z, m, z_skip, &x_outer);
        let packed = pack_witness(&z, m);

        let mut ch_p = FsChallenger::new(b"flock-test-v0");
        let (mut proof, _) = prove(&packed, &x_outer, &mut ch_p);
        proof.s_hat_v[0].c0.lo ^= 1;

        let mut ch_v = FsChallenger::new(b"flock-test-v0");
        let res = verify(claim, z_skip, &x_outer, &proof, &mut ch_v);
        assert!(matches!(res, Err(VerifyError::ClaimMismatch)));
    }

    /// Tensor-algebra transpose is involutive.
    #[test]
    fn transpose_is_involution() {
        let mut rng = Rng::new(0xDEAD);
        let s_hat_v: Vec<F256> = (0..256).map(|_| rng.f256()).collect();
        let twice = tensor_algebra_transpose(&tensor_algebra_transpose(&s_hat_v));
        assert_eq!(s_hat_v, twice);
    }

    #[test]
    fn prove_batched_matches_sequential() {
        use crate::challenger::FsChallenger;
        let mut rng = Rng::new(0x1234_5678);
        for &m in &[8usize, 9, 10, 11] {
            let z = rng.bits(1 << m);
            let x_a: Vec<F256> = (0..(m - 6)).map(|_| rng.f256()).collect();
            let x_b: Vec<F256> = (0..(m - 6)).map(|_| rng.f256()).collect();
            let packed = pack_witness(&z, m);

            let mut ch_seq = FsChallenger::new(b"flock-test-v0");
            let (p_a, o_a) = prove(&packed, &x_a, &mut ch_seq);
            let (p_b, o_b) = prove(&packed, &x_b, &mut ch_seq);

            // After γ-baking the batched transcript diverges from sequential
            // `prove`. s_hat_v + sumcheck_claim still match; rs_eq_ind has γ
            // baked in so it differs from sequential.
            let mut ch_batch = FsChallenger::new(b"flock-test-v0");
            let (results, _gammas_rs) = prove_batched(&packed, &[&x_a, &x_b], &mut ch_batch);

            assert_eq!(results[0].0, p_a, "s_hat_v[0] mismatch at m={m}");
            assert_eq!(results[1].0, p_b, "s_hat_v[1] mismatch at m={m}");
            assert_eq!(results[0].1.sumcheck_claim, o_a.sumcheck_claim);
            assert_eq!(results[1].1.sumcheck_claim, o_b.sumcheck_claim);
            assert_eq!(results[0].1.rs_eq_ind.len(), o_a.rs_eq_ind.len());
            assert_eq!(results[1].1.rs_eq_ind.len(), o_b.rs_eq_ind.len());
        }
    }

    /// The 2-way fold (portable wrapper) matches the naive bit-scan.
    #[test]
    fn mfr_fold_matches_scalar_bit_scan() {
        let mut rng = Rng::new(0xBEEF_D00D);
        for &m in &[9usize, 11, 13, 14] {
            let l = m - LOG_PACKING;
            let pw_len = 1usize << l;
            let pw: Vec<F256> = (0..pw_len).map(|_| rng.f256()).collect();
            let suffix0: Vec<F256> = (0..l).map(|_| rng.f256()).collect();
            let suffix1: Vec<F256> = (0..l).map(|_| rng.f256()).collect();
            let tensor0 = build_eq(&suffix0);
            let tensor1 = build_eq(&suffix1);

            let s0_ref = fold_1b_rows_naive(&pw, &tensor0);
            let s1_ref = fold_1b_rows_naive(&pw, &tensor1);

            let (s0_mfr, s1_mfr) = fold_1b_rows_2way_mfr(&pw, &tensor0, &tensor1);

            assert_eq!(s0_mfr, s0_ref, "s_hat_v0 mismatch at m={m}");
            assert_eq!(s1_mfr, s1_ref, "s_hat_v1 mismatch at m={m}");
        }
    }

    /// The 8-wide (portable wrapper) folds match the naive bit-scan.
    #[test]
    fn mfr_fold_8wide_matches_scalar() {
        let mut rng = Rng::new(0x8888_1357);
        for &m in &[10usize, 12, 13, 16] {
            let l = m - LOG_PACKING;
            let pw_len = 1usize << l;
            let pw: Vec<F256> = (0..pw_len).map(|_| rng.f256()).collect();
            let suffix0: Vec<F256> = (0..l).map(|_| rng.f256()).collect();
            let suffix1: Vec<F256> = (0..l).map(|_| rng.f256()).collect();
            let t0 = build_eq(&suffix0);
            let t1 = build_eq(&suffix1);

            let s0_ref = fold_1b_rows_naive(&pw, &t0);
            let s1_ref = fold_1b_rows_naive(&pw, &t1);
            assert_eq!(
                fold_1b_rows_1way_mfr_8wide_k4(&pw, &t0),
                s0_ref,
                "1-way 8wide m={m}"
            );
            let (s0, s1) = fold_1b_rows_2way_mfr_8wide(&pw, &t0, &t1);
            assert_eq!(s0, s0_ref, "2-way 8wide s0 m={m}");
            assert_eq!(s1, s1_ref, "2-way 8wide s1 m={m}");
        }
    }

    /// The padded folds are byte-identical to the dense folds on honestly
    /// zero-padded witnesses (the portable F256 path ignores padding but is
    /// dense-correct).
    #[test]
    fn fold_1b_padded_matches_dense() {
        let cases: &[(usize, usize, usize)] =
            &[(17, 14, 15_409), (18, 15, 31_401), (19, 16, 42_560)];
        for &(m, k_log, useful_bits) in cases {
            let mut rng = Rng::new(0xCAFE_FACE_u64.wrapping_add((k_log * 31 + m) as u64));
            let total_bits = 1usize << m;
            let block_size = 1usize << k_log;
            let n_blocks = 1usize << (m - k_log);

            let mut z = rng.bits(total_bits);
            for blk in 0..n_blocks {
                for j in useful_bits..block_size {
                    z[blk * block_size + j] = false;
                }
            }
            let packed = pack_witness(&z, m);

            let len = packed.len();
            let t0: Vec<F256> = (0..len).map(|_| rng.f256()).collect();
            let t1: Vec<F256> = (0..len).map(|_| rng.f256()).collect();
            let padding = PaddingSpec {
                k_log,
                useful_bits_per_block: useful_bits,
            };

            let dense8 = fold_1b_rows_2way_mfr_8wide(&packed, &t0, &t1);
            let padded8 = fold_1b_rows_2way_mfr_8wide_padded(&packed, &t0, &t1, &padding);
            assert_eq!(dense8, padded8, "8-wide mismatch: m={m}, k_log={k_log}");

            let dense4 = fold_1b_rows_2way_mfr(&packed, &t0, &t1);
            let padded4 = fold_1b_rows_2way_mfr_padded(&packed, &t0, &t1, &padding);
            assert_eq!(dense4, padded4, "4-wide mismatch: m={m}, k_log={k_log}");
        }
    }

    /// `build_eq_split` factors `build_eq` exactly.
    #[test]
    fn build_eq_split_reconstructs_full() {
        let mut rng = Rng::new(0x9911);
        for &l in &[4usize, 7, 10] {
            let r: Vec<F256> = (0..l).map(|_| rng.f256()).collect();
            let full = build_eq(&r);
            for n_lo in 0..=l {
                let (eq_lo, eq_hi) = build_eq_split(&r, n_lo);
                assert_eq!(eq_lo.len(), 1 << n_lo);
                assert_eq!(eq_hi.len(), 1 << (l - n_lo));
                let mask = (1usize << n_lo) - 1;
                for (i, &f) in full.iter().enumerate() {
                    let recon = eq_lo[i & mask] * eq_hi[i >> n_lo];
                    assert_eq!(recon, f, "reconstruct mismatch l={l} n_lo={n_lo} i={i}");
                }
            }
        }
    }

    /// `fold_1b_rows_split` is byte-identical to the dense fold for every split.
    #[test]
    fn fold_1b_rows_split_matches_16wide() {
        let cases: &[(usize, usize, usize)] =
            &[(17, 14, 15_409), (18, 15, 31_401), (19, 16, 42_560)];
        for &(m, k_log, useful_bits) in cases {
            let l = m - LOG_PACKING;
            let len = 1usize << l;
            let mut rng = Rng::new(0x5757_u64.wrapping_add((m * 131 + k_log) as u64));
            let w: Vec<F256> = (0..len).map(|_| rng.f256()).collect();
            let r: Vec<F256> = (0..l).map(|_| rng.f256()).collect();
            let full_eq = build_eq(&r);
            let padding = PaddingSpec {
                k_log,
                useful_bits_per_block: useful_bits,
            };

            let reference = fold_1b_rows_1way_mfr_16wide_padded(&w, &full_eq, &padding);
            for n_lo in 4..=l {
                let (eq_lo, eq_hi) = build_eq_split(&r, n_lo);
                let got = fold_1b_rows_split(&w, &eq_lo, &eq_hi, &padding);
                assert_eq!(
                    got, reference,
                    "fold_1b_rows_split mismatch: m={m}, k_log={k_log}, n_lo={n_lo}"
                );
            }
            let (eq_lo, eq_hi) = build_eq_split(&r, split_n_lo(l));
            assert_eq!(
                fold_1b_rows_split(&w, &eq_lo, &eq_hi, &padding),
                reference,
                "fold_1b_rows_split mismatch at split_n_lo: m={m}"
            );
        }
    }

    /// `fold_1b_rows_split_2way` matches two separate `fold_1b_rows_split` calls.
    #[test]
    fn fold_1b_rows_split_2way_matches_per_claim() {
        let cases: &[(usize, usize, usize)] =
            &[(17, 14, 15_409), (18, 15, 31_401), (19, 16, 42_560)];
        for &(m, k_log, useful_bits) in cases {
            let l = m - LOG_PACKING;
            let len = 1usize << l;
            let mut rng = Rng::new(0xBEEF_u64.wrapping_add((m * 131 + k_log) as u64));
            let w: Vec<F256> = (0..len).map(|_| rng.f256()).collect();
            let padding = PaddingSpec {
                k_log,
                useful_bits_per_block: useful_bits,
            };
            let n_lo = split_n_lo(l);
            let r0: Vec<F256> = (0..l).map(|_| rng.f256()).collect();
            let r1: Vec<F256> = (0..l).map(|_| rng.f256()).collect();
            let (lo0, hi0) = build_eq_split(&r0, n_lo);
            let (lo1, hi1) = build_eq_split(&r1, n_lo);
            let (got0, got1) = fold_1b_rows_split_2way(&w, &lo0, &hi0, &lo1, &hi1, &padding);
            let want0 = fold_1b_rows_split(&w, &lo0, &hi0, &padding);
            let want1 = fold_1b_rows_split(&w, &lo1, &hi1, &padding);
            assert_eq!(got0, want0, "split_2way mismatch (claim 0) m={m}");
            assert_eq!(got1, want1, "split_2way mismatch (claim 1) m={m}");
        }
    }

    /// `fold_b128_elems_split` matches the materialized `fold_b128_elems`.
    #[test]
    fn fold_b128_elems_split_matches_dense() {
        let mut rng = Rng::new(0xB0B0);
        for &l in &[4usize, 8, 10] {
            let r: Vec<F256> = (0..l).map(|_| rng.f256()).collect();
            let full_eq = build_eq(&r);
            let eq_r: Vec<F256> = (0..256).map(|_| rng.f256()).collect();
            let reference = fold_b128_elems(&full_eq, &eq_r);
            for n_lo in 4..=l {
                let (eq_lo, eq_hi) = build_eq_split(&r, n_lo);
                let got = fold_b128_elems_split(&eq_lo, &eq_hi, &eq_r);
                assert_eq!(
                    got, reference,
                    "fold_b128_elems_split mismatch l={l} n_lo={n_lo}"
                );
            }
        }
    }

    /// AB-claim s_hat_v via `s_hat_v_from_z_vec` (reusing lincheck's
    /// pre-sumcheck partial fold of `z` at `x_outer`) is byte-identical to the
    /// general-purpose `fold_1b_rows` over the materialized suffix tensor.
    #[test]
    fn s_hat_v_from_z_vec_matches_fold_1b_rows_ab() {
        use crate::lincheck::{pack_z_lincheck, partial_fold_packed_z};
        const K_SKIP: usize = 6;
        // (m, k_log) — with LOG_PACKING = 8 the prefix spans K_SKIP + 2 coords,
        // so x_inner_rest[0..2] become ring-switch's prefix factors and the tail
        // fed here is x_inner_rest[2..]. n_log = m − k_log must be ≥ 3.
        let cases: &[(usize, usize)] = &[(13, 10), (15, 11), (17, 13)];
        for &(m, k_log) in cases {
            assert!(k_log >= LOG_PACKING);
            assert!(k_log >= K_SKIP);
            let n_log = m - k_log;
            assert!(n_log >= 3);
            let mut rng = Rng::new(0xCAFE_u64.wrapping_add((m * 131 + k_log) as u64));

            let z = rng.bits(1 << m);
            let packed = pack_witness(&z, m);
            let z_packed_lincheck = pack_z_lincheck(&z, m, k_log);

            // x_inner_rest has k_log − K_SKIP coords; x_outer has n_log coords.
            let x_inner_rest: Vec<F256> = (0..(k_log - K_SKIP)).map(|_| rng.f256()).collect();
            let x_outer: Vec<F256> = (0..n_log).map(|_| rng.f256()).collect();

            // Reference: ring-switch's fold over the materialized suffix tensor.
            let mut x_outer_full = Vec::with_capacity(x_inner_rest.len() + x_outer.len());
            x_outer_full.extend_from_slice(&x_inner_rest);
            x_outer_full.extend_from_slice(&x_outer);
            let suffix = &x_outer_full[2..];
            let suffix_tensor = build_eq(suffix);
            let want = fold_1b_rows_naive(&packed, &suffix_tensor);

            // New path: lincheck-shaped partial fold of z at x_outer, then a
            // strided fold against the inner-rest tail x_inner_rest[2..].
            let eq_x_outer = build_eq(&x_outer);
            let z_vec = partial_fold_packed_z(&z_packed_lincheck, m, k_log, &eq_x_outer);
            let got = s_hat_v_from_z_vec(&z_vec, &x_inner_rest[2..]);

            assert_eq!(got, want, "s_hat_v mismatch at m={m}, k_log={k_log}");
        }
    }

    /// `prove_batched_padded_with_precomputed` is byte-identical to the
    /// no-precompute path when the supplied precomputed `s_hat_v` is honest.
    #[test]
    fn prove_batched_with_precomputed_matches_unprecomputed() {
        use crate::challenger::FsChallenger;
        let mut rng = Rng::new(0xF00D);
        for &m in &[8usize, 9, 10, 11] {
            let z = rng.bits(1 << m);
            let x_a: Vec<F256> = (0..(m - 6)).map(|_| rng.f256()).collect();
            let x_b: Vec<F256> = (0..(m - 6)).map(|_| rng.f256()).collect();
            let packed = pack_witness(&z, m);

            let mut ch_base = FsChallenger::new(b"flock-test-v0");
            let (base, _) = prove_batched(&packed, &[&x_a, &x_b], &mut ch_base);
            let s_hat_v_a = base[0].0.s_hat_v.clone();
            let s_hat_v_b = base[1].0.s_hat_v.clone();

            let padding = PaddingSpec::dense(m);

            for &(pre_a, pre_b) in &[(false, false), (true, false), (false, true), (true, true)] {
                let pa: Option<&[F256]> = if pre_a { Some(&s_hat_v_a) } else { None };
                let pb: Option<&[F256]> = if pre_b { Some(&s_hat_v_b) } else { None };
                let mut ch = FsChallenger::new(b"flock-test-v0");
                let (got, _) = prove_batched_padded_with_precomputed(
                    &packed,
                    &[&x_a, &x_b],
                    &[pa, pb],
                    &padding,
                    &mut ch,
                );
                assert_eq!(
                    got[0].0, base[0].0,
                    "proof[0] mismatch (pre_a={pre_a}, pre_b={pre_b}, m={m})"
                );
                assert_eq!(
                    got[1].0, base[1].0,
                    "proof[1] mismatch (pre_a={pre_a}, pre_b={pre_b}, m={m})"
                );
                assert_eq!(got[0].1.sumcheck_claim, base[0].1.sumcheck_claim);
                assert_eq!(got[1].1.sumcheck_claim, base[1].1.sumcheck_claim);
                assert_eq!(
                    got[0].1.rs_eq_ind.to_dense(),
                    base[0].1.rs_eq_ind.to_dense()
                );
                assert_eq!(
                    got[1].1.rs_eq_ind.to_dense(),
                    base[1].1.rs_eq_ind.to_dense()
                );
            }
        }
    }

    /// Degenerate path: when k_log == LOG_PACKING the tail is empty and the
    /// kernel returns z_vec untouched.
    #[test]
    fn s_hat_v_from_z_vec_degenerate_tail() {
        let mut rng = Rng::new(0xDEAD);
        let z_vec: Vec<F256> = (0..(1 << LOG_PACKING)).map(|_| rng.f256()).collect();
        let got = s_hat_v_from_z_vec(&z_vec, &[]);
        assert_eq!(got, z_vec);
    }

    #[test]
    fn fold_b128_elems_matches_naive() {
        let mut rng = Rng::new(0xF00D);
        for &l in &[1usize, 4, 8, 12] {
            let len = 1usize << l;
            let suffix: Vec<F256> = (0..len).map(|_| rng.f256()).collect();
            let eq_r: Vec<F256> = (0..256).map(|_| rng.f256()).collect();
            let a = fold_b128_elems_naive(&suffix, &eq_r);
            let b = fold_b128_elems(&suffix, &eq_r);
            assert_eq!(a, b, "fold_b128_elems mismatch at L={l}");
        }
    }

    #[test]
    fn s_hat_v_is_linear_in_witness() {
        let mut rng = Rng::new(0x42);
        let m = 9;
        let z1 = rng.bits(1 << m);
        let z2 = rng.bits(1 << m);
        let z_xor: Vec<bool> = z1.iter().zip(&z2).map(|(a, b)| a ^ b).collect();
        let x_outer: Vec<F256> = (0..(m - 6)).map(|_| rng.f256()).collect();
        let suffix_tensor = build_eq(&x_outer[2..]);

        let s1 = fold_1b_rows_naive(&pack_witness(&z1, m), &suffix_tensor);
        let s2 = fold_1b_rows_naive(&pack_witness(&z2, m), &suffix_tensor);
        let sx = fold_1b_rows_naive(&pack_witness(&z_xor, m), &suffix_tensor);

        for (i, ((&a, &b), &c)) in s1.iter().zip(&s2).zip(&sx).enumerate() {
            assert_eq!(a + b, c, "linearity fails at i={i}");
        }
    }

    // -----------------------------------------------------------------------
    // Sparse-tensor fast path: each sparse kernel must produce byte-identical
    // output to its dense counterpart.
    // -----------------------------------------------------------------------

    fn mk_coords(rng: &mut Rng, n: usize, zero_positions: &[usize]) -> Vec<F256> {
        (0..n)
            .map(|i| {
                if zero_positions.contains(&i) {
                    F256::ZERO
                } else {
                    rng.f256()
                }
            })
            .collect()
    }

    #[test]
    fn build_eq_sparse_matches_dense() {
        let mut rng = Rng::new(0xCAFE_F00D);
        let cases: &[(usize, &[usize])] = &[
            (1, &[0]),
            (4, &[1, 3]),
            (6, &[0, 1, 2, 3, 4]),
            (8, &[2, 3, 4, 5, 6]),
            (10, &[]),
            (10, &[0, 5, 9]),
        ];
        for &(n_coords, zero_pos) in cases {
            let coords = mk_coords(&mut rng, n_coords, zero_pos);
            let dense = build_eq(&coords);
            let sparse_eq = build_eq_sparse(&coords);
            let materialized = sparse_eq.materialize();

            let mut covered = vec![false; dense.len()];
            for &(idx, val) in &materialized {
                assert_eq!(
                    val, dense[idx],
                    "sparse value mismatch at idx={idx} (n={n_coords}, zeros={zero_pos:?})"
                );
                assert_ne!(
                    val,
                    F256::ZERO,
                    "sparse entry is zero — should have been skipped"
                );
                covered[idx] = true;
            }
            for (i, &c) in covered.iter().enumerate() {
                if !c {
                    assert_eq!(
                        dense[i],
                        F256::ZERO,
                        "dense[{i}] nonzero but absent from sparse"
                    );
                }
            }
            for w in materialized.windows(2) {
                assert!(w[0].0 < w[1].0, "support not strictly ascending");
            }
            let live_count = n_coords - zero_pos.len();
            assert_eq!(sparse_eq.len(), 1usize << live_count);
        }
    }

    #[test]
    fn fold_1b_rows_sparse_matches_naive() {
        let mut rng = Rng::new(0x5EED_DEAD);
        for &m in &[9usize, 11, 13] {
            let l = m - LOG_PACKING;
            let pw_len = 1usize << l;
            let pw: Vec<F256> = (0..pw_len).map(|_| rng.f256()).collect();
            let zero_pos: Vec<usize> = (0..l.min(3)).collect();
            let suffix = mk_coords(&mut rng, l, &zero_pos);

            let dense_tensor = build_eq(&suffix);
            let sparse_eq = build_eq_sparse(&suffix);

            let dense_s = fold_1b_rows_naive(&pw, &dense_tensor);
            let sparse_s = fold_1b_rows_sparse(&pw, &sparse_eq);

            assert_eq!(dense_s, sparse_s, "s_hat_v mismatch at m={m}");
        }
    }

    #[test]
    fn fold_b128_elems_sparse_matches_dense() {
        let mut rng = Rng::new(0xC0DE_BABE);
        for &l in &[4usize, 8, 12] {
            let len = 1usize << l;
            let zero_pos: Vec<usize> = (0..l.min(3)).collect();
            let suffix = mk_coords(&mut rng, l, &zero_pos);
            let dense_tensor = build_eq(&suffix);
            let sparse_eq = build_eq_sparse(&suffix);
            let eq_r: Vec<F256> = (0..256).map(|_| rng.f256()).collect();

            let dense_out = fold_b128_elems(&dense_tensor, &eq_r);
            let sparse_out = fold_b128_elems_sparse(len, &sparse_eq, &eq_r);

            assert_eq!(dense_out, sparse_out, "rs_eq_ind mismatch at L={l}");
        }
    }

    /// `prove_batched` with a mix of sparse and dense claims matches calling
    /// `prove` per claim (which uses only the dense kernels).
    #[test]
    fn prove_batched_with_sparse_claim_matches_sequential() {
        use crate::challenger::FsChallenger;
        let mut rng = Rng::new(0xBEEF_CAFE);
        for &m in &[11usize, 12, 13] {
            let z = rng.bits(1 << m);
            let packed = pack_witness(&z, m);
            // Suffix is x[2..], length m-8. Zero out the last 3 suffix coords to
            // trip the sparse threshold (needs suffix length ≥ 3, i.e. m ≥ 11).
            let mut x_chain: Vec<F256> = (0..(m - 6)).map(|_| rng.f256()).collect();
            for j in 0..3 {
                x_chain[(m - 6) - 1 - j] = F256::ZERO;
            }
            let x_ab: Vec<F256> = (0..(m - 6)).map(|_| rng.f256()).collect();
            let x_c: Vec<F256> = (0..(m - 6)).map(|_| rng.f256()).collect();

            let mut ch_seq = FsChallenger::new(b"flock-test-sparse");
            let (p_ab, o_ab) = prove(&packed, &x_ab, &mut ch_seq);
            let (p_c, o_c) = prove(&packed, &x_c, &mut ch_seq);
            let (p_chain, o_chain) = prove(&packed, &x_chain, &mut ch_seq);

            let mut ch_batch = FsChallenger::new(b"flock-test-sparse");
            let (results, _) = prove_batched(&packed, &[&x_ab, &x_c, &x_chain], &mut ch_batch);

            assert_eq!(results[0].0, p_ab, "s_hat_v[ab] mismatch at m={m}");
            assert_eq!(results[1].0, p_c, "s_hat_v[c]  mismatch at m={m}");
            assert_eq!(results[2].0, p_chain, "s_hat_v[chain] mismatch at m={m}");
            assert!(
                matches!(results[2].1.rs_eq_ind, RsEqInd::Sparse { .. }),
                "chain claim should be sparse"
            );
            assert!(
                matches!(
                    results[0].1.rs_eq_ind,
                    RsEqInd::Dense(_) | RsEqInd::DeferredDense { .. }
                ),
                "ab claim should be dense"
            );
            assert!(
                matches!(
                    results[1].1.rs_eq_ind,
                    RsEqInd::Dense(_) | RsEqInd::DeferredDense { .. }
                ),
                "c claim should be dense"
            );
            assert_eq!(results[0].1.sumcheck_claim, o_ab.sumcheck_claim);
            assert_eq!(results[1].1.sumcheck_claim, o_c.sumcheck_claim);
            assert_eq!(results[2].1.sumcheck_claim, o_chain.sumcheck_claim);
            let _ = (&o_ab.rs_eq_ind, &o_c.rs_eq_ind, &o_chain.rs_eq_ind);
        }
    }

    /// Cross-check `eval_rs_eq` against the dense
    /// `mle_eval(fold_b128_elems(build_eq(z_vals)), query)` path.
    #[test]
    fn eval_rs_eq_matches_dense() {
        fn mle_eval_naive(values: &[F256], r: &[F256]) -> F256 {
            assert_eq!(values.len(), 1 << r.len());
            let mut buf = values.to_vec();
            for &r_i in r.iter().rev() {
                let half = buf.len() / 2;
                for i in 0..half {
                    let lo = buf[i];
                    let hi = buf[i + half];
                    buf[i] = lo + r_i * (lo + hi);
                }
                buf.truncate(half);
            }
            buf[0]
        }

        let mut rng = Rng::new(0xDEADBEEF);
        for &l_prime in &[3usize, 6, 10, 14] {
            for _trial in 0..3 {
                let z_vals: Vec<F256> = (0..l_prime).map(|_| rng.f256()).collect();
                let query: Vec<F256> = (0..l_prime).map(|_| rng.f256()).collect();
                let r_dprime: Vec<F256> = (0..LOG_PACKING).map(|_| rng.f256()).collect();
                let eq_r_dprime = build_eq(&r_dprime);

                let suffix_tensor = build_eq(&z_vals);
                let rs_eq_ind_dense = fold_b128_elems(&suffix_tensor, &eq_r_dprime);
                let dense_eval = mle_eval_naive(&rs_eq_ind_dense, &query);

                let succinct_eval = eval_rs_eq(&z_vals, &query, &eq_r_dprime);

                assert_eq!(
                    succinct_eval, dense_eval,
                    "eval_rs_eq mismatch at l_prime={l_prime}"
                );
            }
        }
    }
}
