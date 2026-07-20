//! **u64 × u64 → u128 multiplication** R1CS-over-GF(2): one full-width
//! `p = x·y` (no overflow — the exact 128-bit product) per block, batched
//! block-diagonally. Not a hash — a minimal arithmetic statement for
//! benchmarking raw prover throughput (u64 muls proved per second). Reuses
//! the matrix/witness plumbing from [`crate::r1cs_hashes`].
//!
//! ## Circuit
//!
//! Schoolbook multiplication, full 128-bit product:
//!
//! 1. **Partial products** — one AND wire per `pp(j, i) = x_{i−j} · y_j`
//!    for product bit `i ∈ [j, j+64)`: `64 × 64 = 4,096` wires.
//! 2. **Row-by-row carry-save reduction** — 63 rounds; round `j` adds pp row
//!    `j` into a running per-bit partial sum with one full adder per product
//!    bit `i ∈ [j, j+64)`. Each FA emits:
//!    - a **maj** AND wire `m = (s ⊕ c)(s ⊕ p)` (carry-out `= m ⊕ s`, kept
//!      as a ≤ 2-term linear expression, never materialized): 4,032 wires —
//!      unlike a truncated multiplier, every carry matters;
//!    - a **materialized sum** wire `s ⊕ c ⊕ p` (multiplied by the constant
//!      wire): 4,032 wires, of which the 63 with `i = j` are final product
//!      bits and live in the PROD region (3,969 in the SUM region).
//!
//!    A carry produced at round `j`, bit `i` is consumed at round `j + 1`,
//!    bit `i + 1`, so after round `i` product bit `i < 64` is final.
//! 3. **Final ripple-carry over bits 64..128** — after round 63 the top half
//!    still holds pending carries; a bit-serial ripple `FA(s, c, cr)` emits
//!    63 more maj AND wires and materializes product bits 64..127 into the
//!    PROD region. The carry out of bit 127 is identically zero
//!    (`x·y < 2^128`) and is not represented.
//!
//! Every row's support stays ≤ 5 entries, so `A_0`/`B_0` stay far sparser
//! than the hash circuits.
//!
//! ## Slot layout (single block, `K = 2^14 = 16,384`)
//!
//! ```text
//! z[0..64)         x           — free input bits
//! z[64..128)       y           — free input bits
//! z[128..256)      prod        — materialized product bits (p = x·y, 128 bits)
//! z[256]           Z_CONST (= 1)
//! z[257..320)      gap, forced 0 (empty rows)
//! z[320..4416)     pp          — partial products, 64 aligned 64-bit rows
//! z[4416..8448)    maj         — FA carry AND wires, 63 aligned 64-bit rows
//! z[8448..12417)   sum         — non-final FA sums, 63 rows × 63 bits
//! z[12417..12480)  rip         — ripple-carry maj AND wires (bits 64..127)
//! z[12480..16384)  padding, forced 0
//! ```

use flock_core::field::F128;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};

use crate::r1cs_hashes::common::{
    build_block_r1cs_with_matrices, drive_witness_packed_and_lincheck, xor_dedup,
};

// ───────────────────────────────────────────────────────────────────────────
// Compile-time slot layout
// ───────────────────────────────────────────────────────────────────────────

/// Inner-dimension log: `K = 2^14 = 16,384` rows per block.
pub const K_LOG: usize = 14;
pub const K: usize = 1 << K_LOG;
/// Univariate-skip width.
pub const K_SKIP: usize = 6;

pub const WORD_BITS: usize = 64;
/// The product is full-width: 128 bits.
pub const PROD_BITS: usize = 2 * WORD_BITS;

pub const X_BASE: usize = 0;
pub const Y_BASE: usize = WORD_BITS; // 64
pub const PROD_BASE: usize = 2 * WORD_BITS; // 128
pub const Z_CONST_POS: usize = PROD_BASE + PROD_BITS; // 256
/// Word-aligned for the packed witness builder; bits [257, 320) are a gap.
pub const PP_BASE: usize = 5 * WORD_BITS; // 320
/// 64 rows × 64 partial-product wires.
pub const PP_COUNT: usize = WORD_BITS * WORD_BITS; // 4,096
pub const MAJ_BASE: usize = PP_BASE + PP_COUNT; // 4,416
/// 63 rounds × 64 FA carry (maj) wires.
pub const MAJ_COUNT: usize = (WORD_BITS - 1) * WORD_BITS; // 4,032
pub const SUM_BASE: usize = MAJ_BASE + MAJ_COUNT; // 8,448
/// 63 rounds × 63 non-final FA sum wires.
pub const SUM_COUNT: usize = (WORD_BITS - 1) * (WORD_BITS - 1); // 3,969
pub const RIP_BASE: usize = SUM_BASE + SUM_COUNT; // 12,417
/// Ripple-carry maj wires for product bits 64..127.
pub const RIP_COUNT: usize = WORD_BITS - 1; // 63
pub const USEFUL_BITS: usize = RIP_BASE + RIP_COUNT; // 12,480

// Slot accessors.

#[inline]
pub fn x_bit(b: usize) -> usize {
    X_BASE + b
}
#[inline]
pub fn y_bit(b: usize) -> usize {
    Y_BASE + b
}
#[inline]
pub fn prod_bit(b: usize) -> usize {
    PROD_BASE + b
}
/// Partial product `x_{i−j} · y_j` (product bit `i`, row `j`,
/// `j ≤ i < j + 64`). Row `j` is the aligned word at `PP_BASE + 64j`.
#[inline]
pub fn pp_bit(j: usize, i: usize) -> usize {
    PP_BASE + WORD_BITS * j + (i - j)
}
/// FA carry AND wire of round `j` at product bit `i` (`j ∈ [1, 64)`,
/// `j ≤ i < j + 64`). Round `j` is the aligned word at `MAJ_BASE + 64(j−1)`.
#[inline]
pub fn maj_bit(j: usize, i: usize) -> usize {
    MAJ_BASE + WORD_BITS * (j - 1) + (i - j)
}
/// Non-final materialized FA sum of round `j` at product bit `i`
/// (`j < i < j + 64`; the `i = j` sum is final and lives at [`prod_bit`]).
#[inline]
pub fn sum_bit(j: usize, i: usize) -> usize {
    SUM_BASE + (WORD_BITS - 1) * (j - 1) + (i - j - 1)
}
/// Ripple-carry maj AND wire at product bit `i ∈ [64, 127)`.
#[inline]
pub fn rip_bit(i: usize) -> usize {
    RIP_BASE + (i - WORD_BITS)
}

// ───────────────────────────────────────────────────────────────────────────
// Matrix builder
// ───────────────────────────────────────────────────────────────────────────

/// Sorted-deduplicated XOR support — a row of `A` or `B` is one such Vec.
type Sup = Vec<usize>;

/// XOR (symmetric difference) of several supports.
fn xor_sup(parts: &[&Sup]) -> Sup {
    let mut v = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts {
        v.extend_from_slice(p);
    }
    xor_dedup(v)
}

/// Build `(A_0, B_0)` for one block of the u64-mul R1CS. `C_0 = I`
/// (circuit shape); use [`build_block_r1cs`] to wrap these into a
/// [`BlockR1cs`].
pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let mut a_rows: Vec<Sup> = vec![Sup::new(); K];
    let mut b_rows: Vec<Sup> = vec![Sup::new(); K];

    // Z_CONST tautology: z·z = z (boolean-pin).
    a_rows[Z_CONST_POS] = vec![Z_CONST_POS];
    b_rows[Z_CONST_POS] = vec![Z_CONST_POS];

    // x, y: free-witness rows.
    for b in 0..WORD_BITS {
        for s in [x_bit(b), y_bit(b)] {
            a_rows[s] = vec![s];
            b_rows[s] = vec![Z_CONST_POS];
        }
    }

    // Partial products: z[pp(j, i)] = x_{i−j} · y_j.
    for j in 0..WORD_BITS {
        for i in j..j + WORD_BITS {
            let s = pp_bit(j, i);
            a_rows[s] = vec![x_bit(i - j)];
            b_rows[s] = vec![y_bit(j)];
        }
    }

    // Carry-save reduction over the 128 product-bit positions. `s_sup[i]` =
    // current partial-sum wire of product bit `i` (single materialized wire,
    // or empty above the covered span); `c_sup[i]` = pending carry into bit
    // `i` (≤ 2-term expression, never materialized).
    let mut s_sup: Vec<Sup> = (0..PROD_BITS)
        .map(|i| {
            if i < WORD_BITS {
                vec![pp_bit(0, i)]
            } else {
                Sup::new()
            }
        })
        .collect();
    let mut c_sup: Vec<Sup> = vec![Sup::new(); PROD_BITS + 1];

    for j in 1..WORD_BITS {
        let mut next_c: Vec<Sup> = vec![Sup::new(); PROD_BITS + 1];
        for i in j..j + WORD_BITS {
            let p = vec![pp_bit(j, i)];
            let s = std::mem::take(&mut s_sup[i]);
            let c = std::mem::take(&mut c_sup[i]);
            // maj wire m = (s ⊕ c)(s ⊕ p); carry-out = maj(s, c, p) = m ⊕ s.
            // Full product: every carry is kept.
            let m = maj_bit(j, i);
            a_rows[m] = xor_sup(&[&s, &c]);
            b_rows[m] = xor_sup(&[&s, &p]);
            next_c[i + 1] = xor_sup(&[&vec![m], &s]);
            // Materialized FA sum = s ⊕ c ⊕ p. The round-`i` sum of bit `i`
            // is final — it lives in the PROD region.
            let slot = if i == j { prod_bit(i) } else { sum_bit(j, i) };
            a_rows[slot] = xor_sup(&[&s, &c, &p]);
            b_rows[slot] = vec![Z_CONST_POS];
            s_sup[i] = vec![slot];
        }
        c_sup = next_c;
    }

    // Final ripple-carry over bits 64..128: FA(s, c63, cr) per position.
    let mut cr = Sup::new();
    for i in WORD_BITS..PROD_BITS {
        let s = std::mem::take(&mut s_sup[i]); // empty at i = 127
        let c = std::mem::take(&mut c_sup[i]);
        if i < PROD_BITS - 1 {
            let m = rip_bit(i);
            a_rows[m] = xor_sup(&[&s, &c]);
            b_rows[m] = xor_sup(&[&s, &cr]);
            let next_cr = xor_sup(&[&vec![m], &s]);
            a_rows[prod_bit(i)] = xor_sup(&[&s, &c, &cr]);
            b_rows[prod_bit(i)] = vec![Z_CONST_POS];
            cr = next_cr;
        } else {
            // Bit 127: carry out is identically zero (x·y < 2^128) — sum only.
            a_rows[prod_bit(i)] = xor_sup(&[&s, &c, &cr]);
            b_rows[prod_bit(i)] = vec![Z_CONST_POS];
        }
    }

    // prod[0] = pp(0, 0) · 1 — bit 0 is final immediately after row 0.
    a_rows[prod_bit(0)] = vec![pp_bit(0, 0)];
    b_rows[prod_bit(0)] = vec![Z_CONST_POS];

    let to_mat = |rows| SparseBinaryMatrix {
        num_rows: K,
        num_cols: K,
        rows,
    };
    (to_mat(a_rows), to_mat(b_rows))
}

/// Build a [`BlockR1cs`] for `2^n_blocks_log` u64 multiplications batched
/// block-diagonally (one mul per block).
pub fn build_block_r1cs(n_blocks_log: usize) -> BlockR1cs {
    let (a_0, b_0) = build_matrices();
    build_block_r1cs_with_matrices(
        n_blocks_log,
        K_LOG,
        K_SKIP,
        USEFUL_BITS,
        a_0,
        b_0,
        // Constant-wire pin: forces z[Z_CONST_POS] = 1 in every block.
        // Requires padding blocks filled with valid (0·0) multiplications.
        Some(Z_CONST_POS),
    )
}

// ───────────────────────────────────────────────────────────────────────────
// Witness generators
// ───────────────────────────────────────────────────────────────────────────

/// Reference boolean witness for one block, mirroring [`build_matrices`]
/// bit-for-bit. Slow; tests and debugging only — the prove path uses
/// [`build_block_zab`].
pub fn build_block_witness(x: u64, y: u64) -> Vec<bool> {
    let bit = |w: u64, t: usize| (w >> t) & 1 == 1;
    let mut z = vec![false; K];
    z[Z_CONST_POS] = true;
    for b in 0..WORD_BITS {
        z[x_bit(b)] = bit(x, b);
        z[y_bit(b)] = bit(y, b);
    }
    for j in 0..WORD_BITS {
        for i in j..j + WORD_BITS {
            z[pp_bit(j, i)] = bit(x, i - j) && bit(y, j);
        }
    }
    let mut s = [false; PROD_BITS];
    let mut c = [false; PROD_BITS + 1];
    for (i, si) in s.iter_mut().enumerate().take(WORD_BITS) {
        *si = z[pp_bit(0, i)];
    }
    for j in 1..WORD_BITS {
        let mut next_c = [false; PROD_BITS + 1];
        for i in j..j + WORD_BITS {
            let p = z[pp_bit(j, i)];
            let (sv, cv) = (s[i], c[i]);
            let m = (sv ^ cv) & (sv ^ p);
            z[maj_bit(j, i)] = m;
            next_c[i + 1] = m ^ sv;
            let sum = sv ^ cv ^ p;
            let slot = if i == j { prod_bit(i) } else { sum_bit(j, i) };
            z[slot] = sum;
            s[i] = sum;
        }
        // Positions above the round's span keep their sum untouched but must
        // not lose an already-pending carry — there is none: round j−1's
        // carries land within [j, j + 64], all inside round j's span.
        c = next_c;
    }
    let mut cr = false;
    for i in WORD_BITS..PROD_BITS {
        let (sv, cv) = (s[i], c[i]);
        if i < PROD_BITS - 1 {
            let m = (sv ^ cv) & (sv ^ cr);
            z[rip_bit(i)] = m;
            z[prod_bit(i)] = sv ^ cv ^ cr;
            cr = m ^ sv;
        } else {
            z[prod_bit(i)] = sv ^ cv ^ cr;
        }
    }
    z[prod_bit(0)] = z[pp_bit(0, 0)];
    z
}

/// Read the 128-bit product out of a single block of witness.
pub fn read_prod(z: &[bool]) -> u128 {
    (0..PROD_BITS).fold(0u128, |acc, b| acc | ((z[prod_bit(b)] as u128) << b))
}

/// OR `val` (pre-masked to its width ≤ 64) into `buf` at bit offset `off`,
/// handling u64 straddling.
#[inline(always)]
fn or_u64_at_bit(buf: &mut [u64], off: usize, val: u64) {
    let w = off >> 6;
    let s = off & 63;
    buf[w] |= val << s;
    // `(x >> 1) >> (63 − s)` = `x >> (64 − s)` without the s = 0 UB.
    let spill = (val >> 1) >> (63 - s);
    if spill != 0 {
        buf[w + 1] |= spill;
    }
}

/// Fused per-block builder: fill one block's `(z, a, b)` — three zeroed
/// `K/64`-length u64 buffers — with the witness and its `A_0·z` / `B_0·z`
/// images for the full product `p = x·y`. Word-level: each FA round is a
/// handful of u128 ops plus aligned word writes per region.
pub(crate) fn build_block_zab(x: u64, y: u64, z: &mut [u64], a: &mut [u64], b: &mut [u64]) {
    let p = (x as u128) * (y as u128);

    // Aligned words 0..4: x, y, prod lo/hi (b-side = const 1 → all-ones),
    // Z_CONST.
    z[0] = x;
    a[0] = x;
    b[0] = u64::MAX;
    z[1] = y;
    a[1] = y;
    b[1] = u64::MAX;
    z[2] = p as u64;
    a[2] = p as u64;
    b[2] = u64::MAX;
    z[3] = (p >> 64) as u64;
    a[3] = (p >> 64) as u64;
    b[3] = u64::MAX;
    z[4] = 1;
    a[4] = 1;
    b[4] = 1;

    // Round 0: partial-sum word = pp row 0. Round words live in product-bit
    // alignment (bit i of the u128 = product bit i).
    let mut s = if y & 1 == 1 { x as u128 } else { 0 };
    let mut c = 0u128;
    let ppw = PP_BASE / 64; // pp row j is the aligned word ppw + j
    z[ppw] = s as u64;
    a[ppw] = x;
    b[ppw] = if y & 1 == 1 { u64::MAX } else { 0 };

    for j in 1..WORD_BITS {
        let yj = (y >> j) & 1 == 1;

        // pp row j: bit t = x_t · y_j (aligned word).
        z[ppw + j] = if yj { x } else { 0 };
        a[ppw + j] = x;
        b[ppw + j] = if yj { u64::MAX } else { 0 };

        // FA round j over product bits i = j..j+64. Bits < j of s are final
        // product bits (c and pw are zero there, so they pass through sumw);
        // bits ≥ j+64 of s, c, pw are zero, so majw ^ s vanishes outside the
        // round's span and the carry word needs no masking.
        let pw = if yj { (x as u128) << j } else { 0 };
        let axor = s ^ c;
        let bxor = s ^ pw;
        let majw = axor & bxor;
        let sumw = s ^ c ^ pw;

        // maj wires i = j..j+64: exactly the aligned 64-bit window at bit j.
        let mw = MAJ_BASE / 64 + (j - 1);
        z[mw] = (majw >> j) as u64;
        a[mw] = (axor >> j) as u64;
        b[mw] = (bxor >> j) as u64;

        // Non-final sums i = j+1..j+64 (63 bits, unaligned rows).
        let soff = SUM_BASE + (WORD_BITS - 1) * (j - 1);
        let sval = ((sumw >> (j + 1)) as u64) & (u64::MAX >> 1);
        or_u64_at_bit(z, soff, sval);
        or_u64_at_bit(a, soff, sval);
        or_u64_at_bit(b, soff, u64::MAX >> 1);

        // Carry-out = maj(s, c, p) = majw ⊕ s, shifted to its target bit.
        c = (majw ^ s) << 1;
        // The i = j (final) sum bit is already covered by the PROD words.
        s = sumw;
    }

    // Final ripple-carry over bits 64..128 (bit-serial).
    let (mut mz, mut ma, mut mb) = (0u64, 0u64, 0u64);
    let mut cr = 0u64;
    for i in WORD_BITS..PROD_BITS {
        let sv = ((s >> i) & 1) as u64;
        let cv = ((c >> i) & 1) as u64;
        debug_assert_eq!(sv ^ cv ^ cr, ((p >> i) as u64) & 1, "prod bit {i}");
        if i < PROD_BITS - 1 {
            let az = sv ^ cv;
            let bz = sv ^ cr;
            let m = az & bz;
            let t = i - WORD_BITS;
            mz |= m << t;
            ma |= az << t;
            mb |= bz << t;
            cr = m ^ sv;
        }
    }
    or_u64_at_bit(z, RIP_BASE, mz);
    or_u64_at_bit(a, RIP_BASE, ma);
    or_u64_at_bit(b, RIP_BASE, mb);
}

/// Build `(z, a, b, z_lincheck)` packed witness buffers for a batch of
/// multiplications padded to `2^n_blocks_log` slots. Padding slots hold the
/// valid `0·0` block so the pinned constant wire is 1 in every slot.
pub fn generate_witness_with_ab_packed_and_lincheck(
    muls: &[(u64, u64)],
    n_blocks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    drive_witness_packed_and_lincheck(
        muls,
        Some(&(0, 0)),
        n_blocks_log,
        K_LOG,
        |&(x, y), z, a, b| build_block_zab(x, y, z, a, b),
    )
}

/// Smallest `n_blocks_log` for a batch of `n_muls`: `m = K_LOG + n_blocks_log`
/// must reach the Ligerito config floor (`m ≥ 22`), so `n_blocks_log ≥ 8`.
pub fn min_n_blocks_log(n_muls: usize) -> usize {
    assert!(n_muls >= 1);
    let n = n_muls.max(1 << 8);
    n.next_power_of_two().trailing_zeros() as usize
}

// ───────────────────────────────────────────────────────────────────────────
// Setup: R1CS + PCS params + prove/verify wrappers
// ───────────────────────────────────────────────────────────────────────────

/// Reusable prove/verify context for batches of u64 multiplications:
/// the block R1CS, its cached CSC lincheck circuit, and the PCS parameters.
pub struct Mul64Setup {
    pub n_muls: usize,
    pub r1cs: BlockR1cs,
    pub pcs_params: flock_core::pcs::PcsParams,
}

impl Mul64Setup {
    pub fn new(n_muls: usize) -> Self {
        Self::with_profile(n_muls, flock_core::pcs::ligerito::LigeritoProfile::Fast)
    }

    /// Build a setup for a named Ligerito profile (fast/slim/secure).
    pub fn with_profile(
        n_muls: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
    ) -> Self {
        assert!(n_muls >= 1, "n_muls must be ≥ 1");
        let n_log = min_n_blocks_log(n_muls);
        let r1cs = build_block_r1cs(n_log);
        // Warm the CSC fold circuit and pre-fault the prove-cycle scratch
        // buffers so the first prove pays no one-time costs.
        r1cs.csc_lincheck_circuit();
        flock_core::scratch::prewarm_prover(r1cs.m);
        let pcs_params = flock_core::pcs::PcsParams {
            m: r1cs.m,
            log_inv_rate: profile.log_inv_rate(),
            log_batch_size: 6,
            profile,
        };
        Self {
            n_muls,
            r1cs,
            pcs_params,
        }
    }

    pub fn m(&self) -> usize {
        self.r1cs.m
    }
    pub fn n_blocks_log(&self) -> usize {
        self.r1cs.m - self.r1cs.k_log
    }
    pub fn n_block_slots(&self) -> usize {
        1usize << self.n_blocks_log()
    }

    /// Packed `(z, a, b, z_lincheck)` for a batch (see
    /// [`generate_witness_with_ab_packed_and_lincheck`]).
    pub fn generate_witness_ab(
        &self,
        muls: &[(u64, u64)],
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        assert_eq!(muls.len(), self.n_muls);
        generate_witness_with_ab_packed_and_lincheck(muls, self.n_blocks_log())
    }

    /// Fast prover: fused witness build (z, a, b, lincheck stripe emitted
    /// directly), then commit → zerocheck → lincheck → Ligerito open.
    pub fn prove_fast<Ch: flock_core::challenger::Challenger>(
        &self,
        muls: &[(u64, u64)],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
    ) {
        assert_eq!(muls.len(), self.n_muls);
        let (codeword, (z_packed, a_packed, b_packed, z_lincheck)) =
            flock_core::pcs::prefault_codeword_during(&self.pcs_params, || {
                self.generate_witness_ab(muls)
            });
        crate::prover::prove_fast_ligerito_from_witness(
            &self.r1cs,
            &self.pcs_params,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            self.r1cs.csc_lincheck_circuit(),
            codeword,
            challenger,
        )
    }

    /// [`Self::prove_fast`] with a per-phase timing breakdown (witness gen +
    /// commit + zerocheck + lincheck + recursive open). Benchmark-only.
    pub fn prove_fast_timed<Ch: flock_core::challenger::Challenger>(
        &self,
        muls: &[(u64, u64)],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
        crate::prover::ProvePhaseTimings,
    ) {
        assert_eq!(muls.len(), self.n_muls);
        let t0 = std::time::Instant::now();
        let (z_packed, a_packed, b_packed, z_lincheck) = self.generate_witness_ab(muls);
        let witness_s = t0.elapsed().as_secs_f64();
        let (proof, commitment, claim, mut timings) = crate::prover::prove_fast_ligerito_timed(
            &self.r1cs,
            &self.pcs_params,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            self.r1cs.csc_lincheck_circuit(),
            None,
            challenger,
        );
        timings.witness_s = witness_s;
        (proof, commitment, claim, timings)
    }

    pub fn verify<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &flock_core::proof::R1csProofLigerito,
        challenger: &mut Ch,
    ) -> Result<flock_core::proof::R1csClaim, flock_core::verifier::VerifyError> {
        flock_core::verifier::verify_ligerito(
            &self.r1cs,
            commitment,
            proof,
            self.r1cs.csc_lincheck_circuit(),
            &self.pcs_params,
            challenger,
        )
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64 PRNG, deterministic.
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
    }

    /// Interesting corners + random pairs.
    fn test_pairs(n_random: usize, seed: u64) -> Vec<(u64, u64)> {
        let mut pairs = vec![
            (0, 0),
            (0, u64::MAX),
            (u64::MAX, u64::MAX),
            (1, u64::MAX),
            (1u64 << 63, 3),
            (1u64 << 63, 1u64 << 63),
            (0xDEAD_BEEF_CAFE_F00D, 0x0123_4567_89AB_CDEF),
        ];
        let mut rng = Rng::new(seed);
        pairs.extend((0..n_random).map(|_| (rng.next_u64(), rng.next_u64())));
        pairs
    }

    /// Every slot accessor lands in its region exactly once, and the regions
    /// tile [0, USEFUL_BITS) with only the declared gap.
    #[test]
    fn layout_is_injective() {
        assert!(USEFUL_BITS <= K);

        let mut seen = vec![false; K];
        let mut claim = |s: usize| {
            assert!(!seen[s], "slot {s} assigned twice");
            seen[s] = true;
        };
        for b in 0..WORD_BITS {
            claim(x_bit(b));
            claim(y_bit(b));
        }
        for b in 0..PROD_BITS {
            claim(prod_bit(b));
        }
        claim(Z_CONST_POS);
        for j in 0..WORD_BITS {
            for i in j..j + WORD_BITS {
                claim(pp_bit(j, i));
            }
        }
        for j in 1..WORD_BITS {
            for i in j..j + WORD_BITS {
                claim(maj_bit(j, i));
            }
            for i in j + 1..j + WORD_BITS {
                claim(sum_bit(j, i));
            }
        }
        for i in WORD_BITS..PROD_BITS - 1 {
            claim(rip_bit(i));
        }
        let n_claimed = seen.iter().filter(|&&x| x).count();
        assert_eq!(n_claimed, USEFUL_BITS - (PP_BASE - Z_CONST_POS - 1));
        assert!(
            seen[..USEFUL_BITS]
                .iter()
                .enumerate()
                .all(|(s, &x)| x || (Z_CONST_POS < s && s < PP_BASE))
        );
    }

    /// Row-by-row R1CS check `(A·z) ⊙ (B·z) = z` on one block.
    fn satisfies_singleblock(
        a: &SparseBinaryMatrix,
        b: &SparseBinaryMatrix,
        z: &[bool],
    ) -> Result<(), usize> {
        for r in 0..a.rows.len() {
            let av = a.rows[r].iter().fold(false, |acc, &s| acc ^ z[s]);
            let bv = b.rows[r].iter().fold(false, |acc, &s| acc ^ z[s]);
            if (av && bv) != z[r] {
                return Err(r);
            }
        }
        Ok(())
    }

    /// The boolean witness satisfies the circuit and carries the right
    /// full 128-bit product; flipping a product bit breaks it.
    #[test]
    fn witness_satisfies_and_product_correct() {
        let (a_0, b_0) = build_matrices();
        for (x, y) in test_pairs(24, 0x9E37) {
            let mut z = build_block_witness(x, y);
            assert_eq!(
                read_prod(&z),
                (x as u128) * (y as u128),
                "x={x:#x} y={y:#x}"
            );
            satisfies_singleblock(&a_0, &b_0, &z)
                .unwrap_or_else(|r| panic!("row {r} unsatisfied for x={x:#x} y={y:#x}"));
            // Tamper: flip one product bit (incl. the high half) → some row
            // must break.
            let flip = prod_bit((x ^ y) as usize % PROD_BITS);
            z[flip] = !z[flip];
            assert!(
                satisfies_singleblock(&a_0, &b_0, &z).is_err(),
                "tampered witness accepted for x={x:#x} y={y:#x}"
            );
        }
    }

    /// The fused word-level builder matches the boolean reference: same z,
    /// and its a/b outputs equal `A_0·z` / `B_0·z`.
    #[test]
    fn fused_builder_matches_reference() {
        let (a_0, b_0) = build_matrices();
        const W: usize = K / 64;
        for (x, y) in test_pairs(24, 0xB16B) {
            let (mut z, mut a, mut b) = ([0u64; W], [0u64; W], [0u64; W]);
            build_block_zab(x, y, &mut z, &mut a, &mut b);

            let z_ref = build_block_witness(x, y);
            let getbit = |buf: &[u64; W], s: usize| (buf[s >> 6] >> (s & 63)) & 1 == 1;
            for s in 0..K {
                assert_eq!(getbit(&z, s), z_ref[s], "z bit {s} (x={x:#x} y={y:#x})");
                let av = a_0.rows[s].iter().fold(false, |acc, &t| acc ^ z_ref[t]);
                let bv = b_0.rows[s].iter().fold(false, |acc, &t| acc ^ z_ref[t]);
                assert_eq!(getbit(&a, s), av, "a bit {s} (x={x:#x} y={y:#x})");
                assert_eq!(getbit(&b, s), bv, "b bit {s} (x={x:#x} y={y:#x})");
            }
        }
    }

    /// Full-batch `BlockR1cs::satisfies` on the packed witness, incl. the
    /// padding slots (const-wire pin requires them to be valid 0·0 blocks).
    #[test]
    fn packed_batch_satisfies() {
        let n_log = 3;
        let r1cs = build_block_r1cs(n_log);
        let muls = test_pairs(0, 0)[..6].to_vec(); // 6 muls in 8 slots → padding
        let (z, a, b, _stripe) = generate_witness_with_ab_packed_and_lincheck(&muls, n_log);
        assert!(r1cs.satisfies_packed(&z));
        assert_eq!(a, r1cs.apply_a_packed(&z));
        assert_eq!(b, r1cs.apply_b_packed(&z));
    }

    /// End-to-end Ligerito roundtrip + tamper rejection at the smallest
    /// supported shape (m = 22, 256 slots).
    #[test]
    #[ignore] // Heavier — run with `cargo test -p flock-prover --release mul64 -- --ignored`
    fn prove_fast_roundtrip_ligerito() {
        use flock_core::challenger::FsChallenger;

        let n_muls = 200; // < 256 slots → exercises padding blocks
        let setup = Mul64Setup::new(n_muls);
        assert_eq!(setup.m(), 22);
        let mut rng = Rng::new(0x6412_AB01);
        let muls: Vec<(u64, u64)> = (0..n_muls)
            .map(|_| (rng.next_u64(), rng.next_u64()))
            .collect();

        let mut ch_p = FsChallenger::new(b"flock-lig-mul64-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&muls, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-lig-mul64-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("verifier rejected honest mul64 proof: {e:?}"));
        assert_eq!(claim_p, claim_v);

        let mut bad = proof.clone();
        bad.zerocheck.final_a_eval.lo ^= 1;
        let mut ch = FsChallenger::new(b"flock-lig-mul64-v0");
        assert!(
            setup.verify(&commitment, &bad, &mut ch).is_err(),
            "tampered mul64 proof accepted"
        );
    }
}
