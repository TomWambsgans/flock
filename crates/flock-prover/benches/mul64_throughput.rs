//! u64-multiplication proving-throughput sweep (muls proved per second).
//!
//! One full-width `x·y → u128` per K_LOG=14 block (see
//! [`flock_prover::r1cs_mul64`]). Batch sizes come from `MUL64_BENCH_LOG2S`
//! (log2 batch sizes, default
//! "10 12 14 16 18"; minimum 8 — the m = 22 Ligerito config floor), best of
//! `MUL64_BENCH_RUNS` (default 3) after one verified warm-up. Thread count is
//! controlled through `RAYON_NUM_THREADS`. Set `MUL64_BENCH_PHASES=1` to also
//! print a per-phase breakdown of one run per batch size.

use std::hint::black_box;
use std::time::{Duration, Instant};

use flock_prover::challenger::FsChallenger;
use flock_prover::r1cs_mul64::Mul64Setup;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// `1234567 → "1,234,567"`.
fn commas(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn best_of<T, F, O>(inputs: &[T], runs: usize, mut prove: F) -> Duration
where
    F: FnMut(&T) -> O,
{
    let mut best = Duration::MAX;
    for input in &inputs[1..=runs] {
        let start = Instant::now();
        let output = prove(input);
        best = best.min(start.elapsed());
        black_box(output);
    }
    best
}

/// Benchmark one batch size; returns the throughput (muls proved per second).
fn bench_mul64(batch: usize, runs: usize, phases: bool) -> f64 {
    let setup = Mul64Setup::new(batch);
    let input_sets: Vec<Vec<(u64, u64)>> = (0..=runs)
        .map(|run| {
            let mut rng = Rng::new(0x6412_6412 ^ batch as u64 ^ run as u64);
            (0..batch)
                .map(|_| (rng.next_u64(), rng.next_u64()))
                .collect()
        })
        .collect();

    let mut challenger = FsChallenger::new(b"flock-mul64-bench-v0");
    let (proof, commitment, _) = setup.prove_fast(&input_sets[0], &mut challenger);
    let mut challenger = FsChallenger::new(b"flock-mul64-bench-v0");
    setup
        .verify(&commitment, &proof, &mut challenger)
        .expect("u64-mul warm-up proof failed verification");
    black_box(proof);

    if phases {
        let mut challenger = FsChallenger::new(b"flock-mul64-bench-v0");
        let (proof, _, _, t) = setup.prove_fast_timed(&input_sets[0], &mut challenger);
        black_box(proof);
        eprintln!(
            "    phases: witness {:.3} s, commit {:.3} s, zerocheck {:.3} s, \
             lincheck {:.3} s, open {:.3} s",
            t.witness_s, t.commit_s, t.zerocheck_s, t.lincheck_s, t.open_s
        );
    }

    let best = best_of(&input_sets, runs, |inputs| {
        let mut challenger = FsChallenger::new(b"flock-mul64-bench-v0");
        setup.prove_fast(inputs, &mut challenger)
    });
    let seconds = best.as_secs_f64();
    let throughput = batch as f64 / seconds;
    println!(
        "batch {:>9} (m={}): {seconds:>7.3} s/proof  →  {:>9} muls/s",
        commas(batch as u64),
        setup.m(),
        commas(throughput as u64),
    );
    throughput
}

fn parse_log2_batches() -> Vec<u32> {
    let value = std::env::var("MUL64_BENCH_LOG2S").unwrap_or_else(|_| "10 12 14 16 18".to_owned());
    let batches: Vec<u32> = value
        .split([',', ' '])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let log2 = part
                .parse::<u32>()
                .expect("MUL64_BENCH_LOG2S must contain integer log2 batch sizes");
            assert!(
                log2 >= 8,
                "MUL64_BENCH_LOG2S values must be at least 8 (m = 22 config floor)"
            );
            assert!(
                log2 + 14 <= 35,
                "MUL64_BENCH_LOG2S values must be at most 21 (m = 35 config ceiling)"
            );
            log2
        })
        .collect();
    assert!(!batches.is_empty(), "MUL64_BENCH_LOG2S must not be empty");
    batches
}

fn parse_runs() -> usize {
    let runs = std::env::var("MUL64_BENCH_RUNS")
        .unwrap_or_else(|_| "3".to_owned())
        .parse::<usize>()
        .expect("MUL64_BENCH_RUNS must be a positive integer");
    assert!(runs > 0, "MUL64_BENCH_RUNS must be greater than zero");
    runs
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let batches = parse_log2_batches();
    let runs = parse_runs();
    let phases = std::env::var_os("MUL64_BENCH_PHASES").is_some();
    eprintln!(
        "Flock u64-mul proving throughput: {} thread(s), best of {runs} after one warm-up\n",
        rayon::current_num_threads(),
    );

    let mut peak = (0usize, 0f64);
    for &log2 in &batches {
        let batch = 1usize << log2;
        let throughput = bench_mul64(batch, runs, phases);
        if throughput > peak.1 {
            peak = (batch, throughput);
        }
    }
    println!(
        "\nTHROUGHPUT: {} u64 muls proven per second ({} threads, batch {})",
        commas(peak.1 as u64),
        rayon::current_num_threads(),
        commas(peak.0 as u64),
    );
}
