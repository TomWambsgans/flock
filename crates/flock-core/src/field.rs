//! Binary field arithmetic.
//!
//! - [`F8`]   — GF(2^8) with AES polynomial x^8 + x^4 + x^3 + x + 1
//! - [`F128`] — GF(2^128) in GHASH form, polynomial x^128 + x^7 + x^2 + x + 1
//! - [`F256`] — GF(2^256) as the quadratic extension F128[u]/(u²+u+β)
//! - [`F256Unreduced`] — 256-bit unreduced GHASH products, for deferred reduction
//!
//! Note: [`F256`] is the *field* GF(2^256); [`F256Unreduced`] is an unreduced
//! 256-bit product of two F128s (an intermediate in F128 multiplication), not a
//! field element. Despite the shared "256" they are unrelated types.

pub mod gf2_128;
pub mod gf2_256;
pub mod gf2_8;
pub mod phi8;

pub use gf2_8::F8;
pub use gf2_128::{F128, F256Unreduced, mul_by_x};
pub use gf2_256::F256;
pub use phi8::{PHI_8_TABLE, phi8};
