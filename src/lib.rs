#![cfg_attr(feature = "simd", feature(portable_simd))]

pub mod matrix;
pub mod mod24;
pub use mod24::Mod24Solver;
pub mod rand32;
pub mod rand32_rev;
#[cfg(feature = "simd")]
pub mod rand32_simd;

pub mod u56_to_seed;
pub use u56_to_seed::U56ToSeed;

pub mod xorshift128;
