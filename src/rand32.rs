use crate::matrix::BitMatrix32 as M32;

/// Reference implementation of the WvsGlobal RNG used for scrolling
///
/// This has been tested to match identically to the C# output for tens of
/// millions of iterations.
///
/// This is not designed to be fast, simplified, etc. This is simply designed
/// to be a reference implementation to compare against
pub struct Rand32Ref {
    seed1: u32,
    seed2: u32,
    seed3: u32,
}

impl Rand32Ref {
    pub fn new(tick_count: u32) -> Self {
        let seed = tick_count.wrapping_mul(1170746341).wrapping_sub(755606699);
        Self::seeded(seed, seed, seed)
    }

    pub fn seeded(seed1: u32, seed2: u32, seed3: u32) -> Self {
        Self {
            seed1: seed1 | 0x100000,
            seed2: seed2 | 0x1000,
            seed3: seed3 | 0x10,
        }
    }

    pub fn state(&self) -> (u32, u32, u32) {
        (self.seed1, self.seed2, self.seed3)
    }

    pub fn from_state(state: (u32, u32, u32)) -> Self {
        Self {
            seed1: state.0,
            seed2: state.1,
            seed3: state.2,
        }
    }

    pub fn rand(&mut self) -> u32 {
        self.seed1 = ((self.seed1 & 0xFFFFFFFE) << 12)
            ^ ((self.seed1 & 0x7FFC0 ^ (self.seed1 >> 13)) >> 6);
        self.seed2 = ((self.seed2 & 0xFFFFFFF8) << 4)
            ^ (((self.seed2 >> 2) ^ self.seed2 & 0x3F800000) >> 23);
        self.seed3 = ((self.seed3 & 0xFFFFFFF0) << 17)
            ^ (((self.seed3 >> 3) ^ self.seed3 & 0x1FFFFF00) >> 8);
        self.seed1 ^ self.seed2 ^ self.seed3
    }
}

/// Rand32 step for each state field, represented in terms of matrix transform
pub fn rng_matrix() -> [M32; 3] {
    use core::ops::{Shl, Shr};
    let m1 = M32::eye().and(0xFFFFFFFE).shl(12)
        ^ (M32::eye().and(0x7FFC0) ^ M32::eye().shr(13)).shr(6);
    let m2 = M32::eye().and(0xFFFFFFF8).shl(4)
        ^ (M32::eye().shr(2) ^ M32::eye().and(0x3F800000)).shr(23);
    let m3 = M32::eye().and(0xFFFFFFF0).shl(17)
        ^ (M32::eye().shr(3) ^ M32::eye().and(0x1FFFFF00)).shr(8);
    [m1, m2, m3]
}

/// Rand32 back step for each state field, represented in terms of matrix
/// transform. Please note that the formula for xor shift was modified in order
/// to include the unused bits. This is necessary to allow matrix inversion.
pub fn rng_back_matrix() -> [M32; 3] {
    use core::ops::{Shl, Shr};
    let m1 = M32::eye().and(0xFFFFFFFE).shl(12)
        ^ (M32::eye().and(0x7FFC0) ^ M32::eye().shr(13)).shr(6)
        ^ M32::eye().and(1);
    let m2 = M32::eye().and(0xFFFFFFF8).shl(4)
        ^ (M32::eye().shr(2) ^ M32::eye().and(0x3F800000)).shr(23)
        ^ M32::eye().and(0x7);
    let m3 = M32::eye().and(0xFFFFFFF0).shl(17)
        ^ (M32::eye().shr(3) ^ M32::eye().and(0x1FFFFF00)).shr(8)
        ^ M32::eye().and(0xf);
    [
        m1.inv().unwrap().shr(1).shl(1),
        m2.inv().unwrap().shr(3).shl(3),
        m3.inv().unwrap().shr(4).shl(4),
    ]
}
