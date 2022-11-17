use crate::matrix::BitMatrix32 as M32;
use core::simd::{LaneCount, Simd, SupportedLaneCount};

pub struct Rand32Simd<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    last_jump: isize,
    last_delta: [M32; 3],
    seed1: Simd<u32, LANES>,
    seed2: Simd<u32, LANES>,
    seed3: Simd<u32, LANES>,
}

impl<const LANES: usize> Rand32Simd<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn from_state_interval(
        state: (u32, u32, u32),
        interval: usize,
    ) -> Self {
        let (s1, s2, s3) = state;
        let [m1, m2, m3] = crate::rand32::rng_matrix();
        let seed_from_params = |mut seed, step| {
            let array = core::array::from_fn(|_| {
                let value = seed;
                seed = seed * step;
                value
            });
            Simd::from_array(array)
        };

        Self {
            last_jump: 0,
            last_delta: [M32::eye(); 3],
            seed1: seed_from_params(s1, m1.pow(interval)),
            seed2: seed_from_params(s2, m2.pow(interval)),
            seed3: seed_from_params(s3, m3.pow(interval)),
        }
    }

    pub fn jump(&mut self, steps: isize) {
        if self.last_jump != steps {
            self.last_jump = steps;

            if steps < 0 {
                let steps = steps
                    .checked_neg()
                    .expect("what are you doing at MIN_INT?")
                    as usize;

                self.last_delta =
                    crate::rand32::rng_back_matrix().map(|m| m.pow(steps));
            } else {
                let steps = steps as usize;

                self.last_delta =
                    crate::rand32::rng_matrix().map(|m| m.pow(steps));
            }
        }

        self.seed1 = self.seed1 * self.last_delta[0];
        self.seed2 = self.seed2 * self.last_delta[1];
        self.seed3 = self.seed3 * self.last_delta[2];
    }

    pub fn rand_back(&mut self) -> Simd<u32, LANES> {
        use crate::rand32_rev::simd::{prev_s1, prev_s2, prev_s3};
        self.seed1 = prev_s1(self.seed1);
        self.seed2 = prev_s2(self.seed2);
        self.seed3 = prev_s3(self.seed3);
        self.seed1 ^ self.seed2 ^ self.seed3
    }

    pub fn rand(&mut self) -> Simd<u32, LANES> {
        let s = Simd::splat;
        self.seed1 = ((self.seed1 & s(0xFFFFFFFE)) << s(12))
            ^ ((self.seed1 & s(0x7FFC0) ^ (self.seed1 >> s(13))) >> s(6));
        self.seed2 = ((self.seed2 & s(0xFFFFFFF8)) << s(4))
            ^ (((self.seed2 >> s(2)) ^ self.seed2 & s(0x3F800000)) >> s(23));
        self.seed3 = ((self.seed3 & s(0xFFFFFFF0)) << s(17))
            ^ (((self.seed3 >> s(3)) ^ self.seed3 & s(0x1FFFFF00)) >> s(8));
        self.seed1 ^ self.seed2 ^ self.seed3
    }
}

#[cfg(test)]
mod tests {
    use super::Rand32Simd;
    use crate::rand32::Rand32Ref;
    #[test]
    fn test_rand_simd() {
        let mut rng = Rand32Ref::new(0x13371337);
        let state = rng.state();
        const LANES: usize = 8;
        let ref_values: [u32; LANES] = core::array::from_fn(|_| rng.rand());
        let mut rng = Rand32Simd::<LANES>::from_state_interval(state, 1);
        let simd_values = rng.rand().to_array();
        assert_eq!(ref_values, simd_values);
    }

    #[test]
    fn test_rand_simd_back() {
        let mut rng = Rand32Ref::new(0x13371337);
        let state = rng.state();
        const LANES: usize = 8;
        let ref_values: [u32; LANES] = core::array::from_fn(|_| rng.rand());
        let mut rng = Rand32Simd::<LANES>::from_state_interval(state, 1);
        rng.rand();
        rng.rand_back();
        let simd_values = rng.rand().to_array();
        assert_eq!(ref_values, simd_values);
    }

    #[test]
    fn test_rand_simd_jump() {
        let mut rng = Rand32Ref::new(0x13371337);
        let state = rng.state();
        let skip = 0x1337;
        for _ in 0..skip {
            rng.rand();
        }
        const LANES: usize = 8;
        let ref_values: [u32; LANES] = core::array::from_fn(|_| rng.rand());
        let mut rng = Rand32Simd::<LANES>::from_state_interval(state, 1);
        rng.jump(skip);
        let simd_values = rng.rand().to_array();
        assert_eq!(ref_values, simd_values);
    }

    #[test]
    fn test_rand_simd_jump_back() {
        let mut rng = Rand32Ref::new(0x13371337);
        let state = rng.state();
        const LANES: usize = 8;
        let ref_values: [u32; LANES] = core::array::from_fn(|_| rng.rand());
        let mut rng = Rand32Simd::<LANES>::from_state_interval(state, 1);
        rng.jump(0x1337);
        rng.jump(-0x1337);
        let simd_values = rng.rand().to_array();
        assert_eq!(ref_values, simd_values);
    }
}
