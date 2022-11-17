// s = vuts rqpo nmlk jihg fedc ba98 7654 3210
// XOR jihg fedc ba98 7654 321
//                            v utsr qpon mlkj
//                            i hgfe dcba 9876
/// Compute previous value for seed1 in Rand32
pub fn prev_s1(s: u32) -> u32 {
    let bits_19_01 = s >> 13;
    let bits_31_20 = ((bits_19_01 >> 6) & 0xfff) ^ (s >> 1);
    (bits_31_20 << 20) | (bits_19_01 << 1)
}

// s = vuts rqpo nmlk jihg fedc ba98 7654 3210
// XOR rqpo nmlk jihg fedc ba98 7654 3
//                                    vut srqp
//                                    tsr qpon
/// Compute previous value for seed2 in Rand32
pub fn prev_s2(s: u32) -> u32 {
    let bits_27_03 = s >> 7;
    let bits_29_28 = ((s >> 30) ^ (s >> 3)) & 0b11;
    let bits_31_30 = (s >> 5) ^ bits_29_28;
    (bits_31_30 << 30) | (bits_29_28 << 28) | (bits_27_03 << 3)
}

// s = vuts rqpo nmlk jihg fedc ba98 7654 3210
// XOR edcb a987 654
//                  v utsr qpon mlkj ihgf edcb
//                  s rqpo nmlk jihg fedc ba98
/// Compute previous value for seed3 in Rand32
pub fn prev_s3(s: u32) -> u32 {
    let bits_14_04 = s >> 21;
    let mut window = s >> 4;
    let mut bits_31_15 = 0;
    let mut chunk = (bits_14_04 >> 8) & 0b111;
    for _ in 0..6 {
        chunk = (window ^ chunk) & 0b111;
        window >>= 3;
        bits_31_15 = (bits_31_15 >> 3) | (chunk << 29);
    }
    (bits_31_15 << 1) | (bits_14_04 << 4)
}

#[cfg(feature = "simd")]
pub mod simd {
    use core::simd::{LaneCount, Simd, SupportedLaneCount};
    /// Compute previous value for seed1 in Rand32
    pub fn prev_s1<const LANES: usize>(s1: Simd<u32, LANES>) -> Simd<u32, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let s = Simd::splat;
        let bits_19_01 = s1 >> s(13);
        let bits_31_20 = ((bits_19_01 >> s(6)) & s(0xfff)) ^ (s1 >> s(1));
        (bits_31_20 << s(20)) | (bits_19_01 << s(1))
    }

    /// Compute previous value for seed2 in Rand32
    pub fn prev_s2<const LANES: usize>(s2: Simd<u32, LANES>) -> Simd<u32, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let s = Simd::splat;
        let bits_27_03 = s2 >> s(7);
        let bits_29_28 = ((s2 >> s(30)) ^ (s2 >> s(3))) & s(0b11);
        let bits_31_30 = (s2 >> s(5)) ^ bits_29_28;
        (bits_31_30 << s(30)) | (bits_29_28 << s(28)) | (bits_27_03 << s(3))
    }

    /// Compute previous value for seed3 in Rand32
    pub fn prev_s3<const LANES: usize>(s3: Simd<u32, LANES>) -> Simd<u32, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let s = Simd::splat;
        let bits_14_04 = s3 >> s(21);
        let mut window = s3 >> s(4);
        let mut bits_31_15 = s(0);
        let mut chunk = (bits_14_04 >> s(8)) & s(0b111);
        for _ in 0..6 {
            chunk = (window ^ chunk) & s(0b111);
            window >>= s(3);
            bits_31_15 = (bits_31_15 >> s(3)) | (chunk << s(29));
        }
        (bits_31_15 << s(1)) | (bits_14_04 << s(4))
    }
}

/// Convert seed to a timestamp that can produce it
pub fn seed_to_timestamp(s: u32) -> u32 {
    // pow(1170746341, -1, 2**32) == 963516909
    s.wrapping_add(755606699).wrapping_mul(963516909)
}

/// Find the number of steps and the timestamp that produce a given Rand32 state
pub fn find_rng_timestamp(state: (u32, u32, u32)) -> (usize, [u32; 2]) {
    let (mut s1, mut s2, mut s3) = state;
    let m1 = !0x100001;
    let m2 = !0x1007;
    let m3 = !0x1f;
    let m12 = m1 & m2;
    let m13 = m1 & m3;

    let mut steps = 0;
    while (s1 ^ s2) & m12 != 0 || (s1 ^ s3) & m13 != 0 {
        steps += 1;
        s1 = prev_s1(s1);
        s2 = prev_s2(s2);
        s3 = prev_s3(s3);
    }

    let seed = s1 & s2 & !0b1;

    (
        steps,
        [seed_to_timestamp(seed), seed_to_timestamp(seed | 1)],
    )
}

#[cfg(test)]
mod tests {
    use super::{find_rng_timestamp, prev_s1, prev_s2, prev_s3};
    use crate::rand32::Rand32Ref;
    fn next_s1(s: u32) -> u32 {
        ((s & 0xFFFFFFFE) << 12) ^ (((s >> 13) ^ (s & 0x0007FFC0)) >> 6)
    }
    fn next_s2(s: u32) -> u32 {
        ((s & 0xFFFFFFF8) << 4) ^ (((s >> 2) ^ (s & 0x3F800000)) >> 23)
    }
    fn next_s3(s: u32) -> u32 {
        ((s & 0xFFFFFFF0) << 17) ^ (((s >> 3) ^ (s & 0x1FFFFF00)) >> 8)
    }

    #[test]
    fn test_prev_s1() {
        for ii in 0_u32..1 << 16 {
            let ii = (ii << 8) | 0x13371337;
            let mask = (!0) << 1;
            let next = next_s1(ii);
            let prev = prev_s1(next);
            assert_eq!(prev & mask, ii & mask);
        }
    }

    #[test]
    fn test_prev_s2() {
        for ii in 0_u32..1 << 16 {
            let ii = (ii << 8) | 0x13371337;
            let mask = (!0) << 3;
            let next = next_s2(ii);
            let prev = prev_s2(next);
            if prev & mask != ii & mask {
                println!("{:032b} {:032b}", prev & mask, ii & mask);
            }
            assert_eq!(prev & mask, ii & mask);
        }
    }

    #[test]
    fn test_prev_s3() {
        for ii in 0_u32..1 << 16 {
            let ii = (ii << 8) | 0x13371337;
            let mask = (!0) << 4;
            let next = next_s3(ii);
            let prev = prev_s3(next);
            assert_eq!(prev & mask, ii & mask);
        }
    }

    #[test]
    fn test_find_timestamp() {
        let mut rng = Rand32Ref::new(0x1337);
        for _ in 0..10_000_000 {
            rng.rand();
        }
        let (steps, ts) = find_rng_timestamp(rng.state());
        let expected = rng.rand();
        for t in ts {
            let mut rng = Rand32Ref::new(t);
            for _ in 0..steps {
                rng.rand();
            }
            assert_eq!(rng.rand(), expected);
        }
    }
}
