use crate::matrix::{
    BitMatrix128 as M128, BitMatrix32 as M32, Error as MatError,
};
use crate::rand32::rng_matrix;

fn build_inv_matrix() -> Result<M128, MatError> {
    let [m1, m2, m3] = rng_matrix();
    let pows: [[M32; 3]; 4] = core::array::from_fn(|ii| {
        let pow = ii + 1;
        [m1.pow(pow), m2.pow(pow), m3.pow(pow)]
    });
    let mut arr = [0_u128; 128];
    for mat in 0..3 {
        for row in 0..32 {
            let mut row_val = 0;
            for (ii, pow) in pows.iter().enumerate() {
                let val = pow[mat].0[row] as u128;
                let val = val << (ii * 32);
                row_val |= val;
            }

            let o_row = mat * 32 + row;
            arr[o_row] = row_val;
        }
    }
    let mut mat = M128(arr);
    // skip under-constrained input bits
    mat = mat.vskip(64..68).vskip(32..35).vskip(0..1);
    // skip under-specified output bits (&0x00ffffff)
    // note 96+4
    mat = mat
        .hskip(88..96 + 4)
        .hskip(24..32)
        // skip over-specified input
        .hskip(88..128);
    // fill the rest with ones;
    for ii in 88..128 {
        mat.0[ii] = 1 << ii;
    }
    mat.inv()
}

fn two_u56_to_vector(val1: u64, val2: u64) -> u128 {
    let v0 = (val1 >> 32) as u128;
    let v1 = ((val1 << 32) >> 32) as u128;
    let v2 = (val2 >> 32) as u128;
    // note 4 skipped bits
    let v3 = ((val2 << 32) >> (32 + 4)) as u128;
    let vector = v0 | (v1 << 24) | (v2 << 56) | (v3 << 80);
    // mask_off extra bits
    let mask = 128 - 88;
    (vector << mask) >> mask
}

fn vector_to_seed(mut vector: u128) -> (u32, u32, u32) {
    let max_u32 = 0xffff_ffff;
    let s1 = ((vector << 1) & max_u32) as u32;
    vector >>= 32 - 1;
    let s2 = ((vector << 3) & max_u32) as u32;
    vector >>= 32 - 3;
    let s3 = ((vector << 4) & max_u32) as u32;
    (s1, s2, s3)
}

pub struct U56ToSeed {
    inv_matrix: M128,
}

impl U56ToSeed {
    pub fn new() -> Self {
        let inv_matrix = build_inv_matrix()
            .expect("could not build inverse matrix, blame Desu");

        Self { inv_matrix }
    }

    pub fn solve(&self, val1: u64, val2: u64) -> (u32, u32, u32) {
        assert!(
            0 == val1 >> 56 && 0 == val2 >> 56,
            "the observed u56 values must not have high 8 bits set"
        );
        let vector = two_u56_to_vector(val1, val2);
        let out_vector = vector * self.inv_matrix;
        vector_to_seed(out_vector)
    }
}

impl Default for U56ToSeed {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use super::U56ToSeed;
    use crate::rand32::Rand32Ref;

    #[test]
    fn test_example() {
        let u56_solver = U56ToSeed::new();
        let val1 = 0x00d269af_632d45f3;
        let val2 = 0x0009ad9b_493e4d35;
        let state = u56_solver.solve(val1, val2);
        let mut rng = Rand32Ref::from_state(state);
        let val1_ref = (((rng.rand() << 8) as u64) << 24) | rng.rand() as u64;
        assert_eq!(val1, val1_ref);
        let val2_ref = (((rng.rand() << 8) as u64) << 24) | rng.rand() as u64;
        assert_eq!(val2, val2_ref);
    }
}
