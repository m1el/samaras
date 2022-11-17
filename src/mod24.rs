use crate::matrix::{
    BitMatrix128 as M128, BitMatrix32 as M32, Error as MatError,
};
use crate::rand32::{rng_matrix, Rand32Ref};

const MAX_INDEX: usize = 30;
fn build_inv_matrix() -> Result<M128, MatError> {
    let [m1, m2, m3] = rng_matrix();
    let pows: [[M32; 3]; MAX_INDEX] = core::array::from_fn(|ii| {
        let pow = ii + 1;
        [m1.pow(pow), m2.pow(pow), m3.pow(pow)]
    });

    let mut arr = [0_u128; 128];
    for mat in 0..3 {
        for row in 0..32 {
            let mut row_val = 0;
            for (ii, pow) in pows.iter().enumerate() {
                let val = pow[mat].0[row] as u128;
                let val = (val & 0b111) << (ii * 3);
                row_val |= val;
            }

            let o_row = mat * 32 + row;
            arr[o_row] = row_val;
        }
    }
    let mut mat = M128(arr);

    // skip under-constrained input bits
    mat = mat.vskip(64..68).vskip(32..35).vskip(0..1);
    // drop extra bits
    mat = mat.hskip(88..128);

    // fill the rest with ones;
    for ii in 88..128 {
        mat.0[ii] = 1 << ii;
    }

    mat.inv()
}

fn produce_mod24(
    rng: &mut Rand32Ref,
    count: usize,
) -> impl Iterator<Item = u32> + '_ {
    (0..count).map(|_| rng.rand() % 24)
}

fn mod24_to_vector(mod24: &[u32]) -> u128 {
    assert!(mod24.len() > 20, "must provide at least 20 values");
    let mut vec = 0_u128;
    for (ii, m24) in mod24[..MAX_INDEX.min(mod24.len())].iter().enumerate() {
        let value = (m24 & 0b111) as u128;
        vec |= value << (ii * 3);
    }
    let mask = 128 - 88;
    (vec << mask) >> mask
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

pub struct Mod24Solver {
    inv_matrix: M128,
}

#[derive(Debug)]
pub enum Error {
    SequenceTooShort,
    ValidationFailed,
    NotFound,
}

impl Mod24Solver {
    pub fn new() -> Self {
        let inv_matrix = build_inv_matrix()
            .expect("could not build inverse matrix, blame Desu");

        Self { inv_matrix }
    }

    pub fn solve(&self, sequence: &[u32]) -> Result<(u32, u32, u32), Error> {
        let known_values = sequence.len();
        if known_values < 20 {
            return Err(Error::SequenceTooShort);
        }
        let known_bits = known_values * 3;
        let unknown_bits = 88_usize.saturating_sub(known_bits);
        if unknown_bits == 0 {
            let vector = mod24_to_vector(sequence);
            let out_vector = vector * self.inv_matrix;
            let state = vector_to_seed(out_vector);
            let mut rng = Rand32Ref::from_state(state);

            let generated = produce_mod24(&mut rng, known_values);
            if sequence.iter().copied().eq(generated) {
                Ok(state)
            } else {
                Err(Error::ValidationFailed)
            }
        } else {
            let vector = mod24_to_vector(sequence);
            for brute in 0..(1 << unknown_bits) {
                let vector = vector | (brute << known_bits);
                let out_vector = vector * self.inv_matrix;
                let state = vector_to_seed(out_vector);
                let mut rng = Rand32Ref::from_state(state);

                let generated = produce_mod24(&mut rng, known_values);
                if sequence.iter().copied().eq(generated) {
                    return Ok(state);
                }
            }

            Err(Error::NotFound)
        }
    }
}

impl Default for Mod24Solver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{produce_mod24, Mod24Solver, Rand32Ref};

    #[test]
    fn test_example() {
        let mod24_solver = Mod24Solver::new();
        let s1_mask = !0x100001;
        let s2_mask = !0x1007;
        let s3_mask = !0x1f;
        let mut mod24s = Vec::new();
        for seed in 0..0xffff {
            let seed = (seed << 8) ^ 0x13371337;
            let mut rng = Rand32Ref::seeded(seed, seed, seed);
            mod24s.clear();
            mod24s.extend(produce_mod24(&mut rng, 40));
            let (s1, s2, s3) =
                mod24_solver.solve(&mod24s).expect("should find a solution");
            assert_eq!(s1 & s1_mask, seed & s1_mask);
            assert_eq!(s2 & s2_mask, seed & s2_mask);
            assert_eq!(s3 & s3_mask, seed & s3_mask);
        }
    }

    #[test]
    fn test_brute() {
        let mod24_solver = Mod24Solver::new();
        let seed = 0x13371337;
        let s1_mask = !0x100001;
        let s2_mask = !0x1007;
        let s3_mask = !0x1f;
        let mut mod24s = Vec::new();
        for len in 22..30 {
            let mut rng = Rand32Ref::seeded(seed, seed, seed);
            mod24s.clear();
            mod24s.extend(produce_mod24(&mut rng, len));
            let (s1, s2, s3) =
                mod24_solver.solve(&mod24s).expect("should find a solution");
            assert_eq!(s1 & s1_mask, seed & s1_mask);
            assert_eq!(s2 & s2_mask, seed & s2_mask);
            assert_eq!(s3 & s3_mask, seed & s3_mask);
        }
    }
}
