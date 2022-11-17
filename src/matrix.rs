use core::fmt;
#[cfg(feature = "simd")]
use core::simd::{LaneCount, Simd, SupportedLaneCount};

#[derive(Debug)]
pub enum XorShiftOp {
    Ident,
    Shr(u32),
    Shl(u32),
}

#[derive(Debug)]
pub enum Error {
    MatrixNotInvertible { stuck_at_col: usize },
}

macro_rules! impl_bitmatrix {
    ($name:ident, $ty:ty, $fmt_sz:expr, @simd) => {
        impl_bitmatrix!($name, $ty, $fmt_sz);

        #[cfg(feature = "simd")]
        impl<const LANES: usize> core::ops::Mul<$name> for Simd<$ty, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Output = Simd<$ty, LANES>;
            fn mul(self, matrix: $name) -> Self::Output {
                let s = Simd::splat;
                let mut out = s(0);
                let mut vec = self;
                for row in matrix.0 {
                    let bit = s(0) - (vec & s(1));
                    vec >>= s(1);
                    out ^= s(row) & bit;
                }
                out
            }
        }
    };

    ($name:ident, $ty:ty, $fmt_sz:expr) => {
        #[derive(Copy, Clone, PartialEq, Eq)]
        pub struct $name(pub [$ty; $name::BITS]);

        impl fmt::Debug for $name {
            fn fmt(
                &self,
                f: &mut fmt::Formatter<'_>,
            ) -> Result<(), fmt::Error> {
                f.write_str(concat!(stringify!($name), "(\n"))?;
                for row in self.0 {
                    writeln!(
                        f,
                        concat!("  {:0", stringify!($fmt_sz), "b}"),
                        row
                    )?
                }
                f.write_str(")")
            }
        }

        impl $name {
            fn op_mask(op: &XorShiftOp) -> $ty {
                use XorShiftOp::*;
                match op {
                    Ident => !0,
                    Shl(shl) => !0 >> shl,
                    Shr(shr) => !0 << shr,
                }
            }

            const BITS: usize = core::mem::size_of::<$ty>() * 8;

            pub fn zero() -> Self {
                Self([0; $name::BITS])
            }

            pub fn eye() -> Self {
                let arr = core::array::from_fn(|ii| 1 << ii as u32);
                Self(arr)
            }

            pub fn pow(&self, mut pow: usize) -> Self {
                let mut mul = *self;
                let mut out = Self::eye();
                while pow > 0 {
                    if pow & 1 != 0 {
                        out = out * mul;
                    }
                    mul = mul * mul;
                    pow >>= 1;
                }
                out
            }

            pub fn select(self, range: core::ops::Range<u32>) -> Self {
                let right = range.end - range.start;
                let left = range.start + right;
                (self >> right) << left
            }

            pub fn and(mut self, mask: $ty) -> Self {
                for val in self.0.iter_mut() {
                    *val &= mask;
                }
                self
            }

            pub fn shd(mut self, much: usize) -> Self {
                let arr = &mut self.0;
                for ii in much..$name::BITS {
                    arr[ii] = arr[ii - much];
                }
                for ii in 0..much {
                    arr[ii] = 0;
                }
                self
            }

            pub fn shu(mut self, much: usize) -> Self {
                let arr = &mut self.0;
                for ii in much..$name::BITS {
                    arr[ii - much] = arr[ii];
                }
                let delta = $name::BITS - much;
                for ii in 0..much {
                    arr[ii + delta] = 0;
                }
                self
            }

            pub fn vskip(mut self, range: core::ops::Range<usize>) -> Self {
                let arr = &mut self.0;
                let delta = range.end - range.start;
                let end = $name::BITS - delta;
                for ii in range.start..end {
                    arr[ii] = arr[ii + delta];
                }
                for ii in end..$name::BITS {
                    arr[ii] = 0;
                }
                self
            }

            pub fn hskip(self, range: core::ops::Range<usize>) -> Self {
                let mask_low = ($name::BITS - range.start) as u32;
                let mask_high = range.end as u32;
                let sh_low = range.start as u32;
                let lower = (self << mask_low) >> mask_low;
                let higher = if range.end == $name::BITS {
                    Self::zero()
                } else {
                    (self >> mask_high) << sh_low
                };
                lower ^ higher
            }

            pub fn shr(bits: u32) -> Self {
                Self::eye() >> bits
            }

            pub fn shl(bits: u32) -> Self {
                Self::eye() << bits
            }

            pub fn inv(&self) -> Result<Self, Error> {
                let mut tmp = self.0;
                let mut inv = Self::eye().0;

                // reduce to triangular form
                for ii in 0..$name::BITS {
                    // find first non-zero start, swap rows if neede
                    let mask = 1 << ii;
                    if tmp[ii] & mask == 0 {
                        let pos = (ii + 1..$name::BITS)
                            .find(|&jj| tmp[jj] & mask != 0)
                            .ok_or(Error::MatrixNotInvertible {
                                stuck_at_col: ii,
                            })?;
                        tmp.swap(ii, pos);
                        inv.swap(ii, pos);
                    }

                    // clear this bit from the remaining rows
                    for jj in (ii + 1)..$name::BITS {
                        if (tmp[jj] >> ii) & 1 != 0 {
                            tmp[jj] ^= tmp[ii];
                            inv[jj] ^= inv[ii];
                        }
                    }
                }

                // reduce to diagonal
                for ii in 0..$name::BITS {
                    for jj in (ii + 1)..$name::BITS {
                        if (tmp[ii] >> jj) & 1 != 0 {
                            tmp[ii] ^= tmp[jj];
                            inv[ii] ^= inv[jj];
                        }
                    }
                }

                Ok(Self(inv))
            }

            pub fn xorshift_form(&self) -> Vec<(Option<$ty>, XorShiftOp)> {
                let mut tmp = self.0;
                let mut result = Vec::new();
                for ii in 0..$name::BITS {
                    for jj in 0..$name::BITS {
                        let mask = 1 << jj;
                        if tmp[ii] & mask == 0 {
                            continue;
                        }
                        let len = $name::BITS - ii.max(jj);
                        let mut last = 0;
                        for kk in 0..len {
                            let mask = 1 << (jj + kk);
                            if tmp[ii + kk] & mask == 0 {
                                break;
                            }
                            tmp[ii + kk] ^= mask;
                            last = kk + 1;
                        }
                        use core::cmp::Ordering::*;
                        use XorShiftOp::*;
                        let op = match ii.cmp(&jj) {
                            Equal => Ident,
                            Less => Shl((jj - ii) as u32),
                            Greater => Shr((ii - jj) as u32),
                        };
                        let mask = ((!0) >> ($name::BITS - last)) << ii;
                        let mask = if mask == $name::op_mask(&op) {
                            None
                        } else {
                            Some(mask)
                        };
                        result.push((mask, op));
                    }
                }
                result
            }
        }

        impl core::ops::BitXor for $name {
            type Output = $name;
            fn bitxor(self, other: Self) -> Self::Output {
                let arr = core::array::from_fn(|ii| self.0[ii] ^ other.0[ii]);
                $name(arr)
            }
        }

        impl core::ops::Mul for $name {
            type Output = $name;
            fn mul(self, other: Self) -> Self::Output {
                let arr = core::array::from_fn(|ii| self.0[ii] * other);
                $name(arr)
            }
        }

        impl core::ops::Mul<$name> for $ty {
            type Output = $ty;
            fn mul(self, matrix: $name) -> Self::Output {
                let mut out = 0;
                let mut vec = self;
                for row in matrix.0 {
                    let bit = (vec & 1).wrapping_neg();
                    vec >>= 1;
                    out ^= row & bit;
                }
                out
            }
        }

        impl core::ops::Shr<u32> for $name {
            type Output = $name;
            fn shr(self, bits: u32) -> Self::Output {
                let arr = core::array::from_fn(|ii| self.0[ii] >> bits);
                $name(arr)
            }
        }

        impl core::ops::Shl<u32> for $name {
            type Output = $name;
            fn shl(self, bits: u32) -> Self::Output {
                let arr = core::array::from_fn(|ii| self.0[ii] << bits);
                $name(arr)
            }
        }
    };
}

impl_bitmatrix!(BitMatrix32, u32, 32, @simd);
impl_bitmatrix!(BitMatrix64, u64, 64, @simd);
impl_bitmatrix!(BitMatrix128, u128, 128);

#[test]
fn test_inv() {
    use BitMatrix64 as M;
    let matrix = M::eye() ^ M::shl(4);
    let inv = matrix.inv().unwrap();
    assert_eq!(0x1337133713371337 * matrix * inv, 0x1337133713371337);
}
