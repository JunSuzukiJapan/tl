//! テンソル要素型トレイト
//!
//! `CpuTensor<T>` のジェネリックパラメータ `T` が満たすべきトレイト。
//! f32, f64 に対して実装する。

use std::fmt;
use std::ops;

/// テンソルの要素型が満たすべきトレイト
pub trait TensorScalar:
    Copy
    + Default
    + 'static
    + Send
    + Sync
    + fmt::Debug
    + fmt::Display
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Mul<Output = Self>
    + ops::Div<Output = Self>
    + ops::Rem<Output = Self>
    + ops::Neg<Output = Self>
    + ops::AddAssign
    + ops::SubAssign
    + ops::MulAssign
    + ops::DivAssign
    + PartialOrd
    + std::iter::Sum
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
    fn abs(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn tanh(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
    fn neg_infinity() -> Self;
    fn infinity() -> Self;
    fn pi() -> Self;
    fn frac_2_sqrt_pi() -> Self;
    fn size_in_bytes() -> usize;
    /// 乱数生成用: [0, 1) の一様乱数を返す
    fn gen_uniform(rng: &mut impl rand::Rng) -> Self;
    /// usize への変換 (インデックス用)
    fn to_usize(self) -> usize;
}

impl TensorScalar for f32 {
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn one() -> Self { 1.0 }
    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
    #[inline] fn from_f32(v: f32) -> Self { v }
    #[inline] fn to_f32(self) -> f32 { self }
    #[inline] fn abs(self) -> Self { f32::abs(self) }
    #[inline] fn exp(self) -> Self { f32::exp(self) }
    #[inline] fn ln(self) -> Self { f32::ln(self) }
    #[inline] fn sqrt(self) -> Self { f32::sqrt(self) }
    #[inline] fn sin(self) -> Self { f32::sin(self) }
    #[inline] fn cos(self) -> Self { f32::cos(self) }
    #[inline] fn tan(self) -> Self { f32::tan(self) }
    #[inline] fn tanh(self) -> Self { f32::tanh(self) }
    #[inline] fn powf(self, n: Self) -> Self { f32::powf(self, n) }
    #[inline] fn powi(self, n: i32) -> Self { f32::powi(self, n) }
    #[inline] fn clamp(self, min: Self, max: Self) -> Self { f32::clamp(self, min, max) }
    #[inline] fn neg_infinity() -> Self { f32::NEG_INFINITY }
    #[inline] fn infinity() -> Self { f32::INFINITY }
    #[inline] fn pi() -> Self { std::f32::consts::PI }
    #[inline] fn frac_2_sqrt_pi() -> Self { std::f32::consts::FRAC_2_SQRT_PI }
    #[inline] fn size_in_bytes() -> usize { 4 }
    #[inline] fn gen_uniform(rng: &mut impl rand::Rng) -> Self { rng.gen() }
    #[inline] fn to_usize(self) -> usize { self as usize }
}

impl TensorScalar for f64 {
    #[inline] fn zero() -> Self { 0.0 }
    #[inline] fn one() -> Self { 1.0 }
    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn to_f64(self) -> f64 { self }
    #[inline] fn from_f32(v: f32) -> Self { v as f64 }
    #[inline] fn to_f32(self) -> f32 { self as f32 }
    #[inline] fn abs(self) -> Self { f64::abs(self) }
    #[inline] fn exp(self) -> Self { f64::exp(self) }
    #[inline] fn ln(self) -> Self { f64::ln(self) }
    #[inline] fn sqrt(self) -> Self { f64::sqrt(self) }
    #[inline] fn sin(self) -> Self { f64::sin(self) }
    #[inline] fn cos(self) -> Self { f64::cos(self) }
    #[inline] fn tan(self) -> Self { f64::tan(self) }
    #[inline] fn tanh(self) -> Self { f64::tanh(self) }
    #[inline] fn powf(self, n: Self) -> Self { f64::powf(self, n) }
    #[inline] fn powi(self, n: i32) -> Self { f64::powi(self, n) }
    #[inline] fn clamp(self, min: Self, max: Self) -> Self { f64::clamp(self, min, max) }
    #[inline] fn neg_infinity() -> Self { f64::NEG_INFINITY }
    #[inline] fn infinity() -> Self { f64::INFINITY }
    #[inline] fn pi() -> Self { std::f64::consts::PI }
    #[inline] fn frac_2_sqrt_pi() -> Self { std::f64::consts::FRAC_2_SQRT_PI }
    #[inline] fn size_in_bytes() -> usize { 8 }
    #[inline] fn gen_uniform(rng: &mut impl rand::Rng) -> Self { rng.gen() }
    #[inline] fn to_usize(self) -> usize { self as usize }
}
