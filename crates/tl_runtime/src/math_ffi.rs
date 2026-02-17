//! math_ffi - 数学関数 FFI
//!
//! f32, f64, i32, i64 の数学関数を提供

// ========== f32 Functions ==========

#[unsafe(no_mangle)]
/// @ffi_sig (f32) -> f32
pub extern "C" fn tl_f32_abs(x: f32) -> f32 { x.abs() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_acos(x: f32) -> f32 { x.acos() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_acosh(x: f32) -> f32 { x.acosh() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_asin(x: f32) -> f32 { x.asin() }

#[unsafe(no_mangle)]
/// @ffi_sig (f32) -> f32
pub extern "C" fn tl_f32_asinh(x: f32) -> f32 { x.asinh() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_atan(x: f32) -> f32 { x.atan() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_atan2(y: f32, x: f32) -> f32 { y.atan2(x) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_atanh(x: f32) -> f32 { x.atanh() }

#[unsafe(no_mangle)]
/// @ffi_sig (f32) -> f32
pub extern "C" fn tl_f32_cbrt(x: f32) -> f32 { x.cbrt() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_ceil(x: f32) -> f32 { x.ceil() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_copysign(x: f32, y: f32) -> f32 { x.copysign(y) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_cos(x: f32) -> f32 { x.cos() }

#[unsafe(no_mangle)]
/// @ffi_sig (f32) -> f32
pub extern "C" fn tl_f32_cosh(x: f32) -> f32 { x.cosh() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_exp(x: f32) -> f32 { x.exp() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_exp2(x: f32) -> f32 { x.exp2() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_exp_m1(x: f32) -> f32 { x.exp_m1() }

#[unsafe(no_mangle)]
/// @ffi_sig (f32) -> f32
pub extern "C" fn tl_f32_floor(x: f32) -> f32 { x.floor() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_fract(x: f32) -> f32 { x.fract() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_hypot(x: f32, y: f32) -> f32 { x.hypot(y) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_ln(x: f32) -> f32 { x.ln() }

#[unsafe(no_mangle)]
/// @ffi_sig (f32) -> f32
pub extern "C" fn tl_f32_ln_1p(x: f32) -> f32 { x.ln_1p() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_log(x: f32, base: f32) -> f32 { x.log(base) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_log10(x: f32) -> f32 { x.log10() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_log2(x: f32) -> f32 { x.log2() }

#[unsafe(no_mangle)]
/// @ffi_sig (f32, f32) -> f32
pub extern "C" fn tl_f32_powf(x: f32, n: f32) -> f32 { x.powf(n) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_powi(x: f32, n: i32) -> f32 { x.powi(n) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_recip(x: f32) -> f32 { x.recip() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_round(x: f32) -> f32 { x.round() }

#[unsafe(no_mangle)]
/// @ffi_sig (f32) -> f32
pub extern "C" fn tl_f32_signum(x: f32) -> f32 { x.signum() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_sin(x: f32) -> f32 { x.sin() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_sinh(x: f32) -> f32 { x.sinh() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_sqrt(x: f32) -> f32 { x.sqrt() }

#[unsafe(no_mangle)]
/// @ffi_sig (f32) -> f32
pub extern "C" fn tl_f32_tan(x: f32) -> f32 { x.tan() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_tanh(x: f32) -> f32 { x.tanh() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_to_degrees(x: f32) -> f32 { x.to_degrees() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f32_to_radians(x: f32) -> f32 { x.to_radians() }

#[unsafe(no_mangle)]
/// @ffi_sig (f32) -> f32
pub extern "C" fn tl_f32_trunc(x: f32) -> f32 { x.trunc() }

// ========== f64 Functions ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_abs(x: f64) -> f64 { x.abs() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_acos(x: f64) -> f64 { x.acos() }

#[unsafe(no_mangle)]
/// @ffi_sig (f64) -> f64
pub extern "C" fn tl_f64_acosh(x: f64) -> f64 { x.acosh() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_asin(x: f64) -> f64 { x.asin() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_asinh(x: f64) -> f64 { x.asinh() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_atan(x: f64) -> f64 { x.atan() }

#[unsafe(no_mangle)]
/// @ffi_sig (f64, f64) -> f64
pub extern "C" fn tl_f64_atan2(y: f64, x: f64) -> f64 { y.atan2(x) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_atanh(x: f64) -> f64 { x.atanh() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_cbrt(x: f64) -> f64 { x.cbrt() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_ceil(x: f64) -> f64 { x.ceil() }

#[unsafe(no_mangle)]
/// @ffi_sig (f64, f64) -> f64
pub extern "C" fn tl_f64_copysign(x: f64, y: f64) -> f64 { x.copysign(y) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_cos(x: f64) -> f64 { x.cos() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_cosh(x: f64) -> f64 { x.cosh() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_exp(x: f64) -> f64 { x.exp() }

#[unsafe(no_mangle)]
/// @ffi_sig (f64) -> f64
pub extern "C" fn tl_f64_exp2(x: f64) -> f64 { x.exp2() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_exp_m1(x: f64) -> f64 { x.exp_m1() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_floor(x: f64) -> f64 { x.floor() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_fract(x: f64) -> f64 { x.fract() }

#[unsafe(no_mangle)]
/// @ffi_sig (f64, f64) -> f64
pub extern "C" fn tl_f64_hypot(x: f64, y: f64) -> f64 { x.hypot(y) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_ln(x: f64) -> f64 { x.ln() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_ln_1p(x: f64) -> f64 { x.ln_1p() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_log(x: f64, base: f64) -> f64 { x.log(base) }

#[unsafe(no_mangle)]
/// @ffi_sig (f64) -> f64
pub extern "C" fn tl_f64_log10(x: f64) -> f64 { x.log10() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_log2(x: f64) -> f64 { x.log2() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_powf(x: f64, n: f64) -> f64 { x.powf(n) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_powi(x: f64, n: i32) -> f64 { x.powi(n) }

#[unsafe(no_mangle)]
/// @ffi_sig (f64) -> f64
pub extern "C" fn tl_f64_recip(x: f64) -> f64 { x.recip() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_round(x: f64) -> f64 { x.round() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_signum(x: f64) -> f64 { x.signum() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_sin(x: f64) -> f64 { x.sin() }

#[unsafe(no_mangle)]
/// @ffi_sig (f64) -> f64
pub extern "C" fn tl_f64_sinh(x: f64) -> f64 { x.sinh() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_sqrt(x: f64) -> f64 { x.sqrt() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_tan(x: f64) -> f64 { x.tan() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_tanh(x: f64) -> f64 { x.tanh() }

#[unsafe(no_mangle)]
/// @ffi_sig (f64) -> f64
pub extern "C" fn tl_f64_to_degrees(x: f64) -> f64 { x.to_degrees() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_to_radians(x: f64) -> f64 { x.to_radians() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_f64_trunc(x: f64) -> f64 { x.trunc() }

// ========== i32 Functions ==========

#[unsafe(no_mangle)]
/// @ffi_sig (i32) -> i32
pub extern "C" fn tl_i32_abs(x: i32) -> i32 { x.abs() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_i32_div_euclid(x: i32, y: i32) -> i32 { x.div_euclid(y) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_i32_is_negative(x: i32) -> bool { x.is_negative() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_i32_is_positive(x: i32) -> bool { x.is_positive() }

#[unsafe(no_mangle)]
/// @ffi_sig (i32, u32) -> i32
pub extern "C" fn tl_i32_pow(x: i32, n: u32) -> i32 { x.pow(n) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_i32_rem_euclid(x: i32, y: i32) -> i32 { x.rem_euclid(y) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_i32_signum(x: i32) -> i32 { x.signum() }

// ========== i64 Functions ==========

#[unsafe(no_mangle)]
/// @ffi_sig (i64) -> i64
pub extern "C" fn tl_i64_abs(x: i64) -> i64 { x.abs() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_i64_div_euclid(x: i64, y: i64) -> i64 { x.div_euclid(y) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_i64_is_negative(x: i64) -> bool { x.is_negative() }

#[unsafe(no_mangle)]
pub extern "C" fn tl_i64_is_positive(x: i64) -> bool { x.is_positive() }

#[unsafe(no_mangle)]
/// @ffi_sig (i64, u32) -> i64
pub extern "C" fn tl_i64_pow(x: i64, n: u32) -> i64 { x.pow(n) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_i64_rem_euclid(x: i64, y: i64) -> i64 { x.rem_euclid(y) }

#[unsafe(no_mangle)]
pub extern "C" fn tl_i64_signum(x: i64) -> i64 { x.signum() }
