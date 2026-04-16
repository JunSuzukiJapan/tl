use crate::compiler::error::{TlError, CodegenErrorKind};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::{BasicValueEnum, ValueKind};

// ========== F32 メソッド ==========

/// F32 unary method: tl_f32_{method}(self) -> f32
fn compile_f32_unary<'ctx>(
    c: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    method: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_name = format!("tl_f32_{}", method);
    let fn_val = c.module.get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
    let call = c.builder.build_call(fn_val, &[obj.into()], &format!("f32_{}", method))
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid return from {}", fn_name)))),
    };
    Ok((res, Type::F32))
}

/// F32 binary method: tl_f32_{method}(self, other) -> f32
fn compile_f32_binary<'ctx>(
    c: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    method: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(TlError::from(CodegenErrorKind::Internal(format!("f32.{} requires 1 argument", method)))); }
    let fn_name = format!("tl_f32_{}", method);
    let fn_val = c.module.get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
    let call = c.builder.build_call(fn_val, &[obj.into(), args[0].0.into()], &format!("f32_{}", method))
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid return from {}", fn_name)))),
    };
    Ok((res, Type::F32))
}

// F32 unary methods
macro_rules! f32_unary {
    ($name:ident, $method:literal) => {
        pub fn $name<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
            compile_f32_unary(c, o, $method)
        }
    };
}

f32_unary!(compile_f32_abs, "abs");
f32_unary!(compile_f32_acos, "acos");
f32_unary!(compile_f32_acosh, "acosh");
f32_unary!(compile_f32_asin, "asin");
f32_unary!(compile_f32_asinh, "asinh");
f32_unary!(compile_f32_atan, "atan");
f32_unary!(compile_f32_atanh, "atanh");
f32_unary!(compile_f32_cbrt, "cbrt");
f32_unary!(compile_f32_ceil, "ceil");
f32_unary!(compile_f32_cos, "cos");
f32_unary!(compile_f32_cosh, "cosh");
f32_unary!(compile_f32_exp, "exp");
f32_unary!(compile_f32_exp2, "exp2");
f32_unary!(compile_f32_exp_m1, "exp_m1");
f32_unary!(compile_f32_floor, "floor");
f32_unary!(compile_f32_fract, "fract");
f32_unary!(compile_f32_ln, "ln");
f32_unary!(compile_f32_ln_1p, "ln_1p");
f32_unary!(compile_f32_log, "log");
f32_unary!(compile_f32_log10, "log10");
f32_unary!(compile_f32_log2, "log2");
f32_unary!(compile_f32_recip, "recip");
f32_unary!(compile_f32_round, "round");
f32_unary!(compile_f32_signum, "signum");
f32_unary!(compile_f32_sin, "sin");
f32_unary!(compile_f32_sinh, "sinh");
f32_unary!(compile_f32_sqrt, "sqrt");
f32_unary!(compile_f32_tan, "tan");
f32_unary!(compile_f32_tanh, "tanh");
f32_unary!(compile_f32_to_degrees, "to_degrees");
f32_unary!(compile_f32_to_radians, "to_radians");
f32_unary!(compile_f32_trunc, "trunc");

// F32 binary methods
macro_rules! f32_binary {
    ($name:ident, $method:literal) => {
        pub fn $name<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
            compile_f32_binary(c, o, a, $method)
        }
    };
}

f32_binary!(compile_f32_atan2, "atan2");
f32_binary!(compile_f32_copysign, "copysign");
f32_binary!(compile_f32_hypot, "hypot");
f32_binary!(compile_f32_powf, "powf");
f32_binary!(compile_f32_pow, "powf");  // pow delegates to powf for F32
f32_binary!(compile_f32_powi, "powi");

// ========== F64 メソッド ==========

fn compile_f64_unary<'ctx>(
    c: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    method: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_name = format!("tl_f64_{}", method);
    let fn_val = c.module.get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
    let call = c.builder.build_call(fn_val, &[obj.into()], &format!("f64_{}", method))
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid return from {}", fn_name)))),
    };
    Ok((res, Type::F64))
}

fn compile_f64_binary<'ctx>(
    c: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    method: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(TlError::from(CodegenErrorKind::Internal(format!("f64.{} requires 1 argument", method)))); }
    let fn_name = format!("tl_f64_{}", method);
    let fn_val = c.module.get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
    let call = c.builder.build_call(fn_val, &[obj.into(), args[0].0.into()], &format!("f64_{}", method))
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid return from {}", fn_name)))),
    };
    Ok((res, Type::F64))
}

macro_rules! f64_unary {
    ($name:ident, $method:literal) => {
        pub fn $name<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
            compile_f64_unary(c, o, $method)
        }
    };
}

f64_unary!(compile_f64_abs, "abs");
f64_unary!(compile_f64_acos, "acos");
f64_unary!(compile_f64_acosh, "acosh");
f64_unary!(compile_f64_asin, "asin");
f64_unary!(compile_f64_asinh, "asinh");
f64_unary!(compile_f64_atan, "atan");
f64_unary!(compile_f64_atanh, "atanh");
f64_unary!(compile_f64_cbrt, "cbrt");
f64_unary!(compile_f64_ceil, "ceil");
f64_unary!(compile_f64_cos, "cos");
f64_unary!(compile_f64_cosh, "cosh");
f64_unary!(compile_f64_exp, "exp");
f64_unary!(compile_f64_exp2, "exp2");
f64_unary!(compile_f64_exp_m1, "exp_m1");
f64_unary!(compile_f64_floor, "floor");
f64_unary!(compile_f64_fract, "fract");
f64_unary!(compile_f64_ln, "ln");
f64_unary!(compile_f64_ln_1p, "ln_1p");
f64_unary!(compile_f64_log, "log");
f64_unary!(compile_f64_log10, "log10");
f64_unary!(compile_f64_log2, "log2");
f64_unary!(compile_f64_recip, "recip");
f64_unary!(compile_f64_round, "round");
f64_unary!(compile_f64_signum, "signum");
f64_unary!(compile_f64_sin, "sin");
f64_unary!(compile_f64_sinh, "sinh");
f64_unary!(compile_f64_sqrt, "sqrt");
f64_unary!(compile_f64_tan, "tan");
f64_unary!(compile_f64_tanh, "tanh");
f64_unary!(compile_f64_to_degrees, "to_degrees");
f64_unary!(compile_f64_to_radians, "to_radians");
f64_unary!(compile_f64_trunc, "trunc");

macro_rules! f64_binary {
    ($name:ident, $method:literal) => {
        pub fn $name<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
            compile_f64_binary(c, o, a, $method)
        }
    };
}

f64_binary!(compile_f64_atan2, "atan2");
f64_binary!(compile_f64_copysign, "copysign");
f64_binary!(compile_f64_hypot, "hypot");
f64_binary!(compile_f64_powf, "powf");
f64_binary!(compile_f64_pow, "powf");
f64_binary!(compile_f64_powi, "powi");

// ========== I64 メソッド ==========

fn compile_i64_unary<'ctx>(
    c: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    method: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_name = format!("tl_i64_{}", method);
    let fn_val = c.module.get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
    let call = c.builder.build_call(fn_val, &[obj.into()], &format!("i64_{}", method))
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid return from {}", fn_name)))),
    };
    Ok((res, Type::I64))
}

fn compile_i64_binary<'ctx>(
    c: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    method: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(TlError::from(CodegenErrorKind::Internal(format!("i64.{} requires 1 argument", method)))); }
    let fn_name = format!("tl_i64_{}", method);
    let fn_val = c.module.get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
    let call = c.builder.build_call(fn_val, &[obj.into(), args[0].0.into()], &format!("i64_{}", method))
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid return from {}", fn_name)))),
    };
    Ok((res, Type::I64))
}

fn compile_i64_bool_unary<'ctx>(
    c: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    method: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_name = format!("tl_i64_{}", method);
    let fn_val = c.module.get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
    let call = c.builder.build_call(fn_val, &[obj.into()], &format!("i64_{}", method))
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid return from {}", fn_name)))),
    };
    Ok((res, Type::Bool))
}

pub fn compile_i64_abs<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i64_unary(c, o, "abs") }
pub fn compile_i64_signum<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i64_unary(c, o, "signum") }
pub fn compile_i64_is_positive<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i64_bool_unary(c, o, "is_positive") }
pub fn compile_i64_is_negative<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i64_bool_unary(c, o, "is_negative") }
pub fn compile_i64_div_euclid<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i64_binary(c, o, a, "div_euclid") }
pub fn compile_i64_rem_euclid<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i64_binary(c, o, a, "rem_euclid") }
pub fn compile_i64_pow<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i64_binary(c, o, a, "pow") }

// get_offset, sumall — no-op / identity (return self)
pub fn compile_i64_get_offset<'ctx>(_c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { Ok((o, Type::I64)) }
pub fn compile_i64_sumall<'ctx>(_c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { Ok((o, Type::I64)) }

// ========== 型変換メソッド (Compiler-intrinsic) ==========

/// i64.to_f64() -> f64
pub fn compile_i64_to_f64<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let res = c.builder.build_signed_int_to_float(o.into_int_value(), c.context.f64_type(), "i64_to_f64")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::F64))
}

/// i64.to_f32() -> f32
pub fn compile_i64_to_f32<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let res = c.builder.build_signed_int_to_float(o.into_int_value(), c.context.f32_type(), "i64_to_f32")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::F32))
}

/// i64.to_string() -> String (via FFI)
pub fn compile_i64_to_string<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = c.module.get_function("tl_string_from_int")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_string_from_int not found".to_string())))?;
    let call = c.builder.build_call(fn_val, &[o.into()], "i64_to_string")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from i64.to_string".to_string()).into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// i64.min(other: i64) -> i64
pub fn compile_i64_min<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if a.len() != 1 { return Err(CodegenErrorKind::Internal("i64.min requires 1 argument".to_string()).into()); }
    let lhs = o.into_int_value();
    let rhs = a[0].0.into_int_value();
    let cmp = c.builder.build_int_compare(inkwell::IntPredicate::SLT, lhs, rhs, "cmp_min")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = c.builder.build_select(cmp, lhs, rhs, "min_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::I64))
}

/// i64.max(other: i64) -> i64
pub fn compile_i64_max<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if a.len() != 1 { return Err(CodegenErrorKind::Internal("i64.max requires 1 argument".to_string()).into()); }
    let lhs = o.into_int_value();
    let rhs = a[0].0.into_int_value();
    let cmp = c.builder.build_int_compare(inkwell::IntPredicate::SGT, lhs, rhs, "cmp_max")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = c.builder.build_select(cmp, lhs, rhs, "max_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::I64))
}

/// i64.clamp(min: i64, max: i64) -> i64
pub fn compile_i64_clamp<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if a.len() != 2 { return Err(CodegenErrorKind::Internal("i64.clamp requires 2 arguments".to_string()).into()); }
    let val = o.into_int_value();
    let lo = a[0].0.into_int_value();
    let hi = a[1].0.into_int_value();
    // clamp = max(lo, min(val, hi))
    let cmp_hi = c.builder.build_int_compare(inkwell::IntPredicate::SLT, val, hi, "cmp_hi")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let min_val = c.builder.build_select(cmp_hi, val, hi, "min_val")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
    let cmp_lo = c.builder.build_int_compare(inkwell::IntPredicate::SGT, min_val, lo, "cmp_lo")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = c.builder.build_select(cmp_lo, min_val, lo, "clamp_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::I64))
}

/// f32.to_f64() -> f64
pub fn compile_f32_to_f64<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let res = c.builder.build_float_ext(o.into_float_value(), c.context.f64_type(), "f32_to_f64")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::F64))
}

/// f32.to_i64() -> i64
pub fn compile_f32_to_i64<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let res = c.builder.build_float_to_signed_int(o.into_float_value(), c.context.i64_type(), "f32_to_i64")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::I64))
}

/// f32.to_string() -> String (via FFI)
pub fn compile_f32_to_string<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = c.module.get_function("tl_f32_to_string")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_f32_to_string not found".to_string())))?;
    let call = c.builder.build_call(fn_val, &[o.into()], "f32_to_string")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from f32.to_string".to_string()).into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// f32.min(other: f32) -> f32
pub fn compile_f32_min<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if a.len() != 1 { return Err(CodegenErrorKind::Internal("f32.min requires 1 argument".to_string()).into()); }
    let lhs = o.into_float_value();
    let rhs = a[0].0.into_float_value();
    let cmp = c.builder.build_float_compare(inkwell::FloatPredicate::OLT, lhs, rhs, "cmp_min")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = c.builder.build_select(cmp, lhs, rhs, "min_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::F32))
}

/// f32.max(other: f32) -> f32
pub fn compile_f32_max<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if a.len() != 1 { return Err(CodegenErrorKind::Internal("f32.max requires 1 argument".to_string()).into()); }
    let lhs = o.into_float_value();
    let rhs = a[0].0.into_float_value();
    let cmp = c.builder.build_float_compare(inkwell::FloatPredicate::OGT, lhs, rhs, "cmp_max")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = c.builder.build_select(cmp, lhs, rhs, "max_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::F32))
}

/// f32.clamp(min: f32, max: f32) -> f32
pub fn compile_f32_clamp<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if a.len() != 2 { return Err(CodegenErrorKind::Internal("f32.clamp requires 2 arguments".to_string()).into()); }
    let val = o.into_float_value();
    let lo = a[0].0.into_float_value();
    let hi = a[1].0.into_float_value();
    let cmp_hi = c.builder.build_float_compare(inkwell::FloatPredicate::OLT, val, hi, "cmp_hi")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let min_val = c.builder.build_select(cmp_hi, val, hi, "min_val")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_float_value();
    let cmp_lo = c.builder.build_float_compare(inkwell::FloatPredicate::OGT, min_val, lo, "cmp_lo")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = c.builder.build_select(cmp_lo, min_val, lo, "clamp_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::F32))
}

/// f64.to_f32() -> f32
pub fn compile_f64_to_f32<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let res = c.builder.build_float_trunc(o.into_float_value(), c.context.f32_type(), "f64_to_f32")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::F32))
}

/// f64.to_i64() -> i64
pub fn compile_f64_to_i64<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let res = c.builder.build_float_to_signed_int(o.into_float_value(), c.context.i64_type(), "f64_to_i64")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::I64))
}

/// f64.to_string() -> String (via FFI)
pub fn compile_f64_to_string<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = c.module.get_function("tl_f64_to_string")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_f64_to_string not found".to_string())))?;
    let call = c.builder.build_call(fn_val, &[o.into()], "f64_to_string")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from f64.to_string".to_string()).into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// f64.min(other: f64) -> f64
pub fn compile_f64_min<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if a.len() != 1 { return Err(CodegenErrorKind::Internal("f64.min requires 1 argument".to_string()).into()); }
    let lhs = o.into_float_value();
    let rhs = a[0].0.into_float_value();
    let cmp = c.builder.build_float_compare(inkwell::FloatPredicate::OLT, lhs, rhs, "cmp_min")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = c.builder.build_select(cmp, lhs, rhs, "min_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::F64))
}

/// f64.max(other: f64) -> f64
pub fn compile_f64_max<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if a.len() != 1 { return Err(CodegenErrorKind::Internal("f64.max requires 1 argument".to_string()).into()); }
    let lhs = o.into_float_value();
    let rhs = a[0].0.into_float_value();
    let cmp = c.builder.build_float_compare(inkwell::FloatPredicate::OGT, lhs, rhs, "cmp_max")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = c.builder.build_select(cmp, lhs, rhs, "max_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::F64))
}

/// f64.clamp(min: f64, max: f64) -> f64
pub fn compile_f64_clamp<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if a.len() != 2 { return Err(CodegenErrorKind::Internal("f64.clamp requires 2 arguments".to_string()).into()); }
    let val = o.into_float_value();
    let lo = a[0].0.into_float_value();
    let hi = a[1].0.into_float_value();
    let cmp_hi = c.builder.build_float_compare(inkwell::FloatPredicate::OLT, val, hi, "cmp_hi")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let min_val = c.builder.build_select(cmp_hi, val, hi, "min_val")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_float_value();
    let cmp_lo = c.builder.build_float_compare(inkwell::FloatPredicate::OGT, min_val, lo, "cmp_lo")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = c.builder.build_select(cmp_lo, min_val, lo, "clamp_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::F64))
}


// ========== I32 メソッド ==========

fn compile_i32_unary<'ctx>(
    c: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    method: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_name = format!("tl_i32_{}", method);
    let fn_val = c.module.get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
    let call = c.builder.build_call(fn_val, &[obj.into()], &format!("i32_{}", method))
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid return from {}", fn_name)))),
    };
    Ok((res, Type::I64)) // I32 methods return I64 in TL
}

fn compile_i32_binary<'ctx>(
    c: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    method: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(TlError::from(CodegenErrorKind::Internal(format!("i32.{} requires 1 argument", method)))); }
    let fn_name = format!("tl_i32_{}", method);
    let fn_val = c.module.get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
    let call = c.builder.build_call(fn_val, &[obj.into(), args[0].0.into()], &format!("i32_{}", method))
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid return from {}", fn_name)))),
    };
    Ok((res, Type::I64))
}

fn compile_i32_bool_unary<'ctx>(
    c: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    method: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_name = format!("tl_i32_{}", method);
    let fn_val = c.module.get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
    let call = c.builder.build_call(fn_val, &[obj.into()], &format!("i32_{}", method))
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid return from {}", fn_name)))),
    };
    Ok((res, Type::Bool))
}

pub fn compile_i32_abs<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i32_unary(c, o, "abs") }
pub fn compile_i32_signum<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i32_unary(c, o, "signum") }
pub fn compile_i32_is_positive<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i32_bool_unary(c, o, "is_positive") }
pub fn compile_i32_is_negative<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i32_bool_unary(c, o, "is_negative") }
pub fn compile_i32_div_euclid<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i32_binary(c, o, a, "div_euclid") }
pub fn compile_i32_rem_euclid<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i32_binary(c, o, a, "rem_euclid") }
pub fn compile_i32_pow<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { compile_i32_binary(c, o, a, "pow") }
pub fn compile_i32_get_offset<'ctx>(_c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { Ok((o, Type::I64)) }
pub fn compile_i32_sumall<'ctx>(_c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> { Ok((o, Type::I64)) }

/// i32.to_f64() -> f64
pub fn compile_i32_to_f64<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let res = c.builder.build_signed_int_to_float(o.into_int_value(), c.context.f64_type(), "i32_to_f64")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::F64))
}

/// i32.to_f32() -> f32
pub fn compile_i32_to_f32<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let res = c.builder.build_signed_int_to_float(o.into_int_value(), c.context.f32_type(), "i32_to_f32")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::F32))
}

/// i32.to_string() -> String (via FFI, reuses i64 path since i32 is stored as i64)
pub fn compile_i32_to_string<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = c.module.get_function("tl_string_from_int")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_string_from_int not found".to_string())))?;
    let call = c.builder.build_call(fn_val, &[o.into()], "i32_to_string")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from i32.to_string".to_string()).into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// i32.min(other: i32) -> i32
pub fn compile_i32_min<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if a.len() != 1 { return Err(CodegenErrorKind::Internal("i32.min requires 1 argument".to_string()).into()); }
    let lhs = o.into_int_value();
    let rhs = a[0].0.into_int_value();
    let cmp = c.builder.build_int_compare(inkwell::IntPredicate::SLT, lhs, rhs, "cmp_min")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = c.builder.build_select(cmp, lhs, rhs, "min_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::I64))
}

/// i32.max(other: i32) -> i32
pub fn compile_i32_max<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if a.len() != 1 { return Err(CodegenErrorKind::Internal("i32.max requires 1 argument".to_string()).into()); }
    let lhs = o.into_int_value();
    let rhs = a[0].0.into_int_value();
    let cmp = c.builder.build_int_compare(inkwell::IntPredicate::SGT, lhs, rhs, "cmp_max")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = c.builder.build_select(cmp, lhs, rhs, "max_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::I64))
}

/// i32.clamp(min: i32, max: i32) -> i32
pub fn compile_i32_clamp<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if a.len() != 2 { return Err(CodegenErrorKind::Internal("i32.clamp requires 2 arguments".to_string()).into()); }
    let val = o.into_int_value();
    let lo = a[0].0.into_int_value();
    let hi = a[1].0.into_int_value();
    let cmp_hi = c.builder.build_int_compare(inkwell::IntPredicate::SLT, val, hi, "cmp_hi")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let min_val = c.builder.build_select(cmp_hi, val, hi, "min_val")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
    let cmp_lo = c.builder.build_int_compare(inkwell::IntPredicate::SGT, min_val, lo, "cmp_lo")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = c.builder.build_select(cmp_lo, min_val, lo, "clamp_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::I64))
}

// ========== Phase C: 数値型定数・判定メソッド ==========

/// f64.is_nan() -> bool — LLVM fcmp uno で NaN 判定
pub fn compile_f64_is_nan<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = o.into_float_value();
    // fcmp uno (unordered): val != val is true iff val is NaN
    let res = c.builder.build_float_compare(inkwell::FloatPredicate::UNO, val, val, "is_nan")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::Bool))
}

/// f64.is_inf() -> bool — |val| == infinity
pub fn compile_f64_is_inf<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = o.into_float_value();
    let abs_val = c.builder.build_float_neg(val, "neg_tmp").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let is_neg = c.builder.build_float_compare(inkwell::FloatPredicate::OLT, val, c.context.f64_type().const_float(0.0), "is_neg")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let abs = c.builder.build_select(is_neg, abs_val, val, "abs_val")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_float_value();
    let inf = c.context.f64_type().const_float(f64::INFINITY);
    let res = c.builder.build_float_compare(inkwell::FloatPredicate::OEQ, abs, inf, "is_inf")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::Bool))
}

/// f32.is_nan() -> bool
pub fn compile_f32_is_nan<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = o.into_float_value();
    let res = c.builder.build_float_compare(inkwell::FloatPredicate::UNO, val, val, "is_nan")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::Bool))
}

/// f32.is_inf() -> bool
pub fn compile_f32_is_inf<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, _a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = o.into_float_value();
    let abs_val = c.builder.build_float_neg(val, "neg_tmp").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let is_neg = c.builder.build_float_compare(inkwell::FloatPredicate::OLT, val, c.context.f32_type().const_float(0.0), "is_neg")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let abs = c.builder.build_select(is_neg, abs_val, val, "abs_val")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_float_value();
    let inf = c.context.f32_type().const_float(f64::INFINITY);
    let res = c.builder.build_float_compare(inkwell::FloatPredicate::OEQ, abs, inf, "is_inf")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((res.into(), Type::Bool))
}

/// F64::nan() -> f64
pub fn compile_f64_nan<'ctx>(c: &mut CodeGenerator<'ctx>, _a: Vec<(BasicValueEnum<'ctx>, Type)>, _hint: Option<&Type>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = c.context.f64_type().const_float(f64::NAN);
    Ok((val.into(), Type::F64))
}

/// F64::infinity() -> f64
pub fn compile_f64_infinity<'ctx>(c: &mut CodeGenerator<'ctx>, _a: Vec<(BasicValueEnum<'ctx>, Type)>, _hint: Option<&Type>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = c.context.f64_type().const_float(f64::INFINITY);
    Ok((val.into(), Type::F64))
}

/// F64::max_value() -> f64
pub fn compile_f64_max_value<'ctx>(c: &mut CodeGenerator<'ctx>, _a: Vec<(BasicValueEnum<'ctx>, Type)>, _hint: Option<&Type>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = c.context.f64_type().const_float(f64::MAX);
    Ok((val.into(), Type::F64))
}

/// F64::min_value() -> f64
pub fn compile_f64_min_value<'ctx>(c: &mut CodeGenerator<'ctx>, _a: Vec<(BasicValueEnum<'ctx>, Type)>, _hint: Option<&Type>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = c.context.f64_type().const_float(f64::MIN);
    Ok((val.into(), Type::F64))
}

/// I64::max_value() -> i64
pub fn compile_i64_max_value<'ctx>(c: &mut CodeGenerator<'ctx>, _a: Vec<(BasicValueEnum<'ctx>, Type)>, _hint: Option<&Type>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = c.context.i64_type().const_int(i64::MAX as u64, false);
    Ok((val.into(), Type::I64))
}

/// I64::min_value() -> i64
pub fn compile_i64_min_value<'ctx>(c: &mut CodeGenerator<'ctx>, _a: Vec<(BasicValueEnum<'ctx>, Type)>, _hint: Option<&Type>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = c.context.i64_type().const_int(i64::MIN as u64, true);
    Ok((val.into(), Type::I64))
}

/// I32::max_value() -> i32
pub fn compile_i32_max_value<'ctx>(c: &mut CodeGenerator<'ctx>, _a: Vec<(BasicValueEnum<'ctx>, Type)>, _hint: Option<&Type>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = c.context.i32_type().const_int(i32::MAX as u64, false);
    Ok((val.into(), Type::I32))
}

/// I32::min_value() -> i32
pub fn compile_i32_min_value<'ctx>(c: &mut CodeGenerator<'ctx>, _a: Vec<(BasicValueEnum<'ctx>, Type)>, _hint: Option<&Type>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = c.context.i32_type().const_int(i32::MIN as u64, true);
    Ok((val.into(), Type::I32))
}

/// F32::nan() -> f32
pub fn compile_f32_nan<'ctx>(c: &mut CodeGenerator<'ctx>, _a: Vec<(BasicValueEnum<'ctx>, Type)>, _hint: Option<&Type>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = c.context.f32_type().const_float(f64::NAN);
    Ok((val.into(), Type::F32))
}

/// F32::infinity() -> f32
pub fn compile_f32_infinity<'ctx>(c: &mut CodeGenerator<'ctx>, _a: Vec<(BasicValueEnum<'ctx>, Type)>, _hint: Option<&Type>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = c.context.f32_type().const_float(f64::INFINITY);
    Ok((val.into(), Type::F32))
}

/// F32::max_value() -> f32
pub fn compile_f32_max_value<'ctx>(c: &mut CodeGenerator<'ctx>, _a: Vec<(BasicValueEnum<'ctx>, Type)>, _hint: Option<&Type>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = c.context.f32_type().const_float(f32::MAX as f64);
    Ok((val.into(), Type::F32))
}

/// F32::min_value() -> f32
pub fn compile_f32_min_value<'ctx>(c: &mut CodeGenerator<'ctx>, _a: Vec<(BasicValueEnum<'ctx>, Type)>, _hint: Option<&Type>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let val = c.context.f32_type().const_float(f32::MIN as f64);
    Ok((val.into(), Type::F32))
}
