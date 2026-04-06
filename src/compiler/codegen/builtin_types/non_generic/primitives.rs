
use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
use crate::compiler::ast::Type;

use super::primitive_methods as pm;

pub fn register_primitive_types(manager: &mut TypeManager) {
    let string_type = Type::String("String".to_string());
    
    // ===== String =====
    let mut string = CodeGenType::new("String");
    use super::string_methods;
    string.register_evaluated_instance_method("len", string_methods::compile_len, vec![], Type::I64);
    string.register_evaluated_instance_method("contains", string_methods::compile_contains, vec![string_type.clone()], Type::Bool);
    string.register_evaluated_instance_method("concat", string_methods::compile_concat, vec![string_type.clone()], string_type.clone());
    string.register_evaluated_instance_method("char_at", string_methods::compile_char_at, vec![Type::I64], Type::Char("Char".to_string()));
    string.register_evaluated_instance_method("to_i64", string_methods::compile_to_i64, vec![], Type::I64);
    string.register_evaluated_instance_method("print", string_methods::compile_print, vec![], Type::Void);
    string.register_evaluated_instance_method("display", string_methods::compile_display, vec![], Type::Void);
    string.register_evaluated_instance_method("trim", string_methods::compile_trim, vec![], string_type.clone());
    string.register_evaluated_instance_method("starts_with", string_methods::compile_starts_with, vec![string_type.clone()], Type::Bool);
    string.register_evaluated_instance_method("ends_with", string_methods::compile_ends_with, vec![string_type.clone()], Type::Bool);
    string.register_evaluated_instance_method("replace", string_methods::compile_replace, vec![string_type.clone(), string_type.clone()], string_type.clone());
    string.register_evaluated_instance_method("substring", string_methods::compile_substring, vec![Type::I64, Type::I64], string_type.clone());
    string.register_evaluated_instance_method("is_empty", string_methods::compile_is_empty, vec![], Type::Bool);
    string.register_evaluated_instance_method("to_uppercase", string_methods::compile_to_uppercase, vec![], string_type.clone());
    string.register_evaluated_instance_method("to_lowercase", string_methods::compile_to_lowercase, vec![], string_type.clone());
    string.register_evaluated_instance_method("index_of", string_methods::compile_index_of, vec![string_type.clone()], Type::I64);
    string.register_evaluated_instance_method("split", string_methods::compile_split, vec![string_type.clone()], Type::Struct("Vec".to_string(), vec![string_type.clone()]));
    string.register_evaluated_instance_method("to_f64", string_methods::compile_to_f64, vec![], Type::F64);
    string.register_evaluated_instance_method("repeat", string_methods::compile_repeat, vec![Type::I64], string_type.clone());
    string.register_evaluated_instance_method("chars", string_methods::compile_chars, vec![], Type::Struct("Vec".to_string(), vec![Type::I64]));
    string.register_evaluated_static_method("from_chars", string_methods::compile_from_chars, vec![Type::Struct("Vec".to_string(), vec![Type::I64])], string_type.clone());
    manager.register_type(string);

    // ===== F32 =====
    let mut f32_type = CodeGenType::new("F32");
    // Unary methods
    f32_type.register_evaluated_instance_method("abs", pm::compile_f32_abs, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("acos", pm::compile_f32_acos, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("acosh", pm::compile_f32_acosh, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("asin", pm::compile_f32_asin, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("asinh", pm::compile_f32_asinh, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("atan", pm::compile_f32_atan, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("atanh", pm::compile_f32_atanh, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("cbrt", pm::compile_f32_cbrt, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("ceil", pm::compile_f32_ceil, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("cos", pm::compile_f32_cos, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("cosh", pm::compile_f32_cosh, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("exp", pm::compile_f32_exp, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("exp2", pm::compile_f32_exp2, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("exp_m1", pm::compile_f32_exp_m1, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("floor", pm::compile_f32_floor, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("fract", pm::compile_f32_fract, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("ln", pm::compile_f32_ln, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("ln_1p", pm::compile_f32_ln_1p, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("log", pm::compile_f32_log, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("log10", pm::compile_f32_log10, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("log2", pm::compile_f32_log2, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("recip", pm::compile_f32_recip, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("round", pm::compile_f32_round, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("signum", pm::compile_f32_signum, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("sin", pm::compile_f32_sin, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("sinh", pm::compile_f32_sinh, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("sqrt", pm::compile_f32_sqrt, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("tan", pm::compile_f32_tan, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("tanh", pm::compile_f32_tanh, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("to_degrees", pm::compile_f32_to_degrees, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("to_radians", pm::compile_f32_to_radians, vec![], Type::F32);
    f32_type.register_evaluated_instance_method("trunc", pm::compile_f32_trunc, vec![], Type::F32);
    // Binary methods
    f32_type.register_evaluated_instance_method("atan2", pm::compile_f32_atan2, vec![Type::F32], Type::F32);
    f32_type.register_evaluated_instance_method("copysign", pm::compile_f32_copysign, vec![Type::F32], Type::F32);
    f32_type.register_evaluated_instance_method("hypot", pm::compile_f32_hypot, vec![Type::F32], Type::F32);
    f32_type.register_evaluated_instance_method("powf", pm::compile_f32_powf, vec![Type::F32], Type::F32);
    f32_type.register_evaluated_instance_method("pow", pm::compile_f32_pow, vec![Type::F32], Type::F32);
    f32_type.register_evaluated_instance_method("powi", pm::compile_f32_powi, vec![Type::F32], Type::F32);
    // Conversion & comparison methods
    f32_type.register_evaluated_instance_method("to_f64", pm::compile_f32_to_f64, vec![], Type::F64);
    f32_type.register_evaluated_instance_method("to_i64", pm::compile_f32_to_i64, vec![], Type::I64);
    f32_type.register_evaluated_instance_method("to_string", pm::compile_f32_to_string, vec![], string_type.clone());
    f32_type.register_evaluated_instance_method("min", pm::compile_f32_min, vec![Type::F32], Type::F32);
    f32_type.register_evaluated_instance_method("max", pm::compile_f32_max, vec![Type::F32], Type::F32);
    f32_type.register_evaluated_instance_method("clamp", pm::compile_f32_clamp, vec![Type::F32, Type::F32], Type::F32);
    f32_type.register_evaluated_instance_method("is_nan", pm::compile_f32_is_nan, vec![], Type::Bool);
    f32_type.register_evaluated_instance_method("is_inf", pm::compile_f32_is_inf, vec![], Type::Bool);
    f32_type.register_evaluated_static_method("nan", pm::compile_f32_nan, vec![], Type::F32);
    f32_type.register_evaluated_static_method("infinity", pm::compile_f32_infinity, vec![], Type::F32);
    f32_type.register_evaluated_static_method("max_value", pm::compile_f32_max_value, vec![], Type::F32);
    f32_type.register_evaluated_static_method("min_value", pm::compile_f32_min_value, vec![], Type::F32);
    manager.register_type(f32_type);

    // ===== F64 =====
    let mut f64_type = CodeGenType::new("F64");
    f64_type.register_evaluated_instance_method("abs", pm::compile_f64_abs, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("acos", pm::compile_f64_acos, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("acosh", pm::compile_f64_acosh, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("asin", pm::compile_f64_asin, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("asinh", pm::compile_f64_asinh, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("atan", pm::compile_f64_atan, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("atanh", pm::compile_f64_atanh, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("cbrt", pm::compile_f64_cbrt, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("ceil", pm::compile_f64_ceil, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("cos", pm::compile_f64_cos, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("cosh", pm::compile_f64_cosh, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("exp", pm::compile_f64_exp, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("exp2", pm::compile_f64_exp2, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("exp_m1", pm::compile_f64_exp_m1, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("floor", pm::compile_f64_floor, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("fract", pm::compile_f64_fract, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("ln", pm::compile_f64_ln, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("ln_1p", pm::compile_f64_ln_1p, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("log", pm::compile_f64_log, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("log10", pm::compile_f64_log10, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("log2", pm::compile_f64_log2, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("recip", pm::compile_f64_recip, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("round", pm::compile_f64_round, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("signum", pm::compile_f64_signum, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("sin", pm::compile_f64_sin, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("sinh", pm::compile_f64_sinh, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("sqrt", pm::compile_f64_sqrt, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("tan", pm::compile_f64_tan, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("tanh", pm::compile_f64_tanh, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("to_degrees", pm::compile_f64_to_degrees, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("to_radians", pm::compile_f64_to_radians, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("trunc", pm::compile_f64_trunc, vec![], Type::F64);
    f64_type.register_evaluated_instance_method("atan2", pm::compile_f64_atan2, vec![Type::F64], Type::F64);
    f64_type.register_evaluated_instance_method("copysign", pm::compile_f64_copysign, vec![Type::F64], Type::F64);
    f64_type.register_evaluated_instance_method("hypot", pm::compile_f64_hypot, vec![Type::F64], Type::F64);
    f64_type.register_evaluated_instance_method("powf", pm::compile_f64_powf, vec![Type::F64], Type::F64);
    f64_type.register_evaluated_instance_method("pow", pm::compile_f64_pow, vec![Type::F64], Type::F64);
    f64_type.register_evaluated_instance_method("powi", pm::compile_f64_powi, vec![Type::F64], Type::F64);
    // Conversion & comparison methods
    f64_type.register_evaluated_instance_method("to_f32", pm::compile_f64_to_f32, vec![], Type::F32);
    f64_type.register_evaluated_instance_method("to_i64", pm::compile_f64_to_i64, vec![], Type::I64);
    f64_type.register_evaluated_instance_method("to_string", pm::compile_f64_to_string, vec![], string_type.clone());
    f64_type.register_evaluated_instance_method("min", pm::compile_f64_min, vec![Type::F64], Type::F64);
    f64_type.register_evaluated_instance_method("max", pm::compile_f64_max, vec![Type::F64], Type::F64);
    f64_type.register_evaluated_instance_method("clamp", pm::compile_f64_clamp, vec![Type::F64, Type::F64], Type::F64);
    f64_type.register_evaluated_instance_method("is_nan", pm::compile_f64_is_nan, vec![], Type::Bool);
    f64_type.register_evaluated_instance_method("is_inf", pm::compile_f64_is_inf, vec![], Type::Bool);
    f64_type.register_evaluated_static_method("nan", pm::compile_f64_nan, vec![], Type::F64);
    f64_type.register_evaluated_static_method("infinity", pm::compile_f64_infinity, vec![], Type::F64);
    f64_type.register_evaluated_static_method("max_value", pm::compile_f64_max_value, vec![], Type::F64);
    f64_type.register_evaluated_static_method("min_value", pm::compile_f64_min_value, vec![], Type::F64);
    manager.register_type(f64_type);

    // ===== I64 =====
    let mut i64_type = CodeGenType::new("I64");
    i64_type.register_evaluated_instance_method("abs", pm::compile_i64_abs, vec![], Type::I64);
    i64_type.register_evaluated_instance_method("signum", pm::compile_i64_signum, vec![], Type::I64);
    i64_type.register_evaluated_instance_method("get_offset", pm::compile_i64_get_offset, vec![], Type::I64);
    i64_type.register_evaluated_instance_method("sumall", pm::compile_i64_sumall, vec![], Type::I64);
    i64_type.register_evaluated_instance_method("is_positive", pm::compile_i64_is_positive, vec![], Type::Bool);
    i64_type.register_evaluated_instance_method("is_negative", pm::compile_i64_is_negative, vec![], Type::Bool);
    i64_type.register_evaluated_instance_method("div_euclid", pm::compile_i64_div_euclid, vec![Type::I64], Type::I64);
    i64_type.register_evaluated_instance_method("rem_euclid", pm::compile_i64_rem_euclid, vec![Type::I64], Type::I64);
    i64_type.register_evaluated_instance_method("pow", pm::compile_i64_pow, vec![Type::I64], Type::I64);
    // Conversion & comparison methods
    i64_type.register_evaluated_instance_method("to_f64", pm::compile_i64_to_f64, vec![], Type::F64);
    i64_type.register_evaluated_instance_method("to_f32", pm::compile_i64_to_f32, vec![], Type::F32);
    i64_type.register_evaluated_instance_method("to_string", pm::compile_i64_to_string, vec![], string_type.clone());
    i64_type.register_evaluated_instance_method("min", pm::compile_i64_min, vec![Type::I64], Type::I64);
    i64_type.register_evaluated_instance_method("max", pm::compile_i64_max, vec![Type::I64], Type::I64);
    i64_type.register_evaluated_instance_method("clamp", pm::compile_i64_clamp, vec![Type::I64, Type::I64], Type::I64);
    i64_type.register_evaluated_static_method("max_value", pm::compile_i64_max_value, vec![], Type::I64);
    i64_type.register_evaluated_static_method("min_value", pm::compile_i64_min_value, vec![], Type::I64);
    manager.register_type(i64_type);
    
    // ===== I32 =====
    let mut i32_type = CodeGenType::new("I32");
    i32_type.register_evaluated_instance_method("abs", pm::compile_i32_abs, vec![], Type::I64);
    i32_type.register_evaluated_instance_method("signum", pm::compile_i32_signum, vec![], Type::I64);
    i32_type.register_evaluated_instance_method("get_offset", pm::compile_i32_get_offset, vec![], Type::I64);
    i32_type.register_evaluated_instance_method("sumall", pm::compile_i32_sumall, vec![], Type::I64);
    i32_type.register_evaluated_instance_method("is_positive", pm::compile_i32_is_positive, vec![], Type::Bool);
    i32_type.register_evaluated_instance_method("is_negative", pm::compile_i32_is_negative, vec![], Type::Bool);
    i32_type.register_evaluated_instance_method("div_euclid", pm::compile_i32_div_euclid, vec![Type::I64], Type::I64);
    i32_type.register_evaluated_instance_method("rem_euclid", pm::compile_i32_rem_euclid, vec![Type::I64], Type::I64);
    i32_type.register_evaluated_instance_method("pow", pm::compile_i32_pow, vec![Type::I64], Type::I64);
    // Conversion & comparison methods
    i32_type.register_evaluated_instance_method("to_f64", pm::compile_i32_to_f64, vec![], Type::F64);
    i32_type.register_evaluated_instance_method("to_f32", pm::compile_i32_to_f32, vec![], Type::F32);
    i32_type.register_evaluated_instance_method("to_string", pm::compile_i32_to_string, vec![], string_type.clone());
    i32_type.register_evaluated_instance_method("min", pm::compile_i32_min, vec![Type::I64], Type::I64);
    i32_type.register_evaluated_instance_method("max", pm::compile_i32_max, vec![Type::I64], Type::I64);
    i32_type.register_evaluated_instance_method("clamp", pm::compile_i32_clamp, vec![Type::I64, Type::I64], Type::I64);
    i32_type.register_evaluated_static_method("max_value", pm::compile_i32_max_value, vec![], Type::I32);
    i32_type.register_evaluated_static_method("min_value", pm::compile_i32_min_value, vec![], Type::I32);
    manager.register_type(i32_type);
    
    // ===== Bool =====
    let bool_type = CodeGenType::new("Bool");
    manager.register_type(bool_type);
}
