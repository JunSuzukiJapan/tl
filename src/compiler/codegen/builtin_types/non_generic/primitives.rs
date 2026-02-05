
use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
use crate::compiler::ast::Type;



pub fn register_primitive_types(manager: &mut TypeManager) {
    let string_type = Type::String("String".to_string());
    
    // ===== String =====
    let mut string = CodeGenType::new("String");
    string.register_instance_signature("len", vec![], Type::I64);
    string.register_instance_signature("contains", vec![string_type.clone()], Type::Bool);
    string.register_instance_signature("concat", vec![string_type.clone()], string_type.clone());
    string.register_instance_signature("char_at", vec![Type::I64], Type::Char("Char".to_string()));
    string.register_instance_signature("to_i64", vec![], Type::I64);
    string.register_instance_signature("print", vec![], Type::Void);
    string.register_instance_signature("display", vec![], Type::Void);
    manager.register_type(string);

    // ===== F32 =====
    let mut f32_type = CodeGenType::new("F32");
    // Unary methods
    for method in ["abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", 
                   "cbrt", "ceil", "cos", "cosh", "exp", "exp2", "exp_m1",
                   "floor", "fract", "ln", "ln_1p", "log", "log10", "log2",
                   "recip", "round", "signum", "sin", "sinh", "sqrt",
                   "tan", "tanh", "to_degrees", "to_radians", "trunc"] {
        f32_type.register_instance_signature(method, vec![], Type::F32);
    }
    // Binary methods
    for method in ["atan2", "copysign", "hypot", "powf", "pow", "powi"] {
        f32_type.register_instance_signature(method, vec![Type::F32], Type::F32);
    }
    manager.register_type(f32_type);

    // ===== F64 =====
    let mut f64_type = CodeGenType::new("F64");
    for method in ["abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", 
                   "cbrt", "ceil", "cos", "cosh", "exp", "exp2", "exp_m1",
                   "floor", "fract", "ln", "ln_1p", "log", "log10", "log2",
                   "recip", "round", "signum", "sin", "sinh", "sqrt",
                   "tan", "tanh", "to_degrees", "to_radians", "trunc"] {
        f64_type.register_instance_signature(method, vec![], Type::F64);
    }
    for method in ["atan2", "copysign", "hypot", "powf", "pow", "powi"] {
        f64_type.register_instance_signature(method, vec![Type::F64], Type::F64);
    }
    manager.register_type(f64_type);

    // ===== I64 =====
    let mut i64_type = CodeGenType::new("I64");
    i64_type.register_instance_signature("abs", vec![], Type::I64);
    i64_type.register_instance_signature("signum", vec![], Type::I64);
    i64_type.register_instance_signature("get_offset", vec![], Type::I64);
    i64_type.register_instance_signature("sumall", vec![], Type::I64);
    i64_type.register_instance_signature("is_positive", vec![], Type::Bool);
    i64_type.register_instance_signature("is_negative", vec![], Type::Bool);
    i64_type.register_instance_signature("div_euclid", vec![Type::I64], Type::I64);
    i64_type.register_instance_signature("rem_euclid", vec![Type::I64], Type::I64);
    i64_type.register_instance_signature("pow", vec![Type::I64], Type::I64);
    manager.register_type(i64_type);
    
    // ===== I32 =====
    let mut i32_type = CodeGenType::new("I32");
    i32_type.register_instance_signature("abs", vec![], Type::I64);
    i32_type.register_instance_signature("signum", vec![], Type::I64);
    i32_type.register_instance_signature("get_offset", vec![], Type::I64);
    i32_type.register_instance_signature("sumall", vec![], Type::I64);
    i32_type.register_instance_signature("is_positive", vec![], Type::Bool);
    i32_type.register_instance_signature("is_negative", vec![], Type::Bool);
    i32_type.register_instance_signature("div_euclid", vec![Type::I64], Type::I64);
    i32_type.register_instance_signature("rem_euclid", vec![Type::I64], Type::I64);
    i32_type.register_instance_signature("pow", vec![Type::I64], Type::I64);
    manager.register_type(i32_type);
    
    // ===== Bool =====
    let bool_type = CodeGenType::new("Bool");
    manager.register_type(bool_type);
}
