use crate::compiler::ast::Type;

/// Resolves the runtime function name for a static method call based on the type and its generic arguments.
/// 
/// Pattern: tl_{type_name}_{generic_arg1}_{generic_arg2}_{method_name}
/// Example: Vec<i64>::new -> tl_vec_i64_new
/// Example: Vec<String>::new -> tl_vec_string_new
/// Example: Vec<MyStruct>::new -> tl_vec_ptr_new
pub fn resolve_static_method_name(type_name: &str, method: &str, generics: &[Type]) -> String {
    let mut parts = Vec::new();
    parts.push("tl".to_string());
    parts.push(type_name.to_lowercase());

    // Apply defaults for legacy compatibility (Vec -> Vec<i64>, HashMap -> HashMap<String, Tensor>)
    let effective_generics = if generics.is_empty() {
        match type_name {
            "Vec" => vec![Type::I64],
            "HashMap" => vec![Type::String("String".to_string()), Type::Struct("Tensor".to_string(), vec![])],
            _ => generics.to_vec(),
        }
    } else {
        generics.to_vec()
    };

    for g in &effective_generics {
        parts.push(mangle_type_segment(g));
    }
    
    parts.push(method.to_string());
    parts.join("_")
}

fn mangle_type_segment(ty: &Type) -> String {
    match ty {
        Type::I64 => "i64".to_string(),
        Type::I32 => "i32".to_string(),
        Type::F32 => "f32".to_string(),
        // Note: F64 support in Vec not explicit in builtins, but we mangle it anyway. 
        // If runtime function missing, look up will fail (as designed).
        Type::F64 => "f64".to_string(),
        Type::U8 => "u8".to_string(),
        Type::Bool => "bool".to_string(),
        Type::String(_) => "string".to_string(),
        
        // Boxed/Reference types map to "ptr"
        Type::Struct(..) | Type::Enum(..) => "ptr".to_string(),
        
        // Fallback
        _ => "ptr".to_string(),
    }
}
