use crate::compiler::ast::*;
use crate::compiler::error::Span;
use std::collections::HashMap;

/// Register built-in generic implementations (e.g., Vec<T>, Option<T>)
pub fn register_builtin_impls(generic_impls: &mut HashMap<String, Vec<ImplBlock>>) {
    let vec_impl = create_vec_impl();
    generic_impls.insert("Vec".to_string(), vec![vec_impl]);

    let option_impl = create_option_impl();
    generic_impls.insert("Option".to_string(), vec![option_impl]);
}

/// Register built-in struct definitions
pub fn register_builtin_structs(struct_defs: &mut HashMap<String, StructDef>) {
    let t_type = Type::UserDefined("T".to_string(), vec![]);
    
    // struct Vec<T> { ptr: ptr<T>, cap: i64, len: i64 }
    // Note: ptr<T> isn't standard Type in ast.rs. 
    // Usually we use Type::Vec akin to array or pointer?
    // Or we use a specific Pointer type if available.
    // ast.rs doesn't have explicit Pointer(T).
    // It has Tensor(T, rank) which is pointer-ish.
    // Or just treat it as UserDefined("ptr")? 
    // But `ptr` field in our logic uses IndexAccess.
    // IndexAccess works on Tensor, Vec, ScalarArray.
    
    // Let's declare field `ptr` as `Type::Vec(Box::new(t_type))`.
    // Wait, Type::Vec is the high level vector.
    // If I iterate it, it works.
    // If Vec is implemented as { ptr: Vec<T>, ... } Recurisve?
    // No, Type::Vec is just the pointer to data array in this low-level view.
    
    let vec_struct = StructDef {
        name: "Vec".to_string(),
        fields: vec![
            ("ptr".to_string(), Type::Vec(Box::new(t_type.clone()))), // Treat `ptr` field as a raw array/vector
            ("cap".to_string(), Type::I64),
            ("len".to_string(), Type::I64),
        ],
        generics: vec!["T".to_string()],
    };
    eprintln!("DEBUG register_builtin_structs inserting Vec with {} fields", vec_struct.fields.len());
    struct_defs.insert("Vec".to_string(), vec_struct);

    // Note: Option<T> is handled inline in codegen (expr.rs) without a registered struct definition.
    // The Option struct { tag: i64, value: T } is created dynamically based on the concrete type T.
}

fn create_vec_impl() -> ImplBlock {
    let t_type = Type::UserDefined("T".to_string(), vec![]);

    ImplBlock {
        target_type: Type::UserDefined("Vec".to_string(), vec![t_type.clone()]),
        generics: vec!["T".to_string()],
        methods: vec![],
    }
}

fn create_option_impl() -> ImplBlock {
    let t_type = Type::UserDefined("T".to_string(), vec![]);

    ImplBlock {
        target_type: Type::UserDefined("Option".to_string(), vec![t_type.clone()]),
        generics: vec!["T".to_string()],
        methods: vec![],
    }
}
