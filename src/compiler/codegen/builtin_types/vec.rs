use crate::compiler::ast::{StructDef, Type};
// use crate::compiler::codegen::expr; // For future method migration

pub fn get_vec_struct_def() -> StructDef {
    // struct Vec<T> { ptr: ptr<T>, cap: I64, len: I64 }
    let t = Type::UserDefined("T".to_string(), vec![]);
    
    // We expect runtime to handle Vec as Reference Type (pointer),
    // so the fields are mostly for documentation/logic if we parse .tl file.
    // But since we use extern methods for everything, fields are not accessed by methods directly.
    
    StructDef {
        name: "Vec".to_string(),
        fields: vec![
            ("ptr".to_string(), Type::Vec(Box::new(t.clone()))),
            ("cap".to_string(), Type::I64),
            ("len".to_string(), Type::I64),
        ],
        generics: vec!["T".to_string()],
    }
}
