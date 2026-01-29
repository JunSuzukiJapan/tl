use crate::compiler::ast::{StructDef, Type};
// use crate::compiler::codegen::expr; // For future method migration

pub fn get_hashmap_struct_def() -> StructDef {
    // struct HashMap<K, V> { ptr: ptr<T> (unused), len: I64 }
    StructDef {
        name: "HashMap".to_string(),
        fields: vec![
            ("ptr".to_string(), Type::I64), // Placeholder
            ("len".to_string(), Type::I64),
        ],
        generics: vec!["K".to_string(), "V".to_string()],
    }
}
