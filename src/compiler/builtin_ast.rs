use crate::compiler::ast::*;
use crate::compiler::error::Span;

pub fn load_builtin_structs() -> Vec<StructDef> {
    vec![
        create_vec_struct(),
    ]
}

pub fn load_builtin_impls() -> Vec<ImplBlock> {
    vec![
        create_vec_impl(),
    ]
}

fn create_vec_struct() -> StructDef {
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

fn create_vec_impl() -> ImplBlock {
    // impl<T> Vec<T>
    let t = Type::UserDefined("T".to_string(), vec![]);
    let vec_t = Type::UserDefined("Vec".to_string(), vec![t.clone()]);

    // Define extern methods.
    // They will be mangled to tl_vec_{suffix}_{method}
    
    ImplBlock {
        target_type: vec_t.clone(),
        generics: vec!["T".to_string()],
        methods: vec![
            // fn new() -> Vec<T>
            FunctionDef {
                name: "new".to_string(),
                args: vec![],
                return_type: vec_t.clone(),
                body: vec![], // Extern has no body
                generics: vec![],
                is_extern: true,
            },
            // fn push(self, item: T)
            FunctionDef {
                name: "push".to_string(),
                args: vec![
                    ("self".to_string(), vec_t.clone()),
                    ("item".to_string(), t.clone())
                ],
                return_type: Type::Void,
                body: vec![],
                generics: vec![],
                is_extern: true,
            },
            // fn pop(self) -> T
            FunctionDef {
                name: "pop".to_string(),
                args: vec![("self".to_string(), vec_t.clone())],
                return_type: t.clone(),
                body: vec![],
                generics: vec![],
                is_extern: true,
            },
            // fn len(self) -> I64
            FunctionDef {
                name: "len".to_string(),
                args: vec![("self".to_string(), vec_t.clone())],
                return_type: Type::I64,
                body: vec![],
                generics: vec![],
                is_extern: true,
            },
            // fn get(self, index: I64) -> T
            FunctionDef {
                name: "get".to_string(),
                args: vec![
                    ("self".to_string(), vec_t.clone()),
                    ("index".to_string(), Type::I64)
                ],
                return_type: t.clone(),
                body: vec![],
                generics: vec![],
                is_extern: true,
            },
            // fn is_empty(self) -> Bool
            FunctionDef {
                name: "is_empty".to_string(),
                args: vec![("self".to_string(), vec_t.clone())],
                return_type: Type::Bool,
                body: vec![],
                generics: vec![],
                is_extern: true,
            },
            // fn clear(self)
            FunctionDef {
                name: "clear".to_string(),
                args: vec![("self".to_string(), vec_t.clone())],
                return_type: Type::Void,
                body: vec![],
                generics: vec![],
                is_extern: true,
            },
            // fn remove(self, index: I64) -> T
            FunctionDef {
                name: "remove".to_string(),
                args: vec![
                    ("self".to_string(), vec_t.clone()),
                    ("index".to_string(), Type::I64)
                ],
                return_type: t.clone(),
                body: vec![],
                generics: vec![],
                is_extern: true,
            }
        ],
    }
}
