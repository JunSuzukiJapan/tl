use crate::compiler::ast::*;


pub fn load_builtin_enums() -> Vec<EnumDef> {
    vec![
        create_option_enum(),
        create_result_enum(),
    ]
}

pub fn load_builtin_structs() -> Vec<StructDef> {
    vec![
        create_vec_struct(),
        create_hashmap_struct(),
    ]
}

pub fn load_builtin_impls() -> Vec<ImplBlock> {
    vec![
        create_vec_impl(),
        create_hashmap_impl(),
        create_option_impl(),
        create_result_impl(),
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

fn create_hashmap_struct() -> StructDef {
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

fn create_hashmap_impl() -> ImplBlock {
    let k = Type::UserDefined("K".to_string(), vec![]);
    let v = Type::UserDefined("V".to_string(), vec![]);
    let map_t = Type::UserDefined("HashMap".to_string(), vec![k.clone(), v.clone()]);
    
    ImplBlock {
        target_type: map_t.clone(),
        generics: vec!["K".to_string(), "V".to_string()],
        methods: vec![
             // fn new() -> HashMap<K, V>
             FunctionDef {
                 name: "new".to_string(),
                 args: vec![],
                 return_type: map_t.clone(),
                 body: vec![],
                 generics: vec![],
                 is_extern: true,
             },
             // fn insert(self, key: K, value: V) -> Bool
             FunctionDef {
                 name: "insert".to_string(),
                 args: vec![
                     ("self".to_string(), map_t.clone()),
                     ("key".to_string(), k.clone()),
                     ("value".to_string(), v.clone())
                 ],
                 return_type: Type::Bool,
                 body: vec![],
                 generics: vec![],
                 is_extern: true,
             },
             // fn get(self, key: K) -> V
             FunctionDef {
                 name: "get".to_string(),
                 args: vec![
                     ("self".to_string(), map_t.clone()),
                     ("key".to_string(), k.clone())
                 ],
                 return_type: v.clone(),
                 body: vec![],
                 generics: vec![],
                 is_extern: true,
             },
             // fn remove(self, key: K) -> V
             FunctionDef {
                 name: "remove".to_string(),
                 args: vec![
                     ("self".to_string(), map_t.clone()),
                     ("key".to_string(), k.clone())
                 ],
                 return_type: v.clone(),
                 body: vec![],
                 generics: vec![],
                 is_extern: true,
             },
             // fn contains_key(self, key: K) -> Bool
             FunctionDef {
                 name: "contains_key".to_string(),
                 args: vec![
                     ("self".to_string(), map_t.clone()),
                     ("key".to_string(), k.clone())
                 ],
                 return_type: Type::Bool,
                 body: vec![],
                 generics: vec![],
                 is_extern: true,
             },
             // fn len(self) -> I64
             FunctionDef {
                 name: "len".to_string(),
                 args: vec![("self".to_string(), map_t.clone())],
                 return_type: Type::I64,
                 body: vec![],
                 generics: vec![],
                 is_extern: true,
             },
             // fn clear(self) -> Bool
             FunctionDef {
                 name: "clear".to_string(),
                 args: vec![("self".to_string(), map_t.clone())],
                 return_type: Type::Bool,
                 body: vec![],
                 generics: vec![],
                 is_extern: true,
             },
        ]
    }
}

fn create_option_enum() -> EnumDef {
    // enum Option<T> { Some(T), None }
    let t = Type::UserDefined("T".to_string(), vec![]);
    
    EnumDef {
        name: "Option".to_string(),
        generics: vec!["T".to_string()],
        variants: vec![
            VariantDef {
                name: "None".to_string(),
                kind: VariantKind::Unit,
            },
            VariantDef {
                name: "Some".to_string(),
                kind: VariantKind::Tuple(vec![t.clone()]),
            },
        ],
    }
}

fn create_option_impl() -> ImplBlock {
    // impl<T> Option<T>
    let t = Type::UserDefined("T".to_string(), vec![]);
    let opt_t = Type::UserDefined("Option".to_string(), vec![t.clone()]);
    
    ImplBlock {
        target_type: opt_t.clone(),
        generics: vec!["T".to_string()],
        methods: vec![
            // fn is_some(self) -> Bool
            FunctionDef {
                name: "is_some".to_string(),
                args: vec![("self".to_string(), opt_t.clone())],
                return_type: Type::Bool,
                body: vec![],
                generics: vec![],
                is_extern: true,
            },
            // fn is_none(self) -> Bool
            FunctionDef {
                name: "is_none".to_string(),
                args: vec![("self".to_string(), opt_t.clone())],
                return_type: Type::Bool,
                body: vec![],
                generics: vec![],
                is_extern: true,
            },
            // fn unwrap(self) -> T
            FunctionDef {
                 name: "unwrap".to_string(),
                 args: vec![("self".to_string(), opt_t.clone())],
                 return_type: t.clone(),
                 body: vec![],
                 generics: vec![],
                 is_extern: true,
             },
             // fn unwrap_or(self, default: T) -> T
             FunctionDef {
                 name: "unwrap_or".to_string(),
                 args: vec![
                     ("self".to_string(), opt_t.clone()),
                     ("default".to_string(), t.clone())
                 ],
                 return_type: t.clone(),
                 body: vec![],
                 generics: vec![],
                 is_extern: true,
             }
        ]
    }
}

fn create_result_enum() -> EnumDef {
    // enum Result<T, E> { Ok(T), Err(E) }
    let t = Type::UserDefined("T".to_string(), vec![]);
    let e = Type::UserDefined("E".to_string(), vec![]);
    
    EnumDef {
        name: "Result".to_string(),
        generics: vec!["T".to_string(), "E".to_string()],
        variants: vec![
            VariantDef {
                name: "Ok".to_string(),
                kind: VariantKind::Tuple(vec![t.clone()]),
            },
            VariantDef {
                name: "Err".to_string(),
                kind: VariantKind::Tuple(vec![e.clone()]),
            },
        ],
    }
}

fn create_result_impl() -> ImplBlock {
    // impl<T, E> Result<T, E>
    let t = Type::UserDefined("T".to_string(), vec![]);
    let e = Type::UserDefined("E".to_string(), vec![]);
    let res_t = Type::UserDefined("Result".to_string(), vec![t.clone(), e.clone()]);
    
    ImplBlock {
        target_type: res_t.clone(),
        generics: vec!["T".to_string(), "E".to_string()],
        methods: vec![
            // fn is_ok(self) -> Bool
            FunctionDef {
                name: "is_ok".to_string(),
                args: vec![("self".to_string(), res_t.clone())],
                return_type: Type::Bool,
                body: vec![
                    Stmt::dummy(StmtKind::Return(Some(Spanned::dummy(ExprKind::Match {
                        expr: Box::new(Spanned::dummy(ExprKind::Variable("self".to_string()))),
                        arms: vec![
                            (
                                Pattern::EnumPattern {
                                    enum_name: "Result".to_string(),
                                    variant_name: "Ok".to_string(),
                                    bindings: EnumPatternBindings::Tuple(vec!["_".to_string()]),
                                },
                                Spanned::dummy(ExprKind::Bool(true))
                            ),
                            (
                                Pattern::EnumPattern {
                                    enum_name: "Result".to_string(),
                                    variant_name: "Err".to_string(),
                                    bindings: EnumPatternBindings::Tuple(vec!["_".to_string()]),
                                },
                                Spanned::dummy(ExprKind::Bool(false))
                            )
                        ]
                    }))))
                ],
                generics: vec![],
                is_extern: false,
            },
            // fn is_err(self) -> Bool
            FunctionDef {
                name: "is_err".to_string(),
                args: vec![("self".to_string(), res_t.clone())],
                return_type: Type::Bool,
                body: vec![
                    Stmt::dummy(StmtKind::Return(Some(Spanned::dummy(ExprKind::Match {
                        expr: Box::new(Spanned::dummy(ExprKind::Variable("self".to_string()))),
                        arms: vec![
                            (
                                Pattern::EnumPattern {
                                    enum_name: "Result".to_string(),
                                    variant_name: "Ok".to_string(),
                                    bindings: EnumPatternBindings::Tuple(vec!["_".to_string()]),
                                },
                                Spanned::dummy(ExprKind::Bool(false))
                            ),
                            (
                                Pattern::EnumPattern {
                                    enum_name: "Result".to_string(),
                                    variant_name: "Err".to_string(),
                                    bindings: EnumPatternBindings::Tuple(vec!["_".to_string()]),
                                },
                                Spanned::dummy(ExprKind::Bool(true))
                            )
                        ]
                    }))))
                ],
                generics: vec![],
                is_extern: false,
            },
            // fn unwrap(self) -> T
            // Note: unwrap() assumes Ok. If Err, it should panic (handled in codegen).
            FunctionDef {
                name: "unwrap".to_string(),
                args: vec![("self".to_string(), res_t.clone())],
                return_type: t.clone(),
                body: vec![
                    Stmt::dummy(StmtKind::Return(Some(Spanned::dummy(ExprKind::Match {
                        expr: Box::new(Spanned::dummy(ExprKind::Variable("self".to_string()))),
                        arms: vec![
                            (
                                Pattern::EnumPattern {
                                    enum_name: "Result".to_string(),
                                    variant_name: "Ok".to_string(),
                                    bindings: EnumPatternBindings::Tuple(vec!["val".to_string()]),
                                },
                                Spanned::dummy(ExprKind::Variable("val".to_string()))
                            ),
                            (
                                Pattern::EnumPattern {
                                    enum_name: "Result".to_string(),
                                    variant_name: "Err".to_string(),
                                    bindings: EnumPatternBindings::Tuple(vec!["_".to_string()]),
                                },
                                // Recursive call to satisfy type checker and generate valid terminator
                                Spanned::dummy(ExprKind::MethodCall(
                                    Box::new(Spanned::dummy(ExprKind::Variable("self".to_string()))),
                                    "unwrap".to_string(),
                                    vec![]
                                ))
                            )
                        ]
                    }))))
                ],
                generics: vec![],
                is_extern: false,
            },
        ]
    }
}
