use crate::compiler::ast::*;
use crate::compiler::error::Span;
use std::collections::HashMap;

/// Register built-in generic implementations (e.g., Vec<T>)
pub fn register_builtin_impls(generic_impls: &mut HashMap<String, Vec<ImplBlock>>) {
    let vec_impl = create_vec_impl();
    generic_impls.insert("Vec".to_string(), vec![vec_impl]);
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
    struct_defs.insert("Vec".to_string(), vec_struct);
}

fn create_vec_impl() -> ImplBlock {
    let t_type = Type::UserDefined("T".to_string(), vec![]);
    let int_type = Type::I64;
    let bool_type = Type::Bool;
    let void_type = Type::Void;

    // --- Helpers for creating AST nodes ---
    // Explicitly typed to avoid inference locking
    fn expr(kind: ExprKind) -> Expr {
        Spanned::new(kind, Span::default())
    }
    fn stmt(kind: StmtKind) -> Stmt {
        Spanned::new(kind, Span::default())
    }
    
    // self variable expression
    let self_expr = || expr(ExprKind::Variable("self".to_string()));
    
    // access field: self.field
    let field_access = |field: &str| expr(ExprKind::FieldAccess(
        Box::new(self_expr()),
        field.to_string(),
    ));

    // --- pop() -> T ---
    // self.len -= 1
    // return self.ptr[self.len]
    let pop_body = vec![
        stmt(StmtKind::FieldAssign {
             obj: self_expr(),
             field: "len".to_string(),
             op: AssignOp::Assign,
             value: expr(ExprKind::BinOp(
                 Box::new(field_access("len")),
                 BinOp::Sub,
                 Box::new(expr(ExprKind::Int(1)))
             )),
         }),
         stmt(StmtKind::Return(Some(expr(ExprKind::IndexAccess(
             Box::new(field_access("ptr")),
             vec![field_access("len")] // Use the new length as index
         )))))
    ];

    let pop_method = FunctionDef {
        name: "pop".to_string(),
        args: vec![("self".to_string(), Type::UserDefined("Vec".to_string(), vec![t_type.clone()]))],
        return_type: t_type.clone(),
        body: pop_body,
        generics: vec![],
        is_extern: false, 
    };

    // --- is_empty() -> Bool ---
    let is_empty_body = vec![
        stmt(StmtKind::Return(Some(expr(ExprKind::BinOp(
            Box::new(field_access("len")),
            BinOp::Eq,
            Box::new(expr(ExprKind::Int(0)))
        )))))
    ];
    let is_empty_method = FunctionDef {
        name: "is_empty".to_string(),
        args: vec![("self".to_string(), Type::UserDefined("Vec".to_string(), vec![t_type.clone()]))],
        return_type: bool_type.clone(),
        body: is_empty_body,
        generics: vec![],
        is_extern: false,
    };

    // --- clear() -> Void ---
    let clear_body = vec![
        stmt(StmtKind::FieldAssign {
            obj: self_expr(),
            field: "len".to_string(),
            op: AssignOp::Assign,
            value: expr(ExprKind::Int(0))
        })
    ];
    let clear_method = FunctionDef {
        name: "clear".to_string(),
        args: vec![("self".to_string(), Type::UserDefined("Vec".to_string(), vec![t_type.clone()]))],
        return_type: void_type.clone(),
        body: clear_body,
        generics: vec![],
        is_extern: false,
    };
    
    // --- insert(idx, val) ---
    // i = len; while i > idx { ptr[i] = ptr[i-1]; i -= 1; } ptr[idx] = val; len += 1;
    let insert_body = vec![
        // let mut i = self.len;
        stmt(StmtKind::Let {
            name: "i".to_string(),
            type_annotation: Some(int_type.clone()),
            value: field_access("len"),
            mutable: true,
        }),
        // Loop
        stmt(StmtKind::Loop {
            body: vec![
                // if i <= idx { break; }
                stmt(StmtKind::Expr(expr(ExprKind::IfExpr(
                    Box::new(expr(ExprKind::BinOp(
                        Box::new(expr(ExprKind::Variable("i".to_string()))),
                        BinOp::Le,
                        Box::new(expr(ExprKind::Variable("idx".to_string())))
                    ))),
                    vec![stmt(StmtKind::Break)],
                    None,
                )))),
                // ptr[i] = ptr[i-1]
                stmt(StmtKind::Assign {
                    name: "ptr".to_string(),
                    indices: Some(vec![expr(ExprKind::Variable("i".to_string()))]),
                    op: AssignOp::Assign,
                    value: expr(ExprKind::IndexAccess(
                        Box::new(expr(ExprKind::Variable("ptr".to_string()))),
                        vec![expr(ExprKind::BinOp(
                            Box::new(expr(ExprKind::Variable("i".to_string()))),
                            BinOp::Sub,
                            Box::new(expr(ExprKind::Int(1)))
                        ))]
                    )),
                }),
                // i -= 1
                stmt(StmtKind::Assign {
                    name: "i".to_string(),
                    indices: None,
                    op: AssignOp::SubAssign,
                    value: expr(ExprKind::Int(1)),
                }),
            ]
        }),
        // ptr[idx] = val
        stmt(StmtKind::Assign {
            name: "ptr".to_string(),
            indices: Some(vec![expr(ExprKind::Variable("idx".to_string()))]),
            op: AssignOp::Assign,
            value: expr(ExprKind::Variable("val".to_string())),
        }),
        // self.len += 1
        stmt(StmtKind::FieldAssign {
            obj: self_expr(),
            field: "len".to_string(),
            op: AssignOp::AddAssign,
            value: expr(ExprKind::Int(1)),
        }),
    ];
    
    // Wrapper to get ptr alias
    let insert_preamble = vec![
        stmt(StmtKind::Let {
            name: "ptr".to_string(),
            type_annotation: None, 
            value: field_access("ptr"),
            mutable: false, 
        })
    ];
    
    let insert_method = FunctionDef {
        name: "insert".to_string(),
        args: vec![("self".to_string(), Type::UserDefined("Vec".to_string(), vec![t_type.clone()])), ("idx".to_string(), int_type.clone()), ("val".to_string(), t_type.clone())],
        return_type: void_type.clone(),
        body: [insert_preamble, insert_body].concat(),
        generics: vec![],
        is_extern: false,
    };
    
    // --- remove(idx) -> T ---
    let remove_body = vec![
        // let ptr = self.ptr;
        stmt(StmtKind::Let {
            name: "ptr".to_string(),
            type_annotation: None,
            value: field_access("ptr"),
            mutable: false,
        }),
        // let result = ptr[idx];
        stmt(StmtKind::Let {
            name: "result".to_string(),
            type_annotation: Some(t_type.clone()),
            value: expr(ExprKind::IndexAccess(
                Box::new(expr(ExprKind::Variable("ptr".to_string()))),
                vec![expr(ExprKind::Variable("idx".to_string()))]
            )),
            mutable: false,
        }),
        // let mut i = idx;
        stmt(StmtKind::Let {
            name: "i".to_string(),
            type_annotation: Some(int_type.clone()),
            value: expr(ExprKind::Variable("idx".to_string())),
            mutable: true,
        }),
        // Loop
        stmt(StmtKind::Loop {
            body: vec![
                // if i >= self.len - 1 { break; }
                stmt(StmtKind::Expr(expr(ExprKind::IfExpr(
                    Box::new(expr(ExprKind::BinOp(
                        Box::new(expr(ExprKind::Variable("i".to_string()))),
                        BinOp::Ge,
                        Box::new(expr(ExprKind::BinOp(
                            Box::new(field_access("len")),
                            BinOp::Sub,
                            Box::new(expr(ExprKind::Int(1)))
                        )))
                    ))),
                    vec![stmt(StmtKind::Break)],
                    None,
                )))),
                // ptr[i] = ptr[i+1]
                stmt(StmtKind::Assign {
                    name: "ptr".to_string(),
                    indices: Some(vec![expr(ExprKind::Variable("i".to_string()))]),
                    op: AssignOp::Assign,
                    value: expr(ExprKind::IndexAccess(
                        Box::new(expr(ExprKind::Variable("ptr".to_string()))),
                        vec![expr(ExprKind::BinOp(
                            Box::new(expr(ExprKind::Variable("i".to_string()))),
                            BinOp::Add,
                            Box::new(expr(ExprKind::Int(1)))
                        ))]
                    )),
                }),
                // i += 1
                stmt(StmtKind::Assign {
                    name: "i".to_string(),
                    indices: None,
                    op: AssignOp::AddAssign,
                    value: expr(ExprKind::Int(1)),
                }),
            ]
        }),
        // self.len -= 1
        stmt(StmtKind::FieldAssign {
            obj: self_expr(),
            field: "len".to_string(),
            op: AssignOp::SubAssign,
            value: expr(ExprKind::Int(1)),
        }),
        // return result
        stmt(StmtKind::Return(Some(expr(ExprKind::Variable("result".to_string()))))),
    ];
    let remove_method = FunctionDef {
        name: "remove".to_string(),
        args: vec![("self".to_string(), Type::UserDefined("Vec".to_string(), vec![t_type.clone()])), ("idx".to_string(), int_type.clone())],
        return_type: t_type.clone(),
        body: remove_body,
        generics: vec![],
        is_extern: false,
    };

    // --- contains(val) -> Bool ---
    let contains_body = vec![
        // let ptr = self.ptr;
        stmt(StmtKind::Let {
            name: "ptr".to_string(),
            type_annotation: None,
            value: field_access("ptr"),
            mutable: false,
        }),
        // let mut i = 0;
        stmt(StmtKind::Let {
            name: "i".to_string(),
            type_annotation: Some(int_type.clone()),
            value: expr(ExprKind::Int(0)),
            mutable: true,
        }),
        // Loop
        stmt(StmtKind::Loop {
            body: vec![
                // if i >= self.len { return false; }
                stmt(StmtKind::Expr(expr(ExprKind::IfExpr(
                    Box::new(expr(ExprKind::BinOp(
                        Box::new(expr(ExprKind::Variable("i".to_string()))),
                        BinOp::Ge,
                        Box::new(field_access("len"))
                    ))),
                    vec![stmt(StmtKind::Return(Some(expr(ExprKind::Bool(false)))))],
                    None,
                )))),
                // if ptr[i] == val { return true; }
                stmt(StmtKind::Expr(expr(ExprKind::IfExpr(
                    Box::new(expr(ExprKind::BinOp(
                        Box::new(expr(ExprKind::IndexAccess(
                            Box::new(expr(ExprKind::Variable("ptr".to_string()))),
                            vec![expr(ExprKind::Variable("i".to_string()))]
                        ))),
                        BinOp::Eq,
                        Box::new(expr(ExprKind::Variable("val".to_string())))
                    ))),
                    vec![stmt(StmtKind::Return(Some(expr(ExprKind::Bool(true)))))],
                    None,
                )))),
                // i += 1
                stmt(StmtKind::Assign {
                    name: "i".to_string(),
                    indices: None,
                    op: AssignOp::AddAssign,
                    value: expr(ExprKind::Int(1)),
                }),
            ]
        }),
    ];
    let contains_method = FunctionDef {
        name: "contains".to_string(),
        args: vec![("self".to_string(), Type::UserDefined("Vec".to_string(), vec![t_type.clone()])), ("val".to_string(), t_type.clone())],
        return_type: bool_type.clone(),
        body: contains_body,
        generics: vec![],
        is_extern: false,
    };

    ImplBlock {
        target_type: Type::UserDefined("Vec".to_string(), vec![t_type.clone()]),
        generics: vec!["T".to_string()],
        methods: vec![pop_method, is_empty_method, clear_method, insert_method, remove_method, contains_method],
    }
}
