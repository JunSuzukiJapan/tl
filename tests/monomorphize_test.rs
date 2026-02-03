use tl_lang::compiler::monomorphize::Monomorphizer;
use tl_lang::compiler::ast::*;
use std::collections::HashMap;

fn create_empty_module() -> Module {
    Module {
        structs: vec![],
        functions: vec![],
        enums: vec![],
        impls: vec![],
        tensor_decls: vec![],
        relations: vec![],
        rules: vec![],
        queries: vec![],
        imports: vec![],
        submodules: HashMap::new(),
    }
}

fn create_generic_function(name: &str, param: &str) -> FunctionDef {
    FunctionDef {
        name: name.to_string(),
        generics: vec![param.to_string()],
        args: vec![
            ("arg".to_string(), Type::Struct(param.to_string(), vec![]))
        ],
        return_type: Type::Struct(param.to_string(), vec![]),
        body: vec![
            Stmt::dummy(StmtKind::Return(Some(Expr::dummy(ExprKind::Variable("arg".to_string())))))
        ],
        is_extern: false,
    }
}


#[test]
fn test_basic_function_instantiation() {
    let mut mono = Monomorphizer::new();
    let mut module = create_empty_module();
    let func_def = create_generic_function("id", "T");
    module.functions.push(func_def);

    let main_stmts = vec![
        Stmt::dummy(StmtKind::Expr(Expr::dummy(ExprKind::FnCall(
                "id".to_string(),
                vec![ Expr::dummy(ExprKind::Int(10)) ]
        ))))
    ];
    let main_func = FunctionDef {
        name: "main".to_string(),
        generics: vec![],
        args: vec![],
        return_type: Type::Void,
        body: main_stmts,
        is_extern: false,
    };
    module.functions.push(main_func);

    mono.run(&mut module).expect("Monomorphization failed");

    let concrete = module.functions.iter().find(|f| f.name == "id_i64");
    assert!(concrete.is_some(), "id_i64 should be generated");
    let concrete = concrete.unwrap();
    assert_eq!(concrete.args[0].1, Type::I64);
    assert_eq!(concrete.return_type, Type::I64);
}

#[test]
fn test_deduplication() {
        let mut mono = Monomorphizer::new();
    let mut module = create_empty_module();
    let func_def = create_generic_function("id", "T");
    module.functions.push(func_def);

    let main_stmts = vec![
        Stmt::dummy(StmtKind::Expr(Expr::dummy(ExprKind::FnCall(
                "id".to_string(),
                vec![ Expr::dummy(ExprKind::Int(10)) ]
        )))),
        Stmt::dummy(StmtKind::Expr(Expr::dummy(ExprKind::FnCall(
                "id".to_string(),
                vec![ Expr::dummy(ExprKind::Int(20)) ]
        ))))
    ];
    let main_func = FunctionDef {
        name: "main".to_string(),
        generics: vec![],
        args: vec![],
        return_type: Type::Void,
        body: main_stmts,
        is_extern: false,
    };
    module.functions.push(main_func);

    mono.run(&mut module).expect("Monomorphization failed");
    
    let count = module.functions.iter().filter(|f| f.name == "id_i64").count();
    assert_eq!(count, 1, "Should only have one instance of id_i64");
}
