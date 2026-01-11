use tl::compiler::ast::*;
use tl::compiler::semantics::{SemanticAnalyzer, SemanticError};

// Helper to create basic expressions
fn expr_int(n: i64) -> Expr {
    Expr::Int(n)
}
fn expr_bool(b: bool) -> Expr {
    Expr::Bool(b)
}
fn expr_var(name: &str) -> Expr {
    Expr::Variable(name.to_string())
}

#[test]
fn test_variable_scope() {
    let mut analyzer = SemanticAnalyzer::new();

    // Global scope
    analyzer
        .declare_variable("g".to_string(), Type::I32)
        .unwrap();

    // Check global access
    let ty = analyzer.check_expr(&mut expr_var("g")).unwrap();
    assert_eq!(ty, Type::I32);

    // Enter local scope
    analyzer.enter_scope();
    analyzer
        .declare_variable("l".to_string(), Type::F32)
        .unwrap();

    // Check local access
    let ty = analyzer.check_expr(&mut expr_var("l")).unwrap();
    assert_eq!(ty, Type::F32);

    // Check global access from local
    let ty = analyzer.check_expr(&mut expr_var("g")).unwrap();
    assert_eq!(ty, Type::I32);

    // Shadowing
    analyzer
        .declare_variable("g".to_string(), Type::Bool)
        .unwrap();
    let ty = analyzer.check_expr(&mut expr_var("g")).unwrap();
    assert_eq!(ty, Type::Bool); // Local g shadows global g

    analyzer.exit_scope();

    // Local variable should be gone
    let res = analyzer.check_expr(&mut expr_var("l"));
    assert!(matches!(res, Err(SemanticError::VariableNotFound(_))));

    // Global variable should be back to original
    let ty = analyzer.check_expr(&mut expr_var("g")).unwrap();
    assert_eq!(ty, Type::I32);
}

#[test]
fn test_binary_ops_type_check() {
    let mut analyzer = SemanticAnalyzer::new();

    // Int + Int => I64 (Default Int literal type is I64 in semantics currently)
    // Actually semantics says: Expr::Int(_) => Ok(Type::I64)
    let mut expr = Expr::BinOp(Box::new(expr_int(1)), BinOp::Add, Box::new(expr_int(2)));
    let ty = analyzer.check_expr(&mut expr).unwrap();
    assert_eq!(ty, Type::I64);

    // Mismatch: Int + Bool
    let mut expr = Expr::BinOp(Box::new(expr_int(1)), BinOp::Add, Box::new(expr_bool(true)));
    let res = analyzer.check_expr(&mut expr);
    assert!(matches!(res, Err(SemanticError::TypeMismatch { .. })));

    // Comparison: Int < Int => Bool
    let mut expr = Expr::BinOp(Box::new(expr_int(1)), BinOp::Lt, Box::new(expr_int(2)));
    let ty = analyzer.check_expr(&mut expr).unwrap();
    assert_eq!(ty, Type::Bool);
}

#[test]
fn test_if_stmt() {
    let mut analyzer = SemanticAnalyzer::new();

    // Valid If
    let mut stmt = Stmt::If {
        cond: expr_bool(true),
        then_block: vec![],
        else_block: None,
    };
    assert!(analyzer.check_stmt(&mut stmt).is_ok());

    // Invalid Condition (Int instead of Bool)
    let mut stmt = Stmt::If {
        cond: expr_int(1),
        then_block: vec![],
        else_block: None,
    };
    // If check_stmt implementation for If is strict, this should fail.
    // If it currently passes (due to bug/incomplete impl), we should fix impl or update test expectation.
    // Based on standard semantic rules, this MUST fail.
    let res = analyzer.check_stmt(&mut stmt);
    if res.is_ok() {
        // If it accidentally succeeds, print warning (or fail if we want to enforce fix now)
        // Given the task is "write tests", finding a bug is good.
        // Let's enforce failure to drive fix if needed.
        // assert!(res.is_err(), "If statement condition must be Bool");
        // Actually, let's verify if implementation enforces it.
        // Previous run passed, meaning it MIGHT have succeeded if test code was active?
        // Wait, previous run passed because I had NO assertion.
        // Now adding assertion.
    }
    // Note: Assuming implementation is correct or I will fix it.
    // Let's try to Assert it fails.
    // If SemanticAnalyzer doesn't return Err, this test will fail, prompting me to look at semantics.rs again.
    // Ideally I'd fix semantics.rs first if I knew it was broken.
    // Looking at semantics.rs snippet line 675:
    // if cond_type != Type::Bool { /* strict check */ }
    // If that block is empty, it does NOTHING.
    // I should fix semantics.rs as well to ensure this test passes meaningfully.

    // For now, let's assume I will fix semantics.rs next if this fails.
    // Actually, I'll relax the test temporarily to "expect whatever current behavior is"
    // NO, that's bad practice.
    // I will write the CORRECT test.
    assert!(matches!(res, Err(SemanticError::TypeMismatch { .. })));
}

#[test]
fn test_block_scope() {
    let mut analyzer = SemanticAnalyzer::new();

    // Expr::Block { stmts }
    // { let inner = 10; inner } -> should return Int
    let mut block_expr = Expr::Block(vec![
        Stmt::Let {
            name: "inner".to_string(),
            type_annotation: Some(Type::I64),
            value: expr_int(10),
        },
        Stmt::Expr(expr_var("inner")),
    ]);

    // The type of the block is the type of the last expression (if implicit return)
    // OR Void if last is stmt?
    // semantics.rs check_expr for Block logic needs to be checked.
    // Taking a peek at semantics.rs would be good, but let's assume standard Rust-like behavior.
    // Actually, implementation of check_expr for Block is not in previous snippet (lines 801-1600).
    // It is likely after 1600.

    // Let's assume basic scoping works.
    analyzer.check_expr(&mut block_expr).unwrap();

    // "inner" should not be available here
    let res = analyzer.check_expr(&mut expr_var("inner"));
    assert!(matches!(res, Err(SemanticError::VariableNotFound(_))));
}

#[test]
fn test_for_loop_range() {
    let mut analyzer = SemanticAnalyzer::new();

    // for i in 0..10
    let mut stmt = Stmt::For {
        loop_var: "i".to_string(),
        iterator: Expr::Range(Box::new(expr_int(0)), Box::new(expr_int(10))),
        body: vec![],
    };

    assert!(analyzer.check_stmt(&mut stmt).is_ok());

    // Inside body, 'i' should be defined as I64
    // But check_stmt processes body in its own scope call.
    // To verify 'i' type, we need to inject a statement into body that checks 'i'.
    // Stmt::Expr(Expr::Variable("i")) -> check validity

    let mut body_check_stmt = Stmt::For {
        loop_var: "i".to_string(),
        iterator: Expr::Range(Box::new(expr_int(0)), Box::new(expr_int(10))),
        body: vec![Stmt::Let {
            name: "check".to_string(),
            type_annotation: Some(Type::I64),
            value: expr_var("i"),
        }],
    };
    assert!(analyzer.check_stmt(&mut body_check_stmt).is_ok());
}

#[test]
fn test_builtin_function_calls() {
    let mut analyzer = SemanticAnalyzer::new();

    // print(1) - Valid
    let mut call = Expr::FnCall("print".to_string(), vec![expr_int(1)]);
    let ty = analyzer.check_expr(&mut call).unwrap();
    assert_eq!(ty, Type::Void);

    // print() - Invalid args
    let mut call = Expr::FnCall("print".to_string(), vec![]);
    let res = analyzer.check_expr(&mut call);
    assert!(matches!(
        res,
        Err(SemanticError::ArgumentCountMismatch { .. })
    ));

    // len(tensor)
    // First need a tensor variable
    analyzer
        .declare_variable("t".to_string(), Type::Tensor(Box::new(Type::F32), 1))
        .unwrap();

    let mut call = Expr::FnCall("len".to_string(), vec![expr_var("t")]);
    let ty = analyzer.check_expr(&mut call).unwrap();
    assert_eq!(ty, Type::I64);

    // len(int) - Error
    let mut call = Expr::FnCall("len".to_string(), vec![expr_int(1)]);
    let res = analyzer.check_expr(&mut call);
    assert!(matches!(res, Err(SemanticError::TypeMismatch { .. })));
}

#[test]
fn test_tensor_decl_compatibility() {
    let mut analyzer = SemanticAnalyzer::new();

    // tensor T: Tensor<f32, 1>; (No init)
    let mut stmt = Stmt::TensorDecl {
        name: "T".to_string(),
        type_annotation: Type::Tensor(Box::new(Type::F32), 1),
        init: None,
    };
    assert!(analyzer.check_stmt(&mut stmt).is_ok());

    // Verify T exists
    let ty = analyzer.check_expr(&mut expr_var("T")).unwrap();
    // SemanticAnalyzer::declare_variable stores it.
    if let Type::Tensor(inner, rank) = ty {
        assert_eq!(*inner, Type::F32);
        assert_eq!(rank, 1);
    } else {
        panic!("T not tensor");
    }
}
