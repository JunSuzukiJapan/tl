use tl_lang::compiler::lexer::{tokenize};
use tl_lang::compiler::ast::{ExprKind, StmtKind, Type, AssignOp, BinOp, UnOp};
use tl_lang::compiler::parser::{parse_expr, parse_stmt, parse, parse_type};

fn p_expr(input: &str) -> ExprKind {
    let tokens_res = tokenize(input);
    let tokens: Vec<_> = tokens_res.into_iter().map(|r| r.unwrap()).collect();
    let (rest, expr) = parse_expr(&tokens).expect("Parse error");
    assert!(rest.is_empty(), "Trailing tokens: {:?}", rest);
    expr.inner
}

fn p_stmt(input: &str) -> StmtKind {
    let tokens_res = tokenize(input);
    let tokens: Vec<_> = tokens_res.into_iter().map(|r| r.unwrap()).collect();
    let (rest, stmt) = parse_stmt(&tokens).expect("Parse error");
    assert!(rest.is_empty(), "Trailing tokens: {:?}", rest);
    stmt.inner
}

fn p_type(input: &str) -> Type {
    let tokens_res = tokenize(input);
    let tokens: Vec<_> = tokens_res.into_iter().map(|r| r.unwrap()).collect();
    let (rest, ty) = parse_type(&tokens).expect("Parse error");
    assert!(rest.is_empty(), "Trailing tokens: {:?}", rest);
    ty
}

#[test]
fn test_types() {
    let t = p_type("f32");
    assert_eq!(t, Type::F32);

    let t = p_type("Tensor<f32, 2>");
    if let Type::Tensor(inner, rank) = t {
        assert_eq!(*inner, Type::F32);
        assert_eq!(rank, 2);
    } else {
        panic!("Expected Tensor type");
    }

    let t = p_type("(f32, i32)");
    if let Type::Tuple(ts) = t {
        assert_eq!(ts.len(), 2);
    } else {
        panic!("Expected Tuple type");
    }
}

#[test]
fn test_literals() {
    assert_eq!(p_expr("123"), ExprKind::Int(123));
    assert_eq!(p_expr("123.456"), ExprKind::Float(123.456));
    assert_eq!(p_expr("true"), ExprKind::Bool(true));
    assert_eq!(p_expr("false"), ExprKind::Bool(false));
    // String literal includes quotes in parser logic but ast stores content
    // parse_literal_string: "abc" -> StringLiteral("abc")
    assert_eq!(
        p_expr("\"abc\""),
        ExprKind::StringLiteral("abc".to_string())
    );
}

#[test]
fn test_variables() {
    assert_eq!(p_expr("x"), ExprKind::Variable("x".to_string()));
    assert_eq!(
        p_expr("my_var_1"),
        ExprKind::Variable("my_var_1".to_string())
    );
}

#[test]
fn test_binary_ops() {
    // Simple
    if let ExprKind::BinOp(l, op, r) = p_expr("1 + 2") {
        assert_eq!(l.inner, ExprKind::Int(1));
        assert_eq!(op, BinOp::Add);
        assert_eq!(r.inner, ExprKind::Int(2));
    } else {
        panic!("Not a BinOp");
    }

    // Precedence: 1 + 2 * 3 => 1 + (2 * 3)
    if let ExprKind::BinOp(l, op, r) = p_expr("1 + 2 * 3") {
        assert_eq!(l.inner, ExprKind::Int(1));
        assert_eq!(op, BinOp::Add);
        if let ExprKind::BinOp(l2, op2, r2) = &r.inner {
            assert_eq!(l2.inner, ExprKind::Int(2));
            assert_eq!(op2, &BinOp::Mul);
            assert_eq!(r2.inner, ExprKind::Int(3));
        } else {
            panic!("RHS not mul");
        }
    } else {
        panic!("Not add");
    }

    // Comparison
    match p_expr("x <= y") {
        ExprKind::BinOp(_, BinOp::Le, _) => {}
        _ => panic!("Expected Le"),
    }

    // Logical
    match p_expr("a && b || c") {
        // (a && b) || c
        ExprKind::BinOp(l, BinOp::Or, _) => match &l.inner {
            ExprKind::BinOp(_, BinOp::And, _) => {}
            _ => panic!("LHS expected And"),
        },
        _ => panic!("Expected Or"),
    }
}

#[test]
fn test_unary_ops() {
    if let ExprKind::UnOp(op, e) = p_expr("-x") {
        assert_eq!(op, UnOp::Neg);
        assert_eq!(e.inner, ExprKind::Variable("x".to_string()));
    } else {
        panic!("Expected Neg");
    }

    if let ExprKind::UnOp(op, _) = p_expr("!true") {
        assert_eq!(op, UnOp::Not);
    } else {
        panic!("Expected Not");
    }
}

#[test]
fn test_postfix() {
    // Call
    if let ExprKind::FnCall(name, args) = p_expr("foo(1, 2)") {
        assert_eq!(name, "foo");
        assert_eq!(args.len(), 2);
    } else {
        panic!("Expected FnCall");
    }

    // Method
    if let ExprKind::MethodCall(obj, name, args) = p_expr("x.bar(3)") {
        assert_eq!(obj.inner, ExprKind::Variable("x".to_string()));
        assert_eq!(name, "bar");
        assert_eq!(args.len(), 1);
    } else {
        panic!("Expected MethodCall");
    }

    // Index
    if let ExprKind::IndexAccess(obj, indices) = p_expr("A[i, j]") {
        assert_eq!(obj.inner, ExprKind::Variable("A".to_string()));
        assert_eq!(indices.len(), 2);
    } else {
        panic!("Expected IndexAccess");
    }

    // Field
    if let ExprKind::FieldAccess(obj, field) = p_expr("s.field") {
        assert_eq!(obj.inner, ExprKind::Variable("s".to_string()));
        assert_eq!(field, "field");
    } else {
        panic!("Expected FieldAccess");
    }

    // Tuple Access
    if let ExprKind::TupleAccess(obj, idx) = p_expr("t.0") {
        assert_eq!(obj.inner, ExprKind::Variable("t".to_string()));
        assert_eq!(idx, 0);
    } else {
        panic!("Expected TupleAccess");
    }

    // Static Method
    if let ExprKind::StaticMethodCall(type_name, method, _) = p_expr("MyType::new()") {
        // assert_eq!(type_name, "MyType"); // Legacy
        assert_eq!(type_name, Type::Struct("MyType".to_string(), vec![]));
        assert_eq!(method, "new");
    } else {
        panic!("Expected StaticMethodCall");
    }
}

#[test]
fn test_cast() {
    if let ExprKind::As(expr, ty) = p_expr("x as f32") {
        assert_eq!(expr.inner, ExprKind::Variable("x".to_string()));
        assert_eq!(ty, Type::F32);
    } else {
        panic!("Expected Cast");
    }
}

#[test]
fn test_range() {
    if let ExprKind::Range(start, end) = p_expr("0..10") {
        assert_eq!(start.inner, ExprKind::Int(0));
        assert_eq!(end.inner, ExprKind::Int(10));
    } else {
        panic!("Expected Range");
    }
}


#[test]
fn test_statements() {
    match p_stmt("let x = 1;") {
        StmtKind::Let { name, value, .. } => {
            assert_eq!(name, "x");
            assert_eq!(value.inner, ExprKind::Int(1));
        }
        _ => panic!("Expected Let"),
    }

    match p_stmt("x += 5;") {
        StmtKind::Assign { name, op, .. } => {
            assert_eq!(name, "x");
            assert_eq!(op, AssignOp::AddAssign);
        }
        _ => panic!("Expected Assign"),
    }

    match p_stmt("return 42;") {
        StmtKind::Return(Some(val)) => {
            assert_eq!(val.inner, ExprKind::Int(42));
        }
        _ => panic!("Expected Return"),
    }

    match p_stmt("if true { } else { }") {
        StmtKind::Expr(expr) => match expr.inner {
            ExprKind::IfExpr(_, _, _) => {}
            _ => panic!("Expected IfExpr"),
        },
        _ => panic!("Expected Expr containing IfExpr"),
    }

    match p_stmt("for i in 0..10 { }") {
        StmtKind::For { .. } => {}
        _ => panic!("Expected For"),
    }

    match p_stmt("while x < 10 { }") {
        StmtKind::While { .. } => {}
        _ => panic!("Expected While"),
    }
}

#[test]
fn test_top_level() {
    let input = r#"
        mod foo;
        use std::math;
        
        struct Point { x: f32, y: f32 }
        
        impl Point {
            fn new(x: f32, y: f32) -> Point {
                Point { x: x, y: y }
            }
        }
        
        fn main() {
            let p = Point::new(1.0, 2.0);
        }
        
        tensor W: Tensor<f32, 2>;
    "#;

    let tokens_res = tokenize(input);
    let tokens: Vec<_> = tokens_res.into_iter().map(|r| r.unwrap()).collect();
    let module = parse(&tokens).expect("Failed to parse module");
    assert_eq!(module.imports.len(), 1);
    assert!(module.imports.contains(&"std::math".to_string()));
    assert_eq!(module.submodules.len(), 1);
    assert!(module.submodules.contains_key("foo"));

    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.structs[0].name, "Point");

    assert_eq!(module.impls.len(), 1);
    // assert_eq!(module.impls[0].target_type, "Point");
    match &module.impls[0].target_type {
        Type::Struct(name, _) => assert_eq!(name, "Point"),
        _ => panic!("Expected UserDefined or Struct type"),
    }

    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "main");

    // use stmt + tensor decl
    assert_eq!(module.tensor_decls.len(), 1);
}
#[test]
fn test_match_expr() {
    let input = r#"
    match x {
        Ok(v) => v,
        Err(e) => 0,
        _ => -1,
    }
    "#;
    if let ExprKind::Match { expr: target, arms } = p_expr(input) {
        assert_eq!(target.inner, ExprKind::Variable("x".to_string()));
        assert_eq!(arms.len(), 3);
    } else {
        panic!("Expected Match");
    }
}

#[test]
fn test_if_let() {
    let input = "if let Some(x) = opt { x } else { 0 }";
    if let ExprKind::IfLet { pattern, expr, then_block, else_block } = p_expr(input) {
        assert!(matches!(pattern, tl_lang::compiler::ast::Pattern::EnumPattern { .. }));
        assert_eq!(expr.inner, ExprKind::Variable("opt".to_string()));
        assert_eq!(then_block.len(), 1); // Expr stmt
        assert!(else_block.is_some());
    } else {
        panic!("Expected IfLet");
    }
}

#[test]
fn test_generics() {
    // Generic Function
    let input = "fn map<T, U>(x: T) -> U { }";
    let tokens = tokenize(input).into_iter().map(|r| r.unwrap()).collect::<Vec<_>>();
    let module = parse(&tokens).expect("Failed to parse generic fn");
    let func = &module.functions[0];
    assert_eq!(func.name, "map");
    assert_eq!(func.generics, vec!["T", "U"]);

    // Generic Struct
    let input = "struct Container<T> { value: T }";
    let tokens = tokenize(input).into_iter().map(|r| r.unwrap()).collect::<Vec<_>>();
    let module = parse(&tokens).expect("Failed to parse generic struct");
    let strct = &module.structs[0];
    assert_eq!(strct.name, "Container");
    assert_eq!(strct.generics, vec!["T"]);
}

#[test]
fn test_enums() {
    let input = r#"
    enum Option<T> {
        Some(T),
        None,
    }
    "#;
    let tokens = tokenize(input).into_iter().map(|r| r.unwrap()).collect::<Vec<_>>();
    let module = parse(&tokens).expect("Failed to parse enum");
    let enm = &module.enums[0];
    assert_eq!(enm.name, "Option");
    assert_eq!(enm.generics, vec!["T"]);
    assert_eq!(enm.variants.len(), 2);
    // Variant Payload checking if possible via public AST
}

#[test]
fn test_logic_query() {
    match p_expr("?path(a, $x)") {
        ExprKind::UnOp(UnOp::Query, inner) => {
            if let ExprKind::FnCall(name, args) = inner.inner {
                assert_eq!(name, "path");
                assert_eq!(args.len(), 2);
                assert_eq!(args[1].inner, ExprKind::LogicVar("x".to_string()));
            } else {
                panic!("Expected Predicate Call");
            }
        }
        _ => panic!("Expected Query"),
    }
}
