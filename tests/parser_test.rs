use tl::compiler::ast::*;
use tl::compiler::parser::*;

fn p_expr(input: &str) -> Expr {
    let (rest, expr) = parse_expr(input).expect("Parse error");
    assert!(rest.trim().is_empty(), "Trailing text: {}", rest);
    expr
}

fn p_stmt(input: &str) -> Stmt {
    let (rest, stmt) = parse_stmt(input).expect("Parse error");
    assert!(rest.trim().is_empty(), "Trailing text: {}", rest);
    stmt
}

#[test]
fn test_types() {
    let (_, t) = parse_type("f32").unwrap();
    assert_eq!(t, Type::F32);

    let (_, t) = parse_type("Tensor<f32, 2>").unwrap();
    if let Type::Tensor(inner, rank) = t {
        assert_eq!(*inner, Type::F32);
        assert_eq!(rank, 2);
    } else {
        panic!("Expected Tensor type");
    }

    let (_, t) = parse_type("(f32, i32)").unwrap();
    if let Type::Tuple(ts) = t {
        assert_eq!(ts.len(), 2);
    } else {
        panic!("Expected Tuple type");
    }
}

#[test]
fn test_literals() {
    assert_eq!(p_expr("123"), Expr::Int(123));
    assert_eq!(p_expr("123.456"), Expr::Float(123.456));
    assert_eq!(p_expr("true"), Expr::Bool(true));
    assert_eq!(p_expr("false"), Expr::Bool(false));
    // String literal includes quotes in parser logic but ast stores content
    // parse_literal_string: "abc" -> StringLiteral("abc")
    assert_eq!(p_expr("\"abc\""), Expr::StringLiteral("abc".to_string()));
}

#[test]
fn test_variables() {
    assert_eq!(p_expr("x"), Expr::Variable("x".to_string()));
    assert_eq!(p_expr("my_var_1"), Expr::Variable("my_var_1".to_string()));
}

#[test]
fn test_binary_ops() {
    // Simple
    if let Expr::BinOp(l, op, r) = p_expr("1 + 2") {
        assert_eq!(*l, Expr::Int(1));
        assert_eq!(op, BinOp::Add);
        assert_eq!(*r, Expr::Int(2));
    } else {
        panic!("Not a BinOp");
    }

    // Precedence: 1 + 2 * 3 => 1 + (2 * 3)
    if let Expr::BinOp(l, op, r) = p_expr("1 + 2 * 3") {
        assert_eq!(*l, Expr::Int(1));
        assert_eq!(op, BinOp::Add);
        if let Expr::BinOp(l2, op2, r2) = *r {
            assert_eq!(*l2, Expr::Int(2));
            assert_eq!(op2, BinOp::Mul);
            assert_eq!(*r2, Expr::Int(3));
        } else {
            panic!("RHS not mul");
        }
    } else {
        panic!("Not add");
    }

    // Comparison
    match p_expr("x <= y") {
        Expr::BinOp(_, BinOp::Le, _) => {}
        _ => panic!("Expected Le"),
    }

    // Logical
    match p_expr("a && b || c") {
        // (a && b) || c
        Expr::BinOp(l, BinOp::Or, _) => match *l {
            Expr::BinOp(_, BinOp::And, _) => {}
            _ => panic!("LHS expected And"),
        },
        _ => panic!("Expected Or"),
    }
}

#[test]
fn test_unary_ops() {
    if let Expr::UnOp(op, e) = p_expr("-x") {
        assert_eq!(op, UnOp::Neg);
        assert_eq!(*e, Expr::Variable("x".to_string()));
    } else {
        panic!("Expected Neg");
    }

    if let Expr::UnOp(op, _) = p_expr("!true") {
        assert_eq!(op, UnOp::Not);
    } else {
        panic!("Expected Not");
    }
}

#[test]
fn test_postfix() {
    // Call
    if let Expr::FnCall(name, args) = p_expr("foo(1, 2)") {
        assert_eq!(name, "foo");
        assert_eq!(args.len(), 2);
    } else {
        panic!("Expected FnCall");
    }

    // Method
    if let Expr::MethodCall(obj, name, args) = p_expr("x.bar(3)") {
        assert_eq!(*obj, Expr::Variable("x".to_string()));
        assert_eq!(name, "bar");
        assert_eq!(args.len(), 1);
    } else {
        panic!("Expected MethodCall");
    }

    // Index
    if let Expr::IndexAccess(obj, indices) = p_expr("A[i, j]") {
        assert_eq!(*obj, Expr::Variable("A".to_string()));
        assert_eq!(indices.len(), 2);
    } else {
        panic!("Expected IndexAccess");
    }

    // Field
    if let Expr::FieldAccess(obj, field) = p_expr("s.field") {
        assert_eq!(*obj, Expr::Variable("s".to_string()));
        assert_eq!(field, "field");
    } else {
        panic!("Expected FieldAccess");
    }

    // Tuple Access
    if let Expr::TupleAccess(obj, idx) = p_expr("t.0") {
        assert_eq!(*obj, Expr::Variable("t".to_string()));
        assert_eq!(idx, 0);
    } else {
        panic!("Expected TupleAccess");
    }

    // Static Method
    if let Expr::StaticMethodCall(type_name, method, _) = p_expr("MyType::new()") {
        assert_eq!(type_name, "MyType");
        assert_eq!(method, "new");
    } else {
        panic!("Expected StaticMethodCall");
    }
}

#[test]
fn test_cast() {
    if let Expr::As(expr, ty) = p_expr("x as f32") {
        assert_eq!(*expr, Expr::Variable("x".to_string()));
        assert_eq!(ty, Type::F32);
    } else {
        panic!("Expected Cast");
    }
}

#[test]
fn test_range() {
    if let Expr::Range(start, end) = p_expr("0..10") {
        assert_eq!(*start, Expr::Int(0));
        assert_eq!(*end, Expr::Int(10));
    } else {
        panic!("Expected Range");
    }
}

#[test]
fn test_constructors() {
    // Struct init
    if let Expr::StructInit(name, fields) = p_expr("Point { x: 1, y: 2 }") {
        assert_eq!(name, "Point");
        assert_eq!(fields.len(), 2);
    } else {
        panic!("Expected StructInit");
    }

    // Tensor literal
    if let Expr::TensorConstLiteral(elems) = p_expr("[1, 2, 3]") {
        assert_eq!(elems.len(), 3);
    } else {
        panic!("Expected TensorConstLiteral");
    }

    // Tensor comprehension
    if let Expr::TensorComprehension { indices, body } = p_expr("[i, j | i + j]") {
        assert_eq!(indices, vec!["i", "j"]);
        match *body {
            Expr::BinOp(_, _, _) => {}
            _ => panic!("Expected BinOp in body"),
        }
    } else {
        panic!("Expected TensorComprehension");
    }

    // Aggregation
    if let Expr::Aggregation { op, .. } = p_expr("sum(x for i in 0..10)") {
        assert_eq!(op, AggregateOp::Sum);
    } else {
        panic!("Expected Aggregation");
    }
}

#[test]
fn test_statements() {
    match p_stmt("let x = 1;") {
        Stmt::Let { name, value, .. } => {
            assert_eq!(name, "x");
            assert_eq!(value, Expr::Int(1));
        }
        _ => panic!("Expected Let"),
    }

    match p_stmt("x += 5;") {
        Stmt::Assign { name, op, .. } => {
            assert_eq!(name, "x");
            assert_eq!(op, AssignOp::AddAssign);
        }
        _ => panic!("Expected Assign"),
    }

    match p_stmt("return 42;") {
        Stmt::Return(Some(Expr::Int(42))) => {}
        _ => panic!("Expected Return"),
    }

    match p_stmt("if true { } else { }") {
        Stmt::If { .. } => {}
        _ => panic!("Expected If"),
    }

    match p_stmt("for i in 0..10 { }") {
        Stmt::For { .. } => {}
        _ => panic!("Expected For"),
    }

    match p_stmt("while x < 10 { }") {
        Stmt::While { .. } => {}
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

    let module = parse(input).expect("Failed to parse module");
    assert_eq!(module.imports.len(), 1);
    assert_eq!(module.imports[0], "foo");

    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.structs[0].name, "Point");

    assert_eq!(module.impls.len(), 1);
    assert_eq!(module.impls[0].target_type, "Point");

    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "main");

    // use stmt + tensor decl
    assert_eq!(module.tensor_decls.len(), 2);
}
