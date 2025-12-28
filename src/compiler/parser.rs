// src/compiler/parser.rs
use crate::compiler::ast::*;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while},
    character::complete::{alpha1, char, digit1, multispace0},
    combinator::{map, map_res, opt, recognize, value},
    multi::{separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, tuple},
    IResult,
};
// removed unused std::str::FromStr

// --- Whitespace & Comments ---
fn ws<'a, F: 'a, O, E: nom::error::ParseError<&'a str>>(
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(multispace0, inner, multispace0)
}

// --- Identifiers ---
fn identifier(input: &str) -> IResult<&str, String> {
    map(
        recognize(pair(
            alt((alpha1, tag("_"))),
            take_while(|c: char| c.is_alphanumeric() || c == '_'),
        )),
        |s: &str| s.to_string(),
    )(input)
}

// --- Types ---
fn parse_primitive_type(input: &str) -> IResult<&str, Type> {
    alt((
        value(Type::F32, tag("f32")),
        value(Type::F64, tag("f64")),
        value(Type::I32, tag("i32")),
        value(Type::I64, tag("i64")),
        value(Type::Bool, tag("bool")),
        value(Type::Usize, tag("usize")),
        value(Type::Void, tag("void")),
    ))(input)
}

fn parse_tensor_type(input: &str) -> IResult<&str, Type> {
    // Tensor<f32, 2>
    map(
        tuple((
            tag("Tensor"),
            ws(char('<')),
            parse_type,
            ws(char(',')),
            digit1,
            ws(char('>')),
        )),
        |(_, _, inner, _, rank_str, _)| {
            let rank = rank_str.parse::<usize>().unwrap_or(0);
            Type::Tensor(Box::new(inner), rank)
        },
    )(input)
}

fn parse_user_type(input: &str) -> IResult<&str, Type> {
    map(identifier, Type::UserDefined)(input)
}

fn parse_type(input: &str) -> IResult<&str, Type> {
    alt((parse_tensor_type, parse_primitive_type, parse_user_type))(input)
}

// --- Expressions ---

fn parse_literal_int(input: &str) -> IResult<&str, Expr> {
    map_res(digit1, |s: &str| s.parse::<i64>().map(Expr::Int))(input)
}

fn parse_literal_float(input: &str) -> IResult<&str, Expr> {
    map_res(recognize(tuple((digit1, char('.'), digit1))), |s: &str| {
        s.parse::<f64>().map(Expr::Float)
    })(input)
}

fn parse_variable(input: &str) -> IResult<&str, Expr> {
    map(identifier, Expr::Variable)(input)
}

// --- Blocks ---
fn parse_block(input: &str) -> IResult<&str, Vec<Stmt>> {
    delimited(ws(char('{')), nom::multi::many0(parse_stmt), ws(char('}')))(input)
}

// --- Control Flow ---
fn parse_if_expr(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("if")(input)?;
    let (input, cond) = parse_expr(input)?;
    let (input, then_block) = parse_block(input)?;

    let (input, else_block) = opt(preceded(ws(tag("else")), parse_block))(input)?;

    Ok((input, Expr::IfExpr(Box::new(cond), then_block, else_block)))
}

fn parse_tensor_literal(input: &str) -> IResult<&str, Expr> {
    map(
        delimited(
            ws(char('[')),
            separated_list0(ws(char(',')), parse_expr),
            ws(char(']')),
        ),
        Expr::TensorLiteral,
    )(input)
}

// Atom: Literal | Call | IndexAccess | Variable | (Expr)
fn parse_atom(input: &str) -> IResult<&str, Expr> {
    ws(alt((
        parse_literal_float,
        parse_literal_int,
        parse_if_expr,
        parse_tensor_literal, // [1, 2]
        // Call: name(args...)
        map(
            tuple((
                identifier,
                delimited(
                    ws(char('(')),
                    separated_list0(ws(char(',')), parse_expr),
                    ws(char(')')),
                ),
            )),
            |(name, args)| Expr::FnCall(name, args),
        ),
        // IndexAccess: name[i, j]
        map(
            tuple((
                identifier,
                delimited(
                    ws(char('[')),
                    separated_list1(ws(char(',')), identifier),
                    ws(char(']')),
                ),
            )),
            |(name, idx)| Expr::IndexAccess(Box::new(Expr::Variable(name)), idx),
        ),
        parse_variable,
        delimited(ws(char('(')), parse_expr, ws(char(')'))),
        // Block expression? Not in spec explicitly but convenient.
    )))(input)
}

// Term: Atom * Atom or Atom / Atom
fn parse_term(input: &str) -> IResult<&str, Expr> {
    let (input, init) = parse_atom(input)?;

    nom::multi::fold_many0(
        pair(ws(alt((char('*'), char('/')))), parse_atom),
        move || init.clone(),
        |acc, (op, val)| {
            let bin_op = match op {
                '*' => BinOp::Mul,
                '/' => BinOp::Div,
                _ => unreachable!(),
            };
            Expr::BinOp(Box::new(acc), bin_op, Box::new(val))
        },
    )(input)
}

// Expr: Term + Term or Term - Term
fn parse_expr(input: &str) -> IResult<&str, Expr> {
    let (input, init) = parse_term(input)?;

    nom::multi::fold_many0(
        pair(ws(alt((char('+'), char('-')))), parse_term),
        move || init.clone(),
        |acc, (op, val)| {
            let bin_op = match op {
                '+' => BinOp::Add,
                '-' => BinOp::Sub,
                _ => unreachable!(),
            };
            Expr::BinOp(Box::new(acc), bin_op, Box::new(val))
        },
    )(input)
}

// --- Statements ---

// let x[i, j] = ...
fn parse_let_stmt(input: &str) -> IResult<&str, Stmt> {
    let (input, _) = tag("let")(input)?;
    let (input, name) = ws(identifier)(input)?;

    // Optional indices [i, j]
    let (input, indices) = opt(delimited(
        ws(char('[')),
        separated_list0(ws(char(',')), identifier),
        ws(char(']')),
    ))(input)?;

    // Optional type : Type
    let (input, type_annotation) = opt(preceded(ws(char(':')), parse_type))(input)?;

    let (input, _) = ws(char('='))(input)?;
    let (input, value) = parse_expr(input)?;
    let (input, _) = ws(char(';'))(input)?;

    Ok((
        input,
        Stmt::Let {
            name,
            indices,
            type_annotation,
            value,
        },
    ))
}

// x[i] += ... or x = ...
fn parse_assign_stmt(input: &str) -> IResult<&str, Stmt> {
    let (input, name) = ws(identifier)(input)?;

    let (input, indices) = opt(delimited(
        ws(char('[')),
        separated_list0(ws(char(',')), identifier),
        ws(char(']')),
    ))(input)?;

    let (input, op_str) = ws(alt((tag("="), tag("+="), tag("max="), tag("avg="))))(input)?;

    let op = match op_str {
        "=" => AssignOp::Assign,
        "+=" => AssignOp::AddAssign,
        "max=" => AssignOp::MaxAssign,
        "avg=" => AssignOp::AvgAssign,
        _ => unreachable!(),
    };

    let (input, value) = parse_expr(input)?;
    let (input, _) = ws(char(';'))(input)?;

    Ok((
        input,
        Stmt::Assign {
            name,
            indices,
            op,
            value,
        },
    ))
}

// if (stmt version usually wraps expression if return ignored, or purely statement)
// For now, treat top-level if as Stmt::If.
fn parse_if_stmt(input: &str) -> IResult<&str, Stmt> {
    let (input, _) = tag("if")(input)?;
    let (input, cond) = parse_expr(input)?;
    let (input, then_block) = parse_block(input)?;

    let (input, else_block) = opt(preceded(ws(tag("else")), parse_block))(input)?;

    Ok((
        input,
        Stmt::If {
            cond,
            then_block,
            else_block,
        },
    ))
}

fn parse_for_stmt(input: &str) -> IResult<&str, Stmt> {
    // for var in expr { ... }
    let (input, _) = tag("for")(input)?;
    let (input, loop_var) = ws(identifier)(input)?;
    let (input, _) = ws(tag("in"))(input)?;
    let (input, iterator) = parse_expr(input)?; // currently generic expr, could be range
    let (input, body) = parse_block(input)?;

    Ok((
        input,
        Stmt::For {
            loop_var,
            iterator,
            body,
        },
    ))
}

fn parse_return_stmt(input: &str) -> IResult<&str, Stmt> {
    let (input, _) = tag("return")(input)?;
    let (input, val) = parse_expr(input)?;
    let (input, _) = ws(char(';'))(input)?;
    Ok((input, Stmt::Return(val)))
}

fn parse_expr_stmt(input: &str) -> IResult<&str, Stmt> {
    // expr;
    let (input, expr) = parse_expr(input)?;
    // If it's a block or if-expr, semicolon optional? Rust allows.
    // For simplicity, strict semicolon for now except if/for?
    // Let's enforce semicolon for expr stmt.
    let (input, _) = ws(char(';'))(input)?;
    Ok((input, Stmt::Expr(expr)))
}

fn parse_stmt(input: &str) -> IResult<&str, Stmt> {
    ws(alt((
        parse_let_stmt,
        parse_assign_stmt,
        parse_return_stmt,
        parse_if_stmt,
        parse_for_stmt,
        parse_expr_stmt,
    )))(input)
}

// --- Top Level ---
fn parse_fn(input: &str) -> IResult<&str, FunctionDef> {
    // fn name(...) -> Type { ... }
    let (input, _) = tag("fn")(input)?;
    let (input, name) = ws(identifier)(input)?;
    let (input, _args) = delimited(ws(char('(')), multispace0, ws(char(')')))(input)?;
    let (input, _) = ws(tag("->"))(input)?;
    let (input, ret_type) = ws(parse_type)(input)?;
    let (input, _) = ws(char('{'))(input)?;

    // Body
    let (input, body) = nom::multi::many0(parse_stmt)(input)?;

    let (input, _) = ws(char('}'))(input)?;

    Ok((
        input,
        FunctionDef {
            name,
            args: vec![], // TODO
            return_type: ret_type,
            body,
            generics: vec![],
        },
    ))
}

// --- Structs ---
fn parse_struct(input: &str) -> IResult<&str, StructDef> {
    // struct Name<T> { field: Type, ... }
    let (input, _) = tag("struct")(input)?;
    let (input, name) = ws(identifier)(input)?;

    // Generics <T>
    let (input, generics) = opt(delimited(
        ws(char('<')),
        separated_list1(ws(char(',')), identifier),
        ws(char('>')),
    ))(input)?;

    let (input, _) = ws(char('{'))(input)?;

    // Fields: name: Type,
    let (input, fields) = separated_list0(
        ws(char(',')),
        pair(ws(identifier), preceded(ws(char(':')), parse_type)),
    )(input)?;

    let (input, _) = opt(ws(char(',')))(input)?; // trailing comma
    let (input, _) = ws(char('}'))(input)?;

    Ok((
        input,
        StructDef {
            name,
            fields,
            generics: generics.unwrap_or_default(),
        },
    ))
}

// --- Impls ---
fn parse_impl(input: &str) -> IResult<&str, ImplBlock> {
    // impl<T> Name<T> { fn ... }
    let (input, _) = tag("impl")(input)?;

    // Generics <T> (optional on impl)
    let (input, generics) = opt(delimited(
        ws(char('<')),
        separated_list1(ws(char(',')), identifier),
        ws(char('>')),
    ))(input)?;

    let (input, target_type) = ws(identifier)(input)?;

    // Type args for target? e.g. Linear<T>
    // For now skip parsing <T> on target type name in AST or parse it?
    // The AST has target_type: String. Let's just parse the name.
    // If there is <T>, consume it but ignore for now or store in name?
    // Let's consume it.
    let (input, _) = opt(delimited(
        ws(char('<')),
        take_while(|c: char| c != '>'),
        ws(char('>')),
    ))(input)?;

    let (input, _) = ws(char('{'))(input)?;

    let (input, methods) = nom::multi::many0(ws(parse_fn))(input)?;

    let (input, _) = ws(char('}'))(input)?;

    Ok((
        input,
        ImplBlock {
            target_type,
            generics: generics.unwrap_or_default(),
            methods,
        },
    ))
}

// --- Top Level ---
// Re-define parse_module logic
pub fn parse(input: &str) -> anyhow::Result<Module> {
    let mut structs = vec![];
    let mut impls = vec![];
    let mut functions = vec![];
    let mut remaining = input;

    while !remaining.trim().is_empty() {
        // Try struct, then impl, then fn
        if let Ok((next, s)) = ws(parse_struct)(remaining) {
            structs.push(s);
            remaining = next;
        } else if let Ok((next, i)) = ws(parse_impl)(remaining) {
            impls.push(i);
            remaining = next;
        } else if let Ok((next, f)) = ws(parse_fn)(remaining) {
            functions.push(f);
            remaining = next;
        } else {
            return Err(anyhow::anyhow!("Parse error at: {:.30}...", remaining));
        }
    }

    Ok(Module {
        structs,
        impls,
        functions,
    })
}
