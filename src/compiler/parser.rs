// src/compiler/parser.rs
use crate::compiler::ast::*;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{alpha1, char, digit1, space0, space1},
    combinator::{map, map_res, not, opt, recognize, value, verify},
    multi::{separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, separated_pair, tuple},
    IResult,
};
// removed unused std::str::FromStr

// --- Whitespace & Comments ---
// --- Whitespace & Comments ---
fn ws<'a, F: 'a, O, E: nom::error::ParseError<&'a str>>(
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(sp, inner, sp)
}

fn sp<'a, E: nom::error::ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, &'a str, E> {
    let chars = " \t\r\n";
    // Consumes whitespace or comments recursively
    recognize(nom::multi::many0(alt((
        take_while1(move |c| chars.contains(c)),
        preceded(tag("//"), take_while(|c| c != '\n')),
    ))))(input)
}

// --- Identifiers ---

fn simple_identifier(input: &str) -> IResult<&str, String> {
    let (input, s) = recognize(pair(
        alt((alpha1, tag("_"))),
        take_while(|c: char| c.is_alphanumeric() || c == '_'),
    ))(input)?;
    Ok((input, s.to_string()))
}

fn identifier(input: &str) -> IResult<&str, String> {
    // Allow usage of :: for namespaces (e.g. File::open)
    verify(
        map(
            recognize(separated_list1(tag("::"), simple_identifier)),
            |s: &str| s.to_string(),
        ),
        |s: &String| {
            let keywords = vec![
                "fn", "struct", "impl", "let", "if", "else", "return", "for", "in", "while",
                "true", "false", "f32", "f64", "i32", "i64", "bool", "usize", "void", "Tensor",
                "mod", "use",
            ];
            !keywords.contains(&s.as_str())
        },
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
        value(Type::Entity, tag("entity")),
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
    // Check for Enum vs Struct in identifier context? No, just identifier.
    // Type::Struct / Type::Enum / Type::UserDefined are currently ambiguous in parser until resolution.
    // Let's use UserDefined for now for any identifier in type position.
    // But later semantics will resolve it.
    // However, if we want explicit Type::Enum for syntax highlighting or specialized parsing?
    // For now, Type::UserDefined covers both.
    map(identifier, Type::UserDefined)(input)
}

fn parse_tuple_type(input: &str) -> IResult<&str, Type> {
    // (Type, Type, ...)
    let (input, types) = delimited(
        ws(char('(')),
        separated_list0(ws(char(',')), parse_type),
        ws(char(')')),
    )(input)?;

    // If only one type and no trailing comma (ambiguous with grouping), treat as just that type?
    // But for types, (T) is usually same as T.
    // However, for tuple types, we usually want explicit Tuple variant if it's meant to be a tuple.
    // Rust: (i32) is i32. (i32,) is tuple.
    // For simplicity, let's say if we parsed multiple types, it's a tuple.
    // If 0, it's void? Or empty tuple.
    // If 1, return the type directly (grouping behavior).
    // TODO: support (T,) syntax for 1-element tuple type if needed.

    if types.is_empty() {
        Ok((input, Type::Void)) // Unit type
    } else if types.len() == 1 {
        Ok((input, types[0].clone()))
    } else {
        Ok((input, Type::Tuple(types)))
    }
}

pub fn parse_type(input: &str) -> IResult<&str, Type> {
    alt((
        parse_tensor_type,
        parse_primitive_type,
        parse_tuple_type,
        parse_user_type,
    ))(input)
}

// --- Expressions ---

fn parse_literal_int(input: &str) -> IResult<&str, Expr> {
    map_res(digit1, |s: &str| s.parse::<i64>().map(Expr::Int))(input)
}

// Helper for exponent: e/E followed by optional sign and digits
fn parse_exponent(input: &str) -> IResult<&str, &str> {
    recognize(tuple((
        alt((char('e'), char('E'))),
        opt(alt((char('+'), char('-')))),
        digit1,
    )))(input)
}

fn parse_literal_float(input: &str) -> IResult<&str, Expr> {
    // Two patterns:
    // 1. Digits . Digits [Exponent]
    // 2. Digits Exponent
    let pattern1 = recognize(tuple((digit1, char('.'), digit1, opt(parse_exponent))));
    let pattern2 = recognize(tuple((digit1, parse_exponent)));

    map_res(alt((pattern1, pattern2)), |s: &str| {
        s.parse::<f64>().map(Expr::Float)
    })(input)
}

fn parse_literal_bool(input: &str) -> IResult<&str, Expr> {
    alt((
        value(Expr::Bool(true), tag("true")),
        value(Expr::Bool(false), tag("false")),
    ))(input)
}

fn parse_literal_string(input: &str) -> IResult<&str, Expr> {
    let (input, s) = delimited(char('"'), take_while(|c| c != '"'), char('"'))(input)?;
    Ok((input, Expr::StringLiteral(s.to_string())))
}

fn parse_variable(input: &str) -> IResult<&str, Expr> {
    map(identifier, Expr::Variable)(input)
}

// --- Blocks ---
fn parse_block(input: &str) -> IResult<&str, Vec<Stmt>> {
    let (input, _) = ws(char('{'))(input)?;
    let (input, mut stmts) = nom::multi::many0(parse_stmt)(input)?;

    // Optional trailing expression without semicolon
    // But parse_stmt includes parse_expr_stmt which forces semicolon.
    // So here we look for raw parse_expr.
    let (input, trailing) = opt(parse_expr)(input)?;

    let (input, _) = ws(char('}'))(input)?;

    if let Some(expr) = trailing {
        stmts.push(Stmt::Expr(expr));
    }

    Ok((input, stmts))
}

fn parse_block_expr(input: &str) -> IResult<&str, Expr> {
    map(parse_block, Expr::Block)(input)
}

fn parse_comprehension_clause(input: &str) -> IResult<&str, ComprehensionClause> {
    // Ensure we don't start with { (which indicates body) or ] (end)
    let (input, _) = not(ws(char('{')))(input)?;
    let (input, _) = not(ws(char(']')))(input)?;

    // i <- expr
    let generator = map(
        separated_pair(ws(identifier), ws(tag("<-")), parse_expr),
        |(name, range)| ComprehensionClause::Generator { name, range },
    );
    // expr (condition)
    let condition = map(parse_expr, ComprehensionClause::Condition);

    alt((generator, condition))(input)
}

fn parse_tensor_comprehension(input: &str) -> IResult<&str, Expr> {
    // [ indices | clauses... { body } ]
    let (input, _) = ws(char('['))(input)?;

    // 1. Indices (Output dimensions)
    let (input, indices) = separated_list1(ws(char(',')), ws(identifier))(input)?;

    let (input, _) = ws(char('|'))(input)?;

    // 2. Clauses (Generators and Conditions)
    // Parse until '{' or ']'
    // We can't use separated_list1 directly if we want to stop at '{' without consuming it optionally?
    // Actually, separated_list1 will return error if it can't find separator or item?
    // We want "comma separated items until we hit { or ]"
    // Let's use many0 with custom separator check logic or just separated_list0 but we need to ensure we don't consume the body start.
    // parse_comprehension_clause does NOT start with '{'.
    // If we use separated_list0(ws(char(',')), parse_comprehension_clause), it should work as long as `{` is not a start of clause.
    // BUT parse_expr might start with `{` (Block). So Condition might consume body block?
    // Wait, body block is `{ expr }`. Condition is `expr`. `expr` includes `Block`.
    // So `parse_expr` WILL consume `{ ... }`. This is ambiguous if we just list exprs.
    // However, if we see `i <- ...` that is distinct.
    // If we see `expr`, it could be a condition or the body block.
    // But the body block is MANDATORY `{ ... }` syntax for body.
    // Is a block `{ ... }` calculable as a boolean condition? Yes in Rust/TL (returns last expr).
    // So `[ i | { true } ]` -> Is `{ true }` a condition or body?
    // We defined syntax: `... { body } ]`
    // So the last element IS the body if it is a block?
    // Or we say: "Clauses come first. Body comes last."
    // If we encounter a Block `{ ... }` at the top level of the clause list, we might assume it is the body?
    // But `if true { ... }` is also an Expr which starts with `if`.
    // A raw Block `{ ... }` as a condition seems rare but possible.
    // To resolve ambiguity: The Body MUST be the final element and MUST be a Block.
    // AND we can say that clauses are correctly separated by commas.
    // If we are at the end of clauses, we might see a `{`.
    // We can peek. If next char is `{`, we stop parsing clauses?
    // BUT `parse_expr` for a condition also might start with `{`.
    // Let's rely on the rule: Body is `{ expr }`.
    // If we simply peek for `{`, we treat it as Body start.
    // This bans conditions starting with `{` (block expressions) at the top level, unless wrapped in `(...)`.
    // This is a reasonable limitation for clear syntax.

    // Parse clauses: comma separated list of generators or conditions
    let (input, clauses) = separated_list0(ws(char(',')), parse_comprehension_clause)(input)?;

    // Optional Body: { expr }
    // We use peek concept: if `{`, parse block, else None.
    let (input, body_expr) = opt(parse_block_expr)(input)?;

    // Check for `]`
    let (input, _) = ws(char(']'))(input)?;

    // Wrap body if present
    let body = body_expr.map(Box::new);

    Ok((
        input,
        Expr::TensorComprehension {
            indices: indices.into_iter().map(|s| s.to_string()).collect(),
            clauses,
            body,
        },
    ))
}

// --- Control Flow ---
fn parse_if_expr(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("if")(input)?;
    let (input, cond) = parse_expr(input)?;
    let (input, then_block) = parse_block(input)?;

    let (input, else_block) = opt(preceded(ws(tag("else")), parse_block))(input)?;

    Ok((input, Expr::IfExpr(Box::new(cond), then_block, else_block)))
}

fn parse_if_let_expr(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("if")(input)?;
    let (input, _) = ws(tag("let"))(input)?;
    let (input, pattern) = ws(parse_pattern)(input)?;
    let (input, _) = ws(char('='))(input)?;
    let (input, expr) = parse_expr(input)?;
    let (input, then_block) = parse_block(input)?;
    let (input, else_block) = opt(preceded(ws(tag("else")), parse_block))(input)?;

    Ok((
        input,
        Expr::IfLet {
            pattern,
            expr: Box::new(expr),
            then_block,
            else_block,
        },
    ))
}

fn parse_tensor_literal(input: &str) -> IResult<&str, Expr> {
    let (input, elements) = delimited(
        ws(char('[')),
        separated_list0(ws(char(',')), parse_expr),
        ws(char(']')),
    )(input)?;

    // Check if all elements are constants (recursively)
    fn is_const(expr: &Expr) -> bool {
        match expr {
            Expr::Float(_) | Expr::Int(_) | Expr::Bool(_) => true,
            Expr::TensorLiteral(elems) | Expr::TensorConstLiteral(elems) => {
                elems.iter().all(is_const)
            }
            _ => false,
        }
    }

    let all_const = elements.iter().all(is_const);

    if all_const {
        Ok((input, Expr::TensorConstLiteral(elements)))
    } else {
        Ok((input, Expr::TensorLiteral(elements)))
    }
}

// Aggregation: sum(expr for var in range) or sum(expr for var in range where cond)
fn parse_aggregation(input: &str) -> IResult<&str, Expr> {
    // Parse: sum|max|min|avg|count ( expr for var in range [where cond] )
    let (input, op_str) = ws(alt((
        tag("sum"),
        tag("max"),
        tag("min"),
        tag("avg"),
        tag("count"),
    )))(input)?;

    let op = match op_str {
        "sum" => AggregateOp::Sum,
        "max" => AggregateOp::Max,
        "min" => AggregateOp::Min,
        "avg" => AggregateOp::Avg,
        "count" => AggregateOp::Count,
        _ => unreachable!(),
    };

    let (input, _) = ws(char('('))(input)?;
    let (input, expr) = parse_expr(input)?;
    let (input, _) = ws(tag("for"))(input)?;
    let (input, var) = ws(identifier)(input)?;
    let (input, _) = ws(tag("in"))(input)?;
    let (input, range) = parse_expr(input)?;
    let (input, condition) = opt(preceded(ws(tag("where")), parse_expr))(input)?;
    let (input, _) = ws(char(')'))(input)?;

    Ok((
        input,
        Expr::Aggregation {
            op,
            expr: Box::new(expr),
            var,
            range: Box::new(range),
            condition: condition.map(Box::new),
        },
    ))
}

fn parse_struct_init(input: &str) -> IResult<&str, Expr> {
    // Identifier { field: expr, ... }
    let (input, name) = ws(identifier)(input)?;
    let (input, _) = ws(char('{'))(input)?;
    let (input, fields) = separated_list0(
        ws(char(',')),
        pair(ws(identifier), preceded(ws(char(':')), parse_expr)),
    )(input)?;
    let (input, _) = opt(ws(char(',')))(input)?;
    let (input, _) = ws(char('}'))(input)?;

    Ok((input, Expr::StructInit(name, fields)))
}

#[allow(dead_code)]
fn parse_enum_init(input: &str) -> IResult<&str, Expr> {
    // Enum::Variant { field: value ... }
    // Identifier "Enum::Variant" or just "Variant"?
    // Rust requires fully qualified or imported.
    // Our identifier parser handles "A::B".
    // So if we see "A::B { ... }", is it StructInit or EnumInit?
    // StructInit currently takes identifier.
    // If identifier has "::", we could treat it as EnumInit OR StructInit (mod::Struct).
    // Let's assume generic StructInit handles "A::B" as name.
    // BUT we need to produce Expr::EnumInit if it's an Enum.
    // Since Parser doesn't know types, we must produce a "GenericInit" or rely on Semantics to re-classify?
    // OR we can make a heuristic: if it looks like "X::Y { ... }", treat as potentially enum.
    // Actually, `Expr::StructInit` stores a String name. Semantics can resolve "X::Y" to Enum or Struct.
    // So maybe we don't strictly need distinct parse_enum_init IF StructInit covers the syntax.
    // HOWEVER, we added Expr::EnumInit.
    // Let's modify parse_struct_init to return EnumInit if it detects "::".
    // Wait, Structs can differ by module too: `mod::Struct { ... }`.
    // So syntax is identical.
    // Strategy: Parse as StructInit in Parser. In Semantics, if name resolves to Enum, convert to EnumInit.
    // But wait, user requested "Implement Parser support".
    // I can leave it as StructInit and fix in Semantics, OR I can try to disambiguate.
    // Ambiguity is impossible without symbol table.
    // So: I will stick to ParseStructInit, but I'll rename it/comment it or handle conversion in Semantics.
    // Wait, Expr::EnumInit fields: enum_name, variant_name. StructInit only has name.
    // If I stick to StructInit, I need to split name in Semantics.
    // OK, let's keep parse_struct_init as the shared parser.
    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Fail,
    )))
}

fn parse_pattern(input: &str) -> IResult<&str, Pattern> {
    // Wildcard
    let (input, wc) = opt(tag("_"))(input)?;
    if wc.is_some() {
        return Ok((input, Pattern::Wildcard));
    }

    // Enum Pattern: Enum::Variant { x, y } (shorthand) or { f: x }
    // Or just Variant { ... }?
    // Let's support Identifier { ... }
    let (input, name) = ws(identifier)(input)?;
    let (input, block) = opt(delimited(
        ws(char('{')),
        separated_list0(
            ws(char(',')),
            pair(ws(identifier), opt(preceded(ws(char(':')), ws(identifier)))),
        ),
        ws(char('}')),
    ))(input)?;

    if let Some(fields) = block {
        // Parse bindings: x: y -> field x binds to var y.
        // x -> field x binds to var x.
        let bindings = fields
            .into_iter()
            .map(|(field, var_opt)| (field.clone(), var_opt.unwrap_or(field)))
            .collect();

        // Name might be "Enum::Variant" or just "Variant".
        // We need to split.
        if let Some((enum_name, variant_name)) = name.split_once("::") {
            Ok((
                input,
                Pattern::EnumPattern {
                    enum_name: enum_name.to_string(),
                    variant_name: variant_name.to_string(),
                    bindings,
                },
            ))
        } else {
            Ok((
                input,
                Pattern::EnumPattern {
                    enum_name: "".to_string(), // Inferred? Or require full path?
                    variant_name: name,
                    bindings,
                },
            ))
        }
    } else {
        // Unit variant pattern: Enum::Variant
        if let Some((enum_name, variant_name)) = name.split_once("::") {
            Ok((
                input,
                Pattern::EnumPattern {
                    enum_name: enum_name.to_string(),
                    variant_name: variant_name.to_string(),
                    bindings: vec![],
                },
            ))
        } else {
            Ok((
                input,
                Pattern::EnumPattern {
                    enum_name: "".to_string(),
                    variant_name: name,
                    bindings: vec![],
                },
            ))
        }
    }
}

fn parse_match_expr(input: &str) -> IResult<&str, Expr> {
    // match expr { pat => expr, ... }
    let (input, _) = tag("match")(input)?;
    let (input, expr) = parse_expr(input)?;
    let (input, _) = ws(char('{'))(input)?;

    let (input, arms) = separated_list0(
        ws(char(',')),
        pair(ws(parse_pattern), preceded(ws(tag("=>")), parse_expr)),
    )(input)?;

    let (input, _) = opt(ws(char(',')))(input)?;
    let (input, _) = ws(char('}'))(input)?;

    Ok((
        input,
        Expr::Match {
            expr: Box::new(expr),
            arms,
        },
    ))
}

// Primary: Literal | Variable | (Expr) | Block? | IfExpr | Aggregation
fn parse_primary(input: &str) -> IResult<&str, Expr> {
    ws(alt((
        parse_literal_float,
        parse_literal_int,
        parse_literal_bool,
        parse_literal_string,
        parse_if_let_expr,
        parse_if_expr,
        parse_tensor_comprehension, // Try parsing comprehension before literal array
        parse_tensor_literal,
        parse_aggregation, // Must come before parse_variable
        parse_struct_init, // Must come before parse_variable
        parse_match_expr,
        parse_variable,
        parse_block_expr,
        parse_block_expr,
        parse_tuple_or_grouping,
    )))(input)
}

fn parse_tuple_or_grouping(input: &str) -> IResult<&str, Expr> {
    let (input, _) = ws(char('('))(input)?;
    let (input, mut exprs) = separated_list0(ws(char(',')), parse_expr)(input)?;
    // check for trailing comma to distinguish (expr,) from (expr)
    let (input, trailing_comma) = opt(ws(char(',')))(input)?;
    let (input, _) = ws(char(')'))(input)?;

    if exprs.is_empty() {
        // () -> Unit tuple (represented as empty tuple)
        // Or should we have a Unit literal?
        // Let's use empty tuple for now.
        Ok((input, Expr::Tuple(vec![])))
    } else if exprs.len() == 1 && trailing_comma.is_none() {
        // (expr) -> grouping
        Ok((input, exprs.remove(0)))
    } else {
        // (expr, ...) or (expr,) -> tuple
        Ok((input, Expr::Tuple(exprs)))
    }
}

// Postfix: Call, Index, Field, Method
// We parse a primary, then fold many suffixes
fn parse_postfix(input: &str) -> IResult<&str, Expr> {
    let (input, init) = parse_primary(input)?;

    nom::multi::fold_many0(
        ws(alt((
            // Call: (args...)
            map(
                delimited(
                    ws(char('(')),
                    separated_list0(ws(char(',')), parse_expr),
                    ws(char(')')),
                ),
                |args| (0, args, None, None), // Tag 0 for Call
            ),
            // Index: [indices...]
            map(
                delimited(
                    ws(char('[')),
                    separated_list1(ws(char(',')), parse_expr),
                    ws(char(']')),
                ),
                |idxs| (1, vec![], Some(idxs), None), // Tag 1 for Index
            ),
            // Field: .name
            map(
                preceded(char('.'), identifier),
                |name| (2, vec![], None, Some(name)), // Tag 2 for Field/Method
            ),
            // Tuple Access: .0, .1, etc.
            map(
                preceded(char('.'), digit1),
                |idx_str: &str| (3, vec![], None, Some(idx_str.to_string())), // Tag 3 for Tuple Access
            ),
        ))),
        move || init.clone(),
        |acc, (tag, args, idxs, name)| {
            match tag {
                0 => {
                    // Call.
                    match acc {
                        Expr::FieldAccess(obj, method_name) => {
                            Expr::MethodCall(obj, method_name, args)
                        }
                        Expr::Variable(fname) => {
                            // Check if we have Type::method in variable name (from identifier parser)
                            // The identifier parser allows "::".
                            // So "File::open" is returned as identifier string.
                            if fname.contains("::") {
                                let parts: Vec<&str> = fname.split("::").collect();
                                if parts.len() == 2 {
                                    Expr::StaticMethodCall(
                                        parts[0].to_string(),
                                        parts[1].to_string(),
                                        args,
                                    )
                                } else {
                                    // Fallback or Error?
                                    Expr::FnCall(fname, args)
                                }
                            } else {
                                Expr::FnCall(fname, args)
                            }
                        }
                        _ => Expr::FnCall("UNKNOWN_INDIRECT_CALL".to_string(), args),
                    }
                }
                1 => Expr::IndexAccess(Box::new(acc), idxs.unwrap()),
                2 => Expr::FieldAccess(Box::new(acc), name.unwrap()),
                3 => {
                    let idx = name.unwrap().parse::<usize>().unwrap_or(0);
                    Expr::TupleAccess(Box::new(acc), idx)
                }
                _ => unreachable!(),
            }
        },
    )(input)
}

// Cast: expr as Type
fn parse_cast(input: &str) -> IResult<&str, Expr> {
    let (input, lhs) = parse_postfix(input)?;

    // Handle optional "as Type" chain
    nom::multi::fold_many0(
        pair(ws(tag("as")), parse_type),
        move || lhs.clone(),
        |acc, (_, ty)| Expr::As(Box::new(acc), ty),
    )(input)
}

// 1. Unary: - !
fn parse_unary(input: &str) -> IResult<&str, Expr> {
    alt((
        map(pair(ws(char('-')), parse_cast), |(_, expr)| {
            Expr::UnOp(UnOp::Neg, Box::new(expr))
        }),
        map(pair(ws(char('!')), parse_cast), |(_, expr)| {
            Expr::UnOp(UnOp::Not, Box::new(expr))
        }),
        parse_cast,
    ))(input)
}

// 2. Multiplicative: * / %
fn parse_factor(input: &str) -> IResult<&str, Expr> {
    let (input, init) = parse_unary(input)?;

    nom::multi::fold_many0(
        pair(ws(alt((char('*'), char('/'), char('%')))), parse_unary),
        move || init.clone(),
        |acc, (op, val)| {
            let bin_op = match op {
                '*' => BinOp::Mul,
                '/' => BinOp::Div,
                '%' => BinOp::Mod,
                _ => unreachable!(),
            };
            Expr::BinOp(Box::new(acc), bin_op, Box::new(val))
        },
    )(input)
}

// 3. Additive: + -
fn parse_term(input: &str) -> IResult<&str, Expr> {
    let (input, init) = parse_factor(input)?;

    nom::multi::fold_many0(
        pair(ws(alt((char('+'), char('-')))), parse_factor),
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

// 4. Relational: < > <= >=
fn parse_relational(input: &str) -> IResult<&str, Expr> {
    let (input, init) = parse_term(input)?;

    nom::multi::fold_many0(
        pair(
            ws(alt((tag("<="), tag(">="), tag("<"), tag(">")))),
            parse_term,
        ),
        move || init.clone(),
        |acc, (op, val)| {
            let bin_op = match op {
                "<" => BinOp::Lt,
                ">" => BinOp::Gt,
                "<=" => BinOp::Le,
                ">=" => BinOp::Ge,
                _ => unreachable!(),
            };
            Expr::BinOp(Box::new(acc), bin_op, Box::new(val))
        },
    )(input)
}

// 5. Equality: == !=
fn parse_equality(input: &str) -> IResult<&str, Expr> {
    let (input, init) = parse_relational(input)?;

    nom::multi::fold_many0(
        pair(ws(alt((tag("=="), tag("!=")))), parse_relational),
        move || init.clone(),
        |acc, (op, val)| {
            let bin_op = match op {
                "==" => BinOp::Eq,
                "!=" => BinOp::Neq,
                _ => unreachable!(),
            };
            Expr::BinOp(Box::new(acc), bin_op, Box::new(val))
        },
    )(input)
}

// 6. Logical AND: &&
fn parse_logical_and(input: &str) -> IResult<&str, Expr> {
    let (input, init) = parse_equality(input)?;

    nom::multi::fold_many0(
        pair(ws(tag("&&")), parse_equality),
        move || init.clone(),
        |acc, (_, val)| Expr::BinOp(Box::new(acc), BinOp::And, Box::new(val)),
    )(input)
}

// 7. Logical OR: ||
fn parse_logical_or(input: &str) -> IResult<&str, Expr> {
    let (input, init) = parse_logical_and(input)?;

    nom::multi::fold_many0(
        pair(ws(tag("||")), parse_logical_and),
        move || init.clone(),
        |acc, (_, val)| Expr::BinOp(Box::new(acc), BinOp::Or, Box::new(val)),
    )(input)
}

// 8. Range: ..
fn parse_range(input: &str) -> IResult<&str, Expr> {
    let (input, start) = parse_logical_or(input)?;

    let (input, end) = opt(preceded(ws(tag("..")), parse_logical_or))(input)?;

    if let Some(end_expr) = end {
        Ok((input, Expr::Range(Box::new(start), Box::new(end_expr))))
    } else {
        Ok((input, start))
    }
}

// Top level Expr
pub fn parse_expr(input: &str) -> IResult<&str, Expr> {
    parse_range(input)
}

// --- Statements ---

// let x[i, j] = ...
fn parse_let_stmt(input: &str) -> IResult<&str, Stmt> {
    let (input, _) = tag("let")(input)?;
    let (input, _) = space1(input)?;
    let (input, name) = identifier(input)?;

    // Optional indices for tensor comprehension: [i, j]
    let (input, indices) = opt(delimited(
        tuple((char('['), space0)),
        separated_list0(tuple((char(','), space0)), identifier),
        tuple((space0, char(']'))),
    ))(input)?;

    let (input, _) = space0(input)?;
    let (input, type_annotation) = opt(preceded(tuple((char(':'), space0)), parse_type))(input)?;
    let (input, _) = space0(input)?;
    let (input, _) = char('=')(input)?;
    let (input, _) = space0(input)?;
    let (input, value) = parse_expr(input)?;
    let (input, _) = space0(input)?;
    let (input, _) = char(';')(input)?;

    if let Some(idxs) = indices {
        // Transform into TensorComprehension expression if indices are present
        // let C[i,j] = expr  -->  let C = TensorComprehension { indices: [i,j], body: expr }
        Ok((
            input,
            Stmt::Let {
                name,
                type_annotation,
                value: Expr::TensorComprehension {
                    indices: idxs,
                    clauses: Vec::new(),
                    body: Some(Box::new(value)),
                },
            },
        ))
    } else {
        Ok((
            input,
            Stmt::Let {
                name,
                type_annotation,
                value,
            },
        ))
    }
}

// x[i] += ... or x = ...
fn parse_assign_stmt(input: &str) -> IResult<&str, Stmt> {
    let (input, name) = ws(identifier)(input)?;

    let (input, indices) = opt(delimited(
        ws(char('[')),
        separated_list0(ws(char(',')), parse_expr),
        ws(char(']')),
    ))(input)?;

    let (input, op_str) = ws(alt((
        tag("="),
        tag("+="),
        tag("-="),
        tag("*="),
        tag("/="),
        tag("max="),
        tag("avg="),
    )))(input)?;

    let op = match op_str {
        "=" => AssignOp::Assign,
        "+=" => AssignOp::AddAssign,
        "-=" => AssignOp::SubAssign,
        "*=" => AssignOp::MulAssign,
        "/=" => AssignOp::DivAssign,
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

fn parse_while_stmt(input: &str) -> IResult<&str, Stmt> {
    // while expr { ... }
    let (input, _) = tag("while")(input)?;
    let (input, cond) = parse_expr(input)?;
    let (input, body) = parse_block(input)?;

    Ok((input, Stmt::While { cond, body }))
}

fn parse_return_stmt(input: &str) -> IResult<&str, Stmt> {
    let (input, _) = tag("return")(input)?;
    let (input, val) = opt(parse_expr)(input)?;
    let (input, _) = ws(char(';'))(input)?;
    Ok((input, Stmt::Return(val)))
}

fn parse_expr_stmt(input: &str) -> IResult<&str, Stmt> {
    // expr;
    let (input, expr) = parse_expr(input)?;

    // Check if expr is block-like (If, Match, Block)
    // In these cases, semicolon is optional in Rust-like syntax.
    let is_block_like = match &expr {
        Expr::IfExpr(_, _, _) => true,
        Expr::IfLet { .. } => true,
        Expr::Match { .. } => true,
        Expr::Block(_) => true,
        _ => false,
    };

    if is_block_like {
        let (input, _) = opt(ws(char(';')))(input)?;
        Ok((input, Stmt::Expr(expr)))
    } else {
        let (input, _) = ws(char(';'))(input)?;
        Ok((input, Stmt::Expr(expr)))
    }
}

fn parse_block_stmt(input: &str) -> IResult<&str, Stmt> {
    // block expression used as statement (optional semicolon)
    let (input, block) = parse_block_expr(input)?;
    let (input, _) = opt(ws(char(';')))(input)?;
    Ok((input, Stmt::Expr(block)))
}

fn parse_field_assign(input: &str) -> IResult<&str, Stmt> {
    let (input, lhs) = parse_expr(input)?;
    match lhs {
        Expr::FieldAccess(obj, field) => {
            let (input, op_str) =
                ws(alt((tag("="), tag("+="), tag("-="), tag("*="), tag("/="))))(input)?;

            let (input, value) = parse_expr(input)?;
            let (input, _) = ws(char(';'))(input)?;

            if op_str == "=" {
                Ok((
                    input,
                    Stmt::FieldAssign {
                        obj: *obj,
                        field,
                        value,
                    },
                ))
            } else {
                // Desugar x.y += z to x.y.add_assign(z)
                let method = match op_str {
                    "+=" => "add_assign",
                    "-=" => "sub_assign",
                    "*=" => "mul_assign",
                    "/=" => "div_assign",
                    _ => unreachable!(),
                };
                let method_call = Expr::MethodCall(
                    Box::new(Expr::FieldAccess(obj, field)),
                    method.to_string(),
                    vec![value],
                );
                Ok((input, Stmt::Expr(method_call)))
            }
        }
        _ => Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        ))),
    }
}

fn parse_use_stmt(input: &str) -> IResult<&str, Stmt> {
    let (input, _) = tag("use")(input)?;
    let (input, path_segments) = separated_list1(ws(tag("::")), ws(simple_identifier))(input)?;

    // Check for ":: {" to handle multi-import
    let (input, multi_items) = opt(preceded(
        ws(tag("::")),
        delimited(
            ws(char('{')),
            separated_list1(ws(char(',')), ws(simple_identifier)),
            ws(char('}')),
        ),
    ))(input)?;

    if let Some(items) = multi_items {
        let (input, _) = ws(char(';'))(input)?;
        Ok((
            input,
            Stmt::Use {
                path: path_segments,
                alias: None,
                items,
            },
        ))
    } else {
        // Single import, check for "as Alias"
        let (input, alias) = opt(preceded(ws(tag("as")), ws(simple_identifier)))(input)?;
        let (input, _) = ws(char(';'))(input)?;

        Ok((
            input,
            Stmt::Use {
                path: path_segments,
                alias,
                items: vec![],
            },
        ))
    }
}

pub fn parse_stmt(input: &str) -> IResult<&str, Stmt> {
    ws(alt((
        parse_let_stmt,
        parse_assign_stmt,
        parse_field_assign,
        parse_return_stmt,
        parse_if_stmt,
        parse_for_stmt,
        parse_while_stmt,
        parse_block_stmt,
        parse_use_stmt,
        parse_expr_stmt,
    )))(input)
}

// --- Top Level ---
fn parse_fn(input: &str) -> IResult<&str, FunctionDef> {
    // [extern] fn name(arg: Type, ...) -> Type { ... }
    let (input, is_extern) = opt(ws(tag("extern")))(input)?;
    let (input, _) = tag("fn")(input)?;
    let (input, name) = ws(identifier)(input)?;

    // Args: (a: T, b: U)
    let (input, args) = delimited(
        ws(char('(')),
        separated_list0(ws(char(',')), parse_fn_arg),
        ws(char(')')),
    )(input)?;

    let (input, ret_type) = opt(preceded(ws(tag("->")), ws(parse_type)))(input)?;

    // Body logic: if extern, expect semicolon and empty body. Else parse block.
    let (input, body) = if is_extern.is_some() {
        let (input, _) = ws(char(';'))(input)?;
        (input, vec![])
    } else {
        parse_block(input)?
    };

    Ok((
        input,
        FunctionDef {
            name,
            args,
            return_type: ret_type.unwrap_or(Type::Void),
            body,
            generics: vec![],
            is_extern: is_extern.is_some(),
        },
    ))
}

fn parse_fn_arg(input: &str) -> IResult<&str, (String, Type)> {
    let (input, name) = ws(identifier)(input)?;
    let (input, ty) = opt(preceded(ws(char(':')), parse_type))(input)?;

    match ty {
        Some(t) => Ok((input, (name, t))),
        None => {
            if name == "self" {
                Ok((input, (name, Type::UserDefined("Self".to_string()))))
            } else {
                // Fail if no type and not self
                Err(nom::Err::Error(nom::error::Error::new(
                    input,
                    nom::error::ErrorKind::Tag,
                )))
            }
        }
    }
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

fn parse_variant_def(input: &str) -> IResult<&str, VariantDef> {
    // Name { field: Type, ... } or Name
    let (input, name) = ws(identifier)(input)?;

    // Check for struct-like body
    let (input, fields_opt) = opt(delimited(
        ws(char('{')),
        separated_list0(
            ws(char(',')),
            pair(ws(identifier), preceded(ws(char(':')), parse_type)),
        ),
        ws(char('}')),
    ))(input)?;

    Ok((
        input,
        VariantDef {
            name,
            fields: fields_opt.unwrap_or_default(),
        },
    ))
}

fn parse_enum_def(input: &str) -> IResult<&str, EnumDef> {
    // enum Name<T> { Variant, ... }
    let (input, _) = tag("enum")(input)?;
    let (input, name) = ws(identifier)(input)?;

    let (input, generics) = opt(delimited(
        ws(char('<')),
        separated_list1(ws(char(',')), identifier),
        ws(char('>')),
    ))(input)?;

    let (input, _) = ws(char('{'))(input)?;
    let (input, variants) = separated_list0(ws(char(',')), parse_variant_def)(input)?;
    let (input, _) = opt(ws(char(',')))(input)?;
    let (input, _) = ws(char('}'))(input)?;

    Ok((
        input,
        EnumDef {
            name,
            variants,
            generics: generics.unwrap_or_default(),
        },
    ))
}

// --- New Top Level Parsers ---
fn parse_tensor_decl(input: &str) -> IResult<&str, Stmt> {
    // tensor name: Type = Expr;
    let (input, _) = tag("tensor")(input)?;
    let (input, name) = ws(identifier)(input)?;
    let (input, _) = ws(char(':'))(input)?;
    let (input, type_annotation) = ws(parse_type)(input)?;

    let (input, init) = opt(preceded(ws(char('=')), parse_expr))(input)?;
    let (input, _) = opt(ws(char(';')))(input)?; // Optional semicolon

    Ok((
        input,
        Stmt::TensorDecl {
            name,
            type_annotation,
            init,
        },
    ))
}

fn parse_relation_decl(input: &str) -> IResult<&str, RelationDecl> {
    // relation Name(arg: Type, ...)
    let (input, _) = tag("relation")(input)?;
    let (input, name) = ws(identifier)(input)?;
    let (input, args) = delimited(
        ws(char('(')),
        separated_list0(
            ws(char(',')),
            pair(ws(identifier), preceded(ws(char(':')), parse_type)),
        ),
        ws(char(')')),
    )(input)?;
    let (input, _) = opt(ws(char(';')))(input)?;

    Ok((input, RelationDecl { name, args }))
}

fn parse_atom(input: &str) -> IResult<&str, Atom> {
    // Name(arg, ...) - parentheses required but can be empty for nilary
    let (input, predicate) = ws(identifier)(input)?;
    let (input, args) = delimited(
        ws(char('(')),
        separated_list0(ws(char(',')), parse_expr),
        ws(char(')')),
    )(input)?;
    Ok((input, Atom { predicate, args }))
}

fn parse_fact(input: &str) -> IResult<&str, Rule> {
    // Fact: Head(args). - no body, just period terminated
    let (input, head) = parse_atom(input)?;
    let (input, _) = ws(char('.'))(input)?;

    Ok((
        input,
        Rule {
            head,
            body: vec![],
            weight: None,
        },
    ))
}

fn parse_rule(input: &str) -> IResult<&str, Rule> {
    // Head(args) :- Body(args), ...
    // Note: Ambiguity with function call if we don't look ahead for `:-`
    // Using peek could be messy. For now strict order in top level loop or use verify.
    // Parser combinator `tuple` will fail if `:-` is missing.
    let (input, head) = parse_atom(input)?;
    let (input, _) = ws(tag(":-"))(input)?;
    let (input, body) = separated_list1(ws(char(',')), parse_atom)(input)?;

    let (input, _) = alt((ws(char('.')), ws(char(';'))))(input)?;

    Ok((
        input,
        Rule {
            head,
            body,
            weight: None,
        },
    ))
}

fn parse_query(input: &str) -> IResult<&str, Expr> {
    // ?- Expr
    let (input, _) = tag("?-")(input)?;
    let (input, expr) = parse_expr(input)?;
    let (input, _) = opt(ws(char(';')))(input)?;
    Ok((input, expr))
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

// --- Imports ---
use std::collections::HashMap;

fn parse_mod_decl(input: &str) -> IResult<&str, String> {
    let (input, _) = tag("mod")(input)?;
    let (input, name) = ws(identifier)(input)?;
    let (input, _) = ws(char(';'))(input)?;
    Ok((input, name))
}

// --- Top Level ---
// Re-define parse_module logic
pub fn parse(input: &str) -> anyhow::Result<Module> {
    let mut structs = vec![];
    let mut enums = vec![];
    let mut impls = vec![];
    let mut functions = vec![];
    let mut tensor_decls = vec![];
    let mut relations = vec![];
    let mut rules = vec![];
    let mut queries = vec![];
    let mut imports = vec![];

    // Strip BOM if present
    let mut remaining = input.strip_prefix("\u{feff}").unwrap_or(input);

    while !remaining.trim().is_empty() {
        // println!("Parsing at: {:.20}...", remaining); // Debug print

        // Try struct, then impl, then fn, then others
        if let Ok((next, s)) = ws(parse_struct)(remaining) {
            structs.push(s);
            remaining = next;
        } else if let Ok((next, e)) = ws(parse_enum_def)(remaining) {
            enums.push(e);
            remaining = next;
        } else if let Ok((next, i)) = ws(parse_impl)(remaining) {
            impls.push(i);
            remaining = next;
        } else if let Ok((next, f)) = ws(parse_fn)(remaining) {
            functions.push(f);
            remaining = next;
        } else if let Ok((next, m)) = ws(parse_mod_decl)(remaining) {
            imports.push(m);
            remaining = next;
        } else if let Ok((next, t)) = ws(parse_tensor_decl)(remaining) {
            tensor_decls.push(t);
            remaining = next;
        } else if let Ok((next, s)) = ws(parse_stmt)(remaining) {
            tensor_decls.push(s);
            remaining = next;
        } else if let Ok((next, r)) = ws(parse_relation_decl)(remaining) {
            relations.push(r);
            remaining = next;
        } else if let Ok((next, r)) = ws(parse_fact)(remaining) {
            rules.push(r);
            remaining = next;
        } else if let Ok((next, r)) = ws(parse_rule)(remaining) {
            rules.push(r);
            remaining = next;
        } else if let Ok((next, q)) = ws(parse_query)(remaining) {
            queries.push(q);
            remaining = next;
        } else {
            // Explicitly try to consume comments independently if parsers failed?
            // ws() wraps each parser, so it should handle comments.
            // If all failed, it means remaining is NOT matched by ANY.
            // Maybe leading whitespace is not fully consumed by ws() if wrapped?
            // Actually, remaining is passed to ws(parser). ws() does delimited(sp, inner, sp).
            // It consumes leading sp.
            // Use {:?} to show invisible chars like BOM or weird spaces
            return Err(anyhow::anyhow!(
                "Parse error at: {:?}...",
                &remaining[..remaining.len().min(30)]
            ));
        }
    }

    Ok(Module {
        structs,
        enums,
        impls,
        functions,
        tensor_decls,
        relations,
        rules,
        queries,
        imports,
        submodules: HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_full_module() {
        let code = r#"
            fn main() {
                let N = 8;
                let x = 1e-4;
            }
        "#;
        let res = parse(code);
        assert!(res.is_ok(), "Failed to parse module: {:?}", res.err());
    }

    #[test]
    fn test_parse_float_scientific() {
        // Standard floats
        let res = parse_literal_float("1.0");
        assert!(res.is_ok(), "Failed directly on 1.0: {:?}", res);

        // Scientific with dot
        let res = parse_literal_float("1.0e-4");
        assert!(res.is_ok(), "Failed directly on 1.0e-4: {:?}", res);

        let (_, expr) = res.unwrap();
        if let Expr::Float(val) = expr {
            assert!((val - 0.0001).abs() < 1e-10);
        }

        // Scientific without dot
        let res = parse_literal_float("1e-4");
        assert!(res.is_ok(), "Failed directly on 1e-4: {:?}", res);

        let (_, expr) = res.unwrap();
        if let Expr::Float(val) = expr {
            assert!((val - 0.0001).abs() < 1e-10);
        }

        // Capital E
        let res = parse_literal_float("1E5");
        assert!(res.is_ok());
        if let Expr::Float(val) = res.unwrap().1 {
            assert_eq!(val, 100000.0);
        }
    }
}
