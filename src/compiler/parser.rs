// src/compiler/parser.rs
use crate::compiler::ast::*;
use crate::compiler::error::{ParseErrorKind, TlError};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{alpha1, char, digit1, space0, space1},
    combinator::{map, not, opt, recognize, value, verify},
    multi::{separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    IResult,
};
use nom_locate::{position, LocatedSpan};

pub type Span<'a> = LocatedSpan<&'a str>;

pub fn spanned<'a, F, O>(mut parser: F) -> impl FnMut(Span<'a>) -> IResult<Span<'a>, Spanned<O>>
where
    F: FnMut(Span<'a>) -> IResult<Span<'a>, O>,
{
    move |input: Span<'a>| {
        let (input, start) = position(input)?;
        let (input, value) = parser(input)?;
        let span = crate::compiler::error::Span::new(
            start.location_line() as usize,
            start.get_utf8_column(),
        );
        Ok((input, Spanned::new(value, span)))
    }
}

// --- Whitespace & Comments ---
// --- Whitespace & Comments ---
fn ws<'a, F, O, E>(inner: F) -> impl FnMut(Span<'a>) -> IResult<Span<'a>, O, E>
where
    F: FnMut(Span<'a>) -> IResult<Span<'a>, O, E>,
    E: nom::error::ParseError<Span<'a>>,
{
    delimited(sp, inner, sp)
}

fn sp<'a, E: nom::error::ParseError<Span<'a>>>(input: Span<'a>) -> IResult<Span<'a>, Span<'a>, E> {
    let chars = " \t\r\n";
    // Consumes whitespace or comments recursively
    recognize(nom::multi::many0(alt((
        take_while1(move |c| chars.contains(c)),
        preceded(tag("//"), take_while(|c| c != '\n')),
    ))))(input)
}

// --- Identifiers ---

fn simple_identifier(input: Span) -> IResult<Span, String> {
    let (input, s) = recognize(pair(
        alt((alpha1, tag("_"))),
        take_while(|c: char| c.is_alphanumeric() || c == '_'),
    ))(input)?;
    Ok((input, s.fragment().to_string()))
}

fn identifier(input: Span) -> IResult<Span, String> {
    // Allow usage of :: for namespaces (e.g. File::open)
    verify(
        map(
            recognize(separated_list1(tag("::"), simple_identifier)),
            |s: Span| s.fragment().to_string(),
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
fn parse_primitive_type(input: Span) -> IResult<Span, Type> {
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

fn parse_tensor_type(input: Span) -> IResult<Span, Type> {
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
            let rank = rank_str.fragment().parse::<usize>().unwrap_or(0);
            Type::Tensor(Box::new(inner), rank)
        },
    )(input)
}

fn parse_user_type(input: Span) -> IResult<Span, Type> {
    // Check for Enum vs Struct in identifier context? No, just identifier.
    // Type::Struct / Type::Enum / Type::UserDefined are currently ambiguous in parser until resolution.
    // Let's use UserDefined for now for any identifier in type position.
    // But later semantics will resolve it.
    // However, if we want explicit Type::Enum for syntax highlighting or specialized parsing?
    // For now, Type::UserDefined covers both.
    map(identifier, Type::UserDefined)(input)
}

fn parse_tuple_type(input: Span) -> IResult<Span, Type> {
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

pub fn parse_type(input: Span) -> IResult<Span, Type> {
    alt((
        parse_tensor_type,
        parse_primitive_type,
        parse_tuple_type,
        parse_user_type,
    ))(input)
}

// --- Expressions ---

fn parse_int(input: Span) -> IResult<Span, Expr> {
    spanned(map(digit1, |s: Span| {
        let s_str = s.fragment();
        ExprKind::Int(s_str.parse::<i64>().unwrap())
    }))(input)
}

// Helper for exponent: e/E followed by optional sign and digits

fn parse_literal_float(input: Span) -> IResult<Span, Expr> {
    spanned(|input| {
        let (rest, val_span) = nom::number::complete::recognize_float(input)?;
        let s = val_span.fragment();
        // Enforce dot or exponent to distinguish from integers
        if !s.contains('.') && !s.contains('e') && !s.contains('E') {
            return Err(nom::Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Float,
            )));
        }
        // Avoid eating range operator: if float ends with '.' and next char is '.', reject
        if s.ends_with('.') && rest.fragment().starts_with('.') {
            return Err(nom::Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Float,
            )));
        }
        let val = s.parse::<f64>().unwrap();
        Ok((rest, ExprKind::Float(val)))
    })(input)
}

fn parse_bool(input: Span) -> IResult<Span, Expr> {
    spanned(alt((
        map(tag("true"), |_| ExprKind::Bool(true)),
        map(tag("false"), |_| ExprKind::Bool(false)),
    )))(input)
}

fn parse_literal_string(input: Span) -> IResult<Span, Expr> {
    spanned(map(
        delimited(char('"'), take_while(|c: char| c != '"'), char('"')),
        |s: Span| ExprKind::StringLiteral(s.fragment().to_string()),
    ))(input)
}

fn parse_variable(input: Span) -> IResult<Span, Expr> {
    spanned(map(identifier, ExprKind::Variable))(input)
}

// --- Blocks ---
fn parse_block(input: Span) -> IResult<Span, Vec<Stmt>> {
    let (input, _) = ws(char('{'))(input)?;
    let (input, mut stmts) = nom::multi::many0(parse_stmt)(input)?;

    // Optional trailing expression without semicolon
    // But parse_stmt includes parse_expr_stmt which forces semicolon.
    // So here we look for raw parse_expr.
    let (input, trailing) = opt(parse_expr)(input)?;

    let (input, _) = ws(char('}'))(input)?;

    if let Some(expr) = trailing {
        stmts.push(Spanned::dummy(StmtKind::Expr(expr)));
    }

    Ok((input, stmts))
}

fn parse_block_expr(input: Span) -> IResult<Span, Expr> {
    spanned(map(parse_block, ExprKind::Block))(input)
}

fn parse_comprehension_clause(input: Span) -> IResult<Span, ComprehensionClause> {
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

fn parse_tensor_comprehension(input: Span) -> IResult<Span, Expr> {
    spanned(map(
        tuple((
            preceded(
                ws(char('[')),
                separated_list1(ws(char(',')), ws(identifier)),
            ),
            preceded(
                ws(char('|')),
                separated_list0(ws(char(',')), parse_comprehension_clause),
            ),
            opt(parse_block_expr),
            ws(char(']')),
        )),
        |(indices, clauses, body_expr, _)| ExprKind::TensorComprehension {
            indices: indices.into_iter().map(|s| s.to_string()).collect(),
            clauses,
            body: body_expr.map(Box::new),
        },
    ))(input)
}

// --- Control Flow ---
fn parse_if_expr(input: Span) -> IResult<Span, Expr> {
    spanned(map(
        tuple((
            preceded(tag("if"), parse_expr),
            parse_block,
            opt(preceded(ws(tag("else")), parse_block)),
        )),
        |(cond, then_block, else_block)| ExprKind::IfExpr(Box::new(cond), then_block, else_block),
    ))(input)
}

fn parse_if_let_expr(input: Span) -> IResult<Span, Expr> {
    spanned(map(
        tuple((
            preceded(pair(tag("if"), ws(tag("let"))), ws(parse_pattern)),
            preceded(ws(char('=')), parse_expr),
            parse_block,
            opt(preceded(ws(tag("else")), parse_block)),
        )),
        |(pattern, expr, then_block, else_block)| ExprKind::IfLet {
            pattern,
            expr: Box::new(expr),
            then_block,
            else_block,
        },
    ))(input)
}

fn parse_tensor_literal(input: Span) -> IResult<Span, Expr> {
    spanned(map(
        delimited(
            ws(char('[')),
            separated_list0(ws(char(',')), parse_expr),
            ws(char(']')),
        ),
        |elements| {
            // Check if all elements are constants (recursively)
            fn is_const(expr: &Expr) -> bool {
                match &expr.inner {
                    ExprKind::Float(_) | ExprKind::Int(_) | ExprKind::Bool(_) => true,
                    ExprKind::TensorLiteral(ref elems)
                    | ExprKind::TensorConstLiteral(ref elems) => elems.iter().all(is_const),
                    _ => false,
                }
            }

            let all_const = elements.iter().all(is_const);

            if all_const {
                ExprKind::TensorConstLiteral(elements)
            } else {
                ExprKind::TensorLiteral(elements)
            }
        },
    ))(input)
}

// Aggregation: sum(expr for var in range) or sum(expr for var in range where cond)
fn parse_aggregation(input: Span) -> IResult<Span, Expr> {
    spanned(map(
        tuple((
            ws(alt((
                tag("sum"),
                tag("max"),
                tag("min"),
                tag("avg"),
                tag("count"),
            ))),
            delimited(
                ws(char('(')),
                tuple((
                    parse_expr,
                    preceded(ws(tag("for")), ws(identifier)),
                    preceded(ws(tag("in")), parse_expr),
                    opt(preceded(ws(tag("where")), parse_expr)),
                )),
                ws(char(')')),
            ),
        )),
        |(op_str, (expr, var, range, condition))| {
            let op = match *op_str {
                "sum" => AggregateOp::Sum,
                "max" => AggregateOp::Max,
                "min" => AggregateOp::Min,
                "avg" => AggregateOp::Avg,
                "count" => AggregateOp::Count,
                _ => unreachable!(),
            };
            ExprKind::Aggregation {
                op,
                expr: Box::new(expr),
                var,
                range: Box::new(range),
                condition: condition.map(Box::new),
            }
        },
    ))(input)
}

fn parse_struct_init(input: Span) -> IResult<Span, Expr> {
    spanned(map(
        tuple((
            ws(identifier),
            delimited(
                ws(char('{')),
                terminated(
                    separated_list0(
                        ws(char(',')),
                        pair(ws(identifier), preceded(ws(char(':')), parse_expr)),
                    ),
                    opt(ws(char(','))),
                ),
                ws(char('}')),
            ),
        )),
        |(name, fields)| ExprKind::StructInit(name, fields),
    ))(input)
}

#[allow(dead_code)]
fn parse_enum_init(input: Span) -> IResult<Span, Expr> {
    // Enum::Variant { field: value ... }
    // Identifier "Enum::Variant" or just "Variant"?
    // Rust requires fully qualified or imported.
    // Our identifier parser handles "A::B".
    // So if we see "A::B { ... }", is it StructInit or EnumInit?
    // StructInit currently takes identifier.
    // If identifier has "::", we could treat it as EnumInit OR StructInit (mod::Struct).
    // Let's assume generic StructInit handles "A::B" as name.
    // BUT we need to produce ExprKind::EnumInit if it's an Enum.
    // Since Parser doesn't know types, we must produce a "GenericInit" or rely on Semantics to re-classify?
    // OR we can make a heuristic: if it looks like "X::Y { ... }", treat as potentially enum.
    // Actually, `ExprKind::StructInit` stores a String name. Semantics can resolve "X::Y" to Enum or Struct.
    // So maybe we don't strictly need distinct parse_enum_init IF StructInit covers the syntax.
    // HOWEVER, we added ExprKind::EnumInit.
    // Let's modify parse_struct_init to return EnumInit if it detects "::".
    // Wait, Structs can differ by module too: `mod::Struct { ... }`.
    // So syntax is identical.
    // Strategy: Parse as StructInit in Parser. In Semantics, if name resolves to Enum, convert to EnumInit.
    // But wait, user requested "Implement Parser support".
    // I can leave it as StructInit and fix in Semantics, OR I can try to disambiguate.
    // Ambiguity is impossible without symbol table.
    // So: I will stick to ParseStructInit, but I'll rename it/comment it or handle conversion in Semantics.
    // Wait, ExprKind::EnumInit fields: enum_name, variant_name. StructInit only has name.
    // If I stick to StructInit, I need to split name in Semantics.
    // OK, let's keep parse_struct_init as the shared parser.
    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Fail,
    )))
}

fn parse_pattern(input: Span) -> IResult<Span, Pattern> {
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

fn parse_match_expr(input: Span) -> IResult<Span, Expr> {
    spanned(map(
        tuple((
            preceded(tag("match"), parse_expr),
            delimited(
                ws(char('{')),
                separated_list0(
                    ws(char(',')),
                    pair(ws(parse_pattern), preceded(ws(tag("=>")), parse_expr)),
                ),
                terminated(ws(char('}')), opt(ws(char(',')))),
            ),
        )),
        |(expr, arms)| ExprKind::Match {
            expr: Box::new(expr),
            arms,
        },
    ))(input)
}

// Primary: Literal | Variable | (Expr) | Block? | IfExpr | Aggregation
fn parse_atom(input: Span) -> IResult<Span, Expr> {
    ws(alt((
        parse_literal_float,
        parse_int,
        parse_bool,
        parse_literal_string,
        parse_if_let_expr,
        parse_if_expr,
        parse_tensor_comprehension, // Try parsing comprehension before literal array
        parse_tensor_literal,
        parse_aggregation,
        parse_struct_init,
        parse_match_expr,
        parse_block_expr,
        parse_tuple_or_grouping,
        parse_variable, // Variable must be last to avoid shadowing keywords/other structures?
    )))(input)
}

fn parse_tuple_or_grouping(input: Span) -> IResult<Span, Expr> {
    spanned(|input| {
        let (input, _) = ws(char('('))(input)?;
        let (input, mut exprs) = separated_list0(ws(char(',')), parse_expr)(input)?;
        // check for trailing comma to distinguish (expr,) from (expr)
        let (input, trailing_comma) = opt(ws(char(',')))(input)?;
        let (input, _) = ws(char(')'))(input)?;

        if exprs.is_empty() {
            // () -> Unit tuple (represented as empty tuple)
            // Or should we have a Unit literal?
            // Let's use empty tuple for now.
            Ok((input, ExprKind::Tuple(vec![])))
        } else if exprs.len() == 1 && trailing_comma.is_none() {
            // (expr) -> grouping
            // Since we construct a new Spanned, we need to extract the inner ExprKind if we want to re-wrap it,
            // or just return the inner expression. But spanned() will re-wrap whatever we return.
            // If we return the inner expression's Kind, spanned() will wrap it with the outer parens' span.
            // This is correct for grouping: (expr) has the span including parens.
            Ok((input, exprs.remove(0).inner))
        } else {
            // (expr, ...) or (expr,) -> tuple
            Ok((input, ExprKind::Tuple(exprs)))
        }
    })(input)
}

// Postfix: Call, Index, Field, Method
// We parse a primary, then fold many suffixes
fn parse_postfix(input: Span) -> IResult<Span, Expr> {
    let (input, init) = parse_atom(input)?;

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
                |idx_str: Span| (3, vec![], None, Some(idx_str.fragment().to_string())), // Tag 3 for Tuple Access
            ),
        ))),
        move || init.clone(),
        |acc, (tag, args, idxs, name)| {
            let span = acc.span.clone(); // Propagate span from left (start)
            let kind = match tag {
                0 => {
                    // Call.
                    match acc.inner {
                        ExprKind::FieldAccess(obj, method_name) => {
                            ExprKind::MethodCall(obj, method_name, args)
                        }
                        ExprKind::Variable(fname) => {
                            // Check if we have Type::method in variable name (from identifier parser)
                            // The identifier parser allows "::".
                            // So "File::open" is returned as identifier string.
                            if fname.contains("::") {
                                let parts: Vec<&str> = fname.split("::").collect();
                                if parts.len() == 2 {
                                    ExprKind::StaticMethodCall(
                                        parts[0].to_string(),
                                        parts[1].to_string(),
                                        args,
                                    )
                                } else {
                                    // Fallback or Error?
                                    ExprKind::FnCall(fname, args)
                                }
                            } else {
                                ExprKind::FnCall(fname, args)
                            }
                        }
                        _ => ExprKind::FnCall("UNKNOWN_INDIRECT_CALL".to_string(), args),
                    }
                }
                1 => ExprKind::IndexAccess(Box::new(acc), idxs.unwrap()),
                2 => ExprKind::FieldAccess(Box::new(acc), name.unwrap()),
                3 => {
                    let idx = name.unwrap().parse::<usize>().unwrap_or(0);
                    ExprKind::TupleAccess(Box::new(acc), idx)
                }
                _ => unreachable!(),
            };
            Spanned::new(kind, span)
        },
    )(input)
}

// Cast: expr as Type
fn parse_cast(input: Span) -> IResult<Span, Expr> {
    let (input, lhs) = parse_postfix(input)?;

    // Handle optional "as Type" chain
    nom::multi::fold_many0(
        pair(ws(tag("as")), parse_type),
        move || lhs.clone(),
        |acc, (_, ty)| Spanned::new(ExprKind::As(Box::new(acc.clone()), ty), acc.span.clone()),
    )(input)
}

// 1. Unary: - !
fn parse_unary(input: Span) -> IResult<Span, Expr> {
    alt((
        spanned(map(pair(ws(char('-')), parse_cast), |(_, expr)| {
            ExprKind::UnOp(UnOp::Neg, Box::new(expr))
        })),
        spanned(map(pair(ws(char('!')), parse_cast), |(_, expr)| {
            ExprKind::UnOp(UnOp::Not, Box::new(expr))
        })),
        parse_cast,
    ))(input)
}

// 2. Multiplicative: * / %
// 2. Multiplicative: * / %
fn parse_factor(input: Span) -> IResult<Span, Expr> {
    let (input, init) = parse_unary(input)?;

    nom::multi::fold_many0(
        pair(ws(alt((char('*'), char('/'), char('%')))), parse_unary),
        move || init.clone(),
        |acc, (op, val)| {
            let span = acc.span.clone();
            let bin_op = match op {
                '*' => BinOp::Mul,
                '/' => BinOp::Div,
                '%' => BinOp::Mod,
                _ => unreachable!(),
            };
            Spanned::new(ExprKind::BinOp(Box::new(acc), bin_op, Box::new(val)), span)
        },
    )(input)
}

// 3. Additive: + -
// 3. Additive: + -
fn parse_term(input: Span) -> IResult<Span, Expr> {
    let (input, init) = parse_factor(input)?;

    nom::multi::fold_many0(
        pair(ws(alt((char('+'), char('-')))), parse_factor),
        move || init.clone(),
        |acc, (op, val)| {
            let span = acc.span.clone();
            let bin_op = match op {
                '+' => BinOp::Add,
                '-' => BinOp::Sub,
                _ => unreachable!(),
            };
            Spanned::new(ExprKind::BinOp(Box::new(acc), bin_op, Box::new(val)), span)
        },
    )(input)
}

// 4. Relational: < > <= >=
// 4. Relational: < > <= >=
fn parse_relational(input: Span) -> IResult<Span, Expr> {
    let (input, init) = parse_term(input)?;

    nom::multi::fold_many0(
        pair(
            ws(alt((tag("<="), tag(">="), tag("<"), tag(">")))),
            parse_term,
        ),
        move || init.clone(),
        |acc, (op, val)| {
            let span = acc.span.clone();
            let bin_op = match *op {
                "<" => BinOp::Lt,
                ">" => BinOp::Gt,
                "<=" => BinOp::Le,
                ">=" => BinOp::Ge,
                _ => unreachable!(),
            };
            Spanned::new(ExprKind::BinOp(Box::new(acc), bin_op, Box::new(val)), span)
        },
    )(input)
}

// 5. Equality: == !=
// 5. Equality: == !=
fn parse_equality(input: Span) -> IResult<Span, Expr> {
    let (input, init) = parse_relational(input)?;

    nom::multi::fold_many0(
        pair(ws(alt((tag("=="), tag("!=")))), parse_relational),
        move || init.clone(),
        |acc, (op, val)| {
            let span = acc.span.clone();
            let bin_op = match *op {
                "==" => BinOp::Eq,
                "!=" => BinOp::Neq,
                _ => unreachable!(),
            };
            Spanned::new(ExprKind::BinOp(Box::new(acc), bin_op, Box::new(val)), span)
        },
    )(input)
}

// 6. Logical AND: &&
// 6. Logical AND: &&
fn parse_logical_and(input: Span) -> IResult<Span, Expr> {
    let (input, init) = parse_equality(input)?;

    nom::multi::fold_many0(
        pair(ws(tag("&&")), parse_equality),
        move || init.clone(),
        |acc, (_, val)| {
            let span = acc.span.clone();
            Spanned::new(
                ExprKind::BinOp(Box::new(acc), BinOp::And, Box::new(val)),
                span,
            )
        },
    )(input)
}

// 7. Logical OR: ||
// 7. Logical OR: ||
fn parse_logical_or(input: Span) -> IResult<Span, Expr> {
    let (input, init) = parse_logical_and(input)?;

    nom::multi::fold_many0(
        pair(ws(tag("||")), parse_logical_and),
        move || init.clone(),
        |acc, (_, val)| {
            let span = acc.span.clone();
            Spanned::new(
                ExprKind::BinOp(Box::new(acc), BinOp::Or, Box::new(val)),
                span,
            )
        },
    )(input)
}

// 8. Range: ..
// 8. Range: ..
fn parse_range(input: Span) -> IResult<Span, Expr> {
    let (input, start) = parse_logical_or(input)?;

    let (input, end) = opt(preceded(ws(tag("..")), parse_logical_or))(input)?;

    if let Some(end_expr) = end {
        Ok((
            input,
            Spanned::new(
                ExprKind::Range(Box::new(start.clone()), Box::new(end_expr)),
                start.span,
            ),
        ))
    } else {
        Ok((input, start))
    }
}

// Top level Expr
pub fn parse_expr(input: Span) -> IResult<Span, Expr> {
    parse_range(input)
}

// --- Statements ---

// let x[i, j] = ... or let mut x = ...
// let x[i, j] = ... or let mut x = ...
fn parse_let_stmt(input: Span) -> IResult<Span, Stmt> {
    spanned(|input| {
        let (input, _) = tag("let")(input)?;
        let (input, _) = space1(input)?;

        // Check for "mut" keyword
        let (input, is_mut) = opt(terminated(tag("mut"), space1))(input)?;
        let (input, name) = identifier(input)?;

        // Optional indices for tensor comprehension: [i, j]
        let (input, indices) = opt(delimited(
            tuple((char('['), space0)),
            separated_list0(tuple((char(','), space0)), identifier),
            tuple((space0, char(']'))),
        ))(input)?;

        let (input, _) = space0(input)?;
        let (input, type_annotation) =
            opt(preceded(tuple((char(':'), space0)), parse_type))(input)?;
        let (input, _) = space0(input)?;
        let (input, _) = char('=')(input)?;
        let (input, _) = space0(input)?;
        let (input, value) = parse_expr(input)?;
        let (input, _) = space0(input)?;
        let (input, _) = char(';')(input)?;

        let mutable = is_mut.is_some();

        if let Some(idxs) = indices {
            // Transform into TensorComprehension expression if indices are present
            Ok((
                input,
                StmtKind::Let {
                    name,
                    type_annotation,
                    value: Spanned::dummy(ExprKind::TensorComprehension {
                        indices: idxs,
                        clauses: Vec::new(),
                        body: Some(Box::new(value)),
                    }),
                    mutable,
                },
            ))
        } else {
            Ok((
                input,
                StmtKind::Let {
                    name,
                    type_annotation,
                    value,
                    mutable,
                },
            ))
        }
    })(input)
}

// x[i] += ... or x = ...
fn parse_assign_stmt(input: Span) -> IResult<Span, Stmt> {
    spanned(|input| {
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

        let op = match *op_str {
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
            StmtKind::Assign {
                name,
                indices,
                op,
                value,
            },
        ))
    })(input)
}

// if (stmt version usually wraps expression if return ignored, or purely statement)
// For now, treat top-level if as StmtKind::If.
fn parse_if_stmt(input: Span) -> IResult<Span, Stmt> {
    spanned(|input| {
        let (input, _) = tag("if")(input)?;
        let (input, cond) = parse_expr(input)?;
        let (input, then_block) = parse_block(input)?;

        let (input, else_block) = opt(preceded(ws(tag("else")), parse_block))(input)?;

        Ok((
            input,
            StmtKind::If {
                cond,
                then_block,
                else_block,
            },
        ))
    })(input)
}

fn parse_for_stmt(input: Span) -> IResult<Span, Stmt> {
    spanned(|input| {
        // for var in expr { ... }
        let (input, _) = tag("for")(input)?;
        let (input, loop_var) = ws(identifier)(input)?;
        let (input, _) = ws(tag("in"))(input)?;
        let (input, iterator) = parse_expr(input)?; // currently generic expr, could be range
        let (input, body) = parse_block(input)?;

        Ok((
            input,
            StmtKind::For {
                loop_var,
                iterator,
                body,
            },
        ))
    })(input)
}

fn parse_while_stmt(input: Span) -> IResult<Span, Stmt> {
    spanned(|input| {
        // while expr { ... }
        let (input, _) = tag("while")(input)?;
        let (input, cond) = parse_expr(input)?;
        let (input, body) = parse_block(input)?;

        Ok((input, StmtKind::While { cond, body }))
    })(input)
}

fn parse_return_stmt(input: Span) -> IResult<Span, Stmt> {
    spanned(|input| {
        let (input, _) = tag("return")(input)?;
        let (input, val) = opt(parse_expr)(input)?;
        let (input, _) = ws(char(';'))(input)?;
        Ok((input, StmtKind::Return(val)))
    })(input)
}

fn parse_expr_stmt(input: Span) -> IResult<Span, Stmt> {
    spanned(|input| {
        // expr;
        let (input, expr) = parse_expr(input)?;

        // Check if expr is block-like (If, Match, Block)
        // In these cases, semicolon is optional in Rust-like syntax.
        let is_block_like = match &expr.inner {
            // Check inner
            ExprKind::IfExpr(_, _, _) => true,
            ExprKind::IfLet { .. } => true,
            ExprKind::Match { .. } => true,
            ExprKind::Block(_) => true,
            _ => false,
        };

        if is_block_like {
            let (input, _) = opt(ws(char(';')))(input)?;
            Ok((input, StmtKind::Expr(expr)))
        } else {
            let (input, _) = ws(char(';'))(input)?;
            Ok((input, StmtKind::Expr(expr)))
        }
    })(input)
}

fn parse_block_stmt(input: Span) -> IResult<Span, Stmt> {
    spanned(|input| {
        // block expression used as statement (optional semicolon)
        let (input, block) = parse_block_expr(input)?;
        let (input, _) = opt(ws(char(';')))(input)?;
        Ok((input, StmtKind::Expr(block)))
    })(input)
}

fn parse_field_assign(input: Span) -> IResult<Span, Stmt> {
    spanned(|input| {
        let (input, lhs) = parse_expr(input)?;
        match lhs.inner {
            ExprKind::FieldAccess(obj, field) => {
                let (input, op_str) =
                    ws(alt((tag("="), tag("+="), tag("-="), tag("*="), tag("/="))))(input)?;

                let (input, value) = parse_expr(input)?;
                let (input, _) = ws(char(';'))(input)?;

                if *op_str == "=" {
                    Ok((
                        input,
                        StmtKind::FieldAssign {
                            obj: *obj,
                            field,
                            value,
                        },
                    ))
                } else {
                    // Desugar x.y += z to x.y.add_assign(z)
                    let method = match *op_str {
                        "+=" => "add_assign",
                        "-=" => "sub_assign",
                        "*=" => "mul_assign",
                        "/=" => "div_assign",
                        _ => unreachable!(),
                    };
                    // Need a span for the synthetic method call. Use lhs's span.
                    let span = lhs.span.clone();
                    let method_call = Spanned::new(
                        ExprKind::MethodCall(
                            Box::new(Spanned::new(
                                ExprKind::FieldAccess(obj, field),
                                span.clone(),
                            )),
                            method.to_string(),
                            vec![value],
                        ),
                        span,
                    );
                    Ok((input, StmtKind::Expr(method_call)))
                }
            }
            _ => Err(nom::Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Tag,
            ))),
        }
    })(input)
}

fn parse_use_stmt(input: Span) -> IResult<Span, Stmt> {
    spanned(|input| {
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
                StmtKind::Use {
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
                StmtKind::Use {
                    path: path_segments,
                    alias,
                    items: vec![],
                },
            ))
        }
    })(input)
}

pub fn parse_stmt(input: Span) -> IResult<Span, Stmt> {
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
fn parse_fn(input: Span) -> IResult<Span, FunctionDef> {
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

fn parse_fn_arg(input: Span) -> IResult<Span, (String, Type)> {
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
fn parse_struct(input: Span) -> IResult<Span, StructDef> {
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

fn parse_variant_def(input: Span) -> IResult<Span, VariantDef> {
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

fn parse_enum_def(input: Span) -> IResult<Span, EnumDef> {
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
fn parse_tensor_decl(input: Span) -> IResult<Span, Stmt> {
    spanned(|input| {
        // tensor name: Type = Expr;
        let (input, _) = tag("tensor")(input)?;
        let (input, name) = ws(identifier)(input)?;
        let (input, _) = ws(char(':'))(input)?;
        let (input, type_annotation) = ws(parse_type)(input)?;

        let (input, init) = opt(preceded(ws(char('=')), parse_expr))(input)?;
        let (input, _) = opt(ws(char(';')))(input)?; // Optional semicolon

        Ok((
            input,
            StmtKind::TensorDecl {
                name,
                type_annotation,
                init,
            },
        ))
    })(input)
}

fn parse_relation_decl(input: Span) -> IResult<Span, RelationDecl> {
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

fn parse_datalog_atom(input: Span) -> IResult<Span, Atom> {
    // Name(arg, ...) - parentheses required but can be empty for nilary
    let (input, predicate) = ws(identifier)(input)?;
    let (input, args) = delimited(
        ws(char('(')),
        separated_list0(ws(char(',')), parse_expr),
        ws(char(')')),
    )(input)?;
    Ok((input, Atom { predicate, args }))
}

fn parse_fact(input: Span) -> IResult<Span, Rule> {
    // Fact: Head(args). - no body, just period terminated
    let (input, head) = parse_datalog_atom(input)?;
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

fn parse_rule(input: Span) -> IResult<Span, Rule> {
    // Head(args) :- Body(args), ...
    // Link: Head(args) :- Relation1(args), Relation2(args).
    let (input, head) = parse_datalog_atom(input)?;
    let (input, _) = ws(tag(":-"))(input)?;
    let (input, body) = separated_list1(ws(char(',')), parse_datalog_atom)(input)?;
    let (input, _) = ws(alt((char('.'), char(';'))))(input)?;

    Ok((
        input,
        Rule {
            head,
            body,
            weight: None, // No syntax for weight yet in normal rules
        },
    ))
}

fn parse_query(input: Span) -> IResult<Span, Expr> {
    // ?- Expr
    let (input, _) = tag("?-")(input)?;
    let (input, expr) = parse_expr(input)?;
    let (input, _) = opt(ws(char(';')))(input)?;
    Ok((input, expr))
}

// --- Impls ---
fn parse_impl(input: Span) -> IResult<Span, ImplBlock> {
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

fn parse_mod_decl(input: Span) -> IResult<Span, String> {
    let (input, _) = tag("mod")(input)?;
    let (input, name) = ws(identifier)(input)?;
    let (input, _) = ws(char(';'))(input)?;
    Ok((input, name))
}

fn parse_use_decl(input: Span) -> IResult<Span, String> {
    let (input, _) = tag("use")(input)?;
    let (input, _) = space1(input)?;
    // Parse path: ident::ident::...
    // simpler: separated_list1
    let (input, parts) = separated_list1(ws(tag("::")), identifier)(input)?;
    let (input, _) = ws(char(';'))(input)?;
    Ok((input, parts.join("::")))
}

// --- Top Level ---
// Re-define parse_module logic
pub fn parse(input: &str) -> Result<Module, TlError> {
    let input = Span::new(input);
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
    // LocatedSpan doesn't have strip_prefix on itself directly unless deref works,
    // but we can parse it out or just check fragment
    let mut remaining = if input.fragment().starts_with('\u{feff}') {
        let (rem, _) =
            take_while::<_, _, nom::error::Error<Span>>(|c| c == '\u{feff}')(input).unwrap();
        rem
    } else {
        input
    };

    while !remaining.fragment().trim().is_empty() {
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
        } else if let Ok((next, u)) = ws(parse_use_decl)(remaining) {
            // "use" is similar to import in this simpleAST?
            // Existing AST only has `imports: Vec<String>`.
            // Let's store use decls in imports for now?
            // Or maybe separate field? Module struct definition (1493) calls it `imports`.
            // parse_mod_decl pushes to imports.
            // Let's push use decls to imports too.
            imports.push(u);
            remaining = next;
        } else if let Ok((next, t)) = ws(parse_tensor_decl)(remaining) {
            tensor_decls.push(t);
            remaining = next;
        } else if let Ok((next, s)) = ws(parse_stmt)(remaining) {
            tensor_decls.push(s); // Note: Tensor declarations can be statements too? Logic from original code preserved.
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

            // Calculate position from where we failed
            let line = remaining.location_line() as usize;
            let column = remaining.get_utf8_column();
            let span = crate::compiler::error::Span::new(line, column);

            let fragment = remaining.fragment();
            let snippet = &fragment[..fragment.len().min(30)];
            return Err(TlError::Parse {
                kind: ParseErrorKind::InvalidSyntax(format!("at: {:?}...", snippet)),
                span: Some(span),
            });
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
        let res = parse_literal_float(Span::new("1.0"));
        assert!(res.is_ok(), "Failed directly on 1.0: {:?}", res);

        // Scientific with dot
        let res = parse_literal_float(Span::new("1.0e-4"));
        assert!(res.is_ok(), "Failed directly on 1.0e-4: {:?}", res);

        let (_, expr) = res.unwrap();
        if let ExprKind::Float(val) = expr.inner {
            assert!((val - 0.0001).abs() < 1e-10);
        }

        // Scientific without dot
        let res = parse_literal_float(Span::new("1e-4"));
        assert!(res.is_ok(), "Failed directly on 1e-4: {:?}", res);

        let (_, expr) = res.unwrap();
        if let ExprKind::Float(val) = expr.inner {
            assert!((val - 0.0001).abs() < 1e-10);
        }

        // Capital E
        let res = parse_literal_float(Span::new("1E5"));
        assert!(res.is_ok());
        if let ExprKind::Float(val) = res.unwrap().1.inner {
            assert_eq!(val, 100000.0);
        }
    }
}
