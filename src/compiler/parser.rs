// src/compiler/parser.rs
use crate::compiler::ast::*;
use crate::compiler::error::ParseErrorKind;
use crate::compiler::lexer::{Token, SpannedToken}; // Import Lexer
use nom::{
    branch::alt,
    combinator::{map, opt, value, verify, eof, cut},
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult,
};

// Input is a slice of SpannedTokens
pub type Input<'a> = &'a [SpannedToken];

// Custom parsing error
// We need to map nom errors to our structures
#[derive(Debug, PartialEq)]
pub struct ParserError<'a> {
    pub input: Input<'a>,
    pub kind: ParseErrorKind,
}

impl<'a> nom::error::ParseError<Input<'a>> for ParserError<'a> {
    fn from_error_kind(input: Input<'a>, kind: nom::error::ErrorKind) -> Self {
        ParserError {
            input,
            kind: ParseErrorKind::Nom(kind),
        }
    }

    fn append(_: Input<'a>, _: nom::error::ErrorKind, other: Self) -> Self {
        other
    }
}

// Helper: Get Span from Input (current token or EOF)
fn get_span(input: Input) -> crate::compiler::error::Span {
    if let Some(tok) = input.first() {
        crate::compiler::error::Span::with_file(
            "unknown", // TODO: propagete file name using Input Context or pre-fill
            tok.span.line,
            tok.span.column,
        )
    } else {
        // EOF
        crate::compiler::error::Span::default()
    }
}

// Combinator: Spanned
pub fn spanned<'a, F, O>(mut parser: F) -> impl FnMut(Input<'a>) -> IResult<Input<'a>, Spanned<O>, ParserError<'a>>
where
    F: FnMut(Input<'a>) -> IResult<Input<'a>, O, ParserError<'a>>,
{
    move |input: Input<'a>| {
        let start_span = get_span(input);
        let (rest, value) = parser(input)?;
        // End span is tricky if rest is empty (EOF). 
        // We can approximate by taking start_span of the *consumed* tokens?
        // Or just use the start_span for now. Spanned usually wants ranges.
        // But compiler::error::Span is just line/col point? 
        // No, it's usually start point? 
        // Let's assume start point is enough for now or I need end point tracking.
        Ok((rest, Spanned::new(value, start_span)))
    }
}

// Combinator: Match specific token
pub fn expect_token(expected: Token) -> impl Fn(Input) -> IResult<Input, Input, ParserError> {
    move |input: Input| {
        match input.first() {
            Some(tok) if tok.token == expected => Ok((&input[1..], &input[0..1])),
            _ => Err(nom::Err::Error(ParserError {
                input,
                kind: ParseErrorKind::UnexpectedToken(format!("{:?}", expected)),
            })),
        }
    }
}

// Match identifier
pub fn identifier(input: Input) -> IResult<Input, String, ParserError> {
    match input.first() {
        Some(tok) => match &tok.token {
            Token::Identifier(s) => Ok((&input[1..], s.clone())),
            _ => Err(nom::Err::Error(ParserError {
                input,
                kind: ParseErrorKind::UnexpectedToken("Identifier".to_string()),
            })),
        },
        None => Err(nom::Err::Error(ParserError {
            input,
            kind: ParseErrorKind::UnexpectedToken("EOF".to_string()),
        })),
    }
}

// Match primitive identifiers if they are parsed as tokens?
// Lexer emits keywords for f32, etc. So "f32" is Token::F32Type.
// We need to match Token::F32Type -> Type::F32.

// Entry point wrapper (Temporary adapter for existing tests expecting string)
// We need to change the public API `parse` to take `&str` but use `tokenize` internally.


// Helper: Match anything that satisfies predicate
pub fn satisfy_token<F>(pred: F) -> impl Fn(Input) -> IResult<Input, SpannedToken, ParserError> 
where F: Fn(&Token) -> bool {
    move |input: Input| {
        match input.first() {
            Some(tok) if pred(&tok.token) => Ok((&input[1..], tok.clone())),
            _ => Err(nom::Err::Error(ParserError {
                input,
                kind: ParseErrorKind::UnexpectedToken("satisfy".to_string()),
            })),
        }
    }
}

// --- Types ---
fn parse_primitive_type(input: Input) -> IResult<Input, Type, ParserError> {
    alt((
        value(Type::F32, expect_token(Token::F32Type)),
        value(Type::F64, expect_token(Token::F64Type)),
        value(Type::I32, expect_token(Token::I32Type)),
        value(Type::I64, expect_token(Token::I64Type)),
        value(Type::Bool, expect_token(Token::BoolType)),
        value(Type::Usize, expect_token(Token::UsizeType)),
        value(Type::Void, expect_token(Token::VoidType)),
    ))(input)
}

fn parse_tensor_type(input: Input) -> IResult<Input, Type, ParserError> {
    // Tensor<type, rank>
    // Token::Identifier("Tensor") ? No, Identifier("Tensor")
    // But lexer might see Tensor as Identifier cause it's not a keyword.
    // Yes.
    
    // We match Identifier("Tensor")
    let (input, _) = satisfy_token(|t| matches!(t, Token::Identifier(s) if s == "Tensor"))(input)?;
    let (input, _) = expect_token(Token::Lt)(input)?;
    let (input, inner) = parse_type(input)?;
    let (input, _) = expect_token(Token::Comma)(input)?;
    let (input, rank) = match input.first() {
        Some(tok) => match tok.token {
            Token::IntLiteral(n) => Ok((&input[1..], n as usize)),
            _ => Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Expected Int Rank".to_string()) })),
        },
        None => Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("EOF".to_string()) })),
    }?;
    let (input, _) = expect_token(Token::Gt)(input)?;
    
    Ok((input, Type::Tensor(Box::new(inner), rank)))
}

pub fn parse_type(input: Input) -> IResult<Input, Type, ParserError> {
    alt((
        parse_primitive_type,
        parse_tensor_type,
        // Tuple type: (Type, Type, ...)
        map(
            delimited(
                expect_token(Token::LParen),
                separated_list0(expect_token(Token::Comma), parse_type),
                expect_token(Token::RParen),
            ),
            Type::Tuple,
        ),
        // User defined: identifier <generics?>
        map(
            tuple((
                identifier,
                opt(delimited(
                    expect_token(Token::Lt),
                    separated_list1(expect_token(Token::Comma), parse_type),
                    expect_token(Token::Gt),
                )),
            )),
            |(name, generics)| Type::UserDefined(name, generics.unwrap_or_default()),
        ),
    ))(input)
}

// --- Literals ---
pub fn parse_literal(input: Input) -> IResult<Input, Expr, ParserError> {
    let (input, token) = satisfy_token(|t| matches!(t, 
        Token::IntLiteral(_) | Token::FloatLiteral(_) | Token::StringLiteral(_) | Token::True | Token::False
    ))(input)?;
    
    let kind = match token.token {
        Token::IntLiteral(n) => ExprKind::Int(n),
        Token::FloatLiteral(f) => ExprKind::Float(f),
        Token::StringLiteral(s) => ExprKind::StringLiteral(s),
        Token::True => ExprKind::Bool(true),
        Token::False => ExprKind::Bool(false),
        _ => unreachable!(),
    };
    
    Ok((input, Spanned::new(kind, crate::compiler::error::Span::new(token.span.line, token.span.column))))
}





fn parse_tensor_literal(input: Input) -> IResult<Input, Expr, ParserError> {
    let (input, _) = expect_token(Token::LBracket)(input)?;
    let (input, exprs) = split_delimited(input)?;
    Ok((input, Spanned::new(ExprKind::TensorLiteral(exprs), crate::compiler::error::Span::default())))
}

fn parse_tensor_comprehension_or_literal(input: Input) -> IResult<Input, Expr, ParserError> {
    // [ ... ]
    let (rest, _) = expect_token(Token::LBracket)(input)?;
    
    // Check for comprehension: indices | ...
    // Requires lookahead or trial.
    // Try to parse indices (identifiers) then pipe.
    // But identifiers are invalid syntax for tensor literal unless they are variables?
    // Ambiguity: [ a, b ] (literal vec of vars) vs [ a, b | ... ] (indices).
    // Try parse indices + Pipe.
    
    let comp_attempt: IResult<Input, (Vec<String>, Vec<crate::compiler::ast::ComprehensionClause>, Option<Expr>), ParserError> = (move |input| {
        let (input, indices) = separated_list1(expect_token(Token::Comma), identifier)(input)?;
        let (input, _) = expect_token(Token::Pipe)(input)?;
        // Commit to comprehension
        cut(move |input| {
             // Parse clauses and body mixed?
             // List of items separated by comma.
             let (input, items) = separated_list0(expect_token(Token::Comma), |input| {
                 // Try Generator: ident <- expr
                 if let Ok((rest, name)) = identifier(input) {
                     if let Ok((rest2, _)) = expect_token(Token::LArrow)(rest) {
                         let (rest3, range) = parse_expr(rest2)?;
                         return Ok((rest3, (Some(crate::compiler::ast::ComprehensionClause::Generator{name, range}), None)));
                     }
                 }
                 // Try Expr (Condition or Body)
                 let (rest, expr) = parse_expr(input)?;
                 Ok((rest, (None, Some(expr))))
             })(input)?;
             
             // Process items
             let mut clauses = vec![];
             let mut body = None;
             
             for (clause, expr) in items {
                 if let Some(c) = clause {
                     clauses.push(c);
                 } else if let Some(e) = expr {
                     // If it is block, assume body?
                     // Or if it is the last item?
                     // For now, if we have multiple exprs, all but last are conditions?
                     // Or body is implicit?
                     // Test: { i + j }. Block.
                     match e.inner {
                         ExprKind::Block(_) => {
                             if body.is_some() {
                                 // Second block? Condition?
                                 clauses.push(crate::compiler::ast::ComprehensionClause::Condition(e));
                             } else {
                                 body = Some(e);
                             }
                         }
                         _ => {
                              clauses.push(crate::compiler::ast::ComprehensionClause::Condition(e));
                         }
                     }
                 }
             }
             
             Ok((input, (indices.clone(), clauses, body)))
        })(input)
    })(rest);

    match comp_attempt {
        Ok((rest, (indices, clauses, body_box))) => {
            let (rest, _) = expect_token(Token::RBracket)(rest)?;
             Ok((rest, Spanned::new(ExprKind::TensorComprehension { indices, clauses, body: body_box.map(Box::new) }, crate::compiler::error::Span::default())))
        }
        Err(nom::Err::Error(_)) => {
            // Not a comprehension (or syntax error before cut).
            // Try literal.
            // Rewind input to after LBracket? No, parse_tensor_literal expects LBracket.
            // Call parse_tensor_literal(input).
            parse_tensor_literal(input)
        }
        Err(e) => Err(e) // Failure
    }
}

// Helper for tensor literal content to avoid cut issues if I don't use cut yet
// Actually, standard separated_list0 is fine.
fn split_delimited(input: Input) -> IResult<Input, Vec<Expr>, ParserError> {
    let (input, exprs) = separated_list0(expect_token(Token::Comma), parse_expr)(input)?;
    let (input, _) = expect_token(Token::RBracket)(input)?;
    Ok((input, exprs))
}

// --- Expressions ---

// Forward declarations using recursive parsers
pub fn parse_expr(input: Input) -> IResult<Input, Expr, ParserError> {
    parse_binary_logic(input) // Entry point for precedence
}

fn parse_block(input: Input) -> IResult<Input, Expr, ParserError> {
    let (input, stmts) = parse_block_stmts(input)?;
    Ok((input, Spanned::new(ExprKind::Block(stmts), crate::compiler::error::Span::default())))
}



fn parse_path_or_var(input: Input) -> IResult<Input, Expr, ParserError> {
    // Try to parse as Type first?
    // A Type can be Identifier, Primitive, or Tensor<...>.
    // If it is followed by ::, it is a static path.
    // If it is followed by {, it might be StructInit (if Type is Identifier).
    
    // Attempt to parse a Type. 
    // If Type is just Identifier, it consumes Identifier.
    let (rest, ty) = parse_type(input)?;
    
    // Check next token
    if let Ok((rest2, _)) = expect_token(Token::DoubleColon)(rest) {
        // Static Method Call: Type :: Identifier (args)
        let (rest3, method) = identifier(rest2)?;
        // Optional args? StaticMethodCall always has args in AST?
        // If ( follows, parse args.
        // If no (, it might be Unit Variant path? treated as call with empty args?
        let (rest4, args) = if let Ok((r, _)) = expect_token(Token::LParen)(rest3) {
            let (r, args) = separated_list0(expect_token(Token::Comma), parse_expr)(r)?;
            let (r, _) = expect_token(Token::RParen)(r)?;
            (r, args)
        } else {
            (rest3, vec![])
        };
        
        let span = crate::compiler::error::Span::default(); // TODO: Proper spanning
        return Ok((rest4, Spanned::new(ExprKind::StaticMethodCall(ty, method, args), span)));
    } else if let Ok((rest2, _)) = expect_token(Token::LBrace)(rest) {
        // Struct Init: Type { ... }
        // Only if Type is UserDefined (Identifier). f32 { } is invalid.
        if let Type::UserDefined(name, generics) = ty {
            // Parse fields: ident : expr
            let (rest3, fields) = separated_list0(
                expect_token(Token::Comma),
                map(
                    tuple((identifier, expect_token(Token::Colon), parse_expr)),
                    |(id, _, e)| (id, e)
                )
            )(rest2)?;
            // Trailing comma allowed? separated_list0 doesn't handle trailing.
            // Ignore trailing comma for now (or handle it).
            let (rest3, _) = opt(expect_token(Token::Comma))(rest3)?;
            let (rest4, _) = expect_token(Token::RBrace)(rest3)?;
            
            let span = crate::compiler::error::Span::default();
            return Ok((rest4, Spanned::new(ExprKind::StructInit(name, generics, fields), span)));
        }
    }
    
    // Fallback: If it was just a Type which happened to be an Identifier, behave as Variable.
    // If Type was Primitive (f32) and not followed by ::, it's an error in expression context?
    // Or is `let x = f32;` valid? No.
    if let Type::UserDefined(name, generics) = ty {
        if generics.is_empty() {
            let span = crate::compiler::error::Span::default();
            return Ok((rest, Spanned::new(ExprKind::Variable(name), span)));
        }
        // Identifier<T> is not a value unless it's a constructor? 
        // But StructInit handled above.
        // Maybe generic function call? func::<T>()?
        // Parser for fn call is postfix.
        // Getting here means `name<T>`.
        // This implies variable with generic args? `var<T>` isn't valid syntax usually.
        // `func::<T>` uses turbofish.
        // So `parse_type` was too eager parsing `<...>`?
        // `parse_type` parses `Ident < T >`.
        // In expression `a < b`, `parse_type` might match `a < b` if `b` is type.
        // This is the classic matching ambiguity.
        // We should restrict `parse_type` usage here or use `parse_atom` for Variable specifically.
    }
    
    // If we parsed `f32`, and it's not a static call, it's error.
    Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Invalid path/type in expression".to_string()) }))
}

fn parse_path_based_atom(input: Input) -> IResult<Input, Expr, ParserError> {
    // Tries to parse Type followed by :: or {
    // This handles StaticMethodCall and StructInit
    let (rest, ty) = parse_type(input)?;
    
    if let Ok((rest2, _)) = expect_token(Token::DoubleColon)(rest) {
        // Static Call: Type :: Method (args)
        let (rest3, method) = identifier(rest2)?;
        // Method args
        let (rest4, args) = if let Ok((r, _)) = expect_token(Token::LParen)(rest3) {
            let (r, args) = separated_list0(expect_token(Token::Comma), parse_expr)(r)?;
            let (r, _) = expect_token(Token::RParen)(r)?;
            (r, args)
        } else {
            (rest3, vec![])
        };
        Ok((rest4, Spanned::new(ExprKind::StaticMethodCall(ty, method, args), crate::compiler::error::Span::default())))
    } else if let Ok((rest2, _)) = expect_token(Token::LBrace)(rest) {
        // Struct Init: Ident { field: val, ... }
        // Type parses "Ident < T >".
        if let Type::UserDefined(name, generics) = ty {
             let (rest3, fields) = separated_list0(
                expect_token(Token::Comma),
                map(
                    tuple((identifier, expect_token(Token::Colon), parse_expr)),
                    |(id, _, e)| (id, e)
                )
            )(rest2)?;
            let (rest3, _) = opt(expect_token(Token::Comma))(rest3)?;
            let (rest4, _) = expect_token(Token::RBrace)(rest3)?;
            Ok((rest4, Spanned::new(ExprKind::StructInit(name, generics, fields), crate::compiler::error::Span::default())))
        } else {
            Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Struct init requires UserDefined type".to_string()) }))
        }
    } else {
        // Parsed type but no :: or {.
        // This is not a path-based atom (unless we support Type as expression?).
        // Fail so we fallback to Variable parser.
        Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Not a path atom".to_string()) }))
    }
}

fn parse_variable(input: Input) -> IResult<Input, Expr, ParserError> {
    let (input, name) = identifier(input)?;
    Ok((input, Spanned::new(ExprKind::Variable(name), crate::compiler::error::Span::default())))
}

fn parse_atom(input: Input) -> IResult<Input, Expr, ParserError> {
    alt((
        parse_literal,
        parse_tensor_comprehension_or_literal,
        parse_block,
        parse_if_expr,
        map(
            delimited(
                expect_token(Token::LParen),
                separated_list0(expect_token(Token::Comma), parse_expr),
                expect_token(Token::RParen),
            ),
            |exprs| {
                if exprs.len() == 1 {
                    exprs.into_iter().next().unwrap()
                } else {
                    Spanned::new(ExprKind::Tuple(exprs), crate::compiler::error::Span::default())
                }
            }
        ),
        parse_path_based_atom, // Try Type::... or Type { ... } first
        parse_variable,        // Fallback to simple variable
    ))(input)
}

fn parse_postfix(input: Input) -> IResult<Input, Expr, ParserError> {
    let (mut input, mut expr) = parse_atom(input)?;
    
    // Loop to consume chain of postfixes (call, index, field)
    loop {
        // Check for ( (Call)
        if let Ok((rest, _)) = expect_token(Token::LParen)(input) {
            let (rest, args) = separated_list0(expect_token(Token::Comma), parse_expr)(rest)?;
            let (rest, _) = expect_token(Token::RParen)(rest)?;
            
            // Convert expr to MethodCall if it was FieldAccess
            let span = crate::compiler::error::Span::default();
            match expr.inner {
                ExprKind::FieldAccess(obj, method) => {
                    expr = Spanned::new(ExprKind::MethodCall(obj, method, args), span);
                }
                ExprKind::Variable(name) => {
                    expr = Spanned::new(ExprKind::FnCall(name, args), span);
                }
                _ => {
                    // Indirect call? ExprKind::FnCall expects String name.
                    // If we support closures/fn pointers, we need IndirectCall ast?
                    // AST 0.1.8 has FnCall(String, ...). No generic expr call.
                    // Assume it's a FnCall with name "UNKNOWN"? 
                    // Or error?
                    // Current parser uses "UNKNOWN_INDIRECT_CALL".
                    expr = Spanned::new(ExprKind::FnCall("UNKNOWN_INDIRECT_CALL".to_string(), args), span);
                }
            }
            input = rest;
            continue;
        }
        
        // Check for [ (Index)
        if let Ok((rest, _)) = expect_token(Token::LBracket)(input) {
            let (rest, indices) = separated_list1(expect_token(Token::Comma), parse_expr)(rest)?;
            let (rest, _) = expect_token(Token::RBracket)(rest)?;
            let span = crate::compiler::error::Span::default();
            expr = Spanned::new(ExprKind::IndexAccess(Box::new(expr), indices), span);
            input = rest;
            continue;
        }
        
        // Check for . (Field/Method/Tuple)
        if let Ok((rest, _)) = expect_token(Token::Dot)(input) {
            // Identifier or IntLiteral (tuple)
            if let Ok((rest2, field)) = identifier(rest) {
                let span = crate::compiler::error::Span::default();
                expr = Spanned::new(ExprKind::FieldAccess(Box::new(expr), field), span);
                input = rest2;
                continue;
            } else if let Ok((rest2, _)) = satisfy_token(|t| matches!(t, Token::IntLiteral(_)))(rest) {
                // tuple access
                let idx = match rest.first().unwrap().token {
                    Token::IntLiteral(n) => n as usize,
                    _ => 0,
                };
                let span = crate::compiler::error::Span::default();
                expr = Spanned::new(ExprKind::TupleAccess(Box::new(expr), idx), span);
                input = rest2;
                continue;
            }
            // Error if dot not followed by proper token
            return Err(nom::Err::Error(ParserError { input: rest, kind: ParseErrorKind::UnexpectedToken("Expected field or index".to_string())}));
        }
        
        break;
    }
    
    Ok((input, expr))
}

// Cast: expr as Type
fn parse_cast(input: Input) -> IResult<Input, Expr, ParserError> {
    let (mut input, mut expr) = parse_postfix(input)?;
    loop {
        if let Ok((rest, _)) = expect_token(Token::As)(input) {
            let (rest, ty) = parse_type(rest)?;
            let span = crate::compiler::error::Span::default();
            expr = Spanned::new(ExprKind::As(Box::new(expr), ty), span);
            input = rest;
            continue;
        }
        break;
    }
    Ok((input, expr))
}

fn parse_unary(input: Input) -> IResult<Input, Expr, ParserError> {
    if let Ok((rest, op_tok)) = satisfy_token(|t| matches!(t, Token::Minus | Token::Not))(input) {
        let op = match op_tok.token {
            Token::Minus => UnOp::Neg,
            Token::Not => UnOp::Not,
            _ => unreachable!(),
        };
        let (rest, expr) = parse_unary(rest)?;
        Ok((rest, Spanned::new(ExprKind::UnOp(op, Box::new(expr)), crate::compiler::error::Span::default())))
    } else {
        parse_cast(input)
    }
}

// Binary: simple version without precedence table for now (or use Pratt/climbing?)
// Legacy parser handled precedence manually via function chain.
// term -> factor (* / %) -> ...
// I will implement a simplified chain for now or full chain.
// Chain: logical_or -> logical_and -> comparison -> term (+ -) -> factor (* / %) -> unary
// This corresponds to standard precedence.

// Factor: * / %
fn parse_factor(input: Input) -> IResult<Input, Expr, ParserError> {
    let (mut input, mut lhs) = parse_unary(input)?;
    loop {
        if let Ok((rest, op_tok)) = satisfy_token(|t| matches!(t, Token::Star | Token::Slash | Token::Percent))(input) {
            let op = match op_tok.token {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::Percent => BinOp::Mod,
                _ => unreachable!(),
            };
            let (rest, rhs) = parse_unary(rest)?;
            lhs = Spanned::new(ExprKind::BinOp(Box::new(lhs), op, Box::new(rhs)), crate::compiler::error::Span::default());
            input = rest;
        } else {
            break;
        }
    }
    Ok((input, lhs))
}

// Term: + -
fn parse_term(input: Input) -> IResult<Input, Expr, ParserError> {
    let (mut input, mut lhs) = parse_factor(input)?;
    loop {
        if let Ok((rest, op_tok)) = satisfy_token(|t| matches!(t, Token::Plus | Token::Minus))(input) {
            let op = match op_tok.token {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => unreachable!(),
            };
            let (rest, rhs) = parse_factor(rest)?;
            lhs = Spanned::new(ExprKind::BinOp(Box::new(lhs), op, Box::new(rhs)), crate::compiler::error::Span::default());
            input = rest;
        } else {
            break;
        }
    }
    Ok((input, lhs))
}

// Comparison: < > <= >= == !=
fn parse_comparison(input: Input) -> IResult<Input, Expr, ParserError> {
    let (mut input, mut lhs) = parse_term(input)?;
    loop {
        if let Ok((rest, op_tok)) = satisfy_token(|t| matches!(t, 
            Token::Lt | Token::Gt | Token::Le | Token::Ge | Token::Eq | Token::Ne
        ))(input) {
            let op = match op_tok.token {
                Token::Lt => BinOp::Lt,
                Token::Gt => BinOp::Gt,
                Token::Le => BinOp::Le,
                Token::Ge => BinOp::Ge,
                Token::Eq => BinOp::Eq,
                Token::Ne => BinOp::Neq,
                _ => unreachable!(),
            };
            let (rest, rhs) = parse_term(rest)?;
            lhs = Spanned::new(ExprKind::BinOp(Box::new(lhs), op, Box::new(rhs)), crate::compiler::error::Span::default());
            input = rest;
        } else {
            break;
        }
    }
    Ok((input, lhs))
}

// Logical And
fn parse_logical_and(input: Input) -> IResult<Input, Expr, ParserError> {
    let (mut input, mut lhs) = parse_comparison(input)?;
    loop {
        if let Ok((rest, _)) = expect_token(Token::And)(input) {
            let (rest, rhs) = parse_comparison(rest)?;
            lhs = Spanned::new(ExprKind::BinOp(Box::new(lhs), BinOp::And, Box::new(rhs)), crate::compiler::error::Span::default());
            input = rest;
        } else {
            break;
        }
    }
    Ok((input, lhs))
}

// Logical Or
fn parse_logical_or(input: Input) -> IResult<Input, Expr, ParserError> {
    let (mut input, mut lhs) = parse_logical_and(input)?;
    loop {
        if let Ok((rest, _)) = expect_token(Token::Or)(input) {
            let (rest, rhs) = parse_logical_and(rest)?;
            lhs = Spanned::new(ExprKind::BinOp(Box::new(lhs), BinOp::Or, Box::new(rhs)), crate::compiler::error::Span::default());
            input = rest;
        } else {
            break;
        }
    }
    Ok((input, lhs))
}

// Entry for binary ops
fn parse_binary_logic(input: Input) -> IResult<Input, Expr, ParserError> {
    // Range? a..b
    // Range has lower precedence than logic? or higher?
    // Rust: Range has lower precedence than arithmetic/logic usually?
    // Range 0..10.
    // Legacy parser had parse_range wrapping logical_or.
    let (input, start) = parse_logical_or(input)?;
    if let Ok((rest, _)) = expect_token(Token::Range)(input) {
        let (rest, end) = parse_logical_or(rest)?;
        Ok((rest, Spanned::new(ExprKind::Range(Box::new(start), Box::new(end)), crate::compiler::error::Span::default())))
    } else {
        Ok((input, start))
    }
}

// --- Statements ---

fn parse_let_stmt(input: Input) -> IResult<Input, Stmt, ParserError> {
    let (input, _) = expect_token(Token::Let)(input)?;
    let (input, is_mut) = opt(expect_token(Token::Mut))(input)?;
    let (input, name) = identifier(input)?;
    
    // Type annotation? : Type
    let (input, ty) = opt(preceded(expect_token(Token::Colon), parse_type))(input)?;
    
    let (input, _) = expect_token(Token::Assign)(input)?;
    let (input, value) = parse_expr(input)?;
    let (input, _) = expect_token(Token::SemiColon)(input)?;
    
    Ok((input, Spanned::new(StmtKind::Let {
        name,
        type_annotation: ty,
        value,
        mutable: is_mut.is_some(),
    }, crate::compiler::error::Span::default())))
}

fn parse_return_stmt(input: Input) -> IResult<Input, Stmt, ParserError> {
    let (input, _) = expect_token(Token::Return)(input)?;
    let (input, value) = opt(parse_expr)(input)?;
    let (input, _) = expect_token(Token::SemiColon)(input)?;
    Ok((input, Spanned::new(StmtKind::Return(value), crate::compiler::error::Span::default())))
}



fn parse_assign_stmt(input: Input) -> IResult<Input, Stmt, ParserError> {
    // Identifier op expr ;
    // But LHS can be field access or index.
    // This overlaps with ExprStmt parsing.
    // parse_expr will parse "x[i]".
    // Then we see "=".
    // So usually we parse expr, then check if followed by =?
    // Or we use separate parser if we can distinguish.
    // Let's try to parse LHS expr (limited?), then =, then RHS.
    // "x = 1;"
    // parse_expr parses "x".
    // "x + 1 = 2" is invalid LHS.
    // For now, let's match identifier-based assignment specifically?
    // "x = ..."
    // Legacy parser had specific logic.
    // Let's peek?
    // If we have Ident/Postfix then =, it's assignment.
    // If we interpret as Expr, we consume x.
    // Then we see =. parse_expr_stmt expects ;. Fails.
    // So parse_assign_stmt must be tried BEFORE parse_expr_stmt.
    
    let (rest, lhs) = parse_postfix(input)?;
    // Check for assign op
    if let Ok((rest2, op_tok)) = satisfy_token(|t| matches!(t, Token::Assign | Token::PlusAssign | Token::MinusAssign | Token::StarAssign | Token::SlashAssign))(rest) {
        let op = match op_tok.token {
            Token::Assign => AssignOp::Assign,
            Token::PlusAssign => AssignOp::AddAssign,
            Token::MinusAssign => AssignOp::SubAssign,
            Token::StarAssign => AssignOp::MulAssign,
            Token::SlashAssign => AssignOp::DivAssign,
            _ => unreachable!(),
        };
        let (rest3, val) = parse_expr(rest2)?;
        let (rest4, _) = expect_token(Token::SemiColon)(rest3)?;
        
        // Extract name from LHS?
        // Legacy StmtKind::Assign stores `name: String` which is limited to variables?
        // Or `lhs: Expr`?
        // StmtKind::Assign { name: String, ... }.
        // If LHS is `x.y`, we need `StmtKind::FieldAssign`?
        // Legacy had `parse_field_assign`.
        
        match lhs.inner {
            ExprKind::Variable(name) => {
                Ok((rest4, Spanned::new(StmtKind::Assign { name, value: val, op, indices: None }, crate::compiler::error::Span::default())))
            }
            ExprKind::FieldAccess(obj, field) => {
                // If op is Assign.
                if op == AssignOp::Assign {
                    Ok((rest4, Spanned::new(StmtKind::FieldAssign { obj: *obj, field, value: val, op: AssignOp::Assign }, crate::compiler::error::Span::default())))
                } else {
                    // Complex assign on field not supported in legacy?
                    Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Compound align on field not supported yet".to_string()) }))
                }
            }
            _ => Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Invalid LHS for assignment".to_string()) }))
        }
    } else {
        // Not an assignment, fail so alt can try expr_stmt
        Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Not assignment".to_string()) }))
    }
}

// --- Compound Statements & Control Flow ---

fn parse_block_stmts(input: Input) -> IResult<Input, Vec<Stmt>, ParserError> {
    let (input, _) = expect_token(Token::LBrace)(input)?;
    let mut input = input;
    let mut stmts = vec![];
    
    loop {
        if let Ok((rest, _)) = expect_token(Token::RBrace)(input) {
            input = rest;
            break;
        }

        // Try standard Stmt
        match parse_stmt(input) {
            Ok((rest, stmt)) => {
                stmts.push(stmt);
                input = rest;
                continue;
            }
            Err(e) => {
                // If Error, check if it's a trailing expression (missing semi)
                // Try parsing expr
                let (rest, expr) = match parse_expr(input) {
                    Ok(res) => res,
                    Err(_) => return Err(e), // Return original error if not expr
                };
                
                // Must be followed by RBrace to be valid trailing expr
                if let Ok((rest2, _)) = expect_token(Token::RBrace)(rest) {
                    stmts.push(Spanned::new(StmtKind::Expr(expr), crate::compiler::error::Span::default()));
                    input = rest2;
                    break;
                } else {
                    // Expr parsed, but not followed by } or ;
                    // This implies "Expected ;"
                    return Err(nom::Err::Error(ParserError { input: rest, kind: ParseErrorKind::UnexpectedToken("Expected ; or }".to_string()) }));
                }
            }
        }
    }
    Ok((input, stmts))
}

fn parse_if_expr(input: Input) -> IResult<Input, Expr, ParserError> {
    let (input, _) = expect_token(Token::If)(input)?;
    let (input, cond) = parse_expr(input)?;
    let (input, then_block) = parse_block_stmts(input)?;
    let (input, else_block_opt) = opt(preceded(
        expect_token(Token::Else),
        alt((
            map(parse_block_stmts, |s| Some(s)),
            map(parse_if_expr, |e| {
                // else if ... -> Treat as else { if ... }
                // AST expectation: else_block is Option<Vec<Stmt>>.
                // Wrap the IfExpr in a Stmt::Expr.
                Some(vec![Spanned::new(StmtKind::Expr(e), crate::compiler::error::Span::default())])
            }),
        ))
    ))(input)?;
    
    let else_block = else_block_opt.flatten();

    Ok((input, Spanned::new(ExprKind::IfExpr(Box::new(cond), then_block, else_block), crate::compiler::error::Span::default())))
}

fn parse_while_stmt(input: Input) -> IResult<Input, Stmt, ParserError> {
    let (input, _) = expect_token(Token::While)(input)?;
    let (input, cond) = parse_expr(input)?;
    let (input, body) = parse_block_stmts(input)?;
    Ok((input, Spanned::new(StmtKind::While { cond, body }, crate::compiler::error::Span::default())))
}

fn parse_loop_stmt(input: Input) -> IResult<Input, Stmt, ParserError> {
    let (input, _) = expect_token(Token::Loop)(input)?;
    let (input, body) = parse_block_stmts(input)?;
    Ok((input, Spanned::new(StmtKind::Loop { body }, crate::compiler::error::Span::default())))
}

fn parse_for_stmt(input: Input) -> IResult<Input, Stmt, ParserError> {
    let (input, _) = expect_token(Token::For)(input)?;
    let (input, loop_var) = identifier(input)?;
    let (input, _) = expect_token(Token::In)(input)?;
    let (input, iterator) = parse_expr(input)?;
    let (input, body) = parse_block_stmts(input)?;
    Ok((input, Spanned::new(StmtKind::For { loop_var, iterator, body }, crate::compiler::error::Span::default())))
}

fn parse_expr_stmt(input: Input) -> IResult<Input, Stmt, ParserError> {
    let (input, expr) = parse_expr(input)?;
    
    let is_block_like = match expr.inner {
         ExprKind::IfExpr(..) | ExprKind::Block(..) | ExprKind::Match{..} => true,
         _ => false,
    };
    
    if is_block_like {
        let (input, _) = opt(expect_token(Token::SemiColon))(input)?;
        Ok((input, Spanned::new(StmtKind::Expr(expr), crate::compiler::error::Span::default())))
    } else {
        let (input, _) = expect_token(Token::SemiColon)(input)?;
        Ok((input, Spanned::new(StmtKind::Expr(expr), crate::compiler::error::Span::default())))
    }
}

pub fn parse_stmt(input: Input) -> IResult<Input, Stmt, ParserError> {
    alt((
        parse_let_stmt,
        parse_return_stmt,
        parse_while_stmt,
        parse_for_stmt,
        parse_loop_stmt,
        parse_assign_stmt, // Tries to match assignment. If fails (not assignment), falls back.
        parse_expr_stmt,
    ))(input)
}

// --- Top Level Items ---

fn parse_generic_params(input: Input) -> IResult<Input, Vec<String>, ParserError> {
    if let Ok((rest, _)) = expect_token(Token::Lt)(input) {
        let (rest, params) = separated_list1(expect_token(Token::Comma), identifier)(rest)?;
        let (rest, _) = expect_token(Token::Gt)(rest)?;
        Ok((rest, params))
    } else {
        Ok((input, vec![]))
    }
}

fn parse_function_def(input: Input) -> IResult<Input, crate::compiler::ast::FunctionDef, ParserError> {
    // [extern] fn name<T>(args) [: Ret] { body }
    let (input, is_extern) = map(opt(expect_token(Token::Extern)), |o| o.is_some())(input)?;
    let (input, _) = expect_token(Token::Fn)(input)?;
    
    // Commit
    cut(move |input| {
        let (input, name) = identifier(input)?;
        let (input, generics) = parse_generic_params(input)?;
        
        let (input, _) = expect_token(Token::LParen)(input)?;
        let (input, args) = separated_list0(
            expect_token(Token::Comma),
            pair(identifier, preceded(expect_token(Token::Colon), parse_type))
        )(input)?;
        let (input, _) = expect_token(Token::RParen)(input)?;
        
        let (input, return_type) = if let Ok((rest, _)) = expect_token(Token::Arrow)(input) {
            parse_type(rest)?
        } else {
            (input, Type::Void)
        };

        let (input, body) = if is_extern {
            let (input, _) = expect_token(Token::SemiColon)(input)?;
            (input, vec![])
        } else {
            parse_block_stmts(input)?
        };

        Ok((input, crate::compiler::ast::FunctionDef {
            name,
            args,
            return_type,
            body,
            generics,
            is_extern,
        }))
    })(input)
}

fn parse_struct_def(input: Input) -> IResult<Input, crate::compiler::ast::StructDef, ParserError> {
    let (input, _) = expect_token(Token::Struct)(input)?;
    cut(|input| {
        let (input, name) = identifier(input)?;
        let (input, generics) = parse_generic_params(input)?;
        
        let (input, _) = expect_token(Token::LBrace)(input)?;
        let (input, fields) = separated_list0(
            expect_token(Token::Comma),
            pair(identifier, preceded(expect_token(Token::Colon), parse_type))
        )(input)?;
        let (input, _) = opt(expect_token(Token::Comma))(input)?;
        let (input, _) = expect_token(Token::RBrace)(input)?;

        Ok((input, crate::compiler::ast::StructDef {
            name,
            fields,
            generics,
        }))
    })(input)
}

fn parse_impl_block(input: Input) -> IResult<Input, crate::compiler::ast::ImplBlock, ParserError> {
    let (input, _) = expect_token(Token::Impl)(input)?;
    cut(|input| {
        let (input, generics) = parse_generic_params(input)?;
        let (input, target_type) = parse_type(input)?;
        
        let (input, _) = expect_token(Token::LBrace)(input)?;
        let (input, methods) = many0(parse_function_def)(input)?;
        let (input, _) = expect_token(Token::RBrace)(input)?;

        Ok((input, crate::compiler::ast::ImplBlock {
            target_type,
            generics,
            methods,
        }))
    })(input)
}

// Placeholder for enums
fn parse_enum_def(input: Input) -> IResult<Input, crate::compiler::ast::EnumDef, ParserError> {
     // enum Name<T> { Variant, Variant(Type), ... }
     // Implement later or stub.
     Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Enum parsing not implemented".to_string()) }))
}

// Module items: struct, impl, fn, ...
fn parse_item(input: Input) -> IResult<Input, (), ParserError> {
    // Only used to confirm item parsing logic? 
    // parse_module loops manually.
    Ok((input, ()))
}

fn parse_use_decl(input: Input) -> IResult<Input, String, ParserError> {
    let (input, _) = expect_token(Token::Use)(input)?;
    let (input, path) = separated_list1(expect_token(Token::DoubleColon), identifier)(input)?;
    let (input, _) = expect_token(Token::SemiColon)(input)?;
    Ok((input, path.join("::")))
}

fn parse_mod_decl(input: Input) -> IResult<Input, String, ParserError> {
    let (input, _) = expect_token(Token::Mod)(input)?;
    let (input, name) = identifier(input)?;
    let (input, _) = expect_token(Token::SemiColon)(input)?;
    Ok((input, name))
}

fn parse_tensor_decl(input: Input) -> IResult<Input, Stmt, ParserError> {
    // tensor Name: Type [= Init];
    let (input, _) = satisfy_token(|t| match t {
        Token::Identifier(s) if s == "tensor" => true,
        _ => false,
    })(input)?;
    
    let (input, name) = identifier(input)?;
    let (input, _) = expect_token(Token::Colon)(input)?;
    let (input, ty) = parse_type(input)?;
    let (input, init) = opt(preceded(expect_token(Token::Assign), parse_expr))(input)?;
    let (input, _) = expect_token(Token::SemiColon)(input)?;
    
    Ok((input, Spanned::new(StmtKind::TensorDecl { name, type_annotation: ty, init }, crate::compiler::error::Span::default())))
}

fn parse_module(input: Input) -> IResult<Input, crate::compiler::ast::Module, ParserError> {
    let mut input = input;
    let mut module = crate::compiler::ast::Module {
        structs: vec![],
        enums: vec![],
        impls: vec![],
        functions: vec![],
        tensor_decls: vec![],
        relations: vec![],
        rules: vec![],
        queries: vec![],
        imports: vec![],
        submodules: std::collections::HashMap::new(),
    };

    loop {
        if input.is_empty() {
            break;
        }

        // Try helpers
        match parse_function_def(input) {
            Ok((rest, f)) => { module.functions.push(f); input = rest; continue; }
            Err(nom::Err::Failure(e)) => return Err(nom::Err::Failure(e)),
            Err(nom::Err::Error(_)) | Err(nom::Err::Incomplete(_)) => {}
        }
        match parse_struct_def(input) {
            Ok((rest, s)) => { module.structs.push(s); input = rest; continue; }
            Err(nom::Err::Failure(e)) => return Err(nom::Err::Failure(e)),
            Err(nom::Err::Error(_)) | Err(nom::Err::Incomplete(_)) => {}
        }
        match parse_impl_block(input) {
            Ok((rest, i)) => { module.impls.push(i); input = rest; continue; }
            Err(nom::Err::Failure(e)) => return Err(nom::Err::Failure(e)),
            Err(nom::Err::Error(_)) | Err(nom::Err::Incomplete(_)) => {}
        }
        match parse_use_decl(input) {
            Ok((rest, u)) => { module.imports.push(u); input = rest; continue; }
            Err(nom::Err::Failure(e)) => return Err(nom::Err::Failure(e)),
            Err(nom::Err::Error(_)) | Err(nom::Err::Incomplete(_)) => {}
        }
        match parse_mod_decl(input) {
            Ok((rest, m)) => { 
                module.submodules.insert(m, crate::compiler::ast::Module {
                    structs: vec![],
                    enums: vec![],
                    impls: vec![],
                    functions: vec![],
                    tensor_decls: vec![],
                    relations: vec![],
                    rules: vec![],
                    queries: vec![],
                    imports: vec![],
                    submodules: std::collections::HashMap::new(),
                });
                input = rest; 
                continue; 
            }
            Err(nom::Err::Failure(e)) => return Err(nom::Err::Failure(e)),
            Err(nom::Err::Error(_)) | Err(nom::Err::Incomplete(_)) => {}
        }
        match parse_tensor_decl(input) {
            Ok((rest, t)) => { module.tensor_decls.push(t); input = rest; continue; }
            Err(nom::Err::Failure(e)) => return Err(nom::Err::Failure(e)),
            Err(nom::Err::Error(_)) | Err(nom::Err::Incomplete(_)) => {}
        }

        // Statements (for scripts)
        // Try pare_stmt
        if let Ok((rest, stmt)) = parse_stmt(input) {
            match stmt.inner {
                StmtKind::TensorDecl{..} => module.tensor_decls.push(stmt),
                _ => {
                   // Ignore or error?
                   // If we are in parse_module, usually we expect items.
                   // But scripts allow stmts.
                   // Given AST limitations, we might skip generic stmts here or push to tensor_decls if they are conceptually top level "script actions"?
                   // No, tensor_decls is typed Vec<Stmt>.
                   // If I push "Expr(fn_call)" to tensor_decls, it matches the type (Stmt).
                   // Maybe `tensor_decls` is actually "top level statements"? 
                   // Let's assume yes and push ALL top level statements to `tensor_decls` as a hack/fallback?
                   // Or just Expr/Assign/etc.
                   // Let's prevent infinite loop if parse_stmt consumes distinct tokens.
                   module.tensor_decls.push(stmt);
                }
            }
            input = rest;
            continue;
        }

        return Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Unknown top level item".to_string()) }));
    }
    
    Ok((input, module))
}

pub fn parse(input: Input) -> Result<crate::compiler::ast::Module, ParserError> {
     let (rest, module) = parse_module(input).map_err(|e| match e {
         nom::Err::Error(e) | nom::Err::Failure(e) => e,
         nom::Err::Incomplete(_) => ParserError { input: &[], kind: ParseErrorKind::Generic("Incomplete input".to_string()) },
     })?;
     
     if !rest.is_empty() {
         return Err(ParserError { input: rest, kind: ParseErrorKind::Generic("Trailing tokens".to_string()) });
     }
     Ok(module)
}

pub fn parse_from_source(source: &str) -> Result<crate::compiler::ast::Module, crate::compiler::error::TlError> {
    use crate::compiler::lexer::tokenize;
    use crate::compiler::error::{TlError, ParseErrorKind};

    let tokens_res = tokenize(source);
    let mut tokens = Vec::new();
    for res in tokens_res {
        match res {
            Ok(t) => tokens.push(t),
            Err(e) => return Err(TlError::Parse { kind: ParseErrorKind::Generic(format!("Lexical error: {}", e)), span: None }),
        }
    }
    
    match parse(&tokens) {
        Ok(m) => Ok(m),
        Err(e) => {
             let span = e.input.first().map(|t| {
                 crate::compiler::error::Span {
                     file: None,
                     line: t.span.line,
                     column: t.span.column,
                 }
             });
             Err(TlError::Parse { kind: e.kind, span })
        }
    }
}
