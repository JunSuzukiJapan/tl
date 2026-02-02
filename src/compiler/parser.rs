// src/compiler/parser.rs
use crate::compiler::ast::*;
use crate::compiler::error::ParseErrorKind;
use crate::compiler::lexer::{Token, SpannedToken}; // Import Lexer
use nom::{
    branch::alt,
    combinator::{map, opt, value, cut},
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, tuple},
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
// Helper for path segments (Identifier or Type keywords used as namespace)
fn identifier_or_type_token(input: Input) -> IResult<Input, String, ParserError> {
    match input.first() {
        Some(tok) => match &tok.token {
            Token::Identifier(s) => Ok((&input[1..], s.clone())),
            Token::StringType => Ok((&input[1..], "String".to_string())),
            Token::CharType => Ok((&input[1..], "Char".to_string())),
            Token::F32Type => Ok((&input[1..], "f32".to_string())),
            Token::F64Type => Ok((&input[1..], "f64".to_string())),
            Token::I32Type => Ok((&input[1..], "i32".to_string())),
            Token::I64Type => Ok((&input[1..], "i64".to_string())),
            Token::BoolType => Ok((&input[1..], "bool".to_string())),
            Token::VoidType => Ok((&input[1..], "void".to_string())), // Unlikely but consistent
            _ => Err(nom::Err::Error(ParserError {
                input,
                kind: ParseErrorKind::UnexpectedToken("Identifier or Type".to_string()),
            })),
        },
        None => Err(nom::Err::Error(ParserError {
            input,
            kind: ParseErrorKind::UnexpectedToken("EOF".to_string()),
        })),
    }
}
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
        value(Type::String("String".to_string()), expect_token(Token::StringType)),
        value(Type::Char("Char".to_string()), expect_token(Token::CharType)),
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
        // Reference type: &Type
        map(
            preceded(expect_token(Token::Ampersand), parse_type),
            |t| Type::Ref(Box::new(t))
        ),
        // Tuple type: (Type, Type, ...)
        map(
            delimited(
                expect_token(Token::LParen),
                separated_list0(expect_token(Token::Comma), parse_type),
                expect_token(Token::RParen),
            ),
            Type::Tuple,
        ),
            // User defined: identifier<generics?> -> Now Struct
        map(
            tuple((
                identifier,
                opt(delimited(
                    expect_token(Token::Lt),
                    separated_list1(expect_token(Token::Comma), parse_type),
                    expect_token(Token::Gt),
                )),
            )),
            |(name, generics)| Type::Struct(name, generics.unwrap_or_default()),
        ),
    ))(input)
}


// --- Literals ---
pub fn parse_literal(input: Input) -> IResult<Input, Expr, ParserError> {
    let (input, token) = satisfy_token(|t| matches!(t, 
        Token::IntLiteral(_) | Token::FloatLiteral(_) | Token::StringLiteral(_) | Token::CharLiteral(_) | Token::True | Token::False
    ))(input)?;
    
    let kind = match token.token {
        Token::IntLiteral(n) => ExprKind::Int(n),
        Token::FloatLiteral(f) => ExprKind::Float(f),
        Token::StringLiteral(s) => ExprKind::StringLiteral(s),
        Token::CharLiteral(c) => ExprKind::CharLiteral(c),
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

             // Check for optional comma
             let (input, _) = opt(expect_token(Token::Comma))(input)?;

             // 4. Check for trailing block (body) { ... }
             // This supports syntax like [ ... | cond { body } ] without comma
             let (input, items) = if let Ok((_, _)) = expect_token(Token::LBrace)(input) {
                  // Found braces, parse as block expr
                  let (rest, body_expr) = parse_block(input)?;
                  let mut new_items = items;
                  new_items.push((None, Some(body_expr)));
                  (rest, new_items)
             } else {
                  (input, items)
             };
             
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





fn parse_path_based_atom(input: Input) -> IResult<Input, Expr, ParserError> {
    // Parse a path: identifier<generics>? (:: identifier<generics>?)*
    // Then determine what it is based on what follows:
    // - :: method ( ... )  -> Static method call on preceding type
    // - { ... }            -> Struct/Enum init
    // - other              -> Error (not a valid path-based atom)
    
    // Parse first identifier (or typetoken)
    let (mut rest, first) = identifier_or_type_token(input)?;
    let mut path_segments = vec![first.clone()];
    let mut generics: Vec<Type> = vec![];
    
    // Check for generics after first identifier: Type<T, U>
    if let Ok((rest2, _)) = expect_token(Token::Lt)(rest) {
        // Parse generics
        let (rest3, generic_args) = separated_list1(expect_token(Token::Comma), parse_type)(rest2)?;
        let (rest4, _) = expect_token(Token::Gt)(rest3)?;
        generics = generic_args;
        rest = rest4;
    }
    
    // Collect :: identifier<generics>? segments
    loop {
        if let Ok((rest2, _)) = expect_token(Token::DoubleColon)(rest) {
            if let Ok((rest3, seg)) = identifier_or_type_token(rest2) {
                path_segments.push(seg);
                rest = rest3;
                // Check for generics after this segment
                if let Ok((rest4, _)) = expect_token(Token::Lt)(rest) {
                    if let Ok((rest5, generic_args)) = separated_list1(expect_token(Token::Comma), parse_type)(rest4) {
                        if let Ok((rest6, _)) = expect_token(Token::Gt)(rest5) {
                            generics = generic_args;
                            rest = rest6;
                        }
                    }
                }
            } else {
                // :: but no identifier - syntax error or end of path
                break;
            }
        } else {
            break;
        }
    }
    
    // Now path_segments contains the full path
    // Determine the pattern:
    // - If followed by ( args ) and path has >= 2 segments:
    //   -> Last is method, rest is type path: mod::Type::method()
    // - If followed by ( args ) and path has 1 segment with generics:
    //   -> This is Type<T>::method() form, but we need to handle it
    // - If followed by { ... } and path has >= 1 segment:
    //   -> Struct/Enum init
    
    if let Ok((rest2, _)) = expect_token(Token::LParen)(rest) {
        // Function/Static method call
        let (rest3, args) = separated_list0(expect_token(Token::Comma), parse_expr)(rest2)?;
        let (rest4, _) = expect_token(Token::RParen)(rest3)?;
        
        if path_segments.len() >= 2 {
            // Static method call: Type::method() or mod::Type::method()
            let method = path_segments.pop().unwrap();
            let type_name = path_segments.join("::");
            let ty = Type::Struct(type_name, generics);
            return Ok((rest4, Spanned::new(ExprKind::StaticMethodCall(ty, method, args), crate::compiler::error::Span::default())));
        } else {
            // Single identifier + () - not a path-based atom, it's a fn call
            // This should be handled by parse_variable + postfix
            return Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Not a path atom (single fn call)".to_string()) }));
        }
    }
    
    if let Ok((rest2, _)) = expect_token(Token::LBrace)(rest) {
        // Struct or Enum init
        let type_name = path_segments.join("::");
        
        let (rest3, fields) = separated_list0(
            expect_token(Token::Comma),
            map(
                tuple((identifier, expect_token(Token::Colon), parse_expr)),
                |(id, _, e)| (id, e)
            )
        )(rest2)?;
        let (rest3, _) = opt(expect_token(Token::Comma))(rest3)?;
        let (rest4, _) = expect_token(Token::RBrace)(rest3)?;
        
        // Distinguish StructInit vs EnumInit (Struct Variant)
        // Syntactically identical: Type { ... } vs Enum::Variant { ... }
        // If path has >= 2 segments, likely Enum::Variant. But could be Mod::Struct.
        // Ambiguity resolution needed in semantics?
        // Current parser assumes StructInit only? No, old parser had EnumInit?
        // AST has StructInit and EnumInit.
        // Logic: Try to parse as StructInit for now. Semantics should convert if it resolves to an Enum Variant?
        // Or we assume `Enum::Variant { ... }` is EnumInit.
        // If path length > 1, let's assume it *could* be EnumInit.
        // But Struct can be Mod::Struct.
        // Let's produce StructInit, and rely on TypeRegistry/Semantics to re-classify?
        // AST Refactoring plan says "EnumInit { payload: EnumVariantInit }".
        // Let's modify `parse_path_based_atom` to check if it looks like an Enum Variant?
        // Actually, without symbol table, we can't be sure.
        // Wait, current parser ALREADY assumes StructInit for `{ ... }`.
        
        // HOWEVER, to support `Enum::Variant { ... }`, we might want to check if the LAST segment is a variant name.
        // If so, `enum_name` = rest of path.
        // But `Mod::Struct` has no variant name.
        // Existing `ExprKind::EnumInit` has `enum_name` and `variant_name`.
        // If we emit StructInit(Full::Path, ...), can we convert later?
        // YES. StructInit(Name, ...) -> Check if Name is a Struct or (Enum, Variant).
        // If Enum, convert to EnumInit.
        // So for `parser.rs`, sticking to `StructInit` for `{}` syntax is safest/simplest, UNLESS we explicitly want to support distinct EnumInit syntax now.
        // BUT, `VariantDef` changed. If we emit EnumInit, we need `EnumVariantInit::Struct(fields)`.
        
        // Let's assume we maintain `StructInit` here for now, as re-classifying requires semantic info.
        // The AST change mainly affects how we *store* it if we knew it was an Enum.
        // Wait, if `EnumInit` is used in current parser, where is it instantiated?
        // Ah, `parse_path_based_atom` does NOT instantiate EnumInit currently in the presented code (lines 337-441).它 only emits `StructInit`.
        // So `EnumInit` is currently UNUSED in parser? Or I missed it?
        // Line 272 `EnumInit` existed in AST but was it used?
        // Checking `parse_expr`: it calls `parse_path_based_atom`.
        // `parse_path_based_atom` emits `StructInit` (line 422).
        // It seems the legacy parser might have been different or I am missing something.
        // Wait, line 425 mentions "Type::Variant (tuple variant with no args)".
        // Basically, parser seems to emit StructInit or StaticMethodCall.
        
        // OK. I will leave `StructInit` as is for Parsing. The real change needed is in `parse_enum_def` and `parse_pattern`.
        // `parse_pattern` USES `EnumPattern`.
        return Ok((rest4, Spanned::new(ExprKind::StructInit(type_name, generics, fields), crate::compiler::error::Span::default())));
    }
    
    // Check for :: followed by identifier (but no () after) - this is an enum variant
    // Pattern: Type::Variant (tuple variant with no args) or Type::Variant { ... } (struct variant)
    // Actually we already consumed all :: segments above.
    // If we get here with multiple segments and no () or {}, it might be an enum tuple variant without args
    if path_segments.len() >= 2 {
        // This could be Type::Variant with no args (unit or tuple-variant with 0 args)
        let variant = path_segments.pop().unwrap();
        let type_name = path_segments.join("::");
        let ty = Type::Struct(type_name, generics);
        // Emit as static method call with empty args (for tuple variant)
        // Note: This may need adjustment based on how enum variants are handled elsewhere
        return Ok((rest, Spanned::new(ExprKind::StaticMethodCall(ty, variant, vec![]), crate::compiler::error::Span::default())));
    }
    
    // Single identifier without () or {} - not a path-based atom
    Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Not a path atom".to_string()) }))
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
        parse_block,
        parse_if_expr,
        parse_match_expr,
        parse_path_based_atom,
        map(
            delimited(
                expect_token(Token::LParen),
                separated_list0(expect_token(Token::Comma), parse_expr),
                expect_token(Token::RParen),
            ),
            |exprs| {
                if exprs.len() == 1 {
                    // (expr) -> expr (preserve span?)
                    exprs[0].clone()
                } else {
                    // Tuple
                    Spanned::new(ExprKind::Tuple(exprs), crate::compiler::error::Span::default())
                }
            }
        ),
        parse_variable,
        parse_self,
        parse_logic_var,
        parse_wildcard,
    ))(input)
}

fn parse_wildcard(input: Input) -> IResult<Input, Expr, ParserError> {
    let (input, _) = expect_token(Token::Underscore)(input)?;
    Ok((input, Spanned::new(ExprKind::Wildcard, crate::compiler::error::Span::default())))
}

fn parse_logic_var(input: Input) -> IResult<Input, Expr, ParserError> {
    let (input, _) = expect_token(Token::Dollar)(input)?;
    let (input, name) = identifier(input)?;
    Ok((input, Spanned::new(ExprKind::LogicVar(name), crate::compiler::error::Span::default())))
}

fn parse_self(input: Input) -> IResult<Input, Expr, ParserError> {
    let (input, _) = expect_token(Token::Self_)(input)?;
    Ok((input, Spanned::new(ExprKind::Variable("self".to_string()), crate::compiler::error::Span::default())))
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
    if let Ok((rest, op_tok)) = satisfy_token(|t| matches!(t, Token::Minus | Token::Not | Token::Question | Token::Ampersand))(input) {
        let op = match op_tok.token {
            Token::Minus => UnOp::Neg,
            Token::Not => UnOp::Not,
            Token::Question => UnOp::Query,
            Token::Ampersand => UnOp::Ref,
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

// Primitive Types



fn parse_let_or_tensor_decl(input: Input) -> IResult<Input, Stmt, ParserError> {
    let (input, _) = expect_token(Token::Let)(input)?;
    let (input, is_mut) = opt(expect_token(Token::Mut))(input)?;
    let (input, name) = identifier(input)?;
    
    // Check for [ (Tensor Declaration / Comprehension sugar)
    if let Ok((rest, _)) = expect_token(Token::LBracket)(input) {
         let (rest, indices) = separated_list1(expect_token(Token::Comma), identifier)(rest)?;
         let (rest, _) = expect_token(Token::RBracket)(rest)?;
         let (rest, _) = expect_token(Token::Assign)(rest)?;
         let (rest, rhs) = parse_expr(rest)?; 
         let (rest, _) = expect_token(Token::SemiColon)(rest)?;

         // Desugar to: let C = [ indices | RHS ]
         let comprehension = ExprKind::TensorComprehension {
             indices,
             clauses: vec![], // No explicit generator implies implicit range inference
             body: Some(Box::new(rhs)),
         };
         let expr = Spanned::new(comprehension, crate::compiler::error::Span::default());
         
         Ok((rest, Spanned::new(StmtKind::Let {
             name,
             type_annotation: None, 
             value: expr,
             mutable: is_mut.is_some(),
         }, crate::compiler::error::Span::default())))
    } else {
        // Normal Let
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
}


fn parse_return_stmt(input: Input) -> IResult<Input, Stmt, ParserError> {
    let (input, _) = expect_token(Token::Return)(input)?;
    let (input, value) = opt(parse_expr)(input)?;
    let (input, _) = expect_token(Token::SemiColon)(input)?;
    Ok((input, Spanned::new(StmtKind::Return(value), crate::compiler::error::Span::default())))
}



fn parse_assign_stmt(input: Input) -> IResult<Input, Stmt, ParserError> {
    // 1. Parse LHS Expr
    let (input, lhs) = parse_expr(input)?;
    
    // 2. Parse Operator
    let (input, op) = alt((
        value(AssignOp::Assign, expect_token(Token::Assign)),
        value(AssignOp::AddAssign, expect_token(Token::PlusAssign)),
        value(AssignOp::SubAssign, expect_token(Token::MinusAssign)),
        value(AssignOp::MulAssign, expect_token(Token::StarAssign)),
        value(AssignOp::DivAssign, expect_token(Token::SlashAssign)),
        value(AssignOp::ModAssign, expect_token(Token::PercentAssign)),
    ))(input)?;
    
    // 3. Parse RHS
    let (input, rhs) = parse_expr(input)?;
    let (input, _) = expect_token(Token::SemiColon)(input)?;
    
    // 4. Construct Stmt
    match lhs.inner {
        ExprKind::Variable(name) => Ok((input, Spanned::new(StmtKind::Assign {
            name,
            indices: None,
            op,
            value: rhs,
        }, crate::compiler::error::Span::default()))),
        
        ExprKind::IndexAccess(target, indices) => {
            if let ExprKind::Variable(name) = target.inner {
                Ok((input, Spanned::new(StmtKind::Assign {
                    name,
                    indices: Some(indices),
                    op,
                    value: rhs,
                }, crate::compiler::error::Span::default())))
            } else {
               Err(nom::Err::Error(ParserError { 
                   input, 
                   kind: crate::compiler::error::ParseErrorKind::Generic("Complex assignment target not supported yet".to_string()) 
               }))
            }
        },
        
        ExprKind::FieldAccess(obj, field) => {
            Ok((input, Spanned::new(StmtKind::FieldAssign {
                obj: *obj,
                field,
                op,
                value: rhs,
            }, crate::compiler::error::Span::default())))
        },
        
        _ => Err(nom::Err::Error(ParserError { 
            input, 
            kind: crate::compiler::error::ParseErrorKind::Generic("Invalid left-hand side of assignment".to_string()) 
        })),
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
    cut(|input| {
        // Check for `let` (if let)
        if let Ok((rest, _)) = expect_token(Token::Let)(input) {
             let (rest, pattern) = parse_pattern(rest)?;
             let (rest, _) = expect_token(Token::Assign)(rest)?;
             let (rest, expr) = parse_expr(rest)?;
             let (rest, then_block) = parse_block_stmts(rest)?;
             
             let (rest, else_block_opt) = opt(preceded(
                expect_token(Token::Else),
                alt((
                    map(parse_block_stmts, |s| Some(s)),
                    map(parse_if_expr, |e| {
                        Some(vec![Spanned::new(StmtKind::Expr(e), crate::compiler::error::Span::default())])
                    }),
                ))
            ))(rest)?;
            let else_block = else_block_opt.flatten();
            
            Ok((rest, Spanned::new(ExprKind::IfLet { pattern, expr: Box::new(expr), then_block, else_block }, crate::compiler::error::Span::default())))
        } else {
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
    })(input)
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

fn parse_break_stmt(input: Input) -> IResult<Input, Stmt, ParserError> {
    let (input, _) = expect_token(Token::Break)(input)?;
    let (input, _) = expect_token(Token::SemiColon)(input)?;
    Ok((input, Spanned::new(StmtKind::Break, crate::compiler::error::Span::default())))
}

fn parse_continue_stmt(input: Input) -> IResult<Input, Stmt, ParserError> {
    let (input, _) = expect_token(Token::Continue)(input)?;
    let (input, _) = expect_token(Token::SemiColon)(input)?;
    Ok((input, Spanned::new(StmtKind::Continue, crate::compiler::error::Span::default())))
}

fn parse_expr_stmt(input: Input) -> IResult<Input, Stmt, ParserError> {
    let (input, expr) = parse_expr(input)?;
    
    let is_block_like = match expr.inner {
         ExprKind::IfExpr(..) | ExprKind::Block(..) | ExprKind::Match{..} | ExprKind::IfLet{..} => true,
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
        parse_let_or_tensor_decl,
        parse_return_stmt,
        parse_while_stmt,
        parse_for_stmt,
        parse_loop_stmt,
        parse_break_stmt,
        parse_continue_stmt,
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

fn parse_match_expr(input: Input) -> IResult<Input, Expr, ParserError> {
    let (input, _) = expect_token(Token::Match)(input)?;
    let (input, target) = parse_expr(input)?;
    let (input, _) = expect_token(Token::LBrace)(input)?;
    
    let (input, arms) = separated_list0(
        expect_token(Token::Comma),
        parse_match_arm
    )(input)?;
    
    let (input, _) = opt(expect_token(Token::Comma))(input)?;
    let (input, _) = expect_token(Token::RBrace)(input)?;
    
    let span = crate::compiler::error::Span::default();
    Ok((input, Spanned::new(ExprKind::Match { expr: Box::new(target), arms }, span)))
}

fn parse_match_arm(input: Input) -> IResult<Input, (Pattern, Expr), ParserError> {
    let (input, pattern) = parse_pattern(input)?;
    let (input, _) = expect_token(Token::FatArrow)(input)?;
    let (input, expr) = parse_expr(input)?;
    Ok((input, (pattern, expr)))
}

fn parse_pattern(input: Input) -> IResult<Input, Pattern, ParserError> {
    // Wildcard
    if let Ok((rest, _)) = expect_token(Token::Underscore)(input) {
        return Ok((rest, Pattern::Wildcard));
    }
    
    // Literal
    if let Ok((rest, lit)) = parse_literal(input) {
         return Ok((rest, Pattern::Literal(Box::new(lit))));
    }
    
    // Enum Pattern (Path)
    // Tries to parse Type (Ident or Type::Ident)
    // Then check for { or (
    if let Ok((rest, ty)) = parse_type(input) {
         // Check for ::
         if let Ok((rest2, _)) = expect_token(Token::DoubleColon)(rest) {
             let (rest3, method) = identifier(rest2)?;
             // Enum Pattern: Type::Variant ...
             if let Type::Struct(name, _) = ty {
                  if let Ok((rest4, _)) = expect_token(Token::LBrace)(rest3) {
                      // Struct Pattern { field: var, ... }
                      let (rest5, bindings_vec) = separated_list0(
                         expect_token(Token::Comma),
                         |input| {
                             let (input, field) = identifier(input)?;
                             if let Ok((img, _)) = expect_token(Token::Colon)(input) {
                                 let (img, var) = identifier(img)?;
                                 Ok((img, (field, var)))
                             } else {
                                 // Shorthand { field } -> field: field
                                 Ok((input, (field.clone(), field)))
                             }
                         }
                      )(rest4)?;
                      let (rest5, _) = opt(expect_token(Token::Comma))(rest5)?;
                      let (rest5, _) = expect_token(Token::RBrace)(rest5)?;
                      
                      let bindings = crate::compiler::ast::EnumPatternBindings::Struct(bindings_vec);
                      return Ok((rest5, Pattern::EnumPattern { enum_name: name, variant_name: method, bindings }));
                  } else if let Ok((rest4, _)) = expect_token(Token::LParen)(rest3) {
                      // Tuple Pattern ( ... )
                      let (rest5, vars) = separated_list0(expect_token(Token::Comma), identifier)(rest4)?;
                      let (rest5, _) = expect_token(Token::RParen)(rest5)?;
                      
                      let bindings = crate::compiler::ast::EnumPatternBindings::Tuple(vars);
                      return Ok((rest5, Pattern::EnumPattern { enum_name: name, variant_name: method, bindings }));
                  } else {
                      // Unit Variant
                      let bindings = crate::compiler::ast::EnumPatternBindings::Unit;
                      return Ok((rest3, Pattern::EnumPattern { enum_name: name, variant_name: method, bindings }));
                  }
             }
         } else {
             // Just Type (Identifier). `None`. `Some`.
             // Treat as EnumPattern with empty enum_name?
             if let Type::Struct(name, _) = ty {
                  // Check for { or (
                  if let Ok((rest2, _)) = expect_token(Token::LBrace)(rest) {
                      let (rest3, bindings_vec) = separated_list0(
                         expect_token(Token::Comma),
                         |input| {
                             let (input, field) = identifier(input)?;
                             if let Ok((img, _)) = expect_token(Token::Colon)(input) {
                                 let (img, var) = identifier(img)?;
                                 Ok((img, (field, var)))
                             } else {
                                 Ok((input, (field.clone(), field)))
                             }
                         }
                      )(rest2)?;
                       let (rest3, _) = opt(expect_token(Token::Comma))(rest3)?;
                       let (rest3, _) = expect_token(Token::RBrace)(rest3)?;
                       let bindings = crate::compiler::ast::EnumPatternBindings::Struct(bindings_vec);
                       // We don't know the enum name. Use empty? Or assume variant name IS the type name (for None)?
                       // If checking for `None`, usually `Option::None`. `None` implies it's imported.
                       // For now, assume variant_name = name, enum_name = "".
                       return Ok((rest3, Pattern::EnumPattern { enum_name: String::new(), variant_name: name, bindings }));
                  } else if let Ok((rest2, _)) = expect_token(Token::LParen)(rest) {
                      let (rest3, vars) = separated_list0(expect_token(Token::Comma), identifier)(rest2)?;
                      let (rest3, _) = expect_token(Token::RParen)(rest3)?;
                      let bindings = crate::compiler::ast::EnumPatternBindings::Tuple(vars);
                      return Ok((rest3, Pattern::EnumPattern { enum_name: String::new(), variant_name: name, bindings }));
                  } else {
                       // Unit variant
                       let bindings = crate::compiler::ast::EnumPatternBindings::Unit;
                       return Ok((rest, Pattern::EnumPattern { enum_name: String::new(), variant_name: name, bindings }));
                  }
             }
         }
    }
    
    Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Invalid pattern".to_string()) }))
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

        // Handle optional self, possibly reference (&self)
        let (input, maybe_amp) = map(opt(expect_token(Token::Ampersand)), |o| o.is_some())(input)?;
        let (input, has_self) = map(opt(expect_token(Token::Self_)), |o| o.is_some())(input)?;
        
        // Validation: & without self? syntax error unless it's a type (but types are after colon)
        // If we saw &, expected self. But self is optional.
        if maybe_amp && !has_self {
             return Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Expected self after &".to_string()) }));
        }

        let input = if has_self {
             opt(expect_token(Token::Comma))(input)?.0
        } else {
             input
        };

        let (input, mut args) = separated_list0(
            expect_token(Token::Comma),
            pair(identifier, preceded(expect_token(Token::Colon), parse_type))
        )(input)?;

        if has_self {
            // Add self
            let self_type = crate::compiler::ast::Type::Struct("Self".to_string(), vec![]);
            let final_type = if maybe_amp {
                crate::compiler::ast::Type::Ref(Box::new(self_type))
            } else {
                self_type
            };
            args.insert(0, ("self".to_string(), final_type));
        }

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
        let (input, mut methods) = many0(parse_function_def)(input)?;
        let (input, _) = expect_token(Token::RBrace)(input)?;

        // Resolve `Self` in method arguments/return types
        for method in &mut methods {
             for arg in &mut method.args {
                 if let crate::compiler::ast::Type::Struct(name, _) = &arg.1 {
                     if name == "Self" {
                         arg.1 = target_type.clone();
                     }
                 }
             }
        }

        Ok((input, crate::compiler::ast::ImplBlock {
            target_type,
            generics,
            methods,
        }))
    })(input)
}

// Enum definition
fn parse_enum_def(input: Input) -> IResult<Input, crate::compiler::ast::EnumDef, ParserError> {
    let (rest, _) = expect_token(Token::Enum)(input)?;
    
    cut(move |input| {
        let (input, name) = identifier(input)?;
        let (input, generics) = parse_generic_params(input)?;
        
        let (input, _) = expect_token(Token::LBrace)(input)?;
        let (input, variants) = separated_list0(
            expect_token(Token::Comma),
            move |input| {
                let (input, v_name) = identifier(input)?;
                // Check if Tuple Variant: Variant(Type, ...)
                if let Ok((rest, _)) = expect_token(Token::LParen)(input) {
                    let (rest, types) = separated_list1(expect_token(Token::Comma), parse_type)(rest)?;
                    let (rest, _) = expect_token(Token::RParen)(rest)?;
                    
                    Ok((rest, crate::compiler::ast::VariantDef { name: v_name, kind: crate::compiler::ast::VariantKind::Tuple(types) }))
                } else if let Ok((rest, _)) = expect_token(Token::LBrace)(input) {
                    // Struct Variant: Variant { name: Type, ... }
                    let (rest, fields) = separated_list1(
                        expect_token(Token::Comma),
                        move |input| {
                             let (input, f_name) = identifier(input)?;
                             let (input, _) = expect_token(Token::Colon)(input)?;
                             let (input, f_type) = parse_type(input)?;
                             Ok((input, (f_name, f_type)))
                        }
                    )(rest)?;
                    let (rest, _) = expect_token(Token::RBrace)(rest)?;
                    Ok((rest, crate::compiler::ast::VariantDef { name: v_name, kind: crate::compiler::ast::VariantKind::Struct(fields) }))
                } else {
                     Ok((input, crate::compiler::ast::VariantDef { name: v_name, kind: crate::compiler::ast::VariantKind::Unit }))
                }
            }
        )(input)?;
        
        let (input, _) = opt(expect_token(Token::Comma))(input)?; // Allow trailing comma
        let (input, _) = expect_token(Token::RBrace)(input)?;
        
        Ok((input, crate::compiler::ast::EnumDef {
            name,
            generics,
            variants,
        }))
    })(rest)
}

// Module items: struct, impl, fn, ...


fn parse_use_decl(input: Input) -> IResult<Input, String, ParserError> {
    let (input, _) = expect_token(Token::Use)(input)?;
    let (input, path) = separated_list1(
        expect_token(Token::DoubleColon), 
        alt((
            identifier, 
            map(expect_token(Token::Star), |_| "*".to_string())
        ))
    )(input)?;
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

fn parse_datalog_atom(input: Input) -> IResult<Input, crate::compiler::ast::Atom, ParserError> {
    let (input, predicate) = alt((
        identifier,
        map(expect_token(Token::True), |_| "true".to_string()),
        map(expect_token(Token::False), |_| "false".to_string()),
    ))(input)?;

    let (input, args) = if let Ok((rest, _)) = expect_token(Token::LParen)(input) {
        let (rest, args) = separated_list0(expect_token(Token::Comma), parse_expr)(rest)?;
        let (rest, _) = expect_token(Token::RParen)(rest)?;
        (rest, args)
    } else {
        (input, vec![])
    };
    Ok((input, crate::compiler::ast::Atom { predicate, args }))
}


fn parse_logic_literal(input: Input) -> IResult<Input, crate::compiler::ast::LogicLiteral, ParserError> {
    if let Ok((rest, _)) = expect_token(Token::Not)(input) {
        let (rest, atom) = parse_datalog_atom(rest)?;
        Ok((rest, crate::compiler::ast::LogicLiteral::Neg(atom)))
    } else {
        let (rest, atom) = parse_datalog_atom(input)?;
        Ok((rest, crate::compiler::ast::LogicLiteral::Pos(atom)))
    }
}

fn parse_rule(input: Input) -> IResult<Input, crate::compiler::ast::Rule, ParserError> {
    // Head :- Body. or Head.
    let (rest_after_head, head) = parse_datalog_atom(input)?;
    


    if let Ok((rest, _)) = expect_token(Token::Entails)(rest_after_head) {
        // Rule
        let (rest, body) = separated_list1(expect_token(Token::Comma), parse_logic_literal)(rest)?;
        let (rest, _) = alt((expect_token(Token::Dot), expect_token(Token::SemiColon)))(rest)?;
        Ok((rest, crate::compiler::ast::Rule { head, body, weight: None }))
    } else if let Ok((rest, _)) = alt((expect_token(Token::Dot), expect_token(Token::SemiColon)))(rest_after_head) {
        // Fact
        Ok((rest, crate::compiler::ast::Rule { head, body: vec![], weight: None }))
    } else {
        // Not a rule/fact
        Err(nom::Err::Error(ParserError { input, kind: ParseErrorKind::UnexpectedToken("Expected . or :- after datalog atom".to_string()) }))
    }
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
        // Statements (for scripts) - Check BEFORE rules to avoid ambiguous atom parsing (e.g. `f.` fact vs `f.close()`)
        if let Ok((rest, stmt)) = parse_stmt(input) {
            match stmt.inner {
                StmtKind::TensorDecl{..} => module.tensor_decls.push(stmt),
                _ => {
                   module.tensor_decls.push(stmt);
                }
            }
            input = rest;
            continue;
        }

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
                module.imports.push(m.clone()); // Trigger loading
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
        match parse_enum_def(input) {
            Ok((rest, e)) => { module.enums.push(e); input = rest; continue; }
            Err(nom::Err::Failure(e)) => return Err(nom::Err::Failure(e)),
            Err(nom::Err::Error(_)) | Err(nom::Err::Incomplete(_)) => {}
        }
        match parse_tensor_decl(input) {
            Ok((rest, t)) => { module.tensor_decls.push(t); input = rest; continue; }
            Err(nom::Err::Failure(e)) => return Err(nom::Err::Failure(e)),
            Err(nom::Err::Error(_)) | Err(nom::Err::Incomplete(_)) => {}
        }
        match parse_rule(input) {
            Ok((rest, r)) => { module.rules.push(r); input = rest; continue; }
            Err(nom::Err::Failure(e)) => return Err(nom::Err::Failure(e)),
            Err(nom::Err::Error(_)) | Err(nom::Err::Incomplete(_)) => {}
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
