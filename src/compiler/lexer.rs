// src/compiler/lexer.rs
use logos::Logos;
use std::fmt;

// --------------------------------------------------------------------------------
// [CRITICAL WARNING]
// DO NOT INTRODUCE `&self` SYNTAX.
//
// In TL, structs are Handles (implied pointers). Passing `self` passes the Handle (pointer).
// Passing `&self` would pass the Address of the Handle (pointer to pointer), causing
// Runtime Segfaults because the runtime expects a direct Handle value.
//
// Reference: User Request 2026-02-02
// --------------------------------------------------------------------------------

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\f]+")] // Skip whitespace
#[logos(skip r"//.*")]       // Skip line comments
#[logos(skip r"/\*([^*]|\*+[^*/])*\*+/")] // Skip block comments
pub enum Token {
    // Keywords
    #[token("fn")]
    Fn,
    #[token("struct")]
    Struct,
    #[token("impl")]
    Impl,
    #[token("let")]
    Let,
    #[token("mut")]
    Mut,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("return")]
    Return,
    #[token("for")]
    For,
    #[token("in")]
    In,
    #[token("while")]
    While,
    #[token("loop")]
    Loop,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("match")]
    Match,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("pub")]
    Pub,
    #[token("use")]
    Use,
    #[token("mod")]
    Mod,
    #[token("as")]
    As,
    #[token("extern")]
    Extern,
    #[token("enum")]
    Enum,
    #[token("self")]
    Self_,


    // Types (primitive types also act as keywords in some context)
    #[token("f32")]
    F32Type,
    #[token("f64")]
    F64Type,
    #[token("i8")]
    I8Type,
    #[token("i16")]
    I16Type,
    #[token("i32")]
    I32Type,
    #[token("i64")]
    I64Type,
    #[token("u8")]
    U8Type,
    #[token("u16")]
    U16Type,
    #[token("u32")]
    U32Type,
    #[token("u64")]
    U64Type,
    #[token("bool")]
    BoolType,
    #[token("usize")]
    UsizeType,
    #[token("void")]
    VoidType,
    #[token("String")]
    StringType,
    #[token("Char")]
    CharType,

    // Identifiers
    #[regex(r"[a-zA-Z][a-zA-Z0-9_]*|_[a-zA-Z0-9_]+", |lex| lex.slice().to_string())]
    Identifier(String),

    // Literals
    #[regex("-?[0-9]+", |lex| lex.slice().parse().ok())]
    IntLiteral(i64),

    #[regex(r"-?[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?", |lex| lex.slice().parse().ok())]
    #[regex(r"-?[0-9]+[eE][+-]?[0-9]+", |lex| lex.slice().parse().ok())] // Scientific notation without dot
    FloatLiteral(f64),

    #[regex(r#""([^"\\]|\\[\s\S])*""#, |lex| {
        let s = lex.slice();
        s[1..s.len()-1].to_string() // TODO: Handle escapes properly
    })]
    StringLiteral(String),

    #[regex(r"'([^'\\]|\\.)'", |lex| {
        let s = lex.slice();
        s.chars().nth(1).unwrap() // Simple char extraction
    })]
    CharLiteral(char),

    // Symbols
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token(":")]
    Colon,
    #[token(";")]
    SemiColon,
    #[token("=")]
    Assign,
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token("::")]
    DoubleColon,
    #[token("..")]
    Range,
    
    // Operators
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("==")]
    Eq,
    #[token("!=")]
    Ne,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("<=")]
    Le,
    #[token(">=")]
    Ge,
    #[token("&&")]
    And,
    #[token("||")]
    Or,
    #[token("!")]
    #[token("not")]
    Not,
    #[token("|")]
    Pipe,
    #[token("&")]
    Ampersand,
    #[token("^")]
    Caret,
    #[token("<-")]
    LArrow,
    #[token(":-")]
    Entails,
    
    #[token("_")]
    Underscore,
    #[token("?")]
    Question,
    #[token("$")]
    Dollar,
    
    #[token("+=")]
    PlusAssign,
    #[token("-=")]
    MinusAssign,
    #[token("*=")]
    StarAssign,
    #[token("/=")]
    SlashAssign,
    #[token("%=")]
    PercentAssign,

    // Error
    // Logos automatically handles errors but we can have an explicit Error variant if needed.
    // For now we rely on Result from lexer iteration.
}

// Wrapper for Span
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

impl Span {
    pub fn new(start: usize, end: usize, line: usize, column: usize) -> Self {
        Span { start, end, line, column }
    }
}

// Spanned Token
#[derive(Debug, Clone, PartialEq)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Span,
}

pub fn tokenize(input: &str) -> Vec<Result<SpannedToken, String>> {
    let mut tokens = Vec::new();
    let mut lex = Token::lexer(input);
    
    // Simple line tracking
    // We can pre-calculate line start indices for O(1) lookup or track as we go.
    // Since Logos jumps around or slices, we might need global offset.
    // Logos `span()` gives byte range absolute to input.
    
    // Let's build a line_index: Vec<usize> of line start offsets.
    let mut line_starts = vec![0];
    for (i, c) in input.char_indices() {
        if c == '\n' {
            line_starts.push(i + 1);
        }
    }
    
    while let Some(token_res) = lex.next() {
        let span = lex.span(); // byte range
        let start = span.start;
        let end = span.end;
        
        // Find line from start offset using binary search
        let line_idx = match line_starts.binary_search(&start) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        
        let line = line_idx + 1;
        let line_start = line_starts[line_idx];
        // Column is char count from line_start to start.
        // We need to count chars, not bytes, for column.
        let column = input[line_start..start].chars().count() + 1;
        
        let span_info = Span::new(start, end, line, column);
        
        match token_res {
            Ok(t) => tokens.push(Ok(SpannedToken {
                token: t,
                span: span_info,
            })),
            Err(_) => tokens.push(Err(format!("Invalid token at {:?}", span_info))),
        }
    }
    
    tokens
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keywords_and_identifiers() {
        let input = "fn main let mut match_errors";
        let tokens: Vec<Token> = tokenize(input).into_iter().map(|r| r.unwrap().token).collect();
        assert_eq!(tokens, vec![
            Token::Fn,
            Token::Identifier("main".to_string()),
            Token::Let,
            Token::Mut,
            Token::Identifier("match_errors".to_string()), // Critical test case
        ]);
    }

    #[test]
    fn test_snake_case_identifiers() {
        let input = "from_int String::from_int";
        let tokens: Vec<Token> = tokenize(input).into_iter().map(|r| r.unwrap().token).collect();
        assert_eq!(tokens, vec![
            Token::Identifier("from_int".to_string()),
            Token::StringType,
            Token::DoubleColon,
            Token::Identifier("from_int".to_string()),
        ]);
    }

    #[test]
    fn test_literals() {
        let input = "42 3.14 \"hello\"";
        let tokens: Vec<Token> = tokenize(input).into_iter().map(|r| r.unwrap().token).collect();
        assert_eq!(tokens, vec![
            Token::IntLiteral(42),
            Token::FloatLiteral(std::f64::consts::PI),
            Token::StringLiteral("hello".to_string()),
        ]);
    }

    #[test]
    fn test_logic_tokens() {
        let input = "? $ :- <-";
        let tokens: Vec<Token> = tokenize(input).into_iter().map(|r| r.unwrap().token).collect();
        assert_eq!(tokens, vec![
            Token::Question,
            Token::Dollar,
            Token::Entails,
            Token::LArrow,
        ]);
    }

    #[test]
    fn test_complex_floats() {
        let input = "1.0 10.5e-10 -0.5 3.14E+2";
        let tokens: Vec<Token> = tokenize(input).into_iter().map(|r| r.unwrap().token).collect();
        assert_eq!(tokens, vec![
            Token::FloatLiteral(1.0),
            Token::FloatLiteral(10.5e-10),
            Token::FloatLiteral(-0.5),
            Token::FloatLiteral(3.14E+2),
        ]);
    }

    #[test]
    fn test_operators_combinations() {
        let input = "== = != ! <= < >= > && || .. .";
        let tokens: Vec<Token> = tokenize(input).into_iter().map(|r| r.unwrap().token).collect();
        assert_eq!(tokens, vec![
            Token::Eq,
            Token::Assign,
            Token::Ne,
            Token::Not,
            Token::Le,
            Token::Lt,
            Token::Ge,
            Token::Gt,
            Token::And,
            Token::Or,
            Token::Range,
            Token::Dot,
        ]);
    }

    #[test]
    fn test_comments_and_whitespace() {
        let input = r#"
            // This is a comment
            let x = 10; // Inline comment
            /* Block comments are not supported by this regex lexer yet 
               but lines starting with // are skipped */
            let y = 20;
        "#;
        let tokens: Vec<Token> = tokenize(input).into_iter().map(|r| r.unwrap().token).collect();
        assert_eq!(tokens, vec![
            Token::Let,
            Token::Identifier("x".to_string()),
            Token::Assign,
            Token::IntLiteral(10),
            Token::SemiColon, // Added missing semicolon
            Token::Let,
            Token::Identifier("y".to_string()),
            Token::Assign,
            Token::IntLiteral(20),
            Token::SemiColon, // Added missing semicolon
        ]);
    }
    
    #[test]
    fn test_string_escapes_recognition() {
        // Note: The lexer currently trims quotes but doesn't decode escapes in the Value.
        // It just recognizes the string literal token.
        let input = r#""hello world" "escaped\"quote""#;
        let tokens: Vec<Token> = tokenize(input).into_iter().map(|r| r.unwrap().token).collect();
        
        // We verify that it tokenizes as StringLiteral, checking content is secondary 
        // until we implement full escape processing.
        match &tokens[0] {
            Token::StringLiteral(s) => assert_eq!(s, "hello world"),
            _ => panic!("Expected StringLiteral"),
        }
        match &tokens[1] {
            Token::StringLiteral(s) => assert_eq!(s, r#"escaped\"quote"#),
            _ => panic!("Expected StringLiteral"),
        }
    }
    
    #[test]
    fn test_spans_and_lines() {
        let input = "a\nb\n  c";
        let results = tokenize(input);
        
        let t1 = results[0].as_ref().unwrap();
        assert_eq!(t1.token, Token::Identifier("a".to_string()));
        assert_eq!(t1.span.line, 1);
        assert_eq!(t1.span.column, 1);
        
        let t2 = results[1].as_ref().unwrap();
        assert_eq!(t2.token, Token::Identifier("b".to_string()));
        assert_eq!(t2.span.line, 2);
        assert_eq!(t2.span.column, 1);
        
        let t3 = results[2].as_ref().unwrap();
        assert_eq!(t3.token, Token::Identifier("c".to_string()));
        assert_eq!(t3.span.line, 3);
        assert_eq!(t3.span.column, 3);
    }
}
