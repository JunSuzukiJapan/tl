// src/compiler/lexer.rs
use logos::Logos;
use std::fmt;

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\f]+")] // Skip whitespace
#[logos(skip r"//.*")]       // Skip comments
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

    // Types (primitive types also act as keywords in some context)
    #[token("f32")]
    F32Type,
    #[token("f64")]
    F64Type,
    #[token("i32")]
    I32Type,
    #[token("i64")]
    I64Type,
    #[token("bool")]
    BoolType,
    #[token("usize")]
    UsizeType,
    #[token("void")]
    VoidType,

    // Identifiers
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
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
    Not,
    #[token("|")]
    Pipe,
    #[token("&")]
    Ampersand,
    #[token("^")]
    Caret,
    #[token("<-")]
    LArrow,
    
    #[token("+=")]
    PlusAssign,
    #[token("-=")]
    MinusAssign,
    #[token("*=")]
    StarAssign,
    #[token("/=")]
    SlashAssign,

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
    fn test_literals() {
        let input = "42 3.14 \"hello\"";
        let tokens: Vec<Token> = tokenize(input).into_iter().map(|r| r.unwrap().token).collect();
        assert_eq!(tokens, vec![
            Token::IntLiteral(42),
            Token::FloatLiteral(3.14),
            Token::StringLiteral("hello".to_string()),
        ]);
    }

    #[test]
    fn test_symbols_ops() {
        let input = ":: + - -> =>";
        let tokens: Vec<Token> = tokenize(input).into_iter().map(|r| r.unwrap().token).collect();
        assert_eq!(tokens, vec![
            Token::DoubleColon,
            Token::Plus,
            Token::Minus,
            Token::Arrow,
            Token::FatArrow,
        ]);
    }
}
