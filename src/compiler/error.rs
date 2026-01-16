// src/compiler/error.rs
//! TLコンパイラの統合エラー処理モジュール

use std::fmt;
use thiserror::Error;

/// ソースコード内の位置情報
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Span {
    /// ファイル名（既知の場合）
    pub file: Option<String>,
    /// 行番号 (1-indexed)
    pub line: usize,
    /// 列番号 (1-indexed)
    pub column: usize,
}

impl Span {
    /// 新しいSpanを作成（ファイル名なし）
    pub fn new(line: usize, column: usize) -> Self {
        Span {
            file: None,
            line,
            column,
        }
    }

    /// ファイル名付きのSpanを作成
    pub fn with_file(file: impl Into<String>, line: usize, column: usize) -> Self {
        Span {
            file: Some(file.into()),
            line,
            column,
        }
    }

    /// ファイル名を設定
    pub fn set_file(&mut self, file: impl Into<String>) {
        self.file = Some(file.into());
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref file) = self.file {
            write!(f, "{}:{}:{}", file, self.line, self.column)
        } else {
            write!(f, "{}:{}", self.line, self.column)
        }
    }
}

/// TLコンパイラの統合エラー型
#[derive(Error, Debug)]
pub enum TlError {
    /// パーサーエラー
    #[error("{kind}")]
    Parse {
        kind: ParseErrorKind,
        span: Option<Span>,
    },

    /// セマンティクスエラー
    #[error("{kind}")]
    Semantic {
        kind: SemanticErrorKind,
        span: Option<Span>,
    },

    /// コード生成エラー
    #[error("{kind}")]
    Codegen {
        kind: CodegenErrorKind,
        span: Option<Span>,
    },

    /// I/Oエラー
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl TlError {
    /// Spanを取得
    pub fn span(&self) -> Option<&Span> {
        match self {
            TlError::Parse { span, .. } => span.as_ref(),
            TlError::Semantic { span, .. } => span.as_ref(),
            TlError::Codegen { span, .. } => span.as_ref(),
            TlError::Io(_) => None,
        }
    }

    /// ファイル名を設定（既存のSpanに追加）
    pub fn with_file(mut self, file: impl Into<String>) -> Self {
        let file = file.into();
        match &mut self {
            TlError::Parse { span, .. } => {
                if let Some(s) = span {
                    s.file = Some(file);
                } else {
                    *span = Some(Span {
                        file: Some(file),
                        line: 0,
                        column: 0,
                    });
                }
            }
            TlError::Semantic { span, .. } => {
                if let Some(s) = span {
                    s.file = Some(file);
                } else {
                    *span = Some(Span {
                        file: Some(file),
                        line: 0,
                        column: 0,
                    });
                }
            }
            TlError::Codegen { span, .. } => {
                if let Some(s) = span {
                    s.file = Some(file);
                } else {
                    *span = Some(Span {
                        file: Some(file),
                        line: 0,
                        column: 0,
                    });
                }
            }
            TlError::Io(_) => {}
        }
        self
    }

    /// Rustスタイルのエラー表示
    pub fn display_rust_style(&self) -> String {
        let error_type = match self {
            TlError::Parse { .. } => "parse error",
            TlError::Semantic { .. } => "semantic error",
            TlError::Codegen { .. } => "codegen error",
            TlError::Io(_) => "io error",
        };

        let message = match self {
            TlError::Parse { kind, .. } => kind.to_string(),
            TlError::Semantic { kind, .. } => kind.to_string(),
            TlError::Codegen { kind, .. } => kind.to_string(),
            TlError::Io(e) => e.to_string(),
        };

        if let Some(span) = self.span() {
            if span.line > 0 {
                format!(
                    "error[E0001]: {}\n  --> {}\n  |\n{} | <source line>\n  | ^\n  = {}: {}",
                    message, span, span.line, error_type, message
                )
            } else if let Some(ref file) = span.file {
                format!(
                    "error[E0001]: {}\n  --> {}\n  = {}",
                    message, file, error_type
                )
            } else {
                format!("error[E0001]: {}", message)
            }
        } else {
            format!("error[E0001]: {}", message)
        }
    }
}

/// パーサーエラーの種類
#[derive(Error, Debug, Clone)]
pub enum ParseErrorKind {
    #[error("unexpected token: {0}")]
    UnexpectedToken(String),

    #[error("unexpected end of input")]
    UnexpectedEof,

    #[error("invalid syntax: {0}")]
    InvalidSyntax(String),

    #[error("parse error: {0}")]
    Generic(String),
}

/// セマンティクスエラーの種類
#[derive(Error, Debug, Clone)]
pub enum SemanticErrorKind {
    #[error("variable not found: {0}")]
    VariableNotFound(String),

    #[error("variable has been moved: {0}")]
    VariableMoved(String),

    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error("function not found: {0}")]
    FunctionNotFound(String),

    #[error("struct not found: {0}")]
    StructNotFound(String),

    #[error("duplicate definition: {0}")]
    DuplicateDefinition(String),

    #[error("duplicate match arm for variant: {0}")]
    DuplicateMatchArm(String),

    #[error("unreachable match arm after wildcard")]
    UnreachableMatchArm,

    #[error("non-exhaustive match on enum {enum_name}, missing variants: {missing_variants:?}")]
    NonExhaustiveMatch {
        enum_name: String,
        missing_variants: Vec<String>,
    },

    #[error("incorrect number of arguments for {name}: expected {expected}, found {found}")]
    ArgumentCountMismatch {
        name: String,
        expected: usize,
        found: usize,
    },

    #[error("method not found: {method_name} on type {type_name}")]
    MethodNotFound {
        type_name: String,
        method_name: String,
    },

    #[error("unknown function: {0}")]
    UnknownFunction(String),

    #[error("tuple index out of bounds: index {0} is out of bounds for tuple of size {1}")]
    TupleIndexOutOfBounds(usize, usize),

    #[error("cannot index into non-tuple type: {0}")]
    NotATuple(String),

    #[error("cannot assign to immutable variable: {0}")]
    AssignToImmutable(String),
}

/// コード生成エラーの種類
#[derive(Error, Debug, Clone)]
pub enum CodegenErrorKind {
    #[error("variable not found: {0}")]
    VariableNotFound(String),

    #[error("function not found: {0}")]
    FunctionNotFound(String),

    #[error("type error: {0}")]
    TypeError(String),

    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("invalid generated code: {0}")]
    InvalidCode(String),

    #[error("codegen error: {0}")]
    Generic(String),
}

/// 入力文字列とオフセットから行と列を計算
pub fn offset_to_line_col(input: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, ch) in input.chars().enumerate() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_display() {
        let span = Span::new(10, 5);
        assert_eq!(span.to_string(), "10:5");

        let span = Span::with_file("test.tl", 10, 5);
        assert_eq!(span.to_string(), "test.tl:10:5");
    }

    #[test]
    fn test_offset_to_line_col() {
        let input = "fn main() {\n    let x = 1;\n}";
        // 'l' in 'let' is at offset 16
        let (line, col) = offset_to_line_col(input, 16);
        assert_eq!(line, 2);
        assert_eq!(col, 5);
    }

    #[test]
    fn test_error_with_file() {
        let err = TlError::Semantic {
            kind: SemanticErrorKind::VariableNotFound("x".to_string()),
            span: Some(Span::new(5, 10)),
        };
        let err = err.with_file("example.tl");
        if let Some(span) = err.span() {
            assert_eq!(span.file, Some("example.tl".to_string()));
            assert_eq!(span.line, 5);
            assert_eq!(span.column, 10);
        }
    }
}
