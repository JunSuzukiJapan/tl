// src/compiler/ast.rs
#![allow(dead_code)]
use crate::compiler::error::Span;
use std::collections::HashMap;

/// Spanを持つラッパー型
/// ASTノードに位置情報を付加するために使用
/// Spanを持つラッパー型
/// ASTノードに位置情報を付加するために使用
#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub inner: T,
    pub span: Span,
}

impl<T: PartialEq> PartialEq for Spanned<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T: Eq> Eq for Spanned<T> {}

impl<T> Spanned<T> {
    /// 新しいSpannedを作成
    pub fn new(inner: T, span: Span) -> Self {
        Spanned { inner, span }
    }

    /// ダミーのSpan（位置情報なし）でラップ
    pub fn dummy(inner: T) -> Self {
        Spanned {
            inner,
            span: Span::default(),
        }
    }
}

impl<T> std::ops::Deref for Spanned<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: Default> Default for Spanned<T> {
    fn default() -> Self {
        Spanned::dummy(T::default())
    }
}

/// Dimension: either a constant, a variable, or a symbolic name
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    Constant(usize),  // Fixed dimension (e.g., 3, 64, 1024)
    Var(u32),         // Inference variable (e.g., ?D0, ?D1)
    Symbolic(String), // Named dimension (e.g., "batch", "seq_len")
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum Type {
    // Primitive types
    F32,
    F64,
    F16,
    BF16,
    Bool,
    I32,
    I64,
    I8,
    U8,
    U16,
    U32,
    Usize,
    Entity, // Logic Entity

    // Tensor type: Tensor<Type, Rank>
    Tensor(Box<Type>, usize),

    // Tensor with shape information for inference
    TensorShaped(Box<Type>, Vec<Dim>),

    // Type variable for inference (e.g., ?T0, ?T1)
    TypeVar(u32),

    // Object list: Vec<Type>
    Vec(Box<Type>),

    // User defined struct
    Struct(String),

    // User defined enum
    Enum(String),

    // Optimized small constant array (elements up to 4, stored as scalars)
    ScalarArray(Box<Type>, usize), // (element_type, length)

    // Tuple type: (Type, Type, ...)
    Tuple(Vec<Type>),

    // Generic placeholder
    UserDefined(String),

    Void, // For functions returning nothing
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub args: Vec<(String, Type)>,
    pub return_type: Type,
    pub body: Vec<Stmt>,
    pub generics: Vec<String>, // <T>
    pub is_extern: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<(String, Type)>,
    pub generics: Vec<String>, // <T>
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImplBlock {
    pub target_type: String,
    pub generics: Vec<String>,
    pub methods: Vec<FunctionDef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariantDef {
    pub name: String,
    pub fields: Vec<(String, Type)>, // Named fields or empty for unit/tuple-like
                                     // For now we only support named fields or unit. Tuple variants can be named fields "0", "1"... or just distinct syntax?
                                     // Let's stick to named fields for simplicity (struct variants), or maybe tuple variants too later.
                                     // Rust allows: Unit, Tuple(A,B), Struct{x:A}
                                     // Let's start with Struct-like variants (named fields) and Unit variants (empty fields).
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumDef {
    pub name: String,
    pub variants: Vec<VariantDef>,
    pub generics: Vec<String>,
}

pub type Stmt = Spanned<StmtKind>;

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    TensorDecl {
        name: String,
        type_annotation: Type,
        init: Option<Expr>,
    },
    Let {
        name: String,
        type_annotation: Option<Type>,
        value: Expr,
        mutable: bool,
    },
    Assign {
        name: String,
        indices: Option<Vec<Expr>>,
        op: AssignOp, // =, +=, max=, avg=
        value: Expr,
    },
    FieldAssign {
        obj: Expr,
        field: String,
        value: Expr,
    },
    Expr(Expr),
    Return(Option<Expr>),
    If {
        cond: Expr,
        then_block: Vec<Stmt>,
        else_block: Option<Vec<Stmt>>,
    },
    For {
        loop_var: String,
        iterator: Expr, // range or Vec
        body: Vec<Stmt>,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
    Use {
        path: Vec<String>,
        alias: Option<String>,
        items: Vec<String>,
    },
    Break,
    Continue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
    MaxAssign,
    AvgAssign,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComprehensionClause {
    Generator { name: String, range: Expr }, // i <- 0..5
    Condition(Expr),                         // i != j
}

pub type Expr = Spanned<ExprKind>;

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    // Literals
    Float(f64),
    Int(i64),
    Bool(bool),
    StringLiteral(String),
    Tuple(Vec<Expr>),            // (a, b, c)
    Range(Box<Expr>, Box<Expr>), // start..end
    TensorComprehension {
        indices: Vec<String>,
        clauses: Vec<ComprehensionClause>,
        body: Option<Box<Expr>>,
    },
    TensorLiteral(Vec<Expr>),      // Dynamic tensor with expressions
    TensorConstLiteral(Vec<Expr>), // Static tensor with only constants (optimized)
    Symbol(String),                // Logic Symbol (unquoted identifier)
    LogicVar(String),              // Logic Variable ($name)

    // Variables & Access
    Variable(String),
    IndexAccess(Box<Expr>, Vec<Expr>), // Name[i, j]
    TupleAccess(Box<Expr>, usize),     // Name.0
    FieldAccess(Box<Expr>, String),    // self.field

    // Ops
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    UnOp(UnOp, Box<Expr>),

    // Calls
    FnCall(String, Vec<Expr>),
    MethodCall(Box<Expr>, String, Vec<Expr>),
    StaticMethodCall(String, String, Vec<Expr>), // Type::method(args)

    // Cast: expr as Type
    As(Box<Expr>, Type),

    // Control
    IfExpr(Box<Expr>, Vec<Stmt>, Option<Vec<Stmt>>),
    IfLet {
        pattern: Pattern,
        expr: Box<Expr>,
        then_block: Vec<Stmt>,
        else_block: Option<Vec<Stmt>>,
    },
    Block(Vec<Stmt>),

    // Aggregation: sum(expr for var in range where condition)
    Aggregation {
        op: AggregateOp,
        expr: Box<Expr>,
        var: String,
        range: Box<Expr>,             // e.g., 0..n or a collection
        condition: Option<Box<Expr>>, // where clause
    },

    // Struct Init: Name { field: value, ... }
    StructInit(String, Vec<(String, Expr)>),

    // Enum Init: Enum::Variant { field: value, ... }
    EnumInit {
        enum_name: String,
        variant_name: String,
        fields: Vec<(String, Expr)>,
    },

    // Match expression
    Match {
        expr: Box<Expr>,
        arms: Vec<(Pattern, Expr)>, // (pattern, body)
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    // Enum Pattern: Enum::Variant { x, y } (binds fields to variables)
    // For now, simplify binding: just list variable names that bind to fields by position or name?
    // Let's support: Variant { field: var, ... }
    EnumPattern {
        enum_name: String,
        variant_name: String,
        bindings: Vec<(String, String)>, // (field_name, var_name)
    },
    // Wildcard
    Wildcard,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AggregateOp {
    Sum,
    Max,
    Min,
    Avg,
    Count,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod, // Modulo operator (%)
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or, // Logical
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    pub predicate: String,
    pub args: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelationDecl {
    pub name: String,
    pub args: Vec<(String, Type)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    pub head: Atom,
    pub body: Vec<Atom>,
    pub weight: Option<f64>, // Optional probability/weight for probabilistic rules
}

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub structs: Vec<StructDef>,
    pub enums: Vec<EnumDef>,
    pub impls: Vec<ImplBlock>,
    pub functions: Vec<FunctionDef>,
    pub tensor_decls: Vec<Stmt>, // StmtKind::TensorDecl
    pub relations: Vec<RelationDecl>,
    pub rules: Vec<Rule>,
    pub queries: Vec<Expr>,
    pub imports: Vec<String>,
    pub submodules: HashMap<String, Module>,
}
