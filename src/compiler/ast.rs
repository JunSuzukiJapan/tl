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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum Type {
    // Primitive types
    F32,
    F64,
    F16,
    BF16,
    Bool,
    String(String),
    Char(String),
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

    // Reference type: &Type
    Ref(Box<Type>),

    // Type variable for inference (e.g., ?T0, ?T1)
    TypeVar(u32),

    // User defined struct
    Struct(String, Vec<Type>), // Name, Component Types (Generics)

    // User defined enum
    Enum(String, Vec<Type>), // Name, Component Types

    // Tuple type: (Type, Type, ...)
    Tuple(Vec<Type>),

    // Generic placeholder or unresolved type (Merged into Struct)
    // UserDefined(String, Vec<Type>), // REMOVED

    Void, // For functions returning nothing
    Undefined(u64), // For unresolved generics (unique ID)
}

impl Type {
    pub fn get_base_name(&self) -> String {
        match self {
            Type::F32 => "F32".to_string(),
            Type::F64 => "F64".to_string(),
            Type::F16 => "F16".to_string(),
            Type::BF16 => "BF16".to_string(),
            Type::Bool => "Bool".to_string(),
            Type::String(_) => "String".to_string(),
            Type::Char(_) => "Char".to_string(),
            Type::I32 => "I32".to_string(),
            Type::I64 => "I64".to_string(),
            Type::I8 => "I8".to_string(),
            Type::U8 => "U8".to_string(),
            Type::U16 => "U16".to_string(),
            Type::U32 => "U32".to_string(),
            Type::Usize => "Usize".to_string(),
            Type::Entity => "Entity".to_string(),
            Type::Tensor(_, _) => "Tensor".to_string(),
            Type::TensorShaped(_, _) => "Tensor".to_string(),
            Type::Struct(n, _) => n.clone(),
            Type::Enum(n, _) => n.clone(),

            Type::Tuple(_) => "Tuple".to_string(), // Or handle specially? TypeRegistry didn't support Tuple generic methods usually.
            // Type::UserDefined(n, _) => n.clone(), // REMOVED
            Type::Void => "Void".to_string(),
            Type::Undefined(_) => "Undefined".to_string(),
            Type::TypeVar(_) => "TypeVar".to_string(),
            Type::Ref(inner) => inner.get_base_name(), // Use inner name for Ref? Or maybe just "Ref"? 
            // Usually base name implies the nominal type. &Vec<T> base name is Vec.
        }
    }
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
    pub target_type: Type, // Changed from String to Type to generic support (e.g. Struct<T>)
    pub generics: Vec<String>,
    pub methods: Vec<FunctionDef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariantDef {
    pub name: String,
    pub kind: VariantKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VariantKind {
    Unit,
    Tuple(Vec<Type>),
    Struct(Vec<(String, Type)>),
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
        op: AssignOp,
        value: Expr,
    },
    Expr(Expr),
    Return(Option<Expr>),
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
    Loop {
        body: Vec<Stmt>,
    },
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
    CharLiteral(char),
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
    Wildcard,                      // _ (Anonymous Logic Variable)

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
    StaticMethodCall(Type, String, Vec<Expr>), // Type::method(args)

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


    // Struct Init: Name { field: value, ... }
    StructInit(String, Vec<Type>, Vec<(String, Expr)>),

    // Enum Init: Enum::Variant { ... } or Enum::Variant(...)
    EnumInit {
        enum_name: String,
        variant_name: String,
        generics: Vec<Type>, // Inferred/Explicit generics
        payload: EnumVariantInit,
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
    // Enum Pattern
    EnumPattern {
        enum_name: String,
        variant_name: String,
        bindings: EnumPatternBindings,
    },
    // Wildcard
    Wildcard,
    // Literal
    Literal(Box<Expr>),
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
    Query,
    Ref,
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
    pub body: Vec<LogicLiteral>,
    pub weight: Option<f64>, // Optional probability/weight for probabilistic rules
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogicLiteral {
    Pos(Atom),
    Neg(Atom),
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

#[derive(Debug, Clone, PartialEq)]
pub enum EnumVariantInit {
    Unit,
    Tuple(Vec<Expr>),
    Struct(Vec<(String, Expr)>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum EnumPatternBindings {
    Unit,
    Tuple(Vec<String>), // Bind to vars by position
    Struct(Vec<(String, String)>), // field: var
}
