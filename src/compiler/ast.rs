// src/compiler/ast.rs
#![allow(dead_code)]

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

    // Optimized small constant array (elements up to 4, stored as scalars)
    ScalarArray(Box<Type>, usize), // (element_type, length)

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
pub enum Stmt {
    TensorDecl {
        name: String,
        type_annotation: Type,
        init: Option<Expr>,
    },
    Let {
        name: String,
        type_annotation: Option<Type>,
        value: Expr,
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
    Return(Expr),
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
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    MaxAssign,
    AvgAssign,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // Literals
    Float(f64),
    Int(i64),
    Bool(bool),
    StringLiteral(String),
    TensorComprehension {
        indices: Vec<String>,
        body: Box<Expr>,
    },
    TensorLiteral(Vec<Expr>),      // Dynamic tensor with expressions
    TensorConstLiteral(Vec<Expr>), // Static tensor with only constants (optimized)

    // Variables & Access
    Variable(String),
    IndexAccess(Box<Expr>, Vec<Expr>), // Name[i, j]
    FieldAccess(Box<Expr>, String),    // self.field

    // Ops
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    UnOp(UnOp, Box<Expr>),

    // Calls
    FnCall(String, Vec<Expr>),
    MethodCall(Box<Expr>, String, Vec<Expr>),
    StaticMethodCall(String, String, Vec<Expr>), // Type::method(args)

    // Control
    IfExpr(Box<Expr>, Vec<Stmt>, Option<Vec<Stmt>>),
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
    pub impls: Vec<ImplBlock>,
    pub functions: Vec<FunctionDef>,
    pub tensor_decls: Vec<Stmt>, // Stmt::TensorDecl
    pub relations: Vec<RelationDecl>,
    pub rules: Vec<Rule>,
    pub queries: Vec<Expr>,
}
