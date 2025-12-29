// src/compiler/ast.rs

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

    // Tensor type: Tensor<Type, Rank>
    Tensor(Box<Type>, usize),

    // Object list: Vec<Type>
    Vec(Box<Type>),

    // User defined struct
    Struct(String),

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
    Let {
        name: String,
        indices: Option<Vec<String>>, // For let x[i, j] = ...
        type_annotation: Option<Type>,
        value: Expr,
    },
    Assign {
        name: String,
        indices: Option<Vec<String>>,
        op: AssignOp, // =, +=, max=, avg=
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
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignOp {
    Assign,
    AddAssign,
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
    TensorLiteral(Vec<Expr>), // Nested Arrays

    // Variables & Access
    Variable(String),
    IndexAccess(Box<Expr>, Vec<String>), // Name[i, j]
    FieldAccess(Box<Expr>, String),      // self.field

    // Ops
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    UnOp(UnOp, Box<Expr>),

    // Calls
    FnCall(String, Vec<Expr>),
    MethodCall(Box<Expr>, String, Vec<Expr>),

    // Control
    IfExpr(Box<Expr>, Vec<Stmt>, Option<Vec<Stmt>>),
    Block(Vec<Stmt>),
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
pub struct Module {
    pub structs: Vec<StructDef>,
    pub impls: Vec<ImplBlock>,
    pub functions: Vec<FunctionDef>,
}
