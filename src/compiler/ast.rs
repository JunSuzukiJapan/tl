// src/compiler/ast.rs
#![allow(dead_code)]
use crate::compiler::error::Span;
use std::collections::HashMap;

/// マングル名の開始デリミタ（後方互換用。新コードでは `MANGLER` を直接使用すること）
pub use crate::compiler::mangler::MANGLER;

// ── 後方互換ラッパー（既存の呼び出し元が壊れないように残す） ──

/// 型引数を角括弧で囲んでマングル名を生成（後方互換ラッパー）
pub fn mangle_wrap_args(base: &str, args: &[String]) -> String {
    MANGLER.wrap_args(base, args)
}

/// マングル名からベース名を抽出（後方互換ラッパー）
pub fn mangle_base_name(mangled: &str) -> &str {
    MANGLER.base_name(mangled)
}

/// マングル名が型引数を含むか判定（後方互換ラッパー）
pub fn mangle_has_args(mangled: &str) -> bool {
    MANGLER.has_args(mangled)
}



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
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Usize,
    Entity, // Logic Entity

    // ───────────────────────────────────────────────────────────────────────
    // 【設計原則】勾配追跡の型レベル分離
    //
    // TLでは勾配の有無を型システムで静的に区別する:
    //   - Tensor    = 勾配なし（推論用）
    //   - GradTensor = 勾配あり（学習用）
    //
    // この設計により、PyTorchの no_grad のようなランタイムフラグは不要。
    // 勾配の有無はコンパイル時に決定されるため、実行時に切り替えることはない。
    // ───────────────────────────────────────────────────────────────────────

    /// 勾配を追跡しないテンソル（推論・データ処理用）
    Tensor(Box<Type>, usize),

    /// 勾配追跡付きテンソル（学習用）。freeze/unfreeze, clip_grad_value 等はこの型専用。
    GradTensor(Box<Type>, usize),

    // Tensor with shape information for inference
    TensorShaped(Box<Type>, Vec<Dim>),

    // Reference type: &Type - REMOVED (not in spec)
    // Ref(Box<Type>),
    
    // Raw Pointer type: ptr<Type> (or *Type)
    Ptr(Box<Type>),

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

    // Path type: Unresolved user type reference (Mod::Struct<T>)
    Path(Vec<String>, Vec<Type>),

    // Fixed-size array type: [T; N]
    Array(Box<Type>, usize),

    // Range type: Range (Option<i64> based start/end)
    Range,

    /// Monomorphized generic type with complete type information.
    /// Retains the original generic definition and the parameter substitution map.
    SpecializedType {
        gen_type: Box<Type>,                // The original generic type (e.g., Struct("Vec", [Path("T", [])]))
        type_args: Vec<Type>,               // Concrete type arguments in original order (e.g., [I64])
        type_map: Vec<(String, Type)>,      // Concrete type mapping (e.g., [("T", I64)])
        mangled_name: String,               // The fully mangled name (e.g., "Option[i64]") for LLVM linkage
    },

    Void, // For functions returning nothing
    Never, // For diverging expressions (panic!, unreachable, etc.)
    Undefined(u64), // For unresolved generics (unique ID)

    /// Closure / function type: Fn(arg_types) -> return_type
    Fn(Vec<Type>, Box<Type>),

    // Represents a dynamic trait object pointer (`dyn Trait`)
    // V6.0 memory management handles it as a standard reference counting type
    TraitObject(String), // Trait Name
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
            Type::I8 => "I8".to_string(),
            Type::I16 => "I16".to_string(),
            Type::I32 => "I32".to_string(),
            Type::I64 => "I64".to_string(),
            Type::U8 => "U8".to_string(),
            Type::U16 => "U16".to_string(),
            Type::U32 => "U32".to_string(),
            Type::U64 => "U64".to_string(),
            Type::Usize => "Usize".to_string(),
            Type::Entity => "Entity".to_string(),
            Type::Tensor(_, _) => "Tensor".to_string(),
            Type::GradTensor(_, _) => "GradTensor".to_string(),
            Type::TensorShaped(_, _) => "Tensor".to_string(),
            Type::Struct(n, _) => n.clone(),
            Type::Enum(n, _) => n.clone(),
            Type::SpecializedType { gen_type, .. } => gen_type.get_base_name(),
            Type::Path(p, _) => p.last().cloned().unwrap_or_default(),

            Type::Tuple(_) => "Tuple".to_string(), // Or handle specially? TypeRegistry didn't support Tuple generic methods usually.
            // Type::UserDefined(n, _) => n.clone(), // REMOVED
            Type::Void => "Void".to_string(),
            Type::Never => "Never".to_string(),
            Type::Undefined(_) => "Undefined".to_string(),
            Type::TypeVar(_) => "TypeVar".to_string(),
            // Type::Ref(inner) => inner.get_base_name(), // REMOVED
            Type::Ptr(_inner) => "Ptr".to_string(),
            Type::Range => "Range".to_string(),
            Type::Array(inner, _) => format!("Array_{}", inner.get_base_name()),
            Type::Fn(_, _) => "Fn".to_string(),
            Type::TraitObject(t) => format!("dyn {}", t),
        }
    }

    /// Returns (name, args) for struct-like types (Struct or UnifiedType with is_enum=false).
    /// In codegen, prefer this over direct pattern matching to handle both representations.
    pub fn as_struct_like(&self) -> Option<(&str, &[Type])> {
        match self {
            Type::Struct(name, args) => Some((name, args)),
            Type::SpecializedType { gen_type, type_args, .. } if gen_type.is_struct_type() => Some((gen_type.mangled_name_or_name().unwrap_or(""), type_args)),
            _ => None,
        }
    }

    /// Returns (name, args) for enum-like types (Enum or UnifiedType with is_enum=true).
    pub fn as_enum_like(&self) -> Option<(&str, &[Type])> {
        match self {
            Type::Enum(name, args) => Some((name, args)),
            Type::SpecializedType { gen_type, type_args, .. } if gen_type.is_enum_type() => Some((gen_type.mangled_name_or_name().unwrap_or(""), type_args)),
            _ => None,
        }
    }

    /// Returns (name, args) for any named type (Struct, Enum, or UnifiedType).
    pub fn as_named_type(&self) -> Option<(&str, &[Type])> {
        match self {
            Type::Struct(name, args) | Type::Enum(name, args) => Some((name, args)),
            Type::SpecializedType { mangled_name, type_args, .. } => Some((mangled_name, type_args)),
            _ => None,
        }
    }

    /// Returns the mangled name for codegen use.
    /// For UnifiedType, returns the stored mangled_name.
    /// For Struct/Enum, returns the name as-is (which may be a mangled name or base name).
    pub fn mangled_name_or_name(&self) -> Option<&str> {
        match self {
            Type::Struct(name, _) | Type::Enum(name, _) => Some(name),
            Type::SpecializedType { mangled_name, .. } => Some(mangled_name),
            _ => None,
        }
    }

    /// Returns the effective name for codegen (mangled for UnifiedType, as-is for Struct/Enum).
    /// This is the name used for LLVM IR symbol generation and struct_defs lookup.
    pub fn codegen_name(&self) -> Option<String> {
        match self {
            Type::Struct(name, args) if !args.is_empty() => {
                // If has args, the codegen name would need mangling (caller should handle)
                Some(name.clone())
            }
            Type::Struct(name, _) | Type::Enum(name, _) => Some(name.clone()),
            Type::SpecializedType { mangled_name, .. } => {
                Some(mangled_name.clone())
            },
            _ => None,
        }
    }

    /// Check if this type is an enum (either Type::Enum or Type::UnifiedType with is_enum=true).
    pub fn is_enum_type(&self) -> bool {
        match self {
            Type::Enum(_, _) => true,
            Type::SpecializedType { gen_type, .. } => gen_type.is_enum_type(),
            _ => false,
        }
    }

    /// Check if this type is a struct (either Type::Struct or Type::UnifiedType with is_enum=false).
    pub fn is_struct_type(&self) -> bool {
        match self {
            Type::Struct(_, _) => true,
            Type::SpecializedType { gen_type, .. } => gen_type.is_struct_type(),
            _ => false,
        }
    }

    /// Flatten SpecializedType to Struct/Enum(mangled_name, type_args).
    /// This preserves both the mangled name (for LLVM symbol lookup) and the type_args
    /// (for element type access in codegen), while converting to a form that existing
    /// pattern matches can handle.
    /// Non-SpecializedType values are returned as-is.
    pub fn flatten_specialized(&self) -> Type {
        match self {
            Type::SpecializedType { gen_type, type_args, mangled_name, .. } => {
                if gen_type.is_enum_type() {
                    Type::Enum(mangled_name.clone(), type_args.clone())
                } else {
                    Type::Struct(mangled_name.clone(), type_args.clone())
                }
            }
            _ => self.clone(),
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
    pub generic_bounds: Vec<(String, Vec<TraitBound>)>, // <T: Trait1 + Trait2>
    pub where_clause: Option<WhereClause>,
    pub is_extern: bool,
    pub is_pub: bool,
    pub is_async: bool, // async fn
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<(String, Type)>,
    pub generics: Vec<String>, // <T>
    pub generic_bounds: Vec<(String, Vec<TraitBound>)>,
    pub where_clause: Option<WhereClause>,
    pub is_pub: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImplBlock {
    pub target_type: Type, // Changed from String to Type to generic support (e.g. Struct<T>)
    pub generics: Vec<String>,
    pub generic_bounds: Vec<(String, Vec<TraitBound>)>,
    pub where_clause: Option<WhereClause>,
    pub methods: Vec<FunctionDef>,
}

// ── Trait 関連 AST ノード ──

/// Trait定義: trait Iterator<T> { fn next(self) -> Option<T>; }
#[derive(Debug, Clone, PartialEq)]
pub struct TraitDef {
    pub name: String,
    pub generics: Vec<String>,
    pub methods: Vec<TraitMethodDef>,
    pub associated_types: Vec<String>, // 関連型名 (e.g. "Item")
    pub is_pub: bool,
}

/// トレイトメソッド定義（シグネチャ + オプショナルなデフォルト本体）
#[derive(Debug, Clone, PartialEq)]
pub struct TraitMethodDef {
    pub name: String,
    pub args: Vec<(String, Type)>,
    pub return_type: Type,
    pub has_self: bool,
    pub default_body: Option<Vec<Stmt>>, // デフォルトメソッド本体
}

/// impl Trait for Type ブロック
#[derive(Debug, Clone, PartialEq)]
pub struct TraitImplBlock {
    pub trait_name: String,
    pub trait_generics: Vec<Type>,       // impl Display for Vec<i64> の <i64>
    pub target_type: Type,               // 実装対象の型
    pub generics: Vec<String>,           // impl<T> の T
    pub generic_bounds: Vec<(String, Vec<TraitBound>)>,
    pub where_clause: Option<WhereClause>,
    pub methods: Vec<FunctionDef>,
    pub associated_types: Vec<(String, Type)>, // 関連型の具象化 (e.g. ("Item", I64))
}

/// トレイト境界: Display, Iterator<i64>, etc.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraitBound {
    pub trait_name: String,
    pub type_args: Vec<Type>,
}

/// where句
#[derive(Debug, Clone, PartialEq)]
pub struct WhereClause {
    pub predicates: Vec<WherePredicate>,
}

/// where句の述語: T: Display + Clone
#[derive(Debug, Clone, PartialEq)]
pub struct WherePredicate {
    pub type_param: String,
    pub bounds: Vec<TraitBound>,
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
    Array(Type, usize),  // [T; N] — fixed-size array variant
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumDef {
    pub name: String,
    pub variants: Vec<VariantDef>,
    pub generics: Vec<String>,
    pub is_pub: bool,
}

pub type Stmt = Spanned<StmtKind>;

#[derive(Debug, Clone, PartialEq)]
pub enum LValue {
    Variable(String),
    FieldAccess(Box<LValue>, String),
    IndexAccess(Box<LValue>, Vec<Expr>),
}

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
        lhs: LValue,
        op: AssignOp, // =, +=, max=, avg=
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
    Range(Option<Box<Expr>>, Option<Box<Expr>>), // start..end, start.., ..end, ..
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
    Try(Box<Expr>), // expr?
    Await(Box<Expr>), // expr.await

    // Calls
    FnCall(String, Vec<Expr>),
    MethodCall(Box<Expr>, String, Vec<Expr>),
    StaticMethodCall(Type, String, Vec<Expr>), // Type::method(args)
    StaticConstAccess(Type, String),           // Type::CONSTANT (e.g. f64::INFINITY)

    // Cast: expr as Type
    As(Box<Expr>, Type),
    TypeOf(Box<Expr>, Option<Type>),

    // Control
    IfExpr(Box<Expr>, Vec<Stmt>, Option<Vec<Stmt>>),
    IfLet {
        pattern: Pattern,
        expr: Box<Expr>,
        then_block: Vec<Stmt>,
        else_block: Option<Vec<Stmt>>,
    },
    Block(Vec<Stmt>),


    // Struct Init: Type { field: value, ... }
    StructInit(Type, Vec<(String, Expr)>),

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

    /// Closure expression: |args| body
    Closure {
        args: Vec<(String, Option<Type>)>,   // arg name + optional type annotation
        return_type: Option<Type>,            // optional explicit return type
        body: Vec<Stmt>,                      // body statements
        captures: Vec<(String, Type, bool)>,  // captured variables (name, type, is_mutable)
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
    BitAnd, // &
    BitOr,  // |
    BitXor, // ^
}



#[derive(Debug, Clone, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
    Query,
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
    pub traits: Vec<TraitDef>,
    pub trait_impls: Vec<TraitImplBlock>,
    pub functions: Vec<FunctionDef>,
    pub tensor_decls: Vec<Stmt>, // StmtKind::TensorDecl
    pub relations: Vec<RelationDecl>,
    pub rules: Vec<Rule>,
    pub queries: Vec<Expr>,
    pub imports: Vec<String>,
    pub submodules: HashMap<String, Module>,
}

impl Module {
    /// 空の Module を生成する。
    pub fn new() -> Self {
        Module {
            structs: Vec::new(),
            enums: Vec::new(),
            impls: Vec::new(),
            traits: Vec::new(),
            trait_impls: Vec::new(),
            functions: Vec::new(),
            tensor_decls: Vec::new(),
            relations: Vec::new(),
            rules: Vec::new(),
            queries: Vec::new(),
            imports: Vec::new(),
            submodules: HashMap::new(),
        }
    }

    /// `other` のすべてのフィールドを self に結合する。
    /// ビルトインのインジェクトやマルチファイルのマージに使用する。
    pub fn merge(&mut self, other: Module) {
        self.structs.extend(other.structs);
        self.enums.extend(other.enums);
        self.impls.extend(other.impls);
        self.traits.extend(other.traits);
        self.trait_impls.extend(other.trait_impls);
        self.functions.extend(other.functions);
        self.tensor_decls.extend(other.tensor_decls);
        self.relations.extend(other.relations);
        self.rules.extend(other.rules);
        self.queries.extend(other.queries);
        self.imports.extend(other.imports);
        self.submodules.extend(other.submodules);
    }
}

impl Default for Module {
    fn default() -> Self {
        Self::new()
    }
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
