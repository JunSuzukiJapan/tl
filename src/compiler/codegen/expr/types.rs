//! codegen/expr/types.rs
//!
//! TL コンパイラ codegen で使われるメソッド・組み込み関数の型エイリアス、
//! enum（BuiltinFn / InstanceMethod / StaticMethod）、および
//! 各マネージャー構造体（BuiltinManager / InstanceMethodManager / StaticMethodManager）を定義する。
use crate::compiler::error::TlError;

use std::collections::HashMap;

use inkwell::values::BasicValueEnum;

use crate::compiler::ast::{Expr, Type};
use crate::compiler::codegen::CodeGenerator;

// ── 型エイリアス ──────────────────────────────────────────────────────────────

/// 引数を事前評価済みの組み込み関数の fn ポインタ型。
pub type BuiltinFnEval = for<'a, 'ctx> fn(
    &'a mut CodeGenerator<'ctx>,
    Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError>;

/// 引数を未評価（AST ノードのまま）で受け取る組み込み関数の fn ポインタ型。
pub type BuiltinFnUneval = for<'a, 'ctx> fn(
    &'a mut CodeGenerator<'ctx>,
    &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), TlError>;

/// インスタンスメソッド（引数評価済み）の fn ポインタ型。
pub type InstanceMethodEval = for<'a, 'ctx> fn(
    &'a mut CodeGenerator<'ctx>,
    BasicValueEnum<'ctx>, // レシーバ値
    Type,                 // レシーバ型
    Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError>;

/// インスタンスメソッド（引数未評価）の fn ポインタ型。
pub type InstanceMethodUneval = for<'a, 'ctx> fn(
    &'a mut CodeGenerator<'ctx>,
    &Expr,   // レシーバ式
    &str,    // メソッド名
    &[Expr], // 引数
) -> Result<(BasicValueEnum<'ctx>, Type), TlError>;

/// 静的メソッド（引数評価済み）の fn ポインタ型。
pub type StaticMethodEval = for<'a, 'ctx> fn(
    &'a mut CodeGenerator<'ctx>,
    Vec<(BasicValueEnum<'ctx>, Type)>,
    Option<&Type>, // 型ヒント
) -> Result<(BasicValueEnum<'ctx>, Type), TlError>;

/// 静的メソッド（引数未評価）の fn ポインタ型。
pub type StaticMethodUneval = for<'a, 'ctx> fn(
    &'a mut CodeGenerator<'ctx>,
    &[Expr],       // 引数
    Option<&Type>, // 型ヒント
) -> Result<(BasicValueEnum<'ctx>, Type), TlError>;

// ── enum ──────────────────────────────────────────────────────────────────────

/// 組み込み関数の実装バリアント。
#[derive(Clone, Copy)]
pub enum BuiltinFn {
    Evaluated(BuiltinFnEval),
    Unevaluated(BuiltinFnUneval),
}

/// インスタンスメソッドの実装バリアント。
#[derive(Clone, Copy)]
pub enum InstanceMethod {
    Evaluated(InstanceMethodEval),
    Unevaluated(InstanceMethodUneval),
    /// シグネチャのみ（意味解析用）。実装なし。
    SignatureOnly,
}

/// 静的メソッドの実装バリアント。
#[derive(Clone, Copy)]
pub enum StaticMethod {
    Evaluated(StaticMethodEval),
    Unevaluated(StaticMethodUneval),
    /// シグネチャのみ（意味解析用）。実装なし。
    SignatureOnly,
}

// ── InstanceMethodManager ─────────────────────────────────────────────────────

/// 型に紐付くインスタンスメソッドを名前でルックアップするレジストリ。
pub struct InstanceMethodManager {
    methods: HashMap<String, InstanceMethod>,
}

impl InstanceMethodManager {
    pub fn new() -> Self {
        InstanceMethodManager { methods: HashMap::new() }
    }

    pub fn register_eval(&mut self, name: &str, func: InstanceMethodEval) {
        self.methods.insert(name.to_string(), InstanceMethod::Evaluated(func));
    }

    pub fn register_uneval(&mut self, name: &str, func: InstanceMethodUneval) {
        self.methods.insert(name.to_string(), InstanceMethod::Unevaluated(func));
    }

    pub fn get(&self, name: &str) -> Option<&InstanceMethod> {
        self.methods.get(name)
    }
}

// ── StaticMethodManager ───────────────────────────────────────────────────────

/// 型に紐付く静的メソッドを名前でルックアップするレジストリ。
pub struct StaticMethodManager {
    methods: HashMap<String, StaticMethod>,
}

impl StaticMethodManager {
    pub fn new() -> Self {
        StaticMethodManager { methods: HashMap::new() }
    }

    pub fn register_eval(&mut self, name: &str, func: StaticMethodEval) {
        self.methods.insert(name.to_string(), StaticMethod::Evaluated(func));
    }

    pub fn register_uneval(&mut self, name: &str, func: StaticMethodUneval) {
        self.methods.insert(name.to_string(), StaticMethod::Unevaluated(func));
    }

    pub fn get(&self, name: &str) -> Option<&StaticMethod> {
        self.methods.get(name)
    }
}

// ── BuiltinManager ────────────────────────────────────────────────────────────

/// グローバル組み込み関数（`print`, `panic` 等）を名前で管理するレジストリ。
/// `BuiltinManager::new()` を呼ぶと全組み込み関数が自動登録される。
pub struct BuiltinManager {
    pub(super) functions: HashMap<String, BuiltinFn>,
}

impl BuiltinManager {
    /// `BuiltinManager::new` は `mod.rs` 側で定義された `register_all` を呼ぶ。
    /// → `mod.rs` 側の `impl BuiltinManager` ブロックで `new()` を定義。
    pub fn get(&self, name: &str) -> Option<&BuiltinFn> {
        self.functions.get(name)
    }

    pub(super) fn register_eval(&mut self, name: &str, func: BuiltinFnEval) {
        self.functions.insert(name.to_string(), BuiltinFn::Evaluated(func));
    }

    pub(super) fn register_uneval(&mut self, name: &str, func: BuiltinFnUneval) {
        self.functions.insert(name.to_string(), BuiltinFn::Unevaluated(func));
    }
}
