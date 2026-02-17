//! Knowledge Base 推論エンジン
//!
//! Datalog 風の推論エンジン。半ナイーブ評価（Semi-Naive Evaluation）で
//! ファクトとルールから新しいファクトを導出し、クエリに応答する。

use crate::string_ffi::StringStruct;
use crate::OpaqueTensor;
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Mutex;
use tl_metal::{DType, MetalTensor};

// ========== データ型 ==========

/// KB 内の値
#[derive(Debug, Clone, PartialEq)]
enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Entity(i64),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(v) => write!(f, "{}", v),
            Value::Float(v) => write!(f, "{}", v),
            Value::Bool(v) => write!(f, "{}", v),
            Value::Str(v) => write!(f, "{}", v),
            Value::Entity(id) => write!(f, "e{}", id),
        }
    }
}

/// ルール引数の種類
#[derive(Debug, Clone)]
enum RuleArg {
    Var(i64),       // 変数（インデックス）
    ConstInt(i64),
    ConstFloat(f64),
    ConstEntity(i64),
}

/// ルール本体のアトム
#[derive(Debug, Clone)]
struct BodyAtom {
    predicate: String,
    args: Vec<RuleArg>,
    negated: bool,
}

/// ルール
#[derive(Debug, Clone)]
struct Rule {
    head_predicate: String,
    head_args: Vec<RuleArg>,
    body: Vec<BodyAtom>,
}

// ========== グローバル KB ストア ==========

struct KBStore {
    /// ファクトテーブル: predicate → Vec<Vec<Value>>
    facts: HashMap<String, Vec<Vec<Value>>>,
    /// ルール集合
    rules: Vec<Rule>,
    /// エンティティ名 → ID マッピング
    entity_names: HashMap<String, i64>,
    /// ID → エンティティ名
    entity_ids: HashMap<i64, String>,
    /// 次のエンティティ ID
    next_entity_id: i64,
    /// 推論完了フラグ
    inferred: bool,

    // === ビルダーバッファ ===
    /// ファクト引数バッファ
    fact_args_buffer: Vec<Value>,
    /// ルールビルダーの状態
    rule_builder: Option<RuleBuilder>,
}

struct RuleBuilder {
    head_predicate: String,
    head_args: Vec<RuleArg>,
    body: Vec<BodyAtom>,
    /// 現在構築中のボディアトム
    current_body_atom: Option<(String, Vec<RuleArg>, bool)>, // (predicate, args, negated)
}

impl KBStore {
    fn new() -> Self {
        KBStore {
            facts: HashMap::new(),
            rules: Vec::new(),
            entity_names: HashMap::new(),
            entity_ids: HashMap::new(),
            next_entity_id: 0,
            inferred: false,
            fact_args_buffer: Vec::new(),
            rule_builder: None,
        }
    }

    fn add_entity(&mut self, name: &str) -> i64 {
        if let Some(&id) = self.entity_names.get(name) {
            return id;
        }
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        self.entity_names.insert(name.to_string(), id);
        self.entity_ids.insert(id, name.to_string());
        id
    }

    fn add_fact(&mut self, predicate: &str, args: Vec<Value>) {
        let entry = self.facts.entry(predicate.to_string()).or_default();
        // 重複チェック
        if !entry.iter().any(|existing| existing == &args) {
            entry.push(args);
        }
    }

    /// 半ナイーブ評価でルールを適用し、固定点に達するまで繰り返す
    fn infer(&mut self) {
        if self.inferred {
            return;
        }

        let mut changed = true;
        let mut iteration = 0;
        const MAX_ITERATIONS: usize = 1000;

        while changed && iteration < MAX_ITERATIONS {
            changed = false;
            iteration += 1;

            for rule_idx in 0..self.rules.len() {
                let rule = self.rules[rule_idx].clone();
                let new_facts = self.apply_rule(&rule);

                for (pred, args) in new_facts {
                    let entry = self.facts.entry(pred).or_default();
                    if !entry.iter().any(|existing| existing == &args) {
                        entry.push(args);
                        changed = true;
                    }
                }
            }
        }

        self.inferred = true;
    }

    /// ルールを1つ適用し、導出される新しいファクトを返す
    fn apply_rule(&self, rule: &Rule) -> Vec<(String, Vec<Value>)> {
        let mut results = Vec::new();

        // 空のバインディングから開始
        let bindings = vec![HashMap::new()];

        // ボディの各アトムでバインディングを展開
        let final_bindings = self.solve_body(&rule.body, bindings);

        // 各バインディングからヘッドのファクトを生成
        for binding in &final_bindings {
            if let Some(args) = self.instantiate_args(&rule.head_args, binding) {
                results.push((rule.head_predicate.clone(), args));
            }
        }

        results
    }

    /// ルール本体のリテラル列を評価し、合致するバインディングのリストを返す
    fn solve_body(
        &self,
        body: &[BodyAtom],
        mut bindings: Vec<HashMap<i64, Value>>,
    ) -> Vec<HashMap<i64, Value>> {
        for atom in body {
            if bindings.is_empty() {
                break;
            }

            if is_builtin_predicate(&atom.predicate) {
                bindings = self.eval_builtin(atom, bindings);
            } else if atom.negated {
                // 否定: マッチするバインディングが見つからない場合のみ残す
                bindings = bindings
                    .into_iter()
                    .filter(|b| {
                        let test_bindings = self.match_atom(atom, vec![b.clone()]);
                        test_bindings.is_empty()
                    })
                    .collect();
            } else {
                bindings = self.match_atom(atom, bindings);
            }
        }
        bindings
    }

    /// アトムをファクトとマッチングし、バインディングを展開
    fn match_atom(
        &self,
        atom: &BodyAtom,
        bindings: Vec<HashMap<i64, Value>>,
    ) -> Vec<HashMap<i64, Value>> {
        let mut results = Vec::new();

        let facts = match self.facts.get(&atom.predicate) {
            Some(f) => f,
            None => return results,
        };

        for binding in &bindings {
            for fact_args in facts {
                if fact_args.len() != atom.args.len() {
                    continue;
                }
                if let Some(new_binding) = self.try_unify(&atom.args, fact_args, binding) {
                    results.push(new_binding);
                }
            }
        }

        results
    }

    /// 引数リストとファクトの値の統一を試みる
    fn try_unify(
        &self,
        rule_args: &[RuleArg],
        fact_values: &[Value],
        binding: &HashMap<i64, Value>,
    ) -> Option<HashMap<i64, Value>> {
        let mut new_binding = binding.clone();

        for (arg, val) in rule_args.iter().zip(fact_values.iter()) {
            match arg {
                RuleArg::Var(idx) => {
                    if let Some(bound_val) = new_binding.get(idx) {
                        if bound_val != val {
                            return None; // 矛盾
                        }
                    } else {
                        new_binding.insert(*idx, val.clone());
                    }
                }
                RuleArg::ConstInt(c) => {
                    if *val != Value::Int(*c) {
                        return None;
                    }
                }
                RuleArg::ConstFloat(c) => {
                    if *val != Value::Float(*c) {
                        return None;
                    }
                }
                RuleArg::ConstEntity(id) => {
                    if *val != Value::Entity(*id) {
                        return None;
                    }
                }
            }
        }

        Some(new_binding)
    }

    /// 組込み述語の評価
    fn eval_builtin(
        &self,
        atom: &BodyAtom,
        bindings: Vec<HashMap<i64, Value>>,
    ) -> Vec<HashMap<i64, Value>> {
        let pred = atom.predicate.as_str();

        match pred {
            "gt" | ">" => self.eval_comparison(atom, bindings, |a, b| a > b),
            "lt" | "<" => self.eval_comparison(atom, bindings, |a, b| a < b),
            "ge" | ">=" => self.eval_comparison(atom, bindings, |a, b| a >= b),
            "le" | "<=" => self.eval_comparison(atom, bindings, |a, b| a <= b),
            "eq" | "==" | "=:=" => self.eval_comparison(atom, bindings, |a, b| a == b),
            "ne" | "!=" | "=\\=" | "\\=" | "\\==" => {
                self.eval_comparison(atom, bindings, |a, b| a != b)
            }
            "add" => self.eval_arithmetic_3(atom, bindings, |a, b| a + b),
            "sub" => self.eval_arithmetic_3(atom, bindings, |a, b| a - b),
            "mul" => self.eval_arithmetic_3(atom, bindings, |a, b| a * b),
            "div" => self.eval_arithmetic_3(atom, bindings, |a, b| if b != 0 { a / b } else { 0 }),
            "mod" => self.eval_arithmetic_3(atom, bindings, |a, b| if b != 0 { a % b } else { 0 }),
            "neg" => self.eval_neg(atom, bindings),
            "is" => self.eval_is(atom, bindings),
            _ => bindings, // 未知の組込み述語はパススルー
        }
    }

    fn eval_comparison(
        &self,
        atom: &BodyAtom,
        bindings: Vec<HashMap<i64, Value>>,
        cmp: impl Fn(i64, i64) -> bool,
    ) -> Vec<HashMap<i64, Value>> {
        if atom.args.len() != 2 {
            return Vec::new();
        }

        bindings
            .into_iter()
            .filter(|b| {
                let left = self.resolve_arg(&atom.args[0], b);
                let right = self.resolve_arg(&atom.args[1], b);
                match (left, right) {
                    (Some(Value::Int(a)), Some(Value::Int(b))) => cmp(a, b),
                    (Some(Value::Entity(a)), Some(Value::Entity(b))) => cmp(a, b),
                    _ => false,
                }
            })
            .collect()
    }

    /// 3引数の算術述語: pred(a, b, result)
    fn eval_arithmetic_3(
        &self,
        atom: &BodyAtom,
        bindings: Vec<HashMap<i64, Value>>,
        op: impl Fn(i64, i64) -> i64,
    ) -> Vec<HashMap<i64, Value>> {
        if atom.args.len() != 3 {
            return Vec::new();
        }

        let mut results = Vec::new();

        for binding in bindings {
            let a = self.resolve_arg(&atom.args[0], &binding);
            let b = self.resolve_arg(&atom.args[1], &binding);

            if let (Some(Value::Int(av)), Some(Value::Int(bv))) = (a, b) {
                let result_val = op(av, bv);

                match &atom.args[2] {
                    RuleArg::Var(idx) => {
                        if let Some(bound) = binding.get(idx) {
                            if *bound == Value::Int(result_val) {
                                results.push(binding.clone());
                            }
                        } else {
                            let mut new_b = binding.clone();
                            new_b.insert(*idx, Value::Int(result_val));
                            results.push(new_b);
                        }
                    }
                    RuleArg::ConstInt(c) => {
                        if *c == result_val {
                            results.push(binding.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        results
    }

    fn eval_neg(
        &self,
        atom: &BodyAtom,
        bindings: Vec<HashMap<i64, Value>>,
    ) -> Vec<HashMap<i64, Value>> {
        if atom.args.len() != 2 {
            return Vec::new();
        }

        let mut results = Vec::new();

        for binding in bindings {
            let a = self.resolve_arg(&atom.args[0], &binding);

            if let Some(Value::Int(av)) = a {
                let neg_val = -av;

                match &atom.args[1] {
                    RuleArg::Var(idx) => {
                        if let Some(bound) = binding.get(idx) {
                            if *bound == Value::Int(neg_val) {
                                results.push(binding.clone());
                            }
                        } else {
                            let mut new_b = binding.clone();
                            new_b.insert(*idx, Value::Int(neg_val));
                            results.push(new_b);
                        }
                    }
                    RuleArg::ConstInt(c) => {
                        if *c == neg_val {
                            results.push(binding.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        results
    }

    fn eval_is(
        &self,
        atom: &BodyAtom,
        bindings: Vec<HashMap<i64, Value>>,
    ) -> Vec<HashMap<i64, Value>> {
        if atom.args.len() != 2 {
            return Vec::new();
        }

        let mut results = Vec::new();

        for binding in bindings {
            let rhs = self.resolve_arg(&atom.args[1], &binding);

            if let Some(rhs_val) = rhs {
                match &atom.args[0] {
                    RuleArg::Var(idx) => {
                        if let Some(bound) = binding.get(idx) {
                            if *bound == rhs_val {
                                results.push(binding.clone());
                            }
                        } else {
                            let mut new_b = binding.clone();
                            new_b.insert(*idx, rhs_val);
                            results.push(new_b);
                        }
                    }
                    _ => {
                        let lhs = self.resolve_arg(&atom.args[0], &binding);
                        if lhs == Some(rhs_val) {
                            results.push(binding.clone());
                        }
                    }
                }
            }
        }

        results
    }

    /// RuleArg をバインディングの値に解決
    fn resolve_arg(&self, arg: &RuleArg, binding: &HashMap<i64, Value>) -> Option<Value> {
        match arg {
            RuleArg::Var(idx) => binding.get(idx).cloned(),
            RuleArg::ConstInt(v) => Some(Value::Int(*v)),
            RuleArg::ConstFloat(v) => Some(Value::Float(*v)),
            RuleArg::ConstEntity(id) => Some(Value::Entity(*id)),
        }
    }

    /// RuleArg リストをバインディングで具体化
    fn instantiate_args(
        &self,
        args: &[RuleArg],
        binding: &HashMap<i64, Value>,
    ) -> Option<Vec<Value>> {
        let mut result = Vec::new();
        for arg in args {
            match self.resolve_arg(arg, binding) {
                Some(val) => result.push(val),
                None => return None, // 未束縛の変数
            }
        }
        Some(result)
    }

    /// クエリ実行: predicate(arg0, arg1, ...) でファクトを検索
    /// mask: ビットフラグで、各引数が指定されているか（1=指定、0=変数）
    /// tags: 各引数の型タグ (0=int/entity, 1=float, ...)
    fn query(
        &self,
        predicate: &str,
        mask: i64,
        args: Option<&[i64]>,
        _tags: *const u8,
    ) -> Vec<Vec<Value>> {
        let facts = match self.facts.get(predicate) {
            Some(f) => f,
            None => return Vec::new(),
        };

        if mask == 0 || args.is_none() {
            return facts.clone();
        }

        let args = args.unwrap();
        let mut results = Vec::new();

        for fact_args in facts {
            let mut matches = true;

            for (i, arg_val) in args.iter().enumerate() {
                if i >= fact_args.len() {
                    matches = false;
                    break;
                }

                if mask & (1 << i) != 0 {
                    // この引数は指定されている
                    let fact_v = &fact_args[i];
                    match fact_v {
                        Value::Int(v) => {
                            if *v != *arg_val {
                                matches = false;
                                break;
                            }
                        }
                        Value::Entity(id) => {
                            if *id != *arg_val {
                                matches = false;
                                break;
                            }
                        }
                        _ => {
                            matches = false;
                            break;
                        }
                    }
                }
            }

            if matches {
                results.push(fact_args.clone());
            }
        }

        results
    }
}

fn is_builtin_predicate(pred: &str) -> bool {
    matches!(
        pred,
        ">"  | "<"
            | ">="
            | "<="
            | "=="
            | "!="
            | "=:="
            | "=\\="
            | "\\="
            | "\\=="
            | "is"
            | "gt"
            | "lt"
            | "ge"
            | "le"
            | "eq"
            | "ne"
            | "add"
            | "sub"
            | "mul"
            | "div"
            | "mod"
            | "neg"
    )
}

// ========== グローバルインスタンス ==========

use once_cell::sync::Lazy;

static KB: Lazy<Mutex<KBStore>> = Lazy::new(|| Mutex::new(KBStore::new()));

// ========== KB Rule 関数 (コンパイラの builtins.rs に合わせたシグネチャ) ==========

/// ルール開始
/// コンパイラシグネチャ: tl_kb_rule_start(head_rel: *const i8)
#[unsafe(no_mangle)]
/// @ffi_sig (String*) -> void
pub extern "C" fn tl_kb_rule_start(pred_name: *mut StringStruct) {
    let name = unsafe { extract_string(pred_name as *const i8) };
    let mut store = KB.lock().unwrap();
    // 前のボディアトムがあればフラッシュ
    store.rule_builder = Some(RuleBuilder {
        head_predicate: name,
        head_args: Vec::new(),
        body: Vec::new(),
        current_body_atom: None,
    });
}

/// ルール終了
#[unsafe(no_mangle)]
/// @ffi_sig () -> void
pub extern "C" fn tl_kb_rule_finish() {
    let mut store = KB.lock().unwrap();
    if let Some(mut builder) = store.rule_builder.take() {
        // 現在のボディアトムをフラッシュ
        if let Some((pred, args, neg)) = builder.current_body_atom.take() {
            builder.body.push(BodyAtom {
                predicate: pred,
                args,
                negated: neg,
            });
        }

        store.rules.push(Rule {
            head_predicate: builder.head_predicate,
            head_args: builder.head_args,
            body: builder.body,
        });
    }
    store.inferred = false; // 再推論が必要
}

/// ルールヘッドに変数引数を追加
/// コンパイラシグネチャ: tl_kb_rule_add_head_arg_var(index: i64)
#[unsafe(no_mangle)]
/// @ffi_sig (i64) -> void
pub extern "C" fn tl_kb_rule_add_head_arg_var(idx: i64) {
    let mut store = KB.lock().unwrap();
    if let Some(ref mut builder) = store.rule_builder {
        builder.head_args.push(RuleArg::Var(idx));
    }
}

/// ルールヘッドに整数定数を追加
#[unsafe(no_mangle)]
pub extern "C" fn tl_kb_rule_add_head_arg_const_int(val: i64) {
    let mut store = KB.lock().unwrap();
    if let Some(ref mut builder) = store.rule_builder {
        builder.head_args.push(RuleArg::ConstInt(val));
    }
}

/// ルールヘッドに浮動小数点定数を追加
#[unsafe(no_mangle)]
/// @ffi_sig (f64) -> void
pub extern "C" fn tl_kb_rule_add_head_arg_const_float(val: f64) {
    let mut store = KB.lock().unwrap();
    if let Some(ref mut builder) = store.rule_builder {
        builder.head_args.push(RuleArg::ConstFloat(val));
    }
}

/// ルールヘッドにエンティティ定数を追加
#[unsafe(no_mangle)]
pub extern "C" fn tl_kb_rule_add_head_arg_const_entity(entity_id: i64) {
    let mut store = KB.lock().unwrap();
    if let Some(ref mut builder) = store.rule_builder {
        builder.head_args.push(RuleArg::ConstEntity(entity_id));
    }
}

/// ルールボディにアトムを追加
/// コンパイラシグネチャ: tl_kb_rule_add_body_atom(rel: *const i8)
#[unsafe(no_mangle)]
/// @ffi_sig (String*) -> void
pub extern "C" fn tl_kb_rule_add_body_atom(pred_name: *mut StringStruct) {
    let name = unsafe { extract_string(pred_name as *const i8) };
    let mut store = KB.lock().unwrap();
    if let Some(ref mut builder) = store.rule_builder {
        // 前のボディアトムをフラッシュ
        if let Some((pred, args, neg)) = builder.current_body_atom.take() {
            builder.body.push(BodyAtom {
                predicate: pred,
                args,
                negated: neg,
            });
        }
        builder.current_body_atom = Some((name, Vec::new(), false));
    }
}

/// ルールボディに否定アトムを追加
#[unsafe(no_mangle)]
/// @ffi_sig (String*) -> void
pub extern "C" fn tl_kb_rule_add_body_atom_neg(pred_name: *mut StringStruct) {
    let name = unsafe { extract_string(pred_name as *const i8) };
    let mut store = KB.lock().unwrap();
    if let Some(ref mut builder) = store.rule_builder {
        // 前のボディアトムをフラッシュ
        if let Some((pred, args, neg)) = builder.current_body_atom.take() {
            builder.body.push(BodyAtom {
                predicate: pred,
                args,
                negated: neg,
            });
        }
        builder.current_body_atom = Some((name, Vec::new(), true));
    }
}

/// ルールボディに変数引数を追加
/// コンパイラシグネチャ: tl_kb_rule_add_body_arg_var(index: i64)
#[unsafe(no_mangle)]
/// @ffi_sig (i64) -> void
pub extern "C" fn tl_kb_rule_add_body_arg_var(idx: i64) {
    let mut store = KB.lock().unwrap();
    if let Some(ref mut builder) = store.rule_builder {
        if let Some((_, ref mut args, _)) = builder.current_body_atom {
            args.push(RuleArg::Var(idx));
        }
    }
}

/// ルールボディに整数定数を追加
#[unsafe(no_mangle)]
/// @ffi_sig (i64) -> void
pub extern "C" fn tl_kb_rule_add_body_arg_const_int(val: i64) {
    let mut store = KB.lock().unwrap();
    if let Some(ref mut builder) = store.rule_builder {
        if let Some((_, ref mut args, _)) = builder.current_body_atom {
            args.push(RuleArg::ConstInt(val));
        }
    }
}

/// ルールボディに浮動小数点定数を追加
#[unsafe(no_mangle)]
/// @ffi_sig (f64) -> void
pub extern "C" fn tl_kb_rule_add_body_arg_const_float(val: f64) {
    let mut store = KB.lock().unwrap();
    if let Some(ref mut builder) = store.rule_builder {
        if let Some((_, ref mut args, _)) = builder.current_body_atom {
            args.push(RuleArg::ConstFloat(val));
        }
    }
}

/// ルールボディにエンティティ定数を追加
#[unsafe(no_mangle)]
/// @ffi_sig (i64) -> void
pub extern "C" fn tl_kb_rule_add_body_arg_const_entity(entity_id: i64) {
    let mut store = KB.lock().unwrap();
    if let Some(ref mut builder) = store.rule_builder {
        if let Some((_, ref mut args, _)) = builder.current_body_atom {
            args.push(RuleArg::ConstEntity(entity_id));
        }
    }
}

// ========== KB Fact 関数 ==========

/// ファクト引数の追加（旧 API — 未使用だが互換性のため残す）
#[unsafe(no_mangle)]
/// @ffi_sig (String*) -> void
pub extern "C" fn tl_kb_add_fact(_pred_name: *mut StringStruct) {
    // 旧 API — tl_kb_add_fact_serialized を使用
}

/// シリアライズされたファクト追加
/// コンパイラは fact_args_buffer に引数を積んでから、この関数を呼ぶ
/// コンパイラシグネチャ: tl_kb_add_fact_serialized(relation: *const i8) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_kb_add_fact_serialized(rel_name: *const i8) {
    let name = unsafe { extract_string(rel_name) };
    let mut store = KB.lock().unwrap();
    let args = store.fact_args_buffer.drain(..).collect::<Vec<_>>();
    store.add_fact(&name, args);
}

/// エンティティ追加
/// コンパイラシグネチャ: tl_kb_add_entity(name: *const i8) -> i64
#[unsafe(no_mangle)]
/// @ffi_sig (String*) -> i64
pub extern "C" fn tl_kb_add_entity(name: *mut StringStruct) -> i64 {
    let name_str = unsafe { extract_string(name as *const i8) };
    let mut store = KB.lock().unwrap();
    store.add_entity(&name_str)
}

/// ファクト引数クリア
#[unsafe(no_mangle)]
pub extern "C" fn tl_kb_fact_args_clear() {
    let mut store = KB.lock().unwrap();
    store.fact_args_buffer.clear();
}

/// ファクト引数に整数を追加
#[unsafe(no_mangle)]
/// @ffi_sig (i64) -> void
pub extern "C" fn tl_kb_fact_args_add_int(val: i64) {
    let mut store = KB.lock().unwrap();
    store.fact_args_buffer.push(Value::Int(val));
}

/// ファクト引数に浮動小数点を追加
#[unsafe(no_mangle)]
pub extern "C" fn tl_kb_fact_args_add_float(val: f64) {
    let mut store = KB.lock().unwrap();
    store.fact_args_buffer.push(Value::Float(val));
}

/// ファクト引数に文字列を追加
#[unsafe(no_mangle)]
/// @ffi_sig (String*) -> void
pub extern "C" fn tl_kb_fact_args_add_string(s: *mut StringStruct) {
    let name = unsafe { extract_string(s as *const i8) };
    let mut store = KB.lock().unwrap();
    store.fact_args_buffer.push(Value::Str(name));
}

/// ファクト引数にブール値を追加
#[unsafe(no_mangle)]
pub extern "C" fn tl_kb_fact_args_add_bool(val: bool) {
    let mut store = KB.lock().unwrap();
    store.fact_args_buffer.push(Value::Bool(val));
}

/// ファクト引数にエンティティを追加
#[unsafe(no_mangle)]
/// @ffi_sig (i64) -> void
pub extern "C" fn tl_kb_fact_args_add_entity(entity_id: i64) {
    let mut store = KB.lock().unwrap();
    store.fact_args_buffer.push(Value::Entity(entity_id));
}

// ========== KB 推論 ==========

/// 推論実行
/// コンパイラシグネチャ: tl_kb_infer() -> void
#[unsafe(no_mangle)]
/// @ffi_sig () -> void
pub extern "C" fn tl_kb_infer() {
    let mut store = KB.lock().unwrap();
    store.infer();
}

// ========== クエリ ==========

/// クエリ実行
/// コンパイラシグネチャ: tl_query(name: *i8, mask: i64, args: *Tensor, tags: *u8) -> *Tensor
#[unsafe(no_mangle)]
/// @ffi_sig (String*, i64, Tensor*, u8) -> Tensor*
pub extern "C" fn tl_query(
    name: *mut StringStruct,
    mask: i64,
    args_tensor: *mut OpaqueTensor,
    tags: *const u8,
) -> *mut OpaqueTensor {
    let pred_name = unsafe { extract_string(name as *const i8) };
    let is_cpu = std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu");

    let store = KB.lock().unwrap();

    // args_tensor から引数の i64 配列を取得 (CPU/GPU 両対応)
    let args_slice: Option<Vec<i64>> = if !args_tensor.is_null() {
        let data: Vec<f32> = if is_cpu {
            let tensor = unsafe { &*(args_tensor as *mut tl_cpu::CpuTensor) };
            tensor.data_f32().to_vec()
        } else {
            let tensor = unsafe { &*args_tensor };
            tensor.to_vec()
        };
        Some(data.iter().map(|&v| v as i64).collect())
    } else {
        None
    };

    let results = store.query(
        &pred_name,
        mask,
        args_slice.as_deref(),
        tags,
    );

    // CPU/GPU 両対応のテンソル作成ヘルパー
    let create_result_tensor = |data: &[f32], shape: &[usize]| -> *mut OpaqueTensor {
        if is_cpu {
            let t = tl_cpu::CpuTensor::from_slice(data, shape, tl_cpu::DType::F32);
            Box::into_raw(Box::new(t)) as *mut OpaqueTensor
        } else {
            let t = MetalTensor::from_slice(data, shape, DType::F32);
            crate::make_metal_tensor(t)
        }
    };

    if results.is_empty() {
        // false を返す — [0.0] のスカラーテンソル
        create_result_tensor(&[0.0], &[1])
    } else {
        // 結果エンコーディング:
        // Ground query (全引数指定): [1.0] (true)
        // Variable query (一部未指定): 変数値のリストをテンソルとして返す

        let total_args = results[0].len();
        let mut free_var_indices = Vec::new();
        for i in 0..total_args {
            if mask & (1 << i) == 0 {
                free_var_indices.push(i);
            }
        }

        if free_var_indices.is_empty() {
            // Ground query — 結果あり = true
            create_result_tensor(&[1.0], &[1])
        } else if free_var_indices.len() == 1 {
            // 1変数クエリ — 値のリストを返す
            let var_idx = free_var_indices[0];
            let data: Vec<f32> = results
                .iter()
                .map(|row| match &row[var_idx] {
                    Value::Int(v) => *v as f32,
                    Value::Float(v) => *v as f32,
                    Value::Entity(id) => *id as f32,
                    Value::Bool(b) => if *b { 1.0 } else { 0.0 },
                    Value::Str(_) => 0.0,
                })
                .collect();
            let shape = vec![data.len()];
            create_result_tensor(&data, &shape)
        } else {
            // 多変数クエリ — 2Dテンソルを返す
            let num_rows = results.len();
            let num_cols = free_var_indices.len();
            let mut data = Vec::with_capacity(num_rows * num_cols);
            for row in &results {
                for &col in &free_var_indices {
                    data.push(match &row[col] {
                        Value::Int(v) => *v as f32,
                        Value::Float(v) => *v as f32,
                        Value::Entity(id) => *id as f32,
                        Value::Bool(b) => if *b { 1.0 } else { 0.0 },
                        Value::Str(_) => 0.0,
                    });
                }
            }
            let shape = vec![num_rows, num_cols];
            create_result_tensor(&data, &shape)
        }
    }
}

// ========== ユーティリティ ==========

/// C 文字列ポインタから Rust String を取得
unsafe fn extract_string(ptr: *const i8) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}
