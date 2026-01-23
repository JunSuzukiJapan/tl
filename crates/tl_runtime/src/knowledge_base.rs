// use crate::{make_tensor, OpaqueTensor};
// use candle_core::{DType, Device, Tensor};
use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::os::raw::c_char; // Removed c_void
use std::sync::Mutex;

// Global KB Registry (Singleton for simplicity in this POC)
// In a real implementation, we might pass a KB pointer around.
static GLOBAL_KB: Lazy<Mutex<KnowledgeBase>> = Lazy::new(|| Mutex::new(KnowledgeBase::new()));

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
    Char(u32),
    String(String),
    Entity(i64), // Internal ID for symbolic atoms
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstantTag {
    Int = 0,
    Float = 1,
    Bool = 2,
    Entity = 3,
    String = 4,
}

impl std::hash::Hash for Constant {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Constant::Int(i) => {
                0u8.hash(state);
                i.hash(state);
            }
            Constant::Float(f) => {
                1u8.hash(state);
                f.to_bits().hash(state);
            }
            Constant::Bool(b) => {
                2u8.hash(state);
                b.hash(state);
            }
            Constant::Char(c) => {
                3u8.hash(state);
                c.hash(state);
            }
            Constant::String(s) => {
                4u8.hash(state);
                s.hash(state);
            }
            Constant::Entity(e) => {
                5u8.hash(state);
                e.hash(state);
            }
        }
    }
}

impl Eq for Constant {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tuple(pub Vec<Constant>);

pub struct KnowledgeBase {
    // facts: Relation Name -> Set of Tuples
    facts: HashMap<String, HashSet<Tuple>>,
    // rules: Relation Name -> List of Rules defining it
    rules: HashMap<String, Vec<RuleImpl>>,
    // entities: Name -> ID
    entities: HashMap<String, i64>,
    // id_to_name: ID -> Name (Reverse lookup for display)
    id_to_name: HashMap<i64, String>,
    entity_count: i64,
}

impl KnowledgeBase {
    pub fn new() -> Self {
        Self {
            facts: HashMap::new(),
            rules: HashMap::new(),
            entities: HashMap::new(),
            id_to_name: HashMap::new(),
            entity_count: 0,
        }
    }

    pub fn get_entity_id(&mut self, name: &str) -> i64 {
        if let Some(&id) = self.entities.get(name) {
            return id;
        }
        let id = self.entity_count;
        self.entities.insert(name.to_string(), id);
        self.id_to_name.insert(id, name.to_string());
        self.entity_count += 1;
        id
    }

    pub fn get_entity_name(&self, id: i64) -> Option<&str> {
        self.id_to_name.get(&id).map(|s| s.as_str())
    }

    pub fn add_fact(&mut self, relation: &str, args: Vec<Constant>) {
        self.facts
            .entry(relation.to_string())
            .or_insert_with(HashSet::new)
            .insert(Tuple(args));
    }

    pub fn add_rule(&mut self, head: String, rule: RuleImpl) {
        self.rules.entry(head).or_insert_with(Vec::new).push(rule);
    }

    pub fn infer(&mut self) {
        let mut changed = true;
        let mut _iteration = 0;

        while changed {
            changed = false;
            _iteration += 1;

            // Collect all new facts first to avoid borrowing issues
            let mut new_facts: Vec<(String, Tuple)> = Vec::new();

            for (head_rel, rules) in &self.rules {
                for rule in rules {
                    // Evaluate rule
                    let derived = self.evaluate_rule(rule);
                    for tuple in derived {
                        if !self.has_fact(head_rel, &tuple) {
                            new_facts.push((head_rel.clone(), tuple));
                        }
                    }
                }
            }

            for (rel, tuple) in new_facts {
                if self.facts.entry(rel).or_default().insert(tuple) {
                    changed = true;
                }
            }
        }
    }

    fn has_fact(&self, rel: &str, tuple: &Tuple) -> bool {
        self.facts.get(rel).map_or(false, |s| s.contains(tuple))
    }

    fn evaluate_rule(&self, rule: &RuleImpl) -> Vec<Tuple> {
        // Initial bindings: empty
        let mut bindings: Vec<HashMap<usize, Constant>> = vec![HashMap::new()];

        for atom in &rule.body {
            let mut new_bindings = Vec::new();
            if let Some(facts) = self.facts.get(&atom.relation) {
                for binding in bindings {
                    for fact in facts {
                        // Check if fact matches current binding & atom constraint
                        if let Some(extended_binding) = self.unify(&binding, atom, &fact.0) {
                            new_bindings.push(extended_binding);
                        }
                    }
                }
            }
            bindings = new_bindings;
            if bindings.is_empty() {
                break;
            }
        }

        // Construct derived tuples from bindings
        let mut results = Vec::new();
        for binding in bindings {
            let mut tuple_args = Vec::new();
            for arg in &rule.head_args {
                let val = match arg {
                    Arg::Var(v) => binding.get(v).cloned().unwrap_or(Constant::Bool(false)),
                    Arg::Const(c) => c.clone(),
                };
                tuple_args.push(val);
            }
            results.push(Tuple(tuple_args));
        }

        results
    }

    fn unify(
        &self,
        current_binding: &HashMap<usize, Constant>,
        atom: &BodyAtom,
        fact_args: &[Constant],
    ) -> Option<HashMap<usize, Constant>> {
        if atom.args.len() != fact_args.len() {
            return None;
        }

        let mut new_binding = current_binding.clone();

        for (i, arg) in atom.args.iter().enumerate() {
            let fact_val = &fact_args[i];
            match arg {
                Arg::Const(c) => {
                    if c != fact_val {
                        return None;
                    }
                }
                Arg::Var(v) => {
                    if let Some(bound_val) = new_binding.get(v) {
                        if bound_val != fact_val {
                            return None; // Conflict
                        }
                    } else {
                        new_binding.insert(*v, fact_val.clone());
                    }
                }
            }
        }

        Some(new_binding)
    }

    pub fn query(&self, relation: &str, args: &[Constant], mask: i64) -> Vec<Vec<Constant>> {
        let mut results = Vec::new();
        if let Some(tuples) = self.facts.get(relation) {
            for tuple in tuples {
                if self.tuple_matches(tuple, args, mask) {
                    let extracted = self.extract_vars(tuple, mask);
                    results.push(extracted);
                }
            }
        }
        results
    }

    fn tuple_matches(&self, tuple: &Tuple, args: &[Constant], mask: i64) -> bool {
        if tuple.0.len() != args.len() {
            return false;
        }
        for (i, val) in tuple.0.iter().enumerate() {
            let is_var = (mask >> i) & 1 == 1;
            if !is_var {
                if val != &args[i] {
                    return false;
                }
            }
        }
        true
    }

    fn extract_vars(&self, tuple: &Tuple, mask: i64) -> Vec<Constant> {
        let mut vars = Vec::new();
        for (i, val) in tuple.0.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                vars.push(val.clone());
            }
        }
        vars
    }
}

pub struct RuleImpl {
    pub head_args: Vec<Arg>,
    pub body: Vec<BodyAtom>,
}

pub struct BodyAtom {
    pub relation: String,
    pub args: Vec<Arg>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    Var(usize),
    Const(Constant),
}

// --- C ABI ---

#[no_mangle]
pub extern "C" fn tl_kb_add_entity(name: *const c_char) -> i64 {
    let c_str = unsafe { CStr::from_ptr(name) };
    let r_str = c_str.to_str().unwrap();
    GLOBAL_KB.lock().unwrap().get_entity_id(r_str)
}

// Serialization state for fact construction
static GLOBAL_FACT_ARGS: Lazy<Mutex<Vec<Constant>>> = Lazy::new(|| Mutex::new(Vec::new()));

#[no_mangle]
pub extern "C" fn tl_kb_fact_args_clear() {
    GLOBAL_FACT_ARGS.lock().unwrap().clear();
}

#[no_mangle]
pub extern "C" fn tl_kb_fact_args_add_int(val: i64) {
    GLOBAL_FACT_ARGS.lock().unwrap().push(Constant::Int(val));
}

#[no_mangle]
pub extern "C" fn tl_kb_fact_args_add_float(val: f64) {
    GLOBAL_FACT_ARGS.lock().unwrap().push(Constant::Float(val));
}

#[no_mangle]
pub extern "C" fn tl_kb_fact_args_add_bool(val: bool) {
    GLOBAL_FACT_ARGS.lock().unwrap().push(Constant::Bool(val));
}

#[no_mangle]
pub extern "C" fn tl_kb_fact_args_add_entity(id: i64) {
    GLOBAL_FACT_ARGS.lock().unwrap().push(Constant::Entity(id));
}

#[no_mangle]
pub extern "C" fn tl_kb_fact_args_add_string(ptr: *const c_char) {
    let c_str = unsafe { CStr::from_ptr(ptr) };
    let r_str = c_str.to_str().unwrap().to_string();
    GLOBAL_FACT_ARGS.lock().unwrap().push(Constant::String(r_str));
}

#[no_mangle]
pub extern "C" fn tl_kb_add_fact_serialized(relation: *const c_char) {
    let c_str = unsafe { CStr::from_ptr(relation) };
    let rel_name = c_str.to_str().unwrap();
    let args = std::mem::take(&mut *GLOBAL_FACT_ARGS.lock().unwrap());

    GLOBAL_KB.lock().unwrap().add_fact(rel_name, args);
}

// Keep old API for compatibility if needed, but it only supports i64 (Entities/Ints)
#[no_mangle]
pub extern "C" fn tl_kb_add_fact(relation: *const c_char, args: *const i64, arity: i64) {
    let c_str = unsafe { CStr::from_ptr(relation) };
    let rel_name = c_str.to_str().unwrap();
    let args_slice = unsafe { std::slice::from_raw_parts(args, arity as usize) };
    let args_vec = args_slice.iter().map(|&x| Constant::Int(x)).collect();

    GLOBAL_KB.lock().unwrap().add_fact(rel_name, args_vec);
}

#[no_mangle]
pub extern "C" fn tl_kb_get_entity_name(id: i64) -> *const c_char {
    let kb = GLOBAL_KB.lock().unwrap();
    if let Some(name) = kb.get_entity_name(id) {
        std::ffi::CString::new(name).unwrap().into_raw()
    } else {
        std::ptr::null()
    }
}

#[no_mangle]
pub extern "C" fn tl_kb_infer() {
    GLOBAL_KB.lock().unwrap().infer();
}

// --- Rule Builder API ---

struct RuleBuilder {
    head_rel: String,
    head_args: Vec<Arg>,
    body: Vec<BodyAtom>,
    current_body_atom_rel: Option<String>,
    current_body_atom_args: Vec<Arg>,
}

static GLOBAL_RULE_BUILDER: Lazy<Mutex<Option<RuleBuilder>>> = Lazy::new(|| Mutex::new(None));

#[no_mangle]
pub extern "C" fn tl_kb_rule_start(head_rel: *const c_char) {
    let c_str = unsafe { CStr::from_ptr(head_rel) };
    let r_str = c_str.to_str().unwrap().to_string();

    let mut builder = GLOBAL_RULE_BUILDER.lock().unwrap();
    *builder = Some(RuleBuilder {
        head_rel: r_str,
        head_args: Vec::new(),
        body: Vec::new(),
        current_body_atom_rel: None,
        current_body_atom_args: Vec::new(),
    });
}

#[no_mangle]
pub extern "C" fn tl_kb_rule_add_head_arg_var(index: i64) {
    let mut builder_lock = GLOBAL_RULE_BUILDER.lock().unwrap();
    if let Some(builder) = builder_lock.as_mut() {
        builder.head_args.push(Arg::Var(index as usize));
    }
}

#[no_mangle]
pub extern "C" fn tl_kb_rule_add_head_arg_const_int(val: i64) {
    let mut builder_lock = GLOBAL_RULE_BUILDER.lock().unwrap();
    if let Some(builder) = builder_lock.as_mut() {
        builder.head_args.push(Arg::Const(Constant::Int(val)));
    }
}

#[no_mangle]
pub extern "C" fn tl_kb_rule_add_head_arg_const_float(val: f64) {
    let mut builder_lock = GLOBAL_RULE_BUILDER.lock().unwrap();
    if let Some(builder) = builder_lock.as_mut() {
        builder.head_args.push(Arg::Const(Constant::Float(val)));
    }
}

#[no_mangle]
pub extern "C" fn tl_kb_rule_add_head_arg_const_entity(id: i64) {
    let mut builder_lock = GLOBAL_RULE_BUILDER.lock().unwrap();
    if let Some(builder) = builder_lock.as_mut() {
        builder.head_args.push(Arg::Const(Constant::Entity(id)));
    }
}

#[no_mangle]
pub extern "C" fn tl_kb_rule_add_body_atom(rel: *const c_char) {
    let c_str = unsafe { CStr::from_ptr(rel) };
    let r_str = c_str.to_str().unwrap().to_string();

    let mut builder_lock = GLOBAL_RULE_BUILDER.lock().unwrap();
    if let Some(builder) = builder_lock.as_mut() {
        if let Some(prev_rel) = builder.current_body_atom_rel.take() {
            let prev_args = std::mem::take(&mut builder.current_body_atom_args);
            builder.body.push(BodyAtom {
                relation: prev_rel,
                args: prev_args,
            });
        }
        builder.current_body_atom_rel = Some(r_str);
    }
}

#[no_mangle]
pub extern "C" fn tl_kb_rule_add_body_arg_var(index: i64) {
    let mut builder_lock = GLOBAL_RULE_BUILDER.lock().unwrap();
    if let Some(builder) = builder_lock.as_mut() {
        builder.current_body_atom_args.push(Arg::Var(index as usize));
    }
}

#[no_mangle]
pub extern "C" fn tl_kb_rule_add_body_arg_const_int(val: i64) {
    let mut builder_lock = GLOBAL_RULE_BUILDER.lock().unwrap();
    if let Some(builder) = builder_lock.as_mut() {
        builder.current_body_atom_args.push(Arg::Const(Constant::Int(val)));
    }
}

#[no_mangle]
pub extern "C" fn tl_kb_rule_add_body_arg_const_float(val: f64) {
    let mut builder_lock = GLOBAL_RULE_BUILDER.lock().unwrap();
    if let Some(builder) = builder_lock.as_mut() {
        builder.current_body_atom_args.push(Arg::Const(Constant::Float(val)));
    }
}

#[no_mangle]
pub extern "C" fn tl_kb_rule_add_body_arg_const_entity(id: i64) {
    let mut builder_lock = GLOBAL_RULE_BUILDER.lock().unwrap();
    if let Some(builder) = builder_lock.as_mut() {
        builder.current_body_atom_args.push(Arg::Const(Constant::Entity(id)));
    }
}

#[no_mangle]
pub extern "C" fn tl_kb_rule_finish() {
    let mut builder_lock = GLOBAL_RULE_BUILDER.lock().unwrap();
    if let Some(mut builder) = builder_lock.take() {
        if let Some(prev_rel) = builder.current_body_atom_rel.take() {
            let prev_args = std::mem::take(&mut builder.current_body_atom_args);
            builder.body.push(BodyAtom {
                relation: prev_rel,
                args: prev_args,
            });
        }
        let rule = RuleImpl {
            head_args: builder.head_args,
            body: builder.body,
        };
        GLOBAL_KB.lock().unwrap().add_rule(builder.head_rel, rule);
    }
}

pub fn perform_kb_query(relation: &str, args: &[Constant], mask: i64) -> Vec<Vec<Constant>> {
    GLOBAL_KB.lock().unwrap().query(relation, args, mask)
}
