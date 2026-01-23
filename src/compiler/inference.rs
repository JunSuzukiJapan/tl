// src/compiler/inference.rs
#![allow(dead_code)]
//! Inference engine for Datalog-style logic rules.

use crate::compiler::ast::{Atom, Expr, ExprKind, Rule};
use std::collections::{HashMap, HashSet};
pub use tl_runtime::context::{TensorContext, TensorValue};

/// A ground value (no variables).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Entity(i64),
}

// Implement Eq and Hash manually for Value with floats
impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::Int(n) => {
                0u8.hash(state);
                n.hash(state);
            }
            Value::Float(f) => {
                1u8.hash(state);
                f.to_bits().hash(state);
            }
            Value::Str(s) => {
                2u8.hash(state);
                s.hash(state);
            }
            Value::Bool(b) => {
                3u8.hash(state);
                b.hash(state);
            }
            Value::Entity(e) => {
                4u8.hash(state);
                e.hash(state);
            }
        }
    }
}

/// A term can be a variable or a ground value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    Var(String),
    Val(Value),
    TensorAccess(String, Vec<Term>), // tensor_name, indices
}

impl Term {
    /// Convert an Expr to a Term (without tensor context).
    pub fn from_expr(expr: &Expr) -> Self {
        Self::from_expr_with_context(expr, None, &HashMap::new())
    }

    /// Convert an Expr to a Term with tensor context for IndexAccess.
    pub fn from_expr_with_context(
        expr: &Expr,
        ctx: Option<&TensorContext>,
        subst: &Substitution,
    ) -> Self {
        match &expr.inner {
            ExprKind::Variable(name) => {
                if let Some(val) = subst.get(name) {
                    Term::Val(val.clone())
                } else {
                    Term::Var(name.clone())
                }
            }
            ExprKind::LogicVar(name) => {
                if let Some(val) = subst.get(name) {
                    Term::Val(val.clone())
                } else {
                    Term::Var(name.clone())
                }
            }
            ExprKind::Symbol(s) => Term::Val(Value::Str(s.clone())),
            ExprKind::Int(n) => Term::Val(Value::Int(*n)),
            ExprKind::Float(f) => Term::Val(Value::Float(*f)),
            ExprKind::Bool(b) => Term::Val(Value::Bool(*b)),
            ExprKind::StringLiteral(s) => Term::Val(Value::Str(s.clone())),
            ExprKind::IndexAccess(base, indices) => {
                if let ExprKind::Variable(tensor_name) = &base.inner {
                    let idx_terms: Vec<Term> = indices
                        .iter()
                        .map(|idx_expr| match &idx_expr.inner {
                            ExprKind::Int(n) => Term::Val(Value::Int(*n)),
                            ExprKind::Variable(name) => {
                                if let Some(val) = subst.get(name) {
                                    Term::Val(val.clone())
                                } else {
                                    Term::Var(name.clone())
                                }
                            }
                            _ => panic!(
                                "Unsupported expression in tensor index (inference): {:?}",
                                idx_expr
                            ),
                        })
                        .collect();

                    if let Some(tensor_ctx) = ctx {
                        let all_ground: Option<Vec<i64>> = idx_terms
                            .iter()
                            .map(|t| match t {
                                Term::Val(Value::Int(n)) => Some(*n),
                                _ => None,
                            })
                            .collect();

                        if let Some(ground_indices) = all_ground {
                            if let Some(tensor) = tensor_ctx.get(tensor_name) {
                                if let Some(val) = tensor.get(&ground_indices) {
                                    return Term::Val(Value::Float(val));
                                }
                            }
                        }
                    }
                    Term::TensorAccess(tensor_name.clone(), idx_terms)
                } else {
                    panic!("Unsupported tensor access base: {:?}", base)
                }
            }
            _ => panic!("Unsupported expression in logic term: {:?}", expr),
        }
    }
}

/// A ground atom (all arguments are values, no variables).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroundAtom {
    pub predicate: String,
    pub args: Vec<Value>,
}

/// Substitution: variable name -> Value
pub type Substitution = HashMap<String, Value>;

/// Apply substitution to a term.
fn apply_subst_term(term: &Term, subst: &Substitution, ctx: &TensorContext) -> Term {
    match term {
        Term::Var(name) => {
            if let Some(val) = subst.get(name) {
                Term::Val(val.clone())
            } else {
                Term::Var(name.clone())
            }
        }
        Term::Val(v) => Term::Val(v.clone()),
        Term::TensorAccess(name, indices) => {
            let resolved_indices: Vec<Term> = indices
                .iter()
                .map(|t| apply_subst_term(t, subst, ctx))
                .collect();

            let mut integer_indices = Vec::new();
            let mut all_ground_integers = true;

            for idx in &resolved_indices {
                if let Term::Val(Value::Int(i)) = idx {
                    integer_indices.push(*i);
                } else {
                    all_ground_integers = false;
                    break;
                }
            }

            if all_ground_integers {
                if let Some(tensor) = ctx.get(name) {
                    if let Some(val) = tensor.get(&integer_indices) {
                        return Term::Val(Value::Float(val));
                    }
                }
            }

            Term::TensorAccess(name.clone(), resolved_indices)
        }
    }
}

/// Convert Atom (from AST) to a list of Terms.
fn atom_to_terms(atom: &Atom) -> Vec<Term> {
    atom.args.iter().map(Term::from_expr).collect()
}

/// Try to unify two terms, extending the substitution.
fn unify_terms(t1: &Term, t2: &Term, subst: &mut Substitution, ctx: &TensorContext) -> bool {
    let t1_resolved = apply_subst_term(t1, subst, ctx);
    let t2_resolved = apply_subst_term(t2, subst, ctx);

    match (&t1_resolved, &t2_resolved) {
        (Term::Val(v1), Term::Val(v2)) => v1 == v2,
        (Term::Var(name), Term::Val(v)) | (Term::Val(v), Term::Var(name)) => {
            subst.insert(name.clone(), v.clone());
            true
        }
        (Term::Var(n1), Term::Var(n2)) => {
            if n1 != n2 {
                // Compatible but not grounded
            }
            true
        }
        (Term::TensorAccess(_, _), _) | (_, Term::TensorAccess(_, _)) => false,
    }
}

/// Unify an atom pattern with a ground atom.
pub fn unify_atom_with_ground(
    pattern: &Atom,
    ground: &GroundAtom,
    ctx: &TensorContext,
) -> Option<Substitution> {
    if pattern.predicate != ground.predicate || pattern.args.len() != ground.args.len() {
        return None;
    }

    let terms = atom_to_terms(pattern);
    let mut subst = Substitution::new();

    for (term, ground_val) in terms.iter().zip(ground.args.iter()) {
        let ground_term = Term::Val(ground_val.clone());
        if !unify_terms(term, &ground_term, &mut subst, ctx) {
            return None;
        }
    }

    Some(subst)
}

/// Apply substitution to an atom to produce a ground atom (if fully ground).
fn apply_subst_atom(atom: &Atom, subst: &Substitution, ctx: &TensorContext) -> Option<GroundAtom> {
    let mut ground_args = Vec::new();
    for expr in &atom.args {
        let term = Term::from_expr(expr);
        let resolved = apply_subst_term(&term, subst, ctx);
        match resolved {
            Term::Val(v) => ground_args.push(v),
            Term::Var(_) => return None,
            Term::TensorAccess(_, _) => return None,
        }
    }
    Some(GroundAtom {
        predicate: atom.predicate.clone(),
        args: ground_args,
    })
}

/// Forward chaining: derive new facts from rules until fixpoint.
pub fn forward_chain(
    initial_facts: HashSet<GroundAtom>,
    rules: &[Rule],
    ctx: &TensorContext,
) -> HashSet<GroundAtom> {
    let mut facts = initial_facts;
    let mut changed = true;

    while changed {
        changed = false;
        for rule in rules {
            let new_facts = evaluate_rule(rule, &facts, ctx);
            for fact in new_facts {
                if !facts.contains(&fact) {
                    facts.insert(fact);
                    changed = true;
                }
            }
        }
    }
    facts
}

pub fn probabilistic_forward_chain(
    initial_facts: HashMap<GroundAtom, f64>,
    rules: &[Rule],
    ctx: &TensorContext,
) -> HashMap<GroundAtom, f64> {
    let mut facts = initial_facts;
    let mut changed = true;

    while changed {
        changed = false;
        for rule in rules {
            let rule_weight = rule.weight.unwrap_or(1.0);
            let fact_set: HashSet<GroundAtom> = facts.keys().cloned().collect();
            let new_atoms = evaluate_rule(rule, &fact_set, ctx);

            for new_atom in new_atoms {
                let new_weight = rule_weight;
                let should_update = match facts.get(&new_atom) {
                    Some(&existing_weight) => new_weight > existing_weight,
                    None => true,
                };

                if should_update {
                    facts.insert(new_atom, new_weight);
                    changed = true;
                }
            }
        }
    }
    facts
}

fn is_builtin_predicate(name: &str) -> bool {
    matches!(
        name,
        "gt" | "lt" | "ge" | "le" | "eq" | "ne" | ">" | "<" | ">=" | "<=" | "==" | "!="
    )
}

fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Int(n) => Some(*n as f64),
        Value::Float(f) => Some(*f),
        _ => None,
    }
}

fn evaluate_builtin_predicate(pred: &str, args: &[Value]) -> bool {
    if args.len() != 2 {
        return false;
    }

    let left = match value_to_f64(&args[0]) {
        Some(v) => v,
        None => return false,
    };
    let right = match value_to_f64(&args[1]) {
        Some(v) => v,
        None => return false,
    };

    match pred {
        "gt" | ">" => left > right,
        "lt" | "<" => left < right,
        "ge" | ">=" => left >= right,
        "le" | "<=" => left <= right,
        "eq" | "==" => (left - right).abs() < f64::EPSILON,
        "ne" | "!=" => (left - right).abs() >= f64::EPSILON,
        _ => false,
    }
}

struct FactIndex<'a> {
    facts_by_pred: HashMap<&'a str, Vec<&'a GroundAtom>>,
}

impl<'a> FactIndex<'a> {
    fn new(facts: &'a HashSet<GroundAtom>) -> Self {
        let mut map = HashMap::new();
        for fact in facts {
            map.entry(fact.predicate.as_str())
                .or_insert_with(Vec::new)
                .push(fact);
        }
        Self { facts_by_pred: map }
    }

    fn candidates(&self, pred: &str) -> Vec<&'a GroundAtom> {
        self.facts_by_pred.get(pred).cloned().unwrap_or_default()
    }
}

fn evaluate_rule(rule: &Rule, facts: &HashSet<GroundAtom>, ctx: &TensorContext) -> Vec<GroundAtom> {
    let index = FactIndex::new(facts);
    let mut substs = vec![Substitution::new()];

    for body_atom in &rule.body {
        let mut new_substs = Vec::new();
        if is_builtin_predicate(&body_atom.predicate) {
            for subst in &substs {
                let mut resolved_args = Vec::new();
                let mut all_ground = true;
                for expr in &body_atom.args {
                    let term = Term::from_expr(expr);
                    let resolved = apply_subst_term(&term, subst, ctx);
                    match resolved {
                        Term::Val(v) => resolved_args.push(v),
                        _ => {
                            all_ground = false;
                            break;
                        }
                    }
                }
                if all_ground && evaluate_builtin_predicate(&body_atom.predicate, &resolved_args) {
                    new_substs.push(subst.clone());
                }
            }
        } else {
            let candidates = index.candidates(&body_atom.predicate);
            for subst in &substs {
                for fact in candidates.iter() {
                    let mut extended = subst.clone();
                    let pattern_terms = atom_to_terms(body_atom);
                    let mut matches = true;
                    for (term, ground_val) in pattern_terms.iter().zip(fact.args.iter()) {
                        let ground_term = Term::Val(ground_val.clone());
                        if !unify_terms(term, &ground_term, &mut extended, ctx) {
                            matches = false;
                            break;
                        }
                    }
                    if matches {
                        new_substs.push(extended);
                    }
                }
            }
        }
        substs = new_substs;
    }

    let mut results = Vec::new();
    for subst in substs {
        if let Some(ground_head) = apply_subst_atom(&rule.head, &subst, ctx) {
            results.push(ground_head);
        }
    }
    results
}

pub fn query(goal: &Atom, facts: &HashSet<GroundAtom>, ctx: &TensorContext) -> Vec<Substitution> {
    let mut results = Vec::new();
    for fact in facts {
        if let Some(subst) = unify_atom_with_ground(goal, fact, ctx) {
            results.push(subst);
        }
    }
    results
}
