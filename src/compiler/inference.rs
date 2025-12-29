// src/compiler/inference.rs
//! Inference engine for Datalog-style logic rules.
//!
//! Supports:
//! - Unification of atoms
//! - Forward chaining (semi-naive evaluation)

use crate::compiler::ast::{Atom, Expr, Rule};
use std::collections::{HashMap, HashSet};

/// A tensor value stored in context (simplified representation)
#[derive(Debug, Clone)]
pub struct TensorValue {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl TensorValue {
    /// Get element at given indices
    pub fn get(&self, indices: &[i64]) -> Option<f64> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let mut flat_idx = 0usize;
        let mut stride = 1usize;
        for (&i, &dim) in indices.iter().rev().zip(self.shape.iter().rev()) {
            let idx = i as usize;
            if idx >= dim {
                return None;
            }
            flat_idx += idx * stride;
            stride *= dim;
        }
        self.data.get(flat_idx).copied()
    }
}

/// Context holding tensor values for hybrid computation
#[derive(Debug, Clone, Default)]
pub struct TensorContext {
    pub tensors: HashMap<String, TensorValue>,
}

impl TensorContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: String, tensor: TensorValue) {
        self.tensors.insert(name, tensor);
    }

    pub fn get(&self, name: &str) -> Option<&TensorValue> {
        self.tensors.get(name)
    }
}

/// A ground value (no variables).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
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
        match expr {
            Expr::Variable(name) => {
                // Check if variable is already bound
                if let Some(val) = subst.get(name) {
                    Term::Val(val.clone())
                } else {
                    Term::Var(name.clone())
                }
            }
            Expr::Int(n) => Term::Val(Value::Int(*n)),
            Expr::Float(f) => Term::Val(Value::Float(*f)),
            Expr::StringLiteral(s) => Term::Val(Value::Str(s.clone())),
            Expr::IndexAccess(base, indices) => {
                // Extract tensor name from base
                if let Expr::Variable(tensor_name) = base.as_ref() {
                    // Convert indices to terms
                    let idx_terms: Vec<Term> = indices
                        .iter()
                        .map(|idx_str| {
                            if let Ok(n) = idx_str.parse::<i64>() {
                                Term::Val(Value::Int(n))
                            } else if let Some(val) = subst.get(idx_str) {
                                Term::Val(val.clone())
                            } else {
                                Term::Var(idx_str.clone())
                            }
                        })
                        .collect();

                    // If all indices are ground and we have context, evaluate immediately
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

            // Try to resolve tensor access if all indices are grounded integers
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
            // Bind one to the other (arbitrary choice)
            if n1 != n2 {
                // We can't resolve this without a value; treat as compatible for now
                // In a full implementation, we'd use union-find or alias tracking
            }
            true
        }
        // TensorAccess: cannot unify directly if not resolved to a value
        (Term::TensorAccess(_, _), _) | (_, Term::TensorAccess(_, _)) => false,
    }
}

/// Unify an atom pattern with a ground atom.
/// Returns Some(substitution) if successful, None otherwise.
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
            Term::Var(_) => return None,             // Not fully ground
            Term::TensorAccess(_, _) => return None, // Unresolved tensor access
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
            // Find all substitutions that satisfy the body
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

/// A ground atom with an associated probability/weight (for probabilistic inference)
#[derive(Debug, Clone)]
pub struct WeightedGroundAtom {
    pub atom: GroundAtom,
    pub weight: f64,
}

/// Probabilistic forward chaining with max-product propagation.
/// Returns facts with their maximum probability weights.
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

            // Convert to HashSet for evaluate_rule
            let fact_set: HashSet<GroundAtom> = facts.keys().cloned().collect();
            let new_atoms = evaluate_rule(rule, &fact_set, ctx);

            for new_atom in new_atoms {
                // Calculate the weight of this derivation
                // For max-product: weight = rule_weight * product of body weights
                // Since we don't track the exact derivation path, use rule_weight
                // as a simplification (full implementation would track derivations)
                let new_weight = rule_weight;

                // Update if not present or if new weight is higher (max-product)
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

/// Check if a predicate is a built-in (comparison or numeric function)
fn is_builtin_predicate(name: &str) -> bool {
    matches!(
        name,
        "gt" | "lt" | "ge" | "le" | "eq" | "ne" | ">" | "<" | ">=" | "<=" | "==" | "!="
    )
}

/// Get numeric value from a Value
fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Int(n) => Some(*n as f64),
        Value::Float(f) => Some(*f),
        _ => None,
    }
}

/// Evaluate a built-in predicate with given arguments
/// Returns true if the predicate holds, false otherwise
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

/// Evaluate a single rule against current facts.
fn evaluate_rule(rule: &Rule, facts: &HashSet<GroundAtom>, ctx: &TensorContext) -> Vec<GroundAtom> {
    // Start with empty substitution
    let initial_substs = vec![Substitution::new()];

    // For each body atom, extend substitutions
    let mut substs = initial_substs;
    for body_atom in &rule.body {
        let mut new_substs = Vec::new();

        // Check if this is a built-in predicate
        if is_builtin_predicate(&body_atom.predicate) {
            // Evaluate built-in predicate for each existing substitution
            for subst in &substs {
                // Resolve arguments with current substitution
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
            // Normal predicate: match against facts
            for subst in &substs {
                for fact in facts {
                    if fact.predicate != body_atom.predicate {
                        continue;
                    }
                    let mut extended = subst.clone();
                    let pattern_terms = atom_to_terms(body_atom);
                    let mut matches = true;
                    for (term, value) in pattern_terms.iter().zip(fact.args.iter()) {
                        let ground_term = Term::Val(value.clone());
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

    // Apply substitutions to head
    let mut results = Vec::new();
    for subst in substs {
        if let Some(ground_head) = apply_subst_atom(&rule.head, &subst, ctx) {
            results.push(ground_head);
        }
    }
    results
}

/// Query: find all substitutions that make the goal true.
pub fn query(goal: &Atom, facts: &HashSet<GroundAtom>, ctx: &TensorContext) -> Vec<Substitution> {
    let mut results = Vec::new();
    for fact in facts {
        if let Some(subst) = unify_atom_with_ground(goal, fact, ctx) {
            results.push(subst);
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_atom(pred: &str, args: Vec<i64>) -> GroundAtom {
        GroundAtom {
            predicate: pred.to_string(),
            args: args.into_iter().map(Value::Int).collect(),
        }
    }

    #[test]
    fn test_unify_simple() {
        let pattern = Atom {
            predicate: "edge".to_string(),
            args: vec![Expr::Variable("X".to_string()), Expr::Int(2)],
        };
        let ground = make_atom("edge", vec![1, 2]);

        let ctx = TensorContext::new();
        let result = unify_atom_with_ground(&pattern, &ground, &ctx);
        assert!(result.is_some());
        let subst = result.unwrap();
        assert_eq!(subst.get("X"), Some(&Value::Int(1)));
    }

    #[test]
    fn test_forward_chain() {
        // edge(1, 2), edge(2, 3)
        // path(X, Y) :- edge(X, Y)
        // path(X, Z) :- edge(X, Y), path(Y, Z)
        // Should derive path(1, 2), path(2, 3), path(1, 3)

        let mut facts = HashSet::new();
        facts.insert(make_atom("edge", vec![1, 2]));
        facts.insert(make_atom("edge", vec![2, 3]));

        let rules = vec![
            Rule {
                head: Atom {
                    predicate: "path".to_string(),
                    args: vec![
                        Expr::Variable("X".to_string()),
                        Expr::Variable("Y".to_string()),
                    ],
                },
                body: vec![Atom {
                    predicate: "edge".to_string(),
                    args: vec![
                        Expr::Variable("X".to_string()),
                        Expr::Variable("Y".to_string()),
                    ],
                }],
                weight: None,
            },
            Rule {
                head: Atom {
                    predicate: "path".to_string(),
                    args: vec![
                        Expr::Variable("X".to_string()),
                        Expr::Variable("Z".to_string()),
                    ],
                },
                body: vec![
                    Atom {
                        predicate: "edge".to_string(),
                        args: vec![
                            Expr::Variable("X".to_string()),
                            Expr::Variable("Y".to_string()),
                        ],
                    },
                    Atom {
                        predicate: "path".to_string(),
                        args: vec![
                            Expr::Variable("Y".to_string()),
                            Expr::Variable("Z".to_string()),
                        ],
                    },
                ],
                weight: None,
            },
        ];

        let ctx = TensorContext::new();
        let result = forward_chain(facts, &rules, &ctx);

        assert!(result.contains(&make_atom("path", vec![1, 2])));
        assert!(result.contains(&make_atom("path", vec![2, 3])));
        assert!(result.contains(&make_atom("path", vec![1, 3])));
    }

    #[test]
    fn test_builtin_comparison() {
        // Test that built-in predicates work in rule evaluation
        // value(1, 10), value(2, 20), value(3, 5)
        // large(x) :- value(x, v), gt(v, 8)
        // Should derive large(1), large(2) but NOT large(3)

        let mut facts = HashSet::new();
        facts.insert(GroundAtom {
            predicate: "value".to_string(),
            args: vec![Value::Int(1), Value::Int(10)],
        });
        facts.insert(GroundAtom {
            predicate: "value".to_string(),
            args: vec![Value::Int(2), Value::Int(20)],
        });
        facts.insert(GroundAtom {
            predicate: "value".to_string(),
            args: vec![Value::Int(3), Value::Int(5)],
        });

        let rules = vec![Rule {
            head: Atom {
                predicate: "large".to_string(),
                args: vec![Expr::Variable("x".to_string())],
            },
            body: vec![
                Atom {
                    predicate: "value".to_string(),
                    args: vec![
                        Expr::Variable("x".to_string()),
                        Expr::Variable("v".to_string()),
                    ],
                },
                Atom {
                    predicate: "gt".to_string(),
                    args: vec![Expr::Variable("v".to_string()), Expr::Int(8)],
                },
            ],
            weight: None,
        }];

        let ctx = TensorContext::new();
        let result = forward_chain(facts, &rules, &ctx);

        // large(1) and large(2) should be derived
        assert!(result.contains(&GroundAtom {
            predicate: "large".to_string(),
            args: vec![Value::Int(1)],
        }));
        assert!(result.contains(&GroundAtom {
            predicate: "large".to_string(),
            args: vec![Value::Int(2)],
        }));
        // large(3) should NOT be derived because 5 <= 8
        assert!(!result.contains(&GroundAtom {
            predicate: "large".to_string(),
            args: vec![Value::Int(3)],
        }));
    }
}
