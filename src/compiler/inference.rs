// src/compiler/inference.rs
//! Inference engine for Datalog-style logic rules.
//!
//! Supports:
//! - Unification of atoms
//! - Forward chaining (semi-naive evaluation)

use crate::compiler::ast::{Atom, Expr, Rule};
use std::collections::{HashMap, HashSet};

/// A ground value (no variables).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Value {
    Int(i64),
    Float(String), // Store as string to allow hashing
    Str(String),
}

/// A term can be a variable or a ground value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    Var(String),
    Val(Value),
}

impl Term {
    /// Convert an Expr to a Term.
    /// Variables are identifiers, literals become Values.
    pub fn from_expr(expr: &Expr) -> Self {
        match expr {
            Expr::Variable(name) => Term::Var(name.clone()),
            Expr::Int(n) => Term::Val(Value::Int(*n)),
            Expr::Float(f) => Term::Val(Value::Float(f.to_string())),
            Expr::StringLiteral(s) => Term::Val(Value::Str(s.clone())),
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
fn apply_subst_term(term: &Term, subst: &Substitution) -> Term {
    match term {
        Term::Var(name) => {
            if let Some(val) = subst.get(name) {
                Term::Val(val.clone())
            } else {
                Term::Var(name.clone())
            }
        }
        Term::Val(v) => Term::Val(v.clone()),
    }
}

/// Convert Atom (from AST) to a list of Terms.
fn atom_to_terms(atom: &Atom) -> Vec<Term> {
    atom.args.iter().map(Term::from_expr).collect()
}

/// Try to unify two terms, extending the substitution.
fn unify_terms(t1: &Term, t2: &Term, subst: &mut Substitution) -> bool {
    let t1_resolved = apply_subst_term(t1, subst);
    let t2_resolved = apply_subst_term(t2, subst);

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
    }
}

/// Unify an atom pattern with a ground atom.
/// Returns Some(substitution) if successful, None otherwise.
pub fn unify_atom_with_ground(pattern: &Atom, ground: &GroundAtom) -> Option<Substitution> {
    if pattern.predicate != ground.predicate || pattern.args.len() != ground.args.len() {
        return None;
    }

    let terms = atom_to_terms(pattern);
    let mut subst = Substitution::new();

    for (term, value) in terms.iter().zip(ground.args.iter()) {
        let ground_term = Term::Val(value.clone());
        if !unify_terms(term, &ground_term, &mut subst) {
            return None;
        }
    }

    Some(subst)
}

/// Apply substitution to an Atom, producing a GroundAtom if fully ground.
pub fn apply_subst_atom(atom: &Atom, subst: &Substitution) -> Option<GroundAtom> {
    let mut ground_args = Vec::new();
    for expr in &atom.args {
        let term = Term::from_expr(expr);
        let resolved = apply_subst_term(&term, subst);
        match resolved {
            Term::Val(v) => ground_args.push(v),
            Term::Var(_) => return None, // Not fully ground
        }
    }
    Some(GroundAtom {
        predicate: atom.predicate.clone(),
        args: ground_args,
    })
}

/// Forward chaining: derive new facts from rules until fixpoint.
pub fn forward_chain(initial_facts: HashSet<GroundAtom>, rules: &[Rule]) -> HashSet<GroundAtom> {
    let mut facts = initial_facts;
    let mut changed = true;

    while changed {
        changed = false;
        for rule in rules {
            // Find all substitutions that satisfy the body
            let new_facts = evaluate_rule(rule, &facts);
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

/// Evaluate a single rule against current facts.
fn evaluate_rule(rule: &Rule, facts: &HashSet<GroundAtom>) -> Vec<GroundAtom> {
    // Start with empty substitution
    let initial_substs = vec![Substitution::new()];

    // For each body atom, extend substitutions
    let mut substs = initial_substs;
    for body_atom in &rule.body {
        let mut new_substs = Vec::new();
        for subst in &substs {
            // Find matching facts
            for fact in facts {
                if fact.predicate != body_atom.predicate {
                    continue;
                }
                // Try to extend substitution
                let mut extended = subst.clone();
                let pattern_terms = atom_to_terms(body_atom);
                let mut matches = true;
                for (term, value) in pattern_terms.iter().zip(fact.args.iter()) {
                    let ground_term = Term::Val(value.clone());
                    if !unify_terms(term, &ground_term, &mut extended) {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    new_substs.push(extended);
                }
            }
        }
        substs = new_substs;
    }

    // Apply substitutions to head
    let mut results = Vec::new();
    for subst in substs {
        if let Some(ground_head) = apply_subst_atom(&rule.head, &subst) {
            results.push(ground_head);
        }
    }
    results
}

/// Query: find all substitutions that make the goal true.
pub fn query(goal: &Atom, facts: &HashSet<GroundAtom>) -> Vec<Substitution> {
    let mut results = Vec::new();
    for fact in facts {
        if let Some(subst) = unify_atom_with_ground(goal, fact) {
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

        let result = unify_atom_with_ground(&pattern, &ground);
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
            },
        ];

        let result = forward_chain(facts, &rules);

        assert!(result.contains(&make_atom("path", vec![1, 2])));
        assert!(result.contains(&make_atom("path", vec![2, 3])));
        assert!(result.contains(&make_atom("path", vec![1, 3])));
    }
}
