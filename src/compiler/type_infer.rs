// src/compiler/type_infer.rs
#![allow(dead_code)]
//! Constraint-based type inference system with shape inference for tensors.

use crate::compiler::ast::{Dim, Expr, ExprKind, FunctionDef, Module, Stmt, StmtKind, Type};
use std::collections::HashMap;

/// Type constraint for unification
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Two types must be equal
    TypeEq(Type, Type),
    /// Two dimensions must be equal
    DimEq(Dim, Dim),
}

/// Type inference error
#[derive(Debug, Clone)]
pub enum TypeError {
    UnificationFailed(Type, Type),
    DimensionMismatch(Dim, Dim),
    UnboundVariable(String),
    ShapeMismatch(Vec<Dim>, Vec<Dim>),
    Other(String),
}

/// Type inference context
pub struct TypeInferencer {
    /// Next fresh type variable ID
    next_type_var: u32,
    /// Next fresh dimension variable ID
    next_dim_var: u32,
    /// Collected constraints
    constraints: Vec<Constraint>,
    /// Type variable substitution
    type_subst: HashMap<u32, Type>,
    /// Dimension variable substitution
    dim_subst: HashMap<u32, Dim>,
    /// Variable types in scope
    var_types: HashMap<String, Type>,
}

impl TypeInferencer {
    pub fn new() -> Self {
        Self {
            next_type_var: 0,
            next_dim_var: 0,
            constraints: Vec::new(),
            type_subst: HashMap::new(),
            dim_subst: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    /// Generate a fresh type variable
    pub fn fresh_type_var(&mut self) -> Type {
        let id = self.next_type_var;
        self.next_type_var += 1;
        Type::TypeVar(id)
    }

    /// Generate a fresh dimension variable
    pub fn fresh_dim_var(&mut self) -> Dim {
        let id = self.next_dim_var;
        self.next_dim_var += 1;
        Dim::Var(id)
    }

    /// Add a type equality constraint
    pub fn add_type_eq(&mut self, t1: Type, t2: Type) {
        self.constraints.push(Constraint::TypeEq(t1, t2));
    }

    /// Add a dimension equality constraint
    pub fn add_dim_eq(&mut self, d1: Dim, d2: Dim) {
        self.constraints.push(Constraint::DimEq(d1, d2));
    }

    /// Run type inference on a module
    pub fn infer_module(&mut self, module: &Module) -> Result<(), TypeError> {
        // Collect constraints from all functions
        for func in &module.functions {
            self.infer_function(func)?;
        }

        // Solve constraints
        self.unify()?;

        Ok(())
    }

    /// Collect constraints from a function
    fn infer_function(&mut self, func: &FunctionDef) -> Result<Type, TypeError> {
        // Add arguments to scope
        for (name, ty) in &func.args {
            self.var_types.insert(name.clone(), ty.clone());
        }

        // Infer body statements
        let mut last_type = Type::Void;
        for stmt in &func.body {
            last_type = self.infer_stmt(stmt)?;
        }

        // Return type constraint
        self.add_type_eq(last_type.clone(), func.return_type.clone());

        Ok(last_type)
    }

    /// Infer type from a statement
    fn infer_stmt(&mut self, stmt: &Stmt) -> Result<Type, TypeError> {
        match &stmt.inner {
            StmtKind::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                let inferred = self.infer_expr(value)?;
                if let Some(declared) = type_annotation {
                    self.add_type_eq(inferred.clone(), declared.clone());
                }
                self.var_types.insert(name.clone(), inferred.clone());
                Ok(inferred)
            }
            StmtKind::Assign { name, value, .. } => {
                // Lookup target variable type
                let target_ty = self
                    .var_types
                    .get(name)
                    .cloned()
                    .ok_or_else(|| TypeError::UnboundVariable(name.clone()))?;
                let value_ty = self.infer_expr(value)?;
                self.add_type_eq(target_ty, value_ty.clone());
                Ok(value_ty)
            }
            StmtKind::Return(expr_opt) => {
                if let Some(expr) = expr_opt {
                    self.infer_expr(expr)
                } else {
                    Ok(Type::Void)
                }
            }
            StmtKind::Expr(expr) => self.infer_expr(expr),
            StmtKind::While { cond, body, .. } => {
                let cond_ty = self.infer_expr(cond)?;
                self.add_type_eq(cond_ty, Type::Bool);
                for s in body {
                    self.infer_stmt(s)?;
                }
                Ok(Type::Void)
            }
            _ => Ok(Type::Void),
        }
    }

    /// Infer type from an expression
    fn infer_expr(&mut self, expr: &Expr) -> Result<Type, TypeError> {
        match &expr.inner {
            ExprKind::Int(_) => Ok(Type::I64),
            ExprKind::Float(_) => Ok(Type::F32),
            ExprKind::Bool(_) => Ok(Type::Bool),
            ExprKind::StringLiteral(_) => Ok(Type::UserDefined("String".to_string(), vec![])),

            ExprKind::Variable(name) => self
                .var_types
                .get(name)
                .cloned()
                .ok_or_else(|| TypeError::UnboundVariable(name.clone())),

            ExprKind::BinOp(lhs, _op, rhs) => {
                let left_ty = self.infer_expr(lhs)?;
                let right_ty = self.infer_expr(rhs)?;
                // Add constraint that operands have same type
                self.add_type_eq(left_ty.clone(), right_ty);
                Ok(left_ty)
            }

            ExprKind::TensorLiteral(elements) => {
                // Infer from first element, constrain all elements to same type
                if elements.is_empty() {
                    let elem_ty = self.fresh_type_var();
                    let dim = Dim::Constant(0);
                    Ok(Type::TensorShaped(Box::new(elem_ty), vec![dim]))
                } else {
                    let first_ty = self.infer_expr(&elements[0])?;
                    for elem in elements.iter().skip(1) {
                        let elem_ty = self.infer_expr(elem)?;
                        self.add_type_eq(first_ty.clone(), elem_ty);
                    }
                    let dim = Dim::Constant(elements.len());
                    Ok(Type::TensorShaped(Box::new(first_ty), vec![dim]))
                }
            }

            ExprKind::IndexAccess(base, _indices) => {
                let base_ty = self.infer_expr(base)?;
                // Index access on tensor returns element type
                match base_ty {
                    Type::Tensor(inner, _) => Ok(*inner),
                    Type::TensorShaped(inner, _) => Ok(*inner),
                    _ => {
                        // For other types, return a fresh type variable
                        Ok(self.fresh_type_var())
                    }
                }
            }

            ExprKind::FnCall(name, args) => {
                // Infer argument types
                for arg in args {
                    self.infer_expr(arg)?;
                }
                // Return fresh type variable for unknown function returns
                Ok(match name.as_str() {
                    "print" => Type::Void,
                    "zeros" | "ones" | "rand" => {
                        let dim = self.fresh_dim_var();
                        Type::TensorShaped(Box::new(Type::F32), vec![dim])
                    }
                    "matmul" => {
                        let m = self.fresh_dim_var();
                        let n = self.fresh_dim_var();
                        Type::TensorShaped(Box::new(Type::F32), vec![m, n])
                    }
                    _ => self.fresh_type_var(),
                })
            }



            _ => Ok(self.fresh_type_var()),
        }
    }

    /// Unification: solve all collected constraints
    pub fn unify(&mut self) -> Result<(), TypeError> {
        while let Some(constraint) = self.constraints.pop() {
            match constraint {
                Constraint::TypeEq(t1, t2) => {
                    self.unify_types(t1, t2)?;
                }
                Constraint::DimEq(d1, d2) => {
                    self.unify_dims(d1, d2)?;
                }
            }
        }
        Ok(())
    }

    /// Unify two types
    fn unify_types(&mut self, t1: Type, t2: Type) -> Result<(), TypeError> {
        let t1 = self.apply_type_subst(t1);
        let t2 = self.apply_type_subst(t2);

        match (&t1, &t2) {
            // Same concrete type: OK
            _ if t1 == t2 => Ok(()),

            // Type variable on left: bind
            (Type::TypeVar(id), _) => {
                self.type_subst.insert(*id, t2);
                Ok(())
            }

            // Type variable on right: bind
            (_, Type::TypeVar(id)) => {
                self.type_subst.insert(*id, t1);
                Ok(())
            }

            // Tensor types: unify inner type and rank
            (Type::Tensor(inner1, rank1), Type::Tensor(inner2, rank2)) => {
                if rank1 != rank2 {
                    return Err(TypeError::UnificationFailed(t1, t2));
                }
                self.unify_types(*inner1.clone(), *inner2.clone())
            }

            // TensorShaped: unify inner type and dimensions
            (Type::TensorShaped(inner1, dims1), Type::TensorShaped(inner2, dims2)) => {
                if dims1.len() != dims2.len() {
                    return Err(TypeError::ShapeMismatch(dims1.clone(), dims2.clone()));
                }
                self.unify_types(*inner1.clone(), *inner2.clone())?;
                for (d1, d2) in dims1.iter().zip(dims2.iter()) {
                    self.unify_dims(d1.clone(), d2.clone())?;
                }
                Ok(())
            }

            // Vec types: unify inner
            (Type::Vec(inner1), Type::Vec(inner2)) => {
                self.unify_types(*inner1.clone(), *inner2.clone())
            }

            // Mixed Tensor and TensorShaped
            (Type::Tensor(inner1, rank), Type::TensorShaped(inner2, dims))
            | (Type::TensorShaped(inner2, dims), Type::Tensor(inner1, rank)) => {
                if *rank != dims.len() {
                    return Err(TypeError::UnificationFailed(t1, t2));
                }
                self.unify_types(*inner1.clone(), *inner2.clone())
            }

            // Otherwise: failure
            _ => Err(TypeError::UnificationFailed(t1, t2)),
        }
    }

    /// Unify two dimensions
    fn unify_dims(&mut self, d1: Dim, d2: Dim) -> Result<(), TypeError> {
        let d1 = self.apply_dim_subst(d1);
        let d2 = self.apply_dim_subst(d2);

        match (&d1, &d2) {
            _ if d1 == d2 => Ok(()),
            (Dim::Var(id), _) => {
                self.dim_subst.insert(*id, d2);
                Ok(())
            }
            (_, Dim::Var(id)) => {
                self.dim_subst.insert(*id, d1);
                Ok(())
            }
            _ => Err(TypeError::DimensionMismatch(d1, d2)),
        }
    }

    /// Apply type substitution to a type
    pub fn apply_type_subst(&self, ty: Type) -> Type {
        match ty {
            Type::TypeVar(id) => {
                if let Some(resolved) = self.type_subst.get(&id) {
                    self.apply_type_subst(resolved.clone())
                } else {
                    Type::TypeVar(id)
                }
            }
            Type::Tensor(inner, rank) => {
                Type::Tensor(Box::new(self.apply_type_subst(*inner)), rank)
            }
            Type::TensorShaped(inner, dims) => {
                let resolved_dims: Vec<Dim> =
                    dims.into_iter().map(|d| self.apply_dim_subst(d)).collect();
                Type::TensorShaped(Box::new(self.apply_type_subst(*inner)), resolved_dims)
            }
            Type::Vec(inner) => Type::Vec(Box::new(self.apply_type_subst(*inner))),
            other => other,
        }
    }

    /// Apply dimension substitution
    pub fn apply_dim_subst(&self, dim: Dim) -> Dim {
        match dim {
            Dim::Var(id) => {
                if let Some(resolved) = self.dim_subst.get(&id) {
                    self.apply_dim_subst(resolved.clone())
                } else {
                    Dim::Var(id)
                }
            }
            other => other,
        }
    }
}

impl Default for TypeInferencer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresh_type_var() {
        let mut infer = TypeInferencer::new();
        let t1 = infer.fresh_type_var();
        let t2 = infer.fresh_type_var();
        assert_eq!(t1, Type::TypeVar(0));
        assert_eq!(t2, Type::TypeVar(1));
    }

    #[test]
    fn test_unify_same_types() {
        let mut infer = TypeInferencer::new();
        infer.add_type_eq(Type::I64, Type::I64);
        assert!(infer.unify().is_ok());
    }

    #[test]
    fn test_unify_type_var() {
        let mut infer = TypeInferencer::new();
        let tv = infer.fresh_type_var();
        infer.add_type_eq(tv.clone(), Type::F32);
        assert!(infer.unify().is_ok());
        assert_eq!(infer.apply_type_subst(tv), Type::F32);
    }

    #[test]
    fn test_unify_tensors() {
        let mut infer = TypeInferencer::new();
        let t1 = Type::Tensor(Box::new(Type::F32), 2);
        let t2 = Type::Tensor(Box::new(Type::F32), 2);
        infer.add_type_eq(t1, t2);
        assert!(infer.unify().is_ok());
    }

    #[test]
    fn test_unify_tensor_mismatch() {
        let mut infer = TypeInferencer::new();
        let t1 = Type::Tensor(Box::new(Type::F32), 2);
        let t2 = Type::Tensor(Box::new(Type::F32), 3);
        infer.add_type_eq(t1, t2);
        assert!(infer.unify().is_err());
    }

    #[test]
    fn test_dim_unification() {
        let mut infer = TypeInferencer::new();
        let dv = infer.fresh_dim_var();
        infer.add_dim_eq(dv.clone(), Dim::Constant(64));
        assert!(infer.unify().is_ok());
        assert_eq!(infer.apply_dim_subst(dv), Dim::Constant(64));
    }

    #[test]
    fn test_tensor_shaped_unify() {
        let mut infer = TypeInferencer::new();
        let d1 = infer.fresh_dim_var();
        let d2 = infer.fresh_dim_var();
        let t1 = Type::TensorShaped(Box::new(Type::F32), vec![d1.clone(), d2.clone()]);
        let t2 = Type::TensorShaped(
            Box::new(Type::F32),
            vec![Dim::Constant(3), Dim::Constant(4)],
        );
        infer.add_type_eq(t1, t2);
        assert!(infer.unify().is_ok());
        assert_eq!(infer.apply_dim_subst(d1), Dim::Constant(3));
        assert_eq!(infer.apply_dim_subst(d2), Dim::Constant(4));
    }
}
