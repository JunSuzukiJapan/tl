// src/compiler/semantics.rs
use crate::compiler::ast::*;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SemanticError {
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    #[error("Type mismatch: expected {expected:?}, found {found:?}")]
    TypeMismatch { expected: Type, found: Type },
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
    #[error("Struct not found: {0}")]
    StructNotFound(String),
    #[error("Duplicate definition: {0}")]
    DuplicateDefinition(String),
    #[error("Incorrect number of arguments for {name}: expected {expected}, found {found}")]
    ArgumentCountMismatch {
        name: String,
        expected: usize,
        found: usize,
    },
}

#[derive(Clone, Debug)]
struct Symbol {
    #[allow(dead_code)]
    name: String,
    ty: Type,
    // potentially more info like mutability, shape info (if constant)
}

struct Scope {
    symbols: HashMap<String, Symbol>,
}

impl Scope {
    fn new() -> Self {
        Scope {
            symbols: HashMap::new(),
        }
    }

    fn insert(&mut self, name: String, ty: Type) {
        self.symbols.insert(name.clone(), Symbol { name, ty });
    }

    fn get(&self, name: &str) -> Option<&Symbol> {
        self.symbols.get(name)
    }
}

pub struct SemanticAnalyzer {
    scopes: Vec<Scope>,                      // Stack of scopes
    functions: HashMap<String, FunctionDef>, // Global function registry
    structs: HashMap<String, StructDef>,     // Global struct registry
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        SemanticAnalyzer {
            scopes: vec![Scope::new()], // Global scope
            functions: HashMap::new(),
            structs: HashMap::new(),
        }
    }

    fn enter_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare_variable(&mut self, name: String, ty: Type) -> Result<(), SemanticError> {
        // Shadowing is allowed in Rust-like languages usually, but let's be strict for now or allow it?
        // Let's allow shadowing (insert into current scope).
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
            Ok(())
        } else {
            unreachable!("No scope available")
        }
    }

    fn lookup_variable(&self, name: &str) -> Result<Type, SemanticError> {
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.get(name) {
                return Ok(symbol.ty.clone());
            }
        }
        Err(SemanticError::VariableNotFound(name.to_string()))
    }

    // --- Main Checking Logic ---

    pub fn check_module(&mut self, module: &Module) -> Result<(), SemanticError> {
        // First pass: verify and register all global items (Structs, Functions)
        for s in &module.structs {
            if self.structs.contains_key(&s.name) {
                return Err(SemanticError::DuplicateDefinition(s.name.clone()));
            }
            self.structs.insert(s.name.clone(), s.clone());
        }

        for f in &module.functions {
            if self.functions.contains_key(&f.name) {
                return Err(SemanticError::DuplicateDefinition(f.name.clone()));
            }
            self.functions.insert(f.name.clone(), f.clone());
        }

        // Second pass: check function bodies
        for f in &module.functions {
            self.check_function(f)?;
        }

        // Check Impl blocks
        for i in &module.impls {
            self.check_impl_block(i)?;
        }

        Ok(())
    }

    fn check_impl_block(&mut self, impl_block: &ImplBlock) -> Result<(), SemanticError> {
        // Check if target struct exists
        if !self.structs.contains_key(&impl_block.target_type) {
            return Err(SemanticError::StructNotFound(
                impl_block.target_type.clone(),
            ));
        }

        // Check methods
        for method in &impl_block.methods {
            // TODO: Add 'self' to scope if method is not static?
            // For now just check body as normal function
            self.check_function(method)?;
        }
        Ok(())
    }

    fn check_function(&mut self, func: &FunctionDef) -> Result<(), SemanticError> {
        self.enter_scope();

        // Register arguments
        // Register arguments
        for (name, ty) in &func.args {
            self.declare_variable(name.clone(), ty.clone())?;
        }

        for stmt in &func.body {
            self.check_stmt(stmt)?;
        }

        // TODO: Check missing return?

        self.exit_scope();
        Ok(())
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> Result<(), SemanticError> {
        match stmt {
            Stmt::TensorDecl {
                name,
                type_annotation,
                init,
            } => {
                if let Some(expr) = init {
                    let _ = self.check_expr(expr)?;
                }
                self.declare_variable(name.clone(), type_annotation.clone())?;
                Ok(())
            }

            Stmt::Let {
                name,
                indices,
                type_annotation,
                value,
            } => {
                let inferred_type = if let Some(idxs) = indices {
                    // Tensor Equation Logic
                    self.enter_scope();

                    // 1. Declare LHS indices
                    // let C[i, k] = ... -> i, k are valid indices (I64/Usize)
                    let mut lhs_indices = HashSet::new();
                    for idx in idxs {
                        if lhs_indices.contains(idx) {
                            return Err(SemanticError::DuplicateDefinition(idx.clone()));
                        }
                        lhs_indices.insert(idx.clone());
                        self.declare_variable(idx.clone(), Type::I64)?;
                    }

                    // 2. Discover RHS indices
                    // Scan value for IndexAccess identifiers.
                    // If not in LHS and not in outer scope, declare as reduction index.
                    let mut used_indices = HashSet::new();
                    self.collect_indices(value, &mut used_indices);

                    for idx in used_indices {
                        // Check if already declared (LHS or Outer Scope)
                        // If lookup succeeds, it's defined.
                        // But wait, if it is defined in outer scope, is it an index or variable?
                        // If it's used in IndexAccess, it's an index.
                        // If we are in "Tensor Equation Mode", we might want to capture outer variables too?
                        // Design decision: Implicit reduction indices must NOT be defined in outer scope to avoid ambiguity?
                        // Or shadowing: if not in LHS, check if defined. If defined, it's a constant/param.
                        // If NOT defined, it's a local reduction index (dummy var).

                        if lhs_indices.contains(&idx) {
                            continue;
                        }

                        if self.lookup_variable(&idx).is_ok() {
                            // Defined in outer scope (or earlier in THIS scope provided we shadowed LHS).
                            // It's a parameter (e.g. constant index, or outer loop var).
                            // Do nothing.
                        } else {
                            // Not in LHS, not in Outer. Treat as reduction index.
                            self.declare_variable(idx, Type::I64)?;
                        }
                    }

                    // 3. Check RHS expression
                    // It should evaluate to a Scalar type (f32, i64, bool) usually,
                    // or a Tensor if we are building a Tensor of Tensors (rare).
                    // Example: A[i, j] -> f32.
                    // A[i, j] * B[j, k] -> f32.
                    let rhs_type = self.check_expr(value)?;

                    self.exit_scope();

                    // 4. Construct Result Type
                    // Tensor<rhs_type, rank=idxs.len()>
                    // We assume dense tensor for now.
                    Type::Tensor(Box::new(rhs_type), idxs.len())
                } else {
                    self.check_expr(value)?
                };

                let final_type = if let Some(ann) = type_annotation {
                    if !self.are_types_compatible(ann, &inferred_type) {
                        return Err(SemanticError::TypeMismatch {
                            expected: ann.clone(),
                            found: inferred_type,
                        });
                    }
                    ann.clone()
                } else {
                    inferred_type
                };

                self.declare_variable(name.clone(), final_type)?;
                Ok(())
            }
            Stmt::Assign {
                name,
                indices,
                op,
                value,
            } => {
                let var_type = self.lookup_variable(name)?;

                if let Some(idxs) = indices {
                    // Indexed assignment: C[i, k] += ...
                    // Similar logic to Let, but we check against var_type element type.

                    // Verify var_type is Tensor
                    let (inner_type, rank) = match &var_type {
                        Type::Tensor(inner, r) => (inner, *r),
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::Tensor(Box::new(Type::Void), 0),
                                found: var_type,
                            })
                        }
                    };

                    if idxs.len() != rank {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: rank,
                            found: idxs.len(),
                        });
                    }

                    self.enter_scope();
                    let mut lhs_indices = HashSet::new();
                    for idx in idxs {
                        if lhs_indices.contains(idx) {
                            return Err(SemanticError::DuplicateDefinition(idx.clone()));
                        }
                        lhs_indices.insert(idx.clone());
                        self.declare_variable(idx.clone(), Type::I64)?;
                    }

                    let mut used_indices = HashSet::new();
                    self.collect_indices(value, &mut used_indices);
                    for idx in used_indices {
                        if !lhs_indices.contains(&idx) {
                            if self.lookup_variable(&idx).is_err() {
                                // Reduction / Contraction index
                                self.declare_variable(idx, Type::I64)?;
                            }
                        }
                    }

                    let rhs_type = self.check_expr(value)?;
                    self.exit_scope();

                    // rhs_type must match inner_type
                    if !self.are_types_compatible(inner_type, &rhs_type) {
                        return Err(SemanticError::TypeMismatch {
                            expected: *inner_type.clone(),
                            found: rhs_type,
                        });
                    }
                    Ok(())
                } else {
                    // Standard assignment
                    let val_type = self.check_expr(value)?;

                    match op {
                        AssignOp::Assign => {
                            if !self.are_types_compatible(&var_type, &val_type) {
                                return Err(SemanticError::TypeMismatch {
                                    expected: var_type,
                                    found: val_type,
                                });
                            }
                        }
                        AssignOp::AddAssign => {
                            // Check compatibility similar to BinOp
                            // We allow:
                            // 1. Same Type
                            // 2. Broadcasting (Tensor += Scalar)
                            // 3. Broadcasting (Tensor += Tensor of lower rank/compatible)

                            // Simple Reuse of BinOp logic or manual check:
                            let result_type = match (&var_type, &val_type) {
                                (Type::Tensor(inner, _), val) if **inner == *val => {
                                    Some(var_type.clone())
                                } // Tensor += Scalar
                                (Type::Tensor(inner1, rank1), Type::Tensor(inner2, rank2))
                                    if inner1 == inner2 =>
                                {
                                    // Tensor += Tensor
                                    // Result rank is max(rank1, rank2).
                                    // For assignment, Result Rank MUST be equal to var_type rank (cannot change rank of variable in place).
                                    // AND rank1 >= rank2 usually for +=?
                                    // Actually A += B is valid if broadcast(A, B) shape is shape(A).
                                    // This usually implies rank(A) >= rank(B).
                                    if rank1 >= rank2 {
                                        Some(var_type.clone())
                                    } else {
                                        None // Cannot assign higher rank tensor to lower rank variable (shape mismatch likely)
                                    }
                                }
                                (t1, t2) if t1 == t2 => Some(t1.clone()), // Primitive += Primitive
                                _ => None,
                            };

                            if result_type.is_none() {
                                return Err(SemanticError::TypeMismatch {
                                    expected: var_type,
                                    found: val_type,
                                });
                            }
                        }
                        _ => {
                            // Max/Avg etc not yet fully checked
                            if !self.are_types_compatible(&var_type, &val_type) {
                                return Err(SemanticError::TypeMismatch {
                                    expected: var_type,
                                    found: val_type,
                                });
                            }
                        }
                    }
                    Ok(())
                }
            }
            Stmt::Return(expr) => {
                let _ = self.check_expr(expr)?;
                // TODO: Check against function return type (need access to current function context)
                Ok(())
            }
            Stmt::Expr(expr) => {
                self.check_expr(expr)?;
                Ok(())
            }
            Stmt::If {
                cond,
                then_block,
                else_block,
            } => {
                let cond_type = self.check_expr(cond)?;
                // Condition must be boolean (or promotable?) Assuming strict bool for control flow
                // Actually specification says bool types are promoted for arithmetic, but if conditions usually need bool.
                if cond_type != Type::Bool {
                    // Strict check for now
                    // return Err(SemanticError::TypeMismatch { expected: Type::Bool, found: cond_type });
                }

                self.enter_scope();
                for s in then_block {
                    self.check_stmt(s)?;
                }
                self.exit_scope();

                if let Some(block) = else_block {
                    self.enter_scope();
                    for s in block {
                        self.check_stmt(s)?;
                    }
                    self.exit_scope();
                }
                Ok(())
            }
            Stmt::For {
                loop_var,
                iterator,
                body,
            } => {
                let iter_type = self.check_expr(iterator)?;

                // For simplified MVP, allow iteration over Tensor<T, 1> or Array
                // Assume iter_type gives element type
                let elem_type = match iter_type {
                    Type::Tensor(t, 1) => *t,
                    _ => {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::F32), 1),
                            found: iter_type,
                        })
                    }
                };

                self.enter_scope();
                self.declare_variable(loop_var.clone(), elem_type)?;
                for s in body {
                    self.check_stmt(s)?;
                }
                self.exit_scope();
                Ok(())
            }
            Stmt::While { cond, body } => {
                let cond_type = self.check_expr(cond)?;
                if cond_type != Type::Bool {
                    return Err(SemanticError::TypeMismatch {
                        expected: Type::Bool,
                        found: cond_type,
                    });
                }
                self.enter_scope();
                for stmt in body {
                    self.check_stmt(stmt)?;
                }
                self.exit_scope();
                Ok(())
            }
        }
    }

    fn check_expr(&mut self, expr: &Expr) -> Result<Type, SemanticError> {
        match expr {
            Expr::Int(_) => Ok(Type::I64),   // Default integer literal type
            Expr::Float(_) => Ok(Type::F32), // Default float literal type
            Expr::Bool(_) => Ok(Type::Bool),
            Expr::StringLiteral(_) => Ok(Type::UserDefined("String".to_string())), // Placeholder
            Expr::Variable(name) => self.lookup_variable(name),
            Expr::BinOp(lhs, op, rhs) => {
                let left = self.check_expr(lhs)?;
                let right = self.check_expr(rhs)?;

                // Determine result type based on Op
                let result_ty = match op {
                    BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                        Type::Bool
                    }
                    BinOp::And | BinOp::Or => Type::Bool,
                    _ => left.clone(), // Arith ops preserve type (or broadcast)
                };

                // Simple strict matching for now
                if left == right {
                    Ok(result_ty)
                } else {
                    match (&left, &right) {
                        (Type::Tensor(inner, _rank), val) if **inner == *val => Ok(result_ty),
                        (val, Type::Tensor(inner, _rank)) if **inner == *val => {
                            // Scalar * Tensor -> Tensor
                            // If comparison, Bool. If arithmetic, Tensor (right).
                            if matches!(result_ty, Type::Bool) {
                                Ok(Type::Bool)
                            } else {
                                Ok(right)
                            }
                        }
                        (Type::Tensor(inner1, rank1), Type::Tensor(inner2, rank2))
                            if inner1 == inner2 =>
                        {
                            if matches!(result_ty, Type::Bool) {
                                Ok(Type::Bool)
                            } else {
                                // Arithmetic: max rank
                                Ok(Type::Tensor(inner1.clone(), std::cmp::max(*rank1, *rank2)))
                            }
                        }
                        _ => Err(SemanticError::TypeMismatch {
                            expected: left,
                            found: right,
                        }),
                    }
                }
            }
            Expr::FnCall(name, args) => {
                if name == "print" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&args[0])?;
                    return Ok(Type::Void);
                } else if name == "transpose" {
                    if args.len() != 3 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 3,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    let t1 = self.check_expr(&args[1])?;
                    let t2 = self.check_expr(&args[2])?;

                    match t0 {
                        Type::Tensor(_, _) => {}
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::Tensor(Box::new(Type::Void), 0),
                                found: t0,
                            })
                        }
                    }
                    if t1 != Type::I64 {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::I64,
                            found: t1,
                        });
                    }
                    if t2 != Type::I64 {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::I64,
                            found: t2,
                        });
                    }
                    return Ok(t0); // Returns same tensor type (rank preserved)
                } else if name == "reshape" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    let t1 = self.check_expr(&args[1])?;

                    match t0 {
                        Type::Tensor(_, _) => {}
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::Tensor(Box::new(Type::Void), 0),
                                found: t0,
                            })
                        }
                    }
                    // Arg 1 (shape) must be a Tensor (specifically Tensor<i64/f32, 1> or similar)
                    match t1 {
                        Type::Tensor(_, _) => {}
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::Tensor(Box::new(Type::Void), 0),
                                found: t1,
                            })
                        }
                    }
                    return Ok(t0); // Returns same tensor type (ignoring rank change needed for strict typing)
                } else if name == "slice" {
                    if args.len() != 3 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 3,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    let t1 = self.check_expr(&args[1])?;
                    let t2 = self.check_expr(&args[2])?;

                    // Arg 0 must be Tensor
                    match t0 {
                        Type::Tensor(_, _) => {}
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::Tensor(Box::new(Type::Void), 0),
                                found: t0,
                            })
                        }
                    }
                    // Arg 1, 2 must be Int
                    if t1 != Type::I64 {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::I64,
                            found: t1,
                        });
                    }
                    if t2 != Type::I64 {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::I64,
                            found: t2,
                        });
                    }
                    return Ok(t0); // Returns same tensor type
                }

                let func = self
                    .functions
                    .get(name)
                    .ok_or_else(|| SemanticError::FunctionNotFound(name.clone()))?
                    .clone();

                if args.len() != func.args.len() {
                    // func.args is empty in current AST parser stub, need to fix that first to check properly
                    // For now, skip arg checking if definitions are empty
                    if !func.args.is_empty() {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: func.args.len(),
                            found: args.len(),
                        });
                    }
                }

                // TODO: Check arg types
                Ok(func.return_type.clone())
            }
            Expr::TensorLiteral(elements) => {
                // Check all elements are same type
                if elements.is_empty() {
                    return Ok(Type::Tensor(Box::new(Type::F32), 1)); // Empty tensor?
                }
                let first_type = self.check_expr(&elements[0])?;
                for e in &elements[1..] {
                    let t = self.check_expr(e)?;
                    if t != first_type {
                        return Err(SemanticError::TypeMismatch {
                            expected: first_type,
                            found: t,
                        });
                    }
                }
                // Construct Tensor type.
                // If elements are primitive, Rank 1. If elements are Tensor<T, N>, Rank N+1.
                match first_type {
                    Type::Tensor(inner, rank) => Ok(Type::Tensor(inner, rank + 1)),
                    primitive => Ok(Type::Tensor(Box::new(primitive), 1)),
                }
            }
            Expr::IndexAccess(target, _indices) => {
                let target_type = self.check_expr(target)?;
                match target_type {
                    Type::Tensor(inner, _rank) => Ok(*inner), // Accessing reduces rank to scalar (in this simple logic)
                    // Actually in Tensor Logic a[i, j] usually denotes the tensor itself with indices bound,
                    // but for type checking purposes, it resolves to the element type if fully indexed,
                    // or acts as the tensor for equations.
                    // Let's assume it validates to the Inner type for now.
                    _ => Err(SemanticError::TypeMismatch {
                        expected: Type::Tensor(Box::new(Type::Void), 0),
                        found: target_type,
                    }),
                }
            }
            Expr::UnOp(op, inner) => {
                let t = self.check_expr(inner)?;
                match op {
                    UnOp::Neg => {
                        // Neg supports Int, Float, Tensor<Int/Float>
                        match t {
                            Type::I64 | Type::F32 => Ok(t),
                            Type::Tensor(ref inner, _) => match **inner {
                                Type::I64 | Type::F32 => Ok(t),
                                _ => Err(SemanticError::TypeMismatch {
                                    expected: Type::F32,
                                    found: t,
                                }),
                            },
                            _ => Err(SemanticError::TypeMismatch {
                                expected: Type::F32,
                                found: t,
                            }),
                        }
                    }
                    UnOp::Not => {
                        // Not supports Bool, Tensor<Bool>
                        match t {
                            Type::Bool => Ok(t),
                            Type::Tensor(ref inner, _) => match **inner {
                                Type::Bool => Ok(t),
                                _ => Err(SemanticError::TypeMismatch {
                                    expected: Type::Bool,
                                    found: t,
                                }),
                            },
                            _ => Err(SemanticError::TypeMismatch {
                                expected: Type::Bool,
                                found: t,
                            }),
                        }
                    }
                }
            }
            Expr::Block(stmts) => {
                self.enter_scope();
                let mut ret_type = Type::Void;
                for (i, stmt) in stmts.iter().enumerate() {
                    if i == stmts.len() - 1 {
                        // If last statement is an expression without semicolon (Expr::Expr), it's the return value
                        if let Stmt::Expr(e) = stmt {
                            ret_type = self.check_expr(e)?;
                        } else {
                            self.check_stmt(stmt)?;
                            ret_type = Type::Void;
                        }
                    } else {
                        self.check_stmt(stmt)?;
                    }
                }
                self.exit_scope();
                Ok(ret_type)
            }
            Expr::IfExpr(cond, then_block, else_block) => {
                let cond_type = self.check_expr(cond)?;
                if cond_type != Type::Bool {
                    // Warning or Error? Strict for now.
                }

                // Check Then Block (expr block)
                self.enter_scope();
                let mut then_type = Type::Void;
                // Wait, IfExpr in AST uses Vec<Stmt> for blocks, reusing ParseBlock Logic.
                // But ParseBlock returns Vec<Stmt>.
                // So we need to evaluate the block stmts similar to Expr::Block logic above.
                // Refactor Block Logic?

                // Inline logic for now
                for (i, stmt) in then_block.iter().enumerate() {
                    if i == then_block.len() - 1 {
                        if let Stmt::Expr(e) = stmt {
                            then_type = self.check_expr(e)?;
                        } else {
                            self.check_stmt(stmt)?;
                        }
                    } else {
                        self.check_stmt(stmt)?;
                    }
                }
                self.exit_scope();

                if let Some(else_stmts) = else_block {
                    self.enter_scope();
                    let mut else_type = Type::Void;
                    for (i, stmt) in else_stmts.iter().enumerate() {
                        if i == else_stmts.len() - 1 {
                            if let Stmt::Expr(e) = stmt {
                                else_type = self.check_expr(e)?;
                            } else {
                                self.check_stmt(stmt)?;
                            }
                        } else {
                            self.check_stmt(stmt)?;
                        }
                    }
                    self.exit_scope();

                    if then_type != else_type {
                        return Err(SemanticError::TypeMismatch {
                            expected: then_type,
                            found: else_type,
                        });
                    }
                    Ok(then_type)
                } else {
                    // If no else, must return Void
                    if then_type != Type::Void {
                        // error? or just imply void? Rust implies mismatch with () if else missing.
                        // We'll enforce Void.
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Void,
                            found: then_type,
                        });
                    }
                    Ok(Type::Void)
                }
            }
            Expr::Aggregation {
                op: _,
                expr,
                var,
                range,
                condition,
            } => {
                // Type check the range expression
                let _range_ty = self.check_expr(range)?;

                // Declare the loop variable in a new scope
                self.enter_scope();
                self.declare_variable(var.clone(), Type::I64)?; // Assume integer index

                // Check the aggregated expression
                let expr_ty = self.check_expr(expr)?;

                // Check condition if present
                if let Some(cond) = condition {
                    let cond_ty = self.check_expr(cond)?;
                    if cond_ty != Type::Bool {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Bool,
                            found: cond_ty,
                        });
                    }
                }

                self.exit_scope();

                // Aggregation returns same type as the expression (for sum/avg)
                // or I64 for count
                Ok(expr_ty)
            }
            Expr::FieldAccess(_, _) => Ok(Type::Void), // TODO
            Expr::MethodCall(_, _, _) => Ok(Type::Void), // TODO
        }
    }

    fn are_types_compatible(&self, t1: &Type, t2: &Type) -> bool {
        // Strict equality for now
        t1 == t2
    }

    fn collect_indices(&self, expr: &Expr, indices: &mut HashSet<String>) {
        match expr {
            Expr::IndexAccess(target, idxs) => {
                self.collect_indices(target, indices);
                for idx in idxs {
                    indices.insert(idx.clone());
                }
            }
            Expr::BinOp(left, _, right) => {
                self.collect_indices(left, indices);
                self.collect_indices(right, indices);
            }
            Expr::UnOp(_, inner) => {
                self.collect_indices(inner, indices);
            }
            Expr::FnCall(_, args) => {
                for arg in args {
                    self.collect_indices(arg, indices);
                }
            }
            Expr::MethodCall(obj, _, args) => {
                self.collect_indices(obj, indices);
                for arg in args {
                    self.collect_indices(arg, indices);
                }
            }
            Expr::IfExpr(cond, _then_block, _else_block) => {
                self.collect_indices(cond, indices);
                // Recurse into blocks? Stmts might have exprs.
                // But IndexAccess usually in expressions.
                // This is simple helper. Ideally we traverse fully.
                // For now, assume indexes strictly inside the logic.
                // Blocks contain Stmts. Need Stmt traversal?
                // Probably overkill for simple Equations.
                // Let's postpone block traversal inside equation.
            }
            Expr::Block(_) => {
                // Skip blocks for now
            }
            _ => {}
        }
    }
}
