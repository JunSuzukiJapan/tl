// src/compiler/semantics.rs
use crate::compiler::ast::*;
use std::collections::HashMap;
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
        for arg in &func.args {
            // self.declare_variable(arg.name.clone(), arg.ty.clone())?; // AST arg needs definition
            // TODO: Update AST to have args. For now assuming empty or manually injecting for testing
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
            Stmt::Let {
                name,
                indices,
                type_annotation,
                value,
            } => {
                let val_type = self.check_expr(value)?;

                // If indices are present (e.g. let A[i, j] = ...),
                // we treat the declared variable as a Tensor of rank = indices.len().
                // The value Expression usually evaluates to the element type (e.g. f32).
                let inferred_type = if let Some(idxs) = indices {
                    // If val_type is scalar, promote to Tensor.
                    // If val_type is already Tensor (e.g. from function call returning Tensor),
                    // checking might be more complex (broadcasting?), but basic definition
                    // let A[i] = 1.0 implies A is Tensor<f32, 1>.
                    match val_type {
                        Type::Tensor(_, _) => {
                            // If RHS is tensor, usually we are just assigning/aliasing?
                            // But A[i] = B[i] means element-wise assignment.
                            // For now, let's assume if we define with indices, we wrap the scalar type.
                            // But wait, if RHS is a[i, k] * b[k, j] -> it evaluates to f32 (scalar).
                            // So yes, wrap it.
                            Type::Tensor(Box::new(val_type), idxs.len()) // Nested tensor? Unlikely.
                        }
                        _ => Type::Tensor(Box::new(val_type), idxs.len()),
                    }
                } else {
                    val_type
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
            }
            Stmt::Assign { name, value, .. } => {
                let var_type = self.lookup_variable(name)?;
                let val_type = self.check_expr(value)?;

                if !self.are_types_compatible(&var_type, &val_type) {
                    return Err(SemanticError::TypeMismatch {
                        expected: var_type,
                        found: val_type,
                    });
                }
            }
            Stmt::Return(expr) => {
                let _ = self.check_expr(expr)?;
                // TODO: Check against function return type (need access to current function context)
            }
            Stmt::Expr(expr) => {
                self.check_expr(expr)?;
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
            }
            Stmt::For {
                loop_var,
                iterator,
                body,
            } => {
                let iter_type = self.check_expr(iterator)?;
                // Assuming iter_type is iterable (e.g. Range or Tensor dimension)
                // For now, let's assume it introduces a 'usize' or 'i32' variable.

                self.enter_scope();
                self.declare_variable(loop_var.clone(), Type::I64)?; // Default loop var type?
                for s in body {
                    self.check_stmt(s)?;
                }
                self.exit_scope();
            }
        }
        Ok(())
    }

    fn check_expr(&mut self, expr: &Expr) -> Result<Type, SemanticError> {
        match expr {
            Expr::Int(_) => Ok(Type::I64),   // Default integer literal type
            Expr::Float(_) => Ok(Type::F32), // Default float literal type
            Expr::Bool(_) => Ok(Type::Bool),
            Expr::StringLiteral(_) => Ok(Type::UserDefined("String".to_string())), // Placeholder
            Expr::Variable(name) => self.lookup_variable(name),
            Expr::BinOp(lhs, _op, rhs) => {
                let left = self.check_expr(lhs)?;
                let right = self.check_expr(rhs)?;
                // Simple strict matching for now
                // TODO: Tensor broadcasting logic / Type promotion
                if left == right {
                    Ok(left)
                } else {
                    // Allow promotion? e.g. f32 * f64 -> f64? or Tensor<f32> * f32 -> Tensor<f32>?
                    // For initial version, be strict or allow basic Tensor * Scalar
                    match (&left, &right) {
                        (Type::Tensor(inner, rank), val) if **inner == *val => Ok(left), // Tensor * Scalar
                        (val, Type::Tensor(inner, rank)) if **inner == *val => Ok(right), // Scalar * Tensor
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
            _ => Ok(Type::Void), // TOD: Blocks, IfExpr, etc.
        }
    }

    fn are_types_compatible(&self, t1: &Type, t2: &Type) -> bool {
        // Strict equality for now
        t1 == t2
    }
}
