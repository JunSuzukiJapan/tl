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
    #[error("Method not found: {method_name} on type {type_name}")]
    MethodNotFound {
        type_name: String,
        method_name: String,
    },
    #[error("Unknown function: {0}")]
    UnknownFunction(String),
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
    scopes: Vec<Scope>,                                     // Stack of scopes
    functions: HashMap<String, FunctionDef>,                // Global function registry
    structs: HashMap<String, StructDef>,                    // Global struct registry
    methods: HashMap<String, HashMap<String, FunctionDef>>, // Struct methods
    current_return_type: Option<Type>, // Expected return type for current function
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        SemanticAnalyzer {
            scopes: vec![Scope::new()], // Global scope
            functions: HashMap::new(),
            structs: HashMap::new(),
            methods: HashMap::new(),
            current_return_type: None,
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

        // Check Impl blocks (Register and Verify methods)
        // Note: Ideally we should separate "Register signatures" and "Check bodies"
        // to support mutual recursion/out-of-order calls.
        // For now, doing Impls first allows Functions (like main) to call methods.
        for i in &module.impls {
            self.check_impl_block(i)?;
        }

        // Second pass: check function bodies
        for f in &module.functions {
            self.check_function(f, None)?;
        }

        // Check top-level statements (e.g. main script)
        for s in &module.tensor_decls {
            self.check_stmt(s)?;
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
        // 1. Register methods
        {
            let struct_methods = self
                .methods
                .entry(impl_block.target_type.clone())
                .or_insert_with(HashMap::new);
            for method in &impl_block.methods {
                if struct_methods.contains_key(&method.name) {
                    return Err(SemanticError::DuplicateDefinition(format!(
                        "{}::{}",
                        impl_block.target_type, method.name
                    )));
                }
                struct_methods.insert(method.name.clone(), {
                    let mut m = method.clone();
                    // Resolve 'Self' in args to target_type
                    for (_, arg_ty) in &mut m.args {
                        if let Type::UserDefined(ref n) = arg_ty {
                            if n == "Self" {
                                *arg_ty = Type::UserDefined(impl_block.target_type.clone());
                            }
                        }
                    }
                    m
                });
            }
        }

        // 2. Check function bodies
        for method in &impl_block.methods {
            self.check_function(
                method,
                Some(Type::UserDefined(impl_block.target_type.clone())),
            )?;
        }
        Ok(())
    }

    fn check_function(
        &mut self,
        func: &FunctionDef,
        self_type: Option<Type>,
    ) -> Result<(), SemanticError> {
        self.enter_scope();

        // Set expected return type for this function
        self.current_return_type = Some(func.return_type.clone());

        // Register arguments
        for (name, ty) in &func.args {
            let actual_ty = if let Type::UserDefined(ref type_name) = ty {
                if type_name == "Self" {
                    // Resolve Self -> Actual Type
                    self_type.clone().ok_or_else(|| {
                        SemanticError::VariableNotFound(
                            "Self type not available in this context".into(),
                        )
                    })?
                } else {
                    ty.clone()
                }
            } else {
                ty.clone()
            };
            self.declare_variable(name.clone(), actual_ty)?;
        }

        for stmt in &func.body {
            self.check_stmt(stmt)?;
        }

        // Clear return type context
        self.current_return_type = None;

        self.exit_scope();
        Ok(())
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> Result<(), SemanticError> {
        match stmt {
            Stmt::FieldAssign { obj, field, value } => {
                // Check object type and verify it's a struct
                let obj_type = self.check_expr(obj)?;
                let struct_name = match obj_type {
                    Type::UserDefined(name) => name,
                    _ => {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::UserDefined("Struct".into()),
                            found: obj_type,
                        });
                    }
                };

                // Verify struct and field exist
                let struct_def = self
                    .structs
                    .get(&struct_name)
                    .ok_or_else(|| SemanticError::StructNotFound(struct_name.clone()))?;

                let field_type = struct_def
                    .fields
                    .iter()
                    .find(|(name, _)| name == field)
                    .map(|(_, ty)| ty.clone())
                    .ok_or_else(|| SemanticError::VariableNotFound(format!("Field {}", field)))?;

                // Check value type matches field type
                let value_type = self.check_expr(value)?;
                if !self.are_types_compatible(&field_type, &value_type) {
                    return Err(SemanticError::TypeMismatch {
                        expected: field_type,
                        found: value_type,
                    });
                }

                Ok(())
            }
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
                    // Indexed assignment: C[i, k] = ...
                    // Currently we treat this as element-wise assignment validation.
                    // (Tensor Equation logic requires more complex handling of loop vars)

                    // Verify var_type is Tensor
                    let (_inner_type, rank) = match &var_type {
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

                    // Check each index expression is integer
                    for idx_expr in idxs {
                        let idx_type = self.check_expr(idx_expr)?;
                        if !matches!(idx_type, Type::I64 | Type::I32) {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: idx_type,
                            });
                        }
                    }

                    // Check value type matches tensor element type (simplified)
                    // (Value check is done below generally, but we might want context)
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
                        op => {
                            let method_name = match op {
                                AssignOp::AddAssign => "add_assign",
                                AssignOp::SubAssign => "sub_assign",
                                AssignOp::MulAssign => "mul_assign",
                                AssignOp::DivAssign => "div_assign",
                                _ => {
                                    return Err(SemanticError::UnknownFunction(format!(
                                        "Unsupported assign op {:?}",
                                        op
                                    )))
                                }
                            };

                            // Check as method call logic for AssignOp (simplified)
                            match &var_type {
                                Type::Tensor(_, _) => {
                                    // Check if valid method name
                                    match method_name {
                                        "add_assign" | "sub_assign" | "mul_assign"
                                        | "div_assign" => {
                                            // OK
                                        }
                                        _ => {
                                            return Err(SemanticError::MethodNotFound {
                                                type_name: format!("{:?}", var_type),
                                                method_name: method_name.to_string(),
                                            })
                                        }
                                    }

                                    let is_compat = match (&var_type, &val_type) {
                                        (Type::Tensor(inner, _), val) if **inner == *val => true,
                                        (Type::Tensor(_, _), Type::Tensor(_, _)) => true,
                                        _ => false,
                                    };

                                    if !is_compat {
                                        return Err(SemanticError::TypeMismatch {
                                            expected: var_type.clone(),
                                            found: val_type.clone(),
                                        });
                                    }
                                }
                                _ => {
                                    return Err(SemanticError::MethodNotFound {
                                        type_name: format!("{:?}", var_type),
                                        method_name: method_name.to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
            Stmt::Return(expr) => {
                let return_type = self.check_expr(expr)?;
                // Check against function return type
                if let Some(ref expected) = self.current_return_type {
                    if !self.are_types_compatible(expected, &return_type) {
                        return Err(SemanticError::TypeMismatch {
                            expected: expected.clone(),
                            found: return_type,
                        });
                    }
                }
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
                // Check if iterator is range(start, end)
                // This is a special case for the compiler intrinsic 'range'
                let is_range = if let Expr::FnCall(name, args) = iterator {
                    if name == "range" && args.len() == 2 {
                        // Check arguments
                        let start_type = self.check_expr(&args[0])?;
                        let end_type = self.check_expr(&args[1])?;
                        // expect integers
                        if !matches!(start_type, Type::I64 | Type::I32)
                            || !matches!(end_type, Type::I64 | Type::I32)
                        {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: start_type, // simplified error
                            });
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                let elem_type = if is_range {
                    Type::I64
                } else {
                    let iter_type = self.check_expr(iterator)?;
                    match iter_type {
                        Type::Tensor(t, 1) => *t,
                        Type::TensorShaped(t, _) => *t, // Allow iterating shaped tensors?
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::Tensor(Box::new(Type::F32), 1),
                                found: iter_type,
                            })
                        }
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
            Expr::StructInit(name, fields) => {
                let struct_def = self
                    .structs
                    .get(name)
                    .ok_or_else(|| SemanticError::StructNotFound(name.clone()))?
                    .clone();

                let mut initialized_fields = HashSet::new();
                for (field_name, field_expr) in fields {
                    if initialized_fields.contains(field_name) {
                        return Err(SemanticError::DuplicateDefinition(format!(
                            "Field {} in struct init",
                            field_name
                        )));
                    }
                    initialized_fields.insert(field_name.clone());

                    // Check if field exists and get type
                    let expected_type = struct_def
                        .fields
                        .iter()
                        .find(|(f, _)| f == field_name)
                        .map(|(_, t)| t)
                        .ok_or_else(|| {
                            SemanticError::VariableNotFound(format!(
                                "Field {} in struct {}",
                                field_name, name
                            ))
                        })?;

                    let found_type = self.check_expr(field_expr)?;
                    if !self.are_types_compatible(expected_type, &found_type) {
                        return Err(SemanticError::TypeMismatch {
                            expected: expected_type.clone(),
                            found: found_type,
                        });
                    }
                }

                // Check for missing fields
                for (field_name, _) in &struct_def.fields {
                    if !initialized_fields.contains(field_name) {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: format!("Struct init {}", name),
                            expected: struct_def.fields.len(),
                            found: initialized_fields.len(),
                        });
                    }
                }

                Ok(Type::Struct(name.clone()))
            }
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
                if name == "cross_entropy" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    let t1 = self.check_expr(&args[1])?;
                    // Ensure both are tensors
                    if !matches!(t0, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t0,
                        });
                    }
                    // t1 is targets, also tensor
                    if !matches!(t1, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t1,
                        });
                    }
                    // Returns scalar tensor
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                }
                if name == "pow" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    let t1 = self.check_expr(&args[1])?;
                    // Allow Tensor, F32, I64
                    match &t0 {
                        Type::Tensor(_, _) | Type::F32 | Type::I64 => {}
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::Tensor(Box::new(Type::Void), 0),
                                found: t0,
                            })
                        }
                    }
                    match &t1 {
                        Type::Tensor(_, _) | Type::F32 | Type::I64 => {}
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::Tensor(Box::new(Type::Void), 0),
                                found: t1,
                            })
                        }
                    }
                    // Return type is same as first arg (preserving shape usually, or broadcasted)
                    // For simplicity assume resulting type is similar to input or just Tensor
                    // Let's return t0 for now.
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                }
                if name == "enable_grad" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: "enable_grad".into(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    if !matches!(t0, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t0,
                        });
                    }
                    return Ok(t0);
                }
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
                if name == "save_all_params" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    match t0 {
                        Type::UserDefined(s) if s == "String" => {}
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::UserDefined("String".into()),
                                found: t0,
                            })
                        }
                    }
                    return Ok(Type::Void);
                }
                if name == "load_all_params" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    match t0 {
                        Type::UserDefined(s) if s == "String" => {}
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::UserDefined("String".into()),
                                found: t0,
                            })
                        }
                    }
                    return Ok(Type::Void);
                }

                if name == "add_parameter" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    match t0 {
                        Type::UserDefined(s) if s == "String" => {}
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::UserDefined("String".into()),
                                found: t0,
                            })
                        }
                    }
                    let t1 = self.check_expr(&args[1])?;
                    if !matches!(t1, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t1,
                        });
                    }
                    return Ok(Type::Void);
                }

                if name == "parameter" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    if !matches!(t0, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::F32), 0),
                            found: t0,
                        });
                    }
                    return Ok(t0);
                }

                // --- StdLib Phase 1 ---
                // --- StdLib Static Methods ---
                if name.contains("::") {
                    let parts: Vec<&str> = name.split("::").collect();
                    if parts.len() == 2 {
                        match (parts[0], parts[1]) {
                            ("File", "open") => {
                                if args.len() != 2 {
                                    return Err(SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 2,
                                        found: args.len(),
                                    });
                                }
                                self.check_expr(&args[0])?; // path
                                self.check_expr(&args[1])?; // mode
                                return Ok(Type::UserDefined("File".to_string()));
                            }
                            ("Env", "get") => {
                                if args.len() != 1 {
                                    return Err(SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 1,
                                        found: args.len(),
                                    });
                                }
                                self.check_expr(&args[0])?;
                                return Ok(Type::UserDefined("String".to_string()));
                            }
                            ("Path", "new") => {
                                if args.len() != 1 {
                                    return Err(SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 1,
                                        found: args.len(),
                                    });
                                }
                                self.check_expr(&args[0])?;
                                return Ok(Type::UserDefined("Path".to_string()));
                            }
                            ("Http", "download") => {
                                if args.len() != 2 {
                                    return Err(SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 2,
                                        found: args.len(),
                                    });
                                }
                                self.check_expr(&args[0])?;
                                self.check_expr(&args[1])?;
                                return Ok(Type::Bool);
                            }
                            ("Http", "get") => {
                                if args.len() != 1 {
                                    return Err(SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 1,
                                        found: args.len(),
                                    });
                                }
                                self.check_expr(&args[0])?;
                                return Ok(Type::UserDefined("String".to_string()));
                            }
                            ("Env", "set") => {
                                if args.len() != 2 {
                                    return Err(SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 2,
                                        found: args.len(),
                                    });
                                }
                                self.check_expr(&args[0])?;
                                self.check_expr(&args[1])?;
                                return Ok(Type::Void);
                            }
                            ("System", "time") => {
                                if !args.is_empty() {
                                    return Err(SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 0,
                                        found: args.len(),
                                    });
                                }
                                return Ok(Type::F32);
                            }
                            ("System", "sleep") => {
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
                            _ => return Err(SemanticError::FunctionNotFound(name.clone())),
                        }
                    }
                }

                // --- StdLib FFI (Legacy/Direct) ---
                if name == "tl_file_open" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let _t0 = self.check_expr(&args[0])?;
                    let _t1 = self.check_expr(&args[1])?;
                    return Ok(Type::UserDefined("File".to_string()));
                }
                if name == "tl_file_read_string" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let _t0 = self.check_expr(&args[0])?;
                    return Ok(Type::UserDefined("String".to_string()));
                }
                if name == "tl_file_write_string" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&args[0])?;
                    self.check_expr(&args[1])?;
                    return Ok(Type::Void);
                }
                if name == "tl_file_close" {
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
                if name == "tl_env_get" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let _t0 = self.check_expr(&args[0])?;
                    return Ok(Type::UserDefined("String".to_string()));
                } else if name == "tl_http_download" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&args[0])?;
                    self.check_expr(&args[1])?;
                    return Ok(Type::Bool);
                } else if name == "softmax" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&args[0])?;
                    self.check_expr(&args[1])?; // dim
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "sin" || name == "cos" || name == "relu" || name == "gelu" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&args[0])?;
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "tril" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&args[0])?; // tensor
                    self.check_expr(&args[1])?; // diagonal (int)
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "embedding" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&args[0])?; // indices
                    self.check_expr(&args[1])?; // weights
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "sum" {
                    if args.len() != 1 && args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1, // or 2
                            found: args.len(),
                        });
                    }
                    self.check_expr(&args[0])?;
                    if args.len() == 2 {
                        self.check_expr(&args[1])?; // dim
                    }
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
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
                    if args.len() < 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }

                    let t0 = self.check_expr(&args[0])?;
                    if !matches!(t0, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t0,
                        });
                    }

                    // Allow arg 1 to be tensor (old) OR args 1..N to be Int (new)
                    if args.len() == 2 {
                        let t1 = self.check_expr(&args[1])?;
                        if matches!(t1, Type::Tensor(_, _)) {
                            // Old behavior
                        } else if matches!(t1, Type::I64 | Type::I32) {
                            // New behavior (reshape to flat?)
                        } else {
                            // Error
                        }
                    }

                    // Validate remaining args are Int (if not using shape tensor)
                    let t1 = self.check_expr(&args[1])?;
                    if matches!(t1, Type::Tensor(_, _)) && args.len() == 2 {
                        // OK
                    } else {
                        // Varargs mode: All remaining args must be Int
                        for arg in &args[1..] {
                            let t = self.check_expr(arg)?;
                            if !matches!(t, Type::I64 | Type::I32) {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::I64,
                                    found: t,
                                });
                            }
                        }
                    }

                    if let Type::Tensor(inner, _) = t0 {
                        return Ok(Type::Tensor(inner, 0));
                    }
                    unreachable!("t0 verified as tensor above");
                    // For reshape, return matched type with rank 0 (dynamic).
                    if let Type::Tensor(inner, _) = t0 {
                        return Ok(Type::Tensor(inner, 0));
                    }
                    unreachable!("t0 verified as tensor above");
                } else if name == "len" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    if !matches!(t0, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t0,
                        });
                    }
                    return Ok(Type::I64);
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
                } else if name == "randn" {
                    // randn(shape, requires_grad)
                    // randn(shape, requires_grad)
                    // Return rank 0 (dynamic) to be compatible with any target
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "exp" || name == "log" || name == "sqrt" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    return Ok(t0);
                } else if name == "matmul" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    return Ok(t0); // Propagate type
                } else if name == "grad" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    return Ok(t0);
                } else if name == "backward" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    return Ok(Type::Void);
                } else if name == "save_all_params" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: "save_all_params".into(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    return Ok(Type::Void);
                } else if name == "tl_get_memory_mb" {
                    if args.len() != 0 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: "tl_get_memory_mb".into(),
                            expected: 0,
                            found: args.len(),
                        });
                    }
                    return Ok(Type::I64);
                } else if name == "varbuilder_get" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let _name_type = self.check_expr(&args[0])?;
                    let _shape_type = self.check_expr(&args[1])?;
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "update_all_params" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let _lr_type = self.check_expr(&args[0])?;
                    return Ok(Type::Void);
                } else if name == "varbuilder_grad" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let _name_type = self.check_expr(&args[0])?;
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "softmax" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    let t1 = self.check_expr(&args[1])?;
                    if !matches!(t0, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t0,
                        });
                    }
                    if t1 != Type::I64 {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::I64,
                            found: t1,
                        });
                    }
                    return Ok(t0);
                } else if name == "cross_entropy" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&args[0])?;
                    let t1 = self.check_expr(&args[1])?;
                    if !matches!(t0, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t0,
                        });
                    }
                    if !matches!(t1, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t1,
                        });
                    }
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                }

                if let Some(func) = self.functions.get(name).cloned() {
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
                    // Check arg types for function
                    for (i, arg) in args.iter().enumerate() {
                        if i < func.args.len() {
                            let arg_type = self.check_expr(arg)?;
                            let expected_type = &func.args[i].1;
                            if !self.are_types_compatible(expected_type, &arg_type) {
                                return Err(SemanticError::TypeMismatch {
                                    expected: expected_type.clone(),
                                    found: arg_type,
                                });
                            }
                        }
                    }
                    return Ok(func.return_type.clone());
                }

                if let Some(struct_def) = self.structs.get(name).cloned() {
                    // Struct constructor
                    if args.len() != struct_def.fields.len() {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: struct_def.fields.len(),
                            found: args.len(),
                        });
                    }
                    // Check field types
                    for (i, arg) in args.iter().enumerate() {
                        let arg_ty = self.check_expr(arg)?;
                        let required_ty = &struct_def.fields[i].1;
                        if !self.are_types_compatible(required_ty, &arg_ty) {
                            return Err(SemanticError::TypeMismatch {
                                expected: required_ty.clone(),
                                found: arg_ty,
                            });
                        }
                    }
                    return Ok(Type::UserDefined(name.clone()));
                }

                return Err(SemanticError::FunctionNotFound(name.clone()));
            }
            Expr::TensorLiteral(elements) | Expr::TensorConstLiteral(elements) => {
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
                    return Err(SemanticError::TypeMismatch {
                        expected: Type::Bool,
                        found: cond_type,
                    });
                }

                // Check Then Block
                self.enter_scope();
                let mut then_type = Type::Void;
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

                // Check Else Block
                let else_type = if let Some(else_stmts) = else_block {
                    self.enter_scope();
                    let mut e_type = Type::Void;
                    for (i, stmt) in else_stmts.iter().enumerate() {
                        if i == else_stmts.len() - 1 {
                            if let Stmt::Expr(e) = stmt {
                                e_type = self.check_expr(e)?;
                            } else {
                                self.check_stmt(stmt)?;
                            }
                        } else {
                            self.check_stmt(stmt)?;
                        }
                    }
                    self.exit_scope();
                    e_type
                } else {
                    Type::Void
                };

                // Merge types
                if then_type == else_type {
                    Ok(then_type)
                } else {
                    // Simple compatibility check (e.g. F32 vs I64 casting?)
                    // For strict semantics, error.
                    // But if one is Void (e.g. if-without-else used as expr?), error.
                    // If Expr::IfExpr is used, it expects a value. If else is missing, it returns Void?
                    Err(SemanticError::TypeMismatch {
                        expected: then_type,
                        found: else_type,
                    })
                }
                // End of Expr::IfExpr
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
                // Aggregation returns same type as the expression (for sum/avg)
                // or I64 for count
                Ok(expr_ty)
            }
            Expr::StaticMethodCall(type_name, method_name, args) => {
                // Check if type exists (built-in or user-defined struct)
                // For now, only struct or built-in classes like File, Path, etc. use this.
                // 1. Check if it is a built-in static method (e.g. File::open, Path::new)
                //    This logic needs to be consistent with CodeGen.
                // 2. Check if it is a user-defined struct method (e.g. Linear::new)

                // Check arguments first
                for arg in args {
                    self.check_expr(arg)?;
                }

                let user_func = if let Some(methods_map) = self.methods.get(type_name) {
                    methods_map.get(method_name).cloned()
                } else {
                    None
                };

                if let Some(func) = user_func {
                    if func.args.len() != args.len() {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: format!("{}::{}", type_name, method_name),
                            expected: func.args.len(),
                            found: args.len(),
                        });
                    }
                    for (_i, (arg_val, (_, arg_type))) in args.iter().zip(&func.args).enumerate() {
                        let val_type = self.check_expr(arg_val)?;
                        if !self.are_types_compatible(&val_type, arg_type) {
                            return Err(SemanticError::TypeMismatch {
                                expected: arg_type.clone(),
                                found: val_type,
                            });
                        }
                    }
                    return Ok(func.return_type);
                }

                // Fallback for builtins
                match (type_name.as_str(), method_name.as_str()) {
                    ("File", "open") => {
                        if args.len() != 2 {
                            return Err(SemanticError::ArgumentCountMismatch {
                                name: "File::open".into(),
                                expected: 2,
                                found: args.len(),
                            });
                        }
                        return Ok(Type::UserDefined("File".to_string()));
                    }
                    ("Path", "new") => {
                        if args.len() != 1 {
                            return Err(SemanticError::ArgumentCountMismatch {
                                name: "Path::new".into(),
                                expected: 1,
                                found: args.len(),
                            });
                        }
                        return Ok(Type::UserDefined("Path".to_string()));
                    }
                    ("System", "time") => {
                        return Ok(Type::F64);
                    }
                    ("System", "sleep") => {
                        return Ok(Type::Void);
                    }
                    ("Env", "get") => {
                        return Ok(Type::UserDefined("String".into()));
                    }
                    ("Http", "get") => {
                        return Ok(Type::UserDefined("String".into()));
                    }
                    ("Http", "download") => {
                        return Ok(Type::Void);
                    }
                    _ => {
                        return Err(SemanticError::FunctionNotFound(format!(
                            "{}::{}",
                            type_name, method_name
                        )));
                    }
                }
            }
            Expr::FieldAccess(obj, field_name) => {
                let obj_type = self.check_expr(obj)?;
                let name = match obj_type {
                    Type::UserDefined(n) => n,
                    Type::Struct(n) => n,
                    _ => {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::UserDefined("Struct".into()),
                            found: obj_type,
                        });
                    }
                };

                if let Some(struct_def) = self.structs.get(&name) {
                    for (f_name, f_type) in &struct_def.fields {
                        if f_name == field_name {
                            return Ok(f_type.clone());
                        }
                    }
                    return Err(SemanticError::VariableNotFound(format!(
                        "Field {}",
                        field_name
                    )));
                }
                return Err(SemanticError::StructNotFound(name));
            }
            Expr::MethodCall(obj, method_name, args) => {
                let obj_type = self.check_expr(obj)?;
                let type_name = match &obj_type {
                    Type::UserDefined(name) => name.clone(),
                    Type::Tensor(_, _) => {
                        // Built-in tensor methods
                        if method_name == "backward" {
                            return Ok(Type::Void);
                        }
                        if method_name == "grad" {
                            return Ok(obj_type);
                        } // grads have same shape
                        if method_name == "clone" {
                            return Ok(obj_type);
                        }
                        if method_name == "detach" {
                            return Ok(obj_type);
                        }
                        if method_name == "get" {
                            if args.len() != 1 {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 1,
                                    found: args.len(),
                                });
                            }
                            return Ok(Type::F32);
                        }
                        if method_name == "sum" {
                            // sum() returns a scalar tensor (rank 0) or maintains type
                            match obj_type {
                                Type::Tensor(inner, _) => {
                                    return Ok(Type::Tensor(inner.clone(), 0))
                                }
                                _ => return Ok(obj_type.clone()),
                            }
                        }
                        return Ok(Type::Void); // Fallback
                    }
                    _ => {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::UserDefined("Struct".into()),
                            found: obj_type,
                        });
                    }
                };

                let method_def = if let Some(methods) = self.methods.get(&type_name) {
                    methods.get(method_name).cloned()
                } else {
                    None
                };

                if let Some(func) = method_def {
                    let implicit_self = !func.args.is_empty() && func.args[0].0 == "self";
                    let expected_arg_count = if implicit_self {
                        func.args.len() - 1
                    } else {
                        func.args.len()
                    };

                    if args.len() != expected_arg_count {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: method_name.clone(),
                            expected: expected_arg_count,
                            found: args.len(),
                        });
                    }

                    let mut func_arg_iter = func.args.iter();
                    if implicit_self {
                        let (_, self_type) = func_arg_iter.next().unwrap();
                        if !self.are_types_compatible(self_type, &obj_type) {
                            return Err(SemanticError::TypeMismatch {
                                expected: self_type.clone(),
                                found: obj_type,
                            });
                        }
                    }

                    // Check remaining args
                    for (arg_expr, (_, expected_type)) in args.iter().zip(func_arg_iter) {
                        let arg_type = self.check_expr(arg_expr)?;
                        if !self.are_types_compatible(expected_type, &arg_type) {
                            return Err(SemanticError::TypeMismatch {
                                expected: expected_type.clone(),
                                found: arg_type,
                            });
                        }
                    }
                    Ok(func.return_type)
                } else {
                    // Check StdLib methods
                    match (type_name.as_str(), method_name.as_str()) {
                        ("File", "read_string") => {
                            if !args.is_empty() {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 0,
                                    found: args.len(),
                                });
                            }
                            Ok(Type::UserDefined("String".to_string()))
                        }
                        ("File", "write_string") => {
                            if args.len() != 1 {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 1,
                                    found: args.len(),
                                });
                            }
                            self.check_expr(&args[0])?;
                            Ok(Type::Void)
                        }
                        ("File", "close") => {
                            if !args.is_empty() {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 0,
                                    found: args.len(),
                                });
                            }
                            Ok(Type::Void)
                        }
                        ("Path", "join") => {
                            if args.len() != 1 {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 1,
                                    found: args.len(),
                                });
                            }
                            self.check_expr(&args[0])?;
                            Ok(Type::UserDefined("Path".to_string()))
                        }
                        ("Path", "exists") => {
                            if !args.is_empty() {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 0,
                                    found: args.len(),
                                });
                            }
                            Ok(Type::Bool)
                        }
                        ("Path", "is_dir") => {
                            if !args.is_empty() {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 0,
                                    found: args.len(),
                                });
                            }
                            Ok(Type::Bool)
                        }
                        ("Path", "is_file") => {
                            if !args.is_empty() {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 0,
                                    found: args.len(),
                                });
                            }
                            Ok(Type::Bool)
                        }
                        ("Path", "to_string") => {
                            if !args.is_empty() {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 0,
                                    found: args.len(),
                                });
                            }
                            Ok(Type::UserDefined("String".to_string()))
                        }
                        ("Path", "free") => {
                            if !args.is_empty() {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 0,
                                    found: args.len(),
                                });
                            }
                            Ok(Type::Void)
                        }
                        _ => Err(SemanticError::FunctionNotFound(format!(
                            "{}::{}",
                            type_name, method_name
                        ))),
                    }
                }
            }
        }
    }

    fn are_types_compatible(&self, t1: &Type, t2: &Type) -> bool {
        match (t1, t2) {
            (Type::Tensor(i1, r1), Type::Tensor(i2, r2)) => {
                // If inner types match
                if i1 != i2 {
                    return false;
                }
                // If either rank is 0, we treat it as dynamic/compatible
                if *r1 == 0 || *r2 == 0 {
                    return true;
                }
                // Otherwise strict match
                r1 == r2
            }
            _ => t1 == t2,
        }
    }

    fn collect_indices(&self, expr: &Expr, indices: &mut HashSet<String>) {
        match expr {
            Expr::IndexAccess(target, idxs) => {
                self.collect_indices(target, indices);
                for idx in idxs {
                    if let Expr::Variable(name) = idx {
                        indices.insert(name.clone());
                    } else if let Expr::IndexAccess(_, _) = idx {
                        // Recurse if index itself has structure? Unlikely for now.
                        self.collect_indices(idx, indices);
                    } else {
                        // Ignore literals/expressions in indices for equation collection
                    }
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
