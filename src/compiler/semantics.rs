// src/compiler/semantics.rs
use crate::compiler::ast::*;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SemanticError {
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    #[error("Variable has been moved: {0}")]
    VariableMoved(String),
    #[error("Type mismatch: expected {expected:?}, found {found:?}")]
    TypeMismatch { expected: Type, found: Type },
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
    #[error("Struct not found: {0}")]
    StructNotFound(String),
    #[error("Duplicate definition: {0}")]
    DuplicateDefinition(String),
    #[error("Duplicate match arm for variant: {0}")]
    DuplicateMatchArm(String),
    #[error("Unreachable match arm after wildcard")]
    UnreachableMatchArm,
    #[error("Non-exhaustive match on enum {enum_name}, missing variants: {missing_variants:?}")]
    NonExhaustiveMatch {
        enum_name: String,
        missing_variants: Vec<String>,
    },
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
    #[error("Tuple index out of bounds: index {0} is out of bounds for tuple of size {1}")]
    TupleIndexOutOfBounds(usize, usize),
    #[error("Cannot index into non-tuple type: {0:?}")]
    NotATuple(Type),
}

#[derive(Clone, Debug)]
struct Symbol {
    #[allow(dead_code)]
    name: String,
    ty: Type,
    is_moved: bool, // Track if variable has been moved
                    // potentially more info like mutability, shape info (if constant)
}

struct Scope {
    symbols: HashMap<String, Symbol>,
    aliases: HashMap<String, String>, // Alias -> Fully Qualified Name
}

impl Scope {
    fn new() -> Self {
        Scope {
            symbols: HashMap::new(),
            aliases: HashMap::new(),
        }
    }

    fn insert(&mut self, name: String, ty: Type) {
        self.symbols.insert(
            name.clone(),
            Symbol {
                name,
                ty,
                is_moved: false,
            },
        );
    }

    fn get(&self, name: &str) -> Option<&Symbol> {
        self.symbols.get(name)
    }

    fn get_mut(&mut self, name: &str) -> Option<&mut Symbol> {
        self.symbols.get_mut(name)
    }

    fn add_alias(&mut self, alias: String, full_name: String) {
        self.aliases.insert(alias, full_name);
    }

    fn get_alias(&self, alias: &str) -> Option<&String> {
        self.aliases.get(alias)
    }
}

pub struct SemanticAnalyzer {
    scopes: Vec<Scope>,                                     // Stack of scopes
    functions: HashMap<String, FunctionDef>,                // Global function registry
    structs: HashMap<String, StructDef>,                    // Global struct registry
    enums: HashMap<String, EnumDef>,                        // Global enum registry
    methods: HashMap<String, HashMap<String, FunctionDef>>, // Struct methods
    current_return_type: Option<Type>, // Expected return type for current function
    current_module: String,            // Current module prefix (e.g. "a::b")
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        let mut analyzer = SemanticAnalyzer {
            scopes: vec![Scope::new()], // Global scope
            functions: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            methods: HashMap::new(),
            current_return_type: None,
            current_module: String::new(),
        };
        analyzer.declare_builtins();
        analyzer
    }

    fn declare_builtins(&mut self) {
        // Register Device Enum
        let device_enum = EnumDef {
            name: "Device".to_string(),
            generics: vec![],
            variants: vec![
                VariantDef {
                    name: "Auto".to_string(),
                    fields: vec![],
                },
                VariantDef {
                    name: "Cpu".to_string(),
                    fields: vec![],
                },
                VariantDef {
                    name: "Metal".to_string(),
                    fields: vec![],
                },
                VariantDef {
                    name: "Cuda".to_string(),
                    fields: vec![],
                },
            ],
        };
        self.enums.insert("Device".to_string(), device_enum);
    }

    pub fn enter_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    pub fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn declare_variable(&mut self, name: String, ty: Type) -> Result<(), SemanticError> {
        let is_global = self.scopes.len() == 1; // Immutable borrow
                                                // Shadowing is allowed
        if let Some(scope) = self.scopes.last_mut() {
            // Mutable borrow
            let final_name = if is_global && !self.current_module.is_empty() {
                format!("{}::{}", self.current_module, name)
            } else {
                name
            };
            scope.insert(final_name, ty);
            Ok(())
        } else {
            unreachable!("No scope available")
        }
    }

    fn lookup_variable(&self, name: &str) -> Result<Type, SemanticError> {
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.get(name) {
                // Check if variable has been moved
                if symbol.is_moved {
                    return Err(SemanticError::VariableMoved(name.to_string()));
                }
                return Ok(symbol.ty.clone());
            }
        }

        // Try global resolution
        let resolved = self.resolve_symbol_name(name);
        if resolved != name {
            if let Some(global_scope) = self.scopes.first() {
                if let Some(symbol) = global_scope.get(&resolved) {
                    if symbol.is_moved {
                        return Err(SemanticError::VariableMoved(name.to_string()));
                    }
                    return Ok(symbol.ty.clone());
                }
            }
        }

        Err(SemanticError::VariableNotFound(name.to_string()))
    }

    /// Mark a variable as moved (ownership transferred)
    fn mark_moved(&mut self, name: &str) {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(symbol) = scope.get_mut(name) {
                symbol.is_moved = true;
                return;
            }
        }
    }

    /// Check if a type requires move semantics (ownership transfer)
    fn is_moveable_type(&self, ty: &Type) -> bool {
        matches!(
            ty,
            Type::Tensor(_, _) | Type::Struct(_) | Type::UserDefined(_)
        )
    }

    // --- Main Checking Logic ---

    // Helper to resolve a name based on current scope aliases and module context
    fn resolve_symbol_name(&self, name: &str) -> String {
        // 1. Check aliases
        let parts: Vec<&str> = name.split("::").collect();
        let first_segment = parts[0];

        for scope in self.scopes.iter().rev() {
            if let Some(full_name) = scope.get_alias(first_segment) {
                if parts.len() > 1 {
                    let suffix = parts[1..].join("::");
                    return format!("{}::{}", full_name, suffix);
                } else {
                    return full_name.clone();
                }
            }
        }

        // 2. Try current module prefix
        if !self.current_module.is_empty() {
            let local_full_name = format!("{}::{}", self.current_module, name);
            if self.functions.contains_key(&local_full_name)
                || self.structs.contains_key(&local_full_name)
            {
                return local_full_name;
            }
            // Also check global variables (tensors) in the first scope?
            if let Some(global_scope) = self.scopes.first() {
                if global_scope.get(&local_full_name).is_some() {
                    return local_full_name;
                }
            }
        }

        // 3. Try as is (global or absolute path)
        name.to_string()
    }

    fn resolve_enum_variant(&self, name: &str) -> Option<(EnumDef, VariantDef)> {
        // Try splitting name to find Enum and Variant
        // e.g. "MyEnum::Variant"
        if let Some((enum_name_part, variant_name)) = name.rsplit_once("::") {
            let resolved_enum_name = self.resolve_symbol_name(enum_name_part);
            if let Some(enum_def) = self.enums.get(&resolved_enum_name) {
                if let Some(variant) = enum_def.variants.iter().find(|v| v.name == variant_name) {
                    return Some((enum_def.clone(), variant.clone()));
                }
            }
            // Also try exact match if enum_name_part is alias?
            // resolve_symbol_name handles aliases.
        }
        // What if user used `use MyEnum::Variant`? Then `Variant` is aliased to `MyEnum::Variant`.
        // `resolve_symbol_name` should resolve "Variant" to "MyEnum::Variant".
        let resolved = self.resolve_symbol_name(name);
        if resolved != name {
            // Recursive call with resolved name
            return self.resolve_enum_variant(&resolved);
        }

        None
    }

    fn resolve_user_type(&self, ty: &Type) -> Type {
        if let Type::UserDefined(name) = ty {
            let resolved_name = self.resolve_symbol_name(name);
            if self.structs.contains_key(&resolved_name) {
                return Type::Struct(resolved_name);
            }
            if self.enums.contains_key(&resolved_name) {
                return Type::Enum(resolved_name);
            }
            // Keep as UserDefined if not found (or for Self/generics)
            Type::UserDefined(resolved_name)
        } else {
            ty.clone()
        }
    }

    fn bind_enum_pattern(
        &mut self,
        enum_name: &str,
        enum_def: &EnumDef,
        pattern: &Pattern,
    ) -> Result<Option<usize>, SemanticError> {
        match pattern {
            Pattern::Wildcard => Ok(None),
            Pattern::EnumPattern {
                enum_name: p_enum,
                variant_name,
                bindings,
            } => {
                if !p_enum.is_empty() {
                    let resolved = self.resolve_symbol_name(p_enum);
                    if resolved != enum_name {
                        return Err(SemanticError::VariableNotFound(format!(
                            "Variant {} not found in enum {}",
                            variant_name, enum_name
                        )));
                    }
                }

                let variant_idx = enum_def
                    .variants
                    .iter()
                    .position(|v| v.name == *variant_name)
                    .ok_or_else(|| {
                        SemanticError::VariableNotFound(format!(
                            "Variant {} not found in enum {}",
                            variant_name, enum_name
                        ))
                    })?;
                let variant_def = &enum_def.variants[variant_idx];

                let mut seen_fields = HashSet::new();
                let mut seen_vars = HashSet::new();
                for (field_name, var_name) in bindings {
                    if !seen_fields.insert(field_name.clone()) {
                        return Err(SemanticError::DuplicateDefinition(format!(
                            "Field {} in enum pattern",
                            field_name
                        )));
                    }
                    if !seen_vars.insert(var_name.clone()) {
                        return Err(SemanticError::DuplicateDefinition(format!(
                            "Binding {} in enum pattern",
                            var_name
                        )));
                    }

                    let field_type = variant_def
                        .fields
                        .iter()
                        .find(|(f, _)| f == field_name)
                        .map(|(_, t)| t)
                        .ok_or_else(|| {
                            SemanticError::VariableNotFound(format!(
                                "Field {} in variant {}",
                                field_name, variant_name
                            ))
                        })?;

                    self.declare_variable(var_name.clone(), field_type.clone())?;
                }

                Ok(Some(variant_idx))
            }
        }
    }

    pub fn check_module(&mut self, module: &mut Module) -> Result<(), SemanticError> {
        self.register_module_symbols(module, "")?;
        self.check_module_bodies(module, "")?;
        Ok(())
    }

    fn register_module_symbols(
        &mut self,
        module: &Module, // Register doesn't need to mutate? Except if we rename structs/fns?
        // Current logic: clones S/F, renames clone, inserts clone. Original AST undefs remain "Struct".
        // BUT if we want original AST definition nodes to have explicit names?
        // Actually, we don't need to rename definition nodes if we register them with full names in symbol table.
        // It's the REFERENCES (Use sites) that need resolution.
        // So register pass can stay immutable.
        prefix: &str,
    ) -> Result<(), SemanticError> {
        // Register structs
        for s in &module.structs {
            let full_name = if prefix.is_empty() {
                s.name.clone()
            } else {
                format!("{}::{}", prefix, s.name)
            };
            if self.structs.contains_key(&full_name) {
                return Err(SemanticError::DuplicateDefinition(full_name));
            }
            let mut s_clone = s.clone();
            s_clone.name = full_name.clone();
            self.structs.insert(full_name, s_clone);
        }

        // Register enums
        for e in &module.enums {
            let full_name = if prefix.is_empty() {
                e.name.clone()
            } else {
                format!("{}::{}", prefix, e.name)
            };
            if self.enums.contains_key(&full_name) {
                return Err(SemanticError::DuplicateDefinition(full_name));
            }
            let mut e_clone = e.clone();
            e_clone.name = full_name.clone();
            self.enums.insert(full_name, e_clone);
        }

        // Register functions
        for f in &module.functions {
            let full_name = if prefix.is_empty() {
                f.name.clone()
            } else {
                format!("{}::{}", prefix, f.name)
            };
            if self.functions.contains_key(&full_name) {
                return Err(SemanticError::DuplicateDefinition(full_name));
            }
            let mut f_clone = f.clone();
            f_clone.name = full_name.clone();

            // Resolve types in arguments and return type
            for (_, ty) in &mut f_clone.args {
                *ty = self.resolve_user_type(ty);
            }
            f_clone.return_type = self.resolve_user_type(&f_clone.return_type);

            self.functions.insert(full_name, f_clone);
        }

        // Submodules
        for (name, submodule) in &module.submodules {
            let sub_prefix = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{}::{}", prefix, name)
            };
            self.register_module_symbols(submodule, &sub_prefix)?;
        }

        Ok(())
    }

    fn check_module_bodies(
        &mut self,
        module: &mut Module,
        prefix: &str,
    ) -> Result<(), SemanticError> {
        let saved_prefix = self.current_module.clone();
        self.current_module = prefix.to_string();

        // Check impl blocks (Register methods first)
        for i in &mut module.impls {
            self.check_impl_block(i)?;
        }

        // Check top-level statements (e.g. tensor_decls)
        for s in &mut module.tensor_decls {
            self.check_stmt(s)?;
        }

        // Check submodules BEFORE functions (so their impls are registered first)
        for (name, sub) in &mut module.submodules {
            let sub_prefix = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{}::{}", prefix, name)
            };
            self.check_module_bodies(sub, &sub_prefix)?;
        }

        // Check function bodies
        for f in &mut module.functions {
            self.check_function(f, None)?;
        }

        self.current_module = saved_prefix;
        Ok(())
    }

    fn check_impl_block(&mut self, impl_block: &mut ImplBlock) -> Result<(), SemanticError> {
        // Resolve target struct name
        let resolved_name = self.resolve_symbol_name(&impl_block.target_type);
        if resolved_name != impl_block.target_type {
            impl_block.target_type = resolved_name.clone();
        }

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
                .or_default();
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
        for method in &mut impl_block.methods {
            self.check_function(
                method,
                Some(Type::UserDefined(impl_block.target_type.clone())),
            )?;
        }
        Ok(())
    }

    fn check_function(
        &mut self,
        func: &mut FunctionDef,
        self_type: Option<Type>,
    ) -> Result<(), SemanticError> {
        self.enter_scope();

        // Set expected return type for this function
        self.current_return_type = Some(func.return_type.clone());

        // Register arguments
        for (name, ty) in &mut func.args {
            let actual_ty = if let Type::UserDefined(ref type_name) = ty {
                if type_name == "Self" {
                    // Resolve Self -> Actual Type
                    self_type.clone().ok_or_else(|| {
                        SemanticError::VariableNotFound(
                            "Self type not available in this context".into(),
                        )
                    })?
                } else {
                    self.resolve_user_type(ty)
                }
            } else {
                self.resolve_user_type(ty)
            };
            // If we resolved types, we should update `ty`?
            // Since we are iterating `&mut func.args`.
            *ty = actual_ty.clone(); // Update arg type in AST
            self.declare_variable(name.clone(), actual_ty)?;
        }

        for stmt in &mut func.body {
            self.check_stmt(stmt)?;
        }

        // Clear return type context
        self.current_return_type = None;

        self.exit_scope();
        Ok(())
    }

    pub fn check_stmt(&mut self, stmt: &mut Stmt) -> Result<(), SemanticError> {
        match stmt {
            Stmt::FieldAssign { obj, field, value } => {
                // Check object type and verify it's a struct
                let obj_type = self.check_expr(obj)?;
                let struct_name = match obj_type {
                    Type::UserDefined(name) => name,
                    Type::Struct(name) => name,
                    _ => {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::UserDefined("Struct".into()),
                            found: obj_type,
                        });
                    }
                };

                // Verify struct and field exist
                // Need to resolve struct_name? Usually FieldAccess object type is already resolved by check_expr logic?
                // Type::UserDefined("Name") -> Name should be fully qualified if check_expr(obj) did its job.
                // But check_expr returns Type. If Type comes from AST, it might simple name.
                // The Type returned by check_expr comes from:
                // - Expr::StructInit -> looked up strict name (resolved).
                // - Expr::Variable -> lookup_variable -> returns Type from scope.
                // - Expr::FnCall -> returns return_type of function.
                // So if "Struct" type is stored in scope/function def with FQN, then obj_type has FQN.
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
                let mut final_ty = type_annotation.clone();
                if let Some(expr) = init {
                    let init_ty = self.check_expr(expr)?;
                    // If annotation is primitive but init is tensor, upgrade annotation to tensor
                    if matches!(
                        type_annotation,
                        Type::F32 | Type::F64 | Type::I32 | Type::I64
                    ) {
                        if let Type::Tensor(ref inner, rank) = init_ty {
                            if self.are_types_compatible(type_annotation, inner) {
                                final_ty = Type::Tensor(Box::new(type_annotation.clone()), rank);
                            }
                        } else if let Type::ScalarArray(ref inner, _len) = init_ty {
                            if self.are_types_compatible(type_annotation, inner) {
                                final_ty = Type::Tensor(Box::new(type_annotation.clone()), 1);
                                // ScalarArray is 1D
                            }
                        }
                    } else if !self.are_types_compatible(type_annotation, &init_ty) {
                        return Err(SemanticError::TypeMismatch {
                            expected: type_annotation.clone(),
                            found: init_ty,
                        });
                    }
                }
                self.declare_variable(name.clone(), final_ty)?;
                Ok(())
            }

            Stmt::Let {
                name,
                type_annotation,
                value,
            } => {
                // 1. Infer free indices (Tensor Equation Mode)
                // self.infer_free_indices takes &Expr.
                // But value is &mut Expr. Can treat as &Expr.
                let free_indices = self.infer_free_indices(value);

                let inferred_type = if !free_indices.is_empty() {
                    // Tensor Equation Logic
                    self.enter_scope();

                    // Declare implicitly inferred indices
                    for idx in &free_indices {
                        self.declare_variable(idx.clone(), Type::I64)?;
                    }

                    // Now check RHS with these indices valid
                    let rhs_type = self.check_expr(value)?;

                    self.exit_scope();

                    // Construct Tensor Type
                    Type::Tensor(Box::new(rhs_type), free_indices.len())
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

                self.declare_variable(name.clone(), final_type.clone())?;

                // Move semantics: If RHS is a variable of moveable type, mark it as moved
                if let Expr::Variable(source_var) = value {
                    if self.is_moveable_type(&final_type) {
                        self.mark_moved(source_var);
                    }
                }

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
                                    // Check if valid method name involves resolving?
                                    // These are built-in methods on Tensor?
                                    // Just checks compatibility for now.
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
            Stmt::Return(expr_opt) => {
                let found_type = if let Some(expr) = expr_opt {
                    self.check_expr(expr)?
                } else {
                    Type::Void
                };

                // Check against function return type
                if let Some(ref expected) = self.current_return_type {
                    if !self.are_types_compatible(expected, &found_type) {
                        return Err(SemanticError::TypeMismatch {
                            expected: expected.clone(),
                            found: found_type,
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
                if cond_type != Type::Bool {
                    return Err(SemanticError::TypeMismatch {
                        expected: Type::Bool,
                        found: cond_type,
                    });
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
                let elem_type = match iterator {
                    Expr::Range(start, end) => {
                        let start_ty = self.check_expr(start)?;
                        let end_ty = self.check_expr(end)?;
                        if !matches!(start_ty, Type::I64 | Type::I32)
                            || !matches!(end_ty, Type::I64 | Type::I32)
                        {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: start_ty,
                            });
                        }
                        Type::I64
                    }
                    Expr::FnCall(name, args) if name == "range" && args.len() == 2 => {
                        // Deprecated range() function check
                        let start_type = self.check_expr(&mut args[0])?;
                        let end_type = self.check_expr(&mut args[1])?;
                        if !matches!(start_type, Type::I64 | Type::I32)
                            || !matches!(end_type, Type::I64 | Type::I32)
                        {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: start_type,
                            });
                        }
                        Type::I64
                    }
                    _ => {
                        let iter_type = self.check_expr(iterator)?;
                        match iter_type {
                            Type::Tensor(t, 1) => *t,
                            Type::TensorShaped(t, _) => *t,
                            _ => {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::Tensor(Box::new(Type::F32), 1),
                                    found: iter_type,
                                })
                            }
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
            Stmt::Use { path, alias, items } => {
                let full_prefix = path.join("::");

                if !items.is_empty() {
                    // use path::{items...}
                    for item in items {
                        // import path::item as item
                        let full_name = format!("{}::{}", full_prefix, item);
                        let alias_name = item.clone();
                        self.scopes
                            .last_mut()
                            .unwrap()
                            .add_alias(alias_name, full_name);
                    }
                } else {
                    // use path [as alias]
                    let alias_name = if let Some(a) = alias {
                        a.clone()
                    } else {
                        path.last()
                            .ok_or(SemanticError::VariableNotFound("Empty use path".into()))?
                            .clone()
                    };
                    self.scopes
                        .last_mut()
                        .unwrap()
                        .add_alias(alias_name, full_prefix);
                }
                Ok(())
            }
        }
    }

    pub fn check_expr(&mut self, expr: &mut Expr) -> Result<Type, SemanticError> {
        match expr {
            Expr::Int(_) => Ok(Type::I64),   // Default integer literal type
            Expr::Float(_) => Ok(Type::F32), // Default float literal type
            Expr::Bool(_) => Ok(Type::Bool),
            Expr::StringLiteral(_) => Ok(Type::UserDefined("String".to_string())), // Placeholder
            Expr::Tuple(exprs) => {
                let mut types = Vec::new();
                for e in exprs {
                    types.push(self.check_expr(e)?);
                }
                Ok(Type::Tuple(types))
            }
            Expr::TupleAccess(expr, idx) => {
                let ty = self.check_expr(expr)?;
                if let Type::Tuple(types) = ty {
                    if *idx < types.len() {
                        Ok(types[*idx].clone())
                    } else {
                        Err(SemanticError::TupleIndexOutOfBounds(*idx, types.len()))
                    }
                } else {
                    Err(SemanticError::NotATuple(ty))
                }
            }
            Expr::StructInit(name, fields) => {
                let resolved_name = self.resolve_symbol_name(name);
                if *name != resolved_name {
                    *name = resolved_name.clone();
                }

                if let Some(struct_def) = self.structs.get(name).cloned() {
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
                } else if let Some((enum_def, variant_def)) = self.resolve_enum_variant(name) {
                    // It is an Enum Variant! Transform to EnumInit.
                    let fields_owned = std::mem::take(fields);
                    *expr = Expr::EnumInit {
                        enum_name: enum_def.name.clone(),
                        variant_name: variant_def.name.clone(),
                        fields: fields_owned,
                    };
                    // Re-check as EnumInit
                    self.check_expr(expr)
                } else {
                    Err(SemanticError::StructNotFound(name.clone()))
                }
            }
            Expr::EnumInit {
                enum_name,
                variant_name,
                fields,
            } => {
                let enum_def = self
                    .enums
                    .get(enum_name)
                    .ok_or_else(|| SemanticError::StructNotFound(enum_name.clone()))?
                    .clone(); // Clone to avoid borrow issues

                let variant_def = enum_def
                    .variants
                    .iter()
                    .find(|v| v.name == *variant_name)
                    .ok_or_else(|| {
                        SemanticError::VariableNotFound(format!(
                            "Variant {} in enum {}",
                            variant_name, enum_name
                        ))
                    })?;

                let mut initialized_fields = HashSet::new();
                for (field_name, field_expr) in fields {
                    if initialized_fields.contains(field_name) {
                        return Err(SemanticError::DuplicateDefinition(format!(
                            "Field {} in enum variant init",
                            field_name
                        )));
                    }
                    initialized_fields.insert(field_name.clone());

                    let expected_type = variant_def
                        .fields
                        .iter()
                        .find(|(f, _)| f == field_name)
                        .map(|(_, t)| t)
                        .ok_or_else(|| {
                            SemanticError::VariableNotFound(format!(
                                "Field {} in variant {}",
                                field_name, variant_name
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
                for (field_name, _) in &variant_def.fields {
                    if !initialized_fields.contains(field_name) {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: format!("Variant init {}", variant_name),
                            expected: variant_def.fields.len(),
                            found: initialized_fields.len(),
                        });
                    }
                }

                Ok(Type::Enum(enum_name.clone()))
            }
            Expr::Match {
                expr: subject_expr,
                arms,
            } => {
                let subject_type = self.check_expr(subject_expr)?;
                let enum_name = match &subject_type {
                    Type::Enum(n) | Type::UserDefined(n) => n.clone(),
                    _ => {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::UserDefined("Enum".into()),
                            found: subject_type,
                        })
                    }
                };

                let enum_def = self.enums.get(&enum_name);
                if enum_def.is_none() {
                    // If it's not an enum, maybe we support matching on other types later?
                    // For now only Enums (UserDefined that maps to Enum).
                    return Err(SemanticError::StructNotFound(format!("Enum {}", enum_name)));
                }
                let enum_def = enum_def.unwrap().clone();

                let mut return_type = Option::<Type>::None;
                let mut seen_variants = HashSet::new();
                let mut saw_wildcard = false;

                for (pattern, arm_expr) in arms {
                    if saw_wildcard {
                        return Err(SemanticError::UnreachableMatchArm);
                    }

                    self.enter_scope();

                    let variant_idx = self.bind_enum_pattern(&enum_name, &enum_def, pattern)?;

                    let arm_type = self.check_expr(arm_expr)?;
                    self.exit_scope();

                    if let Some(idx) = variant_idx {
                        let variant_name = &enum_def.variants[idx].name;
                        if !seen_variants.insert(idx) {
                            return Err(SemanticError::DuplicateMatchArm(variant_name.clone()));
                        }
                    } else {
                        saw_wildcard = true;
                    }

                    if let Some(ref rt) = return_type {
                        if !self.are_types_compatible(rt, &arm_type) {
                            return Err(SemanticError::TypeMismatch {
                                expected: rt.clone(),
                                found: arm_type,
                            });
                        }
                    } else {
                        return_type = Some(arm_type);
                    }
                }

                if !saw_wildcard {
                    let mut missing = Vec::new();
                    for (idx, variant) in enum_def.variants.iter().enumerate() {
                        if !seen_variants.contains(&idx) {
                            missing.push(variant.name.clone());
                        }
                    }
                    if !missing.is_empty() {
                        return Err(SemanticError::NonExhaustiveMatch {
                            enum_name,
                            missing_variants: missing,
                        });
                    }
                }

                Ok(return_type.unwrap_or(Type::Void))
            }
            Expr::IfLet {
                pattern,
                expr,
                then_block,
                else_block,
            } => {
                let subject_type = self.check_expr(expr)?;
                let enum_name = match &subject_type {
                    Type::Enum(n) | Type::UserDefined(n) => n.clone(),
                    _ => {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::UserDefined("Enum".into()),
                            found: subject_type,
                        })
                    }
                };

                let enum_def = self.enums.get(&enum_name);
                if enum_def.is_none() {
                    return Err(SemanticError::StructNotFound(format!("Enum {}", enum_name)));
                }
                let enum_def = enum_def.unwrap().clone();

                // Then block with bindings
                self.enter_scope();
                self.bind_enum_pattern(&enum_name, &enum_def, pattern)?;
                let mut then_type = Type::Void;
                let then_len = then_block.len();
                for (i, stmt) in then_block.iter_mut().enumerate() {
                    if i == then_len - 1 {
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

                let else_type = if let Some(else_stmts) = else_block {
                    self.enter_scope();
                    let mut e_type = Type::Void;
                    let else_len = else_stmts.len();
                    for (i, stmt) in else_stmts.iter_mut().enumerate() {
                        if i == else_len - 1 {
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

                if then_type == else_type {
                    Ok(then_type)
                } else {
                    Err(SemanticError::TypeMismatch {
                        expected: then_type,
                        found: else_type,
                    })
                }
            }
            Expr::Range(start, end) => {
                let s_ty = self.check_expr(start)?;
                let e_ty = self.check_expr(end)?;
                if !matches!(s_ty, Type::I64 | Type::I32) || !matches!(e_ty, Type::I64 | Type::I32)
                {
                    return Err(SemanticError::TypeMismatch {
                        expected: Type::I64,
                        found: s_ty,
                    });
                }
                // Range expression itself doesn't evaluate to a runtime value outside of for-loops yet,
                // but we return Void or a placeholder.
                Ok(Type::Void)
            }
            Expr::Variable(name) => {
                // 1. Try local scopes first (reverse order)
                for scope in self.scopes.iter().rev() {
                    if let Some(symbol) = scope.get(name) {
                        return Ok(symbol.ty.clone());
                    }
                }

                // 2. Try global resolution
                let resolved = self.resolve_symbol_name(name);
                if resolved != *name {
                    if let Some(global_scope) = self.scopes.first() {
                        if let Some(symbol) = global_scope.get(&resolved) {
                            *name = resolved.clone();
                            return Ok(symbol.ty.clone());
                        }
                    }
                }

                // 3. Last attempt: maybe simple name is in global (e.g. top-level, no prefix)
                if let Some(global_scope) = self.scopes.first() {
                    if let Some(symbol) = global_scope.get(name) {
                        return Ok(symbol.ty.clone());
                    }
                }

                // 4. Try as Enum Unit Variant
                if let Some((enum_def, variant_def)) = self.resolve_enum_variant(name) {
                    if !variant_def.fields.is_empty() {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::UserDefined("Unit Variant".into()),
                            found: Type::UserDefined("Struct Variant".into()),
                        });
                    }
                    *expr = Expr::EnumInit {
                        enum_name: enum_def.name.clone(),
                        variant_name: variant_def.name.clone(),
                        fields: vec![],
                    };
                    return self.check_expr(expr);
                }

                Err(SemanticError::VariableNotFound(name.clone()))
            }
            Expr::BinOp(lhs, op, rhs) => {
                let left = self.check_expr(lhs)?;
                let right = self.check_expr(rhs)?;
                // match op {
                //     _ => {}
                // }
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
                // Resolve name first
                let resolved_name = self.resolve_symbol_name(name);
                if *name != resolved_name {
                    *name = resolved_name.clone();
                }

                // Handle set_device builtin
                if name == "set_device" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let arg_ty = self.check_expr(&mut args[0])?;
                    // Expect Device enum
                    match &arg_ty {
                        Type::Enum(e) | Type::UserDefined(e) if e == "Device" => {}
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::UserDefined("Device".into()),
                                found: arg_ty,
                            })
                        }
                    }
                    return Ok(Type::Void);
                }

                if name == "checkpoint" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    // Special handling for arg 0: Must be obj.method (FieldAccess)
                    // We cannot call check_expr on it because methods are not fields.
                    if let Expr::FieldAccess(obj_expr, _method_name) = &mut args[0] {
                        let _obj_type = self.check_expr(obj_expr)?;
                        // Ideally check if method exists, but we trust codegen or runtime for now.
                        // Or check struct definition if obj_type is Struct.
                    } else {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::UserDefined("obj.method".into()),
                            found: Type::Void, // placeholder
                        });
                    }

                    let t1 = self.check_expr(&mut args[1])?;
                    if !matches!(t1, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t1,
                        });
                    }

                    // Return generic tensor
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                }

                if name == "cross_entropy" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    let t1 = self.check_expr(&mut args[1])?;
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
                    let t0 = self.check_expr(&mut args[0])?;
                    let t1 = self.check_expr(&mut args[1])?;
                    if !matches!(t0, Type::Tensor(_, _) | Type::F32 | Type::I64) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t0,
                        });
                    }
                    if !matches!(t1, Type::Tensor(_, _) | Type::F32 | Type::I64) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t1,
                        });
                    }
                    if matches!(t0, Type::F32 | Type::I64) {
                        return Ok(Type::Tensor(Box::new(Type::F32), 0));
                    }
                    return Ok(t0);
                }

                if name == "argmax" {
                    // argmax(tensor, dim) -> tensor(f32 indices, need to be careful with type or just return tensor)
                    // runtime returns Tensor<i64> conceptually but OpaqueTensor wraps generic.
                    // We'll treat it as Tensor<f32/i64> (generic tensor)
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let t = self.check_expr(&mut args[0])?;
                    if !matches!(t, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t,
                        });
                    }
                    let dim = self.check_expr(&mut args[1])?;
                    if !matches!(dim, Type::I64) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::I64,
                            found: dim,
                        });
                    }
                    // Returns generic Tensor
                    return Ok(Type::Tensor(Box::new(Type::F32), 1));
                }

                if name == "item" {
                    // item(tensor) -> i64 (for this specific builtin)
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t = self.check_expr(&mut args[0])?;
                    if !matches!(t, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t,
                        });
                    }
                    return Ok(Type::I64);
                }

                if name == "enable_grad" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: "enable_grad".into(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    if !matches!(t0, Type::Tensor(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t0,
                        });
                    }
                    return Ok(t0);
                }
                if name == "print" || name == "println" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&mut args[0])?;
                    return Ok(Type::Void);
                }
                if name == "save_all_params" {
                    if args.len() == 1 {
                        let t0 = self.check_expr(&mut args[0])?;
                        match t0 {
                            Type::UserDefined(s) if s == "String" => {}
                            _ => {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::UserDefined("String".into()),
                                    found: t0,
                                })
                            }
                        }
                    } else if args.len() == 2 {
                        let t0 = self.check_expr(&mut args[0])?;
                        match t0 {
                            Type::Struct(_) | Type::UserDefined(_) => {}
                            _ => {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::UserDefined("Struct".into()),
                                    found: t0,
                                })
                            }
                        }
                        let t1 = self.check_expr(&mut args[1])?;
                        match t1 {
                            Type::UserDefined(s) if s == "String" => {}
                            _ => {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::UserDefined("String".into()),
                                    found: t1,
                                })
                            }
                        }
                    } else {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2, // or 1
                            found: args.len(),
                        });
                    }
                    return Ok(Type::Void);
                }
                if name == "load_all_params" {
                    if args.len() == 1 {
                        let t0 = self.check_expr(&mut args[0])?;
                        match t0 {
                            Type::UserDefined(s) if s == "String" => {}
                            _ => {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::UserDefined("String".into()),
                                    found: t0,
                                })
                            }
                        }
                    } else if args.len() == 2 {
                        let t0 = self.check_expr(&mut args[0])?;
                        match t0 {
                            Type::Struct(_) | Type::UserDefined(_) => {}
                            _ => {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::UserDefined("Struct".into()),
                                    found: t0,
                                })
                            }
                        }
                        let t1 = self.check_expr(&mut args[1])?;
                        match t1 {
                            Type::UserDefined(s) if s == "String" => {}
                            _ => {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::UserDefined("String".into()),
                                    found: t1,
                                })
                            }
                        }
                    } else {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2, // or 1
                            found: args.len(),
                        });
                    }
                    return Ok(Type::Void);
                }

                if name == "tl_system_time" {
                    if !args.is_empty() {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 0,
                            found: args.len(),
                        });
                    }
                    return Ok(Type::F32);
                }

                if name == "register_modules" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    match t0 {
                        Type::Struct(_) | Type::UserDefined(_) => {
                            // Ideally check if it maps to a known struct, but UserDefined usually implies valid type if checked elsewhere
                        }
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::UserDefined("Struct".into()),
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
                    let t0 = self.check_expr(&mut args[0])?;
                    match t0 {
                        Type::UserDefined(s) if s == "String" => {}
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::UserDefined("String".into()),
                                found: t0,
                            })
                        }
                    }
                    let t1 = self.check_expr(&mut args[1])?;
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
                    let t0 = self.check_expr(&mut args[0])?;
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
                // --- StdLib Static Methods ---
                // Transferred to Expr::StaticMethodCall handling.
                // The parser ensures that identifiers with "::" are parsed as StaticMethodCall.

                // --- StdLib FFI (Legacy/Direct) ---
                if name == "tl_file_open" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let _t0 = self.check_expr(&mut args[0])?;
                    let _t1 = self.check_expr(&mut args[1])?;
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
                    let _t0 = self.check_expr(&mut args[0])?;
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
                    self.check_expr(&mut args[0])?;
                    self.check_expr(&mut args[1])?;
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
                    self.check_expr(&mut args[0])?;
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
                    let _t0 = self.check_expr(&mut args[0])?;
                    return Ok(Type::UserDefined("String".to_string()));
                } else if name == "tl_http_download" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&mut args[0])?;
                    self.check_expr(&mut args[1])?;
                    return Ok(Type::Bool);
                } else if name == "tl_vec_u8_len" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&mut args[0])?;
                    return Ok(Type::I64);
                } else if name == "tl_vec_u8_get" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&mut args[0])?;
                    self.check_expr(&mut args[1])?;
                    return Ok(Type::U8);
                } else if name == "tl_vec_u8_free" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&mut args[0])?;
                    return Ok(Type::Void);
                } else if name == "sin" || name == "cos" || name == "relu" || name == "gelu" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&mut args[0])?;
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "tril" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&mut args[0])?; // tensor
                    self.check_expr(&mut args[1])?; // diagonal (int)
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "embedding" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    self.check_expr(&mut args[0])?; // indices
                    self.check_expr(&mut args[1])?; // weights
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "sum" {
                    if args.len() != 1 && args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1, // or 2
                            found: args.len(),
                        });
                    }
                    self.check_expr(&mut args[0])?;
                    if args.len() == 2 {
                        self.check_expr(&mut args[1])?; // dim
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
                    let t0 = self.check_expr(&mut args[0])?;
                    let t1 = self.check_expr(&mut args[1])?;
                    let t2 = self.check_expr(&mut args[2])?;

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

                    let t0 = self.check_expr(&mut args[0])?;
                    // Allow Tensor OR ScalarArray
                    if !matches!(t0, Type::Tensor(_, _) | Type::ScalarArray(_, _)) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: t0,
                        });
                    }

                    // Allow arg 1 to be tensor/array (old) OR args 1..N to be Int (new)
                    let t1 = self.check_expr(&mut args[1])?;
                    if (matches!(t1, Type::Tensor(_, _)) || matches!(t1, Type::ScalarArray(_, _)))
                        && args.len() == 2
                    {
                        // OK
                    } else {
                        // Varargs mode: All remaining args must be Int
                        for arg in &mut args[1..] {
                            let t = self.check_expr(arg)?;
                            if !matches!(t, Type::I64 | Type::I32) {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::I64,
                                    found: t,
                                });
                            }
                        }
                    }

                    let inner_type = match t0 {
                        Type::Tensor(inner, _) => inner,
                        Type::ScalarArray(inner, _) => inner,
                        _ => unreachable!(),
                    };

                    // Inference Logic: Inspect args[1] AST if it's a literal
                    let new_rank = if let Expr::TensorLiteral(elements) = &args[1] {
                        elements.len()
                    } else if let Expr::TensorConstLiteral(elements) = &args[1] {
                        elements.len()
                    } else if args.len() > 2 {
                        // Varargs mode: reshape(t, d1, d2, ...) -> Rank = args.len() - 1
                        args.len() - 1
                    } else {
                        0 // Fallback to 0 (dynamic/unknown)
                    };

                    return Ok(Type::Tensor(inner_type, new_rank));
                } else if name == "len" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&mut args[0])?;
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
                    let t0 = self.check_expr(&mut args[0])?;
                    let t1 = self.check_expr(&mut args[1])?;
                    let t2 = self.check_expr(&mut args[2])?;

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
                    // LEGACY: Removed in favor of Tensor::randn
                    return Err(SemanticError::FunctionNotFound(
                        "randn is removed. Use Tensor::randn(shape, req_grad)".into(),
                    ));
                    // return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "exp" || name == "log" || name == "sqrt" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    return Ok(t0);
                } else if name == "matmul" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    return Ok(t0); // Propagate type
                } else if name == "grad" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&mut args[0])?;
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
                    if !args.is_empty() {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: "tl_get_memory_mb".into(),
                            expected: 0,
                            found: args.len(),
                        });
                    }
                    return Ok(Type::I64);
                } else if name == "tl_file_read_binary" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let t = self.check_expr(&mut args[0])?;
                    if !matches!(&t, Type::UserDefined(s) if s == "String") {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::UserDefined("String".into()),
                            found: t,
                        });
                    }
                    return Ok(Type::Vec(Box::new(Type::U8)));
                } else if name == "tl_vec_u8_read_i32_be" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    if !matches!(t0, Type::Vec(ref inner) if **inner == Type::U8) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::Vec(Box::new(Type::U8)),
                            found: t0,
                        });
                    }
                    // Arg 1: Index (int)
                    let t1 = self.check_expr(&mut args[1])?;
                    if !matches!(t1, Type::I64 | Type::I32) {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::I64,
                            found: t1,
                        });
                    }
                    return Ok(Type::I64);
                } else if name == "tl_tensor_from_vec_u8" {
                    if args.len() != 4 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 4,
                            found: args.len(),
                        });
                    }
                    for arg in args.iter_mut() {
                        let _ = self.check_expr(arg)?;
                    }
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "tl_tensor_from_u8_labels" {
                    if args.len() != 3 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 3,
                            found: args.len(),
                        });
                    }
                    for arg in args.iter_mut() {
                        let _ = self.check_expr(arg)?;
                    }
                    return Ok(Type::Tensor(Box::new(Type::I64), 1));
                } else if name == "tl_arena_get_offset" || name == "tl_arena_get_capacity" {
                    if !args.is_empty() {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 0,
                            found: args.len(),
                        });
                    }
                    return Ok(Type::I64);
                } else if name == "tl_arena_is_active" {
                    if !args.is_empty() {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: "tl_arena_is_active".into(),
                            expected: 0,
                            found: args.len(),
                        });
                    }
                    return Ok(Type::Bool);
                } else if name == "tl_arena_alloc" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: "tl_arena_alloc".into(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    return Ok(Type::I64); // returns ptr as i64 for testing
                } else if name == "tl_arena_reset" {
                    if !args.is_empty() {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: "tl_arena_reset".into(),
                            expected: 0,
                            found: args.len(),
                        });
                    }
                    return Ok(Type::Void);
                } else if name == "tl_arena_init" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: "tl_arena_init".into(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    return Ok(Type::Void);
                } else if name == "varbuilder_get" {
                    return Err(SemanticError::FunctionNotFound(
                        "varbuilder_get is removed. Use VarBuilder::get(name, shape)".into(),
                    ));
                } else if name == "update_all_params" {
                    if args.len() != 1 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1,
                            found: args.len(),
                        });
                    }
                    let _lr_type = self.check_expr(&mut args[0])?;
                    return Ok(Type::Void);
                } else if name == "varbuilder_grad" {
                    return Err(SemanticError::FunctionNotFound(
                        "varbuilder_grad is removed. Use VarBuilder::grad(name)".into(),
                    ));
                } else if name == "softmax" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    let t1 = self.check_expr(&mut args[1])?;
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
                    let t0 = self.check_expr(&mut args[0])?;
                    let t1 = self.check_expr(&mut args[1])?;
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
                } else if name == "save_weights" {
                    if args.len() != 2 {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 2,
                            found: args.len(),
                        });
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    let t1 = self.check_expr(&mut args[1])?;

                    // Arg 0: Tensor OR Struct
                    match t0 {
                        Type::Tensor(_, _) => {}
                        Type::UserDefined(ref s) if s != "String" => {}
                        Type::Struct(_) => {}
                        _ => {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::UserDefined("Tensor or Struct".into()),
                                found: t0,
                            })
                        }
                    }

                    if !matches!(t1, Type::UserDefined(ref s) if s == "String") {
                        return Err(SemanticError::TypeMismatch {
                            expected: Type::UserDefined("String".into()),
                            found: t1,
                        });
                    }
                    return Ok(Type::Void);
                } else if name == "load_weights" {
                    if args.len() == 1 {
                        let t0 = self.check_expr(&mut args[0])?;
                        if !matches!(t0, Type::UserDefined(ref s) if s == "String") {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::UserDefined("String".into()),
                                found: t0,
                            });
                        }
                        return Ok(Type::Tensor(Box::new(Type::F32), 0));
                    } else if args.len() == 2 {
                        let t0 = self.check_expr(&mut args[0])?;
                        let t1 = self.check_expr(&mut args[1])?;
                        match t0 {
                            Type::UserDefined(ref s) if s != "String" => {}
                            Type::Struct(_) => {}
                            _ => {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::UserDefined("Struct".into()),
                                    found: t0,
                                })
                            }
                        }
                        if !matches!(t1, Type::UserDefined(ref s) if s == "String") {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::UserDefined("String".into()),
                                found: t1,
                            });
                        }
                        return Ok(Type::Void);
                    } else {
                        return Err(SemanticError::ArgumentCountMismatch {
                            name: name.clone(),
                            expected: 1, // or 2
                            found: args.len(),
                        });
                    }
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
                    for (i, arg) in args.iter_mut().enumerate() {
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
                    for (i, arg) in args.iter_mut().enumerate() {
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

                Err(SemanticError::FunctionNotFound(name.clone()))
            }
            Expr::TensorLiteral(elements) | Expr::TensorConstLiteral(elements) => {
                // Check all elements are same type
                if elements.is_empty() {
                    return Ok(Type::Tensor(Box::new(Type::F32), 1)); // Empty tensor?
                }
                let mut first_type = self.check_expr(&mut elements[0])?;
                let mut has_float = matches!(first_type, Type::F32 | Type::F64);

                for e in &mut elements[1..] {
                    let t = self.check_expr(e)?;
                    if t != first_type {
                        // Allow mixing I64 and F32/F64
                        if (first_type == Type::I64 && matches!(t, Type::F32 | Type::F64))
                            || (matches!(first_type, Type::F32 | Type::F64) && t == Type::I64)
                        {
                            has_float = true;
                            // Promote to F32 if not already
                            if first_type == Type::I64 {
                                first_type = Type::F32;
                            }
                        } else {
                            return Err(SemanticError::TypeMismatch {
                                expected: first_type,
                                found: t,
                            });
                        }
                    }
                }

                // Construct Tensor type.
                if has_float {
                    Ok(Type::Tensor(Box::new(Type::F32), 1))
                } else {
                    match first_type {
                        Type::Tensor(inner, rank) => Ok(Type::Tensor(inner, rank + 1)),
                        primitive => Ok(Type::Tensor(Box::new(primitive), 1)),
                    }
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
            Expr::TensorComprehension { indices, body } => {
                // 1. Enter scope to declare indices
                self.enter_scope();

                // 2. Declare loop variables (dimensions) as integers
                for idx in indices.iter() {
                    self.declare_variable(idx.clone(), Type::I64)?;
                }

                // 3. Infer implicit reduction variables (free vars in body NOT in indices)
                let free_indices_in_body = self.infer_free_indices(body);
                let reduction_indices: Vec<String> = free_indices_in_body
                    .into_iter()
                    .filter(|v| !indices.contains(v))
                    .collect();

                // Declare reduction indices too so body checks pass (they are implicit loops)
                for ridx in &reduction_indices {
                    // Check if already declared (might be outer scope var, or truly implicit)
                    if self.lookup_variable(ridx).is_err() {
                        self.declare_variable(ridx.clone(), Type::I64)?;
                    }
                }

                // 4. Check body type
                let body_type = self.check_expr(body)?;

                // 5. Exit scope
                self.exit_scope();

                // 6. Result Type is Tensor<BodyType, Rank = indices.len()>
                Ok(Type::Tensor(Box::new(body_type), indices.len()))
            }

            Expr::Block(stmts) => {
                self.enter_scope();
                let mut ret_type = Type::Void;
                let stmts_len = stmts.len();
                for (i, stmt) in stmts.iter_mut().enumerate() {
                    if i == stmts_len - 1 {
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
                let then_len = then_block.len();
                for (i, stmt) in then_block.iter_mut().enumerate() {
                    if i == then_len - 1 {
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
                    let else_len = else_stmts.len();
                    for (i, stmt) in else_stmts.iter_mut().enumerate() {
                        if i == else_len - 1 {
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
            Expr::As(expr, target_type) => {
                let source_type = self.check_expr(expr)?;

                // Allow trivial cast
                if source_type == *target_type {
                    return Ok(target_type.clone());
                }

                // Cast logic
                match (&source_type, &*target_type) {
                    (Type::Tensor(_inner_src, _), Type::Tensor(inner_dst, target_rank)) => {
                        // Trust the cast target for rank
                        Ok(Type::Tensor(inner_dst.clone(), *target_rank))
                    }
                    (Type::Tensor(_, rank), primitive)
                        if matches!(primitive, Type::F32 | Type::I64 | Type::I32 | Type::Bool) =>
                    {
                        // `tensor as f32` -> Tensor<f32, rank>
                        Ok(Type::Tensor(Box::new(primitive.clone()), *rank))
                    }
                    (Type::I64, Type::F32) => Ok(Type::F32),
                    (Type::F32, Type::I64) => Ok(Type::I64),
                    (Type::I64, Type::I32) => Ok(Type::I32),
                    (Type::I32, Type::I64) => Ok(Type::I64),
                    (Type::Bool, Type::I64) => Ok(Type::I64),
                    (Type::I64, Type::Bool) => Ok(Type::Bool),
                    (Type::Bool, Type::F32) => Ok(Type::F32),
                    (Type::F32, Type::Bool) => Ok(Type::Bool),
                    _ => Err(SemanticError::TypeMismatch {
                        expected: target_type.clone(),
                        found: source_type,
                    }),
                }
            }
            Expr::StaticMethodCall(type_name, method_name, args) => {
                let resolved_type = self.resolve_symbol_name(type_name);
                if *type_name != resolved_type {
                    *type_name = resolved_type.clone();
                }

                // Check if type exists (built-in or user-defined struct)
                // For now, only struct or built-in classes like File, Path, etc. use this.
                // 1. Check if it is a built-in static method (e.g. File::open, Path::new)
                //    This logic needs to be consistent with CodeGen.
                // 2. Check if it is a user-defined struct method (e.g. Linear::new)

                // Check arguments first
                for arg in args.iter_mut() {
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
                    for (arg_val, (_, arg_type)) in args.iter_mut().zip(&func.args) {
                        let val_type = self.check_expr(arg_val)?;
                        if !self.are_types_compatible(&val_type, arg_type) {
                            return Err(SemanticError::TypeMismatch {
                                expected: arg_type.clone(),
                                found: val_type,
                            });
                        }
                    }
                    // Resolve return type: if it's Self or short name, use method's impl target type
                    let resolved_return = match &func.return_type {
                        Type::UserDefined(ret_name) => {
                            if ret_name == "Self"
                                || ret_name == type_name.split("::").last().unwrap_or(type_name)
                            {
                                Type::UserDefined(type_name.clone())
                            } else {
                                func.return_type.clone()
                            }
                        }
                        _ => func.return_type.clone(),
                    };
                    return Ok(resolved_return);
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
                        Ok(Type::UserDefined("File".to_string()))
                    }
                    ("Path", "new") => {
                        if args.len() != 1 {
                            return Err(SemanticError::ArgumentCountMismatch {
                                name: "Path::new".into(),
                                expected: 1,
                                found: args.len(),
                            });
                        }
                        Ok(Type::UserDefined("Path".to_string()))
                    }
                    ("System", "time") => Ok(Type::F32),
                    ("System", "sleep") => Ok(Type::Void),
                    ("Env", "get") => Ok(Type::UserDefined("String".into())),
                    ("Env", "set") => {
                        if args.len() != 2 {
                            return Err(SemanticError::ArgumentCountMismatch {
                                name: "Env::set".into(),
                                expected: 2,
                                found: args.len(),
                            });
                        }
                        Ok(Type::Void)
                    }
                    ("Http", "get") => Ok(Type::UserDefined("String".into())),
                    ("Http", "download") => Ok(Type::Bool),
                    ("Image", "load_grayscale") => Ok(Type::Vec(Box::new(Type::U8))),
                    ("Image", "width") => Ok(Type::I64),
                    ("Image", "height") => Ok(Type::I64),
                    // --- New Static Methods for Refactor ---
                    ("Tensor", "zeros") => {
                        // Tensor::zeros(shape, requires_grad)
                        if args.is_empty() || args.len() > 2 {
                            return Err(SemanticError::ArgumentCountMismatch {
                                name: "Tensor::zeros".into(),
                                expected: 2,
                                found: args.len(),
                            });
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        if args.len() == 2 {
                            let t1 = self.check_expr(&mut args[1])?;
                            if !matches!(t1, Type::Bool) {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::Bool,
                                    found: t1,
                                });
                            }
                        }
                        Ok(Type::Tensor(Box::new(Type::F32), 0))
                    }
                    ("Tensor", "randn") => {
                        // Tensor::randn(shape, requires_grad)
                        if args.is_empty() || args.len() > 2 {
                            return Err(SemanticError::ArgumentCountMismatch {
                                name: "Tensor::randn".into(),
                                expected: 2,
                                found: args.len(),
                            });
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        if args.len() == 2 {
                            let t1 = self.check_expr(&mut args[1])?;
                            if !matches!(t1, Type::Bool) {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::Bool,
                                    found: t1,
                                });
                            }
                        }
                        Ok(Type::Tensor(Box::new(Type::F32), 0))
                    }
                    ("VarBuilder", "get") => {
                        if args.len() < 2 {
                            return Err(SemanticError::ArgumentCountMismatch {
                                name: "VarBuilder::get".into(),
                                expected: 2,
                                found: args.len(),
                            });
                        }
                        let t0 = self.check_expr(&mut args[0])?;
                        if !matches!(t0, Type::UserDefined(ref s) if s == "String") {
                            return Err(SemanticError::TypeMismatch {
                                expected: Type::UserDefined("String".into()),
                                found: t0,
                            });
                        }
                        if args.len() == 2 {
                            let _ = self.check_expr(&mut args[1])?;
                        } else {
                            for arg in &mut args[1..] {
                                let t = self.check_expr(arg)?;
                                if !matches!(t, Type::I64 | Type::I32) {
                                    return Err(SemanticError::TypeMismatch {
                                        expected: Type::I64,
                                        found: t, // used
                                    });
                                }
                            }
                        }
                        Ok(Type::Tensor(Box::new(Type::F32), 0))
                    }
                    ("VarBuilder", "grad") => {
                        if args.len() != 1 {
                            return Err(SemanticError::ArgumentCountMismatch {
                                name: "VarBuilder::grad".into(),
                                expected: 1,
                                found: args.len(),
                            });
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        Ok(Type::Tensor(Box::new(Type::F32), 0))
                    }
                    _ => {
                        // Try as a module function: type_name::method_name might be a qualified function call
                        let full_name = format!("{}::{}", type_name, method_name);
                        if let Some(func) = self.functions.get(&full_name).cloned() {
                            // Check arguments
                            if args.len() != func.args.len() && !func.args.is_empty() {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: full_name,
                                    expected: func.args.len(),
                                    found: args.len(),
                                });
                            }
                            for (i, arg) in args.iter_mut().enumerate() {
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
                            Ok(func.return_type.clone())
                        } else {
                            Err(SemanticError::FunctionNotFound(full_name))
                        }
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
                Err(SemanticError::StructNotFound(name))
            }
            Expr::MethodCall(obj, method_name, args) => {
                let obj_type = self.check_expr(obj)?;
                let type_name = match &obj_type {
                    Type::UserDefined(name) => name.clone(),
                    Type::Struct(name) => name.clone(),
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
                        if method_name == "slice" {
                            if args.len() != 2 {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 2,
                                    found: args.len(),
                                });
                            }
                            return Ok(obj_type.clone());
                        }
                        if method_name == "contiguous" {
                            // contiguous() returns the same tensor type
                            return Ok(obj_type.clone());
                        }
                        if method_name == "matmul" {
                            if args.len() != 1 {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 1,
                                    found: args.len(),
                                });
                            }
                            return Ok(obj_type.clone()); // Returns Tensor
                        }
                        if method_name == "reshape" {
                            if args.len() != 1 {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 1,
                                    found: args.len(),
                                });
                            }
                            match obj_type {
                                Type::Tensor(inner, _) => {
                                    // Inference Logic: Inspect args[0] AST if it's a literal
                                    let new_rank = if let Expr::TensorLiteral(elements) = &args[0] {
                                        elements.len()
                                    } else if let Expr::TensorConstLiteral(elements) = &args[0] {
                                        elements.len()
                                    } else {
                                        0 // Unknown rank
                                    };
                                    return Ok(Type::Tensor(inner.clone(), new_rank));
                                }
                                _ => return Ok(obj_type.clone()),
                            }
                        }
                        if method_name == "relu"
                            || method_name == "gelu"
                            || method_name == "sin"
                            || method_name == "cos"
                            || method_name == "sigmoid"
                            || method_name == "tanh"
                        {
                            if !args.is_empty() {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 0,
                                    found: args.len(),
                                });
                            }
                            return Ok(obj_type.clone());
                        }
                        if method_name == "softmax" || method_name == "log_softmax" {
                            if args.len() != 1 {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 1,
                                    found: args.len(),
                                });
                            }
                            return Ok(obj_type.clone());
                        }
                        if method_name == "item" {
                            if !args.is_empty() {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 0,
                                    found: args.len(),
                                });
                            }
                            return Ok(Type::F32); // item() returns scalar float usually
                        }
                        if method_name == "item_i64" {
                            if !args.is_empty() {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 0,
                                    found: args.len(),
                                });
                            }
                            return Ok(Type::I64);
                        }
                        if method_name == "to_i64" {
                            if !args.is_empty() {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 0,
                                    found: args.len(),
                                });
                            }
                            match obj_type {
                                Type::Tensor(_, rank) => {
                                    return Ok(Type::Tensor(Box::new(Type::I64), rank))
                                }
                                _ => {
                                    return Err(SemanticError::TypeMismatch {
                                        expected: Type::Tensor(Box::new(Type::Void), 0),
                                        found: obj_type.clone(),
                                    })
                                }
                            }
                        }
                        if method_name == "transpose" {
                            if args.len() != 2 {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 2,
                                    found: args.len(),
                                });
                            }
                            // transpose preserves rank and type
                            return Ok(obj_type.clone());
                        }
                        if method_name == "argmax" {
                            if args.len() != 2 {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: 2,
                                    found: args.len(),
                                });
                            }
                            // Returns tensor of indices (now converted to F32 in runtime)
                            match obj_type {
                                Type::Tensor(inner, _) => {
                                    return Ok(Type::Tensor(inner.clone(), 0))
                                } // Rank unknown/flexible, reuse inner type or F32
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

                if let Some(mut func) = method_def {
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

                    let mut func_arg_iter = func.args.iter_mut();
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
                    for (arg_expr, (_, expected_type)) in args.iter_mut().zip(func_arg_iter) {
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
                            self.check_expr(&mut args[0])?;
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
                            self.check_expr(&mut args[0])?;
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
                        // --- New Static Methods for Refactor ---
                        ("Tensor", "randn") => {
                            // Tensor::randn(shape, requires_grad)

                            // Check args (flexible shape handling)
                            if args.len() != 2 {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: "Tensor::randn".into(),
                                    expected: 2,
                                    found: args.len(),
                                });
                            }
                            // Arg 0: shape
                            let _t0 = self.check_expr(&mut args[0])?;
                            // Arg 1: requires_grad
                            let t1 = self.check_expr(&mut args[1])?;

                            if !matches!(t1, Type::Bool) {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::Bool,
                                    found: t1,
                                });
                            }
                            // Return Tensor<f32>
                            Ok(Type::Tensor(Box::new(Type::F32), 0))
                        }
                        ("VarBuilder", "get") => {
                            if args.len() < 2 {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: "VarBuilder::get".into(),
                                    expected: 2,
                                    found: args.len(),
                                });
                            }
                            let t0 = self.check_expr(&mut args[0])?;
                            if !matches!(t0, Type::UserDefined(ref s) if s == "String") {
                                return Err(SemanticError::TypeMismatch {
                                    expected: Type::UserDefined("String".into()),
                                    found: t0,
                                });
                            }
                            // Remaining args must be Ints (if varargs) OR a single Tensor/Array
                            if args.len() == 2 {
                                let _ = self.check_expr(&mut args[1])?;
                            } else {
                                for arg in &mut args[1..] {
                                    let t = self.check_expr(arg)?;
                                    if !matches!(t, Type::I64 | Type::I32) {
                                        return Err(SemanticError::TypeMismatch {
                                            expected: Type::I64,
                                            found: t,
                                        });
                                    }
                                }
                            }
                            Ok(Type::Tensor(Box::new(Type::F32), 0))
                        }
                        ("VarBuilder", "grad") => {
                            if args.len() != 1 {
                                return Err(SemanticError::ArgumentCountMismatch {
                                    name: "VarBuilder::grad".into(),
                                    expected: 1,
                                    found: args.len(),
                                });
                            }
                            let _ = self.check_expr(&mut args[0])?;
                            Ok(Type::Tensor(Box::new(Type::F32), 0))
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
        if matches!(t1, Type::Void) || matches!(t2, Type::Void) {
            return true;
        }
        if t1 == t2 {
            return true;
        }
        match (t1, t2) {
            (Type::Tensor(i1, r1), Type::Tensor(i2, r2)) => {
                // If either rank is 0, we treat it as dynamic/compatible rank
                let ranks_match = *r1 == 0 || *r2 == 0 || r1 == r2;
                ranks_match && self.are_types_compatible(i1, i2)
            }
            (Type::Tensor(inner, _rank), primitive)
                if self.are_types_compatible(inner, primitive) =>
            {
                true // Allow scalar to tensor promotion in some contexts
            }
            (primitive, Type::Tensor(inner, _rank))
                if self.are_types_compatible(primitive, inner) =>
            {
                true
            }
            (Type::Tuple(ts1), Type::Tuple(ts2)) => {
                if ts1.len() != ts2.len() {
                    return false;
                }
                for (t1, t2) in ts1.iter().zip(ts2.iter()) {
                    if !self.are_types_compatible(t1, t2) {
                        return false;
                    }
                }
                true
            }
            (Type::UserDefined(n1), Type::Struct(n2)) => n1 == n2,
            (Type::Struct(n1), Type::UserDefined(n2)) => n1 == n2,
            (Type::UserDefined(n1), Type::Enum(n2)) => n1 == n2,
            (Type::Enum(n1), Type::UserDefined(n2)) => n1 == n2,
            (Type::Enum(n1), Type::Enum(n2)) => n1 == n2,
            (Type::UserDefined(n1), Type::UserDefined(n2)) => {
                if n1 == n2 {
                    return true;
                }
                // Partial match for module imports: "Linear" vs "mod::Linear"
                if n1.ends_with(&format!("::{}", n2)) || n2.ends_with(&format!("::{}", n1)) {
                    return true;
                }
                false
            }

            // Promotions
            (Type::F64, Type::F32) => true,
            (Type::I64, Type::I32) => true,
            (Type::F32, Type::I64) => true, // Allow int to float promotion
            (Type::F64, Type::I64) => true,
            _ => false,
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
            Expr::As(expr, _) => {
                self.collect_indices(expr, indices);
            }
            Expr::IfExpr(cond, _then_block, _else_block) => {
                self.collect_indices(cond, indices);
            }
            Expr::Block(_) => {}
            Expr::TensorComprehension { .. } => {} // New scope, indices are bound? Or recurse body?
            // Comprehension indices are bound, but body free vars are free.
            // But usually we don't nest equations like this.
            // For now skip or simple recurse body?
            // Skip for safety to avoid over-capturing.
            _ => {}
        }
    }
    // Helper to infer free indices (used in Stmt::Let for Tensor Equation)
    fn infer_free_indices(&self, expr: &Expr) -> Vec<String> {
        let mut indices = HashSet::new();
        self.collect_indices(expr, &mut indices);

        let mut free_indices: Vec<String> = indices
            .into_iter()
            .filter(|idx| {
                // If it exists in scope, it is NOT free
                self.lookup_variable(idx).is_err()
            })
            .collect();
        free_indices.sort();
        free_indices
    }
}
