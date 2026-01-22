// src/compiler/semantics.rs
use crate::compiler::ast::*;
use crate::compiler::error::{SemanticErrorKind, Span, TlError};
use crate::compiler::type_registry::{TypeRegistry, ParamType, ReturnType, MethodSignature};
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
    #[error("Cannot assign to immutable variable: {0}")]
    AssignToImmutable(String),
    #[error("break outside of loop")]
    BreakOutsideLoop,
    #[error("continue outside of loop")]
    ContinueOutsideLoop,
}

impl SemanticError {
    /// SemanticErrorをTlErrorに変換（Span情報付き）
    pub fn to_tl_error(self, span: Option<Span>) -> TlError {
        let kind = match self {
            SemanticError::VariableNotFound(name) => SemanticErrorKind::VariableNotFound(name),
            SemanticError::VariableMoved(name) => SemanticErrorKind::VariableMoved(name),
            SemanticError::TypeMismatch { expected, found } => SemanticErrorKind::TypeMismatch {
                expected: format!("{:?}", expected),
                found: format!("{:?}", found),
            },
            SemanticError::FunctionNotFound(name) => SemanticErrorKind::FunctionNotFound(name),
            SemanticError::StructNotFound(name) => SemanticErrorKind::StructNotFound(name),
            SemanticError::DuplicateDefinition(name) => {
                SemanticErrorKind::DuplicateDefinition(name)
            }
            SemanticError::DuplicateMatchArm(name) => SemanticErrorKind::DuplicateMatchArm(name),
            SemanticError::UnreachableMatchArm => SemanticErrorKind::UnreachableMatchArm,
            SemanticError::NonExhaustiveMatch {
                enum_name,
                missing_variants,
            } => SemanticErrorKind::NonExhaustiveMatch {
                enum_name,
                missing_variants,
            },
            SemanticError::ArgumentCountMismatch {
                name,
                expected,
                found,
            } => SemanticErrorKind::ArgumentCountMismatch {
                name,
                expected,
                found,
            },
            SemanticError::MethodNotFound {
                type_name,
                method_name,
            } => SemanticErrorKind::MethodNotFound {
                type_name,
                method_name,
            },
            SemanticError::UnknownFunction(name) => SemanticErrorKind::UnknownFunction(name),
            SemanticError::TupleIndexOutOfBounds(idx, size) => {
                SemanticErrorKind::TupleIndexOutOfBounds(idx, size)
            }
            SemanticError::NotATuple(ty) => SemanticErrorKind::NotATuple(format!("{:?}", ty)),
            SemanticError::AssignToImmutable(name) => SemanticErrorKind::AssignToImmutable(name),
            SemanticError::BreakOutsideLoop => SemanticErrorKind::BreakOutsideLoop,
            SemanticError::ContinueOutsideLoop => SemanticErrorKind::ContinueOutsideLoop,
        };
        TlError::Semantic { kind, span }
    }
}

impl From<SemanticError> for TlError {
    fn from(err: SemanticError) -> Self {
        err.to_tl_error(None)
    }
}

#[derive(Clone, Debug)]
struct Symbol {
    #[allow(dead_code)]
    name: String,
    ty: Type,
    is_moved: bool,   // Track if variable has been moved
    is_mutable: bool, // Track if variable is mutable (let mut)
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

    fn insert(&mut self, name: String, ty: Type, is_mutable: bool) {
        self.symbols.insert(
            name.clone(),
            Symbol {
                name,
                ty,
                is_moved: false,
                is_mutable,
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
    relations: HashMap<String, RelationDecl>,               // Logic relations
    current_return_type: Option<Type>, // Expected return type for current function
    current_module: String,            // Current module prefix (e.g. "a::b")
    loop_depth: usize,                 // Track nesting level of loops for break/continue
    type_registry: TypeRegistry,       // Centralized type and method signature registry
}

impl SemanticAnalyzer {
    pub fn new(_source: String) -> Self {
        let mut analyzer = SemanticAnalyzer {
            scopes: vec![Scope::new()], // Global scope
            functions: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            methods: HashMap::new(),
            relations: HashMap::new(),
            current_return_type: None,
            current_module: String::new(),
            loop_depth: 0,
            type_registry: TypeRegistry::new(),
        };
        analyzer.declare_builtins();
        analyzer
    }

    fn err<T>(&self, error: SemanticError, span: Option<Span>) -> Result<T, TlError> {
        Err(error.to_tl_error(span))
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

    pub fn declare_variable(
        &mut self,
        name: String,
        ty: Type,
        is_mutable: bool,
    ) -> Result<(), TlError> {
        let is_global = self.scopes.len() == 1; // Immutable borrow
                                                // Shadowing is allowed
        if let Some(scope) = self.scopes.last_mut() {
            // Mutable borrow
            let final_name = if is_global && !self.current_module.is_empty() {
                format!("{}::{}", self.current_module, name)
            } else {
                name
            };
            scope.insert(final_name, ty, is_mutable);
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

    /// Check if a variable is mutable
    fn is_variable_mutable(&self, name: &str) -> Result<bool, SemanticError> {
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.get(name) {
                return Ok(symbol.is_mutable);
            }
        }
        Err(SemanticError::VariableNotFound(name.to_string()))
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

                    self.declare_variable(var_name.clone(), field_type.clone(), true)
                        .unwrap();
                }

                Ok(Some(variant_idx))
            }
        }
    }

    pub fn check_module(&mut self, module: &mut Module) -> Result<(), TlError> {
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
    ) -> Result<(), TlError> {
        // Register structs
        for s in &module.structs {
            let full_name = if prefix.is_empty() {
                s.name.clone()
            } else {
                format!("{}::{}", prefix, s.name)
            };
            if self.structs.contains_key(&full_name) {
                return self.err(SemanticError::DuplicateDefinition(full_name), None);
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
                return self.err(SemanticError::DuplicateDefinition(full_name), None);
            }
            let mut e_clone = e.clone();
            e_clone.name = full_name.clone();
            self.enums.insert(full_name, e_clone);
        }

        // Register relations
        // println!("DEBUG: Module relations count: {}", module.relations.len());
        for r in &module.relations {
            // println!("DEBUG: Registering relation: {}", r.name);
            let full_name = if prefix.is_empty() {
                r.name.clone()
            } else {
                format!("{}::{}", prefix, r.name)
            };
            if self.relations.contains_key(&full_name) {
                return self.err(SemanticError::DuplicateDefinition(full_name), None);
            }
            let mut r_clone = r.clone();
            r_clone.name = full_name.clone();
            self.relations.insert(full_name, r_clone);
        }

        // Register functions
        for f in &module.functions {
            let full_name = if prefix.is_empty() {
                f.name.clone()
            } else {
                format!("{}::{}", prefix, f.name)
            };
            if self.functions.contains_key(&full_name) {
                return self.err(SemanticError::DuplicateDefinition(full_name), None);
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

    fn check_module_bodies(&mut self, module: &mut Module, prefix: &str) -> Result<(), TlError> {
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

    fn check_impl_block(&mut self, impl_block: &mut ImplBlock) -> Result<(), TlError> {
        // Resolve target struct name
        let resolved_name = self.resolve_symbol_name(&impl_block.target_type);
        if resolved_name != impl_block.target_type {
            impl_block.target_type = resolved_name.clone();
        }

        // Check if target struct exists
        if !self.structs.contains_key(&impl_block.target_type) {
            return self.err(
                SemanticError::StructNotFound(impl_block.target_type.clone()),
                None,
            );
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
                    return self.err(
                        SemanticError::DuplicateDefinition(format!(
                            "{}::{}",
                            impl_block.target_type, method.name
                        )),
                        None,
                    );
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
    ) -> Result<(), TlError> {
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
            self.declare_variable(name.clone(), actual_ty, true)?;
        }

        for stmt in &mut func.body {
            self.check_stmt(stmt)?;
        }

        // Clear return type context
        self.current_return_type = None;

        self.exit_scope();
        Ok(())
    }

    fn check_block_stmts(&mut self, stmts: &mut Vec<Stmt>) -> Result<Type, TlError> {
        let mut ret_type = Type::Void;
        let stmts_len = stmts.len();
        for (i, stmt) in stmts.iter_mut().enumerate() {
            if i == stmts_len - 1 {
                if let StmtKind::Expr(e) = &mut stmt.inner {
                    ret_type = self.check_expr(e)?;
                } else if let StmtKind::If {
                    cond,
                    then_block,
                    else_block,
                } = &mut stmt.inner
                {
                    // Check Cond
                    let cond_type = self.check_expr(cond)?;
                    if cond_type != Type::Bool {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::Bool,
                                found: cond_type,
                            },
                            Some(cond.span.clone()),
                        );
                    }

                    self.enter_scope();
                    let then_type = self.check_block_stmts(then_block)?;
                    self.exit_scope();

                    let else_type = if let Some(block) = else_block {
                        self.enter_scope();
                        let t = self.check_block_stmts(block)?;
                        self.exit_scope();
                        t
                    } else {
                        Type::Void
                    };

                    // If types differ, usually logic is Void unless they match?
                    // Codegen IfExpr matches types.
                    // If one is Void and other is not, result is Void (in original IfExpr logic).
                    // But if we want to return value, they MUST match.
                    // Or if else is missing -> Void.
                    // If else is Void -> Void.
                    // So if then_type != else_type, return Error?
                    // Unless one is implicit void?
                    if then_type == else_type {
                        ret_type = then_type;
                    } else {
                        // If mismatch, should error?
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: then_type,
                                found: else_type,
                            },
                            Some(stmt.span.clone()),
                        );
                    }
                } else {
                    self.check_stmt(stmt)?;
                    ret_type = Type::Void;
                }
            } else {
                self.check_stmt(stmt)?;
            }
        }
        Ok(ret_type)
    }

    pub fn check_stmt(&mut self, stmt: &mut Stmt) -> Result<(), TlError> {
        match &mut stmt.inner {
            StmtKind::FieldAssign {
                obj,
                field,
                op,
                value,
            } => {
                // Check object type and verify it's a struct
                let obj_type = self.check_expr(obj)?;
                let struct_name = match obj_type {
                    Type::UserDefined(name) => name,
                    Type::Struct(name) => name,
                    _ => {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::UserDefined("Struct".into()),
                                found: obj_type,
                            },
                            Some(stmt.span.clone()),
                        );
                    }
                };

                // Verify struct and field exist
                // Need to resolve struct_name? Usually FieldAccess object type is already resolved by check_expr logic?
                // Type::UserDefined("Name") -> Name should be fully qualified if check_expr(obj) did its job.
                // But check_expr returns Type. If Type comes from AST, it might simple name.
                // The Type returned by check_expr comes from:
                // - ExprKind::StructInit -> looked up strict name (resolved).
                // - ExprKind::Variable -> lookup_variable -> returns Type from scope.
                // - ExprKind::FnCall -> returns return_type of function.
                // So if "Struct" type is stored in scope/function def with FQN, then obj_type has FQN.
                let struct_def = self
                    .structs
                    .get(&struct_name)
                    .ok_or_else(|| SemanticError::StructNotFound(struct_name.clone()))
                    .map_err(|e| e.to_tl_error(Some(stmt.span.clone())))?;

                let field_type = struct_def
                    .fields
                    .iter()
                    .find(|(name, _)| name == field)
                    .map(|(_, ty)| ty.clone())
                    .ok_or_else(|| SemanticError::VariableNotFound(format!("Field {}", field)))
                    .map_err(|e| e.to_tl_error(Some(stmt.span.clone())))?;

                // Check value type matches field type
                let value_type = self.check_expr(value)?;

                match op {
                    AssignOp::Assign => {
                        if !self.are_types_compatible(&field_type, &value_type) {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: field_type,
                                    found: value_type,
                                },
                                Some(stmt.span.clone()),
                            );
                        }
                    }
                    _ => {
                        // For compound assignments, allow Tensor OR Numeric types
                        let is_numeric =
                            matches!(field_type, Type::I64 | Type::I32 | Type::F32 | Type::F64);
                        let is_tensor = matches!(field_type, Type::Tensor(_, _));

                        if is_tensor {
                            // Tensor compound assignment logic
                            let is_compat = match (&field_type, &value_type) {
                                (Type::Tensor(inner, _), val) if **inner == *val => true, // Tensor += scalar
                                (Type::Tensor(_, _), Type::Tensor(_, _)) => true,         // Tensor += Tensor
                                _ => false,
                            };
                            if !is_compat {
                                return self.err(
                                    SemanticError::TypeMismatch {
                                        expected: field_type,
                                        found: value_type,
                                    },
                                    Some(stmt.span.clone()),
                                );
                            }
                        } else if is_numeric {
                            // Numeric compound assignment
                            if !self.are_types_compatible(&field_type, &value_type) {
                                return self.err(
                                    SemanticError::TypeMismatch {
                                        expected: field_type,
                                        found: value_type,
                                    },
                                    Some(stmt.span.clone()),
                                );
                            }
                        } else {
                            return self.err(
                                SemanticError::MethodNotFound {
                                    type_name: format!("{:?}", field_type),
                                    method_name: format!("{:?}", op),
                                },
                                Some(stmt.span.clone()),
                            );
                        }
                    }
                }

                Ok(())
            }
            StmtKind::TensorDecl {
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
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: type_annotation.clone(),
                                found: init_ty,
                            },
                            Some(stmt.span.clone()),
                        );
                    }
                }
                self.declare_variable(name.clone(), final_ty, true)?;
                Ok(())
            }

            StmtKind::Let {
                name,
                type_annotation,
                value,
                mutable,
            } => {
                // 1. Infer free indices (Tensor Equation Mode)
                // self.infer_free_indices takes &Expr.
                // But value is &mut Expr. Can treat as &Expr.
                let free_indices = self.infer_free_indices(value);

                let inferred_type = if !free_indices.is_empty() {
                    // Tensor Equation Logic
                    self.enter_scope();

                    // Declare implicitly inferred indices (always mutable for loop vars)
                    for idx in &free_indices {
                        self.declare_variable(idx.clone(), Type::I64, true)?;
                    }

                    // Now check RHS with these indices valid
                    let rhs_type = self.check_expr(value)?;

                    self.exit_scope();

                    // Construct Tensor Type
                    if let ExprKind::TensorComprehension { .. } = &value.inner {
                        rhs_type
                    } else {
                        Type::Tensor(Box::new(rhs_type), free_indices.len())
                    }
                } else {
                    self.check_expr(value)?
                };

                let final_type = if let Some(ann) = type_annotation {
                    if !self.are_types_compatible(ann, &inferred_type) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: ann.clone(),
                                found: inferred_type,
                            },
                            Some(stmt.span.clone()),
                        );
                    }
                    ann.clone()
                } else {
                    inferred_type
                };

                self.declare_variable(name.clone(), final_type.clone(), *mutable)?;

                // Move semantics: If RHS is a variable of moveable type, mark it as moved
                if let ExprKind::Variable(source_var) = &value.inner {
                    if self.is_moveable_type(&final_type) {
                        self.mark_moved(source_var);
                    }
                }

                Ok(())
            }
            StmtKind::Assign {
                name,
                indices,
                op,
                value,
            } => {
                // Check if variable is mutable
                let is_mutable = self.is_variable_mutable(name)?;
                if !is_mutable {
                    return self.err(
                        SemanticError::AssignToImmutable(name.clone()),
                        Some(stmt.span.clone()),
                    );
                }

                let var_type = self.lookup_variable(name)?;

                if let Some(idxs) = indices {
                    // Indexed assignment: C[i, k] = ...
                    // Verify var_type is Tensor
                    let (_inner_type, rank) = match &var_type {
                        Type::Tensor(inner, r) => (inner, *r),
                        _ => {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: Type::Tensor(Box::new(Type::Void), 0),
                                    found: var_type,
                                },
                                Some(stmt.span.clone()),
                            )
                        }
                    };

                    if rank != 0 && idxs.len() != rank {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: rank,
                                found: idxs.len(),
                            },
                            Some(stmt.span.clone()),
                        );
                    }

                    // Check each index expression is integer
                    for idx_expr in idxs {
                        let idx_type = self.check_expr(idx_expr)?;
                        if !matches!(idx_type, Type::I64 | Type::I32) {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: Type::I64,
                                    found: idx_type,
                                },
                                Some(stmt.span.clone()),
                            );
                        }
                    }
                } else {
                    // Standard assignment
                    let val_type = self.check_expr(value)?;

                    match op {
                        AssignOp::Assign => {
                            if !self.are_types_compatible(&var_type, &val_type) {
                                return self.err(
                                    SemanticError::TypeMismatch {
                                        expected: var_type,
                                        found: val_type,
                                    },
                                    Some(stmt.span.clone()),
                                );
                            }
                        }
                        _ => {
                            // Compound assignment: Check if numeric or tensor
                            let is_numeric =
                                matches!(var_type, Type::I64 | Type::I32 | Type::F32 | Type::F64);
                            let is_tensor = matches!(var_type, Type::Tensor(_, _));

                            if is_tensor {
                                let is_compat = match (&var_type, &val_type) {
                                    (Type::Tensor(inner, _), val) if **inner == *val => true,
                                    (Type::Tensor(_, _), Type::Tensor(_, _)) => true,
                                    _ => false,
                                };
                                if !is_compat {
                                    return self.err(
                                        SemanticError::TypeMismatch {
                                            expected: var_type.clone(),
                                            found: val_type.clone(),
                                        },
                                        Some(stmt.span.clone()),
                                    );
                                }
                            } else if is_numeric {
                                if !self.are_types_compatible(&var_type, &val_type) {
                                    return self.err(
                                        SemanticError::TypeMismatch {
                                            expected: var_type.clone(),
                                            found: val_type.clone(),
                                        },
                                        Some(stmt.span.clone()),
                                    );
                                }
                            } else {
                                let method_name = match op {
                                    AssignOp::AddAssign => "add_assign",
                                    AssignOp::SubAssign => "sub_assign",
                                    AssignOp::MulAssign => "mul_assign",
                                    AssignOp::DivAssign => "div_assign",
                                    AssignOp::ModAssign => "mod_assign",
                                    _ => "unknown_assign",
                                };
                                return self.err(
                                    SemanticError::MethodNotFound {
                                        type_name: format!("{:?}", var_type),
                                        method_name: method_name.to_string(),
                                    },
                                    Some(stmt.span.clone()),
                                );
                            }
                        }
                    }
                }
                Ok(())
            }
            StmtKind::Return(expr_opt) => {
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
                        }
                        .to_tl_error(Some(stmt.span.clone())));
                    }
                }
                Ok(())
            }
            StmtKind::Expr(expr) => {
                self.check_expr(expr)?;
                Ok(())
            }
            StmtKind::If {
                cond,
                then_block,
                else_block,
            } => {
                let cond_type = self.check_expr(cond)?;
                if cond_type != Type::Bool {
                    return self.err(
                        SemanticError::TypeMismatch {
                            expected: Type::Bool,
                            found: cond_type,
                        },
                        Some(stmt.span.clone()),
                    );
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
            StmtKind::For {
                loop_var,
                iterator,
                body,
            } => {
                // Check if iterator is range(start, end)
                // This is a special case for the compiler intrinsic 'range'
                let elem_type = match &mut iterator.inner {
                    ExprKind::Range(start, end) => {
                        let start_ty = self.check_expr(start.as_mut())?;
                        let end_ty = self.check_expr(end.as_mut())?;
                        if !matches!(start_ty, Type::I64 | Type::I32)
                            || !matches!(end_ty, Type::I64 | Type::I32)
                        {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: Type::I64,
                                    found: start_ty,
                                },
                                Some(stmt.span.clone()),
                            );
                        }
                        Type::I64
                    }
                    ExprKind::FnCall(name, args) if name == "range" && args.len() == 2 => {
                        // Deprecated range() function check
                        let start_type = self.check_expr(&mut args[0])?;
                        let end_type = self.check_expr(&mut args[1])?;
                        if !matches!(start_type, Type::I64 | Type::I32)
                            || !matches!(end_type, Type::I64 | Type::I32)
                        {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: Type::I64,
                                    found: start_type,
                                },
                                Some(stmt.span.clone()),
                            );
                        }
                        Type::I64
                    }
                    _ => {
                        let iter_type = self.check_expr(iterator)?;
                        match iter_type {
                            Type::Tensor(t, 1) => *t,
                            Type::TensorShaped(t, _) => *t,
                            _ => {
                                return self.err(
                                    SemanticError::TypeMismatch {
                                        expected: Type::Tensor(Box::new(Type::F32), 1),
                                        found: iter_type,
                                    },
                                    Some(stmt.span.clone()),
                                );
                            }
                        }
                    }
                };

                self.loop_depth += 1;
                self.enter_scope();
                self.declare_variable(loop_var.clone(), elem_type, true)?;
                for s in body {
                    self.check_stmt(s)?;
                }
                self.exit_scope();
                self.loop_depth -= 1;
                Ok(())
            }
            StmtKind::While { cond, body } => {
                let cond_type = self.check_expr(cond)?;
                if cond_type != Type::Bool {
                    return self.err(
                        SemanticError::TypeMismatch {
                            expected: Type::Bool,
                            found: cond_type,
                        },
                        Some(stmt.span.clone()),
                    );
                }
                self.loop_depth += 1;
                self.enter_scope();
                for stmt in body {
                    self.check_stmt(stmt)?;
                }
                self.exit_scope();
                self.loop_depth -= 1;
                Ok(())
            }
            StmtKind::Loop { body } => {
                self.loop_depth += 1;
                self.enter_scope();
                for stmt in body {
                    self.check_stmt(stmt)?;
                }
                self.exit_scope();
                self.loop_depth -= 1;
                Ok(())
            }
            StmtKind::Use { path, alias, items } => {
                let full_prefix = path.join("::");

                if !items.is_empty() {
                    // use path::{items...}
                    for item in items {
                        if item == "*" {
                            // Glob import: import all relations starting with full_prefix::
                            // Collect keys first to avoid borrow issues
                            let matching_relations: Vec<String> = self
                                .relations
                                .keys()
                                .filter(|k| k.starts_with(&format!("{}::", full_prefix)))
                                .cloned()
                                .collect();

                            for rel_full_name in matching_relations {
                                // Extract simple name: e.g. facts::father -> father
                                if let Some(simple_name) =
                                    rel_full_name.strip_prefix(&format!("{}::", full_prefix))
                                {
                                    self.scopes
                                        .last_mut()
                                        .unwrap()
                                        .add_alias(simple_name.to_string(), rel_full_name);
                                }
                            }
                        } else {
                            // import path::item as item
                            let full_name = format!("{}::{}", full_prefix, item);
                            let alias_name = item.clone();
                            self.scopes
                                .last_mut()
                                .unwrap()
                                .add_alias(alias_name, full_name);
                        }
                    }
                } else {
                    // use path [as alias]
                    let alias_name = if let Some(a) = alias {
                        a.clone()
                    } else {
                        path.last()
                            .ok_or(SemanticError::VariableNotFound("Empty use path".into()))
                            .map_err(|e| e.to_tl_error(Some(stmt.span.clone())))?
                            .clone()
                    };
                    self.scopes
                        .last_mut()
                        .unwrap()
                        .add_alias(alias_name, full_prefix);
                }
                Ok(())
            }
            StmtKind::Break => {
                if self.loop_depth == 0 {
                    return self.err(SemanticError::BreakOutsideLoop, Some(stmt.span.clone()));
                }
                Ok(())
            }
            StmtKind::Continue => {
                if self.loop_depth == 0 {
                    return self.err(SemanticError::ContinueOutsideLoop, Some(stmt.span.clone()));
                }
                Ok(())
            }
        }
    }

    /// Check method call arguments against a signature from the type registry.
    /// Returns the inferred return type if successful.
    fn check_method_call_with_registry(
        &mut self,
        receiver_type: &Type,
        method_name: &str,
        args: &mut [Expr],
        span: &Span,
    ) -> Result<Option<Type>, TlError> {
        let type_key = TypeRegistry::type_to_key(receiver_type);
        
        if let Some(sig) = self.type_registry.get_method(&type_key, method_name) {
            // Clone the signature to avoid borrow issues
            let sig = sig.clone();
            
            // Check argument count
            if !sig.is_varargs && args.len() != sig.params.len() {
                return self.err(
                    SemanticError::ArgumentCountMismatch {
                        name: method_name.to_string(),
                        expected: sig.params.len(),
                        found: args.len(),
                    },
                    Some(span.clone()),
                );
            }
            if sig.is_varargs && args.len() < sig.min_args {
                return self.err(
                    SemanticError::ArgumentCountMismatch {
                        name: method_name.to_string(),
                        expected: sig.min_args,
                        found: args.len(),
                    },
                    Some(span.clone()),
                );
            }

            // Check each argument type
            for (i, arg) in args.iter_mut().enumerate() {
                let arg_type = self.check_expr(arg)?;
                
                let expected_param = if i < sig.params.len() {
                    &sig.params[i]
                } else if sig.is_varargs && !sig.params.is_empty() {
                    // For varargs, use the last param type for remaining args
                    sig.params.last().unwrap()
                } else {
                    continue;
                };

                if !TypeRegistry::matches_param_type(&arg_type, expected_param, receiver_type) {
                    let expected_desc = match expected_param {
                        ParamType::Exact(ty) => ty.clone(),
                        ParamType::AnyTensor => Type::Tensor(Box::new(Type::Void), 0),
                        ParamType::ShapeArray => Type::ScalarArray(Box::new(Type::I64), 0),
                        ParamType::AnyInt => Type::I64,
                        ParamType::AnyNumeric => Type::F32,
                        ParamType::Bool => Type::Bool,
                        ParamType::SameAsReceiver => receiver_type.clone(),
                        ParamType::TensorOf(inner) => Type::Tensor(inner.clone(), 0),
                        ParamType::AnyTensorOrNumeric => Type::Tensor(Box::new(Type::F32), 0),
                    };
                    return self.err(
                        SemanticError::TypeMismatch {
                            expected: expected_desc,
                            found: arg_type,
                        },
                        Some(arg.span.clone()),
                    );
                }
            }

            // Infer return type
            let return_type = self.infer_return_type_from_sig(receiver_type, &sig, args);
            return Ok(Some(return_type));
        }

        // Not found in registry
        Ok(None)
    }

    /// Infer return type based on the signature's ReturnType specification
    fn infer_return_type_from_sig(
        &self,
        receiver_type: &Type,
        sig: &MethodSignature,
        args: &[Expr],
    ) -> Type {
        match &sig.return_type {
            ReturnType::Exact(ty) => ty.clone(),
            ReturnType::SameAsReceiver => receiver_type.clone(),
            ReturnType::TensorSameElementType(rank) => {
                if let Type::Tensor(inner, _) = receiver_type {
                    Type::Tensor(inner.clone(), *rank)
                } else {
                    Type::Tensor(Box::new(Type::F32), *rank)
                }
            }
            ReturnType::TensorDynamicRank => {
                if let Type::Tensor(inner, _) = receiver_type {
                    Type::Tensor(inner.clone(), 0)
                } else {
                    Type::Tensor(Box::new(Type::F32), 0)
                }
            }
            ReturnType::InferFromShapeArg => {
                // Try to infer rank from shape argument
                let new_rank = if !args.is_empty() {
                    if let ExprKind::TensorLiteral(elements) = &args[0].inner {
                        elements.len()
                    } else if let ExprKind::TensorConstLiteral(elements) = &args[0].inner {
                        elements.len()
                    } else {
                        0
                    }
                } else {
                    0
                };
                if let Type::Tensor(inner, _) = receiver_type {
                    Type::Tensor(inner.clone(), new_rank)
                } else {
                    Type::Tensor(Box::new(Type::F32), new_rank)
                }
            }
            ReturnType::ExtractedScalar => {
                if let Type::Tensor(inner, _) = receiver_type {
                    *inner.clone()
                } else {
                    Type::F32
                }
            }
            ReturnType::TensorRankIncr => {
                if let Type::Tensor(inner, rank) = receiver_type {
                    Type::Tensor(inner.clone(), rank + 1)
                } else if let Type::TensorShaped(inner, shape) = receiver_type {
                    Type::Tensor(inner.clone(), shape.len() + 1)
                } else {
                    Type::Tensor(Box::new(Type::F32), 1)
                }
            }
            ReturnType::Void => Type::Void,
        }
    }

    pub fn check_expr(&mut self, expr: &mut Expr) -> Result<Type, TlError> {
        match &mut expr.inner {
            ExprKind::Int(_) => Ok(Type::I64), // Default integer literal type
            ExprKind::Float(_) => Ok(Type::F32), // Default float literal type
            ExprKind::Bool(_) => Ok(Type::Bool),
            ExprKind::StringLiteral(_) => Ok(Type::UserDefined("String".to_string())), // Placeholder
            ExprKind::Symbol(_) => Ok(Type::Entity),
            ExprKind::LogicVar(_) => Ok(Type::Entity),
            ExprKind::Tuple(exprs) => {
                let mut types = Vec::new();
                for e in exprs {
                    types.push(self.check_expr(e)?);
                }
                Ok(Type::Tuple(types))
            }
            ExprKind::TupleAccess(expr, idx) => {
                let ty = self.check_expr(expr)?;
                if let Type::Tuple(types) = ty {
                    if *idx < types.len() {
                        Ok(types[*idx].clone())
                    } else {
                        self.err(
                            SemanticError::TupleIndexOutOfBounds(*idx, types.len()),
                            Some(expr.span.clone()),
                        )
                    }
                } else {
                    self.err(SemanticError::NotATuple(ty), Some(expr.span.clone()))
                }
            }
            ExprKind::StructInit(name, fields) => {
                let resolved_name = self.resolve_symbol_name(name);
                if *name != resolved_name {
                    *name = resolved_name.clone();
                }

                if let Some(struct_def) = self.structs.get(name).cloned() {
                    let mut initialized_fields = HashSet::new();
                    for (field_name, field_expr) in fields {
                        if initialized_fields.contains(field_name) {
                            return self.err(
                                SemanticError::DuplicateDefinition(format!(
                                    "Field {} in struct init",
                                    field_name
                                )),
                                Some(expr.span.clone()),
                            );
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
                            })
                            .map_err(|e| e.to_tl_error(Some(expr.span.clone())))?;

                        let found_type = self.check_expr(field_expr)?;
                        if !self.are_types_compatible(expected_type, &found_type) {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: expected_type.clone(),
                                    found: found_type,
                                },
                                Some(field_expr.span.clone()),
                            );
                        }
                    }

                    // Check for missing fields
                    for (field_name, _) in &struct_def.fields {
                        if !initialized_fields.contains(field_name) {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: format!("Struct init {}", name),
                                    expected: struct_def.fields.len(),
                                    found: initialized_fields.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                    }

                    Ok(Type::Struct(name.clone()))
                } else if let Some((enum_def, variant_def)) = self.resolve_enum_variant(name) {
                    // It is an Enum Variant! Transform to EnumInit.
                    let fields_owned = std::mem::take(fields);
                    expr.inner = ExprKind::EnumInit {
                        enum_name: enum_def.name.clone(),
                        variant_name: variant_def.name.clone(),
                        fields: fields_owned,
                    };
                    // Re-check as EnumInit
                    self.check_expr(expr)
                } else {
                    self.err(
                        SemanticError::StructNotFound(name.clone()),
                        Some(expr.span.clone()),
                    )
                }
            }
            ExprKind::EnumInit {
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
                    })
                    .map_err(|e| e.to_tl_error(Some(expr.span.clone())))?;

                let mut initialized_fields = HashSet::new();
                for (field_name, field_expr) in fields {
                    if initialized_fields.contains(field_name) {
                        return self.err(
                            SemanticError::DuplicateDefinition(format!(
                                "Field {} in enum variant init",
                                field_name
                            )),
                            Some(expr.span.clone()),
                        );
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
                        })
                        .map_err(|e| e.to_tl_error(Some(expr.span.clone())))?;

                    let found_type = self.check_expr(field_expr)?;
                    if !self.are_types_compatible(expected_type, &found_type) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: expected_type.clone(),
                                found: found_type,
                            },
                            Some(field_expr.span.clone()),
                        );
                    }
                }

                // Check for missing fields
                for (field_name, _) in &variant_def.fields {
                    if !initialized_fields.contains(field_name) {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: format!("Variant init {}", variant_name),
                                expected: variant_def.fields.len(),
                                found: initialized_fields.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                }

                Ok(Type::Enum(enum_name.clone()))
            }
            ExprKind::Match {
                expr: subject_expr,
                arms,
            } => {
                let subject_type = self.check_expr(subject_expr)?;
                let enum_name = match &subject_type {
                    Type::Enum(n) | Type::UserDefined(n) => n.clone(),
                    _ => {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::UserDefined("Enum".into()),
                                found: subject_type,
                            },
                            Some(subject_expr.span.clone()),
                        );
                    }
                };

                let enum_def = self.enums.get(&enum_name);
                if enum_def.is_none() {
                    // If it's not an enum, maybe we support matching on other types later?
                    // For now only Enums (UserDefined that maps to Enum).
                    return self.err(
                        SemanticError::StructNotFound(format!("Enum {}", enum_name)),
                        Some(subject_expr.span.clone()),
                    );
                }
                let enum_def = enum_def.unwrap().clone();

                let mut return_type = Option::<Type>::None;
                let mut seen_variants = HashSet::new();
                let mut saw_wildcard = false;

                for (pattern, arm_expr) in arms {
                    if saw_wildcard {
                        return self.err(
                            SemanticError::UnreachableMatchArm,
                            Some(arm_expr.span.clone()),
                        );
                    }

                    self.enter_scope();

                    let variant_idx = self
                        .bind_enum_pattern(&enum_name, &enum_def, pattern)
                        .map_err(|e| e.to_tl_error(Some(arm_expr.span.clone())))?;

                    let arm_type = self.check_expr(arm_expr)?;
                    self.exit_scope();

                    if let Some(idx) = variant_idx {
                        let variant_name = &enum_def.variants[idx].name;
                        if !seen_variants.insert(idx) {
                            return self.err(
                                SemanticError::DuplicateMatchArm(variant_name.clone()),
                                Some(arm_expr.span.clone()),
                            );
                        }
                    } else {
                        saw_wildcard = true;
                    }

                    if let Some(ref rt) = return_type {
                        if !self.are_types_compatible(rt, &arm_type) {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: rt.clone(),
                                    found: arm_type,
                                },
                                Some(arm_expr.span.clone()),
                            );
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
                        return self.err(
                            SemanticError::NonExhaustiveMatch {
                                enum_name,
                                missing_variants: missing,
                            },
                            Some(expr.span.clone()),
                        );
                    }
                }

                Ok(return_type.unwrap_or(Type::Void))
            }
            ExprKind::IfLet {
                pattern,
                expr: subject_expr,
                then_block,
                else_block,
            } => {
                let subject_type = self.check_expr(subject_expr)?;
                let enum_name = match &subject_type {
                    Type::Enum(n) | Type::UserDefined(n) => n.clone(),
                    _ => {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::UserDefined("Enum".into()),
                                found: subject_type,
                            },
                            Some(subject_expr.span.clone()),
                        );
                    }
                };

                let enum_def = self.enums.get(&enum_name);
                if enum_def.is_none() {
                    return self.err(
                        SemanticError::StructNotFound(format!("Enum {}", enum_name)),
                        Some(subject_expr.span.clone()),
                    );
                }
                let enum_def = enum_def.unwrap().clone();

                // Then block with bindings
                self.enter_scope();
                self.bind_enum_pattern(&enum_name, &enum_def, pattern)
                    .map_err(|e| e.to_tl_error(Some(subject_expr.span.clone())))?;
                let mut then_type = Type::Void;
                let then_len = then_block.len();
                for (i, stmt) in then_block.iter_mut().enumerate() {
                    if i == then_len - 1 {
                        if let StmtKind::Expr(e) = &mut stmt.inner {
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
                            if let StmtKind::Expr(e) = &mut stmt.inner {
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
                    self.err(
                        SemanticError::TypeMismatch {
                            expected: then_type,
                            found: else_type,
                        },
                        Some(expr.span.clone()),
                    )
                }
            }
            ExprKind::Range(start, end) => {
                let s_ty = self.check_expr(start)?;
                let e_ty = self.check_expr(end)?;
                if !matches!(s_ty, Type::I64 | Type::I32) {
                    return self.err(
                        SemanticError::TypeMismatch {
                            expected: Type::I64,
                            found: s_ty,
                        },
                        Some(start.span.clone()),
                    );
                }
                if !matches!(e_ty, Type::I64 | Type::I32) {
                    return self.err(
                        SemanticError::TypeMismatch {
                            expected: Type::I64,
                            found: e_ty,
                        },
                        Some(end.span.clone()),
                    );
                }
                // Range expression itself doesn't evaluate to a runtime value outside of for-loops yet,
                // but we return Void or a placeholder.
                Ok(Type::Void)
            }
            ExprKind::Variable(name) => {
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
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::UserDefined("Unit Variant".into()),
                                found: Type::UserDefined("Struct Variant".into()),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    expr.inner = ExprKind::EnumInit {
                        enum_name: enum_def.name.clone(),
                        variant_name: variant_def.name.clone(),
                        fields: vec![],
                    };
                    return self.check_expr(expr);
                }

                self.err(
                    SemanticError::VariableNotFound(name.clone()),
                    Some(expr.span.clone()),
                )
            }
            ExprKind::BinOp(lhs, op, rhs) => {
                let left = self.check_expr(lhs.as_mut())?;
                let right = self.check_expr(rhs.as_mut())?;
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
                        _ => self.err(
                            SemanticError::TypeMismatch {
                                expected: left,
                                found: right,
                            },
                            Some(expr.span.clone()),
                        ),
                    }
                }
            }
            ExprKind::FnCall(name, args) => {
                // println!("DEBUG: check_expr FnCall name='{}'", name);
                // Resolve name first
                let resolved_name = self.resolve_symbol_name(name);
                if *name != resolved_name {
                    *name = resolved_name.clone();
                }

                if name == "print" || name == "println" {
                    if args.len() == 0 {
                        // println() with no args is valid (just newline)
                        if name == "println" {
                            return Ok(Type::Void);
                        }
                        // print() with no args is an error
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: 0,
                            },
                            Some(expr.span.clone()),
                        );
                    }

                    // Verify all arguments are valid expressions
                    for arg in args.iter_mut() {
                        self.check_expr(arg)?;
                    }

                    // Optional: Check if format string placeholders match arg count?
                    // We can do it here or leave it to codegen/runtime panic.
                    // Doing it here is better for dev experience.
                    if args.len() > 1 {
                        if let ExprKind::StringLiteral(s) = &args[0].inner {
                            if s.contains("{}") {
                                let parts: Vec<&str> = s.split("{}").collect();
                                let placeholder_count = parts.len() - 1;
                                let arg_count = args.len() - 1;
                                if arg_count != placeholder_count {
                                    return self.err(
                                        SemanticError::ArgumentCountMismatch {
                                            name: format!("{} (format placeholders)", name),
                                            expected: placeholder_count,
                                            found: arg_count,
                                        },
                                        Some(expr.span.clone()),
                                    );
                                }
                            }
                        }
                        // If not a format string, we could error?
                        // "print(a, b)" without format string is currently not supported by codegen logic
                        // (codegen expects fmt string if mult args).
                        // So we should enforce: if len > 1, arg[0] MUST be string?
                        // Except maybe we want print(a, b) to just print a then b?
                        // Current codegen implementation ONLY supports format string for >1 args.
                        // "Normal print" block: if args.len() != 1 -> Error.
                        // So semantics SHOULD enforce this.
                        if !matches!(&args[0].inner, ExprKind::StringLiteral(_)) {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: Type::UserDefined("StringLiteral".into()),
                                    found: Type::Void, // Hacky message?
                                },
                                Some(args[0].span.clone()),
                            );
                        }
                        // Actually, let's just let check_expr handle types,
                        // but we need to ensure arg[0] is StringLiteral if len > 1.
                        if !matches!(&args[0].inner, ExprKind::StringLiteral(_)) {
                            // We can't easily get the Type of arg[0] here without checking again,
                            // but we just checked all args.
                            // However, we rely on Expr structure.
                            // If arg[0] is Variable("s") which is a string, it's NOT a literal.
                            // Codegen requires Literal for compile-time formatting.
                            // So yes, strictly ExprKind::StringLiteral.
                        }
                    }

                    return Ok(Type::Void);
                }
                if name == "read_line" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: "read_line".into(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    self.check_expr(&mut args[0])?;
                    return Ok(Type::UserDefined("String".to_string()));
                }

                // --- StdLib Phase 1 ---
                // --- StdLib Static Methods ---
                // Transferred to ExprKind::StaticMethodCall handling.
                // The parser ensures that identifiers with "::" are parsed as StaticMethodCall.

                // --- StdLib FFI (Legacy/Direct) ---
                if name == "args_get" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let arg_ty = self.check_expr(&mut args[0])?;
                    if !matches!(arg_ty, Type::I64 | Type::I32) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: arg_ty,
                            },
                            Some(args[0].span.clone()),
                        );
                    }
                    return Ok(Type::UserDefined("String".to_string()));
                } else if name == "char_at" {
                    if args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let string_ty = self.check_expr(&mut args[0])?;
                    let index_ty = self.check_expr(&mut args[1])?;
                    if string_ty != Type::UserDefined("String".to_string()) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::UserDefined("String".to_string()),
                                found: string_ty,
                            },
                            Some(args[0].span.clone()),
                        );
                    }
                    if !matches!(index_ty, Type::I64 | Type::I32) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: index_ty,
                            },
                            Some(args[1].span.clone()),
                        );
                    }
                    return Ok(Type::UserDefined("String".to_string())); // Returns a single character as a String
                } else if name == "len" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let arg_ty = self.check_expr(&mut args[0])?;
                    if arg_ty != Type::UserDefined("String".to_string())
                        && !matches!(arg_ty, Type::Tensor(_, _))
                    {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::UserDefined("String".to_string()),
                                found: arg_ty,
                            },
                            Some(args[0].span.clone()),
                        );
                    }
                    return Ok(Type::I64);
                } else if name == "tl_file_open" {
                    if args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let _t0 = self.check_expr(&mut args[0])?;
                    let _t1 = self.check_expr(&mut args[1])?;
                    return Ok(Type::UserDefined("File".to_string()));
                }
                if name == "tl_file_read_string" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let _t0 = self.check_expr(&mut args[0])?;
                    return Ok(Type::UserDefined("String".to_string()));
                }
                if name == "tl_file_write_string" {
                    if args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    self.check_expr(&mut args[0])?;
                    self.check_expr(&mut args[1])?;
                    return Ok(Type::Void);
                }
                if name == "tl_file_close" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    self.check_expr(&mut args[0])?;
                    return Ok(Type::Void);
                }
                if name == "tl_env_get" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let _t0 = self.check_expr(&mut args[0])?;
                    return Ok(Type::UserDefined("String".to_string()));
                } else if name == "tl_args_count" {
                    if !args.is_empty() {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 0,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    return Ok(Type::I64);
                } else if name == "tl_args_get" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let arg_ty = self.check_expr(&mut args[0])?;
                    if !matches!(arg_ty, Type::I64 | Type::I32) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: arg_ty,
                            },
                            Some(args[0].span.clone()),
                        );
                    }
                    return Ok(Type::UserDefined("String".to_string()));
                } else if name == "tl_string_to_i64" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let arg_ty = self.check_expr(&mut args[0])?;
                    if arg_ty != Type::UserDefined("String".to_string()) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::UserDefined("String".to_string()),
                                found: arg_ty,
                            },
                            Some(args[0].span.clone()),
                        );
                    }
                    return Ok(Type::I64);
                } else if name == "tl_http_download" {
                    if args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    self.check_expr(&mut args[0])?;
                    self.check_expr(&mut args[1])?;
                    return Ok(Type::Bool);
                } else if name == "tl_vec_u8_len" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    self.check_expr(&mut args[0])?;
                    return Ok(Type::I64);
                } else if name == "tl_vec_u8_get" {
                    if args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    self.check_expr(&mut args[0])?;
                    self.check_expr(&mut args[1])?;
                    return Ok(Type::U8);
                } else if name == "tl_vec_u8_free" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    self.check_expr(&mut args[0])?;
                    return Ok(Type::Void);
                } else if name == "sin" || name == "cos" || name == "relu" || name == "gelu" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    self.check_expr(&mut args[0])?;
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "tril" {
                    if args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    self.check_expr(&mut args[0])?; // tensor
                    self.check_expr(&mut args[1])?; // diagonal (int)
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "embedding" {
                    if args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    self.check_expr(&mut args[0])?; // indices
                    self.check_expr(&mut args[1])?; // weights
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "sum" {
                    if args.len() != 1 && args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1, // or 2
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    self.check_expr(&mut args[0])?;
                    if args.len() == 2 {
                        self.check_expr(&mut args[1])?; // dim
                    }
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "transpose" {
                    if args.len() != 3 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 3,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    let t1 = self.check_expr(&mut args[1])?;
                    let t2 = self.check_expr(&mut args[2])?;

                    match t0 {
                        Type::Tensor(_, _) => {}
                        _ => {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: Type::Tensor(Box::new(Type::Void), 0),
                                    found: t0,
                                },
                                Some(args[0].span.clone()),
                            )
                        }
                    }
                    if t1 != Type::I64 {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: t1,
                            },
                            Some(args[1].span.clone()),
                        );
                    }
                    if t2 != Type::I64 {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: t2,
                            },
                            Some(args[2].span.clone()),
                        );
                    }
                    return Ok(t0); // Returns same tensor type (rank preserved)
                } else if name == "reshape" {
                    if args.len() < 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }

                    let t0 = self.check_expr(&mut args[0])?;
                    // Allow Tensor OR ScalarArray
                    if !matches!(t0, Type::Tensor(_, _) | Type::ScalarArray(_, _)) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::Tensor(Box::new(Type::Void), 0),
                                found: t0,
                            },
                            Some(args[0].span.clone()),
                        );
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
                                return self.err(
                                    SemanticError::TypeMismatch {
                                        expected: Type::I64,
                                        found: t,
                                    },
                                    Some(arg.span.clone()),
                                );
                            }
                        }
                    }

                    let inner_type = match t0 {
                        Type::Tensor(inner, _) => inner,
                        Type::ScalarArray(inner, _) => inner,
                        _ => unreachable!(),
                    };

                    // Inference Logic: Inspect args[1] AST if it's a literal
                    let new_rank = if let ExprKind::TensorLiteral(elements) = &args[1].inner {
                        elements.len()
                    } else if let ExprKind::TensorConstLiteral(elements) = &args[1].inner {
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
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    if !matches!(t0, Type::Tensor(_, _)) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::Tensor(Box::new(Type::Void), 0),
                                found: t0,
                            },
                            Some(args[0].span.clone()),
                        );
                    }
                    return Ok(Type::I64);
                } else if name == "slice" {
                    if args.len() != 3 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 3,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    let t1 = self.check_expr(&mut args[1])?;
                    let t2 = self.check_expr(&mut args[2])?;

                    // Arg 0 must be Tensor
                    match t0 {
                        Type::Tensor(_, _) => {}
                        _ => {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: Type::Tensor(Box::new(Type::Void), 0),
                                    found: t0,
                                },
                                Some(args[0].span.clone()),
                            )
                        }
                    }
                    // Arg 1, 2 must be Int
                    if t1 != Type::I64 {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: t1,
                            },
                            Some(args[1].span.clone()),
                        );
                    }
                    if t2 != Type::I64 {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: t2,
                            },
                            Some(args[2].span.clone()),
                        );
                    }
                    return Ok(t0); // Returns same tensor type
                } else if name == "randn" {
                    // LEGACY: Removed in favor of Tensor::randn
                    return self.err(
                        SemanticError::FunctionNotFound(
                            "randn is removed. Use Tensor::randn(shape, req_grad)".into(),
                        ),
                        Some(expr.span.clone()),
                    );
                    // return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "exp" || name == "log" || name == "sqrt" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    return Ok(t0);
                } else if name == "matmul" {
                    if args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    return Ok(t0); // Propagate type
                } else if name == "grad" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    return Ok(t0);
                } else if name == "backward" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    return Ok(Type::Void);
                } else if name == "save_all_params" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: "save_all_params".into(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    return Ok(Type::Void);
                } else if name == "tl_get_memory_mb" {
                    if !args.is_empty() {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: "tl_get_memory_mb".into(),
                                expected: 0,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    return Ok(Type::I64);
                } else if name == "tl_get_metal_pool_bytes"
                    || name == "tl_get_metal_pool_mb"
                    || name == "tl_get_metal_pool_count"
                {
                    if !args.is_empty() {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 0,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    return Ok(Type::I64);
                } else if name == "tl_metal_sync" {
                    if !args.is_empty() {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 0,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    return Ok(Type::Void);
                } else if name == "tl_file_read_binary" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let t = self.check_expr(&mut args[0])?;
                    if !matches!(&t, Type::UserDefined(s) if s == "String") {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::UserDefined("String".into()),
                                found: t,
                            },
                            Some(args[0].span.clone()),
                        );
                    }
                    return Ok(Type::Vec(Box::new(Type::U8)));
                } else if name == "tl_vec_u8_read_i32_be" {
                    if args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    if !matches!(t0, Type::Vec(ref inner) if **inner == Type::U8) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::Vec(Box::new(Type::U8)),
                                found: t0,
                            },
                            Some(args[0].span.clone()),
                        );
                    }
                    // Arg 1: Index (int)
                    let t1 = self.check_expr(&mut args[1])?;
                    if !matches!(t1, Type::I64 | Type::I32) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: t1,
                            },
                            Some(args[1].span.clone()),
                        );
                    }
                    return Ok(Type::I64);
                } else if name == "tl_tensor_from_vec_u8" {
                    if args.len() != 4 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 4,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    for arg in args.iter_mut() {
                        let _ = self.check_expr(arg)?;
                    }
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "tl_tensor_from_u8_labels" {
                    if args.len() != 3 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 3,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    for arg in args.iter_mut() {
                        let _ = self.check_expr(arg)?;
                    }
                    return Ok(Type::Tensor(Box::new(Type::I64), 1));
                } else if name == "tl_arena_get_offset" || name == "tl_arena_get_capacity" {
                    if !args.is_empty() {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 0,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    return Ok(Type::I64);
                } else if name == "tl_arena_is_active" {
                    if !args.is_empty() {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: "tl_arena_is_active".into(),
                                expected: 0,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    return Ok(Type::Bool);
                } else if name == "tl_arena_alloc" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: "tl_arena_alloc".into(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    return Ok(Type::I64); // returns ptr as i64 for testing
                } else if name == "tl_arena_reset" {
                    if !args.is_empty() {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: "tl_arena_reset".into(),
                                expected: 0,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    return Ok(Type::Void);
                } else if name == "tl_arena_init" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: "tl_arena_init".into(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    return Ok(Type::Void);
                } else if name == "args_count" {
                    if !args.is_empty() {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 0,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    return Ok(Type::I64);
                } else if name == "args_get" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    if !matches!(t0, Type::I64 | Type::I32) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: t0,
                            },
                            Some(args[0].span.clone()),
                        );
                    }
                    return Ok(Type::UserDefined("String".to_string()));
                } else if name == "varbuilder_get" {
                    return self.err(
                        SemanticError::FunctionNotFound(
                            "varbuilder_get is removed. Use VarBuilder::get(name, shape)".into(),
                        ),
                        Some(expr.span.clone()),
                    );
                } else if name == "update_all_params" {
                    if args.len() != 1 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let _lr_type = self.check_expr(&mut args[0])?;
                    return Ok(Type::Void);
                } else if name == "varbuilder_grad" {
                    return self.err(
                        SemanticError::FunctionNotFound(
                            "varbuilder_grad is removed. Use VarBuilder::grad(name)".into(),
                        ),
                        Some(expr.span.clone()),
                    );
                } else if name == "softmax" {
                    if args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    let t1 = self.check_expr(&mut args[1])?;
                    if !matches!(t0, Type::Tensor(_, _)) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::Tensor(Box::new(Type::Void), 0),
                                found: t0,
                            },
                            Some(args[0].span.clone()),
                        );
                    }
                    if t1 != Type::I64 {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::I64,
                                found: t1,
                            },
                            Some(args[1].span.clone()),
                        );
                    }
                    return Ok(t0);
                } else if name == "cross_entropy" {
                    if args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    let t1 = self.check_expr(&mut args[1])?;
                    if !matches!(t0, Type::Tensor(_, _)) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::Tensor(Box::new(Type::Void), 0),
                                found: t0,
                            },
                            Some(args[0].span.clone()),
                        );
                    }
                    if !matches!(t1, Type::Tensor(_, _)) {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::Tensor(Box::new(Type::Void), 0),
                                found: t1,
                            },
                            Some(args[1].span.clone()),
                        );
                    }
                    return Ok(Type::Tensor(Box::new(Type::F32), 0));
                } else if name == "save_weights" {
                    if args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    let t0 = self.check_expr(&mut args[0])?;
                    let t1 = self.check_expr(&mut args[1])?;

                    // Arg 0: Tensor OR Struct
                    match t0 {
                        Type::Tensor(_, _) => {}
                        Type::UserDefined(ref s) if s != "String" => {}
                        Type::Struct(_) => {}
                        _ => {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: Type::UserDefined("Tensor or Struct".into()),
                                    found: t0,
                                },
                                Some(expr.span.clone()),
                            );
                        }
                    }

                    if !matches!(t1, Type::UserDefined(ref s) if s == "String") {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::UserDefined("String".into()),
                                found: t1,
                            },
                            Some(args[1].span.clone()),
                        );
                    }
                    return Ok(Type::Void);
                } else if name == "load_weights" {
                    if args.len() == 1 {
                        let t0 = self.check_expr(&mut args[0])?;
                        if !matches!(t0, Type::UserDefined(ref s) if s == "String") {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: Type::UserDefined("String".into()),
                                    found: t0,
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        return Ok(Type::Tensor(Box::new(Type::F32), 0));
                    } else if args.len() == 2 {
                        let t0 = self.check_expr(&mut args[0])?;
                        let t1 = self.check_expr(&mut args[1])?;
                        match t0 {
                            Type::UserDefined(ref s) if s != "String" => {}
                            Type::Struct(_) => {}
                            _ => {
                                return self.err(
                                    SemanticError::TypeMismatch {
                                        expected: Type::UserDefined("Struct".into()),
                                        found: t0,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                        }
                        if !matches!(t1, Type::UserDefined(ref s) if s == "String") {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: Type::UserDefined("String".into()),
                                    found: t1,
                                },
                                Some(args[1].span.clone()),
                            );
                        }
                        return Ok(Type::Void);
                    } else {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: 1, // or 2
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                }

                if name.starts_with("tl_") {
                    let arg_len = args.len();
                    let mut check_all_args = |args: &mut [Expr]| -> Result<(), TlError> {
                        for arg in args.iter_mut() {
                            self.check_expr(arg)?;
                        }
                        Ok(())
                    };

                    let tensor_i64 = Type::Tensor(Box::new(Type::I64), 0);
                    let tensor_f32 = Type::Tensor(Box::new(Type::F32), 0);
                    let tensor_f32_4 = Type::Tensor(Box::new(Type::F32), 4);

                    match name.as_str() {
                        "tl_tokenizer_new" => {
                            if arg_len != 1 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 1,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::I64);
                        }
                        "tl_tokenizer_encode" => {
                            if arg_len != 2 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 2,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(tensor_i64);
                        }
                        "tl_tokenizer_decode" => {
                            if arg_len != 2 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 2,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::UserDefined("String".to_string()));
                        }
                        "tl_gguf_load" => {
                            if arg_len != 1 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 1,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::UserDefined("Map".to_string()));
                        }
                        "tl_tensor_map_get_quantized" => {
                            if arg_len != 2 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 2,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::I64);
                        }
                        "tl_kv_cache_new" => {
                            if arg_len != 1 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 1,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::I64);
                        }
                        "tl_kv_cache_free" => {
                            if arg_len != 1 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 1,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::Void);
                        }
                        "tl_kv_cache_get_k" | "tl_kv_cache_get_v" => {
                            if arg_len != 2 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 2,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(tensor_f32_4);
                        }
                        "tl_kv_cache_update" => {
                            if arg_len != 4 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 4,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::Void);
                        }
                        "tl_file_exists_i64" => {
                            if arg_len != 1 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 1,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::I64);
                        }
                        "tl_read_file" | "tl_read_line" | "tl_prompt" => {
                            if arg_len != 1 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 1,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::UserDefined("String".to_string()));
                        }
                        "tl_write_file" | "tl_download_file" => {
                            if arg_len != 2 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 2,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::I64);
                        }
                        "tl_path_exists" => {
                            if arg_len != 1 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 1,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::Bool);
                        }
                        "tl_print_i64" | "tl_print_string" => {
                            if arg_len != 1 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 1,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::Void);
                        }
                        "tl_string_concat" => {
                            if arg_len != 2 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 2,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::UserDefined("String".to_string()));
                        }
                        "tl_string_from_int" => {
                            if arg_len != 1 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 1,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::UserDefined("String".to_string()));
                        }
                        "tl_string_contains" => {
                            if arg_len != 2 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 2,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(Type::Bool);
                        }
                        "tl_clear_grads" => {
                            if arg_len != 0 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 0,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            return Ok(Type::Void);
                        }
                        "tl_qtensor_matmul" => {
                            if arg_len != 2 {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: name.clone(),
                                        expected: 2,
                                        found: arg_len,
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            check_all_args(args)?;
                            return Ok(tensor_f32);
                        }
                        _ => {}
                    }

                    if name.starts_with("tl_tensor_") {
                        check_all_args(args)?;
                        let return_ty = match name.as_str() {
                            "tl_tensor_len" | "tl_tensor_item_i64" => Type::I64,
                            "tl_tensor_argmax" | "tl_tensor_cat_i64" | "tl_tensor_sample" => {
                                tensor_i64
                            }
                            "tl_tensor_get_shape" => tensor_i64,
                            "tl_tensor_map_get_quantized" => Type::I64,
                            "tl_tensor_print_1" | "tl_tensor_print_2" | "tl_tensor_print_3" => {
                                Type::Void
                            }
                            _ => tensor_f32,
                        };
                        return Ok(return_ty);
                    }
                }

                if let Some(func) = self.functions.get(name).cloned() {
                    if args.len() != func.args.len() {
                        // func.args is empty in current AST parser stub, need to fix that first to check properly
                        // For now, skip arg checking if definitions are empty
                        if !func.args.is_empty() {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: name.clone(),
                                    expected: func.args.len(),
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                    }
                    // Check arg types for function
                    for (i, arg) in args.iter_mut().enumerate() {
                        if i < func.args.len() {
                            let arg_type = self.check_expr(arg)?;
                            let expected_type = &func.args[i].1;
                            if !self.are_types_compatible(expected_type, &arg_type) {
                                return self.err(
                                    SemanticError::TypeMismatch {
                                        expected: expected_type.clone(),
                                        found: arg_type,
                                    },
                                    Some(args[i].span.clone()),
                                );
                            }
                        }
                    }
                    return Ok(func.return_type.clone());
                }

                if let Some(struct_def) = self.structs.get(name).cloned() {
                    // Struct constructor
                    if args.len() != struct_def.fields.len() {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: struct_def.fields.len(),
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    // Check field types
                    for (i, arg) in args.iter_mut().enumerate() {
                        let arg_ty = self.check_expr(arg)?;
                        let required_ty = &struct_def.fields[i].1;
                        if !self.are_types_compatible(required_ty, &arg_ty) {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: required_ty.clone(),
                                    found: arg_ty,
                                },
                                Some(args[i].span.clone()),
                            );
                        }
                    }
                    return Ok(Type::UserDefined(name.clone()));
                } else if let Some(relation) = self.relations.get(name).cloned() {
                    // Logic Query / Relation Call
                    // relation.args gives us expected arity (excluding mask).
                    if args.len() != relation.args.len() {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: name.clone(),
                                expected: relation.args.len(),
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }

                    // Check arguments (Entity/Symbol/LogicVar)
                    for arg in args.iter_mut() {
                        let arg_ty = self.check_expr(arg)?;
                        if !matches!(arg_ty, Type::Entity | Type::I64) {
                            // Potentially error, or allow implicit casting?
                        }
                    }

                    // Return Tensor result (e.g. 1D tensor for now, or boolean tensor)
                    // tl_query returns *mut Tensor. Semantic type is Tensor.
                    return Ok(Type::Tensor(Box::new(Type::F32), 1));
                }

                self.err(
                    SemanticError::FunctionNotFound(name.clone()),
                    Some(expr.span.clone()),
                )
            }
            ExprKind::TensorLiteral(elements) | ExprKind::TensorConstLiteral(elements) => {
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
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: first_type,
                                    found: t,
                                },
                                Some(e.span.clone()),
                            );
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
            ExprKind::IndexAccess(target, _indices) => {
                let target_type = self.check_expr(target)?;
                match target_type {
                    Type::Tensor(inner, _rank) => Ok(*inner), // Accessing reduces rank to scalar (in this simple logic)
                    // Actually in Tensor Logic a[i, j] usually denotes the tensor itself with indices bound,
                    // but for type checking purposes, it resolves to the element type if fully indexed,
                    // or acts as the tensor for equations.
                    // Let's assume it validates to the Inner type for now.
                    _ => self.err(
                        SemanticError::TypeMismatch {
                            expected: Type::Tensor(Box::new(Type::Void), 0),
                            found: target_type,
                        },
                        Some(target.span.clone()),
                    ),
                }
            }
            ExprKind::UnOp(op, inner) => {
                let t = self.check_expr(inner)?;
                match op {
                    UnOp::Neg => {
                        // Neg supports Int, Float, Tensor<Int/Float>
                        match t {
                            Type::I64 | Type::F32 => Ok(t),
                            Type::Tensor(ref tensor_inner, _) => match **tensor_inner {
                                Type::I64 | Type::F32 => Ok(t),
                                _ => self.err(
                                    SemanticError::TypeMismatch {
                                        expected: Type::F32,
                                        found: t,
                                    },
                                    Some(inner.span.clone()),
                                ),
                            },
                            _ => self.err(
                                SemanticError::TypeMismatch {
                                    expected: Type::F32,
                                    found: t,
                                },
                                Some(inner.span.clone()),
                            ),
                        }
                    }
                    UnOp::Not => {
                        // Not supports Bool, Tensor<Bool>
                        match t {
                            Type::Bool => Ok(t),
                            Type::Tensor(ref tensor_inner, _) => match **tensor_inner {
                                Type::Bool => Ok(t),
                                _ => self.err(
                                    SemanticError::TypeMismatch {
                                        expected: Type::Bool,
                                        found: t,
                                    },
                                    Some(inner.span.clone()),
                                ),
                            },
                            _ => self.err(
                                SemanticError::TypeMismatch {
                                    expected: Type::Bool,
                                    found: t,
                                },
                                Some(inner.span.clone()),
                            ),
                        }
                    }
                }
            }
            ExprKind::TensorComprehension {
                indices,
                clauses,
                body,
            } => {
                // 1. Enter scope to declare generators
                self.enter_scope();

                // 2. Process clauses (Generators first, then conditions)
                for clause in clauses.iter_mut() {
                    match clause {
                        ComprehensionClause::Generator { name, range } => {
                            self.check_expr(range)?;
                            self.declare_variable(name.clone(), Type::I64, true)?;
                        }
                        ComprehensionClause::Condition(cond) => {
                            let cond_ty = self.check_expr(cond)?;
                            if cond_ty != Type::Bool {
                                return self.err(
                                    SemanticError::TypeMismatch {
                                        expected: Type::Bool,
                                        found: cond_ty,
                                    },
                                    Some(cond.span.clone()),
                                );
                            }
                        }
                    }
                }

                // 3. Check Body
                let body_type = if let Some(b) = body {
                    self.check_expr(b)?
                } else {
                    // Implicit body: Type::F32
                    Type::F32
                };

                // 4. Exit scope
                self.exit_scope();

                // 5. Result Type is Tensor<BodyType, Rank = indices.len()>
                Ok(Type::Tensor(Box::new(body_type), indices.len()))
            }

            ExprKind::Block(stmts) => {
                self.enter_scope();
                let ret = self.check_block_stmts(stmts)?;
                self.exit_scope();
                Ok(ret)
            }
            ExprKind::IfExpr(cond, then_block, else_block) => {
                let cond_type = self.check_expr(cond)?;
                if cond_type != Type::Bool {
                    return self.err(
                        SemanticError::TypeMismatch {
                            expected: Type::Bool,
                            found: cond_type,
                        },
                        Some(cond.span.clone()),
                    );
                }

                // Check Then Block
                self.enter_scope();
                let mut then_type = Type::Void;
                let then_len = then_block.len();
                for (i, stmt) in then_block.iter_mut().enumerate() {
                    if i == then_len - 1 {
                        if let StmtKind::Expr(e) = &mut stmt.inner {
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
                            if let StmtKind::Expr(e) = &mut stmt.inner {
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
                if self.are_types_compatible(&then_type, &else_type) {
                    // Return the more specific type (e.g. if one is rank 0 and another is rank N)
                    let ret_ty = match (&then_type, &else_type) {
                        (Type::Tensor(_i1, r1), Type::Tensor(_, r2)) if *r1 == 0 && *r2 != 0 => else_type,
                        (Type::Tensor(_i1, r1), Type::Tensor(_, r2)) if *r1 != 0 && *r2 == 0 => then_type,
                        _ => then_type,
                    };
                    Ok(ret_ty)
                } else {
                    self.err(
                        SemanticError::TypeMismatch {
                            expected: then_type,
                            found: else_type,
                        },
                        Some(expr.span.clone()),
                    )
                }
                // End of ExprKind::IfExpr
            }

            ExprKind::As(expr, target_type) => {
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
                    (Type::I64, Type::Usize) => Ok(Type::Usize),
                    (Type::Usize, Type::I64) => Ok(Type::I64),
                    (Type::I64, Type::I32) => Ok(Type::I32),
                    (Type::I32, Type::I64) => Ok(Type::I64),
                    (Type::Bool, Type::I64) => Ok(Type::I64),
                    (Type::I64, Type::Bool) => Ok(Type::Bool),
                    (Type::Bool, Type::F32) => Ok(Type::F32),
                    (Type::F32, Type::Bool) => Ok(Type::Bool),
                    _ => self.err(
                        SemanticError::TypeMismatch {
                            expected: target_type.clone(),
                            found: source_type,
                        },
                        Some(expr.span.clone()),
                    ),
                }
            }
            ExprKind::StaticMethodCall(type_name, method_name, args) => {
                let resolved_type = self.resolve_symbol_name(type_name);
                if *type_name != resolved_type {
                    *type_name = resolved_type.clone();
                }

                // Special handling for Param::checkpoint to allow method references
                if type_name == "Param" && method_name == "checkpoint" {
                    if args.len() != 2 {
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: "Param::checkpoint".into(),
                                expected: 2,
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }

                    // Check arg 0 (method ref)
                    let mut is_valid_method_ref = false;
                    if let ExprKind::FieldAccess(obj, method_name) = &mut args[0].inner {
                        // Check obj type
                        if let Ok(obj_type) = self.check_expr(obj) {
                            let t_name_opt = match obj_type {
                                Type::UserDefined(n) => Some(n),
                                Type::Struct(n) => Some(n),
                                Type::Tensor(_, _) => Some("Tensor".to_string()),
                                _ => None,
                            };

                            if let Some(t_name) = t_name_opt {
                                if let Some(methods) = self.methods.get(&t_name) {
                                    if methods.contains_key(method_name) {
                                        is_valid_method_ref = true;
                                    }
                                }
                            }
                        }
                    }

                    if !is_valid_method_ref {
                        // Fallback
                        let _ = self.check_expr(&mut args[0])?;
                    }

                    // Check arg 1
                    let arg1_type = self.check_expr(&mut args[1])?;
                    return Ok(arg1_type);
                }

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
                        return self.err(
                            SemanticError::ArgumentCountMismatch {
                                name: format!("{}::{}", type_name, method_name),
                                expected: func.args.len(),
                                found: args.len(),
                            },
                            Some(expr.span.clone()),
                        );
                    }
                    for (arg_val, (_, arg_type)) in args.iter_mut().zip(&func.args) {
                        let val_type = self.check_expr(arg_val)?;
                        if !self.are_types_compatible(&val_type, arg_type) {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: arg_type.clone(),
                                    found: val_type,
                                },
                                Some(arg_val.span.clone()),
                            ); // Need index for args[i]. But loop iterates over args.iter_mut().zip.
                               // Wait, args is specific arg.
                               // arg_val is &mut Expr. So arg_val.span is available.
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
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "File::open".into(),
                                    expected: 2,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        Ok(Type::UserDefined("File".to_string()))
                    }
                    ("Path", "new") => {
                        if args.len() != 1 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Path::new".into(),
                                    expected: 1,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        Ok(Type::UserDefined("Path".to_string()))
                    }
                    ("System", "time") => Ok(Type::F32),
                    ("System", "sleep") => Ok(Type::Void),
                    ("System", "memory_mb") => Ok(Type::I64),
                    ("System", "metal_pool_bytes") => Ok(Type::I64),
                    ("System", "metal_pool_mb") => Ok(Type::I64),
                    ("System", "metal_pool_count") => Ok(Type::I64),
                    ("System", "metal_sync") => Ok(Type::Void),
                    ("System", "pool_count") => Ok(Type::I64),
                    ("System", "refcount_count") => Ok(Type::I64),
                    ("System", "scope_depth") => Ok(Type::I64),
                    ("Env", "get") => Ok(Type::UserDefined("String".into())),
                    ("Env", "set") => {
                        if args.len() != 2 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Env::set".into(),
                                    expected: 2,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        Ok(Type::Void)
                    }
                    ("Http", "get") => Ok(Type::UserDefined("String".into())),
                    ("Http", "download") => Ok(Type::Bool),
                    ("Image", "load_grayscale") => Ok(Type::Vec(Box::new(Type::U8))),
                    ("Image", "width") => Ok(Type::I64),
                    ("Image", "height") => Ok(Type::I64),
                    ("Args", "count") => Ok(Type::I64),
                    ("Args", "get") => Ok(Type::UserDefined("String".into())),
                    ("Arena", "get_offset") => Ok(Type::I64),
                    ("Arena", "alloc") => Ok(Type::I64),
                    ("Arena", "init") => Ok(Type::Void),
                    ("Arena", "is_active") => Ok(Type::Bool),
                    ("Tokenizer", "new") => Ok(Type::Struct("Tokenizer".into())),
                    ("KVCache", "new") => Ok(Type::Struct("KVCache".into())),
                    ("Map", "load") => Ok(Type::UserDefined("Map".into())),
                    ("File", "exists") => Ok(Type::Bool),
                    ("File", "read") => Ok(Type::UserDefined("String".into())),
                    ("File", "write") => Ok(Type::Bool),
                    ("File", "download") => Ok(Type::Bool),
                    ("File", "read_binary") => Ok(Type::Vec(Box::new(Type::U8))),
                    ("Path", "exists") => Ok(Type::Bool),
                    ("String", "from_int") => Ok(Type::UserDefined("String".into())),
                    // --- New Static Methods for Refactor ---
                    ("Tensor", "zeros") => {
                        // Tensor::zeros(shape, requires_grad)
                        if args.is_empty() || args.len() > 2 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Tensor::zeros".into(),
                                    expected: 2,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        if args.len() == 2 {
                            let t1 = self.check_expr(&mut args[1])?;
                            if !matches!(t1, Type::Bool) {
                                return self.err(
                                    SemanticError::TypeMismatch {
                                        expected: Type::Bool,
                                        found: t1,
                                    },
                                    Some(args[1].span.clone()),
                                );
                            }
                        }
                        // Rank inference is hard without const eval. Return dynamic rank 1 for now or 0?
                        // Actually, Tensor types don't enforce rank strictly in checking yet (dynamic).
                        Ok(Type::Tensor(Box::new(Type::F32), 0))
                    }
                    ("Tensor", "ones") => {
                        // Tensor::ones(shape, requires_grad)
                        if args.is_empty() || args.len() > 2 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Tensor::ones".into(),
                                    expected: 2,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        if args.len() == 2 {
                            let t1 = self.check_expr(&mut args[1])?;
                            if !matches!(t1, Type::Bool) {
                                return self.err(
                                    SemanticError::TypeMismatch {
                                        expected: Type::Bool,
                                        found: t1,
                                    },
                                    Some(args[1].span.clone()),
                                );
                            }
                        }
                        Ok(Type::Tensor(Box::new(Type::F32), 0))
                    }
                    ("Tensor", "randn") => {
                        // Tensor::randn(shape, requires_grad)
                        if args.is_empty() || args.len() > 2 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Tensor::randn".into(),
                                    expected: 2,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        if args.len() == 2 {
                            let t1 = self.check_expr(&mut args[1])?;
                            if !matches!(t1, Type::Bool) {
                                return self.err(
                                    SemanticError::TypeMismatch {
                                        expected: Type::Bool,
                                        found: t1,
                                    },
                                    Some(args[1].span.clone()),
                                );
                            }
                        }
                        Ok(Type::Tensor(Box::new(Type::F32), 0))
                    }
                    ("Tensor", "load") => {
                        // Tensor::load(path)
                        if args.len() != 1 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Tensor::load".into(),
                                    expected: 1,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        Ok(Type::Tensor(Box::new(Type::F32), 0))
                    }
                    ("Tensor", "clear_grads") => Ok(Type::Void),
                    ("Tensor", "matmul")
                    | ("Tensor", "add")
                    | ("Tensor", "mul")
                    | ("Tensor", "silu")
                    | ("Tensor", "rms_norm")
                    | ("Tensor", "embedding")
                    | ("Tensor", "scale")
                    | ("Tensor", "transpose")
                    | ("Tensor", "transpose_2d")
                    | ("Tensor", "apply_rope")
                    | ("Tensor", "repeat_interleave")
                    | ("Tensor", "new_causal_mask")
                    | ("Tensor", "narrow")
                    | ("Tensor", "cat_4d")
                    | ("Tensor", "matmul_4d")
                    | ("Tensor", "add_4d")
                    | ("Tensor", "softmax")
                    | ("Tensor", "rope_new_cos")
                    | ("Tensor", "rope_new_sin")
                    | ("Tensor", "cat2")
                    | ("Tensor", "reshape_dims")
                    | ("Tensor", "reshape_2d")
                    | ("Tensor", "reshape_3d_to_2d")
                    | ("Tensor", "get_shape")
                    | ("Tensor", "from_vec_u8") => Ok(Type::Tensor(Box::new(Type::F32), 0)),
                    ("Tensor", "matmul_quantized") => Ok(Type::Tensor(Box::new(Type::F32), 0)),
                    ("Tensor", "argmax") | ("Tensor", "cat_i64") | ("Tensor", "sample") => {
                        Ok(Type::Tensor(Box::new(Type::I64), 0))
                    }
                    ("Tensor", "item_i64") => Ok(Type::I64),
                    ("Tensor", "len") => Ok(Type::I64),
                    ("Tensor", "from_u8_labels") => Ok(Type::Tensor(Box::new(Type::I64), 0)),
                    ("VarBuilder", "get") => {
                        if args.len() < 2 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "VarBuilder::get".into(),
                                    expected: 2,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let t0 = self.check_expr(&mut args[0])?;
                        if !matches!(t0, Type::UserDefined(ref s) if s == "String") {
                            return self.err(
                                SemanticError::TypeMismatch {
                                    expected: Type::UserDefined("String".into()),
                                    found: t0,
                                },
                                Some(args[0].span.clone()),
                            );
                        }
                        if args.len() == 2 {
                            let _ = self.check_expr(&mut args[1])?;
                        } else {
                            for arg in &mut args[1..] {
                                let t = self.check_expr(arg)?;
                                if !matches!(t, Type::I64 | Type::I32) {
                                    return self.err(
                                        SemanticError::TypeMismatch {
                                            expected: Type::I64,
                                            found: t, // used
                                        },
                                        Some(arg.span.clone()),
                                    );
                                }
                            }
                        }
                        Ok(Type::Tensor(Box::new(Type::F32), 0))
                    }
                    ("VarBuilder", "grad") => {
                        if args.len() != 1 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "VarBuilder::grad".into(),
                                    expected: 1,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        Ok(Type::Tensor(Box::new(Type::F32), 0))
                    }
                    // Param static methods
                    ("Param", "save_all") | ("Param", "load_all") => {
                        if args.is_empty() || args.len() > 2 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: format!("Param::{}", method_name),
                                    expected: 2, // 1 or 2
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        if args.len() == 2 {
                            let _ = self.check_expr(&mut args[1])?;
                        }
                        Ok(Type::Void)
                    }
                    ("Param", "save") => {
                        if args.len() != 2 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Param::save".into(),
                                    expected: 2,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        let _ = self.check_expr(&mut args[1])?;
                        Ok(Type::Void)
                    }
                    ("Param", "load") => {
                        if args.is_empty() || args.len() > 2 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Param::load".into(),
                                    expected: 2, // 1 or 2
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let _t0 = self.check_expr(&mut args[0])?;
                        if args.len() == 2 {
                            let _ = self.check_expr(&mut args[1])?;
                            Ok(Type::Void) // load(struct, path) -> Void
                        } else {
                            Ok(Type::Tensor(Box::new(Type::F32), 0)) // load(path) -> Tensor
                        }
                    }
                    ("Param", "add") => {
                        if args.len() != 2 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Param::add".into(),
                                    expected: 2,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        let _ = self.check_expr(&mut args[1])?;
                        Ok(Type::Void)
                    }
                    ("Param", "register") => {
                        if args.len() != 1 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Param::register".into(),
                                    expected: 1,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let t = self.check_expr(&mut args[0])?;
                        Ok(t)
                    }
                    ("Param", "update_all") => {
                        if args.len() != 1 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Param::update_all".into(),
                                    expected: 1,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        Ok(Type::Void)
                    }
                    ("Param", "register_modules") => {
                        if args.len() != 1 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Param::register_modules".into(),
                                    expected: 1,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        Ok(Type::Void)
                    }
                    ("Param", "checkpoint") => {
                        if args.len() != 2 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Param::checkpoint".into(),
                                    expected: 2,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }

                        // Check if first arg is a valid method reference (obj.method)
                        let mut is_valid_method_ref = false;
                        if let ExprKind::FieldAccess(obj, method_name) = &mut args[0].inner {
                            // We need to check obj type, but checking obj expr might fail if it contains errors?
                            // Try check_expr on obj
                            if let Ok(obj_type) = self.check_expr(obj) {
                                let type_name = match obj_type {
                                    Type::UserDefined(n) => Some(n),
                                    Type::Struct(n) => Some(n),
                                    Type::Tensor(_, _) => Some("Tensor".to_string()),
                                    _ => None,
                                };

                                if let Some(t_name) = type_name {
                                    if let Some(methods) = self.methods.get(&t_name) {
                                        if methods.contains_key(method_name) {
                                            is_valid_method_ref = true;
                                        }
                                    }
                                }
                            }
                        }

                        if !is_valid_method_ref {
                            // Fallback to normal check (e.g. for function pointers or fields)
                            // This will fail if it was a FieldAccess to a non-existent field (which is what usually happens for methods)
                            let _ = self.check_expr(&mut args[0])?;
                        }

                        let arg1_type = self.check_expr(&mut args[1])?;
                        Ok(arg1_type)
                    }
                    ("Param", "set_device") => {
                        if args.len() != 1 {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: "Param::set_device".into(),
                                    expected: 1,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }
                        let _ = self.check_expr(&mut args[0])?;
                        Ok(Type::Void)
                    }
                    _ => {
                        // Try as a module function: type_name::method_name might be a qualified function call
                        let full_name = format!("{}::{}", type_name, method_name);
                        if let Some(func) = self.functions.get(&full_name).cloned() {
                            // Check arguments
                            if args.len() != func.args.len() && !func.args.is_empty() {
                                return self.err(
                                    SemanticError::ArgumentCountMismatch {
                                        name: full_name,
                                        expected: func.args.len(),
                                        found: args.len(),
                                    },
                                    Some(expr.span.clone()),
                                );
                            }
                            for (i, arg) in args.iter_mut().enumerate() {
                                if i < func.args.len() {
                                    let arg_type = self.check_expr(arg)?;
                                    let expected_type = &func.args[i].1;
                                    if !self.are_types_compatible(expected_type, &arg_type) {
                                        return self.err(
                                            SemanticError::TypeMismatch {
                                                expected: expected_type.clone(),
                                                found: arg_type,
                                            },
                                            Some(args[i].span.clone()),
                                        );
                                    }
                                }
                            }
                            Ok(func.return_type.clone())
                        } else {
                            self.err(
                                SemanticError::FunctionNotFound(full_name),
                                Some(expr.span.clone()),
                            )
                        }
                    }
                }
            }
            ExprKind::FieldAccess(obj, field_name) => {
                let obj_type = self.check_expr(obj)?;
                let name = match obj_type {
                    Type::UserDefined(n) => n,
                    Type::Struct(n) => n,
                    _ => {
                        return self.err(
                            SemanticError::TypeMismatch {
                                expected: Type::UserDefined("Struct".into()),
                                found: obj_type,
                            },
                            Some(expr.span.clone()),
                        );
                    }
                };

                if let Some(struct_def) = self.structs.get(&name) {
                    for (f_name, f_type) in &struct_def.fields {
                        if f_name == field_name {
                            return Ok(f_type.clone());
                        }
                    }
                    return self.err(
                        SemanticError::VariableNotFound(format!("Field {}", field_name)),
                        Some(expr.span.clone()),
                    );
                }
                self.err(SemanticError::StructNotFound(name), Some(expr.span.clone()))
            }
            ExprKind::MethodCall(obj, method_name, args) => {
                let obj_type = self.check_expr(obj)?;

                // 1. Try checking with the unified TypeRegistry
                if let Some(ret_ty) =
                    self.check_method_call_with_registry(&obj_type, method_name, args, &expr.span)?
                {
                    return Ok(ret_ty);
                }

                // 2. Check for UserDefined methods (impl blocks)
                let type_name = match &obj_type {
                    Type::UserDefined(name) => name.clone(),
                    Type::Struct(name) => name.clone(),
                    Type::Enum(name) => name.clone(),
                    _ => TypeRegistry::type_to_key(&obj_type),
                };

                if let Some(methods) = self.methods.get(&type_name) {
                    if let Some(method_def) = methods.get(method_name) {
                        // Clone the signature to avoid borrow checker issues with self.methods
                        let args_types: Vec<(String, Type)> = method_def.args.clone();
                        let ret_type = method_def.return_type.clone();

                        let implicit_self = !args_types.is_empty() && args_types[0].0 == "self";
                        let expected_arg_count = if implicit_self {
                            args_types.len() - 1
                        } else {
                            args_types.len()
                        };

                        if args.len() != expected_arg_count {
                            return self.err(
                                SemanticError::ArgumentCountMismatch {
                                    name: method_name.clone(),
                                    expected: expected_arg_count,
                                    found: args.len(),
                                },
                                Some(expr.span.clone()),
                            );
                        }

                        let start_idx = if implicit_self { 1 } else { 0 };
                        for (i, arg) in args.iter_mut().enumerate() {
                            let arg_type = self.check_expr(arg)?;
                            let expected_type = &args_types[start_idx + i].1;
                            if !self.are_types_compatible(&arg_type, expected_type) {
                                return self.err(
                                    SemanticError::TypeMismatch {
                                        expected: expected_type.clone(),
                                        found: arg_type,
                                    },
                                    Some(arg.span.clone()),
                                );
                            }
                        }
                        return Ok(ret_type);
                    }
                }

                // 3. Method not found
                self.err(
                    SemanticError::MethodNotFound {
                        type_name,
                        method_name: method_name.clone(),
                    },
                    Some(expr.span.clone()),
                )
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
                true // Allow scalar to tensor promotion
            }
            (primitive, Type::Tensor(inner, _rank))
                if self.are_types_compatible(primitive, inner) =>
            {
                true
            }
            // Allow I32/I64/F32/F64 inter-compatibility for some operations
            (Type::I64, Type::F32) | (Type::F32, Type::I64) => true,
            (Type::I32, Type::I64) | (Type::I64, Type::I32) => true,
            (Type::I32, Type::F32) | (Type::F32, Type::I32) => true,
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
            (Type::F64, Type::I64) => true,
            _ => false,
        }
    }

    fn collect_indices(&self, expr: &Expr, indices: &mut HashSet<String>) {
        match &expr.inner {
            ExprKind::IndexAccess(target, idxs) => {
                self.collect_indices(target, indices);
                for idx in idxs {
                    if let ExprKind::Variable(name) = &idx.inner {
                        indices.insert(name.clone());
                    } else if let ExprKind::IndexAccess(_, _) = &idx.inner {
                        // Recurse if index itself has structure? Unlikely for now.
                        self.collect_indices(idx, indices);
                    } else {
                        // Ignore literals/expressions in indices for equation collection
                    }
                }
            }
            ExprKind::BinOp(left, _, right) => {
                self.collect_indices(left, indices);
                self.collect_indices(right, indices);
            }
            ExprKind::UnOp(_, inner) => {
                self.collect_indices(inner, indices);
            }
            ExprKind::FnCall(_, args) => {
                for arg in args {
                    self.collect_indices(arg, indices);
                }
            }
            ExprKind::MethodCall(obj, _, args) => {
                self.collect_indices(obj, indices);
                for arg in args {
                    self.collect_indices(arg, indices);
                }
            }
            ExprKind::As(expr, _) => {
                self.collect_indices(expr, indices);
            }
            ExprKind::IfExpr(cond, _then_block, _else_block) => {
                self.collect_indices(cond, indices);
            }
            ExprKind::Block(_) => {}
            ExprKind::TensorComprehension {
                indices: _,
                clauses,
                body,
            } => {
                // Collect indices from generator ranges and conditions
                for clause in clauses {
                    match clause {
                        ComprehensionClause::Generator { name: _, range } => {
                            self.collect_indices(range, indices);
                        }
                        ComprehensionClause::Condition(cond) => {
                            self.collect_indices(cond, indices);
                        }
                    }
                }

                // Body - filter out bound generators
                // Wait, condition also sees bound generators.
                // The logic here is: collect FREE indices from the whole expression.
                // Any index used in Body/Range/Condition that is NOT generated by this comprehension is Free.

                // 1. Collect all used indices in clauses and body
                let mut sub_indices = HashSet::new();
                for clause in clauses {
                    match clause {
                        ComprehensionClause::Generator { name: _, range } => {
                            self.collect_indices(range, &mut sub_indices);
                        }
                        ComprehensionClause::Condition(cond) => {
                            self.collect_indices(cond, &mut sub_indices);
                        }
                    }
                }

                if let Some(b) = body {
                    self.collect_indices(b, &mut sub_indices);
                }

                // 2. Remove locally generated indices
                for clause in clauses {
                    if let ComprehensionClause::Generator { name, .. } = clause {
                        sub_indices.remove(name);
                    }
                }

                indices.extend(sub_indices);
            }
            _ => {}
        }
    }
    // Helper to infer free indices (used in StmtKind::Let for Tensor Equation)
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
