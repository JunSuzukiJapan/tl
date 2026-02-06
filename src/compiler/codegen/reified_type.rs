// src/compiler/codegen/reified_type.rs
//
// ReifiedType: A structure that holds complete type information including the mangled name,
// original base type name, and concrete type arguments. This eliminates the need for 
// unreliable string parsing to recover type information from mangled names.

use crate::compiler::ast::Type;
use std::collections::HashMap;

/// Holds complete type information for a monomorphized generic type.
/// 
/// # Example
/// For `Entry<String, i64>`:
/// - `mangled_name`: "Entry_String_i64"
/// - `base_name`: "Entry"
/// - `type_args`: [Type::String, Type::I64]
#[derive(Clone, Debug, PartialEq)]
pub struct ReifiedType {
    /// The mangled name used in LLVM IR (e.g., "Entry_String_i64")
    pub mangled_name: String,
    /// The original base type name (e.g., "Entry")
    pub base_name: String,
    /// The concrete type arguments (e.g., [String, I64])
    pub type_args: Vec<Type>,
}

impl ReifiedType {
    /// Create a new ReifiedType
    pub fn new(mangled_name: String, base_name: String, type_args: Vec<Type>) -> Self {
        Self {
            mangled_name,
            base_name,
            type_args,
        }
    }
    
    /// Create a ReifiedType for a non-generic type (no type arguments)
    pub fn simple(name: &str) -> Self {
        Self {
            mangled_name: name.to_string(),
            base_name: name.to_string(),
            type_args: vec![],
        }
    }
    
    /// Check if this type has generic arguments
    pub fn is_generic(&self) -> bool {
        !self.type_args.is_empty()
    }
    
    /// Get the number of type arguments
    pub fn arity(&self) -> usize {
        self.type_args.len()
    }
    
    /// Build a substitution map from generic parameter names to concrete types
    pub fn build_subst_map(&self, param_names: &[String]) -> HashMap<String, Type> {
        let mut map = HashMap::new();
        for (i, param) in param_names.iter().enumerate() {
            if let Some(arg) = self.type_args.get(i) {
                map.insert(param.clone(), arg.clone());
            }
        }
        map
    }
}

/// Registry for looking up ReifiedType by mangled name
#[derive(Default, Clone, Debug)]
pub struct ReifiedTypeRegistry {
    /// Map from mangled name to ReifiedType
    types: HashMap<String, ReifiedType>,
}

impl ReifiedTypeRegistry {
    pub fn new() -> Self {
        Self { types: HashMap::new() }
    }
    
    /// Register a new reified type
    pub fn register(&mut self, reified: ReifiedType) {
        self.types.insert(reified.mangled_name.clone(), reified);
    }
    
    /// Register from components
    pub fn register_from_parts(&mut self, mangled_name: &str, base_name: &str, type_args: &[Type]) {
        let reified = ReifiedType::new(
            mangled_name.to_string(),
            base_name.to_string(),
            type_args.to_vec(),
        );
        self.types.insert(mangled_name.to_string(), reified);
    }
    
    /// Lookup a reified type by mangled name
    pub fn lookup(&self, mangled_name: &str) -> Option<&ReifiedType> {
        self.types.get(mangled_name)
    }
    
    /// Lookup and clone
    pub fn lookup_cloned(&self, mangled_name: &str) -> Option<ReifiedType> {
        self.types.get(mangled_name).cloned()
    }
    
    /// Get type arguments for a mangled name
    pub fn get_type_args(&self, mangled_name: &str) -> Option<&Vec<Type>> {
        self.types.get(mangled_name).map(|r| &r.type_args)
    }
    
    /// Get base name for a mangled name
    pub fn get_base_name(&self, mangled_name: &str) -> Option<&str> {
        self.types.get(mangled_name).map(|r| r.base_name.as_str())
    }
}
