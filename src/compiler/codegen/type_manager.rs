use std::collections::HashMap;
pub use crate::compiler::codegen::expr::{
    StaticMethod, InstanceMethod, StaticMethodEval, StaticMethodUneval, InstanceMethodEval, InstanceMethodUneval
};
use crate::compiler::ast::Type;

/// An overload of a method with its implementation and signature
#[derive(Clone)]
pub struct StaticOverload {
    pub impl_fn: StaticMethod,
    pub arg_types: Vec<Type>,
    pub return_type: Type,
}

#[derive(Clone)]
pub struct InstanceOverload {
    pub impl_fn: InstanceMethod,
    pub arg_types: Vec<Type>,
    pub return_type: Type,
}

/// Represents a type definition within the CodeGenerator, managing its methods.
/// Methods are keyed by TL name (e.g., "sum", "load") with overloads stored in a Vec.
pub struct CodeGenType {
    pub name: String,
    // TL name -> list of overloads
    pub static_methods: HashMap<String, Vec<StaticOverload>>,
    pub instance_methods: HashMap<String, Vec<InstanceOverload>>,
}

impl CodeGenType {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            static_methods: HashMap::new(),
            instance_methods: HashMap::new(),
        }
    }

    /// Register a static method by TL name
    pub fn register_static_method(
        &mut self,
        tl_name: &str,
        method: StaticMethod,
        args: Vec<Type>,
        ret: Type,
    ) {
        self.static_methods
            .entry(tl_name.to_string())
            .or_default()
            .push(StaticOverload {
                impl_fn: method,
                arg_types: args,
                return_type: ret,
            });
    }

    /// Register an instance method by TL name
    pub fn register_instance_method(
        &mut self,
        tl_name: &str,
        method: InstanceMethod,
        args: Vec<Type>,
        ret: Type,
    ) {
        self.instance_methods
            .entry(tl_name.to_string())
            .or_default()
            .push(InstanceOverload {
                impl_fn: method,
                arg_types: args,
                return_type: ret,
            });
    }

    // Convenience methods for registering evaluated/unevaluated methods
    pub fn register_evaluated_static_method(
        &mut self,
        tl_name: &str,
        method: StaticMethodEval,
        args: Vec<Type>,
        ret: Type,
    ) {
        self.register_static_method(tl_name, StaticMethod::Evaluated(method), args, ret);
    }

    pub fn register_unevaluated_static_method(
        &mut self,
        tl_name: &str,
        method: StaticMethodUneval,
        args: Vec<Type>,
        ret: Type,
    ) {
        self.register_static_method(tl_name, StaticMethod::Unevaluated(method), args, ret);
    }

    pub fn register_evaluated_instance_method(
        &mut self,
        tl_name: &str,
        method: InstanceMethodEval,
        args: Vec<Type>,
        ret: Type,
    ) {
        self.register_instance_method(tl_name, InstanceMethod::Evaluated(method), args, ret);
    }

    pub fn register_unevaluated_instance_method(
        &mut self,
        tl_name: &str,
        method: InstanceMethodUneval,
        args: Vec<Type>,
        ret: Type,
    ) {
        self.register_instance_method(tl_name, InstanceMethod::Unevaluated(method), args, ret);
    }

    /// Register a static method signature only (no implementation).
    /// Used for semantics analysis when the codegen is handled elsewhere.
    pub fn register_static_signature(&mut self, tl_name: &str, args: Vec<Type>, ret: Type) {
        self.static_methods
            .entry(tl_name.to_string())
            .or_default()
            .push(StaticOverload {
                impl_fn: StaticMethod::SignatureOnly,
                arg_types: args,
                return_type: ret,
            });
    }

    /// Register an instance method signature only (no implementation).
    /// Used for semantics analysis when the codegen is handled elsewhere.
    pub fn register_instance_signature(&mut self, tl_name: &str, args: Vec<Type>, ret: Type) {
        self.instance_methods
            .entry(tl_name.to_string())
            .or_default()
            .push(InstanceOverload {
                impl_fn: InstanceMethod::SignatureOnly,
                arg_types: args,
                return_type: ret,
            });
    }

    /// Get all overloads for a static method by TL name
    pub fn get_static_overloads(&self, tl_name: &str) -> Option<&Vec<StaticOverload>> {
        self.static_methods.get(tl_name)
    }

    /// Get all overloads for an instance method by TL name
    pub fn get_instance_overloads(&self, tl_name: &str) -> Option<&Vec<InstanceOverload>> {
        self.instance_methods.get(tl_name)
    }

    /// Get all static method signatures by TL name (for semantic analysis)
    pub fn get_static_signatures(&self, tl_name: &str) -> Vec<(&Vec<Type>, &Type)> {
        self.static_methods
            .get(tl_name)
            .map(|overloads| {
                overloads.iter()
                    .map(|o| (&o.arg_types, &o.return_type))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all instance method signatures by TL name (for semantic analysis)
    pub fn get_instance_signatures(&self, tl_name: &str) -> Vec<(&Vec<Type>, &Type)> {
        self.instance_methods
            .get(tl_name)
            .map(|overloads| {
                overloads.iter()
                    .map(|o| (&o.arg_types, &o.return_type))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if a static method exists
    pub fn has_static_method(&self, tl_name: &str) -> bool {
        self.static_methods.contains_key(tl_name)
    }

    /// Check if an instance method exists
    pub fn has_instance_method(&self, tl_name: &str) -> bool {
        self.instance_methods.contains_key(tl_name)
    }

    /// Find matching static overload by argument types
    pub fn find_static_overload(&self, tl_name: &str, arg_types: &[Type]) -> Option<&StaticOverload> {
        self.static_methods.get(tl_name)?.iter().find(|o| {
            o.arg_types.len() == arg_types.len() &&
            o.arg_types.iter().zip(arg_types).all(|(expected, actual)| types_compatible(expected, actual))
        })
    }

    /// Find matching instance overload by argument types
    pub fn find_instance_overload(&self, tl_name: &str, arg_types: &[Type]) -> Option<&InstanceOverload> {
        self.instance_methods.get(tl_name)?.iter().find(|o| {
            o.arg_types.len() == arg_types.len() &&
            o.arg_types.iter().zip(arg_types).all(|(expected, actual)| types_compatible(expected, actual))
        })
    }
}

/// Check if two types are compatible for overload resolution
fn types_compatible(expected: &Type, actual: &Type) -> bool {
    match (expected, actual) {
        // Exact match
        (a, b) if a == b => true,
        // Tensor matches any tensor
        (Type::Tensor(_, _), Type::Tensor(_, _)) => true,
        // Struct with same base name matches
        (Type::Struct(n1, _), Type::Struct(n2, _)) if n1 == n2 => true,
        // String variants
        (Type::String(_), Type::String(_)) => true,
        _ => false,
    }
}

use crate::compiler::builtin_loader::BuiltinTypeData;

/// Central manager for all types and their methods in the CodeGenerator.
pub struct TypeManager {
    types: HashMap<String, CodeGenType>,
    pub builtin_data: HashMap<String, BuiltinTypeData>,
}

impl TypeManager {
    pub fn new() -> Self {
        Self { 
            types: HashMap::new(),
            builtin_data: HashMap::new(),
        }
    }

    pub fn register_type(&mut self, type_obj: CodeGenType) {
        self.types.insert(type_obj.name.clone(), type_obj);
    }

    /// Register a builtin type defined in .tl (AST + Impls).
    pub fn register_builtin(&mut self, data: BuiltinTypeData) {
        self.builtin_data.insert(data.name.clone(), data);
    }

    pub fn get_type(&self, name: &str) -> Option<&CodeGenType> {
        self.types.get(name)
    }
    
    pub fn get_type_mut(&mut self, name: &str) -> Option<&mut CodeGenType> {
        self.types.get_mut(name)
    }
    
    pub fn ensure_type(&mut self, name: &str) -> &mut CodeGenType {
        self.types.entry(name.to_string()).or_insert_with(|| CodeGenType::new(name))
    }

    /// Get instance method signatures for a type by TL name
    pub fn get_instance_signatures_for_type(&self, type_name: &str, tl_name: &str) -> Vec<(&Vec<Type>, &Type)> {
        self.types.get(type_name)
            .map(|t| t.get_instance_signatures(tl_name))
            .unwrap_or_default()
    }

    /// Get static method signatures for a type by TL name
    pub fn get_static_signatures_for_type(&self, type_name: &str, tl_name: &str) -> Vec<(&Vec<Type>, &Type)> {
        self.types.get(type_name)
            .map(|t| t.get_static_signatures(tl_name))
            .unwrap_or_default()
    }

    /// Check if an instance method exists for a type
    pub fn has_instance_method(&self, type_name: &str, tl_name: &str) -> bool {
        self.types.get(type_name)
            .map(|t| t.has_instance_method(tl_name))
            .unwrap_or(false)
    }

    /// Check if a static method exists for a type
    pub fn has_static_method(&self, type_name: &str, tl_name: &str) -> bool {
        self.types.get(type_name)
            .map(|t| t.has_static_method(tl_name))
            .unwrap_or(false)
    }

    /// Find matching instance overload
    pub fn find_instance_overload<'a>(&'a self, type_name: &str, tl_name: &str, arg_types: &[Type]) -> Option<&'a InstanceOverload> {
        self.types.get(type_name)?.find_instance_overload(tl_name, arg_types)
    }

    /// Find matching static overload
    pub fn find_static_overload<'a>(&'a self, type_name: &str, tl_name: &str, arg_types: &[Type]) -> Option<&'a StaticOverload> {
        self.types.get(type_name)?.find_static_overload(tl_name, arg_types)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::codegen::CodeGenerator;
    use inkwell::values::BasicValueEnum;

    // Mock function matching InstanceMethodEval signature
    fn mock_method<'a, 'ctx>(
        _gen: &'a mut CodeGenerator<'ctx>,
        _val: BasicValueEnum<'ctx>,
        _ty: Type,
        _args: Vec<(BasicValueEnum<'ctx>, Type)>,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        Err("Mock method".into())
    }

    #[test]
    fn test_register_and_get_method() {
        let mut ty = CodeGenType::new("TestType");
        ty.register_evaluated_instance_method("test", mock_method, vec![], Type::I64);
        
        assert!(ty.has_instance_method("test"));
        let sigs = ty.get_instance_signatures("test");
        assert_eq!(sigs.len(), 1);
        assert_eq!(*sigs[0].1, Type::I64);
    }

    #[test]
    fn test_overloads() {
        let mut ty = CodeGenType::new("TestType");
        // Register two overloads for "sum"
        ty.register_evaluated_instance_method("sum", mock_method, vec![], Type::I64);
        ty.register_evaluated_instance_method("sum", mock_method, vec![Type::I64], Type::I64);
        ty.register_evaluated_instance_method("sum", mock_method, vec![Type::I64, Type::Bool], Type::I64);
        
        let sigs = ty.get_instance_signatures("sum");
        assert_eq!(sigs.len(), 3);
    }

    #[test]
    fn test_find_overload() {
        let mut ty = CodeGenType::new("Tensor");
        ty.register_evaluated_instance_method("sum", mock_method, vec![], Type::F32);
        ty.register_evaluated_instance_method("sum", mock_method, vec![Type::I64], Type::F32);
        
        // Find 0-arg overload
        let overload = ty.find_instance_overload("sum", &[]);
        assert!(overload.is_some());
        assert_eq!(overload.unwrap().arg_types.len(), 0);
        
        // Find 1-arg overload
        let overload = ty.find_instance_overload("sum", &[Type::I64]);
        assert!(overload.is_some());
        assert_eq!(overload.unwrap().arg_types.len(), 1);
        
        // Non-matching args
        let overload = ty.find_instance_overload("sum", &[Type::Bool]);
        assert!(overload.is_none());
    }

    #[test]
    fn test_type_manager() {
        let mut mgr = TypeManager::new();
        let mut ty = CodeGenType::new("Tensor");
        ty.register_evaluated_instance_method("sum", mock_method, vec![], Type::F32);
        mgr.register_type(ty);
        
        assert!(mgr.has_instance_method("Tensor", "sum"));
        let sigs = mgr.get_instance_signatures_for_type("Tensor", "sum");
        assert_eq!(sigs.len(), 1);
    }
}
