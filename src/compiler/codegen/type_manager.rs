use std::collections::HashMap;
pub use crate::compiler::codegen::expr::{
    StaticMethod, InstanceMethod, StaticMethodEval, StaticMethodUneval, InstanceMethodEval, InstanceMethodUneval
};

/// Represents a type definition within the CodeGenerator, managing its methods.
pub struct CodeGenType {
    pub name: String,
    // Store (Method, ArgTypes, ReturnType)
    pub static_methods: HashMap<String, (StaticMethod, Vec<crate::compiler::ast::Type>, crate::compiler::ast::Type)>,
    pub instance_methods: HashMap<String, (InstanceMethod, Vec<crate::compiler::ast::Type>, crate::compiler::ast::Type)>,
}

impl CodeGenType {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            static_methods: HashMap::new(),
            instance_methods: HashMap::new(),
        }
    }

    pub fn register_evaluated_static_method(&mut self, name: &str, method: crate::compiler::codegen::type_manager::StaticMethodEval, args: Vec<crate::compiler::ast::Type>, ret: crate::compiler::ast::Type) {
        self.static_methods.insert(name.to_string(), (StaticMethod::Evaluated(method), args, ret));
    }

    pub fn register_unevaluated_static_method(&mut self, name: &str, method: crate::compiler::codegen::type_manager::StaticMethodUneval, args: Vec<crate::compiler::ast::Type>, ret: crate::compiler::ast::Type) {
        self.static_methods.insert(name.to_string(), (StaticMethod::Unevaluated(method), args, ret));
    }

    pub fn register_evaluated_instance_method(&mut self, name: &str, method: crate::compiler::codegen::type_manager::InstanceMethodEval, args: Vec<crate::compiler::ast::Type>, ret: crate::compiler::ast::Type) {
        self.instance_methods.insert(name.to_string(), (InstanceMethod::Evaluated(method), args, ret));
    }

    pub fn register_unevaluated_instance_method(&mut self, name: &str, method: crate::compiler::codegen::type_manager::InstanceMethodUneval, args: Vec<crate::compiler::ast::Type>, ret: crate::compiler::ast::Type) {
        self.instance_methods.insert(name.to_string(), (InstanceMethod::Unevaluated(method), args, ret));
    }

    pub fn get_static_method(&self, name: &str) -> Option<&StaticMethod> {
        self.static_methods.get(name).map(|(m, _, _)| m)
    }

    pub fn get_instance_method(&self, name: &str) -> Option<&InstanceMethod> {
        self.instance_methods.get(name).map(|(m, _, _)| m)
    }

    pub fn get_static_signature(&self, name: &str) -> Option<(&Vec<crate::compiler::ast::Type>, &crate::compiler::ast::Type)> {
        self.static_methods.get(name).map(|(_, args, ret)| (args, ret))
    }

    pub fn get_instance_signature(&self, name: &str) -> Option<(&Vec<crate::compiler::ast::Type>, &crate::compiler::ast::Type)> {
        self.instance_methods.get(name).map(|(_, args, ret)| (args, ret))
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
    /// This stores the data to be used by CodeGenerator for AST injection.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::codegen::CodeGenerator;
    use crate::compiler::ast::Type;
    use inkwell::values::BasicValueEnum;
    use crate::compiler::codegen::expr::InstanceMethod;

    // Mock function matching InstanceMethodEval signature
    fn mock_method<'a, 'ctx>(
        _gen: &'a mut CodeGenerator<'ctx>,
        _val: BasicValueEnum<'ctx>,
        _ty: Type,
        _args: Vec<(BasicValueEnum<'ctx>, Type)>,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        Err("Mock called".to_string())
    }

    #[test]
    fn test_codegen_type_methods() {
        let mut ty = CodeGenType::new("TestType");
        assert_eq!(ty.name, "TestType");
        
        // Test registering instance method
        ty.register_evaluated_instance_method("test_method", mock_method, vec![], Type::Void);
        
        let method = ty.get_instance_method("test_method");
        assert!(method.is_some());
        
        if let Some(InstanceMethod::Evaluated(_)) = method {
             // Correctly retrieved
        } else {
             panic!("Expected Evaluated instance method");
        }

        // Test missing method
        assert!(ty.get_instance_method("non_existent").is_none());
    }

    #[test]
    fn test_type_manager_lifecycle() {
        let mut tm = TypeManager::new();
        
        // Test ensure_type (creation)
        let ty_ref = tm.ensure_type("NewType");
        assert_eq!(ty_ref.name, "NewType");
        
        // Test retrieval
        let ty_opt = tm.get_type("NewType");
        assert!(ty_opt.is_some());
        assert_eq!(ty_opt.unwrap().name, "NewType");
        
        // Test ensure_type (existing)
        let ty_ref2 = tm.ensure_type("NewType");
        assert_eq!(ty_ref2.name, "NewType");
        
        // Register explicit type
        let mut explicit_ty = CodeGenType::new("Explicit");
        explicit_ty.register_evaluated_instance_method("foo", mock_method, vec![], Type::Void);
        tm.register_type(explicit_ty);
        
        let retrieved = tm.get_type("Explicit").unwrap();
        assert!(retrieved.get_instance_method("foo").is_some());
    }
}
