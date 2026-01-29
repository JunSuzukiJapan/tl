use std::collections::HashMap;
use crate::compiler::codegen::expr::{StaticMethod, InstanceMethod};

/// Represents a type definition within the CodeGenerator, managing its methods.
pub struct CodeGenType {
    pub name: String,
    pub static_methods: HashMap<String, StaticMethod>,
    pub instance_methods: HashMap<String, InstanceMethod>,
}

impl CodeGenType {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            static_methods: HashMap::new(),
            instance_methods: HashMap::new(),
        }
    }

    pub fn register_static_method(&mut self, name: &str, method: StaticMethod) {
        self.static_methods.insert(name.to_string(), method);
    }
    
    pub fn register_instance_method(&mut self, name: &str, method: InstanceMethod) {
        self.instance_methods.insert(name.to_string(), method);
    }

    pub fn get_static_method(&self, name: &str) -> Option<&StaticMethod> {
        self.static_methods.get(name)
    }

    pub fn get_instance_method(&self, name: &str) -> Option<&InstanceMethod> {
        self.instance_methods.get(name)
    }
}

/// Central manager for all types and their methods in the CodeGenerator.
pub struct TypeManager {
    types: HashMap<String, CodeGenType>,
}

impl TypeManager {
    pub fn new() -> Self {
        Self { types: HashMap::new() }
    }

    pub fn register_type(&mut self, type_obj: CodeGenType) {
        self.types.insert(type_obj.name.clone(), type_obj);
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
