
use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};

pub const SOURCE: &str = include_str!("primitives.tl");

pub fn register_primitive_types(manager: &mut TypeManager) {
    // Register types for CodeGen to recognize them (though they are built-ins)
    // We don't register custom codegen methods here; we rely on runtime functions or mono wrappers.
    // However, we might want to register them to allow "Type::F32" to be looked up by name if needed.
    
    let f32_type = CodeGenType::new("F32");
    manager.register_type(f32_type);

    let f64_type = CodeGenType::new("F64");
    manager.register_type(f64_type);

    let i64_type = CodeGenType::new("I64");
    manager.register_type(i64_type);
    
    let i32_type = CodeGenType::new("I32");
    manager.register_type(i32_type);
    
    let bool_type = CodeGenType::new("Bool");
    manager.register_type(bool_type);
}
