use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
// use crate::compiler::codegen::expr;

pub const SOURCE: &str = include_str!("string.tl");

pub fn register_string_types(manager: &mut TypeManager) {
    let string_type = CodeGenType::new("String");
    // Currently String has no methods exposed directly via TypeManager.
    // Most String operations are built-in operators or stdlib functions.
    // We register it to ensure it is recognized as a valid type managed by the system.
    manager.register_type(string_type);
}
