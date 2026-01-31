use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
// use crate::compiler::codegen::expr;

pub const SOURCE: &str = include_str!("string.tl");

pub fn load_string_data() -> crate::compiler::builtin_loader::BuiltinTypeData {
    crate::compiler::builtin_loader::BuiltinLoader::load_builtin_type(SOURCE, "String")
        .expect("Failed to load String type data")
}

pub fn register_string_types(manager: &mut TypeManager) {
    let string_type = CodeGenType::new("String");
    manager.register_type(string_type);
    
    // Also register builtins?
    let data = load_string_data();
    manager.register_builtin(data);
}
