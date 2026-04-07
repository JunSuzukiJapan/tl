pub const SOURCE: &str = include_str!("type_info.tl");
use crate::compiler::codegen::type_manager::TypeManager;
use crate::compiler::builtin_loader::BuiltinLoader;

pub fn register_type_struct(type_manager: &mut TypeManager) {
    let source = SOURCE;
    
    let builtin_data = BuiltinLoader::load_module_data(source, "Type")
        .expect("Failed to load non-generic builtin Type");
    type_manager.register_builtin(builtin_data);
}
