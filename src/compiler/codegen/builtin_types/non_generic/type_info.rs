use crate::compiler::codegen::type_manager::TypeManager;
use crate::compiler::builtin_loader::BuiltinLoader;

pub fn register_type_struct(type_manager: &mut TypeManager) {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("non_generic/type_info.tl");
    
    let builtin_data = BuiltinLoader::load_module_data(&source, "Type")
        .expect("Failed to load non-generic builtin Type");
    type_manager.register_builtin(builtin_data);
}
