use crate::compiler::builtin_loader::BuiltinLoader;

pub fn load_safetensors_data() -> Vec<crate::compiler::builtin_loader::BuiltinTypeData> {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("non_generic/safetensors.tl");
    vec![
        BuiltinLoader::load_module_data(&source, "SafeTensorsFile")
            .expect("Failed to load SafeTensorsFile"),
        BuiltinLoader::load_module_data(&source, "SafeTensorInfo")
            .expect("Failed to load SafeTensorInfo"),
    ]
}
