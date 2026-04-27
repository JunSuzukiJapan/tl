use crate::compiler::builtin_loader::BuiltinLoader;

pub fn load_npy_data() -> Vec<crate::compiler::builtin_loader::BuiltinTypeData> {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("non_generic/npy.tl");
    vec![
        BuiltinLoader::load_module_data(&source, "NpyFile")
            .expect("Failed to load NpyFile"),
        BuiltinLoader::load_module_data(&source, "NpyHeader")
            .expect("Failed to load NpyHeader"),
    ]
}
