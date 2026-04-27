use crate::compiler::builtin_loader::BuiltinLoader;

pub fn load_npz_data() -> Vec<crate::compiler::builtin_loader::BuiltinTypeData> {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("non_generic/npz.tl");
    vec![
        BuiltinLoader::load_module_data(&source, "NpzFile")
            .expect("Failed to load NpzFile"),
        BuiltinLoader::load_module_data(&source, "NpzEntry")
            .expect("Failed to load NpzEntry"),
    ]
}
