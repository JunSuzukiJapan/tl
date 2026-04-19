use crate::compiler::builtin_loader::BuiltinLoader;

pub fn load_duration() -> crate::compiler::builtin_loader::BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("non_generic/duration.tl");
    BuiltinLoader::load_module_data(&source, "Duration")
        .expect("Failed to load non-generic builtin Duration")
}

pub fn load_instant() -> crate::compiler::builtin_loader::BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("non_generic/instant.tl");
    BuiltinLoader::load_module_data(&source, "Instant")
        .expect("Failed to load non-generic builtin Instant")
}

pub fn load_datetime() -> crate::compiler::builtin_loader::BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("non_generic/datetime.tl");
    BuiltinLoader::load_module_data(&source, "DateTime")
        .expect("Failed to load non-generic builtin DateTime")
}
