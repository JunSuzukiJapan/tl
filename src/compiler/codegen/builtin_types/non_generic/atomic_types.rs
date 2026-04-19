use crate::compiler::builtin_loader::BuiltinLoader;


pub fn load_atomic_i64() -> crate::compiler::builtin_loader::BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("non_generic/atomic_i64.tl");
    BuiltinLoader::load_module_data(&source, "AtomicI64")
        .expect("Failed to load non-generic builtin AtomicI64")
}

pub fn load_atomic_i32() -> crate::compiler::builtin_loader::BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("non_generic/atomic_i32.tl");
    BuiltinLoader::load_module_data(&source, "AtomicI32")
        .expect("Failed to load non-generic builtin AtomicI32")
}
