use crate::compiler::builtin_loader::BuiltinLoader;


pub const SOURCE_I64: &str = include_str!("atomic_i64.tl");
pub const SOURCE_I32: &str = include_str!("atomic_i32.tl");

pub fn load_atomic_i64() -> crate::compiler::builtin_loader::BuiltinTypeData {
    BuiltinLoader::load_module_data(SOURCE_I64, "AtomicI64")
        .expect("Failed to load non-generic builtin AtomicI64")
}

pub fn load_atomic_i32() -> crate::compiler::builtin_loader::BuiltinTypeData {
    BuiltinLoader::load_module_data(SOURCE_I32, "AtomicI32")
        .expect("Failed to load non-generic builtin AtomicI32")
}
