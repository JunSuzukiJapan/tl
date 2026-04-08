use crate::compiler::builtin_loader::BuiltinLoader;

pub const SOURCE_DURATION: &str = include_str!("duration.tl");
pub const SOURCE_INSTANT: &str = include_str!("instant.tl");
pub const SOURCE_DATETIME: &str = include_str!("datetime.tl");

pub fn load_duration() -> crate::compiler::builtin_loader::BuiltinTypeData {
    BuiltinLoader::load_module_data(SOURCE_DURATION, "Duration")
        .expect("Failed to load non-generic builtin Duration")
}

pub fn load_instant() -> crate::compiler::builtin_loader::BuiltinTypeData {
    BuiltinLoader::load_module_data(SOURCE_INSTANT, "Instant")
        .expect("Failed to load non-generic builtin Instant")
}

pub fn load_datetime() -> crate::compiler::builtin_loader::BuiltinTypeData {
    BuiltinLoader::load_module_data(SOURCE_DATETIME, "DateTime")
        .expect("Failed to load non-generic builtin DateTime")
}
