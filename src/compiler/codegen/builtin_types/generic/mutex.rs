use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};

pub const SOURCE: &str = include_str!("mutex.tl");

pub fn load_mutex_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "Mutex")
        .expect("Failed to load Mutex type data")
}
