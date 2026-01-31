use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};

pub const SOURCE: &str = include_str!("vec.tl");

pub fn load_vec_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "Vec")
        .expect("Failed to load Vec type data")
}
