use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};

pub const SOURCE: &str = include_str!("hashmap.tl");

pub fn load_hashmap_data() -> BuiltinTypeData {
    let mut data = BuiltinLoader::load_builtin_type(SOURCE, "HashMap")
        .expect("Failed to load HashMap type data");
    data.destructor = Some("free".to_string());
    data
}

