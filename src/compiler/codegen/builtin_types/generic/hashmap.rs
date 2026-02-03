use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::StructDef;

pub const SOURCE: &str = include_str!("hashmap.tl");

pub fn load_hashmap_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "HashMap")
        .expect("Failed to load HashMap type data")
}

pub fn load_entry_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "Entry")
        .expect("Failed to load Entry type data")
}

pub fn get_hashmap_struct_def() -> StructDef {
    load_hashmap_data().struct_def.expect("HashMap struct missing")
}
