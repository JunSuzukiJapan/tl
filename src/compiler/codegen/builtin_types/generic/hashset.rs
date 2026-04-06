use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::StructDef;

pub const SOURCE: &str = include_str!("hashset.tl");

pub fn load_hashset_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "HashSet")
        .expect("Failed to load HashSet type data")
}

pub fn load_set_entry_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "SetEntry")
        .expect("Failed to load SetEntry type data")
}

pub fn get_hashset_struct_def() -> StructDef {
    load_hashset_data().struct_def.expect("HashSet struct missing")
}
