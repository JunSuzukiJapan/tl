use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::StructDef;

pub const SOURCE: &str = include_str!("btreemap.tl");

pub fn load_btreemap_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "BTreeMap")
        .expect("Failed to load BTreeMap type data")
}

pub fn load_btree_node_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "BTreeNode")
        .expect("Failed to load BTreeNode type data")
}

pub fn get_btreemap_struct_def() -> StructDef {
    load_btreemap_data().struct_def.expect("BTreeMap struct missing")
}
