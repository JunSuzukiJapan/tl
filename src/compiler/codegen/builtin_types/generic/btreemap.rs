use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::StructDef;

pub fn load_btreemap_data() -> BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/btreemap.tl");
    BuiltinLoader::load_builtin_type(&source, "BTreeMap")
        .expect("Failed to load BTreeMap type data")
}

pub fn load_btree_node_data() -> BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/btreemap.tl");
    BuiltinLoader::load_builtin_type(&source, "BTreeNode")
        .expect("Failed to load BTreeNode type data")
}

pub fn get_btreemap_struct_def() -> StructDef {
    load_btreemap_data().struct_def.expect("BTreeMap struct missing")
}
