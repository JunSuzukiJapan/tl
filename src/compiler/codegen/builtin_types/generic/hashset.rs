use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::StructDef;

pub fn load_hashset_data() -> BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/hashset.tl");
    BuiltinLoader::load_builtin_type(&source, "HashSet")
        .expect("Failed to load HashSet type data")
}

pub fn load_set_entry_data() -> BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/hashset.tl");
    BuiltinLoader::load_builtin_type(&source, "SetEntry")
        .expect("Failed to load SetEntry type data")
}

pub fn get_hashset_struct_def() -> StructDef {
    load_hashset_data().struct_def.expect("HashSet struct missing")
}
