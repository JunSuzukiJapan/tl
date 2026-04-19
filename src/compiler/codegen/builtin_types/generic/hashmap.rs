use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::StructDef;

pub fn load_hashmap_data() -> BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/hashmap.tl");
    BuiltinLoader::load_builtin_type(&source, "HashMap")
        .expect("Failed to load HashMap type data")
}

pub fn load_entry_data() -> BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/hashmap.tl");
    BuiltinLoader::load_builtin_type(&source, "Entry")
        .expect("Failed to load Entry type data")
}

pub fn get_hashmap_struct_def() -> StructDef {
    load_hashmap_data().struct_def.expect("HashMap struct missing")
}
