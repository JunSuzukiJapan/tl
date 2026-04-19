use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::StructDef;

pub fn load_string_builder_data() -> BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/string_builder.tl");
    BuiltinLoader::load_builtin_type(&source, "StringBuilder")
        .expect("Failed to load StringBuilder type data")
}

pub fn get_string_builder_struct_def() -> StructDef {
    load_string_builder_data().struct_def.expect("StringBuilder struct missing")
}
