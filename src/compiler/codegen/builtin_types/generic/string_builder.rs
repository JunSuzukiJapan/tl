use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::StructDef;

pub const SOURCE: &str = include_str!("string_builder.tl");

pub fn load_string_builder_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "StringBuilder")
        .expect("Failed to load StringBuilder type data")
}

pub fn get_string_builder_struct_def() -> StructDef {
    load_string_builder_data().struct_def.expect("StringBuilder struct missing")
}
