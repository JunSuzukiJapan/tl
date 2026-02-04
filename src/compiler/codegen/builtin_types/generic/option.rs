use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::EnumDef;

// Updated 2026-02-04: Added abort() to unwrap
pub const SOURCE: &str = include_str!("option.tl");

pub fn load_option_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "Option")
        .expect("Failed to load Option type data")
}

// Deprecated: Wraps load_option_data for legacy compatibility
pub fn get_option_enum_def() -> EnumDef {
    load_option_data().enum_def.expect("Option enum missing")
}

