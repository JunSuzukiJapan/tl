use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::EnumDef;

pub const SOURCE: &str = include_str!("result.tl");

pub fn load_result_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "Result")
        .expect("Failed to load Result type data")
}

// Deprecated: Wraps load_result_data for legacy compatibility
pub fn get_result_enum_def() -> EnumDef {
    load_result_data().enum_def.expect("Result enum missing")
}
