use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::EnumDef;

pub fn load_result_data() -> BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/result.tl");
    BuiltinLoader::load_builtin_type(&source, "Result")
        .expect("Failed to load Result type data")
}

// Deprecated: Wraps load_result_data for legacy compatibility
pub fn get_result_enum_def() -> EnumDef {
    load_result_data().enum_def.expect("Result enum missing")
}
