use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::codegen::type_manager::TypeManager;

pub const SOURCE: &str = include_str!("result.tl");

pub fn load_result_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "Result")
        .expect("Failed to load Result type data")
}
