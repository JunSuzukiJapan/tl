use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::StructDef;

pub const SOURCE: &str = include_str!("vec.tl");

pub fn load_vec_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "Vec")
        .expect("Failed to load Vec type data")
}

pub fn get_vec_struct_def() -> StructDef {
    load_vec_data().struct_def.expect("Vec struct missing")
}
