use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::StructDef;

pub fn load_vec_data() -> BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/vec.tl");
    BuiltinLoader::load_builtin_type(&source, "Vec")
        .expect("Failed to load Vec type data")
}

pub fn get_vec_struct_def() -> StructDef {
    load_vec_data().struct_def.expect("Vec struct missing")
}
