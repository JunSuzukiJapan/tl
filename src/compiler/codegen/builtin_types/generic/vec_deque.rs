use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::StructDef;

pub fn load_vec_deque_data() -> BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/vec_deque.tl");
    BuiltinLoader::load_builtin_type(&source, "VecDeque")
        .expect("Failed to load VecDeque type data")
}

pub fn get_vec_deque_struct_def() -> StructDef {
    load_vec_deque_data().struct_def.expect("VecDeque struct missing")
}
