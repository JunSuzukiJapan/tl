use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};
use crate::compiler::ast::StructDef;

pub const SOURCE: &str = include_str!("vec_deque.tl");

pub fn load_vec_deque_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "VecDeque")
        .expect("Failed to load VecDeque type data")
}

pub fn get_vec_deque_struct_def() -> StructDef {
    load_vec_deque_data().struct_def.expect("VecDeque struct missing")
}
