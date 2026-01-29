use crate::compiler::ast::*;
use crate::compiler::codegen::builtin_types;

use std::collections::HashMap;

pub fn register_builtin_impls(_generic_impls: &mut HashMap<String, Vec<ImplBlock>>) {
    // Legacy generic impls are removed.
    // Methods are now registered via TypeManager.
}

/// Register built-in struct definitions
pub fn register_builtin_structs(struct_defs: &mut HashMap<String, StructDef>) {
    let defs = vec![
        builtin_types::vec::get_vec_struct_def(),
        builtin_types::hashmap::get_hashmap_struct_def(),
    ];
    
    for def in defs {
        struct_defs.insert(def.name.clone(), def);
    }
}
