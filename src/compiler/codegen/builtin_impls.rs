use crate::compiler::ast::*;
use crate::compiler::builtin_ast;

use std::collections::HashMap;

pub fn register_builtin_impls(generic_impls: &mut HashMap<String, Vec<ImplBlock>>) {
    // Load from AST
    for imp in builtin_ast::load_builtin_impls() {
         let name = match &imp.target_type {
             Type::UserDefined(n, _) => n.clone(),
             Type::Struct(n, _) => n.clone(),
             Type::Enum(n, _) => n.clone(),
             Type::Vec(_) => "Vec".to_string(), // Fallback if target is Type::Vec
             _ => "Unknown".to_string(),
         };
         generic_impls.entry(name).or_default().push(imp);
    }

    let option_impl = create_option_impl();
    generic_impls.insert("Option".to_string(), vec![option_impl]);
}

/// Register built-in struct definitions
pub fn register_builtin_structs(struct_defs: &mut HashMap<String, StructDef>) {
    // Load from AST
    for def in builtin_ast::load_builtin_structs() {
        struct_defs.insert(def.name.clone(), def);
    }
}



fn create_option_impl() -> ImplBlock {
    let t_type = Type::UserDefined("T".to_string(), vec![]);

    ImplBlock {
        target_type: Type::UserDefined("Option".to_string(), vec![t_type.clone()]),
        generics: vec!["T".to_string()],
        methods: vec![],
    }
}
