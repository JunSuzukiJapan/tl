pub mod generic;
pub mod non_generic;
pub mod resolver;

// Re-export specific modules for easier access
pub use generic::option;
pub use generic::result;
pub use generic::vec;

pub use non_generic::io;
pub use non_generic::system;
pub use non_generic::llm;
pub use non_generic::tensor;
pub use non_generic::param;

use crate::compiler::codegen::CodeGenerator;

/// Load and register all built-in types into the CodeGenerator.
pub fn load_all_builtins(codegen: &mut CodeGenerator) {
    // 1. Load Generic Types (Option, Result)

    // Option
    let option_data = option::load_option_data();
    codegen.type_manager.register_builtin(option_data.clone());
    if let Some(def) = option_data.enum_def {
        codegen.enum_defs.insert(def.name.clone(), def);
    }
    codegen.generic_impls.entry("Option".to_string()).or_default().extend(option_data.impl_blocks);

    // Result
    let result_data = result::load_result_data();
    codegen.type_manager.register_builtin(result_data.clone());
    if let Some(def) = result_data.enum_def {
        codegen.enum_defs.insert(def.name.clone(), def);
    }
    codegen.generic_impls.entry("Result".to_string()).or_default().extend(result_data.impl_blocks);

    // Vec
    let vec_data = generic::vec::load_vec_data();
    codegen.type_manager.register_builtin(vec_data.clone());
    if let Some(def) = vec_data.struct_def {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    codegen.generic_impls.entry("Vec".to_string()).or_default().extend(vec_data.impl_blocks);

    // 2. Register Non-Generic Types (IO, System, LLM, Tensor, Param)
    // These register directly into TypeManager
    io::register_io_types(&mut codegen.type_manager);
    system::register_system_types(&mut codegen.type_manager);
    llm::register_llm_types(&mut codegen.type_manager);
    tensor::register_tensor_types(&mut codegen.type_manager);
    param::register_param_types(&mut codegen.type_manager);
    
    // 3. Register Hardcoded Enums (Device)
    // Used by runtime for Tensor device selection
    let device_enum = crate::compiler::ast::EnumDef {
        name: "Device".to_string(),
        generics: vec![],
        variants: vec![
            crate::compiler::ast::VariantDef {
                name: "Auto".to_string(),
                kind: crate::compiler::ast::VariantKind::Unit,
            },
            crate::compiler::ast::VariantDef {
                name: "Cpu".to_string(),
                kind: crate::compiler::ast::VariantKind::Unit,
            },
            crate::compiler::ast::VariantDef {
                name: "Cuda".to_string(),
                kind: crate::compiler::ast::VariantKind::Unit,
            },
        ],
    };
    codegen.enum_defs.insert(device_enum.name.clone(), device_enum);
}
