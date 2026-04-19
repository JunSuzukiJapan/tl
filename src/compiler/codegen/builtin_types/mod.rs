pub mod generic;
pub mod non_generic;
pub mod resolver;
pub mod assets;

// Re-export specific modules for easier access
pub use generic::option;
pub use generic::result;
pub use generic::vec;
pub use generic::hashmap;
pub use generic::hashset;
pub use generic::vec_deque;
pub use generic::btreemap;
pub use generic::string_builder;
pub use generic::traits;
pub use generic::future;
pub use generic::mutex;

pub use non_generic::io;
pub use non_generic::system;
pub use non_generic::llm;
pub use non_generic::tensor;
pub use non_generic::param;
pub use non_generic::regex;
pub use non_generic::type_info;

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

    // HashMap
    let hashmap_data = generic::hashmap::load_hashmap_data();
    codegen.type_manager.register_builtin(hashmap_data.clone());
    if let Some(def) = hashmap_data.struct_def {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    codegen.generic_impls.entry("HashMap".to_string()).or_default().extend(hashmap_data.impl_blocks);
    
    // Entry
    let entry_data = generic::hashmap::load_entry_data();
    codegen.type_manager.register_builtin(entry_data.clone());
    if let Some(def) = entry_data.enum_def {
        codegen.enum_defs.insert(def.name.clone(), def);
    }
    codegen.generic_impls.entry("Entry".to_string()).or_default().extend(entry_data.impl_blocks);

    // HashSet
    let hashset_data = generic::hashset::load_hashset_data();
    codegen.type_manager.register_builtin(hashset_data.clone());
    if let Some(def) = hashset_data.struct_def {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    codegen.generic_impls.entry("HashSet".to_string()).or_default().extend(hashset_data.impl_blocks);
    
    // SetEntry
    let set_entry_data = generic::hashset::load_set_entry_data();
    codegen.type_manager.register_builtin(set_entry_data.clone());
    if let Some(def) = set_entry_data.enum_def {
        codegen.enum_defs.insert(def.name.clone(), def);
    }
    codegen.generic_impls.entry("SetEntry".to_string()).or_default().extend(set_entry_data.impl_blocks);

    // VecDeque
    let vec_deque_data = generic::vec_deque::load_vec_deque_data();
    codegen.type_manager.register_builtin(vec_deque_data.clone());
    if let Some(def) = vec_deque_data.struct_def {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    codegen.generic_impls.entry("VecDeque".to_string()).or_default().extend(vec_deque_data.impl_blocks);

    // BTreeMap
    let btreemap_data = generic::btreemap::load_btreemap_data();
    codegen.type_manager.register_builtin(btreemap_data.clone());
    if let Some(def) = btreemap_data.struct_def {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    codegen.generic_impls.entry("BTreeMap".to_string()).or_default().extend(btreemap_data.impl_blocks);

    // BTreeNode
    let btree_node_data = generic::btreemap::load_btree_node_data();
    codegen.type_manager.register_builtin(btree_node_data.clone());
    if let Some(def) = btree_node_data.struct_def {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    codegen.generic_impls.entry("BTreeNode".to_string()).or_default().extend(btree_node_data.impl_blocks);

    // StringBuilder
    let str_builder_data = generic::string_builder::load_string_builder_data();
    codegen.type_manager.register_builtin(str_builder_data.clone());
    if let Some(def) = str_builder_data.struct_def {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    codegen.generic_impls.entry("StringBuilder".to_string()).or_default().extend(str_builder_data.impl_blocks);

    // Mutex
    let mutex_data = generic::mutex::load_mutex_data();
    codegen.type_manager.register_builtin(mutex_data.clone());
    if let Some(def) = mutex_data.struct_def {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    codegen.generic_impls.entry("Mutex".to_string()).or_default().extend(mutex_data.impl_blocks);

    // Channel
    let channel_data = generic::channel::load_channel_data();
    codegen.type_manager.register_builtin(channel_data.clone());
    if let Some(def) = channel_data.struct_def {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    codegen.generic_impls.entry("Channel".to_string()).or_default().extend(channel_data.impl_blocks);

    // 2. Register Non-Generic Types (IO, System, LLM, Tensor, Param, Primitives)
    // These register directly into TypeManager
    non_generic::primitives::register_primitive_types(&mut codegen.type_manager);
    io::register_io_types(&mut codegen.type_manager);
    system::register_system_types(&mut codegen.type_manager);
    non_generic::regex::register_regex_types(&mut codegen.type_manager);
    non_generic::type_info::register_type_struct(&mut codegen.type_manager);
    let i64_data = non_generic::atomic_types::load_atomic_i64();
    codegen.type_manager.register_builtin(i64_data.clone());
    if let Some(def) = i64_data.struct_def.clone() { codegen.struct_defs.insert(def.name.clone(), def); }
    codegen.generic_impls.entry("AtomicI64".to_string()).or_default().extend(i64_data.impl_blocks);
    
    let i32_data = non_generic::atomic_types::load_atomic_i32();
    codegen.type_manager.register_builtin(i32_data.clone());
    if let Some(def) = i32_data.struct_def.clone() { codegen.struct_defs.insert(def.name.clone(), def); }
    codegen.generic_impls.entry("AtomicI32".to_string()).or_default().extend(i32_data.impl_blocks);

    // Duration
    let duration_data = non_generic::time_types::load_duration();
    codegen.type_manager.register_builtin(duration_data.clone());
    for def in duration_data.extra_structs {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    if let Some(def) = duration_data.struct_def.clone() { codegen.struct_defs.insert(def.name.clone(), def); }
    codegen.generic_impls.entry("Duration".to_string()).or_default().extend(duration_data.impl_blocks);

    // Instant
    let instant_data = non_generic::time_types::load_instant();
    codegen.type_manager.register_builtin(instant_data.clone());
    for def in instant_data.extra_structs {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    if let Some(def) = instant_data.struct_def.clone() { codegen.struct_defs.insert(def.name.clone(), def); }
    codegen.generic_impls.entry("Instant".to_string()).or_default().extend(instant_data.impl_blocks);

    // DateTime
    let datetime_data = non_generic::time_types::load_datetime();
    codegen.type_manager.register_builtin(datetime_data.clone());
    for def in datetime_data.extra_structs {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    if let Some(def) = datetime_data.struct_def.clone() { codegen.struct_defs.insert(def.name.clone(), def); }
    codegen.generic_impls.entry("DateTime".to_string()).or_default().extend(datetime_data.impl_blocks);

    // Net
    let (listener_data, stream_data) = non_generic::net::load_net_data();
    
    codegen.type_manager.register_builtin(listener_data.clone());
    for def in listener_data.extra_structs {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    if let Some(def) = listener_data.struct_def.clone() { codegen.struct_defs.insert(def.name.clone(), def); }
    codegen.generic_impls.entry("TcpListener".to_string()).or_default().extend(listener_data.impl_blocks.clone());

    codegen.type_manager.register_builtin(stream_data.clone());
    for def in stream_data.extra_structs {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    if let Some(def) = stream_data.struct_def.clone() { codegen.struct_defs.insert(def.name.clone(), def); }
    codegen.generic_impls.entry("TcpStream".to_string()).or_default().extend(stream_data.impl_blocks.clone());

    // Thread is now fully generic and natively evaluated in expr.rs
    // Register LLM Structs (from source)
    let llm_data = llm::load_llm_data();
    for def in llm_data.extra_structs {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    // Also check struct_def (though current load_module_data returns None for it)
    if let Some(def) = llm_data.struct_def {
        codegen.struct_defs.insert(def.name.clone(), def);
    }
    
    // Register methods
    llm::register_llm_types(&mut codegen.type_manager);
    tensor::register_tensor_types(&mut codegen.type_manager);
    param::register_param_types(&mut codegen.type_manager);
    
    // 3. Register Hardcoded Enums (Device)
    // Used by runtime for Tensor device selection
    let device_enum = crate::compiler::ast::EnumDef {
        name: "Device".to_string(),
        generics: vec![],
        is_pub: false,
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
