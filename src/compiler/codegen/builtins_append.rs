    // tl_tensor_save(path: *const i8, t: *mut OpaqueTensor) -> void
    let save_type = void_type.fn_type(&[i8_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_tensor_save", save_type, None);

    // tl_tensor_load(path: *const i8) -> *mut OpaqueTensor
    let load_type = void_ptr.fn_type(&[i8_ptr.into()], false);
    module.add_function("tl_tensor_load", load_type, None);

    // Mappings
    if let Some(f) = module.get_function("tl_tensor_save") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_save as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_load") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_load as usize);
    }

    // Return types
    fn_return_types.insert("tl_tensor_save".to_string(), Type::Void);
    fn_return_types.insert("tl_tensor_load".to_string(), tensor_type.clone());
