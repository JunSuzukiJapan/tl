use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
use crate::compiler::codegen::expr;

pub fn register_llm_types(manager: &mut TypeManager) {
    // Register Tokenizer
    let mut tokenizer = CodeGenType::new("Tokenizer");
    tokenizer.register_static_method("new", expr::StaticMethod::Evaluated(expr::compile_tokenizer_new));
    tokenizer.register_instance_method("encode", expr::InstanceMethod::Evaluated(expr::compile_tokenizer_encode));
    tokenizer.register_instance_method("decode", expr::InstanceMethod::Evaluated(expr::compile_tokenizer_decode));
    manager.register_type(tokenizer);

    // Register KVCache
    let mut kv_cache = CodeGenType::new("KVCache");
    kv_cache.register_static_method("new", expr::StaticMethod::Evaluated(expr::compile_kv_cache_new));
    kv_cache.register_instance_method("free", expr::InstanceMethod::Evaluated(expr::compile_kv_cache_free));
    kv_cache.register_instance_method("get_k", expr::InstanceMethod::Evaluated(expr::compile_kv_cache_get_k));
    kv_cache.register_instance_method("get_v", expr::InstanceMethod::Evaluated(expr::compile_kv_cache_get_v));
    manager.register_type(kv_cache);
}
