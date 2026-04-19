use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};

pub fn load_mutex_data() -> BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/mutex.tl");
    BuiltinLoader::load_builtin_type(&source, "Mutex")
        .expect("Failed to load Mutex type data")
}
