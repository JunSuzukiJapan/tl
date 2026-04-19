use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};

pub fn load_channel_data() -> BuiltinTypeData {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/channel.tl");
    BuiltinLoader::load_builtin_type(&source, "Channel")
        .expect("Failed to load Channel type data")
}
