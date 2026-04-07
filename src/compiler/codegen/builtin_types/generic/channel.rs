use crate::compiler::builtin_loader::{BuiltinLoader, BuiltinTypeData};

pub const SOURCE: &str = include_str!("channel.tl");

pub fn load_channel_data() -> BuiltinTypeData {
    BuiltinLoader::load_builtin_type(SOURCE, "Channel")
        .expect("Failed to load Channel type data")
}
