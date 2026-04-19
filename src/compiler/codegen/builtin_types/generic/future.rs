// Future trait and Poll enum source
pub fn get_source() -> String {
    crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/future.tl")
}
