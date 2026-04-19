// Standard trait definitions for TL language
pub fn get_source() -> String {
    crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("generic/traits.tl")
}
