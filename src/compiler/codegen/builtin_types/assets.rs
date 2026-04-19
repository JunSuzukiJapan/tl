use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "src/compiler/codegen/builtin_types/"]
pub struct BuiltinAssets;

impl BuiltinAssets {
    /// 実行時にロードされた仮想ファイルシステムからTLソースコードを取得する
    pub fn get_source(path: &str) -> String {
        let file = Self::get(path)
            .unwrap_or_else(|| panic!("Builtin file '{}' not found in assets", path));
        std::str::from_utf8(file.data.as_ref()).unwrap().to_string()
    }
}
