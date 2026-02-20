// src/compiler/mangler.rs
//! マングル名の生成・解析を一元管理するモジュール。
//! 型引数の区切り文字やフォーマットを `Mangler` 構造体に集約し、
//! ハードコードされたリテラルの散在を防ぐ。

use crate::compiler::ast::Type;

/// マングル名の生成・解析を担当する構造体。
/// 区切り文字（デリミタ）を内部に保持し、全てのマングル操作を提供する。
pub struct Mangler {
    /// 型引数の開始デリミタ（例: `[`）
    pub open: &'static str,
    /// 型引数の終了デリミタ（例: `]`）
    pub close: &'static str,
    /// 開始デリミタのバイト値（バイト走査用）
    open_byte: u8,
    /// 終了デリミタのバイト値（バイト走査用）
    close_byte: u8,
}

impl Mangler {
    /// デフォルトのデリミタ `[` / `]` で Mangler を生成する。
    pub const fn new() -> Self {
        Self {
            open: "[",
            close: "]",
            open_byte: b'[',
            close_byte: b']',
        }
    }

    /// 型引数を角括弧で囲んでマングル名を生成する。
    ///
    /// # Examples
    /// ```text
    /// wrap_args("Vec", &["i64"]) → "Vec[i64]"
    /// wrap_args("HashMap", &["i64", "String"]) → "HashMap[i64][String]"
    /// ```
    pub fn wrap_args(&self, base: &str, args: &[String]) -> String {
        if args.is_empty() {
            return base.to_string();
        }
        let suffix: String = args
            .iter()
            .map(|a| format!("{}{}{}", self.open, a, self.close))
            .collect();
        format!("{}{}", base, suffix)
    }

    /// マングル名からベース名を抽出する。
    ///
    /// # Examples
    /// ```text
    /// base_name("Vec[i64]") → "Vec"
    /// base_name("HashMap[i64][String]") → "HashMap"
    /// base_name("plain") → "plain"
    /// ```
    pub fn base_name<'a>(&self, mangled: &'a str) -> &'a str {
        mangled.split(self.open).next().unwrap_or(mangled)
    }

    /// マングル名が型引数を含むかを判定する。
    pub fn has_args(&self, mangled: &str) -> bool {
        mangled.contains(self.open)
    }

    /// マングル名からトップレベルの型引数文字列を抽出する（ネスト対応）。
    ///
    /// # Examples
    /// ```text
    /// extract_args("Vec[i64]") → vec!["i64"]
    /// extract_args("HashMap[i64][String]") → vec!["i64", "String"]
    /// extract_args("Vec[Entry[i64][String]]") → vec!["Entry[i64][String]"]
    /// ```
    pub fn extract_args<'a>(&self, mangled: &'a str) -> Vec<&'a str> {
        let mut args = Vec::new();
        let base_end = match mangled.find(self.open) {
            Some(pos) => pos,
            None => return args,
        };
        let rest = &mangled[base_end..];
        let bytes = rest.as_bytes();
        let mut depth = 0;
        let mut start = 0;
        for (i, &b) in bytes.iter().enumerate() {
            if b == self.open_byte {
                if depth == 0 {
                    start = i + 1;
                }
                depth += 1;
            } else if b == self.close_byte {
                depth -= 1;
                if depth == 0 {
                    args.push(&rest[start..i]);
                }
            }
        }
        args
    }

    /// マングル型文字列を `Type` にパースする（再帰対応）。
    ///
    /// # Examples
    /// ```text
    /// parse_type_str("i64") → Type::I64
    /// parse_type_str("Pair[i64][i64]") → Type::Struct("Pair[i64][i64]", [I64, I64])
    /// ```
    pub fn parse_type_str(&self, s: &str) -> Type {
        match s.to_lowercase().as_str() {
            "i64" => Type::I64,
            "i32" => Type::I32,
            "f32" => Type::F32,
            "f64" => Type::F64,
            "bool" => Type::Bool,
            "u8" => Type::U8,
            "usize" => Type::Usize,
            "string" => Type::String("String".to_string()),
            "void" => Type::Void,
            _ => {
                if self.has_args(s) {
                    let inner_strs = self.extract_args(s);
                    let inner_types: Vec<Type> =
                        inner_strs.iter().map(|t| self.parse_type_str(t)).collect();
                    Type::Struct(s.to_string(), inner_types)
                } else {
                    Type::Struct(s.to_string(), vec![])
                }
            }
        }
    }

    /// マングル型文字列のスライスを `Vec<Type>` にパースする。
    pub fn parse_type_strs(&self, strs: &[&str]) -> Vec<Type> {
        strs.iter().map(|s| self.parse_type_str(s)).collect()
    }

    /// ベース名とマングルされた名前が一致するかを判定する。
    /// `name.starts_with(&format!("{}{}", base, self.open))` のショートカット。
    pub fn starts_with_mangled(&self, name: &str, base: &str) -> bool {
        if name.len() <= base.len() {
            return false;
        }
        name.starts_with(base) && name[base.len()..].starts_with(self.open)
    }

    /// 単一の引数をラップする。`format!("{}{}{}", self.open, arg, self.close)` のショートカット。
    pub fn wrap_single(&self, arg: &str) -> String {
        format!("{}{}{}", self.open, arg, self.close)
    }
}

/// グローバルシングルトン。全てのマングル操作はこの定数を経由する。
pub static MANGLER: Mangler = Mangler::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_args_empty() {
        assert_eq!(MANGLER.wrap_args("Vec", &[]), "Vec");
    }

    #[test]
    fn test_wrap_args_single() {
        assert_eq!(
            MANGLER.wrap_args("Vec", &["i64".to_string()]),
            "Vec[i64]"
        );
    }

    #[test]
    fn test_wrap_args_multiple() {
        assert_eq!(
            MANGLER.wrap_args("HashMap", &["i64".to_string(), "String".to_string()]),
            "HashMap[i64][String]"
        );
    }

    #[test]
    fn test_base_name() {
        assert_eq!(MANGLER.base_name("Vec[i64]"), "Vec");
        assert_eq!(MANGLER.base_name("HashMap[i64][String]"), "HashMap");
        assert_eq!(MANGLER.base_name("plain"), "plain");
    }

    #[test]
    fn test_has_args() {
        assert!(MANGLER.has_args("Vec[i64]"));
        assert!(!MANGLER.has_args("plain"));
    }

    #[test]
    fn test_extract_args() {
        assert_eq!(MANGLER.extract_args("Vec[i64]"), vec!["i64"]);
        assert_eq!(
            MANGLER.extract_args("HashMap[i64][String]"),
            vec!["i64", "String"]
        );
        assert_eq!(
            MANGLER.extract_args("Vec[Entry[i64][String]]"),
            vec!["Entry[i64][String]"]
        );
    }

    #[test]
    fn test_starts_with_mangled() {
        assert!(MANGLER.starts_with_mangled("Vec[i64]", "Vec"));
        assert!(!MANGLER.starts_with_mangled("Vec", "Vec"));
        assert!(!MANGLER.starts_with_mangled("Vector[i64]", "Vec"));
    }

    #[test]
    fn test_wrap_single() {
        assert_eq!(MANGLER.wrap_single("i64"), "[i64]");
    }

    #[test]
    fn test_parse_type_str() {
        assert_eq!(MANGLER.parse_type_str("i64"), Type::I64);
        assert_eq!(MANGLER.parse_type_str("f32"), Type::F32);
        assert_eq!(MANGLER.parse_type_str("bool"), Type::Bool);
        assert_eq!(
            MANGLER.parse_type_str("Pair[i64][i64]"),
            Type::Struct("Pair[i64][i64]".to_string(), vec![Type::I64, Type::I64])
        );
    }
}
