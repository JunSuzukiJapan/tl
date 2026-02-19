//! Specialization Registry
//!
//! ジェネリック型の特殊化を追跡し、必要なメソッドを生成するためのレジストリ。

use crate::compiler::ast::{Type, MANGLE_OPEN, MANGLE_CLOSE, mangle_wrap_args};
use std::collections::HashSet;

/// 特殊化された型を追跡するレジストリ
pub struct SpecializationRegistry {
    /// 特殊化の一覧: (基底名, 型引数)
    specializations: Vec<(String, Vec<Type>)>,
    /// マングル名のセット（重複チェック用）
    mangled_names: HashSet<String>,
}

impl SpecializationRegistry {
    /// 新しいレジストリを作成
    pub fn new() -> Self {
        Self {
            specializations: Vec::new(),
            mangled_names: HashSet::new(),
        }
    }

    /// 特殊化を登録。既に登録済みなら false を返す
    pub fn register(&mut self, base_name: &str, type_args: &[Type]) -> bool {
        let mangled = mangle_type_name(base_name, type_args);
        if self.mangled_names.contains(&mangled) {
            return false; // 既に登録済み
        }
        self.mangled_names.insert(mangled);
        self.specializations.push((base_name.to_string(), type_args.to_vec()));
        true // 新規登録
    }

    /// マングル名が登録済みかチェック
    pub fn contains(&self, base_name: &str, type_args: &[Type]) -> bool {
        let mangled = mangle_type_name(base_name, type_args);
        self.mangled_names.contains(&mangled)
    }

    /// 未処理の特殊化を取り出す（内部リストをクリア）
    pub fn drain_pending(&mut self) -> Vec<(String, Vec<Type>)> {
        std::mem::take(&mut self.specializations)
    }

    /// 登録されている特殊化の数
    pub fn len(&self) -> usize {
        self.mangled_names.len()
    }

    /// 未処理の特殊化があるかどうか（ループ条件用）
    /// NOTE: mangled_names は履歴として保持されるので、specializations をチェックする
    pub fn is_empty(&self) -> bool {
        self.specializations.is_empty()
    }
}

impl Default for SpecializationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Type を文字列サフィックスに変換（マングリング用）
pub fn type_to_suffix(ty: &Type) -> String {
    match ty {
        Type::I64 => "i64".to_string(),
        Type::I32 => "i32".to_string(),
        Type::U8 => "u8".to_string(),
        Type::F32 => "f32".to_string(),
        Type::F64 => "f64".to_string(),
        Type::Bool => "bool".to_string(),
        Type::Usize => "usize".to_string(),
        Type::Void => "void".to_string(),
        Type::String(_) => "String".to_string(),
        Type::Char(_) => "Char".to_string(),
        Type::Struct(name, args) => {
            if args.is_empty() {
                name.clone()
            } else {
                mangle_type_name(name, args)
            }
        }
        Type::Enum(name, args) => {
            if args.is_empty() {
                name.clone()
            } else {
                mangle_type_name(name, args)
            }
        }
        Type::Tensor(inner, rank) => {
            let args = vec![type_to_suffix(inner), rank.to_string()];
            mangle_wrap_args("Tensor", &args)
        }
        Type::Tuple(types) => {
            let args: Vec<String> = types.iter().map(type_to_suffix).collect();
            mangle_wrap_args("Tuple", &args)
        }
        Type::Path(segments, args) => {
            let base = segments.join("_");
            if args.is_empty() {
                base
            } else {
                let args_str: Vec<String> = args.iter().map(type_to_suffix).collect();
                mangle_wrap_args(&base, &args_str)
            }
        }
        Type::Array(inner, size) => {
            let args = vec![type_to_suffix(inner), size.to_string()];
            mangle_wrap_args("Array", &args)
        }
        _ => "unknown".to_string(),
    }
}

/// マングル名を生成
pub fn mangle_type_name(base_name: &str, type_args: &[Type]) -> String {
    if type_args.is_empty() {
        base_name.to_string()
    } else {
        let args_str: Vec<String> = type_args.iter().map(type_to_suffix).collect();
        mangle_wrap_args(base_name, &args_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_new() {
        let mut registry = SpecializationRegistry::new();
        assert!(registry.register("Vec", &[Type::I64]));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_register_duplicate() {
        let mut registry = SpecializationRegistry::new();
        assert!(registry.register("Vec", &[Type::I64]));
        assert!(!registry.register("Vec", &[Type::I64])); // 重複
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_mangle_type_name() {
        assert_eq!(mangle_type_name("Vec", &[Type::I64]), "Vec[i64]");
        assert_eq!(mangle_type_name("HashMap", &[Type::I64, Type::I64]), "HashMap[i64][i64]");
        assert_eq!(mangle_type_name("Option", &[]), "Option");
    }
}
