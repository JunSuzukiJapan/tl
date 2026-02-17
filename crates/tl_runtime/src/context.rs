//! context スタブモジュール
//!
//! 実行コンテキストのスタブ

use std::collections::HashMap;
use crate::OpaqueTensor;

/// コンテキスト構造体（スタブ）
pub struct ExecutionContext {
    // スタブ
}

/// テンソルコンテキスト（コンパイラ互換用）
#[derive(Clone)]
pub struct TensorContext {
    tensors: HashMap<String, TensorValue>,
}

/// テンソル値（コンパイラ互換用）
#[derive(Clone)]
pub enum TensorValue {
    Tensor(*mut OpaqueTensor),
    Null,
}

impl TensorValue {
    /// インデックスでテンソル要素を取得
    pub fn get(&self, indices: &[i64]) -> Option<f64> {
        match self {
            TensorValue::Tensor(ptr) => {
                if ptr.is_null() {
                    return None;
                }
                unsafe {
                    let tensor = &**ptr;
                    let shape = tensor.shape();
                    
                    // インデックスを線形アドレスに変換
                    if indices.len() != shape.len() {
                        return None;
                    }
                    
                    let mut linear_idx = 0usize;
                    let mut stride = 1usize;
                    for i in (0..indices.len()).rev() {
                        let idx = indices[i] as usize;
                        if idx >= shape[i] {
                            return None;
                        }
                        linear_idx += idx * stride;
                        stride *= shape[i];
                    }
                    
                    // テンソルからデータを取得
                    let data: Vec<f32> = tensor.to_vec();
                    if linear_idx < data.len() {
                        Some(data[linear_idx] as f64)
                    } else {
                        None
                    }
                }
            }
            TensorValue::Null => None,
        }
    }
}

impl TensorContext {
    pub fn new() -> Self {
        TensorContext {
            tensors: HashMap::new(),
        }
    }
    
    pub fn get(&self, name: &str) -> Option<&TensorValue> {
        self.tensors.get(name)
    }
    
    pub fn insert(&mut self, name: String, value: TensorValue) {
        self.tensors.insert(name, value);
    }
}

impl Default for TensorContext {
    fn default() -> Self {
        Self::new()
    }
}

/// コンテキスト作成（スタブ）
#[unsafe(no_mangle)]
/// @ffi_sig () -> *mut ExecutionContext
pub extern "C" fn tl_context_new() -> *mut ExecutionContext {
    Box::into_raw(Box::new(ExecutionContext {}))
}

/// コンテキスト解放（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_context_free(ctx: *mut ExecutionContext) {
    if !ctx.is_null() {
        unsafe {
            let _ = Box::from_raw(ctx);
        }
    }
}
