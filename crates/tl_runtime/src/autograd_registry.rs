//! Autograd Registry — FFI 層と MetalVar を接続するグローバルレジストリ
//!
//! thread_local! で MetalVar の計算グラフを追跡。
//! requires_grad=true のテンソルを MetalVar にラップし、
//! 演算時に自動で計算グラフを構築する。
//!
//! 重要: 演算結果のテンソルデータは常に FFI 層の通常計算を使用。
//! MetalVar は計算グラフ（backward パス）の追跡のみに使う。

use std::cell::RefCell;
use std::collections::HashMap;
use tl_metal::autograd::MetalVar;
use tl_metal::MetalTensor;
use crate::OpaqueTensor;

thread_local! {
    /// テンソルポインタ → MetalVar のマッピング
    static REGISTRY: RefCell<HashMap<usize, MetalVar>> = RefCell::new(HashMap::new());
}

/// テンソルを requires_grad として登録
pub fn register_requires_grad(tensor_ptr: *mut OpaqueTensor) {
    if tensor_ptr.is_null() { return; }
    let tensor = unsafe { &*tensor_ptr };
    let var = MetalVar::new(tensor.clone_data(), true);
    let key = tensor_ptr as usize;
    REGISTRY.with(|r| {
        r.borrow_mut().insert(key, var);
    });
}

/// 演算結果の MetalVar を登録
fn register_var(tensor_ptr: *mut OpaqueTensor, var: MetalVar) {
    let key = tensor_ptr as usize;
    REGISTRY.with(|r| {
        r.borrow_mut().insert(key, var);
    });
}

/// テンソルが autograd 追跡されているか
pub fn is_tracked(tensor_ptr: *mut OpaqueTensor) -> bool {
    if tensor_ptr.is_null() { return false; }
    let key = tensor_ptr as usize;
    REGISTRY.with(|r| r.borrow().contains_key(&key))
}

/// MetalVar を取得
fn get_var(tensor_ptr: *mut OpaqueTensor) -> Option<MetalVar> {
    if tensor_ptr.is_null() { return None; }
    let key = tensor_ptr as usize;
    REGISTRY.with(|r| r.borrow().get(&key).cloned())
}

/// backward 実行
pub fn backward(tensor_ptr: *mut OpaqueTensor) {
    if let Some(var) = get_var(tensor_ptr) {
        var.backward();
    }
}

/// 勾配を取得 — MetalTensor として返す
pub fn grad(tensor_ptr: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if let Some(var) = get_var(tensor_ptr) {
        if let Some(grad_tensor) = var.grad() {
            return Box::into_raw(Box::new(grad_tensor));
        }
    }
    // フォールバック: ゼロテンソル
    if tensor_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*tensor_ptr };
    let shape = MetalTensor::shape(tensor);
    let zeros = MetalTensor::zeros(shape, MetalTensor::dtype(tensor));
    Box::into_raw(Box::new(zeros))
}

/// 計算グラフから切り離し
pub fn detach(tensor_ptr: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if tensor_ptr.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*tensor_ptr };
    let cloned = tensor.clone_data();
    Box::into_raw(Box::new(cloned))
}

/// requires_grad を有効にして登録
pub fn enable_grad(tensor_ptr: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if tensor_ptr.is_null() { return std::ptr::null_mut(); }
    register_requires_grad(tensor_ptr);
    tensor_ptr
}

// ========== 演算ヘルパー ==========

/// 二項演算のヘルパー
pub fn binary_op<F>(
    a: *mut OpaqueTensor,
    b: *mut OpaqueTensor,
    var_op: F,
    fallback_result: *mut OpaqueTensor,
) -> *mut OpaqueTensor
where
    F: FnOnce(&MetalVar, &MetalVar) -> MetalVar,
{
    let a_tracked = is_tracked(a);
    let b_tracked = is_tracked(b);
    
    if !a_tracked && !b_tracked {
        return fallback_result;
    }
    
    let a_var = if a_tracked {
        get_var(a).unwrap()
    } else {
        let tensor = unsafe { &*a };
        MetalVar::new(tensor.clone_data(), false)
    };
    
    let b_var = if b_tracked {
        get_var(b).unwrap()
    } else {
        let tensor = unsafe { &*b };
        MetalVar::new(tensor.clone_data(), false)
    };
    
    let result_var = var_op(&a_var, &b_var);
    register_var(fallback_result, result_var);
    fallback_result
}

/// 単項演算のヘルパー
pub fn unary_op<F>(
    a: *mut OpaqueTensor,
    var_op: F,
    fallback_result: *mut OpaqueTensor,
) -> *mut OpaqueTensor
where
    F: FnOnce(&MetalVar) -> MetalVar,
{
    if !is_tracked(a) {
        return fallback_result;
    }
    
    let a_var = get_var(a).unwrap();
    let result_var = var_op(&a_var);
    register_var(fallback_result, result_var);
    fallback_result
}
