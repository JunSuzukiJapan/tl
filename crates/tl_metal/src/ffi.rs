//! Metal FFI functions
//!
//! tl_runtime から呼び出される Metal バックエンドの FFI エントリポイント。
//! CPU バックエンド (tl_cpu) と同じシグネチャを持ち、JIT リンク時に切り替えられる。

use crate::tensor::MetalTensor;

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::sync::Arc;

// OpaqueTensor は MetalTensor のエイリアスとして扱う
pub type OpaqueTensor = MetalTensor;

/// テンソルを Arc で包んでポインタを返す（V5.0 メモリ管理）
// fn make_tensor(t: MetalTensor) -> *mut OpaqueTensor {
//     let arc = Arc::new(UnsafeCell::new(t));
//     Arc::into_raw(arc) as *mut OpaqueTensor
// }

#[no_mangle]
pub extern "C" fn tl_metal_clone(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    // Deep copy: データを完全にコピーして独立した新しいテンソルを作成。
    // Arc::clone() だけでは同じデータを共有するため、学習ループでの
    // パラメータ更新時に共有データの不整合が発生し SIGBUS になる。
    match tensor.clone_data() {
        Ok(cloned) => crate::ffi_ops::make_tensor(cloned),
        Err(e) => {
            eprintln!("Metal clone failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_metal_shallow_clone(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    unsafe {
        let arc = Arc::from_raw(t as *const UnsafeCell<MetalTensor>);
        let cloned = arc.clone();
        let _ = Arc::into_raw(arc); // Keep original alive
        Arc::into_raw(cloned) as *mut OpaqueTensor
    }
}

#[no_mangle]
pub extern "C" fn tl_metal_release(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    // Arc RC-1: release_if_live は直接 Arc::from_raw → drop
    crate::ffi_ops::release_if_live(t);
}

#[no_mangle]
pub extern "C" fn tl_metal_numel(t: *mut OpaqueTensor) -> i64 {
    if t.is_null() { return 0; }
    let tensor = unsafe { &*t };
    tensor.shape().iter().product::<usize>() as i64
}

#[no_mangle]
pub extern "C" fn tl_metal_data(t: *mut OpaqueTensor) -> *mut c_void {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    
    // StorageModeShared バッファは CPU/GPU 共有メモリなので、
    // sync_stream() 後に contents() ポインタを直接返せる。
    // 全要素 Vec コピーとスレッドローカルキャッシュが不要に。
    crate::command_stream::sync_stream();
    tensor.buffer().contents() as *mut c_void
}
