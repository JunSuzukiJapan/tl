//! Step 2: FFI メモリ管理テスト
//! make_tensor / release_if_live / acquire_tensor の Arc ベース RC 管理を検証

use serial_test::serial;
use std::sync::atomic::Ordering;
use tl_cuda::ffi_ops::{
    acquire_tensor, make_tensor, release_if_live, ACQUIRE_COUNT, MAKE_COUNT, RELEASE_COUNT,
};
use tl_cuda::{CudaTensor, DType};

// ========== make_tensor + release_if_live ==========

#[test]
#[serial]
fn test_make_tensor_returns_non_null() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let ptr = make_tensor(t);
    assert!(!ptr.is_null());
    // クリーンアップ
    release_if_live(ptr);
}

#[test]
#[serial]
fn test_make_tensor_preserves_data() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let ptr = make_tensor(t);
    assert!(!ptr.is_null());

    // ポインタから CudaTensor にアクセスしてデータを検証
    let tensor = unsafe { &*ptr };
    let data: Vec<f32> = tensor.to_vec();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);

    release_if_live(ptr);
}

#[test]
#[serial]
fn test_release_null_is_safe() {
    // null ポインタの release がパニックしないことを確認
    release_if_live(std::ptr::null_mut());
}

#[test]
#[serial]
fn test_acquire_increases_rc() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0], &[2], DType::F32);
    let ptr = make_tensor(t); // RC=1

    acquire_tensor(ptr); // RC=2

    // 1回目の release → RC=1 (drop されない)
    release_if_live(ptr);
    // 2回目の release → RC=0 (drop)
    release_if_live(ptr);
}

#[test]
#[serial]
fn test_acquire_null_is_safe() {
    acquire_tensor(std::ptr::null_mut());
}

// ========== デバッグカウンタ ==========

#[test]
#[serial]
fn test_debug_counters() {
    tl_cuda::ffi_ops::tl_cuda_debug_reset_counters();

    let t1 = CudaTensor::from_slice(&[1.0f32], &[1], DType::F32);
    let ptr1 = make_tensor(t1);
    let t2 = CudaTensor::from_slice(&[2.0f32], &[1], DType::F32);
    let ptr2 = make_tensor(t2);

    assert_eq!(MAKE_COUNT.load(Ordering::SeqCst), 2);

    acquire_tensor(ptr1);
    assert_eq!(ACQUIRE_COUNT.load(Ordering::SeqCst), 1);

    release_if_live(ptr1); // RC: 2->1
    release_if_live(ptr1); // RC: 1->0
    release_if_live(ptr2); // RC: 1->0
    assert_eq!(RELEASE_COUNT.load(Ordering::SeqCst), 3);
}

// ========== FFI テンソル作成 ==========

#[test]
#[serial]
fn test_ffi_new_f32() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = [2usize, 3];
    let ptr = tl_cuda::ffi_ops::tl_cuda_new(data.as_ptr(), 2, shape.as_ptr());
    assert!(!ptr.is_null());

    let tensor = unsafe { &*ptr };
    assert_eq!(tensor.shape(), &[2, 3]);
    let result: Vec<f32> = tensor.to_vec();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    release_if_live(ptr);
}

#[test]
#[serial]
fn test_ffi_zeros() {
    let shape = [3usize, 4];
    let ptr = tl_cuda::ffi_ops::tl_cuda_zeros(2, shape.as_ptr(), false);
    assert!(!ptr.is_null());

    let tensor = unsafe { &*ptr };
    assert_eq!(tensor.shape(), &[3, 4]);
    let result: Vec<f32> = tensor.to_vec();
    assert!(result.iter().all(|&v| v == 0.0));

    release_if_live(ptr);
}

#[test]
#[serial]
fn test_ffi_ones_with_grad() {
    let shape = [2usize, 2];
    let ptr = tl_cuda::ffi_ops::tl_cuda_ones(2, shape.as_ptr(), true);
    assert!(!ptr.is_null());

    let tensor = unsafe { &*ptr };
    assert_eq!(tensor.shape(), &[2, 2]);
    assert!(tensor.requires_grad());
    let result: Vec<f32> = tensor.to_vec();
    assert!(result.iter().all(|&v| v == 1.0));

    release_if_live(ptr);
}

#[test]
#[serial]
fn test_ffi_clone_release() {
    use tl_cuda::ffi::*;

    let t = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);
    let ptr = make_tensor(t);

    // clone → 新しいポインタ（同じ Arc）
    let cloned = tl_cuda_clone(ptr);
    assert!(!cloned.is_null());

    // 両方から同じデータにアクセスできる
    let orig = unsafe { &*ptr };
    let copy = unsafe { &*cloned };
    assert_eq!(orig.to_vec::<f32>(), copy.to_vec::<f32>());

    // release
    tl_cuda_release(cloned);
    tl_cuda_release(ptr);
}

#[test]
#[serial]
fn test_ffi_numel() {
    use tl_cuda::ffi::*;

    let t = CudaTensor::from_slice(&[1.0f32; 24], &[2, 3, 4], DType::F32);
    let ptr = make_tensor(t);

    assert_eq!(tl_cuda_numel(ptr), 24);

    tl_cuda_release(ptr);
}

#[test]
#[serial]
fn test_ffi_data() {
    use tl_cuda::ffi::*;

    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let ptr = make_tensor(t);

    let data_ptr = tl_cuda_data(ptr);
    assert!(!data_ptr.is_null());

    // data_ptr は f32 スライスとして読める
    let values = unsafe { std::slice::from_raw_parts(data_ptr as *const f32, 3) };
    assert_eq!(values, &[1.0, 2.0, 3.0]);

    tl_cuda_release(ptr);
}
