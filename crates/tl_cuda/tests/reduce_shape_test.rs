//! Steps 5-6: リダクション + 形状操作テスト

use serial_test::serial;
use tl_cuda::{CudaTensor, DType};

fn approx_eq(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len(), "len: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < eps, "idx {}: {} vs {}", i, x, y);
    }
}

// ========== Reduce (全要素) ==========

#[test]
#[serial]
fn test_sumall() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let s = t.sumall_impl().unwrap();
    assert!((s - 10.0).abs() < 1e-6);
}

#[test]
#[serial]
fn test_mean_all() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let m = t.mean_all_impl().unwrap();
    assert!((m - 2.5).abs() < 1e-6);
}

#[test]
#[serial]
fn test_max_all() {
    let t = CudaTensor::from_slice(&[1.0f32, 5.0, 3.0, 2.0], &[4], DType::F32);
    let r = t.max_all_impl().unwrap();
    approx_eq(&r.to_vec::<f32>(), &[5.0], 1e-6);
}

#[test]
#[serial]
fn test_min_all() {
    let t = CudaTensor::from_slice(&[1.0f32, 5.0, -3.0, 2.0], &[4], DType::F32);
    let r = t.min_all_impl().unwrap();
    approx_eq(&r.to_vec::<f32>(), &[-3.0], 1e-6);
}

// ========== Reduce (軸) ==========

#[test]
#[serial]
fn test_sum_axis() {
    // shape [2, 3], sum over axis 0 → [3]
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let r = t.sum_impl(0).unwrap();
    assert_eq!(r.shape(), &[3]);
    approx_eq(&r.to_vec::<f32>(), &[5.0, 7.0, 9.0], 1e-6);
}

#[test]
#[serial]
fn test_sum_axis1() {
    // shape [2, 3], sum over axis 1 → [2]
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let r = t.sum_impl(1).unwrap();
    assert_eq!(r.shape(), &[2]);
    approx_eq(&r.to_vec::<f32>(), &[6.0, 15.0], 1e-6);
}

#[test]
#[serial]
fn test_argmax_axis() {
    let t = CudaTensor::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], DType::F32);
    let r = t.argmax_impl(1).unwrap();
    assert_eq!(r.shape(), &[2]);
    let result: Vec<i64> = r.to_vec();
    assert_eq!(result, vec![1, 2]); // idx of 5.0 and 6.0
}

// ========== 形状操作 ==========

#[test]
#[serial]
fn test_reshape() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let r = t.reshape_impl(&[3, 2]).unwrap();
    assert_eq!(r.shape(), &[3, 2]);
    approx_eq(&r.to_vec::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-6);
}

#[test]
#[serial]
fn test_transpose() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let r = t.transpose_impl(0, 1).unwrap();
    assert_eq!(r.shape(), &[3, 2]);
    approx_eq(&r.to_vec::<f32>(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-6);
}

#[test]
#[serial]
fn test_squeeze_unsqueeze() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], DType::F32);
    let squeezed = t.squeeze_impl(0).unwrap();
    assert_eq!(squeezed.shape(), &[3]);

    let unsqueezed = squeezed.unsqueeze_impl(0).unwrap();
    assert_eq!(unsqueezed.shape(), &[1, 3]);
}

#[test]
#[serial]
fn test_narrow() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], DType::F32);
    let r = t.narrow_impl(0, 1, 3).unwrap();
    assert_eq!(r.shape(), &[3]);
    approx_eq(&r.to_vec::<f32>(), &[2.0, 3.0, 4.0], 1e-6);
}

#[test]
#[serial]
fn test_cat() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[4.0f32, 5.0], &[2], DType::F32);
    let r = a.cat_impl(&b, 0).unwrap();
    assert_eq!(r.shape(), &[5]);
    approx_eq(&r.to_vec::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0], 1e-6);
}

#[test]
#[serial]
fn test_broadcast_to() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], DType::F32);
    let r = t.broadcast_to_impl(&[2, 3]).unwrap();
    assert_eq!(r.shape(), &[2, 3]);
    approx_eq(&r.to_vec::<f32>(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 1e-6);
}
