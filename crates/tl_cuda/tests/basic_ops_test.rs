//! Step 3: 基本演算テスト
//! binary ops, unary ops, scalar ops, comparison ops

use serial_test::serial;
use tl_cuda::{CudaTensor, DType};

fn approx_eq(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < eps,
            "mismatch at index {}: {} vs {} (eps={})",
            i,
            x,
            y,
            eps
        );
    }
}

// ========== 二項演算 ==========

#[test]
#[serial]
fn test_add() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], DType::F32);
    let c = a.add_impl(&b).unwrap();
    assert_eq!(c.shape(), &[3]);
    approx_eq(&c.to_vec::<f32>(), &[5.0, 7.0, 9.0], 1e-6);
}

#[test]
#[serial]
fn test_sub() {
    let a = CudaTensor::from_slice(&[5.0f32, 3.0, 1.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let c = a.sub_impl(&b).unwrap();
    approx_eq(&c.to_vec::<f32>(), &[4.0, 1.0, -2.0], 1e-6);
}

#[test]
#[serial]
fn test_mul() {
    let a = CudaTensor::from_slice(&[2.0f32, 3.0, 4.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[3.0f32, 2.0, 1.0], &[3], DType::F32);
    let c = a.mul_impl(&b).unwrap();
    approx_eq(&c.to_vec::<f32>(), &[6.0, 6.0, 4.0], 1e-6);
}

#[test]
#[serial]
fn test_div() {
    let a = CudaTensor::from_slice(&[6.0f32, 8.0, 10.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[2.0f32, 4.0, 5.0], &[3], DType::F32);
    let c = a.div_impl(&b).unwrap();
    approx_eq(&c.to_vec::<f32>(), &[3.0, 2.0, 2.0], 1e-6);
}

#[test]
#[serial]
fn test_neg() {
    let a = CudaTensor::from_slice(&[1.0f32, -2.0, 3.0], &[3], DType::F32);
    let c = a.neg_impl().unwrap();
    approx_eq(&c.to_vec::<f32>(), &[-1.0, 2.0, -3.0], 1e-6);
}

#[test]
#[serial]
fn test_abs() {
    let a = CudaTensor::from_slice(&[-1.0f32, 2.0, -3.0], &[3], DType::F32);
    let c = a.abs_impl().unwrap();
    approx_eq(&c.to_vec::<f32>(), &[1.0, 2.0, 3.0], 1e-6);
}

// ========== スカラー演算 ==========

#[test]
#[serial]
fn test_add_scalar() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let c = a.add_scalar_impl(10.0).unwrap();
    approx_eq(&c.to_vec::<f32>(), &[11.0, 12.0, 13.0], 1e-6);
}

#[test]
#[serial]
fn test_mul_scalar() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let c = a.mul_scalar_impl(3.0).unwrap();
    approx_eq(&c.to_vec::<f32>(), &[3.0, 6.0, 9.0], 1e-6);
}

#[test]
#[serial]
fn test_div_scalar() {
    let a = CudaTensor::from_slice(&[6.0f32, 9.0, 12.0], &[3], DType::F32);
    let c = a.div_scalar_impl(3.0).unwrap();
    approx_eq(&c.to_vec::<f32>(), &[2.0, 3.0, 4.0], 1e-6);
}

#[test]
#[serial]
fn test_pow_scalar() {
    let a = CudaTensor::from_slice(&[2.0f32, 3.0, 4.0], &[3], DType::F32);
    let c = a.pow_scalar_impl(2.0).unwrap();
    approx_eq(&c.to_vec::<f32>(), &[4.0, 9.0, 16.0], 1e-6);
}

#[test]
#[serial]
fn test_clamp() {
    let a = CudaTensor::from_slice(&[-2.0f32, 0.5, 1.5, 3.0], &[4], DType::F32);
    let c = a.clamp_impl(0.0, 2.0).unwrap();
    approx_eq(&c.to_vec::<f32>(), &[0.0, 0.5, 1.5, 2.0], 1e-6);
}

// ========== broadcast ==========

#[test]
#[serial]
fn test_add_broadcast() {
    // [2, 3] + [3] → broadcast to [2, 3]
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let b = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);
    let c = a.add_impl(&b).unwrap();
    assert_eq!(c.shape(), &[2, 3]);
    approx_eq(
        &c.to_vec::<f32>(),
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
        1e-6,
    );
}

// ========== 2D テンソル ==========

#[test]
#[serial]
fn test_add_2d() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
    let b = CudaTensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], DType::F32);
    let c = a.add_impl(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    approx_eq(&c.to_vec::<f32>(), &[6.0, 8.0, 10.0, 12.0], 1e-6);
}

// ========== 比較演算 ==========

#[test]
#[serial]
fn test_eq() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[1.0f32, 0.0, 3.0], &[3], DType::F32);
    let c = a.eq_impl(&b).unwrap();
    approx_eq(&c.to_vec::<f32>(), &[1.0, 0.0, 1.0], 1e-6);
}

#[test]
#[serial]
fn test_lt() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[2.0f32, 2.0, 1.0], &[3], DType::F32);
    let c = a.lt_impl(&b).unwrap();
    approx_eq(&c.to_vec::<f32>(), &[1.0, 0.0, 0.0], 1e-6);
}
