//! tl_cuda インプレース演算テスト
//! 既存テストで未カバーのインプレース演算・データ操作を検証

use tl_cuda::{CudaTensor, DType};
use serial_test::serial;

// ========== ヘルパー関数 ==========

#[allow(dead_code)]
fn assert_approx_eq(a: f32, b: f32, eps: f32) {
    assert!((a - b).abs() < eps, "Expected {} ≈ {}, diff = {}", a, b, (a - b).abs());
}

fn assert_tensor_approx_eq(t: &CudaTensor, expected: &[f32], eps: f32) {
    let data = t.to_vec::<f32>();
    assert_eq!(data.len(), expected.len(), "Length mismatch: {} vs {}", data.len(), expected.len());
    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < eps, "At index {}: {} ≈ {}, diff = {}", i, a, b, (a - b).abs());
    }
}

// =====================================================================
// 1. 二項インプレース演算テスト
// =====================================================================

#[test]
#[serial]
fn test_inplace_add_assign() {
    let mut a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);

    // add_impl で結果を a に書き戻す
    let result = a.add(&b).unwrap();
    a = result;
    assert_tensor_approx_eq(&a, &[11.0, 22.0, 33.0], 1e-5);
}

#[test]
#[serial]
fn test_inplace_sub_assign() {
    let mut a = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);

    let result = a.sub(&b).unwrap();
    a = result;
    assert_tensor_approx_eq(&a, &[9.0, 18.0, 27.0], 1e-5);
}

#[test]
#[serial]
fn test_inplace_mul_assign() {
    let mut a = CudaTensor::from_slice(&[2.0f32, 3.0, 4.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[5.0f32, 6.0, 7.0], &[3], DType::F32);

    let result = a.mul(&b).unwrap();
    a = result;
    assert_tensor_approx_eq(&a, &[10.0, 18.0, 28.0], 1e-5);
}

#[test]
#[serial]
fn test_inplace_div_assign() {
    let mut a = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[2.0f32, 4.0, 5.0], &[3], DType::F32);

    let result = a.div(&b).unwrap();
    a = result;
    assert_tensor_approx_eq(&a, &[5.0, 5.0, 6.0], 1e-5);
}

// =====================================================================
// 2. スカラーインプレース演算テスト
// =====================================================================

#[test]
#[serial]
fn test_inplace_add_scalar() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let result = a.add_scalar(10.0).unwrap();
    assert_tensor_approx_eq(&result, &[11.0, 12.0, 13.0], 1e-5);

    // 負の値の加算 = 減算
    let result2 = a.add_scalar(-1.0).unwrap();
    assert_tensor_approx_eq(&result2, &[0.0, 1.0, 2.0], 1e-5);
}

#[test]
#[serial]
fn test_inplace_sub_scalar() {
    let a = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);
    let result = a.sub_scalar(5.0).unwrap();
    assert_tensor_approx_eq(&result, &[5.0, 15.0, 25.0], 1e-5);
}

#[test]
#[serial]
fn test_inplace_mul_scalar() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let result = a.mul_scalar(3.0).unwrap();
    assert_tensor_approx_eq(&result, &[3.0, 6.0, 9.0], 1e-5);
}

#[test]
#[serial]
fn test_inplace_div_scalar() {
    let a = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);
    let result = a.div_scalar(10.0).unwrap();
    assert_tensor_approx_eq(&result, &[1.0, 2.0, 3.0], 1e-5);
}

// =====================================================================
// 3. スカラー演算の連鎖テスト
// =====================================================================

#[test]
#[serial]
fn test_scalar_ops_chained() {
    let a = CudaTensor::from_slice(&[2.0f32, 4.0, 6.0], &[3], DType::F32);
    // (a + 1) * 2 - 3
    let r = a.add_scalar(1.0).unwrap()
             .mul_scalar(2.0).unwrap()
             .sub_scalar(3.0).unwrap();
    assert_tensor_approx_eq(&r, &[3.0, 7.0, 11.0], 1e-5);
}

// =====================================================================
// 4. データ操作テスト
// =====================================================================

#[test]
#[serial]
fn test_clone_data() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
    let b = a.clone_data().unwrap();

    // データが同じであること
    assert_tensor_approx_eq(&b, &[1.0, 2.0, 3.0, 4.0], 1e-5);
    assert_eq!(b.shape(), &[2, 2]);

    // 独立したコピーであること（元を変更してもコピーに影響しない）
    let a_modified = a.add_scalar(10.0).unwrap();
    assert_tensor_approx_eq(&a_modified, &[11.0, 12.0, 13.0, 14.0], 1e-5);
    // b は変わらない
    assert_tensor_approx_eq(&b, &[1.0, 2.0, 3.0, 4.0], 1e-5);
}

#[test]
#[serial]
fn test_shallow_clone() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = a.shallow_clone();

    assert_eq!(b.shape(), &[3]);
    assert_tensor_approx_eq(&b, &[1.0, 2.0, 3.0], 1e-5);
}

#[test]
#[serial]
fn test_contiguous() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let b = a.clone_data().unwrap();

    assert_eq!(b.shape(), &[2, 3]);
    assert_tensor_approx_eq(&b, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5);
}

// =====================================================================
// 5. 連続インプレース更新テスト
// =====================================================================

#[test]
#[serial]
fn test_multiple_inplace_updates() {
    let mut a = CudaTensor::from_slice(&[1.0f32, 1.0, 1.0], &[3], DType::F32);

    // 10回加算: 1 + 10*1 = 11
    let increment = CudaTensor::from_slice(&[1.0f32, 1.0, 1.0], &[3], DType::F32);
    for _ in 0..10 {
        let result = a.add(&increment).unwrap();
        a = result;
    }
    assert_tensor_approx_eq(&a, &[11.0, 11.0, 11.0], 1e-5);
}

// =====================================================================
// 6. 2D テンソルのインプレース演算テスト
// =====================================================================

#[test]
#[serial]
fn test_inplace_2d_add() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let b = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], DType::F32);
    let c = a.add(&b).unwrap();
    assert_eq!(c.shape(), &[2, 3]);
    assert_tensor_approx_eq(&c, &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0], 1e-5);
}

// =====================================================================
// 7. ゼロ除算のスカラー演算テスト
// =====================================================================

#[test]
#[serial]
fn test_mul_scalar_zero() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let result = a.mul_scalar(0.0).unwrap();
    assert_tensor_approx_eq(&result, &[0.0, 0.0, 0.0], 1e-5);
}

#[test]
#[serial]
fn test_add_scalar_zero() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let result = a.add_scalar(0.0).unwrap();
    assert_tensor_approx_eq(&result, &[1.0, 2.0, 3.0], 1e-5);
}
