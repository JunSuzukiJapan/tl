//! Steps 7-8: 行列積 + 特殊演算テスト

use serial_test::serial;
use tl_cuda::{CudaTensor, DType};

fn approx_eq(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len(), "len: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < eps, "idx {}: {} vs {}", i, x, y);
    }
}

// ========== matmul ==========

#[test]
#[serial]
fn test_matmul_2x2() {
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
    let b = CudaTensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], DType::F32);
    let c = a.matmul_impl(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    approx_eq(&c.to_vec::<f32>(), &[19.0, 22.0, 43.0, 50.0], 1e-4);
}

#[test]
#[serial]
fn test_matmul_2x3_3x2() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let b = CudaTensor::from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2], DType::F32);
    let c = a.matmul_impl(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
    approx_eq(&c.to_vec::<f32>(), &[58.0, 64.0, 139.0, 154.0], 1e-4);
}

// ========== softmax ==========

#[test]
#[serial]
fn test_softmax() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], DType::F32);
    let r = t.softmax_impl(-1).unwrap();
    let v = r.to_vec::<f32>();
    assert_eq!(v.len(), 3);
    let sum: f32 = v.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {}", sum);
    // 値は昇順
    assert!(v[0] < v[1] && v[1] < v[2]);
}

// ========== embedding ==========

#[test]
#[serial]
fn test_embedding() {
    // vocab_size=3, embed_dim=2
    let weight = CudaTensor::from_slice(
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[3, 2],
        DType::F32,
    );
    let indices = CudaTensor::from_slice(&[0i64, 2], &[2], DType::I64);
    let r = weight.embedding_impl(&indices).unwrap();
    assert_eq!(r.shape(), &[2, 2]);
    approx_eq(&r.to_vec::<f32>(), &[10.0, 20.0, 50.0, 60.0], 1e-6);
}

// ========== tril ==========

#[test]
#[serial]
fn test_tril() {
    let t = CudaTensor::from_slice(
        &[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        &[3, 3],
        DType::F32,
    );
    let r = t.tril_impl(0).unwrap();
    approx_eq(
        &r.to_vec::<f32>(),
        &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        1e-6,
    );
}

// ========== cross_entropy ==========

#[test]
#[serial]
fn test_cross_entropy() {
    // batch=2, classes=3
    let logits = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3], DType::F32);
    let targets = CudaTensor::from_slice(&[2i64, 0], &[2], DType::I64);
    let loss = logits.cross_entropy_impl(&targets).unwrap();
    let v = loss.to_vec::<f32>();
    assert!(v[0] > 0.0); // loss should be positive
}

// ========== where_cond ==========

#[test]
#[serial]
fn test_where_cond() {
    let cond = CudaTensor::from_slice(&[1.0f32, 0.0, 1.0], &[3], DType::F32);
    let x = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);
    let y = CudaTensor::from_slice(&[-1.0f32, -2.0, -3.0], &[3], DType::F32);
    let r = CudaTensor::where_cond_impl(&cond, &x, &y).unwrap();
    approx_eq(&r.to_vec::<f32>(), &[10.0, -2.0, 30.0], 1e-6);
}
