//! Step 4: 活性化関数テスト

use serial_test::serial;
use tl_cuda::{CudaTensor, DType};

fn approx_eq(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < eps,
            "idx {}: {} vs {} (eps={})",
            i,
            x,
            y,
            eps
        );
    }
}

#[test]
#[serial]
fn test_relu() {
    let t = CudaTensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], &[4], DType::F32);
    let r = t.relu_impl().unwrap();
    approx_eq(&r.to_vec::<f32>(), &[0.0, 0.0, 1.0, 2.0], 1e-6);
}

#[test]
#[serial]
fn test_sigmoid() {
    let t = CudaTensor::from_slice(&[0.0f32], &[1], DType::F32);
    let r = t.sigmoid_impl().unwrap();
    approx_eq(&r.to_vec::<f32>(), &[0.5], 1e-6);
}

#[test]
#[serial]
fn test_tanh() {
    let t = CudaTensor::from_slice(&[0.0f32], &[1], DType::F32);
    let r = t.tanh_impl().unwrap();
    approx_eq(&r.to_vec::<f32>(), &[0.0], 1e-6);
}

#[test]
#[serial]
fn test_exp() {
    let t = CudaTensor::from_slice(&[0.0f32, 1.0], &[2], DType::F32);
    let r = t.exp_impl().unwrap();
    approx_eq(&r.to_vec::<f32>(), &[1.0, std::f32::consts::E], 1e-5);
}

#[test]
#[serial]
fn test_log() {
    let t = CudaTensor::from_slice(&[1.0f32, std::f32::consts::E], &[2], DType::F32);
    let r = t.log_impl().unwrap();
    approx_eq(&r.to_vec::<f32>(), &[0.0, 1.0], 1e-5);
}

#[test]
#[serial]
fn test_sqrt() {
    let t = CudaTensor::from_slice(&[4.0f32, 9.0, 16.0], &[3], DType::F32);
    let r = t.sqrt_impl().unwrap();
    approx_eq(&r.to_vec::<f32>(), &[2.0, 3.0, 4.0], 1e-6);
}

#[test]
#[serial]
fn test_gelu() {
    let t = CudaTensor::from_slice(&[0.0f32], &[1], DType::F32);
    let r = t.gelu_impl().unwrap();
    approx_eq(&r.to_vec::<f32>(), &[0.0], 1e-6);
}

#[test]
#[serial]
fn test_silu() {
    let t = CudaTensor::from_slice(&[0.0f32], &[1], DType::F32);
    let r = t.silu_impl().unwrap();
    approx_eq(&r.to_vec::<f32>(), &[0.0], 1e-6);
}

#[test]
#[serial]
fn test_sin_cos() {
    let t = CudaTensor::from_slice(&[0.0f32, std::f32::consts::FRAC_PI_2], &[2], DType::F32);
    let s = t.sin_impl().unwrap();
    let c = t.cos_impl().unwrap();
    approx_eq(&s.to_vec::<f32>(), &[0.0, 1.0], 1e-5);
    approx_eq(&c.to_vec::<f32>(), &[1.0, 0.0], 1e-5);
}

#[test]
#[serial]
fn test_floor_ceil_round() {
    let t = CudaTensor::from_slice(&[1.3f32, 2.7, -1.5], &[3], DType::F32);
    let f = t.floor_impl().unwrap();
    let c = t.ceil_impl().unwrap();
    let r = t.round_impl().unwrap();
    approx_eq(&f.to_vec::<f32>(), &[1.0, 2.0, -2.0], 1e-6);
    approx_eq(&c.to_vec::<f32>(), &[2.0, 3.0, -1.0], 1e-6);
    approx_eq(&r.to_vec::<f32>(), &[1.0, 3.0, -2.0], 1e-6);
}
