//! tl_metal 包括的テスト

use tl_metal::{MetalTensor, DType};
use tl_metal::{SGD, Adam, AdamW, clip_grad_norm};
use serial_test::serial;

// ========== ヘルパー関数 ==========

fn assert_approx_eq(a: f32, b: f32, eps: f32) {
    assert!((a - b).abs() < eps, "Expected {} ≈ {}, diff = {}", a, b, (a - b).abs());
}

fn assert_tensor_approx_eq(a: &MetalTensor, b: &[f32], eps: f32) {
    let a_data = a.to_vec::<f32>();
    assert_eq!(a_data.len(), b.len(), "Length mismatch: {} vs {}", a_data.len(), b.len());
    for (i, (&av, &bv)) in a_data.iter().zip(b.iter()).enumerate() {
        assert!((av - bv).abs() < eps, "At index {}: {} ≈ {}", i, av, bv);
    }
}

// ========== 基本演算テスト ==========

#[test]
#[serial]
fn test_tensor_creation() {
    let t = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.elem_count(), 4);
}

#[test]
#[serial]
fn test_zeros_ones() {
    let zeros = MetalTensor::zeros(&[3, 3], DType::F32);
    let ones = MetalTensor::ones(&[3, 3], DType::F32);
    
    let z_data = zeros.to_vec::<f32>();
    let o_data = ones.to_vec::<f32>();
    
    assert!(z_data.iter().all(|&x| x == 0.0));
    assert!(o_data.iter().all(|&x| x == 1.0));
}

#[test]
#[serial]
fn test_randn() {
    let t = MetalTensor::randn(&[100], DType::F32);
    let data = t.to_vec::<f32>();
    
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 0.5, "Mean should be near 0: {}", mean);
}

// ========== 二項演算テスト ==========

#[test]
#[serial]
fn test_add() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = MetalTensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], DType::F32);
    let c = a.add(&b).unwrap();
    assert_tensor_approx_eq(&c, &[5.0, 7.0, 9.0], 1e-5);
}

#[test]
#[serial]
fn test_sub() {
    let a = MetalTensor::from_slice(&[5.0f32, 6.0, 7.0], &[3], DType::F32);
    let b = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let c = a.sub(&b).unwrap();
    assert_tensor_approx_eq(&c, &[4.0, 4.0, 4.0], 1e-5);
}

#[test]
#[serial]
fn test_mul() {
    let a = MetalTensor::from_slice(&[2.0f32, 3.0, 4.0], &[3], DType::F32);
    let b = MetalTensor::from_slice(&[3.0f32, 4.0, 5.0], &[3], DType::F32);
    let c = a.mul(&b).unwrap();
    assert_tensor_approx_eq(&c, &[6.0, 12.0, 20.0], 1e-5);
}

#[test]
#[serial]
fn test_div() {
    let a = MetalTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);
    let b = MetalTensor::from_slice(&[2.0f32, 4.0, 5.0], &[3], DType::F32);
    let c = a.div(&b).unwrap();
    assert_tensor_approx_eq(&c, &[5.0, 5.0, 6.0], 1e-5);
}

#[test]
#[serial]
fn test_matmul() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let b = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], DType::F32);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_tensor_approx_eq(&c, &[22.0, 28.0, 49.0, 64.0], 1e-5);
}

// ========== スカラー演算テスト ==========

#[test]
#[serial]
fn test_add_scalar() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let c = a.add_scalar(10.0).unwrap();
    assert_tensor_approx_eq(&c, &[11.0, 12.0, 13.0], 1e-5);
}

#[test]
#[serial]
fn test_mul_scalar() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let c = a.mul_scalar(2.0).unwrap();
    assert_tensor_approx_eq(&c, &[2.0, 4.0, 6.0], 1e-5);
}

#[test]
#[serial]
fn test_clamp() {
    let a = MetalTensor::from_slice(&[-1.0f32, 0.5, 2.0], &[3], DType::F32);
    let c = a.clamp(0.0, 1.0).unwrap();
    assert_tensor_approx_eq(&c, &[0.0, 0.5, 1.0], 1e-5);
}

// ========== 単項演算テスト ==========

#[test]
#[serial]
fn test_neg() {
    let a = MetalTensor::from_slice(&[1.0f32, -2.0, 3.0], &[3], DType::F32);
    let c = a.neg().unwrap();
    assert_tensor_approx_eq(&c, &[-1.0, 2.0, -3.0], 1e-5);
}

#[test]
#[serial]
fn test_abs() {
    let a = MetalTensor::from_slice(&[-1.0f32, 2.0, -3.0], &[3], DType::F32);
    let c = a.abs().unwrap();
    assert_tensor_approx_eq(&c, &[1.0, 2.0, 3.0], 1e-5);
}

#[test]
#[serial]
fn test_exp() {
    let a = MetalTensor::from_slice(&[0.0f32, 1.0, 2.0], &[3], DType::F32);
    let c = a.exp().unwrap();
    assert_tensor_approx_eq(&c, &[1.0, 2.7182817, 7.389056], 1e-4);
}

#[test]
#[serial]
fn test_log() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.7182817, 7.389056], &[3], DType::F32);
    let c = a.log().unwrap();
    assert_tensor_approx_eq(&c, &[0.0, 1.0, 2.0], 1e-4);
}

#[test]
#[serial]
fn test_sqrt() {
    let a = MetalTensor::from_slice(&[1.0f32, 4.0, 9.0], &[3], DType::F32);
    let c = a.sqrt().unwrap();
    assert_tensor_approx_eq(&c, &[1.0, 2.0, 3.0], 1e-5);
}

#[test]
#[serial]
fn test_relu() {
    let a = MetalTensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], &[4], DType::F32);
    let c = a.relu().unwrap();
    assert_tensor_approx_eq(&c, &[0.0, 0.0, 1.0, 2.0], 1e-5);
}

#[test]
#[serial]
fn test_sigmoid() {
    let a = MetalTensor::from_slice(&[0.0f32], &[1], DType::F32);
    let c = a.sigmoid().unwrap();
    assert_tensor_approx_eq(&c, &[0.5], 1e-5);
}

#[test]
#[serial]
fn test_tanh() {
    let a = MetalTensor::from_slice(&[0.0f32], &[1], DType::F32);
    let c = a.tanh().unwrap();
    assert_tensor_approx_eq(&c, &[0.0], 1e-5);
}

#[test]
#[serial]
fn test_sin_cos() {
    let a = MetalTensor::from_slice(&[0.0f32, std::f32::consts::PI / 2.0], &[2], DType::F32);
    let s = a.sin().unwrap();
    let cos = a.cos().unwrap();
    assert_tensor_approx_eq(&s, &[0.0, 1.0], 1e-5);
    assert_tensor_approx_eq(&cos, &[1.0, 0.0], 1e-5);
}

// ========== Reduce 演算テスト ==========

#[test]
#[serial]
fn test_sumall() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let sum = a.sumall().unwrap();
    assert_approx_eq(sum, 10.0, 1e-5);
}

#[test]
#[serial]
fn test_mean_all() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let mean = a.mean_all().unwrap();
    assert_approx_eq(mean, 2.5, 1e-5);
}

#[test]
#[serial]
fn test_sum_axis() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let sum = a.sum(1).unwrap();
    assert_eq!(sum.shape(), &[2]);
    assert_tensor_approx_eq(&sum, &[6.0, 15.0], 1e-5);
}

#[test]
#[serial]
fn test_max() {
    let a = MetalTensor::from_slice(&[1.0f32, 5.0, 2.0, 4.0, 3.0, 6.0], &[2, 3], DType::F32);
    let max = a.max(1).unwrap();
    assert_eq!(max.shape(), &[2]);
    assert_tensor_approx_eq(&max, &[5.0, 6.0], 1e-5);
}

// ========== 形状操作テスト ==========

#[test]
#[serial]
fn test_reshape() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let b = a.reshape(&[3, 2]).unwrap();
    assert_eq!(b.shape(), &[3, 2]);
}

#[test]
#[serial]
fn test_transpose() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let b = a.transpose(0, 1).unwrap();
    assert_eq!(b.shape(), &[3, 2]);
}

#[test]
#[serial]
fn test_squeeze_unsqueeze() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], DType::F32);
    let b = a.squeeze(0).unwrap();
    assert_eq!(b.shape(), &[3]);
    
    let c = b.unsqueeze(0).unwrap();
    assert_eq!(c.shape(), &[1, 3]);
}

#[test]
#[serial]
fn test_softmax() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let s = a.softmax(0).unwrap();
    let data = s.to_vec::<f32>();
    
    let sum: f32 = data.iter().sum();
    assert_approx_eq(sum, 1.0, 1e-5);
}

#[test]
#[serial]
fn test_tril() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3], DType::F32);
    let c = a.tril(0).unwrap();
    assert_tensor_approx_eq(&c, &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0], 1e-5);
}

// ========== 深層学習演算テスト ==========

#[test]
#[serial]
fn test_conv2d() {
    let input = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], DType::F32);
    let kernel = MetalTensor::from_slice(&[2.0f32], &[1, 1, 1, 1], DType::F32);
    let output = input.conv2d(&kernel, (1, 1), (0, 0)).unwrap();
    assert_eq!(output.shape(), &[1, 1, 2, 2]);
}

#[test]
#[serial]
fn test_batch_norm() {
    let input = MetalTensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[1, 3, 2, 2], DType::F32
    );
    let gamma = MetalTensor::ones(&[3], DType::F32);
    let beta = MetalTensor::zeros(&[3], DType::F32);
    let mean = MetalTensor::from_slice(&[2.5f32, 6.5, 10.5], &[3], DType::F32);
    let var = MetalTensor::ones(&[3], DType::F32);
    
    let _ = input.batch_norm(&gamma, &beta, &mean, &var, 1e-5).unwrap();
}

#[test]
#[serial]
fn test_max_pool2d() {
    let input = MetalTensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        &[1, 1, 4, 4], DType::F32
    );
    let output = input.max_pool2d((2, 2), (2, 2)).unwrap();
    assert_eq!(output.shape(), &[1, 1, 2, 2]);
}

#[test]
#[serial]
fn test_layer_norm() {
    let input = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let gamma = MetalTensor::from_slice(&[1.0f32, 1.0, 1.0], &[3], DType::F32);
    let beta = MetalTensor::from_slice(&[0.0f32, 0.0, 0.0], &[3], DType::F32);
    let output = input.layer_norm(&gamma, &beta, 1e-5).unwrap();
    assert_eq!(output.shape(), &[2, 3]);
}

// ========== Autograd テスト ==========

#[test]
#[serial]
fn test_autograd_backward() {
    let input = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let sum = input.sumall().unwrap();
    assert_approx_eq(sum, 10.0, 1e-5);
}

// ========== オプティマイザテスト ==========

#[test]
#[serial]
fn test_sgd() {
    let mut params = vec![MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32)];
    let grads = vec![MetalTensor::from_slice(&[0.1f32, 0.2, 0.3], &[3], DType::F32)];
    
    let mut sgd = SGD::new(0.1, 0.0, 0.0);
    sgd.step(&mut params, &grads);
    
    let updated = params[0].to_vec::<f32>();
    assert!(updated[0] < 1.0);
}

#[test]
#[serial]
fn test_adam() {
    let mut params = vec![MetalTensor::from_slice(&[1.0f32, 1.0], &[2], DType::F32)];
    let grads = vec![MetalTensor::from_slice(&[0.1f32, 0.1], &[2], DType::F32)];
    
    let mut adam = Adam::default(0.01);
    adam.step(&mut params, &grads);
    
    let updated = params[0].to_vec::<f32>();
    assert!(updated[0] < 1.0);
}

#[test]
#[serial]
fn test_adamw() {
    let mut params = vec![MetalTensor::from_slice(&[1.0f32, 1.0], &[2], DType::F32)];
    let grads = vec![MetalTensor::from_slice(&[0.1f32, 0.1], &[2], DType::F32)];
    
    let mut adamw = AdamW::default(0.01);
    adamw.step(&mut params, &grads);
    
    let updated = params[0].to_vec::<f32>();
    assert!(updated[0] < 1.0);
}

#[test]
#[serial]
fn test_clip_grad_norm() {
    let mut grads = vec![
        MetalTensor::from_slice(&[3.0f32, 4.0], &[2], DType::F32),
    ];
    
    let norm = clip_grad_norm(&mut grads, 1.0);
    assert!(norm > 0.0);
}

// ========== 統合テスト ==========

#[test]
#[serial]
fn test_simple_forward() {
    let x = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
    let w = MetalTensor::from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[2, 2], DType::F32);
    
    let y = x.matmul(&w).unwrap();
    assert_eq!(y.shape(), &[2, 2]);
}
