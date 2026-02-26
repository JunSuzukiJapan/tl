//! tl_cuda 包括的テスト

use serial_test::serial;
use tl_cuda::{clip_grad_norm, Adam, AdamW, SGD};
use tl_cuda::{CudaTensor, DType};

// ========== ヘルパー関数 ==========

fn assert_approx_eq(a: f32, b: f32, eps: f32) {
    assert!(
        (a - b).abs() < eps,
        "Expected {} ≈ {}, diff = {}",
        a,
        b,
        (a - b).abs()
    );
}

fn assert_tensor_approx_eq(a: &CudaTensor, b: &[f32], eps: f32) {
    let a_data = a.to_vec::<f32>();
    assert_eq!(
        a_data.len(),
        b.len(),
        "Length mismatch: {} vs {}",
        a_data.len(),
        b.len()
    );
    for (i, (&av, &bv)) in a_data.iter().zip(b.iter()).enumerate() {
        assert!((av - bv).abs() < eps, "At index {}: {} ≈ {}", i, av, bv);
    }
}

// ========== 基本演算テスト ==========

#[test]
#[serial]
fn test_tensor_creation() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.elem_count(), 4);
}

#[test]
#[serial]
fn test_zeros_ones() {
    let zeros = CudaTensor::zeros(&[3, 3], DType::F32);
    let ones = CudaTensor::ones(&[3, 3], DType::F32);

    let z_data = zeros.to_vec::<f32>();
    let o_data = ones.to_vec::<f32>();

    assert!(z_data.iter().all(|&x| x == 0.0));
    assert!(o_data.iter().all(|&x| x == 1.0));
}

#[test]
#[serial]
fn test_randn() {
    let t = CudaTensor::randn(&[100], DType::F32);
    let data = t.to_vec::<f32>();

    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 0.5, "Mean should be near 0: {}", mean);
}

// ========== 二項演算テスト ==========

#[test]
#[serial]
fn test_add() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], DType::F32);
    let c = a.add_impl(&b).unwrap();
    assert_tensor_approx_eq(&c, &[5.0, 7.0, 9.0], 1e-5);
}

#[test]
#[serial]
fn test_sub() {
    let a = CudaTensor::from_slice(&[5.0f32, 6.0, 7.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let c = a.sub_impl(&b).unwrap();
    assert_tensor_approx_eq(&c, &[4.0, 4.0, 4.0], 1e-5);
}

#[test]
#[serial]
fn test_mul() {
    let a = CudaTensor::from_slice(&[2.0f32, 3.0, 4.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[3.0f32, 4.0, 5.0], &[3], DType::F32);
    let c = a.mul_impl(&b).unwrap();
    assert_tensor_approx_eq(&c, &[6.0, 12.0, 20.0], 1e-5);
}

#[test]
#[serial]
fn test_div() {
    let a = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[2.0f32, 4.0, 5.0], &[3], DType::F32);
    let c = a.div_impl(&b).unwrap();
    assert_tensor_approx_eq(&c, &[5.0, 5.0, 6.0], 1e-5);
}

#[test]
#[serial]
fn test_matmul() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let b = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], DType::F32);
    let c = a.matmul_impl(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_tensor_approx_eq(&c, &[22.0, 28.0, 49.0, 64.0], 1e-5);
}

// ========== スカラー演算テスト ==========

#[test]
#[serial]
fn test_add_scalar() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let c = a.add_scalar_impl(10.0).unwrap();
    assert_tensor_approx_eq(&c, &[11.0, 12.0, 13.0], 1e-5);
}

#[test]
#[serial]
fn test_mul_scalar() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let c = a.mul_scalar_impl(2.0).unwrap();
    assert_tensor_approx_eq(&c, &[2.0, 4.0, 6.0], 1e-5);
}

#[test]
#[serial]
fn test_clamp() {
    let a = CudaTensor::from_slice(&[-1.0f32, 0.5, 2.0], &[3], DType::F32);
    let c = a.clamp_impl(0.0, 1.0).unwrap();
    assert_tensor_approx_eq(&c, &[0.0, 0.5, 1.0], 1e-5);
}

// ========== 単項演算テスト ==========

#[test]
#[serial]
fn test_neg() {
    let a = CudaTensor::from_slice(&[1.0f32, -2.0, 3.0], &[3], DType::F32);
    let c = a.neg_impl().unwrap();
    assert_tensor_approx_eq(&c, &[-1.0, 2.0, -3.0], 1e-5);
}

#[test]
#[serial]
fn test_abs() {
    let a = CudaTensor::from_slice(&[-1.0f32, 2.0, -3.0], &[3], DType::F32);
    let c = a.abs_impl().unwrap();
    assert_tensor_approx_eq(&c, &[1.0, 2.0, 3.0], 1e-5);
}

#[test]
#[serial]
fn test_exp() {
    let a = CudaTensor::from_slice(&[0.0f32, 1.0, 2.0], &[3], DType::F32);
    let c = a.exp_impl().unwrap();
    assert_tensor_approx_eq(&c, &[1.0, 2.7182817, 7.389056], 1e-4);
}

#[test]
#[serial]
fn test_log() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.7182817, 7.389056], &[3], DType::F32);
    let c = a.log_impl().unwrap();
    assert_tensor_approx_eq(&c, &[0.0, 1.0, 2.0], 1e-4);
}

#[test]
#[serial]
fn test_sqrt() {
    let a = CudaTensor::from_slice(&[1.0f32, 4.0, 9.0], &[3], DType::F32);
    let c = a.sqrt_impl().unwrap();
    assert_tensor_approx_eq(&c, &[1.0, 2.0, 3.0], 1e-5);
}

#[test]
#[serial]
fn test_relu() {
    let a = CudaTensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], &[4], DType::F32);
    let c = a.relu_impl().unwrap();
    assert_tensor_approx_eq(&c, &[0.0, 0.0, 1.0, 2.0], 1e-5);
}

#[test]
#[serial]
fn test_sigmoid() {
    let a = CudaTensor::from_slice(&[0.0f32], &[1], DType::F32);
    let c = a.sigmoid_impl().unwrap();
    assert_tensor_approx_eq(&c, &[0.5], 1e-5);
}

#[test]
#[serial]
fn test_tanh() {
    let a = CudaTensor::from_slice(&[0.0f32], &[1], DType::F32);
    let c = a.tanh_impl().unwrap();
    assert_tensor_approx_eq(&c, &[0.0], 1e-5);
}

#[test]
#[serial]
fn test_sin_cos() {
    let a = CudaTensor::from_slice(&[0.0f32, std::f32::consts::PI / 2.0], &[2], DType::F32);
    let s = a.sin_impl().unwrap();
    let cos = a.cos_impl().unwrap();
    assert_tensor_approx_eq(&s, &[0.0, 1.0], 1e-5);
    assert_tensor_approx_eq(&cos, &[1.0, 0.0], 1e-5);
}

// ========== Reduce 演算テスト ==========

#[test]
#[serial]
fn test_sumall() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let sum_t = a.sum_all_tensor_impl().unwrap();
    let sum = sum_t.to_vec::<f32>()[0];
    assert_approx_eq(sum, 10.0, 1e-5);
}

#[test]
#[serial]
fn test_mean_all() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let mean = a.mean_all_impl().unwrap();
    assert_approx_eq(mean, 2.5, 1e-5);
}

#[test]
#[serial]
fn test_sum_axis() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let sum = a.sum_impl(1).unwrap();
    assert_eq!(sum.shape(), &[2]);
    assert_tensor_approx_eq(&sum, &[6.0, 15.0], 1e-5);
}

#[test]
#[serial]
fn test_max() {
    let a = CudaTensor::from_slice(&[1.0f32, 5.0, 2.0, 4.0, 3.0, 6.0], &[2, 3], DType::F32);
    let max = a.max_impl(1).unwrap();
    assert_eq!(max.shape(), &[2]);
    assert_tensor_approx_eq(&max, &[5.0, 6.0], 1e-5);
}

// ========== 形状操作テスト ==========

#[test]
#[serial]
fn test_reshape() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let b = a.reshape_impl(&[3, 2]).unwrap();
    assert_eq!(b.shape(), &[3, 2]);
}

#[test]
#[serial]
fn test_transpose() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let b = a.transpose_impl(0, 1).unwrap();
    assert_eq!(b.shape(), &[3, 2]);
}

#[test]
#[serial]
fn test_squeeze_unsqueeze() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], DType::F32);
    let b = a.squeeze_impl(0).unwrap();
    assert_eq!(b.shape(), &[3]);

    let c = b.unsqueeze_impl(0).unwrap();
    assert_eq!(c.shape(), &[1, 3]);
}

#[test]
#[serial]
fn test_softmax() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let s = a.softmax_impl(0).unwrap();
    let data = s.to_vec::<f32>();

    let sum: f32 = data.iter().sum();
    assert_approx_eq(sum, 1.0, 1e-5);
}

#[test]
#[serial]
fn test_tril() {
    let a = CudaTensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        DType::F32,
    );
    let c = a.tril_impl(0).unwrap();
    assert_tensor_approx_eq(&c, &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0], 1e-5);
}

// ========== 深層学習演算テスト ==========

#[test]
#[serial]
fn test_conv2d() {
    let input = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], DType::F32);
    let kernel = CudaTensor::from_slice(&[2.0f32], &[1, 1, 1, 1], DType::F32);
    let output = input.conv2d_impl(&kernel, (1, 1), (0, 0)).unwrap();
    assert_eq!(output.shape(), &[1, 1, 2, 2]);
}

#[test]
#[serial]
fn test_batch_norm() {
    let input = CudaTensor::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[1, 3, 2, 2],
        DType::F32,
    );
    let gamma = CudaTensor::ones(&[3], DType::F32);
    let beta = CudaTensor::zeros(&[3], DType::F32);
    let mean = CudaTensor::from_slice(&[2.5f32, 6.5, 10.5], &[3], DType::F32);
    let var = CudaTensor::ones(&[3], DType::F32);

    let _ = input
        .batch_norm_impl(&gamma, &beta, &mean, &var, 1e-5)
        .unwrap();
}

#[test]
#[serial]
fn test_max_pool2d() {
    let input = CudaTensor::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
        &[1, 1, 4, 4],
        DType::F32,
    );
    let output = input.max_pool2d_impl((2, 2), (2, 2)).unwrap();
    assert_eq!(output.shape(), &[1, 1, 2, 2]);
}

#[test]
#[serial]
fn test_layer_norm() {
    let input = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let gamma = CudaTensor::from_slice(&[1.0f32, 1.0, 1.0], &[3], DType::F32);
    let beta = CudaTensor::from_slice(&[0.0f32, 0.0, 0.0], &[3], DType::F32);
    let output = input.layer_norm_impl(&gamma, &beta, 1e-5).unwrap();
    assert_eq!(output.shape(), &[2, 3]);
}

// ========== Autograd テスト ==========

#[test]
#[serial]
fn test_autograd_backward() {
    let input = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let sum_t = input.sum_all_tensor_impl().unwrap();
    let sum = sum_t.to_vec::<f32>()[0];
    assert_approx_eq(sum, 10.0, 1e-5);
}

// ========== オプティマイザテスト ==========

#[test]
#[serial]
fn test_sgd() {
    let mut params = vec![CudaTensor::from_slice(
        &[1.0f32, 2.0, 3.0],
        &[3],
        DType::F32,
    )];
    let grads = vec![CudaTensor::from_slice(
        &[0.1f32, 0.2, 0.3],
        &[3],
        DType::F32,
    )];

    let mut sgd = SGD::new(0.1, 0.0, 0.0);
    sgd.step(&mut params, &grads);

    let updated = params[0].to_vec::<f32>();
    assert!(updated[0] < 1.0);
}

#[test]
#[serial]
fn test_adam() {
    let mut params = vec![CudaTensor::from_slice(&[1.0f32, 1.0], &[2], DType::F32)];
    let grads = vec![CudaTensor::from_slice(&[0.1f32, 0.1], &[2], DType::F32)];

    let mut adam = Adam::default(0.01);
    adam.step(&mut params, &grads);

    let updated = params[0].to_vec::<f32>();
    assert!(updated[0] < 1.0);
}

#[test]
#[serial]
fn test_adamw() {
    let mut params = vec![CudaTensor::from_slice(&[1.0f32, 1.0], &[2], DType::F32)];
    let grads = vec![CudaTensor::from_slice(&[0.1f32, 0.1], &[2], DType::F32)];

    let mut adamw = AdamW::default(0.01);
    adamw.step(&mut params, &grads);

    let updated = params[0].to_vec::<f32>();
    assert!(updated[0] < 1.0);
}

#[test]
#[serial]
fn test_clip_grad_norm() {
    let mut grads = vec![CudaTensor::from_slice(&[3.0f32, 4.0], &[2], DType::F32)];

    let norm = clip_grad_norm(&mut grads, 1.0);
    assert!(norm > 0.0);
}

// ========== 統合テスト ==========

#[test]
#[serial]
fn test_simple_forward() {
    let x = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
    let w = CudaTensor::from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[2, 2], DType::F32);

    let y = x.matmul_impl(&w).unwrap();
    assert_eq!(y.shape(), &[2, 2]);
}
