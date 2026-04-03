//! 新規追加した CUDA 操作の網羅的テスト
//!
//! Phase A: 活性化, 論理演算, 損失関数, masked_fill, fill_, dot
//! Phase B: var, std, prod, cumsum, norm, topk
//! Phase C: NN層, LLM推論
//! Phase D: 融合カーネル

use serial_test::serial;
use tl_cuda::tensor::CudaTensor;
use tl_cuda::DType;

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
            "idx {}: got {} expected {} (eps={})",
            i,
            x,
            y,
            eps
        );
    }
}

// ====================================================================
// Phase A: 活性化関数
// ====================================================================

#[test]
#[serial]
fn test_leaky_relu() {
    let t = CudaTensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], DType::F32);
    let r = t.leaky_relu_impl(0.1).unwrap();
    approx_eq(&r.to_vec::<f32>(), &[-0.2, -0.1, 0.0, 1.0, 2.0], 1e-6);
}

#[test]
#[serial]
fn test_leaky_relu_different_slope() {
    let t = CudaTensor::from_slice(&[-4.0f32, 0.0, 3.0], &[3], DType::F32);
    let r = t.leaky_relu_impl(0.2).unwrap();
    approx_eq(&r.to_vec::<f32>(), &[-0.8, 0.0, 3.0], 1e-6);
}

#[test]
#[serial]
fn test_elu() {
    let t = CudaTensor::from_slice(&[-1.0f32, 0.0, 1.0], &[3], DType::F32);
    let r = t.elu_impl(1.0).unwrap();
    let expected_neg = 1.0 * ((-1.0f32).exp() - 1.0);
    approx_eq(&r.to_vec::<f32>(), &[expected_neg, 0.0, 1.0], 1e-5);
}

#[test]
#[serial]
fn test_elu_alpha_2() {
    let t = CudaTensor::from_slice(&[-1.0f32, 0.0, 2.0], &[3], DType::F32);
    let r = t.elu_impl(2.0).unwrap();
    let expected_neg = 2.0 * ((-1.0f32).exp() - 1.0);
    approx_eq(&r.to_vec::<f32>(), &[expected_neg, 0.0, 2.0], 1e-5);
}

#[test]
#[serial]
fn test_mish() {
    let t = CudaTensor::from_slice(&[0.0f32, 1.0], &[2], DType::F32);
    // mish(x) = x * tanh(softplus(x))
    let mish_0 = 0.0f32;
    let mish_1 = 1.0 * (1.0f32 + 1.0f32.exp()).ln().tanh();
    let r = t.mish_impl().unwrap();
    approx_eq(&r.to_vec::<f32>(), &[mish_0, mish_1], 1e-5);
}

#[test]
#[serial]
fn test_hardswish() {
    let t = CudaTensor::from_slice(&[-4.0f32, -3.0, 0.0, 3.0, 4.0], &[5], DType::F32);
    // hardswish(x) = x * relu6(x+3)/6
    let r = t.hardswish_impl().unwrap();
    let v = r.to_vec::<f32>();
    assert!((v[0] - 0.0).abs() < 1e-6, "hardswish(-4) should be 0");
    assert!((v[1] - 0.0).abs() < 1e-6, "hardswish(-3) should be 0");
    assert!((v[2] - 0.0).abs() < 1e-6, "hardswish(0) should be 0");
    assert!((v[3] - 3.0).abs() < 1e-6, "hardswish(3) should be 3");
    assert!((v[4] - 4.0).abs() < 1e-6, "hardswish(4) should be 4");
}

#[test]
#[serial]
fn test_hardsigmoid() {
    let t = CudaTensor::from_slice(&[-4.0f32, 0.0, 4.0], &[3], DType::F32);
    let r = t.hardsigmoid_impl().unwrap();
    let v = r.to_vec::<f32>();
    assert!((v[0] - 0.0).abs() < 1e-6, "hardsigmoid(-4) should be 0");
    assert!((v[1] - 0.5).abs() < 1e-6, "hardsigmoid(0) should be 0.5");
    assert!((v[2] - 1.0).abs() < 1e-6, "hardsigmoid(4) should be 1");
}

// ====================================================================
// Phase A: 論理演算
// ====================================================================

#[test]
#[serial]
fn test_logical_and() {
    let a = CudaTensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0], &[4], DType::F32);
    let b = CudaTensor::from_slice(&[1.0f32, 1.0, 0.0, 0.0], &[4], DType::F32);
    let r = a.logical_and_impl(&b).unwrap();
    approx_eq(&r.to_vec::<f32>(), &[1.0, 0.0, 0.0, 0.0], 1e-6);
}

#[test]
#[serial]
fn test_logical_or() {
    let a = CudaTensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0], &[4], DType::F32);
    let b = CudaTensor::from_slice(&[1.0f32, 1.0, 0.0, 0.0], &[4], DType::F32);
    let r = a.logical_or_impl(&b).unwrap();
    approx_eq(&r.to_vec::<f32>(), &[1.0, 1.0, 1.0, 0.0], 1e-6);
}

#[test]
#[serial]
fn test_logical_not() {
    let t = CudaTensor::from_slice(&[1.0f32, 0.0, 5.0, 0.0], &[4], DType::F32);
    let r = t.logical_not_impl().unwrap();
    approx_eq(&r.to_vec::<f32>(), &[0.0, 1.0, 0.0, 1.0], 1e-6);
}

// ====================================================================
// Phase A: 内積
// ====================================================================

#[test]
#[serial]
fn test_dot() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], DType::F32);
    let r = a.dot_impl(&b).unwrap();
    // 1*4 + 2*5 + 3*6 = 32
    let v = r.to_vec::<f32>();
    assert!(
        (v[0] - 32.0).abs() < 1e-4,
        "dot product should be 32, got {}",
        v[0]
    );
}

// ====================================================================
// Phase A: masked_fill / fill_
// ====================================================================

#[test]
#[serial]
fn test_masked_fill() {
    let x = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let mask = CudaTensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0], &[4], DType::F32);
    let r = x.masked_fill_scalar_impl(&mask, -999.0_f64).unwrap();
    approx_eq(&r.to_vec::<f32>(), &[-999.0, 2.0, -999.0, 4.0], 1e-6);
}

#[test]
#[serial]
fn test_fill() {
    let x = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let r = x.fill_impl(42.0).unwrap();
    approx_eq(&r.to_vec::<f32>(), &[42.0, 42.0, 42.0], 1e-6);
}

// ====================================================================
// Phase A: 損失関数
// ====================================================================

#[test]
#[serial]
fn test_mse_loss() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[1.5f32, 2.5, 3.5], &[3], DType::F32);
    let r = a.mse_loss_impl(&b).unwrap();
    // MSE = mean((0.5)^2, (0.5)^2, (0.5)^2) = 0.25
    let v = r.to_vec::<f32>();
    assert!(
        (v[0] - 0.25).abs() < 1e-4,
        "MSE should be 0.25, got {}",
        v[0]
    );
}

#[test]
#[serial]
fn test_l1_loss() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[1.5f32, 2.5, 2.0], &[3], DType::F32);
    let r = a.l1_loss_impl(&b).unwrap();
    // L1 = mean(0.5, 0.5, 1.0) = 2.0/3.0
    let v = r.to_vec::<f32>();
    assert!(
        (v[0] - 2.0 / 3.0).abs() < 1e-4,
        "L1 should be ~0.667, got {}",
        v[0]
    );
}

#[test]
#[serial]
fn test_bce_loss() {
    let pred = CudaTensor::from_slice(&[0.5f32, 0.5], &[2], DType::F32);
    let target = CudaTensor::from_slice(&[1.0f32, 0.0], &[2], DType::F32);
    let r = pred.bce_loss_impl(&target).unwrap();
    // BCE = -mean(1*ln(0.5) + 0*ln(0.5), 0*ln(0.5) + 1*ln(0.5)) = -ln(0.5) = 0.6931...
    let v = r.to_vec::<f32>();
    assert!(
        (v[0] - 0.6931).abs() < 1e-2,
        "BCE should be ~0.693, got {}",
        v[0]
    );
}

#[test]
#[serial]
fn test_nll_loss() {
    // NLL: -mean(log(a) * b)
    let a = CudaTensor::from_slice(&[0.5f32, 0.5], &[2], DType::F32);
    let b = CudaTensor::from_slice(&[1.0f32, 0.0], &[2], DType::F32);
    let r = a.nll_loss_impl(&b).unwrap();
    let v = r.to_vec::<f32>();
    // -mean(ln(0.5)*1 + ln(0.5)*0) = -(ln(0.5))/2 = 0.3466
    assert!(
        v[0].is_finite(),
        "NLL should produce a finite value, got {}",
        v[0]
    );
}

#[test]
#[serial]
fn test_kl_div_loss() {
    let p = CudaTensor::from_slice(&[0.5f32, 0.5], &[2], DType::F32);
    let q = CudaTensor::from_slice(&[0.5f32, 0.5], &[2], DType::F32);
    let r = p.kl_div_loss_impl(&q).unwrap();
    // KL(p||q) = 0 when p == q
    let v = r.to_vec::<f32>();
    assert!(
        (v[0]).abs() < 1e-4,
        "KL divergence of identical distributions should be 0, got {}",
        v[0]
    );
}

// ====================================================================
// Phase B: 縮約演算
// ====================================================================

#[test]
#[serial]
fn test_var() {
    let t = CudaTensor::from_slice(
        &[2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0],
        &[8],
        DType::F32,
    );
    let r = t.var_impl().unwrap();
    let v = r.to_vec::<f32>();
    // mean = 5.0, var = mean of squared diffs
    assert!(v[0] > 0.0, "variance should be positive, got {}", v[0]);
    assert!(
        (v[0] - 4.0).abs() < 1e-3,
        "variance should be ~4.0, got {}",
        v[0]
    );
}

#[test]
#[serial]
fn test_std() {
    let t = CudaTensor::from_slice(
        &[2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0],
        &[8],
        DType::F32,
    );
    let r = t.std_impl().unwrap();
    let v = r.to_vec::<f32>();
    // std = sqrt(var) = sqrt(4) = 2
    assert!(
        (v[0] - 2.0).abs() < 1e-3,
        "std should be ~2.0, got {}",
        v[0]
    );
}

#[test]
#[serial]
fn test_prod() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let r = t.prod_impl().unwrap();
    let v = r.to_vec::<f32>();
    assert!(
        (v[0] - 24.0).abs() < 1e-4,
        "product should be 24, got {}",
        v[0]
    );
}

#[test]
#[serial]
fn test_cumsum() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let r = t.cumsum_impl(0).unwrap();
    approx_eq(&r.to_vec::<f32>(), &[1.0, 3.0, 6.0, 10.0], 1e-5);
}

#[test]
#[serial]
fn test_cumsum_2d() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let r = t.cumsum_impl(1).unwrap();
    // dim=1: cumsum along columns within each row
    approx_eq(&r.to_vec::<f32>(), &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0], 1e-5);
}

#[test]
#[serial]
fn test_norm_l2() {
    let t = CudaTensor::from_slice(&[3.0f32, 4.0], &[2], DType::F32);
    let r = t.norm_impl(2.0).unwrap();
    let v = r.to_vec::<f32>();
    assert!(
        (v[0] - 5.0).abs() < 1e-4,
        "L2 norm should be 5.0, got {}",
        v[0]
    );
}

#[test]
#[serial]
fn test_norm_l1() {
    let t = CudaTensor::from_slice(&[-3.0f32, 4.0], &[2], DType::F32);
    let r = t.norm_impl(1.0).unwrap();
    let v = r.to_vec::<f32>();
    assert!(
        (v[0] - 7.0).abs() < 1e-4,
        "L1 norm should be 7.0, got {}",
        v[0]
    );
}

#[test]
#[serial]
fn test_topk() {
    let t = CudaTensor::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0], &[6], DType::F32);
    let r = t.topk_impl(3).unwrap();
    let v = r.to_vec::<f32>();
    assert_eq!(v.len(), 3, "topk(3) should return 3 elements");
    // top-3 values should be 9, 5, 4 (sorted desc)
    assert!((v[0] - 9.0).abs() < 1e-4, "1st should be 9, got {}", v[0]);
    assert!((v[1] - 5.0).abs() < 1e-4, "2nd should be 5, got {}", v[1]);
    assert!((v[2] - 4.0).abs() < 1e-4, "3rd should be 4, got {}", v[2]);
}

// ====================================================================
// Phase B: to_vec_f32, stack, broadcast_to
// ====================================================================

#[test]
#[serial]
fn test_to_vec_f32_impl() {
    let t = CudaTensor::from_slice(&[1.5f32, 2.5, 3.5], &[3], DType::F32);
    let v = t.to_vec_f32_impl().unwrap();
    approx_eq(&v, &[1.5, 2.5, 3.5], 1e-6);
}

#[test]
#[serial]
fn test_stack() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], DType::F32);
    let r = a.stack_impl(&b, 0).unwrap();
    assert_eq!(r.shape(), &[2, 3]);
    approx_eq(&r.to_vec::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-6);
}

// ====================================================================
// Phase C: NN 層
// ====================================================================

#[test]
#[serial]
fn test_group_norm() {
    // [1, 4, 1] input, 2 groups
    let x = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4, 1], DType::F32);
    let gamma = CudaTensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], DType::F32);
    let beta = CudaTensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], DType::F32);
    let r = x.group_norm_impl(&gamma, &beta, 2, 1e-5).unwrap();
    let v = r.to_vec::<f32>();
    assert_eq!(v.len(), 4);
    // group 0: [1,2] → normalized
    // group 1: [3,4] → normalized
    // Within each group, values should have mean ~0 and std ~1
    assert!((v[0] + v[1]).abs() < 1e-3, "group 0 mean should be ~0");
    assert!((v[2] + v[3]).abs() < 1e-3, "group 1 mean should be ~0");
}

#[test]
#[serial]
fn test_instance_norm() {
    let x = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2], DType::F32);
    let gamma = CudaTensor::from_slice(&[1.0f32, 1.0], &[2], DType::F32);
    let beta = CudaTensor::from_slice(&[0.0f32, 0.0], &[2], DType::F32);
    let r = x.instance_norm_impl(&gamma, &beta, 1e-5).unwrap();
    let v = r.to_vec::<f32>();
    assert_eq!(v.len(), 4);
    // Each channel normalized independently: mean ~0
    assert!((v[0] + v[1]).abs() < 1e-3, "channel 0 mean should be ~0");
}

#[test]
#[serial]
fn test_adaptive_avg_pool2d() {
    // [1, 1, 4, 4] → [1, 1, 2, 2]
    let x = CudaTensor::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
        &[1, 1, 4, 4],
        DType::F32,
    );
    let r = x.adaptive_avg_pool2d_impl((2, 2)).unwrap();
    assert_eq!(r.shape(), &[1, 1, 2, 2]);
    let v = r.to_vec::<f32>();
    // Top-left 2x2 avg = (1+2+5+6)/4 = 3.5
    assert!(
        (v[0] - 3.5).abs() < 1e-4,
        "top-left pool should be 3.5, got {}",
        v[0]
    );
}

#[test]
#[serial]
fn test_pad() {
    let x = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let r = x.pad_impl(&[1, 2], 0.0).unwrap();
    assert_eq!(r.shape(), &[6]); // 3 + 1 + 2 = 6
    approx_eq(&r.to_vec::<f32>(), &[0.0, 1.0, 2.0, 3.0, 0.0, 0.0], 1e-6);
}

#[test]
#[serial]
fn test_pad_with_value() {
    let x = CudaTensor::from_slice(&[1.0f32, 2.0], &[2], DType::F32);
    let r = x.pad_impl(&[1, 1], -1.0).unwrap();
    approx_eq(&r.to_vec::<f32>(), &[-1.0, 1.0, 2.0, -1.0], 1e-6);
}

#[test]
#[serial]
fn test_conv1d() {
    // [1, 1, 5] input, [1, 1, 3] kernel (no padding, stride 1)
    let x = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], DType::F32);
    let w = CudaTensor::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], DType::F32);
    let r = x.conv1d_impl(&w, 1, 0).unwrap();
    assert_eq!(r.shape(), &[1, 1, 3]);
    // [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
    approx_eq(&r.to_vec::<f32>(), &[6.0, 9.0, 12.0], 1e-5);
}

#[test]
#[serial]
fn test_conv_transpose2d() {
    // minimal test: [1,1,1,1] input, [1,1,2,2] weight
    let x = CudaTensor::from_slice(&[1.0f32], &[1, 1, 1, 1], DType::F32);
    let w = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], DType::F32);
    let r = x.conv_transpose2d_impl(&w, (1, 1), (0, 0)).unwrap();
    assert_eq!(r.shape(), &[1, 1, 2, 2]);
    approx_eq(&r.to_vec::<f32>(), &[1.0, 2.0, 3.0, 4.0], 1e-5);
}

#[test]
#[serial]
fn test_interpolate_nearest() {
    // [1,1,2,2] → [1,1,4,4] nearest
    let x = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], DType::F32);
    let r = x.interpolate_impl((4, 4), "nearest").unwrap();
    assert_eq!(r.shape(), &[1, 1, 4, 4]);
    let v = r.to_vec::<f32>();
    assert_eq!(v.len(), 16);
    // Each pixel should be repeated
    assert!((v[0] - 1.0).abs() < 1e-5, "top-left should be 1.0");
}

#[test]
#[serial]
fn test_linear_no_bias() {
    // [1,3] @ [3,2] = [1,2]
    let x = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], DType::F32);
    // weight [3,2]: col-major thinking → row0=[1,0], row1=[0,1], row2=[0,0]
    let w = CudaTensor::from_slice(
        &[
            1.0f32, 0.0, // row0
            0.0, 1.0, // row1
            0.0, 0.0,
        ], // row2
        &[3, 2],
        DType::F32,
    );
    let r = x.linear_impl(&w, None).unwrap();
    assert_eq!(r.shape(), &[1, 2]);
    // [1,2,3] @ [[1,0],[0,1],[0,0]] = [1, 2]
    approx_eq(&r.to_vec::<f32>(), &[1.0, 2.0], 1e-5);
}

#[test]
#[serial]
fn test_linear_with_bias() {
    let x = CudaTensor::from_slice(&[1.0f32, 2.0], &[1, 2], DType::F32);
    let w = CudaTensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0], &[2, 2], DType::F32);
    let bias = CudaTensor::from_slice(&[10.0f32, 20.0], &[2], DType::F32);
    let r = x.linear_impl(&w, Some(&bias)).unwrap();
    assert_eq!(r.shape(), &[1, 2]);
    // x @ w = [1, 2] @ [[1,0],[1,0]] = [3, 0] + bias [10, 20] = [13, 20]
    approx_eq(&r.to_vec::<f32>(), &[13.0, 20.0], 1e-5);
}

// ====================================================================
// Phase C: LLM 推論
// ====================================================================

#[test]
#[serial]
fn test_sdpa() {
    // Q, K, V: [1, 2, 4] (1 head, seq=2, dim=4)
    let q = CudaTensor::from_slice(
        &[1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        &[1, 2, 4],
        DType::F32,
    );
    let k = CudaTensor::from_slice(
        &[1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        &[1, 2, 4],
        DType::F32,
    );
    let v = CudaTensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 2, 4],
        DType::F32,
    );
    let r = q.sdpa_impl(&k, &v, None).unwrap();
    assert_eq!(r.shape(), &[1, 2, 4]);
    let out = r.to_vec::<f32>();
    assert_eq!(out.len(), 8);
    // Just check it's finite and reasonable
    for &val in &out {
        assert!(val.is_finite(), "SDPA output should be finite");
    }
}

#[test]
#[serial]
fn test_top_k_sample() {
    let logits = CudaTensor::from_slice(&[1.0f32, 5.0, 2.0, 3.0, 4.0], &[5], DType::F32);
    let r = logits.top_k_sample_impl(1).unwrap();
    let v = r.to_vec::<f32>();
    assert_eq!(v.len(), 1);
    // top-1 should return index of max = 1
    assert!(
        (v[0] - 1.0).abs() < 1e-4,
        "top-1 sample should return idx 1, got {}",
        v[0]
    );
}

#[test]
#[serial]
fn test_top_p_sample() {
    let logits = CudaTensor::from_slice(&[1.0f32, 100.0, 2.0], &[3], DType::F32);
    let r = logits.top_p_sample_impl(0.9).unwrap();
    let v = r.to_vec::<f32>();
    assert_eq!(v.len(), 1);
    // Should return index 1 since logit[1]=100 dominates
    assert!(
        (v[0] - 1.0).abs() < 1e-4,
        "top-p should return idx 1, got {}",
        v[0]
    );
}

#[test]
#[serial]
fn test_temperature_scale() {
    let logits = CudaTensor::from_slice(&[2.0f32, 4.0, 6.0], &[3], DType::F32);
    let r = logits.temperature_scale_impl(2.0).unwrap();
    approx_eq(&r.to_vec::<f32>(), &[1.0, 2.0, 3.0], 1e-5);
}

#[test]
#[serial]
fn test_repetition_penalty() {
    let logits = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], DType::F32);
    let tokens = CudaTensor::from_slice(&[1.0f32, 3.0], &[2], DType::F32);
    let r = logits.repetition_penalty_impl(&tokens, 2.0).unwrap();
    let v = r.to_vec::<f32>();
    assert_eq!(v.len(), 5);
    // Token indices 1 and 3 should have their logits penalized (divided by 2.0 since positive)
    assert!((v[0] - 1.0).abs() < 1e-4, "idx 0 unchanged");
    assert!((v[1] - 1.0).abs() < 1e-4, "idx 1 penalized: 2/2=1");
    assert!((v[2] - 3.0).abs() < 1e-4, "idx 2 unchanged");
    assert!((v[3] - 2.0).abs() < 1e-4, "idx 3 penalized: 4/2=2");
    assert!((v[4] - 5.0).abs() < 1e-4, "idx 4 unchanged");
}

// ====================================================================
// Phase D: 融合カーネル
// ====================================================================

#[test]
#[serial]
fn test_fused_add_relu() {
    use tl_backend::fused_ops::GpuFusedOps;
    let a = CudaTensor::from_slice(&[-1.0f32, 2.0, -3.0, 4.0], &[4], DType::F32);
    let b = CudaTensor::from_slice(&[0.5f32, -3.0, 5.0, -1.0], &[4], DType::F32);
    let r = a.fused_add_relu(&b).unwrap();
    // add: [-0.5, -1.0, 2.0, 3.0] → relu: [0.0, 0.0, 2.0, 3.0]
    approx_eq(&r.to_vec::<f32>(), &[0.0, 0.0, 2.0, 3.0], 1e-6);
}

#[test]
#[serial]
fn test_fused_silu_mul() {
    use tl_backend::fused_ops::GpuFusedOps;
    let x = CudaTensor::from_slice(&[0.0f32, 1.0, -1.0, 2.0], &[4], DType::F32);
    let up = CudaTensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], DType::F32);
    let r = x.fused_silu_mul(&up).unwrap();
    let v = r.to_vec::<f32>();
    // silu(0) * 1 = 0
    assert!((v[0] - 0.0).abs() < 1e-5, "fused_silu_mul(0,1) = 0");
    // silu(1) * 1 = 1 * sigmoid(1) = 0.7311
    assert!((v[1] - 0.7311).abs() < 1e-3, "fused_silu_mul(1,1) ≈ 0.73");
}

#[test]
#[serial]
fn test_fused_rms_norm() {
    use tl_backend::fused_ops::GpuFusedOps;
    let x = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], DType::F32);
    let w = CudaTensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], DType::F32);
    let r = x.fused_rms_norm(&w, 1e-5).unwrap();
    let v = r.to_vec::<f32>();
    assert_eq!(v.len(), 4);
    // RMS = sqrt(mean(1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // normalized: each / RMS
    let rms = (30.0f32 / 4.0).sqrt();
    for i in 0..4 {
        let expected = (i + 1) as f32 / rms;
        assert!(
            (v[i] - expected).abs() < 1e-3,
            "RMS norm elem {} expected {}, got {}",
            i,
            expected,
            v[i]
        );
    }
}

#[test]
#[serial]
fn test_fused_add_rms_norm() {
    use tl_backend::fused_ops::GpuFusedOps;
    let x = CudaTensor::from_slice(&[1.0f32, 2.0], &[1, 2], DType::F32);
    let residual = CudaTensor::from_slice(&[1.0f32, 0.0], &[1, 2], DType::F32);
    let w = CudaTensor::from_slice(&[1.0f32, 1.0], &[2], DType::F32);
    let r = x.fused_add_rms_norm(&residual, &w, 1e-5).unwrap();
    let v = r.to_vec::<f32>();
    assert_eq!(v.len(), 2);
    // x + residual = [2, 2], RMS = 2, normalized = [1, 1]
    approx_eq(&v, &[1.0, 1.0], 1e-3);
}

#[test]
#[serial]
fn test_fused_bias_gelu() {
    use tl_backend::fused_ops::GpuFusedOps;
    let x = CudaTensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], DType::F32);
    let bias = CudaTensor::from_slice(&[0.0f32, 0.0], &[2], DType::F32);
    let r = x.fused_bias_gelu(&bias).unwrap();
    // gelu(0+0) = 0
    approx_eq(&r.to_vec::<f32>(), &[0.0, 0.0, 0.0, 0.0], 1e-6);
}

// ====================================================================
// Phase C: dropout (確率的なのでスモークテストのみ)
// ====================================================================

#[test]
#[serial]
fn test_dropout_training_false() {
    let x = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let r = x.dropout_impl(0.5, false).unwrap();
    // training=false → そのまま返す
    approx_eq(&r.to_vec::<f32>(), &[1.0, 2.0, 3.0], 1e-6);
}

#[test]
#[serial]
fn test_dropout2d_training_false() {
    let x = CudaTensor::from_slice(&[1.0f32; 12], &[1, 3, 4], DType::F32);
    let r = x.dropout2d_impl(0.5, false).unwrap();
    approx_eq(&r.to_vec::<f32>(), &[1.0f32; 12], 1e-6);
}
