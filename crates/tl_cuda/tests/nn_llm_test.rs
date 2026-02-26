//! Steps 9-10: NN 演算 + LLM 演算テスト

use serial_test::serial;
use tl_cuda::{CudaTensor, DType};

fn approx_eq(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len(), "len: {} vs {}", a.len(), b.len());
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

// ========== Conv2D ==========

#[test]
#[serial]
fn test_conv2d_basic() {
    // input: [1, 1, 3, 3], weight: [1, 1, 2, 2], stride=1, padding=0 → [1, 1, 2, 2]
    let input = CudaTensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
        DType::F32,
    );
    let weight = CudaTensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[1, 1, 2, 2], DType::F32);
    let r = input.conv2d_impl(&weight, (1, 1), (0, 0)).unwrap();
    assert_eq!(r.shape(), &[1, 1, 2, 2]);
    // r[0,0]=1*1+2*0+4*0+5*1=6, r[0,1]=2+6=8, r[1,0]=4+8=12, r[1,1]=5+9=14
    approx_eq(&r.to_vec::<f32>(), &[6.0, 8.0, 12.0, 14.0], 1e-5);
}

// ========== Batch Norm ==========

#[test]
#[serial]
fn test_batch_norm() {
    // input: [1, 2, 1, 1]
    let input = CudaTensor::from_slice(&[2.0f32, 4.0], &[1, 2, 1, 1], DType::F32);
    let gamma = CudaTensor::from_slice(&[1.0f32, 1.0], &[2], DType::F32);
    let beta = CudaTensor::from_slice(&[0.0f32, 0.0], &[2], DType::F32);
    let mean = CudaTensor::from_slice(&[2.0f32, 4.0], &[2], DType::F32);
    let var = CudaTensor::from_slice(&[1.0f32, 1.0], &[2], DType::F32);
    let r = input
        .batch_norm_impl(&gamma, &beta, &mean, &var, 1e-5)
        .unwrap();
    // (2-2)/sqrt(1+eps) ≈ 0, (4-4)/sqrt(1+eps) ≈ 0
    let v = r.to_vec::<f32>();
    assert!(v[0].abs() < 1e-3);
    assert!(v[1].abs() < 1e-3);
}

// ========== Layer Norm ==========

#[test]
#[serial]
fn test_layer_norm() {
    let input = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], DType::F32);
    let gamma = CudaTensor::from_slice(&[1.0f32, 1.0, 1.0], &[3], DType::F32);
    let beta = CudaTensor::from_slice(&[0.0f32, 0.0, 0.0], &[3], DType::F32);
    let r = input.layer_norm_impl(&gamma, &beta, 1e-5).unwrap();
    let v = r.to_vec::<f32>();
    // normalized: mean ≈ 0
    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
    assert!(mean.abs() < 1e-5, "mean = {}", mean);
}

// ========== Max Pool 2D ==========

#[test]
#[serial]
fn test_max_pool2d() {
    // input: [1, 1, 4, 4], kernel=2x2, stride=2 → [1, 1, 2, 2]
    let input = CudaTensor::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
        &[1, 1, 4, 4],
        DType::F32,
    );
    let r = input.max_pool2d_impl((2, 2), (2, 2)).unwrap();
    assert_eq!(r.shape(), &[1, 1, 2, 2]);
    approx_eq(&r.to_vec::<f32>(), &[6.0, 8.0, 14.0, 16.0], 1e-6);
}

// ========== Avg Pool 2D ==========

#[test]
#[serial]
fn test_avg_pool2d() {
    let input = CudaTensor::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
        &[1, 1, 4, 4],
        DType::F32,
    );
    let r = input.avg_pool2d_impl((2, 2), (2, 2)).unwrap();
    assert_eq!(r.shape(), &[1, 1, 2, 2]);
    // (1+2+5+6)/4=3.5, (3+4+7+8)/4=5.5, (9+10+13+14)/4=11.5, (11+12+15+16)/4=13.5
    approx_eq(&r.to_vec::<f32>(), &[3.5, 5.5, 11.5, 13.5], 1e-6);
}

// ========== Dropout ==========

#[test]
#[serial]
fn test_dropout_inference() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let r = t.dropout_impl(0.5, false).unwrap(); // not training → no change
    approx_eq(&r.to_vec::<f32>(), &[1.0, 2.0, 3.0], 1e-6);
}

// ========== RMS Norm ==========

#[test]
#[serial]
fn test_rms_norm() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], DType::F32);
    let r = t.rms_norm_impl(1e-5).unwrap();
    let v = r.to_vec::<f32>();
    // rms = sqrt(mean(x²)) = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.16
    let rms = (14.0f32 / 3.0).sqrt();
    approx_eq(&v, &[1.0 / rms, 2.0 / rms, 3.0 / rms], 1e-4);
}

// ========== Causal Mask ==========

#[test]
#[serial]
fn test_causal_mask() {
    let mask = CudaTensor::causal_mask_impl(3).unwrap();
    assert_eq!(mask.shape(), &[3, 3]);
    let v = mask.to_vec::<f32>();
    assert_eq!(v[0], 0.0); // [0,0]
    assert!(v[1].is_infinite() && v[1] < 0.0); // [0,1] = -inf
    assert_eq!(v[3], 0.0); // [1,0]
    assert_eq!(v[4], 0.0); // [1,1]
    assert!(v[5].is_infinite() && v[5] < 0.0); // [1,2] = -inf
}

// ========== RoPE ==========

#[test]
#[serial]
fn test_rope_cos_sin() {
    let (cos, sin) = CudaTensor::rope_cos_sin_impl(4, 8, 10000.0).unwrap();
    assert_eq!(cos.shape(), &[4, 8]);
    assert_eq!(sin.shape(), &[4, 8]);
    // pos=0 → all cos=1, sin=0
    let cos_data = cos.to_vec::<f32>();
    let sin_data = sin.to_vec::<f32>();
    for i in 0..8 {
        assert!(
            (cos_data[i] - 1.0).abs() < 1e-5,
            "cos[0,{}]={}",
            i,
            cos_data[i]
        );
        assert!(sin_data[i].abs() < 1e-5, "sin[0,{}]={}", i, sin_data[i]);
    }
}
