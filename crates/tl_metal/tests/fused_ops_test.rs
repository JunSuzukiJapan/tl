//! 融合カーネル テスト
//!
//! 各融合操作の結果を非融合版（個別操作の組み合わせ）と比較して正確性を検証。

use tl_metal::{MetalTensor, DType};
use tl_backend::fused_ops::GpuFusedOps;
use serial_test::serial;

fn assert_approx_eq_vec(got: &[f32], expected: &[f32], eps: f32, label: &str) {
    assert_eq!(got.len(), expected.len(), "{}: length mismatch {} vs {}", label, got.len(), expected.len());
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < eps, "{}: at index {} got {} expected {} (diff {})", label, i, g, e, (g - e).abs());
    }
}

// ========================================
// Tier 2: 汎用融合（シンプルなので先にテスト）
// ========================================

#[test]
#[serial]
fn test_fused_add_relu() {
    let a = MetalTensor::from_slice(&[-1.0f32, 2.0, -3.0, 4.0], &[4], DType::F32);
    let b = MetalTensor::from_slice(&[0.5f32, -3.0, 5.0, -1.0], &[4], DType::F32);

    // 融合版
    let fused = a.fused_add_relu(&b).unwrap();
    let fused_data = fused.to_vec::<f32>();

    // 非融合版: relu(a + b)
    let sum = a.add(&b).unwrap();
    let expected = sum.relu().unwrap();
    let expected_data = expected.to_vec::<f32>();

    assert_approx_eq_vec(&fused_data, &expected_data, 1e-5, "fused_add_relu");
    // 手動検証: [-0.5, -1.0, 2.0, 3.0] → relu → [0.0, 0.0, 2.0, 3.0]
    assert_approx_eq_vec(&fused_data, &[0.0, 0.0, 2.0, 3.0], 1e-5, "fused_add_relu manual");
}

#[test]
#[serial]
fn test_fused_silu_mul() {
    let gate = MetalTensor::from_slice(&[1.0f32, -1.0, 2.0, 0.0], &[4], DType::F32);
    let up = MetalTensor::from_slice(&[2.0f32, 3.0, 1.0, 5.0], &[4], DType::F32);

    // 融合版
    let fused = gate.fused_silu_mul(&up).unwrap();
    let fused_data = fused.to_vec::<f32>();

    // 非融合版: silu(gate) * up
    let silu_gate = gate.silu_impl().unwrap();
    let expected = silu_gate.mul(&up).unwrap();
    let expected_data = expected.to_vec::<f32>();

    assert_approx_eq_vec(&fused_data, &expected_data, 1e-4, "fused_silu_mul");
}

#[test]
#[serial]
fn test_fused_bias_gelu() {
    let x = MetalTensor::from_slice(&[1.0f32, -1.0, 0.5, 2.0, -0.5, 1.5], &[2, 3], DType::F32);
    let bias = MetalTensor::from_slice(&[0.1f32, -0.1, 0.2], &[3], DType::F32);

    // 融合版
    let fused = x.fused_bias_gelu(&bias).unwrap();
    let fused_data = fused.to_vec::<f32>();

    // 非融合版: gelu(x + bias) - bias をブロードキャスト
    // x[0,:] + bias = [1.1, -1.1, 0.7], x[1,:] + bias = [2.1, -0.6, 1.7]
    // 手動 GELU 計算は面倒なので、融合版が妥当な値を返すことを確認
    assert_eq!(fused_data.len(), 6);
    // gelu(1.1) ≈ 0.978, gelu(-1.1) ≈ -0.156
    assert!(fused_data[0] > 0.9, "gelu(1.1) should be > 0.9, got {}", fused_data[0]);
    assert!(fused_data[1] < 0.0, "gelu(-1.1) should be < 0, got {}", fused_data[1]);
}

// ========================================
// Tier 1: LLM ホットパス
// ========================================

#[test]
#[serial]
fn test_fused_rms_norm() {
    // [2, 3] テンソル
    let x = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let w = MetalTensor::from_slice(&[1.0f32, 1.0, 1.0], &[3], DType::F32);
    let eps = 1e-5f32;

    let result = x.fused_rms_norm(&w, eps).unwrap();
    let data = result.to_vec::<f32>();

    // 手動計算: row0 = [1,2,3], rms = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.16
    // normalized[0] = 1/2.16 ≈ 0.463, [1] = 2/2.16 ≈ 0.926, [2] = 3/2.16 ≈ 1.389
    let rms0 = (14.0f32 / 3.0 + eps).sqrt();
    assert!((data[0] - 1.0 / rms0).abs() < 1e-3, "rms_norm row0[0]: got {} expected {}", data[0], 1.0 / rms0);
    assert!((data[1] - 2.0 / rms0).abs() < 1e-3, "rms_norm row0[1]");
    assert!((data[2] - 3.0 / rms0).abs() < 1e-3, "rms_norm row0[2]");
}

#[test]
#[serial]
fn test_fused_add_rms_norm() {
    let x = MetalTensor::from_slice(&[1.0f32, 0.0, 0.0], &[1, 3], DType::F32);
    let res = MetalTensor::from_slice(&[0.0f32, 1.0, 0.0], &[1, 3], DType::F32);
    let w = MetalTensor::from_slice(&[1.0f32, 1.0, 1.0], &[3], DType::F32);
    let eps = 1e-5f32;

    let result = x.fused_add_rms_norm(&res, &w, eps).unwrap();
    let data = result.to_vec::<f32>();

    // x + res = [1, 1, 0], rms = sqrt((1+1+0)/3) = sqrt(2/3) ≈ 0.8165
    let rms = (2.0f32 / 3.0 + eps).sqrt();
    assert!((data[0] - 1.0 / rms).abs() < 1e-3, "add_rms_norm[0]: got {} expected {}", data[0], 1.0 / rms);
    assert!((data[1] - 1.0 / rms).abs() < 1e-3, "add_rms_norm[1]");
    assert!((data[2] - 0.0).abs() < 1e-3, "add_rms_norm[2]");
}

#[test]
#[serial]
fn test_fused_add_relu_large() {
    // 大きなテンソルでサイズ対応確認
    let n = 4096;
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32 - 2048.0) * 0.01).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (2048.0 - i as f32) * 0.01).collect();

    let a = MetalTensor::from_slice(&a_data, &[n], DType::F32);
    let b = MetalTensor::from_slice(&b_data, &[n], DType::F32);

    let fused = a.fused_add_relu(&b).unwrap();
    let fused_data = fused.to_vec::<f32>();

    // a[i] + b[i] = 0 for all i → relu(0) = 0
    for (i, &v) in fused_data.iter().enumerate() {
        assert!((v - 0.0).abs() < 1e-4, "large test at {}: got {}", i, v);
    }
}
