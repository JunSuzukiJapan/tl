//! 自動カーネル融合テスト

use tl_metal::{MetalTensor, DType};
use tl_metal::fusion::LazyTensor;
use serial_test::serial;

fn assert_approx(got: &[f32], expected: &[f32], eps: f32, label: &str) {
    assert_eq!(got.len(), expected.len(), "{}: len mismatch {} vs {}", label, got.len(), expected.len());
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < eps, "{}: [{}] got {} expected {} diff {}", label, i, g, e, (g - e).abs());
    }
}

/// 基本: add + relu の融合
#[test]
#[serial]
fn test_fusion_add_relu() {
    let a = MetalTensor::from_slice(&[-1.0f32, 2.0, -3.0, 4.0], &[4], DType::F32);
    let b = MetalTensor::from_slice(&[0.5f32, -3.0, 5.0, -1.0], &[4], DType::F32);

    // 融合版（LazyTensor）
    let la = LazyTensor::from_tensor(a.clone());
    let lb = LazyTensor::from_tensor(b.clone());
    let fused = la.add(&lb).relu();
    let result = fused.materialize();
    let fused_data = result.to_vec::<f32>();

    // 非融合版
    let expected = a.add(&b).unwrap().relu_impl().unwrap();
    let expected_data = expected.to_vec::<f32>();

    assert_approx(&fused_data, &expected_data, 1e-5, "add_relu");
}

/// silu + mul 融合（SwiGLU パターン）
#[test]
#[serial]
fn test_fusion_silu_mul() {
    let gate = MetalTensor::from_slice(&[1.0f32, -1.0, 2.0, 0.0], &[4], DType::F32);
    let up = MetalTensor::from_slice(&[2.0f32, 3.0, 1.0, 5.0], &[4], DType::F32);

    // 融合版
    let lg = LazyTensor::from_tensor(gate.clone());
    let lu = LazyTensor::from_tensor(up.clone());
    let fused = lg.silu().mul(&lu);
    let result = fused.materialize();
    let fused_data = result.to_vec::<f32>();

    // 非融合版
    let silu_gate = gate.silu_impl().unwrap();
    let expected = silu_gate.mul(&up).unwrap();
    let expected_data = expected.to_vec::<f32>();

    assert_approx(&fused_data, &expected_data, 1e-4, "silu_mul");
}

/// 長いチェーン: add → mul_scalar → relu → neg
#[test]
#[serial]
fn test_fusion_long_chain() {
    let a = MetalTensor::from_slice(&[1.0f32, -2.0, 3.0, -4.0], &[4], DType::F32);
    let b = MetalTensor::from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[4], DType::F32);

    // 融合版
    let la = LazyTensor::from_tensor(a.clone());
    let lb = LazyTensor::from_tensor(b.clone());
    let fused = la.add(&lb).mul_scalar(2.0).relu().neg();
    let result = fused.materialize();
    let fused_data = result.to_vec::<f32>();

    // 非融合版: neg(relu((a+b)*2))
    // a+b = [1.5, -1.5, 3.5, -3.5]
    // *2  = [3.0, -3.0, 7.0, -7.0]
    // relu = [3.0, 0.0, 7.0, 0.0]
    // neg  = [-3.0, 0.0, -7.0, 0.0]
    assert_approx(&fused_data, &[-3.0, 0.0, -7.0, 0.0], 1e-5, "long_chain");
}

/// パイプラインキャッシュヒット
#[test]
#[serial]
fn test_fusion_cache_hit() {
    // 初回
    let a1 = MetalTensor::from_slice(&[1.0f32, 2.0], &[2], DType::F32);
    let b1 = MetalTensor::from_slice(&[3.0f32, 4.0], &[2], DType::F32);
    let la1 = LazyTensor::from_tensor(a1);
    let lb1 = LazyTensor::from_tensor(b1);
    let r1 = la1.add(&lb1).relu().materialize();

    let cache_before = tl_metal::fusion::cache::get_cache().len();

    // 2回目（同じパターン → キャッシュヒット）
    let a2 = MetalTensor::from_slice(&[10.0f32, 20.0], &[2], DType::F32);
    let b2 = MetalTensor::from_slice(&[30.0f32, 40.0], &[2], DType::F32);
    let la2 = LazyTensor::from_tensor(a2);
    let lb2 = LazyTensor::from_tensor(b2);
    let r2 = la2.add(&lb2).relu().materialize();

    let cache_after = tl_metal::fusion::cache::get_cache().len();

    // キャッシュサイズが増えていない = ヒット
    assert_eq!(cache_before, cache_after, "Cache should hit for same pattern");

    // 結果も正しい
    let data1 = r1.to_vec::<f32>();
    let data2 = r2.to_vec::<f32>();
    assert_approx(&data1, &[4.0, 6.0], 1e-5, "cache1");
    assert_approx(&data2, &[40.0, 60.0], 1e-5, "cache2");
}

/// 大きなテンソル
#[test]
#[serial]
fn test_fusion_large_tensor() {
    let n = 4096;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 2048.0) * 0.001).collect();
    let t = MetalTensor::from_slice(&data, &[n], DType::F32);

    let lt = LazyTensor::from_tensor(t.clone());
    let fused = lt.relu().mul_scalar(3.0);
    let result = fused.materialize();
    let fused_data = result.to_vec::<f32>();

    // 手動計算
    let expected: Vec<f32> = data.iter().map(|&x| x.max(0.0) * 3.0).collect();
    assert_approx(&fused_data, &expected, 1e-4, "large");
}

/// リーフのまま materialize
#[test]
#[serial]
fn test_fusion_leaf_only() {
    let t = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let lt = LazyTensor::from_tensor(t);
    let result = lt.materialize();
    let data = result.to_vec::<f32>();
    assert_approx(&data, &[1.0, 2.0, 3.0], 1e-5, "leaf_only");
}
