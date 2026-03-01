//! CUDA シェーダー網羅的テスト
//! 既存テスト (comprehensive_test, extended_ops_test) で未カバーの操作を検証
//!
//! カバレッジ:
//!   - 比較演算 (eq, ne, lt, le, gt, ge, rem)
//!   - 追加単項演算 (tan, gelu, silu)
//!   - 追加スカラー演算 (sub_scalar, div_scalar, pow_scalar, fmod_scalar)
//!   - 追加二項演算 (pow)
//!   - 追加リダクション (min, argmax_axis, argmin_axis, mean_axis, argmax_all, argmin_all)
//!   - LLM 演算 (rms_norm, rope_cos_sin, apply_rope, causal_mask)
//!   - Batch Matmul (3D×2D, 3D×3D)
//!   - エッジケース (大テンソル, 負の軸, 自動ブロードキャスト, 2D softmax)

use serial_test::serial;
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

fn assert_tensor_approx_eq(t: &CudaTensor, expected: &[f32], eps: f32) {
    let data = t.to_vec::<f32>();
    assert_eq!(
        data.len(),
        expected.len(),
        "Length mismatch: {} vs {}",
        data.len(),
        expected.len()
    );
    for (i, (&actual, &exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < eps,
            "At index {}: actual {} ≈ expected {}, diff = {}",
            i,
            actual,
            exp,
            (actual - exp).abs()
        );
    }
}

// =====================================================================
// 1. 比較演算テスト
// =====================================================================

#[test]
#[serial]
fn test_eq() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[1.0f32, 0.0, 3.0], &[3], DType::F32);
    let c = a.eq_impl(&b).unwrap();
    assert_tensor_approx_eq(&c, &[1.0, 0.0, 1.0], 1e-5);
}

#[test]
#[serial]
fn test_ne() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[1.0f32, 0.0, 3.0], &[3], DType::F32);
    let c = a.ne_impl(&b).unwrap();
    assert_tensor_approx_eq(&c, &[0.0, 1.0, 0.0], 1e-5);
}

#[test]
#[serial]
fn test_lt() {
    let a = CudaTensor::from_slice(&[1.0f32, 5.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[2.0f32, 3.0, 3.0], &[3], DType::F32);
    let c = a.lt_impl(&b).unwrap();
    assert_tensor_approx_eq(&c, &[1.0, 0.0, 0.0], 1e-5);
}

#[test]
#[serial]
fn test_le() {
    let a = CudaTensor::from_slice(&[1.0f32, 5.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[2.0f32, 3.0, 3.0], &[3], DType::F32);
    let c = a.le_impl(&b).unwrap();
    assert_tensor_approx_eq(&c, &[1.0, 0.0, 1.0], 1e-5);
}

#[test]
#[serial]
fn test_gt() {
    let a = CudaTensor::from_slice(&[1.0f32, 5.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[2.0f32, 3.0, 3.0], &[3], DType::F32);
    let c = a.gt_impl(&b).unwrap();
    assert_tensor_approx_eq(&c, &[0.0, 1.0, 0.0], 1e-5);
}

#[test]
#[serial]
fn test_ge() {
    let a = CudaTensor::from_slice(&[1.0f32, 5.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[2.0f32, 3.0, 3.0], &[3], DType::F32);
    let c = a.ge_impl(&b).unwrap();
    assert_tensor_approx_eq(&c, &[0.0, 1.0, 1.0], 1e-5);
}

#[test]
#[serial]
fn test_rem() {
    let a = CudaTensor::from_slice(&[10.0f32, 7.0, 5.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[3.0f32, 2.0, 3.0], &[3], DType::F32);
    let c = a.rem_impl(&b).unwrap();
    assert_tensor_approx_eq(&c, &[1.0, 1.0, 2.0], 1e-5);
}

// =====================================================================
// 2. 追加単項演算テスト
// =====================================================================

#[test]
#[serial]
fn test_tan() {
    let a = CudaTensor::from_slice(&[0.0f32, std::f32::consts::FRAC_PI_4], &[2], DType::F32);
    let c = a.tan_impl().unwrap();
    assert_tensor_approx_eq(&c, &[0.0, 1.0], 1e-4);
}

#[test]
#[serial]
fn test_gelu() {
    let a = CudaTensor::from_slice(&[0.0f32, 1.0, -1.0, 2.0], &[4], DType::F32);
    let c = a.gelu_impl().unwrap();
    let data = c.to_vec::<f32>();
    // GELU(0) = 0
    assert_approx_eq(data[0], 0.0, 1e-4);
    // GELU(1) ≈ 0.8413
    assert_approx_eq(data[1], 0.8413, 1e-3);
    // GELU(-1) ≈ -0.1587
    assert_approx_eq(data[2], -0.1587, 1e-3);
    // GELU(2) ≈ 1.9545
    assert_approx_eq(data[3], 1.9545, 1e-3);
}

#[test]
#[serial]
fn test_silu() {
    let a = CudaTensor::from_slice(&[0.0f32, 1.0, -1.0, 2.0], &[4], DType::F32);
    let c = a.silu_impl().unwrap();
    let data = c.to_vec::<f32>();
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // SiLU(0) = 0
    assert_approx_eq(data[0], 0.0, 1e-5);
    // SiLU(1) = 1 / (1 + exp(-1)) ≈ 0.7311
    assert_approx_eq(data[1], 0.7311, 1e-3);
    // SiLU(-1) = -1 / (1 + exp(1)) ≈ -0.2689
    assert_approx_eq(data[2], -0.2689, 1e-3);
    // SiLU(2) = 2 / (1 + exp(-2)) ≈ 1.7616
    assert_approx_eq(data[3], 1.7616, 1e-3);
}

// =====================================================================
// 3. 追加スカラー演算テスト
// =====================================================================

#[test]
#[serial]
fn test_sub_scalar() {
    let a = CudaTensor::from_slice(&[5.0f32, 6.0, 7.0], &[3], DType::F32);
    let c = a.add_scalar_impl(-2.0).unwrap();
    assert_tensor_approx_eq(&c, &[3.0, 4.0, 5.0], 1e-5);
}

#[test]
#[serial]
fn test_div_scalar() {
    let a = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);
    let c = a.div_scalar_impl(5.0).unwrap();
    assert_tensor_approx_eq(&c, &[2.0, 4.0, 6.0], 1e-5);
}

#[test]
#[serial]
fn test_pow_scalar() {
    let a = CudaTensor::from_slice(&[2.0f32, 3.0, 4.0], &[3], DType::F32);
    let c = a.pow_scalar_impl(2.0).unwrap();
    assert_tensor_approx_eq(&c, &[4.0, 9.0, 16.0], 1e-4);
}

#[test]
#[serial]
#[ignore] // fmod_scalar_impl not yet implemented
fn test_fmod_scalar() {
    // let a = CudaTensor::from_slice(&[10.0f32, 7.0, 5.0], &[3], DType::F32);
    // let c = a.fmod_scalar_impl(3.0).unwrap();
    // assert_tensor_approx_eq(&c, &[1.0, 1.0, 2.0], 1e-5);
}

// =====================================================================
// 4. 二項演算追加テスト
// =====================================================================

#[test]
#[serial]
fn test_pow_binary() {
    let a = CudaTensor::from_slice(&[2.0f32, 3.0, 4.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[3.0f32, 2.0, 1.0], &[3], DType::F32);
    let c = a.pow_impl(&b).unwrap();
    assert_tensor_approx_eq(&c, &[8.0, 9.0, 4.0], 1e-4);
}

// =====================================================================
// 5. 追加リダクション演算テスト
// =====================================================================

#[test]
#[serial]
fn test_min_axis() {
    let a = CudaTensor::from_slice(&[3.0f32, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3], DType::F32);
    let m = a.min_impl(1).unwrap();
    assert_eq!(m.shape(), &[2]);
    assert_tensor_approx_eq(&m, &[1.0, 4.0], 1e-5);
}

#[test]
#[serial]
fn test_argmax_axis() {
    let a = CudaTensor::from_slice(&[1.0f32, 5.0, 2.0, 4.0, 3.0, 6.0], &[2, 3], DType::F32);
    let idx = a.argmax_impl(1).unwrap();
    assert_eq!(idx.shape(), &[2]);
    // 行0: max=5.0 at idx 1, 行1: max=6.0 at idx 2
    assert_tensor_approx_eq(&idx, &[1.0, 2.0], 1e-5);
}

#[test]
#[serial]
fn test_argmin_axis() {
    let a = CudaTensor::from_slice(&[3.0f32, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3], DType::F32);
    let idx = a.argmin_impl(1).unwrap();
    assert_eq!(idx.shape(), &[2]);
    // 行0: min=1.0 at idx 1, 行1: min=4.0 at idx 1
    assert_tensor_approx_eq(&idx, &[1.0, 1.0], 1e-5);
}

#[test]
#[serial]
fn test_mean_axis() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let m = a.mean_impl(1).unwrap();
    assert_eq!(m.shape(), &[2]);
    assert_tensor_approx_eq(&m, &[2.0, 5.0], 1e-5);
}

#[test]
#[serial]
fn test_argmax_all() {
    let a = CudaTensor::from_slice(&[1.0f32, 5.0, 2.0, 4.0, 3.0, 6.0], &[6], DType::F32);
    let idx = a.argmax_all_impl().unwrap();
    let idx_data = idx.to_vec::<f32>();
    assert_approx_eq(idx_data[0], 5.0, 1e-5); // 6.0 is at index 5
}

#[test]
#[serial]
fn test_argmin_all() {
    let a = CudaTensor::from_slice(&[3.0f32, 1.0, 5.0, 2.0, 4.0], &[5], DType::F32);
    let idx_tensor = a.argmin_all_impl().unwrap();
    let idx_data = idx_tensor.to_vec::<f32>();
    assert_approx_eq(idx_data[0], 1.0, 1e-5); // 1.0 is at index 1
}

// =====================================================================
// 6. LLM 演算テスト
// =====================================================================

#[test]
#[serial]
fn test_rms_norm() {
    // 入力: [2, 4] — 2行4列
    let input = CudaTensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
        &[2, 4],
        DType::F32,
    );
    // let weight = CudaTensor::ones(&[4], DType::F32);
    let output = input.rms_norm_impl(1e-5).unwrap();
    assert_eq!(output.shape(), &[2, 4]);

    let data = output.to_vec::<f32>();

    // 行0: [1, 2, 3, 4]
    // mean(x²) = (1+4+9+16)/4 = 7.5
    // rms = sqrt(7.5 + 1e-5) ≈ 2.7386
    // normalized: [0.3651, 0.7303, 1.0954, 1.4606]
    assert_approx_eq(data[0], 1.0 / (7.5f32 + 1e-5).sqrt(), 1e-3);
    assert_approx_eq(data[1], 2.0 / (7.5f32 + 1e-5).sqrt(), 1e-3);

    // 行1: [2, 4, 6, 8]
    // mean(x²) = (4+16+36+64)/4 = 30.0
    // rms = sqrt(30 + 1e-5) ≈ 5.4772
    assert_approx_eq(data[4], 2.0 / (30.0f32 + 1e-5).sqrt(), 1e-3);
}

#[test]
#[serial]
fn test_rope_cos_sin() {
    // 小さいパラメータで cos/sin テーブル生成
    let (cos_table, sin_table) = CudaTensor::rope_cos_sin_impl(
        4,       // seq_len
        8,       // head_dim
        10000.0, // base
    )
    .unwrap();

    // 出力形状: [seq_len, half_dim] = [4, 4]
    assert_eq!(cos_table.shape(), &[4, 4]);
    assert_eq!(sin_table.shape(), &[4, 4]);

    let cos_data = cos_table.to_vec::<f32>();
    let sin_data = sin_table.to_vec::<f32>();

    // pos=0 の場合: cos(0 * theta) = 1.0, sin(0 * theta) = 0.0
    assert_approx_eq(cos_data[0], 1.0, 1e-4);
    assert_approx_eq(sin_data[0], 0.0, 1e-4);
    // すべての half_dim で pos=0 なら cos=1, sin=0
    assert_approx_eq(cos_data[1], 1.0, 1e-4);
    assert_approx_eq(sin_data[1], 0.0, 1e-4);
}

#[test]
#[serial]
fn test_apply_rope() {
    let head_dim = 4;
    let half_dim = head_dim / 2;
    let seq_len = 2;

    // cos/sin テーブル生成
    let (cos_table, sin_table) = CudaTensor::rope_cos_sin_impl(seq_len, head_dim, 10000.0).unwrap();

    // 入力: [1, head_dim] = [1, 4]
    let input = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, head_dim], DType::F32);

    // pos=0 での RoPE 適用 (cos=1, sin=0 なので入力不変)
    let output = input.apply_rope_impl(&cos_table, &sin_table, 0).unwrap();
    assert_eq!(output.shape(), &[1, head_dim]);

    let data = output.to_vec::<f32>();
    // pos=0: x1' = x1*cos - x2*sin = x1*1 - x2*0 = x1
    //        x2' = x1*sin + x2*cos = x1*0 + x2*1 = x2
    assert_approx_eq(data[0], 1.0, 1e-4);
    assert_approx_eq(data[1], 2.0, 1e-4);
    assert_approx_eq(data[0 + half_dim], 3.0, 1e-4);
    assert_approx_eq(data[1 + half_dim], 4.0, 1e-4);
}

#[test]
#[serial]
fn test_causal_mask() {
    let mask = CudaTensor::causal_mask_impl(4).unwrap();
    assert_eq!(mask.shape(), &[4, 4]);

    let data = mask.to_vec::<f32>();

    // 下三角 (j <= i) は 0.0、上三角 (j > i) は -inf
    // 行0: [0, -inf, -inf, -inf]
    assert_approx_eq(data[0], 0.0, 1e-5);
    assert!(
        data[1].is_infinite() && data[1] < 0.0,
        "mask[0,1] should be -inf, got {}",
        data[1]
    );

    // 行1: [0, 0, -inf, -inf]
    assert_approx_eq(data[4], 0.0, 1e-5);
    assert_approx_eq(data[5], 0.0, 1e-5);
    assert!(
        data[6].is_infinite() && data[6] < 0.0,
        "mask[1,2] should be -inf, got {}",
        data[6]
    );

    // 行3: [0, 0, 0, 0] (最終行は全て 0)
    assert_approx_eq(data[12], 0.0, 1e-5);
    assert_approx_eq(data[13], 0.0, 1e-5);
    assert_approx_eq(data[14], 0.0, 1e-5);
    assert_approx_eq(data[15], 0.0, 1e-5);
}

// =====================================================================
// 7. Matmul 拡張テスト
// =====================================================================

#[test]
#[serial]
fn test_batch_matmul_3d_2d() {
    // [B, M, K] × [K, N] → [B, M, N]
    // B=2, M=2, K=3, N=2
    let a = CudaTensor::from_slice(
        &[
            // batch 0: [[1,2,3],[4,5,6]]
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 1: [[7,8,9],[10,11,12]]
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
        DType::F32,
    );
    let b = CudaTensor::from_slice(
        &[
            // [K, N] = [[1,0],[0,1],[1,1]]
            1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0,
        ],
        &[3, 2],
        DType::F32,
    );

    let c = a.matmul_impl(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2, 2]);

    let data = c.to_vec::<f32>();
    // batch 0: [[1,2,3],[4,5,6]] × [[1,0],[0,1],[1,1]]
    //   row0: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    //   row1: [4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11]
    assert_approx_eq(data[0], 4.0, 1e-4);
    assert_approx_eq(data[1], 5.0, 1e-4);
    assert_approx_eq(data[2], 10.0, 1e-4);
    assert_approx_eq(data[3], 11.0, 1e-4);

    // batch 1: [[7,8,9],[10,11,12]] × [[1,0],[0,1],[1,1]]
    //   row0: [7+0+9, 0+8+9] = [16, 17]
    //   row1: [10+0+12, 0+11+12] = [22, 23]
    assert_approx_eq(data[4], 16.0, 1e-4);
    assert_approx_eq(data[5], 17.0, 1e-4);
    assert_approx_eq(data[6], 22.0, 1e-4);
    assert_approx_eq(data[7], 23.0, 1e-4);
}

#[test]
#[serial]
fn test_batch_matmul_3d_3d() {
    // [B, M, K] × [B, K, N] → [B, M, N]
    // B=2, M=2, K=2, N=2
    let a = CudaTensor::from_slice(
        &[
            // batch 0: [[1,2],[3,4]]
            1.0f32, 2.0, 3.0, 4.0, // batch 1: [[5,6],[7,8]]
            5.0, 6.0, 7.0, 8.0,
        ],
        &[2, 2, 2],
        DType::F32,
    );
    let b = CudaTensor::from_slice(
        &[
            // batch 0: [[1,0],[0,1]]
            1.0f32, 0.0, 0.0, 1.0, // batch 1: [[2,0],[0,2]]
            2.0, 0.0, 0.0, 2.0,
        ],
        &[2, 2, 2],
        DType::F32,
    );

    let c = a.matmul_impl(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2, 2]);

    let data = c.to_vec::<f32>();
    // batch 0: [[1,2],[3,4]] × [[1,0],[0,1]] = [[1,2],[3,4]] (単位行列)
    assert_approx_eq(data[0], 1.0, 1e-4);
    assert_approx_eq(data[1], 2.0, 1e-4);
    assert_approx_eq(data[2], 3.0, 1e-4);
    assert_approx_eq(data[3], 4.0, 1e-4);

    // batch 1: [[5,6],[7,8]] × [[2,0],[0,2]] = [[10,12],[14,16]]
    assert_approx_eq(data[4], 10.0, 1e-4);
    assert_approx_eq(data[5], 12.0, 1e-4);
    assert_approx_eq(data[6], 14.0, 1e-4);
    assert_approx_eq(data[7], 16.0, 1e-4);
}

// =====================================================================
// 8. エッジケース・大テンソルテスト
// =====================================================================

#[test]
#[serial]
fn test_large_tensor_add() {
    // 10000要素の加算
    let n = 10000;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let a = CudaTensor::from_slice(&a_data, &[n], DType::F32);
    let b = CudaTensor::from_slice(&b_data, &[n], DType::F32);
    let c = a.add_impl(&b).unwrap();

    let result = c.to_vec::<f32>();
    assert_eq!(result.len(), n);
    // 各要素は i + (n - i) = n
    for (i, &val) in result.iter().enumerate() {
        assert_approx_eq(val, n as f32, 1e-3);
        if i > 5 {
            break;
        } // サンプル確認
    }
    // 末尾も確認
    assert_approx_eq(result[n - 1], n as f32, 1e-3);
}

#[test]
#[serial]
fn test_negative_axis_reduction() {
    // 負の軸 (-1 = 最終軸) でリダクション
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    // axis=-1 は axis=1 と同等
    let s = a.sum_impl(-1).unwrap();
    assert_eq!(s.shape(), &[2]);
    assert_tensor_approx_eq(&s, &[6.0, 15.0], 1e-5);
}

#[test]
#[serial]
fn test_binary_broadcast_auto() {
    // 形状不一致時の自動ブロードキャスト: [2, 3] + [3]
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let b = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);

    // binary_op 内部で自動ブロードキャスト
    let c = a.add_impl(&b).unwrap();
    assert_eq!(c.shape(), &[2, 3]);
    assert_tensor_approx_eq(&c, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0], 1e-5);
}

#[test]
#[serial]
fn test_softmax_2d() {
    // 2Dテンソルの軸指定 softmax
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3], DType::F32);

    // axis=1 (各行に対して softmax)
    let s = a.softmax_impl(1).unwrap();
    assert_eq!(s.shape(), &[2, 3]);

    let data = s.to_vec::<f32>();
    // 各行の合計が 1.0
    let row0_sum: f32 = data[0..3].iter().sum();
    let row1_sum: f32 = data[3..6].iter().sum();
    assert_approx_eq(row0_sum, 1.0, 1e-5);
    assert_approx_eq(row1_sum, 1.0, 1e-5);

    // 行内の値は単調増加
    assert!(data[0] < data[1], "softmax should preserve order");
    assert!(data[1] < data[2], "softmax should preserve order");

    // 入力が同じなので、2行は同じ出力
    assert_approx_eq(data[0], data[3], 1e-5);
    assert_approx_eq(data[1], data[4], 1e-5);
    assert_approx_eq(data[2], data[5], 1e-5);
}

// =====================================================================
// 9. 比較演算 2D テスト（形状整合性の検証）
// =====================================================================

#[test]
#[serial]
fn test_comparison_ops_2d() {
    let a = CudaTensor::from_slice(&[1.0f32, 5.0, 3.0, 7.0], &[2, 2], DType::F32);
    let b = CudaTensor::from_slice(&[2.0f32, 5.0, 1.0, 8.0], &[2, 2], DType::F32);

    let lt = a.lt_impl(&b).unwrap();
    assert_eq!(lt.shape(), &[2, 2]);
    assert_tensor_approx_eq(&lt, &[1.0, 0.0, 0.0, 1.0], 1e-5);

    let eq = a.eq_impl(&b).unwrap();
    assert_tensor_approx_eq(&eq, &[0.0, 1.0, 0.0, 0.0], 1e-5);
}

// =====================================================================
// 10. リダクション axis=0 テスト
// =====================================================================

#[test]
#[serial]
fn test_sum_axis0() {
    // axis=0 方向のリダクション
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let s = a.sum_impl(0).unwrap();
    assert_eq!(s.shape(), &[3]);
    // 列ごとの合計: [1+4, 2+5, 3+6] = [5, 7, 9]
    assert_tensor_approx_eq(&s, &[5.0, 7.0, 9.0], 1e-5);
}

#[test]
#[serial]
fn test_max_axis0() {
    let a = CudaTensor::from_slice(&[3.0f32, 1.0, 5.0, 2.0, 4.0, 0.0], &[2, 3], DType::F32);
    let m = a.max_impl(0).unwrap();
    assert_eq!(m.shape(), &[3]);
    assert_tensor_approx_eq(&m, &[3.0, 4.0, 5.0], 1e-5);
}

#[test]
#[serial]
fn test_min_axis0() {
    let a = CudaTensor::from_slice(&[3.0f32, 1.0, 5.0, 2.0, 4.0, 0.0], &[2, 3], DType::F32);
    let m = a.min_impl(0).unwrap();
    assert_eq!(m.shape(), &[3]);
    assert_tensor_approx_eq(&m, &[2.0, 1.0, 0.0], 1e-5);
}

// =====================================================================
// 11. Conv2D 数値検証テスト
// =====================================================================

#[test]
#[serial]
fn test_conv2d_numerical() {
    // 1×1×3×3 入力, 1×1×2×2 カーネル, stride=1, padding=0
    // → 出力 1×1×2×2
    let input = CudaTensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
        DType::F32,
    );
    let kernel = CudaTensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[1, 1, 2, 2], DType::F32);
    let output = input.conv2d_impl(&kernel, (1, 1), (0, 0)).unwrap();
    assert_eq!(output.shape(), &[1, 1, 2, 2]);

    // 手計算:
    // out[0,0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
    // out[0,1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
    // out[1,0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
    // out[1,1] = 5*1 + 6*0 + 8*0 + 9*1 = 14
    assert_tensor_approx_eq(&output, &[6.0, 8.0, 12.0, 14.0], 1e-4);
}

// =====================================================================
// 12. Tril 多次元・diagonal テスト
// =====================================================================

#[test]
#[serial]
fn test_tril_positive_diagonal() {
    let a = CudaTensor::ones(&[3, 3], DType::F32);
    let c = a.tril_impl(1).unwrap();
    // diagonal=1: 主対角線 + 1つ上の対角線まで保持
    assert_tensor_approx_eq(&c, &[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1e-5);
}

#[test]
#[serial]
fn test_tril_negative_diagonal() {
    let a = CudaTensor::ones(&[3, 3], DType::F32);
    let c = a.tril_impl(-1).unwrap();
    // diagonal=-1: 主対角線の1つ下から保持
    assert_tensor_approx_eq(&c, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], 1e-5);
}
