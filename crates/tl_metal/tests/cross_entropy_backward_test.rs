//! Cross Entropy Forward + Backward 網羅的テスト
//!
//! Metal GPU カーネルの数値的正しさを CPU 基準値と比較して検証する。
//! - forward: fused softmax + NLL loss カーネル
//! - backward: fused softmax + gradient カーネル
//! - ignore_index, バッチサイズ 1, 大バッチ, edge cases
//!
//! IMPORTANT: from_slice に渡すリテラルは必ず f32 suffix を付けること。
//! &[1.0] は f64 に推論され、バッファオーバーフローの原因になる。

use tl_metal::{MetalTensor, DType};
use serial_test::serial;

// ========== ヘルパー ==========

fn assert_approx(a: f32, b: f32, eps: f32, msg: &str) {
    assert!(
        (a - b).abs() < eps,
        "{}: expected {} ≈ {}, diff = {}",
        msg, b, a, (a - b).abs()
    );
}

fn assert_vec_approx(actual: &[f32], expected: &[f32], eps: f32, msg: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch {} vs {}", msg, actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < eps,
            "{} [{}]: expected {} ≈ {}, diff = {}",
            msg, i, e, a, (a - e).abs()
        );
    }
}

/// CPU で cross entropy forward を計算（期待値の基準）
fn cpu_cross_entropy_forward(logits: &[f32], targets: &[f32], batch_size: usize, num_classes: usize) -> f32 {
    let mut loss = 0.0f32;
    for i in 0..batch_size {
        let target_idx = targets[i] as i64;
        if target_idx < 0 || target_idx as usize >= num_classes {
            continue;
        }
        let offset = i * num_classes;
        let row = &logits[offset..offset + num_classes];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
        let log_softmax = (row[target_idx as usize] - max_val) - sum_exp.ln();
        loss -= log_softmax;
    }
    loss / batch_size as f32
}

/// CPU で cross entropy backward を計算（期待値の基準）
fn cpu_cross_entropy_backward(
    logits: &[f32],
    targets: &[f32],
    grad_output: f32,
    batch_size: usize,
    num_classes: usize,
) -> Vec<f32> {
    let mut valid_count = 0usize;
    for i in 0..batch_size {
        if (targets[i] as i64) >= 0 {
            valid_count += 1;
        }
    }
    if valid_count == 0 { valid_count = 1; }

    let mut grad = vec![0.0f32; batch_size * num_classes];
    for i in 0..batch_size {
        let target_idx = targets[i] as i64;
        let offset = i * num_classes;

        if target_idx < 0 {
            continue;
        }

        let row = &logits[offset..offset + num_classes];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();

        let inv_valid = grad_output / valid_count as f32;
        for j in 0..num_classes {
            let softmax_val = (row[j] - max_val).exp() / sum_exp;
            let one_hot = if j == target_idx as usize { 1.0 } else { 0.0 };
            grad[offset + j] = (softmax_val - one_hot) * inv_valid;
        }
    }
    grad
}

// =====================================================================
// 1. Cross Entropy Forward テスト
// =====================================================================

#[test]
#[serial]
fn test_ce_forward_basic() {
    let logits = MetalTensor::from_slice(
        &[1.0f32, 2.0, 3.0,  4.0, 5.0, 6.0],
        &[2, 3],
        DType::F32,
    );
    let targets = MetalTensor::from_slice(&[0.0f32, 2.0], &[2], DType::F32);

    let loss = logits.cross_entropy_impl(&targets).unwrap();
    let loss_val = loss.to_vec::<f32>()[0];

    let expected = cpu_cross_entropy_forward(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[0.0, 2.0],
        2, 3,
    );
    assert_approx(loss_val, expected, 1e-4, "CE forward basic");
}

#[test]
#[serial]
fn test_ce_forward_batch_1() {
    let logits = MetalTensor::from_slice(&[2.0f32, 0.5, -1.0, 3.0], &[1, 4], DType::F32);
    let targets = MetalTensor::from_slice(&[3.0f32], &[1], DType::F32);

    let loss = logits.cross_entropy_impl(&targets).unwrap();
    let loss_val = loss.to_vec::<f32>()[0];

    let expected = cpu_cross_entropy_forward(
        &[2.0, 0.5, -1.0, 3.0],
        &[3.0],
        1, 4,
    );
    assert_approx(loss_val, expected, 1e-4, "CE forward batch=1");
}

#[test]
#[serial]
fn test_ce_forward_large_batch() {
    let batch_size = 8;
    let num_classes = 5;
    let mut logits_data = Vec::new();
    for i in 0..batch_size {
        for j in 0..num_classes {
            logits_data.push((i * num_classes + j) as f32 * 0.1 - 2.0);
        }
    }
    let targets_data: Vec<f32> = (0..batch_size).map(|i| (i % num_classes) as f32).collect();

    let logits = MetalTensor::from_slice(&logits_data, &[batch_size, num_classes], DType::F32);
    let targets = MetalTensor::from_slice(&targets_data, &[batch_size], DType::F32);

    let loss = logits.cross_entropy_impl(&targets).unwrap();
    let loss_val = loss.to_vec::<f32>()[0];

    let expected = cpu_cross_entropy_forward(
        &logits_data, &targets_data, batch_size, num_classes,
    );
    assert_approx(loss_val, expected, 1e-3, "CE forward large batch");
}

#[test]
#[serial]
fn test_ce_forward_uniform_logits() {
    let logits = MetalTensor::from_slice(
        &[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        &[2, 5],
        DType::F32,
    );
    let targets = MetalTensor::from_slice(&[0.0f32, 4.0], &[2], DType::F32);

    let loss = logits.cross_entropy_impl(&targets).unwrap();
    let loss_val = loss.to_vec::<f32>()[0];

    let expected = (5.0f32).ln();
    assert_approx(loss_val, expected, 1e-4, "CE forward uniform logits → ln(5)");
}

// =====================================================================
// 2. Cross Entropy Backward テスト
// =====================================================================

#[test]
#[serial]
fn test_ce_backward_basic() {
    let logits_data = [1.0f32, 2.0, 3.0,  4.0, 5.0, 6.0];
    let targets_data = [0.0f32, 2.0];

    let logits = MetalTensor::from_slice(&logits_data, &[2, 3], DType::F32);
    let targets = MetalTensor::from_slice(&targets_data, &[2], DType::F32);
    let grad_out = MetalTensor::from_slice(&[1.0f32], &[1], DType::F32);

    let grad = logits.cross_entropy_backward_impl(&targets, &grad_out).unwrap();
    let grad_data = grad.to_vec::<f32>();

    let expected = cpu_cross_entropy_backward(&logits_data, &targets_data, 1.0, 2, 3);
    assert_vec_approx(&grad_data, &expected, 1e-4, "CE backward basic");
}

#[test]
#[serial]
fn test_ce_backward_batch_1() {
    let logits_data = [2.0f32, 0.5, -1.0, 3.0];
    let targets_data = [3.0f32];

    let logits = MetalTensor::from_slice(&logits_data, &[1, 4], DType::F32);
    let targets = MetalTensor::from_slice(&targets_data, &[1], DType::F32);
    let grad_out = MetalTensor::from_slice(&[1.0f32], &[1], DType::F32);

    let grad = logits.cross_entropy_backward_impl(&targets, &grad_out).unwrap();
    let grad_data = grad.to_vec::<f32>();

    let expected = cpu_cross_entropy_backward(&logits_data, &targets_data, 1.0, 1, 4);
    assert_vec_approx(&grad_data, &expected, 1e-4, "CE backward batch=1");
}

#[test]
#[serial]
fn test_ce_backward_ignore_index() {
    let logits_data = [1.0f32, 2.0, 3.0,  4.0, 5.0, 6.0,  7.0, 8.0, 9.0];
    let targets_data = [1.0f32, -1.0, 2.0];

    let logits = MetalTensor::from_slice(&logits_data, &[3, 3], DType::F32);
    let targets = MetalTensor::from_slice(&targets_data, &[3], DType::F32);
    let grad_out = MetalTensor::from_slice(&[1.0f32], &[1], DType::F32);

    let grad = logits.cross_entropy_backward_impl(&targets, &grad_out).unwrap();
    let grad_data = grad.to_vec::<f32>();

    // 2行目 (index 3..6) が全て 0 であることを確認
    assert_approx(grad_data[3], 0.0, 1e-6, "ignore row[0]");
    assert_approx(grad_data[4], 0.0, 1e-6, "ignore row[1]");
    assert_approx(grad_data[5], 0.0, 1e-6, "ignore row[2]");

    // 他の行は非ゼロ
    assert!(grad_data[0].abs() > 1e-6 || grad_data[1].abs() > 1e-6, "non-ignore rows should have gradients");
}

#[test]
#[serial]
fn test_ce_backward_grad_output_scaling() {
    let logits_data = [1.0f32, 2.0, 3.0];
    let targets_data = [1.0f32];

    let logits = MetalTensor::from_slice(&logits_data, &[1, 3], DType::F32);
    let targets = MetalTensor::from_slice(&targets_data, &[1], DType::F32);

    let grad_1 = logits.cross_entropy_backward_impl(
        &targets,
        &MetalTensor::from_slice(&[1.0f32], &[1], DType::F32),
    ).unwrap();
    let grad_2 = logits.cross_entropy_backward_impl(
        &targets,
        &MetalTensor::from_slice(&[2.0f32], &[1], DType::F32),
    ).unwrap();

    let g1 = grad_1.to_vec::<f32>();
    let g2 = grad_2.to_vec::<f32>();

    for i in 0..3 {
        assert_approx(g2[i], g1[i] * 2.0, 1e-5, &format!("grad_output scaling [{}]", i));
    }
}

#[test]
#[serial]
fn test_ce_backward_gradient_sum_property() {
    let logits_data = [1.0f32, 2.0, 3.0, 4.0, 5.0,
                       0.5, 1.5, 2.5, 3.5, 4.5];
    let targets_data = [2.0f32, 0.0];

    let logits = MetalTensor::from_slice(&logits_data, &[2, 5], DType::F32);
    let targets = MetalTensor::from_slice(&targets_data, &[2], DType::F32);
    let grad_out = MetalTensor::from_slice(&[1.0f32], &[1], DType::F32);

    let grad = logits.cross_entropy_backward_impl(&targets, &grad_out).unwrap();
    let grad_data = grad.to_vec::<f32>();

    let row0_sum: f32 = grad_data[0..5].iter().sum();
    let row1_sum: f32 = grad_data[5..10].iter().sum();
    assert_approx(row0_sum, 0.0, 1e-5, "row 0 gradient sum should be ~0");
    assert_approx(row1_sum, 0.0, 1e-5, "row 1 gradient sum should be ~0");
}

#[test]
#[serial]
fn test_ce_backward_large_batch() {
    let batch_size = 8;
    let num_classes = 5;
    let mut logits_data = Vec::new();
    for i in 0..batch_size {
        for j in 0..num_classes {
            logits_data.push((i * num_classes + j) as f32 * 0.1 - 2.0);
        }
    }
    let targets_data: Vec<f32> = (0..batch_size).map(|i| (i % num_classes) as f32).collect();

    let logits = MetalTensor::from_slice(&logits_data, &[batch_size, num_classes], DType::F32);
    let targets = MetalTensor::from_slice(&targets_data, &[batch_size], DType::F32);
    let grad_out = MetalTensor::from_slice(&[1.0f32], &[1], DType::F32);

    let grad = logits.cross_entropy_backward_impl(&targets, &grad_out).unwrap();
    let grad_data = grad.to_vec::<f32>();

    let expected = cpu_cross_entropy_backward(&logits_data, &targets_data, 1.0, batch_size, num_classes);
    assert_vec_approx(&grad_data, &expected, 1e-3, "CE backward large batch");
}

// =====================================================================
// 3. Forward + Backward 統合テスト
// =====================================================================

#[test]
#[serial]
fn test_ce_forward_backward_consistency() {
    let logits_data = [0.0f32, 1.0, 2.0,  3.0, 2.0, 1.0];
    let targets_data = [2.0f32, 0.0];

    let logits = MetalTensor::from_slice(&logits_data, &[2, 3], DType::F32);
    let targets = MetalTensor::from_slice(&targets_data, &[2], DType::F32);

    // forward
    let loss = logits.cross_entropy_impl(&targets).unwrap();
    let loss_val = loss.to_vec::<f32>()[0];
    let expected_loss = cpu_cross_entropy_forward(&logits_data, &targets_data, 2, 3);
    assert_approx(loss_val, expected_loss, 1e-4, "forward consistency");

    // backward
    let grad_out = MetalTensor::from_slice(&[1.0f32], &[1], DType::F32);
    let grad = logits.cross_entropy_backward_impl(&targets, &grad_out).unwrap();
    let grad_data = grad.to_vec::<f32>();
    let expected_grad = cpu_cross_entropy_backward(&logits_data, &targets_data, 1.0, 2, 3);
    assert_vec_approx(&grad_data, &expected_grad, 1e-4, "backward consistency");
}
