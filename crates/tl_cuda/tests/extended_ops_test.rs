//! tl_cuda 拡張機能テスト
//! 後から追加された機能の網羅的テスト

use serial_test::serial;
use tl_backend::GpuTensor;
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

// ========== Broadcast 系テスト ==========

#[test]
#[serial]
fn test_broadcast_to_simple() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = a.broadcast_to_impl(&[2, 3]).unwrap();
    assert_eq!(b.shape(), &[2, 3]);
    assert_tensor_approx_eq(&b, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 1e-5);
}

#[test]
#[serial]
fn test_broadcast_to_expand_dim() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], DType::F32);
    let b = a.broadcast_to_impl(&[2, 3]).unwrap();
    assert_eq!(b.shape(), &[2, 3]);
    assert_tensor_approx_eq(&b, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 1e-5);
}

#[test]
#[serial]
fn test_cat_axis0() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = CudaTensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], DType::F32);
    let c = a.cat_impl(&b, 0).unwrap();
    assert_eq!(c.shape(), &[6]);
    assert_tensor_approx_eq(&c, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5);
}

#[test]
#[serial]
fn test_cat_axis1() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
    let b = CudaTensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], DType::F32);
    let c = a.cat_impl(&b, 1).unwrap();
    assert_eq!(c.shape(), &[2, 4]);
}

#[test]
#[serial]
fn test_narrow() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], DType::F32);
    let b = a.narrow_impl(0, 1, 3).unwrap();
    assert_eq!(b.shape(), &[3]);
    assert_tensor_approx_eq(&b, &[2.0, 3.0, 4.0], 1e-5);
}

#[test]
#[serial]
fn test_arange() {
    let t = CudaTensor::arange(0, 5, tl_backend::DType::F32).unwrap();
    assert_eq!(t.shape(), &[5]);
    assert_tensor_approx_eq(&t, &[0.0, 1.0, 2.0, 3.0, 4.0], 1e-5);
}

#[test]
#[serial]
fn test_contiguous() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = a.contiguous_impl().unwrap();
    assert_eq!(b.shape(), &[3]);
    assert_tensor_approx_eq(&b, &[1.0, 2.0, 3.0], 1e-5);
}

// ========== Special 系テスト ==========

#[test]
#[serial]
fn test_where_cond() {
    let cond = CudaTensor::from_slice(&[1.0f32, 0.0, 1.0], &[3], DType::F32);
    let x = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);
    let y = CudaTensor::from_slice(&[100.0f32, 200.0, 300.0], &[3], DType::F32);
    let result = CudaTensor::where_cond_impl(&cond, &x, &y).unwrap();
    assert_tensor_approx_eq(&result, &[10.0, 200.0, 30.0], 1e-5);
}

#[test]
#[serial]
fn test_repeat_interleave() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = a.repeat_interleave_impl(2, 0).unwrap();
    assert_eq!(b.shape(), &[6]);
    assert_tensor_approx_eq(&b, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 1e-5);
}

#[test]
#[serial]
fn test_index_select() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], DType::F32);
    let indices = CudaTensor::from_slice(&[0i64, 2, 4], &[3], DType::I64);
    let b = a.index_select_impl(0, &indices).unwrap();
    assert_eq!(b.shape(), &[3]);
    assert_tensor_approx_eq(&b, &[1.0, 3.0, 5.0], 1e-5);
}

#[test]
#[serial]
fn test_cross_entropy() {
    // cross_entropy_impl requires 2D [N, C] logits and 1D [N] integer labels
    let pred = CudaTensor::from_slice(&[0.7f32, 0.2, 0.1], &[1, 3], DType::F32);
    let target = CudaTensor::from_slice(&[0i64], &[1], DType::I64);
    let loss = pred.cross_entropy_impl(&target).unwrap();
    assert_eq!(loss.shape(), &[1]);
    let loss_val = loss.to_vec::<f32>()[0];
    assert!(loss_val > 0.0, "loss = {}", loss_val);
}

#[test]
#[serial]
fn test_to_dtype() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = a.to_dtype(DType::F32).unwrap();
    assert_eq!(b.shape(), &[3]);
    assert_tensor_approx_eq(&b, &[1.0, 2.0, 3.0], 1e-5);
}

// ========== Index 系テスト ==========

#[test]
#[serial]
fn test_slice() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let b = a.slice_impl(1, 1, 2).unwrap();
    assert_eq!(b.shape(), &[2, 2]);
    assert_tensor_approx_eq(&b, &[2.0, 3.0, 5.0, 6.0], 1e-5);
}

#[test]
#[serial]
fn test_embedding() {
    let emb = CudaTensor::from_slice(
        &[
            0.1f32, 0.2, 0.3, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3,
        ],
        &[4, 3],
        DType::F32,
    );

    let indices = CudaTensor::from_slice(&[0i64, 2, 1], &[3], DType::I64);

    let result = emb.embedding_impl(&indices).unwrap();
    assert_eq!(result.shape(), &[3, 3]);
    assert_tensor_approx_eq(
        &result,
        &[0.1, 0.2, 0.3, 2.1, 2.2, 2.3, 1.1, 1.2, 1.3],
        1e-5,
    );
}

// ========== NN 系テスト ==========

#[test]
#[serial]
fn test_batch_norm() {
    let input = CudaTensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 2, 2, 2],
        DType::F32,
    );

    let gamma = CudaTensor::from_slice(&[1.0f32, 1.0], &[2], DType::F32);
    let beta = CudaTensor::from_slice(&[0.0f32, 0.0], &[2], DType::F32);
    let mean = CudaTensor::from_slice(&[2.5f32, 6.5], &[2], DType::F32);
    let var = CudaTensor::from_slice(&[1.25f32, 1.25], &[2], DType::F32);

    let output = input
        .batch_norm_impl(&gamma, &beta, &mean, &var, 1e-5)
        .unwrap();
    assert_eq!(output.shape(), &[1, 2, 2, 2]);

    let result = output.to_vec::<f32>();
    assert_approx_eq(result[0], -1.3416, 1e-3);
    assert_approx_eq(result[3], 1.3416, 1e-3);
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
    assert_tensor_approx_eq(&output, &[6.0, 8.0, 14.0, 16.0], 1e-5);
}

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

    let output = input.avg_pool2d_impl((2, 2), (2, 2)).unwrap();
    assert_eq!(output.shape(), &[1, 1, 2, 2]);
    assert_tensor_approx_eq(&output, &[3.5, 5.5, 11.5, 13.5], 1e-5);
}

#[test]
#[serial]
fn test_layer_norm() {
    let input = CudaTensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        DType::F32,
    );

    let gamma = CudaTensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], DType::F32);
    let beta = CudaTensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], DType::F32);

    let output = input.layer_norm_impl(&gamma, &beta, 1e-5).unwrap();
    assert_eq!(output.shape(), &[2, 4]);

    let result = output.to_vec::<f32>();
    assert_approx_eq(result[0], -1.3416, 1e-3);
    assert_approx_eq(result[3], 1.3416, 1e-3);
    assert_approx_eq(result[4], -1.3416, 1e-3);
    assert_approx_eq(result[7], 1.3416, 1e-3);
}

#[test]
#[serial]
fn test_dropout() {
    let input = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);

    let output = input.dropout_impl(0.5, false).unwrap();
    assert_eq!(output.shape(), &[4]);
    assert_tensor_approx_eq(&output, &[1.0, 2.0, 3.0, 4.0], 1e-5);
}

#[test]
#[serial]
fn test_dropout_training() {
    let data: Vec<f32> = (1..1001).map(|i| i as f32).collect();
    let input = CudaTensor::from_slice(&data, &[1000], DType::F32);

    let output = input.dropout_impl(0.5, true).unwrap();
    assert_eq!(output.shape(), &[1000]);
    assert_eq!(output.to_vec::<f32>().len(), 1000);
}

// ========== 統合テスト ==========

#[test]
#[serial]
fn test_broadcast_add() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    let b = CudaTensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], DType::F32);

    let b_broadcast = b.broadcast_to_impl(&[2, 3]).unwrap();
    let c = a.add_impl(&b_broadcast).unwrap();
    assert_tensor_approx_eq(&c, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0], 1e-5);
}

#[test]
#[serial]
fn test_slice_and_cat() {
    let a = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], DType::F32);

    let left = a.slice_impl(0, 0, 3).unwrap();
    let right = a.slice_impl(0, 3, 3).unwrap();

    let reconstructed = left.cat_impl(&right, 0).unwrap();
    assert_tensor_approx_eq(&reconstructed, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5);
}
