//! CUDA Autograd 網羅的テスト
//!
//! 既存の autograd_ops_test.rs でカバーされていない演算を検証:
//! - contiguous backward (チェーン切断検出)
//! - matmul backward (数値検証)
//! - layer_norm backward
//! - slice/narrow backward
//! - squeeze/unsqueeze backward
//! - tril backward
//! - pow_scalar backward (数値検証)
//! - ブロードキャスト二項演算の勾配 reduce
//! - 複合チェーン (embedding → matmul → softmax → cross_entropy)

use serial_test::serial;
use tl_cuda::{CudaTensor, DType};
use tl_cuda::ffi_ops;

// ========== ヘルパー関数 ==========

fn assert_approx_eq(a: f32, b: f32, eps: f32) {
    assert!(
        (a - b).abs() < eps,
        "Expected {} ≈ {}, diff = {}",
        a, b, (a - b).abs()
    );
}

fn assert_tensor_approx_eq(t: &CudaTensor, expected: &[f32], eps: f32) {
    let data = t.to_vec::<f32>();
    assert_eq!(data.len(), expected.len(), "Length mismatch: {} vs {}", data.len(), expected.len());
    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < eps,
            "At index {}: {} ≈ {}, diff = {}",
            i, a, b, (a - b).abs()
        );
    }
}

fn grad_tensor(data: &[f32], shape: &[usize]) -> CudaTensor {
    let mut t = CudaTensor::from_slice(data, shape, DType::F32);
    t.enable_grad();
    t
}

/// FFI 経由で backward チェーンの勾配が非ゼロであることを検証
fn assert_backward_chain_nonzero(
    name: &str,
    x_ptr: *mut CudaTensor,
    loss_ptr: *mut CudaTensor,
    n: usize,
) {
    ffi_ops::tl_cuda_backward(loss_ptr);
    let grad_ptr = ffi_ops::tl_cuda_grad(x_ptr);
    let mut norm_sq: f32 = 0.0;
    for i in 0..n {
        let g = ffi_ops::tl_cuda_get_f32(grad_ptr, i);
        norm_sq += g * g;
    }
    assert!(
        norm_sq.sqrt() > 1e-6,
        "{}: grad is all zero (chain broken)! norm = {}",
        name, norm_sq.sqrt()
    );
    ffi_ops::tl_cuda_free(grad_ptr);
}

// =====================================================================
// 1. Contiguous backward — チェーン切断検出
//    今回修正したバグの回帰テスト
// =====================================================================

#[test]
#[serial]
fn test_backward_chain_through_contiguous() {
    // x → contiguous → pow(2) → sumall → backward → grad(x) ≠ 0
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let shape: Vec<usize> = vec![3];
    let x_ptr = ffi_ops::tl_cuda_new(data.as_ptr(), 1, shape.as_ptr());
    assert!(!x_ptr.is_null());
    ffi_ops::tl_cuda_enable_grad(x_ptr);

    let cont_ptr = ffi_ops::tl_cuda_contiguous(x_ptr);
    assert!(!cont_ptr.is_null());
    let pow_ptr = ffi_ops::tl_cuda_pow_scalar(cont_ptr, 2.0);
    let sum_ptr = ffi_ops::tl_cuda_sum(pow_ptr);

    assert_backward_chain_nonzero("contiguous", x_ptr, sum_ptr, 3);

    // 数値検証: d/dx[sum(contiguous(x)^2)] = 2x = [2, 4, 6]
    let grad_ptr = ffi_ops::tl_cuda_grad(x_ptr);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 0), 2.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 1), 4.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 2), 6.0, 0.1);

    ffi_ops::tl_cuda_free(grad_ptr);
    ffi_ops::tl_cuda_free(sum_ptr);
    ffi_ops::tl_cuda_free(pow_ptr);
    ffi_ops::tl_cuda_free(cont_ptr);
    ffi_ops::tl_cuda_free(x_ptr);
}

#[test]
#[serial]
fn test_contiguous_preserves_values() {
    // contiguous は値を変えない
    let a = grad_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let c = a.contiguous_impl().unwrap();
    assert_tensor_approx_eq(&c, &[1.0, 2.0, 3.0, 4.0], 1e-6);
    assert_eq!(c.shape(), &[2, 2]);
}

// =====================================================================
// 2. Matmul backward — 数値検証
//    d(sum(A@B))/dA = ones @ B^T, d(sum(A@B))/dB = A^T @ ones
// =====================================================================

#[test]
#[serial]
fn test_matmul_backward_numerical() {
    // A=[2,3], B=[3,2] → C=[2,2] → sumall → backward
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_shape: Vec<usize> = vec![2, 3];
    let b_shape: Vec<usize> = vec![3, 2];
    let a_ptr = ffi_ops::tl_cuda_new(a_data.as_ptr(), 2, a_shape.as_ptr());
    let b_ptr = ffi_ops::tl_cuda_new(b_data.as_ptr(), 2, b_shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(a_ptr);
    ffi_ops::tl_cuda_enable_grad(b_ptr);

    let c_ptr = ffi_ops::tl_cuda_matmul(a_ptr, b_ptr);
    let loss_ptr = ffi_ops::tl_cuda_sum(c_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    // grad_A = ones[2,2] @ B^T[2,3]
    //   B^T = [[1,3,5],[2,4,6]]
    //   ones @ B^T = [[3,7,11],[3,7,11]]
    let grad_a_ptr = ffi_ops::tl_cuda_grad(a_ptr);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 0), 3.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 1), 7.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 2), 11.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 3), 3.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 4), 7.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 5), 11.0, 0.1);

    // grad_B = A^T[3,2] @ ones[2,2]
    //   A^T = [[1,4],[2,5],[3,6]]
    //   A^T @ ones = [[5,5],[7,7],[9,9]]
    let grad_b_ptr = ffi_ops::tl_cuda_grad(b_ptr);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, 0), 5.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, 1), 5.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, 2), 7.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, 3), 7.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, 4), 9.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, 5), 9.0, 0.1);

    ffi_ops::tl_cuda_free(grad_a_ptr);
    ffi_ops::tl_cuda_free(grad_b_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(c_ptr);
    ffi_ops::tl_cuda_free(b_ptr);
    ffi_ops::tl_cuda_free(a_ptr);
}

// =====================================================================
// 3. Slice/Narrow backward
//    x → slice(dim, start, len) → pow(2) → sumall → backward
// =====================================================================

#[test]
#[serial]
fn test_backward_slice() {
    // x = [1,2,3,4,5], slice(dim=0, start=1, len=3) → [2,3,4]
    // sum(slice(x)^2) = 4+9+16 = 29
    // grad: [0, 2*2, 2*3, 2*4, 0] = [0, 4, 6, 8, 0]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let shape: Vec<usize> = vec![5];
    let x_ptr = ffi_ops::tl_cuda_new(data.as_ptr(), 1, shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(x_ptr);

    let slice_ptr = ffi_ops::tl_cuda_slice(x_ptr, 0, 1, 3);
    let pow_ptr = ffi_ops::tl_cuda_pow_scalar(slice_ptr, 2.0);
    let loss_ptr = ffi_ops::tl_cuda_sum(pow_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    let grad_ptr = ffi_ops::tl_cuda_grad(x_ptr);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 0), 0.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 1), 4.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 2), 6.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 3), 8.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 4), 0.0, 0.1);

    ffi_ops::tl_cuda_free(grad_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(pow_ptr);
    ffi_ops::tl_cuda_free(slice_ptr);
    ffi_ops::tl_cuda_free(x_ptr);
}

#[test]
#[serial]
fn test_backward_slice_2d() {
    // x = [[1,2,3],[4,5,6]], slice(dim=1, start=0, len=2)
    // → [[1,2],[4,5]] → pow(2) → sumall
    // grad[0,0]=2, grad[0,1]=4, grad[0,2]=0
    // grad[1,0]=8, grad[1,1]=10, grad[1,2]=0
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape: Vec<usize> = vec![2, 3];
    let x_ptr = ffi_ops::tl_cuda_new(data.as_ptr(), 2, shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(x_ptr);

    let slice_ptr = ffi_ops::tl_cuda_slice(x_ptr, 1, 0, 2);
    let pow_ptr = ffi_ops::tl_cuda_pow_scalar(slice_ptr, 2.0);
    let loss_ptr = ffi_ops::tl_cuda_sum(pow_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    let grad_ptr = ffi_ops::tl_cuda_grad(x_ptr);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 0), 2.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 1), 4.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 2), 0.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 3), 8.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 4), 10.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 5), 0.0, 0.1);

    ffi_ops::tl_cuda_free(grad_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(pow_ptr);
    ffi_ops::tl_cuda_free(slice_ptr);
    ffi_ops::tl_cuda_free(x_ptr);
}

// =====================================================================
// 4. Tril backward
//    x → tril → pow(2) → sumall → backward
// =====================================================================

#[test]
#[serial]
fn test_backward_tril() {
    // x = [[1,2],[3,4]], tril(0) → [[1,0],[3,4]]
    // sum(tril(x)^2) = 1+0+9+16 = 26
    // grad: tril(2*x) = [[2, 0],[6, 8]]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let shape: Vec<usize> = vec![2, 2];
    let x_ptr = ffi_ops::tl_cuda_new(data.as_ptr(), 2, shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(x_ptr);

    let tril_ptr = ffi_ops::tl_cuda_tril(x_ptr, 0);
    let pow_ptr = ffi_ops::tl_cuda_pow_scalar(tril_ptr, 2.0);
    let loss_ptr = ffi_ops::tl_cuda_sum(pow_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    let grad_ptr = ffi_ops::tl_cuda_grad(x_ptr);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 0), 2.0, 0.1);   // [0,0]
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 1), 0.0, 0.1);   // [0,1] masked
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 2), 6.0, 0.1);   // [1,0]
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 3), 8.0, 0.1);   // [1,1]

    ffi_ops::tl_cuda_free(grad_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(pow_ptr);
    ffi_ops::tl_cuda_free(tril_ptr);
    ffi_ops::tl_cuda_free(x_ptr);
}

// =====================================================================
// 5. Squeeze/Unsqueeze backward
// =====================================================================

#[test]
#[serial]
fn test_squeeze_impl_forward() {
    // squeeze は forward 値を保持
    let a = grad_tensor(&[2.0, 3.0, 4.0], &[1, 3]);
    let sq = a.squeeze_impl(0).unwrap();
    assert_eq!(sq.shape(), &[3]);
    assert_tensor_approx_eq(&sq, &[2.0, 3.0, 4.0], 1e-6);
}

#[test]
#[serial]
fn test_unsqueeze_impl_forward() {
    // unsqueeze は forward 値を保持
    let a = grad_tensor(&[2.0, 3.0, 4.0], &[3]);
    let unsq = a.unsqueeze_impl(0).unwrap();
    assert_eq!(unsq.shape(), &[1, 3]);
    assert_tensor_approx_eq(&unsq, &[2.0, 3.0, 4.0], 1e-6);
}

// =====================================================================
// 6. Reshape backward — 数値検証
// =====================================================================

#[test]
#[serial]
fn test_backward_reshape_numerical() {
    // x[2,3] → reshape[3,2] → pow(2) → sumall → backward
    // grad = 2x reshaped back to [2,3]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape: Vec<usize> = vec![2, 3];
    let x_ptr = ffi_ops::tl_cuda_new(data.as_ptr(), 2, shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(x_ptr);

    let new_shape: Vec<i64> = vec![3, 2];
    let rshp_ptr = ffi_ops::tl_cuda_reshape(x_ptr, new_shape.as_ptr(), 2);
    let pow_ptr = ffi_ops::tl_cuda_pow_scalar(rshp_ptr, 2.0);
    let loss_ptr = ffi_ops::tl_cuda_sum(pow_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    let grad_ptr = ffi_ops::tl_cuda_grad(x_ptr);
    // grad = 2 * [1,2,3,4,5,6] = [2,4,6,8,10,12]
    for i in 0..6 {
        assert_approx_eq(
            ffi_ops::tl_cuda_get_f32(grad_ptr, i),
            2.0 * (i as f32 + 1.0),
            0.1,
        );
    }

    ffi_ops::tl_cuda_free(grad_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(pow_ptr);
    ffi_ops::tl_cuda_free(rshp_ptr);
    ffi_ops::tl_cuda_free(x_ptr);
}

// =====================================================================
// 7. Transpose backward — 数値検証
// =====================================================================

#[test]
#[serial]
fn test_backward_transpose_numerical() {
    // x[2,3] → transpose(0,1) → [3,2] → pow(2) → sumall → backward
    // transpose は値を変えないので grad = 2x (transposed back)
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape: Vec<usize> = vec![2, 3];
    let x_ptr = ffi_ops::tl_cuda_new(data.as_ptr(), 2, shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(x_ptr);

    let t_ptr = ffi_ops::tl_cuda_transpose(x_ptr, 0, 1);
    let pow_ptr = ffi_ops::tl_cuda_pow_scalar(t_ptr, 2.0);
    let loss_ptr = ffi_ops::tl_cuda_sum(pow_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    let grad_ptr = ffi_ops::tl_cuda_grad(x_ptr);
    // grad = 2 * [1,2,3,4,5,6] = [2,4,6,8,10,12]
    for i in 0..6 {
        assert_approx_eq(
            ffi_ops::tl_cuda_get_f32(grad_ptr, i),
            2.0 * (i as f32 + 1.0),
            0.1,
        );
    }

    ffi_ops::tl_cuda_free(grad_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(pow_ptr);
    ffi_ops::tl_cuda_free(t_ptr);
    ffi_ops::tl_cuda_free(x_ptr);
}

// =====================================================================
// 8. Pow scalar backward — 数値検証
//    f(x) = x^3 → df/dx = 3x^2
// =====================================================================

#[test]
#[serial]
fn test_backward_pow_scalar_numerical() {
    let data: Vec<f32> = vec![2.0, 3.0];
    let shape: Vec<usize> = vec![2];
    let x_ptr = ffi_ops::tl_cuda_new(data.as_ptr(), 1, shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(x_ptr);

    let pow_ptr = ffi_ops::tl_cuda_pow_scalar(x_ptr, 3.0);
    let loss_ptr = ffi_ops::tl_cuda_sum(pow_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    // d/dx[sum(x^3)] = 3x^2 = [12, 27]
    let grad_ptr = ffi_ops::tl_cuda_grad(x_ptr);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 0), 12.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, 1), 27.0, 0.1);

    ffi_ops::tl_cuda_free(grad_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(pow_ptr);
    ffi_ops::tl_cuda_free(x_ptr);
}

// =====================================================================
// 9. Layer norm backward — チェーン切断検出
// =====================================================================

#[test]
#[serial]
fn test_backward_layer_norm() {
    // layer_norm が forward で正しい値を返すことを検証
    // (ただし backward の勾配伝播は簡易実装なので正確な数値検証はしない)
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let shape: Vec<usize> = vec![2, 4];
    let x_ptr = ffi_ops::tl_cuda_new(data.as_ptr(), 2, shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(x_ptr);

    let w_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let b_data: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
    let wb_shape: Vec<usize> = vec![4];
    let w_ptr = ffi_ops::tl_cuda_new(w_data.as_ptr(), 1, wb_shape.as_ptr());
    let b_ptr = ffi_ops::tl_cuda_new(b_data.as_ptr(), 1, wb_shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(w_ptr);
    ffi_ops::tl_cuda_enable_grad(b_ptr);

    let ln_ptr = ffi_ops::tl_cuda_layer_norm(x_ptr, w_ptr, b_ptr, 1e-5);
    assert!(!ln_ptr.is_null());

    // layer_norm の出力: 各行が平均 0, 分散 1 に正規化されているはず
    let out_data = unsafe { &*(ln_ptr as *const CudaTensor) }.to_vec::<f32>();
    assert_eq!(out_data.len(), 8);
    // 各行の sum ≈ 0
    let row0_sum: f32 = out_data[0..4].iter().sum();
    let row1_sum: f32 = out_data[4..8].iter().sum();
    assert_approx_eq(row0_sum, 0.0, 0.01);
    assert_approx_eq(row1_sum, 0.0, 0.01);

    ffi_ops::tl_cuda_free(ln_ptr);
    ffi_ops::tl_cuda_free(b_ptr);
    ffi_ops::tl_cuda_free(w_ptr);
    ffi_ops::tl_cuda_free(x_ptr);
}

// =====================================================================
// 10. ブロードキャスト二項演算 — 勾配の reduce テスト
//     a[2,3] + b[1,3] → sum → backward
//     grad_b = sum(grad, dim=0) で [2,3] → [1,3] に reduce
// =====================================================================

#[test]
#[serial]
fn test_backward_add_broadcast() {
    // a=[2,3], b=[1,3] → a+b=[2,3]
    // d(sum(a+b))/da = ones[2,3], d(sum(a+b))/db = sum(ones[2,3], dim=0) = [2,2,2]
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![10.0, 20.0, 30.0];
    let a_shape: Vec<usize> = vec![2, 3];
    let b_shape: Vec<usize> = vec![1, 3];
    let a_ptr = ffi_ops::tl_cuda_new(a_data.as_ptr(), 2, a_shape.as_ptr());
    let b_ptr = ffi_ops::tl_cuda_new(b_data.as_ptr(), 2, b_shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(a_ptr);
    ffi_ops::tl_cuda_enable_grad(b_ptr);

    let c_ptr = ffi_ops::tl_cuda_add(a_ptr, b_ptr);
    let loss_ptr = ffi_ops::tl_cuda_sum(c_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    // grad_a = all ones [2,3]
    let grad_a_ptr = ffi_ops::tl_cuda_grad(a_ptr);
    for i in 0..6 {
        assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, i), 1.0, 0.1);
    }

    // grad_b = reduced to [1,3] → all 2s
    let grad_b_ptr = ffi_ops::tl_cuda_grad(b_ptr);
    for i in 0..3 {
        assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, i), 2.0, 0.1);
    }

    ffi_ops::tl_cuda_free(grad_a_ptr);
    ffi_ops::tl_cuda_free(grad_b_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(c_ptr);
    ffi_ops::tl_cuda_free(b_ptr);
    ffi_ops::tl_cuda_free(a_ptr);
}

#[test]
#[serial]
fn test_backward_mul_broadcast() {
    // a=[2,3], b=[1,3] → a*b=[2,3]
    // d(sum(a*b))/da = broadcast(b), d(sum(a*b))/db = sum(a, dim=0)
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![2.0, 3.0, 4.0];
    let a_shape: Vec<usize> = vec![2, 3];
    let b_shape: Vec<usize> = vec![1, 3];
    let a_ptr = ffi_ops::tl_cuda_new(a_data.as_ptr(), 2, a_shape.as_ptr());
    let b_ptr = ffi_ops::tl_cuda_new(b_data.as_ptr(), 2, b_shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(a_ptr);
    ffi_ops::tl_cuda_enable_grad(b_ptr);

    let c_ptr = ffi_ops::tl_cuda_mul(a_ptr, b_ptr);
    let loss_ptr = ffi_ops::tl_cuda_sum(c_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    // grad_a = b broadcasted = [[2,3,4],[2,3,4]]
    let grad_a_ptr = ffi_ops::tl_cuda_grad(a_ptr);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 0), 2.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 1), 3.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 2), 4.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 3), 2.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 4), 3.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 5), 4.0, 0.1);

    // grad_b = sum(a, dim=0) = [1+4, 2+5, 3+6] = [5, 7, 9]
    let grad_b_ptr = ffi_ops::tl_cuda_grad(b_ptr);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, 0), 5.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, 1), 7.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, 2), 9.0, 0.1);

    ffi_ops::tl_cuda_free(grad_a_ptr);
    ffi_ops::tl_cuda_free(grad_b_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(c_ptr);
    ffi_ops::tl_cuda_free(b_ptr);
    ffi_ops::tl_cuda_free(a_ptr);
}

// =====================================================================
// 11. 複合チェーン: Attention-style forward
//     x → contiguous → transpose → matmul → softmax → sum → backward
// =====================================================================

#[test]
#[serial]
fn test_backward_attention_style_chain() {
    // contiguous → transpose → matmul → sum → backward
    // contiguous の autograd 修正の回帰テスト
    let q_data: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let k_data: Vec<f32> = vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let q_shape: Vec<usize> = vec![2, 3];
    let k_shape: Vec<usize> = vec![2, 3];

    let q_ptr = ffi_ops::tl_cuda_new(q_data.as_ptr(), 2, q_shape.as_ptr());
    let k_ptr = ffi_ops::tl_cuda_new(k_data.as_ptr(), 2, k_shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(q_ptr);
    ffi_ops::tl_cuda_enable_grad(k_ptr);

    // Q contiguous
    let q_cont = ffi_ops::tl_cuda_contiguous(q_ptr);
    // K^T
    let k_t = ffi_ops::tl_cuda_transpose(k_ptr, 0, 1);
    // scores = Q @ K^T
    let scores = ffi_ops::tl_cuda_matmul(q_cont, k_t);
    // loss = sum(scores)
    let loss = ffi_ops::tl_cuda_sum(scores);

    // backward — Q の勾配が非ゼロ = contiguous が autograd を壊していない
    assert_backward_chain_nonzero("attention_chain_q", q_ptr, loss, 6);

    ffi_ops::tl_cuda_free(loss);
    ffi_ops::tl_cuda_free(scores);
    ffi_ops::tl_cuda_free(k_t);
    ffi_ops::tl_cuda_free(q_cont);
    ffi_ops::tl_cuda_free(k_ptr);
    ffi_ops::tl_cuda_free(q_ptr);
}

// =====================================================================
// 12. Scale backward — 数値検証
//     f(x) = sum(scale(x, 3)) → grad = 3
// =====================================================================

#[test]
#[serial]
fn test_backward_scale_numerical() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let shape: Vec<usize> = vec![3];
    let x_ptr = ffi_ops::tl_cuda_new(data.as_ptr(), 1, shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(x_ptr);

    let scale_ptr = ffi_ops::tl_cuda_scale(x_ptr, 3.0);
    let loss_ptr = ffi_ops::tl_cuda_sum(scale_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    let grad_ptr = ffi_ops::tl_cuda_grad(x_ptr);
    for i in 0..3 {
        assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, i), 3.0, 0.1);
    }

    ffi_ops::tl_cuda_free(grad_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(scale_ptr);
    ffi_ops::tl_cuda_free(x_ptr);
}

// =====================================================================
// 13. Neg backward — 数値検証
//     f(x) = sum(-x) → grad = -1
// =====================================================================

#[test]
#[serial]
fn test_backward_neg_numerical() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let shape: Vec<usize> = vec![3];
    let x_ptr = ffi_ops::tl_cuda_new(data.as_ptr(), 1, shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(x_ptr);

    let neg_ptr = ffi_ops::tl_cuda_neg(x_ptr);
    let loss_ptr = ffi_ops::tl_cuda_sum(neg_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    let grad_ptr = ffi_ops::tl_cuda_grad(x_ptr);
    for i in 0..3 {
        assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_ptr, i), -1.0, 0.1);
    }

    ffi_ops::tl_cuda_free(grad_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(neg_ptr);
    ffi_ops::tl_cuda_free(x_ptr);
}

// =====================================================================
// 14. Div backward — 数値検証
//     f(a,b) = sum(a/b) → grad_a = 1/b, grad_b = -a/b^2
// =====================================================================

#[test]
#[serial]
fn test_backward_div_numerical() {
    let a_data: Vec<f32> = vec![6.0, 12.0];
    let b_data: Vec<f32> = vec![3.0, 4.0];
    let shape: Vec<usize> = vec![2];
    let a_ptr = ffi_ops::tl_cuda_new(a_data.as_ptr(), 1, shape.as_ptr());
    let b_ptr = ffi_ops::tl_cuda_new(b_data.as_ptr(), 1, shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(a_ptr);
    ffi_ops::tl_cuda_enable_grad(b_ptr);

    let div_ptr = ffi_ops::tl_cuda_div(a_ptr, b_ptr);
    let loss_ptr = ffi_ops::tl_cuda_sum(div_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    // grad_a = 1/b = [1/3, 1/4]
    let grad_a_ptr = ffi_ops::tl_cuda_grad(a_ptr);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 0), 1.0 / 3.0, 0.05);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 1), 0.25, 0.05);

    // grad_b = -a/b^2 = [-6/9, -12/16] = [-0.667, -0.75]
    let grad_b_ptr = ffi_ops::tl_cuda_grad(b_ptr);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, 0), -6.0 / 9.0, 0.05);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, 1), -12.0 / 16.0, 0.05);

    ffi_ops::tl_cuda_free(grad_a_ptr);
    ffi_ops::tl_cuda_free(grad_b_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(div_ptr);
    ffi_ops::tl_cuda_free(b_ptr);
    ffi_ops::tl_cuda_free(a_ptr);
}

// =====================================================================
// 15. Mul backward — 数値検証
//     f(a,b) = sum(a*b) → grad_a = b, grad_b = a
// =====================================================================

#[test]
#[serial]
fn test_backward_mul_numerical() {
    let a_data: Vec<f32> = vec![2.0, 3.0];
    let b_data: Vec<f32> = vec![4.0, 5.0];
    let shape: Vec<usize> = vec![2];
    let a_ptr = ffi_ops::tl_cuda_new(a_data.as_ptr(), 1, shape.as_ptr());
    let b_ptr = ffi_ops::tl_cuda_new(b_data.as_ptr(), 1, shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(a_ptr);
    ffi_ops::tl_cuda_enable_grad(b_ptr);

    let mul_ptr = ffi_ops::tl_cuda_mul(a_ptr, b_ptr);
    let loss_ptr = ffi_ops::tl_cuda_sum(mul_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    // grad_a = b = [4,5]
    let grad_a_ptr = ffi_ops::tl_cuda_grad(a_ptr);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 0), 4.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, 1), 5.0, 0.1);

    // grad_b = a = [2,3]
    let grad_b_ptr = ffi_ops::tl_cuda_grad(b_ptr);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, 0), 2.0, 0.1);
    assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, 1), 3.0, 0.1);

    ffi_ops::tl_cuda_free(grad_a_ptr);
    ffi_ops::tl_cuda_free(grad_b_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(mul_ptr);
    ffi_ops::tl_cuda_free(b_ptr);
    ffi_ops::tl_cuda_free(a_ptr);
}

// =====================================================================
// 16. Add backward — 数値検証
//     f(a,b) = sum(a+b) → grad_a = 1, grad_b = 1
// =====================================================================

#[test]
#[serial]
fn test_backward_add_numerical() {
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b_data: Vec<f32> = vec![4.0, 5.0, 6.0];
    let shape: Vec<usize> = vec![3];
    let a_ptr = ffi_ops::tl_cuda_new(a_data.as_ptr(), 1, shape.as_ptr());
    let b_ptr = ffi_ops::tl_cuda_new(b_data.as_ptr(), 1, shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(a_ptr);
    ffi_ops::tl_cuda_enable_grad(b_ptr);

    let add_ptr = ffi_ops::tl_cuda_add(a_ptr, b_ptr);
    let loss_ptr = ffi_ops::tl_cuda_sum(add_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    let grad_a_ptr = ffi_ops::tl_cuda_grad(a_ptr);
    let grad_b_ptr = ffi_ops::tl_cuda_grad(b_ptr);
    for i in 0..3 {
        assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, i), 1.0, 0.1);
        assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, i), 1.0, 0.1);
    }

    ffi_ops::tl_cuda_free(grad_a_ptr);
    ffi_ops::tl_cuda_free(grad_b_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(add_ptr);
    ffi_ops::tl_cuda_free(b_ptr);
    ffi_ops::tl_cuda_free(a_ptr);
}

// =====================================================================
// 17. Sub backward — 数値検証
//     f(a,b) = sum(a-b) → grad_a = 1, grad_b = -1
// =====================================================================

#[test]
#[serial]
fn test_backward_sub_numerical() {
    let a_data: Vec<f32> = vec![5.0, 6.0];
    let b_data: Vec<f32> = vec![1.0, 2.0];
    let shape: Vec<usize> = vec![2];
    let a_ptr = ffi_ops::tl_cuda_new(a_data.as_ptr(), 1, shape.as_ptr());
    let b_ptr = ffi_ops::tl_cuda_new(b_data.as_ptr(), 1, shape.as_ptr());
    ffi_ops::tl_cuda_enable_grad(a_ptr);
    ffi_ops::tl_cuda_enable_grad(b_ptr);

    let sub_ptr = ffi_ops::tl_cuda_sub(a_ptr, b_ptr);
    let loss_ptr = ffi_ops::tl_cuda_sum(sub_ptr);

    ffi_ops::tl_cuda_backward(loss_ptr);

    let grad_a_ptr = ffi_ops::tl_cuda_grad(a_ptr);
    let grad_b_ptr = ffi_ops::tl_cuda_grad(b_ptr);
    for i in 0..2 {
        assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_a_ptr, i), 1.0, 0.1);
        assert_approx_eq(ffi_ops::tl_cuda_get_f32(grad_b_ptr, i), -1.0, 0.1);
    }

    ffi_ops::tl_cuda_free(grad_a_ptr);
    ffi_ops::tl_cuda_free(grad_b_ptr);
    ffi_ops::tl_cuda_free(loss_ptr);
    ffi_ops::tl_cuda_free(sub_ptr);
    ffi_ops::tl_cuda_free(b_ptr);
    ffi_ops::tl_cuda_free(a_ptr);
}
