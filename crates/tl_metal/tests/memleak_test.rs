//! メモリリークデバッグテスト
//! N-Queens パターン（softmax + sum + backward）を模倣し、
//! make/release カウンタの不均衡を検出する。

use tl_metal::{MetalTensor, DType};
use serial_test::serial;

#[test]
#[serial]
fn test_autograd_leak_pattern() {
    use tl_metal::ffi_ops;

    // カウンタリセット
    ffi_ops::tl_metal_debug_reset_counters();
    ffi_ops::debug_dump_counters("start");

    // N-Queens の 1 iteration を模倣
    for i in 0..3 {
        // board (requires_grad)
        let mut board = MetalTensor::randn(&[8, 8], DType::F32);
        board.enable_grad();
        let board_ptr = ffi_ops::make_tensor(board);

        // softmax
        let probs_ptr = ffi_ops::tl_metal_softmax(board_ptr, 1);
        // sum(0) - dim reduction
        let col_sums_ptr = ffi_ops::tl_metal_sum_dim(probs_ptr, 0, false);
        // sub_scalar(1.0)  
        let col_diff_ptr = ffi_ops::tl_metal_sub_scalar(col_sums_ptr, 1.0);
        // pow(2) - needs a scalar tensor for pow
        let two = MetalTensor::from_slice(&[2.0f32], &[1], DType::F32);
        let two_ptr = ffi_ops::make_tensor(two);
        let col_pow_ptr = ffi_ops::tl_metal_pow(col_diff_ptr, two_ptr);
        // sumall
        let col_loss_ptr = ffi_ops::tl_metal_sum(col_pow_ptr);

        // backward
        ffi_ops::tl_metal_backward(col_loss_ptr);

        // grad
        let grad_ptr = ffi_ops::tl_metal_grad(board_ptr);

        // detach
        let detached_ptr = ffi_ops::tl_metal_detach(board_ptr);

        // release all
        ffi_ops::release_if_live(probs_ptr);
        ffi_ops::release_if_live(col_sums_ptr);
        ffi_ops::release_if_live(col_diff_ptr);
        ffi_ops::release_if_live(two_ptr);
        ffi_ops::release_if_live(col_pow_ptr);
        ffi_ops::release_if_live(col_loss_ptr);
        if !grad_ptr.is_null() {
            ffi_ops::release_if_live(grad_ptr);
        }
        ffi_ops::release_if_live(board_ptr);
        ffi_ops::release_if_live(detached_ptr);

        ffi_ops::debug_dump_counters(&format!("iter_{}", i));
    }

    ffi_ops::debug_dump_counters("end");
}
