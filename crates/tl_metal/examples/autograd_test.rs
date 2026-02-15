//! Autograd テスト（V5.0 Arc ベース）
//! FFI 関数経由で autograd グラフが正しく構築・逆伝播されることを検証する。

use tl_metal::ffi_ops;

fn main() {
    println!("=== Autograd テスト (V5.0 Arc ベース) ===\n");

    // --- 単純な加算の勾配 ---
    println!("--- 加算の勾配 (loss = sum(a + b)) ---");
    let a_ptr = ffi_ops::tl_metal_new([1.0f32, 2.0, 3.0].as_ptr(), 1, [3usize].as_ptr());
    let b_ptr = ffi_ops::tl_metal_new([4.0f32, 5.0, 6.0].as_ptr(), 1, [3usize].as_ptr());

    ffi_ops::tl_metal_enable_grad(a_ptr);
    ffi_ops::tl_metal_enable_grad(b_ptr);

    let c_ptr = ffi_ops::tl_metal_add(a_ptr, b_ptr);
    let loss_ptr = ffi_ops::tl_metal_sum(c_ptr);

    let (a, c, loss) = unsafe { (&*a_ptr, &*c_ptr, &*loss_ptr) };
    let b = unsafe { &*b_ptr };
    println!("a = {:?}", a.to_vec::<f32>());
    println!("b = {:?}", b.to_vec::<f32>());
    println!("c = a + b = {:?}", c.to_vec::<f32>());
    println!("loss = sum(c) = {:?}", loss.to_vec::<f32>());

    ffi_ops::tl_metal_backward(loss_ptr);

    let grad_a_ptr = ffi_ops::tl_metal_grad(a_ptr);
    let grad_b_ptr = ffi_ops::tl_metal_grad(b_ptr);
    assert!(!grad_a_ptr.is_null(), "grad_a is null");
    assert!(!grad_b_ptr.is_null(), "grad_b is null");

    let grad_a: Vec<f32> = unsafe { &*grad_a_ptr }.to_vec();
    let grad_b: Vec<f32> = unsafe { &*grad_b_ptr }.to_vec();
    println!("∂loss/∂a = {:?} (expected: [1, 1, 1])", grad_a);
    println!("∂loss/∂b = {:?} (expected: [1, 1, 1])", grad_b);
    assert_eq!(grad_a, vec![1.0, 1.0, 1.0]);
    assert_eq!(grad_b, vec![1.0, 1.0, 1.0]);
    println!("OK ✓\n");

    // リソース解放
    ffi_ops::tl_metal_free(grad_a_ptr);
    ffi_ops::tl_metal_free(grad_b_ptr);
    ffi_ops::tl_metal_free(loss_ptr);
    ffi_ops::tl_metal_free(c_ptr);
    ffi_ops::tl_metal_free(b_ptr);
    ffi_ops::tl_metal_free(a_ptr);

    // --- 乗算の勾配 ---
    println!("--- 乗算の勾配 (loss = sum(a * b)) ---");
    let a_ptr = ffi_ops::tl_metal_new([2.0f32, 3.0].as_ptr(), 1, [2usize].as_ptr());
    let b_ptr = ffi_ops::tl_metal_new([4.0f32, 5.0].as_ptr(), 1, [2usize].as_ptr());

    ffi_ops::tl_metal_enable_grad(a_ptr);
    ffi_ops::tl_metal_enable_grad(b_ptr);

    let c_ptr = ffi_ops::tl_metal_mul(a_ptr, b_ptr);
    let loss_ptr = ffi_ops::tl_metal_sum(c_ptr);

    ffi_ops::tl_metal_backward(loss_ptr);

    let grad_a: Vec<f32> = unsafe { &*ffi_ops::tl_metal_grad(a_ptr) }.to_vec();
    let grad_b: Vec<f32> = unsafe { &*ffi_ops::tl_metal_grad(b_ptr) }.to_vec();
    println!("∂loss/∂a = {:?} (expected: [4, 5] = b)", grad_a);
    println!("∂loss/∂b = {:?} (expected: [2, 3] = a)", grad_b);
    assert_eq!(grad_a, vec![4.0, 5.0]);
    assert_eq!(grad_b, vec![2.0, 3.0]);
    println!("OK ✓\n");

    ffi_ops::tl_metal_free(loss_ptr);
    ffi_ops::tl_metal_free(c_ptr);
    ffi_ops::tl_metal_free(b_ptr);
    ffi_ops::tl_metal_free(a_ptr);

    // --- ReLU + sum ---
    println!("--- ReLU の勾配 (loss = sum(relu(a))) ---");
    let a_ptr = ffi_ops::tl_metal_new([-1.0f32, 0.5, 2.0].as_ptr(), 1, [3usize].as_ptr());
    ffi_ops::tl_metal_enable_grad(a_ptr);

    let r_ptr = ffi_ops::tl_metal_relu(a_ptr);
    let loss_ptr = ffi_ops::tl_metal_sum(r_ptr);

    ffi_ops::tl_metal_backward(loss_ptr);

    let grad_a: Vec<f32> = unsafe { &*ffi_ops::tl_metal_grad(a_ptr) }.to_vec();
    println!("∂loss/∂a = {:?} (expected: [0, 1, 1])", grad_a);
    assert_eq!(grad_a, vec![0.0, 1.0, 1.0]);
    println!("OK ✓\n");

    ffi_ops::tl_metal_free(loss_ptr);
    ffi_ops::tl_metal_free(r_ptr);
    ffi_ops::tl_metal_free(a_ptr);

    // --- Neg ---
    println!("--- Neg の勾配 (loss = sum(-a)) ---");
    let a_ptr = ffi_ops::tl_metal_new([1.0f32, 2.0, 3.0].as_ptr(), 1, [3usize].as_ptr());
    ffi_ops::tl_metal_enable_grad(a_ptr);

    let neg_ptr = ffi_ops::tl_metal_neg(a_ptr);
    let loss_ptr = ffi_ops::tl_metal_sum(neg_ptr);

    ffi_ops::tl_metal_backward(loss_ptr);

    let grad_a: Vec<f32> = unsafe { &*ffi_ops::tl_metal_grad(a_ptr) }.to_vec();
    println!("∂loss/∂a = {:?} (expected: [-1, -1, -1])", grad_a);
    assert_eq!(grad_a, vec![-1.0, -1.0, -1.0]);
    println!("OK ✓\n");

    ffi_ops::tl_metal_free(loss_ptr);
    ffi_ops::tl_metal_free(neg_ptr);
    ffi_ops::tl_metal_free(a_ptr);

    // --- Mul scalar ---
    println!("--- スカラー乗算の勾配 (loss = sum(a * 3.0)) ---");
    let a_ptr = ffi_ops::tl_metal_new([1.0f32, 2.0, 3.0].as_ptr(), 1, [3usize].as_ptr());
    ffi_ops::tl_metal_enable_grad(a_ptr);

    let scaled = ffi_ops::tl_metal_mul_scalar(a_ptr, 3.0);
    let loss_ptr = ffi_ops::tl_metal_sum(scaled);

    ffi_ops::tl_metal_backward(loss_ptr);

    let grad_a: Vec<f32> = unsafe { &*ffi_ops::tl_metal_grad(a_ptr) }.to_vec();
    println!("∂loss/∂a = {:?} (expected: [3, 3, 3])", grad_a);
    assert_eq!(grad_a, vec![3.0, 3.0, 3.0]);

    println!("OK ✓\n");

    ffi_ops::tl_metal_free(loss_ptr);
    ffi_ops::tl_metal_free(scaled);
    ffi_ops::tl_metal_free(a_ptr);

    println!("=== 全テスト完了 ✓ ===");
}
