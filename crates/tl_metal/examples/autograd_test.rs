//! Autograd テスト

use tl_metal::{DType, MetalTensor};
use tl_backend::GpuOps;

/// sumall をテンソルとして返す（backward 可能）
fn tensor_sum(input: &MetalTensor, input_ptr: *mut MetalTensor) -> MetalTensor {
    let val = input.sumall_impl();
    let mut result = MetalTensor::from_slice(&[val], &[1], DType::F32);
    {
        use tl_metal::autograd::ops::SumallBackward;
        result.set_grad_fn(Box::new(SumallBackward { a: input_ptr, shape: input.shape().to_vec() }));
    }
    result
}

fn main() {
    println!("=== Autograd テスト ===\n");

    // --- 単純な加算の勾配 ---
    println!("--- 加算の勾配 (loss = a + b).sumall() ---");
    let mut a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let mut b = MetalTensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], DType::F32);
    a.enable_grad();
    b.enable_grad();

    let a_ptr: *mut MetalTensor = &mut a;
    let b_ptr: *mut MetalTensor = &mut b;

    let mut c = MetalTensor::add_impl(&a, &b);
    {
        use tl_metal::autograd::ops::AddBackward;
        c.set_grad_fn(Box::new(AddBackward { a: a_ptr, b: b_ptr }));
    }

    let c_ptr: *mut MetalTensor = &mut c;
    let mut loss = tensor_sum(&c, c_ptr);

    println!("a = {:?}", a.to_vec::<f32>());
    println!("b = {:?}", b.to_vec::<f32>());
    println!("c = a + b = {:?}", c.to_vec::<f32>());
    println!("loss = sum(c) = {:?}", loss.to_vec::<f32>());

    loss.backward();

    println!("∂loss/∂a = {:?} (expected: [1, 1, 1])", a.get_grad().unwrap().to_vec::<f32>());
    println!("∂loss/∂b = {:?} (expected: [1, 1, 1])", b.get_grad().unwrap().to_vec::<f32>());

    // --- 乗算の勾配 ---
    println!("\n--- 乗算の勾配 (loss = (a * b).sumall()) ---");
    let mut a = MetalTensor::from_slice(&[2.0f32, 3.0], &[2], DType::F32);
    let mut b = MetalTensor::from_slice(&[4.0f32, 5.0], &[2], DType::F32);
    a.enable_grad();
    b.enable_grad();

    let a_ptr: *mut MetalTensor = &mut a;
    let b_ptr: *mut MetalTensor = &mut b;

    let mut c = MetalTensor::mul_impl(&a, &b);
    {
        use tl_metal::autograd::ops::MulBackward;
        c.set_grad_fn(Box::new(MulBackward {
            a: a_ptr, b: b_ptr,
            a_data: a.shallow_clone(),
            b_data: b.shallow_clone(),
        }));
    }

    let c_ptr: *mut MetalTensor = &mut c;
    let mut loss = tensor_sum(&c, c_ptr);

    println!("a = {:?}", a.to_vec::<f32>());
    println!("b = {:?}", b.to_vec::<f32>());
    println!("c = a * b = {:?}", c.to_vec::<f32>());
    println!("loss = sum(c) = {:?}", loss.to_vec::<f32>());

    loss.backward();

    println!("∂loss/∂a = {:?} (expected: [4, 5] = b)", a.get_grad().unwrap().to_vec::<f32>());
    println!("∂loss/∂b = {:?} (expected: [2, 3] = a)", b.get_grad().unwrap().to_vec::<f32>());

    // --- ReLU ---
    println!("\n--- ReLU の勾配 ---");
    let mut a = MetalTensor::from_slice(&[-1.0f32, 0.5, 2.0], &[3], DType::F32);
    a.enable_grad();
    let a_ptr: *mut MetalTensor = &mut a;

    let mut r = a.relu();
    {
        use tl_metal::autograd::ops::ReluBackward;
        r.set_grad_fn(Box::new(ReluBackward { a: a_ptr, a_data: a.shallow_clone() }));
    }

    let r_ptr: *mut MetalTensor = &mut r;
    let mut loss = tensor_sum(&r, r_ptr);

    println!("a = {:?}", a.to_vec::<f32>());
    println!("relu(a) = {:?}", r.to_vec::<f32>());

    loss.backward();

    println!("∂loss/∂a = {:?} (expected: [0, 1, 1])", a.get_grad().unwrap().to_vec::<f32>());

    println!("\n=== テスト完了 ===");
}
