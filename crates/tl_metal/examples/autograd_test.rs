//! Autograd テスト

use tl_metal::{DType, MetalTensor, MetalVar, GpuOps};

fn main() {
    println!("=== Autograd テスト ===\n");

    // --- 単純な加算の勾配 ---
    println!("--- 加算の勾配 (loss = a + b).sumall() ---");
    let a_data = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b_data = MetalTensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], DType::F32);
    
    let a = MetalVar::new(a_data.clone_data(), true);
    let b = MetalVar::new(b_data.clone_data(), true);
    
    let c = a.add(&b);
    let loss = c.sumall();
    
    println!("a = {:?}", a.data().to_vec::<f32>());
    println!("b = {:?}", b.data().to_vec::<f32>());
    println!("c = a + b = {:?}", c.data().to_vec::<f32>());
    println!("loss = sum(c) = {:?}", loss.data().to_vec::<f32>());
    
    loss.backward();
    
    println!("∂loss/∂a = {:?} (expected: [1, 1, 1])", a.grad().unwrap().to_vec::<f32>());
    println!("∂loss/∂b = {:?} (expected: [1, 1, 1])", b.grad().unwrap().to_vec::<f32>());

    // --- 乗算の勾配 ---
    println!("\n--- 乗算の勾配 (loss = (a * b).sumall()) ---");
    let a = MetalVar::new(MetalTensor::from_slice(&[2.0f32, 3.0], &[2], DType::F32), true);
    let b = MetalVar::new(MetalTensor::from_slice(&[4.0f32, 5.0], &[2], DType::F32), true);
    
    let c = a.mul(&b);
    let loss = c.sumall();
    
    println!("a = {:?}", a.data().to_vec::<f32>());
    println!("b = {:?}", b.data().to_vec::<f32>());
    println!("c = a * b = {:?}", c.data().to_vec::<f32>());
    println!("loss = sum(c) = {:?}", loss.data().to_vec::<f32>());
    
    loss.backward();
    
    println!("∂loss/∂a = {:?} (expected: [4, 5] = b)", a.grad().unwrap().to_vec::<f32>());
    println!("∂loss/∂b = {:?} (expected: [2, 3] = a)", b.grad().unwrap().to_vec::<f32>());

    // --- 複合演算（loss = sum((a * b + 1)^2)）---
    println!("\n--- 複合演算 (loss = sum((a * b + 1)^2)) ---");
    let a = MetalVar::new(MetalTensor::from_slice(&[1.0f32, 2.0], &[2], DType::F32), true);
    let b = MetalVar::new(MetalTensor::from_slice(&[3.0f32, 4.0], &[2], DType::F32), true);
    let one = MetalVar::new(MetalTensor::from_slice(&[1.0f32, 1.0], &[2], DType::F32), false);
    let two = MetalVar::new(MetalTensor::from_slice(&[2.0f32, 2.0], &[2], DType::F32), false);
    
    let ab = a.mul(&b);  // [3, 8]
    let ab_plus_1 = ab.add(&one);  // [4, 9]
    let sq = ab_plus_1.pow(&two);  // [16, 81]
    let loss = sq.sumall();  // 97
    
    println!("a = {:?}", a.data().to_vec::<f32>());
    println!("b = {:?}", b.data().to_vec::<f32>());
    println!("loss = {:?}", loss.data().to_vec::<f32>());
    
    loss.backward();
    
    // d/da = 2 * (a*b + 1) * b = 2 * [4, 9] * [3, 4] = [24, 72]
    // d/db = 2 * (a*b + 1) * a = 2 * [4, 9] * [1, 2] = [8, 36]
    println!("∂loss/∂a = {:?} (expected: [24, 72])", a.grad().unwrap().to_vec::<f32>());
    println!("∂loss/∂b = {:?} (expected: [8, 36])", b.grad().unwrap().to_vec::<f32>());

    // --- ReLU ---
    println!("\n--- ReLU の勾配 ---");
    let a = MetalVar::new(MetalTensor::from_slice(&[-1.0f32, 0.5, 2.0], &[3], DType::F32), true);
    
    let r = a.relu();
    let loss = r.sumall();
    
    println!("a = {:?}", a.data().to_vec::<f32>());
    println!("relu(a) = {:?}", r.data().to_vec::<f32>());
    
    loss.backward();
    
    println!("∂loss/∂a = {:?} (expected: [0, 1, 1])", a.grad().unwrap().to_vec::<f32>());

    // --- Softmax ---
    println!("\n--- Softmax の勾配 ---");
    let a = MetalVar::new(MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], DType::F32), true);
    
    let s = a.softmax(1);
    let loss = s.sumall();  // sum = 1
    
    println!("a = {:?}", a.data().to_vec::<f32>());
    println!("softmax(a) = {:?}", s.data().to_vec::<f32>());
    
    loss.backward();
    
    println!("∂loss/∂a = {:?} (expected: ~0 since sum(softmax) = 1)", a.grad().unwrap().to_vec::<f32>());

    println!("\n=== テスト完了 ===");
}
