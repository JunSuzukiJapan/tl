//! Phase 1 演算テスト

use tl_metal::{DType, MetalTensor, GpuOps};

fn main() {
    println!("=== Phase 1 演算テスト ===\n");

    // テスト データ
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let b = MetalTensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[4], DType::F32);
    
    println!("a = {:?}", a.to_vec::<f32>());
    println!("b = {:?}", b.to_vec::<f32>());

    // 二項演算
    println!("\n--- 二項演算 ---");
    println!("a + b = {:?}", a.add(&b).to_vec::<f32>());
    println!("a - b = {:?}", a.sub(&b).to_vec::<f32>());
    println!("a * b = {:?}", a.mul(&b).to_vec::<f32>());
    println!("a / b = {:?}", a.div(&b).to_vec::<f32>());
    
    let two = MetalTensor::from_slice(&[2.0f32; 4], &[4], DType::F32);
    println!("a ^ 2 = {:?}", a.pow(&two).to_vec::<f32>());

    // 単項演算
    println!("\n--- 単項演算 ---");
    println!("-a = {:?}", a.neg().to_vec::<f32>());
    
    let c = MetalTensor::from_slice(&[-1.0f32, 2.0, -3.0, 4.0], &[4], DType::F32);
    println!("abs(c) = {:?} (c={:?})", c.abs().to_vec::<f32>(), c.to_vec::<f32>());
    println!("exp(a) = {:?}", a.exp().to_vec::<f32>());
    println!("log(a) = {:?}", a.log().to_vec::<f32>());
    println!("sqrt(a) = {:?}", a.sqrt().to_vec::<f32>());
    println!("tanh(a) = {:?}", a.tanh().to_vec::<f32>());
    println!("sigmoid(a) = {:?}", a.sigmoid().to_vec::<f32>());
    println!("relu(c) = {:?}", c.relu().to_vec::<f32>());

    // スカラー演算
    println!("\n--- スカラー演算 ---");
    println!("a + 10 = {:?}", a.add_scalar(10.0).to_vec::<f32>());
    println!("a * 2 = {:?}", a.mul_scalar(2.0).to_vec::<f32>());
    println!("a - 1 = {:?}", a.sub_scalar(1.0).to_vec::<f32>());
    println!("a / 2 = {:?}", a.div_scalar(2.0).to_vec::<f32>());
    
    let d = MetalTensor::from_slice(&[-1.0f32, 0.5, 1.5, 2.0], &[4], DType::F32);
    println!("clamp(d, 0, 1) = {:?} (d={:?})", d.clamp(0.0, 1.0).to_vec::<f32>(), d.to_vec::<f32>());

    // Reduce
    println!("\n--- Reduce 演算 ---");
    println!("sumall(a) = {}", a.sumall());
    println!("mean_all(a) = {}", a.mean_all());

    println!("\n=== テスト完了 ===");
}
