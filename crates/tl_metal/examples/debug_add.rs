//! 単純 add テスト
use tl_metal::{MetalTensor, DType, GpuOps};

fn main() {
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let b = MetalTensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], DType::F32);
    
    println!("a data: {:?}", a.to_vec::<f32>());
    println!("b data: {:?}", b.to_vec::<f32>());
    
    // UFCS 呼び出し（テストと同じ）
    let c_ufcs = GpuOps::add(&a, &b);
    println!("UFCS add result: {:?}", c_ufcs.to_vec::<f32>());
    
    // メソッド呼び出し（examples と同じ）
    let c_method = a.add(&b);
    println!("Method add result: {:?}", c_method.to_vec::<f32>());
}
