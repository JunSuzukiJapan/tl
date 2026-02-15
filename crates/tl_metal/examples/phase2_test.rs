//! Phase 2/3 演算テスト

use tl_metal::{DType, MetalTensor};

fn main() {
    println!("=== Phase 2/3 演算テスト ===\n");

    // --- 形状操作 ---
    println!("--- reshape ---");
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    println!("a (2x3) = {:?}", a.to_vec::<f32>());
    
    let b = a.reshape(&[3, 2]).unwrap();
    println!("reshape(a, [3, 2]) = {:?} shape={:?}", b.to_vec::<f32>(), b.shape());
    
    let c = a.reshape(&[6]).unwrap();
    println!("reshape(a, [6]) = {:?} shape={:?}", c.to_vec::<f32>(), c.shape());

    println!("\n--- transpose ---");
    let d = a.transpose(0, 1).unwrap();
    println!("transpose(a) = {:?} shape={:?}", d.to_vec::<f32>(), d.shape());

    println!("\n--- squeeze / unsqueeze ---");
    let e = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], DType::F32);
    println!("e (1x3) = {:?}", e.to_vec::<f32>());
    let f = e.squeeze(0).unwrap();
    println!("squeeze(e, 0) = {:?} shape={:?}", f.to_vec::<f32>(), f.shape());
    let g = f.unsqueeze(1).unwrap();
    println!("unsqueeze(f, 1) = {:?} shape={:?}", g.to_vec::<f32>(), g.shape());

    // --- 生成関数 ---
    println!("\n--- ones ---");
    let ones = MetalTensor::ones(&[2, 3], DType::F32);
    println!("ones([2, 3]) = {:?}", ones.to_vec::<f32>());

    println!("\n--- randn ---");
    let randn = MetalTensor::randn(&[2, 3], DType::F32);
    println!("randn([2, 3]) = {:?}", randn.to_vec::<f32>());

    // --- softmax ---
    println!("\n--- softmax ---");
    let logits = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3], DType::F32);
    println!("logits (2x3) = {:?}", logits.to_vec::<f32>());
    let probs = logits.softmax(1).unwrap();
    println!("softmax(logits, axis=1) = {:?}", probs.to_vec::<f32>());
    
    // 各行の合計が1になることを確認
    let probs_vec: Vec<f32> = probs.to_vec();
    let row0_sum: f32 = probs_vec[0..3].iter().sum();
    let row1_sum: f32 = probs_vec[3..6].iter().sum();
    println!("Row sums: {} and {} (should be 1.0)", row0_sum, row1_sum);

    println!("\n=== テスト完了 ===");
}
