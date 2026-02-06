//! matmul テスト

use tl_metal::{DType, MetalTensor};

fn main() {
    println!("=== matmul テスト ===\n");

    // 2x3 * 3x2 = 2x2
    let a = MetalTensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        DType::F32,
    );
    let b = MetalTensor::from_slice(
        &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[3, 2],
        DType::F32,
    );

    println!("A (2x3): {:?}", a.to_vec::<f32>());
    println!("B (3x2): {:?}", b.to_vec::<f32>());

    let c = a.matmul(&b);
    println!("\nC = A * B (2x2): {:?}", c.to_vec::<f32>());

    // 期待値:
    // [1*7+2*9+3*11, 1*8+2*10+3*12]   = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12]   = [139, 154]
    println!("期待値: [58, 64, 139, 154]");

    // 大きな行列でパフォーマンステスト
    println!("\n--- 大規模 matmul (512x512 * 512x512) ---");
    let size = 512;
    let data: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32 * 0.1).collect();
    let big_a = MetalTensor::from_slice(&data, &[size, size], DType::F32);
    let big_b = MetalTensor::from_slice(&data, &[size, size], DType::F32);

    let start = std::time::Instant::now();
    let big_c = big_a.matmul(&big_b);
    println!("matmul (512x512): {:?}", start.elapsed());
    println!("結果 shape: {:?}", big_c.shape());

    println!("\n=== テスト完了 ===");
}
