//! Phase 4 演算テスト

use tl_metal::{DType, MetalTensor};

fn main() {
    println!("=== Phase 4 演算テスト ===\n");

    // --- 軸指定 reduce ---
    println!("--- sum(axis) ---");
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    println!("a (2x3) = {:?}", a.to_vec::<f32>());
    println!("sum(a, axis=0) = {:?}", a.sum(0).to_vec::<f32>());
    println!("sum(a, axis=1) = {:?}", a.sum(1).to_vec::<f32>());

    println!("\n--- argmax ---");
    let b = MetalTensor::from_slice(&[0.1f32, 0.9, 0.5, 0.3, 0.7, 0.2], &[2, 3], DType::F32);
    println!("b (2x3) = {:?}", b.to_vec::<f32>());
    println!("argmax(b, axis=1) = {:?}", b.argmax(1).to_vec::<f32>());

    println!("\n--- max ---");
    println!("max(b, axis=1) = {:?}", b.max(1).to_vec::<f32>());

    // --- slice ---
    println!("\n--- slice ---");
    let c = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    println!("c (2x3) = {:?}", c.to_vec::<f32>());
    println!("slice(c, axis=1, start=1, len=2) = {:?}", c.slice(1, 1, 2).to_vec::<f32>());

    // --- embedding ---
    println!("\n--- embedding ---");
    let emb_matrix = MetalTensor::from_slice(
        &[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        &[4, 3],
        DType::F32,
    ); // 4 vocab, 3 dim
    let indices = MetalTensor::from_slice(&[0.0f32, 2.0, 1.0], &[3], DType::F32);
    println!("emb_matrix (4x3) = {:?}", emb_matrix.to_vec::<f32>());
    println!("indices = {:?}", indices.to_vec::<f32>());
    let lookup = emb_matrix.embedding(&indices);
    println!("embedding(emb, indices) = {:?} shape={:?}", lookup.to_vec::<f32>(), lookup.shape());

    println!("\n=== テスト完了 ===");
}
