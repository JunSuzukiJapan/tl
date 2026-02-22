//! キャプチャ・リプレイ使用例
//!
//! ```
//! cargo run -p tl_metal --release --example graph_capture_replay
//! ```

use tl_metal::{MetalTensor, DType};

fn main() {
    // ========================================
    // 1. 入力テンソルを準備
    // ========================================
    let x = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
    let w = MetalTensor::from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[2, 2], DType::F32);

    // ========================================
    // 2. forward pass: y = relu(x @ w)
    // ========================================
    let y = x.matmul(&w).unwrap();
    let y = y.relu().unwrap();

    let data = y.to_vec::<f32>();
    println!("y = {:?}", data);
    // y = [1.5, 1.5, 3.5, 3.5]

    // ========================================
    // 3. キャプチャ・リプレイ（低レベル API）
    // ========================================
    // begin_capture() → encode_capturable() → end_capture() → replay()
    //
    // use tl_metal::graph::{begin_capture, end_capture};
    // use tl_backend::graph::GpuGraph;
    //
    // begin_capture();
    // stream.encode_capturable(|encoder| { ... });
    // let graph = end_capture();
    // graph.replay();  // 何度でも再実行
    
    println!("done!");
}
