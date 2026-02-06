//! tl_metal GPU パフォーマンステスト

use std::time::Instant;
use tl_metal::{buffer_pool, DType, MetalTensor, GpuOps};

fn main() {
    println!("=== tl_metal GPU パフォーマンステスト ===\n");

    // デバイス初期化
    let device = tl_metal::device::get_device();
    println!("Metal device: {}", device.name());

    // 大きなテンソル（100万要素）
    let size = 1_000_000;
    let data_a: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
    let data_b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.001).collect();

    println!("\n--- テンソル作成 ({}要素) ---", size);
    let start = Instant::now();
    let a = MetalTensor::from_slice(&data_a, &[size], DType::F32);
    let b = MetalTensor::from_slice(&data_b, &[size], DType::F32);
    println!("作成時間: {:?}", start.elapsed());

    println!("\n--- GPU 演算テスト ---");

    // add
    let start = Instant::now();
    let c = a.add(&b);
    println!("add: {:?}", start.elapsed());

    // mul
    let start = Instant::now();
    let d = a.mul(&b);
    println!("mul: {:?}", start.elapsed());

    // exp
    let start = Instant::now();
    let _e = a.exp();
    println!("exp: {:?}", start.elapsed());

    // sqrt
    let start = Instant::now();
    let f = a.sqrt();
    println!("sqrt: {:?}", start.elapsed());

    // 結果確認（先頭5要素）
    let result: Vec<f32> = c.to_vec();
    println!("\nadd 結果 (先頭5要素): {:?}", &result[..5]);

    // バッファプール統計
    {
        let pool = buffer_pool::BUFFER_POOL.lock().unwrap();
        println!("\n--- プール統計 ---");
        pool.dump_stats();
    }

    // ループテスト（バッファ再利用の効果確認）
    println!("\n--- ループテスト (10回) ---");
    for i in 0..10 {
        let x = MetalTensor::zeros(&[size], DType::F32);
        let y = MetalTensor::zeros(&[size], DType::F32);
        let z = x.add(&y);
        drop(x);
        drop(y);
        drop(z);

        if i == 0 || i == 9 {
            let pool = buffer_pool::BUFFER_POOL.lock().unwrap();
            println!("ループ {}: Hits={}, Misses={}, Hit rate={:.1}%",
                     i, pool.hits, pool.misses, pool.hit_rate() * 100.0);
        }
    }

    println!("\n=== テスト完了 ===");
}
