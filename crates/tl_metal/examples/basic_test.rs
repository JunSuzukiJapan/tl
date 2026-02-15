//! tl_metal テスト

use tl_metal::{buffer_pool, DType, MetalTensor};

fn main() {
    println!("=== tl_metal テスト ===");

    // デバイス初期化
    let device = tl_metal::device::get_device();
    println!("Metal device: {}", device.name());

    // テンソル作成
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
    let b = MetalTensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], DType::F32);

    println!("a = {:?}", a.to_vec::<f32>());
    println!("b = {:?}", b.to_vec::<f32>());

    // 演算
    let c = a.add(&b).unwrap();
    println!("a + b = {:?}", c.to_vec::<f32>());

    let d = a.mul(&b).unwrap();
    println!("a * b = {:?}", d.to_vec::<f32>());

    // プール統計を確認
    {
        let pool = buffer_pool::BUFFER_POOL.lock().unwrap();
        pool.dump_stats();
    }

    // テンソルを解放してバッファがプールに戻るかテスト
    drop(a);
    drop(b);
    drop(c);
    drop(d);

    {
        let pool = buffer_pool::BUFFER_POOL.lock().unwrap();
        println!("\n--- After drop ---");
        pool.dump_stats();
    }

    // 同じサイズのテンソルを作成 → プールから取得されるはず
    let e = MetalTensor::zeros(&[2, 2], DType::F32);
    println!("\ne = {:?}", e.to_vec::<f32>());

    {
        let pool = buffer_pool::BUFFER_POOL.lock().unwrap();
        println!("\n--- After reuse ---");
        pool.dump_stats();
    }

    println!("\n=== テスト完了 ===");
}
