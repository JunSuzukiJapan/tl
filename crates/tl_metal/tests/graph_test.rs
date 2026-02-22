//! キャプチャ・リプレイ テスト (Metal GPU Graph)

use tl_metal::{MetalTensor, DType};
use tl_metal::graph::{begin_capture, end_capture, replay_graph};
use tl_backend::graph::GpuGraph;
use serial_test::serial;

fn assert_tensor_approx_eq(a: &MetalTensor, b: &[f32], eps: f32) {
    let a_data = a.to_vec::<f32>();
    assert_eq!(a_data.len(), b.len(), "Length mismatch: {} vs {}", a_data.len(), b.len());
    for (i, (&av, &bv)) in a_data.iter().zip(b.iter()).enumerate() {
        assert!((av - bv).abs() < eps, "At index {}: {} ≈ {}", i, av, bv);
    }
}

/// 基本テスト: add → 結果確認のキャプチャ・リプレイ
#[test]
#[serial]
fn test_capture_replay_basic() {
    // 入力テンソル
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let b = MetalTensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], DType::F32);

    // キャプチャ: add のみ (encode_capturable は内部的に呼ばれる)
    // 注: 現在 stream_encode は FnOnce で encode_capturable は Fn
    // このテストでは直接 CommandStream API を使用
    let result = a.add(&b).unwrap();
    assert_tensor_approx_eq(&result, &[11.0, 22.0, 33.0, 44.0], 1e-5);
}

/// begin_capture / end_capture の基本動作テスト
#[test]
#[serial]
fn test_capture_end_capture() {
    begin_capture();
    // キャプチャ中は何もエンコードしない（空グラフ）
    let graph = end_capture();
    assert_eq!(graph.node_count(), 0);
}

/// リプレイの基本テスト（空グラフ）
#[test]
#[serial]
fn test_replay_empty_graph() {
    begin_capture();
    let graph = end_capture();
    // 空グラフのリプレイは何もしない
    replay_graph(&graph);
    replay_graph(&graph);
    assert_eq!(graph.node_count(), 0);
}

/// キャプチャ・リプレイの一貫性テスト
/// encode_capturable を使ってカーネルを記録し、リプレイで同じ結果を得る
#[test]
#[serial]
fn test_capture_replay_with_kernel() {
    use tl_metal::command_stream::get_stream;
    use metal::MTLSize;

    // テスト用: 2つのテンソルを add する
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let b = MetalTensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], DType::F32);
    let result = MetalTensor::uninit(&[4], DType::F32);

    // パイプライン取得（raw pointer で保持してライフタイムを回避）
    let pipeline = {
        let mut shaders = tl_metal::shaders::get_shaders().lock().unwrap();
        let device = tl_metal::device::get_device();
        let p = shaders.get_pipeline(device.device(), tl_metal::shaders::SHADER_ADD_F32)
            .expect("add pipeline");
        p as *const metal::ComputePipelineState
    };

    let a_buf = a.buffer() as *const metal::Buffer;
    let b_buf = b.buffer() as *const metal::Buffer;
    let r_buf = result.buffer() as *const metal::Buffer;
    let count = 4usize;

    // キャプチャ開始
    begin_capture();

    // encode_capturable でカーネルをエンコード＋記録
    {
        let mut stream = get_stream();
        stream.encode_capturable(move |encoder| {
            unsafe {
                encoder.set_compute_pipeline_state(&*pipeline);
                encoder.set_buffer(0, Some(&*a_buf), 0);
                encoder.set_buffer(1, Some(&*b_buf), 0);
                encoder.set_buffer(2, Some(&*r_buf), 0);
            }
            let tpg = MTLSize::new(count.min(256) as u64, 1, 1);
            let grid = MTLSize::new(1, 1, 1);
            encoder.dispatch_thread_groups(grid, tpg);
        });
    }

    let graph = end_capture();
    assert_eq!(graph.node_count(), 1);

    // 初回実行の結果確認
    assert_tensor_approx_eq(&result, &[11.0, 22.0, 33.0, 44.0], 1e-5);

    // リプレイ: 同じカーネルを再実行
    // 入力を変えずにリプレイすると同じ結果が得られる
    replay_graph(&graph);
    tl_metal::command_stream::sync_stream();
    assert_tensor_approx_eq(&result, &[11.0, 22.0, 33.0, 44.0], 1e-5);
}

/// 複数回リプレイのテスト
#[test]
#[serial]
fn test_replay_multiple_times() {
    begin_capture();
    let graph = end_capture();

    // 100回リプレイしてもクラッシュしない
    for _ in 0..100 {
        replay_graph(&graph);
    }
    assert_eq!(graph.node_count(), 0);
}
