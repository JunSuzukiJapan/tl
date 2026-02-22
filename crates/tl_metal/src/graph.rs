//! MetalGraph — Metal バックエンドのキャプチャ・リプレイ実装
//!
//! CUDA Graphs 相当の機能を Metal で実現する。
//! Metal には直接の API がないため、エンコード関数（クロージャ）を
//! 記録し、リプレイ時に高速再エンコードする方式。

use metal::ComputeCommandEncoderRef;
use tl_backend::graph::GpuGraph;

/// キャプチャされた単一カーネル
///
/// # Safety
/// `Sync` は手動実装。内部クロージャは `CommandStream` の `Mutex` で
/// 排他制御されるため、スレッド間で安全に共有可能。
pub(crate) struct CapturedKernel {
    /// エンコード関数（複数回呼び出し可能）
    encode_fn: Box<dyn Fn(&ComputeCommandEncoderRef)>,
}

// Safety: CommandStream は Mutex で排他制御されるため、
// CapturedKernel のクロージャは同時に複数スレッドからアクセスされない。
unsafe impl Send for CapturedKernel {}
unsafe impl Sync for CapturedKernel {}

impl CapturedKernel {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&ComputeCommandEncoderRef) + 'static,
    {
        CapturedKernel {
            encode_fn: Box::new(f),
        }
    }

    /// エンコーダにカーネルをエンコード
    pub fn encode(&self, encoder: &ComputeCommandEncoderRef) {
        (self.encode_fn)(encoder);
    }
}

/// Metal 計算グラフ
///
/// `CommandStream::begin_capture()` ~ `end_capture()` で生成される。
/// `replay()` で記録されたカーネルシーケンスを高速再実行。
pub struct MetalGraph {
    kernels: Vec<CapturedKernel>,
}

impl MetalGraph {
    /// 新しい空のグラフを作成（内部用）
    pub(crate) fn new(kernels: Vec<CapturedKernel>) -> Self {
        MetalGraph { kernels }
    }
}

impl GpuGraph for MetalGraph {
    fn replay(&self) {
        let mut stream = crate::command_stream::get_stream();
        for kernel in &self.kernels {
            let cb = stream.ensure_buffer_pub();
            let encoder = cb.new_compute_command_encoder();
            kernel.encode(encoder);
            encoder.end_encoding();
            stream.inc_batch();
        }
    }

    fn node_count(&self) -> usize {
        self.kernels.len()
    }
}

// ============================================================
// グローバルヘルパー関数
// ============================================================

/// キャプチャ開始
pub fn begin_capture() {
    crate::command_stream::get_stream().begin_capture();
}

/// キャプチャ終了 → MetalGraph を返す
pub fn end_capture() -> MetalGraph {
    crate::command_stream::get_stream().end_capture()
}

/// グラフをリプレイ
pub fn replay_graph(graph: &MetalGraph) {
    graph.replay();
}
