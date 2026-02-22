//! CudaGraph — CUDA バックエンドのキャプチャ・リプレイスタブ
//!
//! 将来 cudaGraphExec_t のラッパーとして実装する。
//! 現時点ではすべてスタブ (unimplemented!) 実装。

use tl_backend::graph::GpuGraph;

/// CUDA 計算グラフ（スタブ）
pub struct CudaGraph {
    node_count: usize,
}

impl CudaGraph {
    pub fn new(node_count: usize) -> Self {
        CudaGraph { node_count }
    }
}

impl GpuGraph for CudaGraph {
    fn replay(&self) {
        unimplemented!("CUDA graph replay not yet implemented")
    }

    fn node_count(&self) -> usize {
        self.node_count
    }
}
