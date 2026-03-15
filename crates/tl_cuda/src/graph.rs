//! CudaGraph — CUDA バックエンドのキャプチャ・リプレイ
//!
//! cudaGraphExec_t のラッパーとして実装。
//! ストリームキャプチャでカーネルシーケンスを記録し、replay() で再実行する。

use crate::cuda_sys::{self, cudaGraph_t, cudaGraphExec_t, cudaStreamCaptureMode, CUDA_SUCCESS};
use crate::stream;
use tl_backend::graph::GpuGraph;

/// CUDA 計算グラフ
pub struct CudaGraph {
    /// 実行可能グラフ
    exec: cudaGraphExec_t,
    /// ノード数
    node_count: usize,
}

unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl CudaGraph {
    /// ストリームキャプチャ結果からグラフを構築する。
    ///
    /// # Safety
    /// `graph` は `cudaStreamEndCapture` で取得した有効なグラフハンドルであること。
    pub unsafe fn from_captured_graph(graph: cudaGraph_t) -> Result<Self, String> {
        // ノード数を取得
        let mut num_nodes: usize = 0;
        let err = cuda_sys::cudaGraphGetNodes(graph, std::ptr::null_mut(), &mut num_nodes);
        if err != CUDA_SUCCESS {
            return Err(format!("cudaGraphGetNodes failed: {}", err));
        }

        // グラフをインスタンス化
        let mut exec: cudaGraphExec_t = std::ptr::null_mut();
        let err = cuda_sys::cudaGraphInstantiate(
            &mut exec,
            graph,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        );
        if err != CUDA_SUCCESS {
            return Err(format!("cudaGraphInstantiate failed: {}", err));
        }

        // 元のグラフを破棄（exec が独立して保持）
        cuda_sys::cudaGraphDestroy(graph);

        Ok(CudaGraph {
            exec,
            node_count: num_nodes,
        })
    }

    /// ストリームキャプチャを開始する。
    pub fn begin_capture() -> Result<(), String> {
        let stream = stream::get_stream().raw();
        let err = unsafe {
            cuda_sys::cudaStreamBeginCapture(
                stream,
                cudaStreamCaptureMode::cudaStreamCaptureModeGlobal,
            )
        };
        if err != CUDA_SUCCESS {
            return Err(format!("cudaStreamBeginCapture failed: {}", err));
        }
        Ok(())
    }

    /// ストリームキャプチャを終了し、CudaGraph を返す。
    pub fn end_capture() -> Result<Self, String> {
        let stream = stream::get_stream().raw();
        let mut graph: cudaGraph_t = std::ptr::null_mut();
        let err = unsafe { cuda_sys::cudaStreamEndCapture(stream, &mut graph) };
        if err != CUDA_SUCCESS {
            return Err(format!("cudaStreamEndCapture failed: {}", err));
        }
        unsafe { Self::from_captured_graph(graph) }
    }
}

impl GpuGraph for CudaGraph {
    fn replay(&self) {
        let stream = stream::get_stream().raw();
        let err = unsafe { cuda_sys::cudaGraphLaunch(self.exec, stream) };
        if err != CUDA_SUCCESS {
            eprintln!("cudaGraphLaunch failed: {}", err);
        }
        // 同期してリプレイ完了を保証
        stream::sync_stream();
    }

    fn node_count(&self) -> usize {
        self.node_count
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            unsafe {
                cuda_sys::cudaGraphExecDestroy(self.exec);
            }
        }
    }
}
