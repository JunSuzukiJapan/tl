//! CpuStream — CPU バックエンドの GpuStream 実装
//!
//! CPU には非同期ストリームの概念はないが、
//! バックエンド互換性のために実装する。

use crate::graph::CpuGraph;
use tl_backend::stream::GpuStream;

/// CPU コマンドストリーム
pub struct CpuStream {
    capturing: Option<CpuGraph>,
}

impl CpuStream {
    pub fn new() -> Self {
        CpuStream { capturing: None }
    }
}

impl GpuStream for CpuStream {
    type Graph = CpuGraph;

    fn synchronize(&mut self) {
        // CPU は同期実行なので何もしない
    }

    fn needs_sync(&self) -> bool {
        false // CPU は常に同期済み
    }

    fn begin_capture(&mut self) {
        self.capturing = Some(CpuGraph::new());
    }

    fn end_capture(&mut self) -> CpuGraph {
        self.capturing.take()
            .expect("end_capture called without begin_capture")
    }

    fn is_capturing(&self) -> bool {
        self.capturing.is_some()
    }
}
