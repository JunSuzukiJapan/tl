//! CpuGraph — CPU バックエンドのキャプチャ・リプレイ
//!
//! CPU ではキャプチャ・リプレイの性能メリットはないが、
//! バックエンド互換性のために実装する。
//! 記録された関数を順番に呼び出すだけの実装。

use tl_backend::graph::GpuGraph;

/// キャプチャされた CPU 操作
struct CapturedOp {
    func: Box<dyn Fn() + Send>,
}

// Safety: CPU はシングルスレッドで操作されるため安全
unsafe impl Sync for CapturedOp {}

/// CPU 計算グラフ
pub struct CpuGraph {
    ops: Vec<CapturedOp>,
}

impl CpuGraph {
    pub fn new() -> Self {
        CpuGraph { ops: Vec::new() }
    }

    // pub(crate) fn push<F: Fn() + Send + 'static>(&mut self, f: F) {
    //     self.ops.push(CapturedOp { func: Box::new(f) });
    // }
}

impl GpuGraph for CpuGraph {
    fn replay(&self) {
        for op in &self.ops {
            (op.func)();
        }
    }

    fn node_count(&self) -> usize {
        self.ops.len()
    }
}
