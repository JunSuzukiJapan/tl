//! GpuStream — GPU コマンドストリームの抽象化
//!
//! Metal の CommandStream / CUDA の cudaStream_t を統一する
//! トレイト。同期・キャプチャ操作の共通インターフェースを提供する。

use crate::graph::GpuGraph;

/// GPU コマンドストリーム
///
/// GPU カーネルの投入・同期・キャプチャを抽象化する。
/// 各バックエンド（Metal / CUDA）が実装する。
pub trait GpuStream {
    /// このストリームが生成するグラフの型
    type Graph: GpuGraph;

    /// すべての未完了コマンドを送信し、完了を待つ。
    /// CPU が GPU 結果を読み取る前に呼ぶ。
    fn synchronize(&mut self);

    /// 未同期のコマンドがあるかどうか。
    fn needs_sync(&self) -> bool;

    /// キャプチャ開始。以後のカーネル投入を記録する。
    fn begin_capture(&mut self);

    /// キャプチャ終了。記録されたカーネルシーケンスをグラフとして返す。
    fn end_capture(&mut self) -> Self::Graph;

    /// キャプチャ中かどうか。
    fn is_capturing(&self) -> bool;
}
