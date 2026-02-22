//! GPU 計算グラフ — キャプチャ・リプレイ抽象化
//!
//! CUDA Graphs 相当の「カーネルシーケンスを記録し、再エンコードなしで再実行」
//! する機能の共通トレイト。Metal / CUDA の各バックエンドが実装する。

/// GPU 計算グラフ
///
/// `begin_capture()` ～ `end_capture()` で記録されたカーネルシーケンスを
/// `replay()` で高速に再実行する。
///
/// ## 使い方
/// ```ignore
/// // 1. キャプチャ
/// stream.begin_capture();
/// // ... GPU カーネル実行 ...
/// let graph = stream.end_capture();
///
/// // 2. リプレイ（何度でも）
/// graph.replay();
/// graph.replay();
/// ```
pub trait GpuGraph: Send + Sync {
    /// グラフに記録されたカーネルシーケンスを再実行する。
    ///
    /// 内部的にはエンコード関数を再呼び出しするが、
    /// パイプライン検索やバッファ設定のオーバーヘッドは最小化される。
    fn replay(&self);

    /// グラフ内のノード（カーネル）数
    fn node_count(&self) -> usize;
}
