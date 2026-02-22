//! CommandStream — 非同期コマンドバッファ管理
//!
//! Metal の Command Buffer に複数のカーネルをエンコードし、
//! データの読み取りが必要になるまで commit/wait を遅延させる。
//!
//! ## 設計方針
//! - 各演算は `encode_kernel()` を呼んでカーネルをエンコードするだけ
//! - `commit()` は呼ぶが `wait_until_completed()` は呼ばない
//! - `synchronize()` は `to_vec()`, `item()` など CPU 読み取り時にのみ呼ぶ
//! - 同一 CommandBuffer 内で複数エンコーダを順次利用（エンコーダ間にバリア自動挿入）
//!
//! ## キャプチャ・リプレイ
//! `begin_capture()` ～ `end_capture()` でカーネルシーケンスを記録し、
//! `MetalGraph::replay()` で高速再実行できる。

use crate::device::get_device;
use crate::graph::{CapturedKernel, MetalGraph};
use metal::{CommandBuffer, CommandBufferRef, ComputeCommandEncoderRef};
use std::sync::{LazyLock, Mutex, MutexGuard};

/// 最大バッチサイズ（この数のカーネルを蓄積したら自動コミット）
const MAX_BATCH_SIZE: usize = 128;

/// グローバルコマンドストリーム
pub struct CommandStream {
    /// 現在のコマンドバッファ（commit 済みだが wait していないものも含む）
    pending_buffers: Vec<CommandBuffer>,
    /// 現在エンコード中のコマンドバッファ
    current_buffer: Option<CommandBuffer>,
    /// 現在のバッチ内のカーネル数
    batch_count: usize,
    /// キャプチャモード: Some の場合、カーネルを記録する
    capturing: Option<Vec<CapturedKernel>>,
}

impl CommandStream {
    pub fn new() -> Self {
        CommandStream {
            pending_buffers: Vec::new(),
            current_buffer: None,
            batch_count: 0,
            capturing: None,
        }
    }

    /// 現在のコマンドバッファを取得（なければ新規作成）
    fn ensure_buffer(&mut self) -> &CommandBufferRef {
        if self.current_buffer.is_none() {
            let device = get_device();
            let cb = device.command_queue().new_command_buffer().to_owned();
            self.current_buffer = Some(cb);
        }
        self.current_buffer.as_ref().unwrap()
    }

    /// 外部（graph.rs）から呼び出し可能な ensure_buffer
    pub fn ensure_buffer_pub(&mut self) -> &CommandBufferRef {
        self.ensure_buffer()
    }

    /// バッチカウントをインクリメントし、必要に応じてコミット
    pub fn inc_batch(&mut self) {
        self.batch_count += 1;
        if self.batch_count >= MAX_BATCH_SIZE {
            self.commit_current();
        }
    }

    /// カーネルをエンコードする（非キャプチャ対応版）
    ///
    /// `encode_fn` にはエンコーダが渡される。
    /// エンコーダのライフタイム管理（set_pipeline, dispatch, end_encoding）は
    /// コールバック内で完結させること。
    ///
    /// 戻り後、カーネルはエンコード済みだがまだ GPU 実行は開始していない可能性がある。
    pub fn encode<F>(&mut self, encode_fn: F)
    where
        F: FnOnce(&ComputeCommandEncoderRef),
    {
        objc::rc::autoreleasepool(|| {
            let cb = self.ensure_buffer();
            let encoder = cb.new_compute_command_encoder();
            encode_fn(encoder);
            encoder.end_encoding();
            self.batch_count += 1;

            // バッチサイズに達したら自動コミット
            if self.batch_count >= MAX_BATCH_SIZE {
                self.commit_current();
            }
        });
    }

    /// キャプチャ対応カーネルエンコード
    ///
    /// キャプチャモード中は実行しつつ記録する。
    /// 非キャプチャモード中は通常の `encode` と同じ動作。
    pub fn encode_capturable<F>(&mut self, encode_fn: F)
    where
        F: Fn(&ComputeCommandEncoderRef) + 'static,
    {
        objc::rc::autoreleasepool(|| {
            // 実行
            let cb = self.ensure_buffer();
            let encoder = cb.new_compute_command_encoder();
            encode_fn(encoder);
            encoder.end_encoding();
            self.batch_count += 1;

            if self.batch_count >= MAX_BATCH_SIZE {
                self.commit_current();
            }
        });

        // キャプチャモード中なら記録
        if let Some(ref mut captured) = self.capturing {
            captured.push(CapturedKernel::new(encode_fn));
        }
    }

    /// 現在のコマンドバッファをコミットし、完了を待つ
    /// バッチング（複数カーネルを1つのCBにまとめる）の恩恵は維持しつつ、
    /// コミット後は確実にGPU完了を保証する。
    fn commit_current(&mut self) {
        if let Some(cb) = self.current_buffer.take() {
            objc::rc::autoreleasepool(|| {
                cb.commit();
                cb.wait_until_completed();
            });
            self.batch_count = 0;
        }
    }

    /// すべての未完了コマンドを送信し、完了を待つ
    pub fn synchronize(&mut self) {
        // 現在のバッファがあればコミット
        self.commit_current();

        // すべての pending バッファの完了を待つ
        for cb in self.pending_buffers.drain(..) {
            cb.wait_until_completed();
        }
    }

    /// 未同期のコマンドがあるかどうか
    pub fn needs_sync(&self) -> bool {
        self.current_buffer.is_some() || !self.pending_buffers.is_empty()
    }

    // ============================================================
    // キャプチャ・リプレイ
    // ============================================================

    /// キャプチャ開始
    ///
    /// 以後の `encode_capturable()` で実行されるカーネルを記録する。
    /// 先行コマンドは同期される。
    pub fn begin_capture(&mut self) {
        self.synchronize();
        self.capturing = Some(Vec::new());
    }

    /// キャプチャ終了 → MetalGraph を返す
    ///
    /// 記録を停止し、蓄積されたカーネルシーケンスを `MetalGraph` として返す。
    pub fn end_capture(&mut self) -> MetalGraph {
        let kernels = self.capturing.take()
            .expect("end_capture called without begin_capture");
        self.synchronize();
        MetalGraph::new(kernels)
    }

    /// キャプチャ中かどうか
    pub fn is_capturing(&self) -> bool {
        self.capturing.is_some()
    }
}

impl tl_backend::stream::GpuStream for CommandStream {
    type Graph = MetalGraph;

    fn synchronize(&mut self) {
        self.synchronize();
    }

    fn needs_sync(&self) -> bool {
        self.needs_sync()
    }

    fn begin_capture(&mut self) {
        self.begin_capture();
    }

    fn end_capture(&mut self) -> MetalGraph {
        self.end_capture()
    }

    fn is_capturing(&self) -> bool {
        self.is_capturing()
    }
}

impl Drop for CommandStream {
    fn drop(&mut self) {
        self.synchronize();
    }
}

/// グローバル CommandStream インスタンス
static GLOBAL_STREAM: LazyLock<Mutex<CommandStream>> =
    LazyLock::new(|| Mutex::new(CommandStream::new()));

/// グローバルストリームのロックを取得
pub fn get_stream() -> MutexGuard<'static, CommandStream> {
    GLOBAL_STREAM.lock().unwrap()
}

/// ストリームを同期（GPU 完了を待つ）
/// `to_vec()`, `item()`, `print()` など CPU 読み取り前に呼ぶ。
pub fn sync_stream() {
    get_stream().synchronize();
}

/// カーネルをストリームにエンコードするヘルパー
///
/// 典型的な使い方:
/// ```ignore
/// stream_encode(|encoder| {
///     encoder.set_compute_pipeline_state(pipeline);
///     encoder.set_buffer(0, Some(input_buf), 0);
///     encoder.set_buffer(1, Some(output_buf), 0);
///     let (grid, tpg) = compute_thread_groups(count, pipeline);
///     encoder.dispatch_thread_groups(grid, tpg);
/// });
/// ```
pub fn stream_encode<F>(encode_fn: F)
where
    F: FnOnce(&ComputeCommandEncoderRef),
{
    get_stream().encode(encode_fn);
}

/// ストリームを同期してからコマンドをエンコードし、再度同期する。
/// reduction のように結果をすぐ CPU で読む必要がある演算用。
pub fn stream_encode_sync<F>(encode_fn: F)
where
    F: FnOnce(&ComputeCommandEncoderRef),
{
    let mut stream = get_stream();
    // まず先行するコマンドを同期
    stream.synchronize();
    // 新しいカーネルをエンコード
    stream.encode(encode_fn);
    // この結果を CPU で読むために同期
    stream.synchronize();
}
