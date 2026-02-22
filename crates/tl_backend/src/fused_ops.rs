//! GpuFusedOps — 融合カーネルのトレイト定義
//!
//! 複数の GPU カーネルを1つに融合し、中間バッファと
//! メモリ帯域幅のオーバーヘッドを排除する。
//! Metal / CUDA の各バックエンドが実装する。

use crate::error::BackendError;

type Result<T> = std::result::Result<T, BackendError>;

/// GPU 融合操作トレイト
///
/// 各メソッドは、複数の個別操作を1つのカーネルで実行する。
/// 非融合版と同じ結果を返すが、中間バッファの割り当てが不要。
pub trait GpuFusedOps: Sized {
    // ================================================================
    // Tier 1: LLM ホットパス
    // ================================================================

    /// `silu(gate) * up` を1カーネルで実行
    ///
    /// LLaMA の FFN (SwiGLU) で使用。
    /// 通常は `gate.silu() * up` で中間テンソルが必要だが、融合版では不要。
    fn fused_silu_mul(&self, up: &Self) -> Result<Self>;

    /// RMSNorm を1パスで実行
    ///
    /// 通常の2パス（二乗平均 → 正規化）を1パスに統合。
    /// `output[i] = input[i] * rsqrt(mean(input^2) + eps) * weight[i]`
    fn fused_rms_norm(&self, weight: &Self, eps: f32) -> Result<Self>;

    /// Residual 加算 + RMSNorm を融合
    ///
    /// `rms_norm(self + residual, weight, eps)` を1カーネルで実行。
    /// Transformer 層の Residual Connection + Normalization パターン。
    fn fused_add_rms_norm(&self, residual: &Self, weight: &Self, eps: f32) -> Result<Self>;

    /// Rotary Positional Embedding を1パスで実行
    ///
    /// cos/sin をオンザフライで計算し、直接適用。
    /// `head_dim` は各ヘッドの次元数。
    fn fused_rotary_emb(&self, freqs: &Self, head_dim: usize) -> Result<Self>;

    // ================================================================
    // Tier 2: 汎用融合
    // ================================================================

    /// `relu(self + other)` を1カーネルで実行
    ///
    /// CNN の標準パターン。`add` → `relu` の2カーネルを1つに。
    fn fused_add_relu(&self, other: &Self) -> Result<Self>;

    /// `gelu(self + bias)` を1カーネルで実行
    ///
    /// Transformer FFN のバイアス加算 + GELU 活性化。
    fn fused_bias_gelu(&self, bias: &Self) -> Result<Self>;
}
