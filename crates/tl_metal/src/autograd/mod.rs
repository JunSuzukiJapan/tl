//! Autograd - 自動微分
//!
//! 計算グラフを構築し、backward で勾配を計算する。
//! MetalTensor に直接 AutogradMeta を埋め込む設計。

pub mod ops;

use crate::tensor::MetalTensor;

/// 勾配関数のトレイト
pub trait GradFn {
    /// 出力勾配から入力勾配を計算
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor>;
    
    /// 入力ノードへの生ポインタ参照
    /// テンソルは JIT 実行中に解放されないため安全
    fn inputs(&self) -> Vec<*mut MetalTensor>;
}
