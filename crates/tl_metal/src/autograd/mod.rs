//! Autograd - 自動微分
//!
//! 計算グラフを構築し、backward で勾配を計算する。
//! MetalTensor に直接 AutogradMeta を埋め込む設計。
//! V5.0: Arc ベースの TensorRef で入力テンソルの生存を保証。

pub mod ops;

use crate::tensor::{MetalTensor, TensorRef};

/// 勾配関数のトレイト
pub trait GradFn {
    /// 出力勾配から入力勾配を計算
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor>;
    
    /// 入力ノードへの Arc 参照
    /// TensorRef (Arc) で入力テンソルの生存を保証する。
    fn inputs(&self) -> Vec<TensorRef>;
}
