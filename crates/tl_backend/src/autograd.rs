//! GpuAutograd トレイト定義 - 自動微分

use crate::tensor::GpuTensor;
use crate::ops::GpuOps;

/// 自動微分対応テンソル
pub trait GpuVar: Clone {
    type Tensor: GpuTensor;
    
    /// テンソルデータを取得
    fn data(&self) -> Self::Tensor;
    
    /// 勾配を取得
    fn grad(&self) -> Option<Self::Tensor>;
    
    /// 勾配をゼロクリア
    fn zero_grad(&self);
    
    /// 計算グラフから切り離す
    fn detach(&self) -> Self;
    
    /// 逆伝播
    fn backward(&self);
    
    /// requires_grad を確認
    fn requires_grad(&self) -> bool;
}

/// 自動微分演算トレイト
pub trait GpuAutograd: GpuOps {
    type Var: GpuVar<Tensor = Self>;
    
    /// テンソルを Var に変換
    fn to_var(&self, requires_grad: bool) -> Self::Var;
}

/// Var 間の演算トレイト
pub trait GpuVarOps: GpuVar {
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;
    fn pow(&self, other: &Self) -> Self;
    fn matmul(&self, other: &Self) -> Self;
    fn sumall(&self) -> Self;
    fn relu(&self) -> Self;
    fn softmax(&self, axis: i32) -> Self;
}
