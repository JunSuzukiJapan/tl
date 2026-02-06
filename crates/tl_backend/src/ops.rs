//! GpuOps トレイト定義 - 全演算操作

use crate::tensor::GpuTensor;

/// GPU 演算インターフェース
pub trait GpuOps: GpuTensor {
    // ========== 二項演算 ==========
    
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;
    fn pow(&self, other: &Self) -> Self;
    fn matmul(&self, other: &Self) -> Self;
    
    // ========== スカラー演算 ==========
    
    fn add_scalar(&self, scalar: f32) -> Self;
    fn mul_scalar(&self, scalar: f32) -> Self;
    fn sub_scalar(&self, scalar: f32) -> Self;
    fn div_scalar(&self, scalar: f32) -> Self;
    fn clamp(&self, min: f32, max: f32) -> Self;
    
    // ========== 単項演算 ==========
    
    fn neg(&self) -> Self;
    fn abs(&self) -> Self;
    fn exp(&self) -> Self;
    fn log(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;
    fn tanh(&self) -> Self;
    fn sigmoid(&self) -> Self;
    fn relu(&self) -> Self;
    fn gelu(&self) -> Self;
    
    // ========== Reduce 演算 ==========
    
    /// 全要素合計
    fn sumall(&self) -> f32;
    
    /// 全要素平均
    fn mean_all(&self) -> f32;
    
    /// 軸指定合計
    fn sum(&self, axis: i32) -> Self;
    
    /// 軸指定最大値
    fn max(&self, axis: i32) -> Self;
    
    /// 軸指定最小値
    fn min(&self, axis: i32) -> Self;
    
    /// 軸指定最大値インデックス
    fn argmax(&self, axis: i32) -> Self;
    
    /// 全体最大値インデックス
    fn argmax_all(&self) -> usize;
    
    /// 軸指定最小値インデックス
    fn argmin(&self, axis: i32) -> Self;
    
    /// 軸指定平均
    fn mean(&self, axis: i32) -> Self;
    
    // ========== 形状操作 ==========
    
    fn reshape(&self, shape: &[usize]) -> Self;
    fn transpose(&self, dim0: usize, dim1: usize) -> Self;
    fn squeeze(&self, dim: usize) -> Self;
    fn unsqueeze(&self, dim: usize) -> Self;
    fn broadcast_to(&self, shape: &[usize]) -> Self;
    fn narrow(&self, axis: usize, start: usize, len: usize) -> Self;
    fn slice(&self, axis: usize, start: usize, len: usize) -> Self;
    fn contiguous(&self) -> Self;
    
    /// テンソル結合
    fn cat(tensors: &[&Self], axis: usize) -> Self;
    
    // ========== 活性化・特殊演算 ==========
    
    fn softmax(&self, axis: i32) -> Self;
    fn embedding(&self, indices: &Self) -> Self;
    fn tril(&self, diagonal: i32) -> Self;
    fn cross_entropy(&self, target: &Self) -> Self;
    fn repeat_interleave(&self, repeats: usize, axis: usize) -> Self;
    fn index_select(&self, axis: usize, indices: &Self) -> Self;
    
    /// 条件分岐
    fn where_cond(condition: &Self, x: &Self, y: &Self) -> Self;
}
