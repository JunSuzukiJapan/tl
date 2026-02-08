//! GpuTensor トレイト定義

use crate::dtype::DType;

/// GPU テンソルの基本インターフェース
pub trait GpuTensor: Clone + Send + Sync + Sized {
    /// 形状を取得
    fn shape(&self) -> &[usize];
    
    /// データ型を取得
    fn dtype(&self) -> DType;
    
    /// 要素数を取得
    fn elem_count(&self) -> usize {
        self.shape().iter().product()
    }
    
    /// F32 データを Vec として取得
    fn to_vec_f32(&self) -> Vec<f32>;
    
    /// I64 データを Vec として取得
    fn to_vec_i64(&self) -> Vec<i64>;
    
    /// F32 スライスからテンソルを作成
    fn from_slice_f32(data: &[f32], shape: &[usize]) -> Self;
    
    /// I64 スライスからテンソルを作成
    fn from_slice_i64(data: &[i64], shape: &[usize]) -> Self;
    
    /// ゼロで初期化
    fn zeros(shape: &[usize], dtype: DType) -> Self;
    
    /// 1 で初期化
    fn ones(shape: &[usize], dtype: DType) -> Self;
    
    /// 正規分布乱数で初期化
    fn randn(shape: &[usize], dtype: DType) -> Self;
    
    /// 連番生成
    fn arange(start: i64, end: i64, dtype: DType) -> Self;
    
    /// データを複製
    fn clone_data(&self) -> Self;

    /// テンソルの Box ポインタを解放し、Drop を発動する。
    /// GPU: Drop → pool_release（プール返却）
    /// CPU: Drop → free（メモリ解放）
    fn release(ptr: *mut Self) {
        if !ptr.is_null() {
            unsafe { let _ = Box::from_raw(ptr); }
        }
    }
}
