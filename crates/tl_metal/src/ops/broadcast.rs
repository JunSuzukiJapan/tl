//! ブロードキャスト・連結・生成
//! broadcast_to, cat は shape.rs の GPU 実装に委譲

use crate::tensor::MetalTensor;
use crate::DType;
use tl_backend::{BackendResult, BackendError};

impl MetalTensor {
    /// ブロードキャスト（形状を拡張）— shape.rs の GPU 実装に委譲
    pub fn broadcast_to(&self, target_shape: &[usize]) -> BackendResult<MetalTensor> {
        let src_shape = self.shape();
        if !Self::can_broadcast(src_shape, target_shape) {
            return Err(BackendError::ShapeMismatch(format!(
                "Cannot broadcast {:?} to {:?}",
                src_shape, target_shape
            )));
        }
        self.broadcast_to_impl(target_shape)
    }

    /// ブロードキャスト可能かチェック
    fn can_broadcast(src: &[usize], dst: &[usize]) -> bool {
        if src.len() > dst.len() {
            return false;
        }
        let offset = dst.len() - src.len();
        for i in 0..src.len() {
            if src[i] != dst[offset + i] && src[i] != 1 {
                return false;
            }
        }
        true
    }

    /// 二つのテンソルのブロードキャスト形状を計算
    pub fn broadcast_shape(a: &[usize], b: &[usize]) -> BackendResult<Vec<usize>> {
        let max_ndim = a.len().max(b.len());
        let mut result = vec![1usize; max_ndim];
        
        for i in 0..max_ndim {
            let ai = if i < max_ndim - a.len() { 1 } else { a[i - (max_ndim - a.len())] };
            let bi = if i < max_ndim - b.len() { 1 } else { b[i - (max_ndim - b.len())] };
            
            if ai == bi {
                result[i] = ai;
            } else if ai == 1 {
                result[i] = bi;
            } else if bi == 1 {
                result[i] = ai;
            } else {
                return Err(BackendError::ShapeMismatch(format!(
                    "Cannot broadcast {:?} and {:?}", a, b
                )));
            }
        }
        Ok(result)
    }

    /// テンソル結合（cat）— shape.rs の GPU 実装に委譲
    pub fn cat(tensors: &[&MetalTensor], axis: usize) -> BackendResult<MetalTensor> {
        MetalTensor::cat_impl(tensors, axis)
    }

    /// narrow（軸のスライス）— shape.rs の GPU 実装に委譲
    pub fn narrow(&self, axis: usize, start: usize, len: usize) -> BackendResult<MetalTensor> {
        self.slice_impl(axis, start, len)
    }

    /// contiguous（メモリ連続化）
    pub fn contiguous(&self) -> BackendResult<MetalTensor> {
        Ok(self.clone_data()?)
    }

    /// arange（連番生成）
    pub fn arange(start: i64, end: i64, dtype: DType) -> BackendResult<MetalTensor> {
        let len = (end - start) as usize;
        match dtype {
            DType::F32 => {
                let data: Vec<f32> = (start..end).map(|x| x as f32).collect();
                Ok(MetalTensor::from_slice(&data, &[len], dtype))
            }
            DType::I64 => {
                let data_f32: Vec<f32> = (start..end).map(|x| x as f32).collect();
                Ok(MetalTensor::from_slice(&data_f32, &[len], DType::F32))
            }
            _ => Err(BackendError::DeviceError(format!("arange for {:?}", dtype))),
        }
    }
}
