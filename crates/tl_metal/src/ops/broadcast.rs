//! ブロードキャスト・連結・生成

use crate::tensor::MetalTensor;
use crate::DType;

impl MetalTensor {
    /// ブロードキャスト（形状を拡張）
    /// 例: [3] → [2, 3], [1, 3] → [2, 3]
    pub fn broadcast_to(&self, target_shape: &[usize]) -> MetalTensor {
        let src_shape = self.shape();
        assert!(
            Self::can_broadcast(src_shape, target_shape),
            "Cannot broadcast {:?} to {:?}",
            src_shape,
            target_shape
        );

        // 形状が同じなら何もしない
        if src_shape == target_shape {
            return self.clone_data();
        }

        let src: Vec<f32> = self.to_vec();
        let dst_size: usize = target_shape.iter().product();
        let mut dst = vec![0.0f32; dst_size];

        // ブロードキャストのストライド計算
        let src_ndim = src_shape.len();
        let dst_ndim = target_shape.len();
        let ndim_diff = dst_ndim - src_ndim;

        // パディングされたソース形状
        let mut padded_src_shape = vec![1usize; dst_ndim];
        for i in 0..src_ndim {
            padded_src_shape[ndim_diff + i] = src_shape[i];
        }

        // ストライド計算
        let mut src_strides = vec![0usize; dst_ndim];
        let mut stride = 1usize;
        for i in (0..dst_ndim).rev() {
            if padded_src_shape[i] == target_shape[i] {
                src_strides[i] = stride;
            } else {
                // ブロードキャストの場合ストライドは0
                src_strides[i] = 0;
            }
            stride *= padded_src_shape[i];
        }

        // dst を埋める
        for dst_idx in 0..dst_size {
            let mut src_idx = 0usize;
            let mut tmp = dst_idx;
            
            for dim in (0..dst_ndim).rev() {
                let coord = tmp % target_shape[dim];
                tmp /= target_shape[dim];
                src_idx += coord * src_strides[dim];
            }
            
            dst[dst_idx] = src[src_idx];
        }

        MetalTensor::from_slice(&dst, target_shape, self.dtype())
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
    pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
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
                panic!("Cannot broadcast {:?} and {:?}", a, b);
            }
        }
        result
    }

    /// テンソル結合（cat）
    pub fn cat(tensors: &[&MetalTensor], axis: usize) -> MetalTensor {
        assert!(!tensors.is_empty(), "Cannot cat empty list");
        
        let first_shape = tensors[0].shape();
        let ndim = first_shape.len();
        assert!(axis < ndim, "axis out of range");

        // 結合軸以外の形状が一致することを確認
        for t in tensors.iter().skip(1) {
            assert_eq!(t.shape().len(), ndim, "All tensors must have same ndim");
            for i in 0..ndim {
                if i != axis {
                    assert_eq!(t.shape()[i], first_shape[i], "Shape mismatch at dim {}", i);
                }
            }
        }

        // 出力形状
        let mut out_shape = first_shape.to_vec();
        out_shape[axis] = tensors.iter().map(|t| t.shape()[axis]).sum();

        // 各テンソルのデータを結合
        let mut all_data: Vec<f32> = Vec::new();
        
        // 簡略化: axis=0 の場合は単純に連結
        if axis == 0 {
            for t in tensors {
                all_data.extend(t.to_vec::<f32>());
            }
        } else {
            // 一般的なケース
            let outer_size: usize = first_shape[..axis].iter().product::<usize>().max(1);
            
            for outer in 0..outer_size {
                for t in tensors {
                    let t_data: Vec<f32> = t.to_vec();
                    let inner_size: usize = t.shape()[axis..].iter().product();
                    let start = outer * inner_size;
                    let end = start + inner_size;
                    all_data.extend(&t_data[start..end]);
                }
            }
        }

        MetalTensor::from_slice(&all_data, &out_shape, tensors[0].dtype())
    }

    /// narrow（軸のスライス）
    pub fn narrow(&self, axis: usize, start: usize, len: usize) -> MetalTensor {
        self.slice(axis, start, len)
    }

    /// contiguous（メモリ連続化）
    /// 現在の実装は常に contiguous なので clone を返す
    pub fn contiguous(&self) -> MetalTensor {
        self.clone_data()
    }

    /// arange（連番生成）
    pub fn arange(start: i64, end: i64, dtype: DType) -> MetalTensor {
        let len = (end - start) as usize;
        match dtype {
            DType::F32 => {
                let data: Vec<f32> = (start..end).map(|x| x as f32).collect();
                MetalTensor::from_slice(&data, &[len], dtype)
            }
            DType::I64 => {
                let data: Vec<i64> = (start..end).collect();
                // I64 として保存（注: 現在は F32 バッファのみ対応）
                let data_f32: Vec<f32> = data.iter().map(|x| *x as f32).collect();
                MetalTensor::from_slice(&data_f32, &[len], DType::F32)
            }
            _ => unimplemented!("arange for {:?}", dtype),
        }
    }
}
