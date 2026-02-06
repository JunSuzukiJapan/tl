//! Metal テンソル

use crate::buffer_pool::{pool_acquire, pool_release};
use crate::device::{get_device, MetalDevice};
use crate::{shape_to_bytes, DType};
use metal::{Buffer, MTLResourceOptions};
use std::sync::Arc;

/// Metal GPU テンソル
pub struct MetalTensor {
    /// GPU バッファ
    buffer: Arc<Buffer>,
    /// 形状
    shape: Vec<usize>,
    /// データ型
    dtype: DType,
    /// デバイス
    device: Arc<MetalDevice>,
}

impl MetalTensor {
    /// 新しいテンソルを作成（未初期化）
    pub fn uninit(shape: &[usize], dtype: DType) -> Self {
        let device = get_device();
        let size = shape_to_bytes(shape, dtype);
        let options = MTLResourceOptions::StorageModeShared;

        // プールから取得を試みる
        let buffer = pool_acquire(size, options).unwrap_or_else(|| {
            // プールになければ新規確保
            Arc::new(device.allocate_buffer(size, options))
        });

        MetalTensor {
            buffer,
            shape: shape.to_vec(),
            dtype,
            device,
        }
    }

    /// ゼロで初期化されたテンソルを作成
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let tensor = Self::uninit(shape, dtype);
        // バッファをゼロクリア
        let ptr = tensor.buffer.contents() as *mut u8;
        let size = shape_to_bytes(shape, dtype);
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }
        tensor
    }

    /// スライスからテンソルを作成
    pub fn from_slice<T: Copy>(data: &[T], shape: &[usize], dtype: DType) -> Self {
        let tensor = Self::uninit(shape, dtype);
        let ptr = tensor.buffer.contents() as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        tensor
    }

    /// 形状
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// データ型
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 要素数
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// バッファへの参照
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// データを CPU にコピー
    pub fn to_vec<T: Copy + Default>(&self) -> Vec<T> {
        let count = self.elem_count();
        let ptr = self.buffer.contents() as *const T;
        let mut result = vec![T::default(); count];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), count);
        }
        result
    }
}

impl Drop for MetalTensor {
    fn drop(&mut self) {
        // バッファをプールに返却（解放しない）
        pool_release(self.buffer.clone());
    }
}
