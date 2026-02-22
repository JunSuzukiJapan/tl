//! Reduce 演算

use crate::device::get_device;
use crate::shaders::{SHADER_SUMALL_F32, SHADER_ARGMAX_F32, SHADER_ARGMIN_F32};
use crate::tensor::MetalTensor;
use crate::DType;
use metal::{MTLResourceOptions, MTLSize};
use tl_backend::{BackendResult, BackendError};

impl MetalTensor {
    /// 全要素の合計
    pub fn sumall_impl(&self) -> BackendResult<f32> {
        if self.dtype() != DType::F32 {
            return Err(BackendError::TypeMismatch(format!("sumall only supports F32, got {:?}", self.dtype())));
        }
        
        let device = get_device();
        let count = self.elem_count();
        if count == 0 {
            return Ok(0.0);
        }

        // スレッドグループサイズ
        let tg_size: usize = 256;
        let num_groups = (count + tg_size - 1) / tg_size;

        // 部分和バッファ
        let partial_sums = device.allocate_buffer(
            num_groups * 4,
            MTLResourceOptions::StorageModeShared,
        );

        // カウントバッファ
        let count_u32 = count as u32;
        let count_buf = device.device().new_buffer_with_data(
            &count_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let self_buf = self.buffer() as *const metal::Buffer;
        let partial_buf_ptr = &partial_sums as *const metal::Buffer;
        let count_buf_ptr = &count_buf as *const metal::Buffer;

        let mut shaders = crate::shaders::get_shaders().lock().unwrap();
        let pipeline = shaders.get_pipeline(device.device(), SHADER_SUMALL_F32)
            .expect("sumall pipeline");

        crate::command_stream::stream_encode_sync(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*partial_buf_ptr), 0);
                encoder.set_buffer(2, Some(&*count_buf_ptr), 0);
            }

            let grid_size = MTLSize::new(num_groups as u64, 1, 1);
            let threads_per_group = MTLSize::new(tg_size as u64, 1, 1);
            encoder.dispatch_thread_groups(grid_size, threads_per_group);
        });

        // CPU で部分和を合計
        let ptr = partial_sums.contents() as *const f32;
        let mut total = 0.0f32;
        unsafe {
            for i in 0..num_groups {
                total += *ptr.add(i);
            }
        }

        Ok(total)
    }

    /// 全要素の平均
    pub fn mean_all_impl(&self) -> BackendResult<f32> {
        let sum = self.sumall_impl()?;
        let count = self.elem_count();
        if count == 0 {
            return Ok(0.0);
        }
        Ok(sum / count as f32)
    }

    /// 全要素の最大値インデックス（GPU アクセラレーション）
    pub fn argmax_all_impl(&self) -> BackendResult<usize> {
        if self.dtype() != DType::F32 {
            return Err(BackendError::TypeMismatch(format!("argmax only supports F32, got {:?}", self.dtype())));
        }
        
        let count = self.elem_count();
        if count == 0 {
            return Err(BackendError::ArgumentError("argmax called on empty tensor".to_string()));
        }

        let device = get_device();

        let tg_size: usize = 256;
        let num_groups = (count + tg_size - 1) / tg_size;

        let partial_max = device.allocate_buffer(num_groups * 4, MTLResourceOptions::StorageModeShared);
        let partial_idx = device.allocate_buffer(num_groups * 4, MTLResourceOptions::StorageModeShared);

        let count_u32 = count as u32;
        let count_buf = device.device().new_buffer_with_data(
            &count_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let self_buf = self.buffer() as *const metal::Buffer;
        let partial_max_ptr = &partial_max as *const metal::Buffer;
        let partial_idx_ptr = &partial_idx as *const metal::Buffer;
        let count_buf_ptr = &count_buf as *const metal::Buffer;

        let mut shaders = crate::shaders::get_shaders().lock().unwrap();
        let pipeline = shaders.get_pipeline(device.device(), SHADER_ARGMAX_F32)
            .expect("argmax pipeline");

        crate::command_stream::stream_encode_sync(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*partial_max_ptr), 0);
                encoder.set_buffer(2, Some(&*partial_idx_ptr), 0);
                encoder.set_buffer(3, Some(&*count_buf_ptr), 0);
            }

            let grid_size = MTLSize::new(num_groups as u64, 1, 1);
            let threads_per_group = MTLSize::new(tg_size as u64, 1, 1);
            encoder.dispatch_thread_groups(grid_size, threads_per_group);
        });

        // CPU で部分結果を集約
        let max_ptr = partial_max.contents() as *const f32;
        let idx_ptr = partial_idx.contents() as *const u32;
        
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = 0usize;
        unsafe {
            for i in 0..num_groups {
                let val = *max_ptr.add(i);
                if val > best_val {
                    best_val = val;
                    best_idx = *idx_ptr.add(i) as usize;
                }
            }
        }

        Ok(best_idx)
    }

    /// 全要素の最小値インデックス（GPU アクセラレーション）
    pub fn argmin_all_impl(&self) -> BackendResult<usize> {
        if self.dtype() != DType::F32 {
            return Err(BackendError::TypeMismatch(format!("argmin only supports F32, got {:?}", self.dtype())));
        }
        
        let count = self.elem_count();
        if count == 0 {
            return Err(BackendError::ArgumentError("argmin called on empty tensor".to_string()));
        }

        let device = get_device();

        let tg_size: usize = 256;
        let num_groups = (count + tg_size - 1) / tg_size;

        let partial_min = device.allocate_buffer(num_groups * 4, MTLResourceOptions::StorageModeShared);
        let partial_idx = device.allocate_buffer(num_groups * 4, MTLResourceOptions::StorageModeShared);

        let count_u32 = count as u32;
        let count_buf = device.device().new_buffer_with_data(
            &count_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let self_buf = self.buffer() as *const metal::Buffer;
        let partial_min_ptr = &partial_min as *const metal::Buffer;
        let partial_idx_ptr = &partial_idx as *const metal::Buffer;
        let count_buf_ptr = &count_buf as *const metal::Buffer;

        let mut shaders = crate::shaders::get_shaders().lock().unwrap();
        let pipeline = shaders.get_pipeline(device.device(), SHADER_ARGMIN_F32)
            .expect("argmin pipeline");

        crate::command_stream::stream_encode_sync(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*partial_min_ptr), 0);
                encoder.set_buffer(2, Some(&*partial_idx_ptr), 0);
                encoder.set_buffer(3, Some(&*count_buf_ptr), 0);
            }

            let grid_size = MTLSize::new(num_groups as u64, 1, 1);
            let threads_per_group = MTLSize::new(tg_size as u64, 1, 1);
            encoder.dispatch_thread_groups(grid_size, threads_per_group);
        });

        // CPU で部分結果を集約
        let min_ptr = partial_min.contents() as *const f32;
        let idx_ptr = partial_idx.contents() as *const u32;
        
        let mut best_val = f32::INFINITY;
        let mut best_idx = 0usize;
        unsafe {
            for i in 0..num_groups {
                let val = *min_ptr.add(i);
                if val < best_val {
                    best_val = val;
                    best_idx = *idx_ptr.add(i) as usize;
                }
            }
        }

        Ok(best_idx)
    }
}
