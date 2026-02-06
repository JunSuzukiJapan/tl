//! Reduce 演算

use crate::device::get_device;
use crate::shaders::{self, SHADER_SUMALL_F32};
use crate::tensor::MetalTensor;
use crate::DType;
use metal::{MTLResourceOptions, MTLSize};

impl MetalTensor {
    /// 全要素の合計
    pub fn sumall_impl(&self) -> f32 {
        assert_eq!(MetalTensor::dtype(self), DType::F32, "sumall only supports F32");
        
        let device = get_device();
        let command_queue = device.command_queue();
        let count = self.elem_count();

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

        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), SHADER_SUMALL_F32)
            .expect("Failed to get shader pipeline");

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(&partial_sums), 0);
        encoder.set_buffer(2, Some(&count_buf), 0);

        let grid_size = MTLSize::new(num_groups as u64, 1, 1);
        let threads_per_group = MTLSize::new(tg_size as u64, 1, 1);
        encoder.dispatch_thread_groups(grid_size, threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // CPU で部分和を合計
        let ptr = partial_sums.contents() as *const f32;
        let mut total = 0.0f32;
        unsafe {
            for i in 0..num_groups {
                total += *ptr.add(i);
            }
        }

        total
    }

    /// 全要素の平均
    pub fn mean_all_impl(&self) -> f32 {
        self.sumall_impl() / self.elem_count() as f32
    }
}
