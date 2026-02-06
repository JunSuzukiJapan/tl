//! スカラー演算

use crate::device::get_device;
use crate::shaders::{self, SHADER_ADD_SCALAR_F32, SHADER_MUL_SCALAR_F32, SHADER_CLAMP_F32};
use crate::tensor::MetalTensor;
use metal::MTLResourceOptions;

impl MetalTensor {
    pub fn add_scalar_impl(&self, scalar: f32) -> MetalTensor {
        let result = MetalTensor::uninit(MetalTensor::shape(self), MetalTensor::dtype(self));
        let device = get_device();
        let command_queue = device.command_queue();

        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), SHADER_ADD_SCALAR_F32)
            .expect("Failed to get shader pipeline");

        let scalar_buf = device.device().new_buffer_with_data(
            &scalar as *const f32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(&scalar_buf), 0);
        encoder.set_buffer(2, Some(result.buffer()), 0);

        let (grid_size, threads_per_group) = shaders::compute_thread_groups(self.elem_count(), pipeline);
        encoder.dispatch_thread_groups(grid_size, threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        result
    }

    pub fn mul_scalar_impl(&self, scalar: f32) -> MetalTensor {
        let result = MetalTensor::uninit(MetalTensor::shape(self), MetalTensor::dtype(self));
        let device = get_device();
        let command_queue = device.command_queue();

        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), SHADER_MUL_SCALAR_F32)
            .expect("Failed to get shader pipeline");

        let scalar_buf = device.device().new_buffer_with_data(
            &scalar as *const f32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(&scalar_buf), 0);
        encoder.set_buffer(2, Some(result.buffer()), 0);

        let (grid_size, threads_per_group) = shaders::compute_thread_groups(self.elem_count(), pipeline);
        encoder.dispatch_thread_groups(grid_size, threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        result
    }

    pub fn sub_scalar_impl(&self, scalar: f32) -> MetalTensor {
        self.add_scalar_impl(-scalar)
    }

    pub fn div_scalar_impl(&self, scalar: f32) -> MetalTensor {
        self.mul_scalar_impl(1.0 / scalar)
    }

    pub fn clamp_impl(&self, min_val: f32, max_val: f32) -> MetalTensor {
        let result = MetalTensor::uninit(MetalTensor::shape(self), MetalTensor::dtype(self));
        let device = get_device();
        let command_queue = device.command_queue();

        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), SHADER_CLAMP_F32)
            .expect("Failed to get shader pipeline");

        let min_buf = device.device().new_buffer_with_data(
            &min_val as *const f32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let max_buf = device.device().new_buffer_with_data(
            &max_val as *const f32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(&min_buf), 0);
        encoder.set_buffer(2, Some(&max_buf), 0);
        encoder.set_buffer(3, Some(result.buffer()), 0);

        let (grid_size, threads_per_group) = shaders::compute_thread_groups(self.elem_count(), pipeline);
        encoder.dispatch_thread_groups(grid_size, threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        result
    }
}
