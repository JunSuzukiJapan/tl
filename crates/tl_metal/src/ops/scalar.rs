//! スカラー演算

use crate::device::get_device;
use crate::shaders::{self, SHADER_ADD_SCALAR_F32, SHADER_MUL_SCALAR_F32, SHADER_CLAMP_F32,
    SHADER_POW_SCALAR_F32, SHADER_FMOD_SCALAR_F32};
use crate::tensor::MetalTensor;
use metal::MTLResourceOptions;
use tl_backend::{BackendResult, BackendError};

impl MetalTensor {
    pub fn add_scalar_impl(&self, scalar: f32) -> BackendResult<MetalTensor> {
        let result = MetalTensor::uninit(MetalTensor::shape(self), MetalTensor::dtype(self));
        let device = get_device();

        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), SHADER_ADD_SCALAR_F32)
            .map_err(BackendError::InternalError)?;

        let scalar_buf = device.device().new_buffer_with_data(
            &scalar as *const f32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let self_buf = self.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;
        let scalar_buf_ptr = &scalar_buf as *const metal::Buffer;
        let elem_count = self.elem_count();

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*scalar_buf_ptr), 0);
                encoder.set_buffer(2, Some(&*result_buf), 0);
            }

            let (grid_size, threads_per_group) = shaders::compute_thread_groups(elem_count, pipeline);
            encoder.dispatch_thread_groups(grid_size, threads_per_group);
        });

        Ok(result)
    }

    pub fn mul_scalar_impl(&self, scalar: f32) -> BackendResult<MetalTensor> {
        let result = MetalTensor::uninit(MetalTensor::shape(self), MetalTensor::dtype(self));
        let device = get_device();

        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), SHADER_MUL_SCALAR_F32)
            .map_err(BackendError::InternalError)?;

        let scalar_buf = device.device().new_buffer_with_data(
            &scalar as *const f32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let self_buf = self.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;
        let scalar_buf_ptr = &scalar_buf as *const metal::Buffer;
        let elem_count = self.elem_count();

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*scalar_buf_ptr), 0);
                encoder.set_buffer(2, Some(&*result_buf), 0);
            }

            let (grid_size, threads_per_group) = shaders::compute_thread_groups(elem_count, pipeline);
            encoder.dispatch_thread_groups(grid_size, threads_per_group);
        });

        Ok(result)
    }

    pub fn sub_scalar_impl(&self, scalar: f32) -> BackendResult<MetalTensor> {
        self.add_scalar_impl(-scalar)
    }

    pub fn div_scalar_impl(&self, scalar: f32) -> BackendResult<MetalTensor> {
        self.mul_scalar_impl(1.0 / scalar)
    }

    pub fn clamp_impl(&self, min_val: f32, max_val: f32) -> BackendResult<MetalTensor> {
        let result = MetalTensor::uninit(MetalTensor::shape(self), MetalTensor::dtype(self));
        let device = get_device();

        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), SHADER_CLAMP_F32)
            .map_err(BackendError::InternalError)?;

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

        let self_buf = self.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;
        let min_buf_ptr = &min_buf as *const metal::Buffer;
        let max_buf_ptr = &max_buf as *const metal::Buffer;
        let elem_count = self.elem_count();

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*min_buf_ptr), 0);
                encoder.set_buffer(2, Some(&*max_buf_ptr), 0);
                encoder.set_buffer(3, Some(&*result_buf), 0);
            }

            let (grid_size, threads_per_group) = shaders::compute_thread_groups(elem_count, pipeline);
            encoder.dispatch_thread_groups(grid_size, threads_per_group);
        });

        Ok(result)
    }

    pub fn pow_scalar_impl(&self, exp: f32) -> BackendResult<MetalTensor> {
        self.scalar_op_1(SHADER_POW_SCALAR_F32, exp)
    }

    pub fn fmod_scalar_impl(&self, s: f32) -> BackendResult<MetalTensor> {
        self.scalar_op_1(SHADER_FMOD_SCALAR_F32, s)
    }

    fn scalar_op_1(&self, shader_name: &str, scalar: f32) -> BackendResult<MetalTensor> {
        let result = MetalTensor::uninit(MetalTensor::shape(self), MetalTensor::dtype(self));
        let device = get_device();

        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), shader_name)
            .map_err(|e| BackendError::InternalError(format!("Failed to get shader pipeline ({}): {}", shader_name, e)))?;

        let scalar_buf = device.device().new_buffer_with_data(
            &scalar as *const f32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let self_buf = self.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;
        let scalar_buf_ptr = &scalar_buf as *const metal::Buffer;
        let elem_count = self.elem_count();

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*scalar_buf_ptr), 0);
                encoder.set_buffer(2, Some(&*result_buf), 0);
            }

            let (grid_size, threads_per_group) = shaders::compute_thread_groups(elem_count, pipeline);
            encoder.dispatch_thread_groups(grid_size, threads_per_group);
        });

        Ok(result)
    }
}
