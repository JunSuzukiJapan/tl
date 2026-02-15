//! 単項演算（GPU Shader 使用）

use crate::device::get_device;
use crate::shaders::{self, SHADER_NEG_F32, SHADER_ABS_F32, SHADER_EXP_F32, SHADER_LOG_F32, 
                     SHADER_SQRT_F32, SHADER_TANH_F32, SHADER_SIGMOID_F32, SHADER_RELU_F32,
                     SHADER_SIN_F32, SHADER_COS_F32, SHADER_TAN_F32, SHADER_GELU_F32,
                     SHADER_SILU_F32};
use crate::tensor::MetalTensor;
use crate::DType;
use tl_backend::{BackendResult, BackendError};

impl MetalTensor {
    pub fn neg_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_NEG_F32) }
    pub fn abs_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_ABS_F32) }
    pub fn exp_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_EXP_F32) }
    pub fn log_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_LOG_F32) }
    pub fn sqrt_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_SQRT_F32) }
    pub fn tanh_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_TANH_F32) }
    pub fn sigmoid_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_SIGMOID_F32) }
    pub fn relu_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_RELU_F32) }
    pub fn sin_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_SIN_F32) }
    pub fn cos_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_COS_F32) }
    pub fn tan_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_TAN_F32) }
    pub fn gelu_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_GELU_F32) }
    pub fn silu_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_SILU_F32) }

    fn unary_op(&self, shader_name: &str) -> BackendResult<MetalTensor> {
        match MetalTensor::dtype(self) {
            DType::F32 => self.unary_op_gpu(shader_name),
            _ => Err(BackendError::DeviceError(format!("{} for {:?}", shader_name, MetalTensor::dtype(self)))),
        }
    }

    fn unary_op_gpu(&self, shader_name: &str) -> BackendResult<MetalTensor> {
        let result = MetalTensor::uninit(MetalTensor::shape(self), MetalTensor::dtype(self));
        let device = get_device();
        let command_queue = device.command_queue();

        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), shader_name)
            .map_err(|e| BackendError::InternalError(format!("Failed to get shader pipeline ({}): {}", shader_name, e)))?;

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(result.buffer()), 0);

        let (grid_size, threads_per_group) = shaders::compute_thread_groups(self.elem_count(), pipeline);
        encoder.dispatch_thread_groups(grid_size, threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(result)
    }
}
