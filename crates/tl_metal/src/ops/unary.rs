//! 単項演算（GPU Shader 使用）

use crate::device::get_device;
use crate::shaders::{self, SHADER_NEG_F32, SHADER_ABS_F32, SHADER_EXP_F32, SHADER_LOG_F32, 
                     SHADER_SQRT_F32, SHADER_TANH_F32, SHADER_SIGMOID_F32, SHADER_RELU_F32,
                     SHADER_SIN_F32, SHADER_COS_F32, SHADER_TAN_F32, SHADER_GELU_F32,
                     SHADER_SILU_F32, SHADER_MISH_F32, SHADER_HARDSWISH_F32,
                     SHADER_HARDSIGMOID_F32, SHADER_LEAKY_RELU_F32, SHADER_ELU_F32,
                     SHADER_LOGICAL_NOT_F32, SHADER_FILL_F32};
use crate::tensor::MetalTensor;
use crate::DType;
use metal::MTLResourceOptions;
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
    pub fn mish_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_MISH_F32) }
    pub fn hardswish_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_HARDSWISH_F32) }
    pub fn hardsigmoid_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_HARDSIGMOID_F32) }
    pub fn logical_not_impl(&self) -> BackendResult<MetalTensor> { self.unary_op(SHADER_LOGICAL_NOT_F32) }

    /// LeakyReLU: max(slope*x, x)
    pub fn leaky_relu_impl(&self, slope: f32) -> BackendResult<MetalTensor> {
        self.unary_op_with_scalar(SHADER_LEAKY_RELU_F32, slope)
    }

    /// ELU: x if x > 0, alpha*(exp(x)-1) otherwise
    pub fn elu_impl(&self, alpha: f32) -> BackendResult<MetalTensor> {
        self.unary_op_with_scalar(SHADER_ELU_F32, alpha)
    }

    /// Fill: 全要素を指定値で埋める (in-place 的)
    pub fn fill_impl(&self, value: f32) -> BackendResult<MetalTensor> {
        let device = get_device();
        let elem_count = self.elem_count();

        let mut shaders_lock = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders_lock
            .get_pipeline(device.device(), SHADER_FILL_F32)
            .map_err(|e| BackendError::InternalError(format!("Failed to get fill pipeline: {}", e)))?;

        let value_buf = device.device().new_buffer_with_data(
            &value as *const f32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let count = elem_count as u32;
        let count_buf = device.device().new_buffer_with_data(
            &count as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let self_buf = self.buffer() as *const metal::Buffer;
        let value_buf_ptr = &value_buf as *const metal::Buffer;
        let count_buf_ptr = &count_buf as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*value_buf_ptr), 0);
                encoder.set_buffer(2, Some(&*count_buf_ptr), 0);
            }

            let (grid_size, threads_per_group) = shaders::compute_thread_groups(elem_count, pipeline);
            encoder.dispatch_thread_groups(grid_size, threads_per_group);
        });

        // Return clone with same shape
        Ok(MetalTensor::from_slice(&self.to_vec::<f32>(), MetalTensor::shape(self), DType::F32))
    }

    fn unary_op(&self, shader_name: &str) -> BackendResult<MetalTensor> {
        match MetalTensor::dtype(self) {
            DType::F32 => self.unary_op_gpu(shader_name),
            _ => Err(BackendError::DeviceError(format!("{} for {:?}", shader_name, MetalTensor::dtype(self)))),
        }
    }

    /// スカラーパラメータ付き単項演算の GPU 実行
    /// シェーダ: input[[buffer(0)]], scalar[[buffer(1)]], output[[buffer(2)]]
    fn unary_op_with_scalar(&self, shader_name: &str, scalar: f32) -> BackendResult<MetalTensor> {
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

    /// 単項演算の GPU 実行 — 非同期ディスパッチ
    fn unary_op_gpu(&self, shader_name: &str) -> BackendResult<MetalTensor> {
        let result = MetalTensor::uninit(MetalTensor::shape(self), MetalTensor::dtype(self));
        let device = get_device();

        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), shader_name)
            .map_err(|e| BackendError::InternalError(format!("Failed to get shader pipeline ({}): {}", shader_name, e)))?;

        let self_buf = self.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;
        let elem_count = self.elem_count();

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*result_buf), 0);
            }

            let (grid_size, threads_per_group) = shaders::compute_thread_groups(elem_count, pipeline);
            encoder.dispatch_thread_groups(grid_size, threads_per_group);
        });

        Ok(result)
    }
}
