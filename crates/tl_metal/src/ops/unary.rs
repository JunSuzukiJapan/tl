//! 単項演算（GPU Shader 使用）

use crate::device::get_device;
use crate::shaders::{self, SHADER_NEG_F32, SHADER_ABS_F32, SHADER_EXP_F32, SHADER_LOG_F32, 
                     SHADER_SQRT_F32, SHADER_TANH_F32, SHADER_SIGMOID_F32, SHADER_RELU_F32};
use crate::tensor::MetalTensor;
use crate::DType;

impl MetalTensor {
    /// 符号反転
    pub fn neg(&self) -> MetalTensor {
        self.unary_op(SHADER_NEG_F32)
    }

    /// 絶対値
    pub fn abs(&self) -> MetalTensor {
        self.unary_op(SHADER_ABS_F32)
    }

    /// 指数関数
    pub fn exp(&self) -> MetalTensor {
        self.unary_op(SHADER_EXP_F32)
    }

    /// 自然対数
    pub fn log(&self) -> MetalTensor {
        self.unary_op(SHADER_LOG_F32)
    }

    /// 平方根
    pub fn sqrt(&self) -> MetalTensor {
        self.unary_op(SHADER_SQRT_F32)
    }

    /// 双曲線正接
    pub fn tanh(&self) -> MetalTensor {
        self.unary_op(SHADER_TANH_F32)
    }

    /// シグモイド関数
    pub fn sigmoid(&self) -> MetalTensor {
        self.unary_op(SHADER_SIGMOID_F32)
    }

    /// ReLU
    pub fn relu(&self) -> MetalTensor {
        self.unary_op(SHADER_RELU_F32)
    }

    /// 単項演算の GPU 実行
    fn unary_op(&self, shader_name: &str) -> MetalTensor {
        match self.dtype() {
            DType::F32 => self.unary_op_gpu(shader_name),
            _ => unimplemented!("{} for {:?}", shader_name, self.dtype()),
        }
    }

    /// 単項演算の GPU 実行（内部）
    fn unary_op_gpu(&self, shader_name: &str) -> MetalTensor {
        let result = MetalTensor::uninit(self.shape(), self.dtype());
        let device = get_device();
        let command_queue = device.command_queue();

        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), shader_name)
            .expect("Failed to get shader pipeline");

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

        result
    }
}
