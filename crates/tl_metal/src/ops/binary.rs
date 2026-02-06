//! 二項演算（GPU Shader 使用）

use crate::device::get_device;
use crate::shaders::{self, SHADER_ADD_F32, SHADER_MUL_F32};
use crate::tensor::MetalTensor;
use crate::DType;

impl MetalTensor {
    /// 要素ごとの加算（GPU）
    pub fn add(&self, other: &MetalTensor) -> MetalTensor {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch");
        assert_eq!(self.dtype(), other.dtype(), "DType mismatch");

        match self.dtype() {
            DType::F32 => self.binary_op_gpu(other, SHADER_ADD_F32),
            _ => unimplemented!("add for {:?}", self.dtype()),
        }
    }

    /// 要素ごとの乗算（GPU）
    pub fn mul(&self, other: &MetalTensor) -> MetalTensor {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch");
        assert_eq!(self.dtype(), other.dtype(), "DType mismatch");

        match self.dtype() {
            DType::F32 => self.binary_op_gpu(other, SHADER_MUL_F32),
            _ => unimplemented!("mul for {:?}", self.dtype()),
        }
    }

    /// 二項演算の GPU 実行
    fn binary_op_gpu(&self, other: &MetalTensor, shader_name: &str) -> MetalTensor {
        let result = MetalTensor::uninit(self.shape(), self.dtype());
        let device = get_device();
        let command_queue = device.command_queue();

        // Shader パイプラインを取得
        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), shader_name)
            .expect("Failed to get shader pipeline");

        // コマンドバッファとエンコーダ
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(other.buffer()), 0);
        encoder.set_buffer(2, Some(result.buffer()), 0);

        // スレッドグループサイズを計算
        let (grid_size, threads_per_group) = shaders::compute_thread_groups(self.elem_count(), pipeline);
        encoder.dispatch_thread_groups(grid_size, threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        result
    }
}
