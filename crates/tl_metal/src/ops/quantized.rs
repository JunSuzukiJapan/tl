//! 量子化関連のオペレーション

use crate::device::get_device;
use crate::tensor::MetalTensor;
use crate::shaders::{get_shaders, SHADER_DEQUANTIZE_Q4_K, compute_thread_groups};

impl MetalTensor {
    /// Q4_K データをデクォンタイズして F32 Tensor を返す
    /// input: [num_blocks * 144] bytes (as u8 tensor)
    /// output: [num_blocks * 256] floats (as F32 tensor)
    pub fn dequantize_q4_k(&self, target_shape: &[usize]) -> MetalTensor {
        // self is raw bytes buffer.
        // Input size validation?
        // block size = 256 elements = 144 bytes.
        // num_elements = target_shape.product()
        // num_blocks = num_elements / 256
        
        let num_elements: usize = target_shape.iter().product();
        assert!(num_elements % 256 == 0, "Q4_K num_elements must be divisible by 256");
        let num_blocks = num_elements / 256;
        
        // Output tensor
        let result = MetalTensor::uninit(target_shape, crate::DType::F32);
        
        // Pipeline
        let device = get_device();
        let command_queue = device.command_queue();
        let mut shaders = get_shaders().lock().unwrap();
        let pipeline = shaders.get_pipeline(device.device(), SHADER_DEQUANTIZE_Q4_K).expect("Failed to get dequantize pipeline");
        
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(result.buffer()), 0);
        
        let num_blocks_u32 = num_blocks as u32;
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &num_blocks_u32 as *const u32 as *const std::ffi::c_void);
        
        // Dispatch: 1 thread per block
        // num_blocks threads
        let (grid_size, threads_per_group) = compute_thread_groups(num_blocks, pipeline);
        encoder.dispatch_thread_groups(grid_size, threads_per_group);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        result
    }
}
