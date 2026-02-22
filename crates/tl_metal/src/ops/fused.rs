//! 融合カーネル — Metal 実装
//!
//! `GpuFusedOps` トレイトの Metal バックエンド実装。
//! 各メソッドは専用の融合 MSL カーネルを使用し、
//! 中間バッファなしで複数操作を1パスで実行する。

use crate::{MetalTensor, DType};
use crate::shaders::{
    get_shaders, compute_thread_groups,
    SHADER_FUSED_SILU_MUL_F32, SHADER_FUSED_ADD_RELU_F32,
    SHADER_FUSED_BIAS_GELU_F32, SHADER_FUSED_RMS_NORM_F32,
    SHADER_FUSED_ADD_RMS_NORM_F32, SHADER_FUSED_ROTARY_EMB_F32,
};
use crate::device::get_device;
use crate::command_stream::stream_encode;
use metal::MTLSize;
use tl_backend::error::BackendError;
use tl_backend::fused_ops::GpuFusedOps;

type Result<T> = std::result::Result<T, BackendError>;

/// パイプライン取得ヘルパー
fn get_fused_pipeline(name: &str) -> &'static metal::ComputePipelineState {
    let mut shaders = get_shaders().lock().unwrap();
    let device = get_device();
    let pipeline = shaders.get_pipeline(device.device(), name)
        .expect(&format!("Failed to get fused pipeline: {}", name));
    // OnceLock により安全に 'static 参照を返せる
    unsafe { &*(pipeline as *const metal::ComputePipelineState) }
}

impl GpuFusedOps for MetalTensor {
    fn fused_silu_mul(&self, up: &Self) -> Result<Self> {
        let count = self.elem_count();
        assert_eq!(count, up.elem_count(), "silu_mul: shape mismatch");

        let output = MetalTensor::uninit(self.shape(), DType::F32);
        let pipeline = get_fused_pipeline(SHADER_FUSED_SILU_MUL_F32);
        let (grid, tpg) = compute_thread_groups(count, pipeline);

        let self_buf = self.buffer() as *const metal::Buffer;
        let up_buf = up.buffer() as *const metal::Buffer;
        let out_buf = output.buffer() as *const metal::Buffer;
        let count_u32 = count as u32;
        let count_ptr = &count_u32 as *const u32;

        stream_encode(move |encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*up_buf), 0);
                encoder.set_buffer(2, Some(&*out_buf), 0);
                encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, count_ptr as *const _);
            }
            encoder.dispatch_thread_groups(grid, tpg);
        });

        Ok(output)
    }

    fn fused_rms_norm(&self, weight: &Self, eps: f32) -> Result<Self> {
        let shape = self.shape();
        assert!(shape.len() >= 1, "rms_norm: requires at least 1D");
        let d = *shape.last().unwrap();
        let n = self.elem_count() / d;

        let output = MetalTensor::uninit(shape, DType::F32);
        let pipeline = get_fused_pipeline(SHADER_FUSED_RMS_NORM_F32);

        let self_buf = self.buffer() as *const metal::Buffer;
        let w_buf = weight.buffer() as *const metal::Buffer;
        let out_buf = output.buffer() as *const metal::Buffer;
        let n_u32 = n as u32;
        let d_u32 = d as u32;
        let n_ptr = &n_u32 as *const u32;
        let d_ptr = &d_u32 as *const u32;
        let eps_ptr = &eps as *const f32;

        let grid = MTLSize::new(d as u64, n as u64, 1);
        let tpg = MTLSize::new(d.min(256) as u64, 1, 1);

        stream_encode(move |encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*w_buf), 0);
                encoder.set_buffer(2, Some(&*out_buf), 0);
                encoder.set_bytes(3, 4, n_ptr as *const _);
                encoder.set_bytes(4, 4, d_ptr as *const _);
                encoder.set_bytes(5, 4, eps_ptr as *const _);
            }
            encoder.dispatch_thread_groups(grid, tpg);
        });

        Ok(output)
    }

    fn fused_add_rms_norm(&self, residual: &Self, weight: &Self, eps: f32) -> Result<Self> {
        let shape = self.shape();
        assert_eq!(self.elem_count(), residual.elem_count(), "add_rms_norm: shape mismatch");
        let d = *shape.last().unwrap();
        let n = self.elem_count() / d;

        let output = MetalTensor::uninit(shape, DType::F32);
        let pipeline = get_fused_pipeline(SHADER_FUSED_ADD_RMS_NORM_F32);

        let self_buf = self.buffer() as *const metal::Buffer;
        let res_buf = residual.buffer() as *const metal::Buffer;
        let w_buf = weight.buffer() as *const metal::Buffer;
        let out_buf = output.buffer() as *const metal::Buffer;
        let n_u32 = n as u32;
        let d_u32 = d as u32;
        let n_ptr = &n_u32 as *const u32;
        let d_ptr = &d_u32 as *const u32;
        let eps_ptr = &eps as *const f32;

        let grid = MTLSize::new(d as u64, n as u64, 1);
        let tpg = MTLSize::new(d.min(256) as u64, 1, 1);

        stream_encode(move |encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*res_buf), 0);
                encoder.set_buffer(2, Some(&*w_buf), 0);
                encoder.set_buffer(3, Some(&*out_buf), 0);
                encoder.set_bytes(4, 4, n_ptr as *const _);
                encoder.set_bytes(5, 4, d_ptr as *const _);
                encoder.set_bytes(6, 4, eps_ptr as *const _);
            }
            encoder.dispatch_thread_groups(grid, tpg);
        });

        Ok(output)
    }

    fn fused_rotary_emb(&self, freqs: &Self, head_dim: usize) -> Result<Self> {
        let count = self.elem_count();
        let output = MetalTensor::uninit(self.shape(), DType::F32);
        let pipeline = get_fused_pipeline(SHADER_FUSED_ROTARY_EMB_F32);
        let (grid, tpg) = compute_thread_groups(count / 2, pipeline);

        let self_buf = self.buffer() as *const metal::Buffer;
        let freq_buf = freqs.buffer() as *const metal::Buffer;
        let out_buf = output.buffer() as *const metal::Buffer;
        let count_u32 = (count / 2) as u32;
        let hd_u32 = head_dim as u32;
        let count_ptr = &count_u32 as *const u32;
        let hd_ptr = &hd_u32 as *const u32;

        stream_encode(move |encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*freq_buf), 0);
                encoder.set_buffer(2, Some(&*out_buf), 0);
                encoder.set_bytes(3, 4, count_ptr as *const _);
                encoder.set_bytes(4, 4, hd_ptr as *const _);
            }
            encoder.dispatch_thread_groups(grid, tpg);
        });

        Ok(output)
    }

    fn fused_add_relu(&self, other: &Self) -> Result<Self> {
        let count = self.elem_count();
        assert_eq!(count, other.elem_count(), "add_relu: shape mismatch");

        let output = MetalTensor::uninit(self.shape(), DType::F32);
        let pipeline = get_fused_pipeline(SHADER_FUSED_ADD_RELU_F32);
        let (grid, tpg) = compute_thread_groups(count, pipeline);

        let self_buf = self.buffer() as *const metal::Buffer;
        let other_buf = other.buffer() as *const metal::Buffer;
        let out_buf = output.buffer() as *const metal::Buffer;
        let count_u32 = count as u32;
        let count_ptr = &count_u32 as *const u32;

        stream_encode(move |encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*other_buf), 0);
                encoder.set_buffer(2, Some(&*out_buf), 0);
                encoder.set_bytes(3, 4, count_ptr as *const _);
            }
            encoder.dispatch_thread_groups(grid, tpg);
        });

        Ok(output)
    }

    fn fused_bias_gelu(&self, bias: &Self) -> Result<Self> {
        let count = self.elem_count();
        let bias_len = bias.elem_count();

        let output = MetalTensor::uninit(self.shape(), DType::F32);
        let pipeline = get_fused_pipeline(SHADER_FUSED_BIAS_GELU_F32);
        let (grid, tpg) = compute_thread_groups(count, pipeline);

        let self_buf = self.buffer() as *const metal::Buffer;
        let bias_buf = bias.buffer() as *const metal::Buffer;
        let out_buf = output.buffer() as *const metal::Buffer;
        let count_u32 = count as u32;
        let bias_len_u32 = bias_len as u32;
        let count_ptr = &count_u32 as *const u32;
        let bias_len_ptr = &bias_len_u32 as *const u32;

        stream_encode(move |encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*bias_buf), 0);
                encoder.set_buffer(2, Some(&*out_buf), 0);
                encoder.set_bytes(3, 4, count_ptr as *const _);
                encoder.set_bytes(4, 4, bias_len_ptr as *const _);
            }
            encoder.dispatch_thread_groups(grid, tpg);
        });

        Ok(output)
    }
}
