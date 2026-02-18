//! インデックス操作 — Metal GPU シェーダー実装
//! slice は shape.rs の GPU 実装に委譲、embedding は専用シェーダー

use crate::device::get_device;
use crate::tensor::MetalTensor;
use metal::{ComputePipelineState, MTLSize};
use tl_backend::{BackendResult, BackendError};

/// Embedding 用 Metal シェーダー
const EMBEDDING_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// embedding lookup: out[t*D+d] = emb[idx[t]*D+d]
kernel void embedding_f32(
    device const float* emb [[buffer(0)]],
    device const float* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint t = gid.y;
    if (d >= dim) return;
    
    uint idx = uint(indices[t]);
    output[t * dim + d] = emb[idx * dim + d];
}
"#;

static EMBEDDING_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();

fn get_embedding_pipeline() -> &'static ComputePipelineState {
    EMBEDDING_PIPELINE.get_or_init(|| {
        let device = get_device();
        let options = metal::CompileOptions::new();
        let library = device
            .device()
            .new_library_with_source(EMBEDDING_SHADER, &options)
            .expect("Failed to compile embedding shader");
        let function = library
            .get_function("embedding_f32", None)
            .expect("embedding_f32 not found");
        device
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create embedding pipeline")
    })
}

impl MetalTensor {
    /// スライス（1軸のみ）— shape.rs の GPU 実装に委譲
    pub fn slice(&self, axis: usize, start: usize, len: usize) -> BackendResult<MetalTensor> {
        self.slice_impl(axis, start, len)
    }

    /// embedding lookup — Metal GPU シェーダー実装
    pub fn embedding_impl(&self, other: &MetalTensor) -> BackendResult<MetalTensor> {
        let self_shape = MetalTensor::shape(self);
        let other_shape = MetalTensor::shape(other);

        // 引数順序の判別: weights は 2D [V, D]、indices は 1D or 2D
        let (weights, indices, indices_shape) = if self_shape.len() == 2 && (other_shape.len() == 1 || other_shape.len() == 2) {
            // self=weights(2D), other=indices(1D/2D)
            (self, other, other_shape.to_vec())
        } else if other_shape.len() == 2 && (self_shape.len() == 1 || self_shape.len() == 2 && self_shape[1] > other_shape[1]) {
            // self=indices, other=weights
            (other, self, self_shape.to_vec())
        } else {
             return Err(BackendError::ShapeMismatch(format!(
                 "embedding_impl expects (2D weights, 1D/2D indices), got shapes {:?} and {:?}",
                 self_shape, other_shape
             )));
        };

        let dim = MetalTensor::shape(weights)[1];

        // 2D indices の場合: flatten → embed → reshape
        let (flat_indices, total_tokens) = if indices_shape.len() == 2 {
            let batch = indices_shape[0];
            let seq = indices_shape[1];
            let total = batch * seq;
            // flatten to [total]
            let flat = MetalTensor::from_buffer_shared(
                indices.buffer_arc().clone(),
                vec![total],
                MetalTensor::dtype(indices),
            );
            (flat, total)
        } else {
            let total = indices_shape[0];
            (MetalTensor::from_buffer_shared(
                indices.buffer_arc().clone(),
                vec![total],
                MetalTensor::dtype(indices),
            ), total)
        };

        // GPU embedding lookup: flat_indices [T] → result [T, D]
        let result = MetalTensor::uninit(&[total_tokens, dim], weights.dtype());
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_embedding_pipeline();

        objc::rc::autoreleasepool(|| {
            let dim_buf = device.device().new_buffer_with_data(
                &(dim as u32) as *const u32 as *const _,
                4,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(weights.buffer()), 0);
            encoder.set_buffer(1, Some(flat_indices.buffer()), 0);
            encoder.set_buffer(2, Some(result.buffer()), 0);
            encoder.set_buffer(3, Some(&dim_buf), 0);

            let tpg = MTLSize::new(dim.min(256) as u64, total_tokens.min(4) as u64, 1);
            let grid = MTLSize::new(
                ((dim + tpg.width as usize - 1) / tpg.width as usize) as u64,
                ((total_tokens + tpg.height as usize - 1) / tpg.height as usize) as u64,
                1,
            );
            encoder.dispatch_thread_groups(grid, tpg);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        });

        // 2D indices の場合: [T, D] → [B, S, D] に reshape
        if indices_shape.len() == 2 {
            let batch = indices_shape[0];
            let seq = indices_shape[1];
            Ok(MetalTensor::from_buffer_shared(
                result.buffer_arc().clone(),
                vec![batch, seq, dim],
                weights.dtype(),
            ))
        } else {
            Ok(result)
        }
    }

}
