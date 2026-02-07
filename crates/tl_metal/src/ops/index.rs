//! インデックス操作 — Metal GPU シェーダー実装
//! slice は shape.rs の GPU 実装に委譲、embedding は専用シェーダー

use crate::device::get_device;
use crate::tensor::MetalTensor;
use metal::{ComputePipelineState, MTLSize};

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
    pub fn slice(&self, axis: usize, start: usize, len: usize) -> MetalTensor {
        self.slice_impl(axis, start, len)
    }

    /// embedding lookup — Metal GPU シェーダー実装
    /// weights: [V, D] (埋め込み行列), indices: [T] (インデックス)
    /// → [T, D]
    ///
    /// 引数順序を自動判別:
    ///   - self=2D, other=1D → self=weights, other=indices (標準)
    ///   - self=1D, other=2D → self=indices, other=weights (TL言語からの呼び出し)
    pub fn embedding_impl(&self, other: &MetalTensor) -> MetalTensor {
        let self_ndim = MetalTensor::shape(self).len();
        let other_ndim = MetalTensor::shape(other).len();

        // 引数順序を自動判別
        let (weights, indices) = if self_ndim == 2 && other_ndim == 1 {
            // 標準: self=weights(2D), other=indices(1D)
            (self, other)
        } else if self_ndim == 1 && other_ndim == 2 {
            // TL 言語: self=indices(1D), other=weights(2D)
            (other, self)
        } else {
            eprintln!(
                "Warning: embedding_impl expects (2D weights, 1D indices), got shapes {:?} and {:?}",
                MetalTensor::shape(self), MetalTensor::shape(other)
            );
            // フォールバック: 最善の推測
            if self_ndim >= other_ndim { (self, other) } else { (other, self) }
        };
        
        let dim = MetalTensor::shape(weights)[1];
        let seq_len = indices.shape()[0];
        
        let result = MetalTensor::uninit(&[seq_len, dim], weights.dtype());
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_embedding_pipeline();

        let dim_buf = device.device().new_buffer_with_data(
            &(dim as u32) as *const u32 as *const _,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(weights.buffer()), 0);
        encoder.set_buffer(1, Some(indices.buffer()), 0);
        encoder.set_buffer(2, Some(result.buffer()), 0);
        encoder.set_buffer(3, Some(&dim_buf), 0);

        let tpg = MTLSize::new(dim.min(256) as u64, seq_len.min(4) as u64, 1);
        let grid = MTLSize::new(
            ((dim + tpg.width as usize - 1) / tpg.width as usize) as u64,
            ((seq_len + tpg.height as usize - 1) / tpg.height as usize) as u64,
            1,
        );
        encoder.dispatch_thread_groups(grid, tpg);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        result
    }
}
