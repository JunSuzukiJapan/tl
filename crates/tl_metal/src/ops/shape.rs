//! 形状操作 — Metal GPU シェーダー実装
//! transpose, broadcast_to, slice, cat を GPU で実行

use crate::device::get_device;
use crate::tensor::MetalTensor;
use metal::{ComputePipelineState, MTLSize};

/// 形状操作用 Metal シェーダー
const SHAPE_OPS_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// transpose 2D: dst[j*rows+i] = src[i*cols+j]
kernel void transpose_2d_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= rows || col >= cols) return;
    dst[col * rows + row] = src[row * cols + col];
}

// broadcast_to: ストライドベースのインデックスマッピング
// padded_src_shape, src_strides は最大 8 次元対応
kernel void broadcast_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& ndim [[buffer(2)]],
    constant uint& dst_size [[buffer(3)]],
    constant uint* dst_shape [[buffer(4)]],
    constant uint* src_strides [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= dst_size) return;
    
    uint src_idx = 0;
    uint tmp = id;
    for (int dim = int(ndim) - 1; dim >= 0; dim--) {
        uint coord = tmp % dst_shape[dim];
        tmp /= dst_shape[dim];
        src_idx += coord * src_strides[dim];
    }
    dst[id] = src[src_idx];
}

// slice: axis 方向のスライスコピー
kernel void slice_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& src_axis_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    constant uint& start [[buffer(5)]],
    constant uint& len [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.x = flat index within (len * inner_size), gid.y = outer
    uint flat = gid.x;
    uint outer = gid.y;
    if (outer >= outer_size || flat >= len * inner_size) return;
    
    uint a = flat / inner_size;
    uint inner = flat % inner_size;
    
    uint src_idx = outer * src_axis_size * inner_size + (start + a) * inner_size + inner;
    uint dst_idx = outer * len * inner_size + a * inner_size + inner;
    dst[dst_idx] = src[src_idx];
}
"#;

static TRANSPOSE_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static BROADCAST_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static SLICE_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();

fn compile_shape_pipeline(function_name: &str) -> ComputePipelineState {
    let device = get_device();
    let options = metal::CompileOptions::new();
    let library = device
        .device()
        .new_library_with_source(SHAPE_OPS_SHADER, &options)
        .unwrap_or_else(|e| panic!("Failed to compile shape shader: {}", e));
    let function = library
        .get_function(function_name, None)
        .unwrap_or_else(|e| panic!("{} not found: {}", function_name, e));
    device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {}", function_name, e))
}

fn get_transpose_pipeline() -> &'static ComputePipelineState {
    TRANSPOSE_PIPELINE.get_or_init(|| compile_shape_pipeline("transpose_2d_f32"))
}
fn get_broadcast_pipeline() -> &'static ComputePipelineState {
    BROADCAST_PIPELINE.get_or_init(|| compile_shape_pipeline("broadcast_f32"))
}
fn get_slice_pipeline() -> &'static ComputePipelineState {
    SLICE_PIPELINE.get_or_init(|| compile_shape_pipeline("slice_f32"))
}

/// u32 パラメータバッファを作成
fn make_u32_buf(v: u32) -> metal::Buffer {
    let device = get_device();
    device.device().new_buffer_with_data(
        &v as *const u32 as *const _,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

/// u32 配列バッファを作成
fn make_u32_array_buf(values: &[u32]) -> metal::Buffer {
    let device = get_device();
    device.device().new_buffer_with_data(
        values.as_ptr() as *const _,
        (values.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

impl MetalTensor {
    /// 形状変更（データコピーなし、参照共有）
    pub fn reshape_impl(&self, new_shape: &[usize]) -> MetalTensor {
        let old_size: usize = MetalTensor::shape(self).iter().product();
        let new_size: usize = new_shape.iter().product();
        assert_eq!(old_size, new_size, "reshape: element count mismatch {} vs {}", old_size, new_size);

        MetalTensor::from_buffer_shared(
            self.buffer_arc().clone(),
            new_shape.to_vec(),
            MetalTensor::dtype(self),
        )
    }

    /// squeeze: サイズ1の次元を削除
    pub fn squeeze_impl(&self, dim: usize) -> MetalTensor {
        let shape = MetalTensor::shape(self);
        assert!(dim < shape.len(), "dim out of range");
        assert_eq!(shape[dim], 1, "squeeze: dimension {} is not 1", dim);

        let mut new_shape: Vec<usize> = shape.to_vec();
        new_shape.remove(dim);
        
        MetalTensor::from_buffer_shared(
            self.buffer_arc().clone(),
            new_shape,
            MetalTensor::dtype(self),
        )
    }

    /// unsqueeze: サイズ1の次元を追加
    pub fn unsqueeze_impl(&self, dim: usize) -> MetalTensor {
        let mut new_shape: Vec<usize> = MetalTensor::shape(self).to_vec();
        assert!(dim <= new_shape.len(), "dim out of range");
        new_shape.insert(dim, 1);
        
        MetalTensor::from_buffer_shared(
            self.buffer_arc().clone(),
            new_shape,
            MetalTensor::dtype(self),
        )
    }

    /// transpose — 2D: Metal GPU シェーダー、N-D: CPU フォールバック
    pub fn transpose_impl(&self, dim0: usize, dim1: usize) -> MetalTensor {
        let shape = MetalTensor::shape(self);
        assert!(dim0 < shape.len() && dim1 < shape.len(), "dim out of range");

        if dim0 == dim1 {
            return self.clone_data();
        }

        // 2D テンソル → 既存 GPU シェーダー
        if shape.len() == 2 {
            let rows = shape[0];
            let cols = shape[1];
            let new_shape = vec![cols, rows];

            let result = MetalTensor::uninit(&new_shape, MetalTensor::dtype(self));
            let device = get_device();
            let command_queue = device.command_queue();
            let pipeline = get_transpose_pipeline();

            let rows_buf = make_u32_buf(rows as u32);
            let cols_buf = make_u32_buf(cols as u32);

            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(self.buffer()), 0);
            encoder.set_buffer(1, Some(result.buffer()), 0);
            encoder.set_buffer(2, Some(&rows_buf), 0);
            encoder.set_buffer(3, Some(&cols_buf), 0);

            let tpg = MTLSize::new(cols.min(16) as u64, rows.min(16) as u64, 1);
            let grid = MTLSize::new(
                ((cols + tpg.width as usize - 1) / tpg.width as usize) as u64,
                ((rows + tpg.height as usize - 1) / tpg.height as usize) as u64,
                1,
            );
            encoder.dispatch_thread_groups(grid, tpg);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            return result;
        }

        // N-D テンソル → CPU フォールバック（ストライドベースのインデックスマッピング）
        let ndim = shape.len();
        let mut new_shape = shape.to_vec();
        new_shape.swap(dim0, dim1);

        // 元テンソルのストライド計算
        let mut src_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            src_strides[i] = src_strides[i + 1] * shape[i + 1];
        }

        // 新テンソルのストライド計算
        let mut dst_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            dst_strides[i] = dst_strides[i + 1] * new_shape[i + 1];
        }

        let total_elems: usize = shape.iter().product();
        let src_data: Vec<f32> = self.to_vec();
        let mut dst_data = vec![0.0f32; total_elems];

        for src_idx in 0..total_elems {
            // src_idx → 多次元座標
            let mut remaining = src_idx;
            let mut coords = vec![0usize; ndim];
            for d in 0..ndim {
                coords[d] = remaining / src_strides[d];
                remaining %= src_strides[d];
            }
            // dim0 と dim1 の座標を入れ替え
            coords.swap(dim0, dim1);
            // 新しいインデックスを計算
            let dst_idx: usize = coords.iter().zip(dst_strides.iter()).map(|(c, s)| c * s).sum();
            dst_data[dst_idx] = src_data[src_idx];
        }

        MetalTensor::from_slice(&dst_data, &new_shape, MetalTensor::dtype(self))
    }

    
    /// broadcast_to — Metal GPU シェーダー実装
    pub fn broadcast_to_impl(&self, shape: &[usize]) -> MetalTensor {
        let src_shape = MetalTensor::shape(self);
        
        // 形状が同じなら何もしない
        if src_shape == shape {
            return self.clone_data();
        }

        let src_ndim = src_shape.len();
        let dst_ndim = shape.len();
        let ndim_diff = dst_ndim - src_ndim;

        // パディングされたソース形状
        let mut padded_src_shape = vec![1u32; dst_ndim];
        for i in 0..src_ndim {
            padded_src_shape[ndim_diff + i] = src_shape[i] as u32;
        }

        // ストライド計算
        let mut src_strides = vec![0u32; dst_ndim];
        let mut stride = 1u32;
        for i in (0..dst_ndim).rev() {
            if padded_src_shape[i] == shape[i] as u32 {
                src_strides[i] = stride;
            }
            stride *= padded_src_shape[i];
        }

        let dst_size: usize = shape.iter().product();
        let result = MetalTensor::uninit(shape, MetalTensor::dtype(self));
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_broadcast_pipeline();

        let ndim_buf = make_u32_buf(dst_ndim as u32);
        let dst_size_buf = make_u32_buf(dst_size as u32);
        let dst_shape_buf = make_u32_array_buf(&shape.iter().map(|&s| s as u32).collect::<Vec<_>>());
        let strides_buf = make_u32_array_buf(&src_strides);

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(result.buffer()), 0);
        encoder.set_buffer(2, Some(&ndim_buf), 0);
        encoder.set_buffer(3, Some(&dst_size_buf), 0);
        encoder.set_buffer(4, Some(&dst_shape_buf), 0);
        encoder.set_buffer(5, Some(&strides_buf), 0);

        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
        let thread_count = dst_size.min(max_threads);
        let tpg = MTLSize::new(thread_count as u64, 1, 1);
        let grid = MTLSize::new(
            ((dst_size + thread_count - 1) / thread_count) as u64,
            1, 1,
        );
        encoder.dispatch_thread_groups(grid, tpg);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        result
    }
    
    /// narrow
    pub fn narrow_impl(&self, axis: usize, start: usize, len: usize) -> MetalTensor {
        self.slice_impl(axis, start, len)
    }
    
    /// slice — Metal GPU シェーダー実装
    pub fn slice_impl(&self, axis: usize, start: usize, len: usize) -> MetalTensor {
        let shape = MetalTensor::shape(self);
        assert!(axis < shape.len(), "axis out of range");
        assert!(start + len <= shape[axis], "slice out of range");
        
        let mut new_shape = shape.to_vec();
        new_shape[axis] = len;
        
        let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
        let src_axis_size = shape[axis];
        let inner_size: usize = if axis + 1 < shape.len() { shape[axis+1..].iter().product() } else { 1 };

        let result = MetalTensor::uninit(&new_shape, MetalTensor::dtype(self));
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_slice_pipeline();

        let outer_buf = make_u32_buf(outer_size as u32);
        let src_axis_buf = make_u32_buf(src_axis_size as u32);
        let inner_buf = make_u32_buf(inner_size as u32);
        let start_buf = make_u32_buf(start as u32);
        let len_buf = make_u32_buf(len as u32);

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(result.buffer()), 0);
        encoder.set_buffer(2, Some(&outer_buf), 0);
        encoder.set_buffer(3, Some(&src_axis_buf), 0);
        encoder.set_buffer(4, Some(&inner_buf), 0);
        encoder.set_buffer(5, Some(&start_buf), 0);
        encoder.set_buffer(6, Some(&len_buf), 0);

        let flat_size = len * inner_size;
        let tpg = MTLSize::new(flat_size.min(256) as u64, outer_size.min(4) as u64, 1);
        let grid = MTLSize::new(
            ((flat_size + tpg.width as usize - 1) / tpg.width as usize) as u64,
            ((outer_size + tpg.height as usize - 1) / tpg.height as usize) as u64,
            1,
        );
        encoder.dispatch_thread_groups(grid, tpg);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        result
    }
    
    /// contiguous
    pub fn contiguous_impl(&self) -> MetalTensor {
        self.clone_data()
    }
    
    /// cat — GPU blit コマンドでバッファ結合（axis=0）、それ以外はスライス+コピー
    pub fn cat_impl(tensors: &[&MetalTensor], axis: usize) -> MetalTensor {
        assert!(!tensors.is_empty(), "cat: empty tensor list");
        
        let first_shape = MetalTensor::shape(tensors[0]);
        let mut new_shape = first_shape.to_vec();
        let mut total_axis_size = 0usize;
        
        for t in tensors {
            let ts = MetalTensor::shape(*t);
            for (i, (a, b)) in first_shape.iter().zip(ts.iter()).enumerate() {
                if i != axis {
                    assert_eq!(a, b, "cat: shape mismatch at dim {}", i);
                }
            }
            total_axis_size += ts[axis];
        }
        new_shape[axis] = total_axis_size;
        
        let result = MetalTensor::uninit(&new_shape, MetalTensor::dtype(tensors[0]));
        let device = get_device();
        let command_queue = device.command_queue();

        if axis == 0 {
            // axis=0: GPU blit でバッファを連結コピー
            let command_buffer = command_queue.new_command_buffer();
            let blit_encoder = command_buffer.new_blit_command_encoder();
            
            let mut offset: u64 = 0;
            for t in tensors {
                let size = (t.elem_count() * 4) as u64; // f32 = 4 bytes
                blit_encoder.copy_from_buffer(
                    t.buffer(), 0,
                    result.buffer(), offset,
                    size,
                );
                offset += size;
            }
            
            blit_encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        } else {
            // 一般ケース: 各テンソルをスライスとして結果バッファにコピー
            let outer_size: usize = first_shape[..axis].iter().product::<usize>().max(1);
            let inner_size: usize = if axis + 1 < first_shape.len() {
                first_shape[axis + 1..].iter().product()
            } else { 1 };

            let command_buffer = command_queue.new_command_buffer();
            let blit_encoder = command_buffer.new_blit_command_encoder();
            
            for outer in 0..outer_size {
                let mut dst_axis_offset = 0usize;
                for t in tensors {
                    let t_axis = MetalTensor::shape(*t)[axis];
                    let chunk_size = (t_axis * inner_size * 4) as u64;
                    let src_off = (outer * t_axis * inner_size * 4) as u64;
                    let dst_off = (outer * total_axis_size * inner_size * 4
                        + dst_axis_offset * inner_size * 4) as u64;
                    
                    blit_encoder.copy_from_buffer(
                        t.buffer(), src_off,
                        result.buffer(), dst_off,
                        chunk_size,
                    );
                    dst_axis_offset += t_axis;
                }
            }
            
            blit_encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        result
    }
}
