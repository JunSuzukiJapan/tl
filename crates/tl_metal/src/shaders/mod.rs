//! Metal Shader パイプライン管理

use metal::{ComputePipelineState, Device, Library, MTLSize};
use std::collections::HashMap;

/// Shader 関数名
pub const SHADER_ADD_F32: &str = "add_f32";
pub const SHADER_MUL_F32: &str = "mul_f32";
pub const SHADER_EXP_F32: &str = "exp_f32";
pub const SHADER_SQRT_F32: &str = "sqrt_f32";

/// Metal Shader ソースコード
const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// 二項演算: 加算
kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] + b[id];
}

// 二項演算: 乗算
kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] * b[id];
}

// 単項演算: exp
kernel void exp_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = exp(a[id]);
}

// 単項演算: sqrt
kernel void sqrt_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = sqrt(a[id]);
}
"#;

/// Shader パイプラインを管理
pub struct ShaderPipelines {
    library: Library,
    pipelines: HashMap<String, ComputePipelineState>,
}

impl ShaderPipelines {
    /// デバイスから Shader をコンパイルして初期化
    pub fn new(device: &Device) -> Result<Self, String> {
        let options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .map_err(|e| format!("Failed to compile shaders: {}", e))?;

        Ok(ShaderPipelines {
            library,
            pipelines: HashMap::new(),
        })
    }

    /// 指定した関数のパイプラインを取得（キャッシュ済みなら再利用）
    pub fn get_pipeline(
        &mut self,
        device: &Device,
        function_name: &str,
    ) -> Result<&ComputePipelineState, String> {
        if !self.pipelines.contains_key(function_name) {
            let function = self
                .library
                .get_function(function_name, None)
                .map_err(|e| format!("Function {} not found: {}", function_name, e))?;

            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| format!("Failed to create pipeline for {}: {}", function_name, e))?;

            self.pipelines.insert(function_name.to_string(), pipeline);
        }

        Ok(self.pipelines.get(function_name).unwrap())
    }
}

/// スレッドグループサイズを計算
pub fn compute_thread_groups(element_count: usize, pipeline: &ComputePipelineState) -> (MTLSize, MTLSize) {
    let thread_execution_width = pipeline.thread_execution_width() as usize;
    let threads_per_group = MTLSize::new(thread_execution_width as u64, 1, 1);
    let num_groups = (element_count + thread_execution_width - 1) / thread_execution_width;
    let grid_size = MTLSize::new(num_groups as u64, 1, 1);
    (grid_size, threads_per_group)
}

/// グローバル Shader パイプライン
static SHADER_PIPELINES: std::sync::OnceLock<std::sync::Mutex<ShaderPipelines>> = std::sync::OnceLock::new();

/// グローバル Shader パイプラインを取得
pub fn get_shaders() -> &'static std::sync::Mutex<ShaderPipelines> {
    SHADER_PIPELINES.get_or_init(|| {
        let device = crate::device::get_device();
        std::sync::Mutex::new(ShaderPipelines::new(device.device()).expect("Failed to init shaders"))
    })
}
