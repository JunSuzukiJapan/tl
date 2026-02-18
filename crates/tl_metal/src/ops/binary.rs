//! 二項演算（GPU Shader 使用）

use crate::device::get_device;
use crate::shaders::{self, SHADER_ADD_F32, SHADER_SUB_F32, SHADER_MUL_F32, SHADER_DIV_F32, SHADER_POW_F32,
    SHADER_EQ_F32, SHADER_NE_F32, SHADER_LT_F32, SHADER_LE_F32, SHADER_GT_F32, SHADER_GE_F32,
    SHADER_FMOD_F32};
use crate::tensor::MetalTensor;
use crate::DType;
use tl_backend::{BackendResult, BackendError};

/// NumPy スタイルの broadcast shape 計算
fn broadcast_shape(a: &[usize], b: &[usize]) -> BackendResult<Vec<usize>> {
    let max_rank = a.len().max(b.len());
    let mut result = vec![1; max_rank];
    
    for i in 0..max_rank {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        
        let out_dim = if a_dim == b_dim {
            a_dim
        } else if a_dim == 1 {
            b_dim
        } else if b_dim == 1 {
            a_dim
        } else {
            return Err(BackendError::ShapeMismatch(format!(
                "Cannot broadcast shapes {:?} and {:?}", a, b
            )));
        };
        result[max_rank - 1 - i] = out_dim;
    }
    Ok(result)
}

impl MetalTensor {
    /// 要素ごとの加算（内部実装）
    pub fn add_impl(&self, other: &MetalTensor) -> BackendResult<MetalTensor> {
        self.binary_op(other, SHADER_ADD_F32)
    }

    /// 要素ごとの減算（内部実装）
    pub fn sub_impl(&self, other: &MetalTensor) -> BackendResult<MetalTensor> {
        self.binary_op(other, SHADER_SUB_F32)
    }

    /// 要素ごとの乗算（内部実装）
    pub fn mul_impl(&self, other: &MetalTensor) -> BackendResult<MetalTensor> {
        self.binary_op(other, SHADER_MUL_F32)
    }

    /// 要素ごとの除算（内部実装）
    pub fn div_impl(&self, other: &MetalTensor) -> BackendResult<MetalTensor> {
        self.binary_op(other, SHADER_DIV_F32)
    }

    /// 要素ごとのべき乗（内部実装）
    pub fn pow_impl(&self, other: &MetalTensor) -> BackendResult<MetalTensor> {
        self.binary_op(other, SHADER_POW_F32)
    }

    /// 二項演算の GPU 実行
    fn binary_op(&self, other: &MetalTensor, shader_name: &str) -> BackendResult<MetalTensor> {
        if self.dtype() != other.dtype() {
            return Err(BackendError::TypeMismatch(format!(
                "DType mismatch in binary op: {:?} vs {:?}", self.dtype(), other.dtype()
            )));
        }
        
        let self_shape = self.shape();
        let other_shape = other.shape();
        
        // Shape が同じ場合は直接演算
        if self_shape == other_shape {
            return match self.dtype() {
                DType::F32 => self.binary_op_gpu(other, shader_name),
                _ => Err(BackendError::DeviceError(format!(
                    "{} for {:?}", shader_name, self.dtype()
                ))),
            };
        }
        
        // Broadcast が必要な場合
        let output_shape = broadcast_shape(self_shape, other_shape)?;
        let a = if self_shape != output_shape {
            self.broadcast_to_impl(&output_shape)?
        } else {
            self.clone()
        };
        let b = if other_shape != output_shape {
            other.broadcast_to_impl(&output_shape)?
        } else {
            other.clone()
        };
        
        match a.dtype() {
            DType::F32 => a.binary_op_gpu(&b, shader_name),
            _ => Err(BackendError::DeviceError(format!(
                "{} for {:?}", shader_name, a.dtype()
            ))),
        }
    }

    /// 二項演算の GPU 実行（内部）
    fn binary_op_gpu(&self, other: &MetalTensor, shader_name: &str) -> BackendResult<MetalTensor> {
        let result = MetalTensor::uninit(self.shape(), self.dtype());
        let device = get_device();
        let command_queue = device.command_queue();

        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), shader_name)
            .map_err(|e| BackendError::InternalError(format!("Failed to get shader pipeline ({}): {}", shader_name, e)))?;

        objc::rc::autoreleasepool(|| {
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(self.buffer()), 0);
            encoder.set_buffer(1, Some(other.buffer()), 0);
            encoder.set_buffer(2, Some(result.buffer()), 0);

            let (grid_size, threads_per_group) = shaders::compute_thread_groups(self.elem_count(), pipeline);
            encoder.dispatch_thread_groups(grid_size, threads_per_group);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        });

        Ok(result)
    }

    // ========== 比較演算（GPU Shader）==========

    /// 要素ごとの等値比較
    pub fn eq_impl(&self, other: &MetalTensor) -> BackendResult<MetalTensor> {
        self.binary_op(other, SHADER_EQ_F32)
    }
    /// 要素ごとの非等値比較
    pub fn ne_impl(&self, other: &MetalTensor) -> BackendResult<MetalTensor> {
        self.binary_op(other, SHADER_NE_F32)
    }
    /// 要素ごとの小なり比較
    pub fn lt_impl(&self, other: &MetalTensor) -> BackendResult<MetalTensor> {
        self.binary_op(other, SHADER_LT_F32)
    }
    /// 要素ごとの小なりイコール比較
    pub fn le_impl(&self, other: &MetalTensor) -> BackendResult<MetalTensor> {
        self.binary_op(other, SHADER_LE_F32)
    }
    /// 要素ごとの大なり比較
    pub fn gt_impl(&self, other: &MetalTensor) -> BackendResult<MetalTensor> {
        self.binary_op(other, SHADER_GT_F32)
    }
    /// 要素ごとの大なりイコール比較
    pub fn ge_impl(&self, other: &MetalTensor) -> BackendResult<MetalTensor> {
        self.binary_op(other, SHADER_GE_F32)
    }
    /// 要素ごとの剰余演算
    pub fn rem_impl(&self, other: &MetalTensor) -> BackendResult<MetalTensor> {
        self.binary_op(other, SHADER_FMOD_F32)
    }
}
