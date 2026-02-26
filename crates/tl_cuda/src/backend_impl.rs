//! tl_backend::GpuTensor トレイト実装 + 未実装 _impl スタブ

use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{self, BackendResult, DType as BackendDType, GpuTensor};

fn to_backend_dtype(dtype: DType) -> BackendDType {
    match dtype {
        DType::F32 => BackendDType::F32,
        DType::I64 => BackendDType::I64,
        DType::I32 => BackendDType::I32,
        DType::F16 => BackendDType::F16,
        DType::U8 => BackendDType::U8,
    }
}

fn from_backend_dtype(dtype: BackendDType) -> DType {
    match dtype {
        BackendDType::F32 => DType::F32,
        BackendDType::I64 => DType::I64,
        BackendDType::I32 => DType::I32,
        BackendDType::F16 => DType::F16,
        BackendDType::U8 => DType::U8,
        _ => DType::F32,
    }
}

impl GpuTensor for CudaTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn dtype(&self) -> BackendDType {
        to_backend_dtype(self.dtype)
    }
    fn to_vec_f32(&self) -> Vec<f32> {
        self.to_vec::<f32>()
    }
    fn to_vec_i64(&self) -> Vec<i64> {
        self.to_vec::<i64>()
    }

    fn from_slice_f32(data: &[f32], shape: &[usize]) -> BackendResult<Self> {
        Ok(CudaTensor::from_slice(data, shape, DType::F32))
    }
    fn from_slice_i64(data: &[i64], shape: &[usize]) -> BackendResult<Self> {
        Ok(CudaTensor::from_slice(data, shape, DType::I64))
    }
    fn zeros(shape: &[usize], dtype: BackendDType) -> BackendResult<Self> {
        Ok(CudaTensor::zeros(shape, from_backend_dtype(dtype)))
    }
    fn ones(shape: &[usize], dtype: BackendDType) -> BackendResult<Self> {
        Ok(CudaTensor::ones(shape, from_backend_dtype(dtype)))
    }
    fn randn(shape: &[usize], dtype: BackendDType) -> BackendResult<Self> {
        Ok(CudaTensor::randn(shape, from_backend_dtype(dtype)))
    }
    fn arange(start: i64, end: i64, dtype: BackendDType) -> BackendResult<Self> {
        let local_dtype = from_backend_dtype(dtype);
        let count = (end - start) as usize;
        match local_dtype {
            DType::F32 => {
                let data: Vec<f32> = (start..end).map(|i| i as f32).collect();
                Ok(CudaTensor::from_slice(&data, &[count], local_dtype))
            }
            DType::I64 => {
                let data: Vec<i64> = (start..end).collect();
                Ok(CudaTensor::from_slice(&data, &[count], local_dtype))
            }
            _ => {
                let data: Vec<f32> = (start..end).map(|i| i as f32).collect();
                Ok(CudaTensor::from_slice(&data, &[count], DType::F32))
            }
        }
    }
    fn clone_data(&self) -> BackendResult<Self> {
        CudaTensor::clone_data(self)
    }
}

// ========== 未実装スタブ（ops/ に移動する前の一時置き場）==========
// add_impl, sub_impl, mul_impl, div_impl, pow_impl, rem_impl → ops/binary.rs
// neg_impl, abs_impl → ops/unary.rs
// add_scalar_impl, mul_scalar_impl, div_scalar_impl, pow_scalar_impl, scale_impl, clamp_impl → ops/scalar.rs
// eq_impl, ne_impl, lt_impl, le_impl, gt_impl, ge_impl → ops/binary.rs

impl CudaTensor {
    // === 型変換 ===
    pub fn to_dtype(&self, target: DType) -> BackendResult<Self> {
        if self.dtype == target {
            return self.clone_data();
        }
        match (self.dtype, target) {
            (DType::F32, DType::I64) => {
                let data = self.to_vec::<f32>();
                let i64_data: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                Ok(CudaTensor::from_slice(&i64_data, &self.shape, DType::I64))
            }
            (DType::I64, DType::F32) => {
                let data = self.to_vec::<i64>();
                let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                Ok(CudaTensor::from_slice(&f32_data, &self.shape, DType::F32))
            }
            (DType::U8, DType::F32) => {
                let data = self.to_vec::<u8>();
                let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                Ok(CudaTensor::from_slice(&f32_data, &self.shape, DType::F32))
            }
            (DType::U8, DType::I64) => {
                let data = self.to_vec::<u8>();
                let i64_data: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                Ok(CudaTensor::from_slice(&i64_data, &self.shape, DType::I64))
            }
            (DType::F32, DType::U8) => {
                let data = self.to_vec::<f32>();
                let u8_data: Vec<u8> = data.iter().map(|&x| x.clamp(0.0, 255.0) as u8).collect();
                Ok(CudaTensor::from_slice(&u8_data, &self.shape, DType::U8))
            }
            _ => {
                // その他: F32 経由で変換
                let f32_data = self.to_vec::<f32>();
                Ok(CudaTensor::from_slice(&f32_data, &self.shape, target))
            }
        }
    }

    // === 量子化 ===
    /// Q4_K データをデクォンタイズして F32 Tensor を返す
    /// input: [num_blocks * 144] bytes (as u8 tensor)
    /// output: [num_blocks * 256] floats (F32 tensor)
    ///
    /// Q4_K ブロック構造 (144 bytes per block, 256 elements):
    ///   - [0..1]   f16 d (scale)
    ///   - [2..3]   f16 dmin (min scale)
    ///   - [4..15]  12 bytes = scales_and_mins (K_SCALE_SIZE=12)
    ///   - [16..143] 128 bytes = 256 nibbles (4-bit quantized values)
    pub fn dequantize_q4_k(&self, target_shape: &[usize]) -> BackendResult<Self> {
        use tl_backend::BackendError;

        let num_elements: usize = target_shape.iter().product();
        if num_elements % 256 != 0 {
            return Err(BackendError::ArgumentError(format!(
                "Q4_K num_elements must be divisible by 256, got {}",
                num_elements
            )));
        }
        let num_blocks = num_elements / 256;
        let raw = self.to_vec::<u8>();
        let expected_bytes = num_blocks * 144;
        if raw.len() < expected_bytes {
            return Err(BackendError::ArgumentError(format!(
                "Q4_K: expected {} bytes, got {}",
                expected_bytes,
                raw.len()
            )));
        }

        let mut output = vec![0.0f32; num_elements];

        for block_idx in 0..num_blocks {
            let base = block_idx * 144;

            // d, dmin (f16 → f32)
            let d = f16_to_f32(raw[base], raw[base + 1]);
            let dmin = f16_to_f32(raw[base + 2], raw[base + 3]);

            // scales_and_mins: 12 bytes
            let sc = &raw[base + 4..base + 16];

            // quantized nibbles: 128 bytes → 256 4-bit values
            let qs = &raw[base + 16..base + 144];

            // 8 sub-blocks of 32 elements each
            let out_base = block_idx * 256;
            for j in 0..8 {
                // Extract scale and min for this sub-block (6-bit each)
                let (scale, min) = decode_q4k_scale_min(sc, j);
                let s = d * scale as f32;
                let m = dmin * min as f32;

                let sub_base = out_base + j * 32;
                let q_offset = j * 16; // 16 bytes = 32 nibbles
                for k in 0..16 {
                    let byte = qs[q_offset + k];
                    let lo = (byte & 0x0F) as f32;
                    let hi = ((byte >> 4) & 0x0F) as f32;
                    output[sub_base + k] = lo * s - m;
                    output[sub_base + k + 16] = hi * s - m;
                }
            }
        }

        Ok(CudaTensor::from_slice(&output, target_shape, DType::F32))
    }
}

/// f16 (2 bytes LE) → f32
fn f16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = (hi as u16) << 8 | lo as u16;
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            // subnormal
            let mut val = frac as f32 / 1024.0;
            val *= 2.0f32.powi(-14);
            if sign == 1 {
                -val
            } else {
                val
            }
        }
    } else if exp == 31 {
        if frac == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        let f32_exp = exp as i32 - 15 + 127;
        let f32_bits = (sign << 31) | ((f32_exp as u32) << 23) | (frac << 13);
        f32::from_bits(f32_bits)
    }
}

/// Q4_K: sub-block j の scale/min を 6-bit でデコード
fn decode_q4k_scale_min(sc: &[u8], j: usize) -> (u8, u8) {
    // scales は下位 4 bit + 上位 2 bit に分散
    if j < 4 {
        let scale = (sc[j] & 0x3F) as u8;
        let min = (sc[j + 4] & 0x3F) as u8;
        (scale, min)
    } else {
        let k = j - 4;
        let scale_lo = (sc[k] >> 6) as u8;
        let scale_hi = ((sc[j + 4] & 0x0F) as u8) << 2;
        let scale = scale_lo | scale_hi;
        let min_lo = (sc[k + 4] >> 6) as u8;
        let min_hi = ((sc[j + 4] >> 4) & 0x0F) << 2;
        let min = min_lo | (min_hi as u8);
        (scale, min)
    }
}
