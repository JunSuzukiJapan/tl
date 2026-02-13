//! Quantized Tensor Module
//! 
//! Handles quantized tensors loaded from GGUF files.
//! Supports dequantization to F32 for compatibility with CPU/Metal backends.


/// Quantization Types (subset of GGML types)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q4_K,
    Q6_K,
    Unknown,
}

use std::sync::Mutex;

/// Quantized Tensor Structure
/// Holds raw data and metadata.
#[derive(Debug)]
pub struct QTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub ggml_type: GGMLType,
    /// Cache for dequantized tensor (on device)
    /// Stored as usize to allow Send/Sync (raw pointers are not Send/Sync)
    /// The cached tensor is owned by QTensor and will be freed when QTensor is dropped.
    pub cache: Mutex<Option<usize>>,
    /// Cache for transposed dequantized tensor (on device)
    /// Used by tl_qtensor_matmul to avoid repeated transpose operations.
    pub cache_transposed: Mutex<Option<usize>>,
}

impl Drop for QTensor {
    fn drop(&mut self) {
        let is_cpu = std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu");
        // Free both caches
        for cache in [&self.cache, &self.cache_transposed] {
            if let Ok(mut cache_guard) = cache.lock() {
                if let Some(ptr_val) = cache_guard.take() {
                    if ptr_val != 0 {
                        unsafe {
                            if is_cpu {
                                let _ = Box::from_raw(ptr_val as *mut tl_cpu::CpuTensor);
                            } else {
                                tl_metal::ffi::tl_metal_release(ptr_val as *mut tl_metal::MetalTensor);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl QTensor {
    pub fn new(data: Vec<u8>, shape: Vec<usize>, ggml_type: GGMLType) -> Self {
        Self { 
            data, 
            shape, 
            ggml_type,
            cache: Mutex::new(None),
            cache_transposed: Mutex::new(None),
        }
    }

    /// Dequantize to F32 data vector
    pub fn dequantize_to_f32(&self) -> Result<Vec<f32>, String> {
        match self.ggml_type {
            GGMLType::F32 => {
                // reinterpret bytes as f32
                if self.data.len() % 4 != 0 {
                    return Err("Data length not multiple of 4 for F32".to_string());
                }
                let count = self.data.len() / 4;
                let mut out = Vec::with_capacity(count);
                unsafe {
                    let ptr = self.data.as_ptr() as *const f32;
                    let slice = std::slice::from_raw_parts(ptr, count);
                    out.extend_from_slice(slice);
                }
                Ok(out)
            },
            GGMLType::F16 => self.dequantize_f16(),
            GGMLType::Q4_0 => self.dequantize_q4_0(),
            GGMLType::Q8_0 => self.dequantize_q8_0(),
            GGMLType::Q4_K => self.dequantize_q4_k(),
            GGMLType::Q6_K => self.dequantize_q6_k(),
            _ => Err(format!("Unsupported quantization type: {:?}", self.ggml_type)),
        }
    }

    /// Dequantize to OpaqueTensor (CPU/GPU auto-detect)
    /// ON-DEMAND DEQUANTIZATION Strategy for Memory Efficiency
    /// 1. Uploads raw quantized data (u8) to GPU and caches it.
    /// 2. Executes dequantize kernel (for Q4_K) to create temporary F32 tensor.
    /// 3. Returns the temporary F32 tensor (caller must free it).
    pub fn dequantize_to_tensor(&self) -> Result<*mut crate::OpaqueTensor, String> {
        let is_cpu = std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu");

        if is_cpu {
            // CPU fallback: perform dequantization and cache the F32 tensor
            if let Ok(mut cache_guard) = self.cache.lock() {
                 if let Some(ptr_val) = *cache_guard {
                     return Ok(ptr_val as *mut crate::OpaqueTensor);
                 }
                 
                 let f32_data = self.dequantize_to_f32()?;
                 let shape = self.shape.clone();
                 let tensor_ptr = 
                    tl_cpu::ffi::tl_cpu_tensor_new(
                        f32_data.as_ptr(), shape.len(), shape.as_ptr()
                    ) as *mut crate::OpaqueTensor;
                 *cache_guard = Some(tensor_ptr as usize);
                 return Ok(tensor_ptr);
            }
            return Err("CPU cache lock failed".to_string());
        }

        // Metal (GPU) Path: CPU dequantize + GPU 転送 + F32 キャッシュ
        // 
        // 戦略: CPU 側で正確に dequantize し、F32 データを GPU に転送。
        // 結果を cache に保持し、2回目以降は即座に返す。
        // これにより:
        //   - 速度: 毎回の dequantize を回避
        //   - 精度: CPU dequantize は確実に正しい
        //   - メモリ: F32 テンソルは QTensor の Drop で解放
        let mut cache_guard = self.cache.lock().unwrap();
        
        // キャッシュヒット: 既に dequantized F32 テンソルがある
        if let Some(ptr_val) = *cache_guard {
            return Ok(ptr_val as *mut crate::OpaqueTensor);
        }
        
        // キャッシュミス: CPU dequantize → GPU 転送 → キャッシュ
        let f32_data = self.dequantize_to_f32()?;
        let out_tensor = tl_metal::MetalTensor::from_slice(&f32_data, &self.shape, tl_metal::DType::F32);
        let ptr = crate::make_metal_tensor(out_tensor);
        *cache_guard = Some(ptr as usize);
        Ok(ptr as *mut _)
    }

    fn dequantize_f16(&self) -> Result<Vec<f32>, String> {
        if self.data.len() % 2 != 0 {
            return Err("Data length not multiple of 2 for F16".to_string());
        }
        let count = self.data.len() / 2;
        let mut out = Vec::with_capacity(count);
        let mut ptr = 0;
        while ptr < self.data.len() {
            let bytes = [self.data[ptr], self.data[ptr+1]];
            out.push(f16::from_le_bytes(bytes).to_f32());
            ptr += 2;
        }
        Ok(out)
    }

    fn dequantize_q4_0(&self) -> Result<Vec<f32>, String> {
        // Q4_0 layout: [d: f16 (2 bytes)] [qs: u8 * 16] -> block of 32 weights in 18 bytes
        // Each byte in qs holds 2 x 4-bit values (low nibble first)
        // weight = (nibble - 8) * d
        let block_size = 32;
        let type_size = 2 + block_size / 2; // 2 (f16 scale) + 16 (packed nibbles) = 18 bytes
        let num_elements: usize = self.shape.iter().product();
        
        if num_elements % block_size != 0 {
            return Err("Number of elements must be multiple of 32 for Q4_0".to_string());
        }
        let num_blocks = num_elements / block_size;
        
        if self.data.len() != num_blocks * type_size {
            return Err(format!(
                "Data size mismatch for Q4_0: expected {}, got {} (num_elements={}, num_blocks={})", 
                num_blocks * type_size, self.data.len(), num_elements, num_blocks
            ));
        }

        let mut out = vec![0.0f32; num_elements];
        let mut ptr = 0;

        for block_idx in 0..num_blocks {
            // Read scale (f16)
            let d_bytes = [self.data[ptr], self.data[ptr + 1]];
            let d = f16::from_le_bytes(d_bytes).to_f32();
            ptr += 2;

            // Decode 16 bytes -> 32 weights
            // llama.cpp layout: lo nibble -> positions 0..15, hi nibble -> positions 16..31
            let y_base = block_idx * block_size;
            for j in 0..16 {
                let byte = self.data[ptr + j];
                let x0 = (byte & 0x0F) as i32 - 8;
                let x1 = ((byte >> 4) & 0x0F) as i32 - 8;
                out[y_base + j]      = d * x0 as f32;
                out[y_base + j + 16] = d * x1 as f32;
            }
            ptr += 16;
        }

        Ok(out)
    }

    fn dequantize_q8_0(&self) -> Result<Vec<f32>, String> {
        // Q8_0 layout: [d: f16] [x: i8 * 32] -> block size 34 bytes (for 32 weights)
        // x = d * x_i8
        let block_size = 32;
        let type_size = 2 + block_size; // f16 (2 bytes) + 32 * i8 (1 byte)
        let num_elements: usize = self.shape.iter().product();
        
        if num_elements % block_size != 0 {
            return Err("Number of elements must be multiple of 32 for Q8_0".to_string());
        }
        let num_blocks = num_elements / block_size;
        
        if self.data.len() != num_blocks * type_size {
            return Err(format!("Data size mismatch for Q8_0: expected {}, got {}", num_blocks * type_size, self.data.len()));
        }

        let mut out = Vec::with_capacity(num_elements);
        let mut ptr = 0;

        for _ in 0..num_blocks {
            // Read scale (f16)
            let d_bytes = [self.data[ptr], self.data[ptr+1]];
            let d = f16::from_le_bytes(d_bytes).to_f32();
            ptr += 2;

            for _ in 0..block_size {
                let x_i8 = self.data[ptr] as i8;
                out.push(d * x_i8 as f32);
                ptr += 1;
            }
        }

        Ok(out)
    }

    /// Dequantize Q6_K: 6-bit K-quant (llama.cpp reference implementation)
    /// Block layout (QK_K=256 weights per block, 210 bytes per block):
    ///   ql[128]    - lower 4 bits (interleaved)
    ///   qh[64]     - upper 2 bits (interleaved)
    ///   scales[16] - 8-bit signed scales for 16 groups of 16
    ///   d (f16)    - super-block scale
    fn dequantize_q6_k(&self) -> Result<Vec<f32>, String> {
        const QK_K: usize = 256;
        const BLOCK_SIZE: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // 128+64+16+2 = 210
        let num_elements: usize = self.shape.iter().product();

        if num_elements % QK_K != 0 {
            return Err(format!("Q6_K: num_elements {} not multiple of {}", num_elements, QK_K));
        }
        let num_blocks = num_elements / QK_K;

        if self.data.len() != num_blocks * BLOCK_SIZE {
            return Err(format!(
                "Q6_K data size mismatch: expected {}, got {}",
                num_blocks * BLOCK_SIZE, self.data.len()
            ));
        }

        let mut out = vec![0.0f32; num_elements];

        for block_idx in 0..num_blocks {
            let base = block_idx * BLOCK_SIZE;
            let d_bytes = [self.data[base + 208], self.data[base + 209]];
            let d = f16::from_le_bytes(d_bytes).to_f32();

            let y_base = block_idx * QK_K;
            let mut ql_off = base;         // ql starts at base, 128 bytes total
            let mut qh_off = base + 128;   // qh starts at base+128, 64 bytes total
            let mut sc_off = base + 192;   // scales starts at base+192, 16 bytes total

            // Process 256 weights in 2 chunks of 128
            for n_chunk in 0..2 {
                let chunk_base = y_base + n_chunk * 128;
                for l in 0..32 {
                    let is = l / 16;  // sub-scale index within chunk (0 or 1)

                    let ql_lo = self.data[ql_off + l];       // ql[l+0]
                    let ql_hi = self.data[ql_off + l + 32];  // ql[l+32]
                    let qh_val = self.data[qh_off + l];      // qh[l]

                    let q1 = ((ql_lo & 0xF) | (((qh_val >> 0) & 3) << 4)) as i32 - 32;
                    let q2 = ((ql_hi & 0xF) | (((qh_val >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql_lo >> 4)   | (((qh_val >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql_hi >> 4)   | (((qh_val >> 6) & 3) << 4)) as i32 - 32;

                    let sc1 = self.data[sc_off + is + 0] as i8;
                    let sc2 = self.data[sc_off + is + 2] as i8;
                    let sc3 = self.data[sc_off + is + 4] as i8;
                    let sc4 = self.data[sc_off + is + 6] as i8;

                    out[chunk_base + l +  0] = d * sc1 as f32 * q1 as f32;
                    out[chunk_base + l + 32] = d * sc2 as f32 * q2 as f32;
                    out[chunk_base + l + 64] = d * sc3 as f32 * q3 as f32;
                    out[chunk_base + l + 96] = d * sc4 as f32 * q4 as f32;
                }
                ql_off += 64;
                qh_off += 32;
                sc_off += 8;
            }
        }

        Ok(out)
    }

    /// Dequantize Q4_K: 4-bit K-quant with scales and mins (llama.cpp reference implementation)
    /// Block layout (QK_K=256 weights per block, 144 bytes per block):
    ///   d (f16)      - super-block scale for quantized scales
    ///   dmin (f16)   - super-block scale for quantized mins
    ///   scales[12]   - scales and mins, quantized with 6 bits
    ///   qs[128]      - 4-bit quants (QK_K/2)
    fn dequantize_q4_k(&self) -> Result<Vec<f32>, String> {
        const QK_K: usize = 256;
        const BLOCK_SIZE: usize = 2 + 2 + 12 + QK_K / 2; // 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = 144
        let num_elements: usize = self.shape.iter().product();

        if num_elements % QK_K != 0 {
            return Err(format!("Q4_K: num_elements {} not multiple of {}", num_elements, QK_K));
        }
        let num_blocks = num_elements / QK_K;

        if self.data.len() != num_blocks * BLOCK_SIZE {
            return Err(format!(
                "Q4_K data size mismatch: expected {}, got {}",
                num_blocks * BLOCK_SIZE, self.data.len()
            ));
        }

        let mut out = vec![0.0f32; num_elements];

        for block_idx in 0..num_blocks {
            let base = block_idx * BLOCK_SIZE;

            // Read d and dmin (f16)
            let d = f16::from_le_bytes([self.data[base], self.data[base + 1]]).to_f32();
            let dmin = f16::from_le_bytes([self.data[base + 2], self.data[base + 3]]).to_f32();

            // scales start at base + 4, 12 bytes
            let scales_off = base + 4;
            // qs start at base + 16, 128 bytes
            let qs_off = base + 16;

            let y_base = block_idx * QK_K;
            let mut is: usize = 0;
            let mut q_ptr: usize = qs_off;

            // Process 256 weights in 4 groups of 64
            let mut j: usize = 0;
            while j < QK_K {
                // get_scale_min_k4(is + 0)
                let (sc1, m1) = Self::get_scale_min_k4(is, &self.data[scales_off..scales_off + 12]);
                let d1 = d * sc1 as f32;
                let min1 = dmin * m1 as f32;

                // get_scale_min_k4(is + 1)
                let (sc2, m2) = Self::get_scale_min_k4(is + 1, &self.data[scales_off..scales_off + 12]);
                let d2 = d * sc2 as f32;
                let min2 = dmin * m2 as f32;

                // Low nibble: 32 weights
                for l in 0..32 {
                    out[y_base + j + l] = d1 * (self.data[q_ptr + l] & 0xF) as f32 - min1;
                }
                // High nibble: 32 weights
                for l in 0..32 {
                    out[y_base + j + 32 + l] = d2 * (self.data[q_ptr + l] >> 4) as f32 - min2;
                }

                q_ptr += 32;
                is += 2;
                j += 64;
            }
        }

        Ok(out)
    }

    /// Helper: extract scale and min from Q4_K scales array
    /// Mirrors llama.cpp's get_scale_min_k4()
    fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
        if j < 4 {
            let sc = scales[j] & 63;
            let m = scales[j + 4] & 63;
            (sc, m)
        } else {
            let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
            let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            (sc, m)
        }
    }
}

// Helper for F16 to F32 conversion
use half::f16;
