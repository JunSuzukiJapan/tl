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
    // Add more as needed
    Unknown,
}

/// Quantized Tensor Structure
/// Holds raw data and metadata.
#[derive(Debug)]
pub struct QTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub ggml_type: GGMLType,
}

impl QTensor {
    pub fn new(data: Vec<u8>, shape: Vec<usize>, ggml_type: GGMLType) -> Self {
        Self { data, shape, ggml_type }
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
            _ => Err(format!("Unsupported quantization type: {:?}", self.ggml_type)),
        }
    }

    /// Dequantize to OpaqueTensor (CPU/GPU auto-detect)
    pub fn dequantize_to_tensor(&self) -> Result<*mut crate::OpaqueTensor, String> {
        let f32_data = self.dequantize_to_f32()?;
        let shape = self.shape.clone();
        let is_cpu = std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu");
        if is_cpu {
            Ok(tl_cpu::ffi::tl_cpu_tensor_new(
                f32_data.as_ptr(), shape.len(), shape.as_ptr()
            ) as *mut crate::OpaqueTensor)
        } else {
            Ok(tl_metal::ffi_ops::tl_metal_new(
                f32_data.as_ptr(), shape.len(), shape.as_ptr()
            ))
        }
    }

    fn dequantize_f16(&self) -> Result<Vec<f32>, String> {
        if self.data.len() % 2 != 0 {
            return Err("Data length not multiple of 2 for F16".to_string());
        }
        let count = self.data.len() / 2;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let bytes = [self.data[i * 2], self.data[i * 2 + 1]];
            out.push(f16::from_le_bytes(bytes).to_f32());
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

        let mut out = Vec::with_capacity(num_elements);
        let mut ptr = 0;

        for _ in 0..num_blocks {
            // Read scale (f16)
            let d_bytes = [self.data[ptr], self.data[ptr + 1]];
            let d = f16::from_le_bytes(d_bytes).to_f32();
            ptr += 2;

            // Decode 16 bytes -> 32 weights
            for j in 0..16 {
                let byte = self.data[ptr + j];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                out.push(d * lo as f32);
                out.push(d * hi as f32);
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
}

// Helper for F16 to F32 conversion
use half::f16;

