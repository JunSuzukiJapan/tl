use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::ffi::CString;
use tl_runtime::gguf::tl_gguf_load;
use tl_runtime::tensor_map::{tl_tensor_map_get_quantized, tl_tensor_map_free};

// Helper to write a simple GGUF file v3
// One tensor "tensor1": F32, shape [10], values 0.0..9.0
fn write_simple_gguf(path: &PathBuf) {
    let mut file = File::create(path).expect("Failed to create GGUF file");
    
    // 1. Magic "GGUF"
    file.write_all(b"GGUF").unwrap();
    
    // 2. Version 3 (u32 LE)
    file.write_all(&3u32.to_le_bytes()).unwrap();
    
    // 3. Tensor Count 1 (u64 LE)
    file.write_all(&1u64.to_le_bytes()).unwrap();
    
    // 4. KV Count 0 (u64 LE)
    file.write_all(&0u64.to_le_bytes()).unwrap();
    
    // 5. Tensor Info
    // Name "tensor1"
    let name = "tensor1";
    file.write_all(&(name.len() as u64).to_le_bytes()).unwrap(); // string length prefix
    file.write_all(name.as_bytes()).unwrap();
    
    // Dims 1 (u32)
    file.write_all(&1u32.to_le_bytes()).unwrap();
    
    // Shape [10] (u64 array)
    file.write_all(&10u64.to_le_bytes()).unwrap();
    
    // Type F32 = 0 (u32)
    file.write_all(&0u32.to_le_bytes()).unwrap();
    
    // Offset 0 (u64)
    file.write_all(&0u64.to_le_bytes()).unwrap();
    
    // Calculate current position to handle alignment
    // Magic(4) + Ver(4) + TCount(8) + KVCount(8) = 24
    // NameLen(8) + "tensor1"(7) = 15
    // Dims(4) + Shape(8) + Type(4) + Offset(8) = 24
    // Total header = 24 + 15 + 24 = 63 bytes.
    let current_pos = 63;
    let alignment = 32;
    let padding = (alignment - (current_pos % alignment)) % alignment;
    
    // 6. Padding
    for _ in 0..padding {
        file.write_all(&[0]).unwrap();
    }
    
    // 7. Data (10 floats = 40 bytes)
    let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    for f in data {
        file.write_all(&f.to_le_bytes()).unwrap();
    }
}

#[test]
fn test_gguf_load_simple() {
    let path = PathBuf::from("test_simple.gguf");
    write_simple_gguf(&path);
    
    let path_str = path.to_str().unwrap();
    let c_path = CString::new(path_str).unwrap();
    
    let s_struct = tl_runtime::string_ffi::tl_string_new(c_path.as_ptr());
    assert!(!s_struct.is_null());
    
    let map_ptr = tl_gguf_load(s_struct);
    assert!(!map_ptr.is_null(), "tl_gguf_load returned null");
    
    let tensor_name = CString::new("tensor1").unwrap();
    let name_struct = tl_runtime::string_ffi::tl_string_new(tensor_name.as_ptr());
    
    let qtensor_ptr = tl_tensor_map_get_quantized(map_ptr, name_struct);
    assert!(!qtensor_ptr.is_null(), "Failed to get quantized tensor 'tensor1'");
    
    // Verify qtensor data by dequantizing (since QTensor struct might not be exposed, 
    // we can use dequantize method if available or inspect via FFI if possible)
    // Here we just check pointer is not null, verifying map functionality.
    // To verify data correctness deeper, we would need to expose accessing QTensor data or 
    // load it as MetalTensor using some helper.
    
    // Cleanup
    tl_tensor_map_free(map_ptr);
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_gguf_load_real_file() {
    // Check various locations for models
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let models_dir = PathBuf::from(home).join(".llm/models");
    
    // List of candidate files to try
    let candidates = [
        "tinyllama-1.1b-chat-q8_0.gguf",
        "tinyllama-1.1b-chat-q4_k_m.gguf",
        "tinyllama/tinyllama-1.1b-chat-v1.0.Q8_0.gguf", 
    ];
    
    let mut found_path = None;
    for c in &candidates {
        let p = models_dir.join(c);
        if p.exists() {
            found_path = Some(p);
            break;
        }
    }
    
    if let Some(path) = found_path {
        println!("Testing with actual model file: {:?}", path);
        let path_str = path.to_str().unwrap();
        let c_path = CString::new(path_str).unwrap();
        
        let s_struct = tl_runtime::string_ffi::tl_string_new(c_path.as_ptr());
        assert!(!s_struct.is_null());
        
        let map_ptr = tl_gguf_load(s_struct);
        assert!(!map_ptr.is_null(), "tl_gguf_load returned null for real file");
        
        // Verify common tensors exist
        // TinyLlama / Llama usuall has token_embd.weight
        let tensor_names = ["token_embd.weight", "output.weight", "blk.0.attn_q.weight"];
        
        for name in &tensor_names {
             let c_name = CString::new(*name).unwrap();
             let name_struct = tl_runtime::string_ffi::tl_string_new(c_name.as_ptr());
             let qtensor_ptr = tl_tensor_map_get_quantized(map_ptr, name_struct);
             
             if !qtensor_ptr.is_null() {
                 println!("Found tensor: {}", name);
             } else {
                 println!("Tensor not found (might be expected for some models): {}", name);
             }
        }
        
        tl_tensor_map_free(map_ptr);
    } else {
        println!("No suitable GGUF model found in ~/.llm/models, skipping real file test.");
        // Do not fail the test if file is missing, as this depends on user env
    }
}
