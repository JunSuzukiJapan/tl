// use crate::memory_manager; // Used implicitly? No, but keep to match other files if needed. Actually it was warned used.
// use crate::mod::{make_tensor, OpaqueTensor}; // Fixed in previous step to `use crate::{make_tensor, OpaqueTensor};`
use crate::{device::get_device, make_tensor, OpaqueTensor};
use candle_core::{DType, Device, Tensor};
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use tokenizers::Tokenizer;

// Opaque wrapper for Tokenizer
pub struct OpaqueTokenizer(pub Tokenizer);

#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_new(path: *const c_char) -> i64 {
    unsafe {
        if path.is_null() {
            return 0;
        }
        let path_str = match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return 0,
        };
        let expanded_path = crate::expand_tilde(path_str);
        match Tokenizer::from_file(&expanded_path) {
            Ok(tokenizer) => {
                let ptr = Box::into_raw(Box::new(OpaqueTokenizer(tokenizer)));
                ptr as i64
            }
            Err(e) => {
                eprintln!("Failed to load tokenizer from {}: {}", path_str, e);
                0
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_encode(tokenizer: i64, prompt: *const c_char) -> *mut OpaqueTensor {
    unsafe {
        let tokenizer_ptr = tokenizer as *mut OpaqueTokenizer;
        let prompt_str = CStr::from_ptr(prompt).to_str().unwrap();
        let t = &(*tokenizer_ptr).0;
        match t.encode(prompt_str, true) {
            Ok(encoding) => {
                let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
                let len = ids.len();
                let device = get_device();
                let t_cpu = Tensor::from_vec(ids, (len,), &Device::Cpu).unwrap(); // [Seq]
                let tensor = if device.is_metal() || device.is_cuda() {
                    t_cpu.to_device(&device).unwrap()
                } else {
                    t_cpu
                };
                make_tensor(tensor)
            }
            Err(e) => {
                eprintln!("Tokenizer error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_decode(tokenizer: i64, ids: *mut OpaqueTensor) -> *mut c_char {
    unsafe {
        let t = &(*(tokenizer as *mut OpaqueTokenizer)).0;

        if ids.is_null() {
            return CString::new("").unwrap().into_raw();
        }

        let tensor_wrapper = &(*ids);

        // Check tensor validity if possible or just access
        let tensor = tensor_wrapper.as_tensor().expect("Standard tensor required");

        // Flatten to 1D and get values
        // Expecting [1, 1] or [N] tensor of I64
        let tensor_i64 = match tensor.flatten_all() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Flatten failed: {}", e);
                return CString::new("").unwrap().into_raw();
            }
        };

        let vec_i64 = tensor_i64.to_vec1::<i64>().unwrap_or_else(|_| {
            // Fallback if float
            tensor_i64
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|&x| x as i64)
                .collect()
        });

        // Convert i64 -> u32
        let vec_u32: Vec<u32> = vec_i64.iter().map(|&x| x as u32).collect();

        match t.decode(&vec_u32, true) {
            Ok(s) => {
                // println!("Decoded: {:?}", s);
                let s_fixed = s.replace("\\n", "\n");
                let c_string = CString::new(s_fixed).unwrap();
                c_string.into_raw()
            }
            Err(e) => {
                eprintln!("Tokenizer decode error: {}", e);
                CString::new("").unwrap().into_raw()
            }
        }
    }
}

// --- GGUF Handling ---
use crate::OpaqueTensorMap;

#[unsafe(no_mangle)]
pub extern "C" fn tl_gguf_load(path: *const c_char) -> *mut OpaqueTensorMap {
    unsafe {
        if path.is_null() {
            return std::ptr::null_mut();
        }
        let path_str = match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        let expanded_path = crate::expand_tilde(path_str);

        let mut file = std::fs::File::open(&expanded_path).expect("failed to open file");
        
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .expect("failed to read gguf");

        let map_ptr = Box::into_raw(Box::new(OpaqueTensorMap(std::collections::HashMap::new())));
        {
            let map = &mut *map_ptr;
            let device = get_device();
            for (tensor_name, qtensor_info) in content.tensor_infos.iter() {
                let qtensor = match qtensor_info
                    .read(&mut file, content.tensor_data_offset, &device)
                    .map_err(|e| e.to_string())
                {
                    Ok(q) => q,
                    Err(e) => {
                        eprintln!("Failed to read qtensor: {}", e);
                        return std::ptr::null_mut();
                    }
                };

                let qtensor_arc = std::sync::Arc::new(qtensor);
                map.0.insert(
                    tensor_name.clone(),
                    crate::LoadedTensor::Quantized(qtensor_arc),
                );
            }
        }
        map_ptr
    }
}

// --- Missing Ops ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cat(
    tensors: *mut Vec<*mut OpaqueTensor>,
    dim: i64,
) -> *mut OpaqueTensor {
    unsafe {
        let vec = &*tensors;
        let mut candle_tensors = Vec::new();
        for &t_ptr in vec {
            candle_tensors.push((*t_ptr).as_tensor().expect("Standard tensor required").clone());
        }

        let result = Tensor::cat(&candle_tensors, dim as usize).unwrap();
        make_tensor(result)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_silu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = (*t).as_tensor().expect("Standard tensor required");
        let result = candle_nn::ops::silu(tensor).unwrap();
        make_tensor(result)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_rms_norm(
    x: *mut OpaqueTensor,
    w: *mut OpaqueTensor,
    eps: f32,
) -> *mut OpaqueTensor {
    unsafe {
        let x_t = (*x).as_tensor().expect("Standard tensor required");
        let w_t = (*w).as_tensor().expect("Standard tensor required");

        // RMSNorm: x * rsqrt(mean(x^2) + eps) * w
        let x_dtype = x_t.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let hidden_size = x_t.dim(x_t.rank() - 1).unwrap();
        let x_f32 = x_t.to_dtype(internal_dtype).unwrap();
        let sum_sq = x_f32.sqr().unwrap().sum_keepdim(x_t.rank() - 1).unwrap();
        let mean_sq = (sum_sq / (hidden_size as f64)).unwrap();
        let rsqrt = (mean_sq + (eps as f64))
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();

        let norm_x = x_f32
            .broadcast_mul(&rsqrt)
            .unwrap()
            .to_dtype(x_dtype)
            .unwrap();
        let result = norm_x.broadcast_mul(w_t).unwrap();

        make_tensor(result)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cat2(
    a: *mut OpaqueTensor,
    b: *mut OpaqueTensor,
    dim: i64,
) -> *mut OpaqueTensor {
    unsafe {
        let a_t = (*a).as_tensor().expect("Standard tensor required");
        let b_t = (*b).as_tensor().expect("Standard tensor required");
        let result = Tensor::cat(&[a_t, b_t], dim as usize).unwrap();
        make_tensor(result)
    }
}

/// Concatenate two 4D tensors along a given dimension
/// This is an alias for tl_tensor_cat2 for type-safety in TL code
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cat_4d(
    a: *mut OpaqueTensor,
    b: *mut OpaqueTensor,
    dim: i64,
) -> *mut OpaqueTensor {
    if a.is_null() {
        eprintln!("ERROR: tl_tensor_cat_4d - a is NULL!");
        return std::ptr::null_mut();
    }
    if b.is_null() {
        eprintln!("ERROR: tl_tensor_cat_4d - b is NULL!");
        return std::ptr::null_mut();
    }

    unsafe {
        let a_t = (*a).as_tensor().expect("Standard tensor required");
        let b_t = (*b).as_tensor().expect("Standard tensor required");
        let result = Tensor::cat(&[a_t, b_t], dim as usize).unwrap();
        make_tensor(result)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_apply_rope(
    x: *mut OpaqueTensor,
    cos: *mut OpaqueTensor,
    sin: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    unsafe {
        let x_t = (*x).as_tensor().expect("Standard tensor required"); // [B, S, H, D]
        let cos_t = (*cos).as_tensor().expect("Standard tensor required"); // [S, D/2]
        let sin_t = (*sin).as_tensor().expect("Standard tensor required"); // [S, D/2]

        // x is [B, S, H, D]. We split into two halves along D.
        let d_m2 = x_t.dim(x_t.rank() - 1).unwrap();
        // x1 = x[..., :D/2], x2 = x[..., D/2:]
        let x1 = x_t.narrow(x_t.rank() - 1, 0, d_m2 / 2).unwrap();
        let x2 = x_t.narrow(x_t.rank() - 1, d_m2 / 2, d_m2 / 2).unwrap();

        // rotate_half(x) = [-x2, x1]
        let x2_neg = x2.neg().unwrap();
        let rotated_cat = Tensor::cat(&[&x2_neg, &x1], x_t.rank() - 1).unwrap();

        // (x * cos) + (rotate_half(x) * sin)
        // x1/x2: [B, S, H, D/2]
        // cos/sin: [S, D/2] -> reshape to [1, S, 1, D/2] for broadcast
        let cos_4d = cos_t
            .reshape((1, cos_t.dim(0).unwrap(), 1, cos_t.dim(1).unwrap()))
            .unwrap();
        let sin_4d = sin_t
            .reshape((1, sin_t.dim(0).unwrap(), 1, sin_t.dim(1).unwrap()))
            .unwrap();

        // Need to broadcast cos/sin to match x shape - they apply to both halves
        // For RoPE: x1 * cos + (-x2) * sin and x2 * cos + x1 * sin
        // Simplified as: x * cos + rotated * sin (where rotated has same shape as x)
        // But our rotated_cat is [-x2, x1] so the math is slightly different
        // Standard RoPE: out = x * cos + rotate_half(x) * sin
        // rotate_half swaps and negates: [..., -x2, x1]

        // Since cos/sin are [1, S, 1, D/2] and x is [B, S, H, D]:
        // We need cos/sin repeated for D dimension
        // cat cos with cos to get [1, S, 1, D]
        let cos_full = Tensor::cat(&[&cos_4d, &cos_4d], 3).unwrap();
        let sin_full = Tensor::cat(&[&sin_4d, &sin_4d], 3).unwrap();

        let x_cos = x_t.broadcast_mul(&cos_full).unwrap();
        let rot_sin = rotated_cat.broadcast_mul(&sin_full).unwrap();

        let result = (x_cos + rot_sin).unwrap();
        make_tensor(result)
    }
}

// Helper to get raw pointer from String for FFI
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_as_ptr(s: *const c_char) -> *const c_char {
    s
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_rope_new_cos(dim: i64, len: i64, theta: f32) -> *mut OpaqueTensor {
    let dim = dim as usize;
    let len = len as usize;
    let theta = theta as f64;

    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| 1.0 / (theta.powf(i as f64 / dim as f64) as f32))
        .collect();
    
    let device = get_device();
    let inv_freq_t = Tensor::from_vec(inv_freq, (1, dim / 2), &device).unwrap();

    let t: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let t_tensor = Tensor::from_vec(t, (len, 1), &device).unwrap();

    let freqs = t_tensor.matmul(&inv_freq_t).unwrap();
    let cos = freqs.cos().unwrap();

    make_tensor(cos)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_rope_new_sin(dim: i64, len: i64, theta: f32) -> *mut OpaqueTensor {
    let dim = dim as usize;
    let len = len as usize;
    let theta = theta as f64;

    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| 1.0 / (theta.powf(i as f64 / dim as f64) as f32))
        .collect();
    let device = get_device();
    let inv_freq_t = Tensor::from_vec(inv_freq, (1, dim / 2), &device).unwrap();

    let t: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let t_tensor = Tensor::from_vec(t, (len, 1), &device).unwrap();

    let freqs = t_tensor.matmul(&inv_freq_t).unwrap();
    let sin = freqs.sin().unwrap();
    make_tensor(sin)
}

// --- KV Cache Support ---

use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    /// Global KVCache Manager - holds all active KVCache instances
    /// When the manager is cleared (e.g., at program exit), all caches are freed
    static ref KV_CACHE_MANAGER: Mutex<KVCacheManager> = Mutex::new(KVCacheManager::new());
}

pub struct KVCache {
    // Layer -> (K, V)
    // Using simple vector where index is layer_id
    pub cache: Vec<Option<(Tensor, Tensor)>>,
}

impl KVCache {
    /// Calculate total memory usage of this cache in bytes
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0usize;
        for layer in &self.cache {
            if let Some((k, v)) = layer {
                // Each element is f32 (4 bytes)
                total += k.elem_count() * 4;
                total += v.elem_count() * 4;
            }
        }
        total
    }
}

pub struct KVCacheManager {
    caches: HashMap<i64, Box<KVCache>>,
    next_id: i64,
    /// Total memory usage across all caches (in bytes)
    total_bytes: usize,
}

impl KVCacheManager {
    pub fn new() -> Self {
        KVCacheManager {
            caches: HashMap::new(),
            next_id: 1,
            total_bytes: 0,
        }
    }

    pub fn create(&mut self, layers: usize) -> i64 {
        let mut vec = Vec::with_capacity(layers);
        for _ in 0..layers {
            vec.push(None);
        }
        let cache = Box::new(KVCache { cache: vec });
        let id = self.next_id;
        self.next_id += 1;
        self.caches.insert(id, cache);
        id
    }

    pub fn free(&mut self, id: i64) {
        if let Some(cache) = self.caches.remove(&id) {
            let freed_bytes = cache.memory_bytes();
            self.total_bytes = self.total_bytes.saturating_sub(freed_bytes);
        }
    }

    pub fn get(&self, id: i64) -> Option<&KVCache> {
        self.caches.get(&id).map(|b| b.as_ref())
    }

    pub fn get_mut(&mut self, id: i64) -> Option<&mut KVCache> {
        self.caches.get_mut(&id).map(|b| b.as_mut())
    }

    pub fn clear_all(&mut self) {
        self.caches.clear();
        self.total_bytes = 0;
    }

    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.total_bytes
    }

    /// Recalculate total memory (useful for verification)
    pub fn recalculate_memory(&mut self) {
        self.total_bytes = self.caches.values().map(|c| c.memory_bytes()).sum();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_new(layers: i64) -> i64 {
    let count = layers as usize;
    let mut mgr = KV_CACHE_MANAGER.lock().unwrap();
    mgr.create(count)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_free(id: i64) {
    let mut mgr = KV_CACHE_MANAGER.lock().unwrap();
    mgr.free(id);
}

/// Clear all KV caches - call this at program exit or when resetting
#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_clear_all() {
    let mut mgr = KV_CACHE_MANAGER.lock().unwrap();
    mgr.clear_all();
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_get_k(cache_id: i64, layer_idx: i64) -> *mut OpaqueTensor {
    let mgr = KV_CACHE_MANAGER.lock().unwrap();
    let idx = layer_idx as usize;

    match mgr.get(cache_id) {
        Some(cache) if idx < cache.cache.len() => match &cache.cache[idx] {
            Some((k, _)) => {
                // println!("DEBUG: Get K Cache {} Layer {}", cache_id, layer_idx);
                make_tensor(k.clone())
            }
            None => std::ptr::null_mut(),
        },
        _ => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_get_v(cache_id: i64, layer_idx: i64) -> *mut OpaqueTensor {
    let mgr = KV_CACHE_MANAGER.lock().unwrap();
    let idx = layer_idx as usize;

    match mgr.get(cache_id) {
        Some(cache) if idx < cache.cache.len() => match &cache.cache[idx] {
            Some((_, v)) => {
                // println!("DEBUG: Get V Cache {} Layer {}", cache_id, layer_idx);
                make_tensor(v.clone())
            }
            None => std::ptr::null_mut(),
        },
        _ => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_update(
    cache_id: i64,
    layer_idx: i64,
    k: *mut OpaqueTensor,
    v: *mut OpaqueTensor,
) {
    if k.is_null() {
        eprintln!("ERROR: tl_kv_cache_update - k is NULL!");
        return;
    }
    if v.is_null() {
        eprintln!("ERROR: tl_kv_cache_update - v is NULL!");
        return;
    }

    unsafe {
        let k_tensor = (*k).as_tensor().expect("Standard tensor required");
        let v_tensor = (*v).as_tensor().expect("Standard tensor required");

        let mut mgr = KV_CACHE_MANAGER.lock().unwrap();
        let idx = layer_idx as usize;

        // This print statement is added based on the user's request to show GGUF keys.
        // Note: 'container' is not available in this scope. This is a placeholder
        // and would need to be adapted if GGUF metadata were accessible here.
        // For now, it's a comment to indicate where such a print might go if context allowed.
        // if let Some(meta) = container.metadata.as_ref() {
        //     for kv in meta.kv.iter() {
        //         println!("GGUF Key: {}", kv.key);
        //     }
        // }

        if let Some(cache) = mgr.get_mut(cache_id) {
            if idx < cache.cache.len() {
                // println!("DEBUG: Update Cache {} Layer {}", cache_id, layer_idx);
                let k_t = k_tensor.clone();
                let v_t = v_tensor.clone();
                cache.cache[idx] = Some((k_t, v_t));
            }
        }

        // Recalculate total memory after update
        mgr.recalculate_memory();
    }
}

/// Get total KV Cache memory usage in bytes (FFI)
#[unsafe(no_mangle)]
pub extern "C" fn tl_kv_cache_get_memory_usage() -> i64 {
    let mgr = KV_CACHE_MANAGER.lock().unwrap();
    mgr.memory_usage() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sample(
    logits: *mut OpaqueTensor,
    temp: f32,
    top_p: f32,
) -> *mut OpaqueTensor {
    unsafe {
        let t = (*logits).as_tensor().expect("Standard tensor required"); // [B, Vocab] or [Vocab]
                              // Ensure we are working with the last token's logits if rank > 1
        let logits_flat = t.flatten_all().unwrap();

        // Apply temperature
        let temp_t = if temp <= 0.0 { 1.0 } else { temp };
        let logits_scaled = (&logits_flat / (temp_t as f64)).unwrap();

        // Softmax to get probabilities
        let probs = candle_nn::ops::softmax(&logits_scaled, 0).unwrap();

        // Convert to Vec for sampling
        let probs_vec: Vec<f32> = probs.to_dtype(DType::F32).unwrap().to_vec1().unwrap();

        // Sample
        let mut rng = thread_rng();

        // Top-P Sampling
        let mut probs_with_indices: Vec<(usize, f32)> =
            probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        // Sort descending
        probs_with_indices
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cum_sum = 0.0;
        let mut cutoff_index = probs_with_indices.len();

        if top_p < 1.0 && top_p > 0.0 {
            for (i, (_, p)) in probs_with_indices.iter().enumerate() {
                cum_sum += p;
                if cum_sum >= top_p {
                    cutoff_index = i + 1;
                    break;
                }
            }
        }

        // Filter and renormalize
        // We only keep the top `cutoff_index` elements.
        // Optimization: Construct a smaller weighted index distribution directly from the top elements.
        let (indices, weights): (Vec<usize>, Vec<f32>) =
            probs_with_indices.into_iter().take(cutoff_index).unzip();

        let dist = match WeightedIndex::new(&weights) {
            Ok(d) => d,
            Err(_) => {
                // If filtering failed (e.g. all zeros), fall back to argmax
                let max_idx = logits_flat
                    .argmax(0)
                    .unwrap()
                    .to_scalar::<u32>()
                    .unwrap_or(0);
                let device = get_device();
                let res = Tensor::from_vec(vec![max_idx as i64], (1,), &device).unwrap();
                return make_tensor(res);
            }
        };

        // Sample index in the *filtered* list
        let sampled_filtered_idx = dist.sample(&mut rng);
        // Map back to original vocabulary index
        let sample_idx = indices[sampled_filtered_idx];

        let device = get_device();
        let res = Tensor::from_vec(vec![sample_idx as i64], (1,), &device).unwrap();
        make_tensor(res)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to reset the global manager state between tests
    fn reset_manager() {
        let mut mgr = KV_CACHE_MANAGER.lock().unwrap();
        mgr.caches.clear();
        mgr.next_id = 1;
    }

    #[test]
    fn test_kv_cache_manager_create_and_free() {
        reset_manager();

        // Create a cache
        let id = tl_kv_cache_new(5);
        assert_eq!(id, 1);

        // Verify it exists
        {
            let mgr = KV_CACHE_MANAGER.lock().unwrap();
            assert!(mgr.get(id).is_some());
            assert_eq!(mgr.get(id).unwrap().cache.len(), 5);
        }

        // Free the cache
        tl_kv_cache_free(id);

        // Verify it's gone
        {
            let mgr = KV_CACHE_MANAGER.lock().unwrap();
            assert!(mgr.get(id).is_none());
        }
    }

    #[test]
    fn test_kv_cache_manager_multiple_caches() {
        reset_manager();

        let id1 = tl_kv_cache_new(3);
        let id2 = tl_kv_cache_new(5);
        let id3 = tl_kv_cache_new(7);

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);

        {
            let mgr = KV_CACHE_MANAGER.lock().unwrap();
            assert_eq!(mgr.caches.len(), 3);
        }

        // Free middle one
        tl_kv_cache_free(id2);

        {
            let mgr = KV_CACHE_MANAGER.lock().unwrap();
            assert_eq!(mgr.caches.len(), 2);
            assert!(mgr.get(id1).is_some());
            assert!(mgr.get(id2).is_none());
            assert!(mgr.get(id3).is_some());
        }

        // Free remaining
        tl_kv_cache_free(id1);
        tl_kv_cache_free(id3);

        {
            let mgr = KV_CACHE_MANAGER.lock().unwrap();
            assert_eq!(mgr.caches.len(), 0);
        }
    }

    #[test]
    fn test_kv_cache_manager_clear_all() {
        reset_manager();

        let id1 = tl_kv_cache_new(3);
        let _id2 = tl_kv_cache_new(5);
        let _id3 = tl_kv_cache_new(7);

        // Free one manually first
        tl_kv_cache_free(id1);

        {
            let mgr = KV_CACHE_MANAGER.lock().unwrap();
            assert_eq!(mgr.caches.len(), 2);
        }

        // Clear all - should not panic even though id1 was already freed
        tl_kv_cache_clear_all();

        {
            let mgr = KV_CACHE_MANAGER.lock().unwrap();
            assert_eq!(mgr.caches.len(), 0);
        }

        // Double clear should be safe
        tl_kv_cache_clear_all();

        {
            let mgr = KV_CACHE_MANAGER.lock().unwrap();
            assert_eq!(mgr.caches.len(), 0);
        }
    }

    #[test]
    fn test_kv_cache_manager_free_nonexistent() {
        reset_manager();

        // Free a cache that doesn't exist - should not panic
        tl_kv_cache_free(999);
        tl_kv_cache_free(0);
        tl_kv_cache_free(-1);

        // Manager should still be functional
        let id = tl_kv_cache_new(3);
        assert_eq!(id, 1);
    }

    #[test]
    fn test_kv_cache_manager_double_free_safety() {
        reset_manager();

        let id = tl_kv_cache_new(5);

        // First free
        tl_kv_cache_free(id);

        // Second free should not panic (no-op)
        tl_kv_cache_free(id);

        // Third free should also be safe
        tl_kv_cache_free(id);

        // Manager should still work
        let id2 = tl_kv_cache_new(3);
        assert_eq!(id2, 2); // ID increments even after free
    }

    #[test]
    fn test_kv_cache_manager_get_invalid() {
        reset_manager();

        let id = tl_kv_cache_new(3);

        // Get from non-existent cache
        let k_ptr = tl_kv_cache_get_k(999, 0);
        assert!(k_ptr.is_null());

        let v_ptr = tl_kv_cache_get_v(999, 0);
        assert!(v_ptr.is_null());

        // Get from valid cache but non-existent layer (before any update)
        let k_ptr = tl_kv_cache_get_k(id, 0);
        assert!(k_ptr.is_null()); // None because never updated

        // Get from valid cache but out-of-range layer
        let k_ptr = tl_kv_cache_get_k(id, 100);
        assert!(k_ptr.is_null());
    }

    #[test]
    fn test_kv_cache_manager_update_with_tensors() {
        reset_manager();

        let id = tl_kv_cache_new(3);

        // Create test tensors
        let k_tensor = Tensor::zeros((1, 4, 10, 64), DType::F32, &Device::Cpu).unwrap();
        let v_tensor = Tensor::zeros((1, 4, 10, 64), DType::F32, &Device::Cpu).unwrap();

        let k_opaque = Box::into_raw(Box::new(OpaqueTensor(crate::TensorVariant::Standard(k_tensor, None, None))));
        let v_opaque = Box::into_raw(Box::new(OpaqueTensor(crate::TensorVariant::Standard(v_tensor, None, None))));

        // Update
        tl_kv_cache_update(id, 0, k_opaque, v_opaque);

        // Verify stored
        {
            let mgr = KV_CACHE_MANAGER.lock().unwrap();
            let cache = mgr.get(id).unwrap();
            assert!(cache.cache[0].is_some());
            let (k, v) = cache.cache[0].as_ref().unwrap();
            assert_eq!(k.shape().dims(), &[1, 4, 10, 64]);
            assert_eq!(v.shape().dims(), &[1, 4, 10, 64]);
        }

        // Cleanup test tensors (these are not managed by TL memory manager)
        unsafe {
            let _ = Box::from_raw(k_opaque);
            let _ = Box::from_raw(v_opaque);
        }

        tl_kv_cache_free(id);
    }

    #[test]
    fn test_kv_cache_manager_update_null_tensors() {
        reset_manager();

        let id = tl_kv_cache_new(3);

        // Update with null pointers - should not panic
        tl_kv_cache_update(id, 0, std::ptr::null_mut(), std::ptr::null_mut());

        // Layer should still be None
        {
            let mgr = KV_CACHE_MANAGER.lock().unwrap();
            let cache = mgr.get(id).unwrap();
            assert!(cache.cache[0].is_none());
        }

        tl_kv_cache_free(id);
    }

    #[test]
    fn test_kv_cache_manager_free_then_clear_all_no_leak() {
        reset_manager();

        // Create multiple caches
        let id1 = tl_kv_cache_new(10);
        let id2 = tl_kv_cache_new(10);
        let id3 = tl_kv_cache_new(10);

        // Free some individually
        tl_kv_cache_free(id1);
        tl_kv_cache_free(id3);

        // Verify count
        {
            let mgr = KV_CACHE_MANAGER.lock().unwrap();
            assert_eq!(mgr.caches.len(), 1);
            assert!(mgr.get(id2).is_some());
        }

        // Clear all - should clear remaining id2 without issue
        tl_kv_cache_clear_all();

        {
            let mgr = KV_CACHE_MANAGER.lock().unwrap();
            assert_eq!(mgr.caches.len(), 0);
        }

        // System should still be functional
        let new_id = tl_kv_cache_new(5);
        assert_eq!(new_id, 4); // IDs continue incrementing
    }

    #[test]
    fn test_kv_cache_id_increment_after_clear() {
        reset_manager();

        let id1 = tl_kv_cache_new(3);
        let id2 = tl_kv_cache_new(3);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);

        tl_kv_cache_clear_all();

        // IDs should continue incrementing, not reset
        let id3 = tl_kv_cache_new(3);
        assert_eq!(id3, 3);
    }
}
