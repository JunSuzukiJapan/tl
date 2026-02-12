//! 拡張テンソル操作 (IO / Memory / Legacy Stubs)
//!
//! 演算操作は `tl_metal::ffi_ops` に移動済み。
//! ここには IO 関連や、tl_runtime レベルで処理すべき関数、
//! および互換性のためのスタブを残す。

use crate::OpaqueTensor;
use tl_metal::{MetalTensor, DType};

// ========== IO / System ==========

fn create_fallback_tensor(is_cpu: bool) -> *mut OpaqueTensor {
    if is_cpu {
        let t = tl_cpu::CpuTensor::from_slice(&[0.0f32], &[1], tl_cpu::DType::F32);
        Box::into_raw(Box::new(t)) as *mut OpaqueTensor
    } else {
        let t = tl_metal::MetalTensor::zeros(&[1], tl_metal::DType::F32);
        Box::into_raw(Box::new(t))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_save(path: *mut super::StringStruct, t: *mut OpaqueTensor) {
    if t.is_null() || path.is_null() {
        return;
    }
    let is_cpu = std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu");
    unsafe {
        if (*path).ptr.is_null() { return; }
        let path_str = std::ffi::CStr::from_ptr((*path).ptr).to_string_lossy();
        let path_buf = crate::file_io::expand_path(&path_str);

        let (data, shape) = if is_cpu {
            let tensor = &*(t as *mut tl_cpu::CpuTensor);
            (tensor.data_f32().to_vec(), tensor.shape().to_vec())
        } else {
            let tensor = &*t; // OpaqueTensor = MetalTensor
            (tensor.to_vec(), MetalTensor::shape(tensor).to_vec())
        };

        let mut bytes = Vec::new();
        // save format: rank(u64) + dims(u64...) + data(f32...)
        bytes.extend_from_slice(&(shape.len() as u64).to_le_bytes());
        for &dim in &shape {
            bytes.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        for &val in &data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        if let Err(e) = std::fs::write(&path_buf, &bytes) {
            eprintln!("Failed to save tensor: {}", e);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_load(path: *mut super::StringStruct) -> *mut OpaqueTensor {
    let is_cpu = std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu");
    unsafe {
        if path.is_null() || (*path).ptr.is_null() {
            return create_fallback_tensor(is_cpu);
        }
        let path_str = std::ffi::CStr::from_ptr((*path).ptr).to_string_lossy();
        let path_buf = crate::file_io::expand_path(&path_str);

        let bytes = match std::fs::read(&path_buf) {
            Ok(b) => b,
            Err(_) => return create_fallback_tensor(is_cpu),
        };

        if bytes.len() < 8 {
            return create_fallback_tensor(is_cpu);
        }

        let mut offset = 0;
        let rank = u64::from_le_bytes(bytes[offset..offset+8].try_into().unwrap()) as usize;
        offset += 8;

        if bytes.len() < offset + rank * 8 {
            return create_fallback_tensor(is_cpu);
        }

        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            let dim = u64::from_le_bytes(bytes[offset..offset+8].try_into().unwrap()) as usize;
            shape.push(dim);
            offset += 8;
        }

        let numel: usize = shape.iter().product();
        let expected_data_size = numel * 4;
        if bytes.len() < offset + expected_data_size {
            return create_fallback_tensor(is_cpu);
        }

        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            let val = f32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
            data.push(val);
            offset += 4;
        }

        if is_cpu {
            let t = tl_cpu::CpuTensor::from_slice(&data, &shape, tl_cpu::DType::F32);
            Box::into_raw(Box::new(t)) as *mut OpaqueTensor
        } else {
            let t = tl_metal::MetalTensor::from_slice(&data, &shape, tl_metal::DType::F32);
            Box::into_raw(Box::new(t))
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_get_memory_bytes() -> i64 {
    let mut usage = std::mem::MaybeUninit::uninit();
    unsafe {
        if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) == 0 {
            let usage = usage.assume_init();
            // macOS: ru_maxrss はバイト単位
            // Linux: ru_maxrss は KB 単位 → バイトに変換
            #[cfg(target_os = "macos")]
            { usage.ru_maxrss as i64 }
            #[cfg(not(target_os = "macos"))]
            { usage.ru_maxrss as i64 * 1024 }
        } else {
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_get_memory_mb() -> f64 {
    tl_get_memory_bytes() as f64 / 1024.0 / 1024.0
}

// ========== Image Stubs ==========
// これらのスタブはランタイム側に残す

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_load_grayscale(_path: *const i8) -> *mut OpaqueTensor {
    // Stub
    let t = tl_metal::MetalTensor::zeros(&[1, 1], DType::F32);
    Box::into_raw(Box::new(t))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_width(_t: *mut OpaqueTensor) -> i64 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_height(_t: *mut OpaqueTensor) -> i64 {
    0
}
