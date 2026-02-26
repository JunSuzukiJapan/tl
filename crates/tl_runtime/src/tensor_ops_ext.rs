//! 拡張テンソル操作 (IO / Memory / Legacy Stubs)
//!
//! 演算操作は `tl_metal::ffi_ops` に移動済み。
//! ここには IO 関連や、tl_runtime レベルで処理すべき関数、
//! および互換性のためのスタブを残す。

use crate::OpaqueTensor;

// ========== IO / System ==========

fn create_fallback_tensor(_is_cpu: bool) -> *mut OpaqueTensor {
    crate::device_ffi::create_runtime_zeros(&[1]) as *mut OpaqueTensor
}

#[unsafe(no_mangle)]
/// @ffi_sig (StringStruct, Tensor*) -> void
pub extern "C" fn tl_tensor_save(path: *mut super::StringStruct, t: *mut OpaqueTensor) {
    if t.is_null() || path.is_null() {
        return;
    }
    let is_cpu = std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu");
    unsafe {
        if (*path).ptr.is_null() {
            return;
        }
        let path_str = std::ffi::CStr::from_ptr((*path).ptr).to_string_lossy();
        let path_buf = crate::file_io::expand_path(&path_str);

        let (data, shape) = if is_cpu {
            let tensor = &*(t as *mut tl_cpu::CpuTensor);
            (tensor.data_f32().to_vec(), tensor.shape().to_vec())
        } else {
            let tensor = &*t; // OpaqueTensor = MetalTensor
            (tensor.to_vec(), tensor.shape().to_vec())
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
/// @ffi_sig (StringStruct) -> Tensor*
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
        let rank = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        if bytes.len() < offset + rank * 8 {
            return create_fallback_tensor(is_cpu);
        }

        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            let dim = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
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
            let val = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            data.push(val);
            offset += 4;
        }

        crate::device_ffi::create_runtime_tensor_f32(&data, &shape) as *mut OpaqueTensor
    }
}

#[unsafe(no_mangle)]
/// @ffi_sig () -> i64
pub extern "C" fn tl_get_memory_bytes() -> i64 {
    // macOS: mach_task_basic_info で現在の RSS を取得
    // 注意: ru_maxrss はピーク RSS で単調増加するため使用不可
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        // time_value_t = { integer_t seconds; integer_t microseconds; } = 2 x i32 = 8 bytes
        // struct mach_task_basic_info 全体 = 48 bytes (MACH_TASK_BASIC_INFO_COUNT = 12)
        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,      // mach_vm_size_t (8 bytes)
            resident_size: u64,     // mach_vm_size_t (8 bytes)
            resident_size_max: u64, // mach_vm_size_t (8 bytes)
            user_time: [i32; 2],    // time_value_t (8 bytes)
            system_time: [i32; 2],  // time_value_t (8 bytes)
            policy: i32,            // policy_t (4 bytes)
            suspend_count: i32,     // integer_t (4 bytes)
        }
        const MACH_TASK_BASIC_INFO: u32 = 20;
        unsafe extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut MachTaskBasicInfo,
                task_info_out_cnt: *mut u32,
            ) -> i32;
        }
        unsafe {
            let mut info: MachTaskBasicInfo = mem::zeroed();
            let mut count = (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<u32>()) as u32;
            let kr = task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                &mut info as *mut _,
                &mut count,
            );
            if kr == 0 {
                return info.resident_size as i64;
            }
        }
        0
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Linux: /proc/self/statm から RSS を取得
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            let parts: Vec<&str> = statm.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(rss_pages) = parts[1].parse::<i64>() {
                    return rss_pages * 4096; // ページサイズ 4KB
                }
            }
        }
        0
    }
}

#[unsafe(no_mangle)]
/// @ffi_sig () -> f64
pub extern "C" fn tl_get_memory_mb() -> f64 {
    tl_get_memory_bytes() as f64 / 1024.0 / 1024.0
}

// ========== Image Stubs ==========
// これらのスタブはランタイム側に残す

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_load_grayscale(_path: *const i8) -> *mut OpaqueTensor {
    // Stub
    crate::device_ffi::create_runtime_zeros(&[1, 1]) as *mut OpaqueTensor
}

#[unsafe(no_mangle)]
/// @ffi_sig (Tensor*) -> i64
pub extern "C" fn tl_image_width(_t: *mut OpaqueTensor) -> i64 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_height(_t: *mut OpaqueTensor) -> i64 {
    0
}
