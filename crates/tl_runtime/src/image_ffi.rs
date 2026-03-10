//! Image FFI — 画像の読み込み/保存/変換

use std::ffi::{c_void, CStr};
use crate::string_ffi::StringStruct;

/// RGB画像ロード: path → Tensor[3,H,W]
/// @ffi_sig (String*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_image_load_rgb(path: *mut StringStruct) -> *mut c_void {
    let path_str = if !path.is_null() {
        unsafe {
            if !(*path).ptr.is_null() {
                CStr::from_ptr((*path).ptr).to_string_lossy().to_string()
            } else { return std::ptr::null_mut(); }
        }
    } else { return std::ptr::null_mut(); };

    let img = match image::open(&path_str) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            eprintln!("Error: Image::load_rgb failed: {}", e);
            return std::ptr::null_mut();
        }
    };

    let (w, h) = (img.width() as usize, img.height() as usize);
    // CHW format: [3, H, W]
    let mut data = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel = img.get_pixel(x as u32, y as u32);
            data[0 * h * w + y * w + x] = pixel[0] as f32 / 255.0;
            data[1 * h * w + y * w + x] = pixel[1] as f32 / 255.0;
            data[2 * h * w + y * w + x] = pixel[2] as f32 / 255.0;
        }
    }
    crate::device_ffi::create_runtime_tensor_f32(&data, &[3, h, w])
}

/// 画像リサイズ: Tensor[3,H,W] → Tensor[3,new_h,new_w]
/// @ffi_sig (Tensor*, i64, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_image_resize(t: *mut c_void, new_w: i64, new_h: i64) -> *mut c_void {
    if t.is_null() { return std::ptr::null_mut(); }
    let data = crate::device_ffi::read_runtime_tensor_to_f32_vec(t);
    // Assume CHW format [3, H, W]
    let c = 3usize;
    let total = data.len();
    let hw = total / c;
    let old_h = (hw as f64).sqrt() as usize; // Approximation; get from shape
    let old_w = hw / old_h.max(1);

    let nw = new_w as usize;
    let nh = new_h as usize;
    let mut result = vec![0.0f32; c * nh * nw];

    // Bilinear interpolation
    for ch in 0..c {
        for y in 0..nh {
            for x in 0..nw {
                let src_y = (y as f32) * (old_h as f32) / (nh as f32);
                let src_x = (x as f32) * (old_w as f32) / (nw as f32);
                let y0 = src_y as usize;
                let x0 = src_x as usize;
                let y1 = (y0 + 1).min(old_h - 1);
                let x1 = (x0 + 1).min(old_w - 1);
                let fy = src_y - y0 as f32;
                let fx = src_x - x0 as f32;
                let v00 = data[ch * old_h * old_w + y0 * old_w + x0];
                let v01 = data[ch * old_h * old_w + y0 * old_w + x1];
                let v10 = data[ch * old_h * old_w + y1 * old_w + x0];
                let v11 = data[ch * old_h * old_w + y1 * old_w + x1];
                result[ch * nh * nw + y * nw + x] =
                    v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;
            }
        }
    }
    crate::device_ffi::create_runtime_tensor_f32(&result, &[c, nh, nw])
}

/// 画像保存: Tensor[3,H,W] → ファイル
/// @ffi_sig (Tensor*, String*) -> void
#[unsafe(no_mangle)]
pub extern "C" fn tl_image_save(t: *mut c_void, path: *mut StringStruct) {
    if t.is_null() || path.is_null() { return; }
    let path_str = unsafe {
        if (*path).ptr.is_null() { return; }
        CStr::from_ptr((*path).ptr).to_string_lossy().to_string()
    };
    let data = crate::device_ffi::read_runtime_tensor_to_f32_vec(t);
    let c = 3usize;
    let total = data.len();
    let hw = total / c;
    let h = (hw as f64).sqrt() as usize;
    let w = hw / h.max(1);

    let mut img_buf = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let val = (data[ch * h * w + y * w + x] * 255.0).clamp(0.0, 255.0) as u8;
                img_buf[(y * w + x) * 3 + ch] = val;
            }
        }
    }
    let img = image::RgbImage::from_raw(w as u32, h as u32, img_buf);
    if let Some(img) = img {
        if let Err(e) = img.save(&path_str) {
            eprintln!("Error: Image::save failed: {}", e);
        }
    }
}

/// 画像正規化: (tensor - mean) / std (チャネルwise)
/// @ffi_sig (Tensor*, Tensor*, Tensor*) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_image_normalize(
    t: *mut c_void,
    mean_tensor: *mut c_void,
    std_tensor: *mut c_void,
) -> *mut c_void {
    if t.is_null() { return std::ptr::null_mut(); }
    let mut data = crate::device_ffi::read_runtime_tensor_to_f32_vec(t);
    let mean = if !mean_tensor.is_null() {
        crate::device_ffi::read_runtime_tensor_to_f32_vec(mean_tensor)
    } else {
        vec![0.485, 0.456, 0.406] // ImageNet default
    };
    let std = if !std_tensor.is_null() {
        crate::device_ffi::read_runtime_tensor_to_f32_vec(std_tensor)
    } else {
        vec![0.229, 0.224, 0.225] // ImageNet default
    };
    let c = mean.len().min(std.len()).min(3);
    let total = data.len();
    let hw = total / c;
    for ch in 0..c {
        for i in 0..hw {
            data[ch * hw + i] = (data[ch * hw + i] - mean[ch]) / std[ch];
        }
    }
    // Preserve shape as [C, H, W]
    let h = (hw as f64).sqrt() as usize;
    let w = hw / h.max(1);
    crate::device_ffi::create_runtime_tensor_f32(&data, &[c, h, w])
}

/// 画像クロップ: Tensor[3,H,W] → Tensor[3,crop_h,crop_w]
/// @ffi_sig (Tensor*, i64, i64, i64, i64) -> Tensor*
#[unsafe(no_mangle)]
pub extern "C" fn tl_image_crop(
    t: *mut c_void,
    x: i64, y: i64, crop_w: i64, crop_h: i64,
) -> *mut c_void {
    if t.is_null() { return std::ptr::null_mut(); }
    let data = crate::device_ffi::read_runtime_tensor_to_f32_vec(t);
    let c = 3usize;
    let total = data.len();
    let hw = total / c;
    let img_h = (hw as f64).sqrt() as usize;
    let img_w = hw / img_h.max(1);

    let cx = x as usize;
    let cy = y as usize;
    let cw = crop_w as usize;
    let ch_crop = crop_h as usize;

    let mut result = vec![0.0f32; c * ch_crop * cw];
    for ch in 0..c {
        for dy in 0..ch_crop {
            for dx in 0..cw {
                let sy = cy + dy;
                let sx = cx + dx;
                if sy < img_h && sx < img_w {
                    result[ch * ch_crop * cw + dy * cw + dx] = data[ch * img_h * img_w + sy * img_w + sx];
                }
            }
        }
    }
    crate::device_ffi::create_runtime_tensor_f32(&result, &[c, ch_crop, cw])
}
