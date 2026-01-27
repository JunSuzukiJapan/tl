use std::ffi::{CStr, CString, c_void};
use std::fs::File;
use std::io::{Read, Write};
use std::os::raw::c_char;
// use std::path::Path;
// --- Strings ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_string_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        crate::tl_log_free(s as *const c_void, std::ptr::null(), 0);
        let _ = CString::from_raw(s);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_prompt(prompt: *const c_char) -> *mut c_char {
    // Print prompt
    if !prompt.is_null() {
        let p = unsafe { CStr::from_ptr(prompt).to_string_lossy() };
        print!("{}", p);
        let _ = std::io::stdout().flush();
    }

    // Read line
    let mut buffer = String::new();
    match std::io::stdin().read_line(&mut buffer) {
        Ok(_) => {
            // Remove trailing newline
            let trimmed = buffer.trim_end().to_string();
            match CString::new(trimmed) {
                Ok(c_str) => {
                    let ptr = c_str.into_raw();
                    crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);
                    ptr
                },
                Err(_) => std::ptr::null_mut(),
            }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_read_line(prompt: *const c_char) -> *mut c_char {
    tl_prompt(prompt)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_string_concat(s1: *const c_char, s2: *const c_char) -> *mut c_char {
    if s1.is_null() || s2.is_null() {
        return std::ptr::null_mut();
    }
    let s1_str = unsafe { CStr::from_ptr(s1).to_string_lossy() };
    let s2_str = unsafe { CStr::from_ptr(s2).to_string_lossy() };
    let joined = format!("{}{}", s1_str, s2_str);
    match CString::new(joined) {
        Ok(c_str) => {
            let ptr = c_str.into_raw();
            crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);
            ptr
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_string_to_i64(s: *const c_char) -> i64 {
    if s.is_null() {
        return 0;
    }
    let s_str = unsafe { CStr::from_ptr(s).to_string_lossy() };
    s_str.parse::<i64>().unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_string_from_int(val: i64) -> *mut c_char {
    let s = val.to_string();
    match CString::new(s) {
        Ok(c_str) => {
            let ptr = c_str.into_raw();
            crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);
            ptr
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_string_contains(haystack: *const c_char, needle: *const c_char) -> bool {
    if haystack.is_null() || needle.is_null() {
        return false;
    }
    let h_str = unsafe { CStr::from_ptr(haystack).to_string_lossy() };
    let n_str = unsafe { CStr::from_ptr(needle).to_string_lossy() };
    h_str.contains(n_str.as_ref())
}

/// Get character at index from string (returns ASCII code, or 0 if out of bounds)
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_char_at(s: *const c_char, index: i64) -> *mut c_char {
    eprintln!("DEBUG: tl_string_char_at s={:p} index={}", s, index);
    if s.is_null() || index < 0 {
        return std::ptr::null_mut();
    }
    let s_str = unsafe { CStr::from_ptr(s).to_string_lossy() };
    eprintln!("DEBUG: s_str len={}", s_str.len());
    let idx = index as usize;
    if idx >= s_str.len() {
        return std::ptr::null_mut();
    }
    // Note: This assumes byte indexing for now, or we can use chars().nth()
    // Using chars().nth() handles UTF-8 correctly but is O(N)
    if let Some(c) = s_str.chars().nth(idx) {
        let char_str = c.to_string();
        let ptr = std::ffi::CString::new(char_str).unwrap().into_raw();
        crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);
        ptr
    } else {
        std::ptr::null_mut()
    }
}

/// Get length of a string
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_len(s: *const c_char) -> i64 {
    if s.is_null() {
        return 0;
    }
    let s_str = unsafe { CStr::from_ptr(s).to_string_lossy() };
    s_str.len() as i64
}

// --- File I/O ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_file_open(path: *const c_char, mode: *const c_char) -> *mut File {
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let expanded_path = crate::expand_tilde(&path_str);
    let mode_str = unsafe { CStr::from_ptr(mode).to_string_lossy() };

    let f = match mode_str.as_ref() {
        "r" => File::open(&expanded_path),
        "w" => File::create(&expanded_path),
        _ => return std::ptr::null_mut(), // TODO: proper append support or error
    };

    match f {
        Ok(file) => {
            let ptr = Box::into_raw(Box::new(file));
            crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);
            ptr
        },
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_file_read_string(file: *mut File) -> *mut c_char {
    if file.is_null() {
        return std::ptr::null_mut();
    }
    let file = unsafe { &mut *file };
    let mut content = String::new();
    // Seek to start? usually implicit for linear read
    if file.read_to_string(&mut content).is_ok() {
        match CString::new(content) {
            Ok(c_str) => {
                let ptr = c_str.into_raw();
                 crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);
                 ptr
            },
            Err(_) => std::ptr::null_mut(),
        }
    } else {
        std::ptr::null_mut()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_file_write_string(file: *mut File, content: *const c_char) {
    if file.is_null() || content.is_null() {
        return;
    }
    let file = unsafe { &mut *file };
    let c_str = unsafe { CStr::from_ptr(content) };
    if let Ok(s) = c_str.to_str() {
        let _ = file.write_all(s.as_bytes());
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_file_close(file: *mut File) {
    if !file.is_null() {
        crate::tl_log_free(file as *const c_void, std::ptr::null(), 0);
        unsafe {
            let _ = Box::from_raw(file); // Dropping closes the file
        }
    }
}

// --- Binary File I/O ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_file_read_binary(path: *const c_char) -> *mut Vec<u8> {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let expanded_path = crate::expand_tilde(&path_str);

    match std::fs::read(&expanded_path) {
        Ok(data) => {
            let ptr = Box::into_raw(Box::new(data));
             crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);
             ptr
        },
        Err(e) => {
            eprintln!("tl_file_read_binary error: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_file_write_binary(path: *const c_char, data: *mut Vec<u8>) -> bool {
    if path.is_null() || data.is_null() {
        return false;
    }
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let expanded_path = crate::expand_tilde(&path_str);
    let data_vec = unsafe { &*data };

    match std::fs::write(&expanded_path, data_vec) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("tl_file_write_binary error: {}", e);
            false
        }
    }
}

// --- Image Loading ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_load_grayscale(path: *const c_char) -> *mut Vec<u8> {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let expanded_path = crate::expand_tilde(&path_str);

    match image::open(&expanded_path) {
        Ok(img) => {
            // Convert to grayscale and get raw pixels
            let gray = img.to_luma8();
            let pixels: Vec<u8> = gray.into_raw();
            let ptr = Box::into_raw(Box::new(pixels));
            crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);
            ptr
        }
        Err(e) => {
            eprintln!("tl_image_load_grayscale error: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_width(path: *const c_char) -> i64 {
    if path.is_null() {
        return 0;
    }
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let expanded_path = crate::expand_tilde(&path_str);

    match image::image_dimensions(&expanded_path) {
        Ok((w, _)) => w as i64,
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_height(path: *const c_char) -> i64 {
    if path.is_null() {
        return 0;
    }
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let expanded_path = crate::expand_tilde(&path_str);

    match image::image_dimensions(&expanded_path) {
        Ok((_, h)) => h as i64,
        Err(_) => 0,
    }
}

// --- Path ---

use std::path::PathBuf;

#[unsafe(no_mangle)]
pub extern "C" fn tl_path_new(path: *const c_char) -> *mut PathBuf {
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let expanded_path = crate::expand_tilde(&path_str);
    let ptr = Box::into_raw(Box::new(std::path::PathBuf::from(expanded_path)));
    crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_path_join(base: *mut PathBuf, part: *const c_char) -> *mut PathBuf {
    if base.is_null() {
        return std::ptr::null_mut();
    }
    let base = unsafe { &*base };
    let part_str = unsafe { CStr::from_ptr(part).to_string_lossy() };
    let ptr = Box::into_raw(Box::new(base.join(part_str.as_ref())));
    crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_path_exists(path: *mut PathBuf) -> bool {
    if path.is_null() {
        return false;
    }
    let path = unsafe { &*path };
    path.exists()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_path_is_dir(path: *mut PathBuf) -> bool {
    if path.is_null() {
        return false;
    }
    let path = unsafe { &*path };
    path.is_dir()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_path_is_file(path: *mut PathBuf) -> bool {
    if path.is_null() {
        return false;
    }
    let path = unsafe { &*path };
    path.is_file()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_path_to_string(path: *mut PathBuf) -> *mut c_char {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let path = unsafe { &*path };
    let s = path.to_string_lossy().into_owned();
    match CString::new(s) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_path_free(path: *mut PathBuf) {
    if !path.is_null() {
        unsafe {
            let _ = Box::from_raw(path);
        }
    }
}

// --- Http ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_http_download(url: *const c_char, dest: *const c_char) -> bool {
    let url_str = unsafe { CStr::from_ptr(url).to_string_lossy() };
    let dest_str = unsafe { CStr::from_ptr(dest).to_string_lossy() };
    let expanded_dest = crate::expand_tilde(&dest_str);

    println!("Downloading {} ...", url_str);

    // Use blocking client
    let mut response = match reqwest::blocking::get(url_str.as_ref()) {
        Ok(res) if res.status().is_success() => res,
        _ => {
            eprintln!("Failed to connect to {}", url_str);
            return false;
        }
    };

    let total_size = response.content_length().map(|ct| ct as f64).unwrap_or(0.0);

    let mut file = match File::create(&expanded_dest) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create file {}: {}", dest_str, e);
            return false;
        }
    };

    let mut downloaded: u64 = 0;
    let mut buffer = [0; 8192];

    while let Ok(n) = response.read(&mut buffer) {
        if n == 0 {
            break;
        }
        if file.write_all(&buffer[..n]).is_err() {
            eprintln!("Write error");
            return false;
        }
        downloaded += n as u64;

        if total_size > 0.0 {
            print!(
                "\rProgress: {:.2} MB / {:.2} MB ({:.1}%)",
                downloaded as f64 / 1024.0 / 1024.0,
                total_size / 1024.0 / 1024.0,
                (downloaded as f64 / total_size) * 100.0
            );
        } else {
            print!(
                "\rDownloaded: {:.2} MB",
                downloaded as f64 / 1024.0 / 1024.0
            );
        }
        let _ = std::io::stdout().flush();
    }
    println!("\nDownload complete: {}", dest_str);
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_http_get(url: *const c_char) -> *mut c_char {
    let url_str = unsafe { CStr::from_ptr(url).to_string_lossy() };
    if let Ok(response) = reqwest::blocking::get(url_str.as_ref()) {
        if let Ok(text) = response.text() {
            match CString::new(text) {
                Ok(c_str) => return c_str.into_raw(),
                Err(_) => return std::ptr::null_mut(),
            }
        }
    }
    std::ptr::null_mut()
}

// --- Env ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_env_get(key: *const c_char) -> *mut c_char {
    let key_str = unsafe { CStr::from_ptr(key).to_string_lossy() };
    match std::env::var(key_str.as_ref()) {
        Ok(val) => match CString::new(val) {
            Ok(c_str) => c_str.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_env_set(key: *const c_char, value: *const c_char) {
    let key_str = unsafe { CStr::from_ptr(key).to_string_lossy() };
    let value_str = unsafe { CStr::from_ptr(value).to_string_lossy() };
    unsafe {
        std::env::set_var(key_str.as_ref(), value_str.as_ref());
    }
}

// --- System ---

lazy_static::lazy_static! {
    static ref START_TIME: std::time::Instant = std::time::Instant::now();
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_system_time() -> f32 {
    START_TIME.elapsed().as_secs_f32()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_system_sleep(seconds: f32) {
    std::thread::sleep(std::time::Duration::from_secs_f32(seconds));
}
