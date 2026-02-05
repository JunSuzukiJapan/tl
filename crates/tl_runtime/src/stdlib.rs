use std::ffi::{CStr, CString, c_void};
use std::fs::File;
use std::io::{Read, Write};

// use std::path::Path;
// --- Strings ---

#[unsafe(no_mangle)]

pub extern "C" fn tl_string_free(s: *mut crate::StringStruct) {
    if s.is_null() {
        return;
    }
    unsafe {
        // Free the inner buffer
        if !(*s).ptr.is_null() {
             let _ = CString::from_raw((*s).ptr);
        }
        // Free the struct itself
        let layout = std::alloc::Layout::new::<crate::StringStruct>();
        std::alloc::dealloc(s as *mut u8, layout);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_prompt(prompt: *mut crate::StringStruct) -> *mut crate::StringStruct {
    // Print prompt
    if !prompt.is_null() {
        unsafe {
            if !(*prompt).ptr.is_null() {
                let p = CStr::from_ptr((*prompt).ptr).to_string_lossy();
                print!("{}", p);
                let _ = std::io::stdout().flush();
            }
        }
    }

    // Read line
    let mut buffer = String::new();
    match std::io::stdin().read_line(&mut buffer) {
        Ok(_) => {
            // Remove trailing newline
            let trimmed = buffer.trim_end().to_string();
            match CString::new(trimmed) {
                Ok(c_str) => crate::tl_string_new(c_str.as_ptr()),
                Err(_) => std::ptr::null_mut(),
            }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_read_line(prompt: *mut crate::StringStruct) -> *mut crate::StringStruct {
    tl_prompt(prompt)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_string_concat(s1: *mut crate::StringStruct, s2: *mut crate::StringStruct) -> *mut crate::StringStruct {
    if s1.is_null() || s2.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let s1_ptr = (*s1).ptr;
        let s2_ptr = (*s2).ptr;
        if s1_ptr.is_null() || s2_ptr.is_null() { return std::ptr::null_mut(); }

        let s1_str = std::ffi::CStr::from_ptr(s1_ptr).to_string_lossy();
        let s2_str = std::ffi::CStr::from_ptr(s2_ptr).to_string_lossy();
        let joined = format!("{}{}", s1_str, s2_str);
        
        // Use tl_string_new to allocate StringStruct
        match std::ffi::CString::new(joined) {
            Ok(c_str) => crate::tl_string_new(c_str.as_ptr()),
            Err(_) => std::ptr::null_mut(),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_string_eq(s1: *mut crate::StringStruct, s2: *mut crate::StringStruct) -> bool {
    if s1.is_null() && s2.is_null() { return true; }
    if s1.is_null() || s2.is_null() { return false; }
    unsafe {
        let p1 = (*s1).ptr as *const u8;
        let p2 = (*s2).ptr as *const u8;
        if p1.is_null() && p2.is_null() { return true; }
        if p1.is_null() || p2.is_null() { return false; }
        
        let mut i = 0;
        loop {
            let b1 = *p1.add(i);
            let b2 = *p2.add(i);
            if b1 != b2 { 
                return false; 
            }
            if b1 == 0 { 
                return true; 
            }
            i += 1;
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_string_to_i64(s: *mut crate::StringStruct) -> i64 {
    if s.is_null() {
        return 0;
    }
    unsafe {
        if (*s).ptr.is_null() { return 0; }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        s_str.parse::<i64>().unwrap_or(0)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_string_from_int(val: i64) -> *mut crate::StringStruct {
    let s = val.to_string();
    match CString::new(s) {
        Ok(c_str) => crate::tl_string_new(c_str.as_ptr()),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_string_contains(haystack: *mut crate::StringStruct, needle: *mut crate::StringStruct) -> bool {
    if haystack.is_null() || needle.is_null() {
        return false;
    }
    unsafe {
        if (*haystack).ptr.is_null() || (*needle).ptr.is_null() { return false; }
        let h_str = CStr::from_ptr((*haystack).ptr).to_string_lossy();
        let n_str = CStr::from_ptr((*needle).ptr).to_string_lossy();
        h_str.contains(n_str.as_ref())
    }
}

/// Get character at index from string (returns Unicode Scalar Value as i32, or 0 if out of bounds)
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_char_at(s: *mut crate::StringStruct, index: i64) -> i32 {
    if s.is_null() {
        return 0;
    }
    unsafe {
        if (*s).ptr.is_null() { return 0; }
        let s_str = CStr::from_ptr((*s).ptr).to_string_lossy();
        if index < 0 { return 0; }
        let idx = index as usize;
        if let Some(c) = s_str.chars().nth(idx) {
            c as i32
        } else {
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_string_from_char(c: i32) -> *mut crate::StringStruct {
    if let Some(ch) = std::char::from_u32(c as u32) {
        let s = ch.to_string();
        match CString::new(s) {
            Ok(c_str) => crate::tl_string_new(c_str.as_ptr()),
            Err(_) => std::ptr::null_mut(),
        }
    } else {
        std::ptr::null_mut()
    }
}

/// Get length of a string
#[unsafe(no_mangle)]
pub extern "C" fn tl_string_len(s: *mut crate::StringStruct) -> i64 {
    if s.is_null() {
        return 0;
    }
    unsafe {
        (*s).len
    }
}

// Hash string
#[unsafe(no_mangle)]
pub extern "C" fn tl_hash_string(s: *mut crate::StringStruct) -> i64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    if s.is_null() {
        return 0;
    }
    unsafe {
        if (*s).ptr.is_null() { return 0; }
        // Hash the bytes (content)
        let len = (*s).len as usize;
        let slice = std::slice::from_raw_parts((*s).ptr, len);

        let mut hasher = DefaultHasher::new();
        slice.hash(&mut hasher);
        hasher.finish() as i64
    }
}





// --- File I/O ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_file_open(path: *const std::os::raw::c_char, mode: *const std::os::raw::c_char) -> *mut File {
    if path.is_null() || mode.is_null() { return std::ptr::null_mut(); }
    unsafe {
        let path_str = CStr::from_ptr(path).to_string_lossy();
        let expanded_path = crate::expand_tilde(&path_str);
        let mode_str = CStr::from_ptr(mode).to_string_lossy();

        let f = match mode_str.as_ref() {
            "r" => File::open(&expanded_path),
            "w" => File::create(&expanded_path),
            _ => return std::ptr::null_mut(),
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
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_file_read_string(file: *mut File) -> *mut crate::StringStruct {
    if file.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let file = &mut *file;
        let mut content = String::new();
        if file.read_to_string(&mut content).is_ok() {
            match CString::new(content) {
                Ok(c_str) => crate::tl_string_new(c_str.as_ptr()),
                Err(_) => std::ptr::null_mut(),
            }
        } else {
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_file_write_string(file: *mut File, content: *mut crate::StringStruct) {
    if file.is_null() || content.is_null() {
        return;
    }
    unsafe {
        let file = &mut *file;
        if (*content).ptr.is_null() { return; }
        let c_str = CStr::from_ptr((*content).ptr);
        if let Ok(s) = c_str.to_str() {
            let _ = file.write_all(s.as_bytes());
        }
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
pub extern "C" fn tl_file_read_binary(path: *mut crate::StringStruct) -> *mut Vec<u8> {
    unsafe {
        if path.is_null() || (*path).ptr.is_null() { return std::ptr::null_mut(); }
        let path_str = CStr::from_ptr((*path).ptr).to_string_lossy();
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
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_file_write_binary(path: *mut crate::StringStruct, data: *mut Vec<u8>) -> bool {
    if path.is_null() || data.is_null() {
        return false;
    }
    unsafe {
        if (*path).ptr.is_null() { return false; }
        let path_str = CStr::from_ptr((*path).ptr).to_string_lossy();
        let expanded_path = crate::expand_tilde(&path_str);
        let data_vec = &*data;

        match std::fs::write(&expanded_path, data_vec) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("tl_file_write_binary error: {}", e);
                false
            }
        }
    }
}

// --- Image Loading ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_load_grayscale(path: *mut crate::StringStruct) -> *mut Vec<u8> {
    unsafe {
        if path.is_null() || (*path).ptr.is_null() { return std::ptr::null_mut(); }
        let path_str = CStr::from_ptr((*path).ptr).to_string_lossy();
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
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_width(path: *mut crate::StringStruct) -> i64 {
    unsafe {
        if path.is_null() || (*path).ptr.is_null() { return 0; }
        let path_str = CStr::from_ptr((*path).ptr).to_string_lossy();
        let expanded_path = crate::expand_tilde(&path_str);

    match image::image_dimensions(&expanded_path) {
        Ok((w, _)) => w as i64,
        Err(_) => 0,
    }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_height(path: *mut crate::StringStruct) -> i64 {
    unsafe {
        if path.is_null() || (*path).ptr.is_null() { return 0; }
        let path_str = CStr::from_ptr((*path).ptr).to_string_lossy();
        let expanded_path = crate::expand_tilde(&path_str);

    match image::image_dimensions(&expanded_path) {
        Ok((_, h)) => h as i64,
        Err(_) => 0,
    }
    }
}

// --- Path ---

use std::path::PathBuf;

#[unsafe(no_mangle)]
pub extern "C" fn tl_path_new(path: *mut crate::StringStruct) -> *mut PathBuf {
    unsafe {
        if path.is_null() || (*path).ptr.is_null() { return std::ptr::null_mut(); }
        let path_str = CStr::from_ptr((*path).ptr).to_string_lossy();
        let expanded_path = crate::expand_tilde(&path_str);
        let ptr = Box::into_raw(Box::new(std::path::PathBuf::from(expanded_path)));
        crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);
        ptr
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_path_join(base: *mut PathBuf, part: *mut crate::StringStruct) -> *mut PathBuf {
    if base.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let base = &*base;
        if part.is_null() || (*part).ptr.is_null() { return std::ptr::null_mut(); }
        let part_str = CStr::from_ptr((*part).ptr).to_string_lossy();
        let ptr = Box::into_raw(Box::new(base.join(part_str.as_ref())));
        crate::tl_log_alloc(ptr as *const c_void, 0, std::ptr::null(), 0);
        ptr
    }
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
pub extern "C" fn tl_path_to_string(path: *mut PathBuf) -> *mut crate::StringStruct {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let path = &*path;
        let s = path.to_string_lossy().into_owned();
        match CString::new(s) {
            Ok(c_str) => crate::tl_string_new(c_str.as_ptr()),
            Err(_) => std::ptr::null_mut(),
        }
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
pub extern "C" fn tl_http_download(url: *mut crate::StringStruct, dest: *mut crate::StringStruct) -> bool {
    unsafe {
        if url.is_null() || (*url).ptr.is_null() || dest.is_null() || (*dest).ptr.is_null() { return false; }
        let url_str = CStr::from_ptr((*url).ptr).to_string_lossy();
        let dest_str = CStr::from_ptr((*dest).ptr).to_string_lossy();
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
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_http_get(url: *mut crate::StringStruct) -> *mut crate::StringStruct {
    unsafe {
        if url.is_null() || (*url).ptr.is_null() { return std::ptr::null_mut(); }
        let url_str = CStr::from_ptr((*url).ptr).to_string_lossy();
        if let Ok(response) = reqwest::blocking::get(url_str.as_ref()) {
            if let Ok(text) = response.text() {
                match CString::new(text) {
                    Ok(c_str) => return crate::tl_string_new(c_str.as_ptr()),
                    Err(_) => return std::ptr::null_mut(),
                }
            }
        }
        std::ptr::null_mut()
    }
}

// --- Env ---

#[unsafe(no_mangle)]
pub extern "C" fn tl_env_get(key: *mut crate::StringStruct) -> *mut crate::StringStruct {
    unsafe {
        if key.is_null() || (*key).ptr.is_null() { return std::ptr::null_mut(); }
        let key_str = CStr::from_ptr((*key).ptr).to_string_lossy();
        match std::env::var(key_str.as_ref()) {
            Ok(val) => match CString::new(val) {
                Ok(c_str) => crate::tl_string_new(c_str.as_ptr()),
                Err(_) => std::ptr::null_mut(),
            },
            Err(_) => std::ptr::null_mut(),
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn tl_env_set(key: *mut crate::StringStruct, value: *mut crate::StringStruct) {
    unsafe {
        if key.is_null() || (*key).ptr.is_null() || value.is_null() || (*value).ptr.is_null() { return; }
        let key_str = CStr::from_ptr((*key).ptr).to_string_lossy();
        let value_str = CStr::from_ptr((*value).ptr).to_string_lossy();
        std::env::set_var(key_str.as_ref(), value_str.as_ref());
    }
}

// --- System ---

static START_TIME: std::sync::LazyLock<std::time::Instant> = std::sync::LazyLock::new(|| std::time::Instant::now());

#[unsafe(no_mangle)]
pub extern "C" fn tl_system_time() -> f32 {
    START_TIME.elapsed().as_secs_f32()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_system_sleep(seconds: f32) {
    std::thread::sleep(std::time::Duration::from_secs_f32(seconds));
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_system_exit(code: i64) {
    std::process::exit(code as i32);
}
