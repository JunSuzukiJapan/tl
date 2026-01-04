use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{Read, Write};
use std::os::raw::c_char;
// use std::path::Path;
// --- Strings ---

#[no_mangle]
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
            ptr
        }
        Err(_) => std::ptr::null_mut(),
    }
}

// --- File I/O ---

#[no_mangle]
pub extern "C" fn tl_file_open(path: *const c_char, mode: *const c_char) -> *mut File {
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let mode_str = unsafe { CStr::from_ptr(mode).to_string_lossy() };

    let f = match mode_str.as_ref() {
        "r" => File::open(path_str.as_ref()),
        "w" => File::create(path_str.as_ref()),
        _ => return std::ptr::null_mut(), // TODO: proper append support or error
    };

    match f {
        Ok(file) => Box::into_raw(Box::new(file)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tl_file_read_string(file: *mut File) -> *mut c_char {
    if file.is_null() {
        return std::ptr::null_mut();
    }
    let file = unsafe { &mut *file };
    let mut content = String::new();
    // Seek to start? usually implicit for linear read
    if file.read_to_string(&mut content).is_ok() {
        match CString::new(content) {
            Ok(c_str) => c_str.into_raw(),
            Err(_) => std::ptr::null_mut(),
        }
    } else {
        std::ptr::null_mut()
    }
}

#[no_mangle]
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

#[no_mangle]
pub extern "C" fn tl_file_close(file: *mut File) {
    if !file.is_null() {
        unsafe {
            let _ = Box::from_raw(file); // Dropping closes the file
        }
    }
}

// --- Path ---

use std::path::PathBuf;

#[no_mangle]
pub extern "C" fn tl_path_new(path: *const c_char) -> *mut PathBuf {
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy() };
    Box::into_raw(Box::new(PathBuf::from(path_str.as_ref())))
}

#[no_mangle]
pub extern "C" fn tl_path_join(base: *mut PathBuf, part: *const c_char) -> *mut PathBuf {
    if base.is_null() {
        return std::ptr::null_mut();
    }
    let base = unsafe { &*base };
    let part_str = unsafe { CStr::from_ptr(part).to_string_lossy() };
    Box::into_raw(Box::new(base.join(part_str.as_ref())))
}

#[no_mangle]
pub extern "C" fn tl_path_exists(path: *mut PathBuf) -> bool {
    if path.is_null() {
        return false;
    }
    let path = unsafe { &*path };
    path.exists()
}

#[no_mangle]
pub extern "C" fn tl_path_is_dir(path: *mut PathBuf) -> bool {
    if path.is_null() {
        return false;
    }
    let path = unsafe { &*path };
    path.is_dir()
}

#[no_mangle]
pub extern "C" fn tl_path_is_file(path: *mut PathBuf) -> bool {
    if path.is_null() {
        return false;
    }
    let path = unsafe { &*path };
    path.is_file()
}

#[no_mangle]
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

#[no_mangle]
pub extern "C" fn tl_path_free(path: *mut PathBuf) {
    if !path.is_null() {
        unsafe {
            let _ = Box::from_raw(path);
        }
    }
}

// --- Http ---

#[no_mangle]
pub extern "C" fn tl_http_download(url: *const c_char, dest: *const c_char) -> bool {
    let url_str = unsafe { CStr::from_ptr(url).to_string_lossy() };
    let dest_str = unsafe { CStr::from_ptr(dest).to_string_lossy() };

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

    let mut file = match File::create(dest_str.as_ref()) {
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

#[no_mangle]
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

#[no_mangle]
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
#[no_mangle]
pub extern "C" fn tl_env_set(key: *const c_char, value: *const c_char) {
    let key_str = unsafe { CStr::from_ptr(key).to_string_lossy() };
    let value_str = unsafe { CStr::from_ptr(value).to_string_lossy() };
    std::env::set_var(key_str.as_ref(), value_str.as_ref());
}

// --- System ---

lazy_static::lazy_static! {
    static ref START_TIME: std::time::Instant = std::time::Instant::now();
}

#[no_mangle]
pub extern "C" fn tl_system_time() -> f32 {
    START_TIME.elapsed().as_secs_f32()
}

#[no_mangle]
pub extern "C" fn tl_system_sleep(seconds: f32) {
    std::thread::sleep(std::time::Duration::from_secs_f32(seconds));
}
