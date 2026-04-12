use std::collections::HashMap;
use std::ffi::CStr;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::string_ffi::StringStruct;

#[repr(C)]
pub struct TlVecU8 {
    pub ptr: *mut u8,
    pub cap: i64,
    pub len: i64,
}

static LISTENER_REGISTRY: OnceLock<Mutex<HashMap<i64, Arc<TcpListener>>>> = OnceLock::new();
static STREAM_REGISTRY: OnceLock<Mutex<HashMap<i64, Arc<Mutex<TcpStream>>>>> = OnceLock::new();
static NEXT_ID: AtomicI64 = AtomicI64::new(1);

fn get_listener_registry() -> &'static Mutex<HashMap<i64, Arc<TcpListener>>> {
    LISTENER_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_stream_registry() -> &'static Mutex<HashMap<i64, Arc<Mutex<TcpStream>>>> {
    STREAM_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}



#[unsafe(no_mangle)]
pub extern "C" fn tl_net_listener_bind(addr: *mut StringStruct) -> i64 {
    unsafe {
        if addr.is_null() || (*addr).ptr.is_null() {
            return 0;
        }
        let addr_str = CStr::from_ptr((*addr).ptr).to_string_lossy();
        match TcpListener::bind(addr_str.as_ref()) {
            Ok(listener) => {
                let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
                let mut reg = get_listener_registry().lock().unwrap();
                reg.insert(id, Arc::new(listener));
                id
            }
            Err(_) => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_net_listener_accept(listener_id: i64) -> i64 {
    let listener = {
        let reg = get_listener_registry().lock().unwrap();
        match reg.get(&listener_id) {
            Some(l) => l.clone(),
            None => return 0,
        }
    };

    match listener.accept() {
        Ok((stream, _addr)) => {
            let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
            let mut reg = get_stream_registry().lock().unwrap();
            reg.insert(id, Arc::new(Mutex::new(stream)));
            id
        }
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_net_stream_connect(addr: *mut StringStruct) -> i64 {
    unsafe {
        if addr.is_null() || (*addr).ptr.is_null() {
            return 0;
        }
        let addr_str = CStr::from_ptr((*addr).ptr).to_string_lossy();
        
        match TcpStream::connect(addr_str.as_ref()) {
            Ok(stream) => {
                let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
                let mut reg = get_stream_registry().lock().unwrap();
                reg.insert(id, Arc::new(Mutex::new(stream)));
                id
            }
            Err(_) => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_net_stream_write(stream_id: i64, buf: *mut TlVecU8) -> i64 {
    let stream_mut = {
        let reg = get_stream_registry().lock().unwrap();
        match reg.get(&stream_id) {
            Some(s) => s.clone(),
            None => return -1,
        }
    };
    
    if buf.is_null() {
        return -1;
    }

    let buf_slice = unsafe { std::slice::from_raw_parts((*buf).ptr, (*buf).len as usize) };
    let mut s = stream_mut.lock().unwrap();
    
    match s.write(buf_slice) {
        Ok(n) => n as i64,
        Err(_) => -1,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_net_stream_read(stream_id: i64, max_len: i64) -> *mut TlVecU8 {
    let stream_mut = {
        let reg = get_stream_registry().lock().unwrap();
        match reg.get(&stream_id) {
            Some(s) => s.clone(),
            None => return std::ptr::null_mut(),
        }
    };

    let mut tmp_buf = vec![0u8; max_len as usize];
    let mut s = stream_mut.lock().unwrap();

    match s.read(&mut tmp_buf) {
        Ok(n) => {
            tmp_buf.truncate(n);
            let len = tmp_buf.len() as i64;
            let cap = len;
            let data_ptr = unsafe {
                if len == 0 {
                    std::ptr::null_mut()
                } else {
                    let layout = std::alloc::Layout::from_size_align(tmp_buf.len(), 1).unwrap();
                    let ptr = std::alloc::alloc(layout);
                    std::ptr::copy_nonoverlapping(tmp_buf.as_ptr(), ptr, tmp_buf.len());
                    ptr
                }
            };
            let vec_ptr = unsafe {
                let layout = std::alloc::Layout::new::<TlVecU8>();
                let ptr = std::alloc::alloc(layout) as *mut TlVecU8;
                (*ptr).ptr = data_ptr;
                (*ptr).cap = cap;
                (*ptr).len = len;
                ptr
            };
            vec_ptr
        }
        Err(_) => std::ptr::null_mut(),
    }
}
