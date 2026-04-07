use crossbeam_channel::{unbounded, bounded, Sender, Receiver};
use std::sync::Arc;

// A generic channel will store u64 (representing pointers or raw primitives up to 64 bits).
// This relies on TL passing `T` as a 64-bit word (either an integer, a float, or a pointer to a struct/tensor).

pub struct TlChannel {
    sender: Sender<u64>,
    receiver: Receiver<u64>,
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_channel_new(cap: i64) -> *mut Arc<TlChannel> {
    let (s, r) = if cap <= 0 {
        unbounded()
    } else {
        bounded(cap as usize)
    };
    
    let ch = Arc::new(TlChannel {
        sender: s,
        receiver: r,
    });
    
    let raw = Box::into_raw(Box::new(ch));
    raw
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_channel_send(ptr: *const Arc<TlChannel>, value: u64) -> bool {
    if ptr.is_null() { return false; }
    let ch = unsafe { &*ptr };
    ch.sender.send(value).is_ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_channel_recv(ptr: *const Arc<TlChannel>) -> u64 {
    if ptr.is_null() { return 0; }
    let ch = unsafe { &*ptr };
    ch.receiver.recv().unwrap_or(0) // Default to 0 on failure/disconnect
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_channel_try_recv(ptr: *const Arc<TlChannel>, success_out: *mut bool) -> u64 {
    if ptr.is_null() { 
        unsafe { if !success_out.is_null() { *success_out = false; } }
        return 0; 
    }
    let ch = unsafe { &*ptr };
    match ch.receiver.try_recv() {
        Ok(v) => {
            unsafe { if !success_out.is_null() { *success_out = true; } }
            v
        }
        Err(_) => {
            unsafe { if !success_out.is_null() { *success_out = false; } }
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_channel_clone(ptr: *const Arc<TlChannel>) -> *mut Arc<TlChannel> {
    if ptr.is_null() { return std::ptr::null_mut(); }
    let arc = unsafe { &*ptr };
    Box::into_raw(Box::new(arc.clone()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_channel_free(ptr: *mut Arc<TlChannel>) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}
