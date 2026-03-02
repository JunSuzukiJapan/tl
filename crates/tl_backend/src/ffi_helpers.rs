//! 共通 FFI ヘルパー
//!
//! Arc ベースのメモリ管理パターンを Metal / CUDA で共通化。
//! `make_tensor`, `release_if_live`, `acquire_tensor` とデバッグカウンタを提供。

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// FFI デバッグカウンタ
pub struct FfiCounters {
    pub make: AtomicUsize,
    pub release: AtomicUsize,
    pub acquire: AtomicUsize,
}

impl FfiCounters {
    pub const fn new() -> Self {
        Self {
            make: AtomicUsize::new(0),
            release: AtomicUsize::new(0),
            acquire: AtomicUsize::new(0),
        }
    }

    /// カウンタをリセット
    pub fn reset(&self) {
        self.make.swap(0, Ordering::SeqCst);
        self.release.swap(0, Ordering::SeqCst);
        self.acquire.swap(0, Ordering::SeqCst);
    }

    /// カウンタをダンプ
    pub fn dump(&self, label: &str) {
        let m = self.make.load(Ordering::SeqCst);
        let r = self.release.load(Ordering::SeqCst);
        let a = self.acquire.load(Ordering::SeqCst);
        eprintln!("[FFI_DBG:{}] make={}, release={}, acquire={}, live={}", label, m, r, a, m + a - r);
    }
}

/// テンソルを Arc で包んでポインタを返す (RC=1)
pub fn make_tensor<T>(t: T, counters: &FfiCounters) -> *mut T {
    counters.make.fetch_add(1, Ordering::Relaxed);
    let arc = Arc::new(UnsafeCell::new(t));
    Arc::into_raw(arc) as *mut T
}

/// Arc RC-1: raw pointer から Arc を復元し、drop で RC を減らす。
pub fn release_if_live<T>(t: *mut T, counters: &FfiCounters) {
    if t.is_null() { return; }
    counters.release.fetch_add(1, Ordering::Relaxed);
    unsafe { let _ = Arc::from_raw(t as *const UnsafeCell<T>); }
}

/// Arc RC+1: raw pointer の参照カウントを 1 増やす。
pub fn acquire_tensor<T>(t: *mut T, counters: &FfiCounters) {
    if t.is_null() { return; }
    counters.acquire.fetch_add(1, Ordering::Relaxed);
    unsafe {
        let arc = Arc::from_raw(t as *const UnsafeCell<T>);
        let cloned = arc.clone(); // RC+1
        let _ = Arc::into_raw(arc); // 元のポインタを維持
        std::mem::forget(cloned); // RC+1 を維持（drop しない）
    }
}

/// BackendResult を安全にポインタに変換するヘルパー
pub fn make_result<T>(
    result: crate::error::Result<T>,
    counters: &FfiCounters,
    label: &str,
) -> *mut T {
    match result {
        Ok(t) => make_tensor(t, counters),
        Err(e) => {
            eprintln!("{} Backend FFI Error: {}", label, e);
            std::ptr::null_mut()
        }
    }
}
