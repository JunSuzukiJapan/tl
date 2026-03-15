//! CUDA Runtime API FFI バインディング
//!
//! 最小限の CUDA Runtime API を定義。

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::c_void;

/// CUDA エラーコード
pub type cudaError_t = i32;

/// CUDA 成功
pub const CUDA_SUCCESS: cudaError_t = 0;

/// CUDA ストリームハンドル
pub type cudaStream_t = *mut c_void;

/// CUDA グラフハンドル
pub type cudaGraph_t = *mut c_void;

/// CUDA グラフ実行可能ハンドル
pub type cudaGraphExec_t = *mut c_void;

/// cudaStreamCaptureMode
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
}

/// cudaMemcpy の方向
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
}

/// CUDA デバイスプロパティ（最小限）
#[repr(C)]
pub struct cudaDeviceProp {
    pub name: [u8; 256],
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: i32,
    pub warpSize: i32,
    pub memPitch: usize,
    pub maxThreadsPerBlock: i32,
    pub maxThreadsDim: [i32; 3],
    pub maxGridSize: [i32; 3],
    pub clockRate: i32,
    pub totalConstMem: usize,
    pub major: i32,
    pub minor: i32,
    // 残りのフィールドはパディングで吸収
    _padding: [u8; 4096],
}

impl Default for cudaDeviceProp {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

extern "C" {
    // === デバイス管理 ===
    pub fn cudaGetDeviceCount(count: *mut i32) -> cudaError_t;
    pub fn cudaSetDevice(device: i32) -> cudaError_t;
    pub fn cudaGetDevice(device: *mut i32) -> cudaError_t;
    pub fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: i32) -> cudaError_t;

    // === メモリ管理 ===
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> cudaError_t;
    pub fn cudaMemsetAsync(
        devPtr: *mut c_void,
        value: i32,
        count: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    // === ストリーム管理 ===
    pub fn cudaStreamCreate(stream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;

    // === エラー処理 ===
    pub fn cudaGetLastError() -> cudaError_t;
    pub fn cudaGetErrorString(error: cudaError_t) -> *const i8;

    // === グラフ管理 ===
    pub fn cudaStreamBeginCapture(stream: cudaStream_t, mode: cudaStreamCaptureMode) -> cudaError_t;
    pub fn cudaStreamEndCapture(stream: cudaStream_t, graph: *mut cudaGraph_t) -> cudaError_t;
    pub fn cudaGraphInstantiate(exec: *mut cudaGraphExec_t, graph: cudaGraph_t, error_node: *mut c_void, log: *mut i8, buf_size: usize) -> cudaError_t;
    pub fn cudaGraphLaunch(exec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t;
    pub fn cudaGraphExecDestroy(exec: cudaGraphExec_t) -> cudaError_t;
    pub fn cudaGraphGetNodes(graph: cudaGraph_t, nodes: *mut c_void, num_nodes: *mut usize) -> cudaError_t;
}

/// CUDA エラーをチェックし Result に変換
pub fn check_cuda(err: cudaError_t) -> Result<(), String> {
    if err == CUDA_SUCCESS {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = cudaGetErrorString(err);
            if ptr.is_null() {
                format!("CUDA error code: {}", err)
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().to_string()
            }
        };
        Err(msg)
    }
}

/// CUDA エラーチェックマクロ風ヘルパー
#[macro_export]
macro_rules! cuda_check {
    ($expr:expr) => {{
        let err = unsafe { $expr };
        $crate::cuda_sys::check_cuda(err)
    }};
}
