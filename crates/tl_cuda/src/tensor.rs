//! CUDA テンソル

use crate::autograd::GradFn;
use crate::buffer_pool::{pool_acquire, pool_release};
use crate::cuda_sys::{self, cudaMemcpyKind, CUDA_SUCCESS};
use crate::device::{get_device, CudaDevice};
use crate::{shape_to_bytes, DType};
use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::sync::Arc;
use tl_backend::BackendResult;

/// Arc ベースのテンソル参照（V5.0 メモリ管理）
/// GradFn 内で入力テンソルの生存を保証する。
pub type TensorRef = Arc<UnsafeCell<CudaTensor>>;

/// 生ポインタから TensorRef を取得する（RC+1）。
pub unsafe fn tensor_ref_from_ptr(ptr: *mut CudaTensor) -> TensorRef {
    let arc: TensorRef = Arc::from_raw(ptr as *const UnsafeCell<CudaTensor>);
    let cloned = arc.clone(); // RC+1
    std::mem::forget(arc); // 元のポインタは生かす
    cloned
}

/// TensorRef から内部の CudaTensor への不変参照を取得する。
#[inline]
pub unsafe fn tensor_ref_get(r: &TensorRef) -> &CudaTensor {
    &*r.get()
}

/// TensorRef から内部の CudaTensor への可変参照を取得する。
#[inline]
pub unsafe fn tensor_ref_get_mut(r: &TensorRef) -> &mut CudaTensor {
    &mut *r.get()
}

/// CUDA GPU バッファ
/// Drop 時に cudaFree を呼んで GPU メモリを解放する。
pub struct CudaBuffer {
    ptr: *mut c_void,
    size: usize,
}

unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    /// 新しい GPU バッファを確保
    pub fn new(size: usize) -> Result<Self, String> {
        let device = get_device();
        let ptr = device.allocate_buffer(size)?;
        Ok(CudaBuffer { ptr, size })
    }

    /// GPU メモリへのポインタ
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// バッファサイズ（バイト数）
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cuda_sys::cudaFree(self.ptr);
            }
        }
    }
}

/// Autograd メタデータ
pub struct AutogradMeta {
    /// 勾配（累積）
    pub grad: Option<CudaTensor>,
    /// 勾配関数（リーフノードは None）
    pub grad_fn: Option<Box<dyn GradFn>>,
    /// 勾配が必要か
    pub requires_grad: bool,
}

/// CUDA GPU テンソル
pub struct CudaTensor {
    /// GPU バッファ
    buffer: Arc<CudaBuffer>,
    /// 形状
    pub shape: Vec<usize>,
    /// データ型
    pub dtype: DType,
    /// デバイス
    #[allow(dead_code)]
    device: Arc<CudaDevice>,
    /// Autograd メタデータ（None = autograd 不要）
    pub autograd: Option<Box<AutogradMeta>>,
}

impl std::fmt::Debug for CudaTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("buffer_ptr", &self.buffer.ptr())
            .finish()
    }
}

impl Clone for CudaTensor {
    fn clone(&self) -> Self {
        // データを複製して新しいテンソルを作成
        self.clone_data().expect("Clone failed")
    }
}

unsafe impl Send for CudaTensor {}
unsafe impl Sync for CudaTensor {}

impl CudaTensor {
    /// 新しいテンソルを作成（未初期化）
    pub fn uninit(shape: &[usize], dtype: DType) -> Self {
        let device = get_device();
        let size = shape_to_bytes(shape, dtype);

        // プールから取得を試みる
        let buffer = pool_acquire(size).unwrap_or_else(|| {
            // プールになければ新規確保
            Arc::new(CudaBuffer::new(size).expect("CUDA buffer allocation failed"))
        });

        CudaTensor {
            buffer,
            shape: shape.to_vec(),
            dtype,
            device,
            autograd: None,
        }
    }

    /// ゼロで初期化されたテンソルを作成
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let tensor = Self::uninit(shape, dtype);
        let size = shape_to_bytes(shape, dtype);
        if size > 0 {
            let err = unsafe { cuda_sys::cudaMemset(tensor.buffer.ptr(), 0, size) };
            if err != CUDA_SUCCESS {
                eprintln!("cudaMemset failed in zeros(): {}", err);
            }
        }
        tensor
    }

    /// スライスからテンソルを作成（Host→Device コピー）
    pub fn from_slice<T: Copy>(data: &[T], shape: &[usize], dtype: DType) -> Self {
        let tensor = Self::uninit(shape, dtype);
        let byte_size = data.len() * std::mem::size_of::<T>();
        if byte_size > 0 {
            let err = unsafe {
                cuda_sys::cudaMemcpy(
                    tensor.buffer.ptr(),
                    data.as_ptr() as *const c_void,
                    byte_size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            };
            if err != CUDA_SUCCESS {
                panic!("cudaMemcpy HostToDevice failed in from_slice(): {}", err);
            }
        }
        tensor
    }

    /// 形状
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// データ型
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 要素数
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// GPU バッファへのポインタ
    pub fn buffer_ptr(&self) -> *mut c_void {
        self.buffer.ptr()
    }

    /// GPU バッファの Arc を取得
    pub fn buffer_arc(&self) -> &Arc<CudaBuffer> {
        &self.buffer
    }

    /// 全て1で初期化
    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        let count = shape.iter().product::<usize>();
        match dtype {
            DType::F32 => {
                let data = vec![1.0f32; count];
                Self::from_slice(&data, shape, dtype)
            }
            DType::I64 => {
                let data = vec![1i64; count];
                Self::from_slice(&data, shape, dtype)
            }
            DType::I32 => {
                let data = vec![1i32; count];
                Self::from_slice(&data, shape, dtype)
            }
            _ => unimplemented!("ones for {:?}", dtype),
        }
    }

    /// 正規乱数で初期化
    pub fn randn(shape: &[usize], dtype: DType) -> Self {
        use rand::Rng;
        let count = shape.iter().product::<usize>();
        match dtype {
            DType::F32 => {
                let mut rng = rand::thread_rng();
                let data: Vec<f32> = (0..count)
                    .map(|_| {
                        let u1: f32 = rng.gen();
                        let u2: f32 = rng.gen();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
                    })
                    .collect();
                Self::from_slice(&data, shape, dtype)
            }
            _ => unimplemented!("randn for {:?}", dtype),
        }
    }

    /// データを CPU にコピー
    pub fn to_vec<T: Copy + Default>(&self) -> Vec<T> {
        // GPU 演算の完了を保証
        crate::stream::sync_stream();

        let count = self.elem_count();
        if count == 0 {
            return Vec::new();
        }

        let t_size = std::mem::size_of::<T>();
        let dtype_size = match self.dtype {
            DType::F32 => 4,
            DType::I64 => 8,
            DType::I32 => 4,
            _ => t_size,
        };

        if t_size == dtype_size {
            // 直接コピー（型サイズが一致）
            let byte_size = count * t_size;
            let mut result = vec![T::default(); count];
            let err = unsafe {
                cuda_sys::cudaMemcpy(
                    result.as_mut_ptr() as *mut c_void,
                    self.buffer.ptr() as *const c_void,
                    byte_size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            if err != CUDA_SUCCESS {
                eprintln!("cudaMemcpy DeviceToHost failed in to_vec(): {} (byte_size={}, buffer_size={}, shape={:?}, dtype={:?})",
                    err, byte_size, self.buffer.size(), self.shape, self.dtype);
                return vec![T::default(); count];
            }
            result
        } else if dtype_size == 4 && t_size == 8 {
            // F32/I32 → i64: まず f32 で読んでから変換
            let byte_size = count * 4;
            let mut f32_buf = vec![0.0f32; count];
            let err = unsafe {
                cuda_sys::cudaMemcpy(
                    f32_buf.as_mut_ptr() as *mut c_void,
                    self.buffer.ptr() as *const c_void,
                    byte_size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            if err != CUDA_SUCCESS {
                eprintln!("cudaMemcpy DeviceToHost failed in to_vec() (f32→i64): {} (byte_size={}, buffer_size={}, shape={:?})",
                    err, byte_size, self.buffer.size(), self.shape);
                return vec![T::default(); count];
            }
            // f32 → i64 変換
            let i64_buf: Vec<i64> = f32_buf.iter().map(|&x| x as i64).collect();
            let mut result = vec![T::default(); count];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    i64_buf.as_ptr() as *const T,
                    result.as_mut_ptr(),
                    count,
                );
            }
            result
        } else if dtype_size == 8 && t_size == 4 {
            // I64 → f32: まず i64 で読んでから変換
            let byte_size = count * 8;
            let mut i64_buf = vec![0i64; count];
            let err = unsafe {
                cuda_sys::cudaMemcpy(
                    i64_buf.as_mut_ptr() as *mut c_void,
                    self.buffer.ptr() as *const c_void,
                    byte_size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            if err != CUDA_SUCCESS {
                eprintln!("cudaMemcpy DeviceToHost failed in to_vec() (i64→f32): {} (byte_size={}, buffer_size={}, shape={:?})",
                    err, byte_size, self.buffer.size(), self.shape);
                return vec![T::default(); count];
            }
            // i64 → f32 変換
            let f32_buf: Vec<f32> = i64_buf.iter().map(|&x| x as f32).collect();
            let mut result = vec![T::default(); count];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    f32_buf.as_ptr() as *const T,
                    result.as_mut_ptr(),
                    count,
                );
            }
            result
        } else {
            // フォールバック: バッファサイズ分だけ読む
            let copy_size = self.buffer.size().min(count * t_size);
            let mut result = vec![T::default(); count];
            let err = unsafe {
                cuda_sys::cudaMemcpy(
                    result.as_mut_ptr() as *mut c_void,
                    self.buffer.ptr() as *const c_void,
                    copy_size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            if err != CUDA_SUCCESS {
                eprintln!(
                    "cudaMemcpy DeviceToHost failed in to_vec() (fallback): {}",
                    err
                );
            }
            result
        }
    }

    /// データを GPU 上で完全にコピー（DeviceToDevice）
    pub fn clone_data(&self) -> BackendResult<CudaTensor> {
        crate::stream::sync_stream();

        let result = CudaTensor::uninit(self.shape(), self.dtype());
        let size = shape_to_bytes(self.shape(), self.dtype());
        if size > 0 {
            let err = unsafe {
                cuda_sys::cudaMemcpy(
                    result.buffer.ptr(),
                    self.buffer.ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                )
            };
            if err != CUDA_SUCCESS {
                return Err(tl_backend::BackendError::InternalError(format!(
                    "cudaMemcpy DeviceToDevice failed: {}",
                    err
                )));
            }
        }
        Ok(result)
    }

    /// GPU バッファを共有する浅いクローン（データコピーなし）
    /// autograd メタデータはコピーしない
    pub fn shallow_clone(&self) -> Self {
        CudaTensor {
            buffer: Arc::clone(&self.buffer),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: Arc::clone(&self.device),
            autograd: if self.requires_grad() {
                Some(Box::new(AutogradMeta {
                    grad: None,
                    grad_fn: None,
                    requires_grad: true,
                }))
            } else {
                None
            },
        }
    }

    /// 既存バッファから新しい形状でテンソルを作成（バッファ共有）
    pub fn from_buffer_shared(shape: Vec<usize>, dtype: DType) -> Self {
        let device = get_device();
        let size = shape_to_bytes(&shape, dtype);
        let buffer = Arc::new(CudaBuffer::new(size).expect("CUDA buffer allocation failed"));
        CudaTensor {
            buffer,
            shape,
            dtype,
            device,
            autograd: None,
        }
    }

    // ========== Autograd メソッド ==========

    /// requires_grad フラグを確認
    pub fn requires_grad(&self) -> bool {
        self.autograd.as_ref().map_or(false, |a| a.requires_grad)
    }

    /// requires_grad を有効化
    pub fn enable_grad(&mut self) {
        if self.autograd.is_none() {
            self.autograd = Some(Box::new(AutogradMeta {
                grad: None,
                grad_fn: None,
                requires_grad: true,
            }));
        } else {
            self.autograd.as_mut().unwrap().requires_grad = true;
        }
    }

    /// grad_fn をセット
    pub fn set_grad_fn(&mut self, grad_fn: Box<dyn GradFn>) {
        self.autograd = Some(Box::new(AutogradMeta {
            grad: None,
            grad_fn: Some(grad_fn),
            requires_grad: true,
        }));
    }

    /// 勾配を取得（shallow clone）
    pub fn get_grad(&self) -> Option<CudaTensor> {
        self.autograd
            .as_ref()
            .and_then(|a| a.grad.as_ref().map(|g| g.shallow_clone()))
    }

    /// 勾配をゼロクリア
    pub fn zero_grad(&mut self) {
        if let Some(ref mut a) = self.autograd {
            a.grad = None;
        }
    }

    /// 勾配を累積
    pub fn accumulate_grad(&mut self, grad: CudaTensor) -> BackendResult<()> {
        if let Some(ref mut meta) = self.autograd {
            let detached_grad = grad.detach();
            if let Some(ref mut g) = meta.grad {
                *g = g.add_impl(&detached_grad)?;
            } else {
                meta.grad = Some(detached_grad);
            }
        }
        Ok(())
    }

    /// backward（逆伝播）
    pub fn backward(&mut self) -> BackendResult<()> {
        if !self.requires_grad() {
            return Ok(());
        }

        let ones = CudaTensor::ones(self.shape(), self.dtype());
        let self_ptr = self as *mut CudaTensor;

        // worklist: (ポインタ, 勾配, Arc参照保持)
        // Arc を保持しないとポインタが dangling になる
        let mut worklist: Vec<(*mut CudaTensor, CudaTensor, Option<TensorRef>)> =
            vec![(self_ptr, ones, None)];
        // visited: (ポインタ, Arc参照保持)
        // cleanup 時に grad_fn を drop → TensorRef drop → CudaTensor 解放を防ぐため
        // Arc 参照を cleanup 完了まで保持する
        let mut visited: Vec<(*mut CudaTensor, Option<TensorRef>)> = Vec::new();

        while let Some((tensor_ptr, grad_output, arc_ref)) = worklist.pop() {
            let tensor = unsafe { &mut *tensor_ptr };
            visited.push((tensor_ptr, arc_ref));

            eprintln!(
                "[BACKWARD] step {}, shape={:?}, grad_shape={:?}",
                visited.len(),
                tensor.shape(),
                grad_output.shape()
            );

            let propagation = if let Some(meta) = tensor.autograd.as_ref() {
                if let Some(gf) = meta.grad_fn.as_ref() {
                    eprintln!("[BACKWARD]   calling grad_fn.backward...");
                    let grads = gf.backward(&grad_output)?;
                    eprintln!("[BACKWARD]   grad_fn.backward done, {} grads", grads.len());
                    let inputs = gf.inputs();
                    eprintln!("[BACKWARD]   inputs: {}", inputs.len());
                    Some((grads, inputs))
                } else {
                    None
                }
            } else {
                None
            };

            eprintln!("[BACKWARD]   accumulate_grad...");
            tensor.accumulate_grad(grad_output)?;
            eprintln!("[BACKWARD]   accumulate_grad done");

            if let Some((grads, inputs)) = propagation {
                for (input_ref, grad) in inputs.into_iter().zip(grads.into_iter()) {
                    let input_ptr = input_ref.get() as *mut CudaTensor;
                    let input = unsafe { &*input_ptr };
                    if input.requires_grad() {
                        eprintln!(
                            "[BACKWARD]   push input shape={:?} grad_shape={:?}",
                            input.shape(),
                            grad.shape()
                        );
                        // input_ref (Arc) を worklist に保持してポインタを生かす
                        worklist.push((input_ptr, grad, Some(input_ref)));
                    }
                }
            }
        }

        eprintln!(
            "[BACKWARD] loop done, {} steps. Cleaning up graph...",
            visited.len()
        );

        // 計算グラフ解放
        // visited の Arc 参照が全テンソルを生かしているため安全にアクセス可能
        let visited_len = visited.len();
        for (i, entry) in visited.iter_mut().enumerate() {
            let tensor = unsafe { &mut *entry.0 };
            if let Some(ref mut meta) = tensor.autograd {
                meta.grad_fn = None;
            }
            if i % 10 == 0 {
                eprintln!("[BACKWARD]   cleanup {}/{}", i, visited_len);
            }
        }
        eprintln!("[BACKWARD] cleanup done. Dropping visited & worklist...");
        drop(visited);
        drop(worklist);
        eprintln!("[BACKWARD] backward() complete.");
        Ok(())
    }

    /// 計算グラフから切り離す
    pub fn detach(&self) -> CudaTensor {
        self.shallow_clone()
    }
}

impl Drop for CudaTensor {
    fn drop(&mut self) {
        // バッファの唯一の所有者なら、プールに返却して再利用
        if Arc::strong_count(&self.buffer) == 1 {
            pool_release(self.buffer.clone());
        }
    }
}
