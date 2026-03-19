//! Metal テンソル

use crate::buffer_pool::{pool_acquire, pool_release};
use crate::device::{get_device, MetalDevice};
use crate::{shape_to_bytes, DType};
use crate::autograd::GradFn;
use metal::{Buffer, MTLResourceOptions};
use std::cell::UnsafeCell;
use std::sync::Arc;
use tl_backend::{BackendResult};


/// Arc ベースのテンソル参照（V5.0 メモリ管理）
/// GradFn 内で入力テンソルの生存を保証する。
pub type TensorRef = Arc<UnsafeCell<MetalTensor>>;

/// 生ポインタから TensorRef を取得する（RC+1）。
/// FFI 境界で `*mut MetalTensor` を Autograd 用の `TensorRef` に変換する際に使用。
/// ポインタは `Arc::into_raw` で作成されたものでなければならない。
pub unsafe fn tensor_ref_from_ptr(ptr: *mut MetalTensor) -> TensorRef {
    let arc: TensorRef = Arc::from_raw(ptr as *const UnsafeCell<MetalTensor>);
    let cloned = arc.clone(); // RC+1
    std::mem::forget(arc);    // 元のポインタは生かす
    cloned
}

/// TensorRef から内部の MetalTensor への不変参照を取得する。
#[inline]
pub unsafe fn tensor_ref_get(r: &TensorRef) -> &MetalTensor {
    &*r.get()
}

/// TensorRef から内部の MetalTensor への可変参照を取得する。
#[inline]
pub unsafe fn tensor_ref_get_mut(r: &TensorRef) -> &mut MetalTensor {
    &mut *r.get()
}

/// Autograd メタデータ
pub struct AutogradMeta {
    /// 勾配（累積）
    pub grad: Option<MetalTensor>,
    /// 勾配関数（リーフノードは None）
    pub grad_fn: Option<Box<dyn GradFn>>,
    /// 勾配が必要か
    pub requires_grad: bool,
}

/// Metal GPU テンソル
pub struct MetalTensor {
    /// GPU バッファ
    buffer: Arc<Buffer>,
    /// 形状
    shape: Vec<usize>,
    /// データ型
    dtype: DType,
    /// デバイス
    #[allow(dead_code)]
    device: Arc<MetalDevice>,
    /// Autograd メタデータ（None = autograd 不要）
    pub autograd: Option<Box<AutogradMeta>>,
}

impl Clone for MetalTensor {
    fn clone(&self) -> Self {
        // データを複製して新しいテンソルを作成
        self.clone_data().expect("Clone failed")
    }
}

impl MetalTensor {
    /// 新しいテンソルを作成（未初期化）
    pub fn uninit(shape: &[usize], dtype: DType) -> Self {
        let device = get_device();
        let size = shape_to_bytes(shape, dtype);
        let options = MTLResourceOptions::StorageModeShared;

        // プールから取得を試みる
        let buffer = pool_acquire(size, options).unwrap_or_else(|| {
            // プールになければ新規確保
            Arc::new(device.allocate_buffer(size, options))
        });

        MetalTensor {
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
        let ptr = tensor.buffer.contents() as *mut u8;
        let size = shape_to_bytes(shape, dtype);
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }
        tensor
    }

    /// スライスからテンソルを作成
    pub fn from_slice<T: Copy + std::fmt::Debug>(data: &[T], shape: &[usize], dtype: DType) -> Self {
        let tensor = Self::uninit(shape, dtype);
        let ptr = tensor.buffer.contents() as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
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

    /// バッファへの参照
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// バッファの Arc を取得
    pub fn buffer_arc(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    /// 既存バッファから新しい形状でテンソルを作成（バッファ共有）
    pub fn from_buffer_shared(buffer: Arc<Buffer>, shape: Vec<usize>, dtype: DType) -> Self {
        let device = get_device();
        MetalTensor {
            buffer,
            shape,
            dtype,
            device,
            autograd: None,
        }
    }

    /// GPU バッファを共有する浅いクローン（データコピーなし）
    /// autograd メタデータはコピーしない
    pub fn shallow_clone(&self) -> Self {
        MetalTensor {
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

    /// 全て1で初期化
    pub fn ones(shape: &[usize], dtype: DType) -> BackendResult<Self> {
        if dtype == DType::F64 {
            return Err(tl_backend::BackendError::TypeMismatch(
                "Metal GPU does not support F64 (double precision). Use CPU backend for F64 tensors.".into()
            ));
        }
        let tensor = Self::uninit(shape, dtype);
        let count = tensor.elem_count();
        match dtype {
            DType::F32 => {
                let ptr = tensor.buffer.contents() as *mut f32;
                unsafe {
                    for i in 0..count {
                        *ptr.add(i) = 1.0;
                    }
                }
            }
            DType::I64 => {
                let ptr = tensor.buffer.contents() as *mut i64;
                unsafe {
                    for i in 0..count {
                        *ptr.add(i) = 1;
                    }
                }
            }
            DType::I32 => {
                let ptr = tensor.buffer.contents() as *mut i32;
                unsafe {
                    for i in 0..count {
                        *ptr.add(i) = 1;
                    }
                }
            }
            DType::U8 => {
                let ptr = tensor.buffer.contents() as *mut u8;
                unsafe {
                    for i in 0..count {
                        *ptr.add(i) = 1;
                    }
                }
            }
            _ => {
                // F16/BF16/U32: 未サポート — F32 にフォールバック
                eprintln!("Warning: ones for {:?} not supported, using F32 fallback", dtype);
                let fallback = Self::uninit(shape, DType::F32);
                let ptr = fallback.buffer.contents() as *mut f32;
                let fc = fallback.elem_count();
                unsafe {
                    for i in 0..fc {
                        *ptr.add(i) = 1.0;
                    }
                }
                return Ok(fallback);
            }
        }
        Ok(tensor)
    }

    /// 正規乱数で初期化
    pub fn randn(shape: &[usize], dtype: DType) -> BackendResult<Self> {
        if dtype == DType::F64 {
            return Err(tl_backend::BackendError::TypeMismatch(
                "Metal GPU does not support F64 (double precision). Use CPU backend for F64 tensors.".into()
            ));
        }
        use rand::Rng;
        let tensor = Self::uninit(shape, dtype);
        let mut rng = rand::thread_rng();
        match dtype {
            DType::F32 => {
                let ptr = tensor.buffer.contents() as *mut f32;
                let count = tensor.elem_count();
                unsafe {
                    for i in 0..count {
                        let u1: f32 = rng.gen();
                        let u2: f32 = rng.gen();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                        *ptr.add(i) = z;
                    }
                }
            }
            _ => {
                // 整数型/半精度: 乱数の意味が薄いため F32 で生成してフォールバック
                eprintln!("Warning: randn for {:?} not directly supported, using F32 fallback", dtype);
                return Self::randn(shape, DType::F32);
            }
        }
        Ok(tensor)
    }

    /// データを CPU にコピー
    /// GPU 上の未完了コマンドを暗黙的に同期してからコピーする。
    pub fn to_vec<T: Copy + Default>(&self) -> Vec<T> {
        // GPU 演算の完了を保証
        crate::command_stream::sync_stream();

        let count = self.elem_count();
        if count == 0 {
            return Vec::new();
        }
        let ptr = self.buffer.contents() as *const T;
        if ptr.is_null() {
            eprintln!("Warning: to_vec called on tensor with null buffer, returning zeros");
            return vec![T::default(); count];
        }
        let mut result = vec![T::default(); count];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), count);
        }
        result
    }

    /// データを GPU 上で完全にコピー（Blit）
    pub fn clone_data(&self) -> BackendResult<MetalTensor> {
        // 先行する GPU 演算を同期（ソースバッファが最新であることを保証）
        crate::command_stream::sync_stream();

        let result = MetalTensor::uninit(self.shape(), self.dtype());
        let device = get_device();
        let command_queue = device.command_queue();
        
        objc::rc::autoreleasepool(|| {
            let command_buffer = command_queue.new_command_buffer();
            let blit_encoder = command_buffer.new_blit_command_encoder();
            
            let size = shape_to_bytes(self.shape(), self.dtype());
            blit_encoder.copy_from_buffer(
                self.buffer(),
                0,
                result.buffer(),
                0,
                size as u64,
            );
            blit_encoder.end_encoding();
            
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        
        Ok(result)
    }

    // ========== Autograd メソッド ==========

    /// requires_grad フラグを確認
    pub fn requires_grad(&self) -> bool {
        self.autograd.as_ref().map_or(false, |a| a.requires_grad)
    }

    pub fn set_requires_grad(&mut self, req_grad: bool) {
        if req_grad {
            if self.autograd.is_none() {
                self.autograd = Some(Box::new(AutogradMeta {
                    grad: None,
                    grad_fn: None,
                    requires_grad: true,
                }));
            } else {
                if let Some(ref mut meta) = self.autograd {
                    meta.requires_grad = true;
                }
            }
        } else {
            if let Some(ref mut meta) = self.autograd {
                meta.requires_grad = false;
                meta.grad = None;       // 勾配データも消す
                meta.grad_fn = None;    // グラフからも切り離す
            }
        }
    }

    pub fn clip_grad_value(&mut self, min: f32, max: f32) -> BackendResult<()> {
        if let Some(ref mut meta) = self.autograd {
            if let Some(ref mut g) = meta.grad {
                // MetalTensor is immutable, so we must clone data, clamp it, and replace.
                let mut data = g.to_vec::<f32>();
                for x in data.iter_mut() {
                    if *x < min { *x = min; } else if *x > max { *x = max; }
                }
                *g = MetalTensor::from_slice(&data, g.shape(), g.dtype());
            }
        }
        Ok(())
    }

    pub fn clip_grad_norm(&mut self, max_norm: f32, norm_type: f32) -> BackendResult<f32> {
        let mut total_norm = 0.0f32;
        if let Some(ref mut meta) = self.autograd {
            if let Some(ref mut g) = meta.grad {
                let data = g.to_vec::<f32>();
                if norm_type == std::f32::INFINITY {
                    total_norm = data.iter().fold(0.0f32, |max, &x| max.max(x.abs()));
                } else {
                    let sum: f32 = data.iter().map(|&x| x.abs().powf(norm_type)).sum();
                    total_norm = sum.powf(1.0 / norm_type);
                }
                let clip_coef = max_norm / (total_norm + 1e-6);
                if clip_coef < 1.0 {
                    let scaled: Vec<f32> = data.iter().map(|&x| x * clip_coef).collect();
                    *g = MetalTensor::from_slice(&scaled, g.shape(), g.dtype());
                }
            }
        }
        Ok(total_norm)
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

    /// 勾配を取得（detach — autograd なし）
    /// パラメータ更新 (step) で grad of grad は不要なため、
    /// requires_grad = false のテンソルを返す。
    pub fn get_grad(&self) -> Option<MetalTensor> {
        self.autograd.as_ref().and_then(|a| {
            a.grad.as_ref().map(|g| {
                // detach: バッファ共有だが autograd メタデータなし
                MetalTensor::from_buffer_shared(
                    std::sync::Arc::clone(g.buffer_arc()),
                    g.shape().to_vec(),
                    g.dtype(),
                )
            })
        })
    }

    /// 勾配をゼロクリア
    pub fn zero_grad(&mut self) {
        if let Some(ref mut a) = self.autograd {
            a.grad = None;
        }
    }

    pub(crate) fn accumulate_grad(&mut self, grad: MetalTensor) -> BackendResult<()> {
        if let Some(ref mut meta) = self.autograd {
            // FIX (2026-02-16): Cycle Breaking.
            // Storing execution history in .grad creates a reference cycle:
            // Self -> .grad -> History -> Self
            // We must detach the gradient before storing it.
            // This prevents Double Backward support via .grad access, but fixes the 
            // massive leak for standard training loops.
            let detached_grad = grad.detach();
            
            if let Some(ref mut g) = meta.grad {
                // Add in-place-ish (conceptually) but metal tensors are immutable structs.
                // We ensure the result is also detached/no-history by using detached inputs.
                // g (detached) + detached_grad (detached) -> Result (detached/no history).
                *g = g.add_impl(&detached_grad)?;
            } else {
                meta.grad = Some(detached_grad);
            }
        }
        Ok(())
    }

    /// backward（逆伝播）— ワークリスト方式（スタックオーバーフロー防止）
    /// TensorRef (Arc) ベースで入力テンソルの生存を保証。
    pub fn backward(&mut self) -> BackendResult<()> {
        if !self.requires_grad() {
            return Ok(());
        }

        // 初期勾配: すべて 1
        let ones = MetalTensor::ones(self.shape(), self.dtype())?;
        let self_ptr = self as *mut MetalTensor;

        // ワークリスト: (テンソル生ポインタ, 出力勾配)
        let mut worklist: Vec<(*mut MetalTensor, MetalTensor)> = vec![(self_ptr, ones)];
        // 訪問済みノードを記録（backward後にgrad_fnをクリアするため）
        let mut visited: Vec<*mut MetalTensor> = Vec::new();

        while let Some((tensor_ptr, grad_output)) = worklist.pop() {
            let tensor = unsafe { &mut *tensor_ptr };
            visited.push(tensor_ptr);

            // grad_fn から入力勾配を計算してワークリストに追加
            let propagation = if let Some(meta) = tensor.autograd.as_ref() {
                if let Some(gf) = meta.grad_fn.as_ref() {
                    let grads = gf.backward(&grad_output)?;
                    let inputs = gf.inputs();
                    Some((grads, inputs))
                } else {
                    None
                }
            } else {
                None
            };

            // 勾配を累積
            tensor.accumulate_grad(grad_output)?;

            if let Some((grads, inputs)) = propagation {
                for (input_ref, grad) in inputs.into_iter().zip(grads.into_iter()) {
                    let input_ptr = input_ref.get() as *mut MetalTensor;
                    let input = unsafe { &*input_ptr };
                    if input.requires_grad() {
                        worklist.push((input_ptr, grad));
                    }
                }
            }
        }

        // === 計算グラフ解放（V5.0 メモリ管理） ===
        // ドキュメントの方針「テンソルは最後まで解放しない（Persistent Pool or Rustの自然なDropに任せる）」に従う。
        // ここで手動で grad_fn = None にして参照を絶つと、
        // ダブルフリーなどの UAF(Use-After-Free) を引き起こし、MetalのSegfaultのトリガーとなるため削除。
        /*
        for &ptr in &visited {
            let tensor = unsafe { &mut *ptr };
            if let Some(ref mut meta) = tensor.autograd {
                meta.grad_fn = None;
            }
        }
        */

        Ok(())
    }

    /// 計算グラフから切り離す（データのみの shallow clone）
    pub fn detach(&self) -> MetalTensor {
        self.shallow_clone()
    }
}

impl Drop for MetalTensor {
    fn drop(&mut self) {
        if std::sync::Arc::strong_count(&self.buffer) == 1 {
            pool_release(self.buffer.clone());
        }
    }
}

// GpuTensor トレイトが Send + Sync を要求
unsafe impl Send for MetalTensor {}
unsafe impl Sync for MetalTensor {}
