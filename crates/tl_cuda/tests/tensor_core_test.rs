//! tl_cuda テンソルコア機能テスト
//! テンソル情報取得、データアクセス、型変換、DType、BufferPool、GpuTensorトレイトを検証

use serial_test::serial;
use tl_cuda::{shape_to_bytes, CudaBufferPool, CudaTensor, DType};

// ========== ヘルパー関数 ==========

fn assert_approx_eq(a: f32, b: f32, eps: f32) {
    assert!(
        (a - b).abs() < eps,
        "Expected {} ≈ {}, diff = {}",
        a,
        b,
        (a - b).abs()
    );
}

fn assert_tensor_approx_eq(t: &CudaTensor, expected: &[f32], eps: f32) {
    let data = t.to_vec::<f32>();
    assert_eq!(
        data.len(),
        expected.len(),
        "Length mismatch: {} vs {}",
        data.len(),
        expected.len()
    );
    for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < eps,
            "At index {}: {} ≈ {}, diff = {}",
            i,
            a,
            b,
            (a - b).abs()
        );
    }
}

// =====================================================================
// 1. テンソル情報テスト
// =====================================================================

#[test]
#[serial]
fn test_tensor_shape_info() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.elem_count(), 6);
    assert_eq!(t.dtype(), DType::F32);
}

#[test]
#[serial]
fn test_tensor_1d_shape() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.elem_count(), 3);
}

#[test]
#[serial]
fn test_tensor_3d_shape() {
    let t = CudaTensor::zeros(&[2, 3, 4], DType::F32);
    assert_eq!(t.shape(), &[2, 3, 4]);
    assert_eq!(t.elem_count(), 24);
}

#[test]
#[serial]
fn test_tensor_4d_shape() {
    let t = CudaTensor::zeros(&[1, 3, 4, 4], DType::F32);
    assert_eq!(t.shape(), &[1, 3, 4, 4]);
    assert_eq!(t.elem_count(), 48);
}

// =====================================================================
// 2. データアクセステスト
// =====================================================================

#[test]
#[serial]
fn test_to_vec_f32() {
    let data = vec![1.5f32, 2.5, 3.5, 4.5];
    let t = CudaTensor::from_slice(&data, &[4], DType::F32);
    let result: Vec<f32> = t.to_vec();
    assert_eq!(result, data);
}

#[test]
#[serial]
fn test_to_vec_roundtrip() {
    let original = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
    let t = CudaTensor::from_slice(&original, &[2, 3], DType::F32);
    let roundtrip: Vec<f32> = t.to_vec();
    for (a, b) in original.iter().zip(roundtrip.iter()) {
        assert_approx_eq(*a, *b, 1e-6);
    }
}

#[test]
#[serial]
fn test_from_slice_i64() {
    let data = vec![1i64, 2, 3, 4];
    let t = CudaTensor::from_slice(&data, &[4], DType::I64);
    assert_eq!(t.shape(), &[4]);
    assert_eq!(t.dtype(), DType::I64);
}

// =====================================================================
// 3. DType テスト
// =====================================================================

#[test]
fn test_dtype_size_in_bytes() {
    assert_eq!(DType::F32.size_in_bytes(), 4);
    assert_eq!(DType::F16.size_in_bytes(), 2);
    assert_eq!(DType::I32.size_in_bytes(), 4);
    assert_eq!(DType::I64.size_in_bytes(), 8);
    assert_eq!(DType::U8.size_in_bytes(), 1);
}

#[test]
fn test_dtype_equality() {
    assert_eq!(DType::F32, DType::F32);
    assert_ne!(DType::F32, DType::I32);
    assert_ne!(DType::F16, DType::F32);
}

#[test]
fn test_shape_to_bytes() {
    assert_eq!(shape_to_bytes(&[2, 3], DType::F32), 24); // 6 * 4
    assert_eq!(shape_to_bytes(&[4], DType::F16), 8); // 4 * 2
    assert_eq!(shape_to_bytes(&[10], DType::I64), 80); // 10 * 8
    assert_eq!(shape_to_bytes(&[3, 3], DType::U8), 9); // 9 * 1
}

#[test]
fn test_shape_to_bytes_empty() {
    // 空の形状の場合: iter().product() = 1 (Rust の空積)
    // 1 * sizeof(F32) = 4
    assert_eq!(shape_to_bytes(&[], DType::F32), 4);
}

// =====================================================================
// 4. BufferPool テスト
// =====================================================================

#[test]
fn test_buffer_pool_new() {
    let pool = CudaBufferPool::new();
    assert_eq!(pool.free_count(), 0);
    assert_eq!(pool.hits, 0);
    assert_eq!(pool.misses, 0);
}

#[test]
fn test_buffer_pool_hit_rate_empty() {
    let pool = CudaBufferPool::new();
    assert_eq!(pool.hit_rate(), 0.0);
}

#[test]
#[serial]
fn test_buffer_pool_acquire_release_cycle() {
    // テンソルを作成して正常動作を確認
    let t1 = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], DType::F32);
    let t2 = CudaTensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[4], DType::F32);

    // テンソルを drop → バッファがプールに返却される
    drop(t1);
    drop(t2);

    // テンソル作成が正常に動作することの確認（プールから再利用される可能性）
    let t3 = CudaTensor::from_slice(&[9.0f32, 10.0, 11.0, 12.0], &[4], DType::F32);
    assert_eq!(t3.shape(), &[4]);
}

// =====================================================================
// 5. GpuTensor トレイトテスト
// =====================================================================

#[test]
#[serial]
fn test_gpu_tensor_from_slice_f32() {
    use tl_backend::GpuTensor;
    let t = CudaTensor::from_slice_f32(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let data = t.to_vec_f32();
    assert_eq!(data.len(), 3);
    assert_approx_eq(data[0], 1.0, 1e-5);
    assert_approx_eq(data[1], 2.0, 1e-5);
    assert_approx_eq(data[2], 3.0, 1e-5);
}

#[test]
#[serial]
fn test_gpu_tensor_from_slice_i64() {
    use tl_backend::GpuTensor;
    let t = CudaTensor::from_slice_i64(&[10, 20, 30], &[3]).unwrap();
    let data = t.to_vec_i64();
    assert_eq!(data.len(), 3);
}

#[test]
#[serial]
fn test_gpu_tensor_zeros() {
    use tl_backend::DType as BackendDType;
    use tl_backend::GpuTensor;
    let t = <CudaTensor as GpuTensor>::zeros(&[3, 3], BackendDType::F32).unwrap();
    let data = t.to_vec_f32();
    assert_eq!(data.len(), 9);
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
#[serial]
fn test_gpu_tensor_ones() {
    use tl_backend::DType as BackendDType;
    use tl_backend::GpuTensor;
    let t = <CudaTensor as GpuTensor>::ones(&[2, 2], BackendDType::F32).unwrap();
    let data = t.to_vec_f32();
    assert_eq!(data.len(), 4);
    assert!(data.iter().all(|&x| x == 1.0));
}

#[test]
#[serial]
fn test_gpu_tensor_randn() {
    use tl_backend::DType as BackendDType;
    use tl_backend::GpuTensor;
    let t = <CudaTensor as GpuTensor>::randn(&[100], BackendDType::F32).unwrap();
    let data = t.to_vec_f32();
    assert_eq!(data.len(), 100);
    // 乱数なので値が有限であることのみ確認
    assert!(data.iter().all(|x| x.is_finite()));
}

#[test]
#[serial]
fn test_gpu_tensor_arange() {
    use tl_backend::DType as BackendDType;
    use tl_backend::GpuTensor;
    let t = <CudaTensor as GpuTensor>::arange(0, 5, BackendDType::F32).unwrap();
    let data = t.to_vec_f32();
    assert_eq!(data.len(), 5);
    for i in 0..5 {
        assert_approx_eq(data[i], i as f32, 1e-5);
    }
}

#[test]
#[serial]
fn test_gpu_tensor_clone_data() {
    use tl_backend::GpuTensor;
    let t = CudaTensor::from_slice_f32(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let cloned = t.clone_data().unwrap();
    let data = cloned.to_vec_f32();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[test]
#[serial]
fn test_gpu_tensor_shape_dtype() {
    use tl_backend::DType as BackendDType;
    use tl_backend::GpuTensor;
    let t = CudaTensor::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(GpuTensor::shape(&t), &[2, 2]);
    assert_eq!(GpuTensor::dtype(&t), BackendDType::F32);
}

// =====================================================================
// 6. テンソル作成バリエーションテスト
// =====================================================================

#[test]
#[serial]
fn test_uninit_tensor() {
    let t = CudaTensor::uninit(&[3, 3], DType::F32);
    assert_eq!(t.shape(), &[3, 3]);
    assert_eq!(t.elem_count(), 9);
    // 未初期化なので値の内容は不定だが、アクセスはできる
    let data: Vec<f32> = t.to_vec();
    assert_eq!(data.len(), 9);
}

#[test]
#[serial]
fn test_from_buffer_shared() {
    // from_buffer_shared は CUDA ではスタブ
    // 引数なしのスタブなので、代わりに同じデータを別形状で作成
    let shared = CudaTensor::from_buffer_shared(vec![3, 2], DType::F32);
    assert_eq!(shared.shape(), &[3, 2]);
}

// =====================================================================
// 7. Autograd メタデータテスト
// =====================================================================

#[test]
#[serial]
fn test_requires_grad_default() {
    let t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    assert!(!t.requires_grad(), "Default should not require grad");
}

#[test]
#[serial]
fn test_enable_grad() {
    let mut t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    t.enable_grad();
    assert!(t.requires_grad(), "Should require grad after enable_grad");
}

#[test]
#[serial]
fn test_detach() {
    let mut t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    t.enable_grad();
    let detached = t.detach();
    // detach はデータのみの shallow clone を返す
    // requires_grad の状態は実装依存
    assert_tensor_approx_eq(&detached, &[1.0, 2.0, 3.0], 1e-5);
    assert_eq!(detached.shape(), &[3]);
}

#[test]
#[serial]
fn test_zero_grad() {
    let mut t = CudaTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    t.enable_grad();
    t.zero_grad();
    // zero_grad 後は勾配が None であること
    assert!(
        t.get_grad().is_none(),
        "Grad should be None after zero_grad"
    );
}
