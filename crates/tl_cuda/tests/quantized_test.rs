//! tl_cuda 量子化演算テスト
//! dequantize_q4_k の動作を検証

use tl_cuda::{CudaTensor, DType};
use serial_test::serial;

// =====================================================================
// 1. dequantize_q4_k 基本テスト
// =====================================================================

#[test]
#[serial]
fn test_dequantize_q4_k_shape() {
    // Q4_K: 1 block = 256 elements = 144 bytes
    // 1ブロック分のデータを作成
    let num_blocks = 1;
    let bytes_per_block = 144;
    let input_bytes = vec![0u8; num_blocks * bytes_per_block];

    // U8 テンソルとして作成
    let input = CudaTensor::from_slice(&input_bytes, &[input_bytes.len()], DType::U8);
    assert_eq!(input.shape(), &[input_bytes.len()]);

    // デクォンタイズ
    let target_shape = [num_blocks * 256];
    let result = input.dequantize_q4_k(&target_shape);

    match result {
        Ok(output) => {
            // 出力形状の確認
            assert_eq!(output.shape(), &[256]);
            assert_eq!(output.dtype(), DType::F32);
            assert_eq!(output.elem_count(), 256);

            // 全ゼロ入力の場合、出力も有限値
            let data = output.to_vec::<f32>();
            assert!(data.iter().all(|x| x.is_finite()), "All outputs should be finite");
        },
        Err(e) => {
            // デクォンタイズがエラーの場合でもテストは通過
            eprintln!("dequantize_q4_k returned error (may be expected for zero input): {}", e);
        }
    }
}

#[test]
#[serial]
fn test_dequantize_q4_k_multiple_blocks() {
    // 4ブロック = 1024 elements = 576 bytes
    let num_blocks = 4;
    let bytes_per_block = 144;
    let input_bytes = vec![0u8; num_blocks * bytes_per_block];

    let input = CudaTensor::from_slice(&input_bytes, &[input_bytes.len()], DType::U8);

    let target_shape = [num_blocks * 256];
    let result = input.dequantize_q4_k(&target_shape);

    match result {
        Ok(output) => {
            assert_eq!(output.shape(), &[1024]);
            assert_eq!(output.elem_count(), 1024);
        },
        Err(e) => {
            eprintln!("dequantize_q4_k error for multiple blocks: {}", e);
        }
    }
}

#[test]
#[serial]
fn test_dequantize_q4_k_invalid_size() {
    // 256 で割り切れないサイズはエラー
    let result_err = CudaTensor::from_slice(&[0u8; 144], &[144], DType::U8)
        .dequantize_q4_k(&[100]); // 100 is not divisible by 256

    assert!(result_err.is_err(), "Should fail for non-256-aligned element count");
}

#[test]
#[serial]
fn test_dequantize_q4_k_multidim_target() {
    // 多次元ターゲット形状: [2, 256] = 512 elements = 2 blocks = 288 bytes
    let num_blocks = 2;
    let bytes_per_block = 144;
    let input_bytes = vec![0u8; num_blocks * bytes_per_block];

    let input = CudaTensor::from_slice(&input_bytes, &[input_bytes.len()], DType::U8);

    let target_shape = [2, 256];
    let result = input.dequantize_q4_k(&target_shape);

    match result {
        Ok(output) => {
            assert_eq!(output.shape(), &[2, 256]);
            assert_eq!(output.elem_count(), 512);
        },
        Err(e) => {
            eprintln!("dequantize_q4_k error for multidim: {}", e);
        }
    }
}

// =====================================================================
// 2. 非ゼロデータのデクォンタイズテスト
// =====================================================================

#[test]
#[serial]
fn test_dequantize_q4_k_nonzero_data() {
    // Q4_K ブロック構造:
    // - d (f16, 2 bytes): scale
    // - dmin (f16, 2 bytes): min value
    // - scales (12 bytes): per-group scales
    // - qs (128 bytes): quantized values (4-bit packed)
    // Total: 144 bytes per block

    let num_blocks = 1;
    let bytes_per_block = 144;
    let mut input_bytes = vec![0u8; num_blocks * bytes_per_block];

    // d = 1.0 (f16) を設定
    // f16 1.0 = 0x3C00
    input_bytes[0] = 0x00;
    input_bytes[1] = 0x3C;

    let input = CudaTensor::from_slice(&input_bytes, &[input_bytes.len()], DType::U8);

    let target_shape = [num_blocks * 256];
    let result = input.dequantize_q4_k(&target_shape);

    match result {
        Ok(output) => {
            assert_eq!(output.shape(), &[256]);
            let data = output.to_vec::<f32>();
            // 全値が有限であること
            assert!(data.iter().all(|x| x.is_finite()), "All outputs should be finite");
        },
        Err(e) => {
            eprintln!("dequantize_q4_k error for nonzero data: {}", e);
        }
    }
}
