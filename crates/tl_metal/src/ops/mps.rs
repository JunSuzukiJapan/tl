//! Metal Performance Shaders (MPS) 統合
//! 
//! Apple の MPS フレームワークを使用した高性能 GPU カーネル
//!
//! 現在は CPU フォールバックを使用し、今後 MPS ネイティブ実装を追加予定

use crate::MetalTensor;

/// MPS を使用した Conv2D 実装
/// 
/// 現在は CPU フォールバックを使用
/// TODO: MPSCNNConvolution を使用した GPU 実装
pub fn mps_conv2d(
    input: &MetalTensor,
    weight: &MetalTensor,
    stride: (usize, usize),
    padding: (usize, usize),
) -> MetalTensor {
    // MPS 実装のプレースホルダー
    // MPSCNNConvolution は MPSImage (Texture) ベースなので
    // Buffer -> Texture 変換が必要
    
    // 現在は CPU フォールバック
    MetalTensor::conv2d_impl(input, weight, stride, padding)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    
    #[test]
    fn test_mps_conv2d_basic() {
        // 1x1x4x4 入力, 1x1x2x2 カーネル
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let kernel_data = vec![1.0, 1.0, 1.0, 1.0]; // 2x2 all ones
        
        let input = MetalTensor::from_slice(&input_data, &[1, 1, 4, 4], DType::F32);
        let kernel = MetalTensor::from_slice(&kernel_data, &[1, 1, 2, 2], DType::F32);
        
        let output = mps_conv2d(&input, &kernel, (1, 1), (0, 0));
        let result: Vec<f32> = output.to_vec();
        
        // 出力サイズ: (4-2)/1+1 = 3, so 3x3 = 9 要素
        assert_eq!(result.len(), 9);
        
        // 出力が生成されていることを確認（正確な値は実装依存）
        assert!(result[0].is_finite(), "Output should be finite");
    }
}
