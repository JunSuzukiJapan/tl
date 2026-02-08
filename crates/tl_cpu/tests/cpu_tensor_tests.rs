//! tl_cpu クレート 網羅テスト

use tl_cpu::tensor::CpuTensor;
use tl_cpu::DType;

// ========== ヘルパー ==========

fn assert_approx(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < eps, "index {}: {} != {} (eps={})", i, x, y, eps);
    }
}

fn t(data: &[f32], shape: &[usize]) -> CpuTensor {
    CpuTensor::from_slice(data, shape, DType::F32)
}

// ========== コンストラクタ ==========

#[test]
fn test_zeros() {
    let z = CpuTensor::zeros(&[2, 3], DType::F32);
    assert_eq!(z.shape(), &[2, 3]);
    assert_eq!(z.data_f32(), &[0.0; 6]);
}

#[test]
fn test_ones() {
    let o = CpuTensor::ones(&[3], DType::F32);
    assert_eq!(o.shape(), &[3]);
    assert_eq!(o.data_f32(), &[1.0, 1.0, 1.0]);
}

#[test]
fn test_from_slice() {
    let t = CpuTensor::from_slice(&[1.0, 2.0, 3.0], &[3], DType::F32);
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.data_f32(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_from_slice_i64_data() {
    let t = CpuTensor::from_slice_i64_data(&[10, 20, 30], &[3], DType::I64);
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.data_f32(), &[10.0, 20.0, 30.0]);
    let i64_vec: Vec<i64> = t.to_vec();
    assert_eq!(i64_vec, vec![10i64, 20, 30]);
}

#[test]
fn test_randn() {
    let r = CpuTensor::randn(&[100], DType::F32);
    assert_eq!(r.shape(), &[100]);
    assert_eq!(r.elem_count(), 100);
    // 平均が極端でないことを確認
    let mean: f32 = r.data_f32().iter().sum::<f32>() / 100.0;
    assert!(mean.abs() < 1.0, "randn mean should be near 0, got {}", mean);
}

#[test]
fn test_arange() {
    let a = CpuTensor::arange(0, 5, DType::F32);
    assert_eq!(a.shape(), &[5]);
    assert_eq!(a.data_f32(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
}

// ========== アクセサ ==========

#[test]
fn test_accessors() {
    let t = CpuTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.elem_count(), 6);
    let v: Vec<f32> = t.to_vec();
    assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_shallow_clone() {
    let a = t(&[1.0, 2.0], &[2]);
    let b = a.shallow_clone();
    assert_eq!(b.data_f32(), a.data_f32());
    assert_eq!(b.shape(), a.shape());
    assert!(!b.requires_grad()); // autograd はコピーしない
}

// ========== 基本演算 ==========

#[test]
fn test_add() {
    let a = t(&[1.0, 2.0, 3.0], &[3]);
    let b = t(&[4.0, 5.0, 6.0], &[3]);
    let c = a.add_impl(&b);
    assert_eq!(c.data_f32(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_sub() {
    let a = t(&[5.0, 3.0, 1.0], &[3]);
    let b = t(&[1.0, 2.0, 3.0], &[3]);
    let c = a.sub_impl(&b);
    assert_eq!(c.data_f32(), &[4.0, 1.0, -2.0]);
}

#[test]
fn test_mul() {
    let a = t(&[2.0, 3.0], &[2]);
    let b = t(&[4.0, 5.0], &[2]);
    let c = a.mul_impl(&b);
    assert_eq!(c.data_f32(), &[8.0, 15.0]);
}

#[test]
fn test_div() {
    let a = t(&[10.0, 9.0], &[2]);
    let b = t(&[2.0, 3.0], &[2]);
    let c = a.div_impl(&b);
    assert_eq!(c.data_f32(), &[5.0, 3.0]);
}

#[test]
fn test_pow() {
    let a = t(&[2.0, 3.0], &[2]);
    let b = t(&[3.0, 2.0], &[2]);
    let c = a.pow_impl(&b);
    assert_approx(c.data_f32(), &[8.0, 9.0], 1e-5);
}

#[test]
fn test_rem() {
    let a = t(&[7.0, 10.0], &[2]);
    let b = t(&[3.0, 4.0], &[2]);
    let c = a.rem_impl(&b);
    assert_approx(c.data_f32(), &[1.0, 2.0], 1e-5);
}

#[test]
fn test_neg() {
    let a = t(&[1.0, -2.0, 3.0], &[3]);
    let c = a.neg_impl();
    assert_eq!(c.data_f32(), &[-1.0, 2.0, -3.0]);
}

#[test]
fn test_abs() {
    let a = t(&[-1.0, 2.0, -3.0], &[3]);
    let c = a.abs_impl();
    assert_eq!(c.data_f32(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_broadcast_scalar() {
    let a = t(&[1.0, 2.0, 3.0], &[3]);
    let b = t(&[10.0], &[1]);
    let c = a.add_impl(&b);
    assert_eq!(c.data_f32(), &[11.0, 12.0, 13.0]);
}

// ========== スカラー演算 ==========

#[test]
fn test_add_scalar() {
    let a = t(&[1.0, 2.0], &[2]);
    let c = a.add_scalar_impl(10.0);
    assert_eq!(c.data_f32(), &[11.0, 12.0]);
}

#[test]
fn test_sub_scalar() {
    let a = t(&[5.0, 3.0], &[2]);
    let c = a.sub_scalar_impl(2.0);
    assert_eq!(c.data_f32(), &[3.0, 1.0]);
}

#[test]
fn test_mul_scalar() {
    let a = t(&[2.0, 3.0], &[2]);
    let c = a.mul_scalar_impl(3.0);
    assert_eq!(c.data_f32(), &[6.0, 9.0]);
}

#[test]
fn test_div_scalar() {
    let a = t(&[10.0, 6.0], &[2]);
    let c = a.div_scalar_impl(2.0);
    assert_eq!(c.data_f32(), &[5.0, 3.0]);
}

#[test]
fn test_clamp() {
    let a = t(&[-1.0, 0.5, 2.0, 5.0], &[4]);
    let c = a.clamp_impl(0.0, 3.0);
    assert_eq!(c.data_f32(), &[0.0, 0.5, 2.0, 3.0]);
}

// ========== 比較演算 ==========

#[test]
fn test_comparisons() {
    let a = t(&[1.0, 2.0, 3.0], &[3]);
    let b = t(&[2.0, 2.0, 1.0], &[3]);

    assert_eq!(a.eq_impl(&b).data_f32(), &[0.0, 1.0, 0.0]);
    assert_eq!(a.neq_impl(&b).data_f32(), &[1.0, 0.0, 1.0]);
    assert_eq!(a.lt_impl(&b).data_f32(), &[1.0, 0.0, 0.0]);
    assert_eq!(a.le_impl(&b).data_f32(), &[1.0, 1.0, 0.0]);
    assert_eq!(a.gt_impl(&b).data_f32(), &[0.0, 0.0, 1.0]);
    assert_eq!(a.ge_impl(&b).data_f32(), &[0.0, 1.0, 1.0]);
}

// ========== 数学関数 ==========

#[test]
fn test_exp() {
    let a = t(&[0.0, 1.0], &[2]);
    let c = a.exp_impl();
    assert_approx(c.data_f32(), &[1.0, std::f32::consts::E], 1e-5);
}

#[test]
fn test_log() {
    let a = t(&[1.0, std::f32::consts::E], &[2]);
    let c = a.log_impl();
    assert_approx(c.data_f32(), &[0.0, 1.0], 1e-5);
}

#[test]
fn test_sqrt() {
    let a = t(&[4.0, 9.0, 16.0], &[3]);
    let c = a.sqrt_impl();
    assert_approx(c.data_f32(), &[2.0, 3.0, 4.0], 1e-5);
}

#[test]
fn test_sin_cos() {
    let a = t(&[0.0, std::f32::consts::FRAC_PI_2], &[2]);
    assert_approx(a.sin_impl().data_f32(), &[0.0, 1.0], 1e-5);
    assert_approx(a.cos_impl().data_f32(), &[1.0, 0.0], 1e-5);
}

#[test]
fn test_tan() {
    let a = t(&[0.0], &[1]);
    assert_approx(a.tan_impl().data_f32(), &[0.0], 1e-5);
}

#[test]
fn test_tanh() {
    let a = t(&[0.0, 1.0], &[2]);
    let c = a.tanh_impl();
    assert_approx(c.data_f32(), &[0.0, 0.7615942], 1e-5);
}

#[test]
fn test_sigmoid() {
    let a = t(&[0.0], &[1]);
    let c = a.sigmoid_impl();
    assert_approx(c.data_f32(), &[0.5], 1e-5);
}

#[test]
fn test_relu() {
    let a = t(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    assert_eq!(a.relu_impl().data_f32(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_gelu() {
    let a = t(&[0.0, 1.0, -1.0], &[3]);
    let c = a.gelu_impl();
    // GELU(0) = 0
    assert_approx(&c.data_f32()[0..1], &[0.0], 1e-4);
    // GELU(1) ≈ 0.914 (tanh 近似式による値)
    assert!((c.data_f32()[1] - 0.914).abs() < 0.02, "GELU(1) = {}", c.data_f32()[1]);
    // GELU(-1) < 0
    assert!(c.data_f32()[2] < 0.0);
}

// ========== リダクション ==========

#[test]
fn test_sumall() {
    let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_approx(&[a.sumall_impl()], &[10.0], 1e-5);
}

#[test]
fn test_mean_all() {
    let a = t(&[2.0, 4.0, 6.0, 8.0], &[4]);
    assert_approx(&[a.mean_all_impl()], &[5.0], 1e-5);
}

#[test]
fn test_sum_axis0() {
    let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = a.sum_impl(0);
    assert_eq!(c.shape(), &[3]);
    assert_approx(c.data_f32(), &[5.0, 7.0, 9.0], 1e-5);
}

#[test]
fn test_sum_axis1() {
    let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = a.sum_impl(1);
    assert_eq!(c.shape(), &[2]);
    assert_approx(c.data_f32(), &[6.0, 15.0], 1e-5);
}

#[test]
fn test_max_axis() {
    let a = t(&[1.0, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]);
    let c = a.max_impl(1);
    assert_eq!(c.shape(), &[2]);
    assert_approx(c.data_f32(), &[5.0, 6.0], 1e-5);
}

#[test]
fn test_min_axis() {
    let a = t(&[3.0, 1.0, 5.0, 4.0, 2.0, 6.0], &[2, 3]);
    let c = a.min_impl(1);
    assert_eq!(c.shape(), &[2]);
    assert_approx(c.data_f32(), &[1.0, 2.0], 1e-5);
}

#[test]
fn test_argmax() {
    let a = t(&[1.0, 3.0, 2.0, 6.0, 4.0, 5.0], &[2, 3]);
    let c = a.argmax_impl(1);
    assert_eq!(c.shape(), &[2]);
    assert_approx(c.data_f32(), &[1.0, 0.0], 1e-5);
}

#[test]
fn test_argmax_all() {
    let a = t(&[1.0, 5.0, 3.0, 2.0], &[4]);
    assert_eq!(a.argmax_all_impl(), 1);
}

#[test]
fn test_argmin() {
    let a = t(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3]);
    let c = a.argmin_impl(1);
    assert_eq!(c.shape(), &[2]);
    assert_approx(c.data_f32(), &[1.0, 1.0], 1e-5);
}

#[test]
fn test_mean_axis() {
    let a = t(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    let c = a.mean_impl(1);
    assert_eq!(c.shape(), &[2]);
    assert_approx(c.data_f32(), &[3.0, 7.0], 1e-5);
}

// ========== 形状操作 ==========

#[test]
fn test_reshape() {
    let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = a.reshape_impl(&[3, 2]);
    assert_eq!(b.shape(), &[3, 2]);
    assert_eq!(b.data_f32(), a.data_f32());
}

#[test]
fn test_transpose_2d() {
    let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = a.transpose_impl(0, 1);
    assert_eq!(b.shape(), &[3, 2]);
    assert_approx(b.data_f32(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-5);
}

#[test]
fn test_squeeze() {
    let a = t(&[1.0, 2.0, 3.0], &[1, 3]);
    let b = a.squeeze_impl(0);
    assert_eq!(b.shape(), &[3]);
}

#[test]
fn test_unsqueeze() {
    let a = t(&[1.0, 2.0, 3.0], &[3]);
    let b = a.unsqueeze_impl(0);
    assert_eq!(b.shape(), &[1, 3]);
    let c = a.unsqueeze_impl(1);
    assert_eq!(c.shape(), &[3, 1]);
}

#[test]
fn test_broadcast_to() {
    let a = t(&[1.0, 2.0], &[2]);
    let b = a.broadcast_to_impl(&[3, 2]);
    assert_eq!(b.shape(), &[3, 2]);
    assert_eq!(b.data_f32(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn test_narrow_and_slice() {
    let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    let b = a.narrow_impl(0, 1, 3);
    assert_eq!(b.shape(), &[3]);
    assert_eq!(b.data_f32(), &[2.0, 3.0, 4.0]);

    let c = a.slice_impl(0, 1, 3);
    assert_eq!(c.shape(), &[3]);
    assert_eq!(c.data_f32(), &[2.0, 3.0, 4.0]);
}

#[test]
fn test_contiguous() {
    let a = t(&[1.0, 2.0], &[2]);
    let b = a.contiguous_impl();
    assert_eq!(b.data_f32(), a.data_f32());
}

#[test]
fn test_cat() {
    let a = t(&[1.0, 2.0], &[2]);
    let b = t(&[3.0, 4.0, 5.0], &[3]);
    let c = CpuTensor::cat_impl(&[&a, &b], 0);
    assert_eq!(c.shape(), &[5]);
    assert_eq!(c.data_f32(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_cat_2d() {
    let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c = CpuTensor::cat_impl(&[&a, &b], 0);
    assert_eq!(c.shape(), &[4, 2]);
    assert_eq!(c.data_f32(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

// ========== 特殊演算 ==========

#[test]
fn test_softmax() {
    let a = t(&[1.0, 2.0, 3.0], &[1, 3]);
    let s = a.softmax_impl(1);
    assert_eq!(s.shape(), &[1, 3]);
    let sum: f32 = s.data_f32().iter().sum();
    assert_approx(&[sum], &[1.0], 1e-5);
    // softmax is monotonically increasing
    assert!(s.data_f32()[0] < s.data_f32()[1]);
    assert!(s.data_f32()[1] < s.data_f32()[2]);
}

#[test]
fn test_embedding() {
    // weights: [3, 2] (3 tokens, embed_dim=2)
    let w = t(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2]);
    // indices: [2] → tokens 0, 2
    let idx = t(&[0.0, 2.0], &[2]);
    let e = w.embedding_impl(&idx);
    assert_eq!(e.shape(), &[2, 2]);
    assert_approx(e.data_f32(), &[0.1, 0.2, 0.5, 0.6], 1e-5);
}

#[test]
fn test_tril() {
    let a = t(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[3, 3]);
    let c = a.tril_impl(0);
    assert_eq!(c.shape(), &[3, 3]);
    assert_approx(c.data_f32(), &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], 1e-5);
}

#[test]
fn test_tril_diagonal1() {
    let a = t(&[1.0; 9], &[3, 3]);
    let c = a.tril_impl(1);
    assert_approx(c.data_f32(), &[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1e-5);
}

#[test]
fn test_cross_entropy() {
    // input: [1, 3] logits, target: [1] class index
    let input = t(&[2.0, 1.0, 0.1], &[1, 3]);
    let target = t(&[0.0], &[1]); // target class = 0
    let loss = input.cross_entropy_impl(&target);
    assert_eq!(loss.shape(), &[1]);
    // loss should be > 0
    assert!(loss.data_f32()[0] > 0.0);
}

#[test]
fn test_matmul_2d() {
    // [2, 3] x [3, 2] = [2, 2]
    let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = t(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
    let c = a.matmul_impl(&b);
    assert_eq!(c.shape(), &[2, 2]);
    // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    assert_approx(c.data_f32(), &[58.0, 64.0, 139.0, 154.0], 1e-4);
}

#[test]
fn test_repeat_interleave() {
    let a = t(&[1.0, 2.0, 3.0], &[3]);
    let c = a.repeat_interleave_impl(2, 0);
    assert_eq!(c.shape(), &[6]);
    assert_eq!(c.data_f32(), &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
}

#[test]
fn test_index_select() {
    let a = t(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5]);
    let idx = t(&[0.0, 2.0, 4.0], &[3]);
    let c = a.index_select_impl(0, &idx);
    assert_eq!(c.shape(), &[3]);
    assert_eq!(c.data_f32(), &[10.0, 30.0, 50.0]);
}

#[test]
fn test_where_cond() {
    let cond = t(&[1.0, 0.0, 1.0], &[3]);
    let x = t(&[10.0, 20.0, 30.0], &[3]);
    let y = t(&[100.0, 200.0, 300.0], &[3]);
    let c = CpuTensor::where_cond_impl(&cond, &x, &y);
    assert_eq!(c.data_f32(), &[10.0, 200.0, 30.0]);
}

// ========== NN 演算 ==========

#[test]
fn test_conv2d_basic() {
    // 1x1x3x3 input, 1x1x2x2 kernel
    let input = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[1, 1, 3, 3]);
    let weight = t(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
    let c = input.conv2d_impl(&weight, (1, 1), (0, 0));
    assert_eq!(c.shape(), &[1, 1, 2, 2]);
    // out[0,0]=1*1+2*0+4*0+5*1=6, out[0,1]=2*1+3*0+5*0+6*1=8
    // out[1,0]=4*1+5*0+7*0+8*1=12, out[1,1]=5*1+6*0+8*0+9*1=14
    assert_approx(c.data_f32(), &[6.0, 8.0, 12.0, 14.0], 1e-5);
}

#[test]
fn test_conv2d_stride() {
    // 1x1x4x4 input, 1x1x2x2 kernel, stride=2
    let input = t(&[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ], &[1, 1, 4, 4]);
    let weight = t(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);
    let c = input.conv2d_impl(&weight, (2, 2), (0, 0));
    assert_eq!(c.shape(), &[1, 1, 2, 2]);
    // out[0,0]=1+2+5+6=14, out[0,1]=3+4+7+8=22
    // out[1,0]=9+10+13+14=46, out[1,1]=11+12+15+16=54
    assert_approx(c.data_f32(), &[14.0, 22.0, 46.0, 54.0], 1e-5);
}

#[test]
fn test_conv2d_padding() {
    // 1x1x2x2 input, 1x1x3x3 kernel, padding=1 → output same size
    let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let weight = t(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], &[1, 1, 3, 3]);
    let c = input.conv2d_impl(&weight, (1, 1), (1, 1));
    assert_eq!(c.shape(), &[1, 1, 2, 2]);
    // identity kernel → same as input
    assert_approx(c.data_f32(), &[1.0, 2.0, 3.0, 4.0], 1e-5);
}

#[test]
fn test_layer_norm() {
    let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let gamma = t(&[1.0, 1.0], &[2]);
    let beta = t(&[0.0, 0.0], &[2]);
    let c = a.layer_norm_impl(&gamma, &beta, 1e-5);
    assert_eq!(c.shape(), &[2, 2]);
    // 各行は独立に正規化される (平均≈0)
    let row1_mean = (c.data_f32()[0] + c.data_f32()[1]) / 2.0;
    let row2_mean = (c.data_f32()[2] + c.data_f32()[3]) / 2.0;
    assert!(row1_mean.abs() < 1e-4, "row1 mean = {}", row1_mean);
    assert!(row2_mean.abs() < 1e-4, "row2 mean = {}", row2_mean);
    // [1,2] → [-1, 1] (正規化)
    assert!(c.data_f32()[0] < 0.0);
    assert!(c.data_f32()[1] > 0.0);
}

#[test]
fn test_batch_norm() {
    // [2, 2, 1, 1] → batch=2, channels=2
    let a = t(&[1.0, 3.0, 5.0, 7.0], &[2, 2, 1, 1]);
    let gamma = t(&[1.0, 1.0], &[2]);
    let beta = t(&[0.0, 0.0], &[2]);
    let mean = t(&[3.0, 5.0], &[2]); // channel means: (1+5)/2=3, (3+7)/2=5
    let var = t(&[4.0, 4.0], &[2]);  // channel var: ((1-3)^2+(5-3)^2)/2=4
    let c = a.batch_norm_impl(&gamma, &beta, &mean, &var, 1e-5);
    assert_eq!(c.shape(), &[2, 2, 1, 1]);
    // channel 0: (1-3)/sqrt(4)=-1, (5-3)/sqrt(4)=1
    // channel 1: (3-5)/sqrt(4)=-1, (7-5)/sqrt(4)=1
    assert_approx(c.data_f32(), &[-1.0, -1.0, 1.0, 1.0], 1e-4);
}

#[test]
fn test_max_pool2d() {
    // 1x1x4x4 → kernel=2x2, stride=2 → 1x1x2x2
    let input = t(&[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ], &[1, 1, 4, 4]);
    let c = input.max_pool2d_impl((2, 2), (2, 2));
    assert_eq!(c.shape(), &[1, 1, 2, 2]);
    assert_approx(c.data_f32(), &[6.0, 8.0, 14.0, 16.0], 1e-5);
}

#[test]
fn test_avg_pool2d() {
    // 1x1x4x4 → kernel=2x2, stride=2 → 1x1x2x2
    let input = t(&[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ], &[1, 1, 4, 4]);
    let c = input.avg_pool2d_impl((2, 2), (2, 2));
    assert_eq!(c.shape(), &[1, 1, 2, 2]);
    // avg([1,2,5,6])=3.5, avg([3,4,7,8])=5.5, ...
    assert_approx(c.data_f32(), &[3.5, 5.5, 11.5, 13.5], 1e-5);
}

// ========== Autograd ==========

#[test]
fn test_autograd_basic() {
    let mut a = t(&[2.0, 3.0], &[2]);
    a.enable_grad();
    assert!(a.requires_grad());
    a.zero_grad();
    assert!(a.get_grad().is_none());
}

#[test]
fn test_detach() {
    let mut a = t(&[1.0], &[1]);
    a.enable_grad();
    let b = a.detach();
    assert!(!b.requires_grad());
}

// ========== FFI ラウンドトリップ ==========

mod ffi_tests {
    use tl_cpu::ffi;

    #[test]
    fn test_ffi_tensor_lifecycle() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let shape = [2usize, 2];
        let t = ffi::tl_cpu_tensor_new(data.as_ptr(), 2, shape.as_ptr());
        assert!(!t.is_null());

        // dim
        assert_eq!(ffi::tl_cpu_tensor_dim(t, 0), 2);
        assert_eq!(ffi::tl_cpu_tensor_dim(t, 1), 2);
        assert_eq!(ffi::tl_cpu_tensor_len(t), 4);

        // clone
        let t2 = ffi::tl_cpu_tensor_clone(t);
        assert!(!t2.is_null());
        assert_eq!(ffi::tl_cpu_tensor_len(t2), 4);

        // free
        ffi::tl_cpu_tensor_free(t);
        ffi::tl_cpu_tensor_free(t2);
    }

    #[test]
    fn test_ffi_arithmetic() {
        let a_data = [1.0f32, 2.0, 3.0];
        let b_data = [4.0f32, 5.0, 6.0];
        let shape = [3usize];
        let a = ffi::tl_cpu_tensor_new(a_data.as_ptr(), 1, shape.as_ptr());
        let b = ffi::tl_cpu_tensor_new(b_data.as_ptr(), 1, shape.as_ptr());

        let c = ffi::tl_cpu_tensor_add(a, b);
        assert!(!c.is_null());
        let tensor = unsafe { &*c };
        assert_eq!(tensor.data_f32(), &[5.0, 7.0, 9.0]);

        ffi::tl_cpu_tensor_free(a);
        ffi::tl_cpu_tensor_free(b);
        ffi::tl_cpu_tensor_free(c);
    }

    #[test]
    fn test_ffi_zeros_ones() {
        let shape = [2usize, 3];
        let z = ffi::tl_cpu_tensor_zeros(2, shape.as_ptr());
        assert_eq!(ffi::tl_cpu_tensor_len(z), 6);

        let o = ffi::tl_cpu_tensor_ones(2, shape.as_ptr(), false);
        let ot = unsafe { &*o };
        assert_eq!(ot.data_f32(), &[1.0; 6]);

        ffi::tl_cpu_tensor_free(z);
        ffi::tl_cpu_tensor_free(o);
    }

    #[test]
    fn test_ffi_math_functions() {
        let data = [4.0f32, 9.0];
        let shape = [2usize];
        let t = ffi::tl_cpu_tensor_new(data.as_ptr(), 1, shape.as_ptr());

        let s = ffi::tl_cpu_tensor_sqrt(t);
        let st = unsafe { &*s };
        let d = st.data_f32();
        assert!((d[0] - 2.0).abs() < 1e-5);
        assert!((d[1] - 3.0).abs() < 1e-5);

        ffi::tl_cpu_tensor_free(t);
        ffi::tl_cpu_tensor_free(s);
    }

    #[test]
    fn test_ffi_gelu() {
        let data = [0.0f32, 1.0];
        let shape = [2usize];
        let t = ffi::tl_cpu_tensor_new(data.as_ptr(), 1, shape.as_ptr());
        let g = ffi::tl_cpu_tensor_gelu(t);
        let gt = unsafe { &*g };
        assert!((gt.data_f32()[0]).abs() < 1e-5); // GELU(0)=0
        assert!((gt.data_f32()[1] - 0.914).abs() < 0.02, "GELU(1) = {}", gt.data_f32()[1]);

        ffi::tl_cpu_tensor_free(t);
        ffi::tl_cpu_tensor_free(g);
    }

    #[test]
    fn test_ffi_tril() {
        let data = [1.0f32; 4];
        let shape = [2usize, 2];
        let t = ffi::tl_cpu_tensor_new(data.as_ptr(), 2, shape.as_ptr());
        let r = ffi::tl_cpu_tensor_tril(t, 0);
        let rt = unsafe { &*r };
        assert_eq!(rt.data_f32(), &[1.0, 0.0, 1.0, 1.0]);

        ffi::tl_cpu_tensor_free(t);
        ffi::tl_cpu_tensor_free(r);
    }

    #[test]
    fn test_ffi_sum_dim() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2usize, 3];
        let t = ffi::tl_cpu_tensor_new(data.as_ptr(), 2, shape.as_ptr());

        let s = ffi::tl_cpu_tensor_sum_dim(t, 1, false);
        let st = unsafe { &*s };
        assert_eq!(st.shape(), &[2]);
        let d = st.data_f32();
        assert!((d[0] - 6.0).abs() < 1e-5);
        assert!((d[1] - 15.0).abs() < 1e-5);

        ffi::tl_cpu_tensor_free(t);
        ffi::tl_cpu_tensor_free(s);
    }

    #[test]
    fn test_ffi_sum_dim_keepdim() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2usize, 3];
        let t = ffi::tl_cpu_tensor_new(data.as_ptr(), 2, shape.as_ptr());

        let s = ffi::tl_cpu_tensor_sum_dim(t, 1, true);
        let st = unsafe { &*s };
        assert_eq!(st.shape(), &[2, 1]);

        ffi::tl_cpu_tensor_free(t);
        ffi::tl_cpu_tensor_free(s);
    }

    #[test]
    fn test_ffi_embedding() {
        let w_data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let w_shape = [3usize, 2];
        let idx_data = [0.0f32, 2.0];
        let idx_shape = [2usize];

        let w = ffi::tl_cpu_tensor_new(w_data.as_ptr(), 2, w_shape.as_ptr());
        let idx = ffi::tl_cpu_tensor_new(idx_data.as_ptr(), 1, idx_shape.as_ptr());
        let e = ffi::tl_cpu_tensor_embedding(idx, w);
        let et = unsafe { &*e };
        assert_eq!(et.shape(), &[2, 2]);
        let d = et.data_f32();
        assert!((d[0] - 0.1).abs() < 1e-5);
        assert!((d[1] - 0.2).abs() < 1e-5);
        assert!((d[2] - 0.5).abs() < 1e-5);
        assert!((d[3] - 0.6).abs() < 1e-5);

        ffi::tl_cpu_tensor_free(w);
        ffi::tl_cpu_tensor_free(idx);
        ffi::tl_cpu_tensor_free(e);
    }

    #[test]
    fn test_ffi_matmul() {
        let a_data = [1.0f32, 2.0, 3.0, 4.0];
        let b_data = [5.0f32, 6.0, 7.0, 8.0];
        let a_shape = [2usize, 2];
        let b_shape = [2usize, 2];
        let a = ffi::tl_cpu_tensor_new(a_data.as_ptr(), 2, a_shape.as_ptr());
        let b = ffi::tl_cpu_tensor_new(b_data.as_ptr(), 2, b_shape.as_ptr());
        let c = ffi::tl_cpu_tensor_matmul(a, b);
        let ct = unsafe { &*c };
        assert_eq!(ct.shape(), &[2, 2]);
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        let d = ct.data_f32();
        assert!((d[0] - 19.0).abs() < 1e-4);
        assert!((d[1] - 22.0).abs() < 1e-4);
        assert!((d[2] - 43.0).abs() < 1e-4);
        assert!((d[3] - 50.0).abs() < 1e-4);

        ffi::tl_cpu_tensor_free(a);
        ffi::tl_cpu_tensor_free(b);
        ffi::tl_cpu_tensor_free(c);
    }

    #[test]
    fn test_ffi_reshape() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2usize, 3];
        let new_shape = [3usize, 2];
        let t = ffi::tl_cpu_tensor_new(data.as_ptr(), 2, shape.as_ptr());
        let r = ffi::tl_cpu_tensor_reshape(t, 2, new_shape.as_ptr());
        let rt = unsafe { &*r };
        assert_eq!(rt.shape(), &[3, 2]);
        assert_eq!(rt.data_f32(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        ffi::tl_cpu_tensor_free(t);
        ffi::tl_cpu_tensor_free(r);
    }

    #[test]
    fn test_ffi_transpose() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2usize, 3];
        let t = ffi::tl_cpu_tensor_new(data.as_ptr(), 2, shape.as_ptr());
        let r = ffi::tl_cpu_tensor_transpose(t, 0, 1);
        let rt = unsafe { &*r };
        assert_eq!(rt.shape(), &[3, 2]);

        ffi::tl_cpu_tensor_free(t);
        ffi::tl_cpu_tensor_free(r);
    }

    #[test]
    fn test_ffi_null_safety() {
        // NULL ポインタでクラッシュしないことを確認
        assert!(ffi::tl_cpu_tensor_add(std::ptr::null_mut(), std::ptr::null_mut()).is_null());
        assert!(ffi::tl_cpu_tensor_sqrt(std::ptr::null_mut()).is_null());
        assert!(ffi::tl_cpu_tensor_gelu(std::ptr::null_mut()).is_null());
        assert!(ffi::tl_cpu_tensor_tril(std::ptr::null_mut(), 0).is_null());
        assert!(ffi::tl_cpu_tensor_sum_dim(std::ptr::null_mut(), 0, false).is_null());
        assert!(ffi::tl_cpu_tensor_embedding(std::ptr::null_mut(), std::ptr::null_mut()).is_null());
        assert_eq!(ffi::tl_cpu_tensor_dim(std::ptr::null_mut(), 0), 0);
        assert_eq!(ffi::tl_cpu_tensor_len(std::ptr::null_mut()), 0);
        ffi::tl_cpu_tensor_free(std::ptr::null_mut()); // クラッシュしない
        ffi::tl_cpu_tensor_release(std::ptr::null_mut()); // クラッシュしない
    }

    #[test]
    fn test_ffi_inplace_ops() {
        let a_data = [1.0f32, 2.0, 3.0];
        let b_data = [10.0f32, 20.0, 30.0];
        let shape = [3usize];
        let a = ffi::tl_cpu_tensor_new(a_data.as_ptr(), 1, shape.as_ptr());
        let b = ffi::tl_cpu_tensor_new(b_data.as_ptr(), 1, shape.as_ptr());

        ffi::tl_cpu_tensor_add_assign(a, b);
        let at = unsafe { &*a };
        assert_eq!(at.data_f32(), &[11.0, 22.0, 33.0]);

        ffi::tl_cpu_tensor_mul_assign_scalar_f32(a, 2.0);
        let at = unsafe { &*a };
        assert_eq!(at.data_f32(), &[22.0, 44.0, 66.0]);

        ffi::tl_cpu_tensor_free(a);
        ffi::tl_cpu_tensor_free(b);
    }

    #[test]
    fn test_ffi_comparison() {
        let a_data = [1.0f32, 2.0, 3.0];
        let b_data = [2.0f32, 2.0, 1.0];
        let shape = [3usize];
        let a = ffi::tl_cpu_tensor_new(a_data.as_ptr(), 1, shape.as_ptr());
        let b = ffi::tl_cpu_tensor_new(b_data.as_ptr(), 1, shape.as_ptr());

        let eq = ffi::tl_cpu_tensor_eq(a, b);
        let eqt = unsafe { &*eq };
        assert_eq!(eqt.data_f32(), &[0.0, 1.0, 0.0]);

        let lt = ffi::tl_cpu_tensor_lt(a, b);
        let ltt = unsafe { &*lt };
        assert_eq!(ltt.data_f32(), &[1.0, 0.0, 0.0]);

        ffi::tl_cpu_tensor_free(a);
        ffi::tl_cpu_tensor_free(b);
        ffi::tl_cpu_tensor_free(eq);
        ffi::tl_cpu_tensor_free(lt);
    }
}

// ========== GpuOps トレイト実装テスト ==========

mod trait_tests {
    use tl_cpu::tensor::CpuTensor;
    use tl_cpu::DType;
    use tl_backend::GpuOps;
    use tl_backend::tensor::GpuTensor;

    #[test]
    fn test_trait_constructors() {
        let z = <CpuTensor as GpuTensor>::zeros(&[2, 3], tl_backend::DType::F32);
        assert_eq!(z.shape(), &[2, 3]);

        let o = <CpuTensor as GpuTensor>::ones(&[3], tl_backend::DType::F32);
        assert_eq!(o.to_vec_f32(), vec![1.0, 1.0, 1.0]);

        let a = <CpuTensor as GpuTensor>::arange(0, 4, tl_backend::DType::F32);
        assert_eq!(a.to_vec_f32(), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_trait_ops() {
        let a = CpuTensor::from_slice(&[1.0, 2.0, 3.0], &[3], DType::F32);
        let b = CpuTensor::from_slice(&[4.0, 5.0, 6.0], &[3], DType::F32);

        let c = GpuOps::add(&a, &b);
        assert_eq!(c.to_vec_f32(), vec![5.0, 7.0, 9.0]);

        let d = GpuOps::mul_scalar(&a, 3.0);
        assert_eq!(d.to_vec_f32(), vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_trait_math() {
        let a = CpuTensor::from_slice(&[4.0, 9.0], &[2], DType::F32);
        let s = GpuOps::sqrt(&a);
        let v = s.to_vec_f32();
        assert!((v[0] - 2.0).abs() < 1e-5);
        assert!((v[1] - 3.0).abs() < 1e-5);

        let g = GpuOps::gelu(&a);
        assert!(g.to_vec_f32()[0] > 0.0); // gelu(4) > 0
    }

    #[test]
    fn test_trait_reshape() {
        let a = CpuTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], DType::F32);
        let b = GpuOps::reshape(&a, &[4]);
        assert_eq!(b.shape(), &[4]);
        assert_eq!(b.to_vec_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_trait_reduction() {
        let a = CpuTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], DType::F32);
        let s = GpuOps::sum(&a, 1);
        assert_eq!(s.shape(), &[2]);
        let v = s.to_vec_f32();
        assert!((v[0] - 6.0).abs() < 1e-5);
        assert!((v[1] - 15.0).abs() < 1e-5);
    }
}
