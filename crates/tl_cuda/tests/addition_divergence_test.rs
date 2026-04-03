use serial_test::serial;
use tl_cuda::{CudaTensor, DType};

// CPU側でのGround Truthの代わりに、明らかなNaNや異常値（全トークンが同じ値になる等）が無いかをスキャンする
fn assert_no_nan_or_inf(tensor: &CudaTensor, name: &str) {
    let vec = tensor.to_vec::<f32>();
    for (i, &v) in vec.iter().enumerate() {
        assert!(
            !v.is_nan() && !v.is_infinite(),
            "[{}] NaN or Inf detected at index {}: {}",
            name,
            i,
            v
        );
    }
}

fn assert_shape(t: &CudaTensor, expected: &[usize]) {
    assert_eq!(t.shape(), expected, "Shape mismatch");
}

#[test]
#[serial]
fn test_layernorm_forward() {
    let batch = 1;
    let seq = 12;
    let embed = 128;
    
    // Create random inputs with varying variance
    let mut data = vec![0.0f32; batch * seq * embed];
    for i in 0..data.len() {
        data[i] = (i as f32 % 10.0) - 5.0; // [-5.0, 4.0]
    }
    
    let x = CudaTensor::from_slice(&data, &[batch, seq, embed], DType::F32);
    let scale = CudaTensor::ones(&[embed], DType::F32);
    let bias = CudaTensor::zeros(&[embed], DType::F32);
    
    // Mean
    let mean_reduced = x.sum_impl(2).unwrap().div_scalar_impl(embed as f32).unwrap();
    let mean = mean_reduced.reshape_impl(&[batch, seq, 1]).unwrap();
    // Variance: (x - mean)^2
    let diff = x.sub_impl(&mean).unwrap();
    let sq_diff = diff.pow_scalar_impl(2.0).unwrap();
    let var_reduced = sq_diff.sum_impl(2).unwrap().div_scalar_impl(embed as f32).unwrap();
    let var = var_reduced.reshape_impl(&[batch, seq, 1]).unwrap();
    
    // sqrt(var + eps)
    let eps = 1e-5;
    let var_eps = var.add_scalar_impl(eps).unwrap();
    let std = var_eps.sqrt_impl().unwrap();
    
    // Normalized
    let norm = diff.div_impl(&std).unwrap();
    
    // Output
    let out = norm.mul_impl(&scale).unwrap().add_impl(&bias).unwrap();
    
    assert_no_nan_or_inf(&out, "layernorm_out");
    assert_shape(&out, &[1, 12, 128]);
}

#[test]
#[serial]
fn test_adam_step() {
    let n = 128;
    let lr = 0.0005;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps = 1e-8f32;
    
    let param_data = vec![0.5f32; n];
    let grad_data = vec![0.001f32; n];
    let m_data = vec![0.0f32; n];
    let v_data = vec![0.0f32; n];
    
    let _param = CudaTensor::from_slice(&param_data, &[n], DType::F32);
    let grad = CudaTensor::from_slice(&grad_data, &[n], DType::F32);
    let m = CudaTensor::from_slice(&m_data, &[n], DType::F32);
    let v = CudaTensor::from_slice(&v_data, &[n], DType::F32);
    
    // m = beta1 * m + (1 - beta1) * grad
    let new_m = m.mul_scalar_impl(beta1).unwrap().add_impl(&grad.mul_scalar_impl(1.0 - beta1).unwrap()).unwrap();
    
    // v = beta2 * v + (1 - beta2) * (grad^2)
    let grad_sq = grad.pow_scalar_impl(2.0).unwrap();
    let new_v = v.mul_scalar_impl(beta2).unwrap().add_impl(&grad_sq.mul_scalar_impl(1.0 - beta2).unwrap()).unwrap();
    
    // sqrt(new_v)
    let sqrt_v = new_v.sqrt_impl().unwrap();
    let denom = sqrt_v.add_scalar_impl(eps).unwrap();
    let update = new_m.mul_scalar_impl(lr).unwrap().div_impl(&denom).unwrap();
    
    // Expected to not have NaNs!
    assert_no_nan_or_inf(&new_v, "adam_v");
    assert_no_nan_or_inf(&sqrt_v, "adam_sqrt_v");
    assert_no_nan_or_inf(&denom, "adam_denom");
    assert_no_nan_or_inf(&update, "adam_update");
    
    // Check values
    let denom_vec = denom.to_vec::<f32>();
    assert!(denom_vec[0] > 0.0);
}

#[test]
#[serial]
fn test_max_impl() {
    let shape = [6, 4];
    let data = vec![
        -10.0f32, -5.0f32, -1.0f32, -2.0f32,
        1.0f32, 2.0f32, 3.0f32, 4.0f32,
        0.0f32, 0.0f32, 0.0f32, 0.0f32,
        -1e9f32, -1e9f32, -1e9f32, -1e9f32,
        5.0f32, -3.0f32, 10.0f32, 2.0f32,
        100.0f32, 200.0f32, -50.0f32, 0.0f32,
    ];
    let x = CudaTensor::from_slice(&data, &shape, DType::F32);
    let max_vals = x.max_impl(1).unwrap();
    assert_shape(&max_vals, &[6]);
    let out = max_vals.to_vec::<f32>();
    println!("max_vals: {:?}", out);
    assert_eq!(out[0], -1.0);
    assert_eq!(out[1], 4.0);
    assert_eq!(out[2], 0.0);
    assert_eq!(out[3], -1e9);
    assert_eq!(out[4], 10.0);
    assert_eq!(out[5], 200.0);
}

#[test]
#[serial]
#[ignore]
fn test_layernorm_autograd() {
    let x_data = vec![0.5f32, -0.2f32, 1.3f32, 0.0f32];
    let weight_data = vec![1.0f32, 1.0f32, 1.0f32, 1.0f32];
    let bias_data = vec![0.0f32, 0.0f32, 0.0f32, 0.0f32];
    
    let mut x = CudaTensor::from_slice(&x_data, &[1, 1, 4], DType::F32);
    let mut w = CudaTensor::from_slice(&weight_data, &[4], DType::F32);
    let mut b = CudaTensor::from_slice(&bias_data, &[4], DType::F32);
    
    x.enable_grad();
    w.enable_grad();
    b.enable_grad();
    
    let embed = 4;
    // Mean
    let mean_reduced = x.sum_impl(2).unwrap().div_scalar_impl(embed as f32).unwrap();
    let mean = mean_reduced.reshape_impl(&[1, 1, 1]).unwrap();
    // Variance
    let diff = x.sub_impl(&mean).unwrap();
    let sq_diff = diff.pow_scalar_impl(2.0).unwrap();
    let var_reduced = sq_diff.sum_impl(2).unwrap().div_scalar_impl(embed as f32).unwrap();
    let var = var_reduced.reshape_impl(&[1, 1, 1]).unwrap();
    // Std
    let eps = 1e-5;
    let var_eps = var.add_scalar_impl(eps).unwrap();
    let std = var_eps.sqrt_impl().unwrap();
    let norm = diff.div_impl(&std).unwrap();
    let out = norm.mul_impl(&w).unwrap().add_impl(&b).unwrap();
    
    let mut loss = out.sum_impl(2).unwrap().sum_impl(1).unwrap().sum_impl(0).unwrap();
    loss.backward().unwrap();
    
    let gx = x.get_grad().unwrap().to_vec::<f32>();
    let gw = w.get_grad().unwrap().to_vec::<f32>();
    let gb = b.get_grad().unwrap().to_vec::<f32>();
    
    println!("gx: {:?}", gx);
    println!("gw: {:?}", gw);
    println!("gb: {:?}", gb);
    
    for v in gx.iter() { assert!(!v.is_nan() && !v.is_infinite()); }
    for v in gw.iter() { assert!(!v.is_nan() && !v.is_infinite()); }
    for v in gb.iter() { assert!(!v.is_nan() && !v.is_infinite()); }
}
