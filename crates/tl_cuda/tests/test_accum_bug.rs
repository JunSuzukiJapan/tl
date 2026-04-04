use tl_cuda::tensor::CudaTensor;
use tl_cuda::DType;

fn main() {
    let mut param = CudaTensor::ones(&[2, 2], DType::F32);
    param.enable_grad();

    let grad1 = CudaTensor::ones(&[2, 2], DType::F32);
    param.accumulate_grad(grad1).unwrap();

    let g_val1 = param.autograd.as_ref().unwrap().grad.as_ref().unwrap().to_vec::<f32>();
    println!("Grad after 1 accumulation: {:?}", g_val1);

    let grad2 = CudaTensor::ones(&[2, 2], DType::F32);
    param.accumulate_grad(grad2).unwrap();

    let g_val2 = param.autograd.as_ref().unwrap().grad.as_ref().unwrap().to_vec::<f32>();
    println!("Grad after 2 accumulations: {:?}", g_val2);
}
