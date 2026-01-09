extern fn tl_tensor_reshape_dims(t: Tensor<f32, 1>, dims: Tensor<i64, 1>, n: i64) -> Tensor<f32, 2>;

fn main() {
    print("Testing ensure_tensor transformation...");
    let data = [1.0, 2.0, 3.0, 4.0]; 
    let shape = [2, 2]; 
    
    print("Calling reshape...");
    let res = tl_tensor_reshape_dims(data, shape, 2);
    
    print("Reshape done.");
}
