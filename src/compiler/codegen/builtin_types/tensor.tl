
struct Tensor {
    handle: i64,
}

impl Tensor {
    // Reshape
    extern fn reshape(self, shape: Tensor) -> Tensor;
    
    // Reduction ops
    extern fn sum(self, dim: i64) -> Tensor;
    extern fn sum_dim(self, dim: i64, keepdim: bool) -> Tensor;
    extern fn mean(self) -> Tensor;
    extern fn argmax(self, dim: i64, keepdim: bool) -> Tensor;
    extern fn argmin(self, dim: i64) -> Tensor;
    extern fn max(self, dim: i64) -> Tensor;
    extern fn min(self, dim: i64) -> Tensor;
    
    // Conversions
    extern fn to_i64(self) -> Tensor;
    
    // Softmax
    extern fn softmax(self, dim: i64) -> Tensor;
    extern fn log_softmax(self, dim: i64) -> Tensor;
    
    // Scalar extraction
    extern fn item(self) -> f32;
    extern fn item_i64(self) -> i64;
    
    // Memory/Movement
    extern fn clone(self) -> Tensor;
    extern fn contiguous(self) -> Tensor;
    extern fn detach(self, requires_grad: bool) -> Tensor;
    extern fn cuda(self) -> Tensor;
    extern fn cpu(self) -> Tensor;
    
    // Gradients
    extern fn backward(self) -> ();
    extern fn grad(self) -> Tensor;
    extern fn enable_grad(self) -> Tensor;
    
    // Inspection
    extern fn len(self) -> i64;
    extern fn dim(self, index: i64) -> i64;
    extern fn get_shape(self) -> Tensor; // Returns Tensor<i64>
    
    // Debug
    extern fn print(self) -> ();
    extern fn print_1(self) -> ();
    extern fn print_2(self) -> ();
    extern fn print_3(self) -> ();
    
    // Math Unary
    extern fn relu(self) -> Tensor;
    extern fn gelu(self) -> Tensor;
    extern fn silu(self) -> Tensor;
    extern fn sigmoid(self) -> Tensor;
    extern fn tanh(self) -> Tensor;
    extern fn exp(self) -> Tensor;
    extern fn log(self) -> Tensor;
    extern fn sqrt(self) -> Tensor;
    extern fn abs(self) -> Tensor;
    extern fn neg(self) -> Tensor;
    extern fn sin(self) -> Tensor;
    extern fn cos(self) -> Tensor;
    extern fn tan(self) -> Tensor;
    
    // Math Binary
    extern fn add(self, other: Tensor) -> Tensor;
    extern fn sub(self, other: Tensor) -> Tensor;
    extern fn mul(self, other: Tensor) -> Tensor;
    extern fn div(self, other: Tensor) -> Tensor;
    extern fn rem(self, other: Tensor) -> Tensor;
    extern fn pow(self, exponent: f32) -> Tensor;
    
    // Assignments
    extern fn add_assign(self, other: Tensor) -> ();
    extern fn sub_assign(self, other: Tensor) -> ();
    extern fn mul_assign(self, other: Tensor) -> ();
    extern fn div_assign(self, other: Tensor) -> ();
    extern fn rem_assign(self, other: Tensor) -> ();
    
    // Other ops
    extern fn narrow(self, dim: i64, start: i64, length: i64) -> Tensor;
    extern fn transpose(self, dim0: i64, dim1: i64) -> Tensor;
    extern fn matmul(self, other: Tensor) -> Tensor;
    extern fn cross_entropy(self, target: Tensor) -> Tensor;
    extern fn rms_norm(self, weight: Tensor, eps: f32) -> Tensor;
    extern fn apply_rope(self, cos: Tensor, sin: Tensor) -> Tensor;
    extern fn repeat_interleave(self, repeats: i64, dim: i64) -> Tensor;
    extern fn clamp(self, min: f32, max: f32) -> Tensor;
    extern fn slice(self, start: i64, end: i64) -> Tensor;
    extern fn conv2d(self, weight: Tensor, padding: i64, stride: i64) -> Tensor;
    extern fn scale(self, factor: f32) -> Tensor;
    
    // 4D ops
    extern fn matmul_4d(self, other: Tensor) -> Tensor;
    extern fn add_4d(self, other: Tensor) -> Tensor;
    extern fn cat_4d(self, other: Tensor, dim: i64) -> Tensor;
    extern fn transpose_2d(self, dim0: i64, dim1: i64) -> Tensor;
    
    // Quantized
    extern fn matmul_quantized(self, other: Tensor) -> Tensor;
    extern fn cat_i64(self, other: Tensor, dim: i64) -> Tensor;
    extern fn sample(self, num_samples: i64, temperature: f32) -> Tensor;
    extern fn tril(self, diagonal: i64) -> Tensor;
    extern fn embedding(self, indices: Tensor) -> Tensor;
}
