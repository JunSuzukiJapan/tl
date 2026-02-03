struct S1 { x: Tensor<f32, 1> }
fn main() {
    let s = S1(Tensor::zeros([1]));
    let s = S1(Tensor::zeros([2])); // Shadowing!
}
