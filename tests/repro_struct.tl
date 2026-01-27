struct S1 { x: Tensor<f32, 1> }
struct S2 { s1: S1 }
fn main() {
    let s2 = S2(S1(Tensor::zeros([1])));
}
