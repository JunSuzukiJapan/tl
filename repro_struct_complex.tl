struct S1 { x: Tensor<f32, 1> }
struct S2 { s1: S1, s2: S1 }
struct S3 { s2a: S2, s2b: S2 }
fn main() {
    let s3 = S3(
        S2(S1(Tensor::zeros([1])), S1(Tensor::zeros([1]))),
        S2(S1(Tensor::zeros([1])), S1(Tensor::zeros([1])))
    );
}
