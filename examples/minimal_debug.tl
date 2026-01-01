struct S {
    x: Tensor<f32, 1>,
}
impl S {
    fn new() -> S {
        return S(randn([1], false));
    }
}
fn main() {
    print("Hello");
}
