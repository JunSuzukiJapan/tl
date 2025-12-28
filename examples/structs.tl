struct Linear<T> {
    weight: Tensor<T, 2>,
    bias: Tensor<T, 1>
}

impl Linear {
    fn new() -> void {
        let x = 1;
    }
}

fn main() -> void {
    let l = Linear;
}
