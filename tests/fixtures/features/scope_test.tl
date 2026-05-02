// STDOUT: scope_test OK

fn update_local(x: Tensor<f32, 1>) {
    let one = x / x; // 1
    x = x + one;     // x becomes x+1 (new tensor)
}

fn main() {
    let a = Tensor::randn([1], false);
    update_local(a);
    println("scope_test OK");
}
