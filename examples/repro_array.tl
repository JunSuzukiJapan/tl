struct S { x: Tensor<f32, 1> }

fn main() {
    let x = randn([1, 2], false);
    print("OK");
}
