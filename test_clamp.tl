fn main() {
    let mut x = Tensor::randn([2, 2], false);
    println("x: {}", x);
    let y = x.clamp(0.0, 1.0);
    println("y: {}", y);
}
