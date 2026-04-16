fn main() {
    let mut x = Tensor::randn([2, 5], true);
    let y = x.softmax(1);
    let z = (y - 0.5).pow(2).sumall();
    z.backward();
    println("x.grad: {}", x.grad());
}
