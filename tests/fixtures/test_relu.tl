fn main() {
    let mut x = Tensor::randn([2, 5], true);
    let y = x.relu();
    let z = y.sumall();
    z.backward();
    println("x: {}", x);
    println("x.grad: {}", x.grad());
}
