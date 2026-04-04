fn main() {
    let mut x = Tensor::randn([2, 3], true);
    let y = x.sum(0);
    let z = y.sumall();
    z.backward();
    println("x: {}", x);
    println("x.grad: {}", x.grad());
}
