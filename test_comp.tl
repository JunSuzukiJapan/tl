fn main() {
    let mut x = Tensor::randn([2, 2], true);
    let y = [k | k <- 0..2, r <- 0..2, c <- 0..2, r == k { x[r, c] }];
    let z = y.sumall();
    z.backward();
    println("y: {}", y);
    println("x.grad: {}", x.grad());
}
