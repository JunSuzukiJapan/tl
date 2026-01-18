fn main() {
    let mut t = Tensor::ones([2, 2]);
    println(t);
    t *= 2.0;
    println(t);
    t += 0.5;
    println(t);
    t -= 0.5;
    println(t);
    t /= 2.0;
    println(t);
}
