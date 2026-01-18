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
    
    // Test modulo
    let mut m = [[10.0, 11.0], [12.0, 13.0]];
    println(m);
    m %= 3.0;
    println(m);  // Should be [[1, 2], [0, 1]]
}
