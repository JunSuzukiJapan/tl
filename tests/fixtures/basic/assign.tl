fn main() {
    let a = Tensor::zeros([1]);
    println(a); // STDOUT: [0]
    // Reassignment
    a = a + 1.0;
    println(a); // STDOUT: [1]
    
    let i = 0;
    while i < 3 {
        a = a + 1.0;
        i = i + 1;
    }
    println(a); // STDOUT: [4]
}
