fn main() {
    let a = Tensor::zeros([1]);
    a.print();
    // Reassignment
    a = a + 1.0;
    a.print();
    
    let i = 0;
    while i < 3 {
        a = a + 1.0;
        i = i + 1;
    }
    a.print();
}
