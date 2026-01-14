fn main() {
    println("Testing char_at...");
    let s = "123";
    let c = char_at(s, 1);
    print("char_at(1): ");
    println(c);

    println("Testing Tensor::set...");
    let t = Tensor::zeros([2, 2]);
    t.set(0, 0, 1.0);
    println("Set 0,0 to 1.0");
    t.set(1, 1, 2.0);
    println("Set 1,1 to 2.0");
    
    // Check values if get works (get is not fully reliable yet, but print might work?)
    // print(t); // Tensor print not implemented?
}
