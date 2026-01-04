fn main() {
    print("Testing logical and relu...");
    
    // Test 1: &&
    if 1.0 < 2.0 && 2.0 < 3.0 {
        print("AND works");
    }
    
    // Test 2: relu
    let t = Tensor::randn([1], true);
    let r = relu(t); // Function or method?
    print(r);
}
