fn main() {
    print("Step 0: Start");
    
    // 1. Add (t + t)
    print("Step 1: Create t (ones)");
    let t = Tensor::ones([1], true);
    print("Step 1: t created");
    
    print("Step 1: y = t + t");
    let y = t + t;
    print("Step 1: y.backward()");
    y.backward();
    
    print("Step 1: t.grad()");
    let g = t.grad();
    
    print("Step 1: g.item()");
    let val = g.item();
    
    print("Step 1: check result");
    println("t+t grad: {}", val); 
    
    if val != 2.0 {
        print("FAIL: Expected 2.0");
        // exit(1);
    } else {
        print("PASS: t+t");
    }
}
