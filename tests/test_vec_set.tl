// Simple Vec test
fn main() {
    print("Start\n");

    let mut v: Vec<i64> = Vec::with_capacity(10);
    
    v.push(1);
    v.push(2);
    v.push(3);
    
    print("Pushed 3 items\n");
    
    // Test set method directly
    v.set(1, 42);
    
    print("After set\n");
    
    // Test index assignment
    v[2] = 99;
    
    print("After index assign\n");
    print("Done\n");
}
