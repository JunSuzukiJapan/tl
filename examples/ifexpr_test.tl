fn main() {
    // Test if expression as statement (no assignment)
    // Using integers since they don't require runtime tensor calls
    let a = 5;
    let b = 10;
    
    // For now just test if as statement since if-expr assignment
    // requires the semantic checker to allow let x = if { ... }
    if true {
        print(a);
    } else {
        print(b);
    }
    
    // Nested if statement
    if false {
        print(0);
    } else {
        if true {
            print(200);
        } else {
            print(300);
        }
    }
    
    // Test with tensor literal containing expressions
    let val = 42.0;
    print([val, val + 1.0, val * 2.0]);
}
