fn main() {
    // Immutable
    let x = 10;
    print(x); // STDOUT: 10
    
    // Mutable
    let mut y = 20;
    print(y); // STDOUT: 20
    y = 30;
    print(y); // STDOUT: 30

    // Shadowing
    let z = 100;
    {
        let z = 200;
        print(z); // STDOUT: 200
    }
    print(z); // STDOUT: 100
}
