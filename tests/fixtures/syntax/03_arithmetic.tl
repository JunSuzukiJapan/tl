fn main() {
    let a = 10;
    let b = 3;

    print(a + b); // STDOUT: 13
    print(a - b); // STDOUT: 7
    print(a * b); // STDOUT: 30
    
    // Integer division? Assumed standard behavior
    // print(a / b); // 3 ? Need to check if integer division is supported or casts to float.
    // Based on previous contexts, likely standard ops.
    
    let f1 = 10.0;
    let f2 = 2.5;
    print(f1 / f2); // STDOUT: 4.0

    // Assignments
    let c = 5;
    c = c + 5; 
    print(c); // STDOUT: 10
    
    // Comparison
    if a > b { print("gt"); } // STDOUT: gt
    if b < a { print("lt"); } // STDOUT: lt
    if a == 10 { print("eq"); } // STDOUT: eq
    if a != 5 { print("neq"); } // STDOUT: neq
}
