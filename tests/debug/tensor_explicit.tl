fn main() {
    // Explicit range with condition
    // New Syntax: [ indices | clauses { body } ]
    // Use (i - (i/2)*2) == 0 for even check
    let B = [i | i <- 0..5, (i - (i/2)*2) == 0 { i * 10 }];
    print(B);

    // Explicit range
    let A = [i | i <- 0..5 { i }];
    
    // Implicit range (inferred from usage of A)
    // [ indices | { body } ]
    let C = [i | { A[i] + 1 }];
    print(C);

     // Explicit range that subsets A
    let D = [i | i <- 0..3 { A[i] }];
    print(D);
}
// Expected Output:
// B: [0.0, 0.0, 20.0, 0.0, 40.0]
// C: [1.0, 2.0, 3.0, 4.0, 5.0]
// D: [0.0, 1.0, 2.0]
