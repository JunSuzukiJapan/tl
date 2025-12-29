fn main() {
    let A: Tensor<f32, 2> = [[1.0, 2.0], [3.0, 4.0]];
    let B: Tensor<f32, 2> = [[1.0, 1.0], [1.0, 1.0]];

    // In-place update (Variable reassignment logic)
    // A = A + B -> [[2, 3], [4, 5]]
    A += B;

    print(A);

    // Scalar update (if supported by generic binop)
    // A += 1.0; (Broadcasting update) -> [[3, 4], [5, 6]]
    // Check if parser allows literals? "let A = A + 1.0" works.
    // "A += 1.0" depends on parser expecting expr. Yes, parse_expr.
    
    // Let's test scalar update too
    A += 10.0;
    print(A);
}
