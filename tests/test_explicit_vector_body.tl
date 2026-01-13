fn main() {
    // Test if explicit vector body works
    // Should result in shape [3, 2, 2]
    // i in 0..3 (0, 1, 2)
    // j in 0..2 (0, 1)
    // Body: [i, j]
    let T = [i, j | i <- 0..3, j <- 0..2 { [i, j] }];
    print(T);
}
