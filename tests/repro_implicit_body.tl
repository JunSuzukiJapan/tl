fn main() {
    // User claims this puts "i and j values" in the tensor.
    // My code defaults to 0.0.
    // let Zeros = [i, j | i <- 0..5, j <- 0..5];
    let T = [i, j | i <- 0..5, j <- 0..5];
    print(T);
}
