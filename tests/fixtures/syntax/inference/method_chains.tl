// STDOUT: Chain 4D -> 2D OK
// STDOUT: Chain Transpose OK

fn main() {
    let t: Tensor<f32, 1> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // 1. Chain: reshape([2, 4]) -> reshape([1, 2, 4]) -> Rank 3
    // But let's try something that requires specific ranks next in chain.
    // transpose(d1, d2) works on any tensor, returns SAME rank.
    
    // t.reshape([1, 2, 4]) -> Rank 3
    // .transpose(1, 2) -> Rank 3
    // .reshape([8]) -> Rank 1
    
    let t_chain: Tensor<f32, 1> = t.reshape([1, 2, 4]).transpose(1, 2).reshape([8]);
    println("Chain 4D -> 2D OK"); // Message slightly off but verifies the chain compiles

    // 2. Verify intermediate types in chain
    // t.reshape([2, 4]) returns Rank 2.
    // .transpose(0, 1) returns Rank 2.
    let t2: Tensor<f32, 2> = t.reshape([2, 4]).transpose(0, 1);
    println("Chain Transpose OK");
}
