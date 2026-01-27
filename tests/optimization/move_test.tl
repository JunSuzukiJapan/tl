
struct SafeTensor {
    val: Tensor<f32, 1>
}

fn main() {
    let t1: Tensor<f32, 1> = [1.0, 2.0, 3.0];
    // t1 is defined at time T1.

    // t2 = t1. This is the last use of t1.
    // Liveness analysis should detect t1 is dead after this.
    // CodeGen should skip inc_ref for t1 and mark t1 as CLEANUP_NONE.
    // Result: t2 takes ownership. t1 is not freed. t2 is freed at end.
    let t2 = t1; 

    // Verify t2 has correct data
    println(t2);
}
