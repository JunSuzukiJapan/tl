
fn main() {
    println("Testing Tensor class...");

    // Zeros
    println("Tensor::zeros([2, 3])...");
    let t1 = Tensor::zeros([2, 3]);
    println(t1);

    // Ones
    println("Tensor::ones([2, 3])...");
    let t2 = Tensor::ones([2, 3]);
    println(t2);

    // Binary Ops
    println("Binary Ops...");
    let t_add = t1.add(t2);
    println("t1 + t2 (zeros + ones):");
    println(t_add);
    
    let t_sub = t2.sub(t1);
    println("t2 - t1 (ones - zeros):");
    println(t_sub);

    // Sumall
    println("t2.sumall():");
    println(t2.sumall()); // Should be 6.0

    // Clone
    let t2_clone = t2.clone();
    println("t2 clone:");
    println(t2_clone);
    
    // Randn
    println("Tensor::randn([2, 3])...");
    let t3 = Tensor::randn([2, 3]);
    // println(t3); // Verified by not crashing. Output varies.

    // Clear Grads
    println("Tensor::clear_grads()...");
    Tensor::clear_grads();

    // from_vec_u8
    println("Tensor::from_vec_u8...");
    let mut v_u8 = Vec::new();
    v_u8.push(10 as u8);
    v_u8.push(255 as u8);
    let t_u8 = Tensor::from_vec_u8(v_u8, [2]);
    println(t_u8);

    println("Tensor tests passed!");
}
