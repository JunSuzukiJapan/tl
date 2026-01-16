fn main() {
    let N = 2;
    // Simple float body
    let l1 = [ i | i <- 0..N { 1.0 } ];
    println(l1);

    // If body
    let l2 = [ i | i <- 0..N { if 1==1 { 1.0 } else { 0.0 } } ];
    println(l2);
}
