fn main() {
    let N = 2;
    // 1-level
    let l1 = [ i | i <- 0..N { i * 10 } ];
    println(l1);
    
    // Nesting
    let l2 = [ i | i <- 0..N { 
        [ j | j <- 0..N { i + j } ]
    }];
    println(l2);
    
    // Mask logic verification
    let mask_val = [ i | i <- 0..N {
        [ j | j <- 0..N {
            if i == j { 1.0 } else { 0.0 }
        }]
    }];
    println(mask_val);
}
