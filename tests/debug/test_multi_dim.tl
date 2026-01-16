fn main() {
    let N = 2;
    // 2D
    let mat = [ r, c | r <- 0..N, c <- 0..N { r + c } ];
    println(mat);
    // print("Shape: "); println(mat.shape());

    // 3D Mask
    let K = 3;
    let mask = [ k, r, c | k <- 0..K, r <- 0..N, c <- 0..N {
        if r + c == k { 1.0 } else { 0.0 }
    }];
    println(mask);
    // print("Mask Shape: "); println(mask.shape());
}
