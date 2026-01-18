let A = [[1.0, 2.0], [3.0, 4.0]];
println(A);
let B = [[1.0, 0.0], [0.0, 1.0]];
let C = [ i, k | i <- 0..2, k <- 0..2, j <- 0..2 { A[i, j] * B[j, k] } ];
println(C);
