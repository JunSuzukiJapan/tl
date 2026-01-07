let A = [[1.0, 2.0], [3.0, 4.0]];
println(A);
let B = [[1.0, 0.0], [0.0, 1.0]];
let C = [ i, k | A[i, j] * B[j, k] ];
println(C);
