tensor A: i64 = [1, 2, 3, 4, 5];
print(A);tensor B: f32 = [10.5, 20.0, 5.5];

// Sum
// let s = sum(i for i in A); // Old syntax?
let s = A.sum();
print(s); // Expect 15 (1+2+3+4+5)

// Avg
// let a = avg(x for x in B); // Old syntax?
let a = B.mean();
print(a); // Expect 12.0 ((10.5+20.0+5.5)/3 = 36/3 = 12)

// Max
// let m = max(x for x in B); // Old syntax?
let m = B.max();
print(m); // Expect 20.0

// Conditional Sum (Where)
// let s_cond = sum(i for i in A where i > 2); // Old syntax?
// Workaround using mask if syntax not supported
// let mask = A > 2;
// let s_cond = (A * mask).sum();
// print(s_cond); // Expect 12 (3+4+5)
