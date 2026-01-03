tensor A: i64 = [1, 2, 3, 4, 5];
print(A);tensor B: f32 = [10.5, 20.0, 5.5];

// Sum
let s = sum(i for i in A);
print(s); // Expect 15 (1+2+3+4+5)

// Avg
let a = avg(x for x in B);
print(a); // Expect 12.0 ((10.5+20.0+5.5)/3 = 36/3 = 12)

// Max
let m = max(x for x in B);
print(m); // Expect 20.0

// Conditional Sum (Where)
let s_cond = sum(i for i in A where i > 2);
print(s_cond); // Expect 12 (3+4+5)
