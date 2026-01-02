// Autograd Test

// 1. Create a tensor with gradient tracking
let w = randn([1], true);
print(w);

// 2. Define a simple function: y = w * w * 3.0
// Since we don't have scalar mul yet for tensors (maybe?), let's do w * w + w.
let w2 = w * w;
let y = w2 * w; // w^3
// let y = y + w; // w^3 + w

// 3. Backward
// y is rank 1 (size 1). backward() should work.
y.backward();

// 4. Get gradient
let g = w.grad();
print(g);

// Expected: if w is random, say 0.5.
// y = w^3. dy/dw = 3w^2.
// Check if g is roughly correct manually (by looking at output).

// 5. Update weights (Gradient Descent)
// w -= g * 0.01;
// We don't have scalar literal broadcasting in all ops yet?
// Compile_bin_op handles broadcast.
// w -= g; // simplest step
// w.sub_assign(g);
w -= g;

print(w);
