fn main() {
    let scalar_i = 123;
    let scalar_f = scalar_i as f32;
    print(scalar_f); // 123.0

    let scalar_b = true;
    let b_i = scalar_b as i64;
    print(b_i); // 1

    let t_i = [1, 2, 3];
    let t_f = t_i as Tensor<f32, 1>;
    print(t_f); // [1.0, 2.0, 3.0]

    let t_back_i = t_f as Tensor<i64, 1>;
    print(t_back_i); // [1, 2, 3]

    // pow test
    let base = 2;
    let exp = 3.0;
    // pow expects (f32, f32) or (Tensor, Tensor) or broadcast.
    // In strict sense, tl semantics for pow on scalars depends on implementation.
    // But let's test casting for pow arguments if they are tensors.
    
    // For scalar math, let's just print.
    print(base as f32);

    // Casting in expressions
    let x = (10 as f32) + 2.5;
    print(x); // 12.5
}
