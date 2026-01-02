
fn update_local(x: Tensor<f32, 1>) {
    let one = x / x; // 1
    x = x + one;     // x becomes x+1 (new tensor)
    print(x);        // Should print x+1
}

fn update_inplace(x: Tensor<f32, 1>) {
    let one = x / x;
    x += one;        // x += 1 (modify opaque tensor content)
    print(x);
}

let a = randn([1], false);
print(a);

print(a); // Original
update_local(a);
print(a); // Should be Original (if pass-by-value/pointer-copy)

update_inplace(a);
print(a); // Should be Modified (if OpaqueTensor content updated)
