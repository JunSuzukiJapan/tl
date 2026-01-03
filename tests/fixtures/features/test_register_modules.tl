struct Sub { x: Tensor<f32, 1> }

// 3 params total: a, b.x, b.y (wait y is not in Sub)
// Let's make it slightly deeper
struct Inner { val: Tensor<f32, 1> } 
struct Outer {
    t1: Tensor<f32, 1>,
    inner: Inner,
    t2: Tensor<f32, 1>
}

fn main() {
    print("Testing register_modules...");
    let m = Outer {
        t1: randn([1], true),
        inner: Inner { val: randn([1], true) },
        t2: randn([1], true)
    };
    
    register_modules(m);
    
    // Should save 3 parameters
    save_all_params("test_auto_reg.safetensors");
}
