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
        t1: Tensor::randn([1], true),
        inner: Inner { val: Tensor::randn([1], true) },
        t2: Tensor::randn([1], true)
    };
    
    // register_modules has been removed in favor of explicit Param method usage or just passing model to save
    
    // Should save 3 parameters
    Param::save(m, "test_auto_reg.safetensors");
}
