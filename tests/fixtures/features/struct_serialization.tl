struct Level3 {
    data: Tensor<f32, 2>,
}

struct Level2 {
    inner: Level3,
    bias: Tensor<f32, 1>,
}

struct Level1 {
    mid: Level2,
    linear: Tensor<f32, 2>,
}

fn main() {
    println("Initializing nested structs...");
    
    // Create random Data
    let t3 = randn([2, 2], false);
    let b2 = randn([2], false);
    let l1 = randn([2, 2], false);

    // Assemble structs
    let s3 = Level3 { data: t3 };
    let s2 = Level2 { inner: s3, bias: b2 };
    let root = Level1 { mid: s2, linear: l1 };

    println("Original Root Linear:");
    print(root.linear);
    println("Original Mid Inner Data:");
    print(root.mid.inner.data);

    let path = "test_deep_struct.safetensors";
    println("Saving to 'test_deep_struct.safetensors'...");
    save_weights(root, path);

    // Create new identical struct with ZEROS to verify load actually happens
    let t3_new = randn([2, 2], false); // Just random garbage
    let b2_new = randn([2], false);
    let l1_new = randn([2, 2], false);

    let s3_new = Level3 { data: t3_new };
    let s2_new = Level2 { inner: s3_new, bias: b2_new };
    let root_new = Level1 { mid: s2_new, linear: l1_new };

    println("New struct (before load) - Should differ:");
    print(root_new.linear);

    println("Loading weights...");
    load_weights(root_new, path);

    println("After load - Should match Original:");
    println("Root Linear:");
    print(root_new.linear);
    println("Mid Inner Data:");
    print(root_new.mid.inner.data);
}
