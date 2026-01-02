struct Sub {
    weight: Tensor<f32, 2>,
}

struct Model {
    layer1: Sub,
    bias: Tensor<f32, 1>,
}

fn main() {
    // Random init 1
    let w = randn([2, 2], false);
    let b = randn([2], false);
    
    let sub = Sub { weight: w };
    let model = Model { layer1: sub, bias: b };

    println("Original params:");
    print(model.layer1.weight);
    print(model.bias);

    let path = "test_struct_weights.safetensors";
    save_weights(model, path);
    println("Saved weights.");

    // Random init 2 (different)
    let w2 = randn([2, 2], false);
    let b2 = randn([2], false);
    let sub2 = Sub { weight: w2 };
    let model2 = Model { layer1: sub2, bias: b2 };

    println("Before load (should be different):");
    print(model2.layer1.weight);
    print(model2.bias);

    load_weights(model2, path);
    println("After load (should match original):");
    print(model2.layer1.weight);
    print(model2.bias);
}
