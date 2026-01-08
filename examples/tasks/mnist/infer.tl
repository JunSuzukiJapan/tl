// MNIST Inference Only - No Training

struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>
}

impl Linear {
    fn new(i: i64, o: i64) -> Linear {
        return Linear(
            (Tensor::randn([i, o], true) * 0.1).detach(true), 
            (Tensor::randn([o], true) * 0.0).detach(true)
        );
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        return matmul(x, self.W) + self.b;
    }
}

fn main() {
    print("=== MNIST Inference Test ===");
    
    // Load test image (digit 0)
    let pixels = Image::load_grayscale("examples/tasks/mnist/data/test.png");
    let plen = tl_vec_u8_len(pixels);
    print("Pixels loaded:"); print(plen);
    
    let input = Tensor::zeros([1, 784], false);
    for i in 0..784 {
        let val = tl_vec_u8_get(pixels, i) as f32;
        input[0, i] = val / 255.0;
    }
    tl_vec_u8_free(pixels);
    print("Tensor created");
    
    // Create model
    let model = Linear::new(784, 10);
    print("Model created");
    
    // Forward
    let logits = model.forward(input);
    print("Forward done");
    
    // Get prediction
    let pred = logits.argmax(1, false).item_i64();
    print("Prediction (random):"); print(pred);
    
    print("Done!");
}
