// MNIST Demo - Simplified version for testing image loading

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

struct MLP {
    fc1: Linear,
    fc2: Linear
}

impl MLP {
    fn new() -> MLP {
        return MLP(Linear::new(784, 64), Linear::new(64, 10));
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        let h = relu(self.fc1.forward(x));
        return self.fc2.forward(h);
    }
}

fn main() {
    print("=== MNIST Demo ===");
    
    // Download a sample image (digit 0)
    let url = "https://raw.githubusercontent.com/rasbt/mnist-pngs/main/train/0/16585.png";
    let path = "examples/tasks/mnist/data/digit_0.png";
    
    print("Downloading sample image...");
    Http::download(url, path);
    
    // Load image as grayscale
    print("Loading image...");
    let pixels = Image::load_grayscale(path);
    let len = tl_vec_u8_len(pixels);
    print("Loaded pixels: ");
    print(len as f32);
    
    // Create tensor
    let input = Tensor::zeros([1, 784], false);
    for i in 0..784 {
        let val = tl_vec_u8_get(pixels, i) as f32;
        input[0, i] = val / 255.0;
    }
    tl_vec_u8_free(pixels);
    
    // Create model and run inference
    let model = MLP::new();
    let logits = model.forward(input);
    let pred = logits.argmax(1, false).item_i64();
    
    print("Input: digit 0");
    print("Prediction (random model): ");
    print(pred as f32);
    
    print("Demo complete!");
}
