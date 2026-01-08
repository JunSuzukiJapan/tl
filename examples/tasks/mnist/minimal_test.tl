// Minimal MNIST test - following train_recall.tl syntax

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
    
    fn step(self, lr: f32) -> Linear {
        let s = self;
        s.W = (s.W - s.W.grad() * lr).detach(true);
        s.b = (s.b - s.b.grad() * lr).detach(true);
        return s;
    }
}

fn main() {
    print("Minimal MNIST test");
    
    // Load existing test image
    let pixels = Image::load_grayscale("examples/tasks/mnist/data/test.png");
    let input = Tensor::zeros([1, 784], false);
    for i in 0..784 {
        let val = tl_vec_u8_get(pixels, i) as f32;
        input[0, i] = val / 255.0;
    }
    tl_vec_u8_free(pixels);
    
    // Create simple linear model
    let model = Linear::new(784, 10);
    let lr = 0.1;
    
    // Target: digit 0
    let target = Tensor::zeros([1], false);
    target[0] = 0.0;
    let target_i64 = target.to_i64();
    
    // Train for a few iterations
    for epoch in 0..50 {
        let logits = model.forward(input);
        let loss = cross_entropy(logits, target_i64);
        backward(loss);
        model = model.step(lr);
        
        if epoch - ((epoch / 10) * 10) == 0 {
            print("Epoch:"); print(epoch);
            print("Loss:"); print(loss.get(0));
        }
    }
    
    // Inference
    let logits = model.forward(input);
    let pred = logits.argmax(1, false).item_i64();
    print("Prediction:");
    print(pred);
    
    print("Done!");
}
