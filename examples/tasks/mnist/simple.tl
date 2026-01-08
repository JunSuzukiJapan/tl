// MNIST Simple Test - Using correct argmax syntax

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
    print("Simple MNIST test");
    
    // Load image pixels
    let pixels = Image::load_grayscale("examples/tasks/mnist/data/test.png");
    let plen = tl_vec_u8_len(pixels);
    print("Pixels:"); print(plen);
    tl_vec_u8_free(pixels);
    
    // Create random input
    let input = Tensor::randn([1, 784], false);
    
    // Create model
    let model = Linear::new(784, 10);
    
    // Forward
    let logits = model.forward(input);
    
    // Get argmax using function syntax
    let logits_flat = reshape(logits, 10);
    let pred_tensor = argmax(logits_flat, 0);
    let pred = pred_tensor.get(0) as i64;
    print("Prediction (random):"); print(pred);
    
    print("Done!");
}
