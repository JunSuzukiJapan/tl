// MNIST Training with Real Data (IDX Loading)

struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>
}

impl Linear {
    fn new(i: i64, o: i64) -> Linear {
        return Linear(
            (Tensor::randn([i, o], true) * 0.1).detach(true), 
            (Tensor::zeros([o], true)).detach(true)
        );
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        return matmul(x, self.W) + self.b;
    }
    
    fn step(self, lr: f32) -> Linear {
        let s = self;
        let gW = s.W.grad();
        let gb = s.b.grad();
        s.W = (s.W - gW * lr).detach(true);
        s.b = (s.b - gb * lr).detach(true);
        return s;
    }
}

fn load_mnist_images(path: String) -> Tensor<f32, 2> {
    print("Loading images from:"); print(path);
    let vec = tl_file_read_binary(path);
    let len = tl_vec_u8_len(vec);
    if len == 0 {
        print("Error: Failed to read file or empty file");
        let t = Tensor::zeros([1, 784], false);
        return t;
    }

    let magic = tl_vec_u8_read_i32_be(vec, 0);
    // expect 2051

    let count = tl_vec_u8_read_i32_be(vec, 4);
    let rows = tl_vec_u8_read_i32_be(vec, 8);
    let cols = tl_vec_u8_read_i32_be(vec, 12);
    
    print("  Count:"); print(count);
    print("  Shape:"); print(rows); print(cols);
    
    let dim = rows * cols;
    let shape = [count, dim]; 
    
    // Load as Rank 2 of [count, 784]
    // The helper returns generic tensor, we cast to known shape
    let t = tl_tensor_from_vec_u8(vec, 16, shape, 2);
    
    tl_vec_u8_free(vec);
    return t as Tensor<f32, 2>;
}

fn load_mnist_labels(path: String) -> Tensor<i64, 1> {
    print("Loading labels from:"); print(path);
    let vec = tl_file_read_binary(path);
    let len = tl_vec_u8_len(vec);
    if len == 0 {
        print("Error: Failed to read file");
        let t = Tensor::zeros([1], false).to_i64();
        return t;
    }
    
    let magic = tl_vec_u8_read_i32_be(vec, 0);
    // expect 2049
    let count = tl_vec_u8_read_i32_be(vec, 4);
    print("  Count:"); print(count);
    
    let t = tl_tensor_from_u8_labels(vec, 8, count);
    
    tl_vec_u8_free(vec);
    return t as Tensor<i64, 1>;
}

fn main() {
    print("=== MNIST Training ===");
    
    let train_x = load_mnist_images("examples/tasks/mnist/data/train-images-idx3-ubyte");
    let train_y = load_mnist_labels("examples/tasks/mnist/data/train-labels-idx1-ubyte");
    
    let test_x = load_mnist_images("examples/tasks/mnist/data/t10k-images-idx3-ubyte");
    let test_y = load_mnist_labels("examples/tasks/mnist/data/t10k-labels-idx1-ubyte");
    
    let model = Linear::new(784, 10);
    let lr = 0.5;
    let batch_size = 64;
    
    let num_batches = 60000 / batch_size; 
    let epoch = 0;
    
    print("Starting training loop...");
    
    while epoch < 3 {
        let b = 0;
        let total_loss = 0.0;
        
        while b < num_batches {
            let start = b * batch_size;
            
            // Slice batch
            let batch_x = train_x.slice(start, batch_size);
            let batch_y = train_y.slice(start, batch_size);
            
            let logits = model.forward(batch_x);
            let loss = cross_entropy(logits, batch_y);
            
            backward(loss);
            model = model.step(lr);
            
            total_loss = total_loss + loss.item();
            
            if b - ((b / 200) * 200) == 0 {
                 print("  Batch:"); print(b);
                 print("  Loss:"); print(loss.item());
            }
            
            b = b + 1;
        }
        
        print("Epoch complete:"); print(epoch);
        print("Avg Loss:"); print(total_loss / num_batches as f32); // Assuming casting works or division works?
        // Division: f32 / i32? No. f32 / f32.
        // num_batches is i64?
        // We don't have explicit cast for variables easily?
        // Just print total_loss.
        
        epoch = epoch + 1;
        
        // --- Evaluation ---
        print("Evaluating...");
        let eval_size = 1000;
        let eval_x = test_x.slice(0, eval_size);
        let eval_y = test_y.slice(0, eval_size);
        
        let logits = model.forward(eval_x);
        let preds = logits.argmax(1, false);
        
        let correct = 0;
        let i = 0;
        while i < eval_size {
            // Need cast to int for comparison? Or just compare items (f32 vs f32 from get)
            // preds is f32 tensor (from argmax refactor)
            // eval_y is i64 tensor.
            // eval_y.get(i) returns c_float (f32).
            // preds.get(i) returns f32.
            // So comparison should work.
            
            if preds.get(i) == eval_y.get(i) {
                correct = correct + 1;
            }
            i = i + 1;
        }
        print("Accuracy:"); print(correct); print("/"); print(eval_size);
    }
}
