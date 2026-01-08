// MNIST MLP Model and Training
// Simple Multi-Layer Perceptron for digit recognition

struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>
}

impl Linear {
    fn new(i: i64, o: i64) -> Linear {
        // Xavier initialization
        let scale = sqrt(2.0 / (i as f32 + o as f32));
        return Linear(
            (Tensor::randn([i, o], true) * scale).detach(true), 
            (Tensor::randn([o], true) * 0.0).detach(true)
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

struct MLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear
}

impl MLP {
    fn new() -> MLP {
        // 784 (28x28) -> 128 -> 64 -> 10
        return MLP(
            Linear::new(784, 128),
            Linear::new(128, 64),
            Linear::new(64, 10)
        );
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        let h1 = relu(self.fc1.forward(x));
        let h2 = relu(self.fc2.forward(h1));
        let out = self.fc3.forward(h2);
        return out;
    }
    
    fn step(self, lr: f32) -> MLP {
        let s = self;
        s.fc1 = s.fc1.step(lr);
        s.fc2 = s.fc2.step(lr);
        s.fc3 = s.fc3.step(lr);
        return s;
    }
}

// ======= Data Loading (IDX format parser) =======

fn read_u32_be(data: Vec<u8>, offset: i64) -> i64 {
    // Read 4 bytes as big-endian u32
    let b0 = tl_vec_u8_get(data, offset) as i64;
    let b1 = tl_vec_u8_get(data, offset + 1) as i64;
    let b2 = tl_vec_u8_get(data, offset + 2) as i64;
    let b3 = tl_vec_u8_get(data, offset + 3) as i64;
    return (b0 * 16777216) + (b1 * 65536) + (b2 * 256) + b3;
}

fn load_mnist_images(path: String) -> Tensor<f32, 2> {
    // Load IDX3 image file (uncompressed)
    let data = tl_file_read_binary(path);
    
    // Parse header
    let magic = read_u32_be(data, 0);
    let num_images = read_u32_be(data, 4);
    let rows = read_u32_be(data, 8);
    let cols = read_u32_be(data, 12);
    
    print("Loaded images: ");
    print(num_images as f32);
    
    // Create tensor [num_images, 784]
    let size = num_images * 784;
    let images = Tensor::zeros([num_images, 784], false);
    
    // Read pixel data (starting at offset 16)
    for i in 0..num_images {
        for p in 0..784 {
            let byte_val = tl_vec_u8_get(data, 16 + i * 784 + p) as f32;
            // Normalize to [0, 1]
            let pixel = byte_val / 255.0;
            images[i, p] = pixel;
        }
    }
    
    tl_vec_u8_free(data);
    return images;
}

fn load_mnist_labels(path: String) -> Tensor<f32, 1> {
    // Load IDX1 label file (uncompressed)
    let data = tl_file_read_binary(path);
    
    // Parse header
    let magic = read_u32_be(data, 0);
    let num_labels = read_u32_be(data, 4);
    
    print("Loaded labels: ");
    print(num_labels as f32);
    
    // Create tensor [num_labels]
    let labels = Tensor::zeros([num_labels], false);
    
    // Read label data (starting at offset 8)
    for i in 0..num_labels {
        let label = tl_vec_u8_get(data, 8 + i) as f32;
        labels[i] = label;
    }
    
    tl_vec_u8_free(data);
    return labels;
}

// ======= Training =======

fn train_step(model: MLP, images: Tensor<f32, 2>, labels: Tensor<f32, 1>, 
              batch_start: i64, batch_size: i64, lr: f32) -> MLP {
    // Extract batch
    let batch_images = images.slice(batch_start, batch_size);
    
    // Forward pass
    let logits = model.forward(batch_images);
    
    // Create target one-hot or use cross entropy
    let batch_labels = labels.slice(batch_start, batch_size).to_i64();
    
    // Cross entropy loss
    let loss = cross_entropy(logits, batch_labels);
    
    // Backward
    loss.backward();
    
    // Update
    let updated = model.step(lr);
    
    return updated;
}

fn evaluate(model: MLP, images: Tensor<f32, 2>, labels: Tensor<f32, 1>, 
            num_samples: i64) -> f32 {
    let correct = 0.0;
    
    for i in 0..num_samples {
        let img = images.slice(i, 1);
        let logits = model.forward(img);
        let pred = logits.argmax(1, false).item_i64();
        let target = labels[i] as i64;
        
        if pred == target {
            correct = correct + 1.0;
        }
    }
    
    return correct / (num_samples as f32);
}

fn main() {
    print("=== MNIST Training ===");
    
    // Load data (assumes uncompressed IDX files)
    let train_images = load_mnist_images("examples/tasks/mnist/data/train-images-idx3-ubyte");
    let train_labels = load_mnist_labels("examples/tasks/mnist/data/train-labels-idx1-ubyte");
    let test_images = load_mnist_images("examples/tasks/mnist/data/t10k-images-idx3-ubyte");
    let test_labels = load_mnist_labels("examples/tasks/mnist/data/t10k-labels-idx1-ubyte");
    
    // Create model
    let model = MLP::new();
    
    let lr = 0.01;
    let batch_size = 32;
    let epochs = 5;
    let num_train = 60000;
    let num_batches = num_train / batch_size;
    
    for epoch in 0..epochs {
        print("Epoch: ");
        print(epoch as f32);
        
        for batch in 0..num_batches {
            let start = batch * batch_size;
            model = train_step(model, train_images, train_labels, start, batch_size, lr);
            
            if batch % 100 == 0 {
                print("  Batch: ");
                print(batch as f32);
            }
        }
        
        // Evaluate on subset of test data
        let acc = evaluate(model, test_images, test_labels, 1000);
        print("Test accuracy: ");
        print(acc);
    }
    
    print("Training complete!");
}
