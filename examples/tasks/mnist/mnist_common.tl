// MNIST Common Module - MLP Model

struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>
}

impl Linear {
    fn new(i: i64, o: i64) -> Linear {
        // He initialization: sqrt(2/fan_in) — optimal for ReLU
        let scale = (2.0 / (i as f32)).sqrt();
        let w = (Tensor::randn([i, o], true) * scale).detach(true);
        let b = Tensor::zeros([o], true).detach(true);
        Linear(w, b)
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        x.matmul(self.W) + self.b
    }
    
    fn step(self, lr: f32) -> Linear {
        let mut s = self;
        let gW = s.W.grad();
        let gb = s.b.grad();
        s.W = (s.W - gW * lr).detach(true);
        s.b = (s.b - gb * lr).detach(true);
        s
    }
}

// 2-layer MLP: input -> hidden (ReLU) -> output
struct MLP {
    l1: Linear,
    l2: Linear
}

impl MLP {
    fn new(input_dim: i64, hidden_dim: i64, output_dim: i64) -> MLP {
        MLP(Linear::new(input_dim, hidden_dim), Linear::new(hidden_dim, output_dim))
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        let h = self.l1.forward(x).relu();
        self.l2.forward(h)
    }
    
    fn step(self, lr: f32) -> MLP {
        let mut s = self;
        s.l1 = s.l1.step(lr);
        s.l2 = s.l2.step(lr);
        s
    }
}

// Read a big-endian i32 from 4 consecutive bytes in a Vec<u8>
fn read_i32_be(vec: Vec<u8>, offset: i64) -> i64 {
    let b0 = vec.get(offset).unwrap() as i64;
    let b1 = vec.get(offset + 1).unwrap() as i64;
    let b2 = vec.get(offset + 2).unwrap() as i64;
    let b3 = vec.get(offset + 3).unwrap() as i64;
    b0 * 16777216 + b1 * 65536 + b2 * 256 + b3
}

fn load_mnist_images(path: String) -> Result<Tensor<f32, 2>, String> {
    println("Loading images from: {}", path);
    let res = File::read_binary(path);
    let vec = match res {
        Result::Ok(v) => v,
        Result::Err(e) => {
            return Result::Err("Failed to read file: " + e + " at " + path);
        }
    };

    let magic = read_i32_be(vec, 0);
    let count = read_i32_be(vec, 4);
    let rows = read_i32_be(vec, 8);
    let cols = read_i32_be(vec, 12);
    
    println("  Count: {} Shape: {} x {}", count, rows, cols);
    
    let dim = rows * cols;
    let shape = [count, dim];
    
    // Use Tensor::from_vec_u8_offset to convert byte data to tensor directly
    // Pixel data starts at offset 16
    let result_raw = Tensor::from_vec_u8_offset(vec, shape, 16);
    
    // Normalize to 0.0 - 1.0
    let result = result_raw / 255.0;
    
    Result::Ok(result)
}

fn load_mnist_labels(path: String) -> Result<Tensor<f32, 1>, String> {
    println("Loading labels from: {}", path);
    let res = File::read_binary(path);
    let vec = match res {
        Result::Ok(v) => v,
        Result::Err(e) => {
            return Result::Err("Failed to read file: " + e + " at " + path);
        }
    };
    
    let magic = read_i32_be(vec, 0);
    let count = read_i32_be(vec, 4);
    println("  Count: {}", count);
    
    // Read labels manually from offset 8
    let result = Tensor::from_vec_u8_offset(vec, [count], 8);
    
    Result::Ok(result)
}
