// MNIST Common Module

struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>
}

impl Linear {
    fn new(i: i64, o: i64) -> Linear {
        // Workaround: StructInit does not correctly handle BinOp results,
        // so we assign to intermediate variables and avoid BinOp in struct fields.
        let w_base = Tensor::randn([i, o], true);
        let w = w_base * 0.001;
        let w_d = w.detach();
        print("w_d after detach:");
        print(w_d);
        
        let b_base = Tensor::zeros([o], true);
        let b_d = b_base.detach();
        print("b_d after detach:");
        print(b_d);
        
        print("Constructing Linear...");
        let result = Linear(w_d, b_d);
        print("Linear constructed, checking result.W:");
        print(result.W);
        result
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        x.matmul(self.W) + self.b
    }
    
    fn step(self, lr: f32) -> Linear {
        let mut s = self;
        let gW = s.W.grad();
        let gb = s.b.grad();
        s.W = (s.W - gW * lr).detach();
        s.b = (s.b - gb * lr).detach();
        s
    }

}

// Read a big-endian i32 from 4 consecutive bytes in a Vec<u8>
fn read_i32_be(vec: Vec<u8>, offset: i64) -> i64 {
    let b0 = vec.get(offset) as i64;
    let b1 = vec.get(offset + 1) as i64;
    let b2 = vec.get(offset + 2) as i64;
    let b3 = vec.get(offset + 3) as i64;
    b0 * 16777216 + b1 * 65536 + b2 * 256 + b3
}

fn load_mnist_images(path: String) -> Tensor<f32, 2> {
    println("Loading images from: {}", path);
    let vec = File::read_binary(path);
    let len = vec.len();
    if len == 0 {
        print("Error: Failed to read file or empty file: " + path);
        // Return dummy
        let t = Tensor::zeros([1, 784], false);
        return t;
    }

    let magic = read_i32_be(vec, 0);
    let count = read_i32_be(vec, 4);
    let rows = read_i32_be(vec, 8);
    let cols = read_i32_be(vec, 12);
    
    println("  Count: {} Shape: {} x {}", count, rows, cols);
    
    let dim = rows * cols;
    
    // Use Tensor::from_vec_u8 to convert byte data to tensor
    // Pixel data starts at offset 16
    // First, we need to create a sub-view from offset 16
    // Since from_vec_u8 doesn't support offset, we'll read manually
    let mut result: Tensor<f32, 2> = Tensor::zeros([count, 784], false);
    
    let total_pixels = count * 784;
    for i in range(0, total_pixels) {
        let byte_val = vec.get(16 + i);
        let row = i / 784;
        let col = i - (row * 784);
        result[row, col] = (byte_val as f32) / 255.0;
    }
    
    vec.free();
    result
}

fn load_mnist_labels(path: String) -> Tensor<i64, 1> {
    println("Loading labels from: {}", path);
    let vec = File::read_binary(path);
    let len = vec.len();
    if len == 0 {
        print("Error: Failed to read file: " + path);
        let t = Tensor::zeros([1], false).to_i64();
        return t;
    }
    
    let magic = read_i32_be(vec, 0);
    let count = read_i32_be(vec, 4);
    println("  Count: {}", count);
    
    let t = Tensor::from_u8_labels(vec, 8, count);
    
    vec.free();
    t as Tensor<i64, 1>
}
