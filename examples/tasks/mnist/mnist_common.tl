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

    let magic = vec.read_i32_be(0);
    let count = vec.read_i32_be(4);
    let rows = vec.read_i32_be(8);
    let cols = vec.read_i32_be(12);
    
    println("  Count: {} Shape: {} x {}", count, rows, cols);
    
    let dim = rows * cols;
    
    // Use the new specialized to_tensor_2d method
    if count == 60000 {
        let t = vec.to_tensor_2d(16, 60000, 784);
        vec.free();
        return t as Tensor<f32, 2>;
    }
    
    // Assume 10000 for test set
    let t = vec.to_tensor_2d(16, 10000, 784);
    
    vec.free();
    t as Tensor<f32, 2>
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
    
    let magic = vec.read_i32_be(0);
    let count = vec.read_i32_be(4);
    println("  Count: {}", count);
    
    let t = Tensor::from_u8_labels(vec, 8, count);
    
    vec.free();
    t as Tensor<i64, 1>
}
