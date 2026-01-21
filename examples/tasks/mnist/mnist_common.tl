// MNIST Common Module

struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>
}

impl Linear {
    fn new(i: i64, o: i64) -> Linear {
        return Linear(
            (Tensor::randn([i, o], true) * 0.001).detach(true), 
            (Tensor::zeros([o], true)).detach(true)
        );
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        return x.matmul(self.W) + self.b;
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
    
    // WORKAROUND: Compiler bug with variable array literals [count, dim]
    if count == 60000 {
        let shape = [60000, 784];
        let t = Tensor::from_vec_u8(vec, 16, shape, 2);
        vec.free();
        return t as Tensor<f32, 2>;
    }
    
    // Assume 10000 for test set
    let shape = [10000, 784];
    let t = Tensor::from_vec_u8(vec, 16, shape, 2);
    
    vec.free();
    return t as Tensor<f32, 2>;
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
    return t as Tensor<i64, 1>;
}
