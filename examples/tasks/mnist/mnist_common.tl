// MNIST Common Module

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
        print("Error: Failed to read file or empty file: " + path);
        // Return dummy
        let t = Tensor::zeros([1, 784], false);
        return t;
    }

    let magic = tl_vec_u8_read_i32_be(vec, 0);
    let count = tl_vec_u8_read_i32_be(vec, 4);
    let rows = tl_vec_u8_read_i32_be(vec, 8);
    let cols = tl_vec_u8_read_i32_be(vec, 12);
    
    print("  Count:"); print(count);
    print("  Shape:"); print(rows); print(cols);
    
    let dim = rows * cols;
    let shape = [count, dim]; 
    
    let t = tl_tensor_from_vec_u8(vec, 16, shape, 2);
    
    tl_vec_u8_free(vec);
    return t as Tensor<f32, 2>;
}

fn load_mnist_labels(path: String) -> Tensor<i64, 1> {
    print("Loading labels from:"); print(path);
    let vec = tl_file_read_binary(path);
    let len = tl_vec_u8_len(vec);
    if len == 0 {
        print("Error: Failed to read file: " + path);
        let t = Tensor::zeros([1], false).to_i64();
        return t;
    }
    
    let magic = tl_vec_u8_read_i32_be(vec, 0);
    let count = tl_vec_u8_read_i32_be(vec, 4);
    print("  Count:"); print(count);
    
    let t = tl_tensor_from_u8_labels(vec, 8, count);
    
    tl_vec_u8_free(vec);
    return t as Tensor<i64, 1>;
}
