// Test with struct

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
    print("Test struct");
    
    let model = Linear::new(784, 10);
    print("Model created");
    
    let input = Tensor::zeros([1, 784], false);
    print("Input created");
    
    let output = model.forward(input);
    print("Forward pass done");
    
    print("Done");
}
