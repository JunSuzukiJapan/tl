// Test with load_weights and slice operations

struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        return Linear((Tensor::randn([i, o], true)*0.1).detach(true), (Tensor::randn([o], true)*0.0).detach(true)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        return x.matmul(self.W) + self.b; 
    } 
}

struct SimpleModel { linear: Linear }
impl SimpleModel {
    fn new() -> SimpleModel {
        return SimpleModel(Linear::new(128, 12));
    }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        return self.linear.forward(x);
    }
}

fn test_with_slice() {
    print("Creating model...");
    let model = SimpleModel::new();
    print("Model created");
    
    print("Creating input...");
    let x = Tensor::randn([1, 10, 128], false);
    print("Input created");
    
    print("Forward pass...");
    let y = model.forward(x);
    print("Forward done");
    
    print("Reshaping...");
    let y_flat = y.reshape(10, 12);
    print("Reshaped");
    
    print("Slicing...");
    let s1 = y_flat.slice(0, 1);
    print("Slice 1:");
    print(s1);
    
    let s2 = y_flat.slice(1, 1);
    print("Slice 2:");
    print(s2);
    
    let s3 = y_flat.slice(2, 1);
    print("Slice 3:");
    print(s3);
    
    let s4 = y_flat.slice(3, 1);
    print("Slice 4:");
    print(s4);
    
    print("All slices done");
}

fn main() {
    print("Starting test...");
    test_with_slice();
    print("Test finished - no segfault!");
}
