// Test gradient tracking with Linear layer only (no Embedding)

struct SimpleModel {
    linear: Linear,
}

struct Linear {
    weight: Tensor<f32, 2>,
    bias: Tensor<f32, 1>,
}

impl Linear {
    fn new(in_features: i64, out_features: i64) -> Linear {
        Linear(
            Tensor::randn([in_features, out_features], true),
            Tensor::randn([out_features], true)
        )
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        x.matmul(self.weight) + self.bias
    }
    
    fn step(self, lr: f32) {
        let gw = self.weight.grad();
        let gb = self.bias.grad();
        self.weight = (self.weight - gw * lr).detach();
        self.bias = (self.bias - gb * lr).detach();
    }
}

impl SimpleModel {
    fn new() -> SimpleModel {
        SimpleModel(Linear::new(10, 5))
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        self.linear.forward(x)
    }
    
    fn step(self, lr: f32) {
        self.linear.step(lr);
    }
}

fn main() {
    print("Test: Linear-only gradient tracking");
    
    let model = SimpleModel::new();
    let lr = 0.01;
    
    // Input: [batch=4, features=10]
    let X = Tensor::randn([4, 10], false);
    // Target: [batch=4, outputs=5]  
    let Y = Tensor::randn([4, 5], false);
    
    print("Running 3 training steps...");
    
    for i in range(0, 3) {
        let logits = model.forward(X);
        
        // Simple MSE loss
        let diff = logits - Y;
        let squared = diff * diff;
        let loss = squared.sumall();
        
        print("Step:");
        print(i);
        print("Loss:");
        print(loss);
        
        loss.backward();
        model.step(lr);
    }
    
    print("Training complete!");
}
