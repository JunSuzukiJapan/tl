// Test VarBuilder-based parameter management for gradient tracking
// This tests the solution to the struct field gradient tracking problem

struct Linear {
    weight_name: String,
    bias_name: String,
    in_features: i64,
    out_features: i64,
}

impl Linear {
    fn new(name: String, in_features: i64, out_features: i64) -> Linear {
        return Linear(
            name + ".weight",
            name + ".bias",
            in_features,
            out_features
        );
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        // Get parameters from VarMap by name
        let weight = VarBuilder::get(self.weight_name, [self.in_features, self.out_features]);
        let bias = VarBuilder::get(self.bias_name, [self.out_features]);
        return x.matmul(weight) + bias;
    }
}

fn main() {
    print("Test: VarBuilder-based gradient tracking");
    
    let linear = Linear::new("linear1", 10, 5);
    let lr = 0.01;
    
    // Input and target
    let X = Tensor::randn([4, 10], false);
    let Y = Tensor::randn([4, 5], false);
    
    print("Running 3 training steps...");
    
    for i in range(0, 3) {
        let logits = linear.forward(X);
        
        // MSE loss
        let diff = logits - Y;
        let squared = diff * diff;
        let loss = squared.sum();
        
        print("Step:");
        print(i);
        print("Loss:");
        print(loss);
        
        // Backward pass
        loss.backward();
        
        // Update all parameters via VarMap
        // update_all_params(lr); // Removed
    }
    
    print("Training complete!");
    print("Checking gradients were computed...");
    
    // Verify gradients exist
    let gw = VarBuilder::grad("linear1.weight");
    let gb = VarBuilder::grad("linear1.bias");
    
    print("Weight gradient:");
    print(gw);
    print("Bias gradient:");
    print(gb);
}
