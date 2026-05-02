// STDOUT: Training complete!
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
        Linear(
            name + ".weight",
            name + ".bias",
            in_features,
            out_features
        )
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        // Get parameters from VarMap by name
        let weight = VarBuilder::get(self.weight_name, [self.in_features, self.out_features]);
        let bias = VarBuilder::get(self.bias_name, [self.out_features]);
        x.matmul(weight) + bias
    }
}

fn main() {
    let linear = Linear::new("linear1", 4, 3);
    let lr = 0.01;
    
    // Input and target
    let X = Tensor::randn([2, 4], false);
    let Y = Tensor::randn([2, 3], false);
    
    for i in range(0, 3) {
        let logits = linear.forward(X);
        let diff = logits - Y;
        let squared = diff * diff;
        let loss = squared.sumall();
        
        loss.backward();
        
        // SGD update: p = (p - grad * lr).detach(true)
        VarBuilder::update_all_params(lr);
    }
    
    println("Training complete!");
    
    // Verify gradients exist after training
    let gw = VarBuilder::grad("linear1.weight");
    let gb = VarBuilder::grad("linear1.bias");
    print(gw);
    print(gb);
}
