// Simple VarBuilder test without String type

fn main() {
    print("Test: VarBuilder functions");
    
    // Test varbuilder_get - create parameter tensors
    let weight = varbuilder_get("test_weight", [10, 5]);
    let bias = varbuilder_get("test_bias", [5]);
    
    print("Created parameters");
    print(weight);
    print(bias);
    
    // Simple computation
    let x = randn([4, 10], false);
    let output = matmul(x, weight) + bias;
    let loss = sum(output);
    
    print("Loss:");
    print(loss);
    
    // Backward
    backward(loss);
    
    // Get gradients
    let gw = varbuilder_grad("test_weight");
    let gb = varbuilder_grad("test_bias");
    
    print("Weight gradient:");
    print(gw);
    print("Bias gradient:");
    print(gb);
    
    // Update parameters
    update_all_params(0.01);
    
    print("Parameters updated!");
}
