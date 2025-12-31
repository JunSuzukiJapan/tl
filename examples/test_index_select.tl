// Test index_select gradient

fn main() {
    print("Test: index_select gradient");
    
    let weight = randn([5, 3], true);
    let indices_f32 = randn([2], false) * 0.0 + 1.0;
    
    // Manual index_select simulation
    let indices_u32 = indices_f32; // Will be cast in embedding
    let selected = embedding(indices_u32, weight);
    
    print("Selected:");
    print(selected);
    
    let loss = sum(selected);
    
    print("Loss:");
    print(loss);
    
    backward(loss);
    
    let gw = weight.grad();
    print("Weight gradient:");
    print(gw);
}
