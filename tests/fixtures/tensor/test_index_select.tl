// Test index_select gradient

fn main() {
    print("Test: index_select gradient");
    
    let weight = Tensor::randn([5, 3], true);
    let indices_f32 = Tensor::randn([2], false) * 0.0 + 1.0;
    
    // Manual index_select simulation
    let indices_u32 = indices_f32; // Will be cast in embedding
    let selected = indices_u32.embedding(weight);
    
    print("Selected:");
    print(selected);
    
    let loss = selected.sumall();
    
    print("Loss:");
    print(loss);
    
    loss.backward();
    
    let gw = weight.grad();
    print("Weight gradient:");
    print(gw);
}
