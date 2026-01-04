// だいたんに修正: varbuilder_getの第2引数はスカラー値の配列として扱う
fn main() {
    println("=== VarBuilder Simple Test ===");
    
    // Create parameter using varbuilder_get with individual dimensions
    let w = VarBuilder::get("weights", 10, 5);
    println("Parameter created");
    
    // Forward pass
    let y = w + w;
    println("Forward complete");
    
    // Backward
    backward(y);
    println("Backward complete");
    
    // Get gradient  
    let grad_w = VarBuilder::grad("weights");
    println("Got gradientgrad");
    
    // Update parameters
    update_all_params(0.01);
    println("Parameters updated");
    
    println("=== Test Complete ===");
}
