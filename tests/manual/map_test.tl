// Minimal test for Map String handling
fn main() {
    println("Map String test");
    
    let model_path = "~/.llm/models/llama3/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
    println("Loading map...");
    let weights = Map::load(model_path);
    println("Map loaded");
    
    // Test get with literal string
    let tensor_name = "output_norm.weight";
    println("Getting tensor...");
    let norm_w = weights.get_1d(tensor_name);
    println("Got tensor");
}
