// MNIST Inference - MLP Model

use mnist_common::*;

fn main() {
    print("=== MNIST Inference (MLP) ===");
    
    let test_x = load_mnist_images("data/t10k-images-idx3-ubyte");
    let test_y = load_mnist_labels("data/t10k-labels-idx1-ubyte");
    
    // Initialize model structure (must match training)
    let model = MLP::new(784, 256, 10);
    
    print("Loading weights...");
    Param::load(model, "mnist_weights.safetensors");
    
    print("Evaluating on test set (first 1000 samples)...");
    
    let mut correct = 0;
    let total = 1000;
    let mut i = 0;
    
    while i < total {
        let x = test_x.slice(0, i, i + 1, 1);
        let y = test_y.slice(0, i, i + 1, 1); 
        
        let logits = model.forward(x);
        let preds = logits.argmax(1);
        
        if preds.item() == y.item() {
             correct = correct + 1;
        }
        
        i = i + 1;
    }

    println("Accuracy: {}/{}", correct, total);
}
