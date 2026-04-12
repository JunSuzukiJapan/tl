// MNIST Inference - MLP Model

use mnist_common::*;

fn main() {
    print("=== MNIST Inference (MLP) ===");
    
    let mut test_img_path = "examples/tasks/mnist/data/t10k-images-idx3-ubyte";
    if File::exists("data/t10k-images-idx3-ubyte") {
        test_img_path = "data/t10k-images-idx3-ubyte";
    }
    let mut test_lbl_path = "examples/tasks/mnist/data/t10k-labels-idx1-ubyte";
    if File::exists("data/t10k-labels-idx1-ubyte") {
        test_lbl_path = "data/t10k-labels-idx1-ubyte";
    }
    
    let test_x = load_mnist_images(test_img_path).unwrap();
    let test_y = load_mnist_labels(test_lbl_path).unwrap();
    
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
