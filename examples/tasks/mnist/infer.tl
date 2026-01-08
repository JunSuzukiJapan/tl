// MNIST Inference
mod mnist_common;

use mnist_common::Linear;
use mnist_common::load_mnist_images;
use mnist_common::load_mnist_labels;

fn main() {
    print("=== MNIST Inference ===");
    
    let test_x = load_mnist_images("examples/tasks/mnist/data/t10k-images-idx3-ubyte");
    let test_y = load_mnist_labels("examples/tasks/mnist/data/t10k-labels-idx1-ubyte");
    
    // Initialize model structure
    let model = Linear::new(784, 10);
    
    print("Loading weights...");
    load_weights(model, "examples/tasks/mnist/mnist_weights.safetensors");
    
    print("Evaluating on test set (first 1000 samples)...");
    
    let correct = 0;
    let total = 1000;
    let i = 0;
    
    while i < total {
        let x = test_x.slice(i, 1);
        let y = test_y.slice(i, 1); 
        
        let logits = model.forward(x);
        let preds = logits.argmax(1, false);
        
        // preds is float (index), y is int (label)
        // item() returns float for both (due to runtime casting)
        if preds.item() == y.item() {
             correct = correct + 1;
        }
        
        i = i + 1;
    }

    print("Accuracy:");
    print(correct);
    print("/");
    print(total);
}
