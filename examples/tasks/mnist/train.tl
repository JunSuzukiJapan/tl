// MNIST Training
mod mnist_common;

use mnist_common::Linear;
use mnist_common::load_mnist_images;
use mnist_common::load_mnist_labels;

fn main() {
    print("=== MNIST Training ===");
    
    let train_x = load_mnist_images("examples/tasks/mnist/data/train-images-idx3-ubyte");
    let train_y = load_mnist_labels("examples/tasks/mnist/data/train-labels-idx1-ubyte");
    
    let model = Linear::new(784, 10);
    let lr = 0.5;
    let batch_size = 64;
    
    let num_batches = 10; // 60000 / batch_size; 
    let epoch = 0;
    
    print("Starting training loop...");
    
    while epoch < 3 {
        let b = 0;
        let running_loss = 0.0;
        
        while b < num_batches {
            let start = b * batch_size;
            let bx = train_x.slice(start, batch_size);
            let by = train_y.slice(start, batch_size);
            
            let logits = model.forward(bx);
            let loss = cross_entropy(logits, by);
            
            loss.backward();
            model = model.step(lr);
            
            running_loss = running_loss + loss.item();
            
            b = b + 1;
        }
        
        print("Epoch complete:"); print(epoch);
        print("Avg Loss:"); print(running_loss / (num_batches as f32));
        
        epoch = epoch + 1;
    }
    
    print("Saving model weights...");
    save_weights(model, "examples/tasks/mnist/mnist_weights.safetensors");
    print("Model saved.");
}
