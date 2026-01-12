// MNIST Training
mod mnist_common;

use mnist_common::Linear;
use mnist_common::load_mnist_images;
use mnist_common::load_mnist_labels;

fn main() {
    print("=== MNIST Training ===");
    set_device(Device::Auto);
    
    let train_x = load_mnist_images("examples/tasks/mnist/data/train-images-idx3-ubyte");
    let train_y = load_mnist_labels("examples/tasks/mnist/data/train-labels-idx1-ubyte");
    
    let model = Linear::new(784, 10);
    // let lr = 0.1;
    let lr = 0.05; // Slightly reduced for stability with more updates
    let batch_size = 64;
    
    // 60000 / 64 = 937.5 -> 937 batches
    let num_batches = 937; 
    let epoch = 5;
    
    print("Starting training loop...");
    
    // Check initial weights (pixel 400 - center of image)
    print("Initial W (row 400):");
    print(model.W.slice(400, 1));
    
    while epoch < 10 {  // 10 epochs
        let b = 0;
        let running_loss = 0.0;
        
        while b < num_batches {
            let start = b * batch_size;
            let bx = train_x.slice(start, batch_size);
            
            if b == 0 {
                // Check input scale via sum
                print("Input Sum (Batch 0):");
                print(bx.sum().item());
            }

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
        
        print("W check (row 400):");
        print(model.W.slice(400, 1));

        epoch = epoch + 1;
    }
    
    print("Saving model weights...");
    save_weights(model, "examples/tasks/mnist/mnist_weights.safetensors");
    print("Model saved.");
}
