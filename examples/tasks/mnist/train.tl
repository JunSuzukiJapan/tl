// MNIST Training - MLP Model with LR Scheduling
use mnist_common::*;

fn main() {
    print("=== MNIST Training (MLP) ===");
    Param::set_device(Device::Auto);
    
    let train_x = load_mnist_images("examples/tasks/mnist/data/train-images-idx3-ubyte");
    let train_y = load_mnist_labels("examples/tasks/mnist/data/train-labels-idx1-ubyte");
    
    // 2-layer MLP: 784 -> 256 -> 10
    let mut model = MLP::new(784, 256, 10);
    let batch_size = 64;
    
    // 60000 / 64 = 937.5 -> 937 batches
    let num_batches = 937; 
    let num_epochs = 30;
    
    print("Starting training loop...");
    
    let mut epoch = 0;
    while epoch < num_epochs {
        // Learning rate schedule: warm start then decay
        let lr = if epoch < 10 {
            0.05
        } else if epoch < 20 {
            0.01
        } else {
            0.003
        };
        
        let mut b = 0;
        let mut running_loss = 0.0;
        
        while b < num_batches {
            let start = b * batch_size;
            let bx = train_x.slice(0, start, start + batch_size, 1);
            let by = train_y.slice(0, start, start + batch_size, 1);
            
            let logits = model.forward(bx);
            let loss = logits.cross_entropy(by);
            
            loss.backward();
            
            model = model.step(lr);
            
            // Clear autograd graph to prevent memory accumulation
            Tensor::clear_grads();
            
            running_loss = running_loss + loss.item();
            
            b = b + 1;
        }
        
        println("Epoch {} (lr={}): Avg Loss: {}", epoch, lr, running_loss / (num_batches as f32));
        
        epoch = epoch + 1;
    }
    
    print("Saving model weights...");
    Param::save(model, "mnist_weights.safetensors");
    print("Model saved.");
}
