mod mnist_common;
use mnist_common::load_mnist_images;
use mnist_common::load_mnist_labels;

fn main() {
    print("Loading data...");
    let x = load_mnist_images("examples/tasks/mnist/data/train-images-idx3-ubyte");
    let y = load_mnist_labels("examples/tasks/mnist/data/train-labels-idx1-ubyte");
    
    let bx = x.slice(0, 10);
    let by = y.slice(0, 10);
    
    print("Creating weights...");
    let w = Tensor::randn([784, 10], true);
    
    print("Forward...");
    let logits = matmul(bx, w);
    
    print("Cross Entropy...");
    let loss = cross_entropy(logits, by);
    print("Loss:"); print(loss.item());
    
    print("Backward...");
    loss.backward();
    
    let g = w.grad();
    print("Grad sum abs:");
    print(g.abs().sum().item());
    
    if g.abs().sum().item() == 0.0 {
        print("GRADIENT MISSING!");
    } else {
        print("Gradient OK");
    }
}
