// MNIST Dataset Downloader
// Downloads the MNIST dataset files from the official mirror

// File is built-in

fn main() {
    let base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    let data_dir = "examples/tasks/mnist/data/";
    
    println("Downloading MNIST dataset...");
    
    // Training images
    let train_images_url = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz";
    let train_images_path = "examples/tasks/mnist/data/train-images-idx3-ubyte.gz";
    if File::download(train_images_url, train_images_path) {
        println("Downloaded train-images");
    }
    
    // Training labels
    let train_labels_url = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz";
    let train_labels_path = "examples/tasks/mnist/data/train-labels-idx1-ubyte.gz";
    if File::download(train_labels_url, train_labels_path) {
        println("Downloaded train-labels");
    }
    
    // Test images
    let test_images_url = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz";
    let test_images_path = "examples/tasks/mnist/data/t10k-images-idx3-ubyte.gz";
    if File::download(test_images_url, test_images_path) {
        println("Downloaded test-images");
    }
    
    // Test labels
    let test_labels_url = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz";
    let test_labels_path = "examples/tasks/mnist/data/t10k-labels-idx1-ubyte.gz";
    if File::download(test_labels_url, test_labels_path) {
        println("Downloaded test-labels");
    }
    
    println("Download complete!");
}
