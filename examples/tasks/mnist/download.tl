// MNIST Dataset Downloader
// Downloads the MNIST dataset files from the official mirror

fn download_mnist() {
    let base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    let data_dir = "examples/tasks/mnist/data/";
    
    // Create data directory (using Path operations)
    let data_path = Path::new(data_dir);
    
    // Training images
    let train_images_url = base_url + "train-images-idx3-ubyte.gz";
    let train_images_path = data_dir + "train-images-idx3-ubyte.gz";
    
    // Training labels
    let train_labels_url = base_url + "train-labels-idx1-ubyte.gz";
    let train_labels_path = data_dir + "train-labels-idx1-ubyte.gz";
    
    // Test images
    let test_images_url = base_url + "t10k-images-idx3-ubyte.gz";
    let test_images_path = data_dir + "t10k-images-idx3-ubyte.gz";
    
    // Test labels
    let test_labels_url = base_url + "t10k-labels-idx1-ubyte.gz";
    let test_labels_path = data_dir + "t10k-labels-idx1-ubyte.gz";
    
    print("Downloading MNIST dataset...");
    
    // Download each file
    if http_download(train_images_url, train_images_path) {
        print("Downloaded train-images");
    }
    
    if http_download(train_labels_url, train_labels_path) {
        print("Downloaded train-labels");
    }
    
    if http_download(test_images_url, test_images_path) {
        print("Downloaded test-images");
    }
    
    if http_download(test_labels_url, test_labels_path) {
        print("Downloaded test-labels");
    }
    
    print("Download complete!");
}

fn main() {
    download_mnist();
}
