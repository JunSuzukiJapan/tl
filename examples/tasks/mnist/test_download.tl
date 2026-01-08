// Minimal test for image loading

fn main() {
    print("Testing image loading...");
    
    // Download test image
    let url = "https://raw.githubusercontent.com/rasbt/mnist-pngs/main/train/0/16585.png";
    let path = "examples/tasks/mnist/data/test.png";
    
    http_download(url, path);
    
    print("Download complete");
}
