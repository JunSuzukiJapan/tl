// Minimal test for Http and Image

fn main() {
    print("Testing Http and Image classes...");
    
    // Test Http::download
    let url = "https://raw.githubusercontent.com/rasbt/mnist-pngs/main/train/0/16585.png";
    let path = "examples/tasks/mnist/data/test.png";
    
    let ok = Http::download(url, path);
    print("Download result:");
    print(ok as f32);
    
    // Test Image::load_grayscale  
    let pixels = Image::load_grayscale(path);
    let len = tl_vec_u8_len(pixels);
    print("Pixel count:");
    print(len as f32);
    
    tl_vec_u8_free(pixels);
    print("Done!");
}
