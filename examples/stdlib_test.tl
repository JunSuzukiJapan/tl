// Basic File I/O Test

// Declare functions (until implicit prelude is ready)
// These should match the C signatures in builtins.rs

// Manual decls for testing raw linkage
// In the future, "File" struct will have methods

fn test_file_io() {
    print("Testing File I/O...");
    let path = "test_output.txt";
    let content = "Hello TensorLogic StdLib!";
    
    // Write
    let f_w = File::open(path, "w");
    f_w.write_string(content);
    f_w.close();
    
    // Read
    let f_r = File::open(path, "r");
    let read_content = f_r.read_string();
    f_r.close();
    
    print(read_content);
    if read_content == content {
        print("File I/O SUCCESS");
    } else {
        print("File I/O FAILED");
    }
}

fn test_env() {
    print("Testing Env...");
    // Assuming PATH exists
    let path_env = Env::get("PATH");
    // Just print for visual validation
    // print(path_env); 
    if path_env != "" {
        print("Env SUCCESS (PATH found)");
    } else {
        print("Env FAILED");
    }
}

fn main() {
    test_file_io();
    test_env();
    // HTTP test skipped to avoid network dependency in basic CI, 
    // unless user explicitly runs it.
}
