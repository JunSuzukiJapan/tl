fn assert(cond: bool, msg: String) {
    if !cond {
        print("Assertion Failed: " + msg);
        System::exit(1);
    }
}

fn test_env() {
    print("Testing Env...");
    let home = Env::get("HOME");
    print("HOME: " + home);
    assert(home.len() > 0, "HOME env var should not be empty");
}

fn test_path() {
    print("Testing Path...");
    let root_exists = Path::exists("/");
    assert(root_exists, "Root path / must exist");
    
    let fake_exists = Path::exists("/non/existent/path/xyz");
    assert(!fake_exists, "Fake path should not exist");
}

fn test_file_io() {
    print("Testing File IO...");
    let path = "temp_test.txt";
    
    // Cleanup
    if File::exists(path) {
        print("Temp file exists, overwriting...");
    }

    // Static Write
    let content = "Hello TL IO World\nLine 2";
    print("Content created. Len: " + String::from_int(content.len()));
    print("Writing to path: " + path);
    let wrote = File::write(path, content);
    print("Write finished");
    assert(wrote, "File::write static failed");
    print("Checking existence...");
    
    // Check
    let exists = File::exists(path);
    print("Exists check done: " + String::from_int(if exists { 1 } else { 0 }));
    assert(exists, "File should exist after write");
    
    // Static Read
    print("Reading file...");
    let read_back = File::read(path);
    print("Read done. content len: " + String::from_int(read_back.len()));
    
    print("Asserting read_back == content...");
    assert(read_back == content, "Static read content mismatch");
    print("Assert passed!");
    
    // Instance Open & Read
    print("Opening file instance...");
    let f = File::open(path, "r");
    print("File opened.");
    let s = f.read_string();
    print("Instance read done.");
    assert(s == content, "Instance read_string content mismatch");
    f.close();
    print("Instance closed.");
    
    // Instance Open & Write
    let f2 = File::open(path, "w");
    print("Opening file instance 2.");
    let new_content = "Overwritten Content";
    f2.write_string(new_content);
    f2.close();
    
    let read_new = File::read(path);
    assert(read_new == new_content, "Instance write_string content mismatch");
    
    // Binary Read
    let bin = File::read_binary(path);
    let size_val = bin.len();
    let size_str = String::from_int(size_val);
    print("Binary size: " + size_str);
    assert(bin.len() == new_content.len(), "Binary length mismatch");
}

fn test_network() {
    print("Testing Network (Basic)...");
    print("Skipping actual network request to avoid flakiness.");
}

fn main() {
    test_env();
    test_path();
    test_file_io();
    test_network();
    print("IO Tests Passed");
}
