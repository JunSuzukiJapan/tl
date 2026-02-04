// SKIP: Path struct codegen not implemented
fn main() {
    print("Testing Path...");
    let p = Path::new("src");
    print("Path: " + p.to_string());
    
    if p.exists() {
        print("src exists");
    }
    if p.is_dir() {
        print("src is directory");
    }
    
    let p2 = p.join("main.rs");
    print("Joined path: " + p2.to_string());
    if p2.exists() {
        print("src/main.rs exists");
    }
    if p2.is_file() {
        print("src/main.rs is file");
    }
    
    p.free();
    p2.free();
    print("Path test SUCCESS");
}
