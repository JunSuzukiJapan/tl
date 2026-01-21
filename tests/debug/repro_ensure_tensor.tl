
fn main() {
    print("Testing ensure_tensor transformation...");
    let data = [1.0, 2.0, 3.0, 4.0]; 
    let shape = [2, 2]; 
    
    print("Calling reshape...");
    let res = data.reshape(shape);
    
    print("Reshape done.");
}
