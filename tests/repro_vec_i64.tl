fn main() {
    println("Testing Vec<i64>...");
    let v: Vec<i64> = Vec::new();
    v.push(10);
    v.push(20);
    
    let first = v.get(0);
    println("Got first");
    
    // implicit free
    println("Exiting main...");
}
