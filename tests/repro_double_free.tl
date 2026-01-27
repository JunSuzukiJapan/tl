
struct Point {
    x: i64,
    y: i64,
}

fn main() {
    println("Testing Vec<Point> Double Free...");
    let v: Vec<Point> = Vec::new();
    
    let p1 = Point { x: 10, y: 20 };
    v.push(p1);
    
    // access element
    let first = v.get(0);
    println("Got first");
    
    // Scope exit should free p1, v, and first.
    // If v.get(0) aliased p1 without refcounting, and both first/p1 are in scope,
    // or if v frees elements + scope frees p1... we get Double Free.
    println("Exiting main...");
}
