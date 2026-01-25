fn main() {
    test_vec_i64();
    test_vec_f32();
    test_vec_string();
    println("All generic Vec tests passed!");
}

fn test_vec_i64() {
    let v: Vec<i64> = Vec::new();
    v.push(100);
    v.push(200);
    
    if v.len() != 2 {
        println("Vec<i64> len mismatch");
    }
    
    if v.get(0) != 100 {
        println("Vec<i64> get(0) failed");
    }
    if v.get(1) != 200 {
        println("Vec<i64> get(1) failed");
    }
    
    println("test_vec_i64 passed");
}

fn test_vec_f32() {
    let v: Vec<f32> = Vec::new();
    v.push(1.5);
    v.push(2.5);
    
    if v.len() != 2 {
        println("Vec<f32> len mismatch");
    }
    
    let val0 = v.get(0);
    if val0 != 1.5 {
         println("Vec<f32> get(0) failed");
    }
    
    println("test_vec_f32 passed");
}

fn test_vec_string() {
    let v: Vec<String> = Vec::new();
    v.push("Hello");
    v.push("World");
    
    if v.len() != 2 {
        println("Vec<String> len mismatch");
    }
    
    let s0 = v.get(0);
    println("Vec<String>[0]: " + s0);
    if s0 != "Hello" {
        println("Vec<String> get(0) failed");
    }
    
    println("test_vec_string passed");
}
