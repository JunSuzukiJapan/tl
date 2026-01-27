
struct Point {
    x: i64,
    y: i64,
}

fn test_scalar_vec() {
    println("Testing Vec<i64>...");
    let mut v: Vec<i64> = Vec::new();
    v.push(10);
    v.push(20);
    // Move
    let v2 = v;
    println(v2.get(0));
    println(v2.get(1));

    println("Testing Vec<f32>...");
    let mut vf: Vec<f32> = Vec::new();
    vf.push(1.5);
    vf.push(2.5);
    let vf2 = vf;
    println(vf2.get(0));

    println("Testing Vec<bool>...");
    let mut vb: Vec<bool> = Vec::new();
    vb.push(true);
    vb.push(false);
    let vb2 = vb;
    if vb2.get(0) { println("true"); } else { println("false"); }
}

fn test_string_vec() {
    println("Testing Vec<String>...");
    let mut vs: Vec<String> = Vec::new();
    vs.push("Hello");
    vs.push("World");
    
    // Move Check
    let vs2 = vs;
    println(vs2.get(0));
    println(vs2.get(1));
}

fn test_tensor_vec() {
    println("Testing Vec<Tensor>...");
    let mut vt: Vec<Tensor<f32, 1>> = Vec::new();
    let t1: Tensor<f32, 1> = [1.0, 2.0];
    // println(t1); 
    vt.push(t1);
    
    let t2: Tensor<f32, 1> = [3.0, 4.0];
    vt.push(t2);

    // Move Check
    let vt2 = vt;
    let t_out: Tensor<f32, 1> = vt2.get(0); 
    // println(t_out);
    println("Tensor Vec moved successfully");
}

fn test_struct_vec() {
    println("Testing Vec<Point>...");
    let mut vp: Vec<Point> = Vec::new();
    let p1 = Point { x: 1, y: 2 };
    vp.push(p1);
    
    // Move Check
    let vp2 = vp;
    let p_out = vp2.get(0);
    println(p_out.x);
    println(p_out.y);
}

fn test_nested_vec() {
    println("Testing Vec<Vec<i64>>...");
    let mut v_nested: Vec<Vec<i64>> = Vec::new();
    
    let mut inner1: Vec<i64> = Vec::new();
    inner1.push(1);
    
    let mut inner2: Vec<i64> = Vec::new();
    inner2.push(2);
    
    v_nested.push(inner1);
    v_nested.push(inner2);
    
    // Move Check
    let v_nested2 = v_nested;
    let out1 = v_nested2.get(0);
    println(out1.get(0));
}

fn main() {
    test_scalar_vec();
    test_string_vec();
    test_tensor_vec();
    test_struct_vec();
    test_nested_vec();
    println("All Vec variations passed!");
}
