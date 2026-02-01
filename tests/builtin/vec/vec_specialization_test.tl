fn main() {
    println("Testing Vec specialization...");

    // 1. Vec<i64>
    println("Testing Vec<i64>...");
    let mut v_i64: Vec<i64> = Vec<i64>::new();
    v_i64.push(10);
    v_i64.push(20);
    v_i64.push(30);
    println(v_i64.len());
    println(v_i64.get(0));
    println(v_i64.pop());

    // 2. Vec<f32>
    println("Testing Vec<f32>...");
    let mut v_f32: Vec<f32> = Vec<f32>::new();
    v_f32.push(1.5);
    v_f32.push(2.5);
    println(v_f32.len());
    // println(v_f32.get(0)); // float printing might be limited?

    // 3. Vec<u8>
    println("Testing Vec<u8>...");
    let mut v_u8: Vec<u8> = Vec<u8>::new();
    v_u8.push(10 as u8);
    v_u8.push(255 as u8);
    println(v_u8.len());
    // println(v_u8.get(0)); 
    println(v_u8.pop() as i64);

    // 4. Vec<String>
    println("Testing Vec<String>...");
    let mut v_str: Vec<String> = Vec<String>::new();
    v_str.push("Hello");
    v_str.push("World");
    println(v_str.len());
    println(v_str.get(0));
    println(v_str.pop());

    println("Specialize tests passed!");
}
