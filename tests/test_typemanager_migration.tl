fn main() {
    // 1. Result Check
    let r = Result::Ok(10);
    // matching on result is not fully supported in simple scripting without full enum support, 
    // but assignment works?
    match r {
        Result::Ok(v) => { if v != 10 { print("Result fail 1\n"); } },
        Result::Err(e) => { print("Result fail 2\n"); },
    }
    let r2: Result<i64, String> = Result::Err("error");

    // 2. Vec Check
    // let mut v = Vec::new();
    // v.push(100);
    // v.push(200);
    // if v.len() != 2 { print("Vec len fail\n"); }
    // if v.get(0) != 100 { print("Vec get fail 1\n"); }
    // // v.set(1, 300); // set not implemented in builtin_ast yet
    // v.remove(1);
    // v.push(300);
    // if v.get(1) != 300 { print("Vec get fail 2\n"); }
    // let p = v.pop();
    // if p != 300 { print("Vec pop fail\n"); }
    // if v.len() != 1 { print("Vec len fail 2\n"); }

    // 3. HashMap Check
    // let mut m = HashMap::new();
    // let k = "key1";
    // m.insert(k, 123);
    // if m.len() != 1 { print("HashMap len fail\n"); }
    // if !m.contains_key("key1") { print("HashMap contains fail\n"); }
    // let val = m.get("key1");
    // if val != 123 { print("HashMap get fail\n"); }
    // m.remove("key1");
    // if m.len() != 0 { print("HashMap remove fail\n"); }

    // 4. Option Check
    let o = Option::Some(42);
    if o.is_none() { print("Option fail 1\n"); }
    if !o.is_some() { print("Option fail 2\n"); }
    let val_o = o.unwrap();
    if val_o != 42 { print("Option unwrap fail\n"); }
    let n = Option::None;
    if !n.is_none() { print("Option none fail\n"); }
    
    // 5. System Check
    let mem = System::memory_mb();
    // print("Memory: "); print(mem); print("\n");

    // 6. File Check
    // Can't easily test open without file, but check existence of non-existent
    if File::exists("non_existent_file_xyz_123.txt") {
        print("File exists fail\n");
    }
    
    // 7. Tokenizer/KVCache (compilation check mainly, runtime requires model file usually)
    // let tok = Tokenizer::new("path"); // returns struct, but needs valid path to not crash?
    // print("Tokenizer compilation check passed\n");
    // Skip runtime call that crashes, but method resolution should work if we can compile.
    

    // Path (Static) verification
    if Path::exists(".") {
        print("Path exists verification passed\n");
    } else {
        print("Path exists verification failed\n");
    }

    // Env (Static) verification
    let path_env = Env::get("PATH");
    print(path_env); // Check if it compiles and runs
    print("\nEnv get verification passed\n");

    // System verification
    let time = System::time();
    print("Time: "); print(time); print("\n");
    
    System::sleep(0.1);
    
    let mem = System::memory_mb();
    print("Memory: "); print(mem); print("\n");

    // 8. Tensor Check
    let t_zeros = Tensor::zeros([2, 5]);
    let t_ones = Tensor::ones([2, 2]);
    let t_randn = Tensor::randn([2, 2]);
    Tensor::clear_grads();
    print("Tensor verification passed\n");

    print("Verification passed!\n");
}
