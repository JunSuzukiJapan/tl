fn main() {
    // Option test
    let x = Option::Some(10);
    if x.is_some() {
        print("Option::Some(10) is some\n");
    } else {
        print("Option::Some(10) FAILED (is_some returned false)\n");
    }

    if x.is_none() {
        print("Option::Some(10) FAILED (is_none returned true)\n");
    } else {
        print("Option::Some(10) is not none\n");
    }

    let val = x.unwrap();
    if val == 10 {
        print("Option::unwrap() OK\n");
    } else {
        print("Option::unwrap() FAILED\n");
    }

    let x2 = Option::Some(10);
    let u = x2.unwrap_or(20);
    if u == 10 {
        print("Option::unwrap_or (some) OK\n");
    } else {
        print("Option::unwrap_or (some) FAILED\n");
    }

    let y = Option::None();
    if y.is_none() {
        print("Option::None() is none\n");
    } else {
        print("Option::None() FAILED (is_none returned false)\n");
    }

    // none() type inference might be tricky, let's explicitely type variable if needed, 
    // but here we just assign to y.
    
    // File test
    let fname = "test_instance_methods.txt";
    let f = File::open(fname, "w");
    f.write_string("hello world");
    f.close();
    
    if File::exists(fname) {
        print("File exists OK\n");
    } else {
        print("File exists FAILED\n");
    }
    
    let f2 = File::open(fname, "r");
    let content = f2.read_string();
    if content == "hello world" {
        print("File read OK\n");
    } else {
        print("File read FAILED: ");
        print(content);
        print("\n");
    }
    f2.close();
}
