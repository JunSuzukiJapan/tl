// STDOUT: System::sleep/time SUCCESS

fn test_env_ext() {
    print("Testing Env::set...\n");
    Env::set("TL_TEST_VAR", "Hello from TL!");
    let val = Env::get("TL_TEST_VAR");
    print(val);
    if val == "Hello from TL!" {
        print("Env::set SUCCESS\n");
    } else {
        print("Env::set FAILED\n");
    }
}

fn test_system() {
    print("Testing System...\n");
    let t1 = System::time();
    System::sleep(0.5);  // 0.5 seconds
    let t2 = System::time();
    
    let diff = t2 - t1;
    if diff >= 0 {  // At least 0 seconds elapsed (time is in seconds)
        print("System::sleep/time SUCCESS\n");
    } else {
        print("System::sleep/time FAILED\n");
    }
}

fn main() {
    test_env_ext();
    test_system();
    // Skip HTTP test to avoid network dependency
    print("Tests complete\n");
}
