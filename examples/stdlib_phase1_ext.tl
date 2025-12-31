fn test_env_ext() {
    print("Testing Env::set...");
    Env::set("TL_TEST_VAR", "Hello from TL!");
    let val = Env::get("TL_TEST_VAR");
    print(val);
    if val == "Hello from TL!" {
        print("Env::set SUCCESS");
    } else {
        print("Env::set FAILED");
    }
}

fn test_system() {
    print("Testing System...");
    let t1 = System::time();
    System::sleep(0.5);
    let t2 = System::time();
    
    let diff = t2 - t1;
    if diff >= 0.4 {
        print("System::sleep/time SUCCESS");
    } else {
        print("System::sleep/time FAILED");
    }
}

fn test_http() {
    print("Testing Http::download...");
    let success = Http::download("https://httpbin.org/robots.txt", "test_dl.txt");
    if success {
        print("Http::download SUCCESS");
        let f = File::open("test_dl.txt", "r");
        let content = f.read_string();
        f.close();
        // Check if we got something
        if content != "" {
            print("Downloaded content verified");
        }
    } else {
        print("Http::download FAILED");
    }
}

fn main() {
    test_env_ext();
    test_system();
    test_http();
}
