fn string_concat(a: String, b: String) -> String { a.concat(b) }

fn main() {
    println("Step 3: Start");
    Param::set_device(Device::Auto);
    println("Set device done");
    
    let config_path = "chatbot_config.txt";
    if File::exists(config_path) {
        println("Config exists");
    } else {
        println("Config missing");
    }

    let s1 = "Hello";
    let s2 = " World";
    let s3 = string_concat(s1, s2);
    println(s3);

    let t_path = "dummy_tokenizer.json";
    let tok = Tokenizer::new(t_path);
    println("Tokenizer created");
}
