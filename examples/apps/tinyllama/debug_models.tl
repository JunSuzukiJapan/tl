fn main() {
    println("Debugging Model Loading...");
    
    // Test 1: tokenizer.json
    // Use full path to verify if ~ expansion is the issue
    // User said: ~/.llm/models/tinyllama/tokenizer.json
    // I will try assuming the user's home dir structure or just use the ~ path first to reproduce.
    
    // Check file existence first using File API if possible (though io.tl definition of exists takes String)
    /*
    if File::exists(tok_path) {
        println("File exists (via File::exists)");
    } else {
        println("File::exists returned false!");
    }
    */

    let tok = Tokenizer::new(tok_path);
    println("Tokenizer loaded.");
    println("Vocab size: ".concat(String::from_int(tok.vocab_size())));

    // Test 2: Model GGUF
    let model_path = "/Users/junsuzuki/.llm/models/tinyllama/tinyllama-1.1b-chat-q4_0.gguf";
    println("Loading Model from: ".concat(model_path));
    
    /*
    if File::exists(model_path) {
         println("Model file exists.");
    } else {
         println("Model file NOT found.");
    }
    */

    let weights = Map::load(model_path);
    println("Model Map loaded.");
    
    let emb = weights.get("token_embd.weight");
    println("Got embedding tensor.");
    emb.print_1();
    
    println("Debug Complete.");
}
