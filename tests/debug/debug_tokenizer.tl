
fn main() {
    let tok = Tokenizer::new("~/.llm/models/tokenizer.json");
    let input_str = "Hello";
    
    // Exact prompt from chatbot.tl
    // let prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n".concat(tl_string_concat(input_str, "</s>\n<|assistant|>\n"));
    // Since concat is extern, I'll just hardcode for this test or chain them.
    // "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nHello</s>\n<|assistant|>\n"
    
    let p1 = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nHello</s>\n<|assistant|>\n";
    
    print("Prompt:");
    print(p1);
    
    let tokens = tok.encode(p1);
    print("Tokens:");
    tokens.print_1();
    
    let len = tokens.len();
    print("Length:");
    print(len);
    
    // Print individual IDs to be sure
    let i = 0;
    while i < len {
         let sub = tokens.narrow(0, i, 1);
         let val = sub.item_i64();
         print(val);
         i = i + 1;
    }
}
