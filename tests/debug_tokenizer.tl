extern fn tl_tokenizer_new(path: String) -> i64;
extern fn tl_tokenizer_encode(tok: i64, prompt: String) -> Tensor<i64, 1>;
extern fn tl_tensor_print_1(t: Tensor<i64, 1>);
extern fn tl_tensor_len(t: Tensor<i64, 1>) -> i64;
extern fn tl_tensor_item_i64(t: Tensor<i64, 1>) -> i64;
extern fn tl_tensor_narrow(t: Tensor<i64, 1>, d: i64, s: i64, l: i64) -> Tensor<i64, 1>;

fn main() {
    let tok = tl_tokenizer_new("~/.llm/models/tokenizer.json");
    let input_str = "Hello";
    
    // Exact prompt from chatbot.tl
    // let prompt = tl_string_concat("<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n", tl_string_concat(input_str, "</s>\n<|assistant|>\n"));
    // Since concat is extern, I'll just hardcode for this test or chain them.
    // "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nHello</s>\n<|assistant|>\n"
    
    let p1 = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nHello</s>\n<|assistant|>\n";
    
    print("Prompt:");
    print(p1);
    
    let tokens = tl_tokenizer_encode(tok, p1);
    print("Tokens:");
    tl_tensor_print_1(tokens);
    
    let len = tl_tensor_len(tokens);
    print("Length:");
    print(len);
    
    // Print individual IDs to be sure
    let i = 0;
    while i < len {
         let sub = tl_tensor_narrow(tokens, 0, i, 1);
         let val = tl_tensor_item_i64(sub);
         print(val);
         i = i + 1;
    }
}
