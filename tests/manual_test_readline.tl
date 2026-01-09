
// Test file for tl_read_line
extern fn tl_prompt(prompt: String) -> String;
extern fn tl_print_string(s: String);
extern fn tl_string_concat(s1: String, s2: String) -> String;

fn main() {
    let name = tl_prompt("What is your name? ");
    let msg = tl_string_concat("Hello, ", name);
    print(msg);
    return 0;
}
