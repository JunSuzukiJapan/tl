extern fn tl_file_exists_i64(path: String) -> i64;
extern fn tl_read_line(prompt: String) -> String;
extern fn println(s: String);

fn main() {
    let model_path = "test_path";
    println("Start");

    if tl_file_exists_i64(model_path) == 0 {
        println("Not found");
        loop {
            let choice = tl_read_line("Select> ");
            if choice == "1" {
                println("One");
                return;
            }
        }
    }
}
