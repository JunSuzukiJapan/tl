
fn main() {
    let model_path = "test_path";
    println("Start");

    if !File::exists(model_path) {
        println("Not found");
        loop {
            let choice = read_line("Select> ");
            if choice == "1" {
                println("One");
                return;
            }
        }
    }
}
