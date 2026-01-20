extern fn tl_file_exists_i64(path: String) -> i64;
extern fn tl_read_file(path: String) -> String;
// ... (SKIP for brevity, just minimal test)
fn main() { println("Main at top"); }
struct S {}
