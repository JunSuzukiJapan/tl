// STDOUT: File read success
// STDOUT: Hello from TL

fn main() {
    let f = File::open("tests/fixtures/features/hello.txt", "r");
    let s = f.read_string();
    println("File read success");
    println(s);
    f.close();
}
