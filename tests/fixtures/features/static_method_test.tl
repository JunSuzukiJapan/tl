// STDOUT: Path check passed
// STDOUT: System time passed

fn main() {
    let p = Path::new(".");
    if p.is_dir() {
        println("Path check passed");
    }

    let t = System::time();
    println("System time passed");
}
