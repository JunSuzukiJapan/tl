// STDOUT: Path check passed
// STDOUT: System time passed

let p = Path::new(".");
if p.is_dir() {
    println("Path check passed");
}

let t = System::time();
println("System time passed");
