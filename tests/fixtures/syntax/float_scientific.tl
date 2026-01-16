// STDOUT: 0.0001\n0.0001\n100000\n-200.5
// STDERR:

fn main() {
    let a = 1e-4;
    let b = 1.0e-4;
    let c = 1E5;
    let d = -2.005E2;

    println(a);
    println(b);
    println(c);
    println(d);
}
