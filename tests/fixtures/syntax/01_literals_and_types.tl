// Literals and Types Test
fn main() {
    // Integer
    let i: i64 = 42;
    print(i); // STDOUT: 42

    // Float
    let f: f64 = 3.14;
    print(f); // STDOUT: 3.14

    // Boolean
    let b: bool = true;
    if b {
        print("true"); // STDOUT: true
    } else {
        print("false");
    }

    let b2: bool = false;
    if b2 {
        print("false");
    } else {
        print("true"); // STDOUT: true
    }

    // String (Inference)
    let s = "Hello Type";
    print(s); // STDOUT: Hello Type
}
