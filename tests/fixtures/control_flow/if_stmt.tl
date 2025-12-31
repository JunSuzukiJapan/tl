// STDOUT: True Branch
// STDOUT: False Branch

fn main() {
    let x = 10;
    if x > 5 {
        print("True Branch");
    } else {
        print("False Branch");
    }

    if x < 5 {
        print("Wrong Branch");
    } else {
        print("False Branch");
    }
}
