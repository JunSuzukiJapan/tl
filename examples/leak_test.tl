
fn main() {
    let i = 0.0;
    while i < 1000.0 {
        let A = [[1.0, 2.0], [3.0, 4.0]];
        let B = A + A;
        i = i + 1.0;
    }
    print(i);
}
