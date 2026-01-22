fn main() {
    let mut x = 10;
    x += 5;
    print(); // dummy
    // x.print() should be 15 if x was Tensor, but x is I64.
    // Let's just use it in a way that proves it works.
    if x == 15 {
        "Scalar variable += works".print();
    }

    struct Point { x: i64 }
    let mut p = Point { x: 10 };
    p.x += 5;
    if p.x == 15 {
        "Scalar field += works".print();
    }
}
