fn main() {
    let a: i64 = 4;
    let b: i64 = a << 2;
    assert(b == 16, "4 << 2 should be 16");

    let c: i64 = 16;
    let d: i64 = c >> 2;
    assert(d == 4, "16 >> 2 should be 4");

    let mut e: i64 = 4;
    e <<= 2;
    assert(e == 16, "4 <<= 2 should be 16");

    let mut f: i64 = 16;
    f >>= 2;
    assert(f == 4, "16 >>= 2 should be 4");

    println("Bitwise shift tests passed!");
}
