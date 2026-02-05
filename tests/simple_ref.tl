// SKIP: Known compiler issue - generic type inference

struct S { x: i64 }

fn f(s: &S) {
    let x = s.x;
}

fn main() {
    let s = S { x: 10 };
    f(&s);
}
