struct Empty {}

struct Wrapper {
    e: Empty,
}

fn main() {
    let e = Empty {};
    let w = Wrapper { e: Empty {} };
}
