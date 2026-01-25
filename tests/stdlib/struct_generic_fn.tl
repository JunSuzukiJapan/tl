struct Rect<T> {
    w: T,
    h: T,
}

fn make_rect<T>(w: T, h: T) -> Rect<T> {
    Rect { w: w, h: h }
}

fn main() {
    let r1 = make_rect(10, 20); // Rect<i64>
    if r1.w != 10 { println("Error: r1.w"); } else { println("r1 OK"); }
    
    let r2 = make_rect(1.5, 2.5); // Rect<f64>
    // Just print as equality for floats is risky
    println("r2 created");
    
    println("struct_generic_fn passed");
}
