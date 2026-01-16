struct Point {
    x: f32,
    y: f32
}

fn main() {
    print("Starting struct arena test...");
    let mut i = 0;
    while i < 10 {
        let p = Point { x: 1.0, y: 2.0 };
        let sum = p.x + p.y;
        
        // Manual modulo: i % 2
        let half = i / 2;
        let rem = i - half * 2;
        
        if rem == 0 {
            print(sum);
        }
        i = i + 1;
    }
    print("Struct arena test finished.");
}
