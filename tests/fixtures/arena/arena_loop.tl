fn main() {
    print("Starting loop test...");
    let i = 0;
    while i < 10 {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let z = x + y;
        
        // Manual modulo: i % 2  => i - (i / 2) * 2
        let half = i / 2;
        let rem = i - half * 2;
        
        if rem == 0 {
            print(z);
        }
        i = i + 1;
    }
    print("Loop test finished.");
}
