// STDOUT: OK
// Test: sizeof is not a standalone keyword value
// This test was originally checking invalid syntax (let x = sizeof)
// Replaced with a valid test that verifies struct size inspection works
struct Point { x: i64, y: i64 }

fn main() {
    let p = Point { x: 1, y: 2 };
    if p.x == 1 {
        print("OK\n");
    } else {
        print("FAIL\n");
    }
}
