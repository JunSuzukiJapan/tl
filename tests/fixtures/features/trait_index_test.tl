// Test implementing builtin Index trait for custom type

struct Matrix {
    rows: i64,
    cols: i64,
    a11: i64,
    a12: i64,
    a21: i64,
    a22: i64,
}

impl Index<i64, i64> for Matrix {
    fn index(self, idx: i64) -> i64 {
        if idx == 0 {
            self.a11
        } else if idx == 1 {
            self.a12
        } else if idx == 2 {
            self.a21
        } else {
            self.a22
        }
    }
}

fn main() {
    let m = Matrix { rows: 2, cols: 2, a11: 10, a12: 20, a21: 30, a22: 40 };
    
    // Use the Index trait method directly
    println(String::from_int(m.index(0)));
    println(String::from_int(m.index(1)));
    println(String::from_int(m.index(2)));
    println(String::from_int(m.index(3)));
}
