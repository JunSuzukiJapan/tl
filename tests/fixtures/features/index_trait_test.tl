// Index / IndexMut trait test
// ユーザー定義型で expr[index] と expr[index] = value をテスト

struct Matrix {
    data: Vec<i64>,
    rows: i64,
    cols: i64,
}

impl Matrix {
    fn new(rows: i64, cols: i64) -> Self {
        let mut data: Vec<i64> = Vec::with_capacity(rows * cols);
        let mut i = 0;
        while i < rows * cols {
            data.push(0);
            i = i + 1;
        }
        Matrix { data: data, rows: rows, cols: cols }
    }

    // Index trait: matrix[i] returns the element
    fn index(self, i: i64) -> i64 {
        self.data.get(i).unwrap()
    }

    // IndexMut trait: matrix[i] = value
    fn set(self, i: i64, value: i64) {
        self.data.set(i, value);
    }
}

fn main() {
    let mut m = Matrix::new(2, 3);

    // IndexMut: write via []= syntax
    m[0] = 10;
    m[1] = 20;
    m[2] = 30;
    m[3] = 40;
    m[4] = 50;
    m[5] = 60;

    // Index: read via [] syntax
    print(m[0]);
    print(m[1]);
    print(m[2]);
    print(m[3]);
    print(m[4]);
    print(m[5]);

    println("Index trait test passed!");
}
