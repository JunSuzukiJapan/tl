// Vec methods integration test
// Tests: insert, remove, clear

// STDOUT: [insert] 1 10 2 3
// STDOUT: [remove] 10
// STDOUT: [after_remove] 1 2 3
// STDOUT: [clear] 0

fn main() {
    // insert
    let mut v: Vec<i64> = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    v.insert(1, 10);
    print("[insert] ");
    let mut i = 0;
    while i < v.len() {
        if i > 0 {
            print(" ");
        }
        print("{}", v.get(i).unwrap());
        i = i + 1;
    }
    println("");

    // remove
    let removed = v.remove(1);
    println("[remove] {}", removed);
    print("[after_remove] ");
    i = 0;
    while i < v.len() {
        if i > 0 {
            print(" ");
        }
        print("{}", v.get(i).unwrap());
        i = i + 1;
    }
    println("");

    // clear
    v.clear();
    println("[clear] {}", v.len());
}
