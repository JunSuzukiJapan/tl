// HashMap methods integration test
// Tests: remove, contains_key, keys, values

// STDOUT: [contains_key] true
// STDOUT: [contains_key_neg] false
// STDOUT: [remove_len] 1
// STDOUT: [after_remove] false
// STDOUT: [keys_len] 2
// STDOUT: [values_len] 2

fn main() {
    let mut m: HashMap<i64, i64> = HashMap::new();
    m.insert(1, 10);
    m.insert(2, 20);

    // contains_key
    println("[contains_key] {}", m.contains_key(1));
    println("[contains_key_neg] {}", m.contains_key(99));

    // remove
    m.remove(1);
    println("[remove_len] {}", m.len());
    println("[after_remove] {}", m.contains_key(1));

    // Re-add for keys/values test
    m.insert(3, 30);

    // keys
    let ks = m.keys();
    println("[keys_len] {}", ks.len());

    // values
    let vs = m.values();
    println("[values_len] {}", vs.len());
}
