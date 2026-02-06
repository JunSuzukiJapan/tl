fn main() {
    println("Test 1: HashMap::new");
    let m: HashMap<i64, i64> = HashMap::new();
    println("Test 2: len");
    let l = m.len();
    println("len: {}", l);
    println("Test 3: insert");
    m.insert(1, 100);
    println("Test 4: get");
    let v = m.get(1);
    match v {
        Option<i64>::Some(val) => println("got: {}", val),
        Option<i64>::None => println("not found"),
    }
    println("Test 5: Done");
}
