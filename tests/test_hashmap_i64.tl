// HashMap<i64, i64> test only (no String)
fn main() {
    let mut map: HashMap<i64, i64> = HashMap::new();
    map.insert(1, 100);
    map.insert(2, 200);
    
    let v1 = map.get(1);
    match v1 {
        Option::Some(v) => println("Found 1: {}", v),
        Option::None => println("1 Not Found"),
    }
    
    let v3 = map.get(3);
    match v3 {
        Option::Some(v) => println("Found 3: {}", v),
        Option::None => println("3 Not Found"),
    }
    
    println("HashMap<i64,i64> test complete");
}
