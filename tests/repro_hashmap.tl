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

    let mut map_str: HashMap<String, i64> = HashMap::new();
    map_str.insert("hello", 42);
    map_str.insert("world", 99);

    let v_hello = map_str.get("hello");
    match v_hello {
        Option::Some(v) => println("Found hello: {}", v),
        Option::None => println("hello Not Found"),
    }
    
    let v_world = map_str.get("world");
     match v_world {
        Option::Some(v) => println("Found world: {}", v),
        Option::None => println("world Not Found"),
    }

    let v_unknown = map_str.get("foo");
    match v_unknown {
        Option::Some(v) => println("Found foo: {}", v),
        Option::None => println("foo Not Found"),
    }
}
