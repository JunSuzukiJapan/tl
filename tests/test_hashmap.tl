fn main() {
    let map: HashMap<String, Tensor> = HashMap::new();
    let t1 = [42.0];
    map.insert("hello", t1); 
    
    let t2 = [100.0];
    map.insert("world", t2);

    let val = map.get("hello");
    print("Value for hello:");
    print(val); 

    let len = map.len();
    print("Length:");
    print(len);

    let t3 = [99.0];
    map.insert("hello", t3);
    let val2 = map.get("hello");
    print("Updated:");
    print(val2);

    map.clear();
    print("Cleared length:");
    print(map.len());
}
