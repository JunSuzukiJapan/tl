fn main() {
    let mut vec1 = Vec::new();
    vec1.push(1);
    vec1.push(2);
    
    let mut vec2 = Vec::new();
    vec2.push(3);
    vec2.push(4);
    
    let zipped = vec1.zip(vec2);
    
    assert(zipped.len() == 2, "len == 2");
    
    let tup0 = zipped[0].unwrap();
    let tup1 = zipped[1].unwrap();
    assert(tup0.0 == 1, "0.0");
    assert(tup0.1 == 3, "0.1");
    assert(tup1.0 == 2, "1.0");
    assert(tup1.1 == 4, "1.1");
    print("zip OK\n");
}
