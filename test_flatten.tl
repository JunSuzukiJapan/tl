fn main() {
    let mut vec_of_vecs = Vec::new();
    let mut vec1 = Vec::new();
    vec1.push(1);
    vec1.push(2);
    let mut vec2 = Vec::new();
    vec2.push(3);
    vec2.push(4);
    
    vec_of_vecs.push(vec1);
    vec_of_vecs.push(vec2);
    
    let flat = vec_of_vecs.flatten();
    
    assert(flat.len() == 4, "len == 4");
    assert(flat[0].unwrap() == 1, "0 == 1");
    assert(flat[1].unwrap() == 2, "1 == 2");
    assert(flat[2].unwrap() == 3, "2 == 3");
    assert(flat[3].unwrap() == 4, "3 == 4");
    print("flatten OK\n");
}
