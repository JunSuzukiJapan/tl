fn main() {
    let mut v: Vec<i64> = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    
    if v.len() != 4 {
        print("Error: len mismatch");
    }
    
    let first = v[0];
    if first != 1 {
        print("Error: v[0] mismatch");
    }

    let third = v[2];
    if third != 3 {
        print("Error: v[2] mismatch");
    }

    let popped = v.pop();
    if popped != 4 {
         print("Error: pop mismatch");
    }
    
    if v.len() != 3 {
         print("Error: len after pop mismatch");
    }

    let mut nested: Vec<Vec<i64>> = Vec::new();
    nested.push(v);
    
    let v2 = nested[0];
    if v2[0] != 1 {
        print("Error: nested[0][0] mismatch");
    }

    print("Vec refactor verification pass");
}
