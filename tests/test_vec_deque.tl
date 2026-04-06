fn main() {
    let mut deque: VecDeque<i64> = VecDeque::new();
    
    // Test push_back and push_front
    deque.push_back(10);
    deque.push_back(20);
    deque.push_front(30);
    deque.push_front(40);
    
    // Order should be: 40, 30, 10, 20
    if deque.len() == 4 {
        println("OK: len == 4");
    } else {
        println("FAIL: len");
    }
    
    match deque.get(0) {
        Option::Some(v) => if v == 40 { println("OK: get(0) == 40"); } else { println("FAIL: get(0) = {}", v); },
        Option::None => println("FAIL: get(0) None"),
    }
    
    match deque.get(3) {
        Option::Some(v) => if v == 20 { println("OK: get(3) == 20"); } else { println("FAIL: get(3) = {}", v); },
        Option::None => println("FAIL: get(3) None"),
    }

    // Test pop_front
    match deque.pop_front() {
        Option::Some(v) => if v == 40 { println("OK: pop_front() == 40"); } else { println("FAIL: pop_front() = {}", v); },
        Option::None => println("FAIL: pop_front() None"),
    }
    
    // Test pop_back
    match deque.pop_back() {
        Option::Some(v) => if v == 20 { println("OK: pop_back() == 20"); } else { println("FAIL: pop_back() = {}", v); },
        Option::None => println("FAIL: pop_back() None"),
    }
    
    if deque.len() == 2 {
        println("OK: len == 2 after pops");
    } else {
        println("FAIL: len after pops");
    }
    
    // Test resize (by pushing many elements)
    let mut i = 0;
    while i < 100 {
        deque.push_back(i);
        i = i + 1;
    }
    
    // deque has 2 elements + 100 elements = 102
    if deque.len() == 102 {
        println("OK: len == 102 after resize");
    } else {
        println("FAIL: len after resize");
    }
    
    // test get after resize
    match deque.get(101) {
        Option::Some(v) => if v == 99 { println("OK: get(101) == 99"); } else { println("FAIL: get(101) = {}", v); },
        Option::None => println("FAIL: get(101) None"),
    }

    deque.clear();
    if deque.is_empty() {
        println("OK: is_empty after clear");
    } else {
        println("FAIL: not empty after clear");
    }
}
