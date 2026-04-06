fn main() {
    let mut set: HashSet<i64> = HashSet::new();
    
    // Test insert and contains
    set.insert(10);
    set.insert(20);
    set.insert(30);
    
    if set.contains(10) {
        println("OK: contains(10)");
    } else {
        println("FAIL: contains(10)");
    }
    
    if set.contains(40) {
        println("FAIL: contains(40)");
    } else {
        println("OK: contains(40) is false");
    }
    
    if set.insert(20) {
        println("FAIL: duplicate insert should return false");
    } else {
        println("OK: duplicate insert handled");
    }
    
    if set.len() == 3 {
        println("OK: len == 3");
    } else {
        println("FAIL: len");
    }
    
    // Test remove
    if set.remove(20) {
        println("OK: remove(20) returned true");
    } else {
        println("FAIL: remove(20) returned false");
    }
    
    if set.contains(20) {
        println("FAIL: contains(20) after remove");
    } else {
        println("OK: contains(20) after remove");
    }
    
    if set.len() == 2 {
        println("OK: len == 2 after remove");
    } else {
        println("FAIL: len after remove");
    }
    
    // Test clear
    set.clear();
    if set.is_empty() {
        println("OK: is_empty after clear");
    } else {
        println("FAIL: not empty after clear");
    }
    
    // Test resize by adding many elements
    let mut i = 0;
    while i < 100 {
        set.insert(i);
        i = i + 1;
    }
    if set.len() == 100 {
        println("OK: len == 100 after resize");
    } else {
        println("FAIL: len after resize");
    }
    if set.contains(99) {
        println("OK: contains(99) after resize");
    } else {
        println("FAIL: contains(99) after resize");
    }
}
