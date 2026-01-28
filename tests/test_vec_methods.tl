fn main() {
    let v: Vec<i64> = Vec::new();
    v.push(10);
    v.push(20);
    v.push(30);

    print("Initial len:");
    print(v.len()); // 3

    let p = v.pop();
    print("Popped:");
    print(p); // 30

    print("Len after pop:");
    print(v.len()); // 2

    v.insert(0, 5);
    print("After insert:");
    print(v.get(0)); // 5

    let r = v.remove(1); // remove index 1 (original 10)
    print("Removed:");
    print(r); // 10

    print("Len after remove:");
    print(v.len()); // 2

    let has_20 = v.contains(20);
    print("Contains 20:");
    print(has_20); // true

    print("Is empty:");
    print(v.is_empty()); // false

    v.clear();
    print("Len after clear:");
    print(v.len()); // 0

    print("Is empty after clear:");
    print(v.is_empty()); // true
}
