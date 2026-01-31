// src/compiler/codegen/builtin_types/hashmap.tl

struct HashMap<K, V> {
    ptr: i64, // Opaque HashMap pointer
    len: i64,
}

impl<K, V> HashMap<K, V> {
    extern fn new() -> HashMap<K, V>;
    extern fn insert(self, key: K, value: V) -> bool;
    extern fn get(self, key: K) -> V;
    extern fn remove(self, key: K) -> V;
    extern fn contains_key(self, key: K) -> bool;
    extern fn len(self) -> i64;
    extern fn clear(self) -> bool;
}
