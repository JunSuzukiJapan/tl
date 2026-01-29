// src/compiler/codegen/builtin_types/hashmap.tl

struct HashMap<K, V> {
    ptr: ptr<T>, // Opaque HashMap pointer
    len: I64,
}

impl HashMap<K, V> {
    extern fn new() -> HashMap<K, V>;
    extern fn insert(self, key: K, value: V) -> Bool;
    extern fn get(self, key: K) -> V;
    extern fn remove(self, key: K) -> V;
    extern fn contains_key(self, key: K) -> Bool;
    extern fn len(self) -> I64;
    extern fn clear(self) -> Bool;
    extern fn free(self);
}
