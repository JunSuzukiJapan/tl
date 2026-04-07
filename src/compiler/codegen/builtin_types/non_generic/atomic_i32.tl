extern fn tl_atomic_i32_new(val: i32) -> i64;
extern fn tl_atomic_i32_load(ptr: i64) -> i32;
extern fn tl_atomic_i32_store(ptr: i64, val: i32);
extern fn tl_atomic_i32_add(ptr: i64, val: i32) -> i32;
extern fn tl_atomic_i32_sub(ptr: i64, val: i32) -> i32;
extern fn tl_atomic_i32_swap(ptr: i64, val: i32) -> i32;
extern fn tl_atomic_i32_compare_exchange(ptr: i64, current: i32, new_val: i32) -> bool;

struct AtomicI32 {
    id: i64,
}

impl AtomicI32 {
    fn new(val: i32) -> AtomicI32 {
        AtomicI32 { id: tl_atomic_i32_new(val) }
    }

    fn load(self) -> i32 {
        tl_atomic_i32_load(self.id)
    }

    fn store(self, val: i32) {
        tl_atomic_i32_store(self.id, val);
    }

    fn add(self, val: i32) -> i32 {
        tl_atomic_i32_add(self.id, val)
    }

    fn sub(self, val: i32) -> i32 {
        tl_atomic_i32_sub(self.id, val)
    }

    fn swap(self, val: i32) -> i32 {
        tl_atomic_i32_swap(self.id, val)
    }

    fn compare_exchange(self, current: i32, new_val: i32) -> bool {
        tl_atomic_i32_compare_exchange(self.id, current, new_val)
    }
}
