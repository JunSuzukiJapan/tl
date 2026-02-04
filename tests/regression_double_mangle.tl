// Regression test: Double-mangling prevention
// This test verifies that struct field access works correctly after
// the FieldAccess fix (load struct ptr before GEP) and 
// double-mangling fixes in compile_struct_init/emit_struct_copy.
//
// Previously, Vec::push would not update self.len correctly due to
// incorrect code generation for self.len = self.len + 1.

fn main() {
    // Test 1: Vec::push correctly updates len
    let mut v: Vec<i64> = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    
    // This should print 3, not 0 (which was the bug)
    let len = v.len();
    if len != 3 {
        print(-1); // ERROR: len should be 3
        return;
    }
    
    // Test 2: HashMap::new() works (tests double-mangling fix)
    let m: HashMap<i64, i64> = HashMap::new();
    
    // If we got here without crash, the double-mangling fix is working
    print(len); // Should print 3
}
