// Malloc externs provided by vec.tl

// Intrinsic function for hashing
extern fn tl_core_hash<T>(val: T) -> i64;

fn hash<T>(val: T) -> i64 {
    tl_core_hash(val)
}

enum Entry<K, V> {
    Empty,
    Occupied(K, V),
    Deleted,
}

struct HashMap<K, V> {
    entries: Vec<Entry<K, V>>,
    len: i64,
    capacity: i64,
}

impl<K, V> HashMap<K, V> {
    fn new() -> HashMap<K, V> {
        // Initialize entries with 4 Entry::Empty elements (smaller initial size)
        let mut entries = Vec<Entry<K, V>>::with_capacity(4);
        let mut i = 0;
        while i < 4 {
            entries.push(Entry<K, V>::Empty);
            i = i + 1;
        }
        HashMap<K, V> {
            entries: entries,
            len: 0,
            capacity: 4,
        }
    }
    fn len(self) -> i64 {
        self.len
    }

    fn is_empty(self) -> bool {
        self.len == 0
    }

    fn resize(mut self, new_cap: i64) {
        let old_entries = self.entries;
        // Re-initialize self with empty entries of new capacity
        // Note: Vec::with_capacity creates a vector with capacity but len=0.
        // We need 'len' to be 'new_cap' filled with Empty.
        // Since Vec doesn't support 'fill' yet, we manually push.
        
        let mut new_vec = Vec<Entry<K, V>>::with_capacity(new_cap);
        let mut i = 0;
        while i < new_cap {
            new_vec.push(Entry<K, V>::Empty);
            i = i + 1;
        }
        
        self.entries = new_vec;
        self.len = 0; // Reset length, will re-count on insert
        // Re-hash and insert all old occupied entries
        let old_len = old_entries.len();
        let mut j = 0;
        while j < old_len {
            // Safe manual indexing or pattern matching?
            // Since we don't have pattern matching in `if let` easily for Vec indexing yet without `get`,
            // let's use `get` or direct index. `old_entries.get(j)` returns Option.
            if let Option::Some(e) = old_entries.get(j) {
                match e {
                    Entry<K, V>::Occupied(k, v) => {
                        self.insert(k, v);
                    },
                    Entry<K, V>::Empty => {},
                    Entry<K, V>::Deleted => {}
                }
            }
            j = j + 1;
        }
    }

    fn insert(mut self, key: K, value: V) {
        if self.entries.len() == 0 {
            self.resize(4);
        } else {
            let cap = self.entries.len();
            // Load factor check: (len + 1) / cap > 0.75  => 4*(len+1) > 3*cap
            if 4 * (self.len + 1) > 3 * cap {
                 self.resize(cap * 2);
            }
        }

        let cap = self.entries.len();
        let h = hash(key);
        // Ensure non-negative index through abs or mask? 
        // hash returns i64. using abs for safety.
        let mut idx = h % cap;
        if idx < 0 { idx = -idx; }

        let mut start_idx = idx;
        loop {
            // Need mutable reference to entry? 
            // Currently TL Vec access `v[i]` copies for non-ref types, 
            // but we need to update. `v[i] = ...` works.
            
            // We use a temporary to check content
            // NOTE: This creates a copy of Entry. 
            // For efficiency with large structs, we'd want references, but for now this is fine.
            let entry = self.entries.get(idx).unwrap(); // Should be safe
            
            match (entry) {
                Entry<K, V>::Empty => {
                    self.entries[idx] = Entry<K, V>::Occupied(key, value);
                    self.len = self.len + 1;
                    return;
                },
                Entry<K, V>::Deleted => {
                    self.entries[idx] = Entry<K, V>::Occupied(key, value);
                    self.len = self.len + 1;
                    return;
                },
                Entry<K, V>::Occupied(k, unused) => {}
            }
            
            idx = (idx + 1) % cap;
            if idx == start_idx {
                // Table full? Should have been resized.
                // But theoretically possible if logic bug.
                return; 
            }
        }
    }

    fn get(self, key: K) -> Option<V> {
        if self.len == 0 {
            return Option::None;
        }

        let cap = self.entries.len();
        let h = hash(key);
        let mut idx = h % cap;
        if idx < 0 { idx = -idx; }
        
        let start_idx = idx;

        loop {
            let entry = self.entries.get(idx).unwrap();
            match (entry) {
                Entry<K, V>::Empty => {
                     return Option::None;
                },
                Entry<K, V>::Occupied(k, v) => {
                    if k == key {
                        return Option::Some(v);
                    }
                },
                Entry<K, V>::Deleted => {
                    // Continue probing
                }
            }

            idx = (idx + 1) % cap;
            if idx == start_idx {
                return Option::None;
            }
        }
    }

    fn remove(mut self, key: K) {
    }
}
