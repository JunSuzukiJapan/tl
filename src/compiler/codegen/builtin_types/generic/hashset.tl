// Malloc externs provided by vec.tl

// Intrinsic function for hashing (directly implemented in compiler)
// Provided by hashmap.tl
enum SetEntry<T> {
    Empty,
    Occupied(T),
    Deleted,
}

struct HashSet<T> {
    entries: Vec<SetEntry<T>>,
    len: i64,
    capacity: i64,
}

impl<T> HashSet<T> {
    fn new() -> HashSet<T> {
        let mut entries = Vec<SetEntry<T>>::with_capacity(4);
        let mut i = 0;
        while i < 4 {
            entries.push(SetEntry<T>::Empty);
            i = i + 1;
        }
        HashSet<T> {
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
        
        let mut new_vec = Vec<SetEntry<T>>::with_capacity(new_cap);
        let mut i = 0;
        while i < new_cap {
            new_vec.push(SetEntry<T>::Empty);
            i = i + 1;
        }
        
        self.entries = new_vec;
        self.len = 0;
        
        let old_len = old_entries.len();
        let mut j = 0;
        while j < old_len {
            if let Option::Some(e) = old_entries.get(j) {
                match e {
                    SetEntry<T>::Occupied(val) => {
                        let _tmp = self.insert(val);
                    },
                    SetEntry<T>::Empty => {},
                    SetEntry<T>::Deleted => {}
                }
            }
            j = j + 1;
        }
    }

    fn insert(mut self, value: T) -> bool {
        if self.entries.len() == 0 {
            self.resize(4);
        } else {
            let cap = self.entries.len();
            // Load factor check: 75%
            if 4 * (self.len + 1) > 3 * cap {
                 self.resize(cap * 2);
            }
        }

        let cap = self.entries.len();
        let h = tl_core_hash(value);
        let mut idx = h % cap;
        if idx < 0 { idx = -idx; }

        let mut start_idx = idx;
        loop {
            let entry = self.entries.get(idx).unwrap();
            match (entry) {
                SetEntry<T>::Empty => {
                    self.entries[idx] = SetEntry<T>::Occupied(value);
                    self.len = self.len + 1;
                    return true;
                },
                SetEntry<T>::Deleted => {
                    self.entries[idx] = SetEntry<T>::Occupied(value);
                    self.len = self.len + 1;
                    return true;
                },
                SetEntry<T>::Occupied(val) => {
                    if val == value {
                        return false;
                    }
                }
            }
            
            idx = (idx + 1) % cap;
            if idx == start_idx {
                return false; 
            }
        }
    }

    fn contains(self, value: T) -> bool {
        if self.len == 0 {
            return false;
        }

        let cap = self.entries.len();
        let h = tl_core_hash(value);
        let mut idx = h % cap;
        if idx < 0 { idx = -idx; }
        
        let start_idx = idx;

        loop {
            let entry = self.entries.get(idx).unwrap();
            match (entry) {
                SetEntry<T>::Empty => {
                     return false;
                },
                SetEntry<T>::Occupied(val) => {
                    if val == value {
                        return true;
                    }
                },
                SetEntry<T>::Deleted => {
                }
            }

            idx = (idx + 1) % cap;
            if idx == start_idx {
                return false;
            }
        }
    }

    fn remove(mut self, value: T) -> bool {
        if self.len == 0 {
            return false;
        }
        let cap = self.entries.len();
        let h = tl_core_hash(value);
        let mut idx = h % cap;
        if idx < 0 { idx = -idx; }
        let start_idx = idx;

        loop {
            let entry = self.entries.get(idx).unwrap();
            match (entry) {
                SetEntry<T>::Empty => {
                    return false;
                },
                SetEntry<T>::Occupied(val) => {
                    if val == value {
                        self.entries[idx] = SetEntry<T>::Deleted;
                        self.len = self.len - 1;
                        return true;
                    }
                },
                SetEntry<T>::Deleted => {}
            }
            idx = (idx + 1) % cap;
            if idx == start_idx {
                return false;
            }
        }
    }

    fn clear(mut self) {
        let cap = self.entries.len();
        let mut i = 0;
        while i < cap {
            self.entries[i] = SetEntry<T>::Empty;
            i = i + 1;
        }
        self.len = 0;
    }
}
