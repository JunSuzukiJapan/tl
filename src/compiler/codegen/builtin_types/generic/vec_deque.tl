struct VecDeque<T> {
    buffer: Vec<Option<T>>,
    head: i64,
    len: i64,
}

impl<T> VecDeque<T> {
    fn new() -> VecDeque<T> {
        let mut buffer = Vec<Option<T>>::with_capacity(4);
        let mut i = 0;
        while i < 4 {
            buffer.push(Option<T>::None);
            i = i + 1;
        }
        VecDeque<T> {
            buffer: buffer,
            head: 0,
            len: 0,
        }
    }

    fn len(self) -> i64 {
        self.len
    }

    fn is_empty(self) -> bool {
        self.len == 0
    }

    fn capacity(self) -> i64 {
        self.buffer.len()
    }

    fn resize(mut self, new_cap: i64) {
        let mut new_buffer = Vec<Option<T>>::with_capacity(new_cap);
        let mut i = 0;
        while i < new_cap {
            new_buffer.push(Option<T>::None);
            i = i + 1;
        }

        let old_cap = self.buffer.len();
        let mut j = 0;
        while j < self.len {
            let mut idx = self.head + j;
            if idx >= old_cap {
                idx = idx - old_cap;
            }
            if let Option::Some(val_opt) = self.buffer.get(idx) {
                // val_opt is Option<T>
                new_buffer.set(j, val_opt);
            }
            // Clear old buffer
            self.buffer.set(idx, Option<T>::None);
            j = j + 1;
        }

        self.buffer = new_buffer;
        self.head = 0;
    }

    fn push_back(mut self, value: T) {
        if self.len == self.capacity() {
            let cap = self.capacity();
            if cap == 0 {
                self.resize(4);
            } else {
                self.resize(cap * 2);
            }
        }
        let cap = self.capacity();
        let mut tail = self.head + self.len;
        if tail >= cap {
            tail = tail - cap;
        }
        self.buffer.set(tail, Option<T>::Some(value));
        self.len = self.len + 1;
    }

    fn push_front(mut self, value: T) {
        if self.len == self.capacity() {
            let cap = self.capacity();
            if cap == 0 {
                self.resize(4);
            } else {
                self.resize(cap * 2);
            }
        }
        let cap = self.capacity();
        let mut new_head = self.head - 1;
        if new_head < 0 {
            new_head = new_head + cap;
        }
        self.buffer.set(new_head, Option<T>::Some(value));
        self.head = new_head;
        self.len = self.len + 1;
    }

    fn pop_back(mut self) -> Option<T> {
        if self.len == 0 {
            return Option<T>::None;
        }
        let cap = self.capacity();
        let mut tail = self.head + self.len - 1;
        if tail >= cap {
            tail = tail - cap;
        }
        let val = self.buffer.get(tail).unwrap();
        self.buffer.set(tail, Option<T>::None);
        self.len = self.len - 1;
        val
    }

    fn pop_front(mut self) -> Option<T> {
        if self.len == 0 {
            return Option<T>::None;
        }
        let cap = self.capacity();
        let val = self.buffer.get(self.head).unwrap();
        self.buffer.set(self.head, Option<T>::None);
        
        let mut new_head = self.head + 1;
        if new_head >= cap {
            new_head = new_head - cap;
        }
        self.head = new_head;
        self.len = self.len - 1;
        val
    }

    fn get(self, index: i64) -> Option<T> {
        if index < 0 {
            return Option<T>::None;
        }
        if index >= self.len {
            return Option<T>::None;
        }
        let cap = self.capacity();
        let mut real_idx = self.head + index;
        if real_idx >= cap {
            real_idx = real_idx - cap;
        }
        self.buffer.get(real_idx).unwrap()
    }

    fn clear(mut self) {
        let cap = self.capacity();
        let mut i = 0;
        while i < self.len {
            let mut idx = self.head + i;
            if idx >= cap {
                idx = idx - cap;
            }
            self.buffer.set(idx, Option<T>::None);
            i = i + 1;
        }
        self.head = 0;
        self.len = 0;
    }
}
