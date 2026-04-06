struct StringBuilder {
    buffer: Vec<i64>,
}

impl StringBuilder {
    fn new() -> StringBuilder {
        StringBuilder {
            buffer: Vec::new()
        }
    }
    
    fn with_capacity(capacity: i64) -> StringBuilder {
        let mut buf: Vec<i64> = Vec::new();
        // Vec::reserve is not implemented in standard Vec.tl for generic args yet,
        // so we start with an empty Vec. It will grow automatically.
        // In the future, we can add a reserve mechanism.
        StringBuilder {
            buffer: buf,
        }
    }
    
    fn append(mut self, s: String) {
        let chars = s.chars();
        let mut i = 0;
        let n = chars.len();
        while i < n {
            self.buffer.push(chars.get(i).unwrap());
            i = i + 1;
        }
    }
    
    fn append_i64(mut self, n: i64) {
        self.append(n.to_string());
    }
    
    fn build(self) -> String {
        String::from_chars(self.buffer)
    }
    
    fn clear(mut self) {
        self.buffer.clear();
    }
    
    fn is_empty(self) -> bool {
        self.buffer.is_empty()
    }
    
    fn len(self) -> i64 {
        self.buffer.len()
    }
}
