struct BTreeNode<K, V> {
    keys: Vec<K>,
    values: Vec<V>,
    children: Vec<i64>,
    is_leaf: bool,
}

struct BTreeMap<K, V> {
    nodes: Vec<BTreeNode<K, V>>,
    root: i64,
    len: i64,
}

impl<K, V> BTreeMap<K, V> {
    fn new() -> BTreeMap<K, V> {
        let mut map = BTreeMap<K, V> {
            nodes: Vec::new(),
            root: -1,
            len: 0,
        };
        map.root = map.allocate_node(true);
        map
    }

    fn len(self) -> i64 {
        self.len
    }

    fn is_empty(self) -> bool {
        self.len == 0
    }

    fn allocate_node(mut self, is_leaf: bool) -> i64 {
        let node = BTreeNode<K, V> {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: is_leaf
        };
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn search_node(self, node_id: i64, key: K) -> Option<V> {
        if node_id < 0 {
            return Option::None;
        }
        let node = self.nodes.get(node_id).unwrap();
        let n = node.keys.len();
        let mut i = 0;
        while i < n {
            let k = node.keys.get(i).unwrap();
            if key == k {
                return Option::Some(node.values.get(i).unwrap());
            }
            if key < k {
                break;
            }
            i = i + 1;
        }
        if node.is_leaf {
            return Option::None;
        }
        let next_child = node.children.get(i).unwrap();
        self.search_node(next_child, key)
    }

    fn get(self, key: K) -> Option<V> {
        if self.root == -1 {
            return Option::None;
        }
        self.search_node(self.root, key)
    }

    fn contains_key(self, key: K) -> bool {
        match self.get(key) {
            Option::Some(v) => true,
            Option::None => false,
        }
    }

    fn split_child(mut self, parent_id: i64, child_idx: i64) {
        // T = 4, Capacity = 7 elements
        let mut parent = self.nodes.get(parent_id).unwrap();
        let child_id = parent.children.get(child_idx).unwrap();
        let mut child = self.nodes.get(child_id).unwrap();
        
        let mut new_child = BTreeNode<K, V> {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: child.is_leaf
        };
        
        // Move last T-1 = 3 keys/values from child to new_child
        // Indices: child has 7. Mid = 3. Rights = 4, 5, 6
        let mut j = 0;
        while j < 3 {
            new_child.keys.push(child.keys.get(4 + j).unwrap());
            new_child.values.push(child.values.get(4 + j).unwrap());
            j = j + 1;
        }
        
        if !child.is_leaf {
            // Move last T = 4 children. Indices: 4, 5, 6, 7
            let mut k = 0;
            while k < 4 {
                new_child.children.push(child.children.get(4 + k).unwrap());
                k = k + 1;
            }
            // Resize child children to T = 4
            let mut c2 = Vec::new();
            let mut kc = 0;
            while kc < 4 {
                c2.push(child.children.get(kc).unwrap());
                kc = kc + 1;
            }
            child.children = c2;
        }
        
        let mid_key = child.keys.get(3).unwrap();
        let mid_val = child.values.get(3).unwrap();
        
        // Resize child keys/values to T-1 = 3
        let mut k2 = Vec::new();
        let mut v2 = Vec::new();
        let mut kk = 0;
        while kk < 3 {
            k2.push(child.keys.get(kk).unwrap());
            v2.push(child.values.get(kk).unwrap());
            kk = kk + 1;
        }
        child.keys = k2;
        child.values = v2;
        
        self.nodes.push(new_child);
        let new_child_id = self.nodes.len() - 1;
        
        parent.keys.insert(child_idx, mid_key);
        parent.values.insert(child_idx, mid_val);
        parent.children.insert(child_idx + 1, new_child_id);
        
        self.nodes.set(parent_id, parent);
        self.nodes.set(child_id, child);
    }

    fn insert_non_full(mut self, node_id: i64, key: K, value: V) {
        let mut node = self.nodes.get(node_id).unwrap();
        let n = node.keys.len();
        
        let mut j = 0;
        while j < n {
            let k = node.keys.get(j).unwrap();
            if k == key {
                node.values.set(j, value);
                self.nodes.set(node_id, node);
                return;
            }
            j = j + 1;
        }
        
        if node.is_leaf {
            let mut i = n - 1;
            node.keys.push(key); 
            node.values.push(value); 
            while i >= 0 {
                let current_k = node.keys.get(i).unwrap();
                if key < current_k {
                    let tmp_k = node.keys.get(i).unwrap();
                    let tmp_v = node.values.get(i).unwrap();
                    node.keys.set(i + 1, tmp_k);
                    node.values.set(i + 1, tmp_v);
                    i = i - 1;
                } else {
                    break;
                }
            }
            node.keys.set(i + 1, key);
            node.values.set(i + 1, value);
            self.nodes.set(node_id, node);
            self.len = self.len + 1;
        } else {
            let mut i = n - 1;
            while i >= 0 {
                let current_k = node.keys.get(i).unwrap();
                if key < current_k {
                    i = i - 1;
                } else {
                    break;
                }
            }
            i = i + 1;
            
            let child_id = node.children.get(i).unwrap();
            let child = self.nodes.get(child_id).unwrap();
            
            if child.keys.len() == 7 {
                self.nodes.set(node_id, node); // Must save current node before splitting because split_child modifies it
                self.split_child(node_id, i);
                
                let mut updated_parent = self.nodes.get(node_id).unwrap();
                if key > updated_parent.keys.get(i).unwrap() {
                    i = i + 1;
                }
                let next_child_id = updated_parent.children.get(i).unwrap();
                self.insert_non_full(next_child_id, key, value);
            } else {
                self.nodes.set(node_id, node);
                self.insert_non_full(child_id, key, value);
            }
        }
    }

    fn insert(mut self, key: K, value: V) {
        if self.root == -1 {
            self.root = self.allocate_node(true);
        }
        let root_node = self.nodes.get(self.root).unwrap();
        if root_node.keys.len() == 7 {
            let s = self.allocate_node(false);
            let mut new_root = self.nodes.get(s).unwrap();
            new_root.children.push(self.root);
            self.nodes.set(s, new_root);
            self.root = s;
            
            self.split_child(s, 0);
            self.insert_non_full(self.root, key, value);
        } else {
            self.insert_non_full(self.root, key, value);
        }
    }

    fn clear(mut self) {
        self.nodes = Vec::new();
        self.root = self.allocate_node(true);
        self.len = 0;
    }
}
