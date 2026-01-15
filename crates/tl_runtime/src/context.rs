use std::collections::HashMap;

/// A tensor value stored in context (simplified representation)
#[derive(Debug, Clone)]
pub struct TensorValue {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl TensorValue {
    /// Get element at given indices
    pub fn get(&self, indices: &[i64]) -> Option<f64> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let mut flat_idx = 0usize;
        let mut stride = 1usize;
        for (&i, &dim) in indices.iter().rev().zip(self.shape.iter().rev()) {
            let idx = i as usize;
            if idx >= dim {
                return None;
            }
            flat_idx += idx * stride;
            stride *= dim;
        }
        self.data.get(flat_idx).copied()
    }
}

/// Context holding tensor values for hybrid computation
#[derive(Debug, Clone, Default)]
pub struct TensorContext {
    pub tensors: HashMap<String, TensorValue>,
}

impl TensorContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: String, tensor: TensorValue) {
        self.tensors.insert(name, tensor);
    }

    pub fn get(&self, name: &str) -> Option<&TensorValue> {
        self.tensors.get(name)
    }
}
