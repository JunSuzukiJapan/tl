// src/compiler/shape_analysis.rs
// Phase 2: Static size analysis for Arena Allocator

use crate::compiler::ast::*;
use std::collections::HashMap;

/// Shape information for compile-time analysis
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeInfo {
    /// Completely static shape: [2, 3, 4]
    Static(Vec<usize>),

    /// Partially static shape: [Some(2), None, Some(4)]
    /// None means runtime-determined
    PartiallyStatic {
        known_dims: Vec<Option<usize>>,
        rank: usize,
    },

    /// Unknown shape (type information only)
    Unknown,
}

impl ShapeInfo {
    /// Calculate size in bytes if fully static
    pub fn static_size(&self, elem_size: usize) -> Option<usize> {
        match self {
            ShapeInfo::Static(dims) => {
                let total_elements: usize = dims.iter().product();
                Some(total_elements * elem_size)
            }
            _ => None,
        }
    }

    /// Check if shape is fully static
    pub fn is_static(&self) -> bool {
        matches!(self, ShapeInfo::Static(_))
    }
}

/// Memory usage formula for dynamic cases
#[derive(Debug, Clone)]
pub struct SizeFormula {
    /// Constant part (bytes)
    pub constant: usize,

    /// Dynamic factors: (variable_name, coefficient)
    /// Total size = constant * product(var * coeff for each factor)
    pub factors: Vec<(String, usize)>,
}

impl SizeFormula {
    pub fn new(constant: usize) -> Self {
        SizeFormula {
            constant,
            factors: Vec::new(),
        }
    }

    pub fn add_factor(&mut self, var_name: String, coeff: usize) {
        self.factors.push((var_name, coeff));
    }
}

/// Memory profile for a block of statements
#[derive(Debug, Clone)]
pub struct BlockMemoryProfile {
    /// Total static size if fully determinable
    pub total_static_size: Option<usize>,

    /// Formula for partially dynamic size
    pub size_formula: Option<SizeFormula>,

    /// Maximum allocation count (for pool sizing)
    pub max_allocations: usize,
}

impl BlockMemoryProfile {
    pub fn new() -> Self {
        BlockMemoryProfile {
            total_static_size: None,
            size_formula: None,
            max_allocations: 0,
        }
    }

    pub fn is_static(&self) -> bool {
        self.total_static_size.is_some()
    }
}

/// Analyzer for shape and memory usage
pub struct ShapeAnalyzer {
    /// Variable name -> Shape information
    shapes: HashMap<String, ShapeInfo>,
}

impl ShapeAnalyzer {
    pub fn new() -> Self {
        ShapeAnalyzer {
            shapes: HashMap::new(),
        }
    }

    /// Analyze expression and infer its shape
    pub fn analyze_expr(&mut self, expr: &Expr) -> ShapeInfo {
        match expr {
            // Tensor literals: fully static from length
            Expr::TensorLiteral(elements) | Expr::TensorConstLiteral(elements) => {
                ShapeInfo::Static(vec![elements.len()])
            }

            // Variable reference
            Expr::Variable(name) => self.shapes.get(name).cloned().unwrap_or(ShapeInfo::Unknown),

            // Function calls
            Expr::FnCall(fname, args) => self.infer_fn_call_shape(fname, args),

            // Static method calls (e.g., Tensor::new)
            Expr::StaticMethodCall(type_name, method, args) => {
                if type_name == "Tensor" && method == "new" {
                    self.infer_tensor_new_shape(args)
                } else {
                    ShapeInfo::Unknown
                }
            }

            _ => ShapeInfo::Unknown,
        }
    }

    /// Infer shape from function calls
    fn infer_fn_call_shape(&self, fname: &str, args: &[Expr]) -> ShapeInfo {
        // Handle special functions that we can analyze
        match fname {
            _ => ShapeInfo::Unknown,
        }
    }

    /// Infer shape from Tensor::new([dim1, dim2, ...])
    fn infer_tensor_new_shape(&self, args: &[Expr]) -> ShapeInfo {
        if args.is_empty() {
            return ShapeInfo::Unknown;
        }

        // Tensor::new takes a TensorLiteral of dimension values
        match &args[0] {
            Expr::TensorLiteral(dims) | Expr::TensorConstLiteral(dims) => {
                let mut shape = vec![];
                let mut all_static = true;

                for dim_expr in dims {
                    match dim_expr {
                        Expr::Int(n) => {
                            shape.push(Some(*n as usize));
                        }
                        Expr::Variable(_) => {
                            // Dynamic parameter - will be determined at runtime
                            shape.push(None);
                            all_static = false;
                        }
                        _ => return ShapeInfo::Unknown,
                    }
                }

                if all_static {
                    ShapeInfo::Static(shape.iter().filter_map(|x| *x).collect())
                } else {
                    ShapeInfo::PartiallyStatic {
                        known_dims: shape,
                        rank: dims.len(),
                    }
                }
            }
            _ => ShapeInfo::Unknown,
        }
    }

    /// Analyze a statement and update shape information
    pub fn analyze_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let { name, value, .. } => {
                let shape = self.analyze_expr(value);
                self.shapes.insert(name.clone(), shape);
            }
            Stmt::Assign { name, value, .. } => {
                let shape = self.analyze_expr(value);
                self.shapes.insert(name.clone(), shape);
            }
            _ => {}
        }
    }

    /// Analyze entire block and produce memory profile
    pub fn analyze_block(&mut self, block: &[Stmt]) -> BlockMemoryProfile {
        let mut profile = BlockMemoryProfile::new();
        let mut total_size = 0usize;
        let mut has_dynamic = false;
        let mut allocation_count = 0;

        for stmt in block {
            self.analyze_stmt(stmt);

            // Count allocations and sizes for Let statements
            if let Stmt::Let { value, .. } = stmt {
                let shape = self.analyze_expr(value);

                match shape {
                    ShapeInfo::Static(ref dims) => {
                        // Assume f32 = 4 bytes per element
                        let size: usize = dims.iter().product::<usize>() * 4;
                        total_size += size;
                        allocation_count += 1;
                    }
                    ShapeInfo::PartiallyStatic { .. } => {
                        has_dynamic = true;
                        allocation_count += 1;
                    }
                    ShapeInfo::Unknown => {
                        has_dynamic = true;
                    }
                }
            }
        }

        profile.max_allocations = allocation_count;

        if !has_dynamic && allocation_count > 0 {
            // Fully static case
            profile.total_static_size = Some(total_size);
        } else if has_dynamic {
            // Dynamic case - create formula (TODO: build proper formula)
            profile.size_formula = Some(SizeFormula::new(total_size));
        }

        profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_shape() {
        let shape = ShapeInfo::Static(vec![2, 3, 4]);
        assert_eq!(shape.static_size(4), Some(96)); // 2*3*4*4 = 96 bytes
    }

    #[test]
    fn test_analyzer_basic() {
        let mut analyzer = ShapeAnalyzer::new();

        // let x = [1.0, 2.0, 3.0];
        let expr = Expr::TensorLiteral(vec![Expr::Float(1.0), Expr::Float(2.0), Expr::Float(3.0)]);

        let shape = analyzer.analyze_expr(&expr);
        assert_eq!(shape, ShapeInfo::Static(vec![3]));
    }
}
