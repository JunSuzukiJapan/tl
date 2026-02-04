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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub fn is_static(&self) -> bool {
        matches!(self, ShapeInfo::Static(_))
    }
}

/// Memory usage formula for dynamic cases
#[derive(Debug, Clone)]
#[allow(dead_code)]
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

    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
        match &expr.inner {
            // Tensor literals: fully static from length
            ExprKind::TensorLiteral(elements) | ExprKind::TensorConstLiteral(elements) => {
                if elements.is_empty() {
                    return ShapeInfo::Static(vec![0]);
                }
                let mut shape = vec![elements.len()];
                // Peek at first element to get sub-shape
                let sub_shape = self.analyze_expr(&elements[0]);
                match sub_shape {
                    ShapeInfo::Static(s) => {
                        shape.extend(s);
                    }
                    _ => {}
                }
                ShapeInfo::Static(shape)
            }

            // Variable reference
            ExprKind::Variable(name) => {
                self.shapes.get(name).cloned().unwrap_or(ShapeInfo::Unknown)
            }

            // Function calls
            ExprKind::FnCall(fname, args) => self.infer_fn_call_shape(fname, args),

            // Static method calls (e.g., Tensor::new)
            ExprKind::StaticMethodCall(type_ty, method, args) => {
                let type_name = type_ty.get_base_name();
                if type_name == "Tensor" && method == "new" {
                    self.infer_tensor_new_shape(args)
                } else {
                    ShapeInfo::Unknown
                }
            }

            // Binary operations: if both are static and same, result is same
            ExprKind::BinOp(left, _, right) => {
                let s1 = self.analyze_expr(left);
                let s2 = self.analyze_expr(right);
                match (s1, s2) {
                    (ShapeInfo::Static(d1), ShapeInfo::Static(d2)) if d1 == d2 => {
                        ShapeInfo::Static(d1)
                    }
                    _ => ShapeInfo::Unknown,
                }
            }

            _ => ShapeInfo::Unknown,
        }
    }

    /// Infer shape from function calls
    fn infer_fn_call_shape(&self, fname: &str, _args: &[Expr]) -> ShapeInfo {
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
        match &args[0].inner {
            ExprKind::TensorLiteral(dims) | ExprKind::TensorConstLiteral(dims) => {
                let mut shape = vec![];
                let mut all_static = true;

                for dim_expr in dims {
                    match &dim_expr.inner {
                        ExprKind::Int(n) => {
                            shape.push(Some(*n as usize));
                        }
                        ExprKind::Variable(_) => {
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

    /// Count number of tensor allocations in an expression
    pub fn count_allocations(&self, expr: &Expr) -> usize {
        match &expr.inner {
            ExprKind::TensorLiteral(_) | ExprKind::TensorConstLiteral(_) => 1,
            ExprKind::BinOp(l, _, r) => 1 + self.count_allocations(l) + self.count_allocations(r),
            ExprKind::UnOp(_, e) => 1 + self.count_allocations(e),
            ExprKind::FnCall(_, args) => {
                100 + args
                    .iter()
                    .map(|a| self.count_allocations(a))
                    .sum::<usize>()
            }
            ExprKind::MethodCall(obj, _, args) => {
                100 + self.count_allocations(obj)
                    + args
                        .iter()
                        .map(|a| self.count_allocations(a))
                        .sum::<usize>()
            }
            ExprKind::StaticMethodCall(_, _, args) => {
                100 + args
                    .iter()
                    .map(|a| self.count_allocations(a))
                    .sum::<usize>()
            }
            ExprKind::IndexAccess(obj, indices) => {
                0 + self.count_allocations(obj)
                    + indices
                        .iter()
                        .map(|a| self.count_allocations(a))
                        .sum::<usize>()
            }
            _ => 0,
        }
    }

    /// Analyze a statement and update shape information
    pub fn analyze_stmt(&mut self, stmt: &Stmt) {
        match &stmt.inner {
            StmtKind::Let { name, value, .. } => {
                let shape = self.analyze_expr(value);
                self.shapes.insert(name.clone(), shape);
            }
            StmtKind::Assign {
                lhs, value, op, ..
            } => {
                if *op == AssignOp::Assign {
                     if let LValue::Variable(name) = lhs {
                        let shape = self.analyze_expr(value);
                        self.shapes.insert(name.clone(), shape);
                     }
                }
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

            match &stmt.inner {
                StmtKind::Let { value, .. }
                | StmtKind::Assign { value, .. }
                | StmtKind::Expr(value) => {

                    allocation_count += self.count_allocations(value);

                    let shape = self.analyze_expr(value);
                    match shape {
                        ShapeInfo::Static(dims) => {
                            total_size += dims.iter().product::<usize>() * 4;
                        }
                        ShapeInfo::PartiallyStatic { .. } | ShapeInfo::Unknown => {
                            has_dynamic = true;
                        }
                    }
                }
                StmtKind::Return(value_opt) => {
                    if let Some(value) = value_opt {
                        allocation_count += self.count_allocations(value);

                        let shape = self.analyze_expr(value);
                        match shape {
                            ShapeInfo::Static(dims) => {
                                total_size += dims.iter().product::<usize>() * 4;
                            }
                            ShapeInfo::PartiallyStatic { .. } | ShapeInfo::Unknown => {
                                has_dynamic = true;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        profile.max_allocations = allocation_count;

        if !has_dynamic && allocation_count > 0 {
            profile.total_static_size = Some(total_size);
        } else if has_dynamic {
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
        let expr = Spanned::dummy(ExprKind::TensorLiteral(vec![
            Spanned::dummy(ExprKind::Float(1.0)),
            Spanned::dummy(ExprKind::Float(2.0)),
            Spanned::dummy(ExprKind::Float(3.0)),
        ]));

        let shape = analyzer.analyze_expr(&expr);
        assert_eq!(shape, ShapeInfo::Static(vec![3]));
    }
}
