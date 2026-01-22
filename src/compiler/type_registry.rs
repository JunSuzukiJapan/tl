// src/compiler/type_registry.rs
//! Centralized type and method signature registry for semantic analysis.
//!
//! This module provides a unified way to manage type information and method signatures,
//! enabling consistent type checking across the compiler.

use crate::compiler::ast::Type;
use std::collections::HashMap;

/// Represents a method signature with parameter types and return type.
#[derive(Debug, Clone)]
pub struct MethodSignature {
    /// Method name
    pub name: String,
    /// Parameter types (not including self/receiver)
    pub params: Vec<ParamType>,
    /// Return type pattern
    pub return_type: ReturnType,
    /// Whether this method accepts variable arguments
    pub is_varargs: bool,
    /// Minimum number of arguments (for varargs)
    pub min_args: usize,
}

/// Parameter type specification
#[derive(Debug, Clone)]
pub enum ParamType {
    /// Exact type match required
    Exact(Type),
    /// Any tensor type (any element type, any rank)
    AnyTensor,
    /// Tensor with specific element type
    TensorOf(Box<Type>),
    /// Shape array (tensor literal or scalar array)
    ShapeArray,
    /// Any integer type (I32 or I64)
    AnyInt,
    /// Any numeric type (F32, F64, I32, I64)
    AnyNumeric,
    /// Same as receiver type
    SameAsReceiver,
    /// Boolean
    Bool,
    /// Any tensor or numeric type
    AnyTensorOrNumeric,
}

/// Return type specification
#[derive(Debug, Clone)]
pub enum ReturnType {
    /// Exact type
    Exact(Type),
    /// Same as receiver
    SameAsReceiver,
    /// Tensor with same element type as receiver, but different rank
    TensorSameElementType(usize),
    /// Tensor with dynamic rank (0)
    TensorDynamicRank,
    /// Inferred from shape argument (for reshape)
    InferFromShapeArg,
    /// Scalar extracted from tensor
    ExtractedScalar,
    /// Tensor with same element type as receiver, but rank incremented by 1
    TensorRankIncr,
    /// Void (no return value)
    Void,
}

/// Type registry that manages types and their method signatures
pub struct TypeRegistry {
    /// Type name -> Method name -> Signature
    methods: HashMap<String, HashMap<String, MethodSignature>>,
}

impl TypeRegistry {
    /// Create a new empty type registry
    pub fn new() -> Self {
        let mut registry = Self {
            methods: HashMap::new(),
        };
        registry.register_builtins();
        registry
    }

    /// Register all built-in types and their methods
    fn register_builtins(&mut self) {
        self.register_tensor_methods();
        self.register_numeric_methods();
        self.register_string_methods();
        self.register_file_methods();
        self.register_path_methods();
        self.register_varbuilder_methods();
        self.register_vec_methods();
        self.register_ml_methods(); // Tokenizer, KVCache, Map
    }

    /// Register Tensor methods
    fn register_tensor_methods(&mut self) {
        let tensor_methods = vec![
            // reshape(shape) -> Tensor
            MethodSignature {
                name: "reshape".to_string(),
                params: vec![ParamType::ShapeArray],
                return_type: ReturnType::InferFromShapeArg,
                is_varargs: false,
                min_args: 1,
            },
            // sum(...) -> Tensor (scalar if no args, or dim reduction)
            MethodSignature {
                name: "sum".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::TensorSameElementType(0),
                is_varargs: true,
                min_args: 0,
            },
            // sum_dim(dim, keepdim) -> Tensor
            MethodSignature {
                name: "sum_dim".to_string(),
                params: vec![ParamType::AnyInt, ParamType::Bool],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            // mean() -> Tensor (scalar)
            MethodSignature {
                name: "mean".to_string(),
                params: vec![],
                return_type: ReturnType::TensorSameElementType(0),
                is_varargs: false,
                min_args: 0,
            },
            // argmax(dim) -> Tensor<I64>
            MethodSignature {
                name: "argmax".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::Exact(Type::Tensor(Box::new(Type::I64), 0)),
                is_varargs: false,
                min_args: 1,
            },
            // softmax(dim) -> Tensor
            MethodSignature {
                name: "softmax".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            // item() -> F32/I64 (scalar extraction)
            MethodSignature {
                name: "item".to_string(),
                params: vec![],
                return_type: ReturnType::ExtractedScalar,
                is_varargs: false,
                min_args: 0,
            },
            // clone() -> Tensor
            MethodSignature {
                name: "clone".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            // contiguous() -> Tensor
            MethodSignature {
                name: "contiguous".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            // detach(requires_grad) -> Tensor
            MethodSignature {
                name: "detach".to_string(),
                params: vec![ParamType::Bool],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            // backward() -> Void
            MethodSignature {
                name: "backward".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
            // grad() -> Tensor
            MethodSignature {
                name: "grad".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            // len() -> I64
            MethodSignature {
                name: "len".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::I64),
                is_varargs: false,
                min_args: 0,
            },
            // dim(index) -> I64
            MethodSignature {
                name: "dim".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::Exact(Type::I64),
                is_varargs: false,
                min_args: 1,
            },
            // print() -> Void
            MethodSignature {
                name: "print".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
            // print_1() -> Void
            MethodSignature {
                name: "print_1".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
            // print_2() -> Void
            MethodSignature {
                name: "print_2".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
            // print_3() -> Void
            MethodSignature {
                name: "print_3".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
            // Unary element-wise ops
            MethodSignature {
                name: "relu".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "gelu".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "silu".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "sigmoid".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "tanh".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "exp".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "log".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "sqrt".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "abs".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "neg".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "sin".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "cos".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "tan".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            // narrow(dim, start, length) -> Tensor
            MethodSignature {
                name: "narrow".to_string(),
                params: vec![ParamType::AnyInt, ParamType::AnyInt, ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 3,
            },
            // transpose(dim0, dim1) -> Tensor
            MethodSignature {
                name: "transpose".to_string(),
                params: vec![ParamType::AnyInt, ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 2,
            },
            // matmul(other) -> Tensor
            MethodSignature {
                name: "matmul".to_string(),
                params: vec![ParamType::AnyTensor],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            // cross_entropy(target) -> Tensor
            MethodSignature {
                name: "cross_entropy".to_string(),
                params: vec![ParamType::AnyTensor],
                return_type: ReturnType::TensorSameElementType(0),
                is_varargs: false,
                min_args: 1,
            },

            // argmin(dim) -> Tensor<I64>
            MethodSignature {
                name: "argmin".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::Exact(Type::Tensor(Box::new(Type::I64), 0)),
                is_varargs: false,
                min_args: 1,
            },
            // max/min/mean(dim?) -> Tensor
            MethodSignature {
                name: "max".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::TensorSameElementType(0),
                is_varargs: true,
                min_args: 0,
            },
            MethodSignature {
                name: "min".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::TensorSameElementType(0),
                is_varargs: true,
                min_args: 0,
            },
            // log_softmax(dim) -> Tensor
            MethodSignature {
                name: "log_softmax".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            // cuda() / cpu() -> Tensor
            MethodSignature {
                name: "cuda".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "cpu".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            // get_shape() -> Tensor<I64>(1)
            MethodSignature {
                name: "get_shape".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::Tensor(Box::new(Type::I64), 1)),
                is_varargs: false,
                min_args: 0,
            },
            // item_i64() -> I64
            MethodSignature {
                name: "item_i64".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::I64),
                is_varargs: false,
                min_args: 0,
            },
            // rms_norm(weight, eps) -> Tensor
            MethodSignature {
                name: "rms_norm".to_string(),
                params: vec![ParamType::AnyTensor, ParamType::AnyNumeric],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 2,
            },
            // apply_rope(cos, sin) -> Tensor
            MethodSignature {
                name: "apply_rope".to_string(),
                params: vec![ParamType::AnyTensor, ParamType::AnyTensor],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 2,
            },
            // repeat_interleave(repeats, dim) -> Tensor
            MethodSignature {
                name: "repeat_interleave".to_string(),
                params: vec![ParamType::AnyInt, ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 2,
            },
            // pow(exponent) -> Tensor (accepts scalar or tensor)
            MethodSignature {
                name: "pow".to_string(),
                params: vec![ParamType::AnyNumeric],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            // clamp(min, max) -> Tensor
            MethodSignature {
                name: "clamp".to_string(),
                params: vec![ParamType::AnyNumeric, ParamType::AnyNumeric],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 2,
            },
            // slice(start, end) -> Tensor
            MethodSignature {
                name: "slice".to_string(),
                params: vec![ParamType::AnyInt, ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 2,
            },
            // conv2d(weight, padding, stride) -> Tensor
            MethodSignature {
                name: "conv2d".to_string(),
                params: vec![ParamType::AnyTensor, ParamType::AnyInt, ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 3,
            },
            // Compound assignment methods (used for desugaring field += val)
            MethodSignature {
                name: "add_assign".to_string(),
                params: vec![ParamType::AnyTensorOrNumeric],
                return_type: ReturnType::Exact(Type::Void),
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "sub_assign".to_string(),
                params: vec![ParamType::AnyTensorOrNumeric],
                return_type: ReturnType::Exact(Type::Void),
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "mul_assign".to_string(),
                params: vec![ParamType::AnyTensorOrNumeric],
                return_type: ReturnType::Exact(Type::Void),
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "div_assign".to_string(),
                params: vec![ParamType::AnyTensorOrNumeric],
                return_type: ReturnType::Exact(Type::Void),
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "mod_assign".to_string(),
                params: vec![ParamType::AnyTensorOrNumeric],
                return_type: ReturnType::Exact(Type::Void),
                is_varargs: false,
                min_args: 1,
            },
            // scale(factor) -> Tensor
            MethodSignature {
                name: "scale".to_string(),
                params: vec![ParamType::AnyNumeric],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            // matmul_4d(other) -> Tensor
            MethodSignature {
                name: "matmul_4d".to_string(),
                params: vec![ParamType::AnyTensor],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            // add_4d(other) -> Tensor
            MethodSignature {
                name: "add_4d".to_string(),
                params: vec![ParamType::AnyTensor],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            // cat_4d(other, dim) -> Tensor
            MethodSignature {
                name: "cat_4d".to_string(),
                params: vec![ParamType::AnyTensor, ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 2,
            },
            // transpose_2d(dim0, dim1) -> Tensor
            MethodSignature {
                name: "transpose_2d".to_string(),
                params: vec![ParamType::AnyInt, ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 2,
            },
            // matmul_quantized(other) -> Tensor
            MethodSignature {
                name: "matmul_quantized".to_string(),
                params: vec![ParamType::AnyTensorOrNumeric],
                return_type: ReturnType::Exact(Type::Tensor(Box::new(Type::F32), 0)),
                is_varargs: false,
                min_args: 1,
            },
            // cat_i64(other, dim) -> Tensor
            MethodSignature {
                name: "cat_i64".to_string(),
                params: vec![ParamType::AnyTensor, ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 2,
            },
            // sample(num_samples, temperature) -> Tensor<I64>
            MethodSignature {
                name: "sample".to_string(),
                params: vec![ParamType::AnyNumeric, ParamType::AnyNumeric],
                return_type: ReturnType::Exact(Type::Tensor(Box::new(Type::I64), 1)),
                is_varargs: false,
                min_args: 2,
            },
            // tril(diagonal) -> Tensor
            MethodSignature {
                name: "tril".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            // repeat_interleave(repeats, dim) -> Tensor
            MethodSignature {
                name: "repeat_interleave".to_string(),
                params: vec![ParamType::AnyInt, ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 2,
            },
            // embedding(indices) -> Tensor
            MethodSignature {
                name: "embedding".to_string(),
                params: vec![ParamType::AnyTensor],
                return_type: ReturnType::TensorRankIncr,
                is_varargs: false,
                min_args: 1,
            },
            // Element-wise binary ops (accepts scalar or tensor)
            MethodSignature {
                name: "add".to_string(),
                params: vec![ParamType::AnyTensorOrNumeric],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "sub".to_string(),
                params: vec![ParamType::AnyTensorOrNumeric],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "mul".to_string(),
                params: vec![ParamType::AnyTensorOrNumeric],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "div".to_string(),
                params: vec![ParamType::AnyTensorOrNumeric],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "mod".to_string(),
                params: vec![ParamType::AnyTensorOrNumeric],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            // set(...) -> Void
            MethodSignature {
                name: "set".to_string(),
                params: vec![ParamType::AnyNumeric], // Varargs for indices and value
                return_type: ReturnType::Void,
                is_varargs: true,
                min_args: 2,
            },
            // detach(requires_grad) -> Tensor
            MethodSignature {
                name: "detach".to_string(),
                params: vec![ParamType::Bool],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: true, // Allow 0 or 1 arg
                min_args: 0,
            },
            // enable_grad() -> Same
            MethodSignature {
                name: "enable_grad".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
        ];

        let mut method_map = HashMap::new();
        for sig in tensor_methods {
            method_map.insert(sig.name.clone(), sig);
        }
        self.methods.insert("Tensor".to_string(), method_map);
    }

    /// Register numeric type methods (F32, F64, I64, I32)
    fn register_numeric_methods(&mut self) {
        let unary_math_names = [
            "abs",
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "cbrt",
            "ceil",
            "cos",
            "cosh",
            "exp",
            "exp2",
            "exp_m1",
            "floor",
            "fract",
            "ln",
            "ln_1p",
            "log10",
            "log2",
            "recip",
            "round",
            "signum",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            "to_degrees",
            "to_radians",
            "trunc",
        ];
        let binary_math_names = ["atan2", "copysign", "hypot", "log", "powf", "pow"];

        // F32 methods
        let mut f32_map = HashMap::new();
        for name in unary_math_names {
            f32_map.insert(
                name.to_string(),
                MethodSignature {
                    name: name.to_string(),
                    params: vec![],
                    return_type: ReturnType::SameAsReceiver,
                    is_varargs: false,
                    min_args: 0,
                },
            );
        }
        for name in binary_math_names {
            f32_map.insert(
                name.to_string(),
                MethodSignature {
                    name: name.to_string(),
                    params: vec![ParamType::AnyNumeric],
                    return_type: ReturnType::SameAsReceiver,
                    is_varargs: false,
                    min_args: 1,
                },
            );
        }
        // powi(i32/i64)
        f32_map.insert(
            "powi".to_string(),
            MethodSignature {
                name: "powi".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
        );

        self.methods.insert("F32".to_string(), f32_map.clone());
        self.methods.insert("F64".to_string(), f32_map);

        // I64 methods
        let mut i64_map = HashMap::new();
        let i64_methods = vec![
            MethodSignature {
                name: "abs".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "signum".to_string(),
                params: vec![],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "is_positive".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::Bool),
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "is_negative".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::Bool),
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "div_euclid".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "rem_euclid".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "pow".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::SameAsReceiver,
                is_varargs: false,
                min_args: 1,
            },
        ];
        for sig in i64_methods {
            i64_map.insert(sig.name.clone(), sig);
        }
        self.methods.insert("I64".to_string(), i64_map.clone());
        self.methods.insert("I32".to_string(), i64_map);
    }

    /// Register File methods
    fn register_file_methods(&mut self) {
        let file_methods = vec![
            MethodSignature {
                name: "read_string".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::UserDefined("String".to_string())),
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "write_string".to_string(),
                params: vec![ParamType::Exact(Type::UserDefined("String".to_string()))],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "read_to_end".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::Vec(Box::new(Type::U8))),
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "write".to_string(),
                params: vec![ParamType::Exact(Type::Vec(Box::new(Type::U8)))],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "close".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "free".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
        ];

        let mut file_map = HashMap::new();
        for sig in file_methods {
            file_map.insert(sig.name.clone(), sig);
        }
        self.methods.insert("File".to_string(), file_map);
    }

    /// Register Path methods
    fn register_path_methods(&mut self) {
        let path_methods = vec![
            MethodSignature {
                name: "join".to_string(),
                params: vec![ParamType::Exact(Type::UserDefined("String".to_string()))],
                return_type: ReturnType::Exact(Type::UserDefined("Path".to_string())),
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "exists".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::Bool),
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "is_dir".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::Bool),
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "is_file".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::Bool),
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "to_string".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::UserDefined("String".to_string())),
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "free".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
        ];

        let mut path_map = HashMap::new();
        for sig in path_methods {
            path_map.insert(sig.name.clone(), sig);
        }
        self.methods.insert("Path".to_string(), path_map);
    }

    /// Register VarBuilder methods
    fn register_varbuilder_methods(&mut self) {
        let vb_methods = vec![
            MethodSignature {
                name: "get".to_string(),
                params: vec![ParamType::Exact(Type::UserDefined("String".to_string())), ParamType::ShapeArray],
                return_type: ReturnType::TensorSameElementType(0),
                is_varargs: false,
                min_args: 2,
            },
            MethodSignature {
                name: "grad".to_string(),
                params: vec![ParamType::Exact(Type::UserDefined("String".to_string()))],
                return_type: ReturnType::TensorSameElementType(0),
                is_varargs: false,
                min_args: 1,
            },
        ];

        let mut vb_map = HashMap::new();
        for sig in vb_methods {
            vb_map.insert(sig.name.clone(), sig);
        }
        self.methods.insert("VarBuilder".to_string(), vb_map);
    }

    /// Register String methods
    fn register_string_methods(&mut self) {
        let string_methods = vec![
            MethodSignature {
                name: "len".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::I64),
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "concat".to_string(),
                params: vec![ParamType::Exact(Type::UserDefined("String".to_string()))],
                return_type: ReturnType::Exact(Type::UserDefined("String".to_string())),
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "char_at".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::Exact(Type::UserDefined("String".to_string())),
                is_varargs: false,
                min_args: 1,
            },
            MethodSignature {
                name: "print".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "to_i64".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::I64),
                is_varargs: false,
                min_args: 0,
            },
            MethodSignature {
                name: "display".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
        ];

        let mut string_map = HashMap::new();
        for sig in string_methods {
            string_map.insert(sig.name.clone(), sig);
        }
        self.methods.insert("String".to_string(), string_map);
    }

    /// Register Vec methods
    fn register_vec_methods(&mut self) {
        let mut vec_u8_methods = HashMap::new();
        vec_u8_methods.insert(
            "len".to_string(),
            MethodSignature {
                name: "len".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::I64),
                is_varargs: false,
                min_args: 0,
            },
        );
        vec_u8_methods.insert(
            "read_i32_be".to_string(),
            MethodSignature {
                name: "read_i32_be".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::I64),
                is_varargs: false,
                min_args: 0,
            },
        );
        vec_u8_methods.insert(
            "free".to_string(),
            MethodSignature {
                name: "free".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
        );
        self.methods.insert("Vec<U8>".to_string(), vec_u8_methods);
    }

    /// Register ML-related special types (Tokenizer, KVCache, Map)
    fn register_ml_methods(&mut self) {
        // Tokenizer
        let mut tokenizer_methods = HashMap::new();
        tokenizer_methods.insert(
            "encode".to_string(),
            MethodSignature {
                name: "encode".to_string(),
                params: vec![ParamType::Exact(Type::UserDefined("String".to_string()))],
                return_type: ReturnType::Exact(Type::Tensor(Box::new(Type::I64), 1)),
                is_varargs: false,
                min_args: 1,
            },
        );
        tokenizer_methods.insert(
            "decode".to_string(),
            MethodSignature {
                name: "decode".to_string(),
                params: vec![ParamType::Exact(Type::Tensor(Box::new(Type::I64), 1))],
                return_type: ReturnType::Exact(Type::UserDefined("String".to_string())),
                is_varargs: false,
                min_args: 1,
            },
        );
        tokenizer_methods.insert(
            "token_id".to_string(),
            MethodSignature {
                name: "token_id".to_string(),
                params: vec![ParamType::Exact(Type::UserDefined("String".to_string()))],
                return_type: ReturnType::Exact(Type::I64),
                is_varargs: false,
                min_args: 1,
            },
        );
        tokenizer_methods.insert(
            "vocab_size".to_string(),
            MethodSignature {
                name: "vocab_size".to_string(),
                params: vec![],
                return_type: ReturnType::Exact(Type::I64),
                is_varargs: false,
                min_args: 0,
            },
        );
        tokenizer_methods.insert(
            "free".to_string(),
            MethodSignature {
                name: "free".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
        );
        self.methods.insert("Tokenizer".to_string(), tokenizer_methods);

        // KVCache
        let mut kvcache_methods = HashMap::new();
        kvcache_methods.insert(
            "get_k".to_string(),
            MethodSignature {
                name: "get_k".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::Exact(Type::Tensor(Box::new(Type::F32), 0)),
                is_varargs: false,
                min_args: 1,
            },
        );
        kvcache_methods.insert(
            "get_v".to_string(),
            MethodSignature {
                name: "get_v".to_string(),
                params: vec![ParamType::AnyInt],
                return_type: ReturnType::Exact(Type::Tensor(Box::new(Type::F32), 0)),
                is_varargs: false,
                min_args: 1,
            },
        );
        kvcache_methods.insert(
            "update".to_string(),
            MethodSignature {
                name: "update".to_string(),
                params: vec![ParamType::AnyInt, ParamType::AnyTensor, ParamType::AnyTensor],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 3,
            },
        );
        kvcache_methods.insert(
            "free".to_string(),
            MethodSignature {
                name: "free".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
        );
        self.methods.insert("KVCache".to_string(), kvcache_methods);

        // Map
        let mut map_methods = HashMap::new();
        map_methods.insert(
            "get".to_string(),
            MethodSignature {
                name: "get".to_string(),
                params: vec![ParamType::Exact(Type::UserDefined("String".to_string()))],
                return_type: ReturnType::Exact(Type::Tensor(Box::new(Type::F32), 0)),
                is_varargs: false,
                min_args: 1,
            },
        );
        map_methods.insert(
            "get_1d".to_string(),
            MethodSignature {
                name: "get_1d".to_string(),
                params: vec![ParamType::Exact(Type::UserDefined("String".to_string()))],
                return_type: ReturnType::Exact(Type::Tensor(Box::new(Type::F32), 1)),
                is_varargs: false,
                min_args: 1,
            },
        );
        map_methods.insert(
            "get_quantized".to_string(),
            MethodSignature {
                name: "get_quantized".to_string(),
                params: vec![ParamType::Exact(Type::UserDefined("String".to_string()))],
                return_type: ReturnType::Exact(Type::I64),
                is_varargs: false,
                min_args: 1,
            },
        );
        map_methods.insert(
            "set".to_string(),
            MethodSignature {
                name: "set".to_string(),
                params: vec![
                    ParamType::Exact(Type::UserDefined("String".to_string())),
                    ParamType::Exact(Type::UserDefined("String".to_string())),
                ],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 2,
            },
        );
        map_methods.insert(
            "free".to_string(),
            MethodSignature {
                name: "free".to_string(),
                params: vec![],
                return_type: ReturnType::Void,
                is_varargs: false,
                min_args: 0,
            },
        );
        self.methods.insert("Map".to_string(), map_methods);
    }

    /// Get method signature for a given type and method name
    pub fn get_method(&self, type_name: &str, method_name: &str) -> Option<&MethodSignature> {
        self.methods
            .get(type_name)
            .and_then(|methods| methods.get(method_name))
    }

    /// Convert Type to registry key string
    pub fn type_to_key(ty: &Type) -> String {
        match ty {
            Type::Tensor(_, _) => "Tensor".to_string(),
            Type::TensorShaped(_, _) => "Tensor".to_string(),
            Type::F32 => "F32".to_string(),
            Type::F64 => "F64".to_string(),
            Type::I64 => "I64".to_string(),
            Type::I32 => "I32".to_string(),
            Type::Bool => "Bool".to_string(),
            Type::UserDefined(name) => name.clone(),
            Type::Struct(name) => name.clone(),
            Type::Enum(name) => name.clone(),
            Type::Vec(inner) => format!("Vec<{}>", Self::type_to_key(inner)),
            _ => "Unknown".to_string(),
        }
    }

    /// Check if an actual type matches a parameter type specification
    pub fn matches_param_type(actual: &Type, expected: &ParamType, receiver: &Type) -> bool {
        match expected {
            ParamType::Exact(ty) => Self::types_compatible(actual, ty),
            ParamType::AnyTensor => matches!(actual, Type::Tensor(_, _) | Type::TensorShaped(_, _)),
            ParamType::TensorOf(inner) => {
                if let Type::Tensor(actual_inner, _) = actual {
                    Self::types_compatible(actual_inner, inner)
                } else {
                    false
                }
            }
            ParamType::ShapeArray => {
                matches!(
                    actual,
                    Type::Tensor(_, _)
                        | Type::TensorShaped(_, _)
                        | Type::ScalarArray(_, _)
                        | Type::Vec(_)
                )
            }
            ParamType::AnyInt => matches!(actual, Type::I64 | Type::I32),
            ParamType::AnyNumeric => {
                matches!(actual, Type::F32 | Type::F64 | Type::I64 | Type::I32)
            }
            ParamType::SameAsReceiver => Self::types_compatible(actual, receiver),
            ParamType::Bool => matches!(actual, Type::Bool),
            ParamType::AnyTensorOrNumeric => {
                matches!(
                    actual,
                    Type::Tensor(_, _)
                        | Type::TensorShaped(_, _)
                        | Type::ScalarArray(_, _)
                        | Type::F32
                        | Type::F64
                        | Type::I64
                        | Type::I32
                )
            }
        }
    }

    /// Check if two types are compatible
    fn types_compatible(a: &Type, b: &Type) -> bool {
        match (a, b) {
            (Type::Tensor(_, _), Type::Tensor(_, _)) => true,
            (Type::TensorShaped(_, _), Type::Tensor(_, _)) => true,
            (Type::Tensor(_, _), Type::TensorShaped(_, _)) => true,
            _ => a == b,
        }
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}
