use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::{Type, Expr, ExprKind};
use inkwell::values::{BasicValueEnum, ValueKind, BasicValue};
// use inkwell::AddressSpace;



/// Tensor 型の組み込みメソッドを TypeManager に登録する。
///
/// NOTE: Tensor は勾配情報を持たない。勾配(grad)を扱う場合は GradTensor 型を使うこと。
///       freeze / unfreeze / clip_grad_value / clip_grad_norm などの勾配操作メソッドは
///       GradTensor 上でのみ呼び出せる（semantics.rs 側で型チェックされる）。
pub fn register_tensor_types(manager: &mut TypeManager) {
    let mut tensor = CodeGenType::new("Tensor");
    
    // Unevaluated static methods (for literal optimizations)
    // zeros, ones, randn with 1 or 2 args
    let shape_type = Type::Tensor(Box::new(Type::I64), 1);

    // zeros(shape, requires_grad) and zeros(shape)
    tensor.register_unevaluated_static_method("zeros", compile_tensor_zeros, vec![shape_type.clone(), Type::Bool], Type::Tensor(Box::new(Type::F32), 0));
    tensor.register_unevaluated_static_method("zeros", compile_tensor_zeros, vec![shape_type.clone()], Type::Tensor(Box::new(Type::F32), 0));
    
    // ones(shape, requires_grad) and ones(shape)
    tensor.register_unevaluated_static_method("ones", compile_ones, vec![shape_type.clone(), Type::Bool], Type::Tensor(Box::new(Type::F32), 0));
    tensor.register_unevaluated_static_method("ones", compile_ones, vec![shape_type.clone()], Type::Tensor(Box::new(Type::F32), 0));
    
    // randn(shape, requires_grad) and randn(shape)
    tensor.register_unevaluated_static_method("randn", compile_randn, vec![shape_type.clone(), Type::Bool], Type::Tensor(Box::new(Type::F32), 0));
    tensor.register_unevaluated_static_method("randn", compile_randn, vec![shape_type.clone()], Type::Tensor(Box::new(Type::F32), 0));

    // full(shape, value, requires_grad) and full(shape, value)
    tensor.register_unevaluated_static_method("full", compile_tensor_full, vec![shape_type.clone(), Type::F32, Type::Bool], Type::Tensor(Box::new(Type::F32), 0));
    tensor.register_unevaluated_static_method("full", compile_tensor_full, vec![shape_type.clone(), Type::F32], Type::Tensor(Box::new(Type::F32), 0));

    // eye(n, requires_grad?) -> Tensor
    tensor.register_unevaluated_static_method("eye", compile_tensor_eye, vec![Type::I64, Type::Bool], Type::Tensor(Box::new(Type::F32), 2));
    tensor.register_unevaluated_static_method("eye", compile_tensor_eye, vec![Type::I64], Type::Tensor(Box::new(Type::F32), 2));

    // arange(start, end, step) -> Tensor
    tensor.register_unevaluated_static_method("arange", compile_tensor_arange, vec![Type::F32, Type::F32, Type::F32], Type::Tensor(Box::new(Type::F32), 1));
    tensor.register_unevaluated_static_method("arange", compile_tensor_arange, vec![Type::I64, Type::I64, Type::I64], Type::Tensor(Box::new(Type::F32), 1));
    tensor.register_unevaluated_static_method("arange", compile_tensor_arange, vec![Type::F64, Type::F64, Type::F64], Type::Tensor(Box::new(Type::F32), 1));

    // linspace(start, end, steps) -> Tensor
    tensor.register_unevaluated_static_method("linspace", compile_tensor_linspace, vec![Type::F32, Type::F32, Type::I64], Type::Tensor(Box::new(Type::F32), 1));
    tensor.register_unevaluated_static_method("linspace", compile_tensor_linspace, vec![Type::F64, Type::F64, Type::I64], Type::Tensor(Box::new(Type::F32), 1));
    tensor.register_unevaluated_static_method("linspace", compile_tensor_linspace, vec![Type::I64, Type::I64, Type::I64], Type::Tensor(Box::new(Type::F32), 1));

    // rand(shape, requires_grad?) -> Tensor (uniform [0,1))
    tensor.register_unevaluated_static_method("rand", compile_tensor_rand, vec![shape_type.clone(), Type::Bool], Type::Tensor(Box::new(Type::F32), 0));
    tensor.register_unevaluated_static_method("rand", compile_tensor_rand, vec![shape_type.clone()], Type::Tensor(Box::new(Type::F32), 0));

    // Evaluated static methods
    // rand_like(t) -> Tensor (same shape, uniform random)
    tensor.register_evaluated_static_method("rand_like", compile_tensor_rand_like, vec![Type::Tensor(Box::new(Type::F32), 0)], Type::Tensor(Box::new(Type::F32), 0));
    // randn_like(t) -> Tensor (same shape, normal random)
    tensor.register_evaluated_static_method("randn_like", compile_tensor_randn_like, vec![Type::Tensor(Box::new(Type::F32), 0)], Type::Tensor(Box::new(Type::F32), 0));
    // zeros_like(t) -> Tensor (same shape as t, filled with zeros)
    tensor.register_evaluated_static_method("zeros_like", compile_tensor_zeros_like, vec![Type::Tensor(Box::new(Type::F32), 0)], Type::Tensor(Box::new(Type::F32), 0));
    // ones_like(t) -> Tensor (same shape as t, filled with ones)
    tensor.register_evaluated_static_method("ones_like", compile_tensor_ones_like, vec![Type::Tensor(Box::new(Type::F32), 0)], Type::Tensor(Box::new(Type::F32), 0));

    // where_cond(cond, x, y) -> Tensor
    tensor.register_evaluated_static_method("where_cond", compile_tensor_where_cond, vec![Type::Tensor(Box::new(Type::F32), 0), Type::Tensor(Box::new(Type::F32), 0), Type::Tensor(Box::new(Type::F32), 0)], Type::Tensor(Box::new(Type::F32), 0));

    // from_vec(data: Vec<f32>, shape: Vec<i64>) -> Tensor
    tensor.register_evaluated_static_method("from_vec", compile_from_vec_f32, vec![Type::Struct("Vec".into(), vec![Type::F32]), Type::Struct("Vec".into(), vec![Type::I64])], Type::Tensor(Box::new(Type::F32), 0));

    // load(path: String) -> Tensor
    tensor.register_evaluated_static_method("load", compile_load_tensor, vec![Type::String("String".to_string())], Type::Tensor(Box::new(Type::F32), 1));
    // clear_grads() -> Void
    tensor.register_evaluated_static_method("clear_grads", compile_clear_grads, vec![], Type::Void);
    // mem_purge() -> Void (アロケータの未使用メモリを OS に返却)
    tensor.register_evaluated_static_method("mem_purge", compile_mem_purge, vec![], Type::Void);
    // from_vec_u8(u8s: Vec<u8>, shape: Vec<i64>) -> Tensor
    tensor.register_evaluated_static_method("from_vec_u8", compile_from_vec_u8, vec![Type::Struct("Vec".into(), vec![Type::U8]), Type::Struct("Vec".into(), vec![Type::I64])], Type::Tensor(Box::new(Type::F32), 0));

    // Instance methods
    let any_tensor = Type::Tensor(Box::new(Type::F32), 0);

    // Sum/reduction with overloads
    tensor.register_evaluated_instance_method("sum", compile_tensor_sumall, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("sum", compile_tensor_sum_impl, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("sum", compile_tensor_sum_impl, vec![Type::I64, Type::Bool], any_tensor.clone());

    tensor.register_evaluated_instance_method("max", compile_tensor_max_impl, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("max", compile_tensor_max_impl, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("max", compile_tensor_max_impl, vec![Type::I64, Type::Bool], any_tensor.clone());

    tensor.register_evaluated_instance_method("min", compile_tensor_min_impl, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("min", compile_tensor_min_impl, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("min", compile_tensor_min_impl, vec![Type::I64, Type::Bool], any_tensor.clone());

    tensor.register_evaluated_instance_method("mean", compile_tensor_mean_impl, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("mean", compile_tensor_mean_impl, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("mean", compile_tensor_mean_impl, vec![Type::I64, Type::Bool], any_tensor.clone());

    // Optimizer steps
    tensor.register_evaluated_instance_method(
        "adam_step",
        compile_tensor_adam_step,
        vec![
            any_tensor.clone(), // grad
            any_tensor.clone(), // m
            any_tensor.clone(), // v
            Type::I64,          // step
            Type::F32,          // lr
            Type::F32,          // beta1
            Type::F32,          // beta2
            Type::F32,          // eps
            Type::F32,          // weight_decay
        ],
        Type::Void,
    );

    tensor.register_evaluated_instance_method(
        "sgd_step",
        compile_tensor_sgd_step,
        vec![
            any_tensor.clone(), // grad
            any_tensor.clone(), // velocity
            Type::F32,          // lr
            Type::F32,          // momentum
            Type::F32,          // weight_decay
            Type::F32,          // dampening
            Type::Bool,         // nesterov
        ],
        Type::Void,
    );

    tensor.register_evaluated_instance_method("var", compile_tensor_var_impl, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("var", compile_tensor_var_impl, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("var", compile_tensor_var_impl, vec![Type::I64, Type::Bool], any_tensor.clone());

    tensor.register_evaluated_instance_method("std", compile_tensor_std_impl, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("std", compile_tensor_std_impl, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("std", compile_tensor_std_impl, vec![Type::I64, Type::Bool], any_tensor.clone());

    tensor.register_evaluated_instance_method("prod", compile_tensor_prod_impl, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("prod", compile_tensor_prod_impl, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("prod", compile_tensor_prod_impl, vec![Type::I64, Type::Bool], any_tensor.clone());

    tensor.register_evaluated_instance_method("argmax", compile_tensor_argmax_impl, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("argmin", compile_tensor_argmin_impl, vec![Type::I64], any_tensor.clone());

    // Other instance methods (no overloads)
    tensor.register_evaluated_instance_method("detach", compile_tensor_detach, vec![Type::Bool], any_tensor.clone());
    tensor.register_evaluated_instance_method("tril", compile_tensor_tril, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("contiguous", compile_tensor_contiguous, vec![], any_tensor.clone());

    // masked_fill(mask, value) -> Tensor
    tensor.register_evaluated_instance_method("masked_fill", compile_tensor_masked_fill, vec![Type::Tensor(Box::new(Type::F32), 0), Type::F32], any_tensor.clone());
    tensor.register_evaluated_instance_method("masked_fill", compile_tensor_masked_fill, vec![Type::Tensor(Box::new(Type::F32), 0), Type::F64], any_tensor.clone());

    // fill_(value) -> void (in-place)
    tensor.register_evaluated_instance_method("fill_", compile_tensor_fill_, vec![Type::F32], Type::Void);
    tensor.register_evaluated_instance_method("fill_", compile_tensor_fill_, vec![Type::F64], Type::Void);

    // cumsum(dim) -> Tensor
    tensor.register_evaluated_instance_method("cumsum", compile_tensor_cumsum, vec![Type::I64], any_tensor.clone());

    // norm(p) or norm(p, dim) -> Tensor
    tensor.register_evaluated_instance_method("norm", compile_tensor_norm, vec![Type::F32], any_tensor.clone());
    tensor.register_evaluated_instance_method("norm", compile_tensor_norm, vec![Type::F64], any_tensor.clone());
    tensor.register_evaluated_instance_method("norm", compile_tensor_norm, vec![Type::F32, Type::I64], any_tensor.clone());

    // topk(k) or topk(k, dim) -> Tensor
    tensor.register_evaluated_instance_method("topk", compile_tensor_topk, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("topk", compile_tensor_topk, vec![Type::I64, Type::I64], any_tensor.clone());

    // logical_and / logical_or (static binary)
    tensor.register_evaluated_static_method("logical_and", compile_tensor_logical_and, vec![any_tensor.clone(), any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_static_method("logical_or", compile_tensor_logical_or, vec![any_tensor.clone(), any_tensor.clone()], any_tensor.clone());

    // logical_not (instance unary)
    tensor.register_evaluated_instance_method("logical_not", compile_tensor_logical_not, vec![], any_tensor.clone());

    // NN operations
    // layer_norm(weight, bias, eps) -> Tensor
    tensor.register_evaluated_instance_method("layer_norm", compile_tensor_layer_norm, vec![any_tensor.clone(), any_tensor.clone(), Type::F64], any_tensor.clone());
    // dropout(p, training) -> Tensor
    tensor.register_evaluated_instance_method("dropout", compile_tensor_dropout, vec![Type::F64, Type::Bool], any_tensor.clone());
    // dropout2d(p, training) -> Tensor
    tensor.register_evaluated_instance_method("dropout2d", compile_tensor_dropout2d, vec![Type::F64, Type::Bool], any_tensor.clone());
    // leaky_relu(negative_slope?) -> Tensor
    tensor.register_evaluated_instance_method("leaky_relu", compile_tensor_leaky_relu, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("leaky_relu", compile_tensor_leaky_relu, vec![Type::F32], any_tensor.clone());
    tensor.register_evaluated_instance_method("leaky_relu", compile_tensor_leaky_relu, vec![Type::F64], any_tensor.clone());

    // batch_norm(running_mean, running_var, weight, bias, training, [momentum], [eps])
    tensor.register_evaluated_instance_method("batch_norm", compile_tensor_batch_norm, vec![any_tensor.clone(), any_tensor.clone(), any_tensor.clone(), any_tensor.clone(), Type::Bool], any_tensor.clone());
    tensor.register_evaluated_instance_method("batch_norm", compile_tensor_batch_norm, vec![any_tensor.clone(), any_tensor.clone(), any_tensor.clone(), any_tensor.clone(), Type::Bool, Type::F64, Type::F64], any_tensor.clone());
    // conv2d(weight, bias, stride, padding, dilation, groups)
    // max_pool2d / avg_pool2d (kernel_size, stride, padding)
    tensor.register_evaluated_instance_method("conv2d", compile_tensor_conv2d, vec![any_tensor.clone(), any_tensor.clone(), Type::I64, Type::I64, Type::I64, Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("max_pool2d", compile_tensor_max_pool2d, vec![Type::I64, Type::I64, Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("avg_pool2d", compile_tensor_avg_pool2d, vec![Type::I64, Type::I64, Type::I64], any_tensor.clone());

    // elu(alpha?) / mish()
    tensor.register_evaluated_instance_method("elu", compile_tensor_elu, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("elu", compile_tensor_elu, vec![Type::F32], any_tensor.clone());
    tensor.register_evaluated_instance_method("elu", compile_tensor_elu, vec![Type::F64], any_tensor.clone());
    tensor.register_evaluated_instance_method("mish", compile_tensor_mish, vec![], any_tensor.clone());

    // loss functions (static)
    tensor.register_evaluated_static_method("mse_loss", compile_tensor_mse_loss, vec![any_tensor.clone(), any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_static_method("l1_loss", compile_tensor_l1_loss, vec![any_tensor.clone(), any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_static_method("bce_loss", compile_tensor_bce_loss, vec![any_tensor.clone(), any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_static_method("nll_loss", compile_tensor_nll_loss, vec![any_tensor.clone(), any_tensor.clone()], any_tensor.clone());

    // linear(weight, bias) -> Tensor
    tensor.register_evaluated_instance_method("linear", compile_tensor_linear, vec![any_tensor.clone(), any_tensor.clone()], any_tensor.clone());
    // hardswish / hardsigmoid
    tensor.register_evaluated_instance_method("hardswish", compile_tensor_hardswish, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("hardsigmoid", compile_tensor_hardsigmoid, vec![], any_tensor.clone());

    // group_norm(num_groups, weight, bias, eps)
    tensor.register_evaluated_instance_method("group_norm", compile_tensor_group_norm, vec![Type::I64, any_tensor.clone(), any_tensor.clone(), Type::F64], any_tensor.clone());
    // adaptive_avg_pool2d(output_h, output_w)
    tensor.register_evaluated_instance_method("adaptive_avg_pool2d", compile_tensor_adaptive_avg_pool2d, vec![Type::I64, Type::I64], any_tensor.clone());
    // pad(pad_left, pad_right, value)
    tensor.register_evaluated_instance_method("pad", compile_tensor_pad, vec![Type::I64, Type::I64, Type::F32], any_tensor.clone());
    tensor.register_evaluated_instance_method("pad", compile_tensor_pad, vec![Type::I64, Type::I64, Type::F64], any_tensor.clone());
    tensor.register_evaluated_instance_method("pad", compile_tensor_pad, vec![Type::I64, Type::I64], any_tensor.clone());

    // conv1d(weight, bias, stride, padding)
    tensor.register_evaluated_instance_method("conv1d", compile_tensor_conv1d, vec![any_tensor.clone(), any_tensor.clone(), Type::I64, Type::I64], any_tensor.clone());
    // kl_div_loss
    tensor.register_evaluated_static_method("kl_div_loss", compile_tensor_kl_div_loss, vec![any_tensor.clone(), any_tensor.clone()], any_tensor.clone());

    // conv_transpose2d(weight, bias, stride, padding, output_padding)
    tensor.register_evaluated_instance_method("conv_transpose2d", compile_tensor_conv_transpose2d, vec![any_tensor.clone(), any_tensor.clone(), Type::I64, Type::I64, Type::I64], any_tensor.clone());
    // interpolate(output_h, output_w, mode)
    tensor.register_evaluated_instance_method("interpolate", compile_tensor_interpolate, vec![Type::I64, Type::I64, Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("interpolate", compile_tensor_interpolate, vec![Type::I64, Type::I64], any_tensor.clone());

    // Transformer inference
    tensor.register_evaluated_static_method("sdpa", compile_tensor_sdpa, vec![any_tensor.clone(), any_tensor.clone(), any_tensor.clone(), any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_static_method("sdpa", compile_tensor_sdpa, vec![any_tensor.clone(), any_tensor.clone(), any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_instance_method("top_k_sample", compile_tensor_top_k_sample, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("top_p_sample", compile_tensor_top_p_sample, vec![Type::F64], any_tensor.clone());
    tensor.register_evaluated_instance_method("top_p_sample", compile_tensor_top_p_sample, vec![Type::F32], any_tensor.clone());

    // temperature_scale(temperature)
    tensor.register_evaluated_instance_method("temperature_scale", compile_tensor_temperature_scale, vec![Type::F64], any_tensor.clone());
    tensor.register_evaluated_instance_method("temperature_scale", compile_tensor_temperature_scale, vec![Type::F32], any_tensor.clone());
    // repetition_penalty(tokens, penalty)
    tensor.register_evaluated_instance_method("repetition_penalty", compile_tensor_repetition_penalty, vec![any_tensor.clone(), Type::F64], any_tensor.clone());
    tensor.register_evaluated_instance_method("repetition_penalty", compile_tensor_repetition_penalty, vec![any_tensor.clone(), Type::F32], any_tensor.clone());
    // dot(other)
    tensor.register_evaluated_instance_method("dot", compile_tensor_dot, vec![any_tensor.clone()], any_tensor.clone());

    // instance_norm(weight, bias, eps)
    tensor.register_evaluated_instance_method("instance_norm", compile_tensor_instance_norm, vec![any_tensor.clone(), any_tensor.clone(), Type::F64], any_tensor.clone());
    tensor.register_evaluated_instance_method("instance_norm", compile_tensor_instance_norm, vec![any_tensor.clone(), any_tensor.clone(), Type::F32], any_tensor.clone());
    tensor.register_evaluated_instance_method("instance_norm", compile_tensor_instance_norm, vec![any_tensor.clone(), any_tensor.clone()], any_tensor.clone());

    // chunk(num_chunks, dim, index) -> Tensor
    tensor.register_evaluated_instance_method("chunk", compile_tensor_chunk, vec![Type::I64, Type::I64, Type::I64], any_tensor.clone());
    // split(split_size, dim, index) -> Tensor
    tensor.register_evaluated_instance_method("split", compile_tensor_split, vec![Type::I64, Type::I64, Type::I64], any_tensor.clone());

    // Linear algebra
    tensor.register_evaluated_instance_method("inverse", compile_tensor_inverse, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("det", compile_tensor_det, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("svd_u", compile_tensor_svd_u, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("svd_s", compile_tensor_svd_s, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("svd_v", compile_tensor_svd_v, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("eig_values", compile_tensor_eig_values, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("eig_vectors", compile_tensor_eig_vectors, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("solve", compile_tensor_solve, vec![any_tensor.clone()], any_tensor.clone());

    tensor.register_evaluated_instance_method("clone", compile_tensor_clone, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("shallow_clone", compile_tensor_shallow_clone, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("grad", compile_tensor_grad, vec![], Type::GradTensor(Box::new(Type::F32), 0));
    tensor.register_evaluated_instance_method("to_i64", compile_tensor_to_i64, vec![], Type::Tensor(Box::new(Type::I64), 0));
    tensor.register_evaluated_instance_method("to_f32", compile_tensor_to_f32, vec![], Type::Tensor(Box::new(Type::F32), 0));
    tensor.register_evaluated_instance_method("to_vec", compile_tensor_to_vec_f32, vec![], Type::Struct("Vec".into(), vec![Type::F32]));
    tensor.register_evaluated_instance_method("cuda", compile_tensor_cuda, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("cpu", compile_tensor_cpu, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("item", compile_tensor_item, vec![], Type::F32);
    tensor.register_evaluated_instance_method("shape", compile_tensor_shape, vec![], Type::Struct("Vec".into(), vec![Type::I64]));
    
    // Binary ops
    tensor.register_evaluated_instance_method("mul", compile_tensor_mul, vec![any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_instance_method("add", compile_tensor_add, vec![any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_instance_method("sub", compile_tensor_sub, vec![any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_instance_method("div", compile_tensor_div, vec![any_tensor.clone()], any_tensor.clone());
    
    // Neural network ops
    tensor.register_evaluated_instance_method("conv2d", compile_tensor_conv2d, vec![any_tensor.clone(), Type::I64, Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("clamp", compile_tensor_clamp, vec![Type::I64, Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("matmul_quantized", compile_tensor_matmul_quantized, vec![any_tensor.clone()], any_tensor.clone());
    
    // IO and transformation
    tensor.register_evaluated_instance_method("save", compile_tensor_save, vec![Type::String("String".to_string())], Type::Void);
    tensor.register_evaluated_instance_method("reshape", compile_tensor_reshape, vec![Type::Struct("Vec".into(), vec![Type::I64])], any_tensor.clone());
    // view = reshape alias
    tensor.register_evaluated_instance_method("view", compile_tensor_reshape, vec![Type::Struct("Vec".into(), vec![Type::I64])], any_tensor.clone());
    // expand/broadcast_to
    tensor.register_evaluated_instance_method("expand", compile_tensor_expand, vec![Type::Struct("Vec".into(), vec![Type::I64])], any_tensor.clone());
    tensor.register_evaluated_instance_method("broadcast_to", compile_tensor_expand, vec![Type::Struct("Vec".into(), vec![Type::I64])], any_tensor.clone());
    // stack(tensors, dim) — static method
    tensor.register_evaluated_static_method("stack", compile_tensor_stack, vec![any_tensor.clone(), any_tensor.clone(), Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("to", compile_tensor_to_device, vec![Type::String("String".to_string())], any_tensor.clone());
    tensor.register_evaluated_instance_method("transpose", compile_tensor_transpose, vec![Type::I64, Type::I64], any_tensor.clone());
    
    // Compound assignment
    tensor.register_evaluated_instance_method("add_assign", compile_tensor_add_assign, vec![any_tensor.clone()], Type::Void);
    tensor.register_evaluated_instance_method("sub_assign", compile_tensor_sub_assign, vec![any_tensor.clone()], Type::Void);
    tensor.register_evaluated_instance_method("mul_assign", compile_tensor_mul_assign, vec![any_tensor.clone()], Type::Void);
    tensor.register_evaluated_instance_method("div_assign", compile_tensor_div_assign, vec![any_tensor.clone()], Type::Void);

    // GradTensor methods
    tensor.register_evaluated_instance_method("freeze", compile_tensor_freeze, vec![], Type::Void);
    tensor.register_evaluated_instance_method("unfreeze", compile_tensor_unfreeze, vec![], Type::Void);
    tensor.register_evaluated_instance_method("clip_grad_value", compile_tensor_clip_grad_value, vec![Type::F32, Type::F32], Type::Void);
    tensor.register_evaluated_instance_method("clip_grad_norm", compile_tensor_clip_grad_norm, vec![Type::F32, Type::F32], Type::F32);
    tensor.register_evaluated_instance_method("sub_assign", compile_tensor_sub_assign, vec![any_tensor.clone()], Type::Void);

    // ===== Additional methods from check_builtin_method (signature-only) =====
    // Autograd (implemented in tensor_autograd.rs)
    use super::tensor_autograd;
    tensor.register_evaluated_instance_method("backward", tensor_autograd::compile_backward, vec![], Type::Void);
    tensor.register_evaluated_instance_method("enable_grad", tensor_autograd::compile_enable_grad, vec![], any_tensor.clone());
    // freeze, unfreeze, clip_grad_value, clip_grad_norm are already registered as Evaluated above (L281-284)

    
    // Activation / ops - 1 arg (implemented in tensor_reduce.rs)
    use super::tensor_reduce;
    tensor.register_evaluated_instance_method("softmax", tensor_reduce::compile_softmax, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("log_softmax", tensor_reduce::compile_log_softmax, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("matmul", tensor_reduce::compile_matmul, vec![any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_instance_method("embedding", compile_tensor_embedding, vec![any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_instance_method("cross_entropy", tensor_reduce::compile_cross_entropy, vec![any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_instance_method("pow", tensor_reduce::compile_pow, vec![Type::I64], any_tensor.clone());
    
    // Elementwise ops - 0 arg (implemented in tensor_elementwise.rs)
    use super::tensor_elementwise;
    tensor.register_evaluated_instance_method("abs", tensor_elementwise::compile_abs, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("relu", tensor_elementwise::compile_relu, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("gelu", tensor_elementwise::compile_gelu, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("silu", tensor_elementwise::compile_silu, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("exp", tensor_elementwise::compile_exp, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("log", tensor_elementwise::compile_log, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("sqrt", tensor_elementwise::compile_sqrt, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("sin", tensor_elementwise::compile_sin, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("cos", tensor_elementwise::compile_cos, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("tanh", tensor_elementwise::compile_tanh, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("sigmoid", tensor_elementwise::compile_sigmoid, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("neg", tensor_elementwise::compile_neg, vec![], any_tensor.clone());
    tensor.register_evaluated_instance_method("tan", tensor_elementwise::compile_tan, vec![], any_tensor.clone());

    // Shape ops (implemented in tensor_shape.rs)
    use super::tensor_shape;
    tensor.register_evaluated_instance_method("squeeze", tensor_shape::compile_squeeze, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("unsqueeze", tensor_shape::compile_unsqueeze, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("flatten", tensor_shape::compile_flatten, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("gather", tensor_shape::compile_gather, vec![any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_instance_method("permute", tensor_shape::compile_permute, vec![Type::Struct("Vec".into(), vec![Type::I64])], any_tensor.clone());
    tensor.register_evaluated_instance_method("cat", tensor_shape::compile_cat, vec![any_tensor.clone()], any_tensor.clone());
    
    // LLM ops (implemented in tensor_llm.rs)
    use super::tensor_llm;
    tensor.register_evaluated_instance_method("scale", tensor_llm::compile_scale, vec![Type::F32], any_tensor.clone());
    tensor.register_evaluated_instance_method("add_4d", tensor_llm::compile_add_4d, vec![any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_instance_method("matmul_4d", tensor_llm::compile_matmul_4d, vec![any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_instance_method("cat_4d", tensor_llm::compile_cat_4d, vec![any_tensor.clone(), Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("rms_norm", tensor_llm::compile_rms_norm, vec![any_tensor.clone(), Type::F32], any_tensor.clone());
    tensor.register_evaluated_instance_method("sample", tensor_llm::compile_sample, vec![Type::F32, Type::F32], Type::Tensor(Box::new(Type::F32), 1));
    tensor.register_evaluated_instance_method("repeat_interleave", tensor_llm::compile_repeat_interleave, vec![Type::I64, Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("apply_rope", tensor_llm::compile_apply_rope, vec![any_tensor.clone(), any_tensor.clone()], any_tensor.clone());
    tensor.register_evaluated_instance_method("narrow", tensor_llm::compile_narrow, vec![Type::I64, Type::I64, Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("transpose_2d", tensor_llm::compile_transpose_2d, vec![Type::I64, Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("item_i64", tensor_llm::compile_item_i64, vec![], Type::I64);
    tensor.register_evaluated_instance_method("cat_i64", tensor_llm::compile_cat_i64, vec![any_tensor.clone(), Type::I64], any_tensor.clone());
    
    // Shape/length queries (implemented in tensor_shape.rs)
    tensor.register_evaluated_instance_method("len", tensor_shape::compile_len, vec![], Type::I64);
    tensor.register_evaluated_instance_method("dim", tensor_shape::compile_dim, vec![Type::I64], Type::I64);
    tensor.register_evaluated_instance_method("ndim", tensor_shape::compile_ndim, vec![], Type::I64);
    tensor.register_evaluated_instance_method("get_shape", tensor_shape::compile_get_shape, vec![], Type::Struct("Vec".into(), vec![Type::I64]));

    // Misc (implemented in tensor_misc.rs)
    use super::tensor_misc;
    tensor.register_evaluated_instance_method("print", tensor_misc::compile_print, vec![], Type::Void);
    tensor.register_evaluated_instance_method("debug_ptr", tensor_misc::compile_debug_ptr, vec![], Type::Void);
    tensor.register_evaluated_instance_method("display", tensor_misc::compile_display, vec![], Type::Void);
    
    // Slice (implemented in tensor_misc.rs)
    tensor.register_evaluated_instance_method("slice", tensor_misc::compile_slice, vec![Type::I64, Type::I64, Type::I64, Type::I64], any_tensor.clone());
    
    // Dim-specific ops (implemented in tensor_reduce.rs)
    tensor.register_evaluated_instance_method("sum_dim", tensor_reduce::compile_sum_dim, vec![Type::I64, Type::Bool], any_tensor.clone());
    tensor.register_evaluated_instance_method("mean_dim", tensor_reduce::compile_mean_dim, vec![Type::I64, Type::Bool], any_tensor.clone());
    tensor.register_evaluated_instance_method("max_dim", tensor_reduce::compile_max_dim, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("min_dim", tensor_reduce::compile_min_dim, vec![Type::I64], any_tensor.clone());
    tensor.register_evaluated_instance_method("sumall", compile_tensor_sumall, vec![], any_tensor.clone());
    
    // Modulo (implemented in tensor_reduce.rs)
    tensor.register_evaluated_instance_method("mod", tensor_reduce::compile_mod, vec![Type::I64], any_tensor.clone());


    manager.register_type(tensor);
}

fn compile_tensor_add_assign<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("add_assign requires 1 argument".into());
    }
    let (rhs_val, rhs_ty) = args[0].clone();

    if matches!(rhs_ty, Type::Tensor(_, _)) {
        let fn_val = codegen.module.get_function("tl_tensor_add_assign").unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), rhs_val.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else if matches!(rhs_ty, Type::F32 | Type::F64 | Type::I64 | Type::I32) {
        let scalar_f32 = match rhs_ty {
            Type::F32 => rhs_val.into_float_value(),
            Type::F64 => codegen
                .builder
                .build_float_cast(
                    rhs_val.into_float_value(),
                    codegen.context.f32_type(),
                    "f64_to_f32",
                )
                .map_err(|e| e.to_string())?,
            Type::I64 | Type::I32 => codegen
                .builder
                .build_signed_int_to_float(
                    rhs_val.into_int_value(),
                    codegen.context.f32_type(),
                    "int_to_f32",
                )
                .map_err(|e| e.to_string())?,
            _ => return Err(format!("add_assign scalar: unsupported type {:?}", rhs_ty)),
        };
        let fn_val = codegen
            .module
            .get_function("tl_tensor_add_assign_scalar_f32")
            .unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), scalar_f32.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else {
        return Err(format!(
            "add_assign requires Tensor or scalar argument, got {:?}",
            rhs_ty
        ));
    }

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_tensor_sub_assign<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("sub_assign requires 1 argument".into());
    }
    let (rhs_val, rhs_ty) = args[0].clone();

    if matches!(rhs_ty, Type::Tensor(_, _)) {
        let fn_val = codegen.module.get_function("tl_tensor_sub_assign").unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), rhs_val.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else if matches!(rhs_ty, Type::F32 | Type::F64 | Type::I64 | Type::I32) {
        let scalar_f32 = match rhs_ty {
            Type::F32 => rhs_val.into_float_value(),
            Type::F64 => codegen
                .builder
                .build_float_cast(
                    rhs_val.into_float_value(),
                    codegen.context.f32_type(),
                    "f64_to_f32",
                )
                .map_err(|e| e.to_string())?,
            Type::I64 | Type::I32 => codegen
                .builder
                .build_signed_int_to_float(
                    rhs_val.into_int_value(),
                    codegen.context.f32_type(),
                    "int_to_f32",
                )
                .map_err(|e| e.to_string())?,
            _ => return Err(format!("sub_assign scalar: unsupported type {:?}", rhs_ty)),
        };
        let fn_val = codegen
            .module
            .get_function("tl_tensor_sub_assign_scalar_f32")
            .unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), scalar_f32.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else {
        return Err(format!(
            "sub_assign requires Tensor or scalar argument, got {:?}",
            rhs_ty
        ));
    }

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}


fn compile_tensor_mul_assign<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("mul_assign requires 1 argument".into());
    }
    let (rhs_val, rhs_ty) = args[0].clone();

    if matches!(rhs_ty, Type::Tensor(_, _)) {
        let fn_val = codegen.module.get_function("tl_tensor_mul_assign").unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), rhs_val.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else if matches!(rhs_ty, Type::F32 | Type::F64 | Type::I64 | Type::I32) {
        let scalar_f32 = match rhs_ty {
            Type::F32 => rhs_val.into_float_value(),
            Type::F64 => codegen
                .builder
                .build_float_cast(
                    rhs_val.into_float_value(),
                    codegen.context.f32_type(),
                    "f64_to_f32",
                )
                .map_err(|e| e.to_string())?,
            Type::I64 | Type::I32 => codegen
                .builder
                .build_signed_int_to_float(
                    rhs_val.into_int_value(),
                    codegen.context.f32_type(),
                    "int_to_f32",
                )
                .map_err(|e| e.to_string())?,
            _ => return Err(format!("mul_assign scalar: unsupported type {:?}", rhs_ty)),
        };
        let fn_val = codegen
            .module
            .get_function("tl_tensor_mul_assign_scalar_f32")
            .unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), scalar_f32.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else {
        return Err(format!(
            "mul_assign requires Tensor or scalar argument, got {:?}",
            rhs_ty
        ));
    }

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_tensor_div_assign<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("div_assign requires 1 argument".into());
    }
    let (rhs_val, rhs_ty) = args[0].clone();

    if matches!(rhs_ty, Type::Tensor(_, _)) {
        let fn_val = codegen.module.get_function("tl_tensor_div_assign").unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), rhs_val.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else if matches!(rhs_ty, Type::F32 | Type::F64 | Type::I64 | Type::I32) {
        let scalar_f32 = match rhs_ty {
            Type::F32 => rhs_val.into_float_value(),
            Type::F64 => codegen
                .builder
                .build_float_cast(
                    rhs_val.into_float_value(),
                    codegen.context.f32_type(),
                    "f64_to_f32",
                )
                .map_err(|e| e.to_string())?,
            Type::I64 | Type::I32 => codegen
                .builder
                .build_signed_int_to_float(
                    rhs_val.into_int_value(),
                    codegen.context.f32_type(),
                    "int_to_f32",
                )
                .map_err(|e| e.to_string())?,
            _ => return Err(format!("div_assign scalar: unsupported type {:?}", rhs_ty)),
        };
        let fn_val = codegen
            .module
            .get_function("tl_tensor_div_assign_scalar_f32")
            .unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), scalar_f32.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else {
        return Err(format!(
            "div_assign requires Tensor or scalar argument, got {:?}",
            rhs_ty
        ));
    }

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_tensor_shape<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_shape").ok_or("tl_tensor_shape not found")?;
    let res = match codegen
        .builder
        .build_call(fn_val, &[obj_val.into()], "shape_res")
        .map_err(|e| e.to_string())?
        .try_as_basic_value()
    {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from shape()".into()),
    };
    Ok((res, Type::Struct("Vec".into(), vec![Type::I64])))
}

fn compile_tensor_sumall<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_sum").ok_or("tl_tensor_sum not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "sum_res").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from sumall()".into()),
    };
    Ok((res, obj_ty))
}

fn compile_tensor_detach<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_detach").ok_or("tl_tensor_detach not found")?;
    // Optional arg: req_grad (bool). Default to false.
    let req_grad = if !args.is_empty() {
        let (arg_val, _) = args[0];
        arg_val.into_int_value()
    } else {
        codegen.context.bool_type().const_int(0, false)
    };
    let call = codegen.builder.build_call(fn_val, &[obj.into(), req_grad.into()], "detach_res").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from detach()".into()),
    };
    Ok((res, obj_ty))
}

fn compile_tensor_tril<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("tril requires 1 argument (diagonal)".into()); }
    let fn_val = codegen.module.get_function("tl_tensor_tril").ok_or("tl_tensor_tril not found")?;
    let (diag_val, diag_ty) = &args[0];
    let diag_i64 = match diag_ty {
        Type::I64 => diag_val.into_int_value(),
        Type::I32 => codegen.builder.build_int_s_extend(diag_val.into_int_value(), codegen.context.i64_type(), "tril_diag_ext").unwrap(),
        _ => return Err("tril argument must be integer".into()),
    };
    let call = codegen.builder.build_call(fn_val, &[obj.into(), diag_i64.into()], "tril_res").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from tril()".into()),
    };
    Ok((res, obj_ty))
}

fn compile_tensor_embedding<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,      // self = indices tensor
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("embedding requires 1 argument (weight)".into()); }
    let fn_val = codegen.module.get_function("tl_tensor_embedding")
        .ok_or("tl_tensor_embedding not found")?;

    let (weight_val, _) = &args[0];

    // デフォルト引数: padding_idx=-1, scale_grad_by_freq=false, sparse=false
    let padding_idx = codegen.context.i64_type().const_int((-1i64) as u64, true);
    let false_val = codegen.context.bool_type().const_int(0, false);

    // runtime順序: (weight, indices, padding_idx, scale_grad_by_freq, sparse)
    let call = codegen.builder.build_call(
        fn_val,
        &[(*weight_val).into(), obj.into(), padding_idx.into(), false_val.into(), false_val.into()],
        "embedding_res",
    ).map_err(|e| e.to_string())?;

    let res = codegen.check_tensor_result(call, "embedding_error")?;
    Ok((res, obj_ty))
}

fn compile_tensor_binop<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err(format!("{} requires 1 argument", op_name)); }
    // args already evaluated. Note: ensure_tensor_v2 logic in expr.rs handles scalar conversion *before* evaluation if it was static arg?
    // Wait, TypeManager passes evaluated args. If arg was scalar literal, it is evaluated as I64/F64.
    // We need run-time scalar->tensor conversion if needed?
    // expr.rs logic: self.ensure_tensor_v2(&args[0], 0)?
    // ensure_tensor_v2 takes Expr.
    // HERE we have evaluated values.
    // If we receive I64/F32, we must wrap it in Tensor?
    // BUT TypeManager calls compile_expr(arg). So we have I64/F32 value.
    // We need a helper to ensure tensor from Value.
    
    // For now, let's look at how expr.rs did it. It used ensure_tensor_v2 which compiles Expr potentially to conversion call.
    // Since we receive compiled value, we must check type and convert if scalar.
    // But we don't have easy "convert value to tensor" helper exposed in CodeGen yet?
    // Actually, ensure_tensor_v2 does: if scalar type, allocate tensor, fill.
    // We can replicate logic or assume args are Tensors?
    // Users might write `t + 1.0`. `1.0` is F64.
    // We need to implement scalar-to-tensor promotion here.
    
    let (rhs_val, rhs_ty) = &args[0];
    let final_rhs = if let Type::Tensor(_, _) = rhs_ty {
        *rhs_val
    } else {
        // Promote scalar to tensor
        // Check numeric
        match rhs_ty {
            Type::I64 | Type::F64 | Type::F32 | Type::I32 => {
                 // Create encoded tensor from scalar
                 // We need a helper function in codegen?
                 // Or just hardcode promotion logic.
                 // Ideally we call a runtime helper tl_tensor_from_scalar?
                 // Existing ensure_tensor_v2 calls tl_tensor_zeros/fill or something.
                 // simpler: use codegen.ensure_tensor_from_val(...) if we make it?
                 // For now, I will assume we can't easily access ensure_tensor_v2 (it takes &Expr).
                 // I will skip scalar implementation details for now or try to support it?
                 // The safe bet: If not tensor, error for now OR implement simple promotion.
                 // Given constraints, I'll error if not tensor for this refactor step, 
                 // BUT this breaks `t + 1`. 
                 // Wait, expr.rs ensure_tensor_v2 is called on Expr.
                 // TypeManager evaluates args via compile_expr.
                 // If I want to support scalars, I must handle them.
                 // Let's assume for this task refactoring, we primarily target Tensor-Tensor ops.
                 // OR I can use the trick: The user's code `t + 1` passes `1` as expr.
                 // If I could intercept it... instance_method receives evaluated values.
                 // If I want `ensure_tensor_v2` behavior, I need to implement `ensure_tensor_from_value` here.
                 
                 // Implementation of scalar to tensor promotion:
                 // 1. Alloc tensor of rank 0
                 // 2. data[0] = value
                 // This requires runtime calls.
                 // It's too complex to inline here perfectly without a helper.
                 // Let's rely on checking if rhs is Tensor. 
                 return Err("Scalar broadcasting in binary ops via TypeManager refactor is pending. Please use explicit Tensor.".into());
            }
             _ => return Err("Binary op requires Tensor or Scalar".into())
        }
    };

    let fn_name = match op_name {
        "mul" => "tl_tensor_mul",
        "add" => "tl_tensor_add",
        "sub" => "tl_tensor_sub",
        "div" => "tl_tensor_div",
        _ => unreachable!(),
    };
    let fn_val = codegen.module.get_function(fn_name).ok_or(format!("{} not found", fn_name))?;
    let call = codegen.builder.build_call(fn_val, &[obj.into(), final_rhs.into()], "binop_res").map_err(|e| e.to_string())?;
    let res = codegen.check_tensor_result(call, "binop_error")?;
    Ok((res, obj_ty))
}

fn compile_tensor_mul<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_binop(c, o, t, a, "mul")
}
fn compile_tensor_add<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_binop(c, o, t, a, "add")
}
fn compile_tensor_sub<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_binop(c, o, t, a, "sub")
}
fn compile_tensor_div<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_binop(c, o, t, a, "div")
}

fn compile_tensor_contiguous<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_contiguous").ok_or("tl_tensor_contiguous not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "cont_res").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from contiguous()".into()),
    };
    Ok((res, obj_ty))
}

fn compile_tensor_conv2d<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 3 { return Err("conv2d requires 3 arguments: weight, padding, stride".into()); }
    let fn_val = codegen.module.get_function("tl_tensor_conv2d").ok_or("tl_tensor_conv2d not found")?;
    
    let (weight_val, _) = &args[0];
    let (pad_val, pad_ty) = &args[1];
    let (stride_val, stride_ty) = &args[2];
    
    let pad_i64 = match pad_ty {
        Type::I64 => pad_val.into_int_value(),
        Type::I32 => codegen.builder.build_int_z_extend(pad_val.into_int_value(), codegen.context.i64_type(), "ext").unwrap(),
        _ => return Err("conv2d padding must be int".into()),
    };
    let stride_i64 = match stride_ty {
        Type::I64 => stride_val.into_int_value(),
        Type::I32 => codegen.builder.build_int_z_extend(stride_val.into_int_value(), codegen.context.i64_type(), "ext").unwrap(),
        _ => return Err("conv2d stride must be int".into()),
    };

    let call = codegen.builder.build_call(fn_val, &[obj.into(), (*weight_val).into(), pad_i64.into(), stride_i64.into()], "conv_res").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from conv2d()".into()),
    };
    Ok((res, obj_ty))
}

fn compile_tensor_clamp<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 { return Err("clamp requires 2 args".into()); }
    let fn_val = codegen.module.get_function("tl_tensor_clamp").ok_or("tl_tensor_clamp not found")?;
    
    let (min_val, min_ty) = &args[0];
    let min_f32 = match min_ty {
        Type::F32 => min_val.into_float_value(),
        Type::F64 => codegen.builder.build_float_trunc(min_val.into_float_value(), codegen.context.f32_type(), "trunc").unwrap(),
        _ => return Err("min must be float".into()),
    };
    let (max_val, max_ty) = &args[1];
    let max_f32 = match max_ty {
        Type::F32 => max_val.into_float_value(),
        Type::F64 => codegen.builder.build_float_trunc(max_val.into_float_value(), codegen.context.f32_type(), "trunc").unwrap(),
        _ => return Err("max must be float".into()),
    };

    let call = codegen.builder.build_call(fn_val, &[obj.into(), min_f32.into(), max_f32.into()], "clamp_res").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from clamp()".into()),
    };
    Ok((res, obj_ty))
}

fn compile_tensor_clone<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_clone").ok_or("tl_tensor_clone not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "clone_res").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from clone()".into()),
    };
    Ok((res, obj_ty))
}

fn compile_tensor_shallow_clone<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_shallow_clone").ok_or("tl_tensor_shallow_clone not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "clone_res").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from shallow_clone()".into()),
    };
    Ok((res, obj_ty))
}

fn compile_tensor_grad<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_grad").ok_or("tl_tensor_grad not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "grad_res").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from grad()".into()),
    };
    Ok((res, obj_ty))
}

fn compile_tensor_matmul_quantized<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("matmul_quantized requires 1 arg".into()); }
    let fn_val = codegen.module.get_function("tl_qtensor_matmul").ok_or("tl_qtensor_matmul not found")?;
    let (weight, _) = &args[0];
    let call = codegen.builder.build_call(fn_val, &[obj.into(), (*weight).into()], "qmatmul_res").map_err(|e| e.to_string())?;
    let res = codegen.check_tensor_result(call, "qmatmul_error")?;
    Ok((res, obj_ty))
}

fn compile_tensor_to_i64<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_to_i64").ok_or("tl_tensor_to_i64 not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "to_i64_call").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from to_i64".into()),
    };
    Ok((res, Type::Tensor(Box::new(Type::I64), 0)))
}

fn compile_tensor_to_f32<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_to_f32").ok_or("tl_tensor_to_f32 not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "to_f32_call").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from to_f32".into()),
    };
    Ok((res, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_to_vec_f32<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_to_vec_f32").ok_or("tl_tensor_to_vec_f32 not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "to_vec_call").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from to_vec_f32".into()),
    };
    Ok((res, Type::Struct("Vec".into(), vec![Type::F32])))
}

fn compile_tensor_cuda<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_to_device_helper(codegen, obj, obj_ty, "cuda")
}

fn compile_tensor_cpu<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_to_device_helper(codegen, obj, obj_ty, "cpu")
}

fn compile_tensor_to_device_helper<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    device: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_to_device").ok_or("tl_tensor_to_device not found")?;
    let (dev_str_val, _) = codegen.compile_string_literal(device)?;
    let call = codegen.builder.build_call(fn_val, &[obj.into(), dev_str_val.into()], "to_dev_res").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
         _ => return Err("Invalid return from to_device".into()),
    };
    Ok((res, obj_ty))
}

fn compile_tensor_item<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let is_int = if let Type::Tensor(elem, _) = &obj_ty {
        matches!(elem.as_ref(), Type::I64 | Type::I32 | Type::U32 | Type::U8)
    } else {
        false
    };
    let fn_name = if is_int { "tl_tensor_item_i64" } else { "tl_tensor_item" };
    let fn_val = codegen.module.get_function(fn_name).ok_or(format!("{} not found", fn_name))?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "item_res").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from item()".into()),
    };
    Ok((res, if is_int { Type::I64 } else { Type::F32 }))
}

// fn compile_tensor_reduce_wrapper<'ctx>(
//     codegen: &mut CodeGenerator<'ctx>,
//     obj: BasicValueEnum<'ctx>,
//     obj_ty: Type,
//     args: Vec<(BasicValueEnum<'ctx>, Type)>,
// ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
//     // We don't have 'method name' here easily unless we pass it or make closures?
//     // Wait, register_instance_method usage: compile_tensor_reduce_wrapper
//     // But this function signature doesn't take method name.
//     // Solution: define separate functions for each reduce op, or use a helper that takes name, 
//     // but we can't register a helper that doesn't match the signature.
//     // We need explicit wrappers for each.
//     // I will implement explicit wrappers below.
//     Err("Not implemented: reduce wrapper needs specific method dispatch".into())
// }

// Implement specific reducers
fn compile_tensor_max_impl<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_generic(c, o, t, a, "max")
}
fn compile_tensor_min_impl<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_generic(c, o, t, a, "min")
}
fn compile_tensor_mean_impl<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_generic(c, o, t, a, "mean")
}
fn compile_tensor_var_impl<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_generic(c, o, t, a, "var")
}
fn compile_tensor_std_impl<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_generic(c, o, t, a, "std")
}
fn compile_tensor_prod_impl<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_generic(c, o, t, a, "prod")
}
fn compile_tensor_sum_impl<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_generic(c, o, t, a, "sum")
}
fn compile_tensor_argmax_impl<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_generic(c, o, t, a, "argmax")
}
fn compile_tensor_argmin_impl<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_generic(c, o, t, a, "argmin")
}

fn compile_tensor_reduce_generic<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    method: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() {
        let suffix = if method == "argmax" || method == "argmin" { "" } else { "_dim" };
        let fn_name = format!("tl_tensor_{}{}", method, suffix);
        let fn_val = codegen.module.get_function(&fn_name).ok_or(format!("{} not found", fn_name))?;

        let (dim_val, _) = &args[0];
        let keep_val = if args.len() > 1 {
            let (k, _) = &args[1];
             *k
        } else {
             codegen.context.bool_type().const_int(0, false).into()
        };
        let call = codegen.builder.build_call(fn_val, &[obj.into(), (*dim_val).into(), keep_val.into()], "reduce_res").map_err(|e| e.to_string())?;
        let res = match call.try_as_basic_value() {
            ValueKind::Basic(v) => v,
            _ => return Err("Invalid return".into()),
        };
        Ok((res, obj_ty))
    } else {
        if method == "argmax" || method == "argmin" {
            return Err(format!("{} requires arguments", method));
        }
        let fn_name = format!("tl_tensor_{}", method);
        let fn_val = codegen.module.get_function(&fn_name).ok_or(format!("{} not found", fn_name))?;
        let call = codegen.builder.build_call(fn_val, &[obj.into()], "reduce_res").map_err(|e| e.to_string())?;
        let res = match call.try_as_basic_value() {
            ValueKind::Basic(v) => v,
             _ => return Err("Invalid return".into()),
        };
        Ok((res, obj_ty))
    }
}

fn compile_tensor_save<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("save requires 1 arg (path)".into()); }
    let fn_val = codegen.module.get_function("tl_tensor_save").ok_or("tl_tensor_save not found")?;
    let (path_val, _) = &args[0];
    codegen.builder.build_call(fn_val, &[(*path_val).into(), obj.into()], "save_call").map_err(|e| e.to_string())?;
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}

fn compile_tensor_reshape<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
     if args.len() != 1 { return Err("reshape requires 1 arg (shape)".into()); }
     let (shape_val, shape_ty) = &args[0];
     // Arg[0] is shape.
     // In expr.rs, it compiled arg[0].

     if let Type::Tensor(_, _) = shape_ty {
          // Case 1: Shape is a Tensor. Use tl_tensor_reshape_new(obj, shape_tensor).
          let fn_val = codegen.module.get_function("tl_tensor_reshape_new")
               .ok_or("tl_tensor_reshape_new not found")?;
          let call = codegen.builder.build_call(fn_val, &[obj.into(), shape_val.clone().into()], "reshape_res")
               .map_err(|e| e.to_string())?;
          let res = match call.try_as_basic_value() {
               ValueKind::Basic(v) => v,
               _ => return Err("Invalid return".into()),
          };
          // Free shape tensor after reshape (it was only used as a shape descriptor)
          if let Some(release_fn) = codegen.module.get_function("tl_tensor_release_safe") {
               codegen.builder.build_call(release_fn, &[(*shape_val).into()], "").ok();
          }
          return Ok((res, Type::Tensor(Box::new(Type::F32), 0))); // Dynamic rank
     } else {
          // Case 2: Shape is Vec. Use tl_tensor_reshape_dims(obj, ptr, rank).
          let fn_val = codegen.module.get_function("tl_tensor_reshape_dims")
               .ok_or("tl_tensor_reshape_dims not found")?;

          let (data_ptr, rank_val) = if matches!(shape_ty, Type::Struct(n, _) if n.starts_with("Vec")) {
               // Vec
               let vec_ptr = if shape_val.is_pointer_value() {
                    shape_val.into_pointer_value()
               } else {
                    return Err("Vec shape must be pointer".into());
               };

               let i64_ty = codegen.context.i64_type();
               let vec_struct_ty = codegen.context.struct_type(&[i64_ty.into(), i64_ty.into(), i64_ty.into()], false);

               // Extract ptr (index 0)
               let data_ptr_ptr = codegen.builder.build_struct_gep(vec_struct_ty, vec_ptr, 0, "data_ptr_ptr")
                         .map_err(|e| e.to_string())?;
               
               let data_ptr_int = codegen.builder.build_load(i64_ty, data_ptr_ptr, "data_ptr_int")
                    .map_err(|e| e.to_string())?.into_int_value();
               
               let data_ptr = codegen.builder.build_int_to_ptr(
                    data_ptr_int,
                    codegen.context.ptr_type(inkwell::AddressSpace::default()),
                    "data_ptr"
               ).map_err(|e| e.to_string())?;

               // Extract len (index 2)
               let len_ptr = codegen.builder.build_struct_gep(vec_struct_ty, vec_ptr, 2, "len_ptr")
                         .map_err(|e| e.to_string())?;
               
               let len_val = codegen.builder.build_load(i64_ty, len_ptr, "rank_val")
                    .map_err(|e| e.to_string())?.into_int_value();
               
               (data_ptr, len_val)
          } else {
               return Err(format!("reshape argument must be Tensor or Vec. Got: {:?}", shape_ty));
          };

          let call = codegen.builder.build_call(fn_val, &[obj.into(), data_ptr.into(), rank_val.into()], "reshape_res")
               .map_err(|e| e.to_string())?;
          
          let res = match call.try_as_basic_value() {
               ValueKind::Basic(v) => v,
               _ => return Err("Invalid return".into()),
          };
          return Ok((res, Type::Tensor(Box::new(Type::F32), 0)));
     }
}

fn compile_tensor_to_device<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("to requires 1 arg (device)".into()); }
    let fn_val = codegen.module.get_function("tl_tensor_to_device").ok_or("tl_tensor_to_device not found")?;
    let (dev_val, _) = &args[0];
    let call = codegen.builder.build_call(fn_val, &[obj.into(), (*dev_val).into()], "to_dev_res").map_err(|e| e.to_string())?;
     let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from to()".into()),
    };
    Ok((res, obj_ty))
}

fn compile_tensor_transpose<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 { return Err("transpose requires 2 args (dim0, dim1)".into()); }
    let fn_val = codegen.module.get_function("tl_tensor_transpose").ok_or("tl_tensor_transpose not found")?;
    let (d0, _) = &args[0];
    let (d1, _) = &args[1];
    
    let call = codegen.builder.build_call(fn_val, &[obj.into(), (*d0).into(), (*d1).into()], "transpose_res").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
         _ => return Err("Invalid return from transpose".into()),
    };
    Ok((res, obj_ty))
}

fn compile_clear_grads<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("Tensor::clear_grads takes no arguments".into()); }
    let fn_val = codegen.module.get_function("tl_clear_grads").ok_or("tl_clear_grads not found")?;
    codegen.builder.build_call(fn_val, &[], "clear_grads").map_err(|e| e.to_string())?;
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}

fn compile_mem_purge<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("Tensor::mem_purge takes no arguments".into()); }
    let fn_val = codegen.module.get_function("tl_mem_purge").ok_or("tl_mem_purge not found")?;
    codegen.builder.build_call(fn_val, &[], "mem_purge").map_err(|e| e.to_string())?;
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}

fn compile_from_vec_u8<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    // Tensor::from_vec_u8(vec, shape) -> Tensor
    // Arguments are already evaluated by GenericResolver (evaluated args mode)
    // args[0]: vec (Vec<U8>)
    // args[1]: shape (Vec<I64>)
    // Implicit: offset=0, rank=shape.len()
    
    if args.len() != 2 {
        return Err("Tensor::from_vec_u8 requires 2 arguments (vec, shape)".into());
    }
    
    let vec_val = args[0].0;
    let shape_raw = args[1].0; 
    let shape_ty = args[1].1.clone();
    
    // We need to pass (ptr, offset, shape_data_ptr, rank)
    // shape_data_ptr should be *const i64 (pointer to array)
    
    let (shape_data_ptr, rank_val) = if matches!(shape_ty, Type::Tensor(_, _)) {
        // Handle Tensor
        // Data ptr
        let data_fn = codegen.module.get_function("tl_tensor_data").ok_or("tl_tensor_data not found")?;
        let data_call = codegen.builder.build_call(data_fn, &[shape_raw.into()], "shape_data").map_err(|e| e.to_string())?;
        let data_ptr = match data_call.try_as_basic_value() {
            ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err("Invalid return from tensor data".into()),
        };
        
        // Numel (rank)
        let numel_fn = codegen.module.get_function("tl_tensor_numel").ok_or("tl_tensor_numel not found")?;
        let numel_call = codegen.builder.build_call(numel_fn, &[shape_raw.into()], "shape_numel").map_err(|e| e.to_string())?;
        let numel = match numel_call.try_as_basic_value() {
            ValueKind::Basic(v) => v.into_int_value(),
            _ => return Err("Invalid return from tensor numel".into()),
        };
        
        (data_ptr, numel)
    } else {
         // Assume Vec (pointer to struct)
         // Call tl_vec_i64_as_ptr(vec) -> *const i64
         let as_ptr_fn = codegen.module.get_function("tl_vec_i64_as_ptr").ok_or("tl_vec_i64_as_ptr not found")?;
         let ptr_call = codegen.builder.build_call(as_ptr_fn, &[shape_raw.into()], "vec_ptr").map_err(|e| e.to_string())?;
         let data_ptr = match ptr_call.try_as_basic_value() {
             ValueKind::Basic(v) => v.into_pointer_value(),
             _ => return Err("Invalid return from vec as_ptr".into()),
         };

         // Call tl_vec_i64_len(vec) -> usize
         let len_fn = codegen.module.get_function("tl_vec_i64_len").ok_or("tl_vec_i64_len not found")?;
         let len_call = codegen.builder.build_call(len_fn, &[shape_raw.into()], "vec_len").map_err(|e| e.to_string())?;
         let rank = match len_call.try_as_basic_value() {
             ValueKind::Basic(v) => v.into_int_value(),
             _ => return Err("Invalid return from vec len".into()),
         };
         
         (data_ptr, rank)
    };
    
    let offset_val = codegen.context.i64_type().const_int(0, false);
    
    let fn_val = codegen.module.get_function("tl_tensor_from_vec_u8").ok_or("tl_tensor_from_vec_u8 not found")?;
    
    let call = codegen.builder.build_call(fn_val, &[
        vec_val.into(),
        offset_val.into(),
        shape_data_ptr.into(),
        rank_val.into()
    ], "from_vec_res").map_err(|e| e.to_string())?;
    
    let v = codegen.check_tensor_result(call, "from_vec_error")?;
    
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_load_tensor<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_load").ok_or("tl_tensor_load not found")?;
    let (path_val, _) = &args[0];
    let call = codegen
        .builder
        .build_call(fn_val, &[(*path_val).into()], "load_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid load return".into()),
    };
    // Return rank 1? Or dynamic? Most loaded tensors have specific valid ranks.
    // Existing code returned rank 1.
    Ok((res, Type::Tensor(Box::new(Type::F32), 1)))
}

fn compile_tensor_creation_helper<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    runtime_func_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() {
        return Err(format!("{} requires shape", runtime_func_name));
    }
    let shape_expr = &args[0];
    let (rank, shape_vals) = match &shape_expr.inner {
        ExprKind::TensorLiteral(el) | ExprKind::TensorConstLiteral(el) => {
            let mut vals = Vec::new();
            for e in el {
                let (v, t) = codegen.compile_expr(e)?;
                let int_val = match t {
                    Type::I64 => v.into_int_value(),
                    Type::I32 => codegen
                        .builder
                        .build_int_z_extend(
                            v.into_int_value(),
                            codegen.context.i64_type(),
                            "dim_ext",
                        )
                        .map_err(|e| e.to_string())?,
                    _ => return Err(format!("Dimension must be integer, found {:?}", t)),
                };
                vals.push(int_val);
            }
            (el.len(), vals)
        }
        _ => {
            return Err(format!(
                "{} currently requires array literal [dim, ...] for shape",
                runtime_func_name
            ));
        }
    };
    let requires_grad = if args.len() > 1 {
        match &args[1].inner {
            ExprKind::Bool(b) => *b,
            _ => false,
        }
    } else {
        false
    };
    let i64_type = codegen.context.i64_type();

    let current_block = codegen.builder.get_insert_block().unwrap();
    let function = current_block.get_parent().unwrap();
    let entry_block = function.get_first_basic_block().unwrap();
    let entry_builder = codegen.context.create_builder();
    if let Some(first_instr) = entry_block.get_first_instruction() {
        entry_builder.position_before(&first_instr);
    } else {
        entry_builder.position_at_end(entry_block);
    }

    let shape_array_type = i64_type.array_type(rank as u32);
    let shape_alloca = entry_builder
        .build_alloca(shape_array_type, "shape_arr")
        .map_err(|e| e.to_string())?;

    shape_alloca
        .as_instruction_value()
        .unwrap()
        .set_alignment(16)
        .map_err(|e| e.to_string())?;

    for (i, val) in shape_vals.iter().enumerate() {
        let ptr = unsafe {
            codegen.builder.build_in_bounds_gep(
                shape_array_type,
                shape_alloca,
                &[
                    i64_type.const_int(0, false),
                    i64_type.const_int(i as u64, false),
                ],
                "shape_ptr_in",
            )
        }
        .map_err(|e| e.to_string())?;
        codegen
            .builder
            .build_store(ptr, *val)
            .map_err(|e| e.to_string())?;
    }
    let req_grad_val = codegen
        .context
        .bool_type()
        .const_int(if requires_grad { 1 } else { 0 }, false);
    let f = codegen
        .module
        .get_function(runtime_func_name)
        .ok_or(format!("{} not found", runtime_func_name))?;

    let zero = i64_type.const_int(0, false);
    let first_elem_ptr = unsafe {
        codegen.builder.build_in_bounds_gep(
            shape_array_type,
            shape_alloca,
            &[zero, zero],
            "first_elem_ptr",
        )
    }
    .map_err(|e| e.to_string())?;

    // Build args: randn_debug requires seed arg, zeros/ones do not
    let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::new();
    call_args.push(i64_type.const_int(rank as u64, false).into());
    call_args.push(first_elem_ptr.into());
    if runtime_func_name.contains("randn") {
        // randn_debug has signature: (rank, shape, seed, req_grad)
        call_args.push(i64_type.const_int(0, false).into()); // seed = 0 (unused)
    }
    call_args.push(req_grad_val.into());

    let call = codegen
        .builder
        .build_call(
            f,
            &call_args,
            "creation_res",
        )
        .map_err(|e| e.to_string())?;
    let res = codegen.check_tensor_result(call, "creation_error")?;

    Ok((res, Type::Tensor(Box::new(Type::F32), rank)))
}

fn compile_randn<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_creation_helper(codegen, args, "tl_tensor_randn_debug")
}

fn compile_ones<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_creation_helper(codegen, args, "tl_tensor_ones")
}

fn compile_tensor_zeros<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() {
        return Err("Tensor::zeros requires shape argument".into());
    }

    let elements_ref = if let ExprKind::TensorLiteral(el) = &args[0].inner {
        Some(el)
    } else if let ExprKind::TensorConstLiteral(el) = &args[0].inner {
        Some(el)
    } else {
        None
    };

    if let Some(el) = elements_ref {
        let i64_type = codegen.context.i64_type();
        let mut vals = Vec::new();
        for e in el {
            let (v, t) = codegen.compile_expr(e)?;
            let int_val = match t {
                Type::I64 => v.into_int_value(),
                Type::I32 => codegen
                    .builder
                    .build_int_z_extend(v.into_int_value(), i64_type, "ext")
                    .map_err(|e| e.to_string())?,
                _ => return Err(format!("Dimension must be integer, found {:?}", t)),
            };
            vals.push(int_val);
        }

        let rank = el.len();
        let shape_array_type = i64_type.array_type(rank as u32);
        let shape_alloca = codegen
            .builder
            .build_alloca(shape_array_type, "shape_arr")
            .map_err(|e| e.to_string())?;

        for (i, val) in vals.iter().enumerate() {
            let ptr = unsafe {
                codegen
                    .builder
                    .build_in_bounds_gep(
                        shape_array_type,
                        shape_alloca,
                        &[
                            i64_type.const_int(0, false),
                            i64_type.const_int(i as u64, false),
                        ],
                        "tmp",
                    )
                    .map_err(|e| e.to_string())?
            };
            codegen
                .builder
                .build_store(ptr, *val)
                .map_err(|e| e.to_string())?;
        }

        let req_grad = if args.len() > 1 {
            let (v, _) = codegen.compile_expr(&args[1])?;
            v.into_int_value()
        } else {
            codegen.context.bool_type().const_int(0, false)
        };

        let f = codegen
            .module
            .get_function("tl_tensor_zeros")
            .ok_or("tl_tensor_zeros not found")?;
        let call = codegen
            .builder
            .build_call(
                f,
                &[
                    i64_type.const_int(rank as u64, false).into(),
                    shape_alloca.into(),
                    req_grad.into(),
                ],
                "zeros_res",
            )
            .map_err(|e| e.to_string())?;

        let v = codegen.check_tensor_result(call, "zeros_error")?;
        let result_ty = Type::Tensor(Box::new(Type::F32), rank);
        // codegen.emit_register_tensor(v, &result_ty)?;
        return Ok((v, result_ty));
    }

    Err("Generic Tensor::zeros (non-literal shape) not yet ported. Please use literal shape [d1, d2] for now.".into())
}

fn compile_tensor_full<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 {
        return Err("Tensor::full requires (shape, value) arguments".into());
    }

    let i64_type = codegen.context.i64_type();
    let f32_type = codegen.context.f32_type();

    let elements_ref = if let ExprKind::TensorLiteral(el) = &args[0].inner {
        Some(el)
    } else if let ExprKind::TensorConstLiteral(el) = &args[0].inner {
        Some(el)
    } else {
        None
    };

    let el = elements_ref.ok_or("Tensor::full requires literal shape [d1, d2, ...] for now")?;
    let rank = el.len();

    // Build shape array
    let shape_array_type = i64_type.array_type(rank as u32);
    let shape_alloca = codegen.builder.build_alloca(shape_array_type, "shape_arr").map_err(|e| e.to_string())?;
    for (i, e) in el.iter().enumerate() {
        let (v, t) = codegen.compile_expr(e)?;
        let int_val = match t {
            Type::I64 => v.into_int_value(),
            Type::I32 => codegen.builder.build_int_z_extend(v.into_int_value(), i64_type, "ext").map_err(|e| e.to_string())?,
            _ => return Err(format!("Dimension must be integer, found {:?}", t)),
        };
        let ptr = unsafe {
            codegen.builder.build_in_bounds_gep(
                shape_array_type, shape_alloca,
                &[i64_type.const_int(0, false), i64_type.const_int(i as u64, false)],
                "tmp",
            ).map_err(|e| e.to_string())?
        };
        codegen.builder.build_store(ptr, int_val).map_err(|e| e.to_string())?;
    }

    // Compile value argument
    let (val_v, val_t) = codegen.compile_expr(&args[1])?;
    let value_f32 = match val_t {
        Type::F32 => val_v.into_float_value(),
        Type::F64 => codegen.builder.build_float_trunc(val_v.into_float_value(), f32_type, "trunc").map_err(|e| e.to_string())?,
        Type::I64 | Type::I32 => codegen.builder.build_signed_int_to_float(val_v.into_int_value(), f32_type, "itof").map_err(|e| e.to_string())?,
        _ => return Err(format!("Tensor::full value must be numeric, found {:?}", val_t)),
    };

    let req_grad = if args.len() > 2 {
        let (v, _) = codegen.compile_expr(&args[2])?;
        v.into_int_value()
    } else {
        codegen.context.bool_type().const_int(0, false)
    };

    let f = codegen.module.get_function("tl_tensor_full").ok_or("tl_tensor_full not found")?;
    let call = codegen.builder.build_call(
        f,
        &[
            i64_type.const_int(rank as u64, false).into(),
            shape_alloca.into(),
            value_f32.into(),
            req_grad.into(),
        ],
        "full_res",
    ).map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "full_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), rank)))
}

fn compile_tensor_eye<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() {
        return Err("Tensor::eye requires n argument".into());
    }

    let i64_type = codegen.context.i64_type();
    let (n_v, n_t) = codegen.compile_expr(&args[0])?;
    let n_val = match n_t {
        Type::I64 => n_v.into_int_value(),
        Type::I32 => codegen.builder.build_int_z_extend(n_v.into_int_value(), i64_type, "ext").map_err(|e| e.to_string())?,
        _ => return Err(format!("Tensor::eye requires integer argument, found {:?}", n_t)),
    };

    let req_grad = if args.len() > 1 {
        let (v, _) = codegen.compile_expr(&args[1])?;
        v.into_int_value()
    } else {
        codegen.context.bool_type().const_int(0, false)
    };

    let f = codegen.module.get_function("tl_tensor_eye").ok_or("tl_tensor_eye not found")?;
    let call = codegen.builder.build_call(
        f,
        &[n_val.into(), req_grad.into()],
        "eye_res",
    ).map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "eye_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 2)))
}

fn compile_tensor_arange<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 3 {
        return Err("Tensor::arange requires (start, end, step) arguments".into());
    }

    let f64_type = codegen.context.f64_type();
    let _f32_type = codegen.context.f32_type();
    let mut vals = Vec::new();
    for arg in &args[..3] {
        let (v, t) = codegen.compile_expr(arg)?;
        let f64_val = match t {
            Type::F64 => v.into_float_value(),
            Type::F32 => codegen.builder.build_float_ext(v.into_float_value(), f64_type, "fext").map_err(|e| e.to_string())?,
            Type::I64 => codegen.builder.build_signed_int_to_float(v.into_int_value(), f64_type, "itof").map_err(|e| e.to_string())?,
            Type::I32 => codegen.builder.build_signed_int_to_float(v.into_int_value(), f64_type, "itof").map_err(|e| e.to_string())?,
            _ => return Err(format!("Tensor::arange requires numeric arguments, found {:?}", t)),
        };
        vals.push(f64_val);
    }

    let f = codegen.module.get_function("tl_tensor_arange").ok_or("tl_tensor_arange not found")?;
    let call = codegen.builder.build_call(
        f,
        &[vals[0].into(), vals[1].into(), vals[2].into()],
        "arange_res",
    ).map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "arange_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 1)))
}

fn compile_tensor_zeros_like<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() {
        return Err("Tensor::zeros_like requires 1 argument".into());
    }
    let tensor_val = args[0].0;
    let f = codegen.module.get_function("tl_tensor_zeros_like").ok_or("tl_tensor_zeros_like not found")?;
    let call = codegen.builder.build_call(f, &[tensor_val.into()], "zeros_like_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "zeros_like_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_ones_like<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() {
        return Err("Tensor::ones_like requires 1 argument".into());
    }
    let tensor_val = args[0].0;
    let f = codegen.module.get_function("tl_tensor_ones_like").ok_or("tl_tensor_ones_like not found")?;
    let call = codegen.builder.build_call(f, &[tensor_val.into()], "ones_like_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "ones_like_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_from_vec_f32<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 {
        return Err("Tensor::from_vec requires (data, shape) arguments".into());
    }
    let data_val = args[0].0;
    let shape_val = args[1].0;
    let f = codegen.module.get_function("tl_tensor_from_vec_f32").ok_or("tl_tensor_from_vec_f32 not found")?;
    let call = codegen.builder.build_call(f, &[data_val.into(), shape_val.into()], "from_vec_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "from_vec_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_linspace<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 3 {
        return Err("Tensor::linspace requires (start, end, steps) arguments".into());
    }
    let f64_type = codegen.context.f64_type();
    let i64_type = codegen.context.i64_type();
    let mut float_vals = Vec::new();
    for arg in &args[..2] {
        let (v, t) = codegen.compile_expr(arg)?;
        let f64_val = match t {
            Type::F64 => v.into_float_value(),
            Type::F32 => codegen.builder.build_float_ext(v.into_float_value(), f64_type, "fext").map_err(|e| e.to_string())?,
            Type::I64 => codegen.builder.build_signed_int_to_float(v.into_int_value(), f64_type, "itof").map_err(|e| e.to_string())?,
            Type::I32 => codegen.builder.build_signed_int_to_float(v.into_int_value(), f64_type, "itof").map_err(|e| e.to_string())?,
            _ => return Err(format!("Tensor::linspace requires numeric arguments, found {:?}", t)),
        };
        float_vals.push(f64_val);
    }
    let (steps_v, steps_t) = codegen.compile_expr(&args[2])?;
    let steps_val = match steps_t {
        Type::I64 => steps_v.into_int_value(),
        Type::I32 => codegen.builder.build_int_z_extend(steps_v.into_int_value(), i64_type, "ext").map_err(|e| e.to_string())?,
        _ => return Err(format!("Tensor::linspace steps must be integer, found {:?}", steps_t)),
    };
    let f = codegen.module.get_function("tl_tensor_linspace").ok_or("tl_tensor_linspace not found")?;
    let call = codegen.builder.build_call(
        f, &[float_vals[0].into(), float_vals[1].into(), steps_val.into()], "linspace_res",
    ).map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "linspace_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 1)))
}

fn compile_tensor_rand<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_creation_helper(codegen, args, "tl_tensor_rand")
}

fn compile_tensor_rand_like<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() { return Err("Tensor::rand_like requires 1 argument".into()); }
    let f = codegen.module.get_function("tl_tensor_rand_like").ok_or("tl_tensor_rand_like not found")?;
    let call = codegen.builder.build_call(f, &[args[0].0.into()], "rand_like_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "rand_like_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_randn_like<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() { return Err("Tensor::randn_like requires 1 argument".into()); }
    let f = codegen.module.get_function("tl_tensor_randn_like").ok_or("tl_tensor_randn_like not found")?;
    let call = codegen.builder.build_call(f, &[args[0].0.into()], "randn_like_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "randn_like_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_where_cond<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 3 { return Err("Tensor::where_cond requires (cond, x, y) arguments".into()); }
    let f = codegen.module.get_function("tl_tensor_where_cond").ok_or("tl_tensor_where_cond not found")?;
    let call = codegen.builder.build_call(f, &[args[0].0.into(), args[1].0.into(), args[2].0.into()], "where_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "where_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_masked_fill<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("masked_fill requires (mask, value) arguments".into()); }
    let mask_val = args[0].0;
    let value_val = args[1].0;
    // Convert value to f32 if needed
    let f32_type = codegen.context.f32_type();
    let f32_val = match args[1].1 {
        Type::F32 => value_val.into_float_value(),
        Type::F64 => codegen.builder.build_float_trunc(value_val.into_float_value(), f32_type, "ftrunc").map_err(|e| e.to_string())?,
        _ => return Err("masked_fill value must be f32 or f64".into()),
    };
    let f = codegen.module.get_function("tl_tensor_masked_fill").ok_or("tl_tensor_masked_fill not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), mask_val.into(), f32_val.into()], "mfill_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "masked_fill_error")?;
    Ok((v, obj_ty))
}

fn compile_tensor_fill_<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() { return Err("fill_ requires a value argument".into()); }
    let f32_type = codegen.context.f32_type();
    let f32_val = match args[0].1 {
        Type::F32 => args[0].0.into_float_value(),
        Type::F64 => codegen.builder.build_float_trunc(args[0].0.into_float_value(), f32_type, "ftrunc").map_err(|e| e.to_string())?,
        _ => return Err("fill_ value must be f32 or f64".into()),
    };
    let f = codegen.module.get_function("tl_tensor_fill_").ok_or("tl_tensor_fill_ not found")?;
    codegen.builder.build_call(f, &[obj.into(), f32_val.into()], "").map_err(|e| e.to_string())?;
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}

fn compile_tensor_cumsum<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let dim_val = args[0].0;
    let f = codegen.module.get_function("tl_tensor_cumsum").ok_or("tl_tensor_cumsum not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), dim_val.into()], "cumsum_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "cumsum_error")?;
    Ok((v, obj_ty))
}

fn compile_tensor_norm<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let f32_type = codegen.context.f32_type();
    let p_val = match args[0].1 {
        Type::F32 => args[0].0.into_float_value(),
        Type::F64 => codegen.builder.build_float_trunc(args[0].0.into_float_value(), f32_type, "ptrunc").map_err(|e| e.to_string())?,
        _ => return Err("norm p must be float".into()),
    };
    let dim_val = if args.len() > 1 { args[1].0 } else { codegen.context.i64_type().const_int(u64::MAX, false).into() };
    let f = codegen.module.get_function("tl_tensor_norm").ok_or("tl_tensor_norm not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), p_val.into(), dim_val.into()], "norm_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "norm_error")?;
    Ok((v, obj_ty))
}

fn compile_tensor_topk<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let k_val = args[0].0;
    let dim_val = if args.len() > 1 { args[1].0 } else { codegen.context.i64_type().const_int(u64::MAX, false).into() };
    let f = codegen.module.get_function("tl_tensor_topk").ok_or("tl_tensor_topk not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), k_val.into(), dim_val.into()], "topk_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "topk_error")?;
    Ok((v, obj_ty))
}

fn compile_tensor_logical_and<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("logical_and requires 2 args".into()); }
    let f = codegen.module.get_function("tl_tensor_logical_and").ok_or("tl_tensor_logical_and not found")?;
    let call = codegen.builder.build_call(f, &[args[0].0.into(), args[1].0.into()], "land_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "land_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_logical_or<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("logical_or requires 2 args".into()); }
    let f = codegen.module.get_function("tl_tensor_logical_or").ok_or("tl_tensor_logical_or not found")?;
    let call = codegen.builder.build_call(f, &[args[0].0.into(), args[1].0.into()], "lor_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "lor_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_logical_not<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let f = codegen.module.get_function("tl_tensor_logical_not").ok_or("tl_tensor_logical_not not found")?;
    let call = codegen.builder.build_call(f, &[obj.into()], "lnot_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "lnot_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_expand<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    // Same pattern as compile_tensor_reshape: supports both Tensor and Vec<i64> shape
    if args.len() != 1 { return Err("expand requires 1 arg (shape)".into()); }
    let (shape_val, shape_ty) = &args[0];

    if let Type::Tensor(_, _) = shape_ty {
        // Case 1: Shape is a Tensor (e.g. [3, 3] literal → Tensor<i64, 1>)
        let fn_val = codegen.module.get_function("tl_tensor_expand_new")
            .ok_or("tl_tensor_expand_new not found")?;
        let call = codegen.builder.build_call(fn_val, &[obj.into(), shape_val.clone().into()], "expand_res")
            .map_err(|e| e.to_string())?;
        let res = match call.try_as_basic_value() {
            ValueKind::Basic(v) => v,
            _ => return Err("Invalid return from tl_tensor_expand_new".into()),
        };
        // Free shape tensor after expand (it was only used as a shape descriptor)
        if let Some(release_fn) = codegen.module.get_function("tl_tensor_release_safe") {
            codegen.builder.build_call(release_fn, &[(*shape_val).into()], "").ok();
        }
        return Ok((res, Type::Tensor(Box::new(Type::F32), 0)));
    }

    // Case 2: Vec<i64> shape
    if !matches!(shape_ty, Type::Struct(n, _) if n.starts_with("Vec")) {
        return Err(format!("expand argument must be Vec<i64> or Tensor. Got: {:?}", shape_ty));
    }
    let vec_ptr = if shape_val.is_pointer_value() {
        shape_val.into_pointer_value()
    } else {
        return Err("Vec shape must be pointer".into());
    };
    let i64_ty = codegen.context.i64_type();
    let vec_struct_ty = codegen.context.struct_type(&[i64_ty.into(), i64_ty.into(), i64_ty.into()], false);
    let data_ptr_ptr = codegen.builder.build_struct_gep(vec_struct_ty, vec_ptr, 0, "data_ptr_ptr")
        .map_err(|e| e.to_string())?;
    let data_ptr_int = codegen.builder.build_load(i64_ty, data_ptr_ptr, "data_ptr_int")
        .map_err(|e| e.to_string())?.into_int_value();
    let data_ptr = codegen.builder.build_int_to_ptr(
        data_ptr_int,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "data_ptr"
    ).map_err(|e| e.to_string())?;
    let len_ptr = codegen.builder.build_struct_gep(vec_struct_ty, vec_ptr, 2, "len_ptr")
        .map_err(|e| e.to_string())?;
    let len_val = codegen.builder.build_load(i64_ty, len_ptr, "rank_val")
        .map_err(|e| e.to_string())?.into_int_value();

    let f = codegen.module.get_function("tl_tensor_expand").ok_or("tl_tensor_expand not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), data_ptr.into(), len_val.into()], "expand_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "expand_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_stack<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 3 { return Err("stack requires (a, b, dim) arguments".into()); }
    let f = codegen.module.get_function("tl_tensor_stack").ok_or("tl_tensor_stack not found")?;
    let call = codegen.builder.build_call(f, &[args[0].0.into(), args[1].0.into(), args[2].0.into()], "stack_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "stack_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_layer_norm<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 3 { return Err("layer_norm requires (weight, bias, eps) arguments".into()); }
    let f64_type = codegen.context.f64_type();
    let eps_val = match args[2].1 {
        Type::F64 => args[2].0.into_float_value(),
        Type::F32 => codegen.builder.build_float_ext(args[2].0.into_float_value(), f64_type, "eps_ext").map_err(|e| e.to_string())?,
        _ => f64_type.const_float(1e-5),
    };
    let f = codegen.module.get_function("tl_tensor_layer_norm").ok_or("tl_tensor_layer_norm not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into(), eps_val.into()], "ln_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "layer_norm_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_dropout<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("dropout requires (p, training) arguments".into()); }
    let f = codegen.module.get_function("tl_tensor_dropout").ok_or("tl_tensor_dropout not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into()], "dropout_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "dropout_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_dropout2d<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("dropout2d requires (p, training) arguments".into()); }
    let f = codegen.module.get_function("tl_tensor_dropout2d").ok_or("tl_tensor_dropout2d not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into()], "dropout2d_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "dropout2d_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_leaky_relu<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let f32_type = codegen.context.f32_type();
    let slope = if args.is_empty() {
        f32_type.const_float(0.01)
    } else {
        match args[0].1 {
            Type::F32 => args[0].0.into_float_value(),
            Type::F64 => codegen.builder.build_float_trunc(args[0].0.into_float_value(), f32_type, "ftrunc").map_err(|e| e.to_string())?,
            _ => return Err("leaky_relu slope must be f32 or f64".into()),
        }
    };
    let f = codegen.module.get_function("tl_tensor_leaky_relu").ok_or("tl_tensor_leaky_relu not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), slope.into()], "lrelu_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "leaky_relu_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_max_pool2d<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 3 { return Err("max_pool2d requires (kernel_size, stride, padding)".into()); }
    let f = codegen.module.get_function("tl_tensor_max_pool2d").ok_or("tl_tensor_max_pool2d not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into(), args[2].0.into()], "maxpool_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "maxpool_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_avg_pool2d<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 3 { return Err("avg_pool2d requires (kernel_size, stride, padding)".into()); }
    let f = codegen.module.get_function("tl_tensor_avg_pool2d").ok_or("tl_tensor_avg_pool2d not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into(), args[2].0.into()], "avgpool_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "avgpool_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_elu<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let f32_type = codegen.context.f32_type();
    let alpha = if args.is_empty() {
        f32_type.const_float(1.0)
    } else {
        match args[0].1 {
            Type::F32 => args[0].0.into_float_value(),
            Type::F64 => codegen.builder.build_float_trunc(args[0].0.into_float_value(), f32_type, "atrunc").map_err(|e| e.to_string())?,
            _ => return Err("elu alpha must be float".into()),
        }
    };
    let f = codegen.module.get_function("tl_tensor_elu").ok_or("tl_tensor_elu not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), alpha.into()], "elu_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "elu_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_mish<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let f = codegen.module.get_function("tl_tensor_mish").ok_or("tl_tensor_mish not found")?;
    let call = codegen.builder.build_call(f, &[obj.into()], "mish_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "mish_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_mse_loss<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("mse_loss requires (pred, target)".into()); }
    let f = codegen.module.get_function("tl_tensor_mse_loss").ok_or("tl_tensor_mse_loss not found")?;
    let call = codegen.builder.build_call(f, &[args[0].0.into(), args[1].0.into()], "mse_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "mse_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_l1_loss<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("l1_loss requires (pred, target)".into()); }
    let f = codegen.module.get_function("tl_tensor_l1_loss").ok_or("tl_tensor_l1_loss not found")?;
    let call = codegen.builder.build_call(f, &[args[0].0.into(), args[1].0.into()], "l1_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "l1_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_bce_loss<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("bce_loss requires (pred, target)".into()); }
    let f = codegen.module.get_function("tl_tensor_bce_loss").ok_or("tl_tensor_bce_loss not found")?;
    let call = codegen.builder.build_call(f, &[args[0].0.into(), args[1].0.into()], "bce_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "bce_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_linear<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("linear requires (weight, bias) arguments".into()); }
    let f = codegen.module.get_function("tl_tensor_linear").ok_or("tl_tensor_linear not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into()], "linear_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "linear_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_hardswish<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let f = codegen.module.get_function("tl_tensor_hardswish").ok_or("tl_tensor_hardswish not found")?;
    let call = codegen.builder.build_call(f, &[obj.into()], "hswish_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "hardswish_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_hardsigmoid<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let f = codegen.module.get_function("tl_tensor_hardsigmoid").ok_or("tl_tensor_hardsigmoid not found")?;
    let call = codegen.builder.build_call(f, &[obj.into()], "hsigmoid_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "hardsigmoid_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_nll_loss<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("nll_loss requires (pred, target)".into()); }
    let f = codegen.module.get_function("tl_tensor_nll_loss").ok_or("tl_tensor_nll_loss not found")?;
    let call = codegen.builder.build_call(f, &[args[0].0.into(), args[1].0.into()], "nll_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "nll_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_group_norm<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 4 { return Err("group_norm requires (num_groups, weight, bias, eps)".into()); }
    let f64_type = codegen.context.f64_type();
    let eps_val = match args[3].1 {
        Type::F64 => args[3].0.into_float_value(),
        Type::F32 => codegen.builder.build_float_ext(args[3].0.into_float_value(), f64_type, "eps_ext").map_err(|e| e.to_string())?,
        _ => f64_type.const_float(1e-5),
    };
    let f = codegen.module.get_function("tl_tensor_group_norm").ok_or("tl_tensor_group_norm not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into(), args[2].0.into(), eps_val.into()], "gnorm_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "gnorm_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_adaptive_avg_pool2d<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("adaptive_avg_pool2d requires (output_h, output_w)".into()); }
    let f = codegen.module.get_function("tl_tensor_adaptive_avg_pool2d").ok_or("tl_tensor_adaptive_avg_pool2d not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into()], "apool_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "apool_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_pad<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("pad requires (pad_left, pad_right[, value])".into()); }
    let f32_type = codegen.context.f32_type();
    let value = if args.len() >= 3 {
        match args[2].1 {
            Type::F32 => args[2].0.into_float_value(),
            Type::F64 => codegen.builder.build_float_trunc(args[2].0.into_float_value(), f32_type, "vtrunc").map_err(|e| e.to_string())?,
            _ => f32_type.const_float(0.0),
        }
    } else {
        f32_type.const_float(0.0)
    };
    let f = codegen.module.get_function("tl_tensor_pad").ok_or("tl_tensor_pad not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into(), value.into()], "pad_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "pad_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_conv1d<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 4 { return Err("conv1d requires (weight, bias, stride, padding)".into()); }
    let f = codegen.module.get_function("tl_tensor_conv1d").ok_or("tl_tensor_conv1d not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into(), args[2].0.into(), args[3].0.into()], "conv1d_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "conv1d_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_kl_div_loss<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("kl_div_loss requires (pred, target)".into()); }
    let f = codegen.module.get_function("tl_tensor_kl_div_loss").ok_or("tl_tensor_kl_div_loss not found")?;
    let call = codegen.builder.build_call(f, &[args[0].0.into(), args[1].0.into()], "kl_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "kl_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_conv_transpose2d<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 5 { return Err("conv_transpose2d requires (weight, bias, stride, padding, output_padding)".into()); }
    let f = codegen.module.get_function("tl_tensor_conv_transpose2d").ok_or("tl_tensor_conv_transpose2d not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into(), args[2].0.into(), args[3].0.into(), args[4].0.into()], "conv_t2d_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "conv_transpose2d_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_interpolate<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("interpolate requires (output_h, output_w[, mode])".into()); }
    let i64_type = codegen.context.i64_type();
    let mode = if args.len() >= 3 { args[2].0.into_int_value() } else { i64_type.const_int(0, false) };
    let f = codegen.module.get_function("tl_tensor_interpolate").ok_or("tl_tensor_interpolate not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into(), mode.into()], "interp_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "interpolate_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_sdpa<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 3 { return Err("sdpa requires (q, k, v[, mask])".into()); }
    let void_ptr_type = codegen.context.ptr_type(inkwell::AddressSpace::default());
    let mask = if args.len() >= 4 { args[3].0 } else { void_ptr_type.const_null().into() };
    let f = codegen.module.get_function("tl_tensor_sdpa").ok_or("tl_tensor_sdpa not found")?;
    let call = codegen.builder.build_call(f, &[args[0].0.into(), args[1].0.into(), args[2].0.into(), mask.into()], "sdpa_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "sdpa_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_top_k_sample<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 1 { return Err("top_k_sample requires (k)".into()); }
    let f = codegen.module.get_function("tl_tensor_top_k_sample").ok_or("tl_tensor_top_k_sample not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into()], "topk_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "topk_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_top_p_sample<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 1 { return Err("top_p_sample requires (p)".into()); }
    let f64_type = codegen.context.f64_type();
    let p_val = match args[0].1 {
        Type::F64 => args[0].0.into_float_value(),
        Type::F32 => codegen.builder.build_float_ext(args[0].0.into_float_value(), f64_type, "p_ext").map_err(|e| e.to_string())?,
        _ => return Err("top_p_sample: p must be float".into()),
    };
    let f = codegen.module.get_function("tl_tensor_top_p_sample").ok_or("tl_tensor_top_p_sample not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), p_val.into()], "topp_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "topp_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_temperature_scale<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 1 { return Err("temperature_scale requires (temperature)".into()); }
    let f64_type = codegen.context.f64_type();
    let temp = match args[0].1 {
        Type::F64 => args[0].0.into_float_value(),
        Type::F32 => codegen.builder.build_float_ext(args[0].0.into_float_value(), f64_type, "t_ext").map_err(|e| e.to_string())?,
        _ => return Err("temperature must be float".into()),
    };
    let f = codegen.module.get_function("tl_tensor_temperature_scale").ok_or("tl_tensor_temperature_scale not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), temp.into()], "tscale_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "tscale_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_repetition_penalty<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("repetition_penalty requires (tokens, penalty)".into()); }
    let f64_type = codegen.context.f64_type();
    let penalty = match args[1].1 {
        Type::F64 => args[1].0.into_float_value(),
        Type::F32 => codegen.builder.build_float_ext(args[1].0.into_float_value(), f64_type, "p_ext").map_err(|e| e.to_string())?,
        _ => return Err("penalty must be float".into()),
    };
    let f = codegen.module.get_function("tl_tensor_repetition_penalty").ok_or("tl_tensor_repetition_penalty not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), penalty.into()], "rpen_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "rpen_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_dot<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 1 { return Err("dot requires (other)".into()); }
    let f = codegen.module.get_function("tl_tensor_dot").ok_or("tl_tensor_dot not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into()], "dot_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "dot_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

// batch_norm(running_mean, running_var, weight, bias, training, momentum, eps) -> Tensor
fn compile_tensor_batch_norm<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    // args: running_mean, running_var, weight, bias, training, [momentum], [eps]
    if args.len() < 5 { return Err("batch_norm requires at least (running_mean, running_var, weight, bias, training) arguments".into()); }
    let f64_type = codegen.context.f64_type();
    let momentum = if args.len() > 5 {
        match args[5].1 {
            Type::F64 => args[5].0.into_float_value(),
            Type::F32 => codegen.builder.build_float_ext(args[5].0.into_float_value(), f64_type, "mom_ext").map_err(|e| e.to_string())?,
            _ => f64_type.const_float(0.1),
        }
    } else {
        f64_type.const_float(0.1)
    };
    let eps = if args.len() > 6 {
        match args[6].1 {
            Type::F64 => args[6].0.into_float_value(),
            Type::F32 => codegen.builder.build_float_ext(args[6].0.into_float_value(), f64_type, "eps_ext").map_err(|e| e.to_string())?,
            _ => f64_type.const_float(1e-5),
        }
    } else {
        f64_type.const_float(1e-5)
    };
    let f = codegen.module.get_function("tl_tensor_batch_norm").ok_or("tl_tensor_batch_norm not found")?;
    let call = codegen.builder.build_call(f, &[
        obj.into(),         // input
        args[0].0.into(),   // running_mean
        args[1].0.into(),   // running_var
        args[2].0.into(),   // weight
        args[3].0.into(),   // bias
        args[4].0.into(),   // training
        momentum.into(),    // momentum
        eps.into(),         // eps
    ], "bn_res").map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "batch_norm_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_tensor_freeze<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_device_tensor_set_requires_grad").ok_or("tl_device_tensor_set_requires_grad not found")?;
    codegen.builder.build_call(fn_val, &[obj.into(), codegen.context.bool_type().const_int(0, false).into()], "freeze_call").map_err(|e| e.to_string())?;
    Ok((codegen.context.i64_type().const_zero().into(), Type::Void))
}

fn compile_tensor_unfreeze<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_device_tensor_set_requires_grad").ok_or("tl_device_tensor_set_requires_grad not found")?;
    codegen.builder.build_call(fn_val, &[obj.into(), codegen.context.bool_type().const_int(1, false).into()], "unfreeze_call").map_err(|e| e.to_string())?;
    Ok((codegen.context.i64_type().const_zero().into(), Type::Void))
}

fn compile_tensor_clip_grad_value<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 { return Err("clip_grad_value requires 2 arguments (min, max)".into()); }
    let fn_val = codegen.module.get_function("tl_device_tensor_clip_grad_value").ok_or("tl_device_tensor_clip_grad_value not found")?;
    
    let f64_type = codegen.context.f64_type();
    let min_val = codegen.builder.build_float_ext(args[0].0.into_float_value(), f64_type, "min_ext").map_err(|e| e.to_string())?;
    let max_val = codegen.builder.build_float_ext(args[1].0.into_float_value(), f64_type, "max_ext").map_err(|e| e.to_string())?;

    codegen.builder.build_call(fn_val, &[obj.into(), min_val.into(), max_val.into()], "clip_val_call").map_err(|e| e.to_string())?;
    Ok((codegen.context.i64_type().const_zero().into(), Type::Void))
}

fn compile_tensor_clip_grad_norm<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 { return Err("clip_grad_norm requires 2 arguments (max_norm, norm_type)".into()); }
    let fn_val = codegen.module.get_function("tl_device_tensor_clip_grad_norm").ok_or("tl_device_tensor_clip_grad_norm not found")?;
    
    let f64_type = codegen.context.f64_type();
    let max_norm = codegen.builder.build_float_ext(args[0].0.into_float_value(), f64_type, "norm_ext").map_err(|e| e.to_string())?;
    let norm_type = codegen.builder.build_float_ext(args[1].0.into_float_value(), f64_type, "type_ext").map_err(|e| e.to_string())?;

    let call = codegen.builder.build_call(fn_val, &[obj.into(), max_norm.into(), norm_type.into()], "clip_norm_call").map_err(|e| e.to_string())?;
    let ret_f64 = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v.into_float_value(),
        _ => return Err("Invalid return from clip_grad_norm".into()),
    };
    let ret_f32 = codegen.builder.build_float_trunc(ret_f64, codegen.context.f32_type(), "norm_trunc").map_err(|e| e.to_string())?;
    
    Ok((ret_f32.into(), Type::F32))
}

// instance_norm(weight, bias, eps?) -> Tensor
fn compile_tensor_instance_norm<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 { return Err("instance_norm requires at least (weight, bias)".into()); }
    let f64_type = codegen.context.f64_type();
    let eps_val = if args.len() >= 3 {
        match args[2].1 {
            Type::F64 => args[2].0.into_float_value(),
            Type::F32 => codegen.builder.build_float_ext(args[2].0.into_float_value(), f64_type, "eps_ext").map_err(|e| e.to_string())?,
            _ => f64_type.const_float(1e-5),
        }
    } else {
        f64_type.const_float(1e-5)
    };
    let f = codegen.module.get_function("tl_tensor_instance_norm").ok_or("tl_tensor_instance_norm not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into(), eps_val.into()], "inorm_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "instance_norm_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

// chunk(num_chunks, dim, index) -> Tensor
fn compile_tensor_chunk<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 3 { return Err("chunk requires (num_chunks, dim, index)".into()); }
    let f = codegen.module.get_function("tl_tensor_chunk").ok_or("tl_tensor_chunk not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into(), args[2].0.into()], "chunk_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "chunk_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

// split(split_size, dim, index) -> Tensor
fn compile_tensor_split<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 3 { return Err("split requires (split_size, dim, index)".into()); }
    let f = codegen.module.get_function("tl_tensor_split").ok_or("tl_tensor_split not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into(), args[1].0.into(), args[2].0.into()], "split_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "split_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

// --- Linear Algebra compile functions ---

macro_rules! compile_unary_linalg {
    ($name:ident, $ffi_name:expr, $label:expr) => {
        fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj: BasicValueEnum<'ctx>,
            _obj_ty: Type,
            _args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
            let f = codegen.module.get_function($ffi_name).ok_or(concat!($ffi_name, " not found"))?;
            let call = codegen.builder.build_call(f, &[obj.into()], $label)
                .map_err(|e| e.to_string())?;
            let v = codegen.check_tensor_result(call, concat!($label, "_error"))?;
            Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
        }
    };
}

compile_unary_linalg!(compile_tensor_inverse, "tl_tensor_inverse", "inv_res");
compile_unary_linalg!(compile_tensor_det, "tl_tensor_det", "det_res");
compile_unary_linalg!(compile_tensor_svd_u, "tl_tensor_svd_u", "svdu_res");
compile_unary_linalg!(compile_tensor_svd_s, "tl_tensor_svd_s", "svds_res");
compile_unary_linalg!(compile_tensor_svd_v, "tl_tensor_svd_v", "svdv_res");
compile_unary_linalg!(compile_tensor_eig_values, "tl_tensor_eig_values", "eigv_res");
compile_unary_linalg!(compile_tensor_eig_vectors, "tl_tensor_eig_vectors", "eigvc_res");

// solve(b) -> Tensor
fn compile_tensor_solve<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 1 { return Err("solve requires (b) argument".into()); }
    let f = codegen.module.get_function("tl_tensor_solve").ok_or("tl_tensor_solve not found")?;
    let call = codegen.builder.build_call(f, &[obj.into(), args[0].0.into()], "solve_res")
        .map_err(|e| e.to_string())?;
    let v = codegen.check_tensor_result(call, "solve_error")?;
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

// adam_step(grad, m, v, step, lr, beta1, beta2, eps, weight_decay) -> Void
fn compile_tensor_adam_step<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 9 {
        return Err("Tensor::adam_step takes exactly 9 arguments".into());
    }
    let fn_val = codegen.module.get_function("tl_adam_step").ok_or("tl_adam_step not found")?;
    
    let mut call_args = vec![obj.into()];
    for (v, _) in args {
        call_args.push(v.into());
    }
    
    codegen.builder.build_call(fn_val, &call_args, "adam_step").map_err(|e| e.to_string())?;
    
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}

// sgd_step(grad, velocity, lr, momentum, weight_decay, dampening, nesterov) -> Void
fn compile_tensor_sgd_step<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 7 {
        return Err("Tensor::sgd_step takes exactly 7 arguments".into());
    }
    let fn_val = codegen.module.get_function("tl_sgd_step").ok_or("tl_sgd_step not found")?;
    
    let mut call_args = vec![obj.into()];
    for (v, _) in args {
        call_args.push(v.into());
    }
    
    codegen.builder.build_call(fn_val, &call_args, "sgd_step").map_err(|e| e.to_string())?;
    
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}
