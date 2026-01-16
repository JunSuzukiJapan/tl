; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

@str_lit = private unnamed_addr constant [39 x i8] c"Testing tensor comprehension memory...\00", align 1
@str_lit.124 = private unnamed_addr constant [13 x i8] c"=== Attempt \00", align 1
@str_lit.125 = private unnamed_addr constant [9 x i8] c"  Epoch \00", align 1
@str_lit.126 = private unnamed_addr constant [15 x i8] c"  Attempt done\00", align 1
@str_lit.127 = private unnamed_addr constant [19 x i8] c"All attempts done!\00", align 1

declare void @tl_print_i64(i64 %0)

declare void @tl_display_i64(i64 %0)

declare void @tl_print_f32(float %0)

declare void @tl_display_f32(float %0)

declare float @tl_f32_abs(float %0)

declare float @tl_f32_acos(float %0)

declare float @tl_f32_acosh(float %0)

declare float @tl_f32_asin(float %0)

declare float @tl_f32_asinh(float %0)

declare float @tl_f32_atan(float %0)

declare float @tl_f32_atanh(float %0)

declare float @tl_f32_cbrt(float %0)

declare float @tl_f32_ceil(float %0)

declare float @tl_f32_cos(float %0)

declare float @tl_f32_cosh(float %0)

declare float @tl_f32_exp(float %0)

declare float @tl_f32_exp2(float %0)

declare float @tl_f32_exp_m1(float %0)

declare float @tl_f32_floor(float %0)

declare float @tl_f32_fract(float %0)

declare float @tl_f32_ln(float %0)

declare float @tl_f32_ln_1p(float %0)

declare float @tl_f32_log10(float %0)

declare float @tl_f32_log2(float %0)

declare float @tl_f32_recip(float %0)

declare float @tl_f32_round(float %0)

declare float @tl_f32_signum(float %0)

declare float @tl_f32_sin(float %0)

declare float @tl_f32_sinh(float %0)

declare float @tl_f32_sqrt(float %0)

declare float @tl_f32_tan(float %0)

declare float @tl_f32_tanh(float %0)

declare float @tl_f32_to_degrees(float %0)

declare float @tl_f32_to_radians(float %0)

declare float @tl_f32_trunc(float %0)

declare float @tl_f32_atan2(float %0, float %1)

declare float @tl_f32_copysign(float %0, float %1)

declare float @tl_f32_hypot(float %0, float %1)

declare float @tl_f32_log(float %0, float %1)

declare float @tl_f32_powf(float %0, float %1)

declare float @tl_f32_powi(float %0, i64 %1)

declare double @tl_f64_abs(double %0)

declare double @tl_f64_acos(double %0)

declare double @tl_f64_acosh(double %0)

declare double @tl_f64_asin(double %0)

declare double @tl_f64_asinh(double %0)

declare double @tl_f64_atan(double %0)

declare double @tl_f64_atanh(double %0)

declare double @tl_f64_cbrt(double %0)

declare double @tl_f64_ceil(double %0)

declare double @tl_f64_cos(double %0)

declare double @tl_f64_cosh(double %0)

declare double @tl_f64_exp(double %0)

declare double @tl_f64_exp2(double %0)

declare double @tl_f64_exp_m1(double %0)

declare double @tl_f64_floor(double %0)

declare double @tl_f64_fract(double %0)

declare double @tl_f64_ln(double %0)

declare double @tl_f64_ln_1p(double %0)

declare double @tl_f64_log10(double %0)

declare double @tl_f64_log2(double %0)

declare double @tl_f64_recip(double %0)

declare double @tl_f64_round(double %0)

declare double @tl_f64_signum(double %0)

declare double @tl_f64_sin(double %0)

declare double @tl_f64_sinh(double %0)

declare double @tl_f64_sqrt(double %0)

declare double @tl_f64_tan(double %0)

declare double @tl_f64_tanh(double %0)

declare double @tl_f64_to_degrees(double %0)

declare double @tl_f64_to_radians(double %0)

declare double @tl_f64_trunc(double %0)

declare double @tl_f64_atan2(double %0, double %1)

declare double @tl_f64_copysign(double %0, double %1)

declare double @tl_f64_hypot(double %0, double %1)

declare double @tl_f64_log(double %0, double %1)

declare double @tl_f64_powf(double %0, double %1)

declare double @tl_f64_powi(double %0, i64 %1)

declare i64 @tl_i64_abs(i64 %0)

declare i64 @tl_i64_signum(i64 %0)

declare i64 @tl_i64_div_euclid(i64 %0, i64 %1)

declare i64 @tl_i64_rem_euclid(i64 %0, i64 %1)

declare i64 @tl_i64_pow(i64 %0, i64 %1)

declare i1 @tl_i64_is_positive(i64 %0)

declare i1 @tl_i64_is_negative(i64 %0)

declare i32 @tl_i32_abs(i32 %0)

declare i32 @tl_i32_signum(i32 %0)

declare i32 @tl_i32_div_euclid(i32 %0, i32 %1)

declare i32 @tl_i32_rem_euclid(i32 %0, i32 %1)

declare i32 @tl_i32_pow(i32 %0, i32 %1)

declare i1 @tl_i32_is_positive(i32 %0)

declare i1 @tl_i32_is_negative(i32 %0)

declare void @tl_print_string(ptr %0)

declare void @tl_display_string(ptr %0)

declare void @tl_print_ptr(ptr %0)

declare void @tl_set_device(ptr %0)

declare void @tl_tensor_enable_grad(ptr %0)

declare ptr @tl_tensor_to_device(ptr %0, ptr %1)

declare ptr @malloc(i64 %0)

declare ptr @calloc(i64 %0, i64 %1)

declare void @free(ptr %0)

declare i64 @tl_tensor_dim(ptr %0, i64 %1)

declare float @tl_tensor_get_f32_md(ptr %0, ptr %1, i64 %2)

declare i64 @tl_tensor_get_i64_md(ptr %0, ptr %1, i64 %2)

declare ptr @tl_tensor_set_f32_md(ptr %0, ptr %1, i64 %2, float %3)

declare ptr @tl_tensor_new(ptr %0, i64 %1, ptr %2)

declare ptr @tl_tensor_from_i64_array(ptr %0, i64 %1)

declare ptr @tl_tensor_new_i64(ptr %0, i64 %1, ptr %2)

declare ptr @tl_tensor_sub(ptr %0, ptr %1)

declare void @tl_tensor_free(ptr %0)

declare ptr @tl_tensor_clone(ptr %0)

declare void @tl_tensor_acquire(ptr %0)

declare void @tl_tensor_release(ptr %0)

declare i64 @tl_vec_void_len(ptr %0)

declare ptr @tl_vec_void_get(ptr %0, i64 %1)

declare void @tl_vec_void_free(ptr %0)

declare ptr @tl_tensor_add(ptr %0, ptr %1)

declare ptr @tl_tensor_mul(ptr %0, ptr %1)

declare void @tl_tensor_print(ptr %0)

declare void @tl_tensor_display(ptr %0)

declare void @tl_tensor_print_1(ptr %0)

declare void @tl_tensor_print_2(ptr %0)

declare void @tl_tensor_print_3(ptr %0)

declare float @tl_tensor_get(ptr %0, i64 %1)

declare ptr @tl_tensor_slice(ptr %0, i64 %1, i64 %2)

declare i64 @tl_tensor_len(ptr %0)

declare ptr @tl_tensor_neg(ptr %0)

declare ptr @tl_tensor_transpose(ptr %0, i64 %1, i64 %2)

declare ptr @tl_tensor_pow(ptr %0, ptr %1)

declare ptr @tl_tensor_pow_scalar(ptr %0, float %1)

declare ptr @tl_tensor_sqrt(ptr %0)

declare ptr @tl_tensor_sin(ptr %0)

declare ptr @tl_tensor_cos(ptr %0)

declare ptr @tl_tensor_relu(ptr %0)

declare ptr @tl_tensor_gelu(ptr %0)

declare ptr @tl_tensor_tril(ptr %0, i32 %1)

declare void @tl_clear_grads()

declare ptr @tl_checkpoint(ptr %0, ptr %1, ptr %2)

declare ptr @tl_tensor_sum_dim(ptr %0, i64 %1, i1 %2)

declare ptr @tl_tensor_embedding(ptr %0, ptr %1)

declare ptr @tl_tensor_sum(ptr %0)

declare ptr @tl_tensor_div(ptr %0, ptr %1)

declare ptr @tl_tensor_matmul(ptr %0, ptr %1)

declare ptr @tl_tensor_exp(ptr %0)

declare ptr @tl_tensor_log(ptr %0)

declare void @tl_tensor_add_assign(ptr %0, ptr %1)

declare void @tl_tensor_sub_assign(ptr %0, ptr %1)

declare void @tl_tensor_mul_assign(ptr %0, ptr %1)

declare void @tl_tensor_div_assign(ptr %0, ptr %1)

declare void @tl_register_tensor(ptr %0, ptr %1)

declare i32 @strcmp(ptr %0, ptr %1)

declare void @tl_tensor_save(ptr %0, ptr %1)

declare ptr @tl_tensor_load(ptr %0)

declare ptr @tl_tensor_map_new()

declare void @tl_tensor_map_insert(ptr %0, ptr %1, ptr %2)

declare void @tl_tensor_map_save(ptr %0, ptr %1)

declare ptr @tl_tensor_map_load(ptr %0)

declare void @tl_tensor_map_free(ptr %0)

declare ptr @tl_tensor_reshape_dims(ptr %0, ptr %1, i64 %2)

declare ptr @tl_tensor_reshape_new(ptr %0, ptr %1)

declare ptr @tl_tensor_randn_debug(i64 %0, ptr %1, i1 %2)

declare ptr @tl_tensor_zeros(i64 %0, ptr %1, i1 %2)

declare ptr @tl_varbuilder_get(ptr %0, i64 %1, ptr %2)

declare ptr @tl_varbuilder_get_from_tensor(ptr %0, ptr %1)

declare void @tl_update_all_params(float %0)

declare ptr @tl_varbuilder_grad(ptr %0)

declare void @tl_tensor_backward(ptr %0)

declare ptr @tl_tensor_grad(ptr %0)

declare ptr @tl_tensor_detach(ptr %0, i1 %1)

declare ptr @tl_tensor_contiguous(ptr %0)

declare ptr @tl_tensor_softmax(ptr %0, i64 %1)

declare ptr @tl_tensor_cross_entropy(ptr %0, ptr %1)

declare void @tl_save_all_params(ptr %0)

declare void @tl_add_parameter(ptr %0, ptr %1)

declare void @tl_load_all_params(ptr %0)

declare ptr @tl_register_parameter(ptr %0)

declare ptr @tl_string_new(ptr %0)

declare ptr @tl_string_concat(ptr %0, ptr %1)

declare ptr @tl_file_open(ptr %0, ptr %1)

declare ptr @tl_file_read_string(ptr %0)

declare void @tl_file_write_string(ptr %0, ptr %1)

declare void @tl_file_close(ptr %0)

declare ptr @tl_path_new(ptr %0)

declare ptr @tl_path_join(ptr %0, ptr %1)

declare i1 @tl_path_exists(ptr %0)

declare i1 @tl_path_is_dir(ptr %0)

declare i1 @tl_path_is_file(ptr %0)

declare ptr @tl_path_to_string(ptr %0)

declare void @tl_path_free(ptr %0)

declare i1 @tl_http_download(ptr %0, ptr %1)

declare ptr @tl_http_get(ptr %0)

declare ptr @tl_env_get(ptr %0)

declare void @tl_env_set(ptr %0, ptr %1)

declare i64 @tl_args_count()

declare ptr @tl_args_get(i64 %0)

declare i64 @tl_string_to_i64(ptr %0)

declare ptr @tl_string_char_at(ptr %0, i64 %1)

declare i64 @tl_string_len(ptr %0)

declare ptr @tl_read_line(ptr %0)

declare float @tl_system_time()

declare void @tl_system_sleep(float %0)

declare i64 @tl_get_memory_mb()

declare void @tl_mem_enter_scope()

declare void @tl_mem_exit_scope()

declare void @tl_mem_register_struct(ptr %0)

declare void @tl_mem_register_tensor(ptr %0)

declare void @tl_mem_unregister(ptr %0)

declare ptr @tl_pool_acquire(i64 %0)

declare void @tl_pool_release(ptr %0, i64 %1)

declare void @tl_arena_init(i64 %0)

declare ptr @tl_arena_alloc(i64 %0)

declare void @tl_arena_free()

declare i1 @tl_arena_is_active()

declare void @tl_arena_reset()

declare i64 @tl_arena_get_offset()

declare i64 @tl_arena_get_capacity()

declare i64 @tl_tokenizer_new(ptr %0)

declare ptr @tl_tokenizer_encode(i64 %0, ptr %1)

declare ptr @tl_tokenizer_decode(i64 %0, ptr %1)

declare ptr @tl_gguf_load(ptr %0)

declare ptr @tl_tensor_map_get(ptr %0, ptr %1)

declare ptr @tl_tensor_cat(ptr %0, i64 %1)

declare ptr @tl_tensor_silu(ptr %0)

declare ptr @tl_tensor_scale(ptr %0, float %1)

declare ptr @tl_tensor_cat2(ptr %0, ptr %1, i64 %2)

declare ptr @tl_tensor_cat_4d(ptr %0, ptr %1, i64 %2)

declare ptr @tl_tensor_rms_norm(ptr %0, ptr %1, float %2)

declare ptr @tl_tensor_apply_rope(ptr %0, ptr %1, ptr %2)

declare ptr @tl_tensor_transpose_2d(ptr %0, i64 %1, i64 %2)

declare ptr @tl_tensor_matmul_4d(ptr %0, ptr %1)

declare ptr @tl_tensor_add_4d(ptr %0, ptr %1)

declare ptr @tl_tensor_silu_4d(ptr %0)

declare ptr @tl_tensor_reshape_2d(ptr %0, ptr %1, i64 %2)

declare ptr @tl_tensor_reshape_3d_to_2d(ptr %0, ptr %1, i64 %2)

declare ptr @tl_tensor_map_get_1d(ptr %0, ptr %1)

declare ptr @tl_tensor_narrow(ptr %0, i64 %1, i64 %2, i64 %3)

declare ptr @tl_tensor_repeat_interleave(ptr %0, i64 %1, i64 %2)

declare ptr @tl_tensor_rope_new_cos(i64 %0, i64 %1, float %2)

declare ptr @tl_tensor_rope_new_sin(i64 %0, i64 %1, float %2)

declare ptr @tl_tensor_new_causal_mask(i64 %0)

declare ptr @tl_tensor_cat_i64(ptr %0, ptr %1, i64 %2)

declare i64 @tl_tensor_device_id(ptr %0)

declare ptr @tl_tensor_tan(ptr %0)

declare ptr @tl_tensor_abs(ptr %0)

declare ptr @tl_tensor_sigmoid(ptr %0)

declare ptr @tl_tensor_tanh(ptr %0)

declare ptr @tl_tensor_max(ptr %0)

declare ptr @tl_tensor_max_dim(ptr %0, i64 %1, i1 %2)

declare ptr @tl_tensor_min(ptr %0)

declare ptr @tl_tensor_min_dim(ptr %0, i64 %1, i1 %2)

declare ptr @tl_tensor_mean(ptr %0)

declare ptr @tl_tensor_mean_dim(ptr %0, i64 %1, i1 %2)

declare ptr @tl_tensor_argmin(ptr %0, i64 %1, i1 %2)

declare ptr @tl_tensor_reshape_dims.1(ptr %0, ptr %1, i64 %2)

declare ptr @tl_tensor_reshape(ptr %0, ptr %1)

declare ptr @tl_varbuilder_get.2(ptr %0, i64 %1, ptr %2)

declare ptr @tl_varbuilder_get_from_tensor.3(ptr %0, ptr %1)

declare void @tl_update_all_params.4(float %0)

declare ptr @tl_varbuilder_grad.5(ptr %0)

declare void @tl_tensor_backward.6(ptr %0)

declare ptr @tl_tensor_grad.7(ptr %0)

declare ptr @tl_tensor_detach.8(ptr %0, i1 %1)

declare ptr @tl_tensor_softmax.9(ptr %0, i64 %1)

declare ptr @tl_tensor_cross_entropy.10(ptr %0, ptr %1)

declare void @tl_tensor_save.11(ptr %0, ptr %1)

declare ptr @tl_tensor_load.12(ptr %0)

declare ptr @tl_tensor_to_f32(ptr %0)

declare ptr @tl_tensor_to_i64(ptr %0)

declare void @tl_save_all_params.13(ptr %0)

declare void @tl_add_parameter.14(ptr %0, ptr %1)

declare ptr @tl_tensor_argmax(ptr %0, i64 %1, i1 %2)

declare float @tl_tensor_item(ptr %0)

declare i64 @tl_tensor_item_i64(ptr %0)

declare ptr @tl_tensor_to_i64.15(ptr %0)

declare void @tl_load_all_params.16(ptr %0)

declare void @tl_tensor_sub_assign.17(ptr %0, ptr %1)

declare void @tl_add_parameter.18(ptr %0, ptr %1)

declare ptr @tl_register_parameter.19(ptr %0)

declare ptr @tl_tensor_conv2d(ptr %0, ptr %1, i64 %2, i64 %3)

declare ptr @tl_tensor_clamp(ptr %0, float %1, float %2)

declare ptr @tl_tensor_ones(i64 %0, ptr %1, i1 %2)

declare ptr @tl_string_concat.20(ptr %0, ptr %1)

declare ptr @tl_file_open.21(ptr %0, ptr %1)

declare ptr @tl_file_read_string.22(ptr %0)

declare void @tl_file_write_string.23(ptr %0, ptr %1)

declare void @tl_file_close.24(ptr %0)

declare ptr @tl_path_new.25(ptr %0)

declare ptr @tl_path_join.26(ptr %0, ptr %1)

declare i1 @tl_path_exists.27(ptr %0)

declare i1 @tl_path_is_dir.28(ptr %0)

declare i1 @tl_path_is_file.29(ptr %0)

declare ptr @tl_path_to_string.30(ptr %0)

declare void @tl_path_free.31(ptr %0)

declare i1 @tl_http_download.32(ptr %0, ptr %1)

declare ptr @tl_http_get.33(ptr %0)

declare ptr @tl_env_get.34(ptr %0)

declare void @tl_env_set.35(ptr %0, ptr %1)

declare float @tl_system_time.36()

declare void @tl_system_sleep.37(float %0)

declare i64 @tl_get_memory_mb.38()

declare void @tl_mem_enter_scope.39()

declare void @tl_mem_exit_scope.40()

declare void @tl_mem_register_struct.41(ptr %0)

declare void @tl_mem_register_tensor.42(ptr %0)

declare void @tl_mem_unregister.43(ptr %0)

declare ptr @tl_pool_acquire.44(i64 %0)

declare void @tl_pool_release.45(ptr %0, i64 %1)

declare i64 @tl_kv_cache_new(i64 %0)

declare void @tl_kv_cache_free(i64 %0)

declare ptr @tl_kv_cache_get_k(i64 %0, i64 %1)

declare ptr @tl_kv_cache_get_v(i64 %0, i64 %1)

declare void @tl_kv_cache_update(i64 %0, i64 %1, ptr %2, ptr %3)

declare void @tl_arena_init.46(i64 %0)

declare i64 @tl_arena_alloc.47(i64 %0)

declare ptr @tl_arena_malloc(i64 %0)

declare i1 @tl_arena_is_active.48()

declare void @tl_arena_free.49()

declare ptr @tl_alloc_tmp(i64 %0)

declare void @tl_free_tmp(ptr %0)

declare ptr @tl_vec_u8_new()

declare ptr @tl_vec_u8_with_capacity(i64 %0)

declare i64 @tl_vec_u8_len(ptr %0)

declare i8 @tl_vec_u8_get(ptr %0, i64 %1)

declare i64 @tl_vec_u8_read_i32_be(ptr %0, i64 %1)

declare ptr @tl_tensor_from_vec_u8(ptr %0, i64 %1, ptr %2, i64 %3)

declare ptr @tl_tensor_from_u8_labels(ptr %0, i64 %1, i64 %2)

declare void @tl_vec_u8_set(ptr %0, i64 %1, i8 %2)

declare void @tl_vec_u8_push(ptr %0, i8 %1)

declare void @tl_vec_u8_free(ptr %0)

declare ptr @tl_file_read_binary(ptr %0)

declare i1 @tl_file_write_binary(ptr %0, ptr %1)

declare ptr @tl_image_load_grayscale(ptr %0)

declare i64 @tl_image_width(ptr %0)

declare i64 @tl_image_height(ptr %0)

declare i64 @tl_tensor_map_get_quantized(ptr %0, ptr %1)

declare ptr @tl_qtensor_matmul(ptr %0, i64 %1)

declare void @tl_qtensor_free(i64 %0)

declare i1 @tl_string_contains(ptr %0, ptr %1)

declare ptr @tl_string_from_int(i64 %0)

declare ptr @tl_tensor_sample(ptr %0, float %1, float %2)

declare ptr @tl_tensor_reshape_dims.50(ptr %0, ptr %1, i64 %2)

declare ptr @tl_tensor_reshape.51(ptr %0, ptr %1)

declare ptr @tl_varbuilder_get.52(ptr %0, i64 %1, ptr %2)

declare ptr @tl_varbuilder_get_from_tensor.53(ptr %0, ptr %1)

declare void @tl_update_all_params.54(float %0)

declare ptr @tl_varbuilder_grad.55(ptr %0)

declare void @tl_tensor_backward.56(ptr %0)

declare ptr @tl_tensor_grad.57(ptr %0)

declare ptr @tl_tensor_detach.58(ptr %0, i1 %1)

declare ptr @tl_tensor_softmax.59(ptr %0, i64 %1)

declare ptr @tl_tensor_cross_entropy.60(ptr %0, ptr %1)

declare void @tl_tensor_save.61(ptr %0, ptr %1)

declare ptr @tl_tensor_load.62(ptr %0)

declare void @tl_save_all_params.63(ptr %0)

declare void @tl_add_parameter.64(ptr %0, ptr %1)

declare ptr @tl_tensor_argmax.65(ptr %0, i64 %1, i1 %2)

declare float @tl_tensor_item.66(ptr %0)

declare i64 @tl_tensor_item_i64.67(ptr %0)

declare ptr @tl_tensor_to_i64.68(ptr %0)

declare void @tl_load_all_params.69(ptr %0)

declare void @tl_tensor_sub_assign.70(ptr %0, ptr %1)

declare void @tl_add_parameter.71(ptr %0, ptr %1)

declare ptr @tl_register_parameter.72(ptr %0)

declare ptr @tl_tensor_conv2d.73(ptr %0, ptr %1, i64 %2, i64 %3)

declare ptr @tl_tensor_clamp.74(ptr %0, float %1, float %2)

declare ptr @tl_tensor_ones.75(i64 %0, ptr %1, i1 %2)

declare ptr @tl_string_concat.76(ptr %0, ptr %1)

declare ptr @tl_file_open.77(ptr %0, ptr %1)

declare ptr @tl_file_read_string.78(ptr %0)

declare void @tl_file_write_string.79(ptr %0, ptr %1)

declare void @tl_file_close.80(ptr %0)

declare ptr @tl_path_new.81(ptr %0)

declare ptr @tl_path_join.82(ptr %0, ptr %1)

declare i1 @tl_path_exists.83(ptr %0)

declare i1 @tl_path_is_dir.84(ptr %0)

declare i1 @tl_path_is_file.85(ptr %0)

declare ptr @tl_path_to_string.86(ptr %0)

declare void @tl_path_free.87(ptr %0)

declare i1 @tl_http_download.88(ptr %0, ptr %1)

declare ptr @tl_http_get.89(ptr %0)

declare ptr @tl_env_get.90(ptr %0)

declare void @tl_env_set.91(ptr %0, ptr %1)

declare float @tl_system_time.92()

declare void @tl_system_sleep.93(float %0)

declare i64 @tl_get_memory_mb.94()

declare void @tl_mem_enter_scope.95()

declare void @tl_mem_exit_scope.96()

declare void @tl_mem_register_struct.97(ptr %0)

declare void @tl_mem_register_tensor.98(ptr %0)

declare void @tl_mem_unregister.99(ptr %0)

declare ptr @tl_pool_acquire.100(i64 %0)

declare void @tl_pool_release.101(ptr %0, i64 %1)

declare void @tl_arena_init.102(i64 %0)

declare i64 @tl_arena_alloc.103(i64 %0)

declare ptr @tl_arena_malloc.104(i64 %0)

declare i1 @tl_arena_is_active.105()

declare void @tl_arena_free.106()

declare ptr @tl_alloc_tmp.107(i64 %0)

declare void @tl_free_tmp.108(ptr %0)

declare ptr @tl_vec_u8_new.109()

declare ptr @tl_vec_u8_with_capacity.110(i64 %0)

declare i64 @tl_vec_u8_len.111(ptr %0)

declare i8 @tl_vec_u8_get.112(ptr %0, i64 %1)

declare i64 @tl_vec_u8_read_i32_be.113(ptr %0, i64 %1)

declare ptr @tl_tensor_from_vec_u8.114(ptr %0, i64 %1, ptr %2, i64 %3)

declare ptr @tl_tensor_from_u8_labels.115(ptr %0, i64 %1, i64 %2)

declare void @tl_vec_u8_set.116(ptr %0, i64 %1, i8 %2)

declare void @tl_vec_u8_push.117(ptr %0, i8 %1)

declare void @tl_vec_u8_free.118(ptr %0)

declare ptr @tl_file_read_binary.119(ptr %0)

declare i1 @tl_file_write_binary.120(ptr %0, ptr %1)

declare ptr @tl_image_load_grayscale.121(ptr %0)

declare i64 @tl_image_width.122(ptr %0)

declare i64 @tl_image_height.123(ptr %0)

define void @main() {
entry:
  %scalar_shape_rhs142 = alloca i64, align 16
  %scalar_data_rhs141 = alloca float, align 16
  %g = alloca ptr, align 16
  %total_loss = alloca ptr, align 16
  %main_diag_loss = alloca ptr, align 16
  %scalar_shape_rhs119 = alloca i64, align 16
  %scalar_data_rhs118 = alloca float, align 16
  %anti_diag_loss = alloca ptr, align 16
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %main_diag_sums = alloca ptr, align 16
  %_comp_res_1 = alloca ptr, align 16
  %idx_arr80 = alloca [2 x i64], align 8
  %c67 = alloca i64, align 16
  %r61 = alloca i64, align 16
  %k55 = alloca i64, align 16
  %anti_diag_sums = alloca ptr, align 16
  %_comp_res_0 = alloca ptr, align 16
  %idx_arr = alloca [2 x i64], align 8
  %c23 = alloca i64, align 16
  %r21 = alloca i64, align 16
  %k19 = alloca i64, align 16
  %probs = alloca ptr, align 16
  %i = alloca i64, align 16
  %board = alloca ptr, align 16
  %shape_arr = alloca [2 x i64], align 16
  %max_attempts = alloca i64, align 16
  %attempt = alloca i64, align 16
  %N = alloca i64, align 16
  call void @tl_mem_enter_scope()
  call void @tl_arena_init(i64 102400)
  %string_obj = call ptr @tl_string_new(ptr @str_lit)
  call void @tl_print_string(ptr %string_obj)
  store i64 8, ptr %N, align 8
  store i64 0, ptr %attempt, align 8
  store i64 1, ptr %max_attempts, align 8
  br label %while_cond

while_cond:                                       ; preds = %after_free201, %entry
  call void @tl_mem_enter_scope()
  %attempt1 = load i64, ptr %attempt, align 8
  %max_attempts2 = load i64, ptr %max_attempts, align 8
  %lttmp = icmp slt i64 %attempt1, %max_attempts2
  %while_cond_check = icmp ne i1 %lttmp, false
  call void @tl_mem_exit_scope()
  br i1 %while_cond_check, label %while_body, label %while_end

while_body:                                       ; preds = %while_cond
  call void @tl_mem_enter_scope()
  %string_obj3 = call ptr @tl_string_new(ptr @str_lit.124)
  call void @tl_display_string(ptr %string_obj3)
  %attempt4 = load i64, ptr %attempt, align 8
  call void @tl_print_i64(i64 %attempt4)
  %N5 = load i64, ptr %N, align 8
  %N6 = load i64, ptr %N, align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %N5, ptr %shape_ptr_in, align 8
  %shape_ptr_in7 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 %N6, ptr %shape_ptr_in7, align 8
  %first_elem_ptr = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  %creation_res = call ptr @tl_tensor_randn_debug(i64 2, ptr %first_elem_ptr, i1 true)
  store ptr %creation_res, ptr %board, align 8
  br label %for_header

while_end:                                        ; preds = %while_cond
  %string_obj203 = call ptr @tl_string_new(ptr @str_lit.127)
  call void @tl_print_string(ptr %string_obj203)
  call void @tl_mem_exit_scope()
  ret void

for_header:                                       ; preds = %after_free194, %while_body
  %for_idx = phi i64 [ %next_idx, %after_free194 ], [ 0, %while_body ]
  %for_cond = icmp slt i64 %for_idx, 2
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %i, align 8
  %i8 = load i64, ptr %i, align 8
  %modtmp = srem i64 %i8, 50
  %eqtmp = icmp eq i64 %modtmp, 0
  br i1 %eqtmp, label %then, label %else

for_end:                                          ; preds = %for_header
  %string_obj196 = call ptr @tl_string_new(ptr @str_lit.126)
  call void @tl_print_string(ptr %string_obj196)
  %attempt197 = load i64, ptr %attempt, align 8
  %addtmp198 = add i64 %attempt197, 1
  store i64 %addtmp198, ptr %attempt, align 8
  %tensor_to_free199 = load ptr, ptr %board, align 8
  %is_null202 = icmp eq ptr %tensor_to_free199, null
  br i1 %is_null202, label %after_free201, label %free_tensor200

then:                                             ; preds = %for_body
  call void @tl_mem_enter_scope()
  %string_obj9 = call ptr @tl_string_new(ptr @str_lit.125)
  call void @tl_display_string(ptr %string_obj9)
  %i10 = load i64, ptr %i, align 8
  call void @tl_print_i64(i64 %i10)
  call void @tl_mem_exit_scope()
  br label %merge

else:                                             ; preds = %for_body
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else, %then
  %board11 = load ptr, ptr %board, align 8
  %method_call = call ptr @tl_tensor_softmax(ptr %board11, i64 1)
  call void @tl_mem_register_tensor(ptr %method_call)
  store ptr %method_call, ptr %probs, align 8
  %probs12 = load ptr, ptr %probs, align 8
  %dim_size = call i64 @tl_tensor_dim(ptr %probs12, i64 0)
  %dim_size13 = call i64 @tl_tensor_dim(ptr %probs12, i64 1)
  %N14 = load i64, ptr %N, align 8
  %multmp = mul i64 2, %N14
  %subtmp = sub i64 %multmp, 1
  %count = sub i64 %subtmp, 0
  %N15 = load i64, ptr %N, align 8
  %count16 = sub i64 %N15, 0
  %N17 = load i64, ptr %N, align 8
  %count18 = sub i64 %N17, 0
  %dim_sz = sub i64 %subtmp, 0
  %sz_acc = mul i64 1, %dim_sz
  %buf_void = call ptr @calloc(i64 %sz_acc, i64 4)
  call void @tl_mem_enter_scope()
  br label %loop_cond_k

loop_cond_k:                                      ; preds = %loop_latch_k, %merge
  %k = phi i64 [ 0, %merge ], [ %next35, %loop_latch_k ]
  %cmp = icmp slt i64 %k, %subtmp
  br i1 %cmp, label %loop_body_k, label %loop_aft_k

loop_body_k:                                      ; preds = %loop_cond_k
  store i64 %k, ptr %k19, align 8
  br label %loop_cond_r

loop_aft_k:                                       ; preds = %loop_cond_k
  %shape = alloca [1 x i64], align 8
  %dim = sub i64 %subtmp, 0
  %shape_ptr = getelementptr [1 x i64], ptr %shape, i64 0, i64 0
  store i64 %dim, ptr %shape_ptr, align 8
  %t = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %shape)
  store ptr %t, ptr %_comp_res_0, align 8
  %tensor_ptr = load ptr, ptr %_comp_res_0, align 8
  call void @tl_tensor_acquire(ptr %tensor_ptr)
  store ptr %tensor_ptr, ptr %anti_diag_sums, align 8
  %probs36 = load ptr, ptr %probs, align 8
  %dim_size37 = call i64 @tl_tensor_dim(ptr %probs36, i64 0)
  %dim_size38 = call i64 @tl_tensor_dim(ptr %probs36, i64 1)
  %N39 = load i64, ptr %N, align 8
  %multmp40 = mul i64 2, %N39
  %subtmp41 = sub i64 %multmp40, 1
  %count42 = sub i64 %subtmp41, 0
  %N43 = load i64, ptr %N, align 8
  %count44 = sub i64 %N43, 0
  %N45 = load i64, ptr %N, align 8
  %count46 = sub i64 %N45, 0
  %dim_sz47 = sub i64 %subtmp41, 0
  %sz_acc48 = mul i64 1, %dim_sz47
  %buf_void49 = call ptr @calloc(i64 %sz_acc48, i64 4)
  call void @tl_mem_enter_scope()
  br label %loop_cond_k50

loop_cond_r:                                      ; preds = %loop_latch_r, %loop_body_k
  %r = phi i64 [ 0, %loop_body_k ], [ %next33, %loop_latch_r ]
  %cmp20 = icmp slt i64 %r, %N15
  br i1 %cmp20, label %loop_body_r, label %loop_aft_r

loop_body_r:                                      ; preds = %loop_cond_r
  store i64 %r, ptr %r21, align 8
  br label %loop_cond_c

loop_aft_r:                                       ; preds = %loop_cond_r
  br label %loop_latch_k

loop_cond_c:                                      ; preds = %loop_latch_c, %loop_body_r
  %c = phi i64 [ 0, %loop_body_r ], [ %next, %loop_latch_c ]
  %cmp22 = icmp slt i64 %c, %N17
  br i1 %cmp22, label %loop_body_c, label %loop_aft_c

loop_body_c:                                      ; preds = %loop_cond_c
  store i64 %c, ptr %c23, align 8
  br label %check_cond

loop_aft_c:                                       ; preds = %loop_cond_c
  br label %loop_latch_r

check_cond:                                       ; preds = %loop_body_c
  %r24 = load i64, ptr %r21, align 8
  %c25 = load i64, ptr %c23, align 8
  %addtmp = add i64 %r24, %c25
  %k26 = load i64, ptr %k19, align 8
  %eqtmp27 = icmp eq i64 %addtmp, %k26
  br i1 %eqtmp27, label %cond_true, label %cond_false

cond_true:                                        ; preds = %check_cond
  call void @tl_mem_enter_scope()
  %probs28 = load ptr, ptr %probs, align 8
  %r29 = load i64, ptr %r21, align 8
  %idx_ptr = getelementptr [2 x i64], ptr %idx_arr, i64 0, i64 0
  store i64 %r29, ptr %idx_ptr, align 8
  %c30 = load i64, ptr %c23, align 8
  %idx_ptr31 = getelementptr [2 x i64], ptr %idx_arr, i64 0, i64 1
  store i64 %c30, ptr %idx_ptr31, align 8
  %get_md_call = call float @tl_tensor_get_f32_md(ptr %probs28, ptr %idx_arr, i64 2)
  call void @tl_mem_exit_scope()
  %lim = sub i64 %subtmp, 0
  %rel_idx = sub i64 %k, 0
  %term = mul i64 %rel_idx, 1
  %off = add i64 0, %term
  %new_str = mul i64 1, %lim
  %elem_off = add i64 %off, 0
  %ptr_bound = getelementptr float, ptr %buf_void, i64 %elem_off
  %curr_val = load float, ptr %ptr_bound, align 4
  %accum = fadd float %curr_val, %get_md_call
  store float %accum, ptr %ptr_bound, align 4
  br label %loop_latch_c

cond_false:                                       ; preds = %check_cond
  br label %loop_latch_c

loop_latch_c:                                     ; preds = %cond_false, %cond_true
  %iv = load i64, ptr %c23, align 8
  %next = add i64 %iv, 1
  br label %loop_cond_c

loop_latch_r:                                     ; preds = %loop_aft_c
  %iv32 = load i64, ptr %r21, align 8
  %next33 = add i64 %iv32, 1
  br label %loop_cond_r

loop_latch_k:                                     ; preds = %loop_aft_r
  %iv34 = load i64, ptr %k19, align 8
  %next35 = add i64 %iv34, 1
  br label %loop_cond_k

loop_cond_k50:                                    ; preds = %loop_latch_k101, %loop_aft_k
  %k53 = phi i64 [ 0, %loop_aft_k ], [ %next103, %loop_latch_k101 ]
  %cmp54 = icmp slt i64 %k53, %subtmp41
  br i1 %cmp54, label %loop_body_k51, label %loop_aft_k52

loop_body_k51:                                    ; preds = %loop_cond_k50
  store i64 %k53, ptr %k55, align 8
  br label %loop_cond_r56

loop_aft_k52:                                     ; preds = %loop_cond_k50
  %shape104 = alloca [1 x i64], align 8
  %dim105 = sub i64 %subtmp41, 0
  %shape_ptr106 = getelementptr [1 x i64], ptr %shape104, i64 0, i64 0
  store i64 %dim105, ptr %shape_ptr106, align 8
  %t107 = call ptr @tl_tensor_new(ptr %buf_void49, i64 1, ptr %shape104)
  store ptr %t107, ptr %_comp_res_1, align 8
  %tensor_ptr108 = load ptr, ptr %_comp_res_1, align 8
  call void @tl_tensor_acquire(ptr %tensor_ptr108)
  store ptr %tensor_ptr108, ptr %main_diag_sums, align 8
  %anti_diag_sums109 = load ptr, ptr %anti_diag_sums, align 8
  store float 1.000000e+00, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_sub(ptr %anti_diag_sums109, ptr %scalar_tensor_rhs)
  call void @tl_mem_register_tensor(ptr %binop_res)
  %method_call110 = call ptr @tl_tensor_relu(ptr %binop_res)
  %is_null = icmp eq ptr %binop_res, null
  br i1 %is_null, label %after_free, label %free_tensor

loop_cond_r56:                                    ; preds = %loop_latch_r98, %loop_body_k51
  %r59 = phi i64 [ 0, %loop_body_k51 ], [ %next100, %loop_latch_r98 ]
  %cmp60 = icmp slt i64 %r59, %N43
  br i1 %cmp60, label %loop_body_r57, label %loop_aft_r58

loop_body_r57:                                    ; preds = %loop_cond_r56
  store i64 %r59, ptr %r61, align 8
  br label %loop_cond_c62

loop_aft_r58:                                     ; preds = %loop_cond_r56
  br label %loop_latch_k101

loop_cond_c62:                                    ; preds = %loop_latch_c95, %loop_body_r57
  %c65 = phi i64 [ 0, %loop_body_r57 ], [ %next97, %loop_latch_c95 ]
  %cmp66 = icmp slt i64 %c65, %N45
  br i1 %cmp66, label %loop_body_c63, label %loop_aft_c64

loop_body_c63:                                    ; preds = %loop_cond_c62
  store i64 %c65, ptr %c67, align 8
  br label %check_cond68

loop_aft_c64:                                     ; preds = %loop_cond_c62
  br label %loop_latch_r98

check_cond68:                                     ; preds = %loop_body_c63
  %r71 = load i64, ptr %r61, align 8
  %c72 = load i64, ptr %c67, align 8
  %subtmp73 = sub i64 %r71, %c72
  %N74 = load i64, ptr %N, align 8
  %addtmp75 = add i64 %subtmp73, %N74
  %subtmp76 = sub i64 %addtmp75, 1
  %k77 = load i64, ptr %k55, align 8
  %eqtmp78 = icmp eq i64 %subtmp76, %k77
  br i1 %eqtmp78, label %cond_true69, label %cond_false70

cond_true69:                                      ; preds = %check_cond68
  call void @tl_mem_enter_scope()
  %probs79 = load ptr, ptr %probs, align 8
  %r81 = load i64, ptr %r61, align 8
  %idx_ptr82 = getelementptr [2 x i64], ptr %idx_arr80, i64 0, i64 0
  store i64 %r81, ptr %idx_ptr82, align 8
  %c83 = load i64, ptr %c67, align 8
  %idx_ptr84 = getelementptr [2 x i64], ptr %idx_arr80, i64 0, i64 1
  store i64 %c83, ptr %idx_ptr84, align 8
  %get_md_call85 = call float @tl_tensor_get_f32_md(ptr %probs79, ptr %idx_arr80, i64 2)
  call void @tl_mem_exit_scope()
  %lim86 = sub i64 %subtmp41, 0
  %rel_idx87 = sub i64 %k53, 0
  %term88 = mul i64 %rel_idx87, 1
  %off89 = add i64 0, %term88
  %new_str90 = mul i64 1, %lim86
  %elem_off91 = add i64 %off89, 0
  %ptr_bound92 = getelementptr float, ptr %buf_void49, i64 %elem_off91
  %curr_val93 = load float, ptr %ptr_bound92, align 4
  %accum94 = fadd float %curr_val93, %get_md_call85
  store float %accum94, ptr %ptr_bound92, align 4
  br label %loop_latch_c95

cond_false70:                                     ; preds = %check_cond68
  br label %loop_latch_c95

loop_latch_c95:                                   ; preds = %cond_false70, %cond_true69
  %iv96 = load i64, ptr %c67, align 8
  %next97 = add i64 %iv96, 1
  br label %loop_cond_c62

loop_latch_r98:                                   ; preds = %loop_aft_c64
  %iv99 = load i64, ptr %r61, align 8
  %next100 = add i64 %iv99, 1
  br label %loop_cond_r56

loop_latch_k101:                                  ; preds = %loop_aft_r58
  %iv102 = load i64, ptr %k55, align 8
  %next103 = add i64 %iv102, 1
  br label %loop_cond_k50

free_tensor:                                      ; preds = %loop_aft_k52
  call void @tl_tensor_release(ptr %binop_res)
  br label %after_free

after_free:                                       ; preds = %free_tensor, %loop_aft_k52
  call void @tl_mem_register_tensor(ptr %method_call110)
  %pow_scalar_res = call ptr @tl_tensor_pow_scalar(ptr %method_call110, float 2.000000e+00)
  call void @tl_mem_register_tensor(ptr %pow_scalar_res)
  %is_null113 = icmp eq ptr %method_call110, null
  br i1 %is_null113, label %after_free112, label %free_tensor111

free_tensor111:                                   ; preds = %after_free
  call void @tl_tensor_release(ptr %method_call110)
  br label %after_free112

after_free112:                                    ; preds = %free_tensor111, %after_free
  %sum_res = call ptr @tl_tensor_sum(ptr %pow_scalar_res)
  %is_null116 = icmp eq ptr %pow_scalar_res, null
  br i1 %is_null116, label %after_free115, label %free_tensor114

free_tensor114:                                   ; preds = %after_free112
  call void @tl_tensor_release(ptr %pow_scalar_res)
  br label %after_free115

after_free115:                                    ; preds = %free_tensor114, %after_free112
  store ptr %sum_res, ptr %anti_diag_loss, align 8
  %main_diag_sums117 = load ptr, ptr %main_diag_sums, align 8
  store float 1.000000e+00, ptr %scalar_data_rhs118, align 4
  %scalar_tensor_rhs120 = call ptr @tl_tensor_new(ptr %scalar_data_rhs118, i64 0, ptr %scalar_shape_rhs119)
  %binop_res121 = call ptr @tl_tensor_sub(ptr %main_diag_sums117, ptr %scalar_tensor_rhs120)
  call void @tl_mem_register_tensor(ptr %binop_res121)
  %method_call122 = call ptr @tl_tensor_relu(ptr %binop_res121)
  %is_null125 = icmp eq ptr %binop_res121, null
  br i1 %is_null125, label %after_free124, label %free_tensor123

free_tensor123:                                   ; preds = %after_free115
  call void @tl_tensor_release(ptr %binop_res121)
  br label %after_free124

after_free124:                                    ; preds = %free_tensor123, %after_free115
  call void @tl_mem_register_tensor(ptr %method_call122)
  %pow_scalar_res126 = call ptr @tl_tensor_pow_scalar(ptr %method_call122, float 2.000000e+00)
  call void @tl_mem_register_tensor(ptr %pow_scalar_res126)
  %is_null129 = icmp eq ptr %method_call122, null
  br i1 %is_null129, label %after_free128, label %free_tensor127

free_tensor127:                                   ; preds = %after_free124
  call void @tl_tensor_release(ptr %method_call122)
  br label %after_free128

after_free128:                                    ; preds = %free_tensor127, %after_free124
  %sum_res130 = call ptr @tl_tensor_sum(ptr %pow_scalar_res126)
  %is_null133 = icmp eq ptr %pow_scalar_res126, null
  br i1 %is_null133, label %after_free132, label %free_tensor131

free_tensor131:                                   ; preds = %after_free128
  call void @tl_tensor_release(ptr %pow_scalar_res126)
  br label %after_free132

after_free132:                                    ; preds = %free_tensor131, %after_free128
  store ptr %sum_res130, ptr %main_diag_loss, align 8
  %anti_diag_loss134 = load ptr, ptr %anti_diag_loss, align 8
  %main_diag_loss135 = load ptr, ptr %main_diag_loss, align 8
  %binop_res136 = call ptr @tl_tensor_add(ptr %anti_diag_loss134, ptr %main_diag_loss135)
  call void @tl_mem_register_tensor(ptr %binop_res136)
  store ptr %binop_res136, ptr %total_loss, align 8
  %total_loss137 = load ptr, ptr %total_loss, align 8
  call void @tl_tensor_backward(ptr %total_loss137)
  %board138 = load ptr, ptr %board, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %board138)
  call void @tl_mem_register_tensor(ptr %grad_res)
  store ptr %grad_res, ptr %g, align 8
  %board139 = load ptr, ptr %board, align 8
  %g140 = load ptr, ptr %g, align 8
  store float 5.000000e-01, ptr %scalar_data_rhs141, align 4
  %scalar_tensor_rhs143 = call ptr @tl_tensor_new(ptr %scalar_data_rhs141, i64 0, ptr %scalar_shape_rhs142)
  %binop_res144 = call ptr @tl_tensor_mul(ptr %g140, ptr %scalar_tensor_rhs143)
  call void @tl_mem_register_tensor(ptr %binop_res144)
  %binop_res145 = call ptr @tl_tensor_sub(ptr %board139, ptr %binop_res144)
  call void @tl_mem_register_tensor(ptr %binop_res145)
  call void @tl_mem_unregister(ptr %binop_res145)
  %old_val_to_free = load ptr, ptr %board, align 8
  %is_not_null = icmp ne ptr %old_val_to_free, null
  %are_diff = icmp ne ptr %old_val_to_free, %binop_res145
  %can_free_1 = and i1 %is_not_null, true
  %can_free = and i1 %can_free_1, %are_diff
  br i1 %can_free, label %free_struct, label %continue_after_free

free_struct:                                      ; preds = %after_free132
  %is_null148 = icmp eq ptr %old_val_to_free, null
  br i1 %is_null148, label %after_free147, label %free_tensor146

continue_after_free:                              ; preds = %after_free147, %after_free132
  call void @tl_mem_unregister(ptr %binop_res145)
  store ptr %binop_res145, ptr %board, align 8
  %board149 = load ptr, ptr %board, align 8
  %detach_res = call ptr @tl_tensor_detach(ptr %board149, i1 false)
  call void @tl_mem_register_tensor(ptr %detach_res)
  call void @tl_mem_unregister(ptr %detach_res)
  %old_val_to_free150 = load ptr, ptr %board, align 8
  %is_not_null151 = icmp ne ptr %old_val_to_free150, null
  %are_diff152 = icmp ne ptr %old_val_to_free150, %detach_res
  %can_free_1153 = and i1 %is_not_null151, true
  %can_free154 = and i1 %can_free_1153, %are_diff152
  br i1 %can_free154, label %free_struct155, label %continue_after_free156

free_tensor146:                                   ; preds = %free_struct
  call void @tl_tensor_release(ptr %old_val_to_free)
  br label %after_free147

after_free147:                                    ; preds = %free_tensor146, %free_struct
  call void @tl_mem_unregister(ptr %old_val_to_free)
  br label %continue_after_free

free_struct155:                                   ; preds = %continue_after_free
  %is_null159 = icmp eq ptr %old_val_to_free150, null
  br i1 %is_null159, label %after_free158, label %free_tensor157

continue_after_free156:                           ; preds = %after_free158, %continue_after_free
  call void @tl_mem_unregister(ptr %detach_res)
  store ptr %detach_res, ptr %board, align 8
  %board160 = load ptr, ptr %board, align 8
  call void @tl_tensor_enable_grad(ptr %board160)
  %tensor_to_free = load ptr, ptr %main_diag_sums, align 8
  %is_null163 = icmp eq ptr %tensor_to_free, null
  br i1 %is_null163, label %after_free162, label %free_tensor161

free_tensor157:                                   ; preds = %free_struct155
  call void @tl_tensor_release(ptr %old_val_to_free150)
  br label %after_free158

after_free158:                                    ; preds = %free_tensor157, %free_struct155
  call void @tl_mem_unregister(ptr %old_val_to_free150)
  br label %continue_after_free156

free_tensor161:                                   ; preds = %continue_after_free156
  call void @tl_tensor_release(ptr %tensor_to_free)
  br label %after_free162

after_free162:                                    ; preds = %free_tensor161, %continue_after_free156
  %tensor_to_free164 = load ptr, ptr %g, align 8
  %is_null167 = icmp eq ptr %tensor_to_free164, null
  br i1 %is_null167, label %after_free166, label %free_tensor165

free_tensor165:                                   ; preds = %after_free162
  call void @tl_tensor_release(ptr %tensor_to_free164)
  br label %after_free166

after_free166:                                    ; preds = %free_tensor165, %after_free162
  %tensor_to_free168 = load ptr, ptr %anti_diag_sums, align 8
  %is_null171 = icmp eq ptr %tensor_to_free168, null
  br i1 %is_null171, label %after_free170, label %free_tensor169

free_tensor169:                                   ; preds = %after_free166
  call void @tl_tensor_release(ptr %tensor_to_free168)
  br label %after_free170

after_free170:                                    ; preds = %free_tensor169, %after_free166
  %tensor_to_free172 = load ptr, ptr %anti_diag_loss, align 8
  %is_null175 = icmp eq ptr %tensor_to_free172, null
  br i1 %is_null175, label %after_free174, label %free_tensor173

free_tensor173:                                   ; preds = %after_free170
  call void @tl_tensor_release(ptr %tensor_to_free172)
  br label %after_free174

after_free174:                                    ; preds = %free_tensor173, %after_free170
  %tensor_to_free176 = load ptr, ptr %_comp_res_0, align 8
  %is_null179 = icmp eq ptr %tensor_to_free176, null
  br i1 %is_null179, label %after_free178, label %free_tensor177

free_tensor177:                                   ; preds = %after_free174
  call void @tl_tensor_release(ptr %tensor_to_free176)
  br label %after_free178

after_free178:                                    ; preds = %free_tensor177, %after_free174
  %tensor_to_free180 = load ptr, ptr %_comp_res_1, align 8
  %is_null183 = icmp eq ptr %tensor_to_free180, null
  br i1 %is_null183, label %after_free182, label %free_tensor181

free_tensor181:                                   ; preds = %after_free178
  call void @tl_tensor_release(ptr %tensor_to_free180)
  br label %after_free182

after_free182:                                    ; preds = %free_tensor181, %after_free178
  %tensor_to_free184 = load ptr, ptr %probs, align 8
  %is_null187 = icmp eq ptr %tensor_to_free184, null
  br i1 %is_null187, label %after_free186, label %free_tensor185

free_tensor185:                                   ; preds = %after_free182
  call void @tl_tensor_release(ptr %tensor_to_free184)
  br label %after_free186

after_free186:                                    ; preds = %free_tensor185, %after_free182
  %tensor_to_free188 = load ptr, ptr %main_diag_loss, align 8
  %is_null191 = icmp eq ptr %tensor_to_free188, null
  br i1 %is_null191, label %after_free190, label %free_tensor189

free_tensor189:                                   ; preds = %after_free186
  call void @tl_tensor_release(ptr %tensor_to_free188)
  br label %after_free190

after_free190:                                    ; preds = %free_tensor189, %after_free186
  %tensor_to_free192 = load ptr, ptr %total_loss, align 8
  %is_null195 = icmp eq ptr %tensor_to_free192, null
  br i1 %is_null195, label %after_free194, label %free_tensor193

free_tensor193:                                   ; preds = %after_free190
  call void @tl_tensor_release(ptr %tensor_to_free192)
  br label %after_free194

after_free194:                                    ; preds = %free_tensor193, %after_free190
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx, 1
  br label %for_header

free_tensor200:                                   ; preds = %for_end
  call void @tl_tensor_release(ptr %tensor_to_free199)
  br label %after_free201

after_free201:                                    ; preds = %free_tensor200, %for_end
  call void @tl_mem_exit_scope()
  br label %while_cond
}
