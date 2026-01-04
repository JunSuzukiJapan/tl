; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

declare void @tl_print_i64(i64)

declare void @tl_print_f32(float)

declare void @tl_print_string(ptr)

declare ptr @malloc(i64)

declare ptr @calloc(i64, i64)

declare void @free(ptr)

declare i64 @tl_tensor_dim(ptr, i64)

declare float @tl_tensor_get_f32_md(ptr, ptr, i64)

declare ptr @tl_tensor_new(ptr, i64, ptr)

declare ptr @tl_tensor_sub(ptr, ptr)

declare void @tl_tensor_free(ptr)

declare ptr @tl_tensor_clone(ptr)

declare ptr @tl_tensor_add(ptr, ptr)

declare ptr @tl_tensor_mul(ptr, ptr)

declare void @tl_tensor_print(ptr)

declare float @tl_tensor_get(ptr, i64)

declare ptr @tl_tensor_slice(ptr, i64, i64)

declare i64 @tl_tensor_len(ptr)

declare ptr @tl_tensor_neg(ptr)

declare ptr @tl_tensor_transpose(ptr, i64, i64)

declare ptr @tl_tensor_pow(ptr, ptr)

declare ptr @tl_tensor_sqrt(ptr)

declare ptr @tl_tensor_sin(ptr)

declare ptr @tl_tensor_cos(ptr)

declare ptr @tl_tensor_relu(ptr)

declare ptr @tl_tensor_gelu(ptr)

declare ptr @tl_tensor_tril(ptr, i32)

declare ptr @tl_tensor_sum_dim(ptr, i64, i1)

declare ptr @tl_tensor_embedding(ptr, ptr)

declare ptr @tl_tensor_sum(ptr)

declare ptr @tl_tensor_div(ptr, ptr)

declare ptr @tl_tensor_matmul(ptr, ptr)

declare ptr @tl_tensor_exp(ptr)

declare ptr @tl_tensor_log(ptr)

declare void @tl_tensor_add_assign(ptr, ptr)

declare void @tl_tensor_sub_assign(ptr, ptr)

declare void @tl_tensor_mul_assign(ptr, ptr)

declare void @tl_tensor_div_assign(ptr, ptr)

declare void @tl_register_tensor(ptr, ptr)

declare i32 @strcmp(ptr, ptr)

declare void @tl_tensor_save(ptr, ptr)

declare ptr @tl_tensor_load(ptr)

declare ptr @tl_tensor_map_new()

declare void @tl_tensor_map_insert(ptr, ptr, ptr)

declare void @tl_tensor_map_save(ptr, ptr)

declare ptr @tl_tensor_map_load(ptr)

declare ptr @tl_tensor_map_get(ptr, ptr)

declare void @tl_tensor_map_free(ptr)

declare ptr @tl_tensor_reshape_dims(ptr, ptr, i64)

declare ptr @tl_tensor_reshape(ptr, ptr)

declare ptr @tl_tensor_randn(ptr, i1)

declare ptr @tl_varbuilder_get(ptr, i64, ptr)

declare ptr @tl_varbuilder_get_from_tensor(ptr, ptr)

declare void @tl_update_all_params(float)

declare ptr @tl_varbuilder_grad(ptr)

declare void @tl_tensor_backward(ptr)

declare ptr @tl_tensor_grad(ptr)

declare ptr @tl_tensor_detach(ptr, i1)

declare ptr @tl_tensor_softmax(ptr, i64)

declare ptr @tl_tensor_cross_entropy(ptr, ptr)

declare void @tl_save_all_params(ptr)

declare void @tl_add_parameter(ptr, ptr)

declare void @tl_load_all_params(ptr)

declare ptr @tl_register_parameter(ptr)

declare ptr @tl_string_concat(ptr, ptr)

declare ptr @tl_file_open(ptr, ptr)

declare ptr @tl_file_read_string(ptr)

declare void @tl_file_write_string(ptr, ptr)

declare void @tl_file_close(ptr)

declare ptr @tl_path_new(ptr)

declare ptr @tl_path_join(ptr, ptr)

declare i1 @tl_path_exists(ptr)

declare i1 @tl_path_is_dir(ptr)

declare i1 @tl_path_is_file(ptr)

declare ptr @tl_path_to_string(ptr)

declare void @tl_path_free(ptr)

declare i1 @tl_http_download(ptr, ptr)

declare ptr @tl_http_get(ptr)

declare ptr @tl_env_get(ptr)

declare void @tl_env_set(ptr, ptr)

declare float @tl_system_time()

declare void @tl_system_sleep(float)

declare i64 @tl_get_memory_mb()

declare void @tl_mem_enter_scope()

declare void @tl_mem_exit_scope()

declare void @tl_mem_register_struct(ptr)

declare void @tl_mem_register_tensor(ptr)

declare void @tl_mem_unregister(ptr)

declare ptr @tl_pool_acquire(i64)

declare void @tl_pool_release(ptr, i64)

declare void @tl_arena_init(i64)

declare ptr @tl_arena_alloc(i64)

declare void @tl_arena_free()

declare i1 @tl_arena_is_active()

declare void @tl_arena_reset()

declare ptr @tl_tensor_reshape_dims.1(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.2(ptr, ptr)

declare ptr @tl_tensor_randn.3(ptr, i1)

declare ptr @tl_varbuilder_get.4(ptr, i64, ptr)

declare ptr @tl_varbuilder_get_from_tensor.5(ptr, ptr)

declare void @tl_update_all_params.6(float)

declare ptr @tl_varbuilder_grad.7(ptr)

declare void @tl_tensor_backward.8(ptr)

declare ptr @tl_tensor_grad.9(ptr)

declare ptr @tl_tensor_detach.10(ptr, i1)

declare ptr @tl_tensor_softmax.11(ptr, i64)

declare ptr @tl_tensor_cross_entropy.12(ptr, ptr)

declare void @tl_tensor_save.13(ptr, ptr)

declare ptr @tl_tensor_load.14(ptr)

declare void @tl_save_all_params.15(ptr)

declare void @tl_add_parameter.16(ptr, ptr)

declare void @tl_load_all_params.17(ptr)

declare void @tl_tensor_sub_assign.18(ptr, ptr)

declare void @tl_add_parameter.19(ptr, ptr)

declare ptr @tl_register_parameter.20(ptr)

declare ptr @tl_string_concat.21(ptr, ptr)

declare ptr @tl_file_open.22(ptr, ptr)

declare ptr @tl_file_read_string.23(ptr)

declare void @tl_file_write_string.24(ptr, ptr)

declare void @tl_file_close.25(ptr)

declare ptr @tl_path_new.26(ptr)

declare ptr @tl_path_join.27(ptr, ptr)

declare i1 @tl_path_exists.28(ptr)

declare i1 @tl_path_is_dir.29(ptr)

declare i1 @tl_path_is_file.30(ptr)

declare ptr @tl_path_to_string.31(ptr)

declare void @tl_path_free.32(ptr)

declare i1 @tl_http_download.33(ptr, ptr)

declare ptr @tl_http_get.34(ptr)

declare ptr @tl_env_get.35(ptr)

declare void @tl_env_set.36(ptr, ptr)

declare float @tl_system_time.37()

declare void @tl_system_sleep.38(float)

declare i64 @tl_get_memory_mb.39()

declare void @tl_mem_enter_scope.40()

declare void @tl_mem_exit_scope.41()

declare void @tl_mem_register_struct.42(ptr)

declare void @tl_mem_register_tensor.43(ptr)

declare void @tl_mem_unregister.44(ptr)

declare ptr @tl_pool_acquire.45(i64)

declare void @tl_pool_release.46(ptr, i64)

declare void @tl_arena_init.47(i64)

declare ptr @tl_arena_alloc.48(i64)

declare ptr @tl_arena_malloc(i64)

declare i1 @tl_arena_is_active.49()

declare void @tl_arena_free.50()

declare ptr @tl_alloc_tmp(i64)

declare void @tl_free_tmp(ptr)

declare ptr @tl_tensor_reshape_dims.51(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.52(ptr, ptr)

declare ptr @tl_tensor_randn.53(ptr, i1)

declare ptr @tl_varbuilder_get.54(ptr, i64, ptr)

declare ptr @tl_varbuilder_get_from_tensor.55(ptr, ptr)

declare void @tl_update_all_params.56(float)

declare ptr @tl_varbuilder_grad.57(ptr)

declare void @tl_tensor_backward.58(ptr)

declare ptr @tl_tensor_grad.59(ptr)

declare ptr @tl_tensor_detach.60(ptr, i1)

declare ptr @tl_tensor_softmax.61(ptr, i64)

declare ptr @tl_tensor_cross_entropy.62(ptr, ptr)

declare void @tl_tensor_save.63(ptr, ptr)

declare ptr @tl_tensor_load.64(ptr)

declare void @tl_save_all_params.65(ptr)

declare void @tl_add_parameter.66(ptr, ptr)

declare void @tl_load_all_params.67(ptr)

declare void @tl_tensor_sub_assign.68(ptr, ptr)

declare void @tl_add_parameter.69(ptr, ptr)

declare ptr @tl_register_parameter.70(ptr)

declare ptr @tl_string_concat.71(ptr, ptr)

declare ptr @tl_file_open.72(ptr, ptr)

declare ptr @tl_file_read_string.73(ptr)

declare void @tl_file_write_string.74(ptr, ptr)

declare void @tl_file_close.75(ptr)

declare ptr @tl_path_new.76(ptr)

declare ptr @tl_path_join.77(ptr, ptr)

declare i1 @tl_path_exists.78(ptr)

declare i1 @tl_path_is_dir.79(ptr)

declare i1 @tl_path_is_file.80(ptr)

declare ptr @tl_path_to_string.81(ptr)

declare void @tl_path_free.82(ptr)

declare i1 @tl_http_download.83(ptr, ptr)

declare ptr @tl_http_get.84(ptr)

declare ptr @tl_env_get.85(ptr)

declare void @tl_env_set.86(ptr, ptr)

declare float @tl_system_time.87()

declare void @tl_system_sleep.88(float)

declare i64 @tl_get_memory_mb.89()

declare void @tl_mem_enter_scope.90()

declare void @tl_mem_exit_scope.91()

declare void @tl_mem_register_struct.92(ptr)

declare void @tl_mem_register_tensor.93(ptr)

declare void @tl_mem_unregister.94(ptr)

declare ptr @tl_pool_acquire.95(i64)

declare void @tl_pool_release.96(ptr, i64)

declare void @tl_arena_init.97(i64)

declare ptr @tl_arena_alloc.98(i64)

declare ptr @tl_arena_malloc.99(i64)

declare i1 @tl_arena_is_active.100()

declare void @tl_arena_free.101()

declare ptr @tl_alloc_tmp.102(i64)

declare void @tl_free_tmp.103(ptr)
