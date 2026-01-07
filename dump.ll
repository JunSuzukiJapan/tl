; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

@str_literal = private unnamed_addr constant [12 x i8] c"Sub result:\00", align 1
@str_literal.104 = private unnamed_addr constant [12 x i8] c"Div result:\00", align 1
@str_literal.105 = private unnamed_addr constant [10 x i8] c"Combined:\00", align 1
@str_literal.106 = private unnamed_addr constant [6 x i8] c"Done.\00", align 1

declare void @tl_print_i64(i64)

declare void @tl_print_f32(float)

declare void @tl_print_string(ptr)

declare void @tl_print_ptr(ptr)

declare ptr @malloc(i64)

declare ptr @calloc(i64, i64)

declare void @free(ptr)

declare i64 @tl_tensor_dim(ptr, i64)

declare float @tl_tensor_get_f32_md(ptr, ptr, i64)

declare ptr @tl_tensor_set_f32_md(ptr, ptr, i64, float)

declare ptr @tl_tensor_new(ptr, i64, ptr)

declare ptr @tl_tensor_from_i64_array(ptr, i64)

declare ptr @tl_tensor_sub(ptr, ptr)

declare void @tl_tensor_free(ptr)

declare ptr @tl_tensor_clone(ptr)

declare ptr @tl_tensor_add(ptr, ptr)

declare ptr @tl_tensor_mul(ptr, ptr)

declare void @tl_tensor_print(ptr)

declare void @tl_tensor_print_1(ptr)

declare void @tl_tensor_print_2(ptr)

declare void @tl_tensor_print_3(ptr)

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

declare void @tl_clear_grads()

declare ptr @tl_checkpoint(ptr, ptr, ptr)

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

declare ptr @tl_tensor_randn_debug(i64, ptr, i1)

declare ptr @tl_varbuilder_get(ptr, i64, ptr)

declare ptr @tl_varbuilder_get_from_tensor(ptr, ptr)

declare void @tl_update_all_params(float)

declare ptr @tl_varbuilder_grad(ptr)

declare void @tl_tensor_backward(ptr)

declare ptr @tl_tensor_grad(ptr)

declare ptr @tl_tensor_detach(ptr, i1)

declare ptr @tl_tensor_contiguous(ptr)

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

declare i64 @tl_arena_get_offset()

declare i64 @tl_arena_get_capacity()

declare ptr @tl_tensor_reshape_dims.1(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.2(ptr, ptr)

declare ptr @tl_varbuilder_get.3(ptr, i64, ptr)

declare ptr @tl_varbuilder_get_from_tensor.4(ptr, ptr)

declare void @tl_update_all_params.5(float)

declare ptr @tl_varbuilder_grad.6(ptr)

declare void @tl_tensor_backward.7(ptr)

declare ptr @tl_tensor_grad.8(ptr)

declare ptr @tl_tensor_detach.9(ptr, i1)

declare ptr @tl_tensor_softmax.10(ptr, i64)

declare ptr @tl_tensor_cross_entropy.11(ptr, ptr)

declare void @tl_tensor_save.12(ptr, ptr)

declare ptr @tl_tensor_load.13(ptr)

declare ptr @tl_tensor_to_f32(ptr)

declare ptr @tl_tensor_to_i64(ptr)

declare void @tl_save_all_params.14(ptr)

declare void @tl_add_parameter.15(ptr, ptr)

declare ptr @tl_tensor_argmax(ptr, i64, i1)

declare i64 @tl_tensor_item_i64(ptr)

declare void @tl_load_all_params.16(ptr)

declare void @tl_tensor_sub_assign.17(ptr, ptr)

declare void @tl_add_parameter.18(ptr, ptr)

declare ptr @tl_register_parameter.19(ptr)

declare ptr @tl_string_concat.20(ptr, ptr)

declare ptr @tl_file_open.21(ptr, ptr)

declare ptr @tl_file_read_string.22(ptr)

declare void @tl_file_write_string.23(ptr, ptr)

declare void @tl_file_close.24(ptr)

declare ptr @tl_path_new.25(ptr)

declare ptr @tl_path_join.26(ptr, ptr)

declare i1 @tl_path_exists.27(ptr)

declare i1 @tl_path_is_dir.28(ptr)

declare i1 @tl_path_is_file.29(ptr)

declare ptr @tl_path_to_string.30(ptr)

declare void @tl_path_free.31(ptr)

declare i1 @tl_http_download.32(ptr, ptr)

declare ptr @tl_http_get.33(ptr)

declare ptr @tl_env_get.34(ptr)

declare void @tl_env_set.35(ptr, ptr)

declare float @tl_system_time.36()

declare void @tl_system_sleep.37(float)

declare i64 @tl_get_memory_mb.38()

declare void @tl_mem_enter_scope.39()

declare void @tl_mem_exit_scope.40()

declare void @tl_mem_register_struct.41(ptr)

declare void @tl_mem_register_tensor.42(ptr)

declare void @tl_mem_unregister.43(ptr)

declare ptr @tl_pool_acquire.44(i64)

declare void @tl_pool_release.45(ptr, i64)

declare void @tl_arena_init.46(i64)

declare i64 @tl_arena_alloc.47(i64)

declare ptr @tl_arena_malloc(i64)

declare i1 @tl_arena_is_active.48()

declare void @tl_arena_free.49()

declare ptr @tl_alloc_tmp(i64)

declare void @tl_free_tmp(ptr)

declare ptr @tl_tensor_reshape_dims.50(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.51(ptr, ptr)

declare ptr @tl_varbuilder_get.52(ptr, i64, ptr)

declare ptr @tl_varbuilder_get_from_tensor.53(ptr, ptr)

declare void @tl_update_all_params.54(float)

declare ptr @tl_varbuilder_grad.55(ptr)

declare void @tl_tensor_backward.56(ptr)

declare ptr @tl_tensor_grad.57(ptr)

declare ptr @tl_tensor_detach.58(ptr, i1)

declare ptr @tl_tensor_softmax.59(ptr, i64)

declare ptr @tl_tensor_cross_entropy.60(ptr, ptr)

declare void @tl_tensor_save.61(ptr, ptr)

declare ptr @tl_tensor_load.62(ptr)

declare void @tl_save_all_params.63(ptr)

declare void @tl_add_parameter.64(ptr, ptr)

declare ptr @tl_tensor_argmax.65(ptr, i64, i1)

declare i64 @tl_tensor_item_i64.66(ptr)

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

declare i64 @tl_arena_alloc.98(i64)

declare ptr @tl_arena_malloc.99(i64)

declare i1 @tl_arena_is_active.100()

declare void @tl_arena_free.101()

declare ptr @tl_alloc_tmp.102(i64)

declare void @tl_free_tmp.103(ptr)

define void @main() {
entry:
  %combined = alloca ptr, align 16
  %scalar_shape_rhs28 = alloca i64, align 16
  %scalar_data_rhs27 = alloca float, align 16
  %scalar_shape_rhs24 = alloca i64, align 16
  %scalar_data_rhs23 = alloca float, align 16
  %half = alloca ptr, align 16
  %scalar_shape_rhs18 = alloca i64, align 16
  %scalar_data_rhs17 = alloca float, align 16
  %neg = alloca ptr, align 16
  %scalar_shape_rhs12 = alloca i64, align 16
  %scalar_data_rhs11 = alloca float, align 16
  %t_d = alloca ptr, align 16
  %indices_arr5 = alloca [1 x i64], align 8
  %indices_arr1 = alloca [1 x i64], align 8
  %indices_arr = alloca [1 x i64], align 8
  %t = alloca ptr, align 16
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %shape_arr = alloca [1 x i64], align 8
  call void @tl_mem_enter_scope()
  call void @tl_arena_init(i64 463872)
  %tmp = getelementptr inbounds [1 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 3, ptr %tmp, align 8
  %randn_res = call ptr @tl_tensor_randn_debug(i64 1, ptr %shape_arr, i1 false)
  call void @tl_mem_register_tensor(ptr %randn_res)
  store float 0.000000e+00, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor_rhs)
  call void @tl_mem_register_tensor(ptr %binop_res)
  call void @tl_mem_unregister(ptr %binop_res)
  store ptr %binop_res, ptr %t, align 8
  %idx_ptr = getelementptr inbounds [1 x i64], ptr %indices_arr, i64 0, i64 0
  store i64 0, ptr %idx_ptr, align 8
  %curr_t = load ptr, ptr %t, align 8
  %new_t = call ptr @tl_tensor_set_f32_md(ptr %curr_t, ptr %indices_arr, i64 1, float 1.000000e+00)
  call void @tl_tensor_free(ptr %curr_t)
  store ptr %new_t, ptr %t, align 8
  call void @tl_mem_register_tensor(ptr %new_t)
  %idx_ptr2 = getelementptr inbounds [1 x i64], ptr %indices_arr1, i64 0, i64 0
  store i64 1, ptr %idx_ptr2, align 8
  %curr_t3 = load ptr, ptr %t, align 8
  %new_t4 = call ptr @tl_tensor_set_f32_md(ptr %curr_t3, ptr %indices_arr1, i64 1, float 2.000000e+00)
  call void @tl_tensor_free(ptr %curr_t3)
  store ptr %new_t4, ptr %t, align 8
  call void @tl_mem_register_tensor(ptr %new_t4)
  %idx_ptr6 = getelementptr inbounds [1 x i64], ptr %indices_arr5, i64 0, i64 0
  store i64 2, ptr %idx_ptr6, align 8
  %curr_t7 = load ptr, ptr %t, align 8
  %new_t8 = call ptr @tl_tensor_set_f32_md(ptr %curr_t7, ptr %indices_arr5, i64 1, float 3.000000e+00)
  call void @tl_tensor_free(ptr %curr_t7)
  store ptr %new_t8, ptr %t, align 8
  call void @tl_mem_register_tensor(ptr %new_t8)
  %t9 = load ptr, ptr %t, align 8
  %detach_res = call ptr @tl_tensor_detach(ptr %t9, i1 true)
  call void @tl_mem_register_tensor(ptr %detach_res)
  call void @tl_mem_unregister(ptr %detach_res)
  store ptr %detach_res, ptr %t_d, align 8
  %t_d10 = load ptr, ptr %t_d, align 8
  store float 1.000000e+00, ptr %scalar_data_rhs11, align 4
  %scalar_tensor_rhs13 = call ptr @tl_tensor_new(ptr %scalar_data_rhs11, i64 0, ptr %scalar_shape_rhs12)
  %binop_res14 = call ptr @tl_tensor_sub(ptr %t_d10, ptr %scalar_tensor_rhs13)
  call void @tl_mem_register_tensor(ptr %binop_res14)
  call void @tl_mem_unregister(ptr %binop_res14)
  store ptr %binop_res14, ptr %neg, align 8
  call void @tl_print_string(ptr @str_literal)
  %neg15 = load ptr, ptr %neg, align 8
  call void @tl_tensor_print(ptr %neg15)
  %t_d16 = load ptr, ptr %t_d, align 8
  store float 2.000000e+00, ptr %scalar_data_rhs17, align 4
  %scalar_tensor_rhs19 = call ptr @tl_tensor_new(ptr %scalar_data_rhs17, i64 0, ptr %scalar_shape_rhs18)
  %binop_res20 = call ptr @tl_tensor_div(ptr %t_d16, ptr %scalar_tensor_rhs19)
  call void @tl_mem_register_tensor(ptr %binop_res20)
  call void @tl_mem_unregister(ptr %binop_res20)
  store ptr %binop_res20, ptr %half, align 8
  call void @tl_print_string(ptr @str_literal.104)
  %half21 = load ptr, ptr %half, align 8
  call void @tl_tensor_print(ptr %half21)
  %t_d22 = load ptr, ptr %t_d, align 8
  store float 1.000000e+00, ptr %scalar_data_rhs23, align 4
  %scalar_tensor_rhs25 = call ptr @tl_tensor_new(ptr %scalar_data_rhs23, i64 0, ptr %scalar_shape_rhs24)
  %binop_res26 = call ptr @tl_tensor_sub(ptr %t_d22, ptr %scalar_tensor_rhs25)
  call void @tl_mem_register_tensor(ptr %binop_res26)
  store float 2.000000e+00, ptr %scalar_data_rhs27, align 4
  %scalar_tensor_rhs29 = call ptr @tl_tensor_new(ptr %scalar_data_rhs27, i64 0, ptr %scalar_shape_rhs28)
  %binop_res30 = call ptr @tl_tensor_div(ptr %binop_res26, ptr %scalar_tensor_rhs29)
  call void @tl_mem_register_tensor(ptr %binop_res30)
  call void @tl_mem_unregister(ptr %binop_res30)
  store ptr %binop_res30, ptr %combined, align 8
  call void @tl_print_string(ptr @str_literal.105)
  %combined31 = load ptr, ptr %combined, align 8
  call void @tl_tensor_print(ptr %combined31)
  call void @tl_print_string(ptr @str_literal.106)
  %tensor_to_free = load ptr, ptr %combined, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free32 = load ptr, ptr %t_d, align 8
  call void @tl_tensor_free(ptr %tensor_to_free32)
  %tensor_to_free33 = load ptr, ptr %neg, align 8
  call void @tl_tensor_free(ptr %tensor_to_free33)
  %tensor_to_free34 = load ptr, ptr %t, align 8
  call void @tl_tensor_free(ptr %tensor_to_free34)
  %tensor_to_free35 = load ptr, ptr %half, align 8
  call void @tl_tensor_free(ptr %tensor_to_free35)
  call void @tl_mem_exit_scope()
  ret void
}
