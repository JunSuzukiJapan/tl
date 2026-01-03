; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@str_literal = private unnamed_addr constant [27 x i8] c"Test: VarBuilder functions\00", align 1
@str_literal.103 = private unnamed_addr constant [12 x i8] c"test_weight\00", align 1
@str_literal.104 = private unnamed_addr constant [10 x i8] c"test_bias\00", align 1
@str_literal.105 = private unnamed_addr constant [19 x i8] c"Created parameters\00", align 1
@str_literal.106 = private unnamed_addr constant [6 x i8] c"Loss:\00", align 1
@str_literal.107 = private unnamed_addr constant [12 x i8] c"test_weight\00", align 1
@str_literal.108 = private unnamed_addr constant [10 x i8] c"test_bias\00", align 1
@str_literal.109 = private unnamed_addr constant [17 x i8] c"Weight gradient:\00", align 1
@str_literal.110 = private unnamed_addr constant [15 x i8] c"Bias gradient:\00", align 1
@str_literal.111 = private unnamed_addr constant [20 x i8] c"Parameters updated!\00", align 1

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

declare ptr @tl_tensor_randn(i64, ptr, i1)

declare ptr @tl_varbuilder_get(ptr, i64, ptr)

declare void @tl_update_all_params(float)

declare ptr @tl_varbuilder_grad(ptr)

declare void @tl_tensor_backward(ptr)

declare ptr @tl_tensor_grad(ptr)

declare ptr @tl_tensor_detach(ptr, i1)

declare ptr @tl_tensor_softmax(ptr, i64)

declare ptr @tl_tensor_cross_entropy(ptr, ptr)

declare void @tl_tensor_save.1(ptr, ptr)

declare ptr @tl_tensor_load.2(ptr)

declare void @tl_save_all_params(ptr)

declare void @tl_add_parameter(ptr, ptr)

declare void @tl_load_all_params(ptr)

declare void @tl_tensor_sub_assign.3(ptr, ptr)

declare void @tl_add_parameter.4(ptr, ptr)

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

declare void @tl_print_i64.5(i64)

declare void @tl_print_f32.6(float)

declare void @tl_print_string.7(ptr)

declare ptr @malloc.8(i64)

declare ptr @calloc.9(i64, i64)

declare void @free.10(ptr)

declare i64 @tl_tensor_dim.11(ptr, i64)

declare float @tl_tensor_get_f32_md.12(ptr, ptr, i64)

declare ptr @tl_tensor_new.13(ptr, i64, ptr)

declare ptr @tl_tensor_sub.14(ptr, ptr)

declare void @tl_tensor_free.15(ptr)

declare ptr @tl_tensor_clone.16(ptr)

declare ptr @tl_tensor_add.17(ptr, ptr)

declare ptr @tl_tensor_mul.18(ptr, ptr)

declare void @tl_tensor_print.19(ptr)

declare float @tl_tensor_get.20(ptr, i64)

declare ptr @tl_tensor_slice.21(ptr, i64, i64)

declare i64 @tl_tensor_len.22(ptr)

declare ptr @tl_tensor_neg.23(ptr)

declare ptr @tl_tensor_transpose.24(ptr, i64, i64)

declare ptr @tl_tensor_pow.25(ptr, ptr)

declare ptr @tl_tensor_sqrt.26(ptr)

declare ptr @tl_tensor_sin.27(ptr)

declare ptr @tl_tensor_cos.28(ptr)

declare ptr @tl_tensor_relu.29(ptr)

declare ptr @tl_tensor_gelu.30(ptr)

declare ptr @tl_tensor_tril.31(ptr, i32)

declare ptr @tl_tensor_sum_dim.32(ptr, i64, i1)

declare ptr @tl_tensor_embedding.33(ptr, ptr)

declare ptr @tl_tensor_sum.34(ptr)

declare ptr @tl_tensor_div.35(ptr, ptr)

declare ptr @tl_tensor_matmul.36(ptr, ptr)

declare ptr @tl_tensor_exp.37(ptr)

declare ptr @tl_tensor_log.38(ptr)

declare void @tl_tensor_add_assign.39(ptr, ptr)

declare void @tl_tensor_sub_assign.40(ptr, ptr)

declare void @tl_tensor_mul_assign.41(ptr, ptr)

declare void @tl_tensor_div_assign.42(ptr, ptr)

declare void @tl_register_tensor.43(ptr, ptr)

declare i32 @strcmp.44(ptr, ptr)

declare void @tl_tensor_save.45(ptr, ptr)

declare ptr @tl_tensor_load.46(ptr)

declare ptr @tl_tensor_map_new.47()

declare void @tl_tensor_map_insert.48(ptr, ptr, ptr)

declare void @tl_tensor_map_save.49(ptr, ptr)

declare ptr @tl_tensor_map_load.50(ptr)

declare ptr @tl_tensor_map_get.51(ptr, ptr)

declare void @tl_tensor_map_free.52(ptr)

declare ptr @tl_tensor_reshape_dims.53(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.54(ptr, ptr)

declare ptr @tl_tensor_randn.55(i64, ptr, i1)

declare ptr @tl_varbuilder_get.56(ptr, i64, ptr)

declare void @tl_update_all_params.57(float)

declare ptr @tl_varbuilder_grad.58(ptr)

declare void @tl_tensor_backward.59(ptr)

declare ptr @tl_tensor_grad.60(ptr)

declare ptr @tl_tensor_detach.61(ptr, i1)

declare ptr @tl_tensor_softmax.62(ptr, i64)

declare ptr @tl_tensor_cross_entropy.63(ptr, ptr)

declare void @tl_tensor_save.64(ptr, ptr)

declare ptr @tl_tensor_load.65(ptr)

declare void @tl_save_all_params.66(ptr)

declare void @tl_add_parameter.67(ptr, ptr)

declare void @tl_load_all_params.68(ptr)

declare void @tl_tensor_sub_assign.69(ptr, ptr)

declare void @tl_add_parameter.70(ptr, ptr)

declare ptr @tl_register_parameter.71(ptr)

declare ptr @tl_string_concat.72(ptr, ptr)

declare ptr @tl_file_open.73(ptr, ptr)

declare ptr @tl_file_read_string.74(ptr)

declare void @tl_file_write_string.75(ptr, ptr)

declare void @tl_file_close.76(ptr)

declare ptr @tl_path_new.77(ptr)

declare ptr @tl_path_join.78(ptr, ptr)

declare i1 @tl_path_exists.79(ptr)

declare i1 @tl_path_is_dir.80(ptr)

declare i1 @tl_path_is_file.81(ptr)

declare ptr @tl_path_to_string.82(ptr)

declare void @tl_path_free.83(ptr)

declare i1 @tl_http_download.84(ptr, ptr)

declare ptr @tl_http_get.85(ptr)

declare ptr @tl_env_get.86(ptr)

declare void @tl_env_set.87(ptr, ptr)

declare float @tl_system_time.88()

declare void @tl_system_sleep.89(float)

declare i64 @tl_get_memory_mb.90()

declare void @tl_mem_enter_scope.91()

declare void @tl_mem_exit_scope.92()

declare void @tl_mem_register_struct.93(ptr)

declare void @tl_mem_register_tensor.94(ptr)

declare void @tl_mem_unregister.95(ptr)

declare ptr @tl_pool_acquire.96(i64)

declare void @tl_pool_release.97(ptr, i64)

declare void @tl_arena_init.98(i64)

declare ptr @tl_arena_alloc.99(i64)

declare void @tl_arena_free.100()

declare i1 @tl_arena_is_active.101()

declare void @tl_arena_reset.102()

define void @main() {
entry:
  %gb = alloca ptr, align 16
  %gw = alloca ptr, align 16
  %loss = alloca ptr, align 16
  %output = alloca ptr, align 16
  %x = alloca ptr, align 16
  %shape_arr16 = alloca [2 x i64], align 16
  %bias = alloca ptr, align 16
  %weight = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal)
  %malloc_size = mul i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), 2
  %arr_malloc = call ptr @malloc(i64 %malloc_size)
  call void @tl_mem_register_struct(ptr %arr_malloc)
  %elem_ptr = getelementptr inbounds i64, ptr %arr_malloc, i64 0
  store i64 10, ptr %elem_ptr, align 8
  %elem_ptr1 = getelementptr inbounds i64, ptr %arr_malloc, i64 1
  store i64 5, ptr %elem_ptr1, align 8
  %malloc_size2 = mul i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), 2
  %arr_malloc3 = call ptr @malloc(i64 %malloc_size2)
  call void @tl_mem_register_struct(ptr %arr_malloc3)
  %elem_ptr4 = getelementptr inbounds i64, ptr %arr_malloc3, i64 0
  store i64 10, ptr %elem_ptr4, align 8
  %elem_ptr5 = getelementptr inbounds i64, ptr %arr_malloc3, i64 1
  store i64 5, ptr %elem_ptr5, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shptr = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 10, ptr %shptr, align 8
  %shptr6 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 5, ptr %shptr6, align 8
  %varbuilder_get_result = call ptr @tl_varbuilder_get(ptr @str_literal.103, i64 2, ptr %shape_arr)
  store ptr %varbuilder_get_result, ptr %weight, align 8
  %arr_malloc7 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64))
  call void @tl_mem_register_struct(ptr %arr_malloc7)
  %elem_ptr8 = getelementptr inbounds i64, ptr %arr_malloc7, i64 0
  store i64 5, ptr %elem_ptr8, align 8
  %arr_malloc9 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64))
  call void @tl_mem_register_struct(ptr %arr_malloc9)
  %elem_ptr10 = getelementptr inbounds i64, ptr %arr_malloc9, i64 0
  store i64 5, ptr %elem_ptr10, align 8
  %shape_arr11 = alloca [1 x i64], align 8
  %shptr12 = getelementptr inbounds [1 x i64], ptr %shape_arr11, i64 0, i64 0
  store i64 5, ptr %shptr12, align 8
  %varbuilder_get_result13 = call ptr @tl_varbuilder_get(ptr @str_literal.104, i64 1, ptr %shape_arr11)
  store ptr %varbuilder_get_result13, ptr %bias, align 8
  call void @tl_print_string(ptr @str_literal.105)
  %weight14 = load ptr, ptr %weight, align 8
  call void @tl_tensor_print(ptr %weight14)
  %bias15 = load ptr, ptr %bias, align 8
  call void @tl_tensor_print(ptr %bias15)
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr16, i64 0, i64 0
  store i64 4, ptr %shape_ptr_in, align 8
  %shape_ptr_in17 = getelementptr inbounds [2 x i64], ptr %shape_arr16, i64 0, i64 1
  store i64 10, ptr %shape_ptr_in17, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr16, i1 false)
  store ptr %randn_res, ptr %x, align 8
  %x18 = load ptr, ptr %x, align 8
  %weight19 = load ptr, ptr %weight, align 8
  %matmul_res = call ptr @tl_tensor_matmul(ptr %x18, ptr %weight19)
  %bias20 = load ptr, ptr %bias, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %matmul_res, ptr %bias20)
  store ptr %binop_res, ptr %output, align 8
  %output21 = load ptr, ptr %output, align 8
  %sum_res = call ptr @tl_tensor_sum(ptr %output21)
  store ptr %sum_res, ptr %loss, align 8
  call void @tl_print_string(ptr @str_literal.106)
  %loss22 = load ptr, ptr %loss, align 8
  call void @tl_tensor_print(ptr %loss22)
  %loss23 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss23)
  %varbuilder_grad_result = call ptr @tl_varbuilder_grad(ptr @str_literal.107)
  store ptr %varbuilder_grad_result, ptr %gw, align 8
  %varbuilder_grad_result24 = call ptr @tl_varbuilder_grad(ptr @str_literal.108)
  store ptr %varbuilder_grad_result24, ptr %gb, align 8
  call void @tl_print_string(ptr @str_literal.109)
  %gw25 = load ptr, ptr %gw, align 8
  call void @tl_tensor_print(ptr %gw25)
  call void @tl_print_string(ptr @str_literal.110)
  %gb26 = load ptr, ptr %gb, align 8
  call void @tl_tensor_print(ptr %gb26)
  call void @tl_update_all_params(float 0x3F847AE140000000)
  call void @tl_print_string(ptr @str_literal.111)
  call void @tl_mem_exit_scope()
  ret void
}
