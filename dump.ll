; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

@str_literal = private unnamed_addr constant [29 x i8] c"Starting memory leak test...\00", align 1
@str_literal.106 = private unnamed_addr constant [21 x i8] c"Initial memory (MB):\00", align 1
@str_literal.107 = private unnamed_addr constant [6 x i8] c"Iter:\00", align 1
@str_literal.108 = private unnamed_addr constant [10 x i8] c"Mem (MB):\00", align 1
@str_literal.109 = private unnamed_addr constant [16 x i8] c"Final Mem (MB):\00", align 1
@str_literal.110 = private unnamed_addr constant [6 x i8] c"Done.\00", align 1

declare void @tl_print_i64(i64)

declare void @tl_print_f32(float)

declare void @tl_print_string(ptr)

declare ptr @malloc(i64)

declare ptr @calloc(i64, i64)

declare void @free(ptr)

declare i64 @tl_tensor_dim(ptr, i64)

declare float @tl_tensor_get_f32_md(ptr, ptr, i64)

declare ptr @tl_tensor_new(ptr, i64, ptr)

declare ptr @tl_tensor_from_i64_array(ptr, i64)

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

declare i64 @tl_arena_alloc.48(i64)

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

declare i64 @tl_arena_alloc.98(i64)

declare ptr @tl_arena_malloc.99(i64)

declare i1 @tl_arena_is_active.100()

declare void @tl_arena_free.101()

declare ptr @tl_alloc_tmp.102(i64)

declare void @tl_free_tmp.103(ptr)

declare i64 @tl_get_memory_mb.104()

declare void @tl_print_i64.105(i64)

define void @main() {
entry:
  %end = alloca i64, align 16
  %cur = alloca i64, align 16
  %loss = alloca ptr, align 16
  %w = alloca ptr, align 16
  %z = alloca ptr, align 16
  %y = alloca ptr, align 16
  %conv_buf10 = alloca [2 x float], align 4
  %x = alloca ptr, align 16
  %conv_buf = alloca [2 x float], align 4
  %i = alloca i64, align 16
  %start_mem = alloca i64, align 16
  call void @tl_mem_enter_scope()
  call void @tl_arena_init(i64 409600)
  call void @tl_print_string(ptr @str_literal)
  %call_tmp = call i64 @tl_get_memory_mb()
  store i64 %call_tmp, ptr %start_mem, align 8
  call void @tl_print_string(ptr @str_literal.106)
  %start_mem1 = load i64, ptr %start_mem, align 8
  call void @tl_print_i64(i64 %start_mem1)
  br label %for_header

for_header:                                       ; preds = %merge, %entry
  %for_idx = phi i64 [ %next_idx, %merge ], [ 0, %entry ]
  %for_cond = icmp slt i64 %for_idx, 5000
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %i, align 8
  %arr_malloc = call ptr @malloc(i64 mul (i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), i64 2))
  call void @tl_mem_register_struct(ptr %arr_malloc)
  %elem_ptr = getelementptr inbounds i64, ptr %arr_malloc, i64 0
  store i64 128, ptr %elem_ptr, align 8
  %elem_ptr2 = getelementptr inbounds i64, ptr %arr_malloc, i64 1
  store i64 128, ptr %elem_ptr2, align 8
  %src = getelementptr inbounds i64, ptr %arr_malloc, i64 0
  %l = load i64, ptr %src, align 8
  %c = sitofp i64 %l to float
  %dst = getelementptr inbounds float, ptr %conv_buf, i64 0
  store float %c, ptr %dst, align 4
  %src3 = getelementptr inbounds i64, ptr %arr_malloc, i64 1
  %l4 = load i64, ptr %src3, align 8
  %c5 = sitofp i64 %l4 to float
  %dst6 = getelementptr inbounds float, ptr %conv_buf, i64 1
  store float %c5, ptr %dst6, align 4
  %shape_arr = alloca [1 x i64], align 8
  %shape_ptr = getelementptr inbounds [1 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 2, ptr %shape_ptr, align 8
  %converted_tensor = call ptr @tl_tensor_new(ptr %conv_buf, i64 1, ptr %shape_arr)
  %static_call = call ptr @tl_tensor_randn(ptr %converted_tensor, i1 true)
  call void @tl_mem_unregister(ptr %static_call)
  store ptr %static_call, ptr %x, align 8
  %arr_malloc7 = call ptr @malloc(i64 mul (i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), i64 2))
  call void @tl_mem_register_struct(ptr %arr_malloc7)
  %elem_ptr8 = getelementptr inbounds i64, ptr %arr_malloc7, i64 0
  store i64 128, ptr %elem_ptr8, align 8
  %elem_ptr9 = getelementptr inbounds i64, ptr %arr_malloc7, i64 1
  store i64 128, ptr %elem_ptr9, align 8
  %src11 = getelementptr inbounds i64, ptr %arr_malloc7, i64 0
  %l12 = load i64, ptr %src11, align 8
  %c13 = sitofp i64 %l12 to float
  %dst14 = getelementptr inbounds float, ptr %conv_buf10, i64 0
  store float %c13, ptr %dst14, align 4
  %src15 = getelementptr inbounds i64, ptr %arr_malloc7, i64 1
  %l16 = load i64, ptr %src15, align 8
  %c17 = sitofp i64 %l16 to float
  %dst18 = getelementptr inbounds float, ptr %conv_buf10, i64 1
  store float %c17, ptr %dst18, align 4
  %shape_arr19 = alloca [1 x i64], align 8
  %shape_ptr20 = getelementptr inbounds [1 x i64], ptr %shape_arr19, i64 0, i64 0
  store i64 2, ptr %shape_ptr20, align 8
  %converted_tensor21 = call ptr @tl_tensor_new(ptr %conv_buf10, i64 1, ptr %shape_arr19)
  %static_call22 = call ptr @tl_tensor_randn(ptr %converted_tensor21, i1 true)
  call void @tl_mem_unregister(ptr %static_call22)
  store ptr %static_call22, ptr %y, align 8
  %x23 = load ptr, ptr %x, align 8
  %y24 = load ptr, ptr %y, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %x23, ptr %y24)
  call void @tl_mem_unregister(ptr %binop_res)
  store ptr %binop_res, ptr %z, align 8
  %z25 = load ptr, ptr %z, align 8
  %x26 = load ptr, ptr %x, align 8
  %binop_res27 = call ptr @tl_tensor_mul(ptr %z25, ptr %x26)
  call void @tl_mem_unregister(ptr %binop_res27)
  store ptr %binop_res27, ptr %w, align 8
  %w28 = load ptr, ptr %w, align 8
  %sum_res = call ptr @tl_tensor_sum(ptr %w28)
  call void @tl_mem_unregister(ptr %sum_res)
  store ptr %sum_res, ptr %loss, align 8
  %loss29 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss29)
  %i30 = load i64, ptr %i, align 8
  %i31 = load i64, ptr %i, align 8
  %divtmp = sdiv i64 %i31, 250
  %multmp = mul i64 %divtmp, 250
  %subtmp = sub i64 %i30, %multmp
  %eqtmp = icmp eq i64 %subtmp, 0
  br i1 %eqtmp, label %then, label %else

for_end:                                          ; preds = %for_header
  %call_tmp39 = call i64 @tl_get_memory_mb()
  store i64 %call_tmp39, ptr %end, align 8
  call void @tl_print_string(ptr @str_literal.109)
  %end40 = load i64, ptr %end, align 8
  call void @tl_print_i64(i64 %end40)
  call void @tl_print_string(ptr @str_literal.110)
  call void @tl_mem_exit_scope()
  ret void

then:                                             ; preds = %for_body
  call void @tl_mem_enter_scope()
  %call_tmp32 = call i64 @tl_get_memory_mb()
  store i64 %call_tmp32, ptr %cur, align 8
  call void @tl_print_string(ptr @str_literal.107)
  %i33 = load i64, ptr %i, align 8
  call void @tl_print_i64(i64 %i33)
  call void @tl_print_string(ptr @str_literal.108)
  %cur34 = load i64, ptr %cur, align 8
  call void @tl_print_i64(i64 %cur34)
  call void @tl_mem_exit_scope()
  br label %merge

else:                                             ; preds = %for_body
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else, %then
  %tensor_to_free = load ptr, ptr %z, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free35 = load ptr, ptr %loss, align 8
  call void @tl_tensor_free(ptr %tensor_to_free35)
  %tensor_to_free36 = load ptr, ptr %w, align 8
  call void @tl_tensor_free(ptr %tensor_to_free36)
  %tensor_to_free37 = load ptr, ptr %x, align 8
  call void @tl_tensor_free(ptr %tensor_to_free37)
  %tensor_to_free38 = load ptr, ptr %y, align 8
  call void @tl_tensor_free(ptr %tensor_to_free38)
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx, 1
  br label %for_header
}
