; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@str_literal = private unnamed_addr constant [21 x i8] c"MatMul result shape:\00", align 1

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

define void @main() {
entry:
  %C = alloca ptr, align 8
  %idx_arr32 = alloca [2 x i64], align 8
  %idx_arr = alloca [2 x i64], align 8
  %k = alloca i64, align 8
  %j = alloca i64, align 8
  %i16 = alloca i64, align 8
  %B = alloca ptr, align 8
  %shape_arr6 = alloca [2 x i64], align 8
  %A = alloca ptr, align 8
  %shape_arr = alloca [2 x i64], align 8
  %N = alloca i64, align 8
  %K = alloca i64, align 8
  %M = alloca i64, align 8
  call void @tl_mem_enter_scope()
  store i64 16, ptr %M, align 8
  store i64 16, ptr %K, align 8
  store i64 16, ptr %N, align 8
  %M1 = load i64, ptr %M, align 8
  %K2 = load i64, ptr %K, align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %M1, ptr %shape_ptr_in, align 8
  %shape_ptr_in3 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 %K2, ptr %shape_ptr_in3, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 false)
  store ptr %randn_res, ptr %A, align 8
  %K4 = load i64, ptr %K, align 8
  %N5 = load i64, ptr %N, align 8
  %shape_ptr_in7 = getelementptr inbounds [2 x i64], ptr %shape_arr6, i64 0, i64 0
  store i64 %K4, ptr %shape_ptr_in7, align 8
  %shape_ptr_in8 = getelementptr inbounds [2 x i64], ptr %shape_arr6, i64 0, i64 1
  store i64 %N5, ptr %shape_ptr_in8, align 8
  %randn_res9 = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr6, i1 false)
  store ptr %randn_res9, ptr %B, align 8
  %A10 = load ptr, ptr %A, align 8
  %dim_size = call i64 @tl_tensor_dim(ptr %A10, i64 0)
  %dim_size11 = call i64 @tl_tensor_dim(ptr %A10, i64 1)
  %B12 = load ptr, ptr %B, align 8
  %dim_size13 = call i64 @tl_tensor_dim(ptr %B12, i64 1)
  %sz_acc = mul i64 1, %dim_size
  %sz_acc14 = mul i64 %sz_acc, %dim_size11
  %sz_acc15 = mul i64 %sz_acc14, %dim_size13
  %buf_void = call ptr @calloc(i64 %sz_acc15, i64 4)
  call void @tl_mem_enter_scope()
  br label %loop_cond

eq_after:                                         ; preds = %loop_aft
  %shape = alloca [3 x i64], align 8
  %shape_ptr = getelementptr [3 x i64], ptr %shape, i64 0, i64 0
  store i64 %dim_size, ptr %shape_ptr, align 8
  %shape_ptr51 = getelementptr [3 x i64], ptr %shape, i64 0, i64 1
  store i64 %dim_size11, ptr %shape_ptr51, align 8
  %shape_ptr52 = getelementptr [3 x i64], ptr %shape, i64 0, i64 2
  store i64 %dim_size13, ptr %shape_ptr52, align 8
  %t = call ptr @tl_tensor_new(ptr %buf_void, i64 3, ptr %shape)
  store ptr %t, ptr %C, align 8
  call void @tl_print_string(ptr @str_literal)
  %C53 = load ptr, ptr %C, align 8
  call void @tl_tensor_print(ptr %C53)
  call void @tl_mem_exit_scope()
  ret void

loop_cond:                                        ; preds = %loop_aft19, %entry
  %i = phi i64 [ 0, %entry ], [ %next50, %loop_aft19 ]
  %cmp = icmp slt i64 %i, %dim_size
  br i1 %cmp, label %loop_body, label %loop_aft

loop_body:                                        ; preds = %loop_cond
  store i64 %i, ptr %i16, align 8
  br label %loop_cond17

loop_aft:                                         ; preds = %loop_cond
  br label %eq_after

loop_cond17:                                      ; preds = %loop_aft24, %loop_body
  %i20 = phi i64 [ 0, %loop_body ], [ %next48, %loop_aft24 ]
  %cmp21 = icmp slt i64 %i20, %dim_size11
  br i1 %cmp21, label %loop_body18, label %loop_aft19

loop_body18:                                      ; preds = %loop_cond17
  store i64 %i20, ptr %j, align 8
  br label %loop_cond22

loop_aft19:                                       ; preds = %loop_cond17
  %iv49 = load i64, ptr %i16, align 8
  %next50 = add i64 %iv49, 1
  br label %loop_cond

loop_cond22:                                      ; preds = %loop_body23, %loop_body18
  %i25 = phi i64 [ 0, %loop_body18 ], [ %next, %loop_body23 ]
  %cmp26 = icmp slt i64 %i25, %dim_size13
  br i1 %cmp26, label %loop_body23, label %loop_aft24

loop_body23:                                      ; preds = %loop_cond22
  store i64 %i25, ptr %k, align 8
  %A27 = load ptr, ptr %A, align 8
  %i28 = load i64, ptr %i16, align 8
  %idx_ptr = getelementptr [2 x i64], ptr %idx_arr, i64 0, i64 0
  store i64 %i28, ptr %idx_ptr, align 8
  %j29 = load i64, ptr %j, align 8
  %idx_ptr30 = getelementptr [2 x i64], ptr %idx_arr, i64 0, i64 1
  store i64 %j29, ptr %idx_ptr30, align 8
  %get_md_call = call float @tl_tensor_get_f32_md(ptr %A27, ptr %idx_arr, i64 2)
  %B31 = load ptr, ptr %B, align 8
  %j33 = load i64, ptr %j, align 8
  %idx_ptr34 = getelementptr [2 x i64], ptr %idx_arr32, i64 0, i64 0
  store i64 %j33, ptr %idx_ptr34, align 8
  %k35 = load i64, ptr %k, align 8
  %idx_ptr36 = getelementptr [2 x i64], ptr %idx_arr32, i64 0, i64 1
  store i64 %k35, ptr %idx_ptr36, align 8
  %get_md_call37 = call float @tl_tensor_get_f32_md(ptr %B31, ptr %idx_arr32, i64 2)
  %fmultmp = fmul float %get_md_call, %get_md_call37
  %iv = load i64, ptr %k, align 8
  %term = mul i64 %iv, 1
  %off = add i64 0, %term
  %str = mul i64 1, %dim_size13
  %iv38 = load i64, ptr %j, align 8
  %term39 = mul i64 %iv38, %str
  %off40 = add i64 %off, %term39
  %str41 = mul i64 %str, %dim_size11
  %iv42 = load i64, ptr %i16, align 8
  %term43 = mul i64 %iv42, %str41
  %off44 = add i64 %off40, %term43
  %str45 = mul i64 %str41, %dim_size
  %ptr = getelementptr float, ptr %buf_void, i64 %off44
  %cur = load float, ptr %ptr, align 4
  %new = fadd float %cur, %fmultmp
  store float %new, ptr %ptr, align 4
  %iv46 = load i64, ptr %k, align 8
  %next = add i64 %iv46, 1
  br label %loop_cond22

loop_aft24:                                       ; preds = %loop_cond22
  %iv47 = load i64, ptr %j, align 8
  %next48 = add i64 %iv47, 1
  br label %loop_cond17
}
