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

declare void @tl_tensor_save(ptr, ptr)

declare ptr @tl_tensor_load(ptr)

declare void @tl_save_all_params(ptr)

declare void @tl_add_parameter(ptr, ptr)

declare void @tl_load_all_params(ptr)

declare void @tl_tensor_sub_assign.1(ptr, ptr)

declare void @tl_add_parameter.2(ptr, ptr)

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

declare void @tl_print_i64.3(i64)

declare void @tl_print_f32.4(float)

declare void @tl_print_string.5(ptr)

declare ptr @malloc.6(i64)

declare ptr @calloc.7(i64, i64)

declare void @free.8(ptr)

declare i64 @tl_tensor_dim.9(ptr, i64)

declare float @tl_tensor_get_f32_md.10(ptr, ptr, i64)

declare ptr @tl_tensor_new.11(ptr, i64, ptr)

declare ptr @tl_tensor_sub.12(ptr, ptr)

declare void @tl_tensor_free.13(ptr)

declare ptr @tl_tensor_clone.14(ptr)

declare ptr @tl_tensor_add.15(ptr, ptr)

declare ptr @tl_tensor_mul.16(ptr, ptr)

declare void @tl_tensor_print.17(ptr)

declare float @tl_tensor_get.18(ptr, i64)

declare ptr @tl_tensor_slice.19(ptr, i64, i64)

declare i64 @tl_tensor_len.20(ptr)

declare ptr @tl_tensor_neg.21(ptr)

declare ptr @tl_tensor_transpose.22(ptr, i64, i64)

declare ptr @tl_tensor_pow.23(ptr, ptr)

declare ptr @tl_tensor_sqrt.24(ptr)

declare ptr @tl_tensor_sin.25(ptr)

declare ptr @tl_tensor_cos.26(ptr)

declare ptr @tl_tensor_relu.27(ptr)

declare ptr @tl_tensor_gelu.28(ptr)

declare ptr @tl_tensor_tril.29(ptr, i32)

declare ptr @tl_tensor_sum_dim.30(ptr, i64, i1)

declare ptr @tl_tensor_embedding.31(ptr, ptr)

declare ptr @tl_tensor_sum.32(ptr)

declare ptr @tl_tensor_div.33(ptr, ptr)

declare ptr @tl_tensor_matmul.34(ptr, ptr)

declare ptr @tl_tensor_exp.35(ptr)

declare ptr @tl_tensor_log.36(ptr)

declare void @tl_tensor_add_assign.37(ptr, ptr)

declare void @tl_tensor_sub_assign.38(ptr, ptr)

declare void @tl_tensor_mul_assign.39(ptr, ptr)

declare void @tl_tensor_div_assign.40(ptr, ptr)

declare void @tl_register_tensor.41(ptr, ptr)

declare i32 @strcmp.42(ptr, ptr)

declare ptr @tl_tensor_reshape_dims.43(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.44(ptr, ptr)

declare ptr @tl_tensor_randn.45(i64, ptr, i1)

declare ptr @tl_varbuilder_get.46(ptr, i64, ptr)

declare void @tl_update_all_params.47(float)

declare ptr @tl_varbuilder_grad.48(ptr)

declare void @tl_tensor_backward.49(ptr)

declare ptr @tl_tensor_grad.50(ptr)

declare ptr @tl_tensor_detach.51(ptr, i1)

declare ptr @tl_tensor_softmax.52(ptr, i64)

declare ptr @tl_tensor_cross_entropy.53(ptr, ptr)

declare void @tl_tensor_save.54(ptr, ptr)

declare ptr @tl_tensor_load.55(ptr)

declare void @tl_save_all_params.56(ptr)

declare void @tl_add_parameter.57(ptr, ptr)

declare void @tl_load_all_params.58(ptr)

declare void @tl_tensor_sub_assign.59(ptr, ptr)

declare void @tl_add_parameter.60(ptr, ptr)

declare ptr @tl_register_parameter.61(ptr)

declare ptr @tl_string_concat.62(ptr, ptr)

declare ptr @tl_file_open.63(ptr, ptr)

declare ptr @tl_file_read_string.64(ptr)

declare void @tl_file_write_string.65(ptr, ptr)

declare void @tl_file_close.66(ptr)

declare ptr @tl_path_new.67(ptr)

declare ptr @tl_path_join.68(ptr, ptr)

declare i1 @tl_path_exists.69(ptr)

declare i1 @tl_path_is_dir.70(ptr)

declare i1 @tl_path_is_file.71(ptr)

declare ptr @tl_path_to_string.72(ptr)

declare void @tl_path_free.73(ptr)

declare i1 @tl_http_download.74(ptr, ptr)

declare ptr @tl_http_get.75(ptr)

declare ptr @tl_env_get.76(ptr)

declare void @tl_env_set.77(ptr, ptr)

declare float @tl_system_time.78()

declare void @tl_system_sleep.79(float)

declare i64 @tl_get_memory_mb.80()

declare void @tl_mem_enter_scope.81()

declare void @tl_mem_exit_scope.82()

declare void @tl_mem_register_struct.83(ptr)

declare void @tl_mem_register_tensor.84(ptr)

declare void @tl_mem_unregister.85(ptr)

define void @main() {
entry:
  %C = alloca ptr, align 8
  %_comp_res_0 = alloca ptr, align 8
  %j = alloca i64, align 8
  %k = alloca i64, align 8
  %i20 = alloca i64, align 8
  %B = alloca ptr, align 8
  %A = alloca ptr, align 8
  %temp_data_heap = call ptr @malloc(i64 16)
  %temp_shape_heap = call ptr @malloc(i64 16)
  %data_elem = getelementptr inbounds float, ptr %temp_data_heap, i64 0
  store float 1.000000e+00, ptr %data_elem, align 4
  %data_elem1 = getelementptr inbounds float, ptr %temp_data_heap, i64 1
  store float 2.000000e+00, ptr %data_elem1, align 4
  %data_elem2 = getelementptr inbounds float, ptr %temp_data_heap, i64 2
  store float 3.000000e+00, ptr %data_elem2, align 4
  %data_elem3 = getelementptr inbounds float, ptr %temp_data_heap, i64 3
  store float 4.000000e+00, ptr %data_elem3, align 4
  %shape_elem = getelementptr inbounds i64, ptr %temp_shape_heap, i64 0
  store i64 2, ptr %shape_elem, align 8
  %shape_elem4 = getelementptr inbounds i64, ptr %temp_shape_heap, i64 1
  store i64 2, ptr %shape_elem4, align 8
  %new_const_tensor = call ptr @tl_tensor_new(ptr %temp_data_heap, i64 2, ptr %temp_shape_heap)
  call void @free(ptr %temp_data_heap)
  call void @free(ptr %temp_shape_heap)
  store ptr %new_const_tensor, ptr %A, align 8
  %A5 = load ptr, ptr %A, align 8
  call void @tl_tensor_print(ptr %A5)
  %temp_data_heap6 = call ptr @malloc(i64 16)
  %temp_shape_heap7 = call ptr @malloc(i64 16)
  %data_elem8 = getelementptr inbounds float, ptr %temp_data_heap6, i64 0
  store float 1.000000e+00, ptr %data_elem8, align 4
  %data_elem9 = getelementptr inbounds float, ptr %temp_data_heap6, i64 1
  store float 0.000000e+00, ptr %data_elem9, align 4
  %data_elem10 = getelementptr inbounds float, ptr %temp_data_heap6, i64 2
  store float 0.000000e+00, ptr %data_elem10, align 4
  %data_elem11 = getelementptr inbounds float, ptr %temp_data_heap6, i64 3
  store float 1.000000e+00, ptr %data_elem11, align 4
  %shape_elem12 = getelementptr inbounds i64, ptr %temp_shape_heap7, i64 0
  store i64 2, ptr %shape_elem12, align 8
  %shape_elem13 = getelementptr inbounds i64, ptr %temp_shape_heap7, i64 1
  store i64 2, ptr %shape_elem13, align 8
  %new_const_tensor14 = call ptr @tl_tensor_new(ptr %temp_data_heap6, i64 2, ptr %temp_shape_heap7)
  call void @free(ptr %temp_data_heap6)
  call void @free(ptr %temp_shape_heap7)
  store ptr %new_const_tensor14, ptr %B, align 8
  %A15 = load ptr, ptr %A, align 8
  %dim_size = call i64 @tl_tensor_dim(ptr %A15, i64 0)
  %dim_size16 = call i64 @tl_tensor_dim(ptr %A15, i64 1)
  %B17 = load ptr, ptr %B, align 8
  %dim_size18 = call i64 @tl_tensor_dim(ptr %B17, i64 1)
  %sz_acc = mul i64 1, %dim_size
  %sz_acc19 = mul i64 %sz_acc, %dim_size18
  %buf_void = call ptr @calloc(i64 %sz_acc19, i64 4)
  br label %loop_cond

eq_after:                                         ; preds = %loop_aft
  %shape = alloca [2 x i64], align 8
  %shape_ptr = getelementptr [2 x i64], ptr %shape, i64 0, i64 0
  store i64 %dim_size, ptr %shape_ptr, align 8
  %shape_ptr51 = getelementptr [2 x i64], ptr %shape, i64 0, i64 1
  store i64 %dim_size18, ptr %shape_ptr51, align 8
  %t = call ptr @tl_tensor_new(ptr %buf_void, i64 2, ptr %shape)
  store ptr %t, ptr %_comp_res_0, align 8
  %tensor_ptr = load ptr, ptr %_comp_res_0, align 8
  store ptr %tensor_ptr, ptr %C, align 8
  %C52 = load ptr, ptr %C, align 8
  call void @tl_tensor_print(ptr %C52)
  ret void

loop_cond:                                        ; preds = %loop_aft23, %entry
  %i = phi i64 [ 0, %entry ], [ %next50, %loop_aft23 ]
  %cmp = icmp slt i64 %i, %dim_size
  br i1 %cmp, label %loop_body, label %loop_aft

loop_body:                                        ; preds = %loop_cond
  store i64 %i, ptr %i20, align 8
  br label %loop_cond21

loop_aft:                                         ; preds = %loop_cond
  br label %eq_after

loop_cond21:                                      ; preds = %loop_aft28, %loop_body
  %i24 = phi i64 [ 0, %loop_body ], [ %next48, %loop_aft28 ]
  %cmp25 = icmp slt i64 %i24, %dim_size18
  br i1 %cmp25, label %loop_body22, label %loop_aft23

loop_body22:                                      ; preds = %loop_cond21
  store i64 %i24, ptr %k, align 8
  br label %loop_cond26

loop_aft23:                                       ; preds = %loop_cond21
  %iv49 = load i64, ptr %i20, align 8
  %next50 = add i64 %iv49, 1
  br label %loop_cond

loop_cond26:                                      ; preds = %loop_body27, %loop_body22
  %i29 = phi i64 [ 0, %loop_body22 ], [ %next, %loop_body27 ]
  %cmp30 = icmp slt i64 %i29, %dim_size16
  br i1 %cmp30, label %loop_body27, label %loop_aft28

loop_body27:                                      ; preds = %loop_cond26
  store i64 %i29, ptr %j, align 8
  %A31 = load ptr, ptr %A, align 8
  %idx_arr = alloca [2 x i64], align 8
  %i32 = load i64, ptr %i20, align 8
  %idx_ptr = getelementptr [2 x i64], ptr %idx_arr, i64 0, i64 0
  store i64 %i32, ptr %idx_ptr, align 8
  %j33 = load i64, ptr %j, align 8
  %idx_ptr34 = getelementptr [2 x i64], ptr %idx_arr, i64 0, i64 1
  store i64 %j33, ptr %idx_ptr34, align 8
  %get_md_call = call float @tl_tensor_get_f32_md(ptr %A31, ptr %idx_arr, i64 2)
  %B35 = load ptr, ptr %B, align 8
  %idx_arr36 = alloca [2 x i64], align 8
  %j37 = load i64, ptr %j, align 8
  %idx_ptr38 = getelementptr [2 x i64], ptr %idx_arr36, i64 0, i64 0
  store i64 %j37, ptr %idx_ptr38, align 8
  %k39 = load i64, ptr %k, align 8
  %idx_ptr40 = getelementptr [2 x i64], ptr %idx_arr36, i64 0, i64 1
  store i64 %k39, ptr %idx_ptr40, align 8
  %get_md_call41 = call float @tl_tensor_get_f32_md(ptr %B35, ptr %idx_arr36, i64 2)
  %fmultmp = fmul float %get_md_call, %get_md_call41
  %iv = load i64, ptr %k, align 8
  %term = mul i64 %iv, 1
  %off = add i64 0, %term
  %str = mul i64 1, %dim_size18
  %iv42 = load i64, ptr %i20, align 8
  %term43 = mul i64 %iv42, %str
  %off44 = add i64 %off, %term43
  %str45 = mul i64 %str, %dim_size
  %ptr = getelementptr float, ptr %buf_void, i64 %off44
  %cur = load float, ptr %ptr, align 4
  %new = fadd float %cur, %fmultmp
  store float %new, ptr %ptr, align 4
  %iv46 = load i64, ptr %j, align 8
  %next = add i64 %iv46, 1
  br label %loop_cond26

loop_aft28:                                       ; preds = %loop_cond26
  %iv47 = load i64, ptr %k, align 8
  %next48 = add i64 %iv47, 1
  br label %loop_cond21
}
