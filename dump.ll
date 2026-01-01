; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

%Data = type { ptr }

@tensor_name = private unnamed_addr constant [2 x i8] c"x\00", align 1
@tensor_name.77 = private unnamed_addr constant [2 x i8] c"y\00", align 1
@str_literal = private unnamed_addr constant [9 x i8] c"Inner OK\00", align 1

declare void @tl_print_i64(i64)

declare void @tl_print_f32(float)

declare void @tl_print_string(ptr)

declare ptr @malloc(i64)

declare ptr @calloc(i64, i64)

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

declare ptr @tl_tensor_pow(ptr, float)

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

declare void @tl_load_all_params(ptr)

declare void @tl_tensor_sub_assign.1(ptr, ptr)

declare void @tl_add_parameter(ptr, ptr)

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

declare void @tl_print_i64.2(i64)

declare void @tl_print_f32.3(float)

declare void @tl_print_string.4(ptr)

declare ptr @malloc.5(i64)

declare ptr @calloc.6(i64, i64)

declare i64 @tl_tensor_dim.7(ptr, i64)

declare float @tl_tensor_get_f32_md.8(ptr, ptr, i64)

declare ptr @tl_tensor_new.9(ptr, i64, ptr)

declare ptr @tl_tensor_sub.10(ptr, ptr)

declare void @tl_tensor_free.11(ptr)

declare ptr @tl_tensor_clone.12(ptr)

declare ptr @tl_tensor_add.13(ptr, ptr)

declare ptr @tl_tensor_mul.14(ptr, ptr)

declare void @tl_tensor_print.15(ptr)

declare float @tl_tensor_get.16(ptr, i64)

declare ptr @tl_tensor_slice.17(ptr, i64, i64)

declare i64 @tl_tensor_len.18(ptr)

declare ptr @tl_tensor_neg.19(ptr)

declare ptr @tl_tensor_transpose.20(ptr, i64, i64)

declare ptr @tl_tensor_pow.21(ptr, float)

declare ptr @tl_tensor_sqrt.22(ptr)

declare ptr @tl_tensor_sin.23(ptr)

declare ptr @tl_tensor_cos.24(ptr)

declare ptr @tl_tensor_relu.25(ptr)

declare ptr @tl_tensor_gelu.26(ptr)

declare ptr @tl_tensor_tril.27(ptr, i32)

declare ptr @tl_tensor_sum_dim.28(ptr, i64, i1)

declare ptr @tl_tensor_embedding.29(ptr, ptr)

declare ptr @tl_tensor_sum.30(ptr)

declare ptr @tl_tensor_div.31(ptr, ptr)

declare ptr @tl_tensor_matmul.32(ptr, ptr)

declare ptr @tl_tensor_exp.33(ptr)

declare ptr @tl_tensor_log.34(ptr)

declare void @tl_tensor_add_assign.35(ptr, ptr)

declare void @tl_tensor_sub_assign.36(ptr, ptr)

declare void @tl_tensor_mul_assign.37(ptr, ptr)

declare void @tl_tensor_div_assign.38(ptr, ptr)

declare void @tl_register_tensor.39(ptr, ptr)

declare i32 @strcmp.40(ptr, ptr)

declare ptr @tl_tensor_reshape_dims.41(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.42(ptr, ptr)

declare ptr @tl_tensor_randn.43(i64, ptr, i1)

declare ptr @tl_varbuilder_get.44(ptr, i64, ptr)

declare void @tl_update_all_params.45(float)

declare ptr @tl_varbuilder_grad.46(ptr)

declare void @tl_tensor_backward.47(ptr)

declare ptr @tl_tensor_grad.48(ptr)

declare ptr @tl_tensor_detach.49(ptr, i1)

declare ptr @tl_tensor_softmax.50(ptr, i64)

declare ptr @tl_tensor_cross_entropy.51(ptr, ptr)

declare void @tl_tensor_save.52(ptr, ptr)

declare ptr @tl_tensor_load.53(ptr)

declare void @tl_save_all_params.54(ptr)

declare void @tl_load_all_params.55(ptr)

declare void @tl_tensor_sub_assign.56(ptr, ptr)

declare void @tl_add_parameter.57(ptr, ptr)

declare ptr @tl_register_parameter.58(ptr)

declare ptr @tl_string_concat.59(ptr, ptr)

declare ptr @tl_file_open.60(ptr, ptr)

declare ptr @tl_file_read_string.61(ptr)

declare void @tl_file_write_string.62(ptr, ptr)

declare void @tl_file_close.63(ptr)

declare ptr @tl_path_new.64(ptr)

declare ptr @tl_path_join.65(ptr, ptr)

declare i1 @tl_path_exists.66(ptr)

declare i1 @tl_path_is_dir.67(ptr)

declare i1 @tl_path_is_file.68(ptr)

declare ptr @tl_path_to_string.69(ptr)

declare void @tl_path_free.70(ptr)

declare i1 @tl_http_download.71(ptr, ptr)

declare ptr @tl_http_get.72(ptr)

declare ptr @tl_env_get.73(ptr)

declare void @tl_env_set.74(ptr, ptr)

declare float @tl_system_time.75()

declare void @tl_system_sleep.76(float)

define void @main() {
entry:
  %d = alloca ptr, align 8
  %shape_arr6 = alloca [2 x i64], align 8
  %y = alloca ptr, align 8
  %shape_arr2 = alloca [2 x i64], align 8
  %x = alloca ptr, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 2, ptr %shape_ptr_in, align 8
  %shape_ptr_in1 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 2, ptr %shape_ptr_in1, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 false)
  store ptr %randn_res, ptr %x, align 8
  call void @tl_register_tensor(ptr @tensor_name, ptr %randn_res)
  br i1 true, label %then, label %else

then:                                             ; preds = %entry
  %shape_ptr_in3 = getelementptr inbounds [2 x i64], ptr %shape_arr2, i64 0, i64 0
  store i64 1, ptr %shape_ptr_in3, align 8
  %shape_ptr_in4 = getelementptr inbounds [2 x i64], ptr %shape_arr2, i64 0, i64 1
  store i64 1, ptr %shape_ptr_in4, align 8
  %randn_res5 = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr2, i1 false)
  store ptr %randn_res5, ptr %y, align 8
  call void @tl_register_tensor(ptr @tensor_name.77, ptr %randn_res5)
  call void @tl_print_string(ptr @str_literal)
  br label %merge

else:                                             ; preds = %entry
  br label %merge

merge:                                            ; preds = %else, %then
  %init_Data = alloca %Data, align 8
  %shape_ptr_in7 = getelementptr inbounds [2 x i64], ptr %shape_arr6, i64 0, i64 0
  store i64 3, ptr %shape_ptr_in7, align 8
  %shape_ptr_in8 = getelementptr inbounds [2 x i64], ptr %shape_arr6, i64 0, i64 1
  store i64 3, ptr %shape_ptr_in8, align 8
  %randn_res9 = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr6, i1 false)
  %Data.t = getelementptr inbounds nuw %Data, ptr %init_Data, i32 0, i32 0
  store ptr %randn_res9, ptr %Data.t, align 8
  %val_Data = load %Data, ptr %init_Data, align 8
  store %Data %val_Data, ptr %d, align 8
  %d10 = load %Data, ptr %d, align 8
  %t = extractvalue %Data %d10, 0
  call void @tl_tensor_print(ptr %t)
  %load_for_free = load ptr, ptr %x, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  ret void
}
