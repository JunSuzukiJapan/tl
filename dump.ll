; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@str_literal = private unnamed_addr constant [12 x i8] c"True Branch\00", align 1
@str_literal.84 = private unnamed_addr constant [13 x i8] c"False Branch\00", align 1
@str_literal.85 = private unnamed_addr constant [13 x i8] c"Wrong Branch\00", align 1
@str_literal.86 = private unnamed_addr constant [13 x i8] c"False Branch\00", align 1

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

declare i64 @tl_get_memory_mb()

declare void @tl_mem_enter_scope()

declare void @tl_mem_exit_scope()

declare void @tl_mem_register_struct(ptr)

declare void @tl_mem_register_tensor(ptr)

declare void @tl_mem_unregister(ptr)

declare void @tl_print_i64.2(i64)

declare void @tl_print_f32.3(float)

declare void @tl_print_string.4(ptr)

declare ptr @malloc.5(i64)

declare ptr @calloc.6(i64, i64)

declare void @free.7(ptr)

declare i64 @tl_tensor_dim.8(ptr, i64)

declare float @tl_tensor_get_f32_md.9(ptr, ptr, i64)

declare ptr @tl_tensor_new.10(ptr, i64, ptr)

declare ptr @tl_tensor_sub.11(ptr, ptr)

declare void @tl_tensor_free.12(ptr)

declare ptr @tl_tensor_clone.13(ptr)

declare ptr @tl_tensor_add.14(ptr, ptr)

declare ptr @tl_tensor_mul.15(ptr, ptr)

declare void @tl_tensor_print.16(ptr)

declare float @tl_tensor_get.17(ptr, i64)

declare ptr @tl_tensor_slice.18(ptr, i64, i64)

declare i64 @tl_tensor_len.19(ptr)

declare ptr @tl_tensor_neg.20(ptr)

declare ptr @tl_tensor_transpose.21(ptr, i64, i64)

declare ptr @tl_tensor_pow.22(ptr, ptr)

declare ptr @tl_tensor_sqrt.23(ptr)

declare ptr @tl_tensor_sin.24(ptr)

declare ptr @tl_tensor_cos.25(ptr)

declare ptr @tl_tensor_relu.26(ptr)

declare ptr @tl_tensor_gelu.27(ptr)

declare ptr @tl_tensor_tril.28(ptr, i32)

declare ptr @tl_tensor_sum_dim.29(ptr, i64, i1)

declare ptr @tl_tensor_embedding.30(ptr, ptr)

declare ptr @tl_tensor_sum.31(ptr)

declare ptr @tl_tensor_div.32(ptr, ptr)

declare ptr @tl_tensor_matmul.33(ptr, ptr)

declare ptr @tl_tensor_exp.34(ptr)

declare ptr @tl_tensor_log.35(ptr)

declare void @tl_tensor_add_assign.36(ptr, ptr)

declare void @tl_tensor_sub_assign.37(ptr, ptr)

declare void @tl_tensor_mul_assign.38(ptr, ptr)

declare void @tl_tensor_div_assign.39(ptr, ptr)

declare void @tl_register_tensor.40(ptr, ptr)

declare i32 @strcmp.41(ptr, ptr)

declare ptr @tl_tensor_reshape_dims.42(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.43(ptr, ptr)

declare ptr @tl_tensor_randn.44(i64, ptr, i1)

declare ptr @tl_varbuilder_get.45(ptr, i64, ptr)

declare void @tl_update_all_params.46(float)

declare ptr @tl_varbuilder_grad.47(ptr)

declare void @tl_tensor_backward.48(ptr)

declare ptr @tl_tensor_grad.49(ptr)

declare ptr @tl_tensor_detach.50(ptr, i1)

declare ptr @tl_tensor_softmax.51(ptr, i64)

declare ptr @tl_tensor_cross_entropy.52(ptr, ptr)

declare void @tl_tensor_save.53(ptr, ptr)

declare ptr @tl_tensor_load.54(ptr)

declare void @tl_save_all_params.55(ptr)

declare void @tl_load_all_params.56(ptr)

declare void @tl_tensor_sub_assign.57(ptr, ptr)

declare void @tl_add_parameter.58(ptr, ptr)

declare ptr @tl_register_parameter.59(ptr)

declare ptr @tl_string_concat.60(ptr, ptr)

declare ptr @tl_file_open.61(ptr, ptr)

declare ptr @tl_file_read_string.62(ptr)

declare void @tl_file_write_string.63(ptr, ptr)

declare void @tl_file_close.64(ptr)

declare ptr @tl_path_new.65(ptr)

declare ptr @tl_path_join.66(ptr, ptr)

declare i1 @tl_path_exists.67(ptr)

declare i1 @tl_path_is_dir.68(ptr)

declare i1 @tl_path_is_file.69(ptr)

declare ptr @tl_path_to_string.70(ptr)

declare void @tl_path_free.71(ptr)

declare i1 @tl_http_download.72(ptr, ptr)

declare ptr @tl_http_get.73(ptr)

declare ptr @tl_env_get.74(ptr)

declare void @tl_env_set.75(ptr, ptr)

declare float @tl_system_time.76()

declare void @tl_system_sleep.77(float)

declare i64 @tl_get_memory_mb.78()

declare void @tl_mem_enter_scope.79()

declare void @tl_mem_exit_scope.80()

declare void @tl_mem_register_struct.81(ptr)

declare void @tl_mem_register_tensor.82(ptr)

declare void @tl_mem_unregister.83(ptr)

define void @main() {
entry:
  %x = alloca i64, align 8
  store i64 10, ptr %x, align 8
  %x1 = load i64, ptr %x, align 8
  %gttmp = icmp sgt i64 %x1, 5
  br i1 %gttmp, label %then, label %else

then:                                             ; preds = %entry
  call void @tl_print_string(ptr @str_literal)
  br label %merge

else:                                             ; preds = %entry
  call void @tl_print_string(ptr @str_literal.84)
  br label %merge

merge:                                            ; preds = %else, %then
  %x2 = load i64, ptr %x, align 8
  %lttmp = icmp slt i64 %x2, 5
  br i1 %lttmp, label %then3, label %else4

then3:                                            ; preds = %merge
  call void @tl_print_string(ptr @str_literal.85)
  br label %merge5

else4:                                            ; preds = %merge
  call void @tl_print_string(ptr @str_literal.86)
  br label %merge5

merge5:                                           ; preds = %else4, %then3
  ret void
}
