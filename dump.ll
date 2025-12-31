; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@str_literal = private unnamed_addr constant [27 x i8] c"Test: VarBuilder functions\00", align 1
@param_name = private unnamed_addr constant [12 x i8] c"test_weight\00", align 1
@tensor_name = private unnamed_addr constant [7 x i8] c"weight\00", align 1
@param_name.71 = private unnamed_addr constant [10 x i8] c"test_bias\00", align 1
@tensor_name.72 = private unnamed_addr constant [5 x i8] c"bias\00", align 1
@str_literal.73 = private unnamed_addr constant [19 x i8] c"Created parameters\00", align 1
@tensor_name.74 = private unnamed_addr constant [2 x i8] c"x\00", align 1
@tensor_name.75 = private unnamed_addr constant [7 x i8] c"output\00", align 1
@tensor_name.76 = private unnamed_addr constant [5 x i8] c"loss\00", align 1
@str_literal.77 = private unnamed_addr constant [6 x i8] c"Loss:\00", align 1
@param_name.78 = private unnamed_addr constant [12 x i8] c"test_weight\00", align 1
@tensor_name.79 = private unnamed_addr constant [3 x i8] c"gw\00", align 1
@param_name.80 = private unnamed_addr constant [10 x i8] c"test_bias\00", align 1
@tensor_name.81 = private unnamed_addr constant [3 x i8] c"gb\00", align 1
@str_literal.82 = private unnamed_addr constant [17 x i8] c"Weight gradient:\00", align 1
@str_literal.83 = private unnamed_addr constant [15 x i8] c"Bias gradient:\00", align 1
@str_literal.84 = private unnamed_addr constant [20 x i8] c"Parameters updated!\00", align 1

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

declare void @tl_tensor_sub_assign.1(ptr, ptr)

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

declare void @tl_tensor_sub_assign.52(ptr, ptr)

declare ptr @tl_string_concat.53(ptr, ptr)

declare ptr @tl_file_open.54(ptr, ptr)

declare ptr @tl_file_read_string.55(ptr)

declare void @tl_file_write_string.56(ptr, ptr)

declare void @tl_file_close.57(ptr)

declare ptr @tl_path_new.58(ptr)

declare ptr @tl_path_join.59(ptr, ptr)

declare i1 @tl_path_exists.60(ptr)

declare i1 @tl_path_is_dir.61(ptr)

declare i1 @tl_path_is_file.62(ptr)

declare ptr @tl_path_to_string.63(ptr)

declare void @tl_path_free.64(ptr)

declare i1 @tl_http_download.65(ptr, ptr)

declare ptr @tl_http_get.66(ptr)

declare ptr @tl_env_get.67(ptr)

declare void @tl_env_set.68(ptr, ptr)

declare float @tl_system_time.69()

declare void @tl_system_sleep.70(float)

define void @main() {
entry:
  %gb = alloca ptr, align 8
  %gw = alloca ptr, align 8
  %loss = alloca ptr, align 8
  %output = alloca ptr, align 8
  %x = alloca ptr, align 8
  %bias = alloca ptr, align 8
  %weight = alloca ptr, align 8
  call void @tl_print_string(ptr @str_literal)
  %shape_arr = alloca [2 x i64], align 8
  %shape_elem_0 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 10, ptr %shape_elem_0, align 8
  %shape_elem_1 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 5, ptr %shape_elem_1, align 8
  %varbuilder_get_result = call ptr @tl_varbuilder_get(ptr @param_name, i64 2, ptr %shape_arr)
  store ptr %varbuilder_get_result, ptr %weight, align 8
  call void @tl_register_tensor(ptr @tensor_name, ptr %varbuilder_get_result)
  %shape_arr1 = alloca [1 x i64], align 8
  %shape_elem_02 = getelementptr inbounds [1 x i64], ptr %shape_arr1, i64 0, i64 0
  store i64 5, ptr %shape_elem_02, align 8
  %varbuilder_get_result3 = call ptr @tl_varbuilder_get(ptr @param_name.71, i64 1, ptr %shape_arr1)
  store ptr %varbuilder_get_result3, ptr %bias, align 8
  call void @tl_register_tensor(ptr @tensor_name.72, ptr %varbuilder_get_result3)
  call void @tl_print_string(ptr @str_literal.73)
  %weight4 = load ptr, ptr %weight, align 8
  call void @tl_tensor_print(ptr %weight4)
  %bias5 = load ptr, ptr %bias, align 8
  call void @tl_tensor_print(ptr %bias5)
  %shape_arr6 = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr6, i64 0, i64 0
  store i64 4, ptr %shape_ptr_in, align 8
  %shape_ptr_in7 = getelementptr inbounds [2 x i64], ptr %shape_arr6, i64 0, i64 1
  store i64 10, ptr %shape_ptr_in7, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr6, i1 false)
  store ptr %randn_res, ptr %x, align 8
  call void @tl_register_tensor(ptr @tensor_name.74, ptr %randn_res)
  %x8 = load ptr, ptr %x, align 8
  %weight9 = load ptr, ptr %weight, align 8
  %matmul_res = call ptr @tl_tensor_matmul(ptr %x8, ptr %weight9)
  %bias10 = load ptr, ptr %bias, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %matmul_res, ptr %bias10)
  store ptr %binop_res, ptr %output, align 8
  call void @tl_register_tensor(ptr @tensor_name.75, ptr %binop_res)
  %output11 = load ptr, ptr %output, align 8
  %sum_res = call ptr @tl_tensor_sum(ptr %output11)
  store ptr %sum_res, ptr %loss, align 8
  call void @tl_register_tensor(ptr @tensor_name.76, ptr %sum_res)
  call void @tl_print_string(ptr @str_literal.77)
  %loss12 = load ptr, ptr %loss, align 8
  call void @tl_tensor_print(ptr %loss12)
  %loss13 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss13)
  %varbuilder_grad_result = call ptr @tl_varbuilder_grad(ptr @param_name.78)
  store ptr %varbuilder_grad_result, ptr %gw, align 8
  call void @tl_register_tensor(ptr @tensor_name.79, ptr %varbuilder_grad_result)
  %varbuilder_grad_result14 = call ptr @tl_varbuilder_grad(ptr @param_name.80)
  store ptr %varbuilder_grad_result14, ptr %gb, align 8
  call void @tl_register_tensor(ptr @tensor_name.81, ptr %varbuilder_grad_result14)
  call void @tl_print_string(ptr @str_literal.82)
  %gw15 = load ptr, ptr %gw, align 8
  call void @tl_tensor_print(ptr %gw15)
  call void @tl_print_string(ptr @str_literal.83)
  %gb16 = load ptr, ptr %gb, align 8
  call void @tl_tensor_print(ptr %gb16)
  call void @tl_update_all_params(float 0x3F847AE140000000)
  call void @tl_print_string(ptr @str_literal.84)
  %load_for_free = load ptr, ptr %bias, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free17 = load ptr, ptr %output, align 8
  call void @tl_tensor_free(ptr %load_for_free17)
  %load_for_free18 = load ptr, ptr %loss, align 8
  call void @tl_tensor_free(ptr %load_for_free18)
  %load_for_free19 = load ptr, ptr %gw, align 8
  call void @tl_tensor_free(ptr %load_for_free19)
  %load_for_free20 = load ptr, ptr %weight, align 8
  call void @tl_tensor_free(ptr %load_for_free20)
  %load_for_free21 = load ptr, ptr %x, align 8
  call void @tl_tensor_free(ptr %load_for_free21)
  %load_for_free22 = load ptr, ptr %gb, align 8
  call void @tl_tensor_free(ptr %load_for_free22)
  ret void
}
