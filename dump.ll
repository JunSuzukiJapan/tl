; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@tensor_name = private unnamed_addr constant [2 x i8] c"A\00", align 1
@tensor_name.43 = private unnamed_addr constant [2 x i8] c"B\00", align 1
@str_literal = private unnamed_addr constant [21 x i8] c"MatMul result shape:\00", align 1
@tensor_name.44 = private unnamed_addr constant [2 x i8] c"D\00", align 1
@tensor_name.45 = private unnamed_addr constant [5 x i8] c"diff\00", align 1
@tensor_name.46 = private unnamed_addr constant [2 x i8] c"s\00", align 1
@str_literal.47 = private unnamed_addr constant [37 x i8] c"Difference sum (should be approx 0):\00", align 1

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

declare ptr @tl_tensor_reshape(ptr, ptr)

declare ptr @tl_tensor_sum(ptr)

declare ptr @tl_tensor_div(ptr, ptr)

declare ptr @tl_tensor_sub.1(ptr, ptr)

declare ptr @tl_tensor_matmul(ptr, ptr)

declare ptr @tl_tensor_exp(ptr)

declare ptr @tl_tensor_log(ptr)

declare ptr @tl_tensor_sqrt(ptr)

declare ptr @tl_tensor_pow(ptr, ptr)

declare void @tl_tensor_add_assign(ptr, ptr)

declare void @tl_tensor_sub_assign(ptr, ptr)

declare void @tl_tensor_mul_assign(ptr, ptr)

declare void @tl_tensor_div_assign(ptr, ptr)

declare void @tl_register_tensor(ptr, ptr)

declare ptr @tl_tensor_randn(i64, ptr, i1)

declare void @tl_tensor_backward(ptr)

declare ptr @tl_tensor_grad(ptr)

declare ptr @tl_tensor_detach(ptr, i1)

declare ptr @tl_tensor_softmax(ptr, i64)

declare ptr @tl_tensor_cross_entropy(ptr, ptr)

declare void @tl_tensor_sub_assign.2(ptr, ptr)

declare void @tl_print_i64.3(i64)

declare void @tl_print_f32.4(float)

declare void @tl_print_string.5(ptr)

declare ptr @malloc.6(i64)

declare ptr @calloc.7(i64, i64)

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

declare ptr @tl_tensor_reshape.22(ptr, ptr)

declare ptr @tl_tensor_sum.23(ptr)

declare ptr @tl_tensor_div.24(ptr, ptr)

declare ptr @tl_tensor_sub.25(ptr, ptr)

declare ptr @tl_tensor_matmul.26(ptr, ptr)

declare ptr @tl_tensor_exp.27(ptr)

declare ptr @tl_tensor_log.28(ptr)

declare ptr @tl_tensor_sqrt.29(ptr)

declare ptr @tl_tensor_pow.30(ptr, ptr)

declare void @tl_tensor_add_assign.31(ptr, ptr)

declare void @tl_tensor_sub_assign.32(ptr, ptr)

declare void @tl_tensor_mul_assign.33(ptr, ptr)

declare void @tl_tensor_div_assign.34(ptr, ptr)

declare void @tl_register_tensor.35(ptr, ptr)

declare ptr @tl_tensor_randn.36(i64, ptr, i1)

declare void @tl_tensor_backward.37(ptr)

declare ptr @tl_tensor_grad.38(ptr)

declare ptr @tl_tensor_detach.39(ptr, i1)

declare ptr @tl_tensor_softmax.40(ptr, i64)

declare ptr @tl_tensor_cross_entropy.41(ptr, ptr)

declare void @tl_tensor_sub_assign.42(ptr, ptr)

define void @main() {
entry:
  %s = alloca ptr, align 8
  %diff = alloca ptr, align 8
  %D = alloca ptr, align 8
  %C = alloca ptr, align 8
  %B = alloca ptr, align 8
  %A = alloca ptr, align 8
  %N = alloca i64, align 8
  store i64 256, ptr %N, align 8
  %N1 = load i64, ptr %N, align 8
  %N2 = load i64, ptr %N, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %N1, ptr %shape_ptr_in, align 8
  %shape_ptr_in3 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 %N2, ptr %shape_ptr_in3, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 false)
  store ptr %randn_res, ptr %A, align 8
  call void @tl_register_tensor(ptr @tensor_name, ptr %randn_res)
  %N4 = load i64, ptr %N, align 8
  %N5 = load i64, ptr %N, align 8
  %shape_arr6 = alloca [2 x i64], align 8
  %shape_ptr_in7 = getelementptr inbounds [2 x i64], ptr %shape_arr6, i64 0, i64 0
  store i64 %N4, ptr %shape_ptr_in7, align 8
  %shape_ptr_in8 = getelementptr inbounds [2 x i64], ptr %shape_arr6, i64 0, i64 1
  store i64 %N5, ptr %shape_ptr_in8, align 8
  %randn_res9 = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr6, i1 false)
  store ptr %randn_res9, ptr %B, align 8
  call void @tl_register_tensor(ptr @tensor_name.43, ptr %randn_res9)
  %load_tensor_ptr = load ptr, ptr %A, align 8
  %load_tensor_ptr10 = load ptr, ptr %B, align 8
  %matmul_res = call ptr @tl_tensor_matmul(ptr %load_tensor_ptr, ptr %load_tensor_ptr10)
  store ptr %matmul_res, ptr %C, align 8
  call void @tl_print_string(ptr @str_literal)
  %C11 = load ptr, ptr %C, align 8
  call void @tl_tensor_print(ptr %C11)
  %A12 = load ptr, ptr %A, align 8
  %B13 = load ptr, ptr %B, align 8
  %matmul_res14 = call ptr @tl_tensor_matmul(ptr %A12, ptr %B13)
  store ptr %matmul_res14, ptr %D, align 8
  call void @tl_register_tensor(ptr @tensor_name.44, ptr %matmul_res14)
  %C15 = load ptr, ptr %C, align 8
  %D16 = load ptr, ptr %D, align 8
  %binop_res = call ptr @tl_tensor_sub(ptr %C15, ptr %D16)
  store ptr %binop_res, ptr %diff, align 8
  call void @tl_register_tensor(ptr @tensor_name.45, ptr %binop_res)
  %diff17 = load ptr, ptr %diff, align 8
  %sum_res = call ptr @tl_tensor_sum(ptr %diff17)
  store ptr %sum_res, ptr %s, align 8
  call void @tl_register_tensor(ptr @tensor_name.46, ptr %sum_res)
  call void @tl_print_string(ptr @str_literal.47)
  %s18 = load ptr, ptr %s, align 8
  call void @tl_tensor_print(ptr %s18)
  %load_for_free = load ptr, ptr %s, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free19 = load ptr, ptr %diff, align 8
  call void @tl_tensor_free(ptr %load_for_free19)
  %load_for_free20 = load ptr, ptr %D, align 8
  call void @tl_tensor_free(ptr %load_for_free20)
  %load_for_free21 = load ptr, ptr %B, align 8
  call void @tl_tensor_free(ptr %load_for_free21)
  %load_for_free22 = load ptr, ptr %A, align 8
  call void @tl_tensor_free(ptr %load_for_free22)
  ret void
}
