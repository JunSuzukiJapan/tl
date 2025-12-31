; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@str_literal = private unnamed_addr constant [20 x i8] c"Testing File I/O...\00", align 1
@str_literal.49 = private unnamed_addr constant [16 x i8] c"test_output.txt\00", align 1
@str_literal.50 = private unnamed_addr constant [26 x i8] c"Hello TensorLogic StdLib!\00", align 1
@str_literal.51 = private unnamed_addr constant [2 x i8] c"w\00", align 1
@str_literal.52 = private unnamed_addr constant [2 x i8] c"r\00", align 1
@str_literal.53 = private unnamed_addr constant [17 x i8] c"File I/O SUCCESS\00", align 1
@str_literal.54 = private unnamed_addr constant [16 x i8] c"File I/O FAILED\00", align 1
@str_literal.55 = private unnamed_addr constant [15 x i8] c"Testing Env...\00", align 1
@str_literal.56 = private unnamed_addr constant [5 x i8] c"PATH\00", align 1
@str_literal.57 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@str_literal.58 = private unnamed_addr constant [25 x i8] c"Env SUCCESS (PATH found)\00", align 1
@str_literal.59 = private unnamed_addr constant [11 x i8] c"Env FAILED\00", align 1

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

declare i32 @strcmp(ptr, ptr)

declare ptr @tl_tensor_randn(i64, ptr, i1)

declare void @tl_tensor_backward(ptr)

declare ptr @tl_tensor_grad(ptr)

declare ptr @tl_tensor_detach(ptr, i1)

declare ptr @tl_tensor_softmax(ptr, i64)

declare ptr @tl_tensor_cross_entropy(ptr, ptr)

declare void @tl_tensor_sub_assign.1(ptr, ptr)

declare ptr @tl_file_open(ptr, ptr)

declare ptr @tl_file_read_string(ptr)

declare void @tl_file_write_string(ptr, ptr)

declare void @tl_file_close(ptr)

declare i1 @tl_http_download(ptr, ptr)

declare ptr @tl_http_get(ptr)

declare ptr @tl_env_get(ptr)

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

declare ptr @tl_tensor_reshape.21(ptr, ptr)

declare ptr @tl_tensor_sum.22(ptr)

declare ptr @tl_tensor_div.23(ptr, ptr)

declare ptr @tl_tensor_matmul.24(ptr, ptr)

declare ptr @tl_tensor_exp.25(ptr)

declare ptr @tl_tensor_log.26(ptr)

declare ptr @tl_tensor_sqrt.27(ptr)

declare ptr @tl_tensor_pow.28(ptr, ptr)

declare void @tl_tensor_add_assign.29(ptr, ptr)

declare void @tl_tensor_sub_assign.30(ptr, ptr)

declare void @tl_tensor_mul_assign.31(ptr, ptr)

declare void @tl_tensor_div_assign.32(ptr, ptr)

declare void @tl_register_tensor.33(ptr, ptr)

declare i32 @strcmp.34(ptr, ptr)

declare ptr @tl_tensor_randn.35(i64, ptr, i1)

declare void @tl_tensor_backward.36(ptr)

declare ptr @tl_tensor_grad.37(ptr)

declare ptr @tl_tensor_detach.38(ptr, i1)

declare ptr @tl_tensor_softmax.39(ptr, i64)

declare ptr @tl_tensor_cross_entropy.40(ptr, ptr)

declare void @tl_tensor_sub_assign.41(ptr, ptr)

declare ptr @tl_file_open.42(ptr, ptr)

declare ptr @tl_file_read_string.43(ptr)

declare void @tl_file_write_string.44(ptr, ptr)

declare void @tl_file_close.45(ptr)

declare i1 @tl_http_download.46(ptr, ptr)

declare ptr @tl_http_get.47(ptr)

declare ptr @tl_env_get.48(ptr)

define void @test_file_io() {
entry:
  %read_content = alloca ptr, align 8
  %f_r = alloca ptr, align 8
  %f_w = alloca ptr, align 8
  %content = alloca ptr, align 8
  %path = alloca ptr, align 8
  call void @tl_print_string(ptr @str_literal)
  store ptr @str_literal.49, ptr %path, align 8
  store ptr @str_literal.50, ptr %content, align 8
  %path1 = load ptr, ptr %path, align 8
  %call_tmp = call ptr @tl_file_open(ptr %path1, ptr @str_literal.51)
  store ptr %call_tmp, ptr %f_w, align 8
  %f_w2 = load ptr, ptr %f_w, align 8
  %content3 = load ptr, ptr %content, align 8
  call void @tl_file_write_string(ptr %f_w2, ptr %content3)
  %f_w4 = load ptr, ptr %f_w, align 8
  call void @tl_file_close(ptr %f_w4)
  %path5 = load ptr, ptr %path, align 8
  %call_tmp6 = call ptr @tl_file_open(ptr %path5, ptr @str_literal.52)
  store ptr %call_tmp6, ptr %f_r, align 8
  %f_r7 = load ptr, ptr %f_r, align 8
  %call_method = call ptr @tl_file_read_string(ptr %f_r7)
  store ptr %call_method, ptr %read_content, align 8
  %f_r8 = load ptr, ptr %f_r, align 8
  call void @tl_file_close(ptr %f_r8)
  %read_content9 = load ptr, ptr %read_content, align 8
  call void @tl_print_string(ptr %read_content9)
  %read_content10 = load ptr, ptr %read_content, align 8
  %content11 = load ptr, ptr %content, align 8
  %strcmp_res = call i32 @strcmp(ptr %read_content10, ptr %content11)
  %streq = icmp eq i32 %strcmp_res, 0
  br i1 %streq, label %then, label %else

then:                                             ; preds = %entry
  call void @tl_print_string(ptr @str_literal.53)
  br label %merge

else:                                             ; preds = %entry
  call void @tl_print_string(ptr @str_literal.54)
  br label %merge

merge:                                            ; preds = %else, %then
  ret void
}

define void @test_env() {
entry:
  %path_env = alloca ptr, align 8
  call void @tl_print_string(ptr @str_literal.55)
  %call_tmp = call ptr @tl_env_get(ptr @str_literal.56)
  store ptr %call_tmp, ptr %path_env, align 8
  %path_env1 = load ptr, ptr %path_env, align 8
  %strcmp_res = call i32 @strcmp(ptr %path_env1, ptr @str_literal.57)
  %strneq = icmp ne i32 %strcmp_res, 0
  br i1 %strneq, label %then, label %else

then:                                             ; preds = %entry
  call void @tl_print_string(ptr @str_literal.58)
  br label %merge

else:                                             ; preds = %entry
  call void @tl_print_string(ptr @str_literal.59)
  br label %merge

merge:                                            ; preds = %else, %then
  ret void
}

define void @main() {
entry:
  call void @test_file_io()
  call void @test_env()
  ret void
}
