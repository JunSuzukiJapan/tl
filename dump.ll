; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@str_literal = private unnamed_addr constant [12 x i8] c"True Branch\00", align 1
@str_literal.41 = private unnamed_addr constant [13 x i8] c"False Branch\00", align 1
@str_literal.42 = private unnamed_addr constant [13 x i8] c"Wrong Branch\00", align 1
@str_literal.43 = private unnamed_addr constant [13 x i8] c"False Branch\00", align 1

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

declare ptr @tl_tensor_randn(i64, ptr, i1)

declare void @tl_tensor_backward(ptr)

declare ptr @tl_tensor_grad(ptr)

declare ptr @tl_tensor_detach(ptr, i1)

declare ptr @tl_tensor_softmax(ptr, i64)

declare ptr @tl_tensor_cross_entropy(ptr, ptr)

declare void @tl_tensor_sub_assign.1(ptr, ptr)

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

declare ptr @tl_tensor_randn.34(i64, ptr, i1)

declare void @tl_tensor_backward.35(ptr)

declare ptr @tl_tensor_grad.36(ptr)

declare ptr @tl_tensor_detach.37(ptr, i1)

declare ptr @tl_tensor_softmax.38(ptr, i64)

declare ptr @tl_tensor_cross_entropy.39(ptr, ptr)

declare void @tl_tensor_sub_assign.40(ptr, ptr)

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
  call void @tl_print_string(ptr @str_literal.41)
  br label %merge

merge:                                            ; preds = %else, %then
  %x2 = load i64, ptr %x, align 8
  %lttmp = icmp slt i64 %x2, 5
  br i1 %lttmp, label %then3, label %else4

then3:                                            ; preds = %merge
  call void @tl_print_string(ptr @str_literal.42)
  br label %merge5

else4:                                            ; preds = %merge
  call void @tl_print_string(ptr @str_literal.43)
  br label %merge5

merge5:                                           ; preds = %else4, %then3
  ret void
}
