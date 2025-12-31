; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

declare void @tl_print_i64(i64)

declare void @tl_print_f32(float)

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

declare ptr @tl_tensor_sub.24(ptr, ptr)

declare ptr @tl_tensor_matmul.25(ptr, ptr)

declare ptr @tl_tensor_exp.26(ptr)

declare ptr @tl_tensor_log.27(ptr)

declare ptr @tl_tensor_sqrt.28(ptr)

declare ptr @tl_tensor_pow.29(ptr, ptr)

declare void @tl_tensor_add_assign.30(ptr, ptr)

declare void @tl_tensor_sub_assign.31(ptr, ptr)

declare void @tl_tensor_mul_assign.32(ptr, ptr)

declare void @tl_tensor_div_assign.33(ptr, ptr)

declare void @tl_register_tensor.34(ptr, ptr)

declare ptr @tl_tensor_randn.35(i64, ptr, i1)

declare void @tl_tensor_backward.36(ptr)

declare ptr @tl_tensor_grad.37(ptr)

declare ptr @tl_tensor_detach.38(ptr, i1)

declare ptr @tl_tensor_softmax.39(ptr, i64)

declare ptr @tl_tensor_cross_entropy.40(ptr, ptr)

declare void @tl_tensor_sub_assign.41(ptr, ptr)

define void @main() {
entry:
  %val = alloca float, align 4
  %b = alloca i64, align 8
  %a = alloca i64, align 8
  store i64 5, ptr %a, align 8
  store i64 10, ptr %b, align 8
  br i1 true, label %then, label %else

then:                                             ; preds = %entry
  %a1 = load i64, ptr %a, align 8
  call void @tl_print_i64(i64 %a1)
  br label %merge

else:                                             ; preds = %entry
  %b2 = load i64, ptr %b, align 8
  call void @tl_print_i64(i64 %b2)
  br label %merge

merge:                                            ; preds = %else, %then
  br i1 false, label %then3, label %else4

then3:                                            ; preds = %merge
  call void @tl_print_i64(i64 0)
  br label %merge5

else4:                                            ; preds = %merge
  br i1 true, label %then6, label %else7

merge5:                                           ; preds = %merge8, %then3
  store float 4.200000e+01, ptr %val, align 4
  %tensor_data = alloca float, i64 3, align 4
  %val9 = load float, ptr %val, align 4
  %elem_ptr = getelementptr inbounds float, ptr %tensor_data, i64 0
  store float %val9, ptr %elem_ptr, align 4
  %val10 = load float, ptr %val, align 4
  %faddtmp = fadd float %val10, 1.000000e+00
  %elem_ptr11 = getelementptr inbounds float, ptr %tensor_data, i64 1
  store float %faddtmp, ptr %elem_ptr11, align 4
  %val12 = load float, ptr %val, align 4
  %fmultmp = fmul float %val12, 2.000000e+00
  %elem_ptr13 = getelementptr inbounds float, ptr %tensor_data, i64 2
  store float %fmultmp, ptr %elem_ptr13, align 4
  %tensor_shape = alloca i64, i64 1, align 8
  %shape_ptr = getelementptr inbounds i64, ptr %tensor_shape, i64 0
  store i64 3, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %tensor_data, i64 1, ptr %tensor_shape)
  call void @tl_tensor_print(ptr %new_tensor)
  ret void

then6:                                            ; preds = %else4
  call void @tl_print_i64(i64 200)
  br label %merge8

else7:                                            ; preds = %else4
  call void @tl_print_i64(i64 300)
  br label %merge8

merge8:                                           ; preds = %else7, %then6
  br label %merge5
}
