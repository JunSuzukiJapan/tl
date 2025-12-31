; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@tensor_name = private unnamed_addr constant [2 x i8] c"a\00", align 1
@tensor_name.42 = private unnamed_addr constant [2 x i8] c"b\00", align 1
@tensor_name.43 = private unnamed_addr constant [2 x i8] c"c\00", align 1
@tensor_name.44 = private unnamed_addr constant [2 x i8] c"d\00", align 1
@tensor_name.45 = private unnamed_addr constant [2 x i8] c"t\00", align 1

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
  %x = alloca float, align 4
  %t = alloca ptr, align 8
  %not_val = alloca i64, align 8
  %val = alloca i64, align 8
  %d = alloca ptr, align 8
  %c = alloca ptr, align 8
  %b = alloca ptr, align 8
  %a = alloca ptr, align 8
  %tensor_data = alloca float, i64 2, align 4
  %elem_ptr = getelementptr inbounds float, ptr %tensor_data, i64 0
  store float 1.000000e+00, ptr %elem_ptr, align 4
  %elem_ptr1 = getelementptr inbounds float, ptr %tensor_data, i64 1
  store float 2.000000e+00, ptr %elem_ptr1, align 4
  %tensor_shape = alloca i64, i64 1, align 8
  %shape_ptr = getelementptr inbounds i64, ptr %tensor_shape, i64 0
  store i64 2, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %tensor_data, i64 1, ptr %tensor_shape)
  store ptr %new_tensor, ptr %a, align 8
  call void @tl_register_tensor(ptr @tensor_name, ptr %new_tensor)
  %tensor_data2 = alloca float, i64 2, align 4
  %elem_ptr3 = getelementptr inbounds float, ptr %tensor_data2, i64 0
  store float -1.000000e+00, ptr %elem_ptr3, align 4
  %elem_ptr4 = getelementptr inbounds float, ptr %tensor_data2, i64 1
  store float -2.000000e+00, ptr %elem_ptr4, align 4
  %tensor_shape5 = alloca i64, i64 1, align 8
  %shape_ptr6 = getelementptr inbounds i64, ptr %tensor_shape5, i64 0
  store i64 2, ptr %shape_ptr6, align 8
  %new_tensor7 = call ptr @tl_tensor_new(ptr %tensor_data2, i64 1, ptr %tensor_shape5)
  store ptr %new_tensor7, ptr %b, align 8
  call void @tl_register_tensor(ptr @tensor_name.42, ptr %new_tensor7)
  %a8 = load ptr, ptr %a, align 8
  %b9 = load ptr, ptr %b, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %a8, ptr %b9)
  store ptr %binop_res, ptr %c, align 8
  call void @tl_register_tensor(ptr @tensor_name.43, ptr %binop_res)
  %c10 = load ptr, ptr %c, align 8
  call void @tl_tensor_print(ptr %c10)
  %a11 = load ptr, ptr %a, align 8
  %neg = call ptr @tl_tensor_neg(ptr %a11)
  store ptr %neg, ptr %d, align 8
  call void @tl_register_tensor(ptr @tensor_name.44, ptr %neg)
  store i1 true, ptr %val, align 1
  %val12 = load i1, ptr %val, align 1
  %nottmp = xor i1 %val12, true
  store i1 %nottmp, ptr %not_val, align 1
  %not_val13 = load i1, ptr %not_val, align 1
  br i1 %not_val13, label %then, label %else

then:                                             ; preds = %entry
  %tensor_data14 = alloca float, i64 1, align 4
  %elem_ptr15 = getelementptr inbounds float, ptr %tensor_data14, i64 0
  store float 0.000000e+00, ptr %elem_ptr15, align 4
  %tensor_shape16 = alloca i64, i64 1, align 8
  %shape_ptr17 = getelementptr inbounds i64, ptr %tensor_shape16, i64 0
  store i64 1, ptr %shape_ptr17, align 8
  %new_tensor18 = call ptr @tl_tensor_new(ptr %tensor_data14, i64 1, ptr %tensor_shape16)
  call void @tl_tensor_print(ptr %new_tensor18)
  br label %merge

else:                                             ; preds = %entry
  %tensor_data19 = alloca float, i64 1, align 4
  %elem_ptr20 = getelementptr inbounds float, ptr %tensor_data19, i64 0
  store float 1.000000e+00, ptr %elem_ptr20, align 4
  %tensor_shape21 = alloca i64, i64 1, align 8
  %shape_ptr22 = getelementptr inbounds i64, ptr %tensor_shape21, i64 0
  store i64 1, ptr %shape_ptr22, align 8
  %new_tensor23 = call ptr @tl_tensor_new(ptr %tensor_data19, i64 1, ptr %tensor_shape21)
  call void @tl_tensor_print(ptr %new_tensor23)
  br label %merge

merge:                                            ; preds = %else, %then
  br i1 true, label %then24, label %else25

then24:                                           ; preds = %merge
  %tensor_data27 = alloca float, i64 1, align 4
  %elem_ptr28 = getelementptr inbounds float, ptr %tensor_data27, i64 0
  store float 1.000000e+02, ptr %elem_ptr28, align 4
  %tensor_shape29 = alloca i64, i64 1, align 8
  %shape_ptr30 = getelementptr inbounds i64, ptr %tensor_shape29, i64 0
  store i64 1, ptr %shape_ptr30, align 8
  %new_tensor31 = call ptr @tl_tensor_new(ptr %tensor_data27, i64 1, ptr %tensor_shape29)
  call void @tl_tensor_print(ptr %new_tensor31)
  br label %merge26

else25:                                           ; preds = %merge
  br label %merge26

merge26:                                          ; preds = %else25, %then24
  %tensor_data32 = alloca float, i64 3, align 4
  %elem_ptr33 = getelementptr inbounds float, ptr %tensor_data32, i64 0
  store float 1.000000e+01, ptr %elem_ptr33, align 4
  %elem_ptr34 = getelementptr inbounds float, ptr %tensor_data32, i64 1
  store float 2.000000e+01, ptr %elem_ptr34, align 4
  %elem_ptr35 = getelementptr inbounds float, ptr %tensor_data32, i64 2
  store float 3.000000e+01, ptr %elem_ptr35, align 4
  %tensor_shape36 = alloca i64, i64 1, align 8
  %shape_ptr37 = getelementptr inbounds i64, ptr %tensor_shape36, i64 0
  store i64 3, ptr %shape_ptr37, align 8
  %new_tensor38 = call ptr @tl_tensor_new(ptr %tensor_data32, i64 1, ptr %tensor_shape36)
  store ptr %new_tensor38, ptr %t, align 8
  call void @tl_register_tensor(ptr @tensor_name.45, ptr %new_tensor38)
  %t39 = load ptr, ptr %t, align 8
  %tensor_len = call i64 @tl_tensor_len(ptr %t39)
  %for_tensor = alloca ptr, align 8
  store ptr %t39, ptr %for_tensor, align 8
  br label %for_header

for_header:                                       ; preds = %for_body, %merge26
  %for_idx = phi i64 [ 0, %merge26 ], [ %next_idx, %for_body ]
  %for_cond = icmp slt i64 %for_idx, %tensor_len
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  %tensor_ptr = load ptr, ptr %for_tensor, align 8
  %elem_val = call float @tl_tensor_get(ptr %tensor_ptr, i64 %for_idx)
  store float %elem_val, ptr %x, align 4
  %tensor_data40 = alloca float, i64 1, align 4
  %x41 = load float, ptr %x, align 4
  %elem_ptr42 = getelementptr inbounds float, ptr %tensor_data40, i64 0
  store float %x41, ptr %elem_ptr42, align 4
  %tensor_shape43 = alloca i64, i64 1, align 8
  %shape_ptr44 = getelementptr inbounds i64, ptr %tensor_shape43, i64 0
  store i64 1, ptr %shape_ptr44, align 8
  %new_tensor45 = call ptr @tl_tensor_new(ptr %tensor_data40, i64 1, ptr %tensor_shape43)
  call void @tl_tensor_print(ptr %new_tensor45)
  %next_idx = add i64 %for_idx, 1
  br label %for_header

for_end:                                          ; preds = %for_header
  %load_for_free = load ptr, ptr %d, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free46 = load ptr, ptr %t, align 8
  call void @tl_tensor_free(ptr %load_for_free46)
  %load_for_free47 = load ptr, ptr %a, align 8
  call void @tl_tensor_free(ptr %load_for_free47)
  %load_for_free48 = load ptr, ptr %c, align 8
  call void @tl_tensor_free(ptr %load_for_free48)
  %load_for_free49 = load ptr, ptr %b, align 8
  call void @tl_tensor_free(ptr %load_for_free49)
  ret void
}
