; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

%Linear = type { ptr, ptr }

@tensor_name = private unnamed_addr constant [2 x i8] c"W\00", align 1
@tensor_name.36 = private unnamed_addr constant [2 x i8] c"b\00", align 1
@tensor_name.37 = private unnamed_addr constant [4 x i8] c"out\00", align 1
@tensor_name.38 = private unnamed_addr constant [2 x i8] c"y\00", align 1
@tensor_name.39 = private unnamed_addr constant [2 x i8] c"X\00", align 1
@tensor_name.40 = private unnamed_addr constant [7 x i8] c"target\00", align 1
@tensor_name.41 = private unnamed_addr constant [7 x i8] c"logits\00", align 1
@tensor_name.42 = private unnamed_addr constant [6 x i8] c"probs\00", align 1
@tensor_name.43 = private unnamed_addr constant [5 x i8] c"loss\00", align 1

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

declare void @tl_register_tensor(ptr, ptr)

declare ptr @tl_tensor_randn(i64, ptr, i1)

declare void @tl_tensor_backward(ptr)

declare ptr @tl_tensor_grad(ptr)

declare ptr @tl_tensor_detach(ptr, i1)

declare ptr @tl_tensor_softmax(ptr, i64)

declare ptr @tl_tensor_cross_entropy(ptr, ptr)

declare void @tl_tensor_sub_assign(ptr, ptr)

declare void @tl_print_i64.2(i64)

declare void @tl_print_f32.3(float)

declare ptr @malloc.4(i64)

declare ptr @calloc.5(i64, i64)

declare i64 @tl_tensor_dim.6(ptr, i64)

declare float @tl_tensor_get_f32_md.7(ptr, ptr, i64)

declare ptr @tl_tensor_new.8(ptr, i64, ptr)

declare ptr @tl_tensor_sub.9(ptr, ptr)

declare void @tl_tensor_free.10(ptr)

declare ptr @tl_tensor_clone.11(ptr)

declare ptr @tl_tensor_add.12(ptr, ptr)

declare ptr @tl_tensor_mul.13(ptr, ptr)

declare void @tl_tensor_print.14(ptr)

declare float @tl_tensor_get.15(ptr, i64)

declare ptr @tl_tensor_slice.16(ptr, i64, i64)

declare i64 @tl_tensor_len.17(ptr)

declare ptr @tl_tensor_neg.18(ptr)

declare ptr @tl_tensor_transpose.19(ptr, i64, i64)

declare ptr @tl_tensor_reshape.20(ptr, ptr)

declare ptr @tl_tensor_sum.21(ptr)

declare ptr @tl_tensor_div.22(ptr, ptr)

declare ptr @tl_tensor_sub.23(ptr, ptr)

declare ptr @tl_tensor_matmul.24(ptr, ptr)

declare ptr @tl_tensor_exp.25(ptr)

declare ptr @tl_tensor_log.26(ptr)

declare ptr @tl_tensor_sqrt.27(ptr)

declare void @tl_register_tensor.28(ptr, ptr)

declare ptr @tl_tensor_randn.29(i64, ptr, i1)

declare void @tl_tensor_backward.30(ptr)

declare ptr @tl_tensor_grad.31(ptr)

declare ptr @tl_tensor_detach.32(ptr, i1)

declare ptr @tl_tensor_softmax.33(ptr, i64)

declare ptr @tl_tensor_cross_entropy.34(ptr, ptr)

declare void @tl_tensor_sub_assign.35(ptr, ptr)

define ptr @Linear_new(i64 %in_dim, i64 %out_dim) {
entry:
  %b = alloca ptr, align 8
  %W = alloca ptr, align 8
  %out_dim2 = alloca i64, align 8
  %in_dim1 = alloca i64, align 8
  store i64 %in_dim, ptr %in_dim1, align 8
  store i64 %out_dim, ptr %out_dim2, align 8
  %in_dim3 = load i64, ptr %in_dim1, align 8
  %out_dim4 = load i64, ptr %out_dim2, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %in_dim3, ptr %shape_ptr_in, align 8
  %shape_ptr_in5 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 %out_dim4, ptr %shape_ptr_in5, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 true)
  store ptr %randn_res, ptr %W, align 8
  call void @tl_register_tensor(ptr @tensor_name, ptr %randn_res)
  %out_dim6 = load i64, ptr %out_dim2, align 8
  %shape_arr7 = alloca [1 x i64], align 8
  %shape_ptr_in8 = getelementptr inbounds [1 x i64], ptr %shape_arr7, i64 0, i64 0
  store i64 %out_dim6, ptr %shape_ptr_in8, align 8
  %randn_res9 = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr7, i1 true)
  store ptr %randn_res9, ptr %b, align 8
  call void @tl_register_tensor(ptr @tensor_name.36, ptr %randn_res9)
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Linear, ptr null, i32 1) to i64))
  %W10 = load ptr, ptr %W, align 8
  %clone_res = call ptr @tl_tensor_clone(ptr %W10)
  %init_field = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 0
  store ptr %clone_res, ptr %init_field, align 8
  %b11 = load ptr, ptr %b, align 8
  %clone_res12 = call ptr @tl_tensor_clone(ptr %b11)
  %init_field13 = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 1
  store ptr %clone_res12, ptr %init_field13, align 8
  %load_for_free = load ptr, ptr %b, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free14 = load ptr, ptr %W, align 8
  call void @tl_tensor_free(ptr %load_for_free14)
  ret ptr %struct_malloc
}

define ptr @forward(ptr %model, ptr %x) {
entry:
  %y = alloca ptr, align 8
  %out = alloca ptr, align 8
  %x2 = alloca ptr, align 8
  %model1 = alloca ptr, align 8
  store ptr %model, ptr %model1, align 8
  store ptr %x, ptr %x2, align 8
  %x3 = load ptr, ptr %x2, align 8
  %model4 = load ptr, ptr %model1, align 8
  %ptr_W = getelementptr inbounds nuw %Linear, ptr %model4, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  %matmul_res = call ptr @tl_tensor_matmul(ptr %x3, ptr %W)
  store ptr %matmul_res, ptr %out, align 8
  call void @tl_register_tensor(ptr @tensor_name.37, ptr %matmul_res)
  %out5 = load ptr, ptr %out, align 8
  %model6 = load ptr, ptr %model1, align 8
  %ptr_b = getelementptr inbounds nuw %Linear, ptr %model6, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %out5, ptr %b)
  store ptr %binop_res, ptr %y, align 8
  call void @tl_register_tensor(ptr @tensor_name.38, ptr %binop_res)
  %y7 = load ptr, ptr %y, align 8
  %load_for_free = load ptr, ptr %out, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  ret ptr %y7
}

define void @main() {
entry:
  %loss = alloca ptr, align 8
  %probs = alloca ptr, align 8
  %logits = alloca ptr, align 8
  %model = alloca ptr, align 8
  %target = alloca ptr, align 8
  %X = alloca ptr, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 2, ptr %shape_ptr_in, align 8
  %shape_ptr_in1 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 2, ptr %shape_ptr_in1, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 false)
  store ptr %randn_res, ptr %X, align 8
  call void @tl_register_tensor(ptr @tensor_name.39, ptr %randn_res)
  %X2 = load ptr, ptr %X, align 8
  call void @tl_tensor_print(ptr %X2)
  %tensor_data = alloca float, i64 2, align 4
  %elem_ptr = getelementptr inbounds float, ptr %tensor_data, i64 0
  store float 0.000000e+00, ptr %elem_ptr, align 4
  %elem_ptr3 = getelementptr inbounds float, ptr %tensor_data, i64 1
  store float 1.000000e+00, ptr %elem_ptr3, align 4
  %tensor_shape = alloca i64, i64 1, align 8
  %shape_val = getelementptr inbounds i64, ptr %tensor_shape, i64 0
  store i64 2, ptr %shape_val, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %tensor_data, i64 1, ptr %tensor_shape)
  store ptr %new_tensor, ptr %target, align 8
  call void @tl_register_tensor(ptr @tensor_name.40, ptr %new_tensor)
  %target4 = load ptr, ptr %target, align 8
  call void @tl_tensor_print(ptr %target4)
  %call_tmp = call ptr @Linear_new(i64 2, i64 3)
  store ptr %call_tmp, ptr %model, align 8
  %model5 = load ptr, ptr %model, align 8
  %X6 = load ptr, ptr %X, align 8
  %call_tmp7 = call ptr @forward(ptr %model5, ptr %X6)
  store ptr %call_tmp7, ptr %logits, align 8
  call void @tl_register_tensor(ptr @tensor_name.41, ptr %call_tmp7)
  %logits8 = load ptr, ptr %logits, align 8
  call void @tl_tensor_print(ptr %logits8)
  %logits9 = load ptr, ptr %logits, align 8
  %softmax_res = call ptr @tl_tensor_softmax(ptr %logits9, i64 1)
  store ptr %softmax_res, ptr %probs, align 8
  call void @tl_register_tensor(ptr @tensor_name.42, ptr %softmax_res)
  %probs10 = load ptr, ptr %probs, align 8
  call void @tl_tensor_print(ptr %probs10)
  %logits11 = load ptr, ptr %logits, align 8
  %target12 = load ptr, ptr %target, align 8
  %ce_res = call ptr @tl_tensor_cross_entropy(ptr %logits11, ptr %target12)
  store ptr %ce_res, ptr %loss, align 8
  call void @tl_register_tensor(ptr @tensor_name.43, ptr %ce_res)
  %loss13 = load ptr, ptr %loss, align 8
  call void @tl_tensor_print(ptr %loss13)
  %loss14 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss14)
  %model15 = load ptr, ptr %model, align 8
  %ptr_W = getelementptr inbounds nuw %Linear, ptr %model15, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %W)
  call void @tl_tensor_print(ptr %grad_res)
  %load_for_free = load ptr, ptr %probs, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free16 = load ptr, ptr %X, align 8
  call void @tl_tensor_free(ptr %load_for_free16)
  %load_for_free17 = load ptr, ptr %target, align 8
  call void @tl_tensor_free(ptr %load_for_free17)
  %load_for_free18 = load ptr, ptr %loss, align 8
  call void @tl_tensor_free(ptr %load_for_free18)
  %load_for_free19 = load ptr, ptr %logits, align 8
  call void @tl_tensor_free(ptr %load_for_free19)
  ret void
}
