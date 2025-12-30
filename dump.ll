; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

%Linear = type { ptr, ptr }
%SGD = type { float }

@tensor_name = private unnamed_addr constant [3 x i8] c"gW\00", align 1
@tensor_name.34 = private unnamed_addr constant [3 x i8] c"gb\00", align 1
@tensor_name.35 = private unnamed_addr constant [7 x i8] c"step_W\00", align 1
@tensor_name.36 = private unnamed_addr constant [6 x i8] c"new_W\00", align 1
@tensor_name.37 = private unnamed_addr constant [7 x i8] c"step_b\00", align 1
@tensor_name.38 = private unnamed_addr constant [6 x i8] c"new_b\00", align 1
@tensor_name.39 = private unnamed_addr constant [3 x i8] c"dW\00", align 1
@tensor_name.40 = private unnamed_addr constant [3 x i8] c"db\00", align 1
@tensor_name.41 = private unnamed_addr constant [2 x i8] c"W\00", align 1
@tensor_name.42 = private unnamed_addr constant [2 x i8] c"b\00", align 1
@tensor_name.43 = private unnamed_addr constant [2 x i8] c"y\00", align 1
@tensor_name.44 = private unnamed_addr constant [2 x i8] c"X\00", align 1
@tensor_name.45 = private unnamed_addr constant [7 x i8] c"true_W\00", align 1
@tensor_name.46 = private unnamed_addr constant [7 x i8] c"true_b\00", align 1
@tensor_name.47 = private unnamed_addr constant [7 x i8] c"y_pred\00", align 1
@tensor_name.48 = private unnamed_addr constant [5 x i8] c"loss\00", align 1
@tensor_name.49 = private unnamed_addr constant [7 x i8] c"y_pred\00", align 1
@tensor_name.50 = private unnamed_addr constant [5 x i8] c"loss\00", align 1
@tensor_name.51 = private unnamed_addr constant [7 x i8] c"y_pred\00", align 1
@tensor_name.52 = private unnamed_addr constant [5 x i8] c"loss\00", align 1
@tensor_name.53 = private unnamed_addr constant [7 x i8] c"y_pred\00", align 1
@tensor_name.54 = private unnamed_addr constant [5 x i8] c"loss\00", align 1
@tensor_name.55 = private unnamed_addr constant [7 x i8] c"y_pred\00", align 1
@tensor_name.56 = private unnamed_addr constant [5 x i8] c"loss\00", align 1

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

declare void @tl_tensor_sub_assign.33(ptr, ptr)

define ptr @SGD_step(ptr %0, ptr %1) {
entry:
  %db = alloca ptr, align 8
  %dW = alloca ptr, align 8
  %new_b = alloca ptr, align 8
  %step_b = alloca ptr, align 8
  %new_W = alloca ptr, align 8
  %step_W = alloca ptr, align 8
  %gb = alloca ptr, align 8
  %gW = alloca ptr, align 8
  %model = alloca ptr, align 8
  %self = alloca ptr, align 8
  store ptr %0, ptr %self, align 8
  store ptr %1, ptr %model, align 8
  %model1 = load ptr, ptr %model, align 8
  %ptr_W = getelementptr inbounds nuw %Linear, ptr %model1, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %W)
  store ptr %grad_res, ptr %gW, align 8
  call void @tl_register_tensor(ptr @tensor_name, ptr %grad_res)
  %model2 = load ptr, ptr %model, align 8
  %ptr_b = getelementptr inbounds nuw %Linear, ptr %model2, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res3 = call ptr @tl_tensor_grad(ptr %b)
  store ptr %grad_res3, ptr %gb, align 8
  call void @tl_register_tensor(ptr @tensor_name.34, ptr %grad_res3)
  %gW4 = load ptr, ptr %gW, align 8
  %self5 = load ptr, ptr %self, align 8
  %ptr_lr = getelementptr inbounds nuw %SGD, ptr %self5, i32 0, i32 0
  %lr = load float, ptr %ptr_lr, align 4
  %scalar_data = alloca float, align 4
  store float %lr, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %gW4, ptr %scalar_tensor)
  store ptr %binop_res, ptr %step_W, align 8
  call void @tl_register_tensor(ptr @tensor_name.35, ptr %binop_res)
  %model6 = load ptr, ptr %model, align 8
  %ptr_W7 = getelementptr inbounds nuw %Linear, ptr %model6, i32 0, i32 0
  %W8 = load ptr, ptr %ptr_W7, align 8
  %step_W9 = load ptr, ptr %step_W, align 8
  %binop_res10 = call ptr @tl_tensor_sub(ptr %W8, ptr %step_W9)
  store ptr %binop_res10, ptr %new_W, align 8
  call void @tl_register_tensor(ptr @tensor_name.36, ptr %binop_res10)
  %gb11 = load ptr, ptr %gb, align 8
  %self12 = load ptr, ptr %self, align 8
  %ptr_lr13 = getelementptr inbounds nuw %SGD, ptr %self12, i32 0, i32 0
  %lr14 = load float, ptr %ptr_lr13, align 4
  %scalar_data15 = alloca float, align 4
  store float %lr14, ptr %scalar_data15, align 4
  %scalar_shape16 = alloca i64, i64 0, align 8
  %scalar_tensor17 = call ptr @tl_tensor_new(ptr %scalar_data15, i64 0, ptr %scalar_shape16)
  %binop_res18 = call ptr @tl_tensor_mul(ptr %gb11, ptr %scalar_tensor17)
  store ptr %binop_res18, ptr %step_b, align 8
  call void @tl_register_tensor(ptr @tensor_name.37, ptr %binop_res18)
  %model19 = load ptr, ptr %model, align 8
  %ptr_b20 = getelementptr inbounds nuw %Linear, ptr %model19, i32 0, i32 1
  %b21 = load ptr, ptr %ptr_b20, align 8
  %step_b22 = load ptr, ptr %step_b, align 8
  %binop_res23 = call ptr @tl_tensor_sub(ptr %b21, ptr %step_b22)
  store ptr %binop_res23, ptr %new_b, align 8
  call void @tl_register_tensor(ptr @tensor_name.38, ptr %binop_res23)
  %new_W24 = load ptr, ptr %new_W, align 8
  %detach_res = call ptr @tl_tensor_detach(ptr %new_W24, i1 true)
  store ptr %detach_res, ptr %dW, align 8
  call void @tl_register_tensor(ptr @tensor_name.39, ptr %detach_res)
  %new_b25 = load ptr, ptr %new_b, align 8
  %detach_res26 = call ptr @tl_tensor_detach(ptr %new_b25, i1 true)
  store ptr %detach_res26, ptr %db, align 8
  call void @tl_register_tensor(ptr @tensor_name.40, ptr %detach_res26)
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Linear, ptr null, i32 1) to i64))
  %dW27 = load ptr, ptr %dW, align 8
  %clone_res = call ptr @tl_tensor_clone(ptr %dW27)
  %init_field = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 0
  store ptr %clone_res, ptr %init_field, align 8
  %db28 = load ptr, ptr %db, align 8
  %clone_res29 = call ptr @tl_tensor_clone(ptr %db28)
  %init_field30 = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 1
  store ptr %clone_res29, ptr %init_field30, align 8
  %load_for_free = load ptr, ptr %step_W, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free31 = load ptr, ptr %new_W, align 8
  call void @tl_tensor_free(ptr %load_for_free31)
  %load_for_free32 = load ptr, ptr %new_b, align 8
  call void @tl_tensor_free(ptr %load_for_free32)
  %load_for_free33 = load ptr, ptr %db, align 8
  call void @tl_tensor_free(ptr %load_for_free33)
  %load_for_free34 = load ptr, ptr %gb, align 8
  call void @tl_tensor_free(ptr %load_for_free34)
  %load_for_free35 = load ptr, ptr %gW, align 8
  call void @tl_tensor_free(ptr %load_for_free35)
  %load_for_free36 = load ptr, ptr %step_b, align 8
  call void @tl_tensor_free(ptr %load_for_free36)
  %load_for_free37 = load ptr, ptr %dW, align 8
  call void @tl_tensor_free(ptr %load_for_free37)
  ret ptr %struct_malloc
}

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
  call void @tl_register_tensor(ptr @tensor_name.41, ptr %randn_res)
  %out_dim6 = load i64, ptr %out_dim2, align 8
  %shape_arr7 = alloca [1 x i64], align 8
  %shape_ptr_in8 = getelementptr inbounds [1 x i64], ptr %shape_arr7, i64 0, i64 0
  store i64 %out_dim6, ptr %shape_ptr_in8, align 8
  %randn_res9 = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr7, i1 true)
  store ptr %randn_res9, ptr %b, align 8
  call void @tl_register_tensor(ptr @tensor_name.42, ptr %randn_res9)
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
  %x2 = alloca ptr, align 8
  %model1 = alloca ptr, align 8
  store ptr %model, ptr %model1, align 8
  store ptr %x, ptr %x2, align 8
  %x3 = load ptr, ptr %x2, align 8
  %model4 = load ptr, ptr %model1, align 8
  %ptr_W = getelementptr inbounds nuw %Linear, ptr %model4, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  %matmul_res = call ptr @tl_tensor_matmul(ptr %x3, ptr %W)
  store ptr %matmul_res, ptr %y, align 8
  call void @tl_register_tensor(ptr @tensor_name.43, ptr %matmul_res)
  %y5 = load ptr, ptr %y, align 8
  %model6 = load ptr, ptr %model1, align 8
  %ptr_b = getelementptr inbounds nuw %Linear, ptr %model6, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %y5, ptr %b)
  %load_for_free = load ptr, ptr %y, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  ret ptr %binop_res
}

define void @main() {
entry:
  %model79 = alloca ptr, align 8
  %loss73 = alloca ptr, align 8
  %y_pred68 = alloca ptr, align 8
  %model64 = alloca ptr, align 8
  %loss58 = alloca ptr, align 8
  %y_pred53 = alloca ptr, align 8
  %model49 = alloca ptr, align 8
  %loss43 = alloca ptr, align 8
  %y_pred38 = alloca ptr, align 8
  %model34 = alloca ptr, align 8
  %loss28 = alloca ptr, align 8
  %y_pred23 = alloca ptr, align 8
  %model19 = alloca ptr, align 8
  %loss = alloca ptr, align 8
  %y_pred = alloca ptr, align 8
  %true_b = alloca ptr, align 8
  %true_W = alloca ptr, align 8
  %X = alloca ptr, align 8
  %sgd = alloca ptr, align 8
  %model = alloca ptr, align 8
  %call_tmp = call ptr @Linear_new(i64 2, i64 1)
  store ptr %call_tmp, ptr %model, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%SGD, ptr null, i32 1) to i64))
  %init_field = getelementptr inbounds nuw %SGD, ptr %struct_malloc, i32 0, i32 0
  store float 0x3F847AE140000000, ptr %init_field, align 4
  store ptr %struct_malloc, ptr %sgd, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 16, ptr %shape_ptr_in, align 8
  %shape_ptr_in1 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 2, ptr %shape_ptr_in1, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 false)
  store ptr %randn_res, ptr %X, align 8
  call void @tl_register_tensor(ptr @tensor_name.44, ptr %randn_res)
  %shape_arr2 = alloca [2 x i64], align 8
  %shape_ptr_in3 = getelementptr inbounds [2 x i64], ptr %shape_arr2, i64 0, i64 0
  store i64 2, ptr %shape_ptr_in3, align 8
  %shape_ptr_in4 = getelementptr inbounds [2 x i64], ptr %shape_arr2, i64 0, i64 1
  store i64 1, ptr %shape_ptr_in4, align 8
  %randn_res5 = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr2, i1 false)
  store ptr %randn_res5, ptr %true_W, align 8
  call void @tl_register_tensor(ptr @tensor_name.45, ptr %randn_res5)
  %shape_arr6 = alloca [1 x i64], align 8
  %shape_ptr_in7 = getelementptr inbounds [1 x i64], ptr %shape_arr6, i64 0, i64 0
  store i64 1, ptr %shape_ptr_in7, align 8
  %randn_res8 = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr6, i1 false)
  store ptr %randn_res8, ptr %true_b, align 8
  call void @tl_register_tensor(ptr @tensor_name.46, ptr %randn_res8)
  %model9 = load ptr, ptr %model, align 8
  %ptr_W = getelementptr inbounds nuw %Linear, ptr %model9, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  call void @tl_tensor_print(ptr %W)
  %model10 = load ptr, ptr %model, align 8
  %X11 = load ptr, ptr %X, align 8
  %call_tmp12 = call ptr @forward(ptr %model10, ptr %X11)
  store ptr %call_tmp12, ptr %y_pred, align 8
  call void @tl_register_tensor(ptr @tensor_name.47, ptr %call_tmp12)
  %y_pred13 = load ptr, ptr %y_pred, align 8
  %y_pred14 = load ptr, ptr %y_pred, align 8
  %binop_res = call ptr @tl_tensor_mul(ptr %y_pred13, ptr %y_pred14)
  %sum_res = call ptr @tl_tensor_sum(ptr %binop_res)
  store ptr %sum_res, ptr %loss, align 8
  call void @tl_register_tensor(ptr @tensor_name.48, ptr %sum_res)
  %loss15 = load ptr, ptr %loss, align 8
  call void @tl_tensor_print(ptr %loss15)
  %loss16 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss16)
  %sgd17 = load ptr, ptr %sgd, align 8
  %model18 = load ptr, ptr %model, align 8
  %call_method = call ptr @SGD_step(ptr %sgd17, ptr %model18)
  store ptr %call_method, ptr %model19, align 8
  %model20 = load ptr, ptr %model19, align 8
  %X21 = load ptr, ptr %X, align 8
  %call_tmp22 = call ptr @forward(ptr %model20, ptr %X21)
  store ptr %call_tmp22, ptr %y_pred23, align 8
  call void @tl_register_tensor(ptr @tensor_name.49, ptr %call_tmp22)
  %y_pred24 = load ptr, ptr %y_pred23, align 8
  %y_pred25 = load ptr, ptr %y_pred23, align 8
  %binop_res26 = call ptr @tl_tensor_mul(ptr %y_pred24, ptr %y_pred25)
  %sum_res27 = call ptr @tl_tensor_sum(ptr %binop_res26)
  store ptr %sum_res27, ptr %loss28, align 8
  call void @tl_register_tensor(ptr @tensor_name.50, ptr %sum_res27)
  %loss29 = load ptr, ptr %loss28, align 8
  call void @tl_tensor_print(ptr %loss29)
  %loss30 = load ptr, ptr %loss28, align 8
  call void @tl_tensor_backward(ptr %loss30)
  %sgd31 = load ptr, ptr %sgd, align 8
  %model32 = load ptr, ptr %model19, align 8
  %call_method33 = call ptr @SGD_step(ptr %sgd31, ptr %model32)
  store ptr %call_method33, ptr %model34, align 8
  %model35 = load ptr, ptr %model34, align 8
  %X36 = load ptr, ptr %X, align 8
  %call_tmp37 = call ptr @forward(ptr %model35, ptr %X36)
  store ptr %call_tmp37, ptr %y_pred38, align 8
  call void @tl_register_tensor(ptr @tensor_name.51, ptr %call_tmp37)
  %y_pred39 = load ptr, ptr %y_pred38, align 8
  %y_pred40 = load ptr, ptr %y_pred38, align 8
  %binop_res41 = call ptr @tl_tensor_mul(ptr %y_pred39, ptr %y_pred40)
  %sum_res42 = call ptr @tl_tensor_sum(ptr %binop_res41)
  store ptr %sum_res42, ptr %loss43, align 8
  call void @tl_register_tensor(ptr @tensor_name.52, ptr %sum_res42)
  %loss44 = load ptr, ptr %loss43, align 8
  call void @tl_tensor_print(ptr %loss44)
  %loss45 = load ptr, ptr %loss43, align 8
  call void @tl_tensor_backward(ptr %loss45)
  %sgd46 = load ptr, ptr %sgd, align 8
  %model47 = load ptr, ptr %model34, align 8
  %call_method48 = call ptr @SGD_step(ptr %sgd46, ptr %model47)
  store ptr %call_method48, ptr %model49, align 8
  %model50 = load ptr, ptr %model49, align 8
  %X51 = load ptr, ptr %X, align 8
  %call_tmp52 = call ptr @forward(ptr %model50, ptr %X51)
  store ptr %call_tmp52, ptr %y_pred53, align 8
  call void @tl_register_tensor(ptr @tensor_name.53, ptr %call_tmp52)
  %y_pred54 = load ptr, ptr %y_pred53, align 8
  %y_pred55 = load ptr, ptr %y_pred53, align 8
  %binop_res56 = call ptr @tl_tensor_mul(ptr %y_pred54, ptr %y_pred55)
  %sum_res57 = call ptr @tl_tensor_sum(ptr %binop_res56)
  store ptr %sum_res57, ptr %loss58, align 8
  call void @tl_register_tensor(ptr @tensor_name.54, ptr %sum_res57)
  %loss59 = load ptr, ptr %loss58, align 8
  call void @tl_tensor_print(ptr %loss59)
  %loss60 = load ptr, ptr %loss58, align 8
  call void @tl_tensor_backward(ptr %loss60)
  %sgd61 = load ptr, ptr %sgd, align 8
  %model62 = load ptr, ptr %model49, align 8
  %call_method63 = call ptr @SGD_step(ptr %sgd61, ptr %model62)
  store ptr %call_method63, ptr %model64, align 8
  %model65 = load ptr, ptr %model64, align 8
  %X66 = load ptr, ptr %X, align 8
  %call_tmp67 = call ptr @forward(ptr %model65, ptr %X66)
  store ptr %call_tmp67, ptr %y_pred68, align 8
  call void @tl_register_tensor(ptr @tensor_name.55, ptr %call_tmp67)
  %y_pred69 = load ptr, ptr %y_pred68, align 8
  %y_pred70 = load ptr, ptr %y_pred68, align 8
  %binop_res71 = call ptr @tl_tensor_mul(ptr %y_pred69, ptr %y_pred70)
  %sum_res72 = call ptr @tl_tensor_sum(ptr %binop_res71)
  store ptr %sum_res72, ptr %loss73, align 8
  call void @tl_register_tensor(ptr @tensor_name.56, ptr %sum_res72)
  %loss74 = load ptr, ptr %loss73, align 8
  call void @tl_tensor_print(ptr %loss74)
  %loss75 = load ptr, ptr %loss73, align 8
  call void @tl_tensor_backward(ptr %loss75)
  %sgd76 = load ptr, ptr %sgd, align 8
  %model77 = load ptr, ptr %model64, align 8
  %call_method78 = call ptr @SGD_step(ptr %sgd76, ptr %model77)
  store ptr %call_method78, ptr %model79, align 8
  %model80 = load ptr, ptr %model79, align 8
  %ptr_W81 = getelementptr inbounds nuw %Linear, ptr %model80, i32 0, i32 0
  %W82 = load ptr, ptr %ptr_W81, align 8
  call void @tl_tensor_print(ptr %W82)
  %model83 = load ptr, ptr %model79, align 8
  %ptr_b = getelementptr inbounds nuw %Linear, ptr %model83, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  call void @tl_tensor_print(ptr %b)
  %load_for_free = load ptr, ptr %loss73, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free84 = load ptr, ptr %y_pred68, align 8
  call void @tl_tensor_free(ptr %load_for_free84)
  %load_for_free85 = load ptr, ptr %true_b, align 8
  call void @tl_tensor_free(ptr %load_for_free85)
  %load_for_free86 = load ptr, ptr %X, align 8
  call void @tl_tensor_free(ptr %load_for_free86)
  %load_for_free87 = load ptr, ptr %true_W, align 8
  call void @tl_tensor_free(ptr %load_for_free87)
  ret void
}
