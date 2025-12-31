; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

%Linear = type { ptr, ptr }
%SimpleModel = type { ptr }

@tensor_name = private unnamed_addr constant [3 x i8] c"gw\00", align 1
@tensor_name.68 = private unnamed_addr constant [3 x i8] c"gb\00", align 1
@str_literal = private unnamed_addr constant [36 x i8] c"Test: Linear-only gradient tracking\00", align 1
@tensor_name.69 = private unnamed_addr constant [2 x i8] c"X\00", align 1
@tensor_name.70 = private unnamed_addr constant [2 x i8] c"Y\00", align 1
@str_literal.71 = private unnamed_addr constant [28 x i8] c"Running 3 training steps...\00", align 1
@tensor_name.72 = private unnamed_addr constant [7 x i8] c"logits\00", align 1
@tensor_name.73 = private unnamed_addr constant [5 x i8] c"diff\00", align 1
@tensor_name.74 = private unnamed_addr constant [8 x i8] c"squared\00", align 1
@tensor_name.75 = private unnamed_addr constant [5 x i8] c"loss\00", align 1
@str_literal.76 = private unnamed_addr constant [6 x i8] c"Step:\00", align 1
@str_literal.77 = private unnamed_addr constant [6 x i8] c"Loss:\00", align 1
@str_literal.78 = private unnamed_addr constant [19 x i8] c"Training complete!\00", align 1

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

declare void @tl_tensor_backward.44(ptr)

declare ptr @tl_tensor_grad.45(ptr)

declare ptr @tl_tensor_detach.46(ptr, i1)

declare ptr @tl_tensor_softmax.47(ptr, i64)

declare ptr @tl_tensor_cross_entropy.48(ptr, ptr)

declare void @tl_tensor_sub_assign.49(ptr, ptr)

declare ptr @tl_string_concat.50(ptr, ptr)

declare ptr @tl_file_open.51(ptr, ptr)

declare ptr @tl_file_read_string.52(ptr)

declare void @tl_file_write_string.53(ptr, ptr)

declare void @tl_file_close.54(ptr)

declare ptr @tl_path_new.55(ptr)

declare ptr @tl_path_join.56(ptr, ptr)

declare i1 @tl_path_exists.57(ptr)

declare i1 @tl_path_is_dir.58(ptr)

declare i1 @tl_path_is_file.59(ptr)

declare ptr @tl_path_to_string.60(ptr)

declare void @tl_path_free.61(ptr)

declare i1 @tl_http_download.62(ptr, ptr)

declare ptr @tl_http_get.63(ptr)

declare ptr @tl_env_get.64(ptr)

declare void @tl_env_set.65(ptr, ptr)

declare float @tl_system_time.66()

declare void @tl_system_sleep.67(float)

define ptr @tl_Linear_new(i64 %in_features, i64 %out_features) {
entry:
  %out_features2 = alloca i64, align 8
  %in_features1 = alloca i64, align 8
  store i64 %in_features, ptr %in_features1, align 8
  store i64 %out_features, ptr %out_features2, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Linear, ptr null, i32 1) to i64))
  %in_features3 = load i64, ptr %in_features1, align 8
  %out_features4 = load i64, ptr %out_features2, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %in_features3, ptr %shape_ptr_in, align 8
  %shape_ptr_in5 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 %out_features4, ptr %shape_ptr_in5, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 true)
  %scalar_data = alloca float, align 4
  store float 0x3FB99999A0000000, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor)
  %init_field = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 0
  store ptr %binop_res, ptr %init_field, align 8
  %out_features6 = load i64, ptr %out_features2, align 8
  %shape_arr7 = alloca [1 x i64], align 8
  %shape_ptr_in8 = getelementptr inbounds [1 x i64], ptr %shape_arr7, i64 0, i64 0
  store i64 %out_features6, ptr %shape_ptr_in8, align 8
  %randn_res9 = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr7, i1 true)
  %scalar_data10 = alloca float, align 4
  store float 0.000000e+00, ptr %scalar_data10, align 4
  %scalar_shape11 = alloca i64, i64 0, align 8
  %scalar_tensor12 = call ptr @tl_tensor_new(ptr %scalar_data10, i64 0, ptr %scalar_shape11)
  %binop_res13 = call ptr @tl_tensor_mul(ptr %randn_res9, ptr %scalar_tensor12)
  %init_field14 = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 1
  store ptr %binop_res13, ptr %init_field14, align 8
  ret ptr %struct_malloc
}

define ptr @tl_Linear_forward(ptr %self, ptr %x) {
entry:
  %x2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %x3 = load ptr, ptr %x2, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_weight = getelementptr inbounds nuw %Linear, ptr %self4, i32 0, i32 0
  %weight = load ptr, ptr %ptr_weight, align 8
  %matmul_res = call ptr @tl_tensor_matmul(ptr %x3, ptr %weight)
  %self5 = load ptr, ptr %self1, align 8
  %ptr_bias = getelementptr inbounds nuw %Linear, ptr %self5, i32 0, i32 1
  %bias = load ptr, ptr %ptr_bias, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %matmul_res, ptr %bias)
  ret ptr %binop_res
}

define void @tl_Linear_step(ptr %self, float %lr) {
entry:
  %gb = alloca ptr, align 8
  %gw = alloca ptr, align 8
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  %ptr_weight = getelementptr inbounds nuw %Linear, ptr %self3, i32 0, i32 0
  %weight = load ptr, ptr %ptr_weight, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %weight)
  store ptr %grad_res, ptr %gw, align 8
  call void @tl_register_tensor(ptr @tensor_name, ptr %grad_res)
  %self4 = load ptr, ptr %self1, align 8
  %ptr_bias = getelementptr inbounds nuw %Linear, ptr %self4, i32 0, i32 1
  %bias = load ptr, ptr %ptr_bias, align 8
  %grad_res5 = call ptr @tl_tensor_grad(ptr %bias)
  store ptr %grad_res5, ptr %gb, align 8
  call void @tl_register_tensor(ptr @tensor_name.68, ptr %grad_res5)
  %self6 = load ptr, ptr %self1, align 8
  %ptr_weight7 = getelementptr inbounds nuw %Linear, ptr %self6, i32 0, i32 0
  %self8 = load ptr, ptr %self1, align 8
  %ptr_weight9 = getelementptr inbounds nuw %Linear, ptr %self8, i32 0, i32 0
  %weight10 = load ptr, ptr %ptr_weight9, align 8
  %gw11 = load ptr, ptr %gw, align 8
  %lr12 = load float, ptr %lr2, align 4
  %scalar_data = alloca float, align 4
  store float %lr12, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %gw11, ptr %scalar_tensor)
  %binop_res13 = call ptr @tl_tensor_sub(ptr %weight10, ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res13, i1 true)
  %old_field_val = load ptr, ptr %ptr_weight7, align 8
  call void @tl_tensor_free(ptr %old_field_val)
  store ptr %detach_res, ptr %ptr_weight7, align 8
  %self14 = load ptr, ptr %self1, align 8
  %ptr_bias15 = getelementptr inbounds nuw %Linear, ptr %self14, i32 0, i32 1
  %self16 = load ptr, ptr %self1, align 8
  %ptr_bias17 = getelementptr inbounds nuw %Linear, ptr %self16, i32 0, i32 1
  %bias18 = load ptr, ptr %ptr_bias17, align 8
  %gb19 = load ptr, ptr %gb, align 8
  %lr20 = load float, ptr %lr2, align 4
  %scalar_data21 = alloca float, align 4
  store float %lr20, ptr %scalar_data21, align 4
  %scalar_shape22 = alloca i64, i64 0, align 8
  %scalar_tensor23 = call ptr @tl_tensor_new(ptr %scalar_data21, i64 0, ptr %scalar_shape22)
  %binop_res24 = call ptr @tl_tensor_mul(ptr %gb19, ptr %scalar_tensor23)
  %binop_res25 = call ptr @tl_tensor_sub(ptr %bias18, ptr %binop_res24)
  %detach_res26 = call ptr @tl_tensor_detach(ptr %binop_res25, i1 true)
  %old_field_val27 = load ptr, ptr %ptr_bias15, align 8
  call void @tl_tensor_free(ptr %old_field_val27)
  store ptr %detach_res26, ptr %ptr_bias15, align 8
  ret void
}

define ptr @tl_SimpleModel_new() {
entry:
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%SimpleModel, ptr null, i32 1) to i64))
  %static_call = call ptr @tl_Linear_new(i64 10, i64 5)
  %init_field = getelementptr inbounds nuw %SimpleModel, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  ret ptr %struct_malloc
}

define ptr @tl_SimpleModel_forward(ptr %self, ptr %x) {
entry:
  %x2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_linear = getelementptr inbounds nuw %SimpleModel, ptr %self3, i32 0, i32 0
  %linear = load ptr, ptr %ptr_linear, align 8
  %x4 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_Linear_forward(ptr %linear, ptr %x4)
  ret ptr %call_method
}

define void @tl_SimpleModel_step(ptr %self, float %lr) {
entry:
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  %ptr_linear = getelementptr inbounds nuw %SimpleModel, ptr %self3, i32 0, i32 0
  %linear = load ptr, ptr %ptr_linear, align 8
  %lr4 = load float, ptr %lr2, align 4
  call void @tl_Linear_step(ptr %linear, float %lr4)
  ret void
}

define void @main() {
entry:
  %loss = alloca ptr, align 8
  %squared = alloca ptr, align 8
  %diff = alloca ptr, align 8
  %logits = alloca ptr, align 8
  %i = alloca i64, align 8
  %Y = alloca ptr, align 8
  %X = alloca ptr, align 8
  %lr = alloca float, align 4
  %model = alloca ptr, align 8
  call void @tl_print_string(ptr @str_literal)
  %static_call = call ptr @tl_SimpleModel_new()
  store ptr %static_call, ptr %model, align 8
  store float 0x3F847AE140000000, ptr %lr, align 4
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 4, ptr %shape_ptr_in, align 8
  %shape_ptr_in1 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 10, ptr %shape_ptr_in1, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 false)
  store ptr %randn_res, ptr %X, align 8
  call void @tl_register_tensor(ptr @tensor_name.69, ptr %randn_res)
  %shape_arr2 = alloca [2 x i64], align 8
  %shape_ptr_in3 = getelementptr inbounds [2 x i64], ptr %shape_arr2, i64 0, i64 0
  store i64 4, ptr %shape_ptr_in3, align 8
  %shape_ptr_in4 = getelementptr inbounds [2 x i64], ptr %shape_arr2, i64 0, i64 1
  store i64 5, ptr %shape_ptr_in4, align 8
  %randn_res5 = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr2, i1 false)
  store ptr %randn_res5, ptr %Y, align 8
  call void @tl_register_tensor(ptr @tensor_name.70, ptr %randn_res5)
  call void @tl_print_string(ptr @str_literal.71)
  br label %for_header

for_header:                                       ; preds = %for_body, %entry
  %for_idx = phi i64 [ 0, %entry ], [ %next_idx, %for_body ]
  %for_cond = icmp slt i64 %for_idx, 3
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  store i64 %for_idx, ptr %i, align 8
  %model6 = load ptr, ptr %model, align 8
  %X7 = load ptr, ptr %X, align 8
  %call_method = call ptr @tl_SimpleModel_forward(ptr %model6, ptr %X7)
  store ptr %call_method, ptr %logits, align 8
  call void @tl_register_tensor(ptr @tensor_name.72, ptr %call_method)
  %logits8 = load ptr, ptr %logits, align 8
  %Y9 = load ptr, ptr %Y, align 8
  %binop_res = call ptr @tl_tensor_sub(ptr %logits8, ptr %Y9)
  store ptr %binop_res, ptr %diff, align 8
  call void @tl_register_tensor(ptr @tensor_name.73, ptr %binop_res)
  %diff10 = load ptr, ptr %diff, align 8
  %diff11 = load ptr, ptr %diff, align 8
  %binop_res12 = call ptr @tl_tensor_mul(ptr %diff10, ptr %diff11)
  store ptr %binop_res12, ptr %squared, align 8
  call void @tl_register_tensor(ptr @tensor_name.74, ptr %binop_res12)
  %squared13 = load ptr, ptr %squared, align 8
  %sum_res = call ptr @tl_tensor_sum(ptr %squared13)
  store ptr %sum_res, ptr %loss, align 8
  call void @tl_register_tensor(ptr @tensor_name.75, ptr %sum_res)
  call void @tl_print_string(ptr @str_literal.76)
  %i14 = load i64, ptr %i, align 8
  call void @tl_print_i64(i64 %i14)
  call void @tl_print_string(ptr @str_literal.77)
  %loss15 = load ptr, ptr %loss, align 8
  call void @tl_tensor_print(ptr %loss15)
  %loss16 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss16)
  %model17 = load ptr, ptr %model, align 8
  %lr18 = load float, ptr %lr, align 4
  call void @tl_SimpleModel_step(ptr %model17, float %lr18)
  %load_for_free = load ptr, ptr %logits, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free19 = load ptr, ptr %diff, align 8
  call void @tl_tensor_free(ptr %load_for_free19)
  %load_for_free20 = load ptr, ptr %squared, align 8
  call void @tl_tensor_free(ptr %load_for_free20)
  %load_for_free21 = load ptr, ptr %loss, align 8
  call void @tl_tensor_free(ptr %load_for_free21)
  %next_idx = add i64 %for_idx, 1
  br label %for_header

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal.78)
  %load_for_free22 = load ptr, ptr %X, align 8
  call void @tl_tensor_free(ptr %load_for_free22)
  %load_for_free23 = load ptr, ptr %Y, align 8
  call void @tl_tensor_free(ptr %load_for_free23)
  ret void
}
