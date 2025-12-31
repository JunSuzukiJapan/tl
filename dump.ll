; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

%Emb = type { ptr }

@str_literal = private unnamed_addr constant [40 x i8] c"Test: Embedding gradient through struct\00", align 1
@tensor_name = private unnamed_addr constant [4 x i8] c"idx\00", align 1
@tensor_name.68 = private unnamed_addr constant [4 x i8] c"out\00", align 1
@tensor_name.69 = private unnamed_addr constant [5 x i8] c"loss\00", align 1
@str_literal.70 = private unnamed_addr constant [6 x i8] c"Loss:\00", align 1
@tensor_name.71 = private unnamed_addr constant [3 x i8] c"gw\00", align 1
@str_literal.72 = private unnamed_addr constant [21 x i8] c"Emb weight gradient:\00", align 1

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

define ptr @tl_Emb_new(i64 %vocab, i64 %d) {
entry:
  %d2 = alloca i64, align 8
  %vocab1 = alloca i64, align 8
  store i64 %vocab, ptr %vocab1, align 8
  store i64 %d, ptr %d2, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Emb, ptr null, i32 1) to i64))
  %vocab3 = load i64, ptr %vocab1, align 8
  %d4 = load i64, ptr %d2, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %vocab3, ptr %shape_ptr_in, align 8
  %shape_ptr_in5 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 %d4, ptr %shape_ptr_in5, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 true)
  %scalar_data = alloca float, align 4
  store float 0x3FB99999A0000000, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor)
  %init_field = getelementptr inbounds nuw %Emb, ptr %struct_malloc, i32 0, i32 0
  store ptr %binop_res, ptr %init_field, align 8
  ret ptr %struct_malloc
}

define ptr @tl_Emb_forward(ptr %self, ptr %idx) {
entry:
  %idx2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %idx, ptr %idx2, align 8
  %idx3 = load ptr, ptr %idx2, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_weight = getelementptr inbounds nuw %Emb, ptr %self4, i32 0, i32 0
  %weight = load ptr, ptr %ptr_weight, align 8
  %emb_res = call ptr @tl_tensor_embedding(ptr %idx3, ptr %weight)
  ret ptr %emb_res
}

define void @main() {
entry:
  %gw = alloca ptr, align 8
  %loss = alloca ptr, align 8
  %out = alloca ptr, align 8
  %idx = alloca ptr, align 8
  %emb = alloca ptr, align 8
  call void @tl_print_string(ptr @str_literal)
  %static_call = call ptr @tl_Emb_new(i64 14, i64 16)
  store ptr %static_call, ptr %emb, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 2, ptr %shape_ptr_in, align 8
  %shape_ptr_in1 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 4, ptr %shape_ptr_in1, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 false)
  %scalar_data = alloca float, align 4
  store float 0.000000e+00, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor)
  %scalar_data2 = alloca float, align 4
  store float 1.000000e+00, ptr %scalar_data2, align 4
  %scalar_shape3 = alloca i64, i64 0, align 8
  %scalar_tensor4 = call ptr @tl_tensor_new(ptr %scalar_data2, i64 0, ptr %scalar_shape3)
  %binop_res5 = call ptr @tl_tensor_add(ptr %binop_res, ptr %scalar_tensor4)
  store ptr %binop_res5, ptr %idx, align 8
  call void @tl_register_tensor(ptr @tensor_name, ptr %binop_res5)
  %emb6 = load ptr, ptr %emb, align 8
  %idx7 = load ptr, ptr %idx, align 8
  %call_method = call ptr @tl_Emb_forward(ptr %emb6, ptr %idx7)
  store ptr %call_method, ptr %out, align 8
  call void @tl_register_tensor(ptr @tensor_name.68, ptr %call_method)
  %out8 = load ptr, ptr %out, align 8
  %sum_res = call ptr @tl_tensor_sum(ptr %out8)
  store ptr %sum_res, ptr %loss, align 8
  call void @tl_register_tensor(ptr @tensor_name.69, ptr %sum_res)
  call void @tl_print_string(ptr @str_literal.70)
  %loss9 = load ptr, ptr %loss, align 8
  call void @tl_tensor_print(ptr %loss9)
  %loss10 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss10)
  %emb11 = load ptr, ptr %emb, align 8
  %ptr_weight = getelementptr inbounds nuw %Emb, ptr %emb11, i32 0, i32 0
  %weight = load ptr, ptr %ptr_weight, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %weight)
  store ptr %grad_res, ptr %gw, align 8
  call void @tl_register_tensor(ptr @tensor_name.71, ptr %grad_res)
  call void @tl_print_string(ptr @str_literal.72)
  %gw12 = load ptr, ptr %gw, align 8
  call void @tl_tensor_print(ptr %gw12)
  %load_for_free = load ptr, ptr %loss, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free13 = load ptr, ptr %idx, align 8
  call void @tl_tensor_free(ptr %load_for_free13)
  %load_for_free14 = load ptr, ptr %out, align 8
  call void @tl_tensor_free(ptr %load_for_free14)
  %load_for_free15 = load ptr, ptr %gw, align 8
  call void @tl_tensor_free(ptr %load_for_free15)
  ret void
}
