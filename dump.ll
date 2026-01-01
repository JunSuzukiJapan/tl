; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

%Linear = type { ptr, ptr }
%Embedding = type { ptr }
%LayerNorm = type { ptr, ptr }
%CausalSelfAttention = type { ptr, ptr }
%MLP = type { ptr, ptr }
%Block = type { ptr, ptr, ptr, ptr }
%GPT = type { ptr, ptr, ptr, ptr }

@tensor_name = private unnamed_addr constant [3 x i8] c"gW\00", align 1
@tensor_name.78 = private unnamed_addr constant [3 x i8] c"gb\00", align 1
@tensor_name.79 = private unnamed_addr constant [2 x i8] c"g\00", align 1
@tensor_name.80 = private unnamed_addr constant [3 x i8] c"gb\00", align 1
@tensor_name.81 = private unnamed_addr constant [2 x i8] c"q\00", align 1
@tensor_name.82 = private unnamed_addr constant [2 x i8] c"k\00", align 1
@tensor_name.83 = private unnamed_addr constant [2 x i8] c"v\00", align 1
@tensor_name.84 = private unnamed_addr constant [2 x i8] c"y\00", align 1
@tensor_name.85 = private unnamed_addr constant [2 x i8] c"x\00", align 1
@tensor_name.86 = private unnamed_addr constant [11 x i8] c"total_loss\00", align 1
@tensor_name.87 = private unnamed_addr constant [5 x i8] c"data\00", align 1
@tensor_name.88 = private unnamed_addr constant [7 x i8] c"target\00", align 1
@tensor_name.89 = private unnamed_addr constant [2 x i8] c"X\00", align 1
@tensor_name.90 = private unnamed_addr constant [2 x i8] c"Y\00", align 1
@tensor_name.91 = private unnamed_addr constant [7 x i8] c"logits\00", align 1
@tensor_name.92 = private unnamed_addr constant [12 x i8] c"logits_flat\00", align 1
@tensor_name.93 = private unnamed_addr constant [7 x i8] c"Y_flat\00", align 1
@tensor_name.94 = private unnamed_addr constant [5 x i8] c"loss\00", align 1
@str_literal = private unnamed_addr constant [7 x i8] c"Epoch:\00", align 1
@str_literal.95 = private unnamed_addr constant [6 x i8] c"Loss:\00", align 1
@str_literal.96 = private unnamed_addr constant [62 x i8] c"Training 2-digit addition (0-99) - Function-separated version\00", align 1
@str_literal.97 = private unnamed_addr constant [19 x i8] c"Training Complete!\00", align 1
@str_literal.98 = private unnamed_addr constant [25 x i8] c"model_2digit.safetensors\00", align 1

declare void @tl_print_i64(i64)

declare void @tl_print_f32(float)

declare void @tl_print_string(ptr)

declare ptr @malloc(i64)

declare ptr @calloc(i64, i64)

declare void @free(ptr)

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

declare ptr @tl_tensor_pow(ptr, ptr)

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

declare void @tl_tensor_save(ptr, ptr)

declare ptr @tl_tensor_load(ptr)

declare void @tl_save_all_params(ptr)

declare void @tl_load_all_params(ptr)

declare void @tl_tensor_sub_assign.1(ptr, ptr)

declare void @tl_add_parameter(ptr, ptr)

declare ptr @tl_register_parameter(ptr)

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

declare void @free.7(ptr)

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

declare ptr @tl_tensor_pow.22(ptr, ptr)

declare ptr @tl_tensor_sqrt.23(ptr)

declare ptr @tl_tensor_sin.24(ptr)

declare ptr @tl_tensor_cos.25(ptr)

declare ptr @tl_tensor_relu.26(ptr)

declare ptr @tl_tensor_gelu.27(ptr)

declare ptr @tl_tensor_tril.28(ptr, i32)

declare ptr @tl_tensor_sum_dim.29(ptr, i64, i1)

declare ptr @tl_tensor_embedding.30(ptr, ptr)

declare ptr @tl_tensor_sum.31(ptr)

declare ptr @tl_tensor_div.32(ptr, ptr)

declare ptr @tl_tensor_matmul.33(ptr, ptr)

declare ptr @tl_tensor_exp.34(ptr)

declare ptr @tl_tensor_log.35(ptr)

declare void @tl_tensor_add_assign.36(ptr, ptr)

declare void @tl_tensor_sub_assign.37(ptr, ptr)

declare void @tl_tensor_mul_assign.38(ptr, ptr)

declare void @tl_tensor_div_assign.39(ptr, ptr)

declare void @tl_register_tensor.40(ptr, ptr)

declare i32 @strcmp.41(ptr, ptr)

declare ptr @tl_tensor_reshape_dims.42(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.43(ptr, ptr)

declare ptr @tl_tensor_randn.44(i64, ptr, i1)

declare ptr @tl_varbuilder_get.45(ptr, i64, ptr)

declare void @tl_update_all_params.46(float)

declare ptr @tl_varbuilder_grad.47(ptr)

declare void @tl_tensor_backward.48(ptr)

declare ptr @tl_tensor_grad.49(ptr)

declare ptr @tl_tensor_detach.50(ptr, i1)

declare ptr @tl_tensor_softmax.51(ptr, i64)

declare ptr @tl_tensor_cross_entropy.52(ptr, ptr)

declare void @tl_tensor_save.53(ptr, ptr)

declare ptr @tl_tensor_load.54(ptr)

declare void @tl_save_all_params.55(ptr)

declare void @tl_load_all_params.56(ptr)

declare void @tl_tensor_sub_assign.57(ptr, ptr)

declare void @tl_add_parameter.58(ptr, ptr)

declare ptr @tl_register_parameter.59(ptr)

declare ptr @tl_string_concat.60(ptr, ptr)

declare ptr @tl_file_open.61(ptr, ptr)

declare ptr @tl_file_read_string.62(ptr)

declare void @tl_file_write_string.63(ptr, ptr)

declare void @tl_file_close.64(ptr)

declare ptr @tl_path_new.65(ptr)

declare ptr @tl_path_join.66(ptr, ptr)

declare i1 @tl_path_exists.67(ptr)

declare i1 @tl_path_is_dir.68(ptr)

declare i1 @tl_path_is_file.69(ptr)

declare ptr @tl_path_to_string.70(ptr)

declare void @tl_path_free.71(ptr)

declare i1 @tl_http_download.72(ptr, ptr)

declare ptr @tl_http_get.73(ptr)

declare ptr @tl_env_get.74(ptr)

declare void @tl_env_set.75(ptr, ptr)

declare float @tl_system_time.76()

declare void @tl_system_sleep.77(float)

define ptr @tl_Linear_new(i64 %i, i64 %o) {
entry:
  %shape_arr7 = alloca [1 x i64], align 8
  %shape_arr = alloca [2 x i64], align 8
  %o2 = alloca i64, align 8
  %i1 = alloca i64, align 8
  store i64 %i, ptr %i1, align 8
  store i64 %o, ptr %o2, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Linear, ptr null, i32 1) to i64))
  %i3 = load i64, ptr %i1, align 8
  %o4 = load i64, ptr %o2, align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %i3, ptr %shape_ptr_in, align 8
  %shape_ptr_in5 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 %o4, ptr %shape_ptr_in5, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 true)
  %scalar_data = alloca float, align 4
  store float 0x3FB99999A0000000, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res, i1 true)
  %init_field = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 0
  store ptr %detach_res, ptr %init_field, align 8
  %o6 = load i64, ptr %o2, align 8
  %shape_ptr_in8 = getelementptr inbounds [1 x i64], ptr %shape_arr7, i64 0, i64 0
  store i64 %o6, ptr %shape_ptr_in8, align 8
  %randn_res9 = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr7, i1 true)
  %scalar_data10 = alloca float, align 4
  store float 0.000000e+00, ptr %scalar_data10, align 4
  %scalar_shape11 = alloca i64, i64 0, align 8
  %scalar_tensor12 = call ptr @tl_tensor_new(ptr %scalar_data10, i64 0, ptr %scalar_shape11)
  %binop_res13 = call ptr @tl_tensor_mul(ptr %randn_res9, ptr %scalar_tensor12)
  %detach_res14 = call ptr @tl_tensor_detach(ptr %binop_res13, i1 true)
  %init_field15 = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 1
  store ptr %detach_res14, ptr %init_field15, align 8
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
  %ptr_W = getelementptr inbounds nuw %Linear, ptr %self4, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  %matmul_res = call ptr @tl_tensor_matmul(ptr %x3, ptr %W)
  %self5 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds nuw %Linear, ptr %self5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %matmul_res, ptr %b)
  ret ptr %binop_res
}

define ptr @tl_Linear_step(ptr %self, float %lr) {
entry:
  %gb = alloca ptr, align 8
  %gW = alloca ptr, align 8
  %s = alloca ptr, align 8
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  store ptr %self3, ptr %s, align 8
  %s4 = load ptr, ptr %s, align 8
  %ptr_W = getelementptr inbounds nuw %Linear, ptr %s4, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %W)
  store ptr %grad_res, ptr %gW, align 8
  call void @tl_register_tensor(ptr @tensor_name, ptr %grad_res)
  %s5 = load ptr, ptr %s, align 8
  %ptr_b = getelementptr inbounds nuw %Linear, ptr %s5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res6 = call ptr @tl_tensor_grad(ptr %b)
  store ptr %grad_res6, ptr %gb, align 8
  call void @tl_register_tensor(ptr @tensor_name.78, ptr %grad_res6)
  %s7 = load ptr, ptr %s, align 8
  %ptr_W8 = getelementptr inbounds nuw %Linear, ptr %s7, i32 0, i32 0
  %s9 = load ptr, ptr %s, align 8
  %ptr_W10 = getelementptr inbounds nuw %Linear, ptr %s9, i32 0, i32 0
  %W11 = load ptr, ptr %ptr_W10, align 8
  %gW12 = load ptr, ptr %gW, align 8
  %lr13 = load float, ptr %lr2, align 4
  %scalar_data = alloca float, align 4
  store float %lr13, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %gW12, ptr %scalar_tensor)
  %binop_res14 = call ptr @tl_tensor_sub(ptr %W11, ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res14, i1 true)
  %old_field_val = load ptr, ptr %ptr_W8, align 8
  call void @tl_tensor_free(ptr %old_field_val)
  store ptr %detach_res, ptr %ptr_W8, align 8
  %s15 = load ptr, ptr %s, align 8
  %ptr_b16 = getelementptr inbounds nuw %Linear, ptr %s15, i32 0, i32 1
  %s17 = load ptr, ptr %s, align 8
  %ptr_b18 = getelementptr inbounds nuw %Linear, ptr %s17, i32 0, i32 1
  %b19 = load ptr, ptr %ptr_b18, align 8
  %gb20 = load ptr, ptr %gb, align 8
  %lr21 = load float, ptr %lr2, align 4
  %scalar_data22 = alloca float, align 4
  store float %lr21, ptr %scalar_data22, align 4
  %scalar_shape23 = alloca i64, i64 0, align 8
  %scalar_tensor24 = call ptr @tl_tensor_new(ptr %scalar_data22, i64 0, ptr %scalar_shape23)
  %binop_res25 = call ptr @tl_tensor_mul(ptr %gb20, ptr %scalar_tensor24)
  %binop_res26 = call ptr @tl_tensor_sub(ptr %b19, ptr %binop_res25)
  %detach_res27 = call ptr @tl_tensor_detach(ptr %binop_res26, i1 true)
  %old_field_val28 = load ptr, ptr %ptr_b16, align 8
  call void @tl_tensor_free(ptr %old_field_val28)
  store ptr %detach_res27, ptr %ptr_b16, align 8
  %s29 = load ptr, ptr %s, align 8
  %load_for_free = load ptr, ptr %gb, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free30 = load ptr, ptr %gW, align 8
  call void @tl_tensor_free(ptr %load_for_free30)
  ret ptr %s29
}

define ptr @tl_Embedding_new(i64 %v, i64 %d) {
entry:
  %shape_arr = alloca [2 x i64], align 8
  %d2 = alloca i64, align 8
  %v1 = alloca i64, align 8
  store i64 %v, ptr %v1, align 8
  store i64 %d, ptr %d2, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Embedding, ptr null, i32 1) to i64))
  %v3 = load i64, ptr %v1, align 8
  %d4 = load i64, ptr %d2, align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %v3, ptr %shape_ptr_in, align 8
  %shape_ptr_in5 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 %d4, ptr %shape_ptr_in5, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 true)
  %scalar_data = alloca float, align 4
  store float 0x3FB99999A0000000, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res, i1 true)
  %init_field = getelementptr inbounds nuw %Embedding, ptr %struct_malloc, i32 0, i32 0
  store ptr %detach_res, ptr %init_field, align 8
  ret ptr %struct_malloc
}

define ptr @tl_Embedding_forward(ptr %self, ptr %i) {
entry:
  %i2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %i, ptr %i2, align 8
  %i3 = load ptr, ptr %i2, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds nuw %Embedding, ptr %self4, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %emb_res = call ptr @tl_tensor_embedding(ptr %i3, ptr %w)
  ret ptr %emb_res
}

define ptr @tl_Embedding_step(ptr %self, float %lr) {
entry:
  %g = alloca ptr, align 8
  %s = alloca ptr, align 8
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  store ptr %self3, ptr %s, align 8
  %s4 = load ptr, ptr %s, align 8
  %ptr_w = getelementptr inbounds nuw %Embedding, ptr %s4, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %w)
  store ptr %grad_res, ptr %g, align 8
  call void @tl_register_tensor(ptr @tensor_name.79, ptr %grad_res)
  %s5 = load ptr, ptr %s, align 8
  %ptr_w6 = getelementptr inbounds nuw %Embedding, ptr %s5, i32 0, i32 0
  %s7 = load ptr, ptr %s, align 8
  %ptr_w8 = getelementptr inbounds nuw %Embedding, ptr %s7, i32 0, i32 0
  %w9 = load ptr, ptr %ptr_w8, align 8
  %g10 = load ptr, ptr %g, align 8
  %lr11 = load float, ptr %lr2, align 4
  %scalar_data = alloca float, align 4
  store float %lr11, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %g10, ptr %scalar_tensor)
  %binop_res12 = call ptr @tl_tensor_sub(ptr %w9, ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res12, i1 true)
  %old_field_val = load ptr, ptr %ptr_w6, align 8
  call void @tl_tensor_free(ptr %old_field_val)
  store ptr %detach_res, ptr %ptr_w6, align 8
  %s13 = load ptr, ptr %s, align 8
  %load_for_free = load ptr, ptr %g, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  ret ptr %s13
}

define ptr @tl_LayerNorm_new(i64 %d) {
entry:
  %shape_arr8 = alloca [1 x i64], align 8
  %shape_arr = alloca [1 x i64], align 8
  %d1 = alloca i64, align 8
  store i64 %d, ptr %d1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%LayerNorm, ptr null, i32 1) to i64))
  %d2 = load i64, ptr %d1, align 8
  %shape_ptr_in = getelementptr inbounds [1 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %d2, ptr %shape_ptr_in, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr, i1 true)
  %scalar_data = alloca float, align 4
  store float 0.000000e+00, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor)
  %scalar_data3 = alloca float, align 4
  store float 1.000000e+00, ptr %scalar_data3, align 4
  %scalar_shape4 = alloca i64, i64 0, align 8
  %scalar_tensor5 = call ptr @tl_tensor_new(ptr %scalar_data3, i64 0, ptr %scalar_shape4)
  %binop_res6 = call ptr @tl_tensor_add(ptr %binop_res, ptr %scalar_tensor5)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res6, i1 true)
  %init_field = getelementptr inbounds nuw %LayerNorm, ptr %struct_malloc, i32 0, i32 0
  store ptr %detach_res, ptr %init_field, align 8
  %d7 = load i64, ptr %d1, align 8
  %shape_ptr_in9 = getelementptr inbounds [1 x i64], ptr %shape_arr8, i64 0, i64 0
  store i64 %d7, ptr %shape_ptr_in9, align 8
  %randn_res10 = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr8, i1 true)
  %scalar_data11 = alloca float, align 4
  store float 0.000000e+00, ptr %scalar_data11, align 4
  %scalar_shape12 = alloca i64, i64 0, align 8
  %scalar_tensor13 = call ptr @tl_tensor_new(ptr %scalar_data11, i64 0, ptr %scalar_shape12)
  %binop_res14 = call ptr @tl_tensor_mul(ptr %randn_res10, ptr %scalar_tensor13)
  %detach_res15 = call ptr @tl_tensor_detach(ptr %binop_res14, i1 true)
  %init_field16 = getelementptr inbounds nuw %LayerNorm, ptr %struct_malloc, i32 0, i32 1
  store ptr %detach_res15, ptr %init_field16, align 8
  ret ptr %struct_malloc
}

define ptr @tl_LayerNorm_forward(ptr %self, ptr %x) {
entry:
  %x2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %x3 = load ptr, ptr %x2, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds nuw %LayerNorm, ptr %self4, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %x3, ptr %b)
  ret ptr %binop_res
}

define ptr @tl_LayerNorm_step(ptr %self, float %lr) {
entry:
  %gb = alloca ptr, align 8
  %s = alloca ptr, align 8
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  store ptr %self3, ptr %s, align 8
  %s4 = load ptr, ptr %s, align 8
  %ptr_b = getelementptr inbounds nuw %LayerNorm, ptr %s4, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %b)
  store ptr %grad_res, ptr %gb, align 8
  call void @tl_register_tensor(ptr @tensor_name.80, ptr %grad_res)
  %s5 = load ptr, ptr %s, align 8
  %ptr_b6 = getelementptr inbounds nuw %LayerNorm, ptr %s5, i32 0, i32 1
  %s7 = load ptr, ptr %s, align 8
  %ptr_b8 = getelementptr inbounds nuw %LayerNorm, ptr %s7, i32 0, i32 1
  %b9 = load ptr, ptr %ptr_b8, align 8
  %gb10 = load ptr, ptr %gb, align 8
  %lr11 = load float, ptr %lr2, align 4
  %scalar_data = alloca float, align 4
  store float %lr11, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %gb10, ptr %scalar_tensor)
  %binop_res12 = call ptr @tl_tensor_sub(ptr %b9, ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res12, i1 true)
  %old_field_val = load ptr, ptr %ptr_b6, align 8
  call void @tl_tensor_free(ptr %old_field_val)
  store ptr %detach_res, ptr %ptr_b6, align 8
  %s13 = load ptr, ptr %s, align 8
  %load_for_free = load ptr, ptr %gb, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  ret ptr %s13
}

define ptr @tl_CausalSelfAttention_new(i64 %d) {
entry:
  %d1 = alloca i64, align 8
  store i64 %d, ptr %d1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%CausalSelfAttention, ptr null, i32 1) to i64))
  %d2 = load i64, ptr %d1, align 8
  %d3 = load i64, ptr %d1, align 8
  %multmp = mul i64 %d3, 3
  %static_call = call ptr @tl_Linear_new(i64 %d2, i64 %multmp)
  %init_field = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d4 = load i64, ptr %d1, align 8
  %multmp5 = mul i64 %d4, 3
  %d6 = load i64, ptr %d1, align 8
  %static_call7 = call ptr @tl_Linear_new(i64 %multmp5, i64 %d6)
  %init_field8 = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call7, ptr %init_field8, align 8
  ret ptr %struct_malloc
}

define ptr @tl_CausalSelfAttention_forward(ptr %self, ptr %x) {
entry:
  %y = alloca ptr, align 8
  %v = alloca ptr, align 8
  %k = alloca ptr, align 8
  %q = alloca ptr, align 8
  %x2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_a = getelementptr inbounds nuw %CausalSelfAttention, ptr %self3, i32 0, i32 0
  %a = load ptr, ptr %ptr_a, align 8
  %x4 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_Linear_forward(ptr %a, ptr %x4)
  store ptr %call_method, ptr %q, align 8
  call void @tl_register_tensor(ptr @tensor_name.81, ptr %call_method)
  %q5 = load ptr, ptr %q, align 8
  %cloned = call ptr @tl_tensor_clone(ptr %q5)
  store ptr %cloned, ptr %k, align 8
  call void @tl_register_tensor(ptr @tensor_name.82, ptr %cloned)
  %q6 = load ptr, ptr %q, align 8
  %cloned7 = call ptr @tl_tensor_clone(ptr %q6)
  store ptr %cloned7, ptr %v, align 8
  call void @tl_register_tensor(ptr @tensor_name.83, ptr %cloned7)
  %q8 = load ptr, ptr %q, align 8
  %k9 = load ptr, ptr %k, align 8
  %transpose_res = call ptr @tl_tensor_transpose(ptr %k9, i64 1, i64 2)
  %matmul_res = call ptr @tl_tensor_matmul(ptr %q8, ptr %transpose_res)
  %scalar_data = alloca float, align 4
  store float 1.250000e-01, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %matmul_res, ptr %scalar_tensor)
  %tril_res = call ptr @tl_tensor_tril(ptr %binop_res, i32 0)
  %softmax_res = call ptr @tl_tensor_softmax(ptr %tril_res, i64 2)
  %v10 = load ptr, ptr %v, align 8
  %matmul_res11 = call ptr @tl_tensor_matmul(ptr %softmax_res, ptr %v10)
  store ptr %matmul_res11, ptr %y, align 8
  call void @tl_register_tensor(ptr @tensor_name.84, ptr %matmul_res11)
  %self12 = load ptr, ptr %self1, align 8
  %ptr_p = getelementptr inbounds nuw %CausalSelfAttention, ptr %self12, i32 0, i32 1
  %p = load ptr, ptr %ptr_p, align 8
  %y13 = load ptr, ptr %y, align 8
  %call_method14 = call ptr @tl_Linear_forward(ptr %p, ptr %y13)
  %load_for_free = load ptr, ptr %k, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free15 = load ptr, ptr %q, align 8
  call void @tl_tensor_free(ptr %load_for_free15)
  %load_for_free16 = load ptr, ptr %v, align 8
  call void @tl_tensor_free(ptr %load_for_free16)
  %load_for_free17 = load ptr, ptr %y, align 8
  call void @tl_tensor_free(ptr %load_for_free17)
  ret ptr %call_method14
}

define ptr @tl_CausalSelfAttention_step(ptr %self, float %lr) {
entry:
  %s = alloca ptr, align 8
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  store ptr %self3, ptr %s, align 8
  %s4 = load ptr, ptr %s, align 8
  %ptr_a = getelementptr inbounds nuw %CausalSelfAttention, ptr %s4, i32 0, i32 0
  %s5 = load ptr, ptr %s, align 8
  %ptr_a6 = getelementptr inbounds nuw %CausalSelfAttention, ptr %s5, i32 0, i32 0
  %a = load ptr, ptr %ptr_a6, align 8
  %lr7 = load float, ptr %lr2, align 4
  %call_method = call ptr @tl_Linear_step(ptr %a, float %lr7)
  store ptr %call_method, ptr %ptr_a, align 8
  %s8 = load ptr, ptr %s, align 8
  %ptr_p = getelementptr inbounds nuw %CausalSelfAttention, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_p10 = getelementptr inbounds nuw %CausalSelfAttention, ptr %s9, i32 0, i32 1
  %p = load ptr, ptr %ptr_p10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_Linear_step(ptr %p, float %lr11)
  store ptr %call_method12, ptr %ptr_p, align 8
  %s13 = load ptr, ptr %s, align 8
  ret ptr %s13
}

define ptr @tl_MLP_new(i64 %d) {
entry:
  %d1 = alloca i64, align 8
  store i64 %d, ptr %d1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%MLP, ptr null, i32 1) to i64))
  %d2 = load i64, ptr %d1, align 8
  %d3 = load i64, ptr %d1, align 8
  %multmp = mul i64 %d3, 4
  %static_call = call ptr @tl_Linear_new(i64 %d2, i64 %multmp)
  %init_field = getelementptr inbounds nuw %MLP, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d4 = load i64, ptr %d1, align 8
  %multmp5 = mul i64 %d4, 4
  %d6 = load i64, ptr %d1, align 8
  %static_call7 = call ptr @tl_Linear_new(i64 %multmp5, i64 %d6)
  %init_field8 = getelementptr inbounds nuw %MLP, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call7, ptr %init_field8, align 8
  ret ptr %struct_malloc
}

define ptr @tl_MLP_forward(ptr %self, ptr %x) {
entry:
  %x2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_p = getelementptr inbounds nuw %MLP, ptr %self3, i32 0, i32 1
  %p = load ptr, ptr %ptr_p, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_f = getelementptr inbounds nuw %MLP, ptr %self4, i32 0, i32 0
  %f = load ptr, ptr %ptr_f, align 8
  %x5 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_Linear_forward(ptr %f, ptr %x5)
  %relu_res = call ptr @tl_tensor_relu(ptr %call_method)
  %call_method6 = call ptr @tl_Linear_forward(ptr %p, ptr %relu_res)
  ret ptr %call_method6
}

define ptr @tl_MLP_step(ptr %self, float %lr) {
entry:
  %s = alloca ptr, align 8
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  store ptr %self3, ptr %s, align 8
  %s4 = load ptr, ptr %s, align 8
  %ptr_f = getelementptr inbounds nuw %MLP, ptr %s4, i32 0, i32 0
  %s5 = load ptr, ptr %s, align 8
  %ptr_f6 = getelementptr inbounds nuw %MLP, ptr %s5, i32 0, i32 0
  %f = load ptr, ptr %ptr_f6, align 8
  %lr7 = load float, ptr %lr2, align 4
  %call_method = call ptr @tl_Linear_step(ptr %f, float %lr7)
  store ptr %call_method, ptr %ptr_f, align 8
  %s8 = load ptr, ptr %s, align 8
  %ptr_p = getelementptr inbounds nuw %MLP, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_p10 = getelementptr inbounds nuw %MLP, ptr %s9, i32 0, i32 1
  %p = load ptr, ptr %ptr_p10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_Linear_step(ptr %p, float %lr11)
  store ptr %call_method12, ptr %ptr_p, align 8
  %s13 = load ptr, ptr %s, align 8
  ret ptr %s13
}

define ptr @tl_Block_new(i64 %d) {
entry:
  %d1 = alloca i64, align 8
  store i64 %d, ptr %d1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Block, ptr null, i32 1) to i64))
  %d2 = load i64, ptr %d1, align 8
  %static_call = call ptr @tl_LayerNorm_new(i64 %d2)
  %init_field = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d3 = load i64, ptr %d1, align 8
  %static_call4 = call ptr @tl_CausalSelfAttention_new(i64 %d3)
  %init_field5 = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call4, ptr %init_field5, align 8
  %d6 = load i64, ptr %d1, align 8
  %static_call7 = call ptr @tl_LayerNorm_new(i64 %d6)
  %init_field8 = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 2
  store ptr %static_call7, ptr %init_field8, align 8
  %d9 = load i64, ptr %d1, align 8
  %static_call10 = call ptr @tl_MLP_new(i64 %d9)
  %init_field11 = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 3
  store ptr %static_call10, ptr %init_field11, align 8
  ret ptr %struct_malloc
}

define ptr @tl_Block_forward(ptr %self, ptr %x) {
entry:
  %x8 = alloca ptr, align 8
  %x2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %x3 = load ptr, ptr %x2, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_a = getelementptr inbounds nuw %Block, ptr %self4, i32 0, i32 1
  %a = load ptr, ptr %ptr_a, align 8
  %self5 = load ptr, ptr %self1, align 8
  %ptr_l1 = getelementptr inbounds nuw %Block, ptr %self5, i32 0, i32 0
  %l1 = load ptr, ptr %ptr_l1, align 8
  %x6 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_LayerNorm_forward(ptr %l1, ptr %x6)
  %call_method7 = call ptr @tl_CausalSelfAttention_forward(ptr %a, ptr %call_method)
  %binop_res = call ptr @tl_tensor_add(ptr %x3, ptr %call_method7)
  store ptr %binop_res, ptr %x8, align 8
  call void @tl_register_tensor(ptr @tensor_name.85, ptr %binop_res)
  %x9 = load ptr, ptr %x8, align 8
  %self10 = load ptr, ptr %self1, align 8
  %ptr_m = getelementptr inbounds nuw %Block, ptr %self10, i32 0, i32 3
  %m = load ptr, ptr %ptr_m, align 8
  %self11 = load ptr, ptr %self1, align 8
  %ptr_l2 = getelementptr inbounds nuw %Block, ptr %self11, i32 0, i32 2
  %l2 = load ptr, ptr %ptr_l2, align 8
  %x12 = load ptr, ptr %x8, align 8
  %call_method13 = call ptr @tl_LayerNorm_forward(ptr %l2, ptr %x12)
  %call_method14 = call ptr @tl_MLP_forward(ptr %m, ptr %call_method13)
  %binop_res15 = call ptr @tl_tensor_add(ptr %x9, ptr %call_method14)
  %load_for_free = load ptr, ptr %x8, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  ret ptr %binop_res15
}

define ptr @tl_Block_step(ptr %self, float %lr) {
entry:
  %s = alloca ptr, align 8
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  store ptr %self3, ptr %s, align 8
  %s4 = load ptr, ptr %s, align 8
  %ptr_l1 = getelementptr inbounds nuw %Block, ptr %s4, i32 0, i32 0
  %s5 = load ptr, ptr %s, align 8
  %ptr_l16 = getelementptr inbounds nuw %Block, ptr %s5, i32 0, i32 0
  %l1 = load ptr, ptr %ptr_l16, align 8
  %lr7 = load float, ptr %lr2, align 4
  %call_method = call ptr @tl_LayerNorm_step(ptr %l1, float %lr7)
  store ptr %call_method, ptr %ptr_l1, align 8
  %s8 = load ptr, ptr %s, align 8
  %ptr_a = getelementptr inbounds nuw %Block, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_a10 = getelementptr inbounds nuw %Block, ptr %s9, i32 0, i32 1
  %a = load ptr, ptr %ptr_a10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_CausalSelfAttention_step(ptr %a, float %lr11)
  store ptr %call_method12, ptr %ptr_a, align 8
  %s13 = load ptr, ptr %s, align 8
  %ptr_l2 = getelementptr inbounds nuw %Block, ptr %s13, i32 0, i32 2
  %s14 = load ptr, ptr %s, align 8
  %ptr_l215 = getelementptr inbounds nuw %Block, ptr %s14, i32 0, i32 2
  %l2 = load ptr, ptr %ptr_l215, align 8
  %lr16 = load float, ptr %lr2, align 4
  %call_method17 = call ptr @tl_LayerNorm_step(ptr %l2, float %lr16)
  store ptr %call_method17, ptr %ptr_l2, align 8
  %s18 = load ptr, ptr %s, align 8
  %ptr_m = getelementptr inbounds nuw %Block, ptr %s18, i32 0, i32 3
  %s19 = load ptr, ptr %s, align 8
  %ptr_m20 = getelementptr inbounds nuw %Block, ptr %s19, i32 0, i32 3
  %m = load ptr, ptr %ptr_m20, align 8
  %lr21 = load float, ptr %lr2, align 4
  %call_method22 = call ptr @tl_MLP_step(ptr %m, float %lr21)
  store ptr %call_method22, ptr %ptr_m, align 8
  %s23 = load ptr, ptr %s, align 8
  ret ptr %s23
}

define ptr @tl_GPT_new(i64 %v, i64 %d) {
entry:
  %d2 = alloca i64, align 8
  %v1 = alloca i64, align 8
  store i64 %v, ptr %v1, align 8
  store i64 %d, ptr %d2, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%GPT, ptr null, i32 1) to i64))
  %v3 = load i64, ptr %v1, align 8
  %d4 = load i64, ptr %d2, align 8
  %static_call = call ptr @tl_Embedding_new(i64 %v3, i64 %d4)
  %init_field = getelementptr inbounds nuw %GPT, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d5 = load i64, ptr %d2, align 8
  %static_call6 = call ptr @tl_Block_new(i64 %d5)
  %init_field7 = getelementptr inbounds nuw %GPT, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call6, ptr %init_field7, align 8
  %d8 = load i64, ptr %d2, align 8
  %static_call9 = call ptr @tl_LayerNorm_new(i64 %d8)
  %init_field10 = getelementptr inbounds nuw %GPT, ptr %struct_malloc, i32 0, i32 2
  store ptr %static_call9, ptr %init_field10, align 8
  %d11 = load i64, ptr %d2, align 8
  %v12 = load i64, ptr %v1, align 8
  %static_call13 = call ptr @tl_Linear_new(i64 %d11, i64 %v12)
  %init_field14 = getelementptr inbounds nuw %GPT, ptr %struct_malloc, i32 0, i32 3
  store ptr %static_call13, ptr %init_field14, align 8
  ret ptr %struct_malloc
}

define ptr @tl_GPT_forward(ptr %self, ptr %i) {
entry:
  %i2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %i, ptr %i2, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_h = getelementptr inbounds nuw %GPT, ptr %self3, i32 0, i32 3
  %h = load ptr, ptr %ptr_h, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_l = getelementptr inbounds nuw %GPT, ptr %self4, i32 0, i32 2
  %l = load ptr, ptr %ptr_l, align 8
  %self5 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds nuw %GPT, ptr %self5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %self6 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds nuw %GPT, ptr %self6, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %i7 = load ptr, ptr %i2, align 8
  %call_method = call ptr @tl_Embedding_forward(ptr %w, ptr %i7)
  %call_method8 = call ptr @tl_Block_forward(ptr %b, ptr %call_method)
  %call_method9 = call ptr @tl_LayerNorm_forward(ptr %l, ptr %call_method8)
  %call_method10 = call ptr @tl_Linear_forward(ptr %h, ptr %call_method9)
  ret ptr %call_method10
}

define ptr @tl_GPT_step(ptr %self, float %lr) {
entry:
  %s = alloca ptr, align 8
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  store ptr %self3, ptr %s, align 8
  %s4 = load ptr, ptr %s, align 8
  %ptr_w = getelementptr inbounds nuw %GPT, ptr %s4, i32 0, i32 0
  %s5 = load ptr, ptr %s, align 8
  %ptr_w6 = getelementptr inbounds nuw %GPT, ptr %s5, i32 0, i32 0
  %w = load ptr, ptr %ptr_w6, align 8
  %lr7 = load float, ptr %lr2, align 4
  %call_method = call ptr @tl_Embedding_step(ptr %w, float %lr7)
  store ptr %call_method, ptr %ptr_w, align 8
  %s8 = load ptr, ptr %s, align 8
  %ptr_b = getelementptr inbounds nuw %GPT, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_b10 = getelementptr inbounds nuw %GPT, ptr %s9, i32 0, i32 1
  %b = load ptr, ptr %ptr_b10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_Block_step(ptr %b, float %lr11)
  store ptr %call_method12, ptr %ptr_b, align 8
  %s13 = load ptr, ptr %s, align 8
  %ptr_l = getelementptr inbounds nuw %GPT, ptr %s13, i32 0, i32 2
  %s14 = load ptr, ptr %s, align 8
  %ptr_l15 = getelementptr inbounds nuw %GPT, ptr %s14, i32 0, i32 2
  %l = load ptr, ptr %ptr_l15, align 8
  %lr16 = load float, ptr %lr2, align 4
  %call_method17 = call ptr @tl_LayerNorm_step(ptr %l, float %lr16)
  store ptr %call_method17, ptr %ptr_l, align 8
  %s18 = load ptr, ptr %s, align 8
  %ptr_h = getelementptr inbounds nuw %GPT, ptr %s18, i32 0, i32 3
  %s19 = load ptr, ptr %s, align 8
  %ptr_h20 = getelementptr inbounds nuw %GPT, ptr %s19, i32 0, i32 3
  %h = load ptr, ptr %ptr_h20, align 8
  %lr21 = load float, ptr %lr2, align 4
  %call_method22 = call ptr @tl_Linear_step(ptr %h, float %lr21)
  store ptr %call_method22, ptr %ptr_h, align 8
  %s23 = load ptr, ptr %s, align 8
  ret ptr %s23
}

define ptr @train_epoch(ptr %model, float %lr, i64 %epoch) {
entry:
  %loss = alloca ptr, align 8
  %Y_flat = alloca ptr, align 8
  %logits_flat = alloca ptr, align 8
  %logits = alloca ptr, align 8
  %Y = alloca ptr, align 8
  %X = alloca ptr, align 8
  %target = alloca ptr, align 8
  %data = alloca ptr, align 8
  %s_d3 = alloca float, align 4
  %s_d2 = alloca float, align 4
  %s_d1 = alloca float, align 4
  %j_d2 = alloca float, align 4
  %j_d1 = alloca float, align 4
  %i_d2 = alloca float, align 4
  %i_d1 = alloca float, align 4
  %sum = alloca i64, align 8
  %j = alloca i64, align 8
  %i = alloca i64, align 8
  %total_loss = alloca ptr, align 8
  %epoch3 = alloca i64, align 8
  %lr2 = alloca float, align 4
  %model1 = alloca ptr, align 8
  store ptr %model, ptr %model1, align 8
  store float %lr, ptr %lr2, align 4
  store i64 %epoch, ptr %epoch3, align 8
  %scalar_data = alloca float, align 4
  store float 0.000000e+00, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %scalar_data4 = alloca float, align 4
  store float 1.000000e+00, ptr %scalar_data4, align 4
  %scalar_shape5 = alloca i64, align 8
  %scalar_tensor6 = call ptr @tl_tensor_new(ptr %scalar_data4, i64 0, ptr %scalar_shape5)
  %pow_res = call ptr @tl_tensor_pow(ptr %scalar_tensor, ptr %scalar_tensor6)
  store ptr %pow_res, ptr %total_loss, align 8
  call void @tl_register_tensor(ptr @tensor_name.86, ptr %pow_res)
  br label %for_header

for_header:                                       ; preds = %for_end9, %entry
  %for_idx = phi i64 [ 0, %entry ], [ %next_idx174, %for_end9 ]
  %for_cond = icmp slt i64 %for_idx, 100
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  store i64 %for_idx, ptr %i, align 8
  br label %for_header7

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal)
  %epoch175 = load i64, ptr %epoch3, align 8
  call void @tl_print_i64(i64 %epoch175)
  call void @tl_print_string(ptr @str_literal.95)
  %total_loss176 = load ptr, ptr %total_loss, align 8
  call void @tl_tensor_print(ptr %total_loss176)
  %model177 = load ptr, ptr %model1, align 8
  %load_for_free178 = load ptr, ptr %total_loss, align 8
  call void @tl_tensor_free(ptr %load_for_free178)
  ret ptr %model177

for_header7:                                      ; preds = %for_body8, %for_body
  %for_idx10 = phi i64 [ 0, %for_body ], [ %next_idx, %for_body8 ]
  %for_cond11 = icmp slt i64 %for_idx10, 100
  br i1 %for_cond11, label %for_body8, label %for_end9

for_body8:                                        ; preds = %for_header7
  store i64 %for_idx10, ptr %j, align 8
  %i12 = load i64, ptr %i, align 8
  %j13 = load i64, ptr %j, align 8
  %addtmp = add i64 %i12, %j13
  store i64 %addtmp, ptr %sum, align 8
  %i14 = load i64, ptr %i, align 8
  %divtmp = sdiv i64 %i14, 10
  %scalar_data15 = alloca float, align 4
  %cast_i64_f32 = sitofp i64 %divtmp to float
  store float %cast_i64_f32, ptr %scalar_data15, align 4
  %scalar_shape16 = alloca i64, align 8
  %scalar_tensor17 = call ptr @tl_tensor_new(ptr %scalar_data15, i64 0, ptr %scalar_shape16)
  %scalar_data18 = alloca float, align 4
  store float 1.000000e+00, ptr %scalar_data18, align 4
  %scalar_shape19 = alloca i64, align 8
  %scalar_tensor20 = call ptr @tl_tensor_new(ptr %scalar_data18, i64 0, ptr %scalar_shape19)
  %pow_res21 = call ptr @tl_tensor_pow(ptr %scalar_tensor17, ptr %scalar_tensor20)
  %get_res = call float @tl_tensor_get(ptr %pow_res21, i64 0)
  store float %get_res, ptr %i_d1, align 4
  %i22 = load i64, ptr %i, align 8
  %i23 = load i64, ptr %i, align 8
  %divtmp24 = sdiv i64 %i23, 10
  %multmp = mul i64 %divtmp24, 10
  %subtmp = sub i64 %i22, %multmp
  %scalar_data25 = alloca float, align 4
  %cast_i64_f3226 = sitofp i64 %subtmp to float
  store float %cast_i64_f3226, ptr %scalar_data25, align 4
  %scalar_shape27 = alloca i64, align 8
  %scalar_tensor28 = call ptr @tl_tensor_new(ptr %scalar_data25, i64 0, ptr %scalar_shape27)
  %scalar_data29 = alloca float, align 4
  store float 1.000000e+00, ptr %scalar_data29, align 4
  %scalar_shape30 = alloca i64, align 8
  %scalar_tensor31 = call ptr @tl_tensor_new(ptr %scalar_data29, i64 0, ptr %scalar_shape30)
  %pow_res32 = call ptr @tl_tensor_pow(ptr %scalar_tensor28, ptr %scalar_tensor31)
  %get_res33 = call float @tl_tensor_get(ptr %pow_res32, i64 0)
  store float %get_res33, ptr %i_d2, align 4
  %j34 = load i64, ptr %j, align 8
  %divtmp35 = sdiv i64 %j34, 10
  %scalar_data36 = alloca float, align 4
  %cast_i64_f3237 = sitofp i64 %divtmp35 to float
  store float %cast_i64_f3237, ptr %scalar_data36, align 4
  %scalar_shape38 = alloca i64, align 8
  %scalar_tensor39 = call ptr @tl_tensor_new(ptr %scalar_data36, i64 0, ptr %scalar_shape38)
  %scalar_data40 = alloca float, align 4
  store float 1.000000e+00, ptr %scalar_data40, align 4
  %scalar_shape41 = alloca i64, align 8
  %scalar_tensor42 = call ptr @tl_tensor_new(ptr %scalar_data40, i64 0, ptr %scalar_shape41)
  %pow_res43 = call ptr @tl_tensor_pow(ptr %scalar_tensor39, ptr %scalar_tensor42)
  %get_res44 = call float @tl_tensor_get(ptr %pow_res43, i64 0)
  store float %get_res44, ptr %j_d1, align 4
  %j45 = load i64, ptr %j, align 8
  %j46 = load i64, ptr %j, align 8
  %divtmp47 = sdiv i64 %j46, 10
  %multmp48 = mul i64 %divtmp47, 10
  %subtmp49 = sub i64 %j45, %multmp48
  %scalar_data50 = alloca float, align 4
  %cast_i64_f3251 = sitofp i64 %subtmp49 to float
  store float %cast_i64_f3251, ptr %scalar_data50, align 4
  %scalar_shape52 = alloca i64, align 8
  %scalar_tensor53 = call ptr @tl_tensor_new(ptr %scalar_data50, i64 0, ptr %scalar_shape52)
  %scalar_data54 = alloca float, align 4
  store float 1.000000e+00, ptr %scalar_data54, align 4
  %scalar_shape55 = alloca i64, align 8
  %scalar_tensor56 = call ptr @tl_tensor_new(ptr %scalar_data54, i64 0, ptr %scalar_shape55)
  %pow_res57 = call ptr @tl_tensor_pow(ptr %scalar_tensor53, ptr %scalar_tensor56)
  %get_res58 = call float @tl_tensor_get(ptr %pow_res57, i64 0)
  store float %get_res58, ptr %j_d2, align 4
  %sum59 = load i64, ptr %sum, align 8
  %divtmp60 = sdiv i64 %sum59, 100
  %scalar_data61 = alloca float, align 4
  %cast_i64_f3262 = sitofp i64 %divtmp60 to float
  store float %cast_i64_f3262, ptr %scalar_data61, align 4
  %scalar_shape63 = alloca i64, align 8
  %scalar_tensor64 = call ptr @tl_tensor_new(ptr %scalar_data61, i64 0, ptr %scalar_shape63)
  %scalar_data65 = alloca float, align 4
  store float 1.000000e+00, ptr %scalar_data65, align 4
  %scalar_shape66 = alloca i64, align 8
  %scalar_tensor67 = call ptr @tl_tensor_new(ptr %scalar_data65, i64 0, ptr %scalar_shape66)
  %pow_res68 = call ptr @tl_tensor_pow(ptr %scalar_tensor64, ptr %scalar_tensor67)
  %get_res69 = call float @tl_tensor_get(ptr %pow_res68, i64 0)
  store float %get_res69, ptr %s_d1, align 4
  %sum70 = load i64, ptr %sum, align 8
  %sum71 = load i64, ptr %sum, align 8
  %divtmp72 = sdiv i64 %sum71, 100
  %multmp73 = mul i64 %divtmp72, 100
  %subtmp74 = sub i64 %sum70, %multmp73
  %divtmp75 = sdiv i64 %subtmp74, 10
  %scalar_data76 = alloca float, align 4
  %cast_i64_f3277 = sitofp i64 %divtmp75 to float
  store float %cast_i64_f3277, ptr %scalar_data76, align 4
  %scalar_shape78 = alloca i64, align 8
  %scalar_tensor79 = call ptr @tl_tensor_new(ptr %scalar_data76, i64 0, ptr %scalar_shape78)
  %scalar_data80 = alloca float, align 4
  store float 1.000000e+00, ptr %scalar_data80, align 4
  %scalar_shape81 = alloca i64, align 8
  %scalar_tensor82 = call ptr @tl_tensor_new(ptr %scalar_data80, i64 0, ptr %scalar_shape81)
  %pow_res83 = call ptr @tl_tensor_pow(ptr %scalar_tensor79, ptr %scalar_tensor82)
  %get_res84 = call float @tl_tensor_get(ptr %pow_res83, i64 0)
  store float %get_res84, ptr %s_d2, align 4
  %sum85 = load i64, ptr %sum, align 8
  %sum86 = load i64, ptr %sum, align 8
  %divtmp87 = sdiv i64 %sum86, 10
  %multmp88 = mul i64 %divtmp87, 10
  %subtmp89 = sub i64 %sum85, %multmp88
  %scalar_data90 = alloca float, align 4
  %cast_i64_f3291 = sitofp i64 %subtmp89 to float
  store float %cast_i64_f3291, ptr %scalar_data90, align 4
  %scalar_shape92 = alloca i64, align 8
  %scalar_tensor93 = call ptr @tl_tensor_new(ptr %scalar_data90, i64 0, ptr %scalar_shape92)
  %scalar_data94 = alloca float, align 4
  store float 1.000000e+00, ptr %scalar_data94, align 4
  %scalar_shape95 = alloca i64, align 8
  %scalar_tensor96 = call ptr @tl_tensor_new(ptr %scalar_data94, i64 0, ptr %scalar_shape95)
  %pow_res97 = call ptr @tl_tensor_pow(ptr %scalar_tensor93, ptr %scalar_tensor96)
  %get_res98 = call float @tl_tensor_get(ptr %pow_res97, i64 0)
  store float %get_res98, ptr %s_d3, align 4
  %tensor_data = alloca float, i64 12, align 4
  %i_d199 = load float, ptr %i_d1, align 4
  %elem_ptr = getelementptr inbounds float, ptr %tensor_data, i64 0
  store float %i_d199, ptr %elem_ptr, align 4
  %i_d2100 = load float, ptr %i_d2, align 4
  %elem_ptr101 = getelementptr inbounds float, ptr %tensor_data, i64 1
  store float %i_d2100, ptr %elem_ptr101, align 4
  %elem_ptr102 = getelementptr inbounds float, ptr %tensor_data, i64 2
  store float 1.000000e+01, ptr %elem_ptr102, align 4
  %j_d1103 = load float, ptr %j_d1, align 4
  %elem_ptr104 = getelementptr inbounds float, ptr %tensor_data, i64 3
  store float %j_d1103, ptr %elem_ptr104, align 4
  %j_d2105 = load float, ptr %j_d2, align 4
  %elem_ptr106 = getelementptr inbounds float, ptr %tensor_data, i64 4
  store float %j_d2105, ptr %elem_ptr106, align 4
  %elem_ptr107 = getelementptr inbounds float, ptr %tensor_data, i64 5
  store float 1.100000e+01, ptr %elem_ptr107, align 4
  %s_d1108 = load float, ptr %s_d1, align 4
  %elem_ptr109 = getelementptr inbounds float, ptr %tensor_data, i64 6
  store float %s_d1108, ptr %elem_ptr109, align 4
  %s_d2110 = load float, ptr %s_d2, align 4
  %elem_ptr111 = getelementptr inbounds float, ptr %tensor_data, i64 7
  store float %s_d2110, ptr %elem_ptr111, align 4
  %s_d3112 = load float, ptr %s_d3, align 4
  %elem_ptr113 = getelementptr inbounds float, ptr %tensor_data, i64 8
  store float %s_d3112, ptr %elem_ptr113, align 4
  %elem_ptr114 = getelementptr inbounds float, ptr %tensor_data, i64 9
  store float 1.200000e+01, ptr %elem_ptr114, align 4
  %elem_ptr115 = getelementptr inbounds float, ptr %tensor_data, i64 10
  store float 1.200000e+01, ptr %elem_ptr115, align 4
  %elem_ptr116 = getelementptr inbounds float, ptr %tensor_data, i64 11
  store float 1.200000e+01, ptr %elem_ptr116, align 4
  %tensor_shape = alloca i64, i64 1, align 8
  %shape_ptr = getelementptr inbounds i64, ptr %tensor_shape, i64 0
  store i64 12, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %tensor_data, i64 1, ptr %tensor_shape)
  store ptr %new_tensor, ptr %data, align 8
  call void @tl_register_tensor(ptr @tensor_name.87, ptr %new_tensor)
  %tensor_data117 = alloca float, i64 12, align 4
  %i_d2118 = load float, ptr %i_d2, align 4
  %elem_ptr119 = getelementptr inbounds float, ptr %tensor_data117, i64 0
  store float %i_d2118, ptr %elem_ptr119, align 4
  %elem_ptr120 = getelementptr inbounds float, ptr %tensor_data117, i64 1
  store float 1.000000e+01, ptr %elem_ptr120, align 4
  %j_d1121 = load float, ptr %j_d1, align 4
  %elem_ptr122 = getelementptr inbounds float, ptr %tensor_data117, i64 2
  store float %j_d1121, ptr %elem_ptr122, align 4
  %j_d2123 = load float, ptr %j_d2, align 4
  %elem_ptr124 = getelementptr inbounds float, ptr %tensor_data117, i64 3
  store float %j_d2123, ptr %elem_ptr124, align 4
  %elem_ptr125 = getelementptr inbounds float, ptr %tensor_data117, i64 4
  store float 1.100000e+01, ptr %elem_ptr125, align 4
  %s_d1126 = load float, ptr %s_d1, align 4
  %elem_ptr127 = getelementptr inbounds float, ptr %tensor_data117, i64 5
  store float %s_d1126, ptr %elem_ptr127, align 4
  %s_d2128 = load float, ptr %s_d2, align 4
  %elem_ptr129 = getelementptr inbounds float, ptr %tensor_data117, i64 6
  store float %s_d2128, ptr %elem_ptr129, align 4
  %s_d3130 = load float, ptr %s_d3, align 4
  %elem_ptr131 = getelementptr inbounds float, ptr %tensor_data117, i64 7
  store float %s_d3130, ptr %elem_ptr131, align 4
  %elem_ptr132 = getelementptr inbounds float, ptr %tensor_data117, i64 8
  store float 1.200000e+01, ptr %elem_ptr132, align 4
  %elem_ptr133 = getelementptr inbounds float, ptr %tensor_data117, i64 9
  store float 1.200000e+01, ptr %elem_ptr133, align 4
  %elem_ptr134 = getelementptr inbounds float, ptr %tensor_data117, i64 10
  store float 1.200000e+01, ptr %elem_ptr134, align 4
  %elem_ptr135 = getelementptr inbounds float, ptr %tensor_data117, i64 11
  store float 1.200000e+01, ptr %elem_ptr135, align 4
  %tensor_shape136 = alloca i64, i64 1, align 8
  %shape_ptr137 = getelementptr inbounds i64, ptr %tensor_shape136, i64 0
  store i64 12, ptr %shape_ptr137, align 8
  %new_tensor138 = call ptr @tl_tensor_new(ptr %tensor_data117, i64 1, ptr %tensor_shape136)
  store ptr %new_tensor138, ptr %target, align 8
  call void @tl_register_tensor(ptr @tensor_name.88, ptr %new_tensor138)
  %data139 = load ptr, ptr %data, align 8
  %dims_alloca = alloca [2 x i64], align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr, align 8
  %dim_ptr140 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr140, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %data139, ptr %dims_ptr, i64 2)
  store ptr %reshape_dims_res, ptr %X, align 8
  call void @tl_register_tensor(ptr @tensor_name.89, ptr %reshape_dims_res)
  %target141 = load ptr, ptr %target, align 8
  %dims_alloca142 = alloca [2 x i64], align 8
  %dim_ptr143 = getelementptr [2 x i64], ptr %dims_alloca142, i64 0, i64 0
  store i64 1, ptr %dim_ptr143, align 8
  %dim_ptr144 = getelementptr [2 x i64], ptr %dims_alloca142, i64 0, i64 1
  store i64 12, ptr %dim_ptr144, align 8
  %dims_ptr145 = getelementptr [2 x i64], ptr %dims_alloca142, i64 0, i64 0
  %reshape_dims_res146 = call ptr @tl_tensor_reshape_dims(ptr %target141, ptr %dims_ptr145, i64 2)
  store ptr %reshape_dims_res146, ptr %Y, align 8
  call void @tl_register_tensor(ptr @tensor_name.90, ptr %reshape_dims_res146)
  %model147 = load ptr, ptr %model1, align 8
  %X148 = load ptr, ptr %X, align 8
  %call_method = call ptr @tl_GPT_forward(ptr %model147, ptr %X148)
  store ptr %call_method, ptr %logits, align 8
  call void @tl_register_tensor(ptr @tensor_name.91, ptr %call_method)
  %logits149 = load ptr, ptr %logits, align 8
  %dims_alloca150 = alloca [2 x i64], align 8
  %dim_ptr151 = getelementptr [2 x i64], ptr %dims_alloca150, i64 0, i64 0
  store i64 12, ptr %dim_ptr151, align 8
  %dim_ptr152 = getelementptr [2 x i64], ptr %dims_alloca150, i64 0, i64 1
  store i64 13, ptr %dim_ptr152, align 8
  %dims_ptr153 = getelementptr [2 x i64], ptr %dims_alloca150, i64 0, i64 0
  %reshape_dims_res154 = call ptr @tl_tensor_reshape_dims(ptr %logits149, ptr %dims_ptr153, i64 2)
  store ptr %reshape_dims_res154, ptr %logits_flat, align 8
  call void @tl_register_tensor(ptr @tensor_name.92, ptr %reshape_dims_res154)
  %Y155 = load ptr, ptr %Y, align 8
  %dims_alloca156 = alloca [1 x i64], align 8
  %dim_ptr157 = getelementptr [1 x i64], ptr %dims_alloca156, i64 0, i64 0
  store i64 12, ptr %dim_ptr157, align 8
  %dims_ptr158 = getelementptr [1 x i64], ptr %dims_alloca156, i64 0, i64 0
  %reshape_dims_res159 = call ptr @tl_tensor_reshape_dims(ptr %Y155, ptr %dims_ptr158, i64 1)
  store ptr %reshape_dims_res159, ptr %Y_flat, align 8
  call void @tl_register_tensor(ptr @tensor_name.93, ptr %reshape_dims_res159)
  %logits_flat160 = load ptr, ptr %logits_flat, align 8
  %Y_flat161 = load ptr, ptr %Y_flat, align 8
  %ce_res = call ptr @tl_tensor_cross_entropy(ptr %logits_flat160, ptr %Y_flat161)
  store ptr %ce_res, ptr %loss, align 8
  call void @tl_register_tensor(ptr @tensor_name.94, ptr %ce_res)
  %loss162 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss162)
  %model163 = load ptr, ptr %model1, align 8
  %lr164 = load float, ptr %lr2, align 4
  %call_method165 = call ptr @tl_GPT_step(ptr %model163, float %lr164)
  store ptr %call_method165, ptr %model1, align 8
  %loss166 = load ptr, ptr %loss, align 8
  %detach_res = call ptr @tl_tensor_detach(ptr %loss166, i1 true)
  %old_val_to_free = load ptr, ptr %total_loss, align 8
  call void @tl_tensor_free(ptr %old_val_to_free)
  store ptr %detach_res, ptr %total_loss, align 8
  %load_for_free = load ptr, ptr %Y, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free167 = load ptr, ptr %logits, align 8
  call void @tl_tensor_free(ptr %load_for_free167)
  %load_for_free168 = load ptr, ptr %Y_flat, align 8
  call void @tl_tensor_free(ptr %load_for_free168)
  %load_for_free169 = load ptr, ptr %logits_flat, align 8
  call void @tl_tensor_free(ptr %load_for_free169)
  %load_for_free170 = load ptr, ptr %X, align 8
  call void @tl_tensor_free(ptr %load_for_free170)
  %load_for_free171 = load ptr, ptr %target, align 8
  call void @tl_tensor_free(ptr %load_for_free171)
  %load_for_free172 = load ptr, ptr %loss, align 8
  call void @tl_tensor_free(ptr %load_for_free172)
  %load_for_free173 = load ptr, ptr %data, align 8
  call void @tl_tensor_free(ptr %load_for_free173)
  %next_idx = add i64 %for_idx10, 1
  br label %for_header7

for_end9:                                         ; preds = %for_header7
  %next_idx174 = add i64 %for_idx, 1
  br label %for_header
}

define void @main() {
entry:
  %epoch = alloca i64, align 8
  %epochs = alloca i64, align 8
  %lr = alloca float, align 4
  %model = alloca ptr, align 8
  %d_model = alloca i64, align 8
  %vocab_size = alloca i64, align 8
  store i64 13, ptr %vocab_size, align 8
  store i64 64, ptr %d_model, align 8
  %vocab_size1 = load i64, ptr %vocab_size, align 8
  %d_model2 = load i64, ptr %d_model, align 8
  %static_call = call ptr @tl_GPT_new(i64 %vocab_size1, i64 %d_model2)
  store ptr %static_call, ptr %model, align 8
  store float 0x3F847AE140000000, ptr %lr, align 4
  store i64 50, ptr %epochs, align 8
  call void @tl_print_string(ptr @str_literal.96)
  %epochs3 = load i64, ptr %epochs, align 8
  br label %for_header

for_header:                                       ; preds = %for_body, %entry
  %for_idx = phi i64 [ 0, %entry ], [ %next_idx, %for_body ]
  %for_cond = icmp slt i64 %for_idx, %epochs3
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  store i64 %for_idx, ptr %epoch, align 8
  %model4 = load ptr, ptr %model, align 8
  %lr5 = load float, ptr %lr, align 4
  %epoch6 = load i64, ptr %epoch, align 8
  %call_tmp = call ptr @train_epoch(ptr %model4, float %lr5, i64 %epoch6)
  store ptr %call_tmp, ptr %model, align 8
  %next_idx = add i64 %for_idx, 1
  br label %for_header

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal.97)
  call void @tl_save_all_params(ptr @str_literal.98)
  ret void
}
