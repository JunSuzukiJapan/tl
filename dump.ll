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
@tensor_name.86 = private unnamed_addr constant [3 x i8] c"gb\00", align 1
@tensor_name.87 = private unnamed_addr constant [2 x i8] c"g\00", align 1
@tensor_name.88 = private unnamed_addr constant [3 x i8] c"gb\00", align 1
@tensor_name.89 = private unnamed_addr constant [2 x i8] c"q\00", align 1
@tensor_name.90 = private unnamed_addr constant [2 x i8] c"k\00", align 1
@tensor_name.91 = private unnamed_addr constant [2 x i8] c"v\00", align 1
@tensor_name.92 = private unnamed_addr constant [2 x i8] c"y\00", align 1
@tensor_name.93 = private unnamed_addr constant [2 x i8] c"x\00", align 1
@str_literal = private unnamed_addr constant [4 x i8] c"w.w\00", align 1
@str_literal.94 = private unnamed_addr constant [7 x i8] c"b.l1.w\00", align 1
@str_literal.95 = private unnamed_addr constant [7 x i8] c"b.l1.b\00", align 1
@str_literal.96 = private unnamed_addr constant [8 x i8] c"b.a.a.W\00", align 1
@str_literal.97 = private unnamed_addr constant [8 x i8] c"b.a.a.b\00", align 1
@str_literal.98 = private unnamed_addr constant [8 x i8] c"b.a.p.W\00", align 1
@str_literal.99 = private unnamed_addr constant [8 x i8] c"b.a.p.b\00", align 1
@str_literal.100 = private unnamed_addr constant [7 x i8] c"b.l2.w\00", align 1
@str_literal.101 = private unnamed_addr constant [7 x i8] c"b.l2.b\00", align 1
@str_literal.102 = private unnamed_addr constant [8 x i8] c"b.m.f.W\00", align 1
@str_literal.103 = private unnamed_addr constant [8 x i8] c"b.m.f.b\00", align 1
@str_literal.104 = private unnamed_addr constant [8 x i8] c"b.m.p.W\00", align 1
@str_literal.105 = private unnamed_addr constant [8 x i8] c"b.m.p.b\00", align 1
@str_literal.106 = private unnamed_addr constant [4 x i8] c"l.w\00", align 1
@str_literal.107 = private unnamed_addr constant [4 x i8] c"l.b\00", align 1
@str_literal.108 = private unnamed_addr constant [4 x i8] c"h.W\00", align 1
@str_literal.109 = private unnamed_addr constant [4 x i8] c"h.b\00", align 1
@str_literal.110 = private unnamed_addr constant [36 x i8] c"Initializing Model for Inference...\00", align 1
@str_literal.111 = private unnamed_addr constant [22 x i8] c"Loading Parameters...\00", align 1
@str_literal.112 = private unnamed_addr constant [25 x i8] c"model_2digit.safetensors\00", align 1
@str_literal.113 = private unnamed_addr constant [19 x i8] c"Parameters Loaded.\00", align 1
@str_literal.114 = private unnamed_addr constant [42 x i8] c"Running Inference verification 2-digit...\00", align 1
@str_literal.115 = private unnamed_addr constant [7 x i8] c"Input:\00", align 1
@str_literal.116 = private unnamed_addr constant [2 x i8] c"+\00", align 1
@str_literal.117 = private unnamed_addr constant [18 x i8] c"Predicted Digits:\00", align 1
@tensor_name.118 = private unnamed_addr constant [6 x i8] c"data1\00", align 1
@tensor_name.119 = private unnamed_addr constant [7 x i8] c"input1\00", align 1
@tensor_name.120 = private unnamed_addr constant [8 x i8] c"logits1\00", align 1
@tensor_name.121 = private unnamed_addr constant [8 x i8] c"logits1\00", align 1
@tensor_name.122 = private unnamed_addr constant [13 x i8] c"next_logits1\00", align 1
@tensor_name.123 = private unnamed_addr constant [6 x i8] c"data2\00", align 1
@tensor_name.124 = private unnamed_addr constant [7 x i8] c"input2\00", align 1
@tensor_name.125 = private unnamed_addr constant [8 x i8] c"logits2\00", align 1
@tensor_name.126 = private unnamed_addr constant [8 x i8] c"logits2\00", align 1
@tensor_name.127 = private unnamed_addr constant [13 x i8] c"next_logits2\00", align 1
@tensor_name.128 = private unnamed_addr constant [6 x i8] c"data3\00", align 1
@tensor_name.129 = private unnamed_addr constant [7 x i8] c"input3\00", align 1
@tensor_name.130 = private unnamed_addr constant [8 x i8] c"logits3\00", align 1
@tensor_name.131 = private unnamed_addr constant [8 x i8] c"logits3\00", align 1
@tensor_name.132 = private unnamed_addr constant [13 x i8] c"next_logits3\00", align 1
@str_literal.133 = private unnamed_addr constant [33 x i8] c"Inference Verification Complete.\00", align 1

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

declare void @tl_add_parameter(ptr, ptr)

declare void @tl_load_all_params(ptr)

declare void @tl_tensor_sub_assign.1(ptr, ptr)

declare void @tl_add_parameter.2(ptr, ptr)

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

declare i64 @tl_get_memory_mb()

declare void @tl_mem_enter_scope()

declare void @tl_mem_exit_scope()

declare void @tl_mem_register_struct(ptr)

declare void @tl_mem_register_tensor(ptr)

declare void @tl_mem_unregister(ptr)

declare void @tl_print_i64.3(i64)

declare void @tl_print_f32.4(float)

declare void @tl_print_string.5(ptr)

declare ptr @malloc.6(i64)

declare ptr @calloc.7(i64, i64)

declare void @free.8(ptr)

declare i64 @tl_tensor_dim.9(ptr, i64)

declare float @tl_tensor_get_f32_md.10(ptr, ptr, i64)

declare ptr @tl_tensor_new.11(ptr, i64, ptr)

declare ptr @tl_tensor_sub.12(ptr, ptr)

declare void @tl_tensor_free.13(ptr)

declare ptr @tl_tensor_clone.14(ptr)

declare ptr @tl_tensor_add.15(ptr, ptr)

declare ptr @tl_tensor_mul.16(ptr, ptr)

declare void @tl_tensor_print.17(ptr)

declare float @tl_tensor_get.18(ptr, i64)

declare ptr @tl_tensor_slice.19(ptr, i64, i64)

declare i64 @tl_tensor_len.20(ptr)

declare ptr @tl_tensor_neg.21(ptr)

declare ptr @tl_tensor_transpose.22(ptr, i64, i64)

declare ptr @tl_tensor_pow.23(ptr, ptr)

declare ptr @tl_tensor_sqrt.24(ptr)

declare ptr @tl_tensor_sin.25(ptr)

declare ptr @tl_tensor_cos.26(ptr)

declare ptr @tl_tensor_relu.27(ptr)

declare ptr @tl_tensor_gelu.28(ptr)

declare ptr @tl_tensor_tril.29(ptr, i32)

declare ptr @tl_tensor_sum_dim.30(ptr, i64, i1)

declare ptr @tl_tensor_embedding.31(ptr, ptr)

declare ptr @tl_tensor_sum.32(ptr)

declare ptr @tl_tensor_div.33(ptr, ptr)

declare ptr @tl_tensor_matmul.34(ptr, ptr)

declare ptr @tl_tensor_exp.35(ptr)

declare ptr @tl_tensor_log.36(ptr)

declare void @tl_tensor_add_assign.37(ptr, ptr)

declare void @tl_tensor_sub_assign.38(ptr, ptr)

declare void @tl_tensor_mul_assign.39(ptr, ptr)

declare void @tl_tensor_div_assign.40(ptr, ptr)

declare void @tl_register_tensor.41(ptr, ptr)

declare i32 @strcmp.42(ptr, ptr)

declare ptr @tl_tensor_reshape_dims.43(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.44(ptr, ptr)

declare ptr @tl_tensor_randn.45(i64, ptr, i1)

declare ptr @tl_varbuilder_get.46(ptr, i64, ptr)

declare void @tl_update_all_params.47(float)

declare ptr @tl_varbuilder_grad.48(ptr)

declare void @tl_tensor_backward.49(ptr)

declare ptr @tl_tensor_grad.50(ptr)

declare ptr @tl_tensor_detach.51(ptr, i1)

declare ptr @tl_tensor_softmax.52(ptr, i64)

declare ptr @tl_tensor_cross_entropy.53(ptr, ptr)

declare void @tl_tensor_save.54(ptr, ptr)

declare ptr @tl_tensor_load.55(ptr)

declare void @tl_save_all_params.56(ptr)

declare void @tl_add_parameter.57(ptr, ptr)

declare void @tl_load_all_params.58(ptr)

declare void @tl_tensor_sub_assign.59(ptr, ptr)

declare void @tl_add_parameter.60(ptr, ptr)

declare ptr @tl_register_parameter.61(ptr)

declare ptr @tl_string_concat.62(ptr, ptr)

declare ptr @tl_file_open.63(ptr, ptr)

declare ptr @tl_file_read_string.64(ptr)

declare void @tl_file_write_string.65(ptr, ptr)

declare void @tl_file_close.66(ptr)

declare ptr @tl_path_new.67(ptr)

declare ptr @tl_path_join.68(ptr, ptr)

declare i1 @tl_path_exists.69(ptr)

declare i1 @tl_path_is_dir.70(ptr)

declare i1 @tl_path_is_file.71(ptr)

declare ptr @tl_path_to_string.72(ptr)

declare void @tl_path_free.73(ptr)

declare i1 @tl_http_download.74(ptr, ptr)

declare ptr @tl_http_get.75(ptr)

declare ptr @tl_env_get.76(ptr)

declare void @tl_env_set.77(ptr, ptr)

declare float @tl_system_time.78()

declare void @tl_system_sleep.79(float)

declare i64 @tl_get_memory_mb.80()

declare void @tl_mem_enter_scope.81()

declare void @tl_mem_exit_scope.82()

declare void @tl_mem_register_struct.83(ptr)

declare void @tl_mem_register_tensor.84(ptr)

declare void @tl_mem_unregister.85(ptr)

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
  call void @tl_register_tensor(ptr @tensor_name.86, ptr %grad_res6)
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
  store ptr %detach_res27, ptr %ptr_b16, align 8
  %s29 = load ptr, ptr %s, align 8
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
  call void @tl_register_tensor(ptr @tensor_name.87, ptr %grad_res)
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
  store ptr %detach_res, ptr %ptr_w6, align 8
  %s13 = load ptr, ptr %s, align 8
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
  call void @tl_register_tensor(ptr @tensor_name.88, ptr %grad_res)
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
  store ptr %detach_res, ptr %ptr_b6, align 8
  %s13 = load ptr, ptr %s, align 8
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
  call void @tl_register_tensor(ptr @tensor_name.89, ptr %call_method)
  %q5 = load ptr, ptr %q, align 8
  %cloned = call ptr @tl_tensor_clone(ptr %q5)
  store ptr %cloned, ptr %k, align 8
  call void @tl_register_tensor(ptr @tensor_name.90, ptr %cloned)
  %q6 = load ptr, ptr %q, align 8
  %cloned7 = call ptr @tl_tensor_clone(ptr %q6)
  store ptr %cloned7, ptr %v, align 8
  call void @tl_register_tensor(ptr @tensor_name.91, ptr %cloned7)
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
  call void @tl_register_tensor(ptr @tensor_name.92, ptr %matmul_res11)
  %self12 = load ptr, ptr %self1, align 8
  %ptr_p = getelementptr inbounds nuw %CausalSelfAttention, ptr %self12, i32 0, i32 1
  %p = load ptr, ptr %ptr_p, align 8
  %y13 = load ptr, ptr %y, align 8
  %call_method14 = call ptr @tl_Linear_forward(ptr %p, ptr %y13)
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
  call void @tl_register_tensor(ptr @tensor_name.93, ptr %binop_res)
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

define i64 @get_memory() {
entry:
  %call_tmp = call i64 @tl_get_memory_mb()
  ret i64 %call_tmp
}

define void @register_gpt_params(ptr %m) {
entry:
  %m1 = alloca ptr, align 8
  store ptr %m, ptr %m1, align 8
  %m2 = load ptr, ptr %m1, align 8
  %ptr_w = getelementptr inbounds nuw %GPT, ptr %m2, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %ptr_w3 = getelementptr inbounds nuw %Embedding, ptr %w, i32 0, i32 0
  %w4 = load ptr, ptr %ptr_w3, align 8
  call void @tl_add_parameter(ptr @str_literal, ptr %w4)
  %m5 = load ptr, ptr %m1, align 8
  %ptr_b = getelementptr inbounds nuw %GPT, ptr %m5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %ptr_l1 = getelementptr inbounds nuw %Block, ptr %b, i32 0, i32 0
  %l1 = load ptr, ptr %ptr_l1, align 8
  %ptr_w6 = getelementptr inbounds nuw %LayerNorm, ptr %l1, i32 0, i32 0
  %w7 = load ptr, ptr %ptr_w6, align 8
  call void @tl_add_parameter(ptr @str_literal.94, ptr %w7)
  %m8 = load ptr, ptr %m1, align 8
  %ptr_b9 = getelementptr inbounds nuw %GPT, ptr %m8, i32 0, i32 1
  %b10 = load ptr, ptr %ptr_b9, align 8
  %ptr_l111 = getelementptr inbounds nuw %Block, ptr %b10, i32 0, i32 0
  %l112 = load ptr, ptr %ptr_l111, align 8
  %ptr_b13 = getelementptr inbounds nuw %LayerNorm, ptr %l112, i32 0, i32 1
  %b14 = load ptr, ptr %ptr_b13, align 8
  call void @tl_add_parameter(ptr @str_literal.95, ptr %b14)
  %m15 = load ptr, ptr %m1, align 8
  %ptr_b16 = getelementptr inbounds nuw %GPT, ptr %m15, i32 0, i32 1
  %b17 = load ptr, ptr %ptr_b16, align 8
  %ptr_a = getelementptr inbounds nuw %Block, ptr %b17, i32 0, i32 1
  %a = load ptr, ptr %ptr_a, align 8
  %ptr_a18 = getelementptr inbounds nuw %CausalSelfAttention, ptr %a, i32 0, i32 0
  %a19 = load ptr, ptr %ptr_a18, align 8
  %ptr_W = getelementptr inbounds nuw %Linear, ptr %a19, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  call void @tl_add_parameter(ptr @str_literal.96, ptr %W)
  %m20 = load ptr, ptr %m1, align 8
  %ptr_b21 = getelementptr inbounds nuw %GPT, ptr %m20, i32 0, i32 1
  %b22 = load ptr, ptr %ptr_b21, align 8
  %ptr_a23 = getelementptr inbounds nuw %Block, ptr %b22, i32 0, i32 1
  %a24 = load ptr, ptr %ptr_a23, align 8
  %ptr_a25 = getelementptr inbounds nuw %CausalSelfAttention, ptr %a24, i32 0, i32 0
  %a26 = load ptr, ptr %ptr_a25, align 8
  %ptr_b27 = getelementptr inbounds nuw %Linear, ptr %a26, i32 0, i32 1
  %b28 = load ptr, ptr %ptr_b27, align 8
  call void @tl_add_parameter(ptr @str_literal.97, ptr %b28)
  %m29 = load ptr, ptr %m1, align 8
  %ptr_b30 = getelementptr inbounds nuw %GPT, ptr %m29, i32 0, i32 1
  %b31 = load ptr, ptr %ptr_b30, align 8
  %ptr_a32 = getelementptr inbounds nuw %Block, ptr %b31, i32 0, i32 1
  %a33 = load ptr, ptr %ptr_a32, align 8
  %ptr_p = getelementptr inbounds nuw %CausalSelfAttention, ptr %a33, i32 0, i32 1
  %p = load ptr, ptr %ptr_p, align 8
  %ptr_W34 = getelementptr inbounds nuw %Linear, ptr %p, i32 0, i32 0
  %W35 = load ptr, ptr %ptr_W34, align 8
  call void @tl_add_parameter(ptr @str_literal.98, ptr %W35)
  %m36 = load ptr, ptr %m1, align 8
  %ptr_b37 = getelementptr inbounds nuw %GPT, ptr %m36, i32 0, i32 1
  %b38 = load ptr, ptr %ptr_b37, align 8
  %ptr_a39 = getelementptr inbounds nuw %Block, ptr %b38, i32 0, i32 1
  %a40 = load ptr, ptr %ptr_a39, align 8
  %ptr_p41 = getelementptr inbounds nuw %CausalSelfAttention, ptr %a40, i32 0, i32 1
  %p42 = load ptr, ptr %ptr_p41, align 8
  %ptr_b43 = getelementptr inbounds nuw %Linear, ptr %p42, i32 0, i32 1
  %b44 = load ptr, ptr %ptr_b43, align 8
  call void @tl_add_parameter(ptr @str_literal.99, ptr %b44)
  %m45 = load ptr, ptr %m1, align 8
  %ptr_b46 = getelementptr inbounds nuw %GPT, ptr %m45, i32 0, i32 1
  %b47 = load ptr, ptr %ptr_b46, align 8
  %ptr_l2 = getelementptr inbounds nuw %Block, ptr %b47, i32 0, i32 2
  %l2 = load ptr, ptr %ptr_l2, align 8
  %ptr_w48 = getelementptr inbounds nuw %LayerNorm, ptr %l2, i32 0, i32 0
  %w49 = load ptr, ptr %ptr_w48, align 8
  call void @tl_add_parameter(ptr @str_literal.100, ptr %w49)
  %m50 = load ptr, ptr %m1, align 8
  %ptr_b51 = getelementptr inbounds nuw %GPT, ptr %m50, i32 0, i32 1
  %b52 = load ptr, ptr %ptr_b51, align 8
  %ptr_l253 = getelementptr inbounds nuw %Block, ptr %b52, i32 0, i32 2
  %l254 = load ptr, ptr %ptr_l253, align 8
  %ptr_b55 = getelementptr inbounds nuw %LayerNorm, ptr %l254, i32 0, i32 1
  %b56 = load ptr, ptr %ptr_b55, align 8
  call void @tl_add_parameter(ptr @str_literal.101, ptr %b56)
  %m57 = load ptr, ptr %m1, align 8
  %ptr_b58 = getelementptr inbounds nuw %GPT, ptr %m57, i32 0, i32 1
  %b59 = load ptr, ptr %ptr_b58, align 8
  %ptr_m = getelementptr inbounds nuw %Block, ptr %b59, i32 0, i32 3
  %m60 = load ptr, ptr %ptr_m, align 8
  %ptr_f = getelementptr inbounds nuw %MLP, ptr %m60, i32 0, i32 0
  %f = load ptr, ptr %ptr_f, align 8
  %ptr_W61 = getelementptr inbounds nuw %Linear, ptr %f, i32 0, i32 0
  %W62 = load ptr, ptr %ptr_W61, align 8
  call void @tl_add_parameter(ptr @str_literal.102, ptr %W62)
  %m63 = load ptr, ptr %m1, align 8
  %ptr_b64 = getelementptr inbounds nuw %GPT, ptr %m63, i32 0, i32 1
  %b65 = load ptr, ptr %ptr_b64, align 8
  %ptr_m66 = getelementptr inbounds nuw %Block, ptr %b65, i32 0, i32 3
  %m67 = load ptr, ptr %ptr_m66, align 8
  %ptr_f68 = getelementptr inbounds nuw %MLP, ptr %m67, i32 0, i32 0
  %f69 = load ptr, ptr %ptr_f68, align 8
  %ptr_b70 = getelementptr inbounds nuw %Linear, ptr %f69, i32 0, i32 1
  %b71 = load ptr, ptr %ptr_b70, align 8
  call void @tl_add_parameter(ptr @str_literal.103, ptr %b71)
  %m72 = load ptr, ptr %m1, align 8
  %ptr_b73 = getelementptr inbounds nuw %GPT, ptr %m72, i32 0, i32 1
  %b74 = load ptr, ptr %ptr_b73, align 8
  %ptr_m75 = getelementptr inbounds nuw %Block, ptr %b74, i32 0, i32 3
  %m76 = load ptr, ptr %ptr_m75, align 8
  %ptr_p77 = getelementptr inbounds nuw %MLP, ptr %m76, i32 0, i32 1
  %p78 = load ptr, ptr %ptr_p77, align 8
  %ptr_W79 = getelementptr inbounds nuw %Linear, ptr %p78, i32 0, i32 0
  %W80 = load ptr, ptr %ptr_W79, align 8
  call void @tl_add_parameter(ptr @str_literal.104, ptr %W80)
  %m81 = load ptr, ptr %m1, align 8
  %ptr_b82 = getelementptr inbounds nuw %GPT, ptr %m81, i32 0, i32 1
  %b83 = load ptr, ptr %ptr_b82, align 8
  %ptr_m84 = getelementptr inbounds nuw %Block, ptr %b83, i32 0, i32 3
  %m85 = load ptr, ptr %ptr_m84, align 8
  %ptr_p86 = getelementptr inbounds nuw %MLP, ptr %m85, i32 0, i32 1
  %p87 = load ptr, ptr %ptr_p86, align 8
  %ptr_b88 = getelementptr inbounds nuw %Linear, ptr %p87, i32 0, i32 1
  %b89 = load ptr, ptr %ptr_b88, align 8
  call void @tl_add_parameter(ptr @str_literal.105, ptr %b89)
  %m90 = load ptr, ptr %m1, align 8
  %ptr_l = getelementptr inbounds nuw %GPT, ptr %m90, i32 0, i32 2
  %l = load ptr, ptr %ptr_l, align 8
  %ptr_w91 = getelementptr inbounds nuw %LayerNorm, ptr %l, i32 0, i32 0
  %w92 = load ptr, ptr %ptr_w91, align 8
  call void @tl_add_parameter(ptr @str_literal.106, ptr %w92)
  %m93 = load ptr, ptr %m1, align 8
  %ptr_l94 = getelementptr inbounds nuw %GPT, ptr %m93, i32 0, i32 2
  %l95 = load ptr, ptr %ptr_l94, align 8
  %ptr_b96 = getelementptr inbounds nuw %LayerNorm, ptr %l95, i32 0, i32 1
  %b97 = load ptr, ptr %ptr_b96, align 8
  call void @tl_add_parameter(ptr @str_literal.107, ptr %b97)
  %m98 = load ptr, ptr %m1, align 8
  %ptr_h = getelementptr inbounds nuw %GPT, ptr %m98, i32 0, i32 3
  %h = load ptr, ptr %ptr_h, align 8
  %ptr_W99 = getelementptr inbounds nuw %Linear, ptr %h, i32 0, i32 0
  %W100 = load ptr, ptr %ptr_W99, align 8
  call void @tl_add_parameter(ptr @str_literal.108, ptr %W100)
  %m101 = load ptr, ptr %m1, align 8
  %ptr_h102 = getelementptr inbounds nuw %GPT, ptr %m101, i32 0, i32 3
  %h103 = load ptr, ptr %ptr_h102, align 8
  %ptr_b104 = getelementptr inbounds nuw %Linear, ptr %h103, i32 0, i32 1
  %b105 = load ptr, ptr %ptr_b104, align 8
  call void @tl_add_parameter(ptr @str_literal.109, ptr %b105)
  ret void
}

define void @main() {
entry:
  %val270 = alloca float, align 4
  %k266 = alloca i64, align 8
  %max_val260 = alloca float, align 4
  %pred3 = alloca i64, align 8
  %next_logits3 = alloca ptr, align 8
  %logits3257 = alloca ptr, align 8
  %logits3 = alloca ptr, align 8
  %input3 = alloca ptr, align 8
  %data3 = alloca ptr, align 8
  %tensor_shape_arr239 = alloca i64, i64 1, align 8
  %tensor_data_arr217 = alloca float, i64 12, align 4
  %val_pred2 = alloca float, align 4
  %scalar_shape213 = alloca i64, align 8
  %scalar_data212 = alloca float, align 4
  %scalar_shape210 = alloca i64, align 8
  %scalar_data208 = alloca float, align 4
  %val196 = alloca float, align 4
  %k192 = alloca i64, align 8
  %max_val186 = alloca float, align 4
  %pred2 = alloca i64, align 8
  %next_logits2 = alloca ptr, align 8
  %logits2183 = alloca ptr, align 8
  %logits2 = alloca ptr, align 8
  %input2 = alloca ptr, align 8
  %data2 = alloca ptr, align 8
  %tensor_shape_arr165 = alloca i64, i64 1, align 8
  %tensor_data_arr143 = alloca float, i64 12, align 4
  %val_pred1 = alloca float, align 4
  %scalar_shape139 = alloca i64, align 8
  %scalar_data138 = alloca float, align 4
  %scalar_shape136 = alloca i64, align 8
  %scalar_data134 = alloca float, align 4
  %val = alloca float, align 4
  %k = alloca i64, align 8
  %max_val = alloca float, align 4
  %pred1 = alloca i64, align 8
  %next_logits1 = alloca ptr, align 8
  %logits1115 = alloca ptr, align 8
  %logits1 = alloca ptr, align 8
  %input1 = alloca ptr, align 8
  %data1 = alloca ptr, align 8
  %tensor_shape_arr = alloca i64, i64 1, align 8
  %tensor_data_arr = alloca float, i64 12, align 4
  %pos = alloca i64, align 8
  %x11 = alloca float, align 4
  %x10 = alloca float, align 4
  %x9 = alloca float, align 4
  %x8 = alloca float, align 4
  %x7 = alloca float, align 4
  %x6 = alloca float, align 4
  %x5 = alloca float, align 4
  %x4 = alloca float, align 4
  %x3 = alloca float, align 4
  %x2 = alloca float, align 4
  %x1 = alloca float, align 4
  %x0 = alloca float, align 4
  %val_pad = alloca float, align 4
  %val_eq = alloca float, align 4
  %val_plus = alloca float, align 4
  %j_d2 = alloca float, align 4
  %scalar_shape61 = alloca i64, align 8
  %scalar_data60 = alloca float, align 4
  %scalar_shape58 = alloca i64, align 8
  %scalar_data56 = alloca float, align 4
  %j_d1 = alloca float, align 4
  %scalar_shape47 = alloca i64, align 8
  %scalar_data46 = alloca float, align 4
  %scalar_shape44 = alloca i64, align 8
  %scalar_data42 = alloca float, align 4
  %i_d2 = alloca float, align 4
  %scalar_shape36 = alloca i64, align 8
  %scalar_data35 = alloca float, align 4
  %scalar_shape33 = alloca i64, align 8
  %scalar_data31 = alloca float, align 4
  %i_d1 = alloca float, align 4
  %scalar_shape26 = alloca i64, align 8
  %scalar_data25 = alloca float, align 4
  %scalar_shape = alloca i64, align 8
  %scalar_data = alloca float, align 4
  %j = alloca i64, align 8
  %i = alloca i64, align 8
  %t = alloca i64, align 8
  %model6 = alloca ptr, align 8
  %model = alloca ptr, align 8
  %block_size = alloca i64, align 8
  %d_model = alloca i64, align 8
  %vocab_size = alloca i64, align 8
  store i64 13, ptr %vocab_size, align 8
  store i64 64, ptr %d_model, align 8
  store i64 12, ptr %block_size, align 8
  call void @tl_print_string(ptr @str_literal.110)
  %vocab_size1 = load i64, ptr %vocab_size, align 8
  %d_model2 = load i64, ptr %d_model, align 8
  %static_call = call ptr @tl_GPT_new(i64 %vocab_size1, i64 %d_model2)
  store ptr %static_call, ptr %model, align 8
  %vocab_size3 = load i64, ptr %vocab_size, align 8
  %d_model4 = load i64, ptr %d_model, align 8
  %static_call5 = call ptr @tl_GPT_new(i64 %vocab_size3, i64 %d_model4)
  store ptr %static_call5, ptr %model6, align 8
  %model7 = load ptr, ptr %model6, align 8
  call void @register_gpt_params(ptr %model7)
  call void @tl_print_string(ptr @str_literal.111)
  call void @tl_load_all_params(ptr @str_literal.112)
  call void @tl_print_string(ptr @str_literal.113)
  call void @tl_print_string(ptr @str_literal.114)
  br label %for_header

for_header:                                       ; preds = %for_end263, %entry
  %for_idx = phi i64 [ 0, %entry ], [ %next_idx281, %for_end263 ]
  %for_cond = icmp slt i64 %for_idx, 4
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %t, align 8
  store i64 0, ptr %i, align 8
  store i64 0, ptr %j, align 8
  %t8 = load i64, ptr %t, align 8
  %eqtmp = icmp eq i64 %t8, 0
  br i1 %eqtmp, label %then, label %else

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal.133)
  ret void

then:                                             ; preds = %for_body
  store i64 12, ptr %i, align 8
  store i64 34, ptr %j, align 8
  br label %merge

else:                                             ; preds = %for_body
  br label %merge

merge:                                            ; preds = %else, %then
  %t9 = load i64, ptr %t, align 8
  %eqtmp10 = icmp eq i64 %t9, 1
  br i1 %eqtmp10, label %then11, label %else12

then11:                                           ; preds = %merge
  store i64 99, ptr %i, align 8
  store i64 1, ptr %j, align 8
  br label %merge13

else12:                                           ; preds = %merge
  br label %merge13

merge13:                                          ; preds = %else12, %then11
  %t14 = load i64, ptr %t, align 8
  %eqtmp15 = icmp eq i64 %t14, 2
  br i1 %eqtmp15, label %then16, label %else17

then16:                                           ; preds = %merge13
  store i64 5, ptr %i, align 8
  store i64 5, ptr %j, align 8
  br label %merge18

else17:                                           ; preds = %merge13
  br label %merge18

merge18:                                          ; preds = %else17, %then16
  %t19 = load i64, ptr %t, align 8
  %eqtmp20 = icmp eq i64 %t19, 3
  br i1 %eqtmp20, label %then21, label %else22

then21:                                           ; preds = %merge18
  store i64 88, ptr %i, align 8
  store i64 99, ptr %j, align 8
  br label %merge23

else22:                                           ; preds = %merge18
  br label %merge23

merge23:                                          ; preds = %else22, %then21
  %i24 = load i64, ptr %i, align 8
  %divtmp = sdiv i64 %i24, 10
  %cast_i64_f32 = sitofp i64 %divtmp to float
  store float %cast_i64_f32, ptr %scalar_data, align 4
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  store float 1.000000e+00, ptr %scalar_data25, align 4
  %scalar_tensor27 = call ptr @tl_tensor_new(ptr %scalar_data25, i64 0, ptr %scalar_shape26)
  %pow_res = call ptr @tl_tensor_pow(ptr %scalar_tensor, ptr %scalar_tensor27)
  %get_res = call float @tl_tensor_get(ptr %pow_res, i64 0)
  store float %get_res, ptr %i_d1, align 4
  %i28 = load i64, ptr %i, align 8
  %i29 = load i64, ptr %i, align 8
  %divtmp30 = sdiv i64 %i29, 10
  %multmp = mul i64 %divtmp30, 10
  %subtmp = sub i64 %i28, %multmp
  %cast_i64_f3232 = sitofp i64 %subtmp to float
  store float %cast_i64_f3232, ptr %scalar_data31, align 4
  %scalar_tensor34 = call ptr @tl_tensor_new(ptr %scalar_data31, i64 0, ptr %scalar_shape33)
  store float 1.000000e+00, ptr %scalar_data35, align 4
  %scalar_tensor37 = call ptr @tl_tensor_new(ptr %scalar_data35, i64 0, ptr %scalar_shape36)
  %pow_res38 = call ptr @tl_tensor_pow(ptr %scalar_tensor34, ptr %scalar_tensor37)
  %get_res39 = call float @tl_tensor_get(ptr %pow_res38, i64 0)
  store float %get_res39, ptr %i_d2, align 4
  %j40 = load i64, ptr %j, align 8
  %divtmp41 = sdiv i64 %j40, 10
  %cast_i64_f3243 = sitofp i64 %divtmp41 to float
  store float %cast_i64_f3243, ptr %scalar_data42, align 4
  %scalar_tensor45 = call ptr @tl_tensor_new(ptr %scalar_data42, i64 0, ptr %scalar_shape44)
  store float 1.000000e+00, ptr %scalar_data46, align 4
  %scalar_tensor48 = call ptr @tl_tensor_new(ptr %scalar_data46, i64 0, ptr %scalar_shape47)
  %pow_res49 = call ptr @tl_tensor_pow(ptr %scalar_tensor45, ptr %scalar_tensor48)
  %get_res50 = call float @tl_tensor_get(ptr %pow_res49, i64 0)
  store float %get_res50, ptr %j_d1, align 4
  %j51 = load i64, ptr %j, align 8
  %j52 = load i64, ptr %j, align 8
  %divtmp53 = sdiv i64 %j52, 10
  %multmp54 = mul i64 %divtmp53, 10
  %subtmp55 = sub i64 %j51, %multmp54
  %cast_i64_f3257 = sitofp i64 %subtmp55 to float
  store float %cast_i64_f3257, ptr %scalar_data56, align 4
  %scalar_tensor59 = call ptr @tl_tensor_new(ptr %scalar_data56, i64 0, ptr %scalar_shape58)
  store float 1.000000e+00, ptr %scalar_data60, align 4
  %scalar_tensor62 = call ptr @tl_tensor_new(ptr %scalar_data60, i64 0, ptr %scalar_shape61)
  %pow_res63 = call ptr @tl_tensor_pow(ptr %scalar_tensor59, ptr %scalar_tensor62)
  %get_res64 = call float @tl_tensor_get(ptr %pow_res63, i64 0)
  store float %get_res64, ptr %j_d2, align 4
  store float 1.000000e+01, ptr %val_plus, align 4
  store float 1.100000e+01, ptr %val_eq, align 4
  store float 1.200000e+01, ptr %val_pad, align 4
  %val_pad65 = load float, ptr %val_pad, align 4
  store float %val_pad65, ptr %x0, align 4
  %val_pad66 = load float, ptr %val_pad, align 4
  store float %val_pad66, ptr %x1, align 4
  %val_pad67 = load float, ptr %val_pad, align 4
  store float %val_pad67, ptr %x2, align 4
  %val_pad68 = load float, ptr %val_pad, align 4
  store float %val_pad68, ptr %x3, align 4
  %val_pad69 = load float, ptr %val_pad, align 4
  store float %val_pad69, ptr %x4, align 4
  %val_pad70 = load float, ptr %val_pad, align 4
  store float %val_pad70, ptr %x5, align 4
  %val_pad71 = load float, ptr %val_pad, align 4
  store float %val_pad71, ptr %x6, align 4
  %val_pad72 = load float, ptr %val_pad, align 4
  store float %val_pad72, ptr %x7, align 4
  %val_pad73 = load float, ptr %val_pad, align 4
  store float %val_pad73, ptr %x8, align 4
  %val_pad74 = load float, ptr %val_pad, align 4
  store float %val_pad74, ptr %x9, align 4
  %val_pad75 = load float, ptr %val_pad, align 4
  store float %val_pad75, ptr %x10, align 4
  %val_pad76 = load float, ptr %val_pad, align 4
  store float %val_pad76, ptr %x11, align 4
  store i64 0, ptr %pos, align 8
  %i_d177 = load float, ptr %i_d1, align 4
  store float %i_d177, ptr %x0, align 4
  %i_d278 = load float, ptr %i_d2, align 4
  store float %i_d278, ptr %x1, align 4
  %val_plus79 = load float, ptr %val_plus, align 4
  store float %val_plus79, ptr %x2, align 4
  %j_d180 = load float, ptr %j_d1, align 4
  store float %j_d180, ptr %x3, align 4
  %j_d281 = load float, ptr %j_d2, align 4
  store float %j_d281, ptr %x4, align 4
  %val_eq82 = load float, ptr %val_eq, align 4
  store float %val_eq82, ptr %x5, align 4
  store i64 6, ptr %pos, align 8
  call void @tl_print_string(ptr @str_literal.115)
  %i83 = load i64, ptr %i, align 8
  call void @tl_print_i64(i64 %i83)
  call void @tl_print_string(ptr @str_literal.116)
  %j84 = load i64, ptr %j, align 8
  call void @tl_print_i64(i64 %j84)
  call void @tl_print_string(ptr @str_literal.117)
  %x085 = load float, ptr %x0, align 4
  %elem_ptr = getelementptr inbounds float, ptr %tensor_data_arr, i64 0
  store float %x085, ptr %elem_ptr, align 4
  %x186 = load float, ptr %x1, align 4
  %elem_ptr87 = getelementptr inbounds float, ptr %tensor_data_arr, i64 1
  store float %x186, ptr %elem_ptr87, align 4
  %x288 = load float, ptr %x2, align 4
  %elem_ptr89 = getelementptr inbounds float, ptr %tensor_data_arr, i64 2
  store float %x288, ptr %elem_ptr89, align 4
  %x390 = load float, ptr %x3, align 4
  %elem_ptr91 = getelementptr inbounds float, ptr %tensor_data_arr, i64 3
  store float %x390, ptr %elem_ptr91, align 4
  %x492 = load float, ptr %x4, align 4
  %elem_ptr93 = getelementptr inbounds float, ptr %tensor_data_arr, i64 4
  store float %x492, ptr %elem_ptr93, align 4
  %x594 = load float, ptr %x5, align 4
  %elem_ptr95 = getelementptr inbounds float, ptr %tensor_data_arr, i64 5
  store float %x594, ptr %elem_ptr95, align 4
  %val_pad96 = load float, ptr %val_pad, align 4
  %elem_ptr97 = getelementptr inbounds float, ptr %tensor_data_arr, i64 6
  store float %val_pad96, ptr %elem_ptr97, align 4
  %val_pad98 = load float, ptr %val_pad, align 4
  %elem_ptr99 = getelementptr inbounds float, ptr %tensor_data_arr, i64 7
  store float %val_pad98, ptr %elem_ptr99, align 4
  %val_pad100 = load float, ptr %val_pad, align 4
  %elem_ptr101 = getelementptr inbounds float, ptr %tensor_data_arr, i64 8
  store float %val_pad100, ptr %elem_ptr101, align 4
  %elem_ptr102 = getelementptr inbounds float, ptr %tensor_data_arr, i64 9
  store float 1.200000e+01, ptr %elem_ptr102, align 4
  %elem_ptr103 = getelementptr inbounds float, ptr %tensor_data_arr, i64 10
  store float 1.200000e+01, ptr %elem_ptr103, align 4
  %elem_ptr104 = getelementptr inbounds float, ptr %tensor_data_arr, i64 11
  store float 1.200000e+01, ptr %elem_ptr104, align 4
  %shape_ptr = getelementptr inbounds i64, ptr %tensor_shape_arr, i64 0
  store i64 12, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %tensor_data_arr, i64 1, ptr %tensor_shape_arr)
  store ptr %new_tensor, ptr %data1, align 8
  call void @tl_register_tensor(ptr @tensor_name.118, ptr %new_tensor)
  %data1105 = load ptr, ptr %data1, align 8
  %dims_alloca = alloca [2 x i64], align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr, align 8
  %dim_ptr106 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr106, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %data1105, ptr %dims_ptr, i64 2)
  store ptr %reshape_dims_res, ptr %input1, align 8
  call void @tl_register_tensor(ptr @tensor_name.119, ptr %reshape_dims_res)
  %model107 = load ptr, ptr %model6, align 8
  %input1108 = load ptr, ptr %input1, align 8
  %call_method = call ptr @tl_GPT_forward(ptr %model107, ptr %input1108)
  store ptr %call_method, ptr %logits1, align 8
  call void @tl_register_tensor(ptr @tensor_name.120, ptr %call_method)
  %logits1109 = load ptr, ptr %logits1, align 8
  %dims_alloca110 = alloca [2 x i64], align 8
  %dim_ptr111 = getelementptr [2 x i64], ptr %dims_alloca110, i64 0, i64 0
  store i64 12, ptr %dim_ptr111, align 8
  %dim_ptr112 = getelementptr [2 x i64], ptr %dims_alloca110, i64 0, i64 1
  store i64 13, ptr %dim_ptr112, align 8
  %dims_ptr113 = getelementptr [2 x i64], ptr %dims_alloca110, i64 0, i64 0
  %reshape_dims_res114 = call ptr @tl_tensor_reshape_dims(ptr %logits1109, ptr %dims_ptr113, i64 2)
  store ptr %reshape_dims_res114, ptr %logits1115, align 8
  call void @tl_register_tensor(ptr @tensor_name.121, ptr %reshape_dims_res114)
  %logits1116 = load ptr, ptr %logits1115, align 8
  %slice_res = call ptr @tl_tensor_slice(ptr %logits1116, i64 5, i64 1)
  store ptr %slice_res, ptr %next_logits1, align 8
  call void @tl_register_tensor(ptr @tensor_name.122, ptr %slice_res)
  store i64 0, ptr %pred1, align 8
  store float -1.000000e+06, ptr %max_val, align 4
  br label %for_header117

for_header117:                                    ; preds = %merge129, %merge23
  %for_idx120 = phi i64 [ 0, %merge23 ], [ %next_idx, %merge129 ]
  %for_cond121 = icmp slt i64 %for_idx120, 13
  br i1 %for_cond121, label %for_body118, label %for_end119

for_body118:                                      ; preds = %for_header117
  call void @tl_mem_enter_scope()
  store i64 %for_idx120, ptr %k, align 8
  %next_logits1122 = load ptr, ptr %next_logits1, align 8
  %k123 = load i64, ptr %k, align 8
  %get_res124 = call float @tl_tensor_get(ptr %next_logits1122, i64 %k123)
  store float %get_res124, ptr %val, align 4
  %val125 = load float, ptr %val, align 4
  %max_val126 = load float, ptr %max_val, align 4
  %fgttmp = fcmp ogt float %val125, %max_val126
  br i1 %fgttmp, label %then127, label %else128

for_end119:                                       ; preds = %for_header117
  %pred1132 = load i64, ptr %pred1, align 8
  call void @tl_print_i64(i64 %pred1132)
  %pred1133 = load i64, ptr %pred1, align 8
  %cast_i64_f32135 = sitofp i64 %pred1133 to float
  store float %cast_i64_f32135, ptr %scalar_data134, align 4
  %scalar_tensor137 = call ptr @tl_tensor_new(ptr %scalar_data134, i64 0, ptr %scalar_shape136)
  store float 1.000000e+00, ptr %scalar_data138, align 4
  %scalar_tensor140 = call ptr @tl_tensor_new(ptr %scalar_data138, i64 0, ptr %scalar_shape139)
  %pow_res141 = call ptr @tl_tensor_pow(ptr %scalar_tensor137, ptr %scalar_tensor140)
  %get_res142 = call float @tl_tensor_get(ptr %pow_res141, i64 0)
  store float %get_res142, ptr %val_pred1, align 4
  %x0144 = load float, ptr %x0, align 4
  %elem_ptr145 = getelementptr inbounds float, ptr %tensor_data_arr143, i64 0
  store float %x0144, ptr %elem_ptr145, align 4
  %x1146 = load float, ptr %x1, align 4
  %elem_ptr147 = getelementptr inbounds float, ptr %tensor_data_arr143, i64 1
  store float %x1146, ptr %elem_ptr147, align 4
  %x2148 = load float, ptr %x2, align 4
  %elem_ptr149 = getelementptr inbounds float, ptr %tensor_data_arr143, i64 2
  store float %x2148, ptr %elem_ptr149, align 4
  %x3150 = load float, ptr %x3, align 4
  %elem_ptr151 = getelementptr inbounds float, ptr %tensor_data_arr143, i64 3
  store float %x3150, ptr %elem_ptr151, align 4
  %x4152 = load float, ptr %x4, align 4
  %elem_ptr153 = getelementptr inbounds float, ptr %tensor_data_arr143, i64 4
  store float %x4152, ptr %elem_ptr153, align 4
  %x5154 = load float, ptr %x5, align 4
  %elem_ptr155 = getelementptr inbounds float, ptr %tensor_data_arr143, i64 5
  store float %x5154, ptr %elem_ptr155, align 4
  %val_pred1156 = load float, ptr %val_pred1, align 4
  %elem_ptr157 = getelementptr inbounds float, ptr %tensor_data_arr143, i64 6
  store float %val_pred1156, ptr %elem_ptr157, align 4
  %val_pad158 = load float, ptr %val_pad, align 4
  %elem_ptr159 = getelementptr inbounds float, ptr %tensor_data_arr143, i64 7
  store float %val_pad158, ptr %elem_ptr159, align 4
  %val_pad160 = load float, ptr %val_pad, align 4
  %elem_ptr161 = getelementptr inbounds float, ptr %tensor_data_arr143, i64 8
  store float %val_pad160, ptr %elem_ptr161, align 4
  %elem_ptr162 = getelementptr inbounds float, ptr %tensor_data_arr143, i64 9
  store float 1.200000e+01, ptr %elem_ptr162, align 4
  %elem_ptr163 = getelementptr inbounds float, ptr %tensor_data_arr143, i64 10
  store float 1.200000e+01, ptr %elem_ptr163, align 4
  %elem_ptr164 = getelementptr inbounds float, ptr %tensor_data_arr143, i64 11
  store float 1.200000e+01, ptr %elem_ptr164, align 4
  %shape_ptr166 = getelementptr inbounds i64, ptr %tensor_shape_arr165, i64 0
  store i64 12, ptr %shape_ptr166, align 8
  %new_tensor167 = call ptr @tl_tensor_new(ptr %tensor_data_arr143, i64 1, ptr %tensor_shape_arr165)
  store ptr %new_tensor167, ptr %data2, align 8
  call void @tl_register_tensor(ptr @tensor_name.123, ptr %new_tensor167)
  %data2168 = load ptr, ptr %data2, align 8
  %dims_alloca169 = alloca [2 x i64], align 8
  %dim_ptr170 = getelementptr [2 x i64], ptr %dims_alloca169, i64 0, i64 0
  store i64 1, ptr %dim_ptr170, align 8
  %dim_ptr171 = getelementptr [2 x i64], ptr %dims_alloca169, i64 0, i64 1
  store i64 12, ptr %dim_ptr171, align 8
  %dims_ptr172 = getelementptr [2 x i64], ptr %dims_alloca169, i64 0, i64 0
  %reshape_dims_res173 = call ptr @tl_tensor_reshape_dims(ptr %data2168, ptr %dims_ptr172, i64 2)
  store ptr %reshape_dims_res173, ptr %input2, align 8
  call void @tl_register_tensor(ptr @tensor_name.124, ptr %reshape_dims_res173)
  %model174 = load ptr, ptr %model6, align 8
  %input2175 = load ptr, ptr %input2, align 8
  %call_method176 = call ptr @tl_GPT_forward(ptr %model174, ptr %input2175)
  store ptr %call_method176, ptr %logits2, align 8
  call void @tl_register_tensor(ptr @tensor_name.125, ptr %call_method176)
  %logits2177 = load ptr, ptr %logits2, align 8
  %dims_alloca178 = alloca [2 x i64], align 8
  %dim_ptr179 = getelementptr [2 x i64], ptr %dims_alloca178, i64 0, i64 0
  store i64 12, ptr %dim_ptr179, align 8
  %dim_ptr180 = getelementptr [2 x i64], ptr %dims_alloca178, i64 0, i64 1
  store i64 13, ptr %dim_ptr180, align 8
  %dims_ptr181 = getelementptr [2 x i64], ptr %dims_alloca178, i64 0, i64 0
  %reshape_dims_res182 = call ptr @tl_tensor_reshape_dims(ptr %logits2177, ptr %dims_ptr181, i64 2)
  store ptr %reshape_dims_res182, ptr %logits2183, align 8
  call void @tl_register_tensor(ptr @tensor_name.126, ptr %reshape_dims_res182)
  %logits2184 = load ptr, ptr %logits2183, align 8
  %slice_res185 = call ptr @tl_tensor_slice(ptr %logits2184, i64 6, i64 1)
  store ptr %slice_res185, ptr %next_logits2, align 8
  call void @tl_register_tensor(ptr @tensor_name.127, ptr %slice_res185)
  store i64 0, ptr %pred2, align 8
  store float -1.000000e+06, ptr %max_val186, align 4
  br label %for_header187

then127:                                          ; preds = %for_body118
  %val130 = load float, ptr %val, align 4
  store float %val130, ptr %max_val, align 4
  %k131 = load i64, ptr %k, align 8
  store i64 %k131, ptr %pred1, align 8
  br label %merge129

else128:                                          ; preds = %for_body118
  br label %merge129

merge129:                                         ; preds = %else128, %then127
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx120, 1
  br label %for_header117

for_header187:                                    ; preds = %merge202, %for_end119
  %for_idx190 = phi i64 [ 0, %for_end119 ], [ %next_idx205, %merge202 ]
  %for_cond191 = icmp slt i64 %for_idx190, 13
  br i1 %for_cond191, label %for_body188, label %for_end189

for_body188:                                      ; preds = %for_header187
  call void @tl_mem_enter_scope()
  store i64 %for_idx190, ptr %k192, align 8
  %next_logits2193 = load ptr, ptr %next_logits2, align 8
  %k194 = load i64, ptr %k192, align 8
  %get_res195 = call float @tl_tensor_get(ptr %next_logits2193, i64 %k194)
  store float %get_res195, ptr %val196, align 4
  %val197 = load float, ptr %val196, align 4
  %max_val198 = load float, ptr %max_val186, align 4
  %fgttmp199 = fcmp ogt float %val197, %max_val198
  br i1 %fgttmp199, label %then200, label %else201

for_end189:                                       ; preds = %for_header187
  %pred2206 = load i64, ptr %pred2, align 8
  call void @tl_print_i64(i64 %pred2206)
  %pred2207 = load i64, ptr %pred2, align 8
  %cast_i64_f32209 = sitofp i64 %pred2207 to float
  store float %cast_i64_f32209, ptr %scalar_data208, align 4
  %scalar_tensor211 = call ptr @tl_tensor_new(ptr %scalar_data208, i64 0, ptr %scalar_shape210)
  store float 1.000000e+00, ptr %scalar_data212, align 4
  %scalar_tensor214 = call ptr @tl_tensor_new(ptr %scalar_data212, i64 0, ptr %scalar_shape213)
  %pow_res215 = call ptr @tl_tensor_pow(ptr %scalar_tensor211, ptr %scalar_tensor214)
  %get_res216 = call float @tl_tensor_get(ptr %pow_res215, i64 0)
  store float %get_res216, ptr %val_pred2, align 4
  %x0218 = load float, ptr %x0, align 4
  %elem_ptr219 = getelementptr inbounds float, ptr %tensor_data_arr217, i64 0
  store float %x0218, ptr %elem_ptr219, align 4
  %x1220 = load float, ptr %x1, align 4
  %elem_ptr221 = getelementptr inbounds float, ptr %tensor_data_arr217, i64 1
  store float %x1220, ptr %elem_ptr221, align 4
  %x2222 = load float, ptr %x2, align 4
  %elem_ptr223 = getelementptr inbounds float, ptr %tensor_data_arr217, i64 2
  store float %x2222, ptr %elem_ptr223, align 4
  %x3224 = load float, ptr %x3, align 4
  %elem_ptr225 = getelementptr inbounds float, ptr %tensor_data_arr217, i64 3
  store float %x3224, ptr %elem_ptr225, align 4
  %x4226 = load float, ptr %x4, align 4
  %elem_ptr227 = getelementptr inbounds float, ptr %tensor_data_arr217, i64 4
  store float %x4226, ptr %elem_ptr227, align 4
  %x5228 = load float, ptr %x5, align 4
  %elem_ptr229 = getelementptr inbounds float, ptr %tensor_data_arr217, i64 5
  store float %x5228, ptr %elem_ptr229, align 4
  %val_pred1230 = load float, ptr %val_pred1, align 4
  %elem_ptr231 = getelementptr inbounds float, ptr %tensor_data_arr217, i64 6
  store float %val_pred1230, ptr %elem_ptr231, align 4
  %val_pred2232 = load float, ptr %val_pred2, align 4
  %elem_ptr233 = getelementptr inbounds float, ptr %tensor_data_arr217, i64 7
  store float %val_pred2232, ptr %elem_ptr233, align 4
  %val_pad234 = load float, ptr %val_pad, align 4
  %elem_ptr235 = getelementptr inbounds float, ptr %tensor_data_arr217, i64 8
  store float %val_pad234, ptr %elem_ptr235, align 4
  %elem_ptr236 = getelementptr inbounds float, ptr %tensor_data_arr217, i64 9
  store float 1.200000e+01, ptr %elem_ptr236, align 4
  %elem_ptr237 = getelementptr inbounds float, ptr %tensor_data_arr217, i64 10
  store float 1.200000e+01, ptr %elem_ptr237, align 4
  %elem_ptr238 = getelementptr inbounds float, ptr %tensor_data_arr217, i64 11
  store float 1.200000e+01, ptr %elem_ptr238, align 4
  %shape_ptr240 = getelementptr inbounds i64, ptr %tensor_shape_arr239, i64 0
  store i64 12, ptr %shape_ptr240, align 8
  %new_tensor241 = call ptr @tl_tensor_new(ptr %tensor_data_arr217, i64 1, ptr %tensor_shape_arr239)
  store ptr %new_tensor241, ptr %data3, align 8
  call void @tl_register_tensor(ptr @tensor_name.128, ptr %new_tensor241)
  %data3242 = load ptr, ptr %data3, align 8
  %dims_alloca243 = alloca [2 x i64], align 8
  %dim_ptr244 = getelementptr [2 x i64], ptr %dims_alloca243, i64 0, i64 0
  store i64 1, ptr %dim_ptr244, align 8
  %dim_ptr245 = getelementptr [2 x i64], ptr %dims_alloca243, i64 0, i64 1
  store i64 12, ptr %dim_ptr245, align 8
  %dims_ptr246 = getelementptr [2 x i64], ptr %dims_alloca243, i64 0, i64 0
  %reshape_dims_res247 = call ptr @tl_tensor_reshape_dims(ptr %data3242, ptr %dims_ptr246, i64 2)
  store ptr %reshape_dims_res247, ptr %input3, align 8
  call void @tl_register_tensor(ptr @tensor_name.129, ptr %reshape_dims_res247)
  %model248 = load ptr, ptr %model6, align 8
  %input3249 = load ptr, ptr %input3, align 8
  %call_method250 = call ptr @tl_GPT_forward(ptr %model248, ptr %input3249)
  store ptr %call_method250, ptr %logits3, align 8
  call void @tl_register_tensor(ptr @tensor_name.130, ptr %call_method250)
  %logits3251 = load ptr, ptr %logits3, align 8
  %dims_alloca252 = alloca [2 x i64], align 8
  %dim_ptr253 = getelementptr [2 x i64], ptr %dims_alloca252, i64 0, i64 0
  store i64 12, ptr %dim_ptr253, align 8
  %dim_ptr254 = getelementptr [2 x i64], ptr %dims_alloca252, i64 0, i64 1
  store i64 13, ptr %dim_ptr254, align 8
  %dims_ptr255 = getelementptr [2 x i64], ptr %dims_alloca252, i64 0, i64 0
  %reshape_dims_res256 = call ptr @tl_tensor_reshape_dims(ptr %logits3251, ptr %dims_ptr255, i64 2)
  store ptr %reshape_dims_res256, ptr %logits3257, align 8
  call void @tl_register_tensor(ptr @tensor_name.131, ptr %reshape_dims_res256)
  %logits3258 = load ptr, ptr %logits3257, align 8
  %slice_res259 = call ptr @tl_tensor_slice(ptr %logits3258, i64 7, i64 1)
  store ptr %slice_res259, ptr %next_logits3, align 8
  call void @tl_register_tensor(ptr @tensor_name.132, ptr %slice_res259)
  store i64 0, ptr %pred3, align 8
  store float -1.000000e+06, ptr %max_val260, align 4
  br label %for_header261

then200:                                          ; preds = %for_body188
  %val203 = load float, ptr %val196, align 4
  store float %val203, ptr %max_val186, align 4
  %k204 = load i64, ptr %k192, align 8
  store i64 %k204, ptr %pred2, align 8
  br label %merge202

else201:                                          ; preds = %for_body188
  br label %merge202

merge202:                                         ; preds = %else201, %then200
  call void @tl_mem_exit_scope()
  %next_idx205 = add i64 %for_idx190, 1
  br label %for_header187

for_header261:                                    ; preds = %merge276, %for_end189
  %for_idx264 = phi i64 [ 0, %for_end189 ], [ %next_idx279, %merge276 ]
  %for_cond265 = icmp slt i64 %for_idx264, 13
  br i1 %for_cond265, label %for_body262, label %for_end263

for_body262:                                      ; preds = %for_header261
  call void @tl_mem_enter_scope()
  store i64 %for_idx264, ptr %k266, align 8
  %next_logits3267 = load ptr, ptr %next_logits3, align 8
  %k268 = load i64, ptr %k266, align 8
  %get_res269 = call float @tl_tensor_get(ptr %next_logits3267, i64 %k268)
  store float %get_res269, ptr %val270, align 4
  %val271 = load float, ptr %val270, align 4
  %max_val272 = load float, ptr %max_val260, align 4
  %fgttmp273 = fcmp ogt float %val271, %max_val272
  br i1 %fgttmp273, label %then274, label %else275

for_end263:                                       ; preds = %for_header261
  %pred3280 = load i64, ptr %pred3, align 8
  call void @tl_print_i64(i64 %pred3280)
  call void @tl_mem_exit_scope()
  %next_idx281 = add i64 %for_idx, 1
  br label %for_header

then274:                                          ; preds = %for_body262
  %val277 = load float, ptr %val270, align 4
  store float %val277, ptr %max_val260, align 4
  %k278 = load i64, ptr %k266, align 8
  store i64 %k278, ptr %pred3, align 8
  br label %merge276

else275:                                          ; preds = %for_body262
  br label %merge276

merge276:                                         ; preds = %else275, %then274
  call void @tl_mem_exit_scope()
  %next_idx279 = add i64 %for_idx264, 1
  br label %for_header261
}
