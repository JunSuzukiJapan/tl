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
@tensor_name.110 = private unnamed_addr constant [11 x i8] c"total_loss\00", align 1
@tensor_name.111 = private unnamed_addr constant [5 x i8] c"data\00", align 1
@tensor_name.112 = private unnamed_addr constant [7 x i8] c"target\00", align 1
@tensor_name.113 = private unnamed_addr constant [2 x i8] c"X\00", align 1
@tensor_name.114 = private unnamed_addr constant [2 x i8] c"Y\00", align 1
@tensor_name.115 = private unnamed_addr constant [7 x i8] c"logits\00", align 1
@tensor_name.116 = private unnamed_addr constant [12 x i8] c"logits_flat\00", align 1
@tensor_name.117 = private unnamed_addr constant [7 x i8] c"Y_flat\00", align 1
@tensor_name.118 = private unnamed_addr constant [5 x i8] c"loss\00", align 1
@str_literal.119 = private unnamed_addr constant [6 x i8] c"Loss:\00", align 1
@str_literal.120 = private unnamed_addr constant [12 x i8] c"Memory(MB):\00", align 1
@str_literal.121 = private unnamed_addr constant [58 x i8] c"Training 2-digit addition (0-99) - With memory monitoring\00", align 1
@str_literal.122 = private unnamed_addr constant [7 x i8] c"Epoch:\00", align 1
@str_literal.123 = private unnamed_addr constant [19 x i8] c"Training Complete!\00", align 1
@str_literal.124 = private unnamed_addr constant [25 x i8] c"model_2digit.safetensors\00", align 1

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

define void @tl_Linear_step(ptr %self, float %lr) {
entry:
  %gb = alloca ptr, align 8
  %gW = alloca ptr, align 8
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  %ptr_W = getelementptr inbounds nuw %Linear, ptr %self3, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %W)
  store ptr %grad_res, ptr %gW, align 8
  call void @tl_register_tensor(ptr @tensor_name, ptr %grad_res)
  %self4 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds nuw %Linear, ptr %self4, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res5 = call ptr @tl_tensor_grad(ptr %b)
  store ptr %grad_res5, ptr %gb, align 8
  call void @tl_register_tensor(ptr @tensor_name.86, ptr %grad_res5)
  %self6 = load ptr, ptr %self1, align 8
  %ptr_W7 = getelementptr inbounds nuw %Linear, ptr %self6, i32 0, i32 0
  %W8 = load ptr, ptr %ptr_W7, align 8
  %gW9 = load ptr, ptr %gW, align 8
  %lr10 = load float, ptr %lr2, align 4
  %scalar_data = alloca float, align 4
  store float %lr10, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %gW9, ptr %scalar_tensor)
  call void @tl_tensor_sub_assign(ptr %W8, ptr %binop_res)
  %self11 = load ptr, ptr %self1, align 8
  %ptr_b12 = getelementptr inbounds nuw %Linear, ptr %self11, i32 0, i32 1
  %b13 = load ptr, ptr %ptr_b12, align 8
  %gb14 = load ptr, ptr %gb, align 8
  %lr15 = load float, ptr %lr2, align 4
  %scalar_data16 = alloca float, align 4
  store float %lr15, ptr %scalar_data16, align 4
  %scalar_shape17 = alloca i64, i64 0, align 8
  %scalar_tensor18 = call ptr @tl_tensor_new(ptr %scalar_data16, i64 0, ptr %scalar_shape17)
  %binop_res19 = call ptr @tl_tensor_mul(ptr %gb14, ptr %scalar_tensor18)
  call void @tl_tensor_sub_assign(ptr %b13, ptr %binop_res19)
  ret void
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

define void @tl_Embedding_step(ptr %self, float %lr) {
entry:
  %g = alloca ptr, align 8
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds nuw %Embedding, ptr %self3, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %w)
  store ptr %grad_res, ptr %g, align 8
  call void @tl_register_tensor(ptr @tensor_name.87, ptr %grad_res)
  %self4 = load ptr, ptr %self1, align 8
  %ptr_w5 = getelementptr inbounds nuw %Embedding, ptr %self4, i32 0, i32 0
  %w6 = load ptr, ptr %ptr_w5, align 8
  %g7 = load ptr, ptr %g, align 8
  %lr8 = load float, ptr %lr2, align 4
  %scalar_data = alloca float, align 4
  store float %lr8, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %g7, ptr %scalar_tensor)
  call void @tl_tensor_sub_assign(ptr %w6, ptr %binop_res)
  ret void
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

define void @tl_LayerNorm_step(ptr %self, float %lr) {
entry:
  %gb = alloca ptr, align 8
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds nuw %LayerNorm, ptr %self3, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %b)
  store ptr %grad_res, ptr %gb, align 8
  call void @tl_register_tensor(ptr @tensor_name.88, ptr %grad_res)
  %self4 = load ptr, ptr %self1, align 8
  %ptr_b5 = getelementptr inbounds nuw %LayerNorm, ptr %self4, i32 0, i32 1
  %b6 = load ptr, ptr %ptr_b5, align 8
  %gb7 = load ptr, ptr %gb, align 8
  %lr8 = load float, ptr %lr2, align 4
  %scalar_data = alloca float, align 4
  store float %lr8, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %gb7, ptr %scalar_tensor)
  call void @tl_tensor_sub_assign(ptr %b6, ptr %binop_res)
  ret void
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

define void @tl_CausalSelfAttention_step(ptr %self, float %lr) {
entry:
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  %ptr_a = getelementptr inbounds nuw %CausalSelfAttention, ptr %self3, i32 0, i32 0
  %a = load ptr, ptr %ptr_a, align 8
  %lr4 = load float, ptr %lr2, align 4
  call void @tl_Linear_step(ptr %a, float %lr4)
  %self5 = load ptr, ptr %self1, align 8
  %ptr_p = getelementptr inbounds nuw %CausalSelfAttention, ptr %self5, i32 0, i32 1
  %p = load ptr, ptr %ptr_p, align 8
  %lr6 = load float, ptr %lr2, align 4
  call void @tl_Linear_step(ptr %p, float %lr6)
  ret void
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

define void @tl_MLP_step(ptr %self, float %lr) {
entry:
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  %ptr_f = getelementptr inbounds nuw %MLP, ptr %self3, i32 0, i32 0
  %f = load ptr, ptr %ptr_f, align 8
  %lr4 = load float, ptr %lr2, align 4
  call void @tl_Linear_step(ptr %f, float %lr4)
  %self5 = load ptr, ptr %self1, align 8
  %ptr_p = getelementptr inbounds nuw %MLP, ptr %self5, i32 0, i32 1
  %p = load ptr, ptr %ptr_p, align 8
  %lr6 = load float, ptr %lr2, align 4
  call void @tl_Linear_step(ptr %p, float %lr6)
  ret void
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

define void @tl_Block_step(ptr %self, float %lr) {
entry:
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  %ptr_l1 = getelementptr inbounds nuw %Block, ptr %self3, i32 0, i32 0
  %l1 = load ptr, ptr %ptr_l1, align 8
  %lr4 = load float, ptr %lr2, align 4
  call void @tl_LayerNorm_step(ptr %l1, float %lr4)
  %self5 = load ptr, ptr %self1, align 8
  %ptr_a = getelementptr inbounds nuw %Block, ptr %self5, i32 0, i32 1
  %a = load ptr, ptr %ptr_a, align 8
  %lr6 = load float, ptr %lr2, align 4
  call void @tl_CausalSelfAttention_step(ptr %a, float %lr6)
  %self7 = load ptr, ptr %self1, align 8
  %ptr_l2 = getelementptr inbounds nuw %Block, ptr %self7, i32 0, i32 2
  %l2 = load ptr, ptr %ptr_l2, align 8
  %lr8 = load float, ptr %lr2, align 4
  call void @tl_LayerNorm_step(ptr %l2, float %lr8)
  %self9 = load ptr, ptr %self1, align 8
  %ptr_m = getelementptr inbounds nuw %Block, ptr %self9, i32 0, i32 3
  %m = load ptr, ptr %ptr_m, align 8
  %lr10 = load float, ptr %lr2, align 4
  call void @tl_MLP_step(ptr %m, float %lr10)
  ret void
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

define void @tl_GPT_step(ptr %self, float %lr) {
entry:
  %lr2 = alloca float, align 4
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds nuw %GPT, ptr %self3, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %lr4 = load float, ptr %lr2, align 4
  call void @tl_Embedding_step(ptr %w, float %lr4)
  %self5 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds nuw %GPT, ptr %self5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %lr6 = load float, ptr %lr2, align 4
  call void @tl_Block_step(ptr %b, float %lr6)
  %self7 = load ptr, ptr %self1, align 8
  %ptr_l = getelementptr inbounds nuw %GPT, ptr %self7, i32 0, i32 2
  %l = load ptr, ptr %ptr_l, align 8
  %lr8 = load float, ptr %lr2, align 4
  call void @tl_LayerNorm_step(ptr %l, float %lr8)
  %self9 = load ptr, ptr %self1, align 8
  %ptr_h = getelementptr inbounds nuw %GPT, ptr %self9, i32 0, i32 3
  %h = load ptr, ptr %ptr_h, align 8
  %lr10 = load float, ptr %lr2, align 4
  call void @tl_Linear_step(ptr %h, float %lr10)
  ret void
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

define void @train_epoch(ptr %model, float %lr, i64 %epoch) {
entry:
  %mem_mb = alloca i64, align 8
  %loss = alloca ptr, align 8
  %Y_flat = alloca ptr, align 8
  %logits_flat = alloca ptr, align 8
  %logits = alloca ptr, align 8
  %Y = alloca ptr, align 8
  %X = alloca ptr, align 8
  %target = alloca ptr, align 8
  %tensor_shape_arr136 = alloca i64, i64 1, align 8
  %tensor_data_arr117 = alloca float, i64 12, align 4
  %data = alloca ptr, align 8
  %tensor_shape_arr = alloca i64, i64 1, align 8
  %tensor_data_arr = alloca float, i64 12, align 4
  %s_d3 = alloca float, align 4
  %scalar_shape95 = alloca i64, align 8
  %scalar_data94 = alloca float, align 4
  %scalar_shape92 = alloca i64, align 8
  %scalar_data90 = alloca float, align 4
  %s_d2 = alloca float, align 4
  %scalar_shape81 = alloca i64, align 8
  %scalar_data80 = alloca float, align 4
  %scalar_shape78 = alloca i64, align 8
  %scalar_data76 = alloca float, align 4
  %s_d1 = alloca float, align 4
  %scalar_shape66 = alloca i64, align 8
  %scalar_data65 = alloca float, align 4
  %scalar_shape63 = alloca i64, align 8
  %scalar_data61 = alloca float, align 4
  %j_d2 = alloca float, align 4
  %scalar_shape55 = alloca i64, align 8
  %scalar_data54 = alloca float, align 4
  %scalar_shape52 = alloca i64, align 8
  %scalar_data50 = alloca float, align 4
  %j_d1 = alloca float, align 4
  %scalar_shape41 = alloca i64, align 8
  %scalar_data40 = alloca float, align 4
  %scalar_shape38 = alloca i64, align 8
  %scalar_data36 = alloca float, align 4
  %i_d2 = alloca float, align 4
  %scalar_shape30 = alloca i64, align 8
  %scalar_data29 = alloca float, align 4
  %scalar_shape27 = alloca i64, align 8
  %scalar_data25 = alloca float, align 4
  %i_d1 = alloca float, align 4
  %scalar_shape19 = alloca i64, align 8
  %scalar_data18 = alloca float, align 4
  %scalar_shape16 = alloca i64, align 8
  %scalar_data15 = alloca float, align 4
  %sum = alloca i64, align 8
  %j = alloca i64, align 8
  %i = alloca i64, align 8
  %total_loss = alloca ptr, align 8
  %scalar_shape5 = alloca i64, align 8
  %scalar_data4 = alloca float, align 4
  %scalar_shape = alloca i64, align 8
  %scalar_data = alloca float, align 4
  %epoch3 = alloca i64, align 8
  %lr2 = alloca float, align 4
  %model1 = alloca ptr, align 8
  store ptr %model, ptr %model1, align 8
  store float %lr, ptr %lr2, align 4
  store i64 %epoch, ptr %epoch3, align 8
  store float 0.000000e+00, ptr %scalar_data, align 4
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  store float 1.000000e+00, ptr %scalar_data4, align 4
  %scalar_tensor6 = call ptr @tl_tensor_new(ptr %scalar_data4, i64 0, ptr %scalar_shape5)
  %pow_res = call ptr @tl_tensor_pow(ptr %scalar_tensor, ptr %scalar_tensor6)
  store ptr %pow_res, ptr %total_loss, align 8
  call void @tl_register_tensor(ptr @tensor_name.110, ptr %pow_res)
  br label %for_header

for_header:                                       ; preds = %for_end9, %entry
  %for_idx = phi i64 [ 0, %entry ], [ %next_idx166, %for_end9 ]
  %for_cond = icmp slt i64 %for_idx, 100
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %i, align 8
  br label %for_header7

for_end:                                          ; preds = %for_header
  %call_tmp = call i64 @get_memory()
  store i64 %call_tmp, ptr %mem_mb, align 8
  call void @tl_print_string(ptr @str_literal.119)
  %total_loss167 = load ptr, ptr %total_loss, align 8
  call void @tl_tensor_print(ptr %total_loss167)
  call void @tl_print_string(ptr @str_literal.120)
  %mem_mb168 = load i64, ptr %mem_mb, align 8
  call void @tl_print_i64(i64 %mem_mb168)
  ret void

for_header7:                                      ; preds = %continue_block, %for_body
  %for_idx10 = phi i64 [ 0, %for_body ], [ %next_idx, %continue_block ]
  %for_cond11 = icmp slt i64 %for_idx10, 100
  br i1 %for_cond11, label %for_body8, label %for_end9

for_body8:                                        ; preds = %for_header7
  call void @tl_mem_enter_scope()
  store i64 %for_idx10, ptr %j, align 8
  %i12 = load i64, ptr %i, align 8
  %j13 = load i64, ptr %j, align 8
  %addtmp = add i64 %i12, %j13
  store i64 %addtmp, ptr %sum, align 8
  %i14 = load i64, ptr %i, align 8
  %divtmp = sdiv i64 %i14, 10
  %cast_i64_f32 = sitofp i64 %divtmp to float
  store float %cast_i64_f32, ptr %scalar_data15, align 4
  %scalar_tensor17 = call ptr @tl_tensor_new(ptr %scalar_data15, i64 0, ptr %scalar_shape16)
  store float 1.000000e+00, ptr %scalar_data18, align 4
  %scalar_tensor20 = call ptr @tl_tensor_new(ptr %scalar_data18, i64 0, ptr %scalar_shape19)
  %pow_res21 = call ptr @tl_tensor_pow(ptr %scalar_tensor17, ptr %scalar_tensor20)
  %get_res = call float @tl_tensor_get(ptr %pow_res21, i64 0)
  store float %get_res, ptr %i_d1, align 4
  %i22 = load i64, ptr %i, align 8
  %i23 = load i64, ptr %i, align 8
  %divtmp24 = sdiv i64 %i23, 10
  %multmp = mul i64 %divtmp24, 10
  %subtmp = sub i64 %i22, %multmp
  %cast_i64_f3226 = sitofp i64 %subtmp to float
  store float %cast_i64_f3226, ptr %scalar_data25, align 4
  %scalar_tensor28 = call ptr @tl_tensor_new(ptr %scalar_data25, i64 0, ptr %scalar_shape27)
  store float 1.000000e+00, ptr %scalar_data29, align 4
  %scalar_tensor31 = call ptr @tl_tensor_new(ptr %scalar_data29, i64 0, ptr %scalar_shape30)
  %pow_res32 = call ptr @tl_tensor_pow(ptr %scalar_tensor28, ptr %scalar_tensor31)
  %get_res33 = call float @tl_tensor_get(ptr %pow_res32, i64 0)
  store float %get_res33, ptr %i_d2, align 4
  %j34 = load i64, ptr %j, align 8
  %divtmp35 = sdiv i64 %j34, 10
  %cast_i64_f3237 = sitofp i64 %divtmp35 to float
  store float %cast_i64_f3237, ptr %scalar_data36, align 4
  %scalar_tensor39 = call ptr @tl_tensor_new(ptr %scalar_data36, i64 0, ptr %scalar_shape38)
  store float 1.000000e+00, ptr %scalar_data40, align 4
  %scalar_tensor42 = call ptr @tl_tensor_new(ptr %scalar_data40, i64 0, ptr %scalar_shape41)
  %pow_res43 = call ptr @tl_tensor_pow(ptr %scalar_tensor39, ptr %scalar_tensor42)
  %get_res44 = call float @tl_tensor_get(ptr %pow_res43, i64 0)
  store float %get_res44, ptr %j_d1, align 4
  %j45 = load i64, ptr %j, align 8
  %j46 = load i64, ptr %j, align 8
  %divtmp47 = sdiv i64 %j46, 10
  %multmp48 = mul i64 %divtmp47, 10
  %subtmp49 = sub i64 %j45, %multmp48
  %cast_i64_f3251 = sitofp i64 %subtmp49 to float
  store float %cast_i64_f3251, ptr %scalar_data50, align 4
  %scalar_tensor53 = call ptr @tl_tensor_new(ptr %scalar_data50, i64 0, ptr %scalar_shape52)
  store float 1.000000e+00, ptr %scalar_data54, align 4
  %scalar_tensor56 = call ptr @tl_tensor_new(ptr %scalar_data54, i64 0, ptr %scalar_shape55)
  %pow_res57 = call ptr @tl_tensor_pow(ptr %scalar_tensor53, ptr %scalar_tensor56)
  %get_res58 = call float @tl_tensor_get(ptr %pow_res57, i64 0)
  store float %get_res58, ptr %j_d2, align 4
  %sum59 = load i64, ptr %sum, align 8
  %divtmp60 = sdiv i64 %sum59, 100
  %cast_i64_f3262 = sitofp i64 %divtmp60 to float
  store float %cast_i64_f3262, ptr %scalar_data61, align 4
  %scalar_tensor64 = call ptr @tl_tensor_new(ptr %scalar_data61, i64 0, ptr %scalar_shape63)
  store float 1.000000e+00, ptr %scalar_data65, align 4
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
  %cast_i64_f3277 = sitofp i64 %divtmp75 to float
  store float %cast_i64_f3277, ptr %scalar_data76, align 4
  %scalar_tensor79 = call ptr @tl_tensor_new(ptr %scalar_data76, i64 0, ptr %scalar_shape78)
  store float 1.000000e+00, ptr %scalar_data80, align 4
  %scalar_tensor82 = call ptr @tl_tensor_new(ptr %scalar_data80, i64 0, ptr %scalar_shape81)
  %pow_res83 = call ptr @tl_tensor_pow(ptr %scalar_tensor79, ptr %scalar_tensor82)
  %get_res84 = call float @tl_tensor_get(ptr %pow_res83, i64 0)
  store float %get_res84, ptr %s_d2, align 4
  %sum85 = load i64, ptr %sum, align 8
  %sum86 = load i64, ptr %sum, align 8
  %divtmp87 = sdiv i64 %sum86, 10
  %multmp88 = mul i64 %divtmp87, 10
  %subtmp89 = sub i64 %sum85, %multmp88
  %cast_i64_f3291 = sitofp i64 %subtmp89 to float
  store float %cast_i64_f3291, ptr %scalar_data90, align 4
  %scalar_tensor93 = call ptr @tl_tensor_new(ptr %scalar_data90, i64 0, ptr %scalar_shape92)
  store float 1.000000e+00, ptr %scalar_data94, align 4
  %scalar_tensor96 = call ptr @tl_tensor_new(ptr %scalar_data94, i64 0, ptr %scalar_shape95)
  %pow_res97 = call ptr @tl_tensor_pow(ptr %scalar_tensor93, ptr %scalar_tensor96)
  %get_res98 = call float @tl_tensor_get(ptr %pow_res97, i64 0)
  store float %get_res98, ptr %s_d3, align 4
  %i_d199 = load float, ptr %i_d1, align 4
  %elem_ptr = getelementptr inbounds float, ptr %tensor_data_arr, i64 0
  store float %i_d199, ptr %elem_ptr, align 4
  %i_d2100 = load float, ptr %i_d2, align 4
  %elem_ptr101 = getelementptr inbounds float, ptr %tensor_data_arr, i64 1
  store float %i_d2100, ptr %elem_ptr101, align 4
  %elem_ptr102 = getelementptr inbounds float, ptr %tensor_data_arr, i64 2
  store float 1.000000e+01, ptr %elem_ptr102, align 4
  %j_d1103 = load float, ptr %j_d1, align 4
  %elem_ptr104 = getelementptr inbounds float, ptr %tensor_data_arr, i64 3
  store float %j_d1103, ptr %elem_ptr104, align 4
  %j_d2105 = load float, ptr %j_d2, align 4
  %elem_ptr106 = getelementptr inbounds float, ptr %tensor_data_arr, i64 4
  store float %j_d2105, ptr %elem_ptr106, align 4
  %elem_ptr107 = getelementptr inbounds float, ptr %tensor_data_arr, i64 5
  store float 1.100000e+01, ptr %elem_ptr107, align 4
  %s_d1108 = load float, ptr %s_d1, align 4
  %elem_ptr109 = getelementptr inbounds float, ptr %tensor_data_arr, i64 6
  store float %s_d1108, ptr %elem_ptr109, align 4
  %s_d2110 = load float, ptr %s_d2, align 4
  %elem_ptr111 = getelementptr inbounds float, ptr %tensor_data_arr, i64 7
  store float %s_d2110, ptr %elem_ptr111, align 4
  %s_d3112 = load float, ptr %s_d3, align 4
  %elem_ptr113 = getelementptr inbounds float, ptr %tensor_data_arr, i64 8
  store float %s_d3112, ptr %elem_ptr113, align 4
  %elem_ptr114 = getelementptr inbounds float, ptr %tensor_data_arr, i64 9
  store float 1.200000e+01, ptr %elem_ptr114, align 4
  %elem_ptr115 = getelementptr inbounds float, ptr %tensor_data_arr, i64 10
  store float 1.200000e+01, ptr %elem_ptr115, align 4
  %elem_ptr116 = getelementptr inbounds float, ptr %tensor_data_arr, i64 11
  store float 1.200000e+01, ptr %elem_ptr116, align 4
  %shape_ptr = getelementptr inbounds i64, ptr %tensor_shape_arr, i64 0
  store i64 12, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %tensor_data_arr, i64 1, ptr %tensor_shape_arr)
  store ptr %new_tensor, ptr %data, align 8
  call void @tl_register_tensor(ptr @tensor_name.111, ptr %new_tensor)
  %i_d2118 = load float, ptr %i_d2, align 4
  %elem_ptr119 = getelementptr inbounds float, ptr %tensor_data_arr117, i64 0
  store float %i_d2118, ptr %elem_ptr119, align 4
  %elem_ptr120 = getelementptr inbounds float, ptr %tensor_data_arr117, i64 1
  store float 1.000000e+01, ptr %elem_ptr120, align 4
  %j_d1121 = load float, ptr %j_d1, align 4
  %elem_ptr122 = getelementptr inbounds float, ptr %tensor_data_arr117, i64 2
  store float %j_d1121, ptr %elem_ptr122, align 4
  %j_d2123 = load float, ptr %j_d2, align 4
  %elem_ptr124 = getelementptr inbounds float, ptr %tensor_data_arr117, i64 3
  store float %j_d2123, ptr %elem_ptr124, align 4
  %elem_ptr125 = getelementptr inbounds float, ptr %tensor_data_arr117, i64 4
  store float 1.100000e+01, ptr %elem_ptr125, align 4
  %s_d1126 = load float, ptr %s_d1, align 4
  %elem_ptr127 = getelementptr inbounds float, ptr %tensor_data_arr117, i64 5
  store float %s_d1126, ptr %elem_ptr127, align 4
  %s_d2128 = load float, ptr %s_d2, align 4
  %elem_ptr129 = getelementptr inbounds float, ptr %tensor_data_arr117, i64 6
  store float %s_d2128, ptr %elem_ptr129, align 4
  %s_d3130 = load float, ptr %s_d3, align 4
  %elem_ptr131 = getelementptr inbounds float, ptr %tensor_data_arr117, i64 7
  store float %s_d3130, ptr %elem_ptr131, align 4
  %elem_ptr132 = getelementptr inbounds float, ptr %tensor_data_arr117, i64 8
  store float 1.200000e+01, ptr %elem_ptr132, align 4
  %elem_ptr133 = getelementptr inbounds float, ptr %tensor_data_arr117, i64 9
  store float 1.200000e+01, ptr %elem_ptr133, align 4
  %elem_ptr134 = getelementptr inbounds float, ptr %tensor_data_arr117, i64 10
  store float 1.200000e+01, ptr %elem_ptr134, align 4
  %elem_ptr135 = getelementptr inbounds float, ptr %tensor_data_arr117, i64 11
  store float 1.200000e+01, ptr %elem_ptr135, align 4
  %shape_ptr137 = getelementptr inbounds i64, ptr %tensor_shape_arr136, i64 0
  store i64 12, ptr %shape_ptr137, align 8
  %new_tensor138 = call ptr @tl_tensor_new(ptr %tensor_data_arr117, i64 1, ptr %tensor_shape_arr136)
  store ptr %new_tensor138, ptr %target, align 8
  call void @tl_register_tensor(ptr @tensor_name.112, ptr %new_tensor138)
  %data139 = load ptr, ptr %data, align 8
  %dims_alloca = alloca [2 x i64], align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr, align 8
  %dim_ptr140 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr140, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %data139, ptr %dims_ptr, i64 2)
  store ptr %reshape_dims_res, ptr %X, align 8
  call void @tl_register_tensor(ptr @tensor_name.113, ptr %reshape_dims_res)
  %target141 = load ptr, ptr %target, align 8
  %dims_alloca142 = alloca [2 x i64], align 8
  %dim_ptr143 = getelementptr [2 x i64], ptr %dims_alloca142, i64 0, i64 0
  store i64 1, ptr %dim_ptr143, align 8
  %dim_ptr144 = getelementptr [2 x i64], ptr %dims_alloca142, i64 0, i64 1
  store i64 12, ptr %dim_ptr144, align 8
  %dims_ptr145 = getelementptr [2 x i64], ptr %dims_alloca142, i64 0, i64 0
  %reshape_dims_res146 = call ptr @tl_tensor_reshape_dims(ptr %target141, ptr %dims_ptr145, i64 2)
  store ptr %reshape_dims_res146, ptr %Y, align 8
  call void @tl_register_tensor(ptr @tensor_name.114, ptr %reshape_dims_res146)
  %model147 = load ptr, ptr %model1, align 8
  %X148 = load ptr, ptr %X, align 8
  %call_method = call ptr @tl_GPT_forward(ptr %model147, ptr %X148)
  store ptr %call_method, ptr %logits, align 8
  call void @tl_register_tensor(ptr @tensor_name.115, ptr %call_method)
  %logits149 = load ptr, ptr %logits, align 8
  %dims_alloca150 = alloca [2 x i64], align 8
  %dim_ptr151 = getelementptr [2 x i64], ptr %dims_alloca150, i64 0, i64 0
  store i64 12, ptr %dim_ptr151, align 8
  %dim_ptr152 = getelementptr [2 x i64], ptr %dims_alloca150, i64 0, i64 1
  store i64 13, ptr %dim_ptr152, align 8
  %dims_ptr153 = getelementptr [2 x i64], ptr %dims_alloca150, i64 0, i64 0
  %reshape_dims_res154 = call ptr @tl_tensor_reshape_dims(ptr %logits149, ptr %dims_ptr153, i64 2)
  store ptr %reshape_dims_res154, ptr %logits_flat, align 8
  call void @tl_register_tensor(ptr @tensor_name.116, ptr %reshape_dims_res154)
  %Y155 = load ptr, ptr %Y, align 8
  %dims_alloca156 = alloca [1 x i64], align 8
  %dim_ptr157 = getelementptr [1 x i64], ptr %dims_alloca156, i64 0, i64 0
  store i64 12, ptr %dim_ptr157, align 8
  %dims_ptr158 = getelementptr [1 x i64], ptr %dims_alloca156, i64 0, i64 0
  %reshape_dims_res159 = call ptr @tl_tensor_reshape_dims(ptr %Y155, ptr %dims_ptr158, i64 1)
  store ptr %reshape_dims_res159, ptr %Y_flat, align 8
  call void @tl_register_tensor(ptr @tensor_name.117, ptr %reshape_dims_res159)
  %logits_flat160 = load ptr, ptr %logits_flat, align 8
  %Y_flat161 = load ptr, ptr %Y_flat, align 8
  %ce_res = call ptr @tl_tensor_cross_entropy(ptr %logits_flat160, ptr %Y_flat161)
  store ptr %ce_res, ptr %loss, align 8
  call void @tl_register_tensor(ptr @tensor_name.118, ptr %ce_res)
  %loss162 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss162)
  %model163 = load ptr, ptr %model1, align 8
  %lr164 = load float, ptr %lr2, align 4
  call void @tl_GPT_step(ptr %model163, float %lr164)
  %loss165 = load ptr, ptr %loss, align 8
  %detach_res = call ptr @tl_tensor_detach(ptr %loss165, i1 true)
  %old_val = load ptr, ptr %total_loss, align 8
  %is_not_null = icmp ne ptr %old_val, null
  br i1 %is_not_null, label %free_block, label %continue_block

for_end9:                                         ; preds = %for_header7
  call void @tl_mem_exit_scope()
  %next_idx166 = add i64 %for_idx, 1
  br label %for_header

free_block:                                       ; preds = %for_body8
  br label %continue_block

continue_block:                                   ; preds = %free_block, %for_body8
  call void @tl_mem_unregister(ptr %detach_res)
  store ptr %detach_res, ptr %total_loss, align 8
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx10, 1
  br label %for_header7
}

define void @main() {
entry:
  %epoch = alloca i64, align 8
  %epochs = alloca i64, align 8
  %lr3 = alloca float, align 4
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
  store float 0x3F50624DE0000000, ptr %lr3, align 4
  store i64 50, ptr %epochs, align 8
  call void @tl_print_string(ptr @str_literal.121)
  %epochs4 = load i64, ptr %epochs, align 8
  br label %for_header

for_header:                                       ; preds = %for_body, %entry
  %for_idx = phi i64 [ 0, %entry ], [ %next_idx, %for_body ]
  %for_cond = icmp slt i64 %for_idx, %epochs4
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %epoch, align 8
  call void @tl_print_string(ptr @str_literal.122)
  %epoch5 = load i64, ptr %epoch, align 8
  call void @tl_print_i64(i64 %epoch5)
  %model6 = load ptr, ptr %model, align 8
  %lr7 = load float, ptr %lr3, align 4
  %epoch8 = load i64, ptr %epoch, align 8
  call void @train_epoch(ptr %model6, float %lr7, i64 %epoch8)
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx, 1
  br label %for_header

for_end:                                          ; preds = %for_header
  %model9 = load ptr, ptr %model, align 8
  call void @register_gpt_params(ptr %model9)
  call void @tl_print_string(ptr @str_literal.123)
  call void @tl_save_all_params(ptr @str_literal.124)
  ret void
}
