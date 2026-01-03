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

@str_literal = private unnamed_addr constant [6 x i8] c"Loss:\00", align 1
@str_literal.103 = private unnamed_addr constant [12 x i8] c"Memory(MB):\00", align 1
@str_literal.104 = private unnamed_addr constant [58 x i8] c"Training 2-digit addition (0-99) - With memory monitoring\00", align 1
@str_literal.105 = private unnamed_addr constant [7 x i8] c"Epoch:\00", align 1
@str_literal.106 = private unnamed_addr constant [19 x i8] c"Training Complete!\00", align 1
@key_str = private unnamed_addr constant [4 x i8] c"w.w\00", align 1
@key_str.107 = private unnamed_addr constant [7 x i8] c"b.l1.w\00", align 1
@key_str.108 = private unnamed_addr constant [7 x i8] c"b.l1.b\00", align 1
@key_str.109 = private unnamed_addr constant [8 x i8] c"b.a.a.W\00", align 1
@key_str.110 = private unnamed_addr constant [8 x i8] c"b.a.a.b\00", align 1
@key_str.111 = private unnamed_addr constant [8 x i8] c"b.a.p.W\00", align 1
@key_str.112 = private unnamed_addr constant [8 x i8] c"b.a.p.b\00", align 1
@key_str.113 = private unnamed_addr constant [7 x i8] c"b.l2.w\00", align 1
@key_str.114 = private unnamed_addr constant [7 x i8] c"b.l2.b\00", align 1
@key_str.115 = private unnamed_addr constant [8 x i8] c"b.m.f.W\00", align 1
@key_str.116 = private unnamed_addr constant [8 x i8] c"b.m.f.b\00", align 1
@key_str.117 = private unnamed_addr constant [8 x i8] c"b.m.p.W\00", align 1
@key_str.118 = private unnamed_addr constant [8 x i8] c"b.m.p.b\00", align 1
@key_str.119 = private unnamed_addr constant [4 x i8] c"l.w\00", align 1
@key_str.120 = private unnamed_addr constant [4 x i8] c"l.b\00", align 1
@key_str.121 = private unnamed_addr constant [4 x i8] c"h.W\00", align 1
@key_str.122 = private unnamed_addr constant [4 x i8] c"h.b\00", align 1
@str_literal.123 = private unnamed_addr constant [25 x i8] c"model_2digit.safetensors\00", align 1

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

declare void @tl_tensor_save(ptr, ptr)

declare ptr @tl_tensor_load(ptr)

declare ptr @tl_tensor_map_new()

declare void @tl_tensor_map_insert(ptr, ptr, ptr)

declare void @tl_tensor_map_save(ptr, ptr)

declare ptr @tl_tensor_map_load(ptr)

declare ptr @tl_tensor_map_get(ptr, ptr)

declare void @tl_tensor_map_free(ptr)

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

declare void @tl_tensor_save.1(ptr, ptr)

declare ptr @tl_tensor_load.2(ptr)

declare void @tl_save_all_params(ptr)

declare void @tl_add_parameter(ptr, ptr)

declare void @tl_load_all_params(ptr)

declare void @tl_tensor_sub_assign.3(ptr, ptr)

declare void @tl_add_parameter.4(ptr, ptr)

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

declare ptr @tl_pool_acquire(i64)

declare void @tl_pool_release(ptr, i64)

declare void @tl_arena_init(i64)

declare ptr @tl_arena_alloc(i64)

declare void @tl_arena_free()

declare i1 @tl_arena_is_active()

declare void @tl_arena_reset()

declare void @tl_print_i64.5(i64)

declare void @tl_print_f32.6(float)

declare void @tl_print_string.7(ptr)

declare ptr @malloc.8(i64)

declare ptr @calloc.9(i64, i64)

declare void @free.10(ptr)

declare i64 @tl_tensor_dim.11(ptr, i64)

declare float @tl_tensor_get_f32_md.12(ptr, ptr, i64)

declare ptr @tl_tensor_new.13(ptr, i64, ptr)

declare ptr @tl_tensor_sub.14(ptr, ptr)

declare void @tl_tensor_free.15(ptr)

declare ptr @tl_tensor_clone.16(ptr)

declare ptr @tl_tensor_add.17(ptr, ptr)

declare ptr @tl_tensor_mul.18(ptr, ptr)

declare void @tl_tensor_print.19(ptr)

declare float @tl_tensor_get.20(ptr, i64)

declare ptr @tl_tensor_slice.21(ptr, i64, i64)

declare i64 @tl_tensor_len.22(ptr)

declare ptr @tl_tensor_neg.23(ptr)

declare ptr @tl_tensor_transpose.24(ptr, i64, i64)

declare ptr @tl_tensor_pow.25(ptr, ptr)

declare ptr @tl_tensor_sqrt.26(ptr)

declare ptr @tl_tensor_sin.27(ptr)

declare ptr @tl_tensor_cos.28(ptr)

declare ptr @tl_tensor_relu.29(ptr)

declare ptr @tl_tensor_gelu.30(ptr)

declare ptr @tl_tensor_tril.31(ptr, i32)

declare ptr @tl_tensor_sum_dim.32(ptr, i64, i1)

declare ptr @tl_tensor_embedding.33(ptr, ptr)

declare ptr @tl_tensor_sum.34(ptr)

declare ptr @tl_tensor_div.35(ptr, ptr)

declare ptr @tl_tensor_matmul.36(ptr, ptr)

declare ptr @tl_tensor_exp.37(ptr)

declare ptr @tl_tensor_log.38(ptr)

declare void @tl_tensor_add_assign.39(ptr, ptr)

declare void @tl_tensor_sub_assign.40(ptr, ptr)

declare void @tl_tensor_mul_assign.41(ptr, ptr)

declare void @tl_tensor_div_assign.42(ptr, ptr)

declare void @tl_register_tensor.43(ptr, ptr)

declare i32 @strcmp.44(ptr, ptr)

declare void @tl_tensor_save.45(ptr, ptr)

declare ptr @tl_tensor_load.46(ptr)

declare ptr @tl_tensor_map_new.47()

declare void @tl_tensor_map_insert.48(ptr, ptr, ptr)

declare void @tl_tensor_map_save.49(ptr, ptr)

declare ptr @tl_tensor_map_load.50(ptr)

declare ptr @tl_tensor_map_get.51(ptr, ptr)

declare void @tl_tensor_map_free.52(ptr)

declare ptr @tl_tensor_reshape_dims.53(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.54(ptr, ptr)

declare ptr @tl_tensor_randn.55(i64, ptr, i1)

declare ptr @tl_varbuilder_get.56(ptr, i64, ptr)

declare void @tl_update_all_params.57(float)

declare ptr @tl_varbuilder_grad.58(ptr)

declare void @tl_tensor_backward.59(ptr)

declare ptr @tl_tensor_grad.60(ptr)

declare ptr @tl_tensor_detach.61(ptr, i1)

declare ptr @tl_tensor_softmax.62(ptr, i64)

declare ptr @tl_tensor_cross_entropy.63(ptr, ptr)

declare void @tl_tensor_save.64(ptr, ptr)

declare ptr @tl_tensor_load.65(ptr)

declare void @tl_save_all_params.66(ptr)

declare void @tl_add_parameter.67(ptr, ptr)

declare void @tl_load_all_params.68(ptr)

declare void @tl_tensor_sub_assign.69(ptr, ptr)

declare void @tl_add_parameter.70(ptr, ptr)

declare ptr @tl_register_parameter.71(ptr)

declare ptr @tl_string_concat.72(ptr, ptr)

declare ptr @tl_file_open.73(ptr, ptr)

declare ptr @tl_file_read_string.74(ptr)

declare void @tl_file_write_string.75(ptr, ptr)

declare void @tl_file_close.76(ptr)

declare ptr @tl_path_new.77(ptr)

declare ptr @tl_path_join.78(ptr, ptr)

declare i1 @tl_path_exists.79(ptr)

declare i1 @tl_path_is_dir.80(ptr)

declare i1 @tl_path_is_file.81(ptr)

declare ptr @tl_path_to_string.82(ptr)

declare void @tl_path_free.83(ptr)

declare i1 @tl_http_download.84(ptr, ptr)

declare ptr @tl_http_get.85(ptr)

declare ptr @tl_env_get.86(ptr)

declare void @tl_env_set.87(ptr, ptr)

declare float @tl_system_time.88()

declare void @tl_system_sleep.89(float)

declare i64 @tl_get_memory_mb.90()

declare void @tl_mem_enter_scope.91()

declare void @tl_mem_exit_scope.92()

declare void @tl_mem_register_struct.93(ptr)

declare void @tl_mem_register_tensor.94(ptr)

declare void @tl_mem_unregister.95(ptr)

declare ptr @tl_pool_acquire.96(i64)

declare void @tl_pool_release.97(ptr, i64)

declare void @tl_arena_init.98(i64)

declare ptr @tl_arena_alloc.99(i64)

declare void @tl_arena_free.100()

declare i1 @tl_arena_is_active.101()

declare void @tl_arena_reset.102()

define ptr @tl_Linear_new(i64 %i, i64 %o) {
entry:
  %scalar_shape_rhs11 = alloca i64, align 16
  %scalar_data_rhs10 = alloca float, align 16
  %shape_arr7 = alloca [1 x i64], align 16
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %shape_arr = alloca [2 x i64], align 16
  %o2 = alloca i64, align 16
  %i1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
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
  store float 0x3FB99999A0000000, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor_rhs)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res, i1 true)
  %init_field = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 0
  store ptr %detach_res, ptr %init_field, align 8
  %o6 = load i64, ptr %o2, align 8
  %shape_ptr_in8 = getelementptr inbounds [1 x i64], ptr %shape_arr7, i64 0, i64 0
  store i64 %o6, ptr %shape_ptr_in8, align 8
  %randn_res9 = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr7, i1 true)
  store float 0.000000e+00, ptr %scalar_data_rhs10, align 4
  %scalar_tensor_rhs12 = call ptr @tl_tensor_new(ptr %scalar_data_rhs10, i64 0, ptr %scalar_shape_rhs11)
  %binop_res13 = call ptr @tl_tensor_mul(ptr %randn_res9, ptr %scalar_tensor_rhs12)
  %detach_res14 = call ptr @tl_tensor_detach(ptr %binop_res13, i1 true)
  %init_field15 = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 1
  store ptr %detach_res14, ptr %init_field15, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 1
  %field_val16 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val16)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_Linear_forward(ptr %self, ptr %x) {
entry:
  %x2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_unregister(ptr %binop_res)
  call void @tl_mem_exit_scope()
  ret ptr %binop_res
}

define void @tl_Linear_step(ptr %self, float %lr) {
entry:
  %scalar_shape_rhs22 = alloca i64, align 16
  %scalar_data_rhs21 = alloca float, align 16
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %gb = alloca ptr, align 16
  %gW = alloca ptr, align 16
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  %ptr_W = getelementptr inbounds nuw %Linear, ptr %self3, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %W)
  store ptr %grad_res, ptr %gW, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds nuw %Linear, ptr %self4, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res5 = call ptr @tl_tensor_grad(ptr %b)
  store ptr %grad_res5, ptr %gb, align 8
  %self6 = load ptr, ptr %self1, align 8
  %ptr_W7 = getelementptr inbounds nuw %Linear, ptr %self6, i32 0, i32 0
  %self8 = load ptr, ptr %self1, align 8
  %ptr_W9 = getelementptr inbounds nuw %Linear, ptr %self8, i32 0, i32 0
  %W10 = load ptr, ptr %ptr_W9, align 8
  %gW11 = load ptr, ptr %gW, align 8
  %lr12 = load float, ptr %lr2, align 4
  store float %lr12, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %gW11, ptr %scalar_tensor_rhs)
  %binop_res13 = call ptr @tl_tensor_sub(ptr %W10, ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res13, i1 true)
  %old_field_val = load ptr, ptr %ptr_W7, align 8
  call void @tl_tensor_free(ptr %old_field_val)
  store ptr %detach_res, ptr %ptr_W7, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  %self14 = load ptr, ptr %self1, align 8
  %ptr_b15 = getelementptr inbounds nuw %Linear, ptr %self14, i32 0, i32 1
  %self16 = load ptr, ptr %self1, align 8
  %ptr_b17 = getelementptr inbounds nuw %Linear, ptr %self16, i32 0, i32 1
  %b18 = load ptr, ptr %ptr_b17, align 8
  %gb19 = load ptr, ptr %gb, align 8
  %lr20 = load float, ptr %lr2, align 4
  store float %lr20, ptr %scalar_data_rhs21, align 4
  %scalar_tensor_rhs23 = call ptr @tl_tensor_new(ptr %scalar_data_rhs21, i64 0, ptr %scalar_shape_rhs22)
  %binop_res24 = call ptr @tl_tensor_mul(ptr %gb19, ptr %scalar_tensor_rhs23)
  %binop_res25 = call ptr @tl_tensor_sub(ptr %b18, ptr %binop_res24)
  %detach_res26 = call ptr @tl_tensor_detach(ptr %binop_res25, i1 true)
  %old_field_val27 = load ptr, ptr %ptr_b15, align 8
  call void @tl_tensor_free(ptr %old_field_val27)
  store ptr %detach_res26, ptr %ptr_b15, align 8
  call void @tl_mem_unregister(ptr %detach_res26)
  call void @tl_mem_exit_scope()
  ret void
}

define ptr @tl_Embedding_new(i64 %v, i64 %d) {
entry:
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %shape_arr = alloca [2 x i64], align 16
  %d2 = alloca i64, align 16
  %v1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
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
  store float 0x3FB99999A0000000, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor_rhs)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res, i1 true)
  %init_field = getelementptr inbounds nuw %Embedding, ptr %struct_malloc, i32 0, i32 0
  store ptr %detach_res, ptr %init_field, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds nuw %Embedding, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_Embedding_forward(ptr %self, ptr %i) {
entry:
  %i2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store ptr %i, ptr %i2, align 8
  %i3 = load ptr, ptr %i2, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds nuw %Embedding, ptr %self4, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %emb_res = call ptr @tl_tensor_embedding(ptr %i3, ptr %w)
  call void @tl_mem_unregister(ptr %emb_res)
  call void @tl_mem_exit_scope()
  ret ptr %emb_res
}

define void @tl_Embedding_step(ptr %self, float %lr) {
entry:
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %g = alloca ptr, align 16
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds nuw %Embedding, ptr %self3, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %w)
  store ptr %grad_res, ptr %g, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_w5 = getelementptr inbounds nuw %Embedding, ptr %self4, i32 0, i32 0
  %self6 = load ptr, ptr %self1, align 8
  %ptr_w7 = getelementptr inbounds nuw %Embedding, ptr %self6, i32 0, i32 0
  %w8 = load ptr, ptr %ptr_w7, align 8
  %g9 = load ptr, ptr %g, align 8
  %lr10 = load float, ptr %lr2, align 4
  store float %lr10, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %g9, ptr %scalar_tensor_rhs)
  %binop_res11 = call ptr @tl_tensor_sub(ptr %w8, ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res11, i1 true)
  %old_field_val = load ptr, ptr %ptr_w5, align 8
  call void @tl_tensor_free(ptr %old_field_val)
  store ptr %detach_res, ptr %ptr_w5, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  call void @tl_mem_exit_scope()
  ret void
}

define ptr @tl_LayerNorm_new(i64 %d) {
entry:
  %scalar_shape_rhs12 = alloca i64, align 16
  %scalar_data_rhs11 = alloca float, align 16
  %shape_arr8 = alloca [1 x i64], align 16
  %scalar_shape_rhs4 = alloca i64, align 16
  %scalar_data_rhs3 = alloca float, align 16
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %shape_arr = alloca [1 x i64], align 16
  %d1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %d, ptr %d1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%LayerNorm, ptr null, i32 1) to i64))
  %d2 = load i64, ptr %d1, align 8
  %shape_ptr_in = getelementptr inbounds [1 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %d2, ptr %shape_ptr_in, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr, i1 true)
  store float 0.000000e+00, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor_rhs)
  store float 1.000000e+00, ptr %scalar_data_rhs3, align 4
  %scalar_tensor_rhs5 = call ptr @tl_tensor_new(ptr %scalar_data_rhs3, i64 0, ptr %scalar_shape_rhs4)
  %binop_res6 = call ptr @tl_tensor_add(ptr %binop_res, ptr %scalar_tensor_rhs5)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res6, i1 true)
  %init_field = getelementptr inbounds nuw %LayerNorm, ptr %struct_malloc, i32 0, i32 0
  store ptr %detach_res, ptr %init_field, align 8
  %d7 = load i64, ptr %d1, align 8
  %shape_ptr_in9 = getelementptr inbounds [1 x i64], ptr %shape_arr8, i64 0, i64 0
  store i64 %d7, ptr %shape_ptr_in9, align 8
  %randn_res10 = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr8, i1 true)
  store float 0.000000e+00, ptr %scalar_data_rhs11, align 4
  %scalar_tensor_rhs13 = call ptr @tl_tensor_new(ptr %scalar_data_rhs11, i64 0, ptr %scalar_shape_rhs12)
  %binop_res14 = call ptr @tl_tensor_mul(ptr %randn_res10, ptr %scalar_tensor_rhs13)
  %detach_res15 = call ptr @tl_tensor_detach(ptr %binop_res14, i1 true)
  %init_field16 = getelementptr inbounds nuw %LayerNorm, ptr %struct_malloc, i32 0, i32 1
  store ptr %detach_res15, ptr %init_field16, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds nuw %LayerNorm, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds nuw %LayerNorm, ptr %struct_malloc, i32 0, i32 1
  %field_val17 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val17)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_LayerNorm_forward(ptr %self, ptr %x) {
entry:
  %x2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %x3 = load ptr, ptr %x2, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds nuw %LayerNorm, ptr %self4, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %x3, ptr %b)
  call void @tl_mem_unregister(ptr %binop_res)
  call void @tl_mem_exit_scope()
  ret ptr %binop_res
}

define void @tl_LayerNorm_step(ptr %self, float %lr) {
entry:
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %gb = alloca ptr, align 16
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds nuw %LayerNorm, ptr %self3, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %b)
  store ptr %grad_res, ptr %gb, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_b5 = getelementptr inbounds nuw %LayerNorm, ptr %self4, i32 0, i32 1
  %self6 = load ptr, ptr %self1, align 8
  %ptr_b7 = getelementptr inbounds nuw %LayerNorm, ptr %self6, i32 0, i32 1
  %b8 = load ptr, ptr %ptr_b7, align 8
  %gb9 = load ptr, ptr %gb, align 8
  %lr10 = load float, ptr %lr2, align 4
  store float %lr10, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %gb9, ptr %scalar_tensor_rhs)
  %binop_res11 = call ptr @tl_tensor_sub(ptr %b8, ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res11, i1 true)
  %old_field_val = load ptr, ptr %ptr_b5, align 8
  call void @tl_tensor_free(ptr %old_field_val)
  store ptr %detach_res, ptr %ptr_b5, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  call void @tl_mem_exit_scope()
  ret void
}

define ptr @tl_CausalSelfAttention_new(i64 %d) {
entry:
  %d1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_09 = getelementptr inbounds nuw %Linear, ptr %field_val, i32 0, i32 0
  %field_val10 = load ptr, ptr %unreg_field_09, align 8
  call void @tl_mem_unregister(ptr %field_val10)
  %unreg_field_1 = getelementptr inbounds nuw %Linear, ptr %field_val, i32 0, i32 1
  %field_val11 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val11)
  %unreg_field_112 = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 1
  %field_val13 = load ptr, ptr %unreg_field_112, align 8
  call void @tl_mem_unregister(ptr %field_val13)
  %unreg_field_014 = getelementptr inbounds nuw %Linear, ptr %field_val13, i32 0, i32 0
  %field_val15 = load ptr, ptr %unreg_field_014, align 8
  call void @tl_mem_unregister(ptr %field_val15)
  %unreg_field_116 = getelementptr inbounds nuw %Linear, ptr %field_val13, i32 0, i32 1
  %field_val17 = load ptr, ptr %unreg_field_116, align 8
  call void @tl_mem_unregister(ptr %field_val17)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_CausalSelfAttention_forward(ptr %self, ptr %x) {
entry:
  %y = alloca ptr, align 16
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %v = alloca ptr, align 16
  %k = alloca ptr, align 16
  %q = alloca ptr, align 16
  %x2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_a = getelementptr inbounds nuw %CausalSelfAttention, ptr %self3, i32 0, i32 0
  %a = load ptr, ptr %ptr_a, align 8
  %x4 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_Linear_forward(ptr %a, ptr %x4)
  call void @tl_mem_register_tensor(ptr %call_method)
  store ptr %call_method, ptr %q, align 8
  %q5 = load ptr, ptr %q, align 8
  %cloned = call ptr @tl_tensor_clone(ptr %q5)
  store ptr %cloned, ptr %k, align 8
  %q6 = load ptr, ptr %q, align 8
  %cloned7 = call ptr @tl_tensor_clone(ptr %q6)
  store ptr %cloned7, ptr %v, align 8
  %q8 = load ptr, ptr %q, align 8
  %k9 = load ptr, ptr %k, align 8
  %transpose_res = call ptr @tl_tensor_transpose(ptr %k9, i64 1, i64 2)
  %matmul_res = call ptr @tl_tensor_matmul(ptr %q8, ptr %transpose_res)
  store float 1.250000e-01, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %matmul_res, ptr %scalar_tensor_rhs)
  %tril_res = call ptr @tl_tensor_tril(ptr %binop_res, i32 0)
  %softmax_res = call ptr @tl_tensor_softmax(ptr %tril_res, i64 2)
  %v10 = load ptr, ptr %v, align 8
  %matmul_res11 = call ptr @tl_tensor_matmul(ptr %softmax_res, ptr %v10)
  store ptr %matmul_res11, ptr %y, align 8
  %self12 = load ptr, ptr %self1, align 8
  %ptr_p = getelementptr inbounds nuw %CausalSelfAttention, ptr %self12, i32 0, i32 1
  %p = load ptr, ptr %ptr_p, align 8
  %y13 = load ptr, ptr %y, align 8
  %call_method14 = call ptr @tl_Linear_forward(ptr %p, ptr %y13)
  call void @tl_mem_register_tensor(ptr %call_method14)
  call void @tl_mem_unregister(ptr %call_method14)
  call void @tl_mem_exit_scope()
  ret ptr %call_method14
}

define void @tl_CausalSelfAttention_step(ptr %self, float %lr) {
entry:
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_exit_scope()
  ret void
}

define ptr @tl_MLP_new(i64 %d) {
entry:
  %d1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds nuw %MLP, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_09 = getelementptr inbounds nuw %Linear, ptr %field_val, i32 0, i32 0
  %field_val10 = load ptr, ptr %unreg_field_09, align 8
  call void @tl_mem_unregister(ptr %field_val10)
  %unreg_field_1 = getelementptr inbounds nuw %Linear, ptr %field_val, i32 0, i32 1
  %field_val11 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val11)
  %unreg_field_112 = getelementptr inbounds nuw %MLP, ptr %struct_malloc, i32 0, i32 1
  %field_val13 = load ptr, ptr %unreg_field_112, align 8
  call void @tl_mem_unregister(ptr %field_val13)
  %unreg_field_014 = getelementptr inbounds nuw %Linear, ptr %field_val13, i32 0, i32 0
  %field_val15 = load ptr, ptr %unreg_field_014, align 8
  call void @tl_mem_unregister(ptr %field_val15)
  %unreg_field_116 = getelementptr inbounds nuw %Linear, ptr %field_val13, i32 0, i32 1
  %field_val17 = load ptr, ptr %unreg_field_116, align 8
  call void @tl_mem_unregister(ptr %field_val17)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_MLP_forward(ptr %self, ptr %x) {
entry:
  %x2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_register_tensor(ptr %call_method)
  %relu_res = call ptr @tl_tensor_relu(ptr %call_method)
  %call_method6 = call ptr @tl_Linear_forward(ptr %p, ptr %relu_res)
  call void @tl_mem_register_tensor(ptr %call_method6)
  call void @tl_mem_unregister(ptr %call_method6)
  call void @tl_mem_exit_scope()
  ret ptr %call_method6
}

define void @tl_MLP_step(ptr %self, float %lr) {
entry:
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_exit_scope()
  ret void
}

define ptr @tl_Block_new(i64 %d) {
entry:
  %d1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_012 = getelementptr inbounds nuw %LayerNorm, ptr %field_val, i32 0, i32 0
  %field_val13 = load ptr, ptr %unreg_field_012, align 8
  call void @tl_mem_unregister(ptr %field_val13)
  %unreg_field_1 = getelementptr inbounds nuw %LayerNorm, ptr %field_val, i32 0, i32 1
  %field_val14 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val14)
  %unreg_field_115 = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 1
  %field_val16 = load ptr, ptr %unreg_field_115, align 8
  call void @tl_mem_unregister(ptr %field_val16)
  %unreg_field_017 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val16, i32 0, i32 0
  %field_val18 = load ptr, ptr %unreg_field_017, align 8
  call void @tl_mem_unregister(ptr %field_val18)
  %unreg_field_019 = getelementptr inbounds nuw %Linear, ptr %field_val18, i32 0, i32 0
  %field_val20 = load ptr, ptr %unreg_field_019, align 8
  call void @tl_mem_unregister(ptr %field_val20)
  %unreg_field_121 = getelementptr inbounds nuw %Linear, ptr %field_val18, i32 0, i32 1
  %field_val22 = load ptr, ptr %unreg_field_121, align 8
  call void @tl_mem_unregister(ptr %field_val22)
  %unreg_field_123 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val16, i32 0, i32 1
  %field_val24 = load ptr, ptr %unreg_field_123, align 8
  call void @tl_mem_unregister(ptr %field_val24)
  %unreg_field_025 = getelementptr inbounds nuw %Linear, ptr %field_val24, i32 0, i32 0
  %field_val26 = load ptr, ptr %unreg_field_025, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  %unreg_field_127 = getelementptr inbounds nuw %Linear, ptr %field_val24, i32 0, i32 1
  %field_val28 = load ptr, ptr %unreg_field_127, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_2 = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 2
  %field_val29 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val29)
  %unreg_field_030 = getelementptr inbounds nuw %LayerNorm, ptr %field_val29, i32 0, i32 0
  %field_val31 = load ptr, ptr %unreg_field_030, align 8
  call void @tl_mem_unregister(ptr %field_val31)
  %unreg_field_132 = getelementptr inbounds nuw %LayerNorm, ptr %field_val29, i32 0, i32 1
  %field_val33 = load ptr, ptr %unreg_field_132, align 8
  call void @tl_mem_unregister(ptr %field_val33)
  %unreg_field_3 = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 3
  %field_val34 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  %unreg_field_035 = getelementptr inbounds nuw %MLP, ptr %field_val34, i32 0, i32 0
  %field_val36 = load ptr, ptr %unreg_field_035, align 8
  call void @tl_mem_unregister(ptr %field_val36)
  %unreg_field_037 = getelementptr inbounds nuw %Linear, ptr %field_val36, i32 0, i32 0
  %field_val38 = load ptr, ptr %unreg_field_037, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_139 = getelementptr inbounds nuw %Linear, ptr %field_val36, i32 0, i32 1
  %field_val40 = load ptr, ptr %unreg_field_139, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_141 = getelementptr inbounds nuw %MLP, ptr %field_val34, i32 0, i32 1
  %field_val42 = load ptr, ptr %unreg_field_141, align 8
  call void @tl_mem_unregister(ptr %field_val42)
  %unreg_field_043 = getelementptr inbounds nuw %Linear, ptr %field_val42, i32 0, i32 0
  %field_val44 = load ptr, ptr %unreg_field_043, align 8
  call void @tl_mem_unregister(ptr %field_val44)
  %unreg_field_145 = getelementptr inbounds nuw %Linear, ptr %field_val42, i32 0, i32 1
  %field_val46 = load ptr, ptr %unreg_field_145, align 8
  call void @tl_mem_unregister(ptr %field_val46)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_Block_forward(ptr %self, ptr %x) {
entry:
  %x8 = alloca ptr, align 16
  %x2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_register_tensor(ptr %call_method)
  %call_method7 = call ptr @tl_CausalSelfAttention_forward(ptr %a, ptr %call_method)
  call void @tl_mem_register_tensor(ptr %call_method7)
  %binop_res = call ptr @tl_tensor_add(ptr %x3, ptr %call_method7)
  store ptr %binop_res, ptr %x8, align 8
  %x9 = load ptr, ptr %x8, align 8
  %self10 = load ptr, ptr %self1, align 8
  %ptr_m = getelementptr inbounds nuw %Block, ptr %self10, i32 0, i32 3
  %m = load ptr, ptr %ptr_m, align 8
  %self11 = load ptr, ptr %self1, align 8
  %ptr_l2 = getelementptr inbounds nuw %Block, ptr %self11, i32 0, i32 2
  %l2 = load ptr, ptr %ptr_l2, align 8
  %x12 = load ptr, ptr %x8, align 8
  %call_method13 = call ptr @tl_LayerNorm_forward(ptr %l2, ptr %x12)
  call void @tl_mem_register_tensor(ptr %call_method13)
  %call_method14 = call ptr @tl_MLP_forward(ptr %m, ptr %call_method13)
  call void @tl_mem_register_tensor(ptr %call_method14)
  %binop_res15 = call ptr @tl_tensor_add(ptr %x9, ptr %call_method14)
  call void @tl_mem_unregister(ptr %binop_res15)
  call void @tl_mem_exit_scope()
  ret ptr %binop_res15
}

define void @tl_Block_step(ptr %self, float %lr) {
entry:
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_exit_scope()
  ret void
}

define ptr @tl_GPT_new(i64 %v, i64 %d) {
entry:
  %d2 = alloca i64, align 16
  %v1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds nuw %GPT, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_015 = getelementptr inbounds nuw %Embedding, ptr %field_val, i32 0, i32 0
  %field_val16 = load ptr, ptr %unreg_field_015, align 8
  call void @tl_mem_unregister(ptr %field_val16)
  %unreg_field_1 = getelementptr inbounds nuw %GPT, ptr %struct_malloc, i32 0, i32 1
  %field_val17 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val17)
  %unreg_field_018 = getelementptr inbounds nuw %Block, ptr %field_val17, i32 0, i32 0
  %field_val19 = load ptr, ptr %unreg_field_018, align 8
  call void @tl_mem_unregister(ptr %field_val19)
  %unreg_field_020 = getelementptr inbounds nuw %LayerNorm, ptr %field_val19, i32 0, i32 0
  %field_val21 = load ptr, ptr %unreg_field_020, align 8
  call void @tl_mem_unregister(ptr %field_val21)
  %unreg_field_122 = getelementptr inbounds nuw %LayerNorm, ptr %field_val19, i32 0, i32 1
  %field_val23 = load ptr, ptr %unreg_field_122, align 8
  call void @tl_mem_unregister(ptr %field_val23)
  %unreg_field_124 = getelementptr inbounds nuw %Block, ptr %field_val17, i32 0, i32 1
  %field_val25 = load ptr, ptr %unreg_field_124, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_026 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val25, i32 0, i32 0
  %field_val27 = load ptr, ptr %unreg_field_026, align 8
  call void @tl_mem_unregister(ptr %field_val27)
  %unreg_field_028 = getelementptr inbounds nuw %Linear, ptr %field_val27, i32 0, i32 0
  %field_val29 = load ptr, ptr %unreg_field_028, align 8
  call void @tl_mem_unregister(ptr %field_val29)
  %unreg_field_130 = getelementptr inbounds nuw %Linear, ptr %field_val27, i32 0, i32 1
  %field_val31 = load ptr, ptr %unreg_field_130, align 8
  call void @tl_mem_unregister(ptr %field_val31)
  %unreg_field_132 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val25, i32 0, i32 1
  %field_val33 = load ptr, ptr %unreg_field_132, align 8
  call void @tl_mem_unregister(ptr %field_val33)
  %unreg_field_034 = getelementptr inbounds nuw %Linear, ptr %field_val33, i32 0, i32 0
  %field_val35 = load ptr, ptr %unreg_field_034, align 8
  call void @tl_mem_unregister(ptr %field_val35)
  %unreg_field_136 = getelementptr inbounds nuw %Linear, ptr %field_val33, i32 0, i32 1
  %field_val37 = load ptr, ptr %unreg_field_136, align 8
  call void @tl_mem_unregister(ptr %field_val37)
  %unreg_field_2 = getelementptr inbounds nuw %Block, ptr %field_val17, i32 0, i32 2
  %field_val38 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_039 = getelementptr inbounds nuw %LayerNorm, ptr %field_val38, i32 0, i32 0
  %field_val40 = load ptr, ptr %unreg_field_039, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_141 = getelementptr inbounds nuw %LayerNorm, ptr %field_val38, i32 0, i32 1
  %field_val42 = load ptr, ptr %unreg_field_141, align 8
  call void @tl_mem_unregister(ptr %field_val42)
  %unreg_field_3 = getelementptr inbounds nuw %Block, ptr %field_val17, i32 0, i32 3
  %field_val43 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val43)
  %unreg_field_044 = getelementptr inbounds nuw %MLP, ptr %field_val43, i32 0, i32 0
  %field_val45 = load ptr, ptr %unreg_field_044, align 8
  call void @tl_mem_unregister(ptr %field_val45)
  %unreg_field_046 = getelementptr inbounds nuw %Linear, ptr %field_val45, i32 0, i32 0
  %field_val47 = load ptr, ptr %unreg_field_046, align 8
  call void @tl_mem_unregister(ptr %field_val47)
  %unreg_field_148 = getelementptr inbounds nuw %Linear, ptr %field_val45, i32 0, i32 1
  %field_val49 = load ptr, ptr %unreg_field_148, align 8
  call void @tl_mem_unregister(ptr %field_val49)
  %unreg_field_150 = getelementptr inbounds nuw %MLP, ptr %field_val43, i32 0, i32 1
  %field_val51 = load ptr, ptr %unreg_field_150, align 8
  call void @tl_mem_unregister(ptr %field_val51)
  %unreg_field_052 = getelementptr inbounds nuw %Linear, ptr %field_val51, i32 0, i32 0
  %field_val53 = load ptr, ptr %unreg_field_052, align 8
  call void @tl_mem_unregister(ptr %field_val53)
  %unreg_field_154 = getelementptr inbounds nuw %Linear, ptr %field_val51, i32 0, i32 1
  %field_val55 = load ptr, ptr %unreg_field_154, align 8
  call void @tl_mem_unregister(ptr %field_val55)
  %unreg_field_256 = getelementptr inbounds nuw %GPT, ptr %struct_malloc, i32 0, i32 2
  %field_val57 = load ptr, ptr %unreg_field_256, align 8
  call void @tl_mem_unregister(ptr %field_val57)
  %unreg_field_058 = getelementptr inbounds nuw %LayerNorm, ptr %field_val57, i32 0, i32 0
  %field_val59 = load ptr, ptr %unreg_field_058, align 8
  call void @tl_mem_unregister(ptr %field_val59)
  %unreg_field_160 = getelementptr inbounds nuw %LayerNorm, ptr %field_val57, i32 0, i32 1
  %field_val61 = load ptr, ptr %unreg_field_160, align 8
  call void @tl_mem_unregister(ptr %field_val61)
  %unreg_field_362 = getelementptr inbounds nuw %GPT, ptr %struct_malloc, i32 0, i32 3
  %field_val63 = load ptr, ptr %unreg_field_362, align 8
  call void @tl_mem_unregister(ptr %field_val63)
  %unreg_field_064 = getelementptr inbounds nuw %Linear, ptr %field_val63, i32 0, i32 0
  %field_val65 = load ptr, ptr %unreg_field_064, align 8
  call void @tl_mem_unregister(ptr %field_val65)
  %unreg_field_166 = getelementptr inbounds nuw %Linear, ptr %field_val63, i32 0, i32 1
  %field_val67 = load ptr, ptr %unreg_field_166, align 8
  call void @tl_mem_unregister(ptr %field_val67)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_GPT_forward(ptr %self, ptr %i) {
entry:
  %i2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_register_tensor(ptr %call_method)
  %call_method8 = call ptr @tl_Block_forward(ptr %b, ptr %call_method)
  call void @tl_mem_register_tensor(ptr %call_method8)
  %call_method9 = call ptr @tl_LayerNorm_forward(ptr %l, ptr %call_method8)
  call void @tl_mem_register_tensor(ptr %call_method9)
  %call_method10 = call ptr @tl_Linear_forward(ptr %h, ptr %call_method9)
  call void @tl_mem_register_tensor(ptr %call_method10)
  call void @tl_mem_unregister(ptr %call_method10)
  call void @tl_mem_exit_scope()
  ret ptr %call_method10
}

define void @tl_GPT_step(ptr %self, float %lr) {
entry:
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_exit_scope()
  ret void
}

define i64 @get_memory() {
entry:
  call void @tl_mem_enter_scope()
  %call_tmp = call i64 @tl_get_memory_mb()
  call void @tl_mem_exit_scope()
  ret i64 %call_tmp
}

define void @train_epoch(ptr %model, float %lr, i64 %epoch) {
entry:
  %mem_mb = alloca i64, align 16
  %loss = alloca ptr, align 16
  %Y_flat = alloca ptr, align 16
  %logits_flat = alloca ptr, align 16
  %logits = alloca ptr, align 16
  %Y = alloca ptr, align 16
  %X = alloca ptr, align 16
  %target = alloca ptr, align 16
  %tensor_shape_arr151 = alloca [1 x i64], align 8
  %data = alloca ptr, align 16
  %tensor_shape_arr = alloca [1 x i64], align 8
  %s_d3 = alloca float, align 16
  %scalar_shape110 = alloca i64, align 16
  %scalar_data109 = alloca float, align 16
  %scalar_shape107 = alloca i64, align 16
  %scalar_data105 = alloca float, align 16
  %s_d2 = alloca float, align 16
  %scalar_shape96 = alloca i64, align 16
  %scalar_data95 = alloca float, align 16
  %scalar_shape93 = alloca i64, align 16
  %scalar_data91 = alloca float, align 16
  %s_d1 = alloca float, align 16
  %scalar_shape81 = alloca i64, align 16
  %scalar_data80 = alloca float, align 16
  %scalar_shape78 = alloca i64, align 16
  %scalar_data76 = alloca float, align 16
  %j_d2 = alloca float, align 16
  %scalar_shape70 = alloca i64, align 16
  %scalar_data69 = alloca float, align 16
  %scalar_shape67 = alloca i64, align 16
  %scalar_data65 = alloca float, align 16
  %j_d1 = alloca float, align 16
  %scalar_shape56 = alloca i64, align 16
  %scalar_data55 = alloca float, align 16
  %scalar_shape53 = alloca i64, align 16
  %scalar_data51 = alloca float, align 16
  %i_d2 = alloca float, align 16
  %scalar_shape45 = alloca i64, align 16
  %scalar_data44 = alloca float, align 16
  %scalar_shape42 = alloca i64, align 16
  %scalar_data40 = alloca float, align 16
  %i_d1 = alloca float, align 16
  %scalar_shape32 = alloca i64, align 16
  %scalar_data31 = alloca float, align 16
  %scalar_shape29 = alloca i64, align 16
  %scalar_data28 = alloca float, align 16
  %sum = alloca i64, align 16
  %j = alloca i64, align 16
  %i = alloca i64, align 16
  %idx = alloca i64, align 16
  %raw = alloca i64, align 16
  %s = alloca i64, align 16
  %offset = alloca i64, align 16
  %stride = alloca i64, align 16
  %total_steps = alloca i64, align 16
  %total_loss = alloca ptr, align 16
  %scalar_shape5 = alloca i64, align 16
  %scalar_data4 = alloca float, align 16
  %scalar_shape = alloca i64, align 16
  %scalar_data = alloca float, align 16
  %epoch3 = alloca i64, align 16
  %lr2 = alloca float, align 16
  %model1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %model, ptr %model1, align 8
  store float %lr, ptr %lr2, align 4
  store i64 %epoch, ptr %epoch3, align 8
  store float 0.000000e+00, ptr %scalar_data, align 4
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  store float 1.000000e+00, ptr %scalar_data4, align 4
  %scalar_tensor6 = call ptr @tl_tensor_new(ptr %scalar_data4, i64 0, ptr %scalar_shape5)
  %pow_res = call ptr @tl_tensor_pow(ptr %scalar_tensor, ptr %scalar_tensor6)
  store ptr %pow_res, ptr %total_loss, align 8
  store i64 1000, ptr %total_steps, align 8
  store i64 137, ptr %stride, align 8
  %epoch7 = load i64, ptr %epoch3, align 8
  %multmp = mul i64 %epoch7, 79
  store i64 %multmp, ptr %offset, align 8
  %total_steps8 = load i64, ptr %total_steps, align 8
  br label %for_header

for_header:                                       ; preds = %continue_block, %entry
  %for_idx = phi i64 [ 0, %entry ], [ %next_idx, %continue_block ]
  %for_cond = icmp slt i64 %for_idx, %total_steps8
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %s, align 8
  %s9 = load i64, ptr %s, align 8
  %stride10 = load i64, ptr %stride, align 8
  %multmp11 = mul i64 %s9, %stride10
  %offset12 = load i64, ptr %offset, align 8
  %addtmp = add i64 %multmp11, %offset12
  store i64 %addtmp, ptr %raw, align 8
  %raw13 = load i64, ptr %raw, align 8
  %raw14 = load i64, ptr %raw, align 8
  %divtmp = sdiv i64 %raw14, 10000
  %multmp15 = mul i64 %divtmp, 10000
  %subtmp = sub i64 %raw13, %multmp15
  store i64 %subtmp, ptr %idx, align 8
  %idx16 = load i64, ptr %idx, align 8
  %divtmp17 = sdiv i64 %idx16, 100
  store i64 %divtmp17, ptr %i, align 8
  %idx18 = load i64, ptr %idx, align 8
  %idx19 = load i64, ptr %idx, align 8
  %divtmp20 = sdiv i64 %idx19, 100
  %multmp21 = mul i64 %divtmp20, 100
  %subtmp22 = sub i64 %idx18, %multmp21
  store i64 %subtmp22, ptr %j, align 8
  %i23 = load i64, ptr %i, align 8
  %j24 = load i64, ptr %j, align 8
  %addtmp25 = add i64 %i23, %j24
  store i64 %addtmp25, ptr %sum, align 8
  %i26 = load i64, ptr %i, align 8
  %divtmp27 = sdiv i64 %i26, 10
  %cast_i64_f32 = sitofp i64 %divtmp27 to float
  store float %cast_i64_f32, ptr %scalar_data28, align 4
  %scalar_tensor30 = call ptr @tl_tensor_new(ptr %scalar_data28, i64 0, ptr %scalar_shape29)
  store float 1.000000e+00, ptr %scalar_data31, align 4
  %scalar_tensor33 = call ptr @tl_tensor_new(ptr %scalar_data31, i64 0, ptr %scalar_shape32)
  %pow_res34 = call ptr @tl_tensor_pow(ptr %scalar_tensor30, ptr %scalar_tensor33)
  %get_res = call float @tl_tensor_get(ptr %pow_res34, i64 0)
  store float %get_res, ptr %i_d1, align 4
  %i35 = load i64, ptr %i, align 8
  %i36 = load i64, ptr %i, align 8
  %divtmp37 = sdiv i64 %i36, 10
  %multmp38 = mul i64 %divtmp37, 10
  %subtmp39 = sub i64 %i35, %multmp38
  %cast_i64_f3241 = sitofp i64 %subtmp39 to float
  store float %cast_i64_f3241, ptr %scalar_data40, align 4
  %scalar_tensor43 = call ptr @tl_tensor_new(ptr %scalar_data40, i64 0, ptr %scalar_shape42)
  store float 1.000000e+00, ptr %scalar_data44, align 4
  %scalar_tensor46 = call ptr @tl_tensor_new(ptr %scalar_data44, i64 0, ptr %scalar_shape45)
  %pow_res47 = call ptr @tl_tensor_pow(ptr %scalar_tensor43, ptr %scalar_tensor46)
  %get_res48 = call float @tl_tensor_get(ptr %pow_res47, i64 0)
  store float %get_res48, ptr %i_d2, align 4
  %j49 = load i64, ptr %j, align 8
  %divtmp50 = sdiv i64 %j49, 10
  %cast_i64_f3252 = sitofp i64 %divtmp50 to float
  store float %cast_i64_f3252, ptr %scalar_data51, align 4
  %scalar_tensor54 = call ptr @tl_tensor_new(ptr %scalar_data51, i64 0, ptr %scalar_shape53)
  store float 1.000000e+00, ptr %scalar_data55, align 4
  %scalar_tensor57 = call ptr @tl_tensor_new(ptr %scalar_data55, i64 0, ptr %scalar_shape56)
  %pow_res58 = call ptr @tl_tensor_pow(ptr %scalar_tensor54, ptr %scalar_tensor57)
  %get_res59 = call float @tl_tensor_get(ptr %pow_res58, i64 0)
  store float %get_res59, ptr %j_d1, align 4
  %j60 = load i64, ptr %j, align 8
  %j61 = load i64, ptr %j, align 8
  %divtmp62 = sdiv i64 %j61, 10
  %multmp63 = mul i64 %divtmp62, 10
  %subtmp64 = sub i64 %j60, %multmp63
  %cast_i64_f3266 = sitofp i64 %subtmp64 to float
  store float %cast_i64_f3266, ptr %scalar_data65, align 4
  %scalar_tensor68 = call ptr @tl_tensor_new(ptr %scalar_data65, i64 0, ptr %scalar_shape67)
  store float 1.000000e+00, ptr %scalar_data69, align 4
  %scalar_tensor71 = call ptr @tl_tensor_new(ptr %scalar_data69, i64 0, ptr %scalar_shape70)
  %pow_res72 = call ptr @tl_tensor_pow(ptr %scalar_tensor68, ptr %scalar_tensor71)
  %get_res73 = call float @tl_tensor_get(ptr %pow_res72, i64 0)
  store float %get_res73, ptr %j_d2, align 4
  %sum74 = load i64, ptr %sum, align 8
  %divtmp75 = sdiv i64 %sum74, 100
  %cast_i64_f3277 = sitofp i64 %divtmp75 to float
  store float %cast_i64_f3277, ptr %scalar_data76, align 4
  %scalar_tensor79 = call ptr @tl_tensor_new(ptr %scalar_data76, i64 0, ptr %scalar_shape78)
  store float 1.000000e+00, ptr %scalar_data80, align 4
  %scalar_tensor82 = call ptr @tl_tensor_new(ptr %scalar_data80, i64 0, ptr %scalar_shape81)
  %pow_res83 = call ptr @tl_tensor_pow(ptr %scalar_tensor79, ptr %scalar_tensor82)
  %get_res84 = call float @tl_tensor_get(ptr %pow_res83, i64 0)
  store float %get_res84, ptr %s_d1, align 4
  %sum85 = load i64, ptr %sum, align 8
  %sum86 = load i64, ptr %sum, align 8
  %divtmp87 = sdiv i64 %sum86, 100
  %multmp88 = mul i64 %divtmp87, 100
  %subtmp89 = sub i64 %sum85, %multmp88
  %divtmp90 = sdiv i64 %subtmp89, 10
  %cast_i64_f3292 = sitofp i64 %divtmp90 to float
  store float %cast_i64_f3292, ptr %scalar_data91, align 4
  %scalar_tensor94 = call ptr @tl_tensor_new(ptr %scalar_data91, i64 0, ptr %scalar_shape93)
  store float 1.000000e+00, ptr %scalar_data95, align 4
  %scalar_tensor97 = call ptr @tl_tensor_new(ptr %scalar_data95, i64 0, ptr %scalar_shape96)
  %pow_res98 = call ptr @tl_tensor_pow(ptr %scalar_tensor94, ptr %scalar_tensor97)
  %get_res99 = call float @tl_tensor_get(ptr %pow_res98, i64 0)
  store float %get_res99, ptr %s_d2, align 4
  %sum100 = load i64, ptr %sum, align 8
  %sum101 = load i64, ptr %sum, align 8
  %divtmp102 = sdiv i64 %sum101, 10
  %multmp103 = mul i64 %divtmp102, 10
  %subtmp104 = sub i64 %sum100, %multmp103
  %cast_i64_f32106 = sitofp i64 %subtmp104 to float
  store float %cast_i64_f32106, ptr %scalar_data105, align 4
  %scalar_tensor108 = call ptr @tl_tensor_new(ptr %scalar_data105, i64 0, ptr %scalar_shape107)
  store float 1.000000e+00, ptr %scalar_data109, align 4
  %scalar_tensor111 = call ptr @tl_tensor_new(ptr %scalar_data109, i64 0, ptr %scalar_shape110)
  %pow_res112 = call ptr @tl_tensor_pow(ptr %scalar_tensor108, ptr %scalar_tensor111)
  %get_res113 = call float @tl_tensor_get(ptr %pow_res112, i64 0)
  store float %get_res113, ptr %s_d3, align 4
  %buf_void = call ptr @calloc(i64 12, i64 4)
  %i_d1114 = load float, ptr %i_d1, align 4
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %i_d1114, ptr %elem_ptr, align 4
  %i_d2115 = load float, ptr %i_d2, align 4
  %elem_ptr116 = getelementptr inbounds float, ptr %buf_void, i64 1
  store float %i_d2115, ptr %elem_ptr116, align 4
  %elem_ptr117 = getelementptr inbounds float, ptr %buf_void, i64 2
  store float 1.000000e+01, ptr %elem_ptr117, align 4
  %j_d1118 = load float, ptr %j_d1, align 4
  %elem_ptr119 = getelementptr inbounds float, ptr %buf_void, i64 3
  store float %j_d1118, ptr %elem_ptr119, align 4
  %j_d2120 = load float, ptr %j_d2, align 4
  %elem_ptr121 = getelementptr inbounds float, ptr %buf_void, i64 4
  store float %j_d2120, ptr %elem_ptr121, align 4
  %elem_ptr122 = getelementptr inbounds float, ptr %buf_void, i64 5
  store float 1.100000e+01, ptr %elem_ptr122, align 4
  %s_d1123 = load float, ptr %s_d1, align 4
  %elem_ptr124 = getelementptr inbounds float, ptr %buf_void, i64 6
  store float %s_d1123, ptr %elem_ptr124, align 4
  %s_d2125 = load float, ptr %s_d2, align 4
  %elem_ptr126 = getelementptr inbounds float, ptr %buf_void, i64 7
  store float %s_d2125, ptr %elem_ptr126, align 4
  %s_d3127 = load float, ptr %s_d3, align 4
  %elem_ptr128 = getelementptr inbounds float, ptr %buf_void, i64 8
  store float %s_d3127, ptr %elem_ptr128, align 4
  %elem_ptr129 = getelementptr inbounds float, ptr %buf_void, i64 9
  store float 1.200000e+01, ptr %elem_ptr129, align 4
  %elem_ptr130 = getelementptr inbounds float, ptr %buf_void, i64 10
  store float 1.200000e+01, ptr %elem_ptr130, align 4
  %elem_ptr131 = getelementptr inbounds float, ptr %buf_void, i64 11
  store float 1.200000e+01, ptr %elem_ptr131, align 4
  %shape_ptr = getelementptr inbounds [1 x i64], ptr %tensor_shape_arr, i64 0, i64 0
  store i64 12, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %tensor_shape_arr)
  store ptr %new_tensor, ptr %data, align 8
  %buf_void132 = call ptr @calloc(i64 12, i64 4)
  %i_d2133 = load float, ptr %i_d2, align 4
  %elem_ptr134 = getelementptr inbounds float, ptr %buf_void132, i64 0
  store float %i_d2133, ptr %elem_ptr134, align 4
  %elem_ptr135 = getelementptr inbounds float, ptr %buf_void132, i64 1
  store float 1.000000e+01, ptr %elem_ptr135, align 4
  %j_d1136 = load float, ptr %j_d1, align 4
  %elem_ptr137 = getelementptr inbounds float, ptr %buf_void132, i64 2
  store float %j_d1136, ptr %elem_ptr137, align 4
  %j_d2138 = load float, ptr %j_d2, align 4
  %elem_ptr139 = getelementptr inbounds float, ptr %buf_void132, i64 3
  store float %j_d2138, ptr %elem_ptr139, align 4
  %elem_ptr140 = getelementptr inbounds float, ptr %buf_void132, i64 4
  store float 1.100000e+01, ptr %elem_ptr140, align 4
  %s_d1141 = load float, ptr %s_d1, align 4
  %elem_ptr142 = getelementptr inbounds float, ptr %buf_void132, i64 5
  store float %s_d1141, ptr %elem_ptr142, align 4
  %s_d2143 = load float, ptr %s_d2, align 4
  %elem_ptr144 = getelementptr inbounds float, ptr %buf_void132, i64 6
  store float %s_d2143, ptr %elem_ptr144, align 4
  %s_d3145 = load float, ptr %s_d3, align 4
  %elem_ptr146 = getelementptr inbounds float, ptr %buf_void132, i64 7
  store float %s_d3145, ptr %elem_ptr146, align 4
  %elem_ptr147 = getelementptr inbounds float, ptr %buf_void132, i64 8
  store float 1.200000e+01, ptr %elem_ptr147, align 4
  %elem_ptr148 = getelementptr inbounds float, ptr %buf_void132, i64 9
  store float 1.200000e+01, ptr %elem_ptr148, align 4
  %elem_ptr149 = getelementptr inbounds float, ptr %buf_void132, i64 10
  store float 1.200000e+01, ptr %elem_ptr149, align 4
  %elem_ptr150 = getelementptr inbounds float, ptr %buf_void132, i64 11
  store float 1.200000e+01, ptr %elem_ptr150, align 4
  %shape_ptr152 = getelementptr inbounds [1 x i64], ptr %tensor_shape_arr151, i64 0, i64 0
  store i64 12, ptr %shape_ptr152, align 8
  %new_tensor153 = call ptr @tl_tensor_new(ptr %buf_void132, i64 1, ptr %tensor_shape_arr151)
  store ptr %new_tensor153, ptr %target, align 8
  %data154 = load ptr, ptr %data, align 8
  %dims_alloca = alloca [2 x i64], align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr, align 8
  %dim_ptr155 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr155, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %data154, ptr %dims_ptr, i64 2)
  store ptr %reshape_dims_res, ptr %X, align 8
  %target156 = load ptr, ptr %target, align 8
  %dims_alloca157 = alloca [2 x i64], align 8
  %dim_ptr158 = getelementptr [2 x i64], ptr %dims_alloca157, i64 0, i64 0
  store i64 1, ptr %dim_ptr158, align 8
  %dim_ptr159 = getelementptr [2 x i64], ptr %dims_alloca157, i64 0, i64 1
  store i64 12, ptr %dim_ptr159, align 8
  %dims_ptr160 = getelementptr [2 x i64], ptr %dims_alloca157, i64 0, i64 0
  %reshape_dims_res161 = call ptr @tl_tensor_reshape_dims(ptr %target156, ptr %dims_ptr160, i64 2)
  store ptr %reshape_dims_res161, ptr %Y, align 8
  %model162 = load ptr, ptr %model1, align 8
  %X163 = load ptr, ptr %X, align 8
  %call_method = call ptr @tl_GPT_forward(ptr %model162, ptr %X163)
  call void @tl_mem_register_tensor(ptr %call_method)
  store ptr %call_method, ptr %logits, align 8
  %logits164 = load ptr, ptr %logits, align 8
  %dims_alloca165 = alloca [2 x i64], align 8
  %dim_ptr166 = getelementptr [2 x i64], ptr %dims_alloca165, i64 0, i64 0
  store i64 12, ptr %dim_ptr166, align 8
  %dim_ptr167 = getelementptr [2 x i64], ptr %dims_alloca165, i64 0, i64 1
  store i64 13, ptr %dim_ptr167, align 8
  %dims_ptr168 = getelementptr [2 x i64], ptr %dims_alloca165, i64 0, i64 0
  %reshape_dims_res169 = call ptr @tl_tensor_reshape_dims(ptr %logits164, ptr %dims_ptr168, i64 2)
  store ptr %reshape_dims_res169, ptr %logits_flat, align 8
  %Y170 = load ptr, ptr %Y, align 8
  %dims_alloca171 = alloca [1 x i64], align 8
  %dim_ptr172 = getelementptr [1 x i64], ptr %dims_alloca171, i64 0, i64 0
  store i64 12, ptr %dim_ptr172, align 8
  %dims_ptr173 = getelementptr [1 x i64], ptr %dims_alloca171, i64 0, i64 0
  %reshape_dims_res174 = call ptr @tl_tensor_reshape_dims(ptr %Y170, ptr %dims_ptr173, i64 1)
  store ptr %reshape_dims_res174, ptr %Y_flat, align 8
  %logits_flat175 = load ptr, ptr %logits_flat, align 8
  %Y_flat176 = load ptr, ptr %Y_flat, align 8
  %ce_res = call ptr @tl_tensor_cross_entropy(ptr %logits_flat175, ptr %Y_flat176)
  store ptr %ce_res, ptr %loss, align 8
  %loss177 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss177)
  %model178 = load ptr, ptr %model1, align 8
  %lr179 = load float, ptr %lr2, align 4
  call void @tl_GPT_step(ptr %model178, float %lr179)
  %loss180 = load ptr, ptr %loss, align 8
  %detach_res = call ptr @tl_tensor_detach(ptr %loss180, i1 true)
  %old_val = load ptr, ptr %total_loss, align 8
  %is_not_null = icmp ne ptr %old_val, null
  br i1 %is_not_null, label %free_block, label %continue_block

for_end:                                          ; preds = %for_header
  %call_tmp = call i64 @get_memory()
  store i64 %call_tmp, ptr %mem_mb, align 8
  call void @tl_print_string(ptr @str_literal)
  %total_loss181 = load ptr, ptr %total_loss, align 8
  call void @tl_tensor_print(ptr %total_loss181)
  call void @tl_print_string(ptr @str_literal.103)
  %mem_mb182 = load i64, ptr %mem_mb, align 8
  call void @tl_print_i64(i64 %mem_mb182)
  call void @tl_mem_exit_scope()
  ret void

free_block:                                       ; preds = %for_body
  call void @tl_tensor_free(ptr %old_val)
  br label %continue_block

continue_block:                                   ; preds = %free_block, %for_body
  call void @tl_mem_unregister(ptr %detach_res)
  store ptr %detach_res, ptr %total_loss, align 8
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx, 1
  br label %for_header
}

define void @main() {
entry:
  %epoch = alloca i64, align 16
  %epochs = alloca i64, align 16
  %lr = alloca float, align 16
  %model = alloca ptr, align 16
  %d_model = alloca i64, align 16
  %vocab_size = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 13, ptr %vocab_size, align 8
  store i64 64, ptr %d_model, align 8
  %vocab_size1 = load i64, ptr %vocab_size, align 8
  %d_model2 = load i64, ptr %d_model, align 8
  %static_call = call ptr @tl_GPT_new(i64 %vocab_size1, i64 %d_model2)
  store ptr %static_call, ptr %model, align 8
  store float 0x3F847AE140000000, ptr %lr, align 4
  store i64 100, ptr %epochs, align 8
  call void @tl_print_string(ptr @str_literal.104)
  %epochs3 = load i64, ptr %epochs, align 8
  br label %for_header

for_header:                                       ; preds = %for_body, %entry
  %for_idx = phi i64 [ 0, %entry ], [ %next_idx, %for_body ]
  %for_cond = icmp slt i64 %for_idx, %epochs3
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %epoch, align 8
  call void @tl_print_string(ptr @str_literal.105)
  %epoch4 = load i64, ptr %epoch, align 8
  call void @tl_print_i64(i64 %epoch4)
  %model5 = load ptr, ptr %model, align 8
  %lr6 = load float, ptr %lr, align 4
  %epoch7 = load i64, ptr %epoch, align 8
  call void @train_epoch(ptr %model5, float %lr6, i64 %epoch7)
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx, 1
  br label %for_header

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal.106)
  %model8 = load ptr, ptr %model, align 8
  %w = getelementptr inbounds nuw %GPT, ptr %model8, i32 0, i32 0
  %sub_ptr = load ptr, ptr %w, align 8
  %w9 = getelementptr inbounds nuw %Embedding, ptr %sub_ptr, i32 0, i32 0
  %w10 = load ptr, ptr %w9, align 8
  call void @tl_add_parameter(ptr @key_str, ptr %w10)
  %b = getelementptr inbounds nuw %GPT, ptr %model8, i32 0, i32 1
  %sub_ptr11 = load ptr, ptr %b, align 8
  %l1 = getelementptr inbounds nuw %Block, ptr %sub_ptr11, i32 0, i32 0
  %sub_ptr12 = load ptr, ptr %l1, align 8
  %w13 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr12, i32 0, i32 0
  %w14 = load ptr, ptr %w13, align 8
  call void @tl_add_parameter(ptr @key_str.107, ptr %w14)
  %b15 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr12, i32 0, i32 1
  %b16 = load ptr, ptr %b15, align 8
  call void @tl_add_parameter(ptr @key_str.108, ptr %b16)
  %a = getelementptr inbounds nuw %Block, ptr %sub_ptr11, i32 0, i32 1
  %sub_ptr17 = load ptr, ptr %a, align 8
  %a18 = getelementptr inbounds nuw %CausalSelfAttention, ptr %sub_ptr17, i32 0, i32 0
  %sub_ptr19 = load ptr, ptr %a18, align 8
  %W = getelementptr inbounds nuw %Linear, ptr %sub_ptr19, i32 0, i32 0
  %W20 = load ptr, ptr %W, align 8
  call void @tl_add_parameter(ptr @key_str.109, ptr %W20)
  %b21 = getelementptr inbounds nuw %Linear, ptr %sub_ptr19, i32 0, i32 1
  %b22 = load ptr, ptr %b21, align 8
  call void @tl_add_parameter(ptr @key_str.110, ptr %b22)
  %p = getelementptr inbounds nuw %CausalSelfAttention, ptr %sub_ptr17, i32 0, i32 1
  %sub_ptr23 = load ptr, ptr %p, align 8
  %W24 = getelementptr inbounds nuw %Linear, ptr %sub_ptr23, i32 0, i32 0
  %W25 = load ptr, ptr %W24, align 8
  call void @tl_add_parameter(ptr @key_str.111, ptr %W25)
  %b26 = getelementptr inbounds nuw %Linear, ptr %sub_ptr23, i32 0, i32 1
  %b27 = load ptr, ptr %b26, align 8
  call void @tl_add_parameter(ptr @key_str.112, ptr %b27)
  %l2 = getelementptr inbounds nuw %Block, ptr %sub_ptr11, i32 0, i32 2
  %sub_ptr28 = load ptr, ptr %l2, align 8
  %w29 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr28, i32 0, i32 0
  %w30 = load ptr, ptr %w29, align 8
  call void @tl_add_parameter(ptr @key_str.113, ptr %w30)
  %b31 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr28, i32 0, i32 1
  %b32 = load ptr, ptr %b31, align 8
  call void @tl_add_parameter(ptr @key_str.114, ptr %b32)
  %m = getelementptr inbounds nuw %Block, ptr %sub_ptr11, i32 0, i32 3
  %sub_ptr33 = load ptr, ptr %m, align 8
  %f = getelementptr inbounds nuw %MLP, ptr %sub_ptr33, i32 0, i32 0
  %sub_ptr34 = load ptr, ptr %f, align 8
  %W35 = getelementptr inbounds nuw %Linear, ptr %sub_ptr34, i32 0, i32 0
  %W36 = load ptr, ptr %W35, align 8
  call void @tl_add_parameter(ptr @key_str.115, ptr %W36)
  %b37 = getelementptr inbounds nuw %Linear, ptr %sub_ptr34, i32 0, i32 1
  %b38 = load ptr, ptr %b37, align 8
  call void @tl_add_parameter(ptr @key_str.116, ptr %b38)
  %p39 = getelementptr inbounds nuw %MLP, ptr %sub_ptr33, i32 0, i32 1
  %sub_ptr40 = load ptr, ptr %p39, align 8
  %W41 = getelementptr inbounds nuw %Linear, ptr %sub_ptr40, i32 0, i32 0
  %W42 = load ptr, ptr %W41, align 8
  call void @tl_add_parameter(ptr @key_str.117, ptr %W42)
  %b43 = getelementptr inbounds nuw %Linear, ptr %sub_ptr40, i32 0, i32 1
  %b44 = load ptr, ptr %b43, align 8
  call void @tl_add_parameter(ptr @key_str.118, ptr %b44)
  %l = getelementptr inbounds nuw %GPT, ptr %model8, i32 0, i32 2
  %sub_ptr45 = load ptr, ptr %l, align 8
  %w46 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr45, i32 0, i32 0
  %w47 = load ptr, ptr %w46, align 8
  call void @tl_add_parameter(ptr @key_str.119, ptr %w47)
  %b48 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr45, i32 0, i32 1
  %b49 = load ptr, ptr %b48, align 8
  call void @tl_add_parameter(ptr @key_str.120, ptr %b49)
  %h = getelementptr inbounds nuw %GPT, ptr %model8, i32 0, i32 3
  %sub_ptr50 = load ptr, ptr %h, align 8
  %W51 = getelementptr inbounds nuw %Linear, ptr %sub_ptr50, i32 0, i32 0
  %W52 = load ptr, ptr %W51, align 8
  call void @tl_add_parameter(ptr @key_str.121, ptr %W52)
  %b53 = getelementptr inbounds nuw %Linear, ptr %sub_ptr50, i32 0, i32 1
  %b54 = load ptr, ptr %b53, align 8
  call void @tl_add_parameter(ptr @key_str.122, ptr %b54)
  call void @tl_save_all_params(ptr @str_literal.123)
  call void @tl_mem_exit_scope()
  ret void
}
