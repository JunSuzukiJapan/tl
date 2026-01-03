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

@str_literal = private unnamed_addr constant [36 x i8] c"Initializing Model for Inference...\00", align 1
@str_literal.103 = private unnamed_addr constant [22 x i8] c"Loading Parameters...\00", align 1
@key_str = private unnamed_addr constant [4 x i8] c"w.w\00", align 1
@key_str.104 = private unnamed_addr constant [7 x i8] c"b.l1.w\00", align 1
@key_str.105 = private unnamed_addr constant [7 x i8] c"b.l1.b\00", align 1
@key_str.106 = private unnamed_addr constant [8 x i8] c"b.a.a.W\00", align 1
@key_str.107 = private unnamed_addr constant [8 x i8] c"b.a.a.b\00", align 1
@key_str.108 = private unnamed_addr constant [8 x i8] c"b.a.p.W\00", align 1
@key_str.109 = private unnamed_addr constant [8 x i8] c"b.a.p.b\00", align 1
@key_str.110 = private unnamed_addr constant [7 x i8] c"b.l2.w\00", align 1
@key_str.111 = private unnamed_addr constant [7 x i8] c"b.l2.b\00", align 1
@key_str.112 = private unnamed_addr constant [8 x i8] c"b.m.f.W\00", align 1
@key_str.113 = private unnamed_addr constant [8 x i8] c"b.m.f.b\00", align 1
@key_str.114 = private unnamed_addr constant [8 x i8] c"b.m.p.W\00", align 1
@key_str.115 = private unnamed_addr constant [8 x i8] c"b.m.p.b\00", align 1
@key_str.116 = private unnamed_addr constant [4 x i8] c"l.w\00", align 1
@key_str.117 = private unnamed_addr constant [4 x i8] c"l.b\00", align 1
@key_str.118 = private unnamed_addr constant [4 x i8] c"h.W\00", align 1
@key_str.119 = private unnamed_addr constant [4 x i8] c"h.b\00", align 1
@str_literal.120 = private unnamed_addr constant [25 x i8] c"model_2digit.safetensors\00", align 1
@str_literal.121 = private unnamed_addr constant [19 x i8] c"Parameters Loaded.\00", align 1
@str_literal.122 = private unnamed_addr constant [19 x i8] c"Parameters Loaded.\00", align 1
@str_literal.123 = private unnamed_addr constant [42 x i8] c"Running Inference verification 2-digit...\00", align 1
@str_literal.124 = private unnamed_addr constant [7 x i8] c"Input:\00", align 1
@str_literal.125 = private unnamed_addr constant [2 x i8] c"+\00", align 1
@str_literal.126 = private unnamed_addr constant [18 x i8] c"Predicted Digits:\00", align 1
@str_literal.127 = private unnamed_addr constant [33 x i8] c"Inference Verification Complete.\00", align 1

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

define ptr @tl_Linear_step(ptr %self, float %lr) {
entry:
  %scalar_shape_rhs23 = alloca i64, align 16
  %scalar_data_rhs22 = alloca float, align 16
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %gb = alloca ptr, align 16
  %gW = alloca ptr, align 16
  %s = alloca ptr, align 16
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  store ptr %self3, ptr %s, align 8
  %s4 = load ptr, ptr %s, align 8
  %ptr_W = getelementptr inbounds nuw %Linear, ptr %s4, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %W)
  store ptr %grad_res, ptr %gW, align 8
  %s5 = load ptr, ptr %s, align 8
  %ptr_b = getelementptr inbounds nuw %Linear, ptr %s5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res6 = call ptr @tl_tensor_grad(ptr %b)
  store ptr %grad_res6, ptr %gb, align 8
  %s7 = load ptr, ptr %s, align 8
  %ptr_W8 = getelementptr inbounds nuw %Linear, ptr %s7, i32 0, i32 0
  %s9 = load ptr, ptr %s, align 8
  %ptr_W10 = getelementptr inbounds nuw %Linear, ptr %s9, i32 0, i32 0
  %W11 = load ptr, ptr %ptr_W10, align 8
  %gW12 = load ptr, ptr %gW, align 8
  %lr13 = load float, ptr %lr2, align 4
  store float %lr13, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %gW12, ptr %scalar_tensor_rhs)
  %binop_res14 = call ptr @tl_tensor_sub(ptr %W11, ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res14, i1 true)
  %old_field_val = load ptr, ptr %ptr_W8, align 8
  store ptr %detach_res, ptr %ptr_W8, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  %s15 = load ptr, ptr %s, align 8
  %ptr_b16 = getelementptr inbounds nuw %Linear, ptr %s15, i32 0, i32 1
  %s17 = load ptr, ptr %s, align 8
  %ptr_b18 = getelementptr inbounds nuw %Linear, ptr %s17, i32 0, i32 1
  %b19 = load ptr, ptr %ptr_b18, align 8
  %gb20 = load ptr, ptr %gb, align 8
  %lr21 = load float, ptr %lr2, align 4
  store float %lr21, ptr %scalar_data_rhs22, align 4
  %scalar_tensor_rhs24 = call ptr @tl_tensor_new(ptr %scalar_data_rhs22, i64 0, ptr %scalar_shape_rhs23)
  %binop_res25 = call ptr @tl_tensor_mul(ptr %gb20, ptr %scalar_tensor_rhs24)
  %binop_res26 = call ptr @tl_tensor_sub(ptr %b19, ptr %binop_res25)
  %detach_res27 = call ptr @tl_tensor_detach(ptr %binop_res26, i1 true)
  %old_field_val28 = load ptr, ptr %ptr_b16, align 8
  store ptr %detach_res27, ptr %ptr_b16, align 8
  call void @tl_mem_unregister(ptr %detach_res27)
  %s29 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s29)
  %unreg_field_0 = getelementptr inbounds nuw %Linear, ptr %s29, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds nuw %Linear, ptr %s29, i32 0, i32 1
  %field_val30 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  call void @tl_mem_exit_scope()
  ret ptr %s29
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

define ptr @tl_Embedding_step(ptr %self, float %lr) {
entry:
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %g = alloca ptr, align 16
  %s = alloca ptr, align 16
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  store ptr %self3, ptr %s, align 8
  %s4 = load ptr, ptr %s, align 8
  %ptr_w = getelementptr inbounds nuw %Embedding, ptr %s4, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %w)
  store ptr %grad_res, ptr %g, align 8
  %s5 = load ptr, ptr %s, align 8
  %ptr_w6 = getelementptr inbounds nuw %Embedding, ptr %s5, i32 0, i32 0
  %s7 = load ptr, ptr %s, align 8
  %ptr_w8 = getelementptr inbounds nuw %Embedding, ptr %s7, i32 0, i32 0
  %w9 = load ptr, ptr %ptr_w8, align 8
  %g10 = load ptr, ptr %g, align 8
  %lr11 = load float, ptr %lr2, align 4
  store float %lr11, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %g10, ptr %scalar_tensor_rhs)
  %binop_res12 = call ptr @tl_tensor_sub(ptr %w9, ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res12, i1 true)
  %old_field_val = load ptr, ptr %ptr_w6, align 8
  store ptr %detach_res, ptr %ptr_w6, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  %s13 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s13)
  %unreg_field_0 = getelementptr inbounds nuw %Embedding, ptr %s13, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  call void @tl_mem_exit_scope()
  ret ptr %s13
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

define ptr @tl_LayerNorm_step(ptr %self, float %lr) {
entry:
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %gb = alloca ptr, align 16
  %s = alloca ptr, align 16
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  store ptr %self3, ptr %s, align 8
  %s4 = load ptr, ptr %s, align 8
  %ptr_b = getelementptr inbounds nuw %LayerNorm, ptr %s4, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %b)
  store ptr %grad_res, ptr %gb, align 8
  %s5 = load ptr, ptr %s, align 8
  %ptr_b6 = getelementptr inbounds nuw %LayerNorm, ptr %s5, i32 0, i32 1
  %s7 = load ptr, ptr %s, align 8
  %ptr_b8 = getelementptr inbounds nuw %LayerNorm, ptr %s7, i32 0, i32 1
  %b9 = load ptr, ptr %ptr_b8, align 8
  %gb10 = load ptr, ptr %gb, align 8
  %lr11 = load float, ptr %lr2, align 4
  store float %lr11, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %gb10, ptr %scalar_tensor_rhs)
  %binop_res12 = call ptr @tl_tensor_sub(ptr %b9, ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res12, i1 true)
  %old_field_val = load ptr, ptr %ptr_b6, align 8
  store ptr %detach_res, ptr %ptr_b6, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  %s13 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s13)
  %unreg_field_0 = getelementptr inbounds nuw %LayerNorm, ptr %s13, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds nuw %LayerNorm, ptr %s13, i32 0, i32 1
  %field_val14 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val14)
  call void @tl_mem_exit_scope()
  ret ptr %s13
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
  call void @tl_mem_unregister(ptr %call_method14)
  call void @tl_mem_exit_scope()
  ret ptr %call_method14
}

define ptr @tl_CausalSelfAttention_step(ptr %self, float %lr) {
entry:
  %s = alloca ptr, align 16
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_unregister(ptr %call_method)
  %s8 = load ptr, ptr %s, align 8
  %ptr_p = getelementptr inbounds nuw %CausalSelfAttention, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_p10 = getelementptr inbounds nuw %CausalSelfAttention, ptr %s9, i32 0, i32 1
  %p = load ptr, ptr %ptr_p10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_Linear_step(ptr %p, float %lr11)
  store ptr %call_method12, ptr %ptr_p, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s13 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s13)
  %unreg_field_0 = getelementptr inbounds nuw %CausalSelfAttention, ptr %s13, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_014 = getelementptr inbounds nuw %Linear, ptr %field_val, i32 0, i32 0
  %field_val15 = load ptr, ptr %unreg_field_014, align 8
  call void @tl_mem_unregister(ptr %field_val15)
  %unreg_field_1 = getelementptr inbounds nuw %Linear, ptr %field_val, i32 0, i32 1
  %field_val16 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val16)
  %unreg_field_117 = getelementptr inbounds nuw %CausalSelfAttention, ptr %s13, i32 0, i32 1
  %field_val18 = load ptr, ptr %unreg_field_117, align 8
  call void @tl_mem_unregister(ptr %field_val18)
  %unreg_field_019 = getelementptr inbounds nuw %Linear, ptr %field_val18, i32 0, i32 0
  %field_val20 = load ptr, ptr %unreg_field_019, align 8
  call void @tl_mem_unregister(ptr %field_val20)
  %unreg_field_121 = getelementptr inbounds nuw %Linear, ptr %field_val18, i32 0, i32 1
  %field_val22 = load ptr, ptr %unreg_field_121, align 8
  call void @tl_mem_unregister(ptr %field_val22)
  call void @tl_mem_exit_scope()
  ret ptr %s13
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
  %relu_res = call ptr @tl_tensor_relu(ptr %call_method)
  %call_method6 = call ptr @tl_Linear_forward(ptr %p, ptr %relu_res)
  call void @tl_mem_unregister(ptr %call_method6)
  call void @tl_mem_exit_scope()
  ret ptr %call_method6
}

define ptr @tl_MLP_step(ptr %self, float %lr) {
entry:
  %s = alloca ptr, align 16
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_unregister(ptr %call_method)
  %s8 = load ptr, ptr %s, align 8
  %ptr_p = getelementptr inbounds nuw %MLP, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_p10 = getelementptr inbounds nuw %MLP, ptr %s9, i32 0, i32 1
  %p = load ptr, ptr %ptr_p10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_Linear_step(ptr %p, float %lr11)
  store ptr %call_method12, ptr %ptr_p, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s13 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s13)
  %unreg_field_0 = getelementptr inbounds nuw %MLP, ptr %s13, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_014 = getelementptr inbounds nuw %Linear, ptr %field_val, i32 0, i32 0
  %field_val15 = load ptr, ptr %unreg_field_014, align 8
  call void @tl_mem_unregister(ptr %field_val15)
  %unreg_field_1 = getelementptr inbounds nuw %Linear, ptr %field_val, i32 0, i32 1
  %field_val16 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val16)
  %unreg_field_117 = getelementptr inbounds nuw %MLP, ptr %s13, i32 0, i32 1
  %field_val18 = load ptr, ptr %unreg_field_117, align 8
  call void @tl_mem_unregister(ptr %field_val18)
  %unreg_field_019 = getelementptr inbounds nuw %Linear, ptr %field_val18, i32 0, i32 0
  %field_val20 = load ptr, ptr %unreg_field_019, align 8
  call void @tl_mem_unregister(ptr %field_val20)
  %unreg_field_121 = getelementptr inbounds nuw %Linear, ptr %field_val18, i32 0, i32 1
  %field_val22 = load ptr, ptr %unreg_field_121, align 8
  call void @tl_mem_unregister(ptr %field_val22)
  call void @tl_mem_exit_scope()
  ret ptr %s13
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
  %call_method7 = call ptr @tl_CausalSelfAttention_forward(ptr %a, ptr %call_method)
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
  %call_method14 = call ptr @tl_MLP_forward(ptr %m, ptr %call_method13)
  %binop_res15 = call ptr @tl_tensor_add(ptr %x9, ptr %call_method14)
  call void @tl_mem_unregister(ptr %binop_res15)
  call void @tl_mem_exit_scope()
  ret ptr %binop_res15
}

define ptr @tl_Block_step(ptr %self, float %lr) {
entry:
  %s = alloca ptr, align 16
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_unregister(ptr %call_method)
  %s8 = load ptr, ptr %s, align 8
  %ptr_a = getelementptr inbounds nuw %Block, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_a10 = getelementptr inbounds nuw %Block, ptr %s9, i32 0, i32 1
  %a = load ptr, ptr %ptr_a10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_CausalSelfAttention_step(ptr %a, float %lr11)
  store ptr %call_method12, ptr %ptr_a, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s13 = load ptr, ptr %s, align 8
  %ptr_l2 = getelementptr inbounds nuw %Block, ptr %s13, i32 0, i32 2
  %s14 = load ptr, ptr %s, align 8
  %ptr_l215 = getelementptr inbounds nuw %Block, ptr %s14, i32 0, i32 2
  %l2 = load ptr, ptr %ptr_l215, align 8
  %lr16 = load float, ptr %lr2, align 4
  %call_method17 = call ptr @tl_LayerNorm_step(ptr %l2, float %lr16)
  store ptr %call_method17, ptr %ptr_l2, align 8
  call void @tl_mem_unregister(ptr %call_method17)
  %s18 = load ptr, ptr %s, align 8
  %ptr_m = getelementptr inbounds nuw %Block, ptr %s18, i32 0, i32 3
  %s19 = load ptr, ptr %s, align 8
  %ptr_m20 = getelementptr inbounds nuw %Block, ptr %s19, i32 0, i32 3
  %m = load ptr, ptr %ptr_m20, align 8
  %lr21 = load float, ptr %lr2, align 4
  %call_method22 = call ptr @tl_MLP_step(ptr %m, float %lr21)
  store ptr %call_method22, ptr %ptr_m, align 8
  call void @tl_mem_unregister(ptr %call_method22)
  %s23 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s23)
  %unreg_field_0 = getelementptr inbounds nuw %Block, ptr %s23, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_024 = getelementptr inbounds nuw %LayerNorm, ptr %field_val, i32 0, i32 0
  %field_val25 = load ptr, ptr %unreg_field_024, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_1 = getelementptr inbounds nuw %LayerNorm, ptr %field_val, i32 0, i32 1
  %field_val26 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  %unreg_field_127 = getelementptr inbounds nuw %Block, ptr %s23, i32 0, i32 1
  %field_val28 = load ptr, ptr %unreg_field_127, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_029 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val28, i32 0, i32 0
  %field_val30 = load ptr, ptr %unreg_field_029, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  %unreg_field_031 = getelementptr inbounds nuw %Linear, ptr %field_val30, i32 0, i32 0
  %field_val32 = load ptr, ptr %unreg_field_031, align 8
  call void @tl_mem_unregister(ptr %field_val32)
  %unreg_field_133 = getelementptr inbounds nuw %Linear, ptr %field_val30, i32 0, i32 1
  %field_val34 = load ptr, ptr %unreg_field_133, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  %unreg_field_135 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val28, i32 0, i32 1
  %field_val36 = load ptr, ptr %unreg_field_135, align 8
  call void @tl_mem_unregister(ptr %field_val36)
  %unreg_field_037 = getelementptr inbounds nuw %Linear, ptr %field_val36, i32 0, i32 0
  %field_val38 = load ptr, ptr %unreg_field_037, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_139 = getelementptr inbounds nuw %Linear, ptr %field_val36, i32 0, i32 1
  %field_val40 = load ptr, ptr %unreg_field_139, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_2 = getelementptr inbounds nuw %Block, ptr %s23, i32 0, i32 2
  %field_val41 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val41)
  %unreg_field_042 = getelementptr inbounds nuw %LayerNorm, ptr %field_val41, i32 0, i32 0
  %field_val43 = load ptr, ptr %unreg_field_042, align 8
  call void @tl_mem_unregister(ptr %field_val43)
  %unreg_field_144 = getelementptr inbounds nuw %LayerNorm, ptr %field_val41, i32 0, i32 1
  %field_val45 = load ptr, ptr %unreg_field_144, align 8
  call void @tl_mem_unregister(ptr %field_val45)
  %unreg_field_3 = getelementptr inbounds nuw %Block, ptr %s23, i32 0, i32 3
  %field_val46 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val46)
  %unreg_field_047 = getelementptr inbounds nuw %MLP, ptr %field_val46, i32 0, i32 0
  %field_val48 = load ptr, ptr %unreg_field_047, align 8
  call void @tl_mem_unregister(ptr %field_val48)
  %unreg_field_049 = getelementptr inbounds nuw %Linear, ptr %field_val48, i32 0, i32 0
  %field_val50 = load ptr, ptr %unreg_field_049, align 8
  call void @tl_mem_unregister(ptr %field_val50)
  %unreg_field_151 = getelementptr inbounds nuw %Linear, ptr %field_val48, i32 0, i32 1
  %field_val52 = load ptr, ptr %unreg_field_151, align 8
  call void @tl_mem_unregister(ptr %field_val52)
  %unreg_field_153 = getelementptr inbounds nuw %MLP, ptr %field_val46, i32 0, i32 1
  %field_val54 = load ptr, ptr %unreg_field_153, align 8
  call void @tl_mem_unregister(ptr %field_val54)
  %unreg_field_055 = getelementptr inbounds nuw %Linear, ptr %field_val54, i32 0, i32 0
  %field_val56 = load ptr, ptr %unreg_field_055, align 8
  call void @tl_mem_unregister(ptr %field_val56)
  %unreg_field_157 = getelementptr inbounds nuw %Linear, ptr %field_val54, i32 0, i32 1
  %field_val58 = load ptr, ptr %unreg_field_157, align 8
  call void @tl_mem_unregister(ptr %field_val58)
  call void @tl_mem_exit_scope()
  ret ptr %s23
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
  %call_method8 = call ptr @tl_Block_forward(ptr %b, ptr %call_method)
  %call_method9 = call ptr @tl_LayerNorm_forward(ptr %l, ptr %call_method8)
  %call_method10 = call ptr @tl_Linear_forward(ptr %h, ptr %call_method9)
  call void @tl_mem_unregister(ptr %call_method10)
  call void @tl_mem_exit_scope()
  ret ptr %call_method10
}

define ptr @tl_GPT_step(ptr %self, float %lr) {
entry:
  %s = alloca ptr, align 16
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
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
  call void @tl_mem_unregister(ptr %call_method)
  %s8 = load ptr, ptr %s, align 8
  %ptr_b = getelementptr inbounds nuw %GPT, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_b10 = getelementptr inbounds nuw %GPT, ptr %s9, i32 0, i32 1
  %b = load ptr, ptr %ptr_b10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_Block_step(ptr %b, float %lr11)
  store ptr %call_method12, ptr %ptr_b, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s13 = load ptr, ptr %s, align 8
  %ptr_l = getelementptr inbounds nuw %GPT, ptr %s13, i32 0, i32 2
  %s14 = load ptr, ptr %s, align 8
  %ptr_l15 = getelementptr inbounds nuw %GPT, ptr %s14, i32 0, i32 2
  %l = load ptr, ptr %ptr_l15, align 8
  %lr16 = load float, ptr %lr2, align 4
  %call_method17 = call ptr @tl_LayerNorm_step(ptr %l, float %lr16)
  store ptr %call_method17, ptr %ptr_l, align 8
  call void @tl_mem_unregister(ptr %call_method17)
  %s18 = load ptr, ptr %s, align 8
  %ptr_h = getelementptr inbounds nuw %GPT, ptr %s18, i32 0, i32 3
  %s19 = load ptr, ptr %s, align 8
  %ptr_h20 = getelementptr inbounds nuw %GPT, ptr %s19, i32 0, i32 3
  %h = load ptr, ptr %ptr_h20, align 8
  %lr21 = load float, ptr %lr2, align 4
  %call_method22 = call ptr @tl_Linear_step(ptr %h, float %lr21)
  store ptr %call_method22, ptr %ptr_h, align 8
  call void @tl_mem_unregister(ptr %call_method22)
  %s23 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s23)
  %unreg_field_0 = getelementptr inbounds nuw %GPT, ptr %s23, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_024 = getelementptr inbounds nuw %Embedding, ptr %field_val, i32 0, i32 0
  %field_val25 = load ptr, ptr %unreg_field_024, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_1 = getelementptr inbounds nuw %GPT, ptr %s23, i32 0, i32 1
  %field_val26 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  %unreg_field_027 = getelementptr inbounds nuw %Block, ptr %field_val26, i32 0, i32 0
  %field_val28 = load ptr, ptr %unreg_field_027, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_029 = getelementptr inbounds nuw %LayerNorm, ptr %field_val28, i32 0, i32 0
  %field_val30 = load ptr, ptr %unreg_field_029, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  %unreg_field_131 = getelementptr inbounds nuw %LayerNorm, ptr %field_val28, i32 0, i32 1
  %field_val32 = load ptr, ptr %unreg_field_131, align 8
  call void @tl_mem_unregister(ptr %field_val32)
  %unreg_field_133 = getelementptr inbounds nuw %Block, ptr %field_val26, i32 0, i32 1
  %field_val34 = load ptr, ptr %unreg_field_133, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  %unreg_field_035 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val34, i32 0, i32 0
  %field_val36 = load ptr, ptr %unreg_field_035, align 8
  call void @tl_mem_unregister(ptr %field_val36)
  %unreg_field_037 = getelementptr inbounds nuw %Linear, ptr %field_val36, i32 0, i32 0
  %field_val38 = load ptr, ptr %unreg_field_037, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_139 = getelementptr inbounds nuw %Linear, ptr %field_val36, i32 0, i32 1
  %field_val40 = load ptr, ptr %unreg_field_139, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_141 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val34, i32 0, i32 1
  %field_val42 = load ptr, ptr %unreg_field_141, align 8
  call void @tl_mem_unregister(ptr %field_val42)
  %unreg_field_043 = getelementptr inbounds nuw %Linear, ptr %field_val42, i32 0, i32 0
  %field_val44 = load ptr, ptr %unreg_field_043, align 8
  call void @tl_mem_unregister(ptr %field_val44)
  %unreg_field_145 = getelementptr inbounds nuw %Linear, ptr %field_val42, i32 0, i32 1
  %field_val46 = load ptr, ptr %unreg_field_145, align 8
  call void @tl_mem_unregister(ptr %field_val46)
  %unreg_field_2 = getelementptr inbounds nuw %Block, ptr %field_val26, i32 0, i32 2
  %field_val47 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val47)
  %unreg_field_048 = getelementptr inbounds nuw %LayerNorm, ptr %field_val47, i32 0, i32 0
  %field_val49 = load ptr, ptr %unreg_field_048, align 8
  call void @tl_mem_unregister(ptr %field_val49)
  %unreg_field_150 = getelementptr inbounds nuw %LayerNorm, ptr %field_val47, i32 0, i32 1
  %field_val51 = load ptr, ptr %unreg_field_150, align 8
  call void @tl_mem_unregister(ptr %field_val51)
  %unreg_field_3 = getelementptr inbounds nuw %Block, ptr %field_val26, i32 0, i32 3
  %field_val52 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val52)
  %unreg_field_053 = getelementptr inbounds nuw %MLP, ptr %field_val52, i32 0, i32 0
  %field_val54 = load ptr, ptr %unreg_field_053, align 8
  call void @tl_mem_unregister(ptr %field_val54)
  %unreg_field_055 = getelementptr inbounds nuw %Linear, ptr %field_val54, i32 0, i32 0
  %field_val56 = load ptr, ptr %unreg_field_055, align 8
  call void @tl_mem_unregister(ptr %field_val56)
  %unreg_field_157 = getelementptr inbounds nuw %Linear, ptr %field_val54, i32 0, i32 1
  %field_val58 = load ptr, ptr %unreg_field_157, align 8
  call void @tl_mem_unregister(ptr %field_val58)
  %unreg_field_159 = getelementptr inbounds nuw %MLP, ptr %field_val52, i32 0, i32 1
  %field_val60 = load ptr, ptr %unreg_field_159, align 8
  call void @tl_mem_unregister(ptr %field_val60)
  %unreg_field_061 = getelementptr inbounds nuw %Linear, ptr %field_val60, i32 0, i32 0
  %field_val62 = load ptr, ptr %unreg_field_061, align 8
  call void @tl_mem_unregister(ptr %field_val62)
  %unreg_field_163 = getelementptr inbounds nuw %Linear, ptr %field_val60, i32 0, i32 1
  %field_val64 = load ptr, ptr %unreg_field_163, align 8
  call void @tl_mem_unregister(ptr %field_val64)
  %unreg_field_265 = getelementptr inbounds nuw %GPT, ptr %s23, i32 0, i32 2
  %field_val66 = load ptr, ptr %unreg_field_265, align 8
  call void @tl_mem_unregister(ptr %field_val66)
  %unreg_field_067 = getelementptr inbounds nuw %LayerNorm, ptr %field_val66, i32 0, i32 0
  %field_val68 = load ptr, ptr %unreg_field_067, align 8
  call void @tl_mem_unregister(ptr %field_val68)
  %unreg_field_169 = getelementptr inbounds nuw %LayerNorm, ptr %field_val66, i32 0, i32 1
  %field_val70 = load ptr, ptr %unreg_field_169, align 8
  call void @tl_mem_unregister(ptr %field_val70)
  %unreg_field_371 = getelementptr inbounds nuw %GPT, ptr %s23, i32 0, i32 3
  %field_val72 = load ptr, ptr %unreg_field_371, align 8
  call void @tl_mem_unregister(ptr %field_val72)
  %unreg_field_073 = getelementptr inbounds nuw %Linear, ptr %field_val72, i32 0, i32 0
  %field_val74 = load ptr, ptr %unreg_field_073, align 8
  call void @tl_mem_unregister(ptr %field_val74)
  %unreg_field_175 = getelementptr inbounds nuw %Linear, ptr %field_val72, i32 0, i32 1
  %field_val76 = load ptr, ptr %unreg_field_175, align 8
  call void @tl_mem_unregister(ptr %field_val76)
  call void @tl_mem_exit_scope()
  ret ptr %s23
}

define i64 @get_memory() {
entry:
  call void @tl_mem_enter_scope()
  %call_tmp = call i64 @tl_get_memory_mb()
  call void @tl_mem_exit_scope()
  ret i64 %call_tmp
}

define void @main() {
entry:
  %val316 = alloca float, align 16
  %k312 = alloca i64, align 16
  %max_val306 = alloca float, align 16
  %pred3 = alloca i64, align 16
  %next_logits3 = alloca ptr, align 16
  %logits3302 = alloca ptr, align 16
  %logits3 = alloca ptr, align 16
  %input3 = alloca ptr, align 16
  %data3 = alloca ptr, align 16
  %tensor_shape_arr283 = alloca [1 x i64], align 8
  %val_pred2 = alloca float, align 16
  %scalar_shape257 = alloca i64, align 16
  %scalar_data256 = alloca float, align 16
  %scalar_shape254 = alloca i64, align 16
  %scalar_data252 = alloca float, align 16
  %val240 = alloca float, align 16
  %k236 = alloca i64, align 16
  %max_val230 = alloca float, align 16
  %pred2 = alloca i64, align 16
  %next_logits2 = alloca ptr, align 16
  %logits2226 = alloca ptr, align 16
  %logits2 = alloca ptr, align 16
  %input2 = alloca ptr, align 16
  %data2 = alloca ptr, align 16
  %tensor_shape_arr207 = alloca [1 x i64], align 8
  %val_pred1 = alloca float, align 16
  %scalar_shape181 = alloca i64, align 16
  %scalar_data180 = alloca float, align 16
  %scalar_shape178 = alloca i64, align 16
  %scalar_data176 = alloca float, align 16
  %val = alloca float, align 16
  %k = alloca i64, align 16
  %max_val = alloca float, align 16
  %pred1 = alloca i64, align 16
  %next_logits1 = alloca ptr, align 16
  %logits1157 = alloca ptr, align 16
  %logits1 = alloca ptr, align 16
  %input1 = alloca ptr, align 16
  %data1 = alloca ptr, align 16
  %tensor_shape_arr = alloca [1 x i64], align 8
  %pos = alloca i64, align 16
  %x11 = alloca float, align 16
  %x10 = alloca float, align 16
  %x9 = alloca float, align 16
  %x8 = alloca float, align 16
  %x7 = alloca float, align 16
  %x6 = alloca float, align 16
  %x5 = alloca float, align 16
  %x4 = alloca float, align 16
  %x3 = alloca float, align 16
  %x2 = alloca float, align 16
  %x1 = alloca float, align 16
  %x0 = alloca float, align 16
  %val_pad = alloca float, align 16
  %val_eq = alloca float, align 16
  %val_plus = alloca float, align 16
  %j_d2 = alloca float, align 16
  %scalar_shape103 = alloca i64, align 16
  %scalar_data102 = alloca float, align 16
  %scalar_shape100 = alloca i64, align 16
  %scalar_data98 = alloca float, align 16
  %j_d1 = alloca float, align 16
  %scalar_shape89 = alloca i64, align 16
  %scalar_data88 = alloca float, align 16
  %scalar_shape86 = alloca i64, align 16
  %scalar_data84 = alloca float, align 16
  %i_d2 = alloca float, align 16
  %scalar_shape78 = alloca i64, align 16
  %scalar_data77 = alloca float, align 16
  %scalar_shape75 = alloca i64, align 16
  %scalar_data73 = alloca float, align 16
  %i_d1 = alloca float, align 16
  %scalar_shape68 = alloca i64, align 16
  %scalar_data67 = alloca float, align 16
  %scalar_shape = alloca i64, align 16
  %scalar_data = alloca float, align 16
  %j = alloca i64, align 16
  %i = alloca i64, align 16
  %t = alloca i64, align 16
  %model = alloca ptr, align 16
  %block_size = alloca i64, align 16
  %d_model = alloca i64, align 16
  %vocab_size = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 13, ptr %vocab_size, align 8
  store i64 64, ptr %d_model, align 8
  store i64 12, ptr %block_size, align 8
  call void @tl_print_string(ptr @str_literal)
  %vocab_size1 = load i64, ptr %vocab_size, align 8
  %d_model2 = load i64, ptr %d_model, align 8
  %static_call = call ptr @tl_GPT_new(i64 %vocab_size1, i64 %d_model2)
  store ptr %static_call, ptr %model, align 8
  call void @tl_print_string(ptr @str_literal.103)
  %model3 = load ptr, ptr %model, align 8
  %w = getelementptr inbounds nuw %GPT, ptr %model3, i32 0, i32 0
  %sub_ptr = load ptr, ptr %w, align 8
  %w4 = getelementptr inbounds nuw %Embedding, ptr %sub_ptr, i32 0, i32 0
  %w5 = load ptr, ptr %w4, align 8
  call void @tl_add_parameter(ptr @key_str, ptr %w5)
  %b = getelementptr inbounds nuw %GPT, ptr %model3, i32 0, i32 1
  %sub_ptr6 = load ptr, ptr %b, align 8
  %l1 = getelementptr inbounds nuw %Block, ptr %sub_ptr6, i32 0, i32 0
  %sub_ptr7 = load ptr, ptr %l1, align 8
  %w8 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr7, i32 0, i32 0
  %w9 = load ptr, ptr %w8, align 8
  call void @tl_add_parameter(ptr @key_str.104, ptr %w9)
  %b10 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr7, i32 0, i32 1
  %b11 = load ptr, ptr %b10, align 8
  call void @tl_add_parameter(ptr @key_str.105, ptr %b11)
  %a = getelementptr inbounds nuw %Block, ptr %sub_ptr6, i32 0, i32 1
  %sub_ptr12 = load ptr, ptr %a, align 8
  %a13 = getelementptr inbounds nuw %CausalSelfAttention, ptr %sub_ptr12, i32 0, i32 0
  %sub_ptr14 = load ptr, ptr %a13, align 8
  %W = getelementptr inbounds nuw %Linear, ptr %sub_ptr14, i32 0, i32 0
  %W15 = load ptr, ptr %W, align 8
  call void @tl_add_parameter(ptr @key_str.106, ptr %W15)
  %b16 = getelementptr inbounds nuw %Linear, ptr %sub_ptr14, i32 0, i32 1
  %b17 = load ptr, ptr %b16, align 8
  call void @tl_add_parameter(ptr @key_str.107, ptr %b17)
  %p = getelementptr inbounds nuw %CausalSelfAttention, ptr %sub_ptr12, i32 0, i32 1
  %sub_ptr18 = load ptr, ptr %p, align 8
  %W19 = getelementptr inbounds nuw %Linear, ptr %sub_ptr18, i32 0, i32 0
  %W20 = load ptr, ptr %W19, align 8
  call void @tl_add_parameter(ptr @key_str.108, ptr %W20)
  %b21 = getelementptr inbounds nuw %Linear, ptr %sub_ptr18, i32 0, i32 1
  %b22 = load ptr, ptr %b21, align 8
  call void @tl_add_parameter(ptr @key_str.109, ptr %b22)
  %l2 = getelementptr inbounds nuw %Block, ptr %sub_ptr6, i32 0, i32 2
  %sub_ptr23 = load ptr, ptr %l2, align 8
  %w24 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr23, i32 0, i32 0
  %w25 = load ptr, ptr %w24, align 8
  call void @tl_add_parameter(ptr @key_str.110, ptr %w25)
  %b26 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr23, i32 0, i32 1
  %b27 = load ptr, ptr %b26, align 8
  call void @tl_add_parameter(ptr @key_str.111, ptr %b27)
  %m = getelementptr inbounds nuw %Block, ptr %sub_ptr6, i32 0, i32 3
  %sub_ptr28 = load ptr, ptr %m, align 8
  %f = getelementptr inbounds nuw %MLP, ptr %sub_ptr28, i32 0, i32 0
  %sub_ptr29 = load ptr, ptr %f, align 8
  %W30 = getelementptr inbounds nuw %Linear, ptr %sub_ptr29, i32 0, i32 0
  %W31 = load ptr, ptr %W30, align 8
  call void @tl_add_parameter(ptr @key_str.112, ptr %W31)
  %b32 = getelementptr inbounds nuw %Linear, ptr %sub_ptr29, i32 0, i32 1
  %b33 = load ptr, ptr %b32, align 8
  call void @tl_add_parameter(ptr @key_str.113, ptr %b33)
  %p34 = getelementptr inbounds nuw %MLP, ptr %sub_ptr28, i32 0, i32 1
  %sub_ptr35 = load ptr, ptr %p34, align 8
  %W36 = getelementptr inbounds nuw %Linear, ptr %sub_ptr35, i32 0, i32 0
  %W37 = load ptr, ptr %W36, align 8
  call void @tl_add_parameter(ptr @key_str.114, ptr %W37)
  %b38 = getelementptr inbounds nuw %Linear, ptr %sub_ptr35, i32 0, i32 1
  %b39 = load ptr, ptr %b38, align 8
  call void @tl_add_parameter(ptr @key_str.115, ptr %b39)
  %l = getelementptr inbounds nuw %GPT, ptr %model3, i32 0, i32 2
  %sub_ptr40 = load ptr, ptr %l, align 8
  %w41 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr40, i32 0, i32 0
  %w42 = load ptr, ptr %w41, align 8
  call void @tl_add_parameter(ptr @key_str.116, ptr %w42)
  %b43 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr40, i32 0, i32 1
  %b44 = load ptr, ptr %b43, align 8
  call void @tl_add_parameter(ptr @key_str.117, ptr %b44)
  %h = getelementptr inbounds nuw %GPT, ptr %model3, i32 0, i32 3
  %sub_ptr45 = load ptr, ptr %h, align 8
  %W46 = getelementptr inbounds nuw %Linear, ptr %sub_ptr45, i32 0, i32 0
  %W47 = load ptr, ptr %W46, align 8
  call void @tl_add_parameter(ptr @key_str.118, ptr %W47)
  %b48 = getelementptr inbounds nuw %Linear, ptr %sub_ptr45, i32 0, i32 1
  %b49 = load ptr, ptr %b48, align 8
  call void @tl_add_parameter(ptr @key_str.119, ptr %b49)
  call void @tl_load_all_params(ptr @str_literal.120)
  call void @tl_print_string(ptr @str_literal.121)
  call void @tl_print_string(ptr @str_literal.122)
  call void @tl_print_string(ptr @str_literal.123)
  br label %for_header

for_header:                                       ; preds = %for_end309, %entry
  %for_idx = phi i64 [ 0, %entry ], [ %next_idx327, %for_end309 ]
  %for_cond = icmp slt i64 %for_idx, 4
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %t, align 8
  store i64 0, ptr %i, align 8
  store i64 0, ptr %j, align 8
  %t50 = load i64, ptr %t, align 8
  %eqtmp = icmp eq i64 %t50, 0
  br i1 %eqtmp, label %then, label %else

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal.127)
  call void @tl_mem_exit_scope()
  ret void

then:                                             ; preds = %for_body
  call void @tl_mem_enter_scope()
  store i64 12, ptr %i, align 8
  store i64 34, ptr %j, align 8
  call void @tl_mem_exit_scope()
  br label %merge

else:                                             ; preds = %for_body
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else, %then
  %t51 = load i64, ptr %t, align 8
  %eqtmp52 = icmp eq i64 %t51, 1
  br i1 %eqtmp52, label %then53, label %else54

then53:                                           ; preds = %merge
  call void @tl_mem_enter_scope()
  store i64 99, ptr %i, align 8
  store i64 1, ptr %j, align 8
  call void @tl_mem_exit_scope()
  br label %merge55

else54:                                           ; preds = %merge
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge55

merge55:                                          ; preds = %else54, %then53
  %t56 = load i64, ptr %t, align 8
  %eqtmp57 = icmp eq i64 %t56, 2
  br i1 %eqtmp57, label %then58, label %else59

then58:                                           ; preds = %merge55
  call void @tl_mem_enter_scope()
  store i64 5, ptr %i, align 8
  store i64 5, ptr %j, align 8
  call void @tl_mem_exit_scope()
  br label %merge60

else59:                                           ; preds = %merge55
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge60

merge60:                                          ; preds = %else59, %then58
  %t61 = load i64, ptr %t, align 8
  %eqtmp62 = icmp eq i64 %t61, 3
  br i1 %eqtmp62, label %then63, label %else64

then63:                                           ; preds = %merge60
  call void @tl_mem_enter_scope()
  store i64 88, ptr %i, align 8
  store i64 99, ptr %j, align 8
  call void @tl_mem_exit_scope()
  br label %merge65

else64:                                           ; preds = %merge60
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge65

merge65:                                          ; preds = %else64, %then63
  %i66 = load i64, ptr %i, align 8
  %divtmp = sdiv i64 %i66, 10
  %cast_i64_f32 = sitofp i64 %divtmp to float
  store float %cast_i64_f32, ptr %scalar_data, align 4
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  store float 1.000000e+00, ptr %scalar_data67, align 4
  %scalar_tensor69 = call ptr @tl_tensor_new(ptr %scalar_data67, i64 0, ptr %scalar_shape68)
  %pow_res = call ptr @tl_tensor_pow(ptr %scalar_tensor, ptr %scalar_tensor69)
  %get_res = call float @tl_tensor_get(ptr %pow_res, i64 0)
  store float %get_res, ptr %i_d1, align 4
  %i70 = load i64, ptr %i, align 8
  %i71 = load i64, ptr %i, align 8
  %divtmp72 = sdiv i64 %i71, 10
  %multmp = mul i64 %divtmp72, 10
  %subtmp = sub i64 %i70, %multmp
  %cast_i64_f3274 = sitofp i64 %subtmp to float
  store float %cast_i64_f3274, ptr %scalar_data73, align 4
  %scalar_tensor76 = call ptr @tl_tensor_new(ptr %scalar_data73, i64 0, ptr %scalar_shape75)
  store float 1.000000e+00, ptr %scalar_data77, align 4
  %scalar_tensor79 = call ptr @tl_tensor_new(ptr %scalar_data77, i64 0, ptr %scalar_shape78)
  %pow_res80 = call ptr @tl_tensor_pow(ptr %scalar_tensor76, ptr %scalar_tensor79)
  %get_res81 = call float @tl_tensor_get(ptr %pow_res80, i64 0)
  store float %get_res81, ptr %i_d2, align 4
  %j82 = load i64, ptr %j, align 8
  %divtmp83 = sdiv i64 %j82, 10
  %cast_i64_f3285 = sitofp i64 %divtmp83 to float
  store float %cast_i64_f3285, ptr %scalar_data84, align 4
  %scalar_tensor87 = call ptr @tl_tensor_new(ptr %scalar_data84, i64 0, ptr %scalar_shape86)
  store float 1.000000e+00, ptr %scalar_data88, align 4
  %scalar_tensor90 = call ptr @tl_tensor_new(ptr %scalar_data88, i64 0, ptr %scalar_shape89)
  %pow_res91 = call ptr @tl_tensor_pow(ptr %scalar_tensor87, ptr %scalar_tensor90)
  %get_res92 = call float @tl_tensor_get(ptr %pow_res91, i64 0)
  store float %get_res92, ptr %j_d1, align 4
  %j93 = load i64, ptr %j, align 8
  %j94 = load i64, ptr %j, align 8
  %divtmp95 = sdiv i64 %j94, 10
  %multmp96 = mul i64 %divtmp95, 10
  %subtmp97 = sub i64 %j93, %multmp96
  %cast_i64_f3299 = sitofp i64 %subtmp97 to float
  store float %cast_i64_f3299, ptr %scalar_data98, align 4
  %scalar_tensor101 = call ptr @tl_tensor_new(ptr %scalar_data98, i64 0, ptr %scalar_shape100)
  store float 1.000000e+00, ptr %scalar_data102, align 4
  %scalar_tensor104 = call ptr @tl_tensor_new(ptr %scalar_data102, i64 0, ptr %scalar_shape103)
  %pow_res105 = call ptr @tl_tensor_pow(ptr %scalar_tensor101, ptr %scalar_tensor104)
  %get_res106 = call float @tl_tensor_get(ptr %pow_res105, i64 0)
  store float %get_res106, ptr %j_d2, align 4
  store float 1.000000e+01, ptr %val_plus, align 4
  store float 1.100000e+01, ptr %val_eq, align 4
  store float 1.200000e+01, ptr %val_pad, align 4
  %val_pad107 = load float, ptr %val_pad, align 4
  store float %val_pad107, ptr %x0, align 4
  %val_pad108 = load float, ptr %val_pad, align 4
  store float %val_pad108, ptr %x1, align 4
  %val_pad109 = load float, ptr %val_pad, align 4
  store float %val_pad109, ptr %x2, align 4
  %val_pad110 = load float, ptr %val_pad, align 4
  store float %val_pad110, ptr %x3, align 4
  %val_pad111 = load float, ptr %val_pad, align 4
  store float %val_pad111, ptr %x4, align 4
  %val_pad112 = load float, ptr %val_pad, align 4
  store float %val_pad112, ptr %x5, align 4
  %val_pad113 = load float, ptr %val_pad, align 4
  store float %val_pad113, ptr %x6, align 4
  %val_pad114 = load float, ptr %val_pad, align 4
  store float %val_pad114, ptr %x7, align 4
  %val_pad115 = load float, ptr %val_pad, align 4
  store float %val_pad115, ptr %x8, align 4
  %val_pad116 = load float, ptr %val_pad, align 4
  store float %val_pad116, ptr %x9, align 4
  %val_pad117 = load float, ptr %val_pad, align 4
  store float %val_pad117, ptr %x10, align 4
  %val_pad118 = load float, ptr %val_pad, align 4
  store float %val_pad118, ptr %x11, align 4
  store i64 0, ptr %pos, align 8
  %i_d1119 = load float, ptr %i_d1, align 4
  store float %i_d1119, ptr %x0, align 4
  %i_d2120 = load float, ptr %i_d2, align 4
  store float %i_d2120, ptr %x1, align 4
  %val_plus121 = load float, ptr %val_plus, align 4
  store float %val_plus121, ptr %x2, align 4
  %j_d1122 = load float, ptr %j_d1, align 4
  store float %j_d1122, ptr %x3, align 4
  %j_d2123 = load float, ptr %j_d2, align 4
  store float %j_d2123, ptr %x4, align 4
  %val_eq124 = load float, ptr %val_eq, align 4
  store float %val_eq124, ptr %x5, align 4
  store i64 6, ptr %pos, align 8
  call void @tl_print_string(ptr @str_literal.124)
  %i125 = load i64, ptr %i, align 8
  call void @tl_print_i64(i64 %i125)
  call void @tl_print_string(ptr @str_literal.125)
  %j126 = load i64, ptr %j, align 8
  call void @tl_print_i64(i64 %j126)
  call void @tl_print_string(ptr @str_literal.126)
  %buf_void = call ptr @calloc(i64 12, i64 4)
  %x0127 = load float, ptr %x0, align 4
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %x0127, ptr %elem_ptr, align 4
  %x1128 = load float, ptr %x1, align 4
  %elem_ptr129 = getelementptr inbounds float, ptr %buf_void, i64 1
  store float %x1128, ptr %elem_ptr129, align 4
  %x2130 = load float, ptr %x2, align 4
  %elem_ptr131 = getelementptr inbounds float, ptr %buf_void, i64 2
  store float %x2130, ptr %elem_ptr131, align 4
  %x3132 = load float, ptr %x3, align 4
  %elem_ptr133 = getelementptr inbounds float, ptr %buf_void, i64 3
  store float %x3132, ptr %elem_ptr133, align 4
  %x4134 = load float, ptr %x4, align 4
  %elem_ptr135 = getelementptr inbounds float, ptr %buf_void, i64 4
  store float %x4134, ptr %elem_ptr135, align 4
  %x5136 = load float, ptr %x5, align 4
  %elem_ptr137 = getelementptr inbounds float, ptr %buf_void, i64 5
  store float %x5136, ptr %elem_ptr137, align 4
  %val_pad138 = load float, ptr %val_pad, align 4
  %elem_ptr139 = getelementptr inbounds float, ptr %buf_void, i64 6
  store float %val_pad138, ptr %elem_ptr139, align 4
  %val_pad140 = load float, ptr %val_pad, align 4
  %elem_ptr141 = getelementptr inbounds float, ptr %buf_void, i64 7
  store float %val_pad140, ptr %elem_ptr141, align 4
  %val_pad142 = load float, ptr %val_pad, align 4
  %elem_ptr143 = getelementptr inbounds float, ptr %buf_void, i64 8
  store float %val_pad142, ptr %elem_ptr143, align 4
  %elem_ptr144 = getelementptr inbounds float, ptr %buf_void, i64 9
  store float 1.200000e+01, ptr %elem_ptr144, align 4
  %elem_ptr145 = getelementptr inbounds float, ptr %buf_void, i64 10
  store float 1.200000e+01, ptr %elem_ptr145, align 4
  %elem_ptr146 = getelementptr inbounds float, ptr %buf_void, i64 11
  store float 1.200000e+01, ptr %elem_ptr146, align 4
  %shape_ptr = getelementptr inbounds [1 x i64], ptr %tensor_shape_arr, i64 0, i64 0
  store i64 12, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %tensor_shape_arr)
  store ptr %new_tensor, ptr %data1, align 8
  %data1147 = load ptr, ptr %data1, align 8
  %dims_alloca = alloca [2 x i64], align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr, align 8
  %dim_ptr148 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr148, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %data1147, ptr %dims_ptr, i64 2)
  store ptr %reshape_dims_res, ptr %input1, align 8
  %model149 = load ptr, ptr %model, align 8
  %input1150 = load ptr, ptr %input1, align 8
  %call_method = call ptr @tl_GPT_forward(ptr %model149, ptr %input1150)
  store ptr %call_method, ptr %logits1, align 8
  %logits1151 = load ptr, ptr %logits1, align 8
  %dims_alloca152 = alloca [2 x i64], align 8
  %dim_ptr153 = getelementptr [2 x i64], ptr %dims_alloca152, i64 0, i64 0
  store i64 12, ptr %dim_ptr153, align 8
  %dim_ptr154 = getelementptr [2 x i64], ptr %dims_alloca152, i64 0, i64 1
  store i64 13, ptr %dim_ptr154, align 8
  %dims_ptr155 = getelementptr [2 x i64], ptr %dims_alloca152, i64 0, i64 0
  %reshape_dims_res156 = call ptr @tl_tensor_reshape_dims(ptr %logits1151, ptr %dims_ptr155, i64 2)
  %old_shadowed = load ptr, ptr %logits1, align 8
  call void @tl_mem_unregister(ptr %old_shadowed)
  store ptr %reshape_dims_res156, ptr %logits1157, align 8
  %logits1158 = load ptr, ptr %logits1157, align 8
  %slice_res = call ptr @tl_tensor_slice(ptr %logits1158, i64 5, i64 1)
  store ptr %slice_res, ptr %next_logits1, align 8
  store i64 0, ptr %pred1, align 8
  store float -1.000000e+06, ptr %max_val, align 4
  br label %for_header159

for_header159:                                    ; preds = %merge171, %merge65
  %for_idx162 = phi i64 [ 0, %merge65 ], [ %next_idx, %merge171 ]
  %for_cond163 = icmp slt i64 %for_idx162, 13
  br i1 %for_cond163, label %for_body160, label %for_end161

for_body160:                                      ; preds = %for_header159
  call void @tl_mem_enter_scope()
  store i64 %for_idx162, ptr %k, align 8
  %next_logits1164 = load ptr, ptr %next_logits1, align 8
  %k165 = load i64, ptr %k, align 8
  %get_res166 = call float @tl_tensor_get(ptr %next_logits1164, i64 %k165)
  store float %get_res166, ptr %val, align 4
  %val167 = load float, ptr %val, align 4
  %max_val168 = load float, ptr %max_val, align 4
  %fgttmp = fcmp ogt float %val167, %max_val168
  br i1 %fgttmp, label %then169, label %else170

for_end161:                                       ; preds = %for_header159
  %pred1174 = load i64, ptr %pred1, align 8
  call void @tl_print_i64(i64 %pred1174)
  %pred1175 = load i64, ptr %pred1, align 8
  %cast_i64_f32177 = sitofp i64 %pred1175 to float
  store float %cast_i64_f32177, ptr %scalar_data176, align 4
  %scalar_tensor179 = call ptr @tl_tensor_new(ptr %scalar_data176, i64 0, ptr %scalar_shape178)
  store float 1.000000e+00, ptr %scalar_data180, align 4
  %scalar_tensor182 = call ptr @tl_tensor_new(ptr %scalar_data180, i64 0, ptr %scalar_shape181)
  %pow_res183 = call ptr @tl_tensor_pow(ptr %scalar_tensor179, ptr %scalar_tensor182)
  %get_res184 = call float @tl_tensor_get(ptr %pow_res183, i64 0)
  store float %get_res184, ptr %val_pred1, align 4
  %buf_void185 = call ptr @calloc(i64 12, i64 4)
  %x0186 = load float, ptr %x0, align 4
  %elem_ptr187 = getelementptr inbounds float, ptr %buf_void185, i64 0
  store float %x0186, ptr %elem_ptr187, align 4
  %x1188 = load float, ptr %x1, align 4
  %elem_ptr189 = getelementptr inbounds float, ptr %buf_void185, i64 1
  store float %x1188, ptr %elem_ptr189, align 4
  %x2190 = load float, ptr %x2, align 4
  %elem_ptr191 = getelementptr inbounds float, ptr %buf_void185, i64 2
  store float %x2190, ptr %elem_ptr191, align 4
  %x3192 = load float, ptr %x3, align 4
  %elem_ptr193 = getelementptr inbounds float, ptr %buf_void185, i64 3
  store float %x3192, ptr %elem_ptr193, align 4
  %x4194 = load float, ptr %x4, align 4
  %elem_ptr195 = getelementptr inbounds float, ptr %buf_void185, i64 4
  store float %x4194, ptr %elem_ptr195, align 4
  %x5196 = load float, ptr %x5, align 4
  %elem_ptr197 = getelementptr inbounds float, ptr %buf_void185, i64 5
  store float %x5196, ptr %elem_ptr197, align 4
  %val_pred1198 = load float, ptr %val_pred1, align 4
  %elem_ptr199 = getelementptr inbounds float, ptr %buf_void185, i64 6
  store float %val_pred1198, ptr %elem_ptr199, align 4
  %val_pad200 = load float, ptr %val_pad, align 4
  %elem_ptr201 = getelementptr inbounds float, ptr %buf_void185, i64 7
  store float %val_pad200, ptr %elem_ptr201, align 4
  %val_pad202 = load float, ptr %val_pad, align 4
  %elem_ptr203 = getelementptr inbounds float, ptr %buf_void185, i64 8
  store float %val_pad202, ptr %elem_ptr203, align 4
  %elem_ptr204 = getelementptr inbounds float, ptr %buf_void185, i64 9
  store float 1.200000e+01, ptr %elem_ptr204, align 4
  %elem_ptr205 = getelementptr inbounds float, ptr %buf_void185, i64 10
  store float 1.200000e+01, ptr %elem_ptr205, align 4
  %elem_ptr206 = getelementptr inbounds float, ptr %buf_void185, i64 11
  store float 1.200000e+01, ptr %elem_ptr206, align 4
  %shape_ptr208 = getelementptr inbounds [1 x i64], ptr %tensor_shape_arr207, i64 0, i64 0
  store i64 12, ptr %shape_ptr208, align 8
  %new_tensor209 = call ptr @tl_tensor_new(ptr %buf_void185, i64 1, ptr %tensor_shape_arr207)
  store ptr %new_tensor209, ptr %data2, align 8
  %data2210 = load ptr, ptr %data2, align 8
  %dims_alloca211 = alloca [2 x i64], align 8
  %dim_ptr212 = getelementptr [2 x i64], ptr %dims_alloca211, i64 0, i64 0
  store i64 1, ptr %dim_ptr212, align 8
  %dim_ptr213 = getelementptr [2 x i64], ptr %dims_alloca211, i64 0, i64 1
  store i64 12, ptr %dim_ptr213, align 8
  %dims_ptr214 = getelementptr [2 x i64], ptr %dims_alloca211, i64 0, i64 0
  %reshape_dims_res215 = call ptr @tl_tensor_reshape_dims(ptr %data2210, ptr %dims_ptr214, i64 2)
  store ptr %reshape_dims_res215, ptr %input2, align 8
  %model216 = load ptr, ptr %model, align 8
  %input2217 = load ptr, ptr %input2, align 8
  %call_method218 = call ptr @tl_GPT_forward(ptr %model216, ptr %input2217)
  store ptr %call_method218, ptr %logits2, align 8
  %logits2219 = load ptr, ptr %logits2, align 8
  %dims_alloca220 = alloca [2 x i64], align 8
  %dim_ptr221 = getelementptr [2 x i64], ptr %dims_alloca220, i64 0, i64 0
  store i64 12, ptr %dim_ptr221, align 8
  %dim_ptr222 = getelementptr [2 x i64], ptr %dims_alloca220, i64 0, i64 1
  store i64 13, ptr %dim_ptr222, align 8
  %dims_ptr223 = getelementptr [2 x i64], ptr %dims_alloca220, i64 0, i64 0
  %reshape_dims_res224 = call ptr @tl_tensor_reshape_dims(ptr %logits2219, ptr %dims_ptr223, i64 2)
  %old_shadowed225 = load ptr, ptr %logits2, align 8
  call void @tl_mem_unregister(ptr %old_shadowed225)
  store ptr %reshape_dims_res224, ptr %logits2226, align 8
  %logits2227 = load ptr, ptr %logits2226, align 8
  %slice_res228 = call ptr @tl_tensor_slice(ptr %logits2227, i64 6, i64 1)
  store ptr %slice_res228, ptr %next_logits2, align 8
  store i64 0, ptr %pred2, align 8
  %old_shadowed229 = load ptr, ptr %max_val, align 8
  call void @tl_mem_unregister(ptr %old_shadowed229)
  store float -1.000000e+06, ptr %max_val230, align 4
  br label %for_header231

then169:                                          ; preds = %for_body160
  call void @tl_mem_enter_scope()
  %val172 = load float, ptr %val, align 4
  store float %val172, ptr %max_val, align 4
  %k173 = load i64, ptr %k, align 8
  store i64 %k173, ptr %pred1, align 8
  call void @tl_mem_exit_scope()
  br label %merge171

else170:                                          ; preds = %for_body160
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge171

merge171:                                         ; preds = %else170, %then169
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx162, 1
  br label %for_header159

for_header231:                                    ; preds = %merge246, %for_end161
  %for_idx234 = phi i64 [ 0, %for_end161 ], [ %next_idx249, %merge246 ]
  %for_cond235 = icmp slt i64 %for_idx234, 13
  br i1 %for_cond235, label %for_body232, label %for_end233

for_body232:                                      ; preds = %for_header231
  call void @tl_mem_enter_scope()
  store i64 %for_idx234, ptr %k236, align 8
  %next_logits2237 = load ptr, ptr %next_logits2, align 8
  %k238 = load i64, ptr %k236, align 8
  %get_res239 = call float @tl_tensor_get(ptr %next_logits2237, i64 %k238)
  store float %get_res239, ptr %val240, align 4
  %val241 = load float, ptr %val240, align 4
  %max_val242 = load float, ptr %max_val230, align 4
  %fgttmp243 = fcmp ogt float %val241, %max_val242
  br i1 %fgttmp243, label %then244, label %else245

for_end233:                                       ; preds = %for_header231
  %pred2250 = load i64, ptr %pred2, align 8
  call void @tl_print_i64(i64 %pred2250)
  %pred2251 = load i64, ptr %pred2, align 8
  %cast_i64_f32253 = sitofp i64 %pred2251 to float
  store float %cast_i64_f32253, ptr %scalar_data252, align 4
  %scalar_tensor255 = call ptr @tl_tensor_new(ptr %scalar_data252, i64 0, ptr %scalar_shape254)
  store float 1.000000e+00, ptr %scalar_data256, align 4
  %scalar_tensor258 = call ptr @tl_tensor_new(ptr %scalar_data256, i64 0, ptr %scalar_shape257)
  %pow_res259 = call ptr @tl_tensor_pow(ptr %scalar_tensor255, ptr %scalar_tensor258)
  %get_res260 = call float @tl_tensor_get(ptr %pow_res259, i64 0)
  store float %get_res260, ptr %val_pred2, align 4
  %buf_void261 = call ptr @calloc(i64 12, i64 4)
  %x0262 = load float, ptr %x0, align 4
  %elem_ptr263 = getelementptr inbounds float, ptr %buf_void261, i64 0
  store float %x0262, ptr %elem_ptr263, align 4
  %x1264 = load float, ptr %x1, align 4
  %elem_ptr265 = getelementptr inbounds float, ptr %buf_void261, i64 1
  store float %x1264, ptr %elem_ptr265, align 4
  %x2266 = load float, ptr %x2, align 4
  %elem_ptr267 = getelementptr inbounds float, ptr %buf_void261, i64 2
  store float %x2266, ptr %elem_ptr267, align 4
  %x3268 = load float, ptr %x3, align 4
  %elem_ptr269 = getelementptr inbounds float, ptr %buf_void261, i64 3
  store float %x3268, ptr %elem_ptr269, align 4
  %x4270 = load float, ptr %x4, align 4
  %elem_ptr271 = getelementptr inbounds float, ptr %buf_void261, i64 4
  store float %x4270, ptr %elem_ptr271, align 4
  %x5272 = load float, ptr %x5, align 4
  %elem_ptr273 = getelementptr inbounds float, ptr %buf_void261, i64 5
  store float %x5272, ptr %elem_ptr273, align 4
  %val_pred1274 = load float, ptr %val_pred1, align 4
  %elem_ptr275 = getelementptr inbounds float, ptr %buf_void261, i64 6
  store float %val_pred1274, ptr %elem_ptr275, align 4
  %val_pred2276 = load float, ptr %val_pred2, align 4
  %elem_ptr277 = getelementptr inbounds float, ptr %buf_void261, i64 7
  store float %val_pred2276, ptr %elem_ptr277, align 4
  %val_pad278 = load float, ptr %val_pad, align 4
  %elem_ptr279 = getelementptr inbounds float, ptr %buf_void261, i64 8
  store float %val_pad278, ptr %elem_ptr279, align 4
  %elem_ptr280 = getelementptr inbounds float, ptr %buf_void261, i64 9
  store float 1.200000e+01, ptr %elem_ptr280, align 4
  %elem_ptr281 = getelementptr inbounds float, ptr %buf_void261, i64 10
  store float 1.200000e+01, ptr %elem_ptr281, align 4
  %elem_ptr282 = getelementptr inbounds float, ptr %buf_void261, i64 11
  store float 1.200000e+01, ptr %elem_ptr282, align 4
  %shape_ptr284 = getelementptr inbounds [1 x i64], ptr %tensor_shape_arr283, i64 0, i64 0
  store i64 12, ptr %shape_ptr284, align 8
  %new_tensor285 = call ptr @tl_tensor_new(ptr %buf_void261, i64 1, ptr %tensor_shape_arr283)
  store ptr %new_tensor285, ptr %data3, align 8
  %data3286 = load ptr, ptr %data3, align 8
  %dims_alloca287 = alloca [2 x i64], align 8
  %dim_ptr288 = getelementptr [2 x i64], ptr %dims_alloca287, i64 0, i64 0
  store i64 1, ptr %dim_ptr288, align 8
  %dim_ptr289 = getelementptr [2 x i64], ptr %dims_alloca287, i64 0, i64 1
  store i64 12, ptr %dim_ptr289, align 8
  %dims_ptr290 = getelementptr [2 x i64], ptr %dims_alloca287, i64 0, i64 0
  %reshape_dims_res291 = call ptr @tl_tensor_reshape_dims(ptr %data3286, ptr %dims_ptr290, i64 2)
  store ptr %reshape_dims_res291, ptr %input3, align 8
  %model292 = load ptr, ptr %model, align 8
  %input3293 = load ptr, ptr %input3, align 8
  %call_method294 = call ptr @tl_GPT_forward(ptr %model292, ptr %input3293)
  store ptr %call_method294, ptr %logits3, align 8
  %logits3295 = load ptr, ptr %logits3, align 8
  %dims_alloca296 = alloca [2 x i64], align 8
  %dim_ptr297 = getelementptr [2 x i64], ptr %dims_alloca296, i64 0, i64 0
  store i64 12, ptr %dim_ptr297, align 8
  %dim_ptr298 = getelementptr [2 x i64], ptr %dims_alloca296, i64 0, i64 1
  store i64 13, ptr %dim_ptr298, align 8
  %dims_ptr299 = getelementptr [2 x i64], ptr %dims_alloca296, i64 0, i64 0
  %reshape_dims_res300 = call ptr @tl_tensor_reshape_dims(ptr %logits3295, ptr %dims_ptr299, i64 2)
  %old_shadowed301 = load ptr, ptr %logits3, align 8
  call void @tl_mem_unregister(ptr %old_shadowed301)
  store ptr %reshape_dims_res300, ptr %logits3302, align 8
  %logits3303 = load ptr, ptr %logits3302, align 8
  %slice_res304 = call ptr @tl_tensor_slice(ptr %logits3303, i64 7, i64 1)
  store ptr %slice_res304, ptr %next_logits3, align 8
  store i64 0, ptr %pred3, align 8
  %old_shadowed305 = load ptr, ptr %max_val230, align 8
  call void @tl_mem_unregister(ptr %old_shadowed305)
  store float -1.000000e+06, ptr %max_val306, align 4
  br label %for_header307

then244:                                          ; preds = %for_body232
  call void @tl_mem_enter_scope()
  %val247 = load float, ptr %val240, align 4
  store float %val247, ptr %max_val230, align 4
  %k248 = load i64, ptr %k236, align 8
  store i64 %k248, ptr %pred2, align 8
  call void @tl_mem_exit_scope()
  br label %merge246

else245:                                          ; preds = %for_body232
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge246

merge246:                                         ; preds = %else245, %then244
  call void @tl_mem_exit_scope()
  %next_idx249 = add i64 %for_idx234, 1
  br label %for_header231

for_header307:                                    ; preds = %merge322, %for_end233
  %for_idx310 = phi i64 [ 0, %for_end233 ], [ %next_idx325, %merge322 ]
  %for_cond311 = icmp slt i64 %for_idx310, 13
  br i1 %for_cond311, label %for_body308, label %for_end309

for_body308:                                      ; preds = %for_header307
  call void @tl_mem_enter_scope()
  store i64 %for_idx310, ptr %k312, align 8
  %next_logits3313 = load ptr, ptr %next_logits3, align 8
  %k314 = load i64, ptr %k312, align 8
  %get_res315 = call float @tl_tensor_get(ptr %next_logits3313, i64 %k314)
  store float %get_res315, ptr %val316, align 4
  %val317 = load float, ptr %val316, align 4
  %max_val318 = load float, ptr %max_val306, align 4
  %fgttmp319 = fcmp ogt float %val317, %max_val318
  br i1 %fgttmp319, label %then320, label %else321

for_end309:                                       ; preds = %for_header307
  %pred3326 = load i64, ptr %pred3, align 8
  call void @tl_print_i64(i64 %pred3326)
  call void @tl_mem_exit_scope()
  %next_idx327 = add i64 %for_idx, 1
  br label %for_header

then320:                                          ; preds = %for_body308
  call void @tl_mem_enter_scope()
  %val323 = load float, ptr %val316, align 4
  store float %val323, ptr %max_val306, align 4
  %k324 = load i64, ptr %k312, align 8
  store i64 %k324, ptr %pred3, align 8
  call void @tl_mem_exit_scope()
  br label %merge322

else321:                                          ; preds = %for_body308
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge322

merge322:                                         ; preds = %else321, %then320
  call void @tl_mem_exit_scope()
  %next_idx325 = add i64 %for_idx310, 1
  br label %for_header307
}
