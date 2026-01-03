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

@str_literal = private unnamed_addr constant [51 x i8] c"Simple loop test - 100 epochs, 100 iterations each\00", align 1
@str_literal.103 = private unnamed_addr constant [7 x i8] c"Epoch:\00", align 1
@str_literal.104 = private unnamed_addr constant [29 x i8] c"Test completed successfully!\00", align 1

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

define void @main() {
entry:
  %loss = alloca ptr, align 16
  %Y_flat = alloca ptr, align 16
  %logits_flat = alloca ptr, align 16
  %logits = alloca ptr, align 16
  %Y = alloca ptr, align 16
  %X = alloca ptr, align 16
  %target = alloca ptr, align 16
  %data = alloca ptr, align 16
  %i = alloca i64, align 16
  %epoch = alloca i64, align 16
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
  call void @tl_print_string(ptr @str_literal)
  br label %for_header

for_header:                                       ; preds = %for_end6, %entry
  %for_idx = phi i64 [ 0, %entry ], [ %next_idx116, %for_end6 ]
  %for_cond = icmp slt i64 %for_idx, 100
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %epoch, align 8
  call void @tl_print_string(ptr @str_literal.103)
  %epoch3 = load i64, ptr %epoch, align 8
  call void @tl_print_i64(i64 %epoch3)
  br label %for_header4

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal.104)
  call void @tl_mem_exit_scope()
  ret void

for_header4:                                      ; preds = %continue_after_free, %for_body
  %for_idx7 = phi i64 [ 0, %for_body ], [ %next_idx, %continue_after_free ]
  %for_cond8 = icmp slt i64 %for_idx7, 100
  br i1 %for_cond8, label %for_body5, label %for_end6

for_body5:                                        ; preds = %for_header4
  call void @tl_mem_enter_scope()
  store i64 %for_idx7, ptr %i, align 8
  %temp_data_heap = call ptr @malloc(i64 48)
  %temp_shape_heap = call ptr @malloc(i64 8)
  %data_elem = getelementptr inbounds float, ptr %temp_data_heap, i64 0
  store float 1.000000e+00, ptr %data_elem, align 4
  %data_elem9 = getelementptr inbounds float, ptr %temp_data_heap, i64 1
  store float 2.000000e+00, ptr %data_elem9, align 4
  %data_elem10 = getelementptr inbounds float, ptr %temp_data_heap, i64 2
  store float 3.000000e+00, ptr %data_elem10, align 4
  %data_elem11 = getelementptr inbounds float, ptr %temp_data_heap, i64 3
  store float 4.000000e+00, ptr %data_elem11, align 4
  %data_elem12 = getelementptr inbounds float, ptr %temp_data_heap, i64 4
  store float 5.000000e+00, ptr %data_elem12, align 4
  %data_elem13 = getelementptr inbounds float, ptr %temp_data_heap, i64 5
  store float 6.000000e+00, ptr %data_elem13, align 4
  %data_elem14 = getelementptr inbounds float, ptr %temp_data_heap, i64 6
  store float 7.000000e+00, ptr %data_elem14, align 4
  %data_elem15 = getelementptr inbounds float, ptr %temp_data_heap, i64 7
  store float 8.000000e+00, ptr %data_elem15, align 4
  %data_elem16 = getelementptr inbounds float, ptr %temp_data_heap, i64 8
  store float 9.000000e+00, ptr %data_elem16, align 4
  %data_elem17 = getelementptr inbounds float, ptr %temp_data_heap, i64 9
  store float 1.000000e+01, ptr %data_elem17, align 4
  %data_elem18 = getelementptr inbounds float, ptr %temp_data_heap, i64 10
  store float 1.100000e+01, ptr %data_elem18, align 4
  %data_elem19 = getelementptr inbounds float, ptr %temp_data_heap, i64 11
  store float 1.200000e+01, ptr %data_elem19, align 4
  %shape_elem = getelementptr inbounds i64, ptr %temp_shape_heap, i64 0
  store i64 12, ptr %shape_elem, align 8
  %new_const_tensor = call ptr @tl_tensor_new(ptr %temp_data_heap, i64 1, ptr %temp_shape_heap)
  call void @free(ptr %temp_data_heap)
  call void @free(ptr %temp_shape_heap)
  store ptr %new_const_tensor, ptr %data, align 8
  %temp_data_heap20 = call ptr @malloc(i64 48)
  %temp_shape_heap21 = call ptr @malloc(i64 8)
  %data_elem22 = getelementptr inbounds float, ptr %temp_data_heap20, i64 0
  store float 2.000000e+00, ptr %data_elem22, align 4
  %data_elem23 = getelementptr inbounds float, ptr %temp_data_heap20, i64 1
  store float 3.000000e+00, ptr %data_elem23, align 4
  %data_elem24 = getelementptr inbounds float, ptr %temp_data_heap20, i64 2
  store float 4.000000e+00, ptr %data_elem24, align 4
  %data_elem25 = getelementptr inbounds float, ptr %temp_data_heap20, i64 3
  store float 5.000000e+00, ptr %data_elem25, align 4
  %data_elem26 = getelementptr inbounds float, ptr %temp_data_heap20, i64 4
  store float 6.000000e+00, ptr %data_elem26, align 4
  %data_elem27 = getelementptr inbounds float, ptr %temp_data_heap20, i64 5
  store float 7.000000e+00, ptr %data_elem27, align 4
  %data_elem28 = getelementptr inbounds float, ptr %temp_data_heap20, i64 6
  store float 8.000000e+00, ptr %data_elem28, align 4
  %data_elem29 = getelementptr inbounds float, ptr %temp_data_heap20, i64 7
  store float 9.000000e+00, ptr %data_elem29, align 4
  %data_elem30 = getelementptr inbounds float, ptr %temp_data_heap20, i64 8
  store float 1.000000e+01, ptr %data_elem30, align 4
  %data_elem31 = getelementptr inbounds float, ptr %temp_data_heap20, i64 9
  store float 1.100000e+01, ptr %data_elem31, align 4
  %data_elem32 = getelementptr inbounds float, ptr %temp_data_heap20, i64 10
  store float 1.200000e+01, ptr %data_elem32, align 4
  %data_elem33 = getelementptr inbounds float, ptr %temp_data_heap20, i64 11
  store float 1.200000e+01, ptr %data_elem33, align 4
  %shape_elem34 = getelementptr inbounds i64, ptr %temp_shape_heap21, i64 0
  store i64 12, ptr %shape_elem34, align 8
  %new_const_tensor35 = call ptr @tl_tensor_new(ptr %temp_data_heap20, i64 1, ptr %temp_shape_heap21)
  call void @free(ptr %temp_data_heap20)
  call void @free(ptr %temp_shape_heap21)
  store ptr %new_const_tensor35, ptr %target, align 8
  %data36 = load ptr, ptr %data, align 8
  %dims_alloca = alloca [2 x i64], align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr, align 8
  %dim_ptr37 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr37, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %data36, ptr %dims_ptr, i64 2)
  store ptr %reshape_dims_res, ptr %X, align 8
  %target38 = load ptr, ptr %target, align 8
  %dims_alloca39 = alloca [2 x i64], align 8
  %dim_ptr40 = getelementptr [2 x i64], ptr %dims_alloca39, i64 0, i64 0
  store i64 1, ptr %dim_ptr40, align 8
  %dim_ptr41 = getelementptr [2 x i64], ptr %dims_alloca39, i64 0, i64 1
  store i64 12, ptr %dim_ptr41, align 8
  %dims_ptr42 = getelementptr [2 x i64], ptr %dims_alloca39, i64 0, i64 0
  %reshape_dims_res43 = call ptr @tl_tensor_reshape_dims(ptr %target38, ptr %dims_ptr42, i64 2)
  store ptr %reshape_dims_res43, ptr %Y, align 8
  %model44 = load ptr, ptr %model, align 8
  %X45 = load ptr, ptr %X, align 8
  %call_method = call ptr @tl_GPT_forward(ptr %model44, ptr %X45)
  store ptr %call_method, ptr %logits, align 8
  %logits46 = load ptr, ptr %logits, align 8
  %dims_alloca47 = alloca [2 x i64], align 8
  %dim_ptr48 = getelementptr [2 x i64], ptr %dims_alloca47, i64 0, i64 0
  store i64 12, ptr %dim_ptr48, align 8
  %dim_ptr49 = getelementptr [2 x i64], ptr %dims_alloca47, i64 0, i64 1
  store i64 13, ptr %dim_ptr49, align 8
  %dims_ptr50 = getelementptr [2 x i64], ptr %dims_alloca47, i64 0, i64 0
  %reshape_dims_res51 = call ptr @tl_tensor_reshape_dims(ptr %logits46, ptr %dims_ptr50, i64 2)
  store ptr %reshape_dims_res51, ptr %logits_flat, align 8
  %Y52 = load ptr, ptr %Y, align 8
  %dims_alloca53 = alloca [1 x i64], align 8
  %dim_ptr54 = getelementptr [1 x i64], ptr %dims_alloca53, i64 0, i64 0
  store i64 12, ptr %dim_ptr54, align 8
  %dims_ptr55 = getelementptr [1 x i64], ptr %dims_alloca53, i64 0, i64 0
  %reshape_dims_res56 = call ptr @tl_tensor_reshape_dims(ptr %Y52, ptr %dims_ptr55, i64 1)
  store ptr %reshape_dims_res56, ptr %Y_flat, align 8
  %logits_flat57 = load ptr, ptr %logits_flat, align 8
  %Y_flat58 = load ptr, ptr %Y_flat, align 8
  %ce_res = call ptr @tl_tensor_cross_entropy(ptr %logits_flat57, ptr %Y_flat58)
  store ptr %ce_res, ptr %loss, align 8
  %loss59 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss59)
  %model60 = load ptr, ptr %model, align 8
  %lr61 = load float, ptr %lr, align 4
  %call_method62 = call ptr @tl_GPT_step(ptr %model60, float %lr61)
  %old_struct_to_free = load ptr, ptr %model, align 8
  %is_not_null = icmp ne ptr %old_struct_to_free, null
  br i1 %is_not_null, label %free_struct, label %continue_after_free

for_end6:                                         ; preds = %for_header4
  call void @tl_mem_exit_scope()
  %next_idx116 = add i64 %for_idx, 1
  br label %for_header

free_struct:                                      ; preds = %for_body5
  br label %continue_after_free

continue_after_free:                              ; preds = %free_struct, %for_body5
  call void @tl_mem_unregister(ptr %call_method62)
  %unreg_field_0 = getelementptr inbounds nuw %GPT, ptr %call_method62, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_063 = getelementptr inbounds nuw %Embedding, ptr %field_val, i32 0, i32 0
  %field_val64 = load ptr, ptr %unreg_field_063, align 8
  call void @tl_mem_unregister(ptr %field_val64)
  %unreg_field_1 = getelementptr inbounds nuw %GPT, ptr %call_method62, i32 0, i32 1
  %field_val65 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val65)
  %unreg_field_066 = getelementptr inbounds nuw %Block, ptr %field_val65, i32 0, i32 0
  %field_val67 = load ptr, ptr %unreg_field_066, align 8
  call void @tl_mem_unregister(ptr %field_val67)
  %unreg_field_068 = getelementptr inbounds nuw %LayerNorm, ptr %field_val67, i32 0, i32 0
  %field_val69 = load ptr, ptr %unreg_field_068, align 8
  call void @tl_mem_unregister(ptr %field_val69)
  %unreg_field_170 = getelementptr inbounds nuw %LayerNorm, ptr %field_val67, i32 0, i32 1
  %field_val71 = load ptr, ptr %unreg_field_170, align 8
  call void @tl_mem_unregister(ptr %field_val71)
  %unreg_field_172 = getelementptr inbounds nuw %Block, ptr %field_val65, i32 0, i32 1
  %field_val73 = load ptr, ptr %unreg_field_172, align 8
  call void @tl_mem_unregister(ptr %field_val73)
  %unreg_field_074 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val73, i32 0, i32 0
  %field_val75 = load ptr, ptr %unreg_field_074, align 8
  call void @tl_mem_unregister(ptr %field_val75)
  %unreg_field_076 = getelementptr inbounds nuw %Linear, ptr %field_val75, i32 0, i32 0
  %field_val77 = load ptr, ptr %unreg_field_076, align 8
  call void @tl_mem_unregister(ptr %field_val77)
  %unreg_field_178 = getelementptr inbounds nuw %Linear, ptr %field_val75, i32 0, i32 1
  %field_val79 = load ptr, ptr %unreg_field_178, align 8
  call void @tl_mem_unregister(ptr %field_val79)
  %unreg_field_180 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val73, i32 0, i32 1
  %field_val81 = load ptr, ptr %unreg_field_180, align 8
  call void @tl_mem_unregister(ptr %field_val81)
  %unreg_field_082 = getelementptr inbounds nuw %Linear, ptr %field_val81, i32 0, i32 0
  %field_val83 = load ptr, ptr %unreg_field_082, align 8
  call void @tl_mem_unregister(ptr %field_val83)
  %unreg_field_184 = getelementptr inbounds nuw %Linear, ptr %field_val81, i32 0, i32 1
  %field_val85 = load ptr, ptr %unreg_field_184, align 8
  call void @tl_mem_unregister(ptr %field_val85)
  %unreg_field_2 = getelementptr inbounds nuw %Block, ptr %field_val65, i32 0, i32 2
  %field_val86 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val86)
  %unreg_field_087 = getelementptr inbounds nuw %LayerNorm, ptr %field_val86, i32 0, i32 0
  %field_val88 = load ptr, ptr %unreg_field_087, align 8
  call void @tl_mem_unregister(ptr %field_val88)
  %unreg_field_189 = getelementptr inbounds nuw %LayerNorm, ptr %field_val86, i32 0, i32 1
  %field_val90 = load ptr, ptr %unreg_field_189, align 8
  call void @tl_mem_unregister(ptr %field_val90)
  %unreg_field_3 = getelementptr inbounds nuw %Block, ptr %field_val65, i32 0, i32 3
  %field_val91 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val91)
  %unreg_field_092 = getelementptr inbounds nuw %MLP, ptr %field_val91, i32 0, i32 0
  %field_val93 = load ptr, ptr %unreg_field_092, align 8
  call void @tl_mem_unregister(ptr %field_val93)
  %unreg_field_094 = getelementptr inbounds nuw %Linear, ptr %field_val93, i32 0, i32 0
  %field_val95 = load ptr, ptr %unreg_field_094, align 8
  call void @tl_mem_unregister(ptr %field_val95)
  %unreg_field_196 = getelementptr inbounds nuw %Linear, ptr %field_val93, i32 0, i32 1
  %field_val97 = load ptr, ptr %unreg_field_196, align 8
  call void @tl_mem_unregister(ptr %field_val97)
  %unreg_field_198 = getelementptr inbounds nuw %MLP, ptr %field_val91, i32 0, i32 1
  %field_val99 = load ptr, ptr %unreg_field_198, align 8
  call void @tl_mem_unregister(ptr %field_val99)
  %unreg_field_0100 = getelementptr inbounds nuw %Linear, ptr %field_val99, i32 0, i32 0
  %field_val101 = load ptr, ptr %unreg_field_0100, align 8
  call void @tl_mem_unregister(ptr %field_val101)
  %unreg_field_1102 = getelementptr inbounds nuw %Linear, ptr %field_val99, i32 0, i32 1
  %field_val103 = load ptr, ptr %unreg_field_1102, align 8
  call void @tl_mem_unregister(ptr %field_val103)
  %unreg_field_2104 = getelementptr inbounds nuw %GPT, ptr %call_method62, i32 0, i32 2
  %field_val105 = load ptr, ptr %unreg_field_2104, align 8
  call void @tl_mem_unregister(ptr %field_val105)
  %unreg_field_0106 = getelementptr inbounds nuw %LayerNorm, ptr %field_val105, i32 0, i32 0
  %field_val107 = load ptr, ptr %unreg_field_0106, align 8
  call void @tl_mem_unregister(ptr %field_val107)
  %unreg_field_1108 = getelementptr inbounds nuw %LayerNorm, ptr %field_val105, i32 0, i32 1
  %field_val109 = load ptr, ptr %unreg_field_1108, align 8
  call void @tl_mem_unregister(ptr %field_val109)
  %unreg_field_3110 = getelementptr inbounds nuw %GPT, ptr %call_method62, i32 0, i32 3
  %field_val111 = load ptr, ptr %unreg_field_3110, align 8
  call void @tl_mem_unregister(ptr %field_val111)
  %unreg_field_0112 = getelementptr inbounds nuw %Linear, ptr %field_val111, i32 0, i32 0
  %field_val113 = load ptr, ptr %unreg_field_0112, align 8
  call void @tl_mem_unregister(ptr %field_val113)
  %unreg_field_1114 = getelementptr inbounds nuw %Linear, ptr %field_val111, i32 0, i32 1
  %field_val115 = load ptr, ptr %unreg_field_1114, align 8
  call void @tl_mem_unregister(ptr %field_val115)
  call void @tl_mem_unregister(ptr %call_method62)
  store ptr %call_method62, ptr %model, align 8
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx7, 1
  br label %for_header4
}
