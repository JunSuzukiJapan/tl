; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

%Linear = type { ptr, ptr }
%Embedding = type { ptr }
%LayerNorm = type { ptr, ptr }
%CausalSelfAttention = type { ptr, ptr, ptr, ptr }
%MLP = type { ptr, ptr }
%Block = type { ptr, ptr, ptr, ptr }
%GPT = type { ptr, ptr, ptr, ptr, ptr, ptr }

@str_literal = private unnamed_addr constant [30 x i8] c"Predictions (Whole Sequence):\00", align 1
@str_literal.107 = private unnamed_addr constant [18 x i8] c"Inference: Input:\00", align 1
@str_literal.108 = private unnamed_addr constant [21 x i8] c"Expected (at pos 1):\00", align 1
@str_literal.109 = private unnamed_addr constant [6 x i8] c"Loss:\00", align 1
@str_literal.110 = private unnamed_addr constant [42 x i8] c"Verifying 2-Digit Addition (Vocab=200)...\00", align 1
@str_literal.111 = private unnamed_addr constant [12 x i8] c"Training...\00", align 1
@str_literal.112 = private unnamed_addr constant [7 x i8] c"epoch:\00", align 1
@str_literal.113 = private unnamed_addr constant [19 x i8] c"Inference check...\00", align 1
@str_literal.114 = private unnamed_addr constant [24 x i8] c"Verification Completed.\00", align 1

declare void @tl_print_i64(i64)

declare void @tl_print_f32(float)

declare void @tl_print_string(ptr)

declare void @tl_print_ptr(ptr)

declare ptr @malloc(i64)

declare ptr @calloc(i64, i64)

declare void @free(ptr)

declare i64 @tl_tensor_dim(ptr, i64)

declare float @tl_tensor_get_f32_md(ptr, ptr, i64)

declare ptr @tl_tensor_new(ptr, i64, ptr)

declare ptr @tl_tensor_from_i64_array(ptr, i64)

declare ptr @tl_tensor_sub(ptr, ptr)

declare void @tl_tensor_free(ptr)

declare ptr @tl_tensor_clone(ptr)

declare ptr @tl_tensor_add(ptr, ptr)

declare ptr @tl_tensor_mul(ptr, ptr)

declare void @tl_tensor_print(ptr)

declare void @tl_tensor_print_1(ptr)

declare void @tl_tensor_print_2(ptr)

declare void @tl_tensor_print_3(ptr)

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

declare void @tl_clear_grads()

declare ptr @tl_checkpoint(ptr, ptr, ptr)

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

declare ptr @tl_tensor_randn_debug(i64, ptr, i1)

declare ptr @tl_varbuilder_get(ptr, i64, ptr)

declare ptr @tl_varbuilder_get_from_tensor(ptr, ptr)

declare void @tl_update_all_params(float)

declare ptr @tl_varbuilder_grad(ptr)

declare void @tl_tensor_backward(ptr)

declare ptr @tl_tensor_grad(ptr)

declare ptr @tl_tensor_detach(ptr, i1)

declare ptr @tl_tensor_contiguous(ptr)

declare ptr @tl_tensor_softmax(ptr, i64)

declare ptr @tl_tensor_cross_entropy(ptr, ptr)

declare void @tl_save_all_params(ptr)

declare void @tl_add_parameter(ptr, ptr)

declare void @tl_load_all_params(ptr)

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

declare i64 @tl_arena_get_offset()

declare i64 @tl_arena_get_capacity()

declare ptr @tl_tensor_reshape_dims.1(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.2(ptr, ptr)

declare ptr @tl_varbuilder_get.3(ptr, i64, ptr)

declare ptr @tl_varbuilder_get_from_tensor.4(ptr, ptr)

declare void @tl_update_all_params.5(float)

declare ptr @tl_varbuilder_grad.6(ptr)

declare void @tl_tensor_backward.7(ptr)

declare ptr @tl_tensor_grad.8(ptr)

declare ptr @tl_tensor_detach.9(ptr, i1)

declare ptr @tl_tensor_softmax.10(ptr, i64)

declare ptr @tl_tensor_cross_entropy.11(ptr, ptr)

declare void @tl_tensor_save.12(ptr, ptr)

declare ptr @tl_tensor_load.13(ptr)

declare void @tl_save_all_params.14(ptr)

declare void @tl_add_parameter.15(ptr, ptr)

declare ptr @tl_tensor_argmax(ptr, i64, i1)

declare i64 @tl_tensor_item_i64(ptr)

declare void @tl_load_all_params.16(ptr)

declare void @tl_tensor_sub_assign.17(ptr, ptr)

declare void @tl_add_parameter.18(ptr, ptr)

declare ptr @tl_register_parameter.19(ptr)

declare ptr @tl_string_concat.20(ptr, ptr)

declare ptr @tl_file_open.21(ptr, ptr)

declare ptr @tl_file_read_string.22(ptr)

declare void @tl_file_write_string.23(ptr, ptr)

declare void @tl_file_close.24(ptr)

declare ptr @tl_path_new.25(ptr)

declare ptr @tl_path_join.26(ptr, ptr)

declare i1 @tl_path_exists.27(ptr)

declare i1 @tl_path_is_dir.28(ptr)

declare i1 @tl_path_is_file.29(ptr)

declare ptr @tl_path_to_string.30(ptr)

declare void @tl_path_free.31(ptr)

declare i1 @tl_http_download.32(ptr, ptr)

declare ptr @tl_http_get.33(ptr)

declare ptr @tl_env_get.34(ptr)

declare void @tl_env_set.35(ptr, ptr)

declare float @tl_system_time.36()

declare void @tl_system_sleep.37(float)

declare i64 @tl_get_memory_mb.38()

declare void @tl_mem_enter_scope.39()

declare void @tl_mem_exit_scope.40()

declare void @tl_mem_register_struct.41(ptr)

declare void @tl_mem_register_tensor.42(ptr)

declare void @tl_mem_unregister.43(ptr)

declare ptr @tl_pool_acquire.44(i64)

declare void @tl_pool_release.45(ptr, i64)

declare void @tl_arena_init.46(i64)

declare i64 @tl_arena_alloc.47(i64)

declare ptr @tl_arena_malloc(i64)

declare i1 @tl_arena_is_active.48()

declare void @tl_arena_free.49()

declare ptr @tl_alloc_tmp(i64)

declare void @tl_free_tmp(ptr)

declare ptr @tl_tensor_reshape_dims.50(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.51(ptr, ptr)

declare ptr @tl_varbuilder_get.52(ptr, i64, ptr)

declare ptr @tl_varbuilder_get_from_tensor.53(ptr, ptr)

declare void @tl_update_all_params.54(float)

declare ptr @tl_varbuilder_grad.55(ptr)

declare void @tl_tensor_backward.56(ptr)

declare ptr @tl_tensor_grad.57(ptr)

declare ptr @tl_tensor_detach.58(ptr, i1)

declare ptr @tl_tensor_softmax.59(ptr, i64)

declare ptr @tl_tensor_cross_entropy.60(ptr, ptr)

declare void @tl_tensor_save.61(ptr, ptr)

declare ptr @tl_tensor_load.62(ptr)

declare void @tl_save_all_params.63(ptr)

declare void @tl_add_parameter.64(ptr, ptr)

declare ptr @tl_tensor_argmax.65(ptr, i64, i1)

declare i64 @tl_tensor_item_i64.66(ptr)

declare void @tl_load_all_params.67(ptr)

declare void @tl_tensor_sub_assign.68(ptr, ptr)

declare void @tl_add_parameter.69(ptr, ptr)

declare ptr @tl_register_parameter.70(ptr)

declare ptr @tl_string_concat.71(ptr, ptr)

declare ptr @tl_file_open.72(ptr, ptr)

declare ptr @tl_file_read_string.73(ptr)

declare void @tl_file_write_string.74(ptr, ptr)

declare void @tl_file_close.75(ptr)

declare ptr @tl_path_new.76(ptr)

declare ptr @tl_path_join.77(ptr, ptr)

declare i1 @tl_path_exists.78(ptr)

declare i1 @tl_path_is_dir.79(ptr)

declare i1 @tl_path_is_file.80(ptr)

declare ptr @tl_path_to_string.81(ptr)

declare void @tl_path_free.82(ptr)

declare i1 @tl_http_download.83(ptr, ptr)

declare ptr @tl_http_get.84(ptr)

declare ptr @tl_env_get.85(ptr)

declare void @tl_env_set.86(ptr, ptr)

declare float @tl_system_time.87()

declare void @tl_system_sleep.88(float)

declare i64 @tl_get_memory_mb.89()

declare void @tl_mem_enter_scope.90()

declare void @tl_mem_exit_scope.91()

declare void @tl_mem_register_struct.92(ptr)

declare void @tl_mem_register_tensor.93(ptr)

declare void @tl_mem_unregister.94(ptr)

declare ptr @tl_pool_acquire.95(i64)

declare void @tl_pool_release.96(ptr, i64)

declare void @tl_arena_init.97(i64)

declare i64 @tl_arena_alloc.98(i64)

declare ptr @tl_arena_malloc.99(i64)

declare i1 @tl_arena_is_active.100()

declare void @tl_arena_free.101()

declare ptr @tl_alloc_tmp.102(i64)

declare void @tl_free_tmp.103(ptr)

define ptr @tl_Linear_new(i64 %i, i64 %o) {
entry:
  %scalar_shape_rhs11 = alloca i64, align 16
  %scalar_data_rhs10 = alloca float, align 16
  %shape_arr7 = alloca [1 x i64], align 8
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %shape_arr = alloca [2 x i64], align 8
  %o2 = alloca i64, align 16
  %i1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %i, ptr %i1, align 8
  store i64 %o, ptr %o2, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Linear, ptr null, i32 1) to i64))
  %i3 = load i64, ptr %i1, align 8
  %o4 = load i64, ptr %o2, align 8
  call void @tl_print_i64(i64 %i3)
  call void @tl_print_i64(i64 %o4)
  %tmp = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %i3, ptr %tmp, align 8
  %tmp5 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 %o4, ptr %tmp5, align 8
  %randn_res = call ptr @tl_tensor_randn_debug(i64 2, ptr %shape_arr, i1 true)
  call void @tl_mem_register_tensor(ptr %randn_res)
  store float 0x3FB99999A0000000, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor_rhs)
  call void @tl_mem_register_tensor(ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res, i1 true)
  call void @tl_mem_register_tensor(ptr %detach_res)
  %init_field = getelementptr inbounds %Linear, ptr %struct_malloc, i32 0, i32 0
  store ptr %detach_res, ptr %init_field, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  %o6 = load i64, ptr %o2, align 8
  call void @tl_print_i64(i64 %o6)
  %tmp8 = getelementptr inbounds [1 x i64], ptr %shape_arr7, i64 0, i64 0
  store i64 %o6, ptr %tmp8, align 8
  %randn_res9 = call ptr @tl_tensor_randn_debug(i64 1, ptr %shape_arr7, i1 true)
  call void @tl_mem_register_tensor(ptr %randn_res9)
  store float 0.000000e+00, ptr %scalar_data_rhs10, align 4
  %scalar_tensor_rhs12 = call ptr @tl_tensor_new(ptr %scalar_data_rhs10, i64 0, ptr %scalar_shape_rhs11)
  %binop_res13 = call ptr @tl_tensor_mul(ptr %randn_res9, ptr %scalar_tensor_rhs12)
  call void @tl_mem_register_tensor(ptr %binop_res13)
  %detach_res14 = call ptr @tl_tensor_detach(ptr %binop_res13, i1 true)
  call void @tl_mem_register_tensor(ptr %detach_res14)
  %init_field15 = getelementptr inbounds %Linear, ptr %struct_malloc, i32 0, i32 1
  store ptr %detach_res14, ptr %init_field15, align 8
  call void @tl_mem_unregister(ptr %detach_res14)
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %Linear, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %struct_malloc, i32 0, i32 1
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
  %ptr_W = getelementptr inbounds %Linear, ptr %self4, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  %matmul_res = call ptr @tl_tensor_matmul(ptr %x3, ptr %W)
  %self5 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds %Linear, ptr %self5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %matmul_res, ptr %b)
  call void @tl_mem_register_tensor(ptr %binop_res)
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
  %ptr_W = getelementptr inbounds %Linear, ptr %s4, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %W)
  call void @tl_mem_register_tensor(ptr %grad_res)
  call void @tl_mem_unregister(ptr %grad_res)
  store ptr %grad_res, ptr %gW, align 8
  %s5 = load ptr, ptr %s, align 8
  %ptr_b = getelementptr inbounds %Linear, ptr %s5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res6 = call ptr @tl_tensor_grad(ptr %b)
  call void @tl_mem_register_tensor(ptr %grad_res6)
  call void @tl_mem_unregister(ptr %grad_res6)
  store ptr %grad_res6, ptr %gb, align 8
  %s7 = load ptr, ptr %s, align 8
  %ptr_W8 = getelementptr inbounds %Linear, ptr %s7, i32 0, i32 0
  %s9 = load ptr, ptr %s, align 8
  %ptr_W10 = getelementptr inbounds %Linear, ptr %s9, i32 0, i32 0
  %W11 = load ptr, ptr %ptr_W10, align 8
  %gW12 = load ptr, ptr %gW, align 8
  %lr13 = load float, ptr %lr2, align 4
  store float %lr13, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %gW12, ptr %scalar_tensor_rhs)
  call void @tl_mem_register_tensor(ptr %binop_res)
  %binop_res14 = call ptr @tl_tensor_sub(ptr %W11, ptr %binop_res)
  call void @tl_mem_register_tensor(ptr %binop_res14)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res14, i1 true)
  call void @tl_mem_register_tensor(ptr %detach_res)
  %old_field_val = load ptr, ptr %ptr_W8, align 8
  %cnt_free_diff = icmp ne ptr %old_field_val, %detach_res
  br i1 %cnt_free_diff, label %free_old_val, label %skip_free

free_old_val:                                     ; preds = %entry
  call void @tl_tensor_free(ptr %old_field_val)
  br label %skip_free

skip_free:                                        ; preds = %free_old_val, %entry
  store ptr %detach_res, ptr %ptr_W8, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  %s15 = load ptr, ptr %s, align 8
  %ptr_b16 = getelementptr inbounds %Linear, ptr %s15, i32 0, i32 1
  %s17 = load ptr, ptr %s, align 8
  %ptr_b18 = getelementptr inbounds %Linear, ptr %s17, i32 0, i32 1
  %b19 = load ptr, ptr %ptr_b18, align 8
  %gb20 = load ptr, ptr %gb, align 8
  %lr21 = load float, ptr %lr2, align 4
  store float %lr21, ptr %scalar_data_rhs22, align 4
  %scalar_tensor_rhs24 = call ptr @tl_tensor_new(ptr %scalar_data_rhs22, i64 0, ptr %scalar_shape_rhs23)
  %binop_res25 = call ptr @tl_tensor_mul(ptr %gb20, ptr %scalar_tensor_rhs24)
  call void @tl_mem_register_tensor(ptr %binop_res25)
  %binop_res26 = call ptr @tl_tensor_sub(ptr %b19, ptr %binop_res25)
  call void @tl_mem_register_tensor(ptr %binop_res26)
  %detach_res27 = call ptr @tl_tensor_detach(ptr %binop_res26, i1 true)
  call void @tl_mem_register_tensor(ptr %detach_res27)
  %old_field_val28 = load ptr, ptr %ptr_b16, align 8
  %cnt_free_diff29 = icmp ne ptr %old_field_val28, %detach_res27
  br i1 %cnt_free_diff29, label %free_old_val30, label %skip_free31

free_old_val30:                                   ; preds = %skip_free
  call void @tl_tensor_free(ptr %old_field_val28)
  br label %skip_free31

skip_free31:                                      ; preds = %free_old_val30, %skip_free
  store ptr %detach_res27, ptr %ptr_b16, align 8
  call void @tl_mem_unregister(ptr %detach_res27)
  %s32 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s32)
  %unreg_field_0 = getelementptr inbounds %Linear, ptr %s32, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %s32, i32 0, i32 1
  %field_val33 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val33)
  %tensor_to_free = load ptr, ptr %gW, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free34 = load ptr, ptr %gb, align 8
  call void @tl_tensor_free(ptr %tensor_to_free34)
  call void @tl_mem_exit_scope()
  ret ptr %s32
}

define ptr @tl_Linear_clone(ptr %self) {
entry:
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Linear, ptr null, i32 1) to i64))
  %self2 = load ptr, ptr %self1, align 8
  %ptr_W = getelementptr inbounds %Linear, ptr %self2, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  %clone_res = call ptr @tl_tensor_clone(ptr %W)
  %init_field = getelementptr inbounds %Linear, ptr %struct_malloc, i32 0, i32 0
  store ptr %clone_res, ptr %init_field, align 8
  call void @tl_mem_unregister(ptr %clone_res)
  %self3 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds %Linear, ptr %self3, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %clone_res4 = call ptr @tl_tensor_clone(ptr %b)
  %init_field5 = getelementptr inbounds %Linear, ptr %struct_malloc, i32 0, i32 1
  store ptr %clone_res4, ptr %init_field5, align 8
  call void @tl_mem_unregister(ptr %clone_res4)
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %Linear, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %struct_malloc, i32 0, i32 1
  %field_val6 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val6)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_Embedding_new(i64 %v, i64 %d) {
entry:
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %shape_arr = alloca [2 x i64], align 8
  %d2 = alloca i64, align 16
  %v1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %v, ptr %v1, align 8
  store i64 %d, ptr %d2, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Embedding, ptr null, i32 1) to i64))
  %v3 = load i64, ptr %v1, align 8
  %d4 = load i64, ptr %d2, align 8
  call void @tl_print_i64(i64 %v3)
  call void @tl_print_i64(i64 %d4)
  %tmp = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %v3, ptr %tmp, align 8
  %tmp5 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 %d4, ptr %tmp5, align 8
  %randn_res = call ptr @tl_tensor_randn_debug(i64 2, ptr %shape_arr, i1 true)
  call void @tl_mem_register_tensor(ptr %randn_res)
  store float 0x3FB99999A0000000, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor_rhs)
  call void @tl_mem_register_tensor(ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res, i1 true)
  call void @tl_mem_register_tensor(ptr %detach_res)
  %init_field = getelementptr inbounds %Embedding, ptr %struct_malloc, i32 0, i32 0
  store ptr %detach_res, ptr %init_field, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %Embedding, ptr %struct_malloc, i32 0, i32 0
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
  %ptr_w = getelementptr inbounds %Embedding, ptr %self4, i32 0, i32 0
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
  %ptr_w = getelementptr inbounds %Embedding, ptr %s4, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %w)
  call void @tl_mem_register_tensor(ptr %grad_res)
  call void @tl_mem_unregister(ptr %grad_res)
  store ptr %grad_res, ptr %g, align 8
  %s5 = load ptr, ptr %s, align 8
  %ptr_w6 = getelementptr inbounds %Embedding, ptr %s5, i32 0, i32 0
  %s7 = load ptr, ptr %s, align 8
  %ptr_w8 = getelementptr inbounds %Embedding, ptr %s7, i32 0, i32 0
  %w9 = load ptr, ptr %ptr_w8, align 8
  %g10 = load ptr, ptr %g, align 8
  %lr11 = load float, ptr %lr2, align 4
  store float %lr11, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %g10, ptr %scalar_tensor_rhs)
  call void @tl_mem_register_tensor(ptr %binop_res)
  %binop_res12 = call ptr @tl_tensor_sub(ptr %w9, ptr %binop_res)
  call void @tl_mem_register_tensor(ptr %binop_res12)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res12, i1 true)
  call void @tl_mem_register_tensor(ptr %detach_res)
  %old_field_val = load ptr, ptr %ptr_w6, align 8
  %cnt_free_diff = icmp ne ptr %old_field_val, %detach_res
  br i1 %cnt_free_diff, label %free_old_val, label %skip_free

free_old_val:                                     ; preds = %entry
  call void @tl_tensor_free(ptr %old_field_val)
  br label %skip_free

skip_free:                                        ; preds = %free_old_val, %entry
  store ptr %detach_res, ptr %ptr_w6, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  %s13 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s13)
  %unreg_field_0 = getelementptr inbounds %Embedding, ptr %s13, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %tensor_to_free = load ptr, ptr %g, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  call void @tl_mem_exit_scope()
  ret ptr %s13
}

define ptr @tl_Embedding_clone(ptr %self) {
entry:
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Embedding, ptr null, i32 1) to i64))
  %self2 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds %Embedding, ptr %self2, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %clone_res = call ptr @tl_tensor_clone(ptr %w)
  %init_field = getelementptr inbounds %Embedding, ptr %struct_malloc, i32 0, i32 0
  store ptr %clone_res, ptr %init_field, align 8
  call void @tl_mem_unregister(ptr %clone_res)
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %Embedding, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_LayerNorm_new(i64 %d) {
entry:
  %scalar_shape_rhs12 = alloca i64, align 16
  %scalar_data_rhs11 = alloca float, align 16
  %shape_arr8 = alloca [1 x i64], align 8
  %scalar_shape_rhs4 = alloca i64, align 16
  %scalar_data_rhs3 = alloca float, align 16
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %shape_arr = alloca [1 x i64], align 8
  %d1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %d, ptr %d1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%LayerNorm, ptr null, i32 1) to i64))
  %d2 = load i64, ptr %d1, align 8
  call void @tl_print_i64(i64 %d2)
  %tmp = getelementptr inbounds [1 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %d2, ptr %tmp, align 8
  %randn_res = call ptr @tl_tensor_randn_debug(i64 1, ptr %shape_arr, i1 true)
  call void @tl_mem_register_tensor(ptr %randn_res)
  store float 0.000000e+00, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor_rhs)
  call void @tl_mem_register_tensor(ptr %binop_res)
  store float 1.000000e+00, ptr %scalar_data_rhs3, align 4
  %scalar_tensor_rhs5 = call ptr @tl_tensor_new(ptr %scalar_data_rhs3, i64 0, ptr %scalar_shape_rhs4)
  %binop_res6 = call ptr @tl_tensor_add(ptr %binop_res, ptr %scalar_tensor_rhs5)
  call void @tl_mem_register_tensor(ptr %binop_res6)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res6, i1 true)
  call void @tl_mem_register_tensor(ptr %detach_res)
  %init_field = getelementptr inbounds %LayerNorm, ptr %struct_malloc, i32 0, i32 0
  store ptr %detach_res, ptr %init_field, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  %d7 = load i64, ptr %d1, align 8
  call void @tl_print_i64(i64 %d7)
  %tmp9 = getelementptr inbounds [1 x i64], ptr %shape_arr8, i64 0, i64 0
  store i64 %d7, ptr %tmp9, align 8
  %randn_res10 = call ptr @tl_tensor_randn_debug(i64 1, ptr %shape_arr8, i1 true)
  call void @tl_mem_register_tensor(ptr %randn_res10)
  store float 0.000000e+00, ptr %scalar_data_rhs11, align 4
  %scalar_tensor_rhs13 = call ptr @tl_tensor_new(ptr %scalar_data_rhs11, i64 0, ptr %scalar_shape_rhs12)
  %binop_res14 = call ptr @tl_tensor_mul(ptr %randn_res10, ptr %scalar_tensor_rhs13)
  call void @tl_mem_register_tensor(ptr %binop_res14)
  %detach_res15 = call ptr @tl_tensor_detach(ptr %binop_res14, i1 true)
  call void @tl_mem_register_tensor(ptr %detach_res15)
  %init_field16 = getelementptr inbounds %LayerNorm, ptr %struct_malloc, i32 0, i32 1
  store ptr %detach_res15, ptr %init_field16, align 8
  call void @tl_mem_unregister(ptr %detach_res15)
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %LayerNorm, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds %LayerNorm, ptr %struct_malloc, i32 0, i32 1
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
  %ptr_b = getelementptr inbounds %LayerNorm, ptr %self4, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %x3, ptr %b)
  call void @tl_mem_register_tensor(ptr %binop_res)
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
  %ptr_b = getelementptr inbounds %LayerNorm, ptr %s4, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %b)
  call void @tl_mem_register_tensor(ptr %grad_res)
  call void @tl_mem_unregister(ptr %grad_res)
  store ptr %grad_res, ptr %gb, align 8
  %s5 = load ptr, ptr %s, align 8
  %ptr_b6 = getelementptr inbounds %LayerNorm, ptr %s5, i32 0, i32 1
  %s7 = load ptr, ptr %s, align 8
  %ptr_b8 = getelementptr inbounds %LayerNorm, ptr %s7, i32 0, i32 1
  %b9 = load ptr, ptr %ptr_b8, align 8
  %gb10 = load ptr, ptr %gb, align 8
  %lr11 = load float, ptr %lr2, align 4
  store float %lr11, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %gb10, ptr %scalar_tensor_rhs)
  call void @tl_mem_register_tensor(ptr %binop_res)
  %binop_res12 = call ptr @tl_tensor_sub(ptr %b9, ptr %binop_res)
  call void @tl_mem_register_tensor(ptr %binop_res12)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res12, i1 true)
  call void @tl_mem_register_tensor(ptr %detach_res)
  %old_field_val = load ptr, ptr %ptr_b6, align 8
  %cnt_free_diff = icmp ne ptr %old_field_val, %detach_res
  br i1 %cnt_free_diff, label %free_old_val, label %skip_free

free_old_val:                                     ; preds = %entry
  call void @tl_tensor_free(ptr %old_field_val)
  br label %skip_free

skip_free:                                        ; preds = %free_old_val, %entry
  store ptr %detach_res, ptr %ptr_b6, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  %s13 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s13)
  %unreg_field_0 = getelementptr inbounds %LayerNorm, ptr %s13, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds %LayerNorm, ptr %s13, i32 0, i32 1
  %field_val14 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val14)
  %tensor_to_free = load ptr, ptr %gb, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  call void @tl_mem_exit_scope()
  ret ptr %s13
}

define ptr @tl_LayerNorm_clone(ptr %self) {
entry:
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%LayerNorm, ptr null, i32 1) to i64))
  %self2 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds %LayerNorm, ptr %self2, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %clone_res = call ptr @tl_tensor_clone(ptr %w)
  %init_field = getelementptr inbounds %LayerNorm, ptr %struct_malloc, i32 0, i32 0
  store ptr %clone_res, ptr %init_field, align 8
  call void @tl_mem_unregister(ptr %clone_res)
  %self3 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds %LayerNorm, ptr %self3, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %clone_res4 = call ptr @tl_tensor_clone(ptr %b)
  %init_field5 = getelementptr inbounds %LayerNorm, ptr %struct_malloc, i32 0, i32 1
  store ptr %clone_res4, ptr %init_field5, align 8
  call void @tl_mem_unregister(ptr %clone_res4)
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %LayerNorm, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds %LayerNorm, ptr %struct_malloc, i32 0, i32 1
  %field_val6 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val6)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_CausalSelfAttention_new(i64 %d) {
entry:
  %d1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %d, ptr %d1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%CausalSelfAttention, ptr null, i32 1) to i64))
  %d2 = load i64, ptr %d1, align 8
  %d3 = load i64, ptr %d1, align 8
  %static_call = call ptr @tl_Linear_new(i64 %d2, i64 %d3)
  %init_field = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d4 = load i64, ptr %d1, align 8
  %d5 = load i64, ptr %d1, align 8
  %static_call6 = call ptr @tl_Linear_new(i64 %d4, i64 %d5)
  %init_field7 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call6, ptr %init_field7, align 8
  %d8 = load i64, ptr %d1, align 8
  %d9 = load i64, ptr %d1, align 8
  %static_call10 = call ptr @tl_Linear_new(i64 %d8, i64 %d9)
  %init_field11 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 2
  store ptr %static_call10, ptr %init_field11, align 8
  %d12 = load i64, ptr %d1, align 8
  %d13 = load i64, ptr %d1, align 8
  %static_call14 = call ptr @tl_Linear_new(i64 %d12, i64 %d13)
  %init_field15 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 3
  store ptr %static_call14, ptr %init_field15, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_016 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 0
  %field_val17 = load ptr, ptr %unreg_field_016, align 8
  call void @tl_mem_unregister(ptr %field_val17)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 1
  %field_val18 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val18)
  %unreg_field_119 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 1
  %field_val20 = load ptr, ptr %unreg_field_119, align 8
  call void @tl_mem_unregister(ptr %field_val20)
  %unreg_field_021 = getelementptr inbounds %Linear, ptr %field_val20, i32 0, i32 0
  %field_val22 = load ptr, ptr %unreg_field_021, align 8
  call void @tl_mem_unregister(ptr %field_val22)
  %unreg_field_123 = getelementptr inbounds %Linear, ptr %field_val20, i32 0, i32 1
  %field_val24 = load ptr, ptr %unreg_field_123, align 8
  call void @tl_mem_unregister(ptr %field_val24)
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 2
  %field_val25 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_026 = getelementptr inbounds %Linear, ptr %field_val25, i32 0, i32 0
  %field_val27 = load ptr, ptr %unreg_field_026, align 8
  call void @tl_mem_unregister(ptr %field_val27)
  %unreg_field_128 = getelementptr inbounds %Linear, ptr %field_val25, i32 0, i32 1
  %field_val29 = load ptr, ptr %unreg_field_128, align 8
  call void @tl_mem_unregister(ptr %field_val29)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 3
  %field_val30 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  %unreg_field_031 = getelementptr inbounds %Linear, ptr %field_val30, i32 0, i32 0
  %field_val32 = load ptr, ptr %unreg_field_031, align 8
  call void @tl_mem_unregister(ptr %field_val32)
  %unreg_field_133 = getelementptr inbounds %Linear, ptr %field_val30, i32 0, i32 1
  %field_val34 = load ptr, ptr %unreg_field_133, align 8
  call void @tl_mem_unregister(ptr %field_val34)
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
  %ptr_q_proj = getelementptr inbounds %CausalSelfAttention, ptr %self3, i32 0, i32 0
  %q_proj = load ptr, ptr %ptr_q_proj, align 8
  %x4 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_Linear_forward(ptr %q_proj, ptr %x4)
  call void @tl_mem_register_tensor(ptr %call_method)
  call void @tl_mem_unregister(ptr %call_method)
  store ptr %call_method, ptr %q, align 8
  %self5 = load ptr, ptr %self1, align 8
  %ptr_k_proj = getelementptr inbounds %CausalSelfAttention, ptr %self5, i32 0, i32 1
  %k_proj = load ptr, ptr %ptr_k_proj, align 8
  %x6 = load ptr, ptr %x2, align 8
  %call_method7 = call ptr @tl_Linear_forward(ptr %k_proj, ptr %x6)
  call void @tl_mem_register_tensor(ptr %call_method7)
  call void @tl_mem_unregister(ptr %call_method7)
  store ptr %call_method7, ptr %k, align 8
  %self8 = load ptr, ptr %self1, align 8
  %ptr_v_proj = getelementptr inbounds %CausalSelfAttention, ptr %self8, i32 0, i32 2
  %v_proj = load ptr, ptr %ptr_v_proj, align 8
  %x9 = load ptr, ptr %x2, align 8
  %call_method10 = call ptr @tl_Linear_forward(ptr %v_proj, ptr %x9)
  call void @tl_mem_register_tensor(ptr %call_method10)
  call void @tl_mem_unregister(ptr %call_method10)
  store ptr %call_method10, ptr %v, align 8
  %q11 = load ptr, ptr %q, align 8
  %k12 = load ptr, ptr %k, align 8
  %transpose_res = call ptr @tl_tensor_transpose(ptr %k12, i64 1, i64 2)
  %matmul_res = call ptr @tl_tensor_matmul(ptr %q11, ptr %transpose_res)
  store float 1.250000e-01, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %matmul_res, ptr %scalar_tensor_rhs)
  call void @tl_mem_register_tensor(ptr %binop_res)
  %tril_res = call ptr @tl_tensor_tril(ptr %binop_res, i32 0)
  %softmax_res = call ptr @tl_tensor_softmax(ptr %tril_res, i64 2)
  call void @tl_tensor_free(ptr %tril_res)
  %v13 = load ptr, ptr %v, align 8
  %matmul_res14 = call ptr @tl_tensor_matmul(ptr %softmax_res, ptr %v13)
  call void @tl_mem_unregister(ptr %matmul_res14)
  store ptr %matmul_res14, ptr %y, align 8
  %self15 = load ptr, ptr %self1, align 8
  %ptr_p_proj = getelementptr inbounds %CausalSelfAttention, ptr %self15, i32 0, i32 3
  %p_proj = load ptr, ptr %ptr_p_proj, align 8
  %y16 = load ptr, ptr %y, align 8
  %call_method17 = call ptr @tl_Linear_forward(ptr %p_proj, ptr %y16)
  call void @tl_mem_register_tensor(ptr %call_method17)
  call void @tl_mem_unregister(ptr %call_method17)
  %tensor_to_free = load ptr, ptr %v, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free18 = load ptr, ptr %y, align 8
  call void @tl_tensor_free(ptr %tensor_to_free18)
  %tensor_to_free19 = load ptr, ptr %q, align 8
  call void @tl_tensor_free(ptr %tensor_to_free19)
  %tensor_to_free20 = load ptr, ptr %k, align 8
  call void @tl_tensor_free(ptr %tensor_to_free20)
  call void @tl_mem_exit_scope()
  ret ptr %call_method17
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
  %ptr_q_proj = getelementptr inbounds %CausalSelfAttention, ptr %s4, i32 0, i32 0
  %s5 = load ptr, ptr %s, align 8
  %ptr_q_proj6 = getelementptr inbounds %CausalSelfAttention, ptr %s5, i32 0, i32 0
  %q_proj = load ptr, ptr %ptr_q_proj6, align 8
  %lr7 = load float, ptr %lr2, align 4
  %call_method = call ptr @tl_Linear_step(ptr %q_proj, float %lr7)
  call void @tl_mem_register_struct(ptr %call_method)
  %old_field_val = load ptr, ptr %ptr_q_proj, align 8
  %cnt_free_diff = icmp ne ptr %old_field_val, %call_method
  br i1 %cnt_free_diff, label %free_old_val, label %skip_free

free_old_val:                                     ; preds = %entry
  %field_gep = getelementptr inbounds %Linear, ptr %old_field_val, i32 0, i32 0
  %field_load = load ptr, ptr %field_gep, align 8
  call void @tl_tensor_free(ptr %field_load)
  %field_gep8 = getelementptr inbounds %Linear, ptr %old_field_val, i32 0, i32 1
  %field_load9 = load ptr, ptr %field_gep8, align 8
  call void @tl_tensor_free(ptr %field_load9)
  br label %skip_free

skip_free:                                        ; preds = %free_old_val, %entry
  store ptr %call_method, ptr %ptr_q_proj, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s10 = load ptr, ptr %s, align 8
  %ptr_k_proj = getelementptr inbounds %CausalSelfAttention, ptr %s10, i32 0, i32 1
  %s11 = load ptr, ptr %s, align 8
  %ptr_k_proj12 = getelementptr inbounds %CausalSelfAttention, ptr %s11, i32 0, i32 1
  %k_proj = load ptr, ptr %ptr_k_proj12, align 8
  %lr13 = load float, ptr %lr2, align 4
  %call_method14 = call ptr @tl_Linear_step(ptr %k_proj, float %lr13)
  call void @tl_mem_register_struct(ptr %call_method14)
  %old_field_val15 = load ptr, ptr %ptr_k_proj, align 8
  %cnt_free_diff16 = icmp ne ptr %old_field_val15, %call_method14
  br i1 %cnt_free_diff16, label %free_old_val17, label %skip_free18

free_old_val17:                                   ; preds = %skip_free
  %field_gep19 = getelementptr inbounds %Linear, ptr %old_field_val15, i32 0, i32 0
  %field_load20 = load ptr, ptr %field_gep19, align 8
  call void @tl_tensor_free(ptr %field_load20)
  %field_gep21 = getelementptr inbounds %Linear, ptr %old_field_val15, i32 0, i32 1
  %field_load22 = load ptr, ptr %field_gep21, align 8
  call void @tl_tensor_free(ptr %field_load22)
  br label %skip_free18

skip_free18:                                      ; preds = %free_old_val17, %skip_free
  store ptr %call_method14, ptr %ptr_k_proj, align 8
  call void @tl_mem_unregister(ptr %call_method14)
  %s23 = load ptr, ptr %s, align 8
  %ptr_v_proj = getelementptr inbounds %CausalSelfAttention, ptr %s23, i32 0, i32 2
  %s24 = load ptr, ptr %s, align 8
  %ptr_v_proj25 = getelementptr inbounds %CausalSelfAttention, ptr %s24, i32 0, i32 2
  %v_proj = load ptr, ptr %ptr_v_proj25, align 8
  %lr26 = load float, ptr %lr2, align 4
  %call_method27 = call ptr @tl_Linear_step(ptr %v_proj, float %lr26)
  call void @tl_mem_register_struct(ptr %call_method27)
  %old_field_val28 = load ptr, ptr %ptr_v_proj, align 8
  %cnt_free_diff29 = icmp ne ptr %old_field_val28, %call_method27
  br i1 %cnt_free_diff29, label %free_old_val30, label %skip_free31

free_old_val30:                                   ; preds = %skip_free18
  %field_gep32 = getelementptr inbounds %Linear, ptr %old_field_val28, i32 0, i32 0
  %field_load33 = load ptr, ptr %field_gep32, align 8
  call void @tl_tensor_free(ptr %field_load33)
  %field_gep34 = getelementptr inbounds %Linear, ptr %old_field_val28, i32 0, i32 1
  %field_load35 = load ptr, ptr %field_gep34, align 8
  call void @tl_tensor_free(ptr %field_load35)
  br label %skip_free31

skip_free31:                                      ; preds = %free_old_val30, %skip_free18
  store ptr %call_method27, ptr %ptr_v_proj, align 8
  call void @tl_mem_unregister(ptr %call_method27)
  %s36 = load ptr, ptr %s, align 8
  %ptr_p_proj = getelementptr inbounds %CausalSelfAttention, ptr %s36, i32 0, i32 3
  %s37 = load ptr, ptr %s, align 8
  %ptr_p_proj38 = getelementptr inbounds %CausalSelfAttention, ptr %s37, i32 0, i32 3
  %p_proj = load ptr, ptr %ptr_p_proj38, align 8
  %lr39 = load float, ptr %lr2, align 4
  %call_method40 = call ptr @tl_Linear_step(ptr %p_proj, float %lr39)
  call void @tl_mem_register_struct(ptr %call_method40)
  %old_field_val41 = load ptr, ptr %ptr_p_proj, align 8
  %cnt_free_diff42 = icmp ne ptr %old_field_val41, %call_method40
  br i1 %cnt_free_diff42, label %free_old_val43, label %skip_free44

free_old_val43:                                   ; preds = %skip_free31
  %field_gep45 = getelementptr inbounds %Linear, ptr %old_field_val41, i32 0, i32 0
  %field_load46 = load ptr, ptr %field_gep45, align 8
  call void @tl_tensor_free(ptr %field_load46)
  %field_gep47 = getelementptr inbounds %Linear, ptr %old_field_val41, i32 0, i32 1
  %field_load48 = load ptr, ptr %field_gep47, align 8
  call void @tl_tensor_free(ptr %field_load48)
  br label %skip_free44

skip_free44:                                      ; preds = %free_old_val43, %skip_free31
  store ptr %call_method40, ptr %ptr_p_proj, align 8
  call void @tl_mem_unregister(ptr %call_method40)
  %s49 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s49)
  %unreg_field_0 = getelementptr inbounds %CausalSelfAttention, ptr %s49, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_050 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 0
  %field_val51 = load ptr, ptr %unreg_field_050, align 8
  call void @tl_mem_unregister(ptr %field_val51)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 1
  %field_val52 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val52)
  %unreg_field_153 = getelementptr inbounds %CausalSelfAttention, ptr %s49, i32 0, i32 1
  %field_val54 = load ptr, ptr %unreg_field_153, align 8
  call void @tl_mem_unregister(ptr %field_val54)
  %unreg_field_055 = getelementptr inbounds %Linear, ptr %field_val54, i32 0, i32 0
  %field_val56 = load ptr, ptr %unreg_field_055, align 8
  call void @tl_mem_unregister(ptr %field_val56)
  %unreg_field_157 = getelementptr inbounds %Linear, ptr %field_val54, i32 0, i32 1
  %field_val58 = load ptr, ptr %unreg_field_157, align 8
  call void @tl_mem_unregister(ptr %field_val58)
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %s49, i32 0, i32 2
  %field_val59 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val59)
  %unreg_field_060 = getelementptr inbounds %Linear, ptr %field_val59, i32 0, i32 0
  %field_val61 = load ptr, ptr %unreg_field_060, align 8
  call void @tl_mem_unregister(ptr %field_val61)
  %unreg_field_162 = getelementptr inbounds %Linear, ptr %field_val59, i32 0, i32 1
  %field_val63 = load ptr, ptr %unreg_field_162, align 8
  call void @tl_mem_unregister(ptr %field_val63)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %s49, i32 0, i32 3
  %field_val64 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val64)
  %unreg_field_065 = getelementptr inbounds %Linear, ptr %field_val64, i32 0, i32 0
  %field_val66 = load ptr, ptr %unreg_field_065, align 8
  call void @tl_mem_unregister(ptr %field_val66)
  %unreg_field_167 = getelementptr inbounds %Linear, ptr %field_val64, i32 0, i32 1
  %field_val68 = load ptr, ptr %unreg_field_167, align 8
  call void @tl_mem_unregister(ptr %field_val68)
  call void @tl_mem_exit_scope()
  ret ptr %s49
}

define ptr @tl_CausalSelfAttention_clone(ptr %self) {
entry:
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%CausalSelfAttention, ptr null, i32 1) to i64))
  %self2 = load ptr, ptr %self1, align 8
  %ptr_q_proj = getelementptr inbounds %CausalSelfAttention, ptr %self2, i32 0, i32 0
  %q_proj = load ptr, ptr %ptr_q_proj, align 8
  %call_method = call ptr @tl_Linear_clone(ptr %q_proj)
  call void @tl_mem_register_struct(ptr %call_method)
  %init_field = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 0
  store ptr %call_method, ptr %init_field, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_k_proj = getelementptr inbounds %CausalSelfAttention, ptr %self3, i32 0, i32 1
  %k_proj = load ptr, ptr %ptr_k_proj, align 8
  %call_method4 = call ptr @tl_Linear_clone(ptr %k_proj)
  call void @tl_mem_register_struct(ptr %call_method4)
  %init_field5 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 1
  store ptr %call_method4, ptr %init_field5, align 8
  %self6 = load ptr, ptr %self1, align 8
  %ptr_v_proj = getelementptr inbounds %CausalSelfAttention, ptr %self6, i32 0, i32 2
  %v_proj = load ptr, ptr %ptr_v_proj, align 8
  %call_method7 = call ptr @tl_Linear_clone(ptr %v_proj)
  call void @tl_mem_register_struct(ptr %call_method7)
  %init_field8 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 2
  store ptr %call_method7, ptr %init_field8, align 8
  %self9 = load ptr, ptr %self1, align 8
  %ptr_p_proj = getelementptr inbounds %CausalSelfAttention, ptr %self9, i32 0, i32 3
  %p_proj = load ptr, ptr %ptr_p_proj, align 8
  %call_method10 = call ptr @tl_Linear_clone(ptr %p_proj)
  call void @tl_mem_register_struct(ptr %call_method10)
  %init_field11 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 3
  store ptr %call_method10, ptr %init_field11, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_012 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 0
  %field_val13 = load ptr, ptr %unreg_field_012, align 8
  call void @tl_mem_unregister(ptr %field_val13)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 1
  %field_val14 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val14)
  %unreg_field_115 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 1
  %field_val16 = load ptr, ptr %unreg_field_115, align 8
  call void @tl_mem_unregister(ptr %field_val16)
  %unreg_field_017 = getelementptr inbounds %Linear, ptr %field_val16, i32 0, i32 0
  %field_val18 = load ptr, ptr %unreg_field_017, align 8
  call void @tl_mem_unregister(ptr %field_val18)
  %unreg_field_119 = getelementptr inbounds %Linear, ptr %field_val16, i32 0, i32 1
  %field_val20 = load ptr, ptr %unreg_field_119, align 8
  call void @tl_mem_unregister(ptr %field_val20)
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 2
  %field_val21 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val21)
  %unreg_field_022 = getelementptr inbounds %Linear, ptr %field_val21, i32 0, i32 0
  %field_val23 = load ptr, ptr %unreg_field_022, align 8
  call void @tl_mem_unregister(ptr %field_val23)
  %unreg_field_124 = getelementptr inbounds %Linear, ptr %field_val21, i32 0, i32 1
  %field_val25 = load ptr, ptr %unreg_field_124, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 3
  %field_val26 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  %unreg_field_027 = getelementptr inbounds %Linear, ptr %field_val26, i32 0, i32 0
  %field_val28 = load ptr, ptr %unreg_field_027, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_129 = getelementptr inbounds %Linear, ptr %field_val26, i32 0, i32 1
  %field_val30 = load ptr, ptr %unreg_field_129, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
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
  %init_field = getelementptr inbounds %MLP, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d4 = load i64, ptr %d1, align 8
  %multmp5 = mul i64 %d4, 4
  %d6 = load i64, ptr %d1, align 8
  %static_call7 = call ptr @tl_Linear_new(i64 %multmp5, i64 %d6)
  %init_field8 = getelementptr inbounds %MLP, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call7, ptr %init_field8, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %MLP, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_09 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 0
  %field_val10 = load ptr, ptr %unreg_field_09, align 8
  call void @tl_mem_unregister(ptr %field_val10)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 1
  %field_val11 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val11)
  %unreg_field_112 = getelementptr inbounds %MLP, ptr %struct_malloc, i32 0, i32 1
  %field_val13 = load ptr, ptr %unreg_field_112, align 8
  call void @tl_mem_unregister(ptr %field_val13)
  %unreg_field_014 = getelementptr inbounds %Linear, ptr %field_val13, i32 0, i32 0
  %field_val15 = load ptr, ptr %unreg_field_014, align 8
  call void @tl_mem_unregister(ptr %field_val15)
  %unreg_field_116 = getelementptr inbounds %Linear, ptr %field_val13, i32 0, i32 1
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
  %ptr_p = getelementptr inbounds %MLP, ptr %self3, i32 0, i32 1
  %p = load ptr, ptr %ptr_p, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_f = getelementptr inbounds %MLP, ptr %self4, i32 0, i32 0
  %f = load ptr, ptr %ptr_f, align 8
  %x5 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_Linear_forward(ptr %f, ptr %x5)
  call void @tl_mem_register_tensor(ptr %call_method)
  %relu_res = call ptr @tl_tensor_relu(ptr %call_method)
  %call_method6 = call ptr @tl_Linear_forward(ptr %p, ptr %relu_res)
  call void @tl_tensor_free(ptr %relu_res)
  call void @tl_mem_register_tensor(ptr %call_method6)
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
  %ptr_f = getelementptr inbounds %MLP, ptr %s4, i32 0, i32 0
  %s5 = load ptr, ptr %s, align 8
  %ptr_f6 = getelementptr inbounds %MLP, ptr %s5, i32 0, i32 0
  %f = load ptr, ptr %ptr_f6, align 8
  %lr7 = load float, ptr %lr2, align 4
  %call_method = call ptr @tl_Linear_step(ptr %f, float %lr7)
  call void @tl_mem_register_struct(ptr %call_method)
  %old_field_val = load ptr, ptr %ptr_f, align 8
  %cnt_free_diff = icmp ne ptr %old_field_val, %call_method
  br i1 %cnt_free_diff, label %free_old_val, label %skip_free

free_old_val:                                     ; preds = %entry
  %field_gep = getelementptr inbounds %Linear, ptr %old_field_val, i32 0, i32 0
  %field_load = load ptr, ptr %field_gep, align 8
  call void @tl_tensor_free(ptr %field_load)
  %field_gep8 = getelementptr inbounds %Linear, ptr %old_field_val, i32 0, i32 1
  %field_load9 = load ptr, ptr %field_gep8, align 8
  call void @tl_tensor_free(ptr %field_load9)
  br label %skip_free

skip_free:                                        ; preds = %free_old_val, %entry
  store ptr %call_method, ptr %ptr_f, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s10 = load ptr, ptr %s, align 8
  %ptr_p = getelementptr inbounds %MLP, ptr %s10, i32 0, i32 1
  %s11 = load ptr, ptr %s, align 8
  %ptr_p12 = getelementptr inbounds %MLP, ptr %s11, i32 0, i32 1
  %p = load ptr, ptr %ptr_p12, align 8
  %lr13 = load float, ptr %lr2, align 4
  %call_method14 = call ptr @tl_Linear_step(ptr %p, float %lr13)
  call void @tl_mem_register_struct(ptr %call_method14)
  %old_field_val15 = load ptr, ptr %ptr_p, align 8
  %cnt_free_diff16 = icmp ne ptr %old_field_val15, %call_method14
  br i1 %cnt_free_diff16, label %free_old_val17, label %skip_free18

free_old_val17:                                   ; preds = %skip_free
  %field_gep19 = getelementptr inbounds %Linear, ptr %old_field_val15, i32 0, i32 0
  %field_load20 = load ptr, ptr %field_gep19, align 8
  call void @tl_tensor_free(ptr %field_load20)
  %field_gep21 = getelementptr inbounds %Linear, ptr %old_field_val15, i32 0, i32 1
  %field_load22 = load ptr, ptr %field_gep21, align 8
  call void @tl_tensor_free(ptr %field_load22)
  br label %skip_free18

skip_free18:                                      ; preds = %free_old_val17, %skip_free
  store ptr %call_method14, ptr %ptr_p, align 8
  call void @tl_mem_unregister(ptr %call_method14)
  %s23 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s23)
  %unreg_field_0 = getelementptr inbounds %MLP, ptr %s23, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_024 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 0
  %field_val25 = load ptr, ptr %unreg_field_024, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 1
  %field_val26 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  %unreg_field_127 = getelementptr inbounds %MLP, ptr %s23, i32 0, i32 1
  %field_val28 = load ptr, ptr %unreg_field_127, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_029 = getelementptr inbounds %Linear, ptr %field_val28, i32 0, i32 0
  %field_val30 = load ptr, ptr %unreg_field_029, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  %unreg_field_131 = getelementptr inbounds %Linear, ptr %field_val28, i32 0, i32 1
  %field_val32 = load ptr, ptr %unreg_field_131, align 8
  call void @tl_mem_unregister(ptr %field_val32)
  call void @tl_mem_exit_scope()
  ret ptr %s23
}

define ptr @tl_MLP_clone(ptr %self) {
entry:
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%MLP, ptr null, i32 1) to i64))
  %self2 = load ptr, ptr %self1, align 8
  %ptr_f = getelementptr inbounds %MLP, ptr %self2, i32 0, i32 0
  %f = load ptr, ptr %ptr_f, align 8
  %call_method = call ptr @tl_Linear_clone(ptr %f)
  call void @tl_mem_register_struct(ptr %call_method)
  %init_field = getelementptr inbounds %MLP, ptr %struct_malloc, i32 0, i32 0
  store ptr %call_method, ptr %init_field, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_p = getelementptr inbounds %MLP, ptr %self3, i32 0, i32 1
  %p = load ptr, ptr %ptr_p, align 8
  %call_method4 = call ptr @tl_Linear_clone(ptr %p)
  call void @tl_mem_register_struct(ptr %call_method4)
  %init_field5 = getelementptr inbounds %MLP, ptr %struct_malloc, i32 0, i32 1
  store ptr %call_method4, ptr %init_field5, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %MLP, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_06 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 0
  %field_val7 = load ptr, ptr %unreg_field_06, align 8
  call void @tl_mem_unregister(ptr %field_val7)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 1
  %field_val8 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val8)
  %unreg_field_19 = getelementptr inbounds %MLP, ptr %struct_malloc, i32 0, i32 1
  %field_val10 = load ptr, ptr %unreg_field_19, align 8
  call void @tl_mem_unregister(ptr %field_val10)
  %unreg_field_011 = getelementptr inbounds %Linear, ptr %field_val10, i32 0, i32 0
  %field_val12 = load ptr, ptr %unreg_field_011, align 8
  call void @tl_mem_unregister(ptr %field_val12)
  %unreg_field_113 = getelementptr inbounds %Linear, ptr %field_val10, i32 0, i32 1
  %field_val14 = load ptr, ptr %unreg_field_113, align 8
  call void @tl_mem_unregister(ptr %field_val14)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_Block_new(i64 %d) {
entry:
  %d1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %d, ptr %d1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Block, ptr null, i32 1) to i64))
  %d2 = load i64, ptr %d1, align 8
  %static_call = call ptr @tl_LayerNorm_new(i64 %d2)
  %init_field = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d3 = load i64, ptr %d1, align 8
  %static_call4 = call ptr @tl_CausalSelfAttention_new(i64 %d3)
  %init_field5 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call4, ptr %init_field5, align 8
  %d6 = load i64, ptr %d1, align 8
  %static_call7 = call ptr @tl_LayerNorm_new(i64 %d6)
  %init_field8 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 2
  store ptr %static_call7, ptr %init_field8, align 8
  %d9 = load i64, ptr %d1, align 8
  %static_call10 = call ptr @tl_MLP_new(i64 %d9)
  %init_field11 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 3
  store ptr %static_call10, ptr %init_field11, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_012 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 0
  %field_val13 = load ptr, ptr %unreg_field_012, align 8
  call void @tl_mem_unregister(ptr %field_val13)
  %unreg_field_1 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 1
  %field_val14 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val14)
  %unreg_field_115 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 1
  %field_val16 = load ptr, ptr %unreg_field_115, align 8
  call void @tl_mem_unregister(ptr %field_val16)
  %unreg_field_017 = getelementptr inbounds %CausalSelfAttention, ptr %field_val16, i32 0, i32 0
  %field_val18 = load ptr, ptr %unreg_field_017, align 8
  call void @tl_mem_unregister(ptr %field_val18)
  %unreg_field_019 = getelementptr inbounds %Linear, ptr %field_val18, i32 0, i32 0
  %field_val20 = load ptr, ptr %unreg_field_019, align 8
  call void @tl_mem_unregister(ptr %field_val20)
  %unreg_field_121 = getelementptr inbounds %Linear, ptr %field_val18, i32 0, i32 1
  %field_val22 = load ptr, ptr %unreg_field_121, align 8
  call void @tl_mem_unregister(ptr %field_val22)
  %unreg_field_123 = getelementptr inbounds %CausalSelfAttention, ptr %field_val16, i32 0, i32 1
  %field_val24 = load ptr, ptr %unreg_field_123, align 8
  call void @tl_mem_unregister(ptr %field_val24)
  %unreg_field_025 = getelementptr inbounds %Linear, ptr %field_val24, i32 0, i32 0
  %field_val26 = load ptr, ptr %unreg_field_025, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  %unreg_field_127 = getelementptr inbounds %Linear, ptr %field_val24, i32 0, i32 1
  %field_val28 = load ptr, ptr %unreg_field_127, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %field_val16, i32 0, i32 2
  %field_val29 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val29)
  %unreg_field_030 = getelementptr inbounds %Linear, ptr %field_val29, i32 0, i32 0
  %field_val31 = load ptr, ptr %unreg_field_030, align 8
  call void @tl_mem_unregister(ptr %field_val31)
  %unreg_field_132 = getelementptr inbounds %Linear, ptr %field_val29, i32 0, i32 1
  %field_val33 = load ptr, ptr %unreg_field_132, align 8
  call void @tl_mem_unregister(ptr %field_val33)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val16, i32 0, i32 3
  %field_val34 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  %unreg_field_035 = getelementptr inbounds %Linear, ptr %field_val34, i32 0, i32 0
  %field_val36 = load ptr, ptr %unreg_field_035, align 8
  call void @tl_mem_unregister(ptr %field_val36)
  %unreg_field_137 = getelementptr inbounds %Linear, ptr %field_val34, i32 0, i32 1
  %field_val38 = load ptr, ptr %unreg_field_137, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_239 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 2
  %field_val40 = load ptr, ptr %unreg_field_239, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_041 = getelementptr inbounds %LayerNorm, ptr %field_val40, i32 0, i32 0
  %field_val42 = load ptr, ptr %unreg_field_041, align 8
  call void @tl_mem_unregister(ptr %field_val42)
  %unreg_field_143 = getelementptr inbounds %LayerNorm, ptr %field_val40, i32 0, i32 1
  %field_val44 = load ptr, ptr %unreg_field_143, align 8
  call void @tl_mem_unregister(ptr %field_val44)
  %unreg_field_345 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 3
  %field_val46 = load ptr, ptr %unreg_field_345, align 8
  call void @tl_mem_unregister(ptr %field_val46)
  %unreg_field_047 = getelementptr inbounds %MLP, ptr %field_val46, i32 0, i32 0
  %field_val48 = load ptr, ptr %unreg_field_047, align 8
  call void @tl_mem_unregister(ptr %field_val48)
  %unreg_field_049 = getelementptr inbounds %Linear, ptr %field_val48, i32 0, i32 0
  %field_val50 = load ptr, ptr %unreg_field_049, align 8
  call void @tl_mem_unregister(ptr %field_val50)
  %unreg_field_151 = getelementptr inbounds %Linear, ptr %field_val48, i32 0, i32 1
  %field_val52 = load ptr, ptr %unreg_field_151, align 8
  call void @tl_mem_unregister(ptr %field_val52)
  %unreg_field_153 = getelementptr inbounds %MLP, ptr %field_val46, i32 0, i32 1
  %field_val54 = load ptr, ptr %unreg_field_153, align 8
  call void @tl_mem_unregister(ptr %field_val54)
  %unreg_field_055 = getelementptr inbounds %Linear, ptr %field_val54, i32 0, i32 0
  %field_val56 = load ptr, ptr %unreg_field_055, align 8
  call void @tl_mem_unregister(ptr %field_val56)
  %unreg_field_157 = getelementptr inbounds %Linear, ptr %field_val54, i32 0, i32 1
  %field_val58 = load ptr, ptr %unreg_field_157, align 8
  call void @tl_mem_unregister(ptr %field_val58)
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
  %ptr_a = getelementptr inbounds %Block, ptr %self4, i32 0, i32 1
  %a = load ptr, ptr %ptr_a, align 8
  %self5 = load ptr, ptr %self1, align 8
  %ptr_l1 = getelementptr inbounds %Block, ptr %self5, i32 0, i32 0
  %l1 = load ptr, ptr %ptr_l1, align 8
  %x6 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_LayerNorm_forward(ptr %l1, ptr %x6)
  call void @tl_mem_register_tensor(ptr %call_method)
  %call_method7 = call ptr @tl_CausalSelfAttention_forward(ptr %a, ptr %call_method)
  call void @tl_tensor_free(ptr %call_method)
  call void @tl_mem_register_tensor(ptr %call_method7)
  %binop_res = call ptr @tl_tensor_add(ptr %x3, ptr %call_method7)
  call void @tl_mem_register_tensor(ptr %binop_res)
  call void @tl_mem_unregister(ptr %binop_res)
  store ptr %binop_res, ptr %x8, align 8
  %x9 = load ptr, ptr %x8, align 8
  %self10 = load ptr, ptr %self1, align 8
  %ptr_m = getelementptr inbounds %Block, ptr %self10, i32 0, i32 3
  %m = load ptr, ptr %ptr_m, align 8
  %self11 = load ptr, ptr %self1, align 8
  %ptr_l2 = getelementptr inbounds %Block, ptr %self11, i32 0, i32 2
  %l2 = load ptr, ptr %ptr_l2, align 8
  %x12 = load ptr, ptr %x8, align 8
  %call_method13 = call ptr @tl_LayerNorm_forward(ptr %l2, ptr %x12)
  call void @tl_mem_register_tensor(ptr %call_method13)
  %call_method14 = call ptr @tl_MLP_forward(ptr %m, ptr %call_method13)
  call void @tl_tensor_free(ptr %call_method13)
  call void @tl_mem_register_tensor(ptr %call_method14)
  %binop_res15 = call ptr @tl_tensor_add(ptr %x9, ptr %call_method14)
  call void @tl_mem_register_tensor(ptr %binop_res15)
  call void @tl_mem_unregister(ptr %binop_res15)
  %tensor_to_free = load ptr, ptr %x8, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
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
  %ptr_l1 = getelementptr inbounds %Block, ptr %s4, i32 0, i32 0
  %s5 = load ptr, ptr %s, align 8
  %ptr_l16 = getelementptr inbounds %Block, ptr %s5, i32 0, i32 0
  %l1 = load ptr, ptr %ptr_l16, align 8
  %lr7 = load float, ptr %lr2, align 4
  %call_method = call ptr @tl_LayerNorm_step(ptr %l1, float %lr7)
  call void @tl_mem_register_struct(ptr %call_method)
  %old_field_val = load ptr, ptr %ptr_l1, align 8
  %cnt_free_diff = icmp ne ptr %old_field_val, %call_method
  br i1 %cnt_free_diff, label %free_old_val, label %skip_free

free_old_val:                                     ; preds = %entry
  %field_gep = getelementptr inbounds %LayerNorm, ptr %old_field_val, i32 0, i32 0
  %field_load = load ptr, ptr %field_gep, align 8
  call void @tl_tensor_free(ptr %field_load)
  %field_gep8 = getelementptr inbounds %LayerNorm, ptr %old_field_val, i32 0, i32 1
  %field_load9 = load ptr, ptr %field_gep8, align 8
  call void @tl_tensor_free(ptr %field_load9)
  br label %skip_free

skip_free:                                        ; preds = %free_old_val, %entry
  store ptr %call_method, ptr %ptr_l1, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s10 = load ptr, ptr %s, align 8
  %ptr_a = getelementptr inbounds %Block, ptr %s10, i32 0, i32 1
  %s11 = load ptr, ptr %s, align 8
  %ptr_a12 = getelementptr inbounds %Block, ptr %s11, i32 0, i32 1
  %a = load ptr, ptr %ptr_a12, align 8
  %lr13 = load float, ptr %lr2, align 4
  %call_method14 = call ptr @tl_CausalSelfAttention_step(ptr %a, float %lr13)
  call void @tl_mem_register_struct(ptr %call_method14)
  %old_field_val15 = load ptr, ptr %ptr_a, align 8
  %cnt_free_diff16 = icmp ne ptr %old_field_val15, %call_method14
  br i1 %cnt_free_diff16, label %free_old_val17, label %skip_free18

free_old_val17:                                   ; preds = %skip_free
  %field_gep19 = getelementptr inbounds %CausalSelfAttention, ptr %old_field_val15, i32 0, i32 0
  %field_load20 = load ptr, ptr %field_gep19, align 8
  %field_gep21 = getelementptr inbounds %Linear, ptr %field_load20, i32 0, i32 0
  %field_load22 = load ptr, ptr %field_gep21, align 8
  call void @tl_tensor_free(ptr %field_load22)
  %field_gep23 = getelementptr inbounds %Linear, ptr %field_load20, i32 0, i32 1
  %field_load24 = load ptr, ptr %field_gep23, align 8
  call void @tl_tensor_free(ptr %field_load24)
  %field_gep25 = getelementptr inbounds %CausalSelfAttention, ptr %old_field_val15, i32 0, i32 1
  %field_load26 = load ptr, ptr %field_gep25, align 8
  %field_gep27 = getelementptr inbounds %Linear, ptr %field_load26, i32 0, i32 0
  %field_load28 = load ptr, ptr %field_gep27, align 8
  call void @tl_tensor_free(ptr %field_load28)
  %field_gep29 = getelementptr inbounds %Linear, ptr %field_load26, i32 0, i32 1
  %field_load30 = load ptr, ptr %field_gep29, align 8
  call void @tl_tensor_free(ptr %field_load30)
  %field_gep31 = getelementptr inbounds %CausalSelfAttention, ptr %old_field_val15, i32 0, i32 2
  %field_load32 = load ptr, ptr %field_gep31, align 8
  %field_gep33 = getelementptr inbounds %Linear, ptr %field_load32, i32 0, i32 0
  %field_load34 = load ptr, ptr %field_gep33, align 8
  call void @tl_tensor_free(ptr %field_load34)
  %field_gep35 = getelementptr inbounds %Linear, ptr %field_load32, i32 0, i32 1
  %field_load36 = load ptr, ptr %field_gep35, align 8
  call void @tl_tensor_free(ptr %field_load36)
  %field_gep37 = getelementptr inbounds %CausalSelfAttention, ptr %old_field_val15, i32 0, i32 3
  %field_load38 = load ptr, ptr %field_gep37, align 8
  %field_gep39 = getelementptr inbounds %Linear, ptr %field_load38, i32 0, i32 0
  %field_load40 = load ptr, ptr %field_gep39, align 8
  call void @tl_tensor_free(ptr %field_load40)
  %field_gep41 = getelementptr inbounds %Linear, ptr %field_load38, i32 0, i32 1
  %field_load42 = load ptr, ptr %field_gep41, align 8
  call void @tl_tensor_free(ptr %field_load42)
  br label %skip_free18

skip_free18:                                      ; preds = %free_old_val17, %skip_free
  store ptr %call_method14, ptr %ptr_a, align 8
  call void @tl_mem_unregister(ptr %call_method14)
  %s43 = load ptr, ptr %s, align 8
  %ptr_l2 = getelementptr inbounds %Block, ptr %s43, i32 0, i32 2
  %s44 = load ptr, ptr %s, align 8
  %ptr_l245 = getelementptr inbounds %Block, ptr %s44, i32 0, i32 2
  %l2 = load ptr, ptr %ptr_l245, align 8
  %lr46 = load float, ptr %lr2, align 4
  %call_method47 = call ptr @tl_LayerNorm_step(ptr %l2, float %lr46)
  call void @tl_mem_register_struct(ptr %call_method47)
  %old_field_val48 = load ptr, ptr %ptr_l2, align 8
  %cnt_free_diff49 = icmp ne ptr %old_field_val48, %call_method47
  br i1 %cnt_free_diff49, label %free_old_val50, label %skip_free51

free_old_val50:                                   ; preds = %skip_free18
  %field_gep52 = getelementptr inbounds %LayerNorm, ptr %old_field_val48, i32 0, i32 0
  %field_load53 = load ptr, ptr %field_gep52, align 8
  call void @tl_tensor_free(ptr %field_load53)
  %field_gep54 = getelementptr inbounds %LayerNorm, ptr %old_field_val48, i32 0, i32 1
  %field_load55 = load ptr, ptr %field_gep54, align 8
  call void @tl_tensor_free(ptr %field_load55)
  br label %skip_free51

skip_free51:                                      ; preds = %free_old_val50, %skip_free18
  store ptr %call_method47, ptr %ptr_l2, align 8
  call void @tl_mem_unregister(ptr %call_method47)
  %s56 = load ptr, ptr %s, align 8
  %ptr_m = getelementptr inbounds %Block, ptr %s56, i32 0, i32 3
  %s57 = load ptr, ptr %s, align 8
  %ptr_m58 = getelementptr inbounds %Block, ptr %s57, i32 0, i32 3
  %m = load ptr, ptr %ptr_m58, align 8
  %lr59 = load float, ptr %lr2, align 4
  %call_method60 = call ptr @tl_MLP_step(ptr %m, float %lr59)
  call void @tl_mem_register_struct(ptr %call_method60)
  %old_field_val61 = load ptr, ptr %ptr_m, align 8
  %cnt_free_diff62 = icmp ne ptr %old_field_val61, %call_method60
  br i1 %cnt_free_diff62, label %free_old_val63, label %skip_free64

free_old_val63:                                   ; preds = %skip_free51
  %field_gep65 = getelementptr inbounds %MLP, ptr %old_field_val61, i32 0, i32 0
  %field_load66 = load ptr, ptr %field_gep65, align 8
  %field_gep67 = getelementptr inbounds %Linear, ptr %field_load66, i32 0, i32 0
  %field_load68 = load ptr, ptr %field_gep67, align 8
  call void @tl_tensor_free(ptr %field_load68)
  %field_gep69 = getelementptr inbounds %Linear, ptr %field_load66, i32 0, i32 1
  %field_load70 = load ptr, ptr %field_gep69, align 8
  call void @tl_tensor_free(ptr %field_load70)
  %field_gep71 = getelementptr inbounds %MLP, ptr %old_field_val61, i32 0, i32 1
  %field_load72 = load ptr, ptr %field_gep71, align 8
  %field_gep73 = getelementptr inbounds %Linear, ptr %field_load72, i32 0, i32 0
  %field_load74 = load ptr, ptr %field_gep73, align 8
  call void @tl_tensor_free(ptr %field_load74)
  %field_gep75 = getelementptr inbounds %Linear, ptr %field_load72, i32 0, i32 1
  %field_load76 = load ptr, ptr %field_gep75, align 8
  call void @tl_tensor_free(ptr %field_load76)
  br label %skip_free64

skip_free64:                                      ; preds = %free_old_val63, %skip_free51
  store ptr %call_method60, ptr %ptr_m, align 8
  call void @tl_mem_unregister(ptr %call_method60)
  %s77 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s77)
  %unreg_field_0 = getelementptr inbounds %Block, ptr %s77, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_078 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 0
  %field_val79 = load ptr, ptr %unreg_field_078, align 8
  call void @tl_mem_unregister(ptr %field_val79)
  %unreg_field_1 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 1
  %field_val80 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val80)
  %unreg_field_181 = getelementptr inbounds %Block, ptr %s77, i32 0, i32 1
  %field_val82 = load ptr, ptr %unreg_field_181, align 8
  call void @tl_mem_unregister(ptr %field_val82)
  %unreg_field_083 = getelementptr inbounds %CausalSelfAttention, ptr %field_val82, i32 0, i32 0
  %field_val84 = load ptr, ptr %unreg_field_083, align 8
  call void @tl_mem_unregister(ptr %field_val84)
  %unreg_field_085 = getelementptr inbounds %Linear, ptr %field_val84, i32 0, i32 0
  %field_val86 = load ptr, ptr %unreg_field_085, align 8
  call void @tl_mem_unregister(ptr %field_val86)
  %unreg_field_187 = getelementptr inbounds %Linear, ptr %field_val84, i32 0, i32 1
  %field_val88 = load ptr, ptr %unreg_field_187, align 8
  call void @tl_mem_unregister(ptr %field_val88)
  %unreg_field_189 = getelementptr inbounds %CausalSelfAttention, ptr %field_val82, i32 0, i32 1
  %field_val90 = load ptr, ptr %unreg_field_189, align 8
  call void @tl_mem_unregister(ptr %field_val90)
  %unreg_field_091 = getelementptr inbounds %Linear, ptr %field_val90, i32 0, i32 0
  %field_val92 = load ptr, ptr %unreg_field_091, align 8
  call void @tl_mem_unregister(ptr %field_val92)
  %unreg_field_193 = getelementptr inbounds %Linear, ptr %field_val90, i32 0, i32 1
  %field_val94 = load ptr, ptr %unreg_field_193, align 8
  call void @tl_mem_unregister(ptr %field_val94)
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %field_val82, i32 0, i32 2
  %field_val95 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val95)
  %unreg_field_096 = getelementptr inbounds %Linear, ptr %field_val95, i32 0, i32 0
  %field_val97 = load ptr, ptr %unreg_field_096, align 8
  call void @tl_mem_unregister(ptr %field_val97)
  %unreg_field_198 = getelementptr inbounds %Linear, ptr %field_val95, i32 0, i32 1
  %field_val99 = load ptr, ptr %unreg_field_198, align 8
  call void @tl_mem_unregister(ptr %field_val99)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val82, i32 0, i32 3
  %field_val100 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val100)
  %unreg_field_0101 = getelementptr inbounds %Linear, ptr %field_val100, i32 0, i32 0
  %field_val102 = load ptr, ptr %unreg_field_0101, align 8
  call void @tl_mem_unregister(ptr %field_val102)
  %unreg_field_1103 = getelementptr inbounds %Linear, ptr %field_val100, i32 0, i32 1
  %field_val104 = load ptr, ptr %unreg_field_1103, align 8
  call void @tl_mem_unregister(ptr %field_val104)
  %unreg_field_2105 = getelementptr inbounds %Block, ptr %s77, i32 0, i32 2
  %field_val106 = load ptr, ptr %unreg_field_2105, align 8
  call void @tl_mem_unregister(ptr %field_val106)
  %unreg_field_0107 = getelementptr inbounds %LayerNorm, ptr %field_val106, i32 0, i32 0
  %field_val108 = load ptr, ptr %unreg_field_0107, align 8
  call void @tl_mem_unregister(ptr %field_val108)
  %unreg_field_1109 = getelementptr inbounds %LayerNorm, ptr %field_val106, i32 0, i32 1
  %field_val110 = load ptr, ptr %unreg_field_1109, align 8
  call void @tl_mem_unregister(ptr %field_val110)
  %unreg_field_3111 = getelementptr inbounds %Block, ptr %s77, i32 0, i32 3
  %field_val112 = load ptr, ptr %unreg_field_3111, align 8
  call void @tl_mem_unregister(ptr %field_val112)
  %unreg_field_0113 = getelementptr inbounds %MLP, ptr %field_val112, i32 0, i32 0
  %field_val114 = load ptr, ptr %unreg_field_0113, align 8
  call void @tl_mem_unregister(ptr %field_val114)
  %unreg_field_0115 = getelementptr inbounds %Linear, ptr %field_val114, i32 0, i32 0
  %field_val116 = load ptr, ptr %unreg_field_0115, align 8
  call void @tl_mem_unregister(ptr %field_val116)
  %unreg_field_1117 = getelementptr inbounds %Linear, ptr %field_val114, i32 0, i32 1
  %field_val118 = load ptr, ptr %unreg_field_1117, align 8
  call void @tl_mem_unregister(ptr %field_val118)
  %unreg_field_1119 = getelementptr inbounds %MLP, ptr %field_val112, i32 0, i32 1
  %field_val120 = load ptr, ptr %unreg_field_1119, align 8
  call void @tl_mem_unregister(ptr %field_val120)
  %unreg_field_0121 = getelementptr inbounds %Linear, ptr %field_val120, i32 0, i32 0
  %field_val122 = load ptr, ptr %unreg_field_0121, align 8
  call void @tl_mem_unregister(ptr %field_val122)
  %unreg_field_1123 = getelementptr inbounds %Linear, ptr %field_val120, i32 0, i32 1
  %field_val124 = load ptr, ptr %unreg_field_1123, align 8
  call void @tl_mem_unregister(ptr %field_val124)
  call void @tl_mem_exit_scope()
  ret ptr %s77
}

define ptr @tl_Block_clone(ptr %self) {
entry:
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Block, ptr null, i32 1) to i64))
  %self2 = load ptr, ptr %self1, align 8
  %ptr_l1 = getelementptr inbounds %Block, ptr %self2, i32 0, i32 0
  %l1 = load ptr, ptr %ptr_l1, align 8
  %call_method = call ptr @tl_LayerNorm_clone(ptr %l1)
  call void @tl_mem_register_struct(ptr %call_method)
  %init_field = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 0
  store ptr %call_method, ptr %init_field, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_a = getelementptr inbounds %Block, ptr %self3, i32 0, i32 1
  %a = load ptr, ptr %ptr_a, align 8
  %call_method4 = call ptr @tl_CausalSelfAttention_clone(ptr %a)
  call void @tl_mem_register_struct(ptr %call_method4)
  %init_field5 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 1
  store ptr %call_method4, ptr %init_field5, align 8
  %self6 = load ptr, ptr %self1, align 8
  %ptr_l2 = getelementptr inbounds %Block, ptr %self6, i32 0, i32 2
  %l2 = load ptr, ptr %ptr_l2, align 8
  %call_method7 = call ptr @tl_LayerNorm_clone(ptr %l2)
  call void @tl_mem_register_struct(ptr %call_method7)
  %init_field8 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 2
  store ptr %call_method7, ptr %init_field8, align 8
  %self9 = load ptr, ptr %self1, align 8
  %ptr_m = getelementptr inbounds %Block, ptr %self9, i32 0, i32 3
  %m = load ptr, ptr %ptr_m, align 8
  %call_method10 = call ptr @tl_MLP_clone(ptr %m)
  call void @tl_mem_register_struct(ptr %call_method10)
  %init_field11 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 3
  store ptr %call_method10, ptr %init_field11, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_012 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 0
  %field_val13 = load ptr, ptr %unreg_field_012, align 8
  call void @tl_mem_unregister(ptr %field_val13)
  %unreg_field_1 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 1
  %field_val14 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val14)
  %unreg_field_115 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 1
  %field_val16 = load ptr, ptr %unreg_field_115, align 8
  call void @tl_mem_unregister(ptr %field_val16)
  %unreg_field_017 = getelementptr inbounds %CausalSelfAttention, ptr %field_val16, i32 0, i32 0
  %field_val18 = load ptr, ptr %unreg_field_017, align 8
  call void @tl_mem_unregister(ptr %field_val18)
  %unreg_field_019 = getelementptr inbounds %Linear, ptr %field_val18, i32 0, i32 0
  %field_val20 = load ptr, ptr %unreg_field_019, align 8
  call void @tl_mem_unregister(ptr %field_val20)
  %unreg_field_121 = getelementptr inbounds %Linear, ptr %field_val18, i32 0, i32 1
  %field_val22 = load ptr, ptr %unreg_field_121, align 8
  call void @tl_mem_unregister(ptr %field_val22)
  %unreg_field_123 = getelementptr inbounds %CausalSelfAttention, ptr %field_val16, i32 0, i32 1
  %field_val24 = load ptr, ptr %unreg_field_123, align 8
  call void @tl_mem_unregister(ptr %field_val24)
  %unreg_field_025 = getelementptr inbounds %Linear, ptr %field_val24, i32 0, i32 0
  %field_val26 = load ptr, ptr %unreg_field_025, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  %unreg_field_127 = getelementptr inbounds %Linear, ptr %field_val24, i32 0, i32 1
  %field_val28 = load ptr, ptr %unreg_field_127, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %field_val16, i32 0, i32 2
  %field_val29 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val29)
  %unreg_field_030 = getelementptr inbounds %Linear, ptr %field_val29, i32 0, i32 0
  %field_val31 = load ptr, ptr %unreg_field_030, align 8
  call void @tl_mem_unregister(ptr %field_val31)
  %unreg_field_132 = getelementptr inbounds %Linear, ptr %field_val29, i32 0, i32 1
  %field_val33 = load ptr, ptr %unreg_field_132, align 8
  call void @tl_mem_unregister(ptr %field_val33)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val16, i32 0, i32 3
  %field_val34 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  %unreg_field_035 = getelementptr inbounds %Linear, ptr %field_val34, i32 0, i32 0
  %field_val36 = load ptr, ptr %unreg_field_035, align 8
  call void @tl_mem_unregister(ptr %field_val36)
  %unreg_field_137 = getelementptr inbounds %Linear, ptr %field_val34, i32 0, i32 1
  %field_val38 = load ptr, ptr %unreg_field_137, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_239 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 2
  %field_val40 = load ptr, ptr %unreg_field_239, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_041 = getelementptr inbounds %LayerNorm, ptr %field_val40, i32 0, i32 0
  %field_val42 = load ptr, ptr %unreg_field_041, align 8
  call void @tl_mem_unregister(ptr %field_val42)
  %unreg_field_143 = getelementptr inbounds %LayerNorm, ptr %field_val40, i32 0, i32 1
  %field_val44 = load ptr, ptr %unreg_field_143, align 8
  call void @tl_mem_unregister(ptr %field_val44)
  %unreg_field_345 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 3
  %field_val46 = load ptr, ptr %unreg_field_345, align 8
  call void @tl_mem_unregister(ptr %field_val46)
  %unreg_field_047 = getelementptr inbounds %MLP, ptr %field_val46, i32 0, i32 0
  %field_val48 = load ptr, ptr %unreg_field_047, align 8
  call void @tl_mem_unregister(ptr %field_val48)
  %unreg_field_049 = getelementptr inbounds %Linear, ptr %field_val48, i32 0, i32 0
  %field_val50 = load ptr, ptr %unreg_field_049, align 8
  call void @tl_mem_unregister(ptr %field_val50)
  %unreg_field_151 = getelementptr inbounds %Linear, ptr %field_val48, i32 0, i32 1
  %field_val52 = load ptr, ptr %unreg_field_151, align 8
  call void @tl_mem_unregister(ptr %field_val52)
  %unreg_field_153 = getelementptr inbounds %MLP, ptr %field_val46, i32 0, i32 1
  %field_val54 = load ptr, ptr %unreg_field_153, align 8
  call void @tl_mem_unregister(ptr %field_val54)
  %unreg_field_055 = getelementptr inbounds %Linear, ptr %field_val54, i32 0, i32 0
  %field_val56 = load ptr, ptr %unreg_field_055, align 8
  call void @tl_mem_unregister(ptr %field_val56)
  %unreg_field_157 = getelementptr inbounds %Linear, ptr %field_val54, i32 0, i32 1
  %field_val58 = load ptr, ptr %unreg_field_157, align 8
  call void @tl_mem_unregister(ptr %field_val58)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
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
  %init_field = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d5 = load i64, ptr %d2, align 8
  %static_call6 = call ptr @tl_Embedding_new(i64 4, i64 %d5)
  %init_field7 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call6, ptr %init_field7, align 8
  %d8 = load i64, ptr %d2, align 8
  %static_call9 = call ptr @tl_Block_new(i64 %d8)
  %init_field10 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 2
  store ptr %static_call9, ptr %init_field10, align 8
  %d11 = load i64, ptr %d2, align 8
  %static_call12 = call ptr @tl_Block_new(i64 %d11)
  %init_field13 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 3
  store ptr %static_call12, ptr %init_field13, align 8
  %d14 = load i64, ptr %d2, align 8
  %static_call15 = call ptr @tl_LayerNorm_new(i64 %d14)
  %init_field16 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 4
  store ptr %static_call15, ptr %init_field16, align 8
  %d17 = load i64, ptr %d2, align 8
  %v18 = load i64, ptr %v1, align 8
  %static_call19 = call ptr @tl_Linear_new(i64 %d17, i64 %v18)
  %init_field20 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 5
  store ptr %static_call19, ptr %init_field20, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_021 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val22 = load ptr, ptr %unreg_field_021, align 8
  call void @tl_mem_unregister(ptr %field_val22)
  %unreg_field_1 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 1
  %field_val23 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val23)
  %unreg_field_024 = getelementptr inbounds %Embedding, ptr %field_val23, i32 0, i32 0
  %field_val25 = load ptr, ptr %unreg_field_024, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_2 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 2
  %field_val26 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  %unreg_field_027 = getelementptr inbounds %Block, ptr %field_val26, i32 0, i32 0
  %field_val28 = load ptr, ptr %unreg_field_027, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_029 = getelementptr inbounds %LayerNorm, ptr %field_val28, i32 0, i32 0
  %field_val30 = load ptr, ptr %unreg_field_029, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  %unreg_field_131 = getelementptr inbounds %LayerNorm, ptr %field_val28, i32 0, i32 1
  %field_val32 = load ptr, ptr %unreg_field_131, align 8
  call void @tl_mem_unregister(ptr %field_val32)
  %unreg_field_133 = getelementptr inbounds %Block, ptr %field_val26, i32 0, i32 1
  %field_val34 = load ptr, ptr %unreg_field_133, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  %unreg_field_035 = getelementptr inbounds %CausalSelfAttention, ptr %field_val34, i32 0, i32 0
  %field_val36 = load ptr, ptr %unreg_field_035, align 8
  call void @tl_mem_unregister(ptr %field_val36)
  %unreg_field_037 = getelementptr inbounds %Linear, ptr %field_val36, i32 0, i32 0
  %field_val38 = load ptr, ptr %unreg_field_037, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_139 = getelementptr inbounds %Linear, ptr %field_val36, i32 0, i32 1
  %field_val40 = load ptr, ptr %unreg_field_139, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_141 = getelementptr inbounds %CausalSelfAttention, ptr %field_val34, i32 0, i32 1
  %field_val42 = load ptr, ptr %unreg_field_141, align 8
  call void @tl_mem_unregister(ptr %field_val42)
  %unreg_field_043 = getelementptr inbounds %Linear, ptr %field_val42, i32 0, i32 0
  %field_val44 = load ptr, ptr %unreg_field_043, align 8
  call void @tl_mem_unregister(ptr %field_val44)
  %unreg_field_145 = getelementptr inbounds %Linear, ptr %field_val42, i32 0, i32 1
  %field_val46 = load ptr, ptr %unreg_field_145, align 8
  call void @tl_mem_unregister(ptr %field_val46)
  %unreg_field_247 = getelementptr inbounds %CausalSelfAttention, ptr %field_val34, i32 0, i32 2
  %field_val48 = load ptr, ptr %unreg_field_247, align 8
  call void @tl_mem_unregister(ptr %field_val48)
  %unreg_field_049 = getelementptr inbounds %Linear, ptr %field_val48, i32 0, i32 0
  %field_val50 = load ptr, ptr %unreg_field_049, align 8
  call void @tl_mem_unregister(ptr %field_val50)
  %unreg_field_151 = getelementptr inbounds %Linear, ptr %field_val48, i32 0, i32 1
  %field_val52 = load ptr, ptr %unreg_field_151, align 8
  call void @tl_mem_unregister(ptr %field_val52)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val34, i32 0, i32 3
  %field_val53 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val53)
  %unreg_field_054 = getelementptr inbounds %Linear, ptr %field_val53, i32 0, i32 0
  %field_val55 = load ptr, ptr %unreg_field_054, align 8
  call void @tl_mem_unregister(ptr %field_val55)
  %unreg_field_156 = getelementptr inbounds %Linear, ptr %field_val53, i32 0, i32 1
  %field_val57 = load ptr, ptr %unreg_field_156, align 8
  call void @tl_mem_unregister(ptr %field_val57)
  %unreg_field_258 = getelementptr inbounds %Block, ptr %field_val26, i32 0, i32 2
  %field_val59 = load ptr, ptr %unreg_field_258, align 8
  call void @tl_mem_unregister(ptr %field_val59)
  %unreg_field_060 = getelementptr inbounds %LayerNorm, ptr %field_val59, i32 0, i32 0
  %field_val61 = load ptr, ptr %unreg_field_060, align 8
  call void @tl_mem_unregister(ptr %field_val61)
  %unreg_field_162 = getelementptr inbounds %LayerNorm, ptr %field_val59, i32 0, i32 1
  %field_val63 = load ptr, ptr %unreg_field_162, align 8
  call void @tl_mem_unregister(ptr %field_val63)
  %unreg_field_364 = getelementptr inbounds %Block, ptr %field_val26, i32 0, i32 3
  %field_val65 = load ptr, ptr %unreg_field_364, align 8
  call void @tl_mem_unregister(ptr %field_val65)
  %unreg_field_066 = getelementptr inbounds %MLP, ptr %field_val65, i32 0, i32 0
  %field_val67 = load ptr, ptr %unreg_field_066, align 8
  call void @tl_mem_unregister(ptr %field_val67)
  %unreg_field_068 = getelementptr inbounds %Linear, ptr %field_val67, i32 0, i32 0
  %field_val69 = load ptr, ptr %unreg_field_068, align 8
  call void @tl_mem_unregister(ptr %field_val69)
  %unreg_field_170 = getelementptr inbounds %Linear, ptr %field_val67, i32 0, i32 1
  %field_val71 = load ptr, ptr %unreg_field_170, align 8
  call void @tl_mem_unregister(ptr %field_val71)
  %unreg_field_172 = getelementptr inbounds %MLP, ptr %field_val65, i32 0, i32 1
  %field_val73 = load ptr, ptr %unreg_field_172, align 8
  call void @tl_mem_unregister(ptr %field_val73)
  %unreg_field_074 = getelementptr inbounds %Linear, ptr %field_val73, i32 0, i32 0
  %field_val75 = load ptr, ptr %unreg_field_074, align 8
  call void @tl_mem_unregister(ptr %field_val75)
  %unreg_field_176 = getelementptr inbounds %Linear, ptr %field_val73, i32 0, i32 1
  %field_val77 = load ptr, ptr %unreg_field_176, align 8
  call void @tl_mem_unregister(ptr %field_val77)
  %unreg_field_378 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 3
  %field_val79 = load ptr, ptr %unreg_field_378, align 8
  call void @tl_mem_unregister(ptr %field_val79)
  %unreg_field_080 = getelementptr inbounds %Block, ptr %field_val79, i32 0, i32 0
  %field_val81 = load ptr, ptr %unreg_field_080, align 8
  call void @tl_mem_unregister(ptr %field_val81)
  %unreg_field_082 = getelementptr inbounds %LayerNorm, ptr %field_val81, i32 0, i32 0
  %field_val83 = load ptr, ptr %unreg_field_082, align 8
  call void @tl_mem_unregister(ptr %field_val83)
  %unreg_field_184 = getelementptr inbounds %LayerNorm, ptr %field_val81, i32 0, i32 1
  %field_val85 = load ptr, ptr %unreg_field_184, align 8
  call void @tl_mem_unregister(ptr %field_val85)
  %unreg_field_186 = getelementptr inbounds %Block, ptr %field_val79, i32 0, i32 1
  %field_val87 = load ptr, ptr %unreg_field_186, align 8
  call void @tl_mem_unregister(ptr %field_val87)
  %unreg_field_088 = getelementptr inbounds %CausalSelfAttention, ptr %field_val87, i32 0, i32 0
  %field_val89 = load ptr, ptr %unreg_field_088, align 8
  call void @tl_mem_unregister(ptr %field_val89)
  %unreg_field_090 = getelementptr inbounds %Linear, ptr %field_val89, i32 0, i32 0
  %field_val91 = load ptr, ptr %unreg_field_090, align 8
  call void @tl_mem_unregister(ptr %field_val91)
  %unreg_field_192 = getelementptr inbounds %Linear, ptr %field_val89, i32 0, i32 1
  %field_val93 = load ptr, ptr %unreg_field_192, align 8
  call void @tl_mem_unregister(ptr %field_val93)
  %unreg_field_194 = getelementptr inbounds %CausalSelfAttention, ptr %field_val87, i32 0, i32 1
  %field_val95 = load ptr, ptr %unreg_field_194, align 8
  call void @tl_mem_unregister(ptr %field_val95)
  %unreg_field_096 = getelementptr inbounds %Linear, ptr %field_val95, i32 0, i32 0
  %field_val97 = load ptr, ptr %unreg_field_096, align 8
  call void @tl_mem_unregister(ptr %field_val97)
  %unreg_field_198 = getelementptr inbounds %Linear, ptr %field_val95, i32 0, i32 1
  %field_val99 = load ptr, ptr %unreg_field_198, align 8
  call void @tl_mem_unregister(ptr %field_val99)
  %unreg_field_2100 = getelementptr inbounds %CausalSelfAttention, ptr %field_val87, i32 0, i32 2
  %field_val101 = load ptr, ptr %unreg_field_2100, align 8
  call void @tl_mem_unregister(ptr %field_val101)
  %unreg_field_0102 = getelementptr inbounds %Linear, ptr %field_val101, i32 0, i32 0
  %field_val103 = load ptr, ptr %unreg_field_0102, align 8
  call void @tl_mem_unregister(ptr %field_val103)
  %unreg_field_1104 = getelementptr inbounds %Linear, ptr %field_val101, i32 0, i32 1
  %field_val105 = load ptr, ptr %unreg_field_1104, align 8
  call void @tl_mem_unregister(ptr %field_val105)
  %unreg_field_3106 = getelementptr inbounds %CausalSelfAttention, ptr %field_val87, i32 0, i32 3
  %field_val107 = load ptr, ptr %unreg_field_3106, align 8
  call void @tl_mem_unregister(ptr %field_val107)
  %unreg_field_0108 = getelementptr inbounds %Linear, ptr %field_val107, i32 0, i32 0
  %field_val109 = load ptr, ptr %unreg_field_0108, align 8
  call void @tl_mem_unregister(ptr %field_val109)
  %unreg_field_1110 = getelementptr inbounds %Linear, ptr %field_val107, i32 0, i32 1
  %field_val111 = load ptr, ptr %unreg_field_1110, align 8
  call void @tl_mem_unregister(ptr %field_val111)
  %unreg_field_2112 = getelementptr inbounds %Block, ptr %field_val79, i32 0, i32 2
  %field_val113 = load ptr, ptr %unreg_field_2112, align 8
  call void @tl_mem_unregister(ptr %field_val113)
  %unreg_field_0114 = getelementptr inbounds %LayerNorm, ptr %field_val113, i32 0, i32 0
  %field_val115 = load ptr, ptr %unreg_field_0114, align 8
  call void @tl_mem_unregister(ptr %field_val115)
  %unreg_field_1116 = getelementptr inbounds %LayerNorm, ptr %field_val113, i32 0, i32 1
  %field_val117 = load ptr, ptr %unreg_field_1116, align 8
  call void @tl_mem_unregister(ptr %field_val117)
  %unreg_field_3118 = getelementptr inbounds %Block, ptr %field_val79, i32 0, i32 3
  %field_val119 = load ptr, ptr %unreg_field_3118, align 8
  call void @tl_mem_unregister(ptr %field_val119)
  %unreg_field_0120 = getelementptr inbounds %MLP, ptr %field_val119, i32 0, i32 0
  %field_val121 = load ptr, ptr %unreg_field_0120, align 8
  call void @tl_mem_unregister(ptr %field_val121)
  %unreg_field_0122 = getelementptr inbounds %Linear, ptr %field_val121, i32 0, i32 0
  %field_val123 = load ptr, ptr %unreg_field_0122, align 8
  call void @tl_mem_unregister(ptr %field_val123)
  %unreg_field_1124 = getelementptr inbounds %Linear, ptr %field_val121, i32 0, i32 1
  %field_val125 = load ptr, ptr %unreg_field_1124, align 8
  call void @tl_mem_unregister(ptr %field_val125)
  %unreg_field_1126 = getelementptr inbounds %MLP, ptr %field_val119, i32 0, i32 1
  %field_val127 = load ptr, ptr %unreg_field_1126, align 8
  call void @tl_mem_unregister(ptr %field_val127)
  %unreg_field_0128 = getelementptr inbounds %Linear, ptr %field_val127, i32 0, i32 0
  %field_val129 = load ptr, ptr %unreg_field_0128, align 8
  call void @tl_mem_unregister(ptr %field_val129)
  %unreg_field_1130 = getelementptr inbounds %Linear, ptr %field_val127, i32 0, i32 1
  %field_val131 = load ptr, ptr %unreg_field_1130, align 8
  call void @tl_mem_unregister(ptr %field_val131)
  %unreg_field_4 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 4
  %field_val132 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val132)
  %unreg_field_0133 = getelementptr inbounds %LayerNorm, ptr %field_val132, i32 0, i32 0
  %field_val134 = load ptr, ptr %unreg_field_0133, align 8
  call void @tl_mem_unregister(ptr %field_val134)
  %unreg_field_1135 = getelementptr inbounds %LayerNorm, ptr %field_val132, i32 0, i32 1
  %field_val136 = load ptr, ptr %unreg_field_1135, align 8
  call void @tl_mem_unregister(ptr %field_val136)
  %unreg_field_5 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 5
  %field_val137 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val137)
  %unreg_field_0138 = getelementptr inbounds %Linear, ptr %field_val137, i32 0, i32 0
  %field_val139 = load ptr, ptr %unreg_field_0138, align 8
  call void @tl_mem_unregister(ptr %field_val139)
  %unreg_field_1140 = getelementptr inbounds %Linear, ptr %field_val137, i32 0, i32 1
  %field_val141 = load ptr, ptr %unreg_field_1140, align 8
  call void @tl_mem_unregister(ptr %field_val141)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_GPT_forward(ptr %self, ptr %i) {
entry:
  %x20 = alloca ptr, align 16
  %x15 = alloca ptr, align 16
  %x = alloca ptr, align 16
  %pos_emb = alloca ptr, align 16
  %tok_emb = alloca ptr, align 16
  %pos = alloca ptr, align 16
  %dims_alloca = alloca [2 x i64], align 8
  %pos_data = alloca ptr, align 16
  %i2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store ptr %i, ptr %i2, align 8
  %arr_malloc = call ptr @malloc(i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 4))
  call void @tl_mem_register_struct(ptr %arr_malloc)
  %elem_ptr = getelementptr inbounds float, ptr %arr_malloc, i64 0
  store float 0.000000e+00, ptr %elem_ptr, align 4
  %elem_ptr3 = getelementptr inbounds float, ptr %arr_malloc, i64 1
  store float 1.000000e+00, ptr %elem_ptr3, align 4
  %elem_ptr4 = getelementptr inbounds float, ptr %arr_malloc, i64 2
  store float 2.000000e+00, ptr %elem_ptr4, align 4
  %elem_ptr5 = getelementptr inbounds float, ptr %arr_malloc, i64 3
  store float 3.000000e+00, ptr %elem_ptr5, align 4
  store ptr %arr_malloc, ptr %pos_data, align 8
  %pos_data_ptr = load ptr, ptr %pos_data, align 8
  %shape_arr = alloca [1 x i64], align 8
  %shape_ptr = getelementptr inbounds [1 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 4, ptr %shape_ptr, align 8
  %converted_tensor = call ptr @tl_tensor_new(ptr %pos_data_ptr, i64 1, ptr %shape_arr)
  %dim_ptr_0 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0, align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 4, ptr %dim_ptr, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %converted_tensor, ptr %dims_ptr, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res)
  store ptr %reshape_dims_res, ptr %pos, align 8
  %self6 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds %GPT, ptr %self6, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %i7 = load ptr, ptr %i2, align 8
  %call_method = call ptr @tl_Embedding_forward(ptr %w, ptr %i7)
  call void @tl_mem_register_tensor(ptr %call_method)
  call void @tl_mem_unregister(ptr %call_method)
  store ptr %call_method, ptr %tok_emb, align 8
  %self8 = load ptr, ptr %self1, align 8
  %ptr_wp = getelementptr inbounds %GPT, ptr %self8, i32 0, i32 1
  %wp = load ptr, ptr %ptr_wp, align 8
  %pos9 = load ptr, ptr %pos, align 8
  %call_method10 = call ptr @tl_Embedding_forward(ptr %wp, ptr %pos9)
  call void @tl_mem_register_tensor(ptr %call_method10)
  call void @tl_mem_unregister(ptr %call_method10)
  store ptr %call_method10, ptr %pos_emb, align 8
  %tok_emb11 = load ptr, ptr %tok_emb, align 8
  %pos_emb12 = load ptr, ptr %pos_emb, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %tok_emb11, ptr %pos_emb12)
  call void @tl_mem_register_tensor(ptr %binop_res)
  call void @tl_mem_unregister(ptr %binop_res)
  store ptr %binop_res, ptr %x, align 8
  %self13 = load ptr, ptr %self1, align 8
  %ptr_b1 = getelementptr inbounds %GPT, ptr %self13, i32 0, i32 2
  %b1 = load ptr, ptr %ptr_b1, align 8
  %x14 = load ptr, ptr %x, align 8
  %checkpoint_res = call ptr @tl_checkpoint(ptr %b1, ptr @tl_Block_forward, ptr %x14)
  call void @tl_mem_register_tensor(ptr %checkpoint_res)
  call void @tl_mem_unregister(ptr %checkpoint_res)
  %old_shadowed = load ptr, ptr %x, align 8
  call void @tl_mem_unregister(ptr %old_shadowed)
  store ptr %checkpoint_res, ptr %x15, align 8
  %self16 = load ptr, ptr %self1, align 8
  %ptr_b2 = getelementptr inbounds %GPT, ptr %self16, i32 0, i32 3
  %b2 = load ptr, ptr %ptr_b2, align 8
  %x17 = load ptr, ptr %x15, align 8
  %checkpoint_res18 = call ptr @tl_checkpoint(ptr %b2, ptr @tl_Block_forward, ptr %x17)
  call void @tl_mem_register_tensor(ptr %checkpoint_res18)
  call void @tl_mem_unregister(ptr %checkpoint_res18)
  %old_shadowed19 = load ptr, ptr %x15, align 8
  call void @tl_mem_unregister(ptr %old_shadowed19)
  store ptr %checkpoint_res18, ptr %x20, align 8
  %self21 = load ptr, ptr %self1, align 8
  %ptr_h = getelementptr inbounds %GPT, ptr %self21, i32 0, i32 5
  %h = load ptr, ptr %ptr_h, align 8
  %self22 = load ptr, ptr %self1, align 8
  %ptr_l = getelementptr inbounds %GPT, ptr %self22, i32 0, i32 4
  %l = load ptr, ptr %ptr_l, align 8
  %x23 = load ptr, ptr %x20, align 8
  %call_method24 = call ptr @tl_LayerNorm_forward(ptr %l, ptr %x23)
  call void @tl_mem_register_tensor(ptr %call_method24)
  %call_method25 = call ptr @tl_Linear_forward(ptr %h, ptr %call_method24)
  call void @tl_tensor_free(ptr %call_method24)
  call void @tl_mem_register_tensor(ptr %call_method25)
  call void @tl_mem_unregister(ptr %call_method25)
  %tensor_to_free = load ptr, ptr %x20, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free26 = load ptr, ptr %pos_emb, align 8
  call void @tl_tensor_free(ptr %tensor_to_free26)
  %tensor_to_free27 = load ptr, ptr %pos, align 8
  call void @tl_tensor_free(ptr %tensor_to_free27)
  %tensor_to_free28 = load ptr, ptr %tok_emb, align 8
  call void @tl_tensor_free(ptr %tensor_to_free28)
  call void @tl_mem_exit_scope()
  ret ptr %call_method25
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
  %ptr_w = getelementptr inbounds %GPT, ptr %s4, i32 0, i32 0
  %s5 = load ptr, ptr %s, align 8
  %ptr_w6 = getelementptr inbounds %GPT, ptr %s5, i32 0, i32 0
  %w = load ptr, ptr %ptr_w6, align 8
  %lr7 = load float, ptr %lr2, align 4
  %call_method = call ptr @tl_Embedding_step(ptr %w, float %lr7)
  call void @tl_mem_register_struct(ptr %call_method)
  %old_field_val = load ptr, ptr %ptr_w, align 8
  %cnt_free_diff = icmp ne ptr %old_field_val, %call_method
  br i1 %cnt_free_diff, label %free_old_val, label %skip_free

free_old_val:                                     ; preds = %entry
  %field_gep = getelementptr inbounds %Embedding, ptr %old_field_val, i32 0, i32 0
  %field_load = load ptr, ptr %field_gep, align 8
  call void @tl_tensor_free(ptr %field_load)
  br label %skip_free

skip_free:                                        ; preds = %free_old_val, %entry
  store ptr %call_method, ptr %ptr_w, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s8 = load ptr, ptr %s, align 8
  %ptr_wp = getelementptr inbounds %GPT, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_wp10 = getelementptr inbounds %GPT, ptr %s9, i32 0, i32 1
  %wp = load ptr, ptr %ptr_wp10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_Embedding_step(ptr %wp, float %lr11)
  call void @tl_mem_register_struct(ptr %call_method12)
  %old_field_val13 = load ptr, ptr %ptr_wp, align 8
  %cnt_free_diff14 = icmp ne ptr %old_field_val13, %call_method12
  br i1 %cnt_free_diff14, label %free_old_val15, label %skip_free16

free_old_val15:                                   ; preds = %skip_free
  %field_gep17 = getelementptr inbounds %Embedding, ptr %old_field_val13, i32 0, i32 0
  %field_load18 = load ptr, ptr %field_gep17, align 8
  call void @tl_tensor_free(ptr %field_load18)
  br label %skip_free16

skip_free16:                                      ; preds = %free_old_val15, %skip_free
  store ptr %call_method12, ptr %ptr_wp, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s19 = load ptr, ptr %s, align 8
  %ptr_b1 = getelementptr inbounds %GPT, ptr %s19, i32 0, i32 2
  %s20 = load ptr, ptr %s, align 8
  %ptr_b121 = getelementptr inbounds %GPT, ptr %s20, i32 0, i32 2
  %b1 = load ptr, ptr %ptr_b121, align 8
  %lr22 = load float, ptr %lr2, align 4
  %call_method23 = call ptr @tl_Block_step(ptr %b1, float %lr22)
  call void @tl_mem_register_struct(ptr %call_method23)
  %old_field_val24 = load ptr, ptr %ptr_b1, align 8
  %cnt_free_diff25 = icmp ne ptr %old_field_val24, %call_method23
  br i1 %cnt_free_diff25, label %free_old_val26, label %skip_free27

free_old_val26:                                   ; preds = %skip_free16
  %field_gep28 = getelementptr inbounds %Block, ptr %old_field_val24, i32 0, i32 0
  %field_load29 = load ptr, ptr %field_gep28, align 8
  %field_gep30 = getelementptr inbounds %LayerNorm, ptr %field_load29, i32 0, i32 0
  %field_load31 = load ptr, ptr %field_gep30, align 8
  call void @tl_tensor_free(ptr %field_load31)
  %field_gep32 = getelementptr inbounds %LayerNorm, ptr %field_load29, i32 0, i32 1
  %field_load33 = load ptr, ptr %field_gep32, align 8
  call void @tl_tensor_free(ptr %field_load33)
  %field_gep34 = getelementptr inbounds %Block, ptr %old_field_val24, i32 0, i32 1
  %field_load35 = load ptr, ptr %field_gep34, align 8
  %field_gep36 = getelementptr inbounds %CausalSelfAttention, ptr %field_load35, i32 0, i32 0
  %field_load37 = load ptr, ptr %field_gep36, align 8
  %field_gep38 = getelementptr inbounds %Linear, ptr %field_load37, i32 0, i32 0
  %field_load39 = load ptr, ptr %field_gep38, align 8
  call void @tl_tensor_free(ptr %field_load39)
  %field_gep40 = getelementptr inbounds %Linear, ptr %field_load37, i32 0, i32 1
  %field_load41 = load ptr, ptr %field_gep40, align 8
  call void @tl_tensor_free(ptr %field_load41)
  %field_gep42 = getelementptr inbounds %CausalSelfAttention, ptr %field_load35, i32 0, i32 1
  %field_load43 = load ptr, ptr %field_gep42, align 8
  %field_gep44 = getelementptr inbounds %Linear, ptr %field_load43, i32 0, i32 0
  %field_load45 = load ptr, ptr %field_gep44, align 8
  call void @tl_tensor_free(ptr %field_load45)
  %field_gep46 = getelementptr inbounds %Linear, ptr %field_load43, i32 0, i32 1
  %field_load47 = load ptr, ptr %field_gep46, align 8
  call void @tl_tensor_free(ptr %field_load47)
  %field_gep48 = getelementptr inbounds %CausalSelfAttention, ptr %field_load35, i32 0, i32 2
  %field_load49 = load ptr, ptr %field_gep48, align 8
  %field_gep50 = getelementptr inbounds %Linear, ptr %field_load49, i32 0, i32 0
  %field_load51 = load ptr, ptr %field_gep50, align 8
  call void @tl_tensor_free(ptr %field_load51)
  %field_gep52 = getelementptr inbounds %Linear, ptr %field_load49, i32 0, i32 1
  %field_load53 = load ptr, ptr %field_gep52, align 8
  call void @tl_tensor_free(ptr %field_load53)
  %field_gep54 = getelementptr inbounds %CausalSelfAttention, ptr %field_load35, i32 0, i32 3
  %field_load55 = load ptr, ptr %field_gep54, align 8
  %field_gep56 = getelementptr inbounds %Linear, ptr %field_load55, i32 0, i32 0
  %field_load57 = load ptr, ptr %field_gep56, align 8
  call void @tl_tensor_free(ptr %field_load57)
  %field_gep58 = getelementptr inbounds %Linear, ptr %field_load55, i32 0, i32 1
  %field_load59 = load ptr, ptr %field_gep58, align 8
  call void @tl_tensor_free(ptr %field_load59)
  %field_gep60 = getelementptr inbounds %Block, ptr %old_field_val24, i32 0, i32 2
  %field_load61 = load ptr, ptr %field_gep60, align 8
  %field_gep62 = getelementptr inbounds %LayerNorm, ptr %field_load61, i32 0, i32 0
  %field_load63 = load ptr, ptr %field_gep62, align 8
  call void @tl_tensor_free(ptr %field_load63)
  %field_gep64 = getelementptr inbounds %LayerNorm, ptr %field_load61, i32 0, i32 1
  %field_load65 = load ptr, ptr %field_gep64, align 8
  call void @tl_tensor_free(ptr %field_load65)
  %field_gep66 = getelementptr inbounds %Block, ptr %old_field_val24, i32 0, i32 3
  %field_load67 = load ptr, ptr %field_gep66, align 8
  %field_gep68 = getelementptr inbounds %MLP, ptr %field_load67, i32 0, i32 0
  %field_load69 = load ptr, ptr %field_gep68, align 8
  %field_gep70 = getelementptr inbounds %Linear, ptr %field_load69, i32 0, i32 0
  %field_load71 = load ptr, ptr %field_gep70, align 8
  call void @tl_tensor_free(ptr %field_load71)
  %field_gep72 = getelementptr inbounds %Linear, ptr %field_load69, i32 0, i32 1
  %field_load73 = load ptr, ptr %field_gep72, align 8
  call void @tl_tensor_free(ptr %field_load73)
  %field_gep74 = getelementptr inbounds %MLP, ptr %field_load67, i32 0, i32 1
  %field_load75 = load ptr, ptr %field_gep74, align 8
  %field_gep76 = getelementptr inbounds %Linear, ptr %field_load75, i32 0, i32 0
  %field_load77 = load ptr, ptr %field_gep76, align 8
  call void @tl_tensor_free(ptr %field_load77)
  %field_gep78 = getelementptr inbounds %Linear, ptr %field_load75, i32 0, i32 1
  %field_load79 = load ptr, ptr %field_gep78, align 8
  call void @tl_tensor_free(ptr %field_load79)
  br label %skip_free27

skip_free27:                                      ; preds = %free_old_val26, %skip_free16
  store ptr %call_method23, ptr %ptr_b1, align 8
  call void @tl_mem_unregister(ptr %call_method23)
  %s80 = load ptr, ptr %s, align 8
  %ptr_b2 = getelementptr inbounds %GPT, ptr %s80, i32 0, i32 3
  %s81 = load ptr, ptr %s, align 8
  %ptr_b282 = getelementptr inbounds %GPT, ptr %s81, i32 0, i32 3
  %b2 = load ptr, ptr %ptr_b282, align 8
  %lr83 = load float, ptr %lr2, align 4
  %call_method84 = call ptr @tl_Block_step(ptr %b2, float %lr83)
  call void @tl_mem_register_struct(ptr %call_method84)
  %old_field_val85 = load ptr, ptr %ptr_b2, align 8
  %cnt_free_diff86 = icmp ne ptr %old_field_val85, %call_method84
  br i1 %cnt_free_diff86, label %free_old_val87, label %skip_free88

free_old_val87:                                   ; preds = %skip_free27
  %field_gep89 = getelementptr inbounds %Block, ptr %old_field_val85, i32 0, i32 0
  %field_load90 = load ptr, ptr %field_gep89, align 8
  %field_gep91 = getelementptr inbounds %LayerNorm, ptr %field_load90, i32 0, i32 0
  %field_load92 = load ptr, ptr %field_gep91, align 8
  call void @tl_tensor_free(ptr %field_load92)
  %field_gep93 = getelementptr inbounds %LayerNorm, ptr %field_load90, i32 0, i32 1
  %field_load94 = load ptr, ptr %field_gep93, align 8
  call void @tl_tensor_free(ptr %field_load94)
  %field_gep95 = getelementptr inbounds %Block, ptr %old_field_val85, i32 0, i32 1
  %field_load96 = load ptr, ptr %field_gep95, align 8
  %field_gep97 = getelementptr inbounds %CausalSelfAttention, ptr %field_load96, i32 0, i32 0
  %field_load98 = load ptr, ptr %field_gep97, align 8
  %field_gep99 = getelementptr inbounds %Linear, ptr %field_load98, i32 0, i32 0
  %field_load100 = load ptr, ptr %field_gep99, align 8
  call void @tl_tensor_free(ptr %field_load100)
  %field_gep101 = getelementptr inbounds %Linear, ptr %field_load98, i32 0, i32 1
  %field_load102 = load ptr, ptr %field_gep101, align 8
  call void @tl_tensor_free(ptr %field_load102)
  %field_gep103 = getelementptr inbounds %CausalSelfAttention, ptr %field_load96, i32 0, i32 1
  %field_load104 = load ptr, ptr %field_gep103, align 8
  %field_gep105 = getelementptr inbounds %Linear, ptr %field_load104, i32 0, i32 0
  %field_load106 = load ptr, ptr %field_gep105, align 8
  call void @tl_tensor_free(ptr %field_load106)
  %field_gep107 = getelementptr inbounds %Linear, ptr %field_load104, i32 0, i32 1
  %field_load108 = load ptr, ptr %field_gep107, align 8
  call void @tl_tensor_free(ptr %field_load108)
  %field_gep109 = getelementptr inbounds %CausalSelfAttention, ptr %field_load96, i32 0, i32 2
  %field_load110 = load ptr, ptr %field_gep109, align 8
  %field_gep111 = getelementptr inbounds %Linear, ptr %field_load110, i32 0, i32 0
  %field_load112 = load ptr, ptr %field_gep111, align 8
  call void @tl_tensor_free(ptr %field_load112)
  %field_gep113 = getelementptr inbounds %Linear, ptr %field_load110, i32 0, i32 1
  %field_load114 = load ptr, ptr %field_gep113, align 8
  call void @tl_tensor_free(ptr %field_load114)
  %field_gep115 = getelementptr inbounds %CausalSelfAttention, ptr %field_load96, i32 0, i32 3
  %field_load116 = load ptr, ptr %field_gep115, align 8
  %field_gep117 = getelementptr inbounds %Linear, ptr %field_load116, i32 0, i32 0
  %field_load118 = load ptr, ptr %field_gep117, align 8
  call void @tl_tensor_free(ptr %field_load118)
  %field_gep119 = getelementptr inbounds %Linear, ptr %field_load116, i32 0, i32 1
  %field_load120 = load ptr, ptr %field_gep119, align 8
  call void @tl_tensor_free(ptr %field_load120)
  %field_gep121 = getelementptr inbounds %Block, ptr %old_field_val85, i32 0, i32 2
  %field_load122 = load ptr, ptr %field_gep121, align 8
  %field_gep123 = getelementptr inbounds %LayerNorm, ptr %field_load122, i32 0, i32 0
  %field_load124 = load ptr, ptr %field_gep123, align 8
  call void @tl_tensor_free(ptr %field_load124)
  %field_gep125 = getelementptr inbounds %LayerNorm, ptr %field_load122, i32 0, i32 1
  %field_load126 = load ptr, ptr %field_gep125, align 8
  call void @tl_tensor_free(ptr %field_load126)
  %field_gep127 = getelementptr inbounds %Block, ptr %old_field_val85, i32 0, i32 3
  %field_load128 = load ptr, ptr %field_gep127, align 8
  %field_gep129 = getelementptr inbounds %MLP, ptr %field_load128, i32 0, i32 0
  %field_load130 = load ptr, ptr %field_gep129, align 8
  %field_gep131 = getelementptr inbounds %Linear, ptr %field_load130, i32 0, i32 0
  %field_load132 = load ptr, ptr %field_gep131, align 8
  call void @tl_tensor_free(ptr %field_load132)
  %field_gep133 = getelementptr inbounds %Linear, ptr %field_load130, i32 0, i32 1
  %field_load134 = load ptr, ptr %field_gep133, align 8
  call void @tl_tensor_free(ptr %field_load134)
  %field_gep135 = getelementptr inbounds %MLP, ptr %field_load128, i32 0, i32 1
  %field_load136 = load ptr, ptr %field_gep135, align 8
  %field_gep137 = getelementptr inbounds %Linear, ptr %field_load136, i32 0, i32 0
  %field_load138 = load ptr, ptr %field_gep137, align 8
  call void @tl_tensor_free(ptr %field_load138)
  %field_gep139 = getelementptr inbounds %Linear, ptr %field_load136, i32 0, i32 1
  %field_load140 = load ptr, ptr %field_gep139, align 8
  call void @tl_tensor_free(ptr %field_load140)
  br label %skip_free88

skip_free88:                                      ; preds = %free_old_val87, %skip_free27
  store ptr %call_method84, ptr %ptr_b2, align 8
  call void @tl_mem_unregister(ptr %call_method84)
  %s141 = load ptr, ptr %s, align 8
  %ptr_l = getelementptr inbounds %GPT, ptr %s141, i32 0, i32 4
  %s142 = load ptr, ptr %s, align 8
  %ptr_l143 = getelementptr inbounds %GPT, ptr %s142, i32 0, i32 4
  %l = load ptr, ptr %ptr_l143, align 8
  %lr144 = load float, ptr %lr2, align 4
  %call_method145 = call ptr @tl_LayerNorm_step(ptr %l, float %lr144)
  call void @tl_mem_register_struct(ptr %call_method145)
  %old_field_val146 = load ptr, ptr %ptr_l, align 8
  %cnt_free_diff147 = icmp ne ptr %old_field_val146, %call_method145
  br i1 %cnt_free_diff147, label %free_old_val148, label %skip_free149

free_old_val148:                                  ; preds = %skip_free88
  %field_gep150 = getelementptr inbounds %LayerNorm, ptr %old_field_val146, i32 0, i32 0
  %field_load151 = load ptr, ptr %field_gep150, align 8
  call void @tl_tensor_free(ptr %field_load151)
  %field_gep152 = getelementptr inbounds %LayerNorm, ptr %old_field_val146, i32 0, i32 1
  %field_load153 = load ptr, ptr %field_gep152, align 8
  call void @tl_tensor_free(ptr %field_load153)
  br label %skip_free149

skip_free149:                                     ; preds = %free_old_val148, %skip_free88
  store ptr %call_method145, ptr %ptr_l, align 8
  call void @tl_mem_unregister(ptr %call_method145)
  %s154 = load ptr, ptr %s, align 8
  %ptr_h = getelementptr inbounds %GPT, ptr %s154, i32 0, i32 5
  %s155 = load ptr, ptr %s, align 8
  %ptr_h156 = getelementptr inbounds %GPT, ptr %s155, i32 0, i32 5
  %h = load ptr, ptr %ptr_h156, align 8
  %lr157 = load float, ptr %lr2, align 4
  %call_method158 = call ptr @tl_Linear_step(ptr %h, float %lr157)
  call void @tl_mem_register_struct(ptr %call_method158)
  %old_field_val159 = load ptr, ptr %ptr_h, align 8
  %cnt_free_diff160 = icmp ne ptr %old_field_val159, %call_method158
  br i1 %cnt_free_diff160, label %free_old_val161, label %skip_free162

free_old_val161:                                  ; preds = %skip_free149
  %field_gep163 = getelementptr inbounds %Linear, ptr %old_field_val159, i32 0, i32 0
  %field_load164 = load ptr, ptr %field_gep163, align 8
  call void @tl_tensor_free(ptr %field_load164)
  %field_gep165 = getelementptr inbounds %Linear, ptr %old_field_val159, i32 0, i32 1
  %field_load166 = load ptr, ptr %field_gep165, align 8
  call void @tl_tensor_free(ptr %field_load166)
  br label %skip_free162

skip_free162:                                     ; preds = %free_old_val161, %skip_free149
  store ptr %call_method158, ptr %ptr_h, align 8
  call void @tl_mem_unregister(ptr %call_method158)
  %s167 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s167)
  %unreg_field_0 = getelementptr inbounds %GPT, ptr %s167, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_0168 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val169 = load ptr, ptr %unreg_field_0168, align 8
  call void @tl_mem_unregister(ptr %field_val169)
  %unreg_field_1 = getelementptr inbounds %GPT, ptr %s167, i32 0, i32 1
  %field_val170 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val170)
  %unreg_field_0171 = getelementptr inbounds %Embedding, ptr %field_val170, i32 0, i32 0
  %field_val172 = load ptr, ptr %unreg_field_0171, align 8
  call void @tl_mem_unregister(ptr %field_val172)
  %unreg_field_2 = getelementptr inbounds %GPT, ptr %s167, i32 0, i32 2
  %field_val173 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val173)
  %unreg_field_0174 = getelementptr inbounds %Block, ptr %field_val173, i32 0, i32 0
  %field_val175 = load ptr, ptr %unreg_field_0174, align 8
  call void @tl_mem_unregister(ptr %field_val175)
  %unreg_field_0176 = getelementptr inbounds %LayerNorm, ptr %field_val175, i32 0, i32 0
  %field_val177 = load ptr, ptr %unreg_field_0176, align 8
  call void @tl_mem_unregister(ptr %field_val177)
  %unreg_field_1178 = getelementptr inbounds %LayerNorm, ptr %field_val175, i32 0, i32 1
  %field_val179 = load ptr, ptr %unreg_field_1178, align 8
  call void @tl_mem_unregister(ptr %field_val179)
  %unreg_field_1180 = getelementptr inbounds %Block, ptr %field_val173, i32 0, i32 1
  %field_val181 = load ptr, ptr %unreg_field_1180, align 8
  call void @tl_mem_unregister(ptr %field_val181)
  %unreg_field_0182 = getelementptr inbounds %CausalSelfAttention, ptr %field_val181, i32 0, i32 0
  %field_val183 = load ptr, ptr %unreg_field_0182, align 8
  call void @tl_mem_unregister(ptr %field_val183)
  %unreg_field_0184 = getelementptr inbounds %Linear, ptr %field_val183, i32 0, i32 0
  %field_val185 = load ptr, ptr %unreg_field_0184, align 8
  call void @tl_mem_unregister(ptr %field_val185)
  %unreg_field_1186 = getelementptr inbounds %Linear, ptr %field_val183, i32 0, i32 1
  %field_val187 = load ptr, ptr %unreg_field_1186, align 8
  call void @tl_mem_unregister(ptr %field_val187)
  %unreg_field_1188 = getelementptr inbounds %CausalSelfAttention, ptr %field_val181, i32 0, i32 1
  %field_val189 = load ptr, ptr %unreg_field_1188, align 8
  call void @tl_mem_unregister(ptr %field_val189)
  %unreg_field_0190 = getelementptr inbounds %Linear, ptr %field_val189, i32 0, i32 0
  %field_val191 = load ptr, ptr %unreg_field_0190, align 8
  call void @tl_mem_unregister(ptr %field_val191)
  %unreg_field_1192 = getelementptr inbounds %Linear, ptr %field_val189, i32 0, i32 1
  %field_val193 = load ptr, ptr %unreg_field_1192, align 8
  call void @tl_mem_unregister(ptr %field_val193)
  %unreg_field_2194 = getelementptr inbounds %CausalSelfAttention, ptr %field_val181, i32 0, i32 2
  %field_val195 = load ptr, ptr %unreg_field_2194, align 8
  call void @tl_mem_unregister(ptr %field_val195)
  %unreg_field_0196 = getelementptr inbounds %Linear, ptr %field_val195, i32 0, i32 0
  %field_val197 = load ptr, ptr %unreg_field_0196, align 8
  call void @tl_mem_unregister(ptr %field_val197)
  %unreg_field_1198 = getelementptr inbounds %Linear, ptr %field_val195, i32 0, i32 1
  %field_val199 = load ptr, ptr %unreg_field_1198, align 8
  call void @tl_mem_unregister(ptr %field_val199)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val181, i32 0, i32 3
  %field_val200 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val200)
  %unreg_field_0201 = getelementptr inbounds %Linear, ptr %field_val200, i32 0, i32 0
  %field_val202 = load ptr, ptr %unreg_field_0201, align 8
  call void @tl_mem_unregister(ptr %field_val202)
  %unreg_field_1203 = getelementptr inbounds %Linear, ptr %field_val200, i32 0, i32 1
  %field_val204 = load ptr, ptr %unreg_field_1203, align 8
  call void @tl_mem_unregister(ptr %field_val204)
  %unreg_field_2205 = getelementptr inbounds %Block, ptr %field_val173, i32 0, i32 2
  %field_val206 = load ptr, ptr %unreg_field_2205, align 8
  call void @tl_mem_unregister(ptr %field_val206)
  %unreg_field_0207 = getelementptr inbounds %LayerNorm, ptr %field_val206, i32 0, i32 0
  %field_val208 = load ptr, ptr %unreg_field_0207, align 8
  call void @tl_mem_unregister(ptr %field_val208)
  %unreg_field_1209 = getelementptr inbounds %LayerNorm, ptr %field_val206, i32 0, i32 1
  %field_val210 = load ptr, ptr %unreg_field_1209, align 8
  call void @tl_mem_unregister(ptr %field_val210)
  %unreg_field_3211 = getelementptr inbounds %Block, ptr %field_val173, i32 0, i32 3
  %field_val212 = load ptr, ptr %unreg_field_3211, align 8
  call void @tl_mem_unregister(ptr %field_val212)
  %unreg_field_0213 = getelementptr inbounds %MLP, ptr %field_val212, i32 0, i32 0
  %field_val214 = load ptr, ptr %unreg_field_0213, align 8
  call void @tl_mem_unregister(ptr %field_val214)
  %unreg_field_0215 = getelementptr inbounds %Linear, ptr %field_val214, i32 0, i32 0
  %field_val216 = load ptr, ptr %unreg_field_0215, align 8
  call void @tl_mem_unregister(ptr %field_val216)
  %unreg_field_1217 = getelementptr inbounds %Linear, ptr %field_val214, i32 0, i32 1
  %field_val218 = load ptr, ptr %unreg_field_1217, align 8
  call void @tl_mem_unregister(ptr %field_val218)
  %unreg_field_1219 = getelementptr inbounds %MLP, ptr %field_val212, i32 0, i32 1
  %field_val220 = load ptr, ptr %unreg_field_1219, align 8
  call void @tl_mem_unregister(ptr %field_val220)
  %unreg_field_0221 = getelementptr inbounds %Linear, ptr %field_val220, i32 0, i32 0
  %field_val222 = load ptr, ptr %unreg_field_0221, align 8
  call void @tl_mem_unregister(ptr %field_val222)
  %unreg_field_1223 = getelementptr inbounds %Linear, ptr %field_val220, i32 0, i32 1
  %field_val224 = load ptr, ptr %unreg_field_1223, align 8
  call void @tl_mem_unregister(ptr %field_val224)
  %unreg_field_3225 = getelementptr inbounds %GPT, ptr %s167, i32 0, i32 3
  %field_val226 = load ptr, ptr %unreg_field_3225, align 8
  call void @tl_mem_unregister(ptr %field_val226)
  %unreg_field_0227 = getelementptr inbounds %Block, ptr %field_val226, i32 0, i32 0
  %field_val228 = load ptr, ptr %unreg_field_0227, align 8
  call void @tl_mem_unregister(ptr %field_val228)
  %unreg_field_0229 = getelementptr inbounds %LayerNorm, ptr %field_val228, i32 0, i32 0
  %field_val230 = load ptr, ptr %unreg_field_0229, align 8
  call void @tl_mem_unregister(ptr %field_val230)
  %unreg_field_1231 = getelementptr inbounds %LayerNorm, ptr %field_val228, i32 0, i32 1
  %field_val232 = load ptr, ptr %unreg_field_1231, align 8
  call void @tl_mem_unregister(ptr %field_val232)
  %unreg_field_1233 = getelementptr inbounds %Block, ptr %field_val226, i32 0, i32 1
  %field_val234 = load ptr, ptr %unreg_field_1233, align 8
  call void @tl_mem_unregister(ptr %field_val234)
  %unreg_field_0235 = getelementptr inbounds %CausalSelfAttention, ptr %field_val234, i32 0, i32 0
  %field_val236 = load ptr, ptr %unreg_field_0235, align 8
  call void @tl_mem_unregister(ptr %field_val236)
  %unreg_field_0237 = getelementptr inbounds %Linear, ptr %field_val236, i32 0, i32 0
  %field_val238 = load ptr, ptr %unreg_field_0237, align 8
  call void @tl_mem_unregister(ptr %field_val238)
  %unreg_field_1239 = getelementptr inbounds %Linear, ptr %field_val236, i32 0, i32 1
  %field_val240 = load ptr, ptr %unreg_field_1239, align 8
  call void @tl_mem_unregister(ptr %field_val240)
  %unreg_field_1241 = getelementptr inbounds %CausalSelfAttention, ptr %field_val234, i32 0, i32 1
  %field_val242 = load ptr, ptr %unreg_field_1241, align 8
  call void @tl_mem_unregister(ptr %field_val242)
  %unreg_field_0243 = getelementptr inbounds %Linear, ptr %field_val242, i32 0, i32 0
  %field_val244 = load ptr, ptr %unreg_field_0243, align 8
  call void @tl_mem_unregister(ptr %field_val244)
  %unreg_field_1245 = getelementptr inbounds %Linear, ptr %field_val242, i32 0, i32 1
  %field_val246 = load ptr, ptr %unreg_field_1245, align 8
  call void @tl_mem_unregister(ptr %field_val246)
  %unreg_field_2247 = getelementptr inbounds %CausalSelfAttention, ptr %field_val234, i32 0, i32 2
  %field_val248 = load ptr, ptr %unreg_field_2247, align 8
  call void @tl_mem_unregister(ptr %field_val248)
  %unreg_field_0249 = getelementptr inbounds %Linear, ptr %field_val248, i32 0, i32 0
  %field_val250 = load ptr, ptr %unreg_field_0249, align 8
  call void @tl_mem_unregister(ptr %field_val250)
  %unreg_field_1251 = getelementptr inbounds %Linear, ptr %field_val248, i32 0, i32 1
  %field_val252 = load ptr, ptr %unreg_field_1251, align 8
  call void @tl_mem_unregister(ptr %field_val252)
  %unreg_field_3253 = getelementptr inbounds %CausalSelfAttention, ptr %field_val234, i32 0, i32 3
  %field_val254 = load ptr, ptr %unreg_field_3253, align 8
  call void @tl_mem_unregister(ptr %field_val254)
  %unreg_field_0255 = getelementptr inbounds %Linear, ptr %field_val254, i32 0, i32 0
  %field_val256 = load ptr, ptr %unreg_field_0255, align 8
  call void @tl_mem_unregister(ptr %field_val256)
  %unreg_field_1257 = getelementptr inbounds %Linear, ptr %field_val254, i32 0, i32 1
  %field_val258 = load ptr, ptr %unreg_field_1257, align 8
  call void @tl_mem_unregister(ptr %field_val258)
  %unreg_field_2259 = getelementptr inbounds %Block, ptr %field_val226, i32 0, i32 2
  %field_val260 = load ptr, ptr %unreg_field_2259, align 8
  call void @tl_mem_unregister(ptr %field_val260)
  %unreg_field_0261 = getelementptr inbounds %LayerNorm, ptr %field_val260, i32 0, i32 0
  %field_val262 = load ptr, ptr %unreg_field_0261, align 8
  call void @tl_mem_unregister(ptr %field_val262)
  %unreg_field_1263 = getelementptr inbounds %LayerNorm, ptr %field_val260, i32 0, i32 1
  %field_val264 = load ptr, ptr %unreg_field_1263, align 8
  call void @tl_mem_unregister(ptr %field_val264)
  %unreg_field_3265 = getelementptr inbounds %Block, ptr %field_val226, i32 0, i32 3
  %field_val266 = load ptr, ptr %unreg_field_3265, align 8
  call void @tl_mem_unregister(ptr %field_val266)
  %unreg_field_0267 = getelementptr inbounds %MLP, ptr %field_val266, i32 0, i32 0
  %field_val268 = load ptr, ptr %unreg_field_0267, align 8
  call void @tl_mem_unregister(ptr %field_val268)
  %unreg_field_0269 = getelementptr inbounds %Linear, ptr %field_val268, i32 0, i32 0
  %field_val270 = load ptr, ptr %unreg_field_0269, align 8
  call void @tl_mem_unregister(ptr %field_val270)
  %unreg_field_1271 = getelementptr inbounds %Linear, ptr %field_val268, i32 0, i32 1
  %field_val272 = load ptr, ptr %unreg_field_1271, align 8
  call void @tl_mem_unregister(ptr %field_val272)
  %unreg_field_1273 = getelementptr inbounds %MLP, ptr %field_val266, i32 0, i32 1
  %field_val274 = load ptr, ptr %unreg_field_1273, align 8
  call void @tl_mem_unregister(ptr %field_val274)
  %unreg_field_0275 = getelementptr inbounds %Linear, ptr %field_val274, i32 0, i32 0
  %field_val276 = load ptr, ptr %unreg_field_0275, align 8
  call void @tl_mem_unregister(ptr %field_val276)
  %unreg_field_1277 = getelementptr inbounds %Linear, ptr %field_val274, i32 0, i32 1
  %field_val278 = load ptr, ptr %unreg_field_1277, align 8
  call void @tl_mem_unregister(ptr %field_val278)
  %unreg_field_4 = getelementptr inbounds %GPT, ptr %s167, i32 0, i32 4
  %field_val279 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val279)
  %unreg_field_0280 = getelementptr inbounds %LayerNorm, ptr %field_val279, i32 0, i32 0
  %field_val281 = load ptr, ptr %unreg_field_0280, align 8
  call void @tl_mem_unregister(ptr %field_val281)
  %unreg_field_1282 = getelementptr inbounds %LayerNorm, ptr %field_val279, i32 0, i32 1
  %field_val283 = load ptr, ptr %unreg_field_1282, align 8
  call void @tl_mem_unregister(ptr %field_val283)
  %unreg_field_5 = getelementptr inbounds %GPT, ptr %s167, i32 0, i32 5
  %field_val284 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val284)
  %unreg_field_0285 = getelementptr inbounds %Linear, ptr %field_val284, i32 0, i32 0
  %field_val286 = load ptr, ptr %unreg_field_0285, align 8
  call void @tl_mem_unregister(ptr %field_val286)
  %unreg_field_1287 = getelementptr inbounds %Linear, ptr %field_val284, i32 0, i32 1
  %field_val288 = load ptr, ptr %unreg_field_1287, align 8
  call void @tl_mem_unregister(ptr %field_val288)
  call void @tl_mem_exit_scope()
  ret ptr %s167
}

define ptr @tl_GPT_clone(ptr %self) {
entry:
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%GPT, ptr null, i32 1) to i64))
  %self2 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds %GPT, ptr %self2, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %call_method = call ptr @tl_Embedding_clone(ptr %w)
  call void @tl_mem_register_struct(ptr %call_method)
  %init_field = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 0
  store ptr %call_method, ptr %init_field, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_wp = getelementptr inbounds %GPT, ptr %self3, i32 0, i32 1
  %wp = load ptr, ptr %ptr_wp, align 8
  %call_method4 = call ptr @tl_Embedding_clone(ptr %wp)
  call void @tl_mem_register_struct(ptr %call_method4)
  %init_field5 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 1
  store ptr %call_method4, ptr %init_field5, align 8
  %self6 = load ptr, ptr %self1, align 8
  %ptr_b1 = getelementptr inbounds %GPT, ptr %self6, i32 0, i32 2
  %b1 = load ptr, ptr %ptr_b1, align 8
  %call_method7 = call ptr @tl_Block_clone(ptr %b1)
  call void @tl_mem_register_struct(ptr %call_method7)
  %init_field8 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 2
  store ptr %call_method7, ptr %init_field8, align 8
  %self9 = load ptr, ptr %self1, align 8
  %ptr_b2 = getelementptr inbounds %GPT, ptr %self9, i32 0, i32 3
  %b2 = load ptr, ptr %ptr_b2, align 8
  %call_method10 = call ptr @tl_Block_clone(ptr %b2)
  call void @tl_mem_register_struct(ptr %call_method10)
  %init_field11 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 3
  store ptr %call_method10, ptr %init_field11, align 8
  %self12 = load ptr, ptr %self1, align 8
  %ptr_l = getelementptr inbounds %GPT, ptr %self12, i32 0, i32 4
  %l = load ptr, ptr %ptr_l, align 8
  %call_method13 = call ptr @tl_LayerNorm_clone(ptr %l)
  call void @tl_mem_register_struct(ptr %call_method13)
  %init_field14 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 4
  store ptr %call_method13, ptr %init_field14, align 8
  %self15 = load ptr, ptr %self1, align 8
  %ptr_h = getelementptr inbounds %GPT, ptr %self15, i32 0, i32 5
  %h = load ptr, ptr %ptr_h, align 8
  %call_method16 = call ptr @tl_Linear_clone(ptr %h)
  call void @tl_mem_register_struct(ptr %call_method16)
  %init_field17 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 5
  store ptr %call_method16, ptr %init_field17, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_018 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val19 = load ptr, ptr %unreg_field_018, align 8
  call void @tl_mem_unregister(ptr %field_val19)
  %unreg_field_1 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 1
  %field_val20 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val20)
  %unreg_field_021 = getelementptr inbounds %Embedding, ptr %field_val20, i32 0, i32 0
  %field_val22 = load ptr, ptr %unreg_field_021, align 8
  call void @tl_mem_unregister(ptr %field_val22)
  %unreg_field_2 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 2
  %field_val23 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val23)
  %unreg_field_024 = getelementptr inbounds %Block, ptr %field_val23, i32 0, i32 0
  %field_val25 = load ptr, ptr %unreg_field_024, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_026 = getelementptr inbounds %LayerNorm, ptr %field_val25, i32 0, i32 0
  %field_val27 = load ptr, ptr %unreg_field_026, align 8
  call void @tl_mem_unregister(ptr %field_val27)
  %unreg_field_128 = getelementptr inbounds %LayerNorm, ptr %field_val25, i32 0, i32 1
  %field_val29 = load ptr, ptr %unreg_field_128, align 8
  call void @tl_mem_unregister(ptr %field_val29)
  %unreg_field_130 = getelementptr inbounds %Block, ptr %field_val23, i32 0, i32 1
  %field_val31 = load ptr, ptr %unreg_field_130, align 8
  call void @tl_mem_unregister(ptr %field_val31)
  %unreg_field_032 = getelementptr inbounds %CausalSelfAttention, ptr %field_val31, i32 0, i32 0
  %field_val33 = load ptr, ptr %unreg_field_032, align 8
  call void @tl_mem_unregister(ptr %field_val33)
  %unreg_field_034 = getelementptr inbounds %Linear, ptr %field_val33, i32 0, i32 0
  %field_val35 = load ptr, ptr %unreg_field_034, align 8
  call void @tl_mem_unregister(ptr %field_val35)
  %unreg_field_136 = getelementptr inbounds %Linear, ptr %field_val33, i32 0, i32 1
  %field_val37 = load ptr, ptr %unreg_field_136, align 8
  call void @tl_mem_unregister(ptr %field_val37)
  %unreg_field_138 = getelementptr inbounds %CausalSelfAttention, ptr %field_val31, i32 0, i32 1
  %field_val39 = load ptr, ptr %unreg_field_138, align 8
  call void @tl_mem_unregister(ptr %field_val39)
  %unreg_field_040 = getelementptr inbounds %Linear, ptr %field_val39, i32 0, i32 0
  %field_val41 = load ptr, ptr %unreg_field_040, align 8
  call void @tl_mem_unregister(ptr %field_val41)
  %unreg_field_142 = getelementptr inbounds %Linear, ptr %field_val39, i32 0, i32 1
  %field_val43 = load ptr, ptr %unreg_field_142, align 8
  call void @tl_mem_unregister(ptr %field_val43)
  %unreg_field_244 = getelementptr inbounds %CausalSelfAttention, ptr %field_val31, i32 0, i32 2
  %field_val45 = load ptr, ptr %unreg_field_244, align 8
  call void @tl_mem_unregister(ptr %field_val45)
  %unreg_field_046 = getelementptr inbounds %Linear, ptr %field_val45, i32 0, i32 0
  %field_val47 = load ptr, ptr %unreg_field_046, align 8
  call void @tl_mem_unregister(ptr %field_val47)
  %unreg_field_148 = getelementptr inbounds %Linear, ptr %field_val45, i32 0, i32 1
  %field_val49 = load ptr, ptr %unreg_field_148, align 8
  call void @tl_mem_unregister(ptr %field_val49)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val31, i32 0, i32 3
  %field_val50 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val50)
  %unreg_field_051 = getelementptr inbounds %Linear, ptr %field_val50, i32 0, i32 0
  %field_val52 = load ptr, ptr %unreg_field_051, align 8
  call void @tl_mem_unregister(ptr %field_val52)
  %unreg_field_153 = getelementptr inbounds %Linear, ptr %field_val50, i32 0, i32 1
  %field_val54 = load ptr, ptr %unreg_field_153, align 8
  call void @tl_mem_unregister(ptr %field_val54)
  %unreg_field_255 = getelementptr inbounds %Block, ptr %field_val23, i32 0, i32 2
  %field_val56 = load ptr, ptr %unreg_field_255, align 8
  call void @tl_mem_unregister(ptr %field_val56)
  %unreg_field_057 = getelementptr inbounds %LayerNorm, ptr %field_val56, i32 0, i32 0
  %field_val58 = load ptr, ptr %unreg_field_057, align 8
  call void @tl_mem_unregister(ptr %field_val58)
  %unreg_field_159 = getelementptr inbounds %LayerNorm, ptr %field_val56, i32 0, i32 1
  %field_val60 = load ptr, ptr %unreg_field_159, align 8
  call void @tl_mem_unregister(ptr %field_val60)
  %unreg_field_361 = getelementptr inbounds %Block, ptr %field_val23, i32 0, i32 3
  %field_val62 = load ptr, ptr %unreg_field_361, align 8
  call void @tl_mem_unregister(ptr %field_val62)
  %unreg_field_063 = getelementptr inbounds %MLP, ptr %field_val62, i32 0, i32 0
  %field_val64 = load ptr, ptr %unreg_field_063, align 8
  call void @tl_mem_unregister(ptr %field_val64)
  %unreg_field_065 = getelementptr inbounds %Linear, ptr %field_val64, i32 0, i32 0
  %field_val66 = load ptr, ptr %unreg_field_065, align 8
  call void @tl_mem_unregister(ptr %field_val66)
  %unreg_field_167 = getelementptr inbounds %Linear, ptr %field_val64, i32 0, i32 1
  %field_val68 = load ptr, ptr %unreg_field_167, align 8
  call void @tl_mem_unregister(ptr %field_val68)
  %unreg_field_169 = getelementptr inbounds %MLP, ptr %field_val62, i32 0, i32 1
  %field_val70 = load ptr, ptr %unreg_field_169, align 8
  call void @tl_mem_unregister(ptr %field_val70)
  %unreg_field_071 = getelementptr inbounds %Linear, ptr %field_val70, i32 0, i32 0
  %field_val72 = load ptr, ptr %unreg_field_071, align 8
  call void @tl_mem_unregister(ptr %field_val72)
  %unreg_field_173 = getelementptr inbounds %Linear, ptr %field_val70, i32 0, i32 1
  %field_val74 = load ptr, ptr %unreg_field_173, align 8
  call void @tl_mem_unregister(ptr %field_val74)
  %unreg_field_375 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 3
  %field_val76 = load ptr, ptr %unreg_field_375, align 8
  call void @tl_mem_unregister(ptr %field_val76)
  %unreg_field_077 = getelementptr inbounds %Block, ptr %field_val76, i32 0, i32 0
  %field_val78 = load ptr, ptr %unreg_field_077, align 8
  call void @tl_mem_unregister(ptr %field_val78)
  %unreg_field_079 = getelementptr inbounds %LayerNorm, ptr %field_val78, i32 0, i32 0
  %field_val80 = load ptr, ptr %unreg_field_079, align 8
  call void @tl_mem_unregister(ptr %field_val80)
  %unreg_field_181 = getelementptr inbounds %LayerNorm, ptr %field_val78, i32 0, i32 1
  %field_val82 = load ptr, ptr %unreg_field_181, align 8
  call void @tl_mem_unregister(ptr %field_val82)
  %unreg_field_183 = getelementptr inbounds %Block, ptr %field_val76, i32 0, i32 1
  %field_val84 = load ptr, ptr %unreg_field_183, align 8
  call void @tl_mem_unregister(ptr %field_val84)
  %unreg_field_085 = getelementptr inbounds %CausalSelfAttention, ptr %field_val84, i32 0, i32 0
  %field_val86 = load ptr, ptr %unreg_field_085, align 8
  call void @tl_mem_unregister(ptr %field_val86)
  %unreg_field_087 = getelementptr inbounds %Linear, ptr %field_val86, i32 0, i32 0
  %field_val88 = load ptr, ptr %unreg_field_087, align 8
  call void @tl_mem_unregister(ptr %field_val88)
  %unreg_field_189 = getelementptr inbounds %Linear, ptr %field_val86, i32 0, i32 1
  %field_val90 = load ptr, ptr %unreg_field_189, align 8
  call void @tl_mem_unregister(ptr %field_val90)
  %unreg_field_191 = getelementptr inbounds %CausalSelfAttention, ptr %field_val84, i32 0, i32 1
  %field_val92 = load ptr, ptr %unreg_field_191, align 8
  call void @tl_mem_unregister(ptr %field_val92)
  %unreg_field_093 = getelementptr inbounds %Linear, ptr %field_val92, i32 0, i32 0
  %field_val94 = load ptr, ptr %unreg_field_093, align 8
  call void @tl_mem_unregister(ptr %field_val94)
  %unreg_field_195 = getelementptr inbounds %Linear, ptr %field_val92, i32 0, i32 1
  %field_val96 = load ptr, ptr %unreg_field_195, align 8
  call void @tl_mem_unregister(ptr %field_val96)
  %unreg_field_297 = getelementptr inbounds %CausalSelfAttention, ptr %field_val84, i32 0, i32 2
  %field_val98 = load ptr, ptr %unreg_field_297, align 8
  call void @tl_mem_unregister(ptr %field_val98)
  %unreg_field_099 = getelementptr inbounds %Linear, ptr %field_val98, i32 0, i32 0
  %field_val100 = load ptr, ptr %unreg_field_099, align 8
  call void @tl_mem_unregister(ptr %field_val100)
  %unreg_field_1101 = getelementptr inbounds %Linear, ptr %field_val98, i32 0, i32 1
  %field_val102 = load ptr, ptr %unreg_field_1101, align 8
  call void @tl_mem_unregister(ptr %field_val102)
  %unreg_field_3103 = getelementptr inbounds %CausalSelfAttention, ptr %field_val84, i32 0, i32 3
  %field_val104 = load ptr, ptr %unreg_field_3103, align 8
  call void @tl_mem_unregister(ptr %field_val104)
  %unreg_field_0105 = getelementptr inbounds %Linear, ptr %field_val104, i32 0, i32 0
  %field_val106 = load ptr, ptr %unreg_field_0105, align 8
  call void @tl_mem_unregister(ptr %field_val106)
  %unreg_field_1107 = getelementptr inbounds %Linear, ptr %field_val104, i32 0, i32 1
  %field_val108 = load ptr, ptr %unreg_field_1107, align 8
  call void @tl_mem_unregister(ptr %field_val108)
  %unreg_field_2109 = getelementptr inbounds %Block, ptr %field_val76, i32 0, i32 2
  %field_val110 = load ptr, ptr %unreg_field_2109, align 8
  call void @tl_mem_unregister(ptr %field_val110)
  %unreg_field_0111 = getelementptr inbounds %LayerNorm, ptr %field_val110, i32 0, i32 0
  %field_val112 = load ptr, ptr %unreg_field_0111, align 8
  call void @tl_mem_unregister(ptr %field_val112)
  %unreg_field_1113 = getelementptr inbounds %LayerNorm, ptr %field_val110, i32 0, i32 1
  %field_val114 = load ptr, ptr %unreg_field_1113, align 8
  call void @tl_mem_unregister(ptr %field_val114)
  %unreg_field_3115 = getelementptr inbounds %Block, ptr %field_val76, i32 0, i32 3
  %field_val116 = load ptr, ptr %unreg_field_3115, align 8
  call void @tl_mem_unregister(ptr %field_val116)
  %unreg_field_0117 = getelementptr inbounds %MLP, ptr %field_val116, i32 0, i32 0
  %field_val118 = load ptr, ptr %unreg_field_0117, align 8
  call void @tl_mem_unregister(ptr %field_val118)
  %unreg_field_0119 = getelementptr inbounds %Linear, ptr %field_val118, i32 0, i32 0
  %field_val120 = load ptr, ptr %unreg_field_0119, align 8
  call void @tl_mem_unregister(ptr %field_val120)
  %unreg_field_1121 = getelementptr inbounds %Linear, ptr %field_val118, i32 0, i32 1
  %field_val122 = load ptr, ptr %unreg_field_1121, align 8
  call void @tl_mem_unregister(ptr %field_val122)
  %unreg_field_1123 = getelementptr inbounds %MLP, ptr %field_val116, i32 0, i32 1
  %field_val124 = load ptr, ptr %unreg_field_1123, align 8
  call void @tl_mem_unregister(ptr %field_val124)
  %unreg_field_0125 = getelementptr inbounds %Linear, ptr %field_val124, i32 0, i32 0
  %field_val126 = load ptr, ptr %unreg_field_0125, align 8
  call void @tl_mem_unregister(ptr %field_val126)
  %unreg_field_1127 = getelementptr inbounds %Linear, ptr %field_val124, i32 0, i32 1
  %field_val128 = load ptr, ptr %unreg_field_1127, align 8
  call void @tl_mem_unregister(ptr %field_val128)
  %unreg_field_4 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 4
  %field_val129 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val129)
  %unreg_field_0130 = getelementptr inbounds %LayerNorm, ptr %field_val129, i32 0, i32 0
  %field_val131 = load ptr, ptr %unreg_field_0130, align 8
  call void @tl_mem_unregister(ptr %field_val131)
  %unreg_field_1132 = getelementptr inbounds %LayerNorm, ptr %field_val129, i32 0, i32 1
  %field_val133 = load ptr, ptr %unreg_field_1132, align 8
  call void @tl_mem_unregister(ptr %field_val133)
  %unreg_field_5 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 5
  %field_val134 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val134)
  %unreg_field_0135 = getelementptr inbounds %Linear, ptr %field_val134, i32 0, i32 0
  %field_val136 = load ptr, ptr %unreg_field_0135, align 8
  call void @tl_mem_unregister(ptr %field_val136)
  %unreg_field_1137 = getelementptr inbounds %Linear, ptr %field_val134, i32 0, i32 1
  %field_val138 = load ptr, ptr %unreg_field_1137, align 8
  call void @tl_mem_unregister(ptr %field_val138)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

declare void @tl_tensor_print_2.104(ptr)

declare void @tl_tensor_print_1.105(ptr)

declare void @tl_clear_grads.106()

define ptr @train_step(ptr %model, float %lr, i64 %i, i64 %j) {
entry:
  %dims_alloca50 = alloca [1 x i64], align 8
  %loss = alloca ptr, align 16
  %Y_flat = alloca ptr, align 16
  %dims_alloca43 = alloca [1 x i64], align 8
  %logits_flat = alloca ptr, align 16
  %dims_alloca37 = alloca [2 x i64], align 8
  %logits = alloca ptr, align 16
  %Y = alloca ptr, align 16
  %dims_alloca28 = alloca [2 x i64], align 8
  %X = alloca ptr, align 16
  %dims_alloca = alloca [2 x i64], align 8
  %sum = alloca i64, align 16
  %m = alloca ptr, align 16
  %j4 = alloca i64, align 16
  %i3 = alloca i64, align 16
  %lr2 = alloca float, align 16
  %model1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %model, ptr %model1, align 8
  store float %lr, ptr %lr2, align 4
  store i64 %i, ptr %i3, align 8
  store i64 %j, ptr %j4, align 8
  %model5 = load ptr, ptr %model1, align 8
  %call_method = call ptr @tl_GPT_clone(ptr %model5)
  call void @tl_mem_register_struct(ptr %call_method)
  store ptr %call_method, ptr %m, align 8
  %i6 = load i64, ptr %i3, align 8
  %j7 = load i64, ptr %j4, align 8
  %addtmp = add i64 %i6, %j7
  store i64 %addtmp, ptr %sum, align 8
  %buf_void = call ptr @tl_alloc_tmp(i64 16)
  %i8 = load i64, ptr %i3, align 8
  %i2f = sitofp i64 %i8 to float
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %i2f, ptr %elem_ptr, align 4
  %j9 = load i64, ptr %j4, align 8
  %i2f10 = sitofp i64 %j9 to float
  %elem_ptr11 = getelementptr inbounds float, ptr %buf_void, i64 1
  store float %i2f10, ptr %elem_ptr11, align 4
  %sum12 = load i64, ptr %sum, align 8
  %i2f13 = sitofp i64 %sum12 to float
  %elem_ptr14 = getelementptr inbounds float, ptr %buf_void, i64 2
  store float %i2f13, ptr %elem_ptr14, align 4
  %elem_ptr15 = getelementptr inbounds float, ptr %buf_void, i64 3
  store float 0.000000e+00, ptr %elem_ptr15, align 4
  %shape_alloc = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr = getelementptr inbounds i64, ptr %shape_alloc, i64 0
  store i64 4, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %shape_alloc)
  call void @tl_free_tmp(ptr %buf_void)
  call void @tl_free_tmp(ptr %shape_alloc)
  %dim_ptr_0 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0, align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 4, ptr %dim_ptr, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %new_tensor, ptr %dims_ptr, i64 2)
  call void @tl_tensor_free(ptr %new_tensor)
  call void @tl_mem_unregister(ptr %reshape_dims_res)
  store ptr %reshape_dims_res, ptr %X, align 8
  %buf_void16 = call ptr @tl_alloc_tmp(i64 16)
  %j17 = load i64, ptr %j4, align 8
  %i2f18 = sitofp i64 %j17 to float
  %elem_ptr19 = getelementptr inbounds float, ptr %buf_void16, i64 0
  store float %i2f18, ptr %elem_ptr19, align 4
  %sum20 = load i64, ptr %sum, align 8
  %i2f21 = sitofp i64 %sum20 to float
  %elem_ptr22 = getelementptr inbounds float, ptr %buf_void16, i64 1
  store float %i2f21, ptr %elem_ptr22, align 4
  %elem_ptr23 = getelementptr inbounds float, ptr %buf_void16, i64 2
  store float 0.000000e+00, ptr %elem_ptr23, align 4
  %elem_ptr24 = getelementptr inbounds float, ptr %buf_void16, i64 3
  store float 0.000000e+00, ptr %elem_ptr24, align 4
  %shape_alloc25 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr26 = getelementptr inbounds i64, ptr %shape_alloc25, i64 0
  store i64 4, ptr %shape_ptr26, align 8
  %new_tensor27 = call ptr @tl_tensor_new(ptr %buf_void16, i64 1, ptr %shape_alloc25)
  call void @tl_free_tmp(ptr %buf_void16)
  call void @tl_free_tmp(ptr %shape_alloc25)
  %dim_ptr_029 = getelementptr [2 x i64], ptr %dims_alloca28, i64 0, i64 0
  store i64 1, ptr %dim_ptr_029, align 8
  %dim_ptr30 = getelementptr [2 x i64], ptr %dims_alloca28, i64 0, i64 1
  store i64 4, ptr %dim_ptr30, align 8
  %dims_ptr31 = getelementptr [2 x i64], ptr %dims_alloca28, i64 0, i64 0
  %reshape_dims_res32 = call ptr @tl_tensor_reshape_dims(ptr %new_tensor27, ptr %dims_ptr31, i64 2)
  call void @tl_tensor_free(ptr %new_tensor27)
  call void @tl_mem_unregister(ptr %reshape_dims_res32)
  store ptr %reshape_dims_res32, ptr %Y, align 8
  %m33 = load ptr, ptr %m, align 8
  %X34 = load ptr, ptr %X, align 8
  %call_method35 = call ptr @tl_GPT_forward(ptr %m33, ptr %X34)
  call void @tl_mem_register_tensor(ptr %call_method35)
  call void @tl_mem_unregister(ptr %call_method35)
  store ptr %call_method35, ptr %logits, align 8
  %logits36 = load ptr, ptr %logits, align 8
  %dim_ptr_038 = getelementptr [2 x i64], ptr %dims_alloca37, i64 0, i64 0
  store i64 4, ptr %dim_ptr_038, align 8
  %dim_ptr39 = getelementptr [2 x i64], ptr %dims_alloca37, i64 0, i64 1
  store i64 200, ptr %dim_ptr39, align 8
  %dims_ptr40 = getelementptr [2 x i64], ptr %dims_alloca37, i64 0, i64 0
  %reshape_dims_res41 = call ptr @tl_tensor_reshape_dims(ptr %logits36, ptr %dims_ptr40, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res41)
  store ptr %reshape_dims_res41, ptr %logits_flat, align 8
  %Y42 = load ptr, ptr %Y, align 8
  %dim_ptr_044 = getelementptr [1 x i64], ptr %dims_alloca43, i64 0, i64 0
  store i64 4, ptr %dim_ptr_044, align 8
  %dims_ptr45 = getelementptr [1 x i64], ptr %dims_alloca43, i64 0, i64 0
  %reshape_dims_res46 = call ptr @tl_tensor_reshape_dims(ptr %Y42, ptr %dims_ptr45, i64 1)
  call void @tl_mem_unregister(ptr %reshape_dims_res46)
  store ptr %reshape_dims_res46, ptr %Y_flat, align 8
  %logits_flat47 = load ptr, ptr %logits_flat, align 8
  %Y_flat48 = load ptr, ptr %Y_flat, align 8
  %ce_res = call ptr @tl_tensor_cross_entropy(ptr %logits_flat47, ptr %Y_flat48)
  call void @tl_mem_unregister(ptr %ce_res)
  store ptr %ce_res, ptr %loss, align 8
  %loss49 = load ptr, ptr %loss, align 8
  %dim_ptr_051 = getelementptr [1 x i64], ptr %dims_alloca50, i64 0, i64 0
  store i64 1, ptr %dim_ptr_051, align 8
  %dims_ptr52 = getelementptr [1 x i64], ptr %dims_alloca50, i64 0, i64 0
  %reshape_dims_res53 = call ptr @tl_tensor_reshape_dims(ptr %loss49, ptr %dims_ptr52, i64 1)
  call void @tl_tensor_print_1(ptr %reshape_dims_res53)
  call void @tl_tensor_free(ptr %reshape_dims_res53)
  %loss54 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss54)
  %m55 = load ptr, ptr %m, align 8
  %lr56 = load float, ptr %lr2, align 4
  %call_method57 = call ptr @tl_GPT_step(ptr %m55, float %lr56)
  call void @tl_mem_register_struct(ptr %call_method57)
  call void @tl_mem_unregister(ptr %call_method57)
  %old_struct_to_free = load ptr, ptr %m, align 8
  %is_not_null = icmp ne ptr %old_struct_to_free, null
  %are_diff = icmp ne ptr %old_struct_to_free, %call_method57
  %can_free_1 = and i1 %is_not_null, true
  %can_free = and i1 %can_free_1, %are_diff
  br i1 %can_free, label %free_struct, label %continue_after_free

free_struct:                                      ; preds = %entry
  %field_gep = getelementptr inbounds %GPT, ptr %old_struct_to_free, i32 0, i32 0
  %field_load = load ptr, ptr %field_gep, align 8
  %field_gep58 = getelementptr inbounds %Embedding, ptr %field_load, i32 0, i32 0
  %field_load59 = load ptr, ptr %field_gep58, align 8
  call void @tl_tensor_free(ptr %field_load59)
  %field_gep60 = getelementptr inbounds %GPT, ptr %old_struct_to_free, i32 0, i32 1
  %field_load61 = load ptr, ptr %field_gep60, align 8
  %field_gep62 = getelementptr inbounds %Embedding, ptr %field_load61, i32 0, i32 0
  %field_load63 = load ptr, ptr %field_gep62, align 8
  call void @tl_tensor_free(ptr %field_load63)
  %field_gep64 = getelementptr inbounds %GPT, ptr %old_struct_to_free, i32 0, i32 2
  %field_load65 = load ptr, ptr %field_gep64, align 8
  %field_gep66 = getelementptr inbounds %Block, ptr %field_load65, i32 0, i32 0
  %field_load67 = load ptr, ptr %field_gep66, align 8
  %field_gep68 = getelementptr inbounds %LayerNorm, ptr %field_load67, i32 0, i32 0
  %field_load69 = load ptr, ptr %field_gep68, align 8
  call void @tl_tensor_free(ptr %field_load69)
  %field_gep70 = getelementptr inbounds %LayerNorm, ptr %field_load67, i32 0, i32 1
  %field_load71 = load ptr, ptr %field_gep70, align 8
  call void @tl_tensor_free(ptr %field_load71)
  %field_gep72 = getelementptr inbounds %Block, ptr %field_load65, i32 0, i32 1
  %field_load73 = load ptr, ptr %field_gep72, align 8
  %field_gep74 = getelementptr inbounds %CausalSelfAttention, ptr %field_load73, i32 0, i32 0
  %field_load75 = load ptr, ptr %field_gep74, align 8
  %field_gep76 = getelementptr inbounds %Linear, ptr %field_load75, i32 0, i32 0
  %field_load77 = load ptr, ptr %field_gep76, align 8
  call void @tl_tensor_free(ptr %field_load77)
  %field_gep78 = getelementptr inbounds %Linear, ptr %field_load75, i32 0, i32 1
  %field_load79 = load ptr, ptr %field_gep78, align 8
  call void @tl_tensor_free(ptr %field_load79)
  %field_gep80 = getelementptr inbounds %CausalSelfAttention, ptr %field_load73, i32 0, i32 1
  %field_load81 = load ptr, ptr %field_gep80, align 8
  %field_gep82 = getelementptr inbounds %Linear, ptr %field_load81, i32 0, i32 0
  %field_load83 = load ptr, ptr %field_gep82, align 8
  call void @tl_tensor_free(ptr %field_load83)
  %field_gep84 = getelementptr inbounds %Linear, ptr %field_load81, i32 0, i32 1
  %field_load85 = load ptr, ptr %field_gep84, align 8
  call void @tl_tensor_free(ptr %field_load85)
  %field_gep86 = getelementptr inbounds %CausalSelfAttention, ptr %field_load73, i32 0, i32 2
  %field_load87 = load ptr, ptr %field_gep86, align 8
  %field_gep88 = getelementptr inbounds %Linear, ptr %field_load87, i32 0, i32 0
  %field_load89 = load ptr, ptr %field_gep88, align 8
  call void @tl_tensor_free(ptr %field_load89)
  %field_gep90 = getelementptr inbounds %Linear, ptr %field_load87, i32 0, i32 1
  %field_load91 = load ptr, ptr %field_gep90, align 8
  call void @tl_tensor_free(ptr %field_load91)
  %field_gep92 = getelementptr inbounds %CausalSelfAttention, ptr %field_load73, i32 0, i32 3
  %field_load93 = load ptr, ptr %field_gep92, align 8
  %field_gep94 = getelementptr inbounds %Linear, ptr %field_load93, i32 0, i32 0
  %field_load95 = load ptr, ptr %field_gep94, align 8
  call void @tl_tensor_free(ptr %field_load95)
  %field_gep96 = getelementptr inbounds %Linear, ptr %field_load93, i32 0, i32 1
  %field_load97 = load ptr, ptr %field_gep96, align 8
  call void @tl_tensor_free(ptr %field_load97)
  %field_gep98 = getelementptr inbounds %Block, ptr %field_load65, i32 0, i32 2
  %field_load99 = load ptr, ptr %field_gep98, align 8
  %field_gep100 = getelementptr inbounds %LayerNorm, ptr %field_load99, i32 0, i32 0
  %field_load101 = load ptr, ptr %field_gep100, align 8
  call void @tl_tensor_free(ptr %field_load101)
  %field_gep102 = getelementptr inbounds %LayerNorm, ptr %field_load99, i32 0, i32 1
  %field_load103 = load ptr, ptr %field_gep102, align 8
  call void @tl_tensor_free(ptr %field_load103)
  %field_gep104 = getelementptr inbounds %Block, ptr %field_load65, i32 0, i32 3
  %field_load105 = load ptr, ptr %field_gep104, align 8
  %field_gep106 = getelementptr inbounds %MLP, ptr %field_load105, i32 0, i32 0
  %field_load107 = load ptr, ptr %field_gep106, align 8
  %field_gep108 = getelementptr inbounds %Linear, ptr %field_load107, i32 0, i32 0
  %field_load109 = load ptr, ptr %field_gep108, align 8
  call void @tl_tensor_free(ptr %field_load109)
  %field_gep110 = getelementptr inbounds %Linear, ptr %field_load107, i32 0, i32 1
  %field_load111 = load ptr, ptr %field_gep110, align 8
  call void @tl_tensor_free(ptr %field_load111)
  %field_gep112 = getelementptr inbounds %MLP, ptr %field_load105, i32 0, i32 1
  %field_load113 = load ptr, ptr %field_gep112, align 8
  %field_gep114 = getelementptr inbounds %Linear, ptr %field_load113, i32 0, i32 0
  %field_load115 = load ptr, ptr %field_gep114, align 8
  call void @tl_tensor_free(ptr %field_load115)
  %field_gep116 = getelementptr inbounds %Linear, ptr %field_load113, i32 0, i32 1
  %field_load117 = load ptr, ptr %field_gep116, align 8
  call void @tl_tensor_free(ptr %field_load117)
  %field_gep118 = getelementptr inbounds %GPT, ptr %old_struct_to_free, i32 0, i32 3
  %field_load119 = load ptr, ptr %field_gep118, align 8
  %field_gep120 = getelementptr inbounds %Block, ptr %field_load119, i32 0, i32 0
  %field_load121 = load ptr, ptr %field_gep120, align 8
  %field_gep122 = getelementptr inbounds %LayerNorm, ptr %field_load121, i32 0, i32 0
  %field_load123 = load ptr, ptr %field_gep122, align 8
  call void @tl_tensor_free(ptr %field_load123)
  %field_gep124 = getelementptr inbounds %LayerNorm, ptr %field_load121, i32 0, i32 1
  %field_load125 = load ptr, ptr %field_gep124, align 8
  call void @tl_tensor_free(ptr %field_load125)
  %field_gep126 = getelementptr inbounds %Block, ptr %field_load119, i32 0, i32 1
  %field_load127 = load ptr, ptr %field_gep126, align 8
  %field_gep128 = getelementptr inbounds %CausalSelfAttention, ptr %field_load127, i32 0, i32 0
  %field_load129 = load ptr, ptr %field_gep128, align 8
  %field_gep130 = getelementptr inbounds %Linear, ptr %field_load129, i32 0, i32 0
  %field_load131 = load ptr, ptr %field_gep130, align 8
  call void @tl_tensor_free(ptr %field_load131)
  %field_gep132 = getelementptr inbounds %Linear, ptr %field_load129, i32 0, i32 1
  %field_load133 = load ptr, ptr %field_gep132, align 8
  call void @tl_tensor_free(ptr %field_load133)
  %field_gep134 = getelementptr inbounds %CausalSelfAttention, ptr %field_load127, i32 0, i32 1
  %field_load135 = load ptr, ptr %field_gep134, align 8
  %field_gep136 = getelementptr inbounds %Linear, ptr %field_load135, i32 0, i32 0
  %field_load137 = load ptr, ptr %field_gep136, align 8
  call void @tl_tensor_free(ptr %field_load137)
  %field_gep138 = getelementptr inbounds %Linear, ptr %field_load135, i32 0, i32 1
  %field_load139 = load ptr, ptr %field_gep138, align 8
  call void @tl_tensor_free(ptr %field_load139)
  %field_gep140 = getelementptr inbounds %CausalSelfAttention, ptr %field_load127, i32 0, i32 2
  %field_load141 = load ptr, ptr %field_gep140, align 8
  %field_gep142 = getelementptr inbounds %Linear, ptr %field_load141, i32 0, i32 0
  %field_load143 = load ptr, ptr %field_gep142, align 8
  call void @tl_tensor_free(ptr %field_load143)
  %field_gep144 = getelementptr inbounds %Linear, ptr %field_load141, i32 0, i32 1
  %field_load145 = load ptr, ptr %field_gep144, align 8
  call void @tl_tensor_free(ptr %field_load145)
  %field_gep146 = getelementptr inbounds %CausalSelfAttention, ptr %field_load127, i32 0, i32 3
  %field_load147 = load ptr, ptr %field_gep146, align 8
  %field_gep148 = getelementptr inbounds %Linear, ptr %field_load147, i32 0, i32 0
  %field_load149 = load ptr, ptr %field_gep148, align 8
  call void @tl_tensor_free(ptr %field_load149)
  %field_gep150 = getelementptr inbounds %Linear, ptr %field_load147, i32 0, i32 1
  %field_load151 = load ptr, ptr %field_gep150, align 8
  call void @tl_tensor_free(ptr %field_load151)
  %field_gep152 = getelementptr inbounds %Block, ptr %field_load119, i32 0, i32 2
  %field_load153 = load ptr, ptr %field_gep152, align 8
  %field_gep154 = getelementptr inbounds %LayerNorm, ptr %field_load153, i32 0, i32 0
  %field_load155 = load ptr, ptr %field_gep154, align 8
  call void @tl_tensor_free(ptr %field_load155)
  %field_gep156 = getelementptr inbounds %LayerNorm, ptr %field_load153, i32 0, i32 1
  %field_load157 = load ptr, ptr %field_gep156, align 8
  call void @tl_tensor_free(ptr %field_load157)
  %field_gep158 = getelementptr inbounds %Block, ptr %field_load119, i32 0, i32 3
  %field_load159 = load ptr, ptr %field_gep158, align 8
  %field_gep160 = getelementptr inbounds %MLP, ptr %field_load159, i32 0, i32 0
  %field_load161 = load ptr, ptr %field_gep160, align 8
  %field_gep162 = getelementptr inbounds %Linear, ptr %field_load161, i32 0, i32 0
  %field_load163 = load ptr, ptr %field_gep162, align 8
  call void @tl_tensor_free(ptr %field_load163)
  %field_gep164 = getelementptr inbounds %Linear, ptr %field_load161, i32 0, i32 1
  %field_load165 = load ptr, ptr %field_gep164, align 8
  call void @tl_tensor_free(ptr %field_load165)
  %field_gep166 = getelementptr inbounds %MLP, ptr %field_load159, i32 0, i32 1
  %field_load167 = load ptr, ptr %field_gep166, align 8
  %field_gep168 = getelementptr inbounds %Linear, ptr %field_load167, i32 0, i32 0
  %field_load169 = load ptr, ptr %field_gep168, align 8
  call void @tl_tensor_free(ptr %field_load169)
  %field_gep170 = getelementptr inbounds %Linear, ptr %field_load167, i32 0, i32 1
  %field_load171 = load ptr, ptr %field_gep170, align 8
  call void @tl_tensor_free(ptr %field_load171)
  %field_gep172 = getelementptr inbounds %GPT, ptr %old_struct_to_free, i32 0, i32 4
  %field_load173 = load ptr, ptr %field_gep172, align 8
  %field_gep174 = getelementptr inbounds %LayerNorm, ptr %field_load173, i32 0, i32 0
  %field_load175 = load ptr, ptr %field_gep174, align 8
  call void @tl_tensor_free(ptr %field_load175)
  %field_gep176 = getelementptr inbounds %LayerNorm, ptr %field_load173, i32 0, i32 1
  %field_load177 = load ptr, ptr %field_gep176, align 8
  call void @tl_tensor_free(ptr %field_load177)
  %field_gep178 = getelementptr inbounds %GPT, ptr %old_struct_to_free, i32 0, i32 5
  %field_load179 = load ptr, ptr %field_gep178, align 8
  %field_gep180 = getelementptr inbounds %Linear, ptr %field_load179, i32 0, i32 0
  %field_load181 = load ptr, ptr %field_gep180, align 8
  call void @tl_tensor_free(ptr %field_load181)
  %field_gep182 = getelementptr inbounds %Linear, ptr %field_load179, i32 0, i32 1
  %field_load183 = load ptr, ptr %field_gep182, align 8
  call void @tl_tensor_free(ptr %field_load183)
  call void @tl_mem_unregister(ptr %old_struct_to_free)
  br label %continue_after_free

continue_after_free:                              ; preds = %free_struct, %entry
  store ptr %call_method57, ptr %m, align 8
  call void @tl_clear_grads()
  %m184 = load ptr, ptr %m, align 8
  call void @tl_mem_unregister(ptr %m184)
  %unreg_field_0 = getelementptr inbounds %GPT, ptr %m184, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_0185 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val186 = load ptr, ptr %unreg_field_0185, align 8
  call void @tl_mem_unregister(ptr %field_val186)
  %unreg_field_1 = getelementptr inbounds %GPT, ptr %m184, i32 0, i32 1
  %field_val187 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val187)
  %unreg_field_0188 = getelementptr inbounds %Embedding, ptr %field_val187, i32 0, i32 0
  %field_val189 = load ptr, ptr %unreg_field_0188, align 8
  call void @tl_mem_unregister(ptr %field_val189)
  %unreg_field_2 = getelementptr inbounds %GPT, ptr %m184, i32 0, i32 2
  %field_val190 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val190)
  %unreg_field_0191 = getelementptr inbounds %Block, ptr %field_val190, i32 0, i32 0
  %field_val192 = load ptr, ptr %unreg_field_0191, align 8
  call void @tl_mem_unregister(ptr %field_val192)
  %unreg_field_0193 = getelementptr inbounds %LayerNorm, ptr %field_val192, i32 0, i32 0
  %field_val194 = load ptr, ptr %unreg_field_0193, align 8
  call void @tl_mem_unregister(ptr %field_val194)
  %unreg_field_1195 = getelementptr inbounds %LayerNorm, ptr %field_val192, i32 0, i32 1
  %field_val196 = load ptr, ptr %unreg_field_1195, align 8
  call void @tl_mem_unregister(ptr %field_val196)
  %unreg_field_1197 = getelementptr inbounds %Block, ptr %field_val190, i32 0, i32 1
  %field_val198 = load ptr, ptr %unreg_field_1197, align 8
  call void @tl_mem_unregister(ptr %field_val198)
  %unreg_field_0199 = getelementptr inbounds %CausalSelfAttention, ptr %field_val198, i32 0, i32 0
  %field_val200 = load ptr, ptr %unreg_field_0199, align 8
  call void @tl_mem_unregister(ptr %field_val200)
  %unreg_field_0201 = getelementptr inbounds %Linear, ptr %field_val200, i32 0, i32 0
  %field_val202 = load ptr, ptr %unreg_field_0201, align 8
  call void @tl_mem_unregister(ptr %field_val202)
  %unreg_field_1203 = getelementptr inbounds %Linear, ptr %field_val200, i32 0, i32 1
  %field_val204 = load ptr, ptr %unreg_field_1203, align 8
  call void @tl_mem_unregister(ptr %field_val204)
  %unreg_field_1205 = getelementptr inbounds %CausalSelfAttention, ptr %field_val198, i32 0, i32 1
  %field_val206 = load ptr, ptr %unreg_field_1205, align 8
  call void @tl_mem_unregister(ptr %field_val206)
  %unreg_field_0207 = getelementptr inbounds %Linear, ptr %field_val206, i32 0, i32 0
  %field_val208 = load ptr, ptr %unreg_field_0207, align 8
  call void @tl_mem_unregister(ptr %field_val208)
  %unreg_field_1209 = getelementptr inbounds %Linear, ptr %field_val206, i32 0, i32 1
  %field_val210 = load ptr, ptr %unreg_field_1209, align 8
  call void @tl_mem_unregister(ptr %field_val210)
  %unreg_field_2211 = getelementptr inbounds %CausalSelfAttention, ptr %field_val198, i32 0, i32 2
  %field_val212 = load ptr, ptr %unreg_field_2211, align 8
  call void @tl_mem_unregister(ptr %field_val212)
  %unreg_field_0213 = getelementptr inbounds %Linear, ptr %field_val212, i32 0, i32 0
  %field_val214 = load ptr, ptr %unreg_field_0213, align 8
  call void @tl_mem_unregister(ptr %field_val214)
  %unreg_field_1215 = getelementptr inbounds %Linear, ptr %field_val212, i32 0, i32 1
  %field_val216 = load ptr, ptr %unreg_field_1215, align 8
  call void @tl_mem_unregister(ptr %field_val216)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val198, i32 0, i32 3
  %field_val217 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val217)
  %unreg_field_0218 = getelementptr inbounds %Linear, ptr %field_val217, i32 0, i32 0
  %field_val219 = load ptr, ptr %unreg_field_0218, align 8
  call void @tl_mem_unregister(ptr %field_val219)
  %unreg_field_1220 = getelementptr inbounds %Linear, ptr %field_val217, i32 0, i32 1
  %field_val221 = load ptr, ptr %unreg_field_1220, align 8
  call void @tl_mem_unregister(ptr %field_val221)
  %unreg_field_2222 = getelementptr inbounds %Block, ptr %field_val190, i32 0, i32 2
  %field_val223 = load ptr, ptr %unreg_field_2222, align 8
  call void @tl_mem_unregister(ptr %field_val223)
  %unreg_field_0224 = getelementptr inbounds %LayerNorm, ptr %field_val223, i32 0, i32 0
  %field_val225 = load ptr, ptr %unreg_field_0224, align 8
  call void @tl_mem_unregister(ptr %field_val225)
  %unreg_field_1226 = getelementptr inbounds %LayerNorm, ptr %field_val223, i32 0, i32 1
  %field_val227 = load ptr, ptr %unreg_field_1226, align 8
  call void @tl_mem_unregister(ptr %field_val227)
  %unreg_field_3228 = getelementptr inbounds %Block, ptr %field_val190, i32 0, i32 3
  %field_val229 = load ptr, ptr %unreg_field_3228, align 8
  call void @tl_mem_unregister(ptr %field_val229)
  %unreg_field_0230 = getelementptr inbounds %MLP, ptr %field_val229, i32 0, i32 0
  %field_val231 = load ptr, ptr %unreg_field_0230, align 8
  call void @tl_mem_unregister(ptr %field_val231)
  %unreg_field_0232 = getelementptr inbounds %Linear, ptr %field_val231, i32 0, i32 0
  %field_val233 = load ptr, ptr %unreg_field_0232, align 8
  call void @tl_mem_unregister(ptr %field_val233)
  %unreg_field_1234 = getelementptr inbounds %Linear, ptr %field_val231, i32 0, i32 1
  %field_val235 = load ptr, ptr %unreg_field_1234, align 8
  call void @tl_mem_unregister(ptr %field_val235)
  %unreg_field_1236 = getelementptr inbounds %MLP, ptr %field_val229, i32 0, i32 1
  %field_val237 = load ptr, ptr %unreg_field_1236, align 8
  call void @tl_mem_unregister(ptr %field_val237)
  %unreg_field_0238 = getelementptr inbounds %Linear, ptr %field_val237, i32 0, i32 0
  %field_val239 = load ptr, ptr %unreg_field_0238, align 8
  call void @tl_mem_unregister(ptr %field_val239)
  %unreg_field_1240 = getelementptr inbounds %Linear, ptr %field_val237, i32 0, i32 1
  %field_val241 = load ptr, ptr %unreg_field_1240, align 8
  call void @tl_mem_unregister(ptr %field_val241)
  %unreg_field_3242 = getelementptr inbounds %GPT, ptr %m184, i32 0, i32 3
  %field_val243 = load ptr, ptr %unreg_field_3242, align 8
  call void @tl_mem_unregister(ptr %field_val243)
  %unreg_field_0244 = getelementptr inbounds %Block, ptr %field_val243, i32 0, i32 0
  %field_val245 = load ptr, ptr %unreg_field_0244, align 8
  call void @tl_mem_unregister(ptr %field_val245)
  %unreg_field_0246 = getelementptr inbounds %LayerNorm, ptr %field_val245, i32 0, i32 0
  %field_val247 = load ptr, ptr %unreg_field_0246, align 8
  call void @tl_mem_unregister(ptr %field_val247)
  %unreg_field_1248 = getelementptr inbounds %LayerNorm, ptr %field_val245, i32 0, i32 1
  %field_val249 = load ptr, ptr %unreg_field_1248, align 8
  call void @tl_mem_unregister(ptr %field_val249)
  %unreg_field_1250 = getelementptr inbounds %Block, ptr %field_val243, i32 0, i32 1
  %field_val251 = load ptr, ptr %unreg_field_1250, align 8
  call void @tl_mem_unregister(ptr %field_val251)
  %unreg_field_0252 = getelementptr inbounds %CausalSelfAttention, ptr %field_val251, i32 0, i32 0
  %field_val253 = load ptr, ptr %unreg_field_0252, align 8
  call void @tl_mem_unregister(ptr %field_val253)
  %unreg_field_0254 = getelementptr inbounds %Linear, ptr %field_val253, i32 0, i32 0
  %field_val255 = load ptr, ptr %unreg_field_0254, align 8
  call void @tl_mem_unregister(ptr %field_val255)
  %unreg_field_1256 = getelementptr inbounds %Linear, ptr %field_val253, i32 0, i32 1
  %field_val257 = load ptr, ptr %unreg_field_1256, align 8
  call void @tl_mem_unregister(ptr %field_val257)
  %unreg_field_1258 = getelementptr inbounds %CausalSelfAttention, ptr %field_val251, i32 0, i32 1
  %field_val259 = load ptr, ptr %unreg_field_1258, align 8
  call void @tl_mem_unregister(ptr %field_val259)
  %unreg_field_0260 = getelementptr inbounds %Linear, ptr %field_val259, i32 0, i32 0
  %field_val261 = load ptr, ptr %unreg_field_0260, align 8
  call void @tl_mem_unregister(ptr %field_val261)
  %unreg_field_1262 = getelementptr inbounds %Linear, ptr %field_val259, i32 0, i32 1
  %field_val263 = load ptr, ptr %unreg_field_1262, align 8
  call void @tl_mem_unregister(ptr %field_val263)
  %unreg_field_2264 = getelementptr inbounds %CausalSelfAttention, ptr %field_val251, i32 0, i32 2
  %field_val265 = load ptr, ptr %unreg_field_2264, align 8
  call void @tl_mem_unregister(ptr %field_val265)
  %unreg_field_0266 = getelementptr inbounds %Linear, ptr %field_val265, i32 0, i32 0
  %field_val267 = load ptr, ptr %unreg_field_0266, align 8
  call void @tl_mem_unregister(ptr %field_val267)
  %unreg_field_1268 = getelementptr inbounds %Linear, ptr %field_val265, i32 0, i32 1
  %field_val269 = load ptr, ptr %unreg_field_1268, align 8
  call void @tl_mem_unregister(ptr %field_val269)
  %unreg_field_3270 = getelementptr inbounds %CausalSelfAttention, ptr %field_val251, i32 0, i32 3
  %field_val271 = load ptr, ptr %unreg_field_3270, align 8
  call void @tl_mem_unregister(ptr %field_val271)
  %unreg_field_0272 = getelementptr inbounds %Linear, ptr %field_val271, i32 0, i32 0
  %field_val273 = load ptr, ptr %unreg_field_0272, align 8
  call void @tl_mem_unregister(ptr %field_val273)
  %unreg_field_1274 = getelementptr inbounds %Linear, ptr %field_val271, i32 0, i32 1
  %field_val275 = load ptr, ptr %unreg_field_1274, align 8
  call void @tl_mem_unregister(ptr %field_val275)
  %unreg_field_2276 = getelementptr inbounds %Block, ptr %field_val243, i32 0, i32 2
  %field_val277 = load ptr, ptr %unreg_field_2276, align 8
  call void @tl_mem_unregister(ptr %field_val277)
  %unreg_field_0278 = getelementptr inbounds %LayerNorm, ptr %field_val277, i32 0, i32 0
  %field_val279 = load ptr, ptr %unreg_field_0278, align 8
  call void @tl_mem_unregister(ptr %field_val279)
  %unreg_field_1280 = getelementptr inbounds %LayerNorm, ptr %field_val277, i32 0, i32 1
  %field_val281 = load ptr, ptr %unreg_field_1280, align 8
  call void @tl_mem_unregister(ptr %field_val281)
  %unreg_field_3282 = getelementptr inbounds %Block, ptr %field_val243, i32 0, i32 3
  %field_val283 = load ptr, ptr %unreg_field_3282, align 8
  call void @tl_mem_unregister(ptr %field_val283)
  %unreg_field_0284 = getelementptr inbounds %MLP, ptr %field_val283, i32 0, i32 0
  %field_val285 = load ptr, ptr %unreg_field_0284, align 8
  call void @tl_mem_unregister(ptr %field_val285)
  %unreg_field_0286 = getelementptr inbounds %Linear, ptr %field_val285, i32 0, i32 0
  %field_val287 = load ptr, ptr %unreg_field_0286, align 8
  call void @tl_mem_unregister(ptr %field_val287)
  %unreg_field_1288 = getelementptr inbounds %Linear, ptr %field_val285, i32 0, i32 1
  %field_val289 = load ptr, ptr %unreg_field_1288, align 8
  call void @tl_mem_unregister(ptr %field_val289)
  %unreg_field_1290 = getelementptr inbounds %MLP, ptr %field_val283, i32 0, i32 1
  %field_val291 = load ptr, ptr %unreg_field_1290, align 8
  call void @tl_mem_unregister(ptr %field_val291)
  %unreg_field_0292 = getelementptr inbounds %Linear, ptr %field_val291, i32 0, i32 0
  %field_val293 = load ptr, ptr %unreg_field_0292, align 8
  call void @tl_mem_unregister(ptr %field_val293)
  %unreg_field_1294 = getelementptr inbounds %Linear, ptr %field_val291, i32 0, i32 1
  %field_val295 = load ptr, ptr %unreg_field_1294, align 8
  call void @tl_mem_unregister(ptr %field_val295)
  %unreg_field_4 = getelementptr inbounds %GPT, ptr %m184, i32 0, i32 4
  %field_val296 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val296)
  %unreg_field_0297 = getelementptr inbounds %LayerNorm, ptr %field_val296, i32 0, i32 0
  %field_val298 = load ptr, ptr %unreg_field_0297, align 8
  call void @tl_mem_unregister(ptr %field_val298)
  %unreg_field_1299 = getelementptr inbounds %LayerNorm, ptr %field_val296, i32 0, i32 1
  %field_val300 = load ptr, ptr %unreg_field_1299, align 8
  call void @tl_mem_unregister(ptr %field_val300)
  %unreg_field_5 = getelementptr inbounds %GPT, ptr %m184, i32 0, i32 5
  %field_val301 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val301)
  %unreg_field_0302 = getelementptr inbounds %Linear, ptr %field_val301, i32 0, i32 0
  %field_val303 = load ptr, ptr %unreg_field_0302, align 8
  call void @tl_mem_unregister(ptr %field_val303)
  %unreg_field_1304 = getelementptr inbounds %Linear, ptr %field_val301, i32 0, i32 1
  %field_val305 = load ptr, ptr %unreg_field_1304, align 8
  call void @tl_mem_unregister(ptr %field_val305)
  %tensor_to_free = load ptr, ptr %logits, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free306 = load ptr, ptr %Y_flat, align 8
  call void @tl_tensor_free(ptr %tensor_to_free306)
  %tensor_to_free307 = load ptr, ptr %loss, align 8
  call void @tl_tensor_free(ptr %tensor_to_free307)
  %tensor_to_free308 = load ptr, ptr %Y, align 8
  call void @tl_tensor_free(ptr %tensor_to_free308)
  %tensor_to_free309 = load ptr, ptr %logits_flat, align 8
  call void @tl_tensor_free(ptr %tensor_to_free309)
  %tensor_to_free310 = load ptr, ptr %X, align 8
  call void @tl_tensor_free(ptr %tensor_to_free310)
  call void @tl_mem_exit_scope()
  ret ptr %m184
}

define void @infer(ptr %model, i64 %i, i64 %j) {
entry:
  %dims_alloca57 = alloca [1 x i64], align 8
  %dims_alloca48 = alloca [2 x i64], align 8
  %preds = alloca ptr, align 16
  %loss = alloca ptr, align 16
  %Y_flat = alloca ptr, align 16
  %dims_alloca40 = alloca [1 x i64], align 8
  %logits_flat = alloca ptr, align 16
  %dims_alloca34 = alloca [2 x i64], align 8
  %Y = alloca ptr, align 16
  %dims_alloca28 = alloca [2 x i64], align 8
  %logits = alloca ptr, align 16
  %X = alloca ptr, align 16
  %dims_alloca = alloca [2 x i64], align 8
  %expected = alloca i64, align 16
  %m = alloca ptr, align 16
  %j3 = alloca i64, align 16
  %i2 = alloca i64, align 16
  %model1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %model, ptr %model1, align 8
  store i64 %i, ptr %i2, align 8
  store i64 %j, ptr %j3, align 8
  %model4 = load ptr, ptr %model1, align 8
  %call_method = call ptr @tl_GPT_clone(ptr %model4)
  call void @tl_mem_register_struct(ptr %call_method)
  store ptr %call_method, ptr %m, align 8
  %i5 = load i64, ptr %i2, align 8
  %j6 = load i64, ptr %j3, align 8
  %addtmp = add i64 %i5, %j6
  store i64 %addtmp, ptr %expected, align 8
  %buf_void = call ptr @tl_alloc_tmp(i64 16)
  %i7 = load i64, ptr %i2, align 8
  %i2f = sitofp i64 %i7 to float
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %i2f, ptr %elem_ptr, align 4
  %j8 = load i64, ptr %j3, align 8
  %i2f9 = sitofp i64 %j8 to float
  %elem_ptr10 = getelementptr inbounds float, ptr %buf_void, i64 1
  store float %i2f9, ptr %elem_ptr10, align 4
  %elem_ptr11 = getelementptr inbounds float, ptr %buf_void, i64 2
  store float 0.000000e+00, ptr %elem_ptr11, align 4
  %elem_ptr12 = getelementptr inbounds float, ptr %buf_void, i64 3
  store float 0.000000e+00, ptr %elem_ptr12, align 4
  %shape_alloc = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr = getelementptr inbounds i64, ptr %shape_alloc, i64 0
  store i64 4, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %shape_alloc)
  call void @tl_free_tmp(ptr %buf_void)
  call void @tl_free_tmp(ptr %shape_alloc)
  %dim_ptr_0 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0, align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 4, ptr %dim_ptr, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %new_tensor, ptr %dims_ptr, i64 2)
  call void @tl_tensor_free(ptr %new_tensor)
  call void @tl_mem_unregister(ptr %reshape_dims_res)
  store ptr %reshape_dims_res, ptr %X, align 8
  %m13 = load ptr, ptr %m, align 8
  %X14 = load ptr, ptr %X, align 8
  %call_method15 = call ptr @tl_GPT_forward(ptr %m13, ptr %X14)
  call void @tl_mem_register_tensor(ptr %call_method15)
  call void @tl_mem_unregister(ptr %call_method15)
  store ptr %call_method15, ptr %logits, align 8
  %buf_void16 = call ptr @tl_alloc_tmp(i64 16)
  %j17 = load i64, ptr %j3, align 8
  %i2f18 = sitofp i64 %j17 to float
  %elem_ptr19 = getelementptr inbounds float, ptr %buf_void16, i64 0
  store float %i2f18, ptr %elem_ptr19, align 4
  %expected20 = load i64, ptr %expected, align 8
  %i2f21 = sitofp i64 %expected20 to float
  %elem_ptr22 = getelementptr inbounds float, ptr %buf_void16, i64 1
  store float %i2f21, ptr %elem_ptr22, align 4
  %elem_ptr23 = getelementptr inbounds float, ptr %buf_void16, i64 2
  store float 0.000000e+00, ptr %elem_ptr23, align 4
  %elem_ptr24 = getelementptr inbounds float, ptr %buf_void16, i64 3
  store float 0.000000e+00, ptr %elem_ptr24, align 4
  %shape_alloc25 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr26 = getelementptr inbounds i64, ptr %shape_alloc25, i64 0
  store i64 4, ptr %shape_ptr26, align 8
  %new_tensor27 = call ptr @tl_tensor_new(ptr %buf_void16, i64 1, ptr %shape_alloc25)
  call void @tl_free_tmp(ptr %buf_void16)
  call void @tl_free_tmp(ptr %shape_alloc25)
  %dim_ptr_029 = getelementptr [2 x i64], ptr %dims_alloca28, i64 0, i64 0
  store i64 1, ptr %dim_ptr_029, align 8
  %dim_ptr30 = getelementptr [2 x i64], ptr %dims_alloca28, i64 0, i64 1
  store i64 4, ptr %dim_ptr30, align 8
  %dims_ptr31 = getelementptr [2 x i64], ptr %dims_alloca28, i64 0, i64 0
  %reshape_dims_res32 = call ptr @tl_tensor_reshape_dims(ptr %new_tensor27, ptr %dims_ptr31, i64 2)
  call void @tl_tensor_free(ptr %new_tensor27)
  call void @tl_mem_unregister(ptr %reshape_dims_res32)
  store ptr %reshape_dims_res32, ptr %Y, align 8
  %logits33 = load ptr, ptr %logits, align 8
  %dim_ptr_035 = getelementptr [2 x i64], ptr %dims_alloca34, i64 0, i64 0
  store i64 4, ptr %dim_ptr_035, align 8
  %dim_ptr36 = getelementptr [2 x i64], ptr %dims_alloca34, i64 0, i64 1
  store i64 200, ptr %dim_ptr36, align 8
  %dims_ptr37 = getelementptr [2 x i64], ptr %dims_alloca34, i64 0, i64 0
  %reshape_dims_res38 = call ptr @tl_tensor_reshape_dims(ptr %logits33, ptr %dims_ptr37, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res38)
  store ptr %reshape_dims_res38, ptr %logits_flat, align 8
  %Y39 = load ptr, ptr %Y, align 8
  %dim_ptr_041 = getelementptr [1 x i64], ptr %dims_alloca40, i64 0, i64 0
  store i64 4, ptr %dim_ptr_041, align 8
  %dims_ptr42 = getelementptr [1 x i64], ptr %dims_alloca40, i64 0, i64 0
  %reshape_dims_res43 = call ptr @tl_tensor_reshape_dims(ptr %Y39, ptr %dims_ptr42, i64 1)
  call void @tl_mem_unregister(ptr %reshape_dims_res43)
  store ptr %reshape_dims_res43, ptr %Y_flat, align 8
  %logits_flat44 = load ptr, ptr %logits_flat, align 8
  %Y_flat45 = load ptr, ptr %Y_flat, align 8
  %ce_res = call ptr @tl_tensor_cross_entropy(ptr %logits_flat44, ptr %Y_flat45)
  call void @tl_mem_unregister(ptr %ce_res)
  store ptr %ce_res, ptr %loss, align 8
  %logits46 = load ptr, ptr %logits, align 8
  %argmax_res = call ptr @tl_tensor_argmax(ptr %logits46, i64 2, i1 false)
  call void @tl_mem_register_tensor(ptr %argmax_res)
  call void @tl_mem_unregister(ptr %argmax_res)
  store ptr %argmax_res, ptr %preds, align 8
  call void @tl_print_string(ptr @str_literal)
  %preds47 = load ptr, ptr %preds, align 8
  %dim_ptr_049 = getelementptr [2 x i64], ptr %dims_alloca48, i64 0, i64 0
  store i64 1, ptr %dim_ptr_049, align 8
  %dim_ptr50 = getelementptr [2 x i64], ptr %dims_alloca48, i64 0, i64 1
  store i64 4, ptr %dim_ptr50, align 8
  %dims_ptr51 = getelementptr [2 x i64], ptr %dims_alloca48, i64 0, i64 0
  %reshape_dims_res52 = call ptr @tl_tensor_reshape_dims(ptr %preds47, ptr %dims_ptr51, i64 2)
  call void @tl_tensor_print_2(ptr %reshape_dims_res52)
  call void @tl_tensor_free(ptr %reshape_dims_res52)
  call void @tl_print_string(ptr @str_literal.107)
  %i53 = load i64, ptr %i2, align 8
  call void @tl_print_i64(i64 %i53)
  %j54 = load i64, ptr %j3, align 8
  call void @tl_print_i64(i64 %j54)
  call void @tl_print_string(ptr @str_literal.108)
  %expected55 = load i64, ptr %expected, align 8
  call void @tl_print_i64(i64 %expected55)
  call void @tl_print_string(ptr @str_literal.109)
  %loss56 = load ptr, ptr %loss, align 8
  %dim_ptr_058 = getelementptr [1 x i64], ptr %dims_alloca57, i64 0, i64 0
  store i64 1, ptr %dim_ptr_058, align 8
  %dims_ptr59 = getelementptr [1 x i64], ptr %dims_alloca57, i64 0, i64 0
  %reshape_dims_res60 = call ptr @tl_tensor_reshape_dims(ptr %loss56, ptr %dims_ptr59, i64 1)
  call void @tl_tensor_print_1(ptr %reshape_dims_res60)
  call void @tl_tensor_free(ptr %reshape_dims_res60)
  %tensor_to_free = load ptr, ptr %X, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free61 = load ptr, ptr %loss, align 8
  call void @tl_tensor_free(ptr %tensor_to_free61)
  %tensor_to_free62 = load ptr, ptr %preds, align 8
  call void @tl_tensor_free(ptr %tensor_to_free62)
  %tensor_to_free63 = load ptr, ptr %Y_flat, align 8
  call void @tl_tensor_free(ptr %tensor_to_free63)
  %tensor_to_free64 = load ptr, ptr %Y, align 8
  call void @tl_tensor_free(ptr %tensor_to_free64)
  %tensor_to_free65 = load ptr, ptr %logits_flat, align 8
  call void @tl_tensor_free(ptr %tensor_to_free65)
  %struct_to_free = load ptr, ptr %m, align 8
  %field_gep = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 0
  %field_load = load ptr, ptr %field_gep, align 8
  %field_gep66 = getelementptr inbounds %Embedding, ptr %field_load, i32 0, i32 0
  %field_load67 = load ptr, ptr %field_gep66, align 8
  call void @tl_tensor_free(ptr %field_load67)
  %field_gep68 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 1
  %field_load69 = load ptr, ptr %field_gep68, align 8
  %field_gep70 = getelementptr inbounds %Embedding, ptr %field_load69, i32 0, i32 0
  %field_load71 = load ptr, ptr %field_gep70, align 8
  call void @tl_tensor_free(ptr %field_load71)
  %field_gep72 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 2
  %field_load73 = load ptr, ptr %field_gep72, align 8
  %field_gep74 = getelementptr inbounds %Block, ptr %field_load73, i32 0, i32 0
  %field_load75 = load ptr, ptr %field_gep74, align 8
  %field_gep76 = getelementptr inbounds %LayerNorm, ptr %field_load75, i32 0, i32 0
  %field_load77 = load ptr, ptr %field_gep76, align 8
  call void @tl_tensor_free(ptr %field_load77)
  %field_gep78 = getelementptr inbounds %LayerNorm, ptr %field_load75, i32 0, i32 1
  %field_load79 = load ptr, ptr %field_gep78, align 8
  call void @tl_tensor_free(ptr %field_load79)
  %field_gep80 = getelementptr inbounds %Block, ptr %field_load73, i32 0, i32 1
  %field_load81 = load ptr, ptr %field_gep80, align 8
  %field_gep82 = getelementptr inbounds %CausalSelfAttention, ptr %field_load81, i32 0, i32 0
  %field_load83 = load ptr, ptr %field_gep82, align 8
  %field_gep84 = getelementptr inbounds %Linear, ptr %field_load83, i32 0, i32 0
  %field_load85 = load ptr, ptr %field_gep84, align 8
  call void @tl_tensor_free(ptr %field_load85)
  %field_gep86 = getelementptr inbounds %Linear, ptr %field_load83, i32 0, i32 1
  %field_load87 = load ptr, ptr %field_gep86, align 8
  call void @tl_tensor_free(ptr %field_load87)
  %field_gep88 = getelementptr inbounds %CausalSelfAttention, ptr %field_load81, i32 0, i32 1
  %field_load89 = load ptr, ptr %field_gep88, align 8
  %field_gep90 = getelementptr inbounds %Linear, ptr %field_load89, i32 0, i32 0
  %field_load91 = load ptr, ptr %field_gep90, align 8
  call void @tl_tensor_free(ptr %field_load91)
  %field_gep92 = getelementptr inbounds %Linear, ptr %field_load89, i32 0, i32 1
  %field_load93 = load ptr, ptr %field_gep92, align 8
  call void @tl_tensor_free(ptr %field_load93)
  %field_gep94 = getelementptr inbounds %CausalSelfAttention, ptr %field_load81, i32 0, i32 2
  %field_load95 = load ptr, ptr %field_gep94, align 8
  %field_gep96 = getelementptr inbounds %Linear, ptr %field_load95, i32 0, i32 0
  %field_load97 = load ptr, ptr %field_gep96, align 8
  call void @tl_tensor_free(ptr %field_load97)
  %field_gep98 = getelementptr inbounds %Linear, ptr %field_load95, i32 0, i32 1
  %field_load99 = load ptr, ptr %field_gep98, align 8
  call void @tl_tensor_free(ptr %field_load99)
  %field_gep100 = getelementptr inbounds %CausalSelfAttention, ptr %field_load81, i32 0, i32 3
  %field_load101 = load ptr, ptr %field_gep100, align 8
  %field_gep102 = getelementptr inbounds %Linear, ptr %field_load101, i32 0, i32 0
  %field_load103 = load ptr, ptr %field_gep102, align 8
  call void @tl_tensor_free(ptr %field_load103)
  %field_gep104 = getelementptr inbounds %Linear, ptr %field_load101, i32 0, i32 1
  %field_load105 = load ptr, ptr %field_gep104, align 8
  call void @tl_tensor_free(ptr %field_load105)
  %field_gep106 = getelementptr inbounds %Block, ptr %field_load73, i32 0, i32 2
  %field_load107 = load ptr, ptr %field_gep106, align 8
  %field_gep108 = getelementptr inbounds %LayerNorm, ptr %field_load107, i32 0, i32 0
  %field_load109 = load ptr, ptr %field_gep108, align 8
  call void @tl_tensor_free(ptr %field_load109)
  %field_gep110 = getelementptr inbounds %LayerNorm, ptr %field_load107, i32 0, i32 1
  %field_load111 = load ptr, ptr %field_gep110, align 8
  call void @tl_tensor_free(ptr %field_load111)
  %field_gep112 = getelementptr inbounds %Block, ptr %field_load73, i32 0, i32 3
  %field_load113 = load ptr, ptr %field_gep112, align 8
  %field_gep114 = getelementptr inbounds %MLP, ptr %field_load113, i32 0, i32 0
  %field_load115 = load ptr, ptr %field_gep114, align 8
  %field_gep116 = getelementptr inbounds %Linear, ptr %field_load115, i32 0, i32 0
  %field_load117 = load ptr, ptr %field_gep116, align 8
  call void @tl_tensor_free(ptr %field_load117)
  %field_gep118 = getelementptr inbounds %Linear, ptr %field_load115, i32 0, i32 1
  %field_load119 = load ptr, ptr %field_gep118, align 8
  call void @tl_tensor_free(ptr %field_load119)
  %field_gep120 = getelementptr inbounds %MLP, ptr %field_load113, i32 0, i32 1
  %field_load121 = load ptr, ptr %field_gep120, align 8
  %field_gep122 = getelementptr inbounds %Linear, ptr %field_load121, i32 0, i32 0
  %field_load123 = load ptr, ptr %field_gep122, align 8
  call void @tl_tensor_free(ptr %field_load123)
  %field_gep124 = getelementptr inbounds %Linear, ptr %field_load121, i32 0, i32 1
  %field_load125 = load ptr, ptr %field_gep124, align 8
  call void @tl_tensor_free(ptr %field_load125)
  %field_gep126 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 3
  %field_load127 = load ptr, ptr %field_gep126, align 8
  %field_gep128 = getelementptr inbounds %Block, ptr %field_load127, i32 0, i32 0
  %field_load129 = load ptr, ptr %field_gep128, align 8
  %field_gep130 = getelementptr inbounds %LayerNorm, ptr %field_load129, i32 0, i32 0
  %field_load131 = load ptr, ptr %field_gep130, align 8
  call void @tl_tensor_free(ptr %field_load131)
  %field_gep132 = getelementptr inbounds %LayerNorm, ptr %field_load129, i32 0, i32 1
  %field_load133 = load ptr, ptr %field_gep132, align 8
  call void @tl_tensor_free(ptr %field_load133)
  %field_gep134 = getelementptr inbounds %Block, ptr %field_load127, i32 0, i32 1
  %field_load135 = load ptr, ptr %field_gep134, align 8
  %field_gep136 = getelementptr inbounds %CausalSelfAttention, ptr %field_load135, i32 0, i32 0
  %field_load137 = load ptr, ptr %field_gep136, align 8
  %field_gep138 = getelementptr inbounds %Linear, ptr %field_load137, i32 0, i32 0
  %field_load139 = load ptr, ptr %field_gep138, align 8
  call void @tl_tensor_free(ptr %field_load139)
  %field_gep140 = getelementptr inbounds %Linear, ptr %field_load137, i32 0, i32 1
  %field_load141 = load ptr, ptr %field_gep140, align 8
  call void @tl_tensor_free(ptr %field_load141)
  %field_gep142 = getelementptr inbounds %CausalSelfAttention, ptr %field_load135, i32 0, i32 1
  %field_load143 = load ptr, ptr %field_gep142, align 8
  %field_gep144 = getelementptr inbounds %Linear, ptr %field_load143, i32 0, i32 0
  %field_load145 = load ptr, ptr %field_gep144, align 8
  call void @tl_tensor_free(ptr %field_load145)
  %field_gep146 = getelementptr inbounds %Linear, ptr %field_load143, i32 0, i32 1
  %field_load147 = load ptr, ptr %field_gep146, align 8
  call void @tl_tensor_free(ptr %field_load147)
  %field_gep148 = getelementptr inbounds %CausalSelfAttention, ptr %field_load135, i32 0, i32 2
  %field_load149 = load ptr, ptr %field_gep148, align 8
  %field_gep150 = getelementptr inbounds %Linear, ptr %field_load149, i32 0, i32 0
  %field_load151 = load ptr, ptr %field_gep150, align 8
  call void @tl_tensor_free(ptr %field_load151)
  %field_gep152 = getelementptr inbounds %Linear, ptr %field_load149, i32 0, i32 1
  %field_load153 = load ptr, ptr %field_gep152, align 8
  call void @tl_tensor_free(ptr %field_load153)
  %field_gep154 = getelementptr inbounds %CausalSelfAttention, ptr %field_load135, i32 0, i32 3
  %field_load155 = load ptr, ptr %field_gep154, align 8
  %field_gep156 = getelementptr inbounds %Linear, ptr %field_load155, i32 0, i32 0
  %field_load157 = load ptr, ptr %field_gep156, align 8
  call void @tl_tensor_free(ptr %field_load157)
  %field_gep158 = getelementptr inbounds %Linear, ptr %field_load155, i32 0, i32 1
  %field_load159 = load ptr, ptr %field_gep158, align 8
  call void @tl_tensor_free(ptr %field_load159)
  %field_gep160 = getelementptr inbounds %Block, ptr %field_load127, i32 0, i32 2
  %field_load161 = load ptr, ptr %field_gep160, align 8
  %field_gep162 = getelementptr inbounds %LayerNorm, ptr %field_load161, i32 0, i32 0
  %field_load163 = load ptr, ptr %field_gep162, align 8
  call void @tl_tensor_free(ptr %field_load163)
  %field_gep164 = getelementptr inbounds %LayerNorm, ptr %field_load161, i32 0, i32 1
  %field_load165 = load ptr, ptr %field_gep164, align 8
  call void @tl_tensor_free(ptr %field_load165)
  %field_gep166 = getelementptr inbounds %Block, ptr %field_load127, i32 0, i32 3
  %field_load167 = load ptr, ptr %field_gep166, align 8
  %field_gep168 = getelementptr inbounds %MLP, ptr %field_load167, i32 0, i32 0
  %field_load169 = load ptr, ptr %field_gep168, align 8
  %field_gep170 = getelementptr inbounds %Linear, ptr %field_load169, i32 0, i32 0
  %field_load171 = load ptr, ptr %field_gep170, align 8
  call void @tl_tensor_free(ptr %field_load171)
  %field_gep172 = getelementptr inbounds %Linear, ptr %field_load169, i32 0, i32 1
  %field_load173 = load ptr, ptr %field_gep172, align 8
  call void @tl_tensor_free(ptr %field_load173)
  %field_gep174 = getelementptr inbounds %MLP, ptr %field_load167, i32 0, i32 1
  %field_load175 = load ptr, ptr %field_gep174, align 8
  %field_gep176 = getelementptr inbounds %Linear, ptr %field_load175, i32 0, i32 0
  %field_load177 = load ptr, ptr %field_gep176, align 8
  call void @tl_tensor_free(ptr %field_load177)
  %field_gep178 = getelementptr inbounds %Linear, ptr %field_load175, i32 0, i32 1
  %field_load179 = load ptr, ptr %field_gep178, align 8
  call void @tl_tensor_free(ptr %field_load179)
  %field_gep180 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 4
  %field_load181 = load ptr, ptr %field_gep180, align 8
  %field_gep182 = getelementptr inbounds %LayerNorm, ptr %field_load181, i32 0, i32 0
  %field_load183 = load ptr, ptr %field_gep182, align 8
  call void @tl_tensor_free(ptr %field_load183)
  %field_gep184 = getelementptr inbounds %LayerNorm, ptr %field_load181, i32 0, i32 1
  %field_load185 = load ptr, ptr %field_gep184, align 8
  call void @tl_tensor_free(ptr %field_load185)
  %field_gep186 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 5
  %field_load187 = load ptr, ptr %field_gep186, align 8
  %field_gep188 = getelementptr inbounds %Linear, ptr %field_load187, i32 0, i32 0
  %field_load189 = load ptr, ptr %field_gep188, align 8
  call void @tl_tensor_free(ptr %field_load189)
  %field_gep190 = getelementptr inbounds %Linear, ptr %field_load187, i32 0, i32 1
  %field_load191 = load ptr, ptr %field_gep190, align 8
  call void @tl_tensor_free(ptr %field_load191)
  call void @tl_mem_unregister(ptr %struct_to_free)
  %tensor_to_free192 = load ptr, ptr %logits, align 8
  call void @tl_tensor_free(ptr %tensor_to_free192)
  call void @tl_mem_exit_scope()
  ret void
}

define void @main() {
entry:
  %epoch = alloca i64, align 16
  %model = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  call void @tl_arena_init(i64 409600)
  call void @tl_print_string(ptr @str_literal.110)
  %static_call = call ptr @tl_GPT_new(i64 200, i64 64)
  call void @tl_mem_unregister(ptr %static_call)
  store ptr %static_call, ptr %model, align 8
  call void @tl_print_string(ptr @str_literal.111)
  br label %for_header

for_header:                                       ; preds = %merge, %entry
  %for_idx = phi i64 [ %next_idx, %merge ], [ 0, %entry ]
  %for_cond = icmp slt i64 %for_idx, 200
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %epoch, align 8
  %model1 = load ptr, ptr %model, align 8
  %call_tmp = call ptr @train_step(ptr %model1, float 0x3FA99999A0000000, i64 12, i64 34)
  call void @tl_mem_unregister(ptr %call_tmp)
  %old_struct_to_free = load ptr, ptr %model, align 8
  %is_not_null = icmp ne ptr %old_struct_to_free, null
  %are_diff = icmp ne ptr %old_struct_to_free, %call_tmp
  %can_free_1 = and i1 %is_not_null, true
  %can_free = and i1 %can_free_1, %are_diff
  br i1 %can_free, label %free_struct, label %continue_after_free

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal.113)
  %model1312 = load ptr, ptr %model, align 8
  call void @infer(ptr %model1312, i64 12, i64 34)
  %model1313 = load ptr, ptr %model, align 8
  call void @infer(ptr %model1313, i64 50, i64 50)
  %model1314 = load ptr, ptr %model, align 8
  call void @infer(ptr %model1314, i64 99, i64 99)
  call void @tl_print_string(ptr @str_literal.114)
  %struct_to_free = load ptr, ptr %model, align 8
  %field_gep1315 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 0
  %field_load1316 = load ptr, ptr %field_gep1315, align 8
  %field_gep1317 = getelementptr inbounds %Embedding, ptr %field_load1316, i32 0, i32 0
  %field_load1318 = load ptr, ptr %field_gep1317, align 8
  call void @tl_tensor_free(ptr %field_load1318)
  %field_gep1319 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 1
  %field_load1320 = load ptr, ptr %field_gep1319, align 8
  %field_gep1321 = getelementptr inbounds %Embedding, ptr %field_load1320, i32 0, i32 0
  %field_load1322 = load ptr, ptr %field_gep1321, align 8
  call void @tl_tensor_free(ptr %field_load1322)
  %field_gep1323 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 2
  %field_load1324 = load ptr, ptr %field_gep1323, align 8
  %field_gep1325 = getelementptr inbounds %Block, ptr %field_load1324, i32 0, i32 0
  %field_load1326 = load ptr, ptr %field_gep1325, align 8
  %field_gep1327 = getelementptr inbounds %LayerNorm, ptr %field_load1326, i32 0, i32 0
  %field_load1328 = load ptr, ptr %field_gep1327, align 8
  call void @tl_tensor_free(ptr %field_load1328)
  %field_gep1329 = getelementptr inbounds %LayerNorm, ptr %field_load1326, i32 0, i32 1
  %field_load1330 = load ptr, ptr %field_gep1329, align 8
  call void @tl_tensor_free(ptr %field_load1330)
  %field_gep1331 = getelementptr inbounds %Block, ptr %field_load1324, i32 0, i32 1
  %field_load1332 = load ptr, ptr %field_gep1331, align 8
  %field_gep1333 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1332, i32 0, i32 0
  %field_load1334 = load ptr, ptr %field_gep1333, align 8
  %field_gep1335 = getelementptr inbounds %Linear, ptr %field_load1334, i32 0, i32 0
  %field_load1336 = load ptr, ptr %field_gep1335, align 8
  call void @tl_tensor_free(ptr %field_load1336)
  %field_gep1337 = getelementptr inbounds %Linear, ptr %field_load1334, i32 0, i32 1
  %field_load1338 = load ptr, ptr %field_gep1337, align 8
  call void @tl_tensor_free(ptr %field_load1338)
  %field_gep1339 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1332, i32 0, i32 1
  %field_load1340 = load ptr, ptr %field_gep1339, align 8
  %field_gep1341 = getelementptr inbounds %Linear, ptr %field_load1340, i32 0, i32 0
  %field_load1342 = load ptr, ptr %field_gep1341, align 8
  call void @tl_tensor_free(ptr %field_load1342)
  %field_gep1343 = getelementptr inbounds %Linear, ptr %field_load1340, i32 0, i32 1
  %field_load1344 = load ptr, ptr %field_gep1343, align 8
  call void @tl_tensor_free(ptr %field_load1344)
  %field_gep1345 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1332, i32 0, i32 2
  %field_load1346 = load ptr, ptr %field_gep1345, align 8
  %field_gep1347 = getelementptr inbounds %Linear, ptr %field_load1346, i32 0, i32 0
  %field_load1348 = load ptr, ptr %field_gep1347, align 8
  call void @tl_tensor_free(ptr %field_load1348)
  %field_gep1349 = getelementptr inbounds %Linear, ptr %field_load1346, i32 0, i32 1
  %field_load1350 = load ptr, ptr %field_gep1349, align 8
  call void @tl_tensor_free(ptr %field_load1350)
  %field_gep1351 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1332, i32 0, i32 3
  %field_load1352 = load ptr, ptr %field_gep1351, align 8
  %field_gep1353 = getelementptr inbounds %Linear, ptr %field_load1352, i32 0, i32 0
  %field_load1354 = load ptr, ptr %field_gep1353, align 8
  call void @tl_tensor_free(ptr %field_load1354)
  %field_gep1355 = getelementptr inbounds %Linear, ptr %field_load1352, i32 0, i32 1
  %field_load1356 = load ptr, ptr %field_gep1355, align 8
  call void @tl_tensor_free(ptr %field_load1356)
  %field_gep1357 = getelementptr inbounds %Block, ptr %field_load1324, i32 0, i32 2
  %field_load1358 = load ptr, ptr %field_gep1357, align 8
  %field_gep1359 = getelementptr inbounds %LayerNorm, ptr %field_load1358, i32 0, i32 0
  %field_load1360 = load ptr, ptr %field_gep1359, align 8
  call void @tl_tensor_free(ptr %field_load1360)
  %field_gep1361 = getelementptr inbounds %LayerNorm, ptr %field_load1358, i32 0, i32 1
  %field_load1362 = load ptr, ptr %field_gep1361, align 8
  call void @tl_tensor_free(ptr %field_load1362)
  %field_gep1363 = getelementptr inbounds %Block, ptr %field_load1324, i32 0, i32 3
  %field_load1364 = load ptr, ptr %field_gep1363, align 8
  %field_gep1365 = getelementptr inbounds %MLP, ptr %field_load1364, i32 0, i32 0
  %field_load1366 = load ptr, ptr %field_gep1365, align 8
  %field_gep1367 = getelementptr inbounds %Linear, ptr %field_load1366, i32 0, i32 0
  %field_load1368 = load ptr, ptr %field_gep1367, align 8
  call void @tl_tensor_free(ptr %field_load1368)
  %field_gep1369 = getelementptr inbounds %Linear, ptr %field_load1366, i32 0, i32 1
  %field_load1370 = load ptr, ptr %field_gep1369, align 8
  call void @tl_tensor_free(ptr %field_load1370)
  %field_gep1371 = getelementptr inbounds %MLP, ptr %field_load1364, i32 0, i32 1
  %field_load1372 = load ptr, ptr %field_gep1371, align 8
  %field_gep1373 = getelementptr inbounds %Linear, ptr %field_load1372, i32 0, i32 0
  %field_load1374 = load ptr, ptr %field_gep1373, align 8
  call void @tl_tensor_free(ptr %field_load1374)
  %field_gep1375 = getelementptr inbounds %Linear, ptr %field_load1372, i32 0, i32 1
  %field_load1376 = load ptr, ptr %field_gep1375, align 8
  call void @tl_tensor_free(ptr %field_load1376)
  %field_gep1377 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 3
  %field_load1378 = load ptr, ptr %field_gep1377, align 8
  %field_gep1379 = getelementptr inbounds %Block, ptr %field_load1378, i32 0, i32 0
  %field_load1380 = load ptr, ptr %field_gep1379, align 8
  %field_gep1381 = getelementptr inbounds %LayerNorm, ptr %field_load1380, i32 0, i32 0
  %field_load1382 = load ptr, ptr %field_gep1381, align 8
  call void @tl_tensor_free(ptr %field_load1382)
  %field_gep1383 = getelementptr inbounds %LayerNorm, ptr %field_load1380, i32 0, i32 1
  %field_load1384 = load ptr, ptr %field_gep1383, align 8
  call void @tl_tensor_free(ptr %field_load1384)
  %field_gep1385 = getelementptr inbounds %Block, ptr %field_load1378, i32 0, i32 1
  %field_load1386 = load ptr, ptr %field_gep1385, align 8
  %field_gep1387 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1386, i32 0, i32 0
  %field_load1388 = load ptr, ptr %field_gep1387, align 8
  %field_gep1389 = getelementptr inbounds %Linear, ptr %field_load1388, i32 0, i32 0
  %field_load1390 = load ptr, ptr %field_gep1389, align 8
  call void @tl_tensor_free(ptr %field_load1390)
  %field_gep1391 = getelementptr inbounds %Linear, ptr %field_load1388, i32 0, i32 1
  %field_load1392 = load ptr, ptr %field_gep1391, align 8
  call void @tl_tensor_free(ptr %field_load1392)
  %field_gep1393 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1386, i32 0, i32 1
  %field_load1394 = load ptr, ptr %field_gep1393, align 8
  %field_gep1395 = getelementptr inbounds %Linear, ptr %field_load1394, i32 0, i32 0
  %field_load1396 = load ptr, ptr %field_gep1395, align 8
  call void @tl_tensor_free(ptr %field_load1396)
  %field_gep1397 = getelementptr inbounds %Linear, ptr %field_load1394, i32 0, i32 1
  %field_load1398 = load ptr, ptr %field_gep1397, align 8
  call void @tl_tensor_free(ptr %field_load1398)
  %field_gep1399 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1386, i32 0, i32 2
  %field_load1400 = load ptr, ptr %field_gep1399, align 8
  %field_gep1401 = getelementptr inbounds %Linear, ptr %field_load1400, i32 0, i32 0
  %field_load1402 = load ptr, ptr %field_gep1401, align 8
  call void @tl_tensor_free(ptr %field_load1402)
  %field_gep1403 = getelementptr inbounds %Linear, ptr %field_load1400, i32 0, i32 1
  %field_load1404 = load ptr, ptr %field_gep1403, align 8
  call void @tl_tensor_free(ptr %field_load1404)
  %field_gep1405 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1386, i32 0, i32 3
  %field_load1406 = load ptr, ptr %field_gep1405, align 8
  %field_gep1407 = getelementptr inbounds %Linear, ptr %field_load1406, i32 0, i32 0
  %field_load1408 = load ptr, ptr %field_gep1407, align 8
  call void @tl_tensor_free(ptr %field_load1408)
  %field_gep1409 = getelementptr inbounds %Linear, ptr %field_load1406, i32 0, i32 1
  %field_load1410 = load ptr, ptr %field_gep1409, align 8
  call void @tl_tensor_free(ptr %field_load1410)
  %field_gep1411 = getelementptr inbounds %Block, ptr %field_load1378, i32 0, i32 2
  %field_load1412 = load ptr, ptr %field_gep1411, align 8
  %field_gep1413 = getelementptr inbounds %LayerNorm, ptr %field_load1412, i32 0, i32 0
  %field_load1414 = load ptr, ptr %field_gep1413, align 8
  call void @tl_tensor_free(ptr %field_load1414)
  %field_gep1415 = getelementptr inbounds %LayerNorm, ptr %field_load1412, i32 0, i32 1
  %field_load1416 = load ptr, ptr %field_gep1415, align 8
  call void @tl_tensor_free(ptr %field_load1416)
  %field_gep1417 = getelementptr inbounds %Block, ptr %field_load1378, i32 0, i32 3
  %field_load1418 = load ptr, ptr %field_gep1417, align 8
  %field_gep1419 = getelementptr inbounds %MLP, ptr %field_load1418, i32 0, i32 0
  %field_load1420 = load ptr, ptr %field_gep1419, align 8
  %field_gep1421 = getelementptr inbounds %Linear, ptr %field_load1420, i32 0, i32 0
  %field_load1422 = load ptr, ptr %field_gep1421, align 8
  call void @tl_tensor_free(ptr %field_load1422)
  %field_gep1423 = getelementptr inbounds %Linear, ptr %field_load1420, i32 0, i32 1
  %field_load1424 = load ptr, ptr %field_gep1423, align 8
  call void @tl_tensor_free(ptr %field_load1424)
  %field_gep1425 = getelementptr inbounds %MLP, ptr %field_load1418, i32 0, i32 1
  %field_load1426 = load ptr, ptr %field_gep1425, align 8
  %field_gep1427 = getelementptr inbounds %Linear, ptr %field_load1426, i32 0, i32 0
  %field_load1428 = load ptr, ptr %field_gep1427, align 8
  call void @tl_tensor_free(ptr %field_load1428)
  %field_gep1429 = getelementptr inbounds %Linear, ptr %field_load1426, i32 0, i32 1
  %field_load1430 = load ptr, ptr %field_gep1429, align 8
  call void @tl_tensor_free(ptr %field_load1430)
  %field_gep1431 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 4
  %field_load1432 = load ptr, ptr %field_gep1431, align 8
  %field_gep1433 = getelementptr inbounds %LayerNorm, ptr %field_load1432, i32 0, i32 0
  %field_load1434 = load ptr, ptr %field_gep1433, align 8
  call void @tl_tensor_free(ptr %field_load1434)
  %field_gep1435 = getelementptr inbounds %LayerNorm, ptr %field_load1432, i32 0, i32 1
  %field_load1436 = load ptr, ptr %field_gep1435, align 8
  call void @tl_tensor_free(ptr %field_load1436)
  %field_gep1437 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 5
  %field_load1438 = load ptr, ptr %field_gep1437, align 8
  %field_gep1439 = getelementptr inbounds %Linear, ptr %field_load1438, i32 0, i32 0
  %field_load1440 = load ptr, ptr %field_gep1439, align 8
  call void @tl_tensor_free(ptr %field_load1440)
  %field_gep1441 = getelementptr inbounds %Linear, ptr %field_load1438, i32 0, i32 1
  %field_load1442 = load ptr, ptr %field_gep1441, align 8
  call void @tl_tensor_free(ptr %field_load1442)
  call void @tl_mem_unregister(ptr %struct_to_free)
  call void @tl_mem_exit_scope()
  ret void

free_struct:                                      ; preds = %for_body
  %field_gep = getelementptr inbounds %GPT, ptr %old_struct_to_free, i32 0, i32 0
  %field_load = load ptr, ptr %field_gep, align 8
  %field_gep2 = getelementptr inbounds %Embedding, ptr %field_load, i32 0, i32 0
  %field_load3 = load ptr, ptr %field_gep2, align 8
  call void @tl_tensor_free(ptr %field_load3)
  %field_gep4 = getelementptr inbounds %GPT, ptr %old_struct_to_free, i32 0, i32 1
  %field_load5 = load ptr, ptr %field_gep4, align 8
  %field_gep6 = getelementptr inbounds %Embedding, ptr %field_load5, i32 0, i32 0
  %field_load7 = load ptr, ptr %field_gep6, align 8
  call void @tl_tensor_free(ptr %field_load7)
  %field_gep8 = getelementptr inbounds %GPT, ptr %old_struct_to_free, i32 0, i32 2
  %field_load9 = load ptr, ptr %field_gep8, align 8
  %field_gep10 = getelementptr inbounds %Block, ptr %field_load9, i32 0, i32 0
  %field_load11 = load ptr, ptr %field_gep10, align 8
  %field_gep12 = getelementptr inbounds %LayerNorm, ptr %field_load11, i32 0, i32 0
  %field_load13 = load ptr, ptr %field_gep12, align 8
  call void @tl_tensor_free(ptr %field_load13)
  %field_gep14 = getelementptr inbounds %LayerNorm, ptr %field_load11, i32 0, i32 1
  %field_load15 = load ptr, ptr %field_gep14, align 8
  call void @tl_tensor_free(ptr %field_load15)
  %field_gep16 = getelementptr inbounds %Block, ptr %field_load9, i32 0, i32 1
  %field_load17 = load ptr, ptr %field_gep16, align 8
  %field_gep18 = getelementptr inbounds %CausalSelfAttention, ptr %field_load17, i32 0, i32 0
  %field_load19 = load ptr, ptr %field_gep18, align 8
  %field_gep20 = getelementptr inbounds %Linear, ptr %field_load19, i32 0, i32 0
  %field_load21 = load ptr, ptr %field_gep20, align 8
  call void @tl_tensor_free(ptr %field_load21)
  %field_gep22 = getelementptr inbounds %Linear, ptr %field_load19, i32 0, i32 1
  %field_load23 = load ptr, ptr %field_gep22, align 8
  call void @tl_tensor_free(ptr %field_load23)
  %field_gep24 = getelementptr inbounds %CausalSelfAttention, ptr %field_load17, i32 0, i32 1
  %field_load25 = load ptr, ptr %field_gep24, align 8
  %field_gep26 = getelementptr inbounds %Linear, ptr %field_load25, i32 0, i32 0
  %field_load27 = load ptr, ptr %field_gep26, align 8
  call void @tl_tensor_free(ptr %field_load27)
  %field_gep28 = getelementptr inbounds %Linear, ptr %field_load25, i32 0, i32 1
  %field_load29 = load ptr, ptr %field_gep28, align 8
  call void @tl_tensor_free(ptr %field_load29)
  %field_gep30 = getelementptr inbounds %CausalSelfAttention, ptr %field_load17, i32 0, i32 2
  %field_load31 = load ptr, ptr %field_gep30, align 8
  %field_gep32 = getelementptr inbounds %Linear, ptr %field_load31, i32 0, i32 0
  %field_load33 = load ptr, ptr %field_gep32, align 8
  call void @tl_tensor_free(ptr %field_load33)
  %field_gep34 = getelementptr inbounds %Linear, ptr %field_load31, i32 0, i32 1
  %field_load35 = load ptr, ptr %field_gep34, align 8
  call void @tl_tensor_free(ptr %field_load35)
  %field_gep36 = getelementptr inbounds %CausalSelfAttention, ptr %field_load17, i32 0, i32 3
  %field_load37 = load ptr, ptr %field_gep36, align 8
  %field_gep38 = getelementptr inbounds %Linear, ptr %field_load37, i32 0, i32 0
  %field_load39 = load ptr, ptr %field_gep38, align 8
  call void @tl_tensor_free(ptr %field_load39)
  %field_gep40 = getelementptr inbounds %Linear, ptr %field_load37, i32 0, i32 1
  %field_load41 = load ptr, ptr %field_gep40, align 8
  call void @tl_tensor_free(ptr %field_load41)
  %field_gep42 = getelementptr inbounds %Block, ptr %field_load9, i32 0, i32 2
  %field_load43 = load ptr, ptr %field_gep42, align 8
  %field_gep44 = getelementptr inbounds %LayerNorm, ptr %field_load43, i32 0, i32 0
  %field_load45 = load ptr, ptr %field_gep44, align 8
  call void @tl_tensor_free(ptr %field_load45)
  %field_gep46 = getelementptr inbounds %LayerNorm, ptr %field_load43, i32 0, i32 1
  %field_load47 = load ptr, ptr %field_gep46, align 8
  call void @tl_tensor_free(ptr %field_load47)
  %field_gep48 = getelementptr inbounds %Block, ptr %field_load9, i32 0, i32 3
  %field_load49 = load ptr, ptr %field_gep48, align 8
  %field_gep50 = getelementptr inbounds %MLP, ptr %field_load49, i32 0, i32 0
  %field_load51 = load ptr, ptr %field_gep50, align 8
  %field_gep52 = getelementptr inbounds %Linear, ptr %field_load51, i32 0, i32 0
  %field_load53 = load ptr, ptr %field_gep52, align 8
  call void @tl_tensor_free(ptr %field_load53)
  %field_gep54 = getelementptr inbounds %Linear, ptr %field_load51, i32 0, i32 1
  %field_load55 = load ptr, ptr %field_gep54, align 8
  call void @tl_tensor_free(ptr %field_load55)
  %field_gep56 = getelementptr inbounds %MLP, ptr %field_load49, i32 0, i32 1
  %field_load57 = load ptr, ptr %field_gep56, align 8
  %field_gep58 = getelementptr inbounds %Linear, ptr %field_load57, i32 0, i32 0
  %field_load59 = load ptr, ptr %field_gep58, align 8
  call void @tl_tensor_free(ptr %field_load59)
  %field_gep60 = getelementptr inbounds %Linear, ptr %field_load57, i32 0, i32 1
  %field_load61 = load ptr, ptr %field_gep60, align 8
  call void @tl_tensor_free(ptr %field_load61)
  %field_gep62 = getelementptr inbounds %GPT, ptr %old_struct_to_free, i32 0, i32 3
  %field_load63 = load ptr, ptr %field_gep62, align 8
  %field_gep64 = getelementptr inbounds %Block, ptr %field_load63, i32 0, i32 0
  %field_load65 = load ptr, ptr %field_gep64, align 8
  %field_gep66 = getelementptr inbounds %LayerNorm, ptr %field_load65, i32 0, i32 0
  %field_load67 = load ptr, ptr %field_gep66, align 8
  call void @tl_tensor_free(ptr %field_load67)
  %field_gep68 = getelementptr inbounds %LayerNorm, ptr %field_load65, i32 0, i32 1
  %field_load69 = load ptr, ptr %field_gep68, align 8
  call void @tl_tensor_free(ptr %field_load69)
  %field_gep70 = getelementptr inbounds %Block, ptr %field_load63, i32 0, i32 1
  %field_load71 = load ptr, ptr %field_gep70, align 8
  %field_gep72 = getelementptr inbounds %CausalSelfAttention, ptr %field_load71, i32 0, i32 0
  %field_load73 = load ptr, ptr %field_gep72, align 8
  %field_gep74 = getelementptr inbounds %Linear, ptr %field_load73, i32 0, i32 0
  %field_load75 = load ptr, ptr %field_gep74, align 8
  call void @tl_tensor_free(ptr %field_load75)
  %field_gep76 = getelementptr inbounds %Linear, ptr %field_load73, i32 0, i32 1
  %field_load77 = load ptr, ptr %field_gep76, align 8
  call void @tl_tensor_free(ptr %field_load77)
  %field_gep78 = getelementptr inbounds %CausalSelfAttention, ptr %field_load71, i32 0, i32 1
  %field_load79 = load ptr, ptr %field_gep78, align 8
  %field_gep80 = getelementptr inbounds %Linear, ptr %field_load79, i32 0, i32 0
  %field_load81 = load ptr, ptr %field_gep80, align 8
  call void @tl_tensor_free(ptr %field_load81)
  %field_gep82 = getelementptr inbounds %Linear, ptr %field_load79, i32 0, i32 1
  %field_load83 = load ptr, ptr %field_gep82, align 8
  call void @tl_tensor_free(ptr %field_load83)
  %field_gep84 = getelementptr inbounds %CausalSelfAttention, ptr %field_load71, i32 0, i32 2
  %field_load85 = load ptr, ptr %field_gep84, align 8
  %field_gep86 = getelementptr inbounds %Linear, ptr %field_load85, i32 0, i32 0
  %field_load87 = load ptr, ptr %field_gep86, align 8
  call void @tl_tensor_free(ptr %field_load87)
  %field_gep88 = getelementptr inbounds %Linear, ptr %field_load85, i32 0, i32 1
  %field_load89 = load ptr, ptr %field_gep88, align 8
  call void @tl_tensor_free(ptr %field_load89)
  %field_gep90 = getelementptr inbounds %CausalSelfAttention, ptr %field_load71, i32 0, i32 3
  %field_load91 = load ptr, ptr %field_gep90, align 8
  %field_gep92 = getelementptr inbounds %Linear, ptr %field_load91, i32 0, i32 0
  %field_load93 = load ptr, ptr %field_gep92, align 8
  call void @tl_tensor_free(ptr %field_load93)
  %field_gep94 = getelementptr inbounds %Linear, ptr %field_load91, i32 0, i32 1
  %field_load95 = load ptr, ptr %field_gep94, align 8
  call void @tl_tensor_free(ptr %field_load95)
  %field_gep96 = getelementptr inbounds %Block, ptr %field_load63, i32 0, i32 2
  %field_load97 = load ptr, ptr %field_gep96, align 8
  %field_gep98 = getelementptr inbounds %LayerNorm, ptr %field_load97, i32 0, i32 0
  %field_load99 = load ptr, ptr %field_gep98, align 8
  call void @tl_tensor_free(ptr %field_load99)
  %field_gep100 = getelementptr inbounds %LayerNorm, ptr %field_load97, i32 0, i32 1
  %field_load101 = load ptr, ptr %field_gep100, align 8
  call void @tl_tensor_free(ptr %field_load101)
  %field_gep102 = getelementptr inbounds %Block, ptr %field_load63, i32 0, i32 3
  %field_load103 = load ptr, ptr %field_gep102, align 8
  %field_gep104 = getelementptr inbounds %MLP, ptr %field_load103, i32 0, i32 0
  %field_load105 = load ptr, ptr %field_gep104, align 8
  %field_gep106 = getelementptr inbounds %Linear, ptr %field_load105, i32 0, i32 0
  %field_load107 = load ptr, ptr %field_gep106, align 8
  call void @tl_tensor_free(ptr %field_load107)
  %field_gep108 = getelementptr inbounds %Linear, ptr %field_load105, i32 0, i32 1
  %field_load109 = load ptr, ptr %field_gep108, align 8
  call void @tl_tensor_free(ptr %field_load109)
  %field_gep110 = getelementptr inbounds %MLP, ptr %field_load103, i32 0, i32 1
  %field_load111 = load ptr, ptr %field_gep110, align 8
  %field_gep112 = getelementptr inbounds %Linear, ptr %field_load111, i32 0, i32 0
  %field_load113 = load ptr, ptr %field_gep112, align 8
  call void @tl_tensor_free(ptr %field_load113)
  %field_gep114 = getelementptr inbounds %Linear, ptr %field_load111, i32 0, i32 1
  %field_load115 = load ptr, ptr %field_gep114, align 8
  call void @tl_tensor_free(ptr %field_load115)
  %field_gep116 = getelementptr inbounds %GPT, ptr %old_struct_to_free, i32 0, i32 4
  %field_load117 = load ptr, ptr %field_gep116, align 8
  %field_gep118 = getelementptr inbounds %LayerNorm, ptr %field_load117, i32 0, i32 0
  %field_load119 = load ptr, ptr %field_gep118, align 8
  call void @tl_tensor_free(ptr %field_load119)
  %field_gep120 = getelementptr inbounds %LayerNorm, ptr %field_load117, i32 0, i32 1
  %field_load121 = load ptr, ptr %field_gep120, align 8
  call void @tl_tensor_free(ptr %field_load121)
  %field_gep122 = getelementptr inbounds %GPT, ptr %old_struct_to_free, i32 0, i32 5
  %field_load123 = load ptr, ptr %field_gep122, align 8
  %field_gep124 = getelementptr inbounds %Linear, ptr %field_load123, i32 0, i32 0
  %field_load125 = load ptr, ptr %field_gep124, align 8
  call void @tl_tensor_free(ptr %field_load125)
  %field_gep126 = getelementptr inbounds %Linear, ptr %field_load123, i32 0, i32 1
  %field_load127 = load ptr, ptr %field_gep126, align 8
  call void @tl_tensor_free(ptr %field_load127)
  call void @tl_mem_unregister(ptr %old_struct_to_free)
  br label %continue_after_free

continue_after_free:                              ; preds = %free_struct, %for_body
  call void @tl_mem_unregister(ptr %call_tmp)
  %unreg_field_0 = getelementptr inbounds %GPT, ptr %call_tmp, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_0128 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val129 = load ptr, ptr %unreg_field_0128, align 8
  call void @tl_mem_unregister(ptr %field_val129)
  %unreg_field_1 = getelementptr inbounds %GPT, ptr %call_tmp, i32 0, i32 1
  %field_val130 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val130)
  %unreg_field_0131 = getelementptr inbounds %Embedding, ptr %field_val130, i32 0, i32 0
  %field_val132 = load ptr, ptr %unreg_field_0131, align 8
  call void @tl_mem_unregister(ptr %field_val132)
  %unreg_field_2 = getelementptr inbounds %GPT, ptr %call_tmp, i32 0, i32 2
  %field_val133 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val133)
  %unreg_field_0134 = getelementptr inbounds %Block, ptr %field_val133, i32 0, i32 0
  %field_val135 = load ptr, ptr %unreg_field_0134, align 8
  call void @tl_mem_unregister(ptr %field_val135)
  %unreg_field_0136 = getelementptr inbounds %LayerNorm, ptr %field_val135, i32 0, i32 0
  %field_val137 = load ptr, ptr %unreg_field_0136, align 8
  call void @tl_mem_unregister(ptr %field_val137)
  %unreg_field_1138 = getelementptr inbounds %LayerNorm, ptr %field_val135, i32 0, i32 1
  %field_val139 = load ptr, ptr %unreg_field_1138, align 8
  call void @tl_mem_unregister(ptr %field_val139)
  %unreg_field_1140 = getelementptr inbounds %Block, ptr %field_val133, i32 0, i32 1
  %field_val141 = load ptr, ptr %unreg_field_1140, align 8
  call void @tl_mem_unregister(ptr %field_val141)
  %unreg_field_0142 = getelementptr inbounds %CausalSelfAttention, ptr %field_val141, i32 0, i32 0
  %field_val143 = load ptr, ptr %unreg_field_0142, align 8
  call void @tl_mem_unregister(ptr %field_val143)
  %unreg_field_0144 = getelementptr inbounds %Linear, ptr %field_val143, i32 0, i32 0
  %field_val145 = load ptr, ptr %unreg_field_0144, align 8
  call void @tl_mem_unregister(ptr %field_val145)
  %unreg_field_1146 = getelementptr inbounds %Linear, ptr %field_val143, i32 0, i32 1
  %field_val147 = load ptr, ptr %unreg_field_1146, align 8
  call void @tl_mem_unregister(ptr %field_val147)
  %unreg_field_1148 = getelementptr inbounds %CausalSelfAttention, ptr %field_val141, i32 0, i32 1
  %field_val149 = load ptr, ptr %unreg_field_1148, align 8
  call void @tl_mem_unregister(ptr %field_val149)
  %unreg_field_0150 = getelementptr inbounds %Linear, ptr %field_val149, i32 0, i32 0
  %field_val151 = load ptr, ptr %unreg_field_0150, align 8
  call void @tl_mem_unregister(ptr %field_val151)
  %unreg_field_1152 = getelementptr inbounds %Linear, ptr %field_val149, i32 0, i32 1
  %field_val153 = load ptr, ptr %unreg_field_1152, align 8
  call void @tl_mem_unregister(ptr %field_val153)
  %unreg_field_2154 = getelementptr inbounds %CausalSelfAttention, ptr %field_val141, i32 0, i32 2
  %field_val155 = load ptr, ptr %unreg_field_2154, align 8
  call void @tl_mem_unregister(ptr %field_val155)
  %unreg_field_0156 = getelementptr inbounds %Linear, ptr %field_val155, i32 0, i32 0
  %field_val157 = load ptr, ptr %unreg_field_0156, align 8
  call void @tl_mem_unregister(ptr %field_val157)
  %unreg_field_1158 = getelementptr inbounds %Linear, ptr %field_val155, i32 0, i32 1
  %field_val159 = load ptr, ptr %unreg_field_1158, align 8
  call void @tl_mem_unregister(ptr %field_val159)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val141, i32 0, i32 3
  %field_val160 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val160)
  %unreg_field_0161 = getelementptr inbounds %Linear, ptr %field_val160, i32 0, i32 0
  %field_val162 = load ptr, ptr %unreg_field_0161, align 8
  call void @tl_mem_unregister(ptr %field_val162)
  %unreg_field_1163 = getelementptr inbounds %Linear, ptr %field_val160, i32 0, i32 1
  %field_val164 = load ptr, ptr %unreg_field_1163, align 8
  call void @tl_mem_unregister(ptr %field_val164)
  %unreg_field_2165 = getelementptr inbounds %Block, ptr %field_val133, i32 0, i32 2
  %field_val166 = load ptr, ptr %unreg_field_2165, align 8
  call void @tl_mem_unregister(ptr %field_val166)
  %unreg_field_0167 = getelementptr inbounds %LayerNorm, ptr %field_val166, i32 0, i32 0
  %field_val168 = load ptr, ptr %unreg_field_0167, align 8
  call void @tl_mem_unregister(ptr %field_val168)
  %unreg_field_1169 = getelementptr inbounds %LayerNorm, ptr %field_val166, i32 0, i32 1
  %field_val170 = load ptr, ptr %unreg_field_1169, align 8
  call void @tl_mem_unregister(ptr %field_val170)
  %unreg_field_3171 = getelementptr inbounds %Block, ptr %field_val133, i32 0, i32 3
  %field_val172 = load ptr, ptr %unreg_field_3171, align 8
  call void @tl_mem_unregister(ptr %field_val172)
  %unreg_field_0173 = getelementptr inbounds %MLP, ptr %field_val172, i32 0, i32 0
  %field_val174 = load ptr, ptr %unreg_field_0173, align 8
  call void @tl_mem_unregister(ptr %field_val174)
  %unreg_field_0175 = getelementptr inbounds %Linear, ptr %field_val174, i32 0, i32 0
  %field_val176 = load ptr, ptr %unreg_field_0175, align 8
  call void @tl_mem_unregister(ptr %field_val176)
  %unreg_field_1177 = getelementptr inbounds %Linear, ptr %field_val174, i32 0, i32 1
  %field_val178 = load ptr, ptr %unreg_field_1177, align 8
  call void @tl_mem_unregister(ptr %field_val178)
  %unreg_field_1179 = getelementptr inbounds %MLP, ptr %field_val172, i32 0, i32 1
  %field_val180 = load ptr, ptr %unreg_field_1179, align 8
  call void @tl_mem_unregister(ptr %field_val180)
  %unreg_field_0181 = getelementptr inbounds %Linear, ptr %field_val180, i32 0, i32 0
  %field_val182 = load ptr, ptr %unreg_field_0181, align 8
  call void @tl_mem_unregister(ptr %field_val182)
  %unreg_field_1183 = getelementptr inbounds %Linear, ptr %field_val180, i32 0, i32 1
  %field_val184 = load ptr, ptr %unreg_field_1183, align 8
  call void @tl_mem_unregister(ptr %field_val184)
  %unreg_field_3185 = getelementptr inbounds %GPT, ptr %call_tmp, i32 0, i32 3
  %field_val186 = load ptr, ptr %unreg_field_3185, align 8
  call void @tl_mem_unregister(ptr %field_val186)
  %unreg_field_0187 = getelementptr inbounds %Block, ptr %field_val186, i32 0, i32 0
  %field_val188 = load ptr, ptr %unreg_field_0187, align 8
  call void @tl_mem_unregister(ptr %field_val188)
  %unreg_field_0189 = getelementptr inbounds %LayerNorm, ptr %field_val188, i32 0, i32 0
  %field_val190 = load ptr, ptr %unreg_field_0189, align 8
  call void @tl_mem_unregister(ptr %field_val190)
  %unreg_field_1191 = getelementptr inbounds %LayerNorm, ptr %field_val188, i32 0, i32 1
  %field_val192 = load ptr, ptr %unreg_field_1191, align 8
  call void @tl_mem_unregister(ptr %field_val192)
  %unreg_field_1193 = getelementptr inbounds %Block, ptr %field_val186, i32 0, i32 1
  %field_val194 = load ptr, ptr %unreg_field_1193, align 8
  call void @tl_mem_unregister(ptr %field_val194)
  %unreg_field_0195 = getelementptr inbounds %CausalSelfAttention, ptr %field_val194, i32 0, i32 0
  %field_val196 = load ptr, ptr %unreg_field_0195, align 8
  call void @tl_mem_unregister(ptr %field_val196)
  %unreg_field_0197 = getelementptr inbounds %Linear, ptr %field_val196, i32 0, i32 0
  %field_val198 = load ptr, ptr %unreg_field_0197, align 8
  call void @tl_mem_unregister(ptr %field_val198)
  %unreg_field_1199 = getelementptr inbounds %Linear, ptr %field_val196, i32 0, i32 1
  %field_val200 = load ptr, ptr %unreg_field_1199, align 8
  call void @tl_mem_unregister(ptr %field_val200)
  %unreg_field_1201 = getelementptr inbounds %CausalSelfAttention, ptr %field_val194, i32 0, i32 1
  %field_val202 = load ptr, ptr %unreg_field_1201, align 8
  call void @tl_mem_unregister(ptr %field_val202)
  %unreg_field_0203 = getelementptr inbounds %Linear, ptr %field_val202, i32 0, i32 0
  %field_val204 = load ptr, ptr %unreg_field_0203, align 8
  call void @tl_mem_unregister(ptr %field_val204)
  %unreg_field_1205 = getelementptr inbounds %Linear, ptr %field_val202, i32 0, i32 1
  %field_val206 = load ptr, ptr %unreg_field_1205, align 8
  call void @tl_mem_unregister(ptr %field_val206)
  %unreg_field_2207 = getelementptr inbounds %CausalSelfAttention, ptr %field_val194, i32 0, i32 2
  %field_val208 = load ptr, ptr %unreg_field_2207, align 8
  call void @tl_mem_unregister(ptr %field_val208)
  %unreg_field_0209 = getelementptr inbounds %Linear, ptr %field_val208, i32 0, i32 0
  %field_val210 = load ptr, ptr %unreg_field_0209, align 8
  call void @tl_mem_unregister(ptr %field_val210)
  %unreg_field_1211 = getelementptr inbounds %Linear, ptr %field_val208, i32 0, i32 1
  %field_val212 = load ptr, ptr %unreg_field_1211, align 8
  call void @tl_mem_unregister(ptr %field_val212)
  %unreg_field_3213 = getelementptr inbounds %CausalSelfAttention, ptr %field_val194, i32 0, i32 3
  %field_val214 = load ptr, ptr %unreg_field_3213, align 8
  call void @tl_mem_unregister(ptr %field_val214)
  %unreg_field_0215 = getelementptr inbounds %Linear, ptr %field_val214, i32 0, i32 0
  %field_val216 = load ptr, ptr %unreg_field_0215, align 8
  call void @tl_mem_unregister(ptr %field_val216)
  %unreg_field_1217 = getelementptr inbounds %Linear, ptr %field_val214, i32 0, i32 1
  %field_val218 = load ptr, ptr %unreg_field_1217, align 8
  call void @tl_mem_unregister(ptr %field_val218)
  %unreg_field_2219 = getelementptr inbounds %Block, ptr %field_val186, i32 0, i32 2
  %field_val220 = load ptr, ptr %unreg_field_2219, align 8
  call void @tl_mem_unregister(ptr %field_val220)
  %unreg_field_0221 = getelementptr inbounds %LayerNorm, ptr %field_val220, i32 0, i32 0
  %field_val222 = load ptr, ptr %unreg_field_0221, align 8
  call void @tl_mem_unregister(ptr %field_val222)
  %unreg_field_1223 = getelementptr inbounds %LayerNorm, ptr %field_val220, i32 0, i32 1
  %field_val224 = load ptr, ptr %unreg_field_1223, align 8
  call void @tl_mem_unregister(ptr %field_val224)
  %unreg_field_3225 = getelementptr inbounds %Block, ptr %field_val186, i32 0, i32 3
  %field_val226 = load ptr, ptr %unreg_field_3225, align 8
  call void @tl_mem_unregister(ptr %field_val226)
  %unreg_field_0227 = getelementptr inbounds %MLP, ptr %field_val226, i32 0, i32 0
  %field_val228 = load ptr, ptr %unreg_field_0227, align 8
  call void @tl_mem_unregister(ptr %field_val228)
  %unreg_field_0229 = getelementptr inbounds %Linear, ptr %field_val228, i32 0, i32 0
  %field_val230 = load ptr, ptr %unreg_field_0229, align 8
  call void @tl_mem_unregister(ptr %field_val230)
  %unreg_field_1231 = getelementptr inbounds %Linear, ptr %field_val228, i32 0, i32 1
  %field_val232 = load ptr, ptr %unreg_field_1231, align 8
  call void @tl_mem_unregister(ptr %field_val232)
  %unreg_field_1233 = getelementptr inbounds %MLP, ptr %field_val226, i32 0, i32 1
  %field_val234 = load ptr, ptr %unreg_field_1233, align 8
  call void @tl_mem_unregister(ptr %field_val234)
  %unreg_field_0235 = getelementptr inbounds %Linear, ptr %field_val234, i32 0, i32 0
  %field_val236 = load ptr, ptr %unreg_field_0235, align 8
  call void @tl_mem_unregister(ptr %field_val236)
  %unreg_field_1237 = getelementptr inbounds %Linear, ptr %field_val234, i32 0, i32 1
  %field_val238 = load ptr, ptr %unreg_field_1237, align 8
  call void @tl_mem_unregister(ptr %field_val238)
  %unreg_field_4 = getelementptr inbounds %GPT, ptr %call_tmp, i32 0, i32 4
  %field_val239 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val239)
  %unreg_field_0240 = getelementptr inbounds %LayerNorm, ptr %field_val239, i32 0, i32 0
  %field_val241 = load ptr, ptr %unreg_field_0240, align 8
  call void @tl_mem_unregister(ptr %field_val241)
  %unreg_field_1242 = getelementptr inbounds %LayerNorm, ptr %field_val239, i32 0, i32 1
  %field_val243 = load ptr, ptr %unreg_field_1242, align 8
  call void @tl_mem_unregister(ptr %field_val243)
  %unreg_field_5 = getelementptr inbounds %GPT, ptr %call_tmp, i32 0, i32 5
  %field_val244 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val244)
  %unreg_field_0245 = getelementptr inbounds %Linear, ptr %field_val244, i32 0, i32 0
  %field_val246 = load ptr, ptr %unreg_field_0245, align 8
  call void @tl_mem_unregister(ptr %field_val246)
  %unreg_field_1247 = getelementptr inbounds %Linear, ptr %field_val244, i32 0, i32 1
  %field_val248 = load ptr, ptr %unreg_field_1247, align 8
  call void @tl_mem_unregister(ptr %field_val248)
  call void @tl_mem_unregister(ptr %call_tmp)
  store ptr %call_tmp, ptr %model, align 8
  %model249 = load ptr, ptr %model, align 8
  %call_tmp250 = call ptr @train_step(ptr %model249, float 0x3FA99999A0000000, i64 50, i64 50)
  call void @tl_mem_unregister(ptr %call_tmp250)
  %old_struct_to_free251 = load ptr, ptr %model, align 8
  %is_not_null252 = icmp ne ptr %old_struct_to_free251, null
  %are_diff253 = icmp ne ptr %old_struct_to_free251, %call_tmp250
  %can_free_1254 = and i1 %is_not_null252, true
  %can_free255 = and i1 %can_free_1254, %are_diff253
  br i1 %can_free255, label %free_struct256, label %continue_after_free257

free_struct256:                                   ; preds = %continue_after_free
  %field_gep258 = getelementptr inbounds %GPT, ptr %old_struct_to_free251, i32 0, i32 0
  %field_load259 = load ptr, ptr %field_gep258, align 8
  %field_gep260 = getelementptr inbounds %Embedding, ptr %field_load259, i32 0, i32 0
  %field_load261 = load ptr, ptr %field_gep260, align 8
  call void @tl_tensor_free(ptr %field_load261)
  %field_gep262 = getelementptr inbounds %GPT, ptr %old_struct_to_free251, i32 0, i32 1
  %field_load263 = load ptr, ptr %field_gep262, align 8
  %field_gep264 = getelementptr inbounds %Embedding, ptr %field_load263, i32 0, i32 0
  %field_load265 = load ptr, ptr %field_gep264, align 8
  call void @tl_tensor_free(ptr %field_load265)
  %field_gep266 = getelementptr inbounds %GPT, ptr %old_struct_to_free251, i32 0, i32 2
  %field_load267 = load ptr, ptr %field_gep266, align 8
  %field_gep268 = getelementptr inbounds %Block, ptr %field_load267, i32 0, i32 0
  %field_load269 = load ptr, ptr %field_gep268, align 8
  %field_gep270 = getelementptr inbounds %LayerNorm, ptr %field_load269, i32 0, i32 0
  %field_load271 = load ptr, ptr %field_gep270, align 8
  call void @tl_tensor_free(ptr %field_load271)
  %field_gep272 = getelementptr inbounds %LayerNorm, ptr %field_load269, i32 0, i32 1
  %field_load273 = load ptr, ptr %field_gep272, align 8
  call void @tl_tensor_free(ptr %field_load273)
  %field_gep274 = getelementptr inbounds %Block, ptr %field_load267, i32 0, i32 1
  %field_load275 = load ptr, ptr %field_gep274, align 8
  %field_gep276 = getelementptr inbounds %CausalSelfAttention, ptr %field_load275, i32 0, i32 0
  %field_load277 = load ptr, ptr %field_gep276, align 8
  %field_gep278 = getelementptr inbounds %Linear, ptr %field_load277, i32 0, i32 0
  %field_load279 = load ptr, ptr %field_gep278, align 8
  call void @tl_tensor_free(ptr %field_load279)
  %field_gep280 = getelementptr inbounds %Linear, ptr %field_load277, i32 0, i32 1
  %field_load281 = load ptr, ptr %field_gep280, align 8
  call void @tl_tensor_free(ptr %field_load281)
  %field_gep282 = getelementptr inbounds %CausalSelfAttention, ptr %field_load275, i32 0, i32 1
  %field_load283 = load ptr, ptr %field_gep282, align 8
  %field_gep284 = getelementptr inbounds %Linear, ptr %field_load283, i32 0, i32 0
  %field_load285 = load ptr, ptr %field_gep284, align 8
  call void @tl_tensor_free(ptr %field_load285)
  %field_gep286 = getelementptr inbounds %Linear, ptr %field_load283, i32 0, i32 1
  %field_load287 = load ptr, ptr %field_gep286, align 8
  call void @tl_tensor_free(ptr %field_load287)
  %field_gep288 = getelementptr inbounds %CausalSelfAttention, ptr %field_load275, i32 0, i32 2
  %field_load289 = load ptr, ptr %field_gep288, align 8
  %field_gep290 = getelementptr inbounds %Linear, ptr %field_load289, i32 0, i32 0
  %field_load291 = load ptr, ptr %field_gep290, align 8
  call void @tl_tensor_free(ptr %field_load291)
  %field_gep292 = getelementptr inbounds %Linear, ptr %field_load289, i32 0, i32 1
  %field_load293 = load ptr, ptr %field_gep292, align 8
  call void @tl_tensor_free(ptr %field_load293)
  %field_gep294 = getelementptr inbounds %CausalSelfAttention, ptr %field_load275, i32 0, i32 3
  %field_load295 = load ptr, ptr %field_gep294, align 8
  %field_gep296 = getelementptr inbounds %Linear, ptr %field_load295, i32 0, i32 0
  %field_load297 = load ptr, ptr %field_gep296, align 8
  call void @tl_tensor_free(ptr %field_load297)
  %field_gep298 = getelementptr inbounds %Linear, ptr %field_load295, i32 0, i32 1
  %field_load299 = load ptr, ptr %field_gep298, align 8
  call void @tl_tensor_free(ptr %field_load299)
  %field_gep300 = getelementptr inbounds %Block, ptr %field_load267, i32 0, i32 2
  %field_load301 = load ptr, ptr %field_gep300, align 8
  %field_gep302 = getelementptr inbounds %LayerNorm, ptr %field_load301, i32 0, i32 0
  %field_load303 = load ptr, ptr %field_gep302, align 8
  call void @tl_tensor_free(ptr %field_load303)
  %field_gep304 = getelementptr inbounds %LayerNorm, ptr %field_load301, i32 0, i32 1
  %field_load305 = load ptr, ptr %field_gep304, align 8
  call void @tl_tensor_free(ptr %field_load305)
  %field_gep306 = getelementptr inbounds %Block, ptr %field_load267, i32 0, i32 3
  %field_load307 = load ptr, ptr %field_gep306, align 8
  %field_gep308 = getelementptr inbounds %MLP, ptr %field_load307, i32 0, i32 0
  %field_load309 = load ptr, ptr %field_gep308, align 8
  %field_gep310 = getelementptr inbounds %Linear, ptr %field_load309, i32 0, i32 0
  %field_load311 = load ptr, ptr %field_gep310, align 8
  call void @tl_tensor_free(ptr %field_load311)
  %field_gep312 = getelementptr inbounds %Linear, ptr %field_load309, i32 0, i32 1
  %field_load313 = load ptr, ptr %field_gep312, align 8
  call void @tl_tensor_free(ptr %field_load313)
  %field_gep314 = getelementptr inbounds %MLP, ptr %field_load307, i32 0, i32 1
  %field_load315 = load ptr, ptr %field_gep314, align 8
  %field_gep316 = getelementptr inbounds %Linear, ptr %field_load315, i32 0, i32 0
  %field_load317 = load ptr, ptr %field_gep316, align 8
  call void @tl_tensor_free(ptr %field_load317)
  %field_gep318 = getelementptr inbounds %Linear, ptr %field_load315, i32 0, i32 1
  %field_load319 = load ptr, ptr %field_gep318, align 8
  call void @tl_tensor_free(ptr %field_load319)
  %field_gep320 = getelementptr inbounds %GPT, ptr %old_struct_to_free251, i32 0, i32 3
  %field_load321 = load ptr, ptr %field_gep320, align 8
  %field_gep322 = getelementptr inbounds %Block, ptr %field_load321, i32 0, i32 0
  %field_load323 = load ptr, ptr %field_gep322, align 8
  %field_gep324 = getelementptr inbounds %LayerNorm, ptr %field_load323, i32 0, i32 0
  %field_load325 = load ptr, ptr %field_gep324, align 8
  call void @tl_tensor_free(ptr %field_load325)
  %field_gep326 = getelementptr inbounds %LayerNorm, ptr %field_load323, i32 0, i32 1
  %field_load327 = load ptr, ptr %field_gep326, align 8
  call void @tl_tensor_free(ptr %field_load327)
  %field_gep328 = getelementptr inbounds %Block, ptr %field_load321, i32 0, i32 1
  %field_load329 = load ptr, ptr %field_gep328, align 8
  %field_gep330 = getelementptr inbounds %CausalSelfAttention, ptr %field_load329, i32 0, i32 0
  %field_load331 = load ptr, ptr %field_gep330, align 8
  %field_gep332 = getelementptr inbounds %Linear, ptr %field_load331, i32 0, i32 0
  %field_load333 = load ptr, ptr %field_gep332, align 8
  call void @tl_tensor_free(ptr %field_load333)
  %field_gep334 = getelementptr inbounds %Linear, ptr %field_load331, i32 0, i32 1
  %field_load335 = load ptr, ptr %field_gep334, align 8
  call void @tl_tensor_free(ptr %field_load335)
  %field_gep336 = getelementptr inbounds %CausalSelfAttention, ptr %field_load329, i32 0, i32 1
  %field_load337 = load ptr, ptr %field_gep336, align 8
  %field_gep338 = getelementptr inbounds %Linear, ptr %field_load337, i32 0, i32 0
  %field_load339 = load ptr, ptr %field_gep338, align 8
  call void @tl_tensor_free(ptr %field_load339)
  %field_gep340 = getelementptr inbounds %Linear, ptr %field_load337, i32 0, i32 1
  %field_load341 = load ptr, ptr %field_gep340, align 8
  call void @tl_tensor_free(ptr %field_load341)
  %field_gep342 = getelementptr inbounds %CausalSelfAttention, ptr %field_load329, i32 0, i32 2
  %field_load343 = load ptr, ptr %field_gep342, align 8
  %field_gep344 = getelementptr inbounds %Linear, ptr %field_load343, i32 0, i32 0
  %field_load345 = load ptr, ptr %field_gep344, align 8
  call void @tl_tensor_free(ptr %field_load345)
  %field_gep346 = getelementptr inbounds %Linear, ptr %field_load343, i32 0, i32 1
  %field_load347 = load ptr, ptr %field_gep346, align 8
  call void @tl_tensor_free(ptr %field_load347)
  %field_gep348 = getelementptr inbounds %CausalSelfAttention, ptr %field_load329, i32 0, i32 3
  %field_load349 = load ptr, ptr %field_gep348, align 8
  %field_gep350 = getelementptr inbounds %Linear, ptr %field_load349, i32 0, i32 0
  %field_load351 = load ptr, ptr %field_gep350, align 8
  call void @tl_tensor_free(ptr %field_load351)
  %field_gep352 = getelementptr inbounds %Linear, ptr %field_load349, i32 0, i32 1
  %field_load353 = load ptr, ptr %field_gep352, align 8
  call void @tl_tensor_free(ptr %field_load353)
  %field_gep354 = getelementptr inbounds %Block, ptr %field_load321, i32 0, i32 2
  %field_load355 = load ptr, ptr %field_gep354, align 8
  %field_gep356 = getelementptr inbounds %LayerNorm, ptr %field_load355, i32 0, i32 0
  %field_load357 = load ptr, ptr %field_gep356, align 8
  call void @tl_tensor_free(ptr %field_load357)
  %field_gep358 = getelementptr inbounds %LayerNorm, ptr %field_load355, i32 0, i32 1
  %field_load359 = load ptr, ptr %field_gep358, align 8
  call void @tl_tensor_free(ptr %field_load359)
  %field_gep360 = getelementptr inbounds %Block, ptr %field_load321, i32 0, i32 3
  %field_load361 = load ptr, ptr %field_gep360, align 8
  %field_gep362 = getelementptr inbounds %MLP, ptr %field_load361, i32 0, i32 0
  %field_load363 = load ptr, ptr %field_gep362, align 8
  %field_gep364 = getelementptr inbounds %Linear, ptr %field_load363, i32 0, i32 0
  %field_load365 = load ptr, ptr %field_gep364, align 8
  call void @tl_tensor_free(ptr %field_load365)
  %field_gep366 = getelementptr inbounds %Linear, ptr %field_load363, i32 0, i32 1
  %field_load367 = load ptr, ptr %field_gep366, align 8
  call void @tl_tensor_free(ptr %field_load367)
  %field_gep368 = getelementptr inbounds %MLP, ptr %field_load361, i32 0, i32 1
  %field_load369 = load ptr, ptr %field_gep368, align 8
  %field_gep370 = getelementptr inbounds %Linear, ptr %field_load369, i32 0, i32 0
  %field_load371 = load ptr, ptr %field_gep370, align 8
  call void @tl_tensor_free(ptr %field_load371)
  %field_gep372 = getelementptr inbounds %Linear, ptr %field_load369, i32 0, i32 1
  %field_load373 = load ptr, ptr %field_gep372, align 8
  call void @tl_tensor_free(ptr %field_load373)
  %field_gep374 = getelementptr inbounds %GPT, ptr %old_struct_to_free251, i32 0, i32 4
  %field_load375 = load ptr, ptr %field_gep374, align 8
  %field_gep376 = getelementptr inbounds %LayerNorm, ptr %field_load375, i32 0, i32 0
  %field_load377 = load ptr, ptr %field_gep376, align 8
  call void @tl_tensor_free(ptr %field_load377)
  %field_gep378 = getelementptr inbounds %LayerNorm, ptr %field_load375, i32 0, i32 1
  %field_load379 = load ptr, ptr %field_gep378, align 8
  call void @tl_tensor_free(ptr %field_load379)
  %field_gep380 = getelementptr inbounds %GPT, ptr %old_struct_to_free251, i32 0, i32 5
  %field_load381 = load ptr, ptr %field_gep380, align 8
  %field_gep382 = getelementptr inbounds %Linear, ptr %field_load381, i32 0, i32 0
  %field_load383 = load ptr, ptr %field_gep382, align 8
  call void @tl_tensor_free(ptr %field_load383)
  %field_gep384 = getelementptr inbounds %Linear, ptr %field_load381, i32 0, i32 1
  %field_load385 = load ptr, ptr %field_gep384, align 8
  call void @tl_tensor_free(ptr %field_load385)
  call void @tl_mem_unregister(ptr %old_struct_to_free251)
  br label %continue_after_free257

continue_after_free257:                           ; preds = %free_struct256, %continue_after_free
  call void @tl_mem_unregister(ptr %call_tmp250)
  %unreg_field_0386 = getelementptr inbounds %GPT, ptr %call_tmp250, i32 0, i32 0
  %field_val387 = load ptr, ptr %unreg_field_0386, align 8
  call void @tl_mem_unregister(ptr %field_val387)
  %unreg_field_0388 = getelementptr inbounds %Embedding, ptr %field_val387, i32 0, i32 0
  %field_val389 = load ptr, ptr %unreg_field_0388, align 8
  call void @tl_mem_unregister(ptr %field_val389)
  %unreg_field_1390 = getelementptr inbounds %GPT, ptr %call_tmp250, i32 0, i32 1
  %field_val391 = load ptr, ptr %unreg_field_1390, align 8
  call void @tl_mem_unregister(ptr %field_val391)
  %unreg_field_0392 = getelementptr inbounds %Embedding, ptr %field_val391, i32 0, i32 0
  %field_val393 = load ptr, ptr %unreg_field_0392, align 8
  call void @tl_mem_unregister(ptr %field_val393)
  %unreg_field_2394 = getelementptr inbounds %GPT, ptr %call_tmp250, i32 0, i32 2
  %field_val395 = load ptr, ptr %unreg_field_2394, align 8
  call void @tl_mem_unregister(ptr %field_val395)
  %unreg_field_0396 = getelementptr inbounds %Block, ptr %field_val395, i32 0, i32 0
  %field_val397 = load ptr, ptr %unreg_field_0396, align 8
  call void @tl_mem_unregister(ptr %field_val397)
  %unreg_field_0398 = getelementptr inbounds %LayerNorm, ptr %field_val397, i32 0, i32 0
  %field_val399 = load ptr, ptr %unreg_field_0398, align 8
  call void @tl_mem_unregister(ptr %field_val399)
  %unreg_field_1400 = getelementptr inbounds %LayerNorm, ptr %field_val397, i32 0, i32 1
  %field_val401 = load ptr, ptr %unreg_field_1400, align 8
  call void @tl_mem_unregister(ptr %field_val401)
  %unreg_field_1402 = getelementptr inbounds %Block, ptr %field_val395, i32 0, i32 1
  %field_val403 = load ptr, ptr %unreg_field_1402, align 8
  call void @tl_mem_unregister(ptr %field_val403)
  %unreg_field_0404 = getelementptr inbounds %CausalSelfAttention, ptr %field_val403, i32 0, i32 0
  %field_val405 = load ptr, ptr %unreg_field_0404, align 8
  call void @tl_mem_unregister(ptr %field_val405)
  %unreg_field_0406 = getelementptr inbounds %Linear, ptr %field_val405, i32 0, i32 0
  %field_val407 = load ptr, ptr %unreg_field_0406, align 8
  call void @tl_mem_unregister(ptr %field_val407)
  %unreg_field_1408 = getelementptr inbounds %Linear, ptr %field_val405, i32 0, i32 1
  %field_val409 = load ptr, ptr %unreg_field_1408, align 8
  call void @tl_mem_unregister(ptr %field_val409)
  %unreg_field_1410 = getelementptr inbounds %CausalSelfAttention, ptr %field_val403, i32 0, i32 1
  %field_val411 = load ptr, ptr %unreg_field_1410, align 8
  call void @tl_mem_unregister(ptr %field_val411)
  %unreg_field_0412 = getelementptr inbounds %Linear, ptr %field_val411, i32 0, i32 0
  %field_val413 = load ptr, ptr %unreg_field_0412, align 8
  call void @tl_mem_unregister(ptr %field_val413)
  %unreg_field_1414 = getelementptr inbounds %Linear, ptr %field_val411, i32 0, i32 1
  %field_val415 = load ptr, ptr %unreg_field_1414, align 8
  call void @tl_mem_unregister(ptr %field_val415)
  %unreg_field_2416 = getelementptr inbounds %CausalSelfAttention, ptr %field_val403, i32 0, i32 2
  %field_val417 = load ptr, ptr %unreg_field_2416, align 8
  call void @tl_mem_unregister(ptr %field_val417)
  %unreg_field_0418 = getelementptr inbounds %Linear, ptr %field_val417, i32 0, i32 0
  %field_val419 = load ptr, ptr %unreg_field_0418, align 8
  call void @tl_mem_unregister(ptr %field_val419)
  %unreg_field_1420 = getelementptr inbounds %Linear, ptr %field_val417, i32 0, i32 1
  %field_val421 = load ptr, ptr %unreg_field_1420, align 8
  call void @tl_mem_unregister(ptr %field_val421)
  %unreg_field_3422 = getelementptr inbounds %CausalSelfAttention, ptr %field_val403, i32 0, i32 3
  %field_val423 = load ptr, ptr %unreg_field_3422, align 8
  call void @tl_mem_unregister(ptr %field_val423)
  %unreg_field_0424 = getelementptr inbounds %Linear, ptr %field_val423, i32 0, i32 0
  %field_val425 = load ptr, ptr %unreg_field_0424, align 8
  call void @tl_mem_unregister(ptr %field_val425)
  %unreg_field_1426 = getelementptr inbounds %Linear, ptr %field_val423, i32 0, i32 1
  %field_val427 = load ptr, ptr %unreg_field_1426, align 8
  call void @tl_mem_unregister(ptr %field_val427)
  %unreg_field_2428 = getelementptr inbounds %Block, ptr %field_val395, i32 0, i32 2
  %field_val429 = load ptr, ptr %unreg_field_2428, align 8
  call void @tl_mem_unregister(ptr %field_val429)
  %unreg_field_0430 = getelementptr inbounds %LayerNorm, ptr %field_val429, i32 0, i32 0
  %field_val431 = load ptr, ptr %unreg_field_0430, align 8
  call void @tl_mem_unregister(ptr %field_val431)
  %unreg_field_1432 = getelementptr inbounds %LayerNorm, ptr %field_val429, i32 0, i32 1
  %field_val433 = load ptr, ptr %unreg_field_1432, align 8
  call void @tl_mem_unregister(ptr %field_val433)
  %unreg_field_3434 = getelementptr inbounds %Block, ptr %field_val395, i32 0, i32 3
  %field_val435 = load ptr, ptr %unreg_field_3434, align 8
  call void @tl_mem_unregister(ptr %field_val435)
  %unreg_field_0436 = getelementptr inbounds %MLP, ptr %field_val435, i32 0, i32 0
  %field_val437 = load ptr, ptr %unreg_field_0436, align 8
  call void @tl_mem_unregister(ptr %field_val437)
  %unreg_field_0438 = getelementptr inbounds %Linear, ptr %field_val437, i32 0, i32 0
  %field_val439 = load ptr, ptr %unreg_field_0438, align 8
  call void @tl_mem_unregister(ptr %field_val439)
  %unreg_field_1440 = getelementptr inbounds %Linear, ptr %field_val437, i32 0, i32 1
  %field_val441 = load ptr, ptr %unreg_field_1440, align 8
  call void @tl_mem_unregister(ptr %field_val441)
  %unreg_field_1442 = getelementptr inbounds %MLP, ptr %field_val435, i32 0, i32 1
  %field_val443 = load ptr, ptr %unreg_field_1442, align 8
  call void @tl_mem_unregister(ptr %field_val443)
  %unreg_field_0444 = getelementptr inbounds %Linear, ptr %field_val443, i32 0, i32 0
  %field_val445 = load ptr, ptr %unreg_field_0444, align 8
  call void @tl_mem_unregister(ptr %field_val445)
  %unreg_field_1446 = getelementptr inbounds %Linear, ptr %field_val443, i32 0, i32 1
  %field_val447 = load ptr, ptr %unreg_field_1446, align 8
  call void @tl_mem_unregister(ptr %field_val447)
  %unreg_field_3448 = getelementptr inbounds %GPT, ptr %call_tmp250, i32 0, i32 3
  %field_val449 = load ptr, ptr %unreg_field_3448, align 8
  call void @tl_mem_unregister(ptr %field_val449)
  %unreg_field_0450 = getelementptr inbounds %Block, ptr %field_val449, i32 0, i32 0
  %field_val451 = load ptr, ptr %unreg_field_0450, align 8
  call void @tl_mem_unregister(ptr %field_val451)
  %unreg_field_0452 = getelementptr inbounds %LayerNorm, ptr %field_val451, i32 0, i32 0
  %field_val453 = load ptr, ptr %unreg_field_0452, align 8
  call void @tl_mem_unregister(ptr %field_val453)
  %unreg_field_1454 = getelementptr inbounds %LayerNorm, ptr %field_val451, i32 0, i32 1
  %field_val455 = load ptr, ptr %unreg_field_1454, align 8
  call void @tl_mem_unregister(ptr %field_val455)
  %unreg_field_1456 = getelementptr inbounds %Block, ptr %field_val449, i32 0, i32 1
  %field_val457 = load ptr, ptr %unreg_field_1456, align 8
  call void @tl_mem_unregister(ptr %field_val457)
  %unreg_field_0458 = getelementptr inbounds %CausalSelfAttention, ptr %field_val457, i32 0, i32 0
  %field_val459 = load ptr, ptr %unreg_field_0458, align 8
  call void @tl_mem_unregister(ptr %field_val459)
  %unreg_field_0460 = getelementptr inbounds %Linear, ptr %field_val459, i32 0, i32 0
  %field_val461 = load ptr, ptr %unreg_field_0460, align 8
  call void @tl_mem_unregister(ptr %field_val461)
  %unreg_field_1462 = getelementptr inbounds %Linear, ptr %field_val459, i32 0, i32 1
  %field_val463 = load ptr, ptr %unreg_field_1462, align 8
  call void @tl_mem_unregister(ptr %field_val463)
  %unreg_field_1464 = getelementptr inbounds %CausalSelfAttention, ptr %field_val457, i32 0, i32 1
  %field_val465 = load ptr, ptr %unreg_field_1464, align 8
  call void @tl_mem_unregister(ptr %field_val465)
  %unreg_field_0466 = getelementptr inbounds %Linear, ptr %field_val465, i32 0, i32 0
  %field_val467 = load ptr, ptr %unreg_field_0466, align 8
  call void @tl_mem_unregister(ptr %field_val467)
  %unreg_field_1468 = getelementptr inbounds %Linear, ptr %field_val465, i32 0, i32 1
  %field_val469 = load ptr, ptr %unreg_field_1468, align 8
  call void @tl_mem_unregister(ptr %field_val469)
  %unreg_field_2470 = getelementptr inbounds %CausalSelfAttention, ptr %field_val457, i32 0, i32 2
  %field_val471 = load ptr, ptr %unreg_field_2470, align 8
  call void @tl_mem_unregister(ptr %field_val471)
  %unreg_field_0472 = getelementptr inbounds %Linear, ptr %field_val471, i32 0, i32 0
  %field_val473 = load ptr, ptr %unreg_field_0472, align 8
  call void @tl_mem_unregister(ptr %field_val473)
  %unreg_field_1474 = getelementptr inbounds %Linear, ptr %field_val471, i32 0, i32 1
  %field_val475 = load ptr, ptr %unreg_field_1474, align 8
  call void @tl_mem_unregister(ptr %field_val475)
  %unreg_field_3476 = getelementptr inbounds %CausalSelfAttention, ptr %field_val457, i32 0, i32 3
  %field_val477 = load ptr, ptr %unreg_field_3476, align 8
  call void @tl_mem_unregister(ptr %field_val477)
  %unreg_field_0478 = getelementptr inbounds %Linear, ptr %field_val477, i32 0, i32 0
  %field_val479 = load ptr, ptr %unreg_field_0478, align 8
  call void @tl_mem_unregister(ptr %field_val479)
  %unreg_field_1480 = getelementptr inbounds %Linear, ptr %field_val477, i32 0, i32 1
  %field_val481 = load ptr, ptr %unreg_field_1480, align 8
  call void @tl_mem_unregister(ptr %field_val481)
  %unreg_field_2482 = getelementptr inbounds %Block, ptr %field_val449, i32 0, i32 2
  %field_val483 = load ptr, ptr %unreg_field_2482, align 8
  call void @tl_mem_unregister(ptr %field_val483)
  %unreg_field_0484 = getelementptr inbounds %LayerNorm, ptr %field_val483, i32 0, i32 0
  %field_val485 = load ptr, ptr %unreg_field_0484, align 8
  call void @tl_mem_unregister(ptr %field_val485)
  %unreg_field_1486 = getelementptr inbounds %LayerNorm, ptr %field_val483, i32 0, i32 1
  %field_val487 = load ptr, ptr %unreg_field_1486, align 8
  call void @tl_mem_unregister(ptr %field_val487)
  %unreg_field_3488 = getelementptr inbounds %Block, ptr %field_val449, i32 0, i32 3
  %field_val489 = load ptr, ptr %unreg_field_3488, align 8
  call void @tl_mem_unregister(ptr %field_val489)
  %unreg_field_0490 = getelementptr inbounds %MLP, ptr %field_val489, i32 0, i32 0
  %field_val491 = load ptr, ptr %unreg_field_0490, align 8
  call void @tl_mem_unregister(ptr %field_val491)
  %unreg_field_0492 = getelementptr inbounds %Linear, ptr %field_val491, i32 0, i32 0
  %field_val493 = load ptr, ptr %unreg_field_0492, align 8
  call void @tl_mem_unregister(ptr %field_val493)
  %unreg_field_1494 = getelementptr inbounds %Linear, ptr %field_val491, i32 0, i32 1
  %field_val495 = load ptr, ptr %unreg_field_1494, align 8
  call void @tl_mem_unregister(ptr %field_val495)
  %unreg_field_1496 = getelementptr inbounds %MLP, ptr %field_val489, i32 0, i32 1
  %field_val497 = load ptr, ptr %unreg_field_1496, align 8
  call void @tl_mem_unregister(ptr %field_val497)
  %unreg_field_0498 = getelementptr inbounds %Linear, ptr %field_val497, i32 0, i32 0
  %field_val499 = load ptr, ptr %unreg_field_0498, align 8
  call void @tl_mem_unregister(ptr %field_val499)
  %unreg_field_1500 = getelementptr inbounds %Linear, ptr %field_val497, i32 0, i32 1
  %field_val501 = load ptr, ptr %unreg_field_1500, align 8
  call void @tl_mem_unregister(ptr %field_val501)
  %unreg_field_4502 = getelementptr inbounds %GPT, ptr %call_tmp250, i32 0, i32 4
  %field_val503 = load ptr, ptr %unreg_field_4502, align 8
  call void @tl_mem_unregister(ptr %field_val503)
  %unreg_field_0504 = getelementptr inbounds %LayerNorm, ptr %field_val503, i32 0, i32 0
  %field_val505 = load ptr, ptr %unreg_field_0504, align 8
  call void @tl_mem_unregister(ptr %field_val505)
  %unreg_field_1506 = getelementptr inbounds %LayerNorm, ptr %field_val503, i32 0, i32 1
  %field_val507 = load ptr, ptr %unreg_field_1506, align 8
  call void @tl_mem_unregister(ptr %field_val507)
  %unreg_field_5508 = getelementptr inbounds %GPT, ptr %call_tmp250, i32 0, i32 5
  %field_val509 = load ptr, ptr %unreg_field_5508, align 8
  call void @tl_mem_unregister(ptr %field_val509)
  %unreg_field_0510 = getelementptr inbounds %Linear, ptr %field_val509, i32 0, i32 0
  %field_val511 = load ptr, ptr %unreg_field_0510, align 8
  call void @tl_mem_unregister(ptr %field_val511)
  %unreg_field_1512 = getelementptr inbounds %Linear, ptr %field_val509, i32 0, i32 1
  %field_val513 = load ptr, ptr %unreg_field_1512, align 8
  call void @tl_mem_unregister(ptr %field_val513)
  call void @tl_mem_unregister(ptr %call_tmp250)
  store ptr %call_tmp250, ptr %model, align 8
  %model514 = load ptr, ptr %model, align 8
  %call_tmp515 = call ptr @train_step(ptr %model514, float 0x3FA99999A0000000, i64 99, i64 99)
  call void @tl_mem_unregister(ptr %call_tmp515)
  %old_struct_to_free516 = load ptr, ptr %model, align 8
  %is_not_null517 = icmp ne ptr %old_struct_to_free516, null
  %are_diff518 = icmp ne ptr %old_struct_to_free516, %call_tmp515
  %can_free_1519 = and i1 %is_not_null517, true
  %can_free520 = and i1 %can_free_1519, %are_diff518
  br i1 %can_free520, label %free_struct521, label %continue_after_free522

free_struct521:                                   ; preds = %continue_after_free257
  %field_gep523 = getelementptr inbounds %GPT, ptr %old_struct_to_free516, i32 0, i32 0
  %field_load524 = load ptr, ptr %field_gep523, align 8
  %field_gep525 = getelementptr inbounds %Embedding, ptr %field_load524, i32 0, i32 0
  %field_load526 = load ptr, ptr %field_gep525, align 8
  call void @tl_tensor_free(ptr %field_load526)
  %field_gep527 = getelementptr inbounds %GPT, ptr %old_struct_to_free516, i32 0, i32 1
  %field_load528 = load ptr, ptr %field_gep527, align 8
  %field_gep529 = getelementptr inbounds %Embedding, ptr %field_load528, i32 0, i32 0
  %field_load530 = load ptr, ptr %field_gep529, align 8
  call void @tl_tensor_free(ptr %field_load530)
  %field_gep531 = getelementptr inbounds %GPT, ptr %old_struct_to_free516, i32 0, i32 2
  %field_load532 = load ptr, ptr %field_gep531, align 8
  %field_gep533 = getelementptr inbounds %Block, ptr %field_load532, i32 0, i32 0
  %field_load534 = load ptr, ptr %field_gep533, align 8
  %field_gep535 = getelementptr inbounds %LayerNorm, ptr %field_load534, i32 0, i32 0
  %field_load536 = load ptr, ptr %field_gep535, align 8
  call void @tl_tensor_free(ptr %field_load536)
  %field_gep537 = getelementptr inbounds %LayerNorm, ptr %field_load534, i32 0, i32 1
  %field_load538 = load ptr, ptr %field_gep537, align 8
  call void @tl_tensor_free(ptr %field_load538)
  %field_gep539 = getelementptr inbounds %Block, ptr %field_load532, i32 0, i32 1
  %field_load540 = load ptr, ptr %field_gep539, align 8
  %field_gep541 = getelementptr inbounds %CausalSelfAttention, ptr %field_load540, i32 0, i32 0
  %field_load542 = load ptr, ptr %field_gep541, align 8
  %field_gep543 = getelementptr inbounds %Linear, ptr %field_load542, i32 0, i32 0
  %field_load544 = load ptr, ptr %field_gep543, align 8
  call void @tl_tensor_free(ptr %field_load544)
  %field_gep545 = getelementptr inbounds %Linear, ptr %field_load542, i32 0, i32 1
  %field_load546 = load ptr, ptr %field_gep545, align 8
  call void @tl_tensor_free(ptr %field_load546)
  %field_gep547 = getelementptr inbounds %CausalSelfAttention, ptr %field_load540, i32 0, i32 1
  %field_load548 = load ptr, ptr %field_gep547, align 8
  %field_gep549 = getelementptr inbounds %Linear, ptr %field_load548, i32 0, i32 0
  %field_load550 = load ptr, ptr %field_gep549, align 8
  call void @tl_tensor_free(ptr %field_load550)
  %field_gep551 = getelementptr inbounds %Linear, ptr %field_load548, i32 0, i32 1
  %field_load552 = load ptr, ptr %field_gep551, align 8
  call void @tl_tensor_free(ptr %field_load552)
  %field_gep553 = getelementptr inbounds %CausalSelfAttention, ptr %field_load540, i32 0, i32 2
  %field_load554 = load ptr, ptr %field_gep553, align 8
  %field_gep555 = getelementptr inbounds %Linear, ptr %field_load554, i32 0, i32 0
  %field_load556 = load ptr, ptr %field_gep555, align 8
  call void @tl_tensor_free(ptr %field_load556)
  %field_gep557 = getelementptr inbounds %Linear, ptr %field_load554, i32 0, i32 1
  %field_load558 = load ptr, ptr %field_gep557, align 8
  call void @tl_tensor_free(ptr %field_load558)
  %field_gep559 = getelementptr inbounds %CausalSelfAttention, ptr %field_load540, i32 0, i32 3
  %field_load560 = load ptr, ptr %field_gep559, align 8
  %field_gep561 = getelementptr inbounds %Linear, ptr %field_load560, i32 0, i32 0
  %field_load562 = load ptr, ptr %field_gep561, align 8
  call void @tl_tensor_free(ptr %field_load562)
  %field_gep563 = getelementptr inbounds %Linear, ptr %field_load560, i32 0, i32 1
  %field_load564 = load ptr, ptr %field_gep563, align 8
  call void @tl_tensor_free(ptr %field_load564)
  %field_gep565 = getelementptr inbounds %Block, ptr %field_load532, i32 0, i32 2
  %field_load566 = load ptr, ptr %field_gep565, align 8
  %field_gep567 = getelementptr inbounds %LayerNorm, ptr %field_load566, i32 0, i32 0
  %field_load568 = load ptr, ptr %field_gep567, align 8
  call void @tl_tensor_free(ptr %field_load568)
  %field_gep569 = getelementptr inbounds %LayerNorm, ptr %field_load566, i32 0, i32 1
  %field_load570 = load ptr, ptr %field_gep569, align 8
  call void @tl_tensor_free(ptr %field_load570)
  %field_gep571 = getelementptr inbounds %Block, ptr %field_load532, i32 0, i32 3
  %field_load572 = load ptr, ptr %field_gep571, align 8
  %field_gep573 = getelementptr inbounds %MLP, ptr %field_load572, i32 0, i32 0
  %field_load574 = load ptr, ptr %field_gep573, align 8
  %field_gep575 = getelementptr inbounds %Linear, ptr %field_load574, i32 0, i32 0
  %field_load576 = load ptr, ptr %field_gep575, align 8
  call void @tl_tensor_free(ptr %field_load576)
  %field_gep577 = getelementptr inbounds %Linear, ptr %field_load574, i32 0, i32 1
  %field_load578 = load ptr, ptr %field_gep577, align 8
  call void @tl_tensor_free(ptr %field_load578)
  %field_gep579 = getelementptr inbounds %MLP, ptr %field_load572, i32 0, i32 1
  %field_load580 = load ptr, ptr %field_gep579, align 8
  %field_gep581 = getelementptr inbounds %Linear, ptr %field_load580, i32 0, i32 0
  %field_load582 = load ptr, ptr %field_gep581, align 8
  call void @tl_tensor_free(ptr %field_load582)
  %field_gep583 = getelementptr inbounds %Linear, ptr %field_load580, i32 0, i32 1
  %field_load584 = load ptr, ptr %field_gep583, align 8
  call void @tl_tensor_free(ptr %field_load584)
  %field_gep585 = getelementptr inbounds %GPT, ptr %old_struct_to_free516, i32 0, i32 3
  %field_load586 = load ptr, ptr %field_gep585, align 8
  %field_gep587 = getelementptr inbounds %Block, ptr %field_load586, i32 0, i32 0
  %field_load588 = load ptr, ptr %field_gep587, align 8
  %field_gep589 = getelementptr inbounds %LayerNorm, ptr %field_load588, i32 0, i32 0
  %field_load590 = load ptr, ptr %field_gep589, align 8
  call void @tl_tensor_free(ptr %field_load590)
  %field_gep591 = getelementptr inbounds %LayerNorm, ptr %field_load588, i32 0, i32 1
  %field_load592 = load ptr, ptr %field_gep591, align 8
  call void @tl_tensor_free(ptr %field_load592)
  %field_gep593 = getelementptr inbounds %Block, ptr %field_load586, i32 0, i32 1
  %field_load594 = load ptr, ptr %field_gep593, align 8
  %field_gep595 = getelementptr inbounds %CausalSelfAttention, ptr %field_load594, i32 0, i32 0
  %field_load596 = load ptr, ptr %field_gep595, align 8
  %field_gep597 = getelementptr inbounds %Linear, ptr %field_load596, i32 0, i32 0
  %field_load598 = load ptr, ptr %field_gep597, align 8
  call void @tl_tensor_free(ptr %field_load598)
  %field_gep599 = getelementptr inbounds %Linear, ptr %field_load596, i32 0, i32 1
  %field_load600 = load ptr, ptr %field_gep599, align 8
  call void @tl_tensor_free(ptr %field_load600)
  %field_gep601 = getelementptr inbounds %CausalSelfAttention, ptr %field_load594, i32 0, i32 1
  %field_load602 = load ptr, ptr %field_gep601, align 8
  %field_gep603 = getelementptr inbounds %Linear, ptr %field_load602, i32 0, i32 0
  %field_load604 = load ptr, ptr %field_gep603, align 8
  call void @tl_tensor_free(ptr %field_load604)
  %field_gep605 = getelementptr inbounds %Linear, ptr %field_load602, i32 0, i32 1
  %field_load606 = load ptr, ptr %field_gep605, align 8
  call void @tl_tensor_free(ptr %field_load606)
  %field_gep607 = getelementptr inbounds %CausalSelfAttention, ptr %field_load594, i32 0, i32 2
  %field_load608 = load ptr, ptr %field_gep607, align 8
  %field_gep609 = getelementptr inbounds %Linear, ptr %field_load608, i32 0, i32 0
  %field_load610 = load ptr, ptr %field_gep609, align 8
  call void @tl_tensor_free(ptr %field_load610)
  %field_gep611 = getelementptr inbounds %Linear, ptr %field_load608, i32 0, i32 1
  %field_load612 = load ptr, ptr %field_gep611, align 8
  call void @tl_tensor_free(ptr %field_load612)
  %field_gep613 = getelementptr inbounds %CausalSelfAttention, ptr %field_load594, i32 0, i32 3
  %field_load614 = load ptr, ptr %field_gep613, align 8
  %field_gep615 = getelementptr inbounds %Linear, ptr %field_load614, i32 0, i32 0
  %field_load616 = load ptr, ptr %field_gep615, align 8
  call void @tl_tensor_free(ptr %field_load616)
  %field_gep617 = getelementptr inbounds %Linear, ptr %field_load614, i32 0, i32 1
  %field_load618 = load ptr, ptr %field_gep617, align 8
  call void @tl_tensor_free(ptr %field_load618)
  %field_gep619 = getelementptr inbounds %Block, ptr %field_load586, i32 0, i32 2
  %field_load620 = load ptr, ptr %field_gep619, align 8
  %field_gep621 = getelementptr inbounds %LayerNorm, ptr %field_load620, i32 0, i32 0
  %field_load622 = load ptr, ptr %field_gep621, align 8
  call void @tl_tensor_free(ptr %field_load622)
  %field_gep623 = getelementptr inbounds %LayerNorm, ptr %field_load620, i32 0, i32 1
  %field_load624 = load ptr, ptr %field_gep623, align 8
  call void @tl_tensor_free(ptr %field_load624)
  %field_gep625 = getelementptr inbounds %Block, ptr %field_load586, i32 0, i32 3
  %field_load626 = load ptr, ptr %field_gep625, align 8
  %field_gep627 = getelementptr inbounds %MLP, ptr %field_load626, i32 0, i32 0
  %field_load628 = load ptr, ptr %field_gep627, align 8
  %field_gep629 = getelementptr inbounds %Linear, ptr %field_load628, i32 0, i32 0
  %field_load630 = load ptr, ptr %field_gep629, align 8
  call void @tl_tensor_free(ptr %field_load630)
  %field_gep631 = getelementptr inbounds %Linear, ptr %field_load628, i32 0, i32 1
  %field_load632 = load ptr, ptr %field_gep631, align 8
  call void @tl_tensor_free(ptr %field_load632)
  %field_gep633 = getelementptr inbounds %MLP, ptr %field_load626, i32 0, i32 1
  %field_load634 = load ptr, ptr %field_gep633, align 8
  %field_gep635 = getelementptr inbounds %Linear, ptr %field_load634, i32 0, i32 0
  %field_load636 = load ptr, ptr %field_gep635, align 8
  call void @tl_tensor_free(ptr %field_load636)
  %field_gep637 = getelementptr inbounds %Linear, ptr %field_load634, i32 0, i32 1
  %field_load638 = load ptr, ptr %field_gep637, align 8
  call void @tl_tensor_free(ptr %field_load638)
  %field_gep639 = getelementptr inbounds %GPT, ptr %old_struct_to_free516, i32 0, i32 4
  %field_load640 = load ptr, ptr %field_gep639, align 8
  %field_gep641 = getelementptr inbounds %LayerNorm, ptr %field_load640, i32 0, i32 0
  %field_load642 = load ptr, ptr %field_gep641, align 8
  call void @tl_tensor_free(ptr %field_load642)
  %field_gep643 = getelementptr inbounds %LayerNorm, ptr %field_load640, i32 0, i32 1
  %field_load644 = load ptr, ptr %field_gep643, align 8
  call void @tl_tensor_free(ptr %field_load644)
  %field_gep645 = getelementptr inbounds %GPT, ptr %old_struct_to_free516, i32 0, i32 5
  %field_load646 = load ptr, ptr %field_gep645, align 8
  %field_gep647 = getelementptr inbounds %Linear, ptr %field_load646, i32 0, i32 0
  %field_load648 = load ptr, ptr %field_gep647, align 8
  call void @tl_tensor_free(ptr %field_load648)
  %field_gep649 = getelementptr inbounds %Linear, ptr %field_load646, i32 0, i32 1
  %field_load650 = load ptr, ptr %field_gep649, align 8
  call void @tl_tensor_free(ptr %field_load650)
  call void @tl_mem_unregister(ptr %old_struct_to_free516)
  br label %continue_after_free522

continue_after_free522:                           ; preds = %free_struct521, %continue_after_free257
  call void @tl_mem_unregister(ptr %call_tmp515)
  %unreg_field_0651 = getelementptr inbounds %GPT, ptr %call_tmp515, i32 0, i32 0
  %field_val652 = load ptr, ptr %unreg_field_0651, align 8
  call void @tl_mem_unregister(ptr %field_val652)
  %unreg_field_0653 = getelementptr inbounds %Embedding, ptr %field_val652, i32 0, i32 0
  %field_val654 = load ptr, ptr %unreg_field_0653, align 8
  call void @tl_mem_unregister(ptr %field_val654)
  %unreg_field_1655 = getelementptr inbounds %GPT, ptr %call_tmp515, i32 0, i32 1
  %field_val656 = load ptr, ptr %unreg_field_1655, align 8
  call void @tl_mem_unregister(ptr %field_val656)
  %unreg_field_0657 = getelementptr inbounds %Embedding, ptr %field_val656, i32 0, i32 0
  %field_val658 = load ptr, ptr %unreg_field_0657, align 8
  call void @tl_mem_unregister(ptr %field_val658)
  %unreg_field_2659 = getelementptr inbounds %GPT, ptr %call_tmp515, i32 0, i32 2
  %field_val660 = load ptr, ptr %unreg_field_2659, align 8
  call void @tl_mem_unregister(ptr %field_val660)
  %unreg_field_0661 = getelementptr inbounds %Block, ptr %field_val660, i32 0, i32 0
  %field_val662 = load ptr, ptr %unreg_field_0661, align 8
  call void @tl_mem_unregister(ptr %field_val662)
  %unreg_field_0663 = getelementptr inbounds %LayerNorm, ptr %field_val662, i32 0, i32 0
  %field_val664 = load ptr, ptr %unreg_field_0663, align 8
  call void @tl_mem_unregister(ptr %field_val664)
  %unreg_field_1665 = getelementptr inbounds %LayerNorm, ptr %field_val662, i32 0, i32 1
  %field_val666 = load ptr, ptr %unreg_field_1665, align 8
  call void @tl_mem_unregister(ptr %field_val666)
  %unreg_field_1667 = getelementptr inbounds %Block, ptr %field_val660, i32 0, i32 1
  %field_val668 = load ptr, ptr %unreg_field_1667, align 8
  call void @tl_mem_unregister(ptr %field_val668)
  %unreg_field_0669 = getelementptr inbounds %CausalSelfAttention, ptr %field_val668, i32 0, i32 0
  %field_val670 = load ptr, ptr %unreg_field_0669, align 8
  call void @tl_mem_unregister(ptr %field_val670)
  %unreg_field_0671 = getelementptr inbounds %Linear, ptr %field_val670, i32 0, i32 0
  %field_val672 = load ptr, ptr %unreg_field_0671, align 8
  call void @tl_mem_unregister(ptr %field_val672)
  %unreg_field_1673 = getelementptr inbounds %Linear, ptr %field_val670, i32 0, i32 1
  %field_val674 = load ptr, ptr %unreg_field_1673, align 8
  call void @tl_mem_unregister(ptr %field_val674)
  %unreg_field_1675 = getelementptr inbounds %CausalSelfAttention, ptr %field_val668, i32 0, i32 1
  %field_val676 = load ptr, ptr %unreg_field_1675, align 8
  call void @tl_mem_unregister(ptr %field_val676)
  %unreg_field_0677 = getelementptr inbounds %Linear, ptr %field_val676, i32 0, i32 0
  %field_val678 = load ptr, ptr %unreg_field_0677, align 8
  call void @tl_mem_unregister(ptr %field_val678)
  %unreg_field_1679 = getelementptr inbounds %Linear, ptr %field_val676, i32 0, i32 1
  %field_val680 = load ptr, ptr %unreg_field_1679, align 8
  call void @tl_mem_unregister(ptr %field_val680)
  %unreg_field_2681 = getelementptr inbounds %CausalSelfAttention, ptr %field_val668, i32 0, i32 2
  %field_val682 = load ptr, ptr %unreg_field_2681, align 8
  call void @tl_mem_unregister(ptr %field_val682)
  %unreg_field_0683 = getelementptr inbounds %Linear, ptr %field_val682, i32 0, i32 0
  %field_val684 = load ptr, ptr %unreg_field_0683, align 8
  call void @tl_mem_unregister(ptr %field_val684)
  %unreg_field_1685 = getelementptr inbounds %Linear, ptr %field_val682, i32 0, i32 1
  %field_val686 = load ptr, ptr %unreg_field_1685, align 8
  call void @tl_mem_unregister(ptr %field_val686)
  %unreg_field_3687 = getelementptr inbounds %CausalSelfAttention, ptr %field_val668, i32 0, i32 3
  %field_val688 = load ptr, ptr %unreg_field_3687, align 8
  call void @tl_mem_unregister(ptr %field_val688)
  %unreg_field_0689 = getelementptr inbounds %Linear, ptr %field_val688, i32 0, i32 0
  %field_val690 = load ptr, ptr %unreg_field_0689, align 8
  call void @tl_mem_unregister(ptr %field_val690)
  %unreg_field_1691 = getelementptr inbounds %Linear, ptr %field_val688, i32 0, i32 1
  %field_val692 = load ptr, ptr %unreg_field_1691, align 8
  call void @tl_mem_unregister(ptr %field_val692)
  %unreg_field_2693 = getelementptr inbounds %Block, ptr %field_val660, i32 0, i32 2
  %field_val694 = load ptr, ptr %unreg_field_2693, align 8
  call void @tl_mem_unregister(ptr %field_val694)
  %unreg_field_0695 = getelementptr inbounds %LayerNorm, ptr %field_val694, i32 0, i32 0
  %field_val696 = load ptr, ptr %unreg_field_0695, align 8
  call void @tl_mem_unregister(ptr %field_val696)
  %unreg_field_1697 = getelementptr inbounds %LayerNorm, ptr %field_val694, i32 0, i32 1
  %field_val698 = load ptr, ptr %unreg_field_1697, align 8
  call void @tl_mem_unregister(ptr %field_val698)
  %unreg_field_3699 = getelementptr inbounds %Block, ptr %field_val660, i32 0, i32 3
  %field_val700 = load ptr, ptr %unreg_field_3699, align 8
  call void @tl_mem_unregister(ptr %field_val700)
  %unreg_field_0701 = getelementptr inbounds %MLP, ptr %field_val700, i32 0, i32 0
  %field_val702 = load ptr, ptr %unreg_field_0701, align 8
  call void @tl_mem_unregister(ptr %field_val702)
  %unreg_field_0703 = getelementptr inbounds %Linear, ptr %field_val702, i32 0, i32 0
  %field_val704 = load ptr, ptr %unreg_field_0703, align 8
  call void @tl_mem_unregister(ptr %field_val704)
  %unreg_field_1705 = getelementptr inbounds %Linear, ptr %field_val702, i32 0, i32 1
  %field_val706 = load ptr, ptr %unreg_field_1705, align 8
  call void @tl_mem_unregister(ptr %field_val706)
  %unreg_field_1707 = getelementptr inbounds %MLP, ptr %field_val700, i32 0, i32 1
  %field_val708 = load ptr, ptr %unreg_field_1707, align 8
  call void @tl_mem_unregister(ptr %field_val708)
  %unreg_field_0709 = getelementptr inbounds %Linear, ptr %field_val708, i32 0, i32 0
  %field_val710 = load ptr, ptr %unreg_field_0709, align 8
  call void @tl_mem_unregister(ptr %field_val710)
  %unreg_field_1711 = getelementptr inbounds %Linear, ptr %field_val708, i32 0, i32 1
  %field_val712 = load ptr, ptr %unreg_field_1711, align 8
  call void @tl_mem_unregister(ptr %field_val712)
  %unreg_field_3713 = getelementptr inbounds %GPT, ptr %call_tmp515, i32 0, i32 3
  %field_val714 = load ptr, ptr %unreg_field_3713, align 8
  call void @tl_mem_unregister(ptr %field_val714)
  %unreg_field_0715 = getelementptr inbounds %Block, ptr %field_val714, i32 0, i32 0
  %field_val716 = load ptr, ptr %unreg_field_0715, align 8
  call void @tl_mem_unregister(ptr %field_val716)
  %unreg_field_0717 = getelementptr inbounds %LayerNorm, ptr %field_val716, i32 0, i32 0
  %field_val718 = load ptr, ptr %unreg_field_0717, align 8
  call void @tl_mem_unregister(ptr %field_val718)
  %unreg_field_1719 = getelementptr inbounds %LayerNorm, ptr %field_val716, i32 0, i32 1
  %field_val720 = load ptr, ptr %unreg_field_1719, align 8
  call void @tl_mem_unregister(ptr %field_val720)
  %unreg_field_1721 = getelementptr inbounds %Block, ptr %field_val714, i32 0, i32 1
  %field_val722 = load ptr, ptr %unreg_field_1721, align 8
  call void @tl_mem_unregister(ptr %field_val722)
  %unreg_field_0723 = getelementptr inbounds %CausalSelfAttention, ptr %field_val722, i32 0, i32 0
  %field_val724 = load ptr, ptr %unreg_field_0723, align 8
  call void @tl_mem_unregister(ptr %field_val724)
  %unreg_field_0725 = getelementptr inbounds %Linear, ptr %field_val724, i32 0, i32 0
  %field_val726 = load ptr, ptr %unreg_field_0725, align 8
  call void @tl_mem_unregister(ptr %field_val726)
  %unreg_field_1727 = getelementptr inbounds %Linear, ptr %field_val724, i32 0, i32 1
  %field_val728 = load ptr, ptr %unreg_field_1727, align 8
  call void @tl_mem_unregister(ptr %field_val728)
  %unreg_field_1729 = getelementptr inbounds %CausalSelfAttention, ptr %field_val722, i32 0, i32 1
  %field_val730 = load ptr, ptr %unreg_field_1729, align 8
  call void @tl_mem_unregister(ptr %field_val730)
  %unreg_field_0731 = getelementptr inbounds %Linear, ptr %field_val730, i32 0, i32 0
  %field_val732 = load ptr, ptr %unreg_field_0731, align 8
  call void @tl_mem_unregister(ptr %field_val732)
  %unreg_field_1733 = getelementptr inbounds %Linear, ptr %field_val730, i32 0, i32 1
  %field_val734 = load ptr, ptr %unreg_field_1733, align 8
  call void @tl_mem_unregister(ptr %field_val734)
  %unreg_field_2735 = getelementptr inbounds %CausalSelfAttention, ptr %field_val722, i32 0, i32 2
  %field_val736 = load ptr, ptr %unreg_field_2735, align 8
  call void @tl_mem_unregister(ptr %field_val736)
  %unreg_field_0737 = getelementptr inbounds %Linear, ptr %field_val736, i32 0, i32 0
  %field_val738 = load ptr, ptr %unreg_field_0737, align 8
  call void @tl_mem_unregister(ptr %field_val738)
  %unreg_field_1739 = getelementptr inbounds %Linear, ptr %field_val736, i32 0, i32 1
  %field_val740 = load ptr, ptr %unreg_field_1739, align 8
  call void @tl_mem_unregister(ptr %field_val740)
  %unreg_field_3741 = getelementptr inbounds %CausalSelfAttention, ptr %field_val722, i32 0, i32 3
  %field_val742 = load ptr, ptr %unreg_field_3741, align 8
  call void @tl_mem_unregister(ptr %field_val742)
  %unreg_field_0743 = getelementptr inbounds %Linear, ptr %field_val742, i32 0, i32 0
  %field_val744 = load ptr, ptr %unreg_field_0743, align 8
  call void @tl_mem_unregister(ptr %field_val744)
  %unreg_field_1745 = getelementptr inbounds %Linear, ptr %field_val742, i32 0, i32 1
  %field_val746 = load ptr, ptr %unreg_field_1745, align 8
  call void @tl_mem_unregister(ptr %field_val746)
  %unreg_field_2747 = getelementptr inbounds %Block, ptr %field_val714, i32 0, i32 2
  %field_val748 = load ptr, ptr %unreg_field_2747, align 8
  call void @tl_mem_unregister(ptr %field_val748)
  %unreg_field_0749 = getelementptr inbounds %LayerNorm, ptr %field_val748, i32 0, i32 0
  %field_val750 = load ptr, ptr %unreg_field_0749, align 8
  call void @tl_mem_unregister(ptr %field_val750)
  %unreg_field_1751 = getelementptr inbounds %LayerNorm, ptr %field_val748, i32 0, i32 1
  %field_val752 = load ptr, ptr %unreg_field_1751, align 8
  call void @tl_mem_unregister(ptr %field_val752)
  %unreg_field_3753 = getelementptr inbounds %Block, ptr %field_val714, i32 0, i32 3
  %field_val754 = load ptr, ptr %unreg_field_3753, align 8
  call void @tl_mem_unregister(ptr %field_val754)
  %unreg_field_0755 = getelementptr inbounds %MLP, ptr %field_val754, i32 0, i32 0
  %field_val756 = load ptr, ptr %unreg_field_0755, align 8
  call void @tl_mem_unregister(ptr %field_val756)
  %unreg_field_0757 = getelementptr inbounds %Linear, ptr %field_val756, i32 0, i32 0
  %field_val758 = load ptr, ptr %unreg_field_0757, align 8
  call void @tl_mem_unregister(ptr %field_val758)
  %unreg_field_1759 = getelementptr inbounds %Linear, ptr %field_val756, i32 0, i32 1
  %field_val760 = load ptr, ptr %unreg_field_1759, align 8
  call void @tl_mem_unregister(ptr %field_val760)
  %unreg_field_1761 = getelementptr inbounds %MLP, ptr %field_val754, i32 0, i32 1
  %field_val762 = load ptr, ptr %unreg_field_1761, align 8
  call void @tl_mem_unregister(ptr %field_val762)
  %unreg_field_0763 = getelementptr inbounds %Linear, ptr %field_val762, i32 0, i32 0
  %field_val764 = load ptr, ptr %unreg_field_0763, align 8
  call void @tl_mem_unregister(ptr %field_val764)
  %unreg_field_1765 = getelementptr inbounds %Linear, ptr %field_val762, i32 0, i32 1
  %field_val766 = load ptr, ptr %unreg_field_1765, align 8
  call void @tl_mem_unregister(ptr %field_val766)
  %unreg_field_4767 = getelementptr inbounds %GPT, ptr %call_tmp515, i32 0, i32 4
  %field_val768 = load ptr, ptr %unreg_field_4767, align 8
  call void @tl_mem_unregister(ptr %field_val768)
  %unreg_field_0769 = getelementptr inbounds %LayerNorm, ptr %field_val768, i32 0, i32 0
  %field_val770 = load ptr, ptr %unreg_field_0769, align 8
  call void @tl_mem_unregister(ptr %field_val770)
  %unreg_field_1771 = getelementptr inbounds %LayerNorm, ptr %field_val768, i32 0, i32 1
  %field_val772 = load ptr, ptr %unreg_field_1771, align 8
  call void @tl_mem_unregister(ptr %field_val772)
  %unreg_field_5773 = getelementptr inbounds %GPT, ptr %call_tmp515, i32 0, i32 5
  %field_val774 = load ptr, ptr %unreg_field_5773, align 8
  call void @tl_mem_unregister(ptr %field_val774)
  %unreg_field_0775 = getelementptr inbounds %Linear, ptr %field_val774, i32 0, i32 0
  %field_val776 = load ptr, ptr %unreg_field_0775, align 8
  call void @tl_mem_unregister(ptr %field_val776)
  %unreg_field_1777 = getelementptr inbounds %Linear, ptr %field_val774, i32 0, i32 1
  %field_val778 = load ptr, ptr %unreg_field_1777, align 8
  call void @tl_mem_unregister(ptr %field_val778)
  call void @tl_mem_unregister(ptr %call_tmp515)
  store ptr %call_tmp515, ptr %model, align 8
  %model779 = load ptr, ptr %model, align 8
  %call_tmp780 = call ptr @train_step(ptr %model779, float 0x3FA99999A0000000, i64 10, i64 5)
  call void @tl_mem_unregister(ptr %call_tmp780)
  %old_struct_to_free781 = load ptr, ptr %model, align 8
  %is_not_null782 = icmp ne ptr %old_struct_to_free781, null
  %are_diff783 = icmp ne ptr %old_struct_to_free781, %call_tmp780
  %can_free_1784 = and i1 %is_not_null782, true
  %can_free785 = and i1 %can_free_1784, %are_diff783
  br i1 %can_free785, label %free_struct786, label %continue_after_free787

free_struct786:                                   ; preds = %continue_after_free522
  %field_gep788 = getelementptr inbounds %GPT, ptr %old_struct_to_free781, i32 0, i32 0
  %field_load789 = load ptr, ptr %field_gep788, align 8
  %field_gep790 = getelementptr inbounds %Embedding, ptr %field_load789, i32 0, i32 0
  %field_load791 = load ptr, ptr %field_gep790, align 8
  call void @tl_tensor_free(ptr %field_load791)
  %field_gep792 = getelementptr inbounds %GPT, ptr %old_struct_to_free781, i32 0, i32 1
  %field_load793 = load ptr, ptr %field_gep792, align 8
  %field_gep794 = getelementptr inbounds %Embedding, ptr %field_load793, i32 0, i32 0
  %field_load795 = load ptr, ptr %field_gep794, align 8
  call void @tl_tensor_free(ptr %field_load795)
  %field_gep796 = getelementptr inbounds %GPT, ptr %old_struct_to_free781, i32 0, i32 2
  %field_load797 = load ptr, ptr %field_gep796, align 8
  %field_gep798 = getelementptr inbounds %Block, ptr %field_load797, i32 0, i32 0
  %field_load799 = load ptr, ptr %field_gep798, align 8
  %field_gep800 = getelementptr inbounds %LayerNorm, ptr %field_load799, i32 0, i32 0
  %field_load801 = load ptr, ptr %field_gep800, align 8
  call void @tl_tensor_free(ptr %field_load801)
  %field_gep802 = getelementptr inbounds %LayerNorm, ptr %field_load799, i32 0, i32 1
  %field_load803 = load ptr, ptr %field_gep802, align 8
  call void @tl_tensor_free(ptr %field_load803)
  %field_gep804 = getelementptr inbounds %Block, ptr %field_load797, i32 0, i32 1
  %field_load805 = load ptr, ptr %field_gep804, align 8
  %field_gep806 = getelementptr inbounds %CausalSelfAttention, ptr %field_load805, i32 0, i32 0
  %field_load807 = load ptr, ptr %field_gep806, align 8
  %field_gep808 = getelementptr inbounds %Linear, ptr %field_load807, i32 0, i32 0
  %field_load809 = load ptr, ptr %field_gep808, align 8
  call void @tl_tensor_free(ptr %field_load809)
  %field_gep810 = getelementptr inbounds %Linear, ptr %field_load807, i32 0, i32 1
  %field_load811 = load ptr, ptr %field_gep810, align 8
  call void @tl_tensor_free(ptr %field_load811)
  %field_gep812 = getelementptr inbounds %CausalSelfAttention, ptr %field_load805, i32 0, i32 1
  %field_load813 = load ptr, ptr %field_gep812, align 8
  %field_gep814 = getelementptr inbounds %Linear, ptr %field_load813, i32 0, i32 0
  %field_load815 = load ptr, ptr %field_gep814, align 8
  call void @tl_tensor_free(ptr %field_load815)
  %field_gep816 = getelementptr inbounds %Linear, ptr %field_load813, i32 0, i32 1
  %field_load817 = load ptr, ptr %field_gep816, align 8
  call void @tl_tensor_free(ptr %field_load817)
  %field_gep818 = getelementptr inbounds %CausalSelfAttention, ptr %field_load805, i32 0, i32 2
  %field_load819 = load ptr, ptr %field_gep818, align 8
  %field_gep820 = getelementptr inbounds %Linear, ptr %field_load819, i32 0, i32 0
  %field_load821 = load ptr, ptr %field_gep820, align 8
  call void @tl_tensor_free(ptr %field_load821)
  %field_gep822 = getelementptr inbounds %Linear, ptr %field_load819, i32 0, i32 1
  %field_load823 = load ptr, ptr %field_gep822, align 8
  call void @tl_tensor_free(ptr %field_load823)
  %field_gep824 = getelementptr inbounds %CausalSelfAttention, ptr %field_load805, i32 0, i32 3
  %field_load825 = load ptr, ptr %field_gep824, align 8
  %field_gep826 = getelementptr inbounds %Linear, ptr %field_load825, i32 0, i32 0
  %field_load827 = load ptr, ptr %field_gep826, align 8
  call void @tl_tensor_free(ptr %field_load827)
  %field_gep828 = getelementptr inbounds %Linear, ptr %field_load825, i32 0, i32 1
  %field_load829 = load ptr, ptr %field_gep828, align 8
  call void @tl_tensor_free(ptr %field_load829)
  %field_gep830 = getelementptr inbounds %Block, ptr %field_load797, i32 0, i32 2
  %field_load831 = load ptr, ptr %field_gep830, align 8
  %field_gep832 = getelementptr inbounds %LayerNorm, ptr %field_load831, i32 0, i32 0
  %field_load833 = load ptr, ptr %field_gep832, align 8
  call void @tl_tensor_free(ptr %field_load833)
  %field_gep834 = getelementptr inbounds %LayerNorm, ptr %field_load831, i32 0, i32 1
  %field_load835 = load ptr, ptr %field_gep834, align 8
  call void @tl_tensor_free(ptr %field_load835)
  %field_gep836 = getelementptr inbounds %Block, ptr %field_load797, i32 0, i32 3
  %field_load837 = load ptr, ptr %field_gep836, align 8
  %field_gep838 = getelementptr inbounds %MLP, ptr %field_load837, i32 0, i32 0
  %field_load839 = load ptr, ptr %field_gep838, align 8
  %field_gep840 = getelementptr inbounds %Linear, ptr %field_load839, i32 0, i32 0
  %field_load841 = load ptr, ptr %field_gep840, align 8
  call void @tl_tensor_free(ptr %field_load841)
  %field_gep842 = getelementptr inbounds %Linear, ptr %field_load839, i32 0, i32 1
  %field_load843 = load ptr, ptr %field_gep842, align 8
  call void @tl_tensor_free(ptr %field_load843)
  %field_gep844 = getelementptr inbounds %MLP, ptr %field_load837, i32 0, i32 1
  %field_load845 = load ptr, ptr %field_gep844, align 8
  %field_gep846 = getelementptr inbounds %Linear, ptr %field_load845, i32 0, i32 0
  %field_load847 = load ptr, ptr %field_gep846, align 8
  call void @tl_tensor_free(ptr %field_load847)
  %field_gep848 = getelementptr inbounds %Linear, ptr %field_load845, i32 0, i32 1
  %field_load849 = load ptr, ptr %field_gep848, align 8
  call void @tl_tensor_free(ptr %field_load849)
  %field_gep850 = getelementptr inbounds %GPT, ptr %old_struct_to_free781, i32 0, i32 3
  %field_load851 = load ptr, ptr %field_gep850, align 8
  %field_gep852 = getelementptr inbounds %Block, ptr %field_load851, i32 0, i32 0
  %field_load853 = load ptr, ptr %field_gep852, align 8
  %field_gep854 = getelementptr inbounds %LayerNorm, ptr %field_load853, i32 0, i32 0
  %field_load855 = load ptr, ptr %field_gep854, align 8
  call void @tl_tensor_free(ptr %field_load855)
  %field_gep856 = getelementptr inbounds %LayerNorm, ptr %field_load853, i32 0, i32 1
  %field_load857 = load ptr, ptr %field_gep856, align 8
  call void @tl_tensor_free(ptr %field_load857)
  %field_gep858 = getelementptr inbounds %Block, ptr %field_load851, i32 0, i32 1
  %field_load859 = load ptr, ptr %field_gep858, align 8
  %field_gep860 = getelementptr inbounds %CausalSelfAttention, ptr %field_load859, i32 0, i32 0
  %field_load861 = load ptr, ptr %field_gep860, align 8
  %field_gep862 = getelementptr inbounds %Linear, ptr %field_load861, i32 0, i32 0
  %field_load863 = load ptr, ptr %field_gep862, align 8
  call void @tl_tensor_free(ptr %field_load863)
  %field_gep864 = getelementptr inbounds %Linear, ptr %field_load861, i32 0, i32 1
  %field_load865 = load ptr, ptr %field_gep864, align 8
  call void @tl_tensor_free(ptr %field_load865)
  %field_gep866 = getelementptr inbounds %CausalSelfAttention, ptr %field_load859, i32 0, i32 1
  %field_load867 = load ptr, ptr %field_gep866, align 8
  %field_gep868 = getelementptr inbounds %Linear, ptr %field_load867, i32 0, i32 0
  %field_load869 = load ptr, ptr %field_gep868, align 8
  call void @tl_tensor_free(ptr %field_load869)
  %field_gep870 = getelementptr inbounds %Linear, ptr %field_load867, i32 0, i32 1
  %field_load871 = load ptr, ptr %field_gep870, align 8
  call void @tl_tensor_free(ptr %field_load871)
  %field_gep872 = getelementptr inbounds %CausalSelfAttention, ptr %field_load859, i32 0, i32 2
  %field_load873 = load ptr, ptr %field_gep872, align 8
  %field_gep874 = getelementptr inbounds %Linear, ptr %field_load873, i32 0, i32 0
  %field_load875 = load ptr, ptr %field_gep874, align 8
  call void @tl_tensor_free(ptr %field_load875)
  %field_gep876 = getelementptr inbounds %Linear, ptr %field_load873, i32 0, i32 1
  %field_load877 = load ptr, ptr %field_gep876, align 8
  call void @tl_tensor_free(ptr %field_load877)
  %field_gep878 = getelementptr inbounds %CausalSelfAttention, ptr %field_load859, i32 0, i32 3
  %field_load879 = load ptr, ptr %field_gep878, align 8
  %field_gep880 = getelementptr inbounds %Linear, ptr %field_load879, i32 0, i32 0
  %field_load881 = load ptr, ptr %field_gep880, align 8
  call void @tl_tensor_free(ptr %field_load881)
  %field_gep882 = getelementptr inbounds %Linear, ptr %field_load879, i32 0, i32 1
  %field_load883 = load ptr, ptr %field_gep882, align 8
  call void @tl_tensor_free(ptr %field_load883)
  %field_gep884 = getelementptr inbounds %Block, ptr %field_load851, i32 0, i32 2
  %field_load885 = load ptr, ptr %field_gep884, align 8
  %field_gep886 = getelementptr inbounds %LayerNorm, ptr %field_load885, i32 0, i32 0
  %field_load887 = load ptr, ptr %field_gep886, align 8
  call void @tl_tensor_free(ptr %field_load887)
  %field_gep888 = getelementptr inbounds %LayerNorm, ptr %field_load885, i32 0, i32 1
  %field_load889 = load ptr, ptr %field_gep888, align 8
  call void @tl_tensor_free(ptr %field_load889)
  %field_gep890 = getelementptr inbounds %Block, ptr %field_load851, i32 0, i32 3
  %field_load891 = load ptr, ptr %field_gep890, align 8
  %field_gep892 = getelementptr inbounds %MLP, ptr %field_load891, i32 0, i32 0
  %field_load893 = load ptr, ptr %field_gep892, align 8
  %field_gep894 = getelementptr inbounds %Linear, ptr %field_load893, i32 0, i32 0
  %field_load895 = load ptr, ptr %field_gep894, align 8
  call void @tl_tensor_free(ptr %field_load895)
  %field_gep896 = getelementptr inbounds %Linear, ptr %field_load893, i32 0, i32 1
  %field_load897 = load ptr, ptr %field_gep896, align 8
  call void @tl_tensor_free(ptr %field_load897)
  %field_gep898 = getelementptr inbounds %MLP, ptr %field_load891, i32 0, i32 1
  %field_load899 = load ptr, ptr %field_gep898, align 8
  %field_gep900 = getelementptr inbounds %Linear, ptr %field_load899, i32 0, i32 0
  %field_load901 = load ptr, ptr %field_gep900, align 8
  call void @tl_tensor_free(ptr %field_load901)
  %field_gep902 = getelementptr inbounds %Linear, ptr %field_load899, i32 0, i32 1
  %field_load903 = load ptr, ptr %field_gep902, align 8
  call void @tl_tensor_free(ptr %field_load903)
  %field_gep904 = getelementptr inbounds %GPT, ptr %old_struct_to_free781, i32 0, i32 4
  %field_load905 = load ptr, ptr %field_gep904, align 8
  %field_gep906 = getelementptr inbounds %LayerNorm, ptr %field_load905, i32 0, i32 0
  %field_load907 = load ptr, ptr %field_gep906, align 8
  call void @tl_tensor_free(ptr %field_load907)
  %field_gep908 = getelementptr inbounds %LayerNorm, ptr %field_load905, i32 0, i32 1
  %field_load909 = load ptr, ptr %field_gep908, align 8
  call void @tl_tensor_free(ptr %field_load909)
  %field_gep910 = getelementptr inbounds %GPT, ptr %old_struct_to_free781, i32 0, i32 5
  %field_load911 = load ptr, ptr %field_gep910, align 8
  %field_gep912 = getelementptr inbounds %Linear, ptr %field_load911, i32 0, i32 0
  %field_load913 = load ptr, ptr %field_gep912, align 8
  call void @tl_tensor_free(ptr %field_load913)
  %field_gep914 = getelementptr inbounds %Linear, ptr %field_load911, i32 0, i32 1
  %field_load915 = load ptr, ptr %field_gep914, align 8
  call void @tl_tensor_free(ptr %field_load915)
  call void @tl_mem_unregister(ptr %old_struct_to_free781)
  br label %continue_after_free787

continue_after_free787:                           ; preds = %free_struct786, %continue_after_free522
  call void @tl_mem_unregister(ptr %call_tmp780)
  %unreg_field_0916 = getelementptr inbounds %GPT, ptr %call_tmp780, i32 0, i32 0
  %field_val917 = load ptr, ptr %unreg_field_0916, align 8
  call void @tl_mem_unregister(ptr %field_val917)
  %unreg_field_0918 = getelementptr inbounds %Embedding, ptr %field_val917, i32 0, i32 0
  %field_val919 = load ptr, ptr %unreg_field_0918, align 8
  call void @tl_mem_unregister(ptr %field_val919)
  %unreg_field_1920 = getelementptr inbounds %GPT, ptr %call_tmp780, i32 0, i32 1
  %field_val921 = load ptr, ptr %unreg_field_1920, align 8
  call void @tl_mem_unregister(ptr %field_val921)
  %unreg_field_0922 = getelementptr inbounds %Embedding, ptr %field_val921, i32 0, i32 0
  %field_val923 = load ptr, ptr %unreg_field_0922, align 8
  call void @tl_mem_unregister(ptr %field_val923)
  %unreg_field_2924 = getelementptr inbounds %GPT, ptr %call_tmp780, i32 0, i32 2
  %field_val925 = load ptr, ptr %unreg_field_2924, align 8
  call void @tl_mem_unregister(ptr %field_val925)
  %unreg_field_0926 = getelementptr inbounds %Block, ptr %field_val925, i32 0, i32 0
  %field_val927 = load ptr, ptr %unreg_field_0926, align 8
  call void @tl_mem_unregister(ptr %field_val927)
  %unreg_field_0928 = getelementptr inbounds %LayerNorm, ptr %field_val927, i32 0, i32 0
  %field_val929 = load ptr, ptr %unreg_field_0928, align 8
  call void @tl_mem_unregister(ptr %field_val929)
  %unreg_field_1930 = getelementptr inbounds %LayerNorm, ptr %field_val927, i32 0, i32 1
  %field_val931 = load ptr, ptr %unreg_field_1930, align 8
  call void @tl_mem_unregister(ptr %field_val931)
  %unreg_field_1932 = getelementptr inbounds %Block, ptr %field_val925, i32 0, i32 1
  %field_val933 = load ptr, ptr %unreg_field_1932, align 8
  call void @tl_mem_unregister(ptr %field_val933)
  %unreg_field_0934 = getelementptr inbounds %CausalSelfAttention, ptr %field_val933, i32 0, i32 0
  %field_val935 = load ptr, ptr %unreg_field_0934, align 8
  call void @tl_mem_unregister(ptr %field_val935)
  %unreg_field_0936 = getelementptr inbounds %Linear, ptr %field_val935, i32 0, i32 0
  %field_val937 = load ptr, ptr %unreg_field_0936, align 8
  call void @tl_mem_unregister(ptr %field_val937)
  %unreg_field_1938 = getelementptr inbounds %Linear, ptr %field_val935, i32 0, i32 1
  %field_val939 = load ptr, ptr %unreg_field_1938, align 8
  call void @tl_mem_unregister(ptr %field_val939)
  %unreg_field_1940 = getelementptr inbounds %CausalSelfAttention, ptr %field_val933, i32 0, i32 1
  %field_val941 = load ptr, ptr %unreg_field_1940, align 8
  call void @tl_mem_unregister(ptr %field_val941)
  %unreg_field_0942 = getelementptr inbounds %Linear, ptr %field_val941, i32 0, i32 0
  %field_val943 = load ptr, ptr %unreg_field_0942, align 8
  call void @tl_mem_unregister(ptr %field_val943)
  %unreg_field_1944 = getelementptr inbounds %Linear, ptr %field_val941, i32 0, i32 1
  %field_val945 = load ptr, ptr %unreg_field_1944, align 8
  call void @tl_mem_unregister(ptr %field_val945)
  %unreg_field_2946 = getelementptr inbounds %CausalSelfAttention, ptr %field_val933, i32 0, i32 2
  %field_val947 = load ptr, ptr %unreg_field_2946, align 8
  call void @tl_mem_unregister(ptr %field_val947)
  %unreg_field_0948 = getelementptr inbounds %Linear, ptr %field_val947, i32 0, i32 0
  %field_val949 = load ptr, ptr %unreg_field_0948, align 8
  call void @tl_mem_unregister(ptr %field_val949)
  %unreg_field_1950 = getelementptr inbounds %Linear, ptr %field_val947, i32 0, i32 1
  %field_val951 = load ptr, ptr %unreg_field_1950, align 8
  call void @tl_mem_unregister(ptr %field_val951)
  %unreg_field_3952 = getelementptr inbounds %CausalSelfAttention, ptr %field_val933, i32 0, i32 3
  %field_val953 = load ptr, ptr %unreg_field_3952, align 8
  call void @tl_mem_unregister(ptr %field_val953)
  %unreg_field_0954 = getelementptr inbounds %Linear, ptr %field_val953, i32 0, i32 0
  %field_val955 = load ptr, ptr %unreg_field_0954, align 8
  call void @tl_mem_unregister(ptr %field_val955)
  %unreg_field_1956 = getelementptr inbounds %Linear, ptr %field_val953, i32 0, i32 1
  %field_val957 = load ptr, ptr %unreg_field_1956, align 8
  call void @tl_mem_unregister(ptr %field_val957)
  %unreg_field_2958 = getelementptr inbounds %Block, ptr %field_val925, i32 0, i32 2
  %field_val959 = load ptr, ptr %unreg_field_2958, align 8
  call void @tl_mem_unregister(ptr %field_val959)
  %unreg_field_0960 = getelementptr inbounds %LayerNorm, ptr %field_val959, i32 0, i32 0
  %field_val961 = load ptr, ptr %unreg_field_0960, align 8
  call void @tl_mem_unregister(ptr %field_val961)
  %unreg_field_1962 = getelementptr inbounds %LayerNorm, ptr %field_val959, i32 0, i32 1
  %field_val963 = load ptr, ptr %unreg_field_1962, align 8
  call void @tl_mem_unregister(ptr %field_val963)
  %unreg_field_3964 = getelementptr inbounds %Block, ptr %field_val925, i32 0, i32 3
  %field_val965 = load ptr, ptr %unreg_field_3964, align 8
  call void @tl_mem_unregister(ptr %field_val965)
  %unreg_field_0966 = getelementptr inbounds %MLP, ptr %field_val965, i32 0, i32 0
  %field_val967 = load ptr, ptr %unreg_field_0966, align 8
  call void @tl_mem_unregister(ptr %field_val967)
  %unreg_field_0968 = getelementptr inbounds %Linear, ptr %field_val967, i32 0, i32 0
  %field_val969 = load ptr, ptr %unreg_field_0968, align 8
  call void @tl_mem_unregister(ptr %field_val969)
  %unreg_field_1970 = getelementptr inbounds %Linear, ptr %field_val967, i32 0, i32 1
  %field_val971 = load ptr, ptr %unreg_field_1970, align 8
  call void @tl_mem_unregister(ptr %field_val971)
  %unreg_field_1972 = getelementptr inbounds %MLP, ptr %field_val965, i32 0, i32 1
  %field_val973 = load ptr, ptr %unreg_field_1972, align 8
  call void @tl_mem_unregister(ptr %field_val973)
  %unreg_field_0974 = getelementptr inbounds %Linear, ptr %field_val973, i32 0, i32 0
  %field_val975 = load ptr, ptr %unreg_field_0974, align 8
  call void @tl_mem_unregister(ptr %field_val975)
  %unreg_field_1976 = getelementptr inbounds %Linear, ptr %field_val973, i32 0, i32 1
  %field_val977 = load ptr, ptr %unreg_field_1976, align 8
  call void @tl_mem_unregister(ptr %field_val977)
  %unreg_field_3978 = getelementptr inbounds %GPT, ptr %call_tmp780, i32 0, i32 3
  %field_val979 = load ptr, ptr %unreg_field_3978, align 8
  call void @tl_mem_unregister(ptr %field_val979)
  %unreg_field_0980 = getelementptr inbounds %Block, ptr %field_val979, i32 0, i32 0
  %field_val981 = load ptr, ptr %unreg_field_0980, align 8
  call void @tl_mem_unregister(ptr %field_val981)
  %unreg_field_0982 = getelementptr inbounds %LayerNorm, ptr %field_val981, i32 0, i32 0
  %field_val983 = load ptr, ptr %unreg_field_0982, align 8
  call void @tl_mem_unregister(ptr %field_val983)
  %unreg_field_1984 = getelementptr inbounds %LayerNorm, ptr %field_val981, i32 0, i32 1
  %field_val985 = load ptr, ptr %unreg_field_1984, align 8
  call void @tl_mem_unregister(ptr %field_val985)
  %unreg_field_1986 = getelementptr inbounds %Block, ptr %field_val979, i32 0, i32 1
  %field_val987 = load ptr, ptr %unreg_field_1986, align 8
  call void @tl_mem_unregister(ptr %field_val987)
  %unreg_field_0988 = getelementptr inbounds %CausalSelfAttention, ptr %field_val987, i32 0, i32 0
  %field_val989 = load ptr, ptr %unreg_field_0988, align 8
  call void @tl_mem_unregister(ptr %field_val989)
  %unreg_field_0990 = getelementptr inbounds %Linear, ptr %field_val989, i32 0, i32 0
  %field_val991 = load ptr, ptr %unreg_field_0990, align 8
  call void @tl_mem_unregister(ptr %field_val991)
  %unreg_field_1992 = getelementptr inbounds %Linear, ptr %field_val989, i32 0, i32 1
  %field_val993 = load ptr, ptr %unreg_field_1992, align 8
  call void @tl_mem_unregister(ptr %field_val993)
  %unreg_field_1994 = getelementptr inbounds %CausalSelfAttention, ptr %field_val987, i32 0, i32 1
  %field_val995 = load ptr, ptr %unreg_field_1994, align 8
  call void @tl_mem_unregister(ptr %field_val995)
  %unreg_field_0996 = getelementptr inbounds %Linear, ptr %field_val995, i32 0, i32 0
  %field_val997 = load ptr, ptr %unreg_field_0996, align 8
  call void @tl_mem_unregister(ptr %field_val997)
  %unreg_field_1998 = getelementptr inbounds %Linear, ptr %field_val995, i32 0, i32 1
  %field_val999 = load ptr, ptr %unreg_field_1998, align 8
  call void @tl_mem_unregister(ptr %field_val999)
  %unreg_field_21000 = getelementptr inbounds %CausalSelfAttention, ptr %field_val987, i32 0, i32 2
  %field_val1001 = load ptr, ptr %unreg_field_21000, align 8
  call void @tl_mem_unregister(ptr %field_val1001)
  %unreg_field_01002 = getelementptr inbounds %Linear, ptr %field_val1001, i32 0, i32 0
  %field_val1003 = load ptr, ptr %unreg_field_01002, align 8
  call void @tl_mem_unregister(ptr %field_val1003)
  %unreg_field_11004 = getelementptr inbounds %Linear, ptr %field_val1001, i32 0, i32 1
  %field_val1005 = load ptr, ptr %unreg_field_11004, align 8
  call void @tl_mem_unregister(ptr %field_val1005)
  %unreg_field_31006 = getelementptr inbounds %CausalSelfAttention, ptr %field_val987, i32 0, i32 3
  %field_val1007 = load ptr, ptr %unreg_field_31006, align 8
  call void @tl_mem_unregister(ptr %field_val1007)
  %unreg_field_01008 = getelementptr inbounds %Linear, ptr %field_val1007, i32 0, i32 0
  %field_val1009 = load ptr, ptr %unreg_field_01008, align 8
  call void @tl_mem_unregister(ptr %field_val1009)
  %unreg_field_11010 = getelementptr inbounds %Linear, ptr %field_val1007, i32 0, i32 1
  %field_val1011 = load ptr, ptr %unreg_field_11010, align 8
  call void @tl_mem_unregister(ptr %field_val1011)
  %unreg_field_21012 = getelementptr inbounds %Block, ptr %field_val979, i32 0, i32 2
  %field_val1013 = load ptr, ptr %unreg_field_21012, align 8
  call void @tl_mem_unregister(ptr %field_val1013)
  %unreg_field_01014 = getelementptr inbounds %LayerNorm, ptr %field_val1013, i32 0, i32 0
  %field_val1015 = load ptr, ptr %unreg_field_01014, align 8
  call void @tl_mem_unregister(ptr %field_val1015)
  %unreg_field_11016 = getelementptr inbounds %LayerNorm, ptr %field_val1013, i32 0, i32 1
  %field_val1017 = load ptr, ptr %unreg_field_11016, align 8
  call void @tl_mem_unregister(ptr %field_val1017)
  %unreg_field_31018 = getelementptr inbounds %Block, ptr %field_val979, i32 0, i32 3
  %field_val1019 = load ptr, ptr %unreg_field_31018, align 8
  call void @tl_mem_unregister(ptr %field_val1019)
  %unreg_field_01020 = getelementptr inbounds %MLP, ptr %field_val1019, i32 0, i32 0
  %field_val1021 = load ptr, ptr %unreg_field_01020, align 8
  call void @tl_mem_unregister(ptr %field_val1021)
  %unreg_field_01022 = getelementptr inbounds %Linear, ptr %field_val1021, i32 0, i32 0
  %field_val1023 = load ptr, ptr %unreg_field_01022, align 8
  call void @tl_mem_unregister(ptr %field_val1023)
  %unreg_field_11024 = getelementptr inbounds %Linear, ptr %field_val1021, i32 0, i32 1
  %field_val1025 = load ptr, ptr %unreg_field_11024, align 8
  call void @tl_mem_unregister(ptr %field_val1025)
  %unreg_field_11026 = getelementptr inbounds %MLP, ptr %field_val1019, i32 0, i32 1
  %field_val1027 = load ptr, ptr %unreg_field_11026, align 8
  call void @tl_mem_unregister(ptr %field_val1027)
  %unreg_field_01028 = getelementptr inbounds %Linear, ptr %field_val1027, i32 0, i32 0
  %field_val1029 = load ptr, ptr %unreg_field_01028, align 8
  call void @tl_mem_unregister(ptr %field_val1029)
  %unreg_field_11030 = getelementptr inbounds %Linear, ptr %field_val1027, i32 0, i32 1
  %field_val1031 = load ptr, ptr %unreg_field_11030, align 8
  call void @tl_mem_unregister(ptr %field_val1031)
  %unreg_field_41032 = getelementptr inbounds %GPT, ptr %call_tmp780, i32 0, i32 4
  %field_val1033 = load ptr, ptr %unreg_field_41032, align 8
  call void @tl_mem_unregister(ptr %field_val1033)
  %unreg_field_01034 = getelementptr inbounds %LayerNorm, ptr %field_val1033, i32 0, i32 0
  %field_val1035 = load ptr, ptr %unreg_field_01034, align 8
  call void @tl_mem_unregister(ptr %field_val1035)
  %unreg_field_11036 = getelementptr inbounds %LayerNorm, ptr %field_val1033, i32 0, i32 1
  %field_val1037 = load ptr, ptr %unreg_field_11036, align 8
  call void @tl_mem_unregister(ptr %field_val1037)
  %unreg_field_51038 = getelementptr inbounds %GPT, ptr %call_tmp780, i32 0, i32 5
  %field_val1039 = load ptr, ptr %unreg_field_51038, align 8
  call void @tl_mem_unregister(ptr %field_val1039)
  %unreg_field_01040 = getelementptr inbounds %Linear, ptr %field_val1039, i32 0, i32 0
  %field_val1041 = load ptr, ptr %unreg_field_01040, align 8
  call void @tl_mem_unregister(ptr %field_val1041)
  %unreg_field_11042 = getelementptr inbounds %Linear, ptr %field_val1039, i32 0, i32 1
  %field_val1043 = load ptr, ptr %unreg_field_11042, align 8
  call void @tl_mem_unregister(ptr %field_val1043)
  call void @tl_mem_unregister(ptr %call_tmp780)
  store ptr %call_tmp780, ptr %model, align 8
  %model1044 = load ptr, ptr %model, align 8
  %call_tmp1045 = call ptr @train_step(ptr %model1044, float 0x3FA99999A0000000, i64 0, i64 0)
  call void @tl_mem_unregister(ptr %call_tmp1045)
  %old_struct_to_free1046 = load ptr, ptr %model, align 8
  %is_not_null1047 = icmp ne ptr %old_struct_to_free1046, null
  %are_diff1048 = icmp ne ptr %old_struct_to_free1046, %call_tmp1045
  %can_free_11049 = and i1 %is_not_null1047, true
  %can_free1050 = and i1 %can_free_11049, %are_diff1048
  br i1 %can_free1050, label %free_struct1051, label %continue_after_free1052

free_struct1051:                                  ; preds = %continue_after_free787
  %field_gep1053 = getelementptr inbounds %GPT, ptr %old_struct_to_free1046, i32 0, i32 0
  %field_load1054 = load ptr, ptr %field_gep1053, align 8
  %field_gep1055 = getelementptr inbounds %Embedding, ptr %field_load1054, i32 0, i32 0
  %field_load1056 = load ptr, ptr %field_gep1055, align 8
  call void @tl_tensor_free(ptr %field_load1056)
  %field_gep1057 = getelementptr inbounds %GPT, ptr %old_struct_to_free1046, i32 0, i32 1
  %field_load1058 = load ptr, ptr %field_gep1057, align 8
  %field_gep1059 = getelementptr inbounds %Embedding, ptr %field_load1058, i32 0, i32 0
  %field_load1060 = load ptr, ptr %field_gep1059, align 8
  call void @tl_tensor_free(ptr %field_load1060)
  %field_gep1061 = getelementptr inbounds %GPT, ptr %old_struct_to_free1046, i32 0, i32 2
  %field_load1062 = load ptr, ptr %field_gep1061, align 8
  %field_gep1063 = getelementptr inbounds %Block, ptr %field_load1062, i32 0, i32 0
  %field_load1064 = load ptr, ptr %field_gep1063, align 8
  %field_gep1065 = getelementptr inbounds %LayerNorm, ptr %field_load1064, i32 0, i32 0
  %field_load1066 = load ptr, ptr %field_gep1065, align 8
  call void @tl_tensor_free(ptr %field_load1066)
  %field_gep1067 = getelementptr inbounds %LayerNorm, ptr %field_load1064, i32 0, i32 1
  %field_load1068 = load ptr, ptr %field_gep1067, align 8
  call void @tl_tensor_free(ptr %field_load1068)
  %field_gep1069 = getelementptr inbounds %Block, ptr %field_load1062, i32 0, i32 1
  %field_load1070 = load ptr, ptr %field_gep1069, align 8
  %field_gep1071 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1070, i32 0, i32 0
  %field_load1072 = load ptr, ptr %field_gep1071, align 8
  %field_gep1073 = getelementptr inbounds %Linear, ptr %field_load1072, i32 0, i32 0
  %field_load1074 = load ptr, ptr %field_gep1073, align 8
  call void @tl_tensor_free(ptr %field_load1074)
  %field_gep1075 = getelementptr inbounds %Linear, ptr %field_load1072, i32 0, i32 1
  %field_load1076 = load ptr, ptr %field_gep1075, align 8
  call void @tl_tensor_free(ptr %field_load1076)
  %field_gep1077 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1070, i32 0, i32 1
  %field_load1078 = load ptr, ptr %field_gep1077, align 8
  %field_gep1079 = getelementptr inbounds %Linear, ptr %field_load1078, i32 0, i32 0
  %field_load1080 = load ptr, ptr %field_gep1079, align 8
  call void @tl_tensor_free(ptr %field_load1080)
  %field_gep1081 = getelementptr inbounds %Linear, ptr %field_load1078, i32 0, i32 1
  %field_load1082 = load ptr, ptr %field_gep1081, align 8
  call void @tl_tensor_free(ptr %field_load1082)
  %field_gep1083 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1070, i32 0, i32 2
  %field_load1084 = load ptr, ptr %field_gep1083, align 8
  %field_gep1085 = getelementptr inbounds %Linear, ptr %field_load1084, i32 0, i32 0
  %field_load1086 = load ptr, ptr %field_gep1085, align 8
  call void @tl_tensor_free(ptr %field_load1086)
  %field_gep1087 = getelementptr inbounds %Linear, ptr %field_load1084, i32 0, i32 1
  %field_load1088 = load ptr, ptr %field_gep1087, align 8
  call void @tl_tensor_free(ptr %field_load1088)
  %field_gep1089 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1070, i32 0, i32 3
  %field_load1090 = load ptr, ptr %field_gep1089, align 8
  %field_gep1091 = getelementptr inbounds %Linear, ptr %field_load1090, i32 0, i32 0
  %field_load1092 = load ptr, ptr %field_gep1091, align 8
  call void @tl_tensor_free(ptr %field_load1092)
  %field_gep1093 = getelementptr inbounds %Linear, ptr %field_load1090, i32 0, i32 1
  %field_load1094 = load ptr, ptr %field_gep1093, align 8
  call void @tl_tensor_free(ptr %field_load1094)
  %field_gep1095 = getelementptr inbounds %Block, ptr %field_load1062, i32 0, i32 2
  %field_load1096 = load ptr, ptr %field_gep1095, align 8
  %field_gep1097 = getelementptr inbounds %LayerNorm, ptr %field_load1096, i32 0, i32 0
  %field_load1098 = load ptr, ptr %field_gep1097, align 8
  call void @tl_tensor_free(ptr %field_load1098)
  %field_gep1099 = getelementptr inbounds %LayerNorm, ptr %field_load1096, i32 0, i32 1
  %field_load1100 = load ptr, ptr %field_gep1099, align 8
  call void @tl_tensor_free(ptr %field_load1100)
  %field_gep1101 = getelementptr inbounds %Block, ptr %field_load1062, i32 0, i32 3
  %field_load1102 = load ptr, ptr %field_gep1101, align 8
  %field_gep1103 = getelementptr inbounds %MLP, ptr %field_load1102, i32 0, i32 0
  %field_load1104 = load ptr, ptr %field_gep1103, align 8
  %field_gep1105 = getelementptr inbounds %Linear, ptr %field_load1104, i32 0, i32 0
  %field_load1106 = load ptr, ptr %field_gep1105, align 8
  call void @tl_tensor_free(ptr %field_load1106)
  %field_gep1107 = getelementptr inbounds %Linear, ptr %field_load1104, i32 0, i32 1
  %field_load1108 = load ptr, ptr %field_gep1107, align 8
  call void @tl_tensor_free(ptr %field_load1108)
  %field_gep1109 = getelementptr inbounds %MLP, ptr %field_load1102, i32 0, i32 1
  %field_load1110 = load ptr, ptr %field_gep1109, align 8
  %field_gep1111 = getelementptr inbounds %Linear, ptr %field_load1110, i32 0, i32 0
  %field_load1112 = load ptr, ptr %field_gep1111, align 8
  call void @tl_tensor_free(ptr %field_load1112)
  %field_gep1113 = getelementptr inbounds %Linear, ptr %field_load1110, i32 0, i32 1
  %field_load1114 = load ptr, ptr %field_gep1113, align 8
  call void @tl_tensor_free(ptr %field_load1114)
  %field_gep1115 = getelementptr inbounds %GPT, ptr %old_struct_to_free1046, i32 0, i32 3
  %field_load1116 = load ptr, ptr %field_gep1115, align 8
  %field_gep1117 = getelementptr inbounds %Block, ptr %field_load1116, i32 0, i32 0
  %field_load1118 = load ptr, ptr %field_gep1117, align 8
  %field_gep1119 = getelementptr inbounds %LayerNorm, ptr %field_load1118, i32 0, i32 0
  %field_load1120 = load ptr, ptr %field_gep1119, align 8
  call void @tl_tensor_free(ptr %field_load1120)
  %field_gep1121 = getelementptr inbounds %LayerNorm, ptr %field_load1118, i32 0, i32 1
  %field_load1122 = load ptr, ptr %field_gep1121, align 8
  call void @tl_tensor_free(ptr %field_load1122)
  %field_gep1123 = getelementptr inbounds %Block, ptr %field_load1116, i32 0, i32 1
  %field_load1124 = load ptr, ptr %field_gep1123, align 8
  %field_gep1125 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1124, i32 0, i32 0
  %field_load1126 = load ptr, ptr %field_gep1125, align 8
  %field_gep1127 = getelementptr inbounds %Linear, ptr %field_load1126, i32 0, i32 0
  %field_load1128 = load ptr, ptr %field_gep1127, align 8
  call void @tl_tensor_free(ptr %field_load1128)
  %field_gep1129 = getelementptr inbounds %Linear, ptr %field_load1126, i32 0, i32 1
  %field_load1130 = load ptr, ptr %field_gep1129, align 8
  call void @tl_tensor_free(ptr %field_load1130)
  %field_gep1131 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1124, i32 0, i32 1
  %field_load1132 = load ptr, ptr %field_gep1131, align 8
  %field_gep1133 = getelementptr inbounds %Linear, ptr %field_load1132, i32 0, i32 0
  %field_load1134 = load ptr, ptr %field_gep1133, align 8
  call void @tl_tensor_free(ptr %field_load1134)
  %field_gep1135 = getelementptr inbounds %Linear, ptr %field_load1132, i32 0, i32 1
  %field_load1136 = load ptr, ptr %field_gep1135, align 8
  call void @tl_tensor_free(ptr %field_load1136)
  %field_gep1137 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1124, i32 0, i32 2
  %field_load1138 = load ptr, ptr %field_gep1137, align 8
  %field_gep1139 = getelementptr inbounds %Linear, ptr %field_load1138, i32 0, i32 0
  %field_load1140 = load ptr, ptr %field_gep1139, align 8
  call void @tl_tensor_free(ptr %field_load1140)
  %field_gep1141 = getelementptr inbounds %Linear, ptr %field_load1138, i32 0, i32 1
  %field_load1142 = load ptr, ptr %field_gep1141, align 8
  call void @tl_tensor_free(ptr %field_load1142)
  %field_gep1143 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1124, i32 0, i32 3
  %field_load1144 = load ptr, ptr %field_gep1143, align 8
  %field_gep1145 = getelementptr inbounds %Linear, ptr %field_load1144, i32 0, i32 0
  %field_load1146 = load ptr, ptr %field_gep1145, align 8
  call void @tl_tensor_free(ptr %field_load1146)
  %field_gep1147 = getelementptr inbounds %Linear, ptr %field_load1144, i32 0, i32 1
  %field_load1148 = load ptr, ptr %field_gep1147, align 8
  call void @tl_tensor_free(ptr %field_load1148)
  %field_gep1149 = getelementptr inbounds %Block, ptr %field_load1116, i32 0, i32 2
  %field_load1150 = load ptr, ptr %field_gep1149, align 8
  %field_gep1151 = getelementptr inbounds %LayerNorm, ptr %field_load1150, i32 0, i32 0
  %field_load1152 = load ptr, ptr %field_gep1151, align 8
  call void @tl_tensor_free(ptr %field_load1152)
  %field_gep1153 = getelementptr inbounds %LayerNorm, ptr %field_load1150, i32 0, i32 1
  %field_load1154 = load ptr, ptr %field_gep1153, align 8
  call void @tl_tensor_free(ptr %field_load1154)
  %field_gep1155 = getelementptr inbounds %Block, ptr %field_load1116, i32 0, i32 3
  %field_load1156 = load ptr, ptr %field_gep1155, align 8
  %field_gep1157 = getelementptr inbounds %MLP, ptr %field_load1156, i32 0, i32 0
  %field_load1158 = load ptr, ptr %field_gep1157, align 8
  %field_gep1159 = getelementptr inbounds %Linear, ptr %field_load1158, i32 0, i32 0
  %field_load1160 = load ptr, ptr %field_gep1159, align 8
  call void @tl_tensor_free(ptr %field_load1160)
  %field_gep1161 = getelementptr inbounds %Linear, ptr %field_load1158, i32 0, i32 1
  %field_load1162 = load ptr, ptr %field_gep1161, align 8
  call void @tl_tensor_free(ptr %field_load1162)
  %field_gep1163 = getelementptr inbounds %MLP, ptr %field_load1156, i32 0, i32 1
  %field_load1164 = load ptr, ptr %field_gep1163, align 8
  %field_gep1165 = getelementptr inbounds %Linear, ptr %field_load1164, i32 0, i32 0
  %field_load1166 = load ptr, ptr %field_gep1165, align 8
  call void @tl_tensor_free(ptr %field_load1166)
  %field_gep1167 = getelementptr inbounds %Linear, ptr %field_load1164, i32 0, i32 1
  %field_load1168 = load ptr, ptr %field_gep1167, align 8
  call void @tl_tensor_free(ptr %field_load1168)
  %field_gep1169 = getelementptr inbounds %GPT, ptr %old_struct_to_free1046, i32 0, i32 4
  %field_load1170 = load ptr, ptr %field_gep1169, align 8
  %field_gep1171 = getelementptr inbounds %LayerNorm, ptr %field_load1170, i32 0, i32 0
  %field_load1172 = load ptr, ptr %field_gep1171, align 8
  call void @tl_tensor_free(ptr %field_load1172)
  %field_gep1173 = getelementptr inbounds %LayerNorm, ptr %field_load1170, i32 0, i32 1
  %field_load1174 = load ptr, ptr %field_gep1173, align 8
  call void @tl_tensor_free(ptr %field_load1174)
  %field_gep1175 = getelementptr inbounds %GPT, ptr %old_struct_to_free1046, i32 0, i32 5
  %field_load1176 = load ptr, ptr %field_gep1175, align 8
  %field_gep1177 = getelementptr inbounds %Linear, ptr %field_load1176, i32 0, i32 0
  %field_load1178 = load ptr, ptr %field_gep1177, align 8
  call void @tl_tensor_free(ptr %field_load1178)
  %field_gep1179 = getelementptr inbounds %Linear, ptr %field_load1176, i32 0, i32 1
  %field_load1180 = load ptr, ptr %field_gep1179, align 8
  call void @tl_tensor_free(ptr %field_load1180)
  call void @tl_mem_unregister(ptr %old_struct_to_free1046)
  br label %continue_after_free1052

continue_after_free1052:                          ; preds = %free_struct1051, %continue_after_free787
  call void @tl_mem_unregister(ptr %call_tmp1045)
  %unreg_field_01181 = getelementptr inbounds %GPT, ptr %call_tmp1045, i32 0, i32 0
  %field_val1182 = load ptr, ptr %unreg_field_01181, align 8
  call void @tl_mem_unregister(ptr %field_val1182)
  %unreg_field_01183 = getelementptr inbounds %Embedding, ptr %field_val1182, i32 0, i32 0
  %field_val1184 = load ptr, ptr %unreg_field_01183, align 8
  call void @tl_mem_unregister(ptr %field_val1184)
  %unreg_field_11185 = getelementptr inbounds %GPT, ptr %call_tmp1045, i32 0, i32 1
  %field_val1186 = load ptr, ptr %unreg_field_11185, align 8
  call void @tl_mem_unregister(ptr %field_val1186)
  %unreg_field_01187 = getelementptr inbounds %Embedding, ptr %field_val1186, i32 0, i32 0
  %field_val1188 = load ptr, ptr %unreg_field_01187, align 8
  call void @tl_mem_unregister(ptr %field_val1188)
  %unreg_field_21189 = getelementptr inbounds %GPT, ptr %call_tmp1045, i32 0, i32 2
  %field_val1190 = load ptr, ptr %unreg_field_21189, align 8
  call void @tl_mem_unregister(ptr %field_val1190)
  %unreg_field_01191 = getelementptr inbounds %Block, ptr %field_val1190, i32 0, i32 0
  %field_val1192 = load ptr, ptr %unreg_field_01191, align 8
  call void @tl_mem_unregister(ptr %field_val1192)
  %unreg_field_01193 = getelementptr inbounds %LayerNorm, ptr %field_val1192, i32 0, i32 0
  %field_val1194 = load ptr, ptr %unreg_field_01193, align 8
  call void @tl_mem_unregister(ptr %field_val1194)
  %unreg_field_11195 = getelementptr inbounds %LayerNorm, ptr %field_val1192, i32 0, i32 1
  %field_val1196 = load ptr, ptr %unreg_field_11195, align 8
  call void @tl_mem_unregister(ptr %field_val1196)
  %unreg_field_11197 = getelementptr inbounds %Block, ptr %field_val1190, i32 0, i32 1
  %field_val1198 = load ptr, ptr %unreg_field_11197, align 8
  call void @tl_mem_unregister(ptr %field_val1198)
  %unreg_field_01199 = getelementptr inbounds %CausalSelfAttention, ptr %field_val1198, i32 0, i32 0
  %field_val1200 = load ptr, ptr %unreg_field_01199, align 8
  call void @tl_mem_unregister(ptr %field_val1200)
  %unreg_field_01201 = getelementptr inbounds %Linear, ptr %field_val1200, i32 0, i32 0
  %field_val1202 = load ptr, ptr %unreg_field_01201, align 8
  call void @tl_mem_unregister(ptr %field_val1202)
  %unreg_field_11203 = getelementptr inbounds %Linear, ptr %field_val1200, i32 0, i32 1
  %field_val1204 = load ptr, ptr %unreg_field_11203, align 8
  call void @tl_mem_unregister(ptr %field_val1204)
  %unreg_field_11205 = getelementptr inbounds %CausalSelfAttention, ptr %field_val1198, i32 0, i32 1
  %field_val1206 = load ptr, ptr %unreg_field_11205, align 8
  call void @tl_mem_unregister(ptr %field_val1206)
  %unreg_field_01207 = getelementptr inbounds %Linear, ptr %field_val1206, i32 0, i32 0
  %field_val1208 = load ptr, ptr %unreg_field_01207, align 8
  call void @tl_mem_unregister(ptr %field_val1208)
  %unreg_field_11209 = getelementptr inbounds %Linear, ptr %field_val1206, i32 0, i32 1
  %field_val1210 = load ptr, ptr %unreg_field_11209, align 8
  call void @tl_mem_unregister(ptr %field_val1210)
  %unreg_field_21211 = getelementptr inbounds %CausalSelfAttention, ptr %field_val1198, i32 0, i32 2
  %field_val1212 = load ptr, ptr %unreg_field_21211, align 8
  call void @tl_mem_unregister(ptr %field_val1212)
  %unreg_field_01213 = getelementptr inbounds %Linear, ptr %field_val1212, i32 0, i32 0
  %field_val1214 = load ptr, ptr %unreg_field_01213, align 8
  call void @tl_mem_unregister(ptr %field_val1214)
  %unreg_field_11215 = getelementptr inbounds %Linear, ptr %field_val1212, i32 0, i32 1
  %field_val1216 = load ptr, ptr %unreg_field_11215, align 8
  call void @tl_mem_unregister(ptr %field_val1216)
  %unreg_field_31217 = getelementptr inbounds %CausalSelfAttention, ptr %field_val1198, i32 0, i32 3
  %field_val1218 = load ptr, ptr %unreg_field_31217, align 8
  call void @tl_mem_unregister(ptr %field_val1218)
  %unreg_field_01219 = getelementptr inbounds %Linear, ptr %field_val1218, i32 0, i32 0
  %field_val1220 = load ptr, ptr %unreg_field_01219, align 8
  call void @tl_mem_unregister(ptr %field_val1220)
  %unreg_field_11221 = getelementptr inbounds %Linear, ptr %field_val1218, i32 0, i32 1
  %field_val1222 = load ptr, ptr %unreg_field_11221, align 8
  call void @tl_mem_unregister(ptr %field_val1222)
  %unreg_field_21223 = getelementptr inbounds %Block, ptr %field_val1190, i32 0, i32 2
  %field_val1224 = load ptr, ptr %unreg_field_21223, align 8
  call void @tl_mem_unregister(ptr %field_val1224)
  %unreg_field_01225 = getelementptr inbounds %LayerNorm, ptr %field_val1224, i32 0, i32 0
  %field_val1226 = load ptr, ptr %unreg_field_01225, align 8
  call void @tl_mem_unregister(ptr %field_val1226)
  %unreg_field_11227 = getelementptr inbounds %LayerNorm, ptr %field_val1224, i32 0, i32 1
  %field_val1228 = load ptr, ptr %unreg_field_11227, align 8
  call void @tl_mem_unregister(ptr %field_val1228)
  %unreg_field_31229 = getelementptr inbounds %Block, ptr %field_val1190, i32 0, i32 3
  %field_val1230 = load ptr, ptr %unreg_field_31229, align 8
  call void @tl_mem_unregister(ptr %field_val1230)
  %unreg_field_01231 = getelementptr inbounds %MLP, ptr %field_val1230, i32 0, i32 0
  %field_val1232 = load ptr, ptr %unreg_field_01231, align 8
  call void @tl_mem_unregister(ptr %field_val1232)
  %unreg_field_01233 = getelementptr inbounds %Linear, ptr %field_val1232, i32 0, i32 0
  %field_val1234 = load ptr, ptr %unreg_field_01233, align 8
  call void @tl_mem_unregister(ptr %field_val1234)
  %unreg_field_11235 = getelementptr inbounds %Linear, ptr %field_val1232, i32 0, i32 1
  %field_val1236 = load ptr, ptr %unreg_field_11235, align 8
  call void @tl_mem_unregister(ptr %field_val1236)
  %unreg_field_11237 = getelementptr inbounds %MLP, ptr %field_val1230, i32 0, i32 1
  %field_val1238 = load ptr, ptr %unreg_field_11237, align 8
  call void @tl_mem_unregister(ptr %field_val1238)
  %unreg_field_01239 = getelementptr inbounds %Linear, ptr %field_val1238, i32 0, i32 0
  %field_val1240 = load ptr, ptr %unreg_field_01239, align 8
  call void @tl_mem_unregister(ptr %field_val1240)
  %unreg_field_11241 = getelementptr inbounds %Linear, ptr %field_val1238, i32 0, i32 1
  %field_val1242 = load ptr, ptr %unreg_field_11241, align 8
  call void @tl_mem_unregister(ptr %field_val1242)
  %unreg_field_31243 = getelementptr inbounds %GPT, ptr %call_tmp1045, i32 0, i32 3
  %field_val1244 = load ptr, ptr %unreg_field_31243, align 8
  call void @tl_mem_unregister(ptr %field_val1244)
  %unreg_field_01245 = getelementptr inbounds %Block, ptr %field_val1244, i32 0, i32 0
  %field_val1246 = load ptr, ptr %unreg_field_01245, align 8
  call void @tl_mem_unregister(ptr %field_val1246)
  %unreg_field_01247 = getelementptr inbounds %LayerNorm, ptr %field_val1246, i32 0, i32 0
  %field_val1248 = load ptr, ptr %unreg_field_01247, align 8
  call void @tl_mem_unregister(ptr %field_val1248)
  %unreg_field_11249 = getelementptr inbounds %LayerNorm, ptr %field_val1246, i32 0, i32 1
  %field_val1250 = load ptr, ptr %unreg_field_11249, align 8
  call void @tl_mem_unregister(ptr %field_val1250)
  %unreg_field_11251 = getelementptr inbounds %Block, ptr %field_val1244, i32 0, i32 1
  %field_val1252 = load ptr, ptr %unreg_field_11251, align 8
  call void @tl_mem_unregister(ptr %field_val1252)
  %unreg_field_01253 = getelementptr inbounds %CausalSelfAttention, ptr %field_val1252, i32 0, i32 0
  %field_val1254 = load ptr, ptr %unreg_field_01253, align 8
  call void @tl_mem_unregister(ptr %field_val1254)
  %unreg_field_01255 = getelementptr inbounds %Linear, ptr %field_val1254, i32 0, i32 0
  %field_val1256 = load ptr, ptr %unreg_field_01255, align 8
  call void @tl_mem_unregister(ptr %field_val1256)
  %unreg_field_11257 = getelementptr inbounds %Linear, ptr %field_val1254, i32 0, i32 1
  %field_val1258 = load ptr, ptr %unreg_field_11257, align 8
  call void @tl_mem_unregister(ptr %field_val1258)
  %unreg_field_11259 = getelementptr inbounds %CausalSelfAttention, ptr %field_val1252, i32 0, i32 1
  %field_val1260 = load ptr, ptr %unreg_field_11259, align 8
  call void @tl_mem_unregister(ptr %field_val1260)
  %unreg_field_01261 = getelementptr inbounds %Linear, ptr %field_val1260, i32 0, i32 0
  %field_val1262 = load ptr, ptr %unreg_field_01261, align 8
  call void @tl_mem_unregister(ptr %field_val1262)
  %unreg_field_11263 = getelementptr inbounds %Linear, ptr %field_val1260, i32 0, i32 1
  %field_val1264 = load ptr, ptr %unreg_field_11263, align 8
  call void @tl_mem_unregister(ptr %field_val1264)
  %unreg_field_21265 = getelementptr inbounds %CausalSelfAttention, ptr %field_val1252, i32 0, i32 2
  %field_val1266 = load ptr, ptr %unreg_field_21265, align 8
  call void @tl_mem_unregister(ptr %field_val1266)
  %unreg_field_01267 = getelementptr inbounds %Linear, ptr %field_val1266, i32 0, i32 0
  %field_val1268 = load ptr, ptr %unreg_field_01267, align 8
  call void @tl_mem_unregister(ptr %field_val1268)
  %unreg_field_11269 = getelementptr inbounds %Linear, ptr %field_val1266, i32 0, i32 1
  %field_val1270 = load ptr, ptr %unreg_field_11269, align 8
  call void @tl_mem_unregister(ptr %field_val1270)
  %unreg_field_31271 = getelementptr inbounds %CausalSelfAttention, ptr %field_val1252, i32 0, i32 3
  %field_val1272 = load ptr, ptr %unreg_field_31271, align 8
  call void @tl_mem_unregister(ptr %field_val1272)
  %unreg_field_01273 = getelementptr inbounds %Linear, ptr %field_val1272, i32 0, i32 0
  %field_val1274 = load ptr, ptr %unreg_field_01273, align 8
  call void @tl_mem_unregister(ptr %field_val1274)
  %unreg_field_11275 = getelementptr inbounds %Linear, ptr %field_val1272, i32 0, i32 1
  %field_val1276 = load ptr, ptr %unreg_field_11275, align 8
  call void @tl_mem_unregister(ptr %field_val1276)
  %unreg_field_21277 = getelementptr inbounds %Block, ptr %field_val1244, i32 0, i32 2
  %field_val1278 = load ptr, ptr %unreg_field_21277, align 8
  call void @tl_mem_unregister(ptr %field_val1278)
  %unreg_field_01279 = getelementptr inbounds %LayerNorm, ptr %field_val1278, i32 0, i32 0
  %field_val1280 = load ptr, ptr %unreg_field_01279, align 8
  call void @tl_mem_unregister(ptr %field_val1280)
  %unreg_field_11281 = getelementptr inbounds %LayerNorm, ptr %field_val1278, i32 0, i32 1
  %field_val1282 = load ptr, ptr %unreg_field_11281, align 8
  call void @tl_mem_unregister(ptr %field_val1282)
  %unreg_field_31283 = getelementptr inbounds %Block, ptr %field_val1244, i32 0, i32 3
  %field_val1284 = load ptr, ptr %unreg_field_31283, align 8
  call void @tl_mem_unregister(ptr %field_val1284)
  %unreg_field_01285 = getelementptr inbounds %MLP, ptr %field_val1284, i32 0, i32 0
  %field_val1286 = load ptr, ptr %unreg_field_01285, align 8
  call void @tl_mem_unregister(ptr %field_val1286)
  %unreg_field_01287 = getelementptr inbounds %Linear, ptr %field_val1286, i32 0, i32 0
  %field_val1288 = load ptr, ptr %unreg_field_01287, align 8
  call void @tl_mem_unregister(ptr %field_val1288)
  %unreg_field_11289 = getelementptr inbounds %Linear, ptr %field_val1286, i32 0, i32 1
  %field_val1290 = load ptr, ptr %unreg_field_11289, align 8
  call void @tl_mem_unregister(ptr %field_val1290)
  %unreg_field_11291 = getelementptr inbounds %MLP, ptr %field_val1284, i32 0, i32 1
  %field_val1292 = load ptr, ptr %unreg_field_11291, align 8
  call void @tl_mem_unregister(ptr %field_val1292)
  %unreg_field_01293 = getelementptr inbounds %Linear, ptr %field_val1292, i32 0, i32 0
  %field_val1294 = load ptr, ptr %unreg_field_01293, align 8
  call void @tl_mem_unregister(ptr %field_val1294)
  %unreg_field_11295 = getelementptr inbounds %Linear, ptr %field_val1292, i32 0, i32 1
  %field_val1296 = load ptr, ptr %unreg_field_11295, align 8
  call void @tl_mem_unregister(ptr %field_val1296)
  %unreg_field_41297 = getelementptr inbounds %GPT, ptr %call_tmp1045, i32 0, i32 4
  %field_val1298 = load ptr, ptr %unreg_field_41297, align 8
  call void @tl_mem_unregister(ptr %field_val1298)
  %unreg_field_01299 = getelementptr inbounds %LayerNorm, ptr %field_val1298, i32 0, i32 0
  %field_val1300 = load ptr, ptr %unreg_field_01299, align 8
  call void @tl_mem_unregister(ptr %field_val1300)
  %unreg_field_11301 = getelementptr inbounds %LayerNorm, ptr %field_val1298, i32 0, i32 1
  %field_val1302 = load ptr, ptr %unreg_field_11301, align 8
  call void @tl_mem_unregister(ptr %field_val1302)
  %unreg_field_51303 = getelementptr inbounds %GPT, ptr %call_tmp1045, i32 0, i32 5
  %field_val1304 = load ptr, ptr %unreg_field_51303, align 8
  call void @tl_mem_unregister(ptr %field_val1304)
  %unreg_field_01305 = getelementptr inbounds %Linear, ptr %field_val1304, i32 0, i32 0
  %field_val1306 = load ptr, ptr %unreg_field_01305, align 8
  call void @tl_mem_unregister(ptr %field_val1306)
  %unreg_field_11307 = getelementptr inbounds %Linear, ptr %field_val1304, i32 0, i32 1
  %field_val1308 = load ptr, ptr %unreg_field_11307, align 8
  call void @tl_mem_unregister(ptr %field_val1308)
  call void @tl_mem_unregister(ptr %call_tmp1045)
  store ptr %call_tmp1045, ptr %model, align 8
  %epoch1309 = load i64, ptr %epoch, align 8
  %epoch1310 = load i64, ptr %epoch, align 8
  %divtmp = sdiv i64 %epoch1310, 20
  %multmp = mul i64 %divtmp, 20
  %subtmp = sub i64 %epoch1309, %multmp
  %eqtmp = icmp eq i64 %subtmp, 0
  br i1 %eqtmp, label %then, label %else

then:                                             ; preds = %continue_after_free1052
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.112)
  %epoch1311 = load i64, ptr %epoch, align 8
  call void @tl_print_i64(i64 %epoch1311)
  call void @tl_mem_exit_scope()
  br label %merge

else:                                             ; preds = %continue_after_free1052
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else, %then
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx, 1
  br label %for_header
}
