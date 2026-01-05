; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

%Linear = type { ptr, ptr }
%Embedding = type { ptr }
%LayerNorm = type { ptr, ptr }
%CausalSelfAttention = type { ptr, ptr, ptr, ptr }
%MLP = type { ptr, ptr }
%Block = type { ptr, ptr, ptr, ptr }
%GPT = type { ptr, ptr, ptr, ptr, ptr }

@str_literal = private unnamed_addr constant [36 x i8] c"Initializing Model for Inference...\00", align 1
@str_literal.104 = private unnamed_addr constant [46 x i8] c"Initializing Runtime: Metal backend selected.\00", align 1
@str_literal.105 = private unnamed_addr constant [22 x i8] c"Loading Parameters...\00", align 1
@key_str = private unnamed_addr constant [4 x i8] c"w.w\00", align 1
@key_str.106 = private unnamed_addr constant [5 x i8] c"wp.w\00", align 1
@key_str.107 = private unnamed_addr constant [7 x i8] c"b.l1.w\00", align 1
@key_str.108 = private unnamed_addr constant [7 x i8] c"b.l1.b\00", align 1
@key_str.109 = private unnamed_addr constant [13 x i8] c"b.a.q_proj.W\00", align 1
@key_str.110 = private unnamed_addr constant [13 x i8] c"b.a.q_proj.b\00", align 1
@key_str.111 = private unnamed_addr constant [13 x i8] c"b.a.k_proj.W\00", align 1
@key_str.112 = private unnamed_addr constant [13 x i8] c"b.a.k_proj.b\00", align 1
@key_str.113 = private unnamed_addr constant [13 x i8] c"b.a.v_proj.W\00", align 1
@key_str.114 = private unnamed_addr constant [13 x i8] c"b.a.v_proj.b\00", align 1
@key_str.115 = private unnamed_addr constant [13 x i8] c"b.a.p_proj.W\00", align 1
@key_str.116 = private unnamed_addr constant [13 x i8] c"b.a.p_proj.b\00", align 1
@key_str.117 = private unnamed_addr constant [7 x i8] c"b.l2.w\00", align 1
@key_str.118 = private unnamed_addr constant [7 x i8] c"b.l2.b\00", align 1
@key_str.119 = private unnamed_addr constant [8 x i8] c"b.m.f.W\00", align 1
@key_str.120 = private unnamed_addr constant [8 x i8] c"b.m.f.b\00", align 1
@key_str.121 = private unnamed_addr constant [8 x i8] c"b.m.p.W\00", align 1
@key_str.122 = private unnamed_addr constant [8 x i8] c"b.m.p.b\00", align 1
@key_str.123 = private unnamed_addr constant [4 x i8] c"l.w\00", align 1
@key_str.124 = private unnamed_addr constant [4 x i8] c"l.b\00", align 1
@key_str.125 = private unnamed_addr constant [4 x i8] c"h.W\00", align 1
@key_str.126 = private unnamed_addr constant [4 x i8] c"h.b\00", align 1
@str_literal.127 = private unnamed_addr constant [25 x i8] c"model_2digit.safetensors\00", align 1
@str_literal.128 = private unnamed_addr constant [19 x i8] c"Parameters Loaded.\00", align 1
@str_literal.129 = private unnamed_addr constant [19 x i8] c"Parameters Loaded.\00", align 1
@str_literal.130 = private unnamed_addr constant [42 x i8] c"Running Inference verification 2-digit...\00", align 1
@str_literal.131 = private unnamed_addr constant [7 x i8] c"Input:\00", align 1
@str_literal.132 = private unnamed_addr constant [3 x i8] c"12\00", align 1
@str_literal.133 = private unnamed_addr constant [2 x i8] c"+\00", align 1
@str_literal.134 = private unnamed_addr constant [3 x i8] c"34\00", align 1
@str_literal.135 = private unnamed_addr constant [18 x i8] c"Predicted Digits:\00", align 1
@str_literal.136 = private unnamed_addr constant [7 x i8] c"Input:\00", align 1
@str_literal.137 = private unnamed_addr constant [3 x i8] c"99\00", align 1
@str_literal.138 = private unnamed_addr constant [2 x i8] c"+\00", align 1
@str_literal.139 = private unnamed_addr constant [2 x i8] c"1\00", align 1
@str_literal.140 = private unnamed_addr constant [18 x i8] c"Predicted Digits:\00", align 1
@str_literal.141 = private unnamed_addr constant [7 x i8] c"Input:\00", align 1
@str_literal.142 = private unnamed_addr constant [2 x i8] c"5\00", align 1
@str_literal.143 = private unnamed_addr constant [2 x i8] c"+\00", align 1
@str_literal.144 = private unnamed_addr constant [2 x i8] c"5\00", align 1
@str_literal.145 = private unnamed_addr constant [18 x i8] c"Predicted Digits:\00", align 1
@str_literal.146 = private unnamed_addr constant [7 x i8] c"Input:\00", align 1
@str_literal.147 = private unnamed_addr constant [3 x i8] c"88\00", align 1
@str_literal.148 = private unnamed_addr constant [2 x i8] c"+\00", align 1
@str_literal.149 = private unnamed_addr constant [3 x i8] c"99\00", align 1
@str_literal.150 = private unnamed_addr constant [18 x i8] c"Predicted Digits:\00", align 1
@str_literal.151 = private unnamed_addr constant [33 x i8] c"Inference Verification Complete.\00", align 1

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

declare ptr @tl_tensor_randn(ptr, i1)

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

declare ptr @tl_tensor_randn.3(ptr, i1)

declare ptr @tl_varbuilder_get.4(ptr, i64, ptr)

declare ptr @tl_varbuilder_get_from_tensor.5(ptr, ptr)

declare void @tl_update_all_params.6(float)

declare ptr @tl_varbuilder_grad.7(ptr)

declare void @tl_tensor_backward.8(ptr)

declare ptr @tl_tensor_grad.9(ptr)

declare ptr @tl_tensor_detach.10(ptr, i1)

declare ptr @tl_tensor_softmax.11(ptr, i64)

declare ptr @tl_tensor_cross_entropy.12(ptr, ptr)

declare void @tl_tensor_save.13(ptr, ptr)

declare ptr @tl_tensor_load.14(ptr)

declare void @tl_save_all_params.15(ptr)

declare void @tl_add_parameter.16(ptr, ptr)

declare void @tl_load_all_params.17(ptr)

declare void @tl_tensor_sub_assign.18(ptr, ptr)

declare void @tl_add_parameter.19(ptr, ptr)

declare ptr @tl_register_parameter.20(ptr)

declare ptr @tl_string_concat.21(ptr, ptr)

declare ptr @tl_file_open.22(ptr, ptr)

declare ptr @tl_file_read_string.23(ptr)

declare void @tl_file_write_string.24(ptr, ptr)

declare void @tl_file_close.25(ptr)

declare ptr @tl_path_new.26(ptr)

declare ptr @tl_path_join.27(ptr, ptr)

declare i1 @tl_path_exists.28(ptr)

declare i1 @tl_path_is_dir.29(ptr)

declare i1 @tl_path_is_file.30(ptr)

declare ptr @tl_path_to_string.31(ptr)

declare void @tl_path_free.32(ptr)

declare i1 @tl_http_download.33(ptr, ptr)

declare ptr @tl_http_get.34(ptr)

declare ptr @tl_env_get.35(ptr)

declare void @tl_env_set.36(ptr, ptr)

declare float @tl_system_time.37()

declare void @tl_system_sleep.38(float)

declare i64 @tl_get_memory_mb.39()

declare void @tl_mem_enter_scope.40()

declare void @tl_mem_exit_scope.41()

declare void @tl_mem_register_struct.42(ptr)

declare void @tl_mem_register_tensor.43(ptr)

declare void @tl_mem_unregister.44(ptr)

declare ptr @tl_pool_acquire.45(i64)

declare void @tl_pool_release.46(ptr, i64)

declare void @tl_arena_init.47(i64)

declare i64 @tl_arena_alloc.48(i64)

declare ptr @tl_arena_malloc(i64)

declare i1 @tl_arena_is_active.49()

declare void @tl_arena_free.50()

declare ptr @tl_alloc_tmp(i64)

declare void @tl_free_tmp(ptr)

declare ptr @tl_tensor_reshape_dims.51(ptr, ptr, i64)

declare ptr @tl_tensor_reshape.52(ptr, ptr)

declare ptr @tl_tensor_randn.53(ptr, i1)

declare ptr @tl_varbuilder_get.54(ptr, i64, ptr)

declare ptr @tl_varbuilder_get_from_tensor.55(ptr, ptr)

declare void @tl_update_all_params.56(float)

declare ptr @tl_varbuilder_grad.57(ptr)

declare void @tl_tensor_backward.58(ptr)

declare ptr @tl_tensor_grad.59(ptr)

declare ptr @tl_tensor_detach.60(ptr, i1)

declare ptr @tl_tensor_softmax.61(ptr, i64)

declare ptr @tl_tensor_cross_entropy.62(ptr, ptr)

declare void @tl_tensor_save.63(ptr, ptr)

declare ptr @tl_tensor_load.64(ptr)

declare void @tl_save_all_params.65(ptr)

declare void @tl_add_parameter.66(ptr, ptr)

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
  %scalar_shape_rhs16 = alloca i64, align 16
  %scalar_data_rhs15 = alloca float, align 16
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %o2 = alloca i64, align 16
  %i1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %i, ptr %i1, align 8
  store i64 %o, ptr %o2, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Linear, ptr null, i32 1) to i64))
  %buf_void = call ptr @tl_alloc_tmp(i64 8)
  %i3 = load i64, ptr %i1, align 8
  %i2f = sitofp i64 %i3 to float
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %i2f, ptr %elem_ptr, align 4
  %o4 = load i64, ptr %o2, align 8
  %i2f5 = sitofp i64 %o4 to float
  %elem_ptr6 = getelementptr inbounds float, ptr %buf_void, i64 1
  store float %i2f5, ptr %elem_ptr6, align 4
  %shape_alloc = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr = getelementptr inbounds i64, ptr %shape_alloc, i64 0
  store i64 2, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %shape_alloc)
  call void @tl_free_tmp(ptr %buf_void)
  call void @tl_free_tmp(ptr %shape_alloc)
  %static_call = call ptr @tl_tensor_randn(ptr %new_tensor, i1 true)
  store float 0x3FB99999A0000000, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %static_call, ptr %scalar_tensor_rhs)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res, i1 true)
  %init_field = getelementptr inbounds %Linear, ptr %struct_malloc, i32 0, i32 0
  store ptr %detach_res, ptr %init_field, align 8
  %buf_void7 = call ptr @tl_alloc_tmp(i64 4)
  %o8 = load i64, ptr %o2, align 8
  %i2f9 = sitofp i64 %o8 to float
  %elem_ptr10 = getelementptr inbounds float, ptr %buf_void7, i64 0
  store float %i2f9, ptr %elem_ptr10, align 4
  %shape_alloc11 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr12 = getelementptr inbounds i64, ptr %shape_alloc11, i64 0
  store i64 1, ptr %shape_ptr12, align 8
  %new_tensor13 = call ptr @tl_tensor_new(ptr %buf_void7, i64 1, ptr %shape_alloc11)
  call void @tl_free_tmp(ptr %buf_void7)
  call void @tl_free_tmp(ptr %shape_alloc11)
  %static_call14 = call ptr @tl_tensor_randn(ptr %new_tensor13, i1 true)
  store float 0.000000e+00, ptr %scalar_data_rhs15, align 4
  %scalar_tensor_rhs17 = call ptr @tl_tensor_new(ptr %scalar_data_rhs15, i64 0, ptr %scalar_shape_rhs16)
  %binop_res18 = call ptr @tl_tensor_mul(ptr %static_call14, ptr %scalar_tensor_rhs17)
  %detach_res19 = call ptr @tl_tensor_detach(ptr %binop_res18, i1 true)
  %init_field20 = getelementptr inbounds %Linear, ptr %struct_malloc, i32 0, i32 1
  store ptr %detach_res19, ptr %init_field20, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %Linear, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %struct_malloc, i32 0, i32 1
  %field_val21 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val21)
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
  store ptr %grad_res, ptr %gW, align 8
  %s5 = load ptr, ptr %s, align 8
  %ptr_b = getelementptr inbounds %Linear, ptr %s5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res6 = call ptr @tl_tensor_grad(ptr %b)
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
  %binop_res14 = call ptr @tl_tensor_sub(ptr %W11, ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res14, i1 true)
  %old_field_val = load ptr, ptr %ptr_W8, align 8
  call void @tl_tensor_free(ptr %old_field_val)
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
  %binop_res26 = call ptr @tl_tensor_sub(ptr %b19, ptr %binop_res25)
  %detach_res27 = call ptr @tl_tensor_detach(ptr %binop_res26, i1 true)
  %old_field_val28 = load ptr, ptr %ptr_b16, align 8
  call void @tl_tensor_free(ptr %old_field_val28)
  store ptr %detach_res27, ptr %ptr_b16, align 8
  call void @tl_mem_unregister(ptr %detach_res27)
  %s29 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s29)
  %unreg_field_0 = getelementptr inbounds %Linear, ptr %s29, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %s29, i32 0, i32 1
  %field_val30 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  call void @tl_mem_exit_scope()
  ret ptr %s29
}

define ptr @tl_Embedding_new(i64 %v, i64 %d) {
entry:
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %d2 = alloca i64, align 16
  %v1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %v, ptr %v1, align 8
  store i64 %d, ptr %d2, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Embedding, ptr null, i32 1) to i64))
  %buf_void = call ptr @tl_alloc_tmp(i64 8)
  %v3 = load i64, ptr %v1, align 8
  %i2f = sitofp i64 %v3 to float
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %i2f, ptr %elem_ptr, align 4
  %d4 = load i64, ptr %d2, align 8
  %i2f5 = sitofp i64 %d4 to float
  %elem_ptr6 = getelementptr inbounds float, ptr %buf_void, i64 1
  store float %i2f5, ptr %elem_ptr6, align 4
  %shape_alloc = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr = getelementptr inbounds i64, ptr %shape_alloc, i64 0
  store i64 2, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %shape_alloc)
  call void @tl_free_tmp(ptr %buf_void)
  call void @tl_free_tmp(ptr %shape_alloc)
  %static_call = call ptr @tl_tensor_randn(ptr %new_tensor, i1 true)
  store float 0x3FB99999A0000000, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %static_call, ptr %scalar_tensor_rhs)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res, i1 true)
  %init_field = getelementptr inbounds %Embedding, ptr %struct_malloc, i32 0, i32 0
  store ptr %detach_res, ptr %init_field, align 8
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
  %binop_res12 = call ptr @tl_tensor_sub(ptr %w9, ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res12, i1 true)
  %old_field_val = load ptr, ptr %ptr_w6, align 8
  call void @tl_tensor_free(ptr %old_field_val)
  store ptr %detach_res, ptr %ptr_w6, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  %s13 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s13)
  %unreg_field_0 = getelementptr inbounds %Embedding, ptr %s13, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  call void @tl_mem_exit_scope()
  ret ptr %s13
}

define ptr @tl_LayerNorm_new(i64 %d) {
entry:
  %scalar_shape_rhs16 = alloca i64, align 16
  %scalar_data_rhs15 = alloca float, align 16
  %scalar_shape_rhs4 = alloca i64, align 16
  %scalar_data_rhs3 = alloca float, align 16
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %d1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %d, ptr %d1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%LayerNorm, ptr null, i32 1) to i64))
  %buf_void = call ptr @tl_alloc_tmp(i64 4)
  %d2 = load i64, ptr %d1, align 8
  %i2f = sitofp i64 %d2 to float
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %i2f, ptr %elem_ptr, align 4
  %shape_alloc = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr = getelementptr inbounds i64, ptr %shape_alloc, i64 0
  store i64 1, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %shape_alloc)
  call void @tl_free_tmp(ptr %buf_void)
  call void @tl_free_tmp(ptr %shape_alloc)
  %static_call = call ptr @tl_tensor_randn(ptr %new_tensor, i1 true)
  store float 0.000000e+00, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %static_call, ptr %scalar_tensor_rhs)
  store float 1.000000e+00, ptr %scalar_data_rhs3, align 4
  %scalar_tensor_rhs5 = call ptr @tl_tensor_new(ptr %scalar_data_rhs3, i64 0, ptr %scalar_shape_rhs4)
  %binop_res6 = call ptr @tl_tensor_add(ptr %binop_res, ptr %scalar_tensor_rhs5)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res6, i1 true)
  %init_field = getelementptr inbounds %LayerNorm, ptr %struct_malloc, i32 0, i32 0
  store ptr %detach_res, ptr %init_field, align 8
  %buf_void7 = call ptr @tl_alloc_tmp(i64 4)
  %d8 = load i64, ptr %d1, align 8
  %i2f9 = sitofp i64 %d8 to float
  %elem_ptr10 = getelementptr inbounds float, ptr %buf_void7, i64 0
  store float %i2f9, ptr %elem_ptr10, align 4
  %shape_alloc11 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr12 = getelementptr inbounds i64, ptr %shape_alloc11, i64 0
  store i64 1, ptr %shape_ptr12, align 8
  %new_tensor13 = call ptr @tl_tensor_new(ptr %buf_void7, i64 1, ptr %shape_alloc11)
  call void @tl_free_tmp(ptr %buf_void7)
  call void @tl_free_tmp(ptr %shape_alloc11)
  %static_call14 = call ptr @tl_tensor_randn(ptr %new_tensor13, i1 true)
  store float 0.000000e+00, ptr %scalar_data_rhs15, align 4
  %scalar_tensor_rhs17 = call ptr @tl_tensor_new(ptr %scalar_data_rhs15, i64 0, ptr %scalar_shape_rhs16)
  %binop_res18 = call ptr @tl_tensor_mul(ptr %static_call14, ptr %scalar_tensor_rhs17)
  %detach_res19 = call ptr @tl_tensor_detach(ptr %binop_res18, i1 true)
  %init_field20 = getelementptr inbounds %LayerNorm, ptr %struct_malloc, i32 0, i32 1
  store ptr %detach_res19, ptr %init_field20, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %LayerNorm, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds %LayerNorm, ptr %struct_malloc, i32 0, i32 1
  %field_val21 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val21)
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
  %binop_res12 = call ptr @tl_tensor_sub(ptr %b9, ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res12, i1 true)
  %old_field_val = load ptr, ptr %ptr_b6, align 8
  call void @tl_tensor_free(ptr %old_field_val)
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
  store ptr %call_method, ptr %q, align 8
  %self5 = load ptr, ptr %self1, align 8
  %ptr_k_proj = getelementptr inbounds %CausalSelfAttention, ptr %self5, i32 0, i32 1
  %k_proj = load ptr, ptr %ptr_k_proj, align 8
  %x6 = load ptr, ptr %x2, align 8
  %call_method7 = call ptr @tl_Linear_forward(ptr %k_proj, ptr %x6)
  call void @tl_mem_register_tensor(ptr %call_method7)
  store ptr %call_method7, ptr %k, align 8
  %self8 = load ptr, ptr %self1, align 8
  %ptr_v_proj = getelementptr inbounds %CausalSelfAttention, ptr %self8, i32 0, i32 2
  %v_proj = load ptr, ptr %ptr_v_proj, align 8
  %x9 = load ptr, ptr %x2, align 8
  %call_method10 = call ptr @tl_Linear_forward(ptr %v_proj, ptr %x9)
  call void @tl_mem_register_tensor(ptr %call_method10)
  store ptr %call_method10, ptr %v, align 8
  %q11 = load ptr, ptr %q, align 8
  %k12 = load ptr, ptr %k, align 8
  %transpose_res = call ptr @tl_tensor_transpose(ptr %k12, i64 1, i64 2)
  %matmul_res = call ptr @tl_tensor_matmul(ptr %q11, ptr %transpose_res)
  store float 0x3FB6A0BA20000000, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %matmul_res, ptr %scalar_tensor_rhs)
  %tril_res = call ptr @tl_tensor_tril(ptr %binop_res, i32 0)
  %softmax_res = call ptr @tl_tensor_softmax(ptr %tril_res, i64 2)
  %v13 = load ptr, ptr %v, align 8
  %matmul_res14 = call ptr @tl_tensor_matmul(ptr %softmax_res, ptr %v13)
  store ptr %matmul_res14, ptr %y, align 8
  %self15 = load ptr, ptr %self1, align 8
  %ptr_p_proj = getelementptr inbounds %CausalSelfAttention, ptr %self15, i32 0, i32 3
  %p_proj = load ptr, ptr %ptr_p_proj, align 8
  %y16 = load ptr, ptr %y, align 8
  %call_method17 = call ptr @tl_Linear_forward(ptr %p_proj, ptr %y16)
  call void @tl_mem_register_tensor(ptr %call_method17)
  call void @tl_mem_unregister(ptr %call_method17)
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
  store ptr %call_method, ptr %ptr_q_proj, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s8 = load ptr, ptr %s, align 8
  %ptr_k_proj = getelementptr inbounds %CausalSelfAttention, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_k_proj10 = getelementptr inbounds %CausalSelfAttention, ptr %s9, i32 0, i32 1
  %k_proj = load ptr, ptr %ptr_k_proj10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_Linear_step(ptr %k_proj, float %lr11)
  store ptr %call_method12, ptr %ptr_k_proj, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s13 = load ptr, ptr %s, align 8
  %ptr_v_proj = getelementptr inbounds %CausalSelfAttention, ptr %s13, i32 0, i32 2
  %s14 = load ptr, ptr %s, align 8
  %ptr_v_proj15 = getelementptr inbounds %CausalSelfAttention, ptr %s14, i32 0, i32 2
  %v_proj = load ptr, ptr %ptr_v_proj15, align 8
  %lr16 = load float, ptr %lr2, align 4
  %call_method17 = call ptr @tl_Linear_step(ptr %v_proj, float %lr16)
  store ptr %call_method17, ptr %ptr_v_proj, align 8
  call void @tl_mem_unregister(ptr %call_method17)
  %s18 = load ptr, ptr %s, align 8
  %ptr_p_proj = getelementptr inbounds %CausalSelfAttention, ptr %s18, i32 0, i32 3
  %s19 = load ptr, ptr %s, align 8
  %ptr_p_proj20 = getelementptr inbounds %CausalSelfAttention, ptr %s19, i32 0, i32 3
  %p_proj = load ptr, ptr %ptr_p_proj20, align 8
  %lr21 = load float, ptr %lr2, align 4
  %call_method22 = call ptr @tl_Linear_step(ptr %p_proj, float %lr21)
  store ptr %call_method22, ptr %ptr_p_proj, align 8
  call void @tl_mem_unregister(ptr %call_method22)
  %s23 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s23)
  %unreg_field_0 = getelementptr inbounds %CausalSelfAttention, ptr %s23, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_024 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 0
  %field_val25 = load ptr, ptr %unreg_field_024, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 1
  %field_val26 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  %unreg_field_127 = getelementptr inbounds %CausalSelfAttention, ptr %s23, i32 0, i32 1
  %field_val28 = load ptr, ptr %unreg_field_127, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_029 = getelementptr inbounds %Linear, ptr %field_val28, i32 0, i32 0
  %field_val30 = load ptr, ptr %unreg_field_029, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  %unreg_field_131 = getelementptr inbounds %Linear, ptr %field_val28, i32 0, i32 1
  %field_val32 = load ptr, ptr %unreg_field_131, align 8
  call void @tl_mem_unregister(ptr %field_val32)
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %s23, i32 0, i32 2
  %field_val33 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val33)
  %unreg_field_034 = getelementptr inbounds %Linear, ptr %field_val33, i32 0, i32 0
  %field_val35 = load ptr, ptr %unreg_field_034, align 8
  call void @tl_mem_unregister(ptr %field_val35)
  %unreg_field_136 = getelementptr inbounds %Linear, ptr %field_val33, i32 0, i32 1
  %field_val37 = load ptr, ptr %unreg_field_136, align 8
  call void @tl_mem_unregister(ptr %field_val37)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %s23, i32 0, i32 3
  %field_val38 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_039 = getelementptr inbounds %Linear, ptr %field_val38, i32 0, i32 0
  %field_val40 = load ptr, ptr %unreg_field_039, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_141 = getelementptr inbounds %Linear, ptr %field_val38, i32 0, i32 1
  %field_val42 = load ptr, ptr %unreg_field_141, align 8
  call void @tl_mem_unregister(ptr %field_val42)
  call void @tl_mem_exit_scope()
  ret ptr %s23
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
  store ptr %call_method, ptr %ptr_f, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s8 = load ptr, ptr %s, align 8
  %ptr_p = getelementptr inbounds %MLP, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_p10 = getelementptr inbounds %MLP, ptr %s9, i32 0, i32 1
  %p = load ptr, ptr %ptr_p10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_Linear_step(ptr %p, float %lr11)
  store ptr %call_method12, ptr %ptr_p, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s13 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s13)
  %unreg_field_0 = getelementptr inbounds %MLP, ptr %s13, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_014 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 0
  %field_val15 = load ptr, ptr %unreg_field_014, align 8
  call void @tl_mem_unregister(ptr %field_val15)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 1
  %field_val16 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val16)
  %unreg_field_117 = getelementptr inbounds %MLP, ptr %s13, i32 0, i32 1
  %field_val18 = load ptr, ptr %unreg_field_117, align 8
  call void @tl_mem_unregister(ptr %field_val18)
  %unreg_field_019 = getelementptr inbounds %Linear, ptr %field_val18, i32 0, i32 0
  %field_val20 = load ptr, ptr %unreg_field_019, align 8
  call void @tl_mem_unregister(ptr %field_val20)
  %unreg_field_121 = getelementptr inbounds %Linear, ptr %field_val18, i32 0, i32 1
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
  call void @tl_mem_register_tensor(ptr %call_method7)
  %binop_res = call ptr @tl_tensor_add(ptr %x3, ptr %call_method7)
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
  call void @tl_mem_register_tensor(ptr %call_method14)
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
  %ptr_l1 = getelementptr inbounds %Block, ptr %s4, i32 0, i32 0
  %s5 = load ptr, ptr %s, align 8
  %ptr_l16 = getelementptr inbounds %Block, ptr %s5, i32 0, i32 0
  %l1 = load ptr, ptr %ptr_l16, align 8
  %lr7 = load float, ptr %lr2, align 4
  %call_method = call ptr @tl_LayerNorm_step(ptr %l1, float %lr7)
  store ptr %call_method, ptr %ptr_l1, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s8 = load ptr, ptr %s, align 8
  %ptr_a = getelementptr inbounds %Block, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_a10 = getelementptr inbounds %Block, ptr %s9, i32 0, i32 1
  %a = load ptr, ptr %ptr_a10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_CausalSelfAttention_step(ptr %a, float %lr11)
  store ptr %call_method12, ptr %ptr_a, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s13 = load ptr, ptr %s, align 8
  %ptr_l2 = getelementptr inbounds %Block, ptr %s13, i32 0, i32 2
  %s14 = load ptr, ptr %s, align 8
  %ptr_l215 = getelementptr inbounds %Block, ptr %s14, i32 0, i32 2
  %l2 = load ptr, ptr %ptr_l215, align 8
  %lr16 = load float, ptr %lr2, align 4
  %call_method17 = call ptr @tl_LayerNorm_step(ptr %l2, float %lr16)
  store ptr %call_method17, ptr %ptr_l2, align 8
  call void @tl_mem_unregister(ptr %call_method17)
  %s18 = load ptr, ptr %s, align 8
  %ptr_m = getelementptr inbounds %Block, ptr %s18, i32 0, i32 3
  %s19 = load ptr, ptr %s, align 8
  %ptr_m20 = getelementptr inbounds %Block, ptr %s19, i32 0, i32 3
  %m = load ptr, ptr %ptr_m20, align 8
  %lr21 = load float, ptr %lr2, align 4
  %call_method22 = call ptr @tl_MLP_step(ptr %m, float %lr21)
  store ptr %call_method22, ptr %ptr_m, align 8
  call void @tl_mem_unregister(ptr %call_method22)
  %s23 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s23)
  %unreg_field_0 = getelementptr inbounds %Block, ptr %s23, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_024 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 0
  %field_val25 = load ptr, ptr %unreg_field_024, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_1 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 1
  %field_val26 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  %unreg_field_127 = getelementptr inbounds %Block, ptr %s23, i32 0, i32 1
  %field_val28 = load ptr, ptr %unreg_field_127, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_029 = getelementptr inbounds %CausalSelfAttention, ptr %field_val28, i32 0, i32 0
  %field_val30 = load ptr, ptr %unreg_field_029, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  %unreg_field_031 = getelementptr inbounds %Linear, ptr %field_val30, i32 0, i32 0
  %field_val32 = load ptr, ptr %unreg_field_031, align 8
  call void @tl_mem_unregister(ptr %field_val32)
  %unreg_field_133 = getelementptr inbounds %Linear, ptr %field_val30, i32 0, i32 1
  %field_val34 = load ptr, ptr %unreg_field_133, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  %unreg_field_135 = getelementptr inbounds %CausalSelfAttention, ptr %field_val28, i32 0, i32 1
  %field_val36 = load ptr, ptr %unreg_field_135, align 8
  call void @tl_mem_unregister(ptr %field_val36)
  %unreg_field_037 = getelementptr inbounds %Linear, ptr %field_val36, i32 0, i32 0
  %field_val38 = load ptr, ptr %unreg_field_037, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_139 = getelementptr inbounds %Linear, ptr %field_val36, i32 0, i32 1
  %field_val40 = load ptr, ptr %unreg_field_139, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %field_val28, i32 0, i32 2
  %field_val41 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val41)
  %unreg_field_042 = getelementptr inbounds %Linear, ptr %field_val41, i32 0, i32 0
  %field_val43 = load ptr, ptr %unreg_field_042, align 8
  call void @tl_mem_unregister(ptr %field_val43)
  %unreg_field_144 = getelementptr inbounds %Linear, ptr %field_val41, i32 0, i32 1
  %field_val45 = load ptr, ptr %unreg_field_144, align 8
  call void @tl_mem_unregister(ptr %field_val45)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val28, i32 0, i32 3
  %field_val46 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val46)
  %unreg_field_047 = getelementptr inbounds %Linear, ptr %field_val46, i32 0, i32 0
  %field_val48 = load ptr, ptr %unreg_field_047, align 8
  call void @tl_mem_unregister(ptr %field_val48)
  %unreg_field_149 = getelementptr inbounds %Linear, ptr %field_val46, i32 0, i32 1
  %field_val50 = load ptr, ptr %unreg_field_149, align 8
  call void @tl_mem_unregister(ptr %field_val50)
  %unreg_field_251 = getelementptr inbounds %Block, ptr %s23, i32 0, i32 2
  %field_val52 = load ptr, ptr %unreg_field_251, align 8
  call void @tl_mem_unregister(ptr %field_val52)
  %unreg_field_053 = getelementptr inbounds %LayerNorm, ptr %field_val52, i32 0, i32 0
  %field_val54 = load ptr, ptr %unreg_field_053, align 8
  call void @tl_mem_unregister(ptr %field_val54)
  %unreg_field_155 = getelementptr inbounds %LayerNorm, ptr %field_val52, i32 0, i32 1
  %field_val56 = load ptr, ptr %unreg_field_155, align 8
  call void @tl_mem_unregister(ptr %field_val56)
  %unreg_field_357 = getelementptr inbounds %Block, ptr %s23, i32 0, i32 3
  %field_val58 = load ptr, ptr %unreg_field_357, align 8
  call void @tl_mem_unregister(ptr %field_val58)
  %unreg_field_059 = getelementptr inbounds %MLP, ptr %field_val58, i32 0, i32 0
  %field_val60 = load ptr, ptr %unreg_field_059, align 8
  call void @tl_mem_unregister(ptr %field_val60)
  %unreg_field_061 = getelementptr inbounds %Linear, ptr %field_val60, i32 0, i32 0
  %field_val62 = load ptr, ptr %unreg_field_061, align 8
  call void @tl_mem_unregister(ptr %field_val62)
  %unreg_field_163 = getelementptr inbounds %Linear, ptr %field_val60, i32 0, i32 1
  %field_val64 = load ptr, ptr %unreg_field_163, align 8
  call void @tl_mem_unregister(ptr %field_val64)
  %unreg_field_165 = getelementptr inbounds %MLP, ptr %field_val58, i32 0, i32 1
  %field_val66 = load ptr, ptr %unreg_field_165, align 8
  call void @tl_mem_unregister(ptr %field_val66)
  %unreg_field_067 = getelementptr inbounds %Linear, ptr %field_val66, i32 0, i32 0
  %field_val68 = load ptr, ptr %unreg_field_067, align 8
  call void @tl_mem_unregister(ptr %field_val68)
  %unreg_field_169 = getelementptr inbounds %Linear, ptr %field_val66, i32 0, i32 1
  %field_val70 = load ptr, ptr %unreg_field_169, align 8
  call void @tl_mem_unregister(ptr %field_val70)
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
  %init_field = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d5 = load i64, ptr %d2, align 8
  %static_call6 = call ptr @tl_Embedding_new(i64 12, i64 %d5)
  %init_field7 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call6, ptr %init_field7, align 8
  %d8 = load i64, ptr %d2, align 8
  %static_call9 = call ptr @tl_Block_new(i64 %d8)
  %init_field10 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 2
  store ptr %static_call9, ptr %init_field10, align 8
  %d11 = load i64, ptr %d2, align 8
  %static_call12 = call ptr @tl_LayerNorm_new(i64 %d11)
  %init_field13 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 3
  store ptr %static_call12, ptr %init_field13, align 8
  %d14 = load i64, ptr %d2, align 8
  %v15 = load i64, ptr %v1, align 8
  %static_call16 = call ptr @tl_Linear_new(i64 %d14, i64 %v15)
  %init_field17 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 4
  store ptr %static_call16, ptr %init_field17, align 8
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
  %unreg_field_077 = getelementptr inbounds %LayerNorm, ptr %field_val76, i32 0, i32 0
  %field_val78 = load ptr, ptr %unreg_field_077, align 8
  call void @tl_mem_unregister(ptr %field_val78)
  %unreg_field_179 = getelementptr inbounds %LayerNorm, ptr %field_val76, i32 0, i32 1
  %field_val80 = load ptr, ptr %unreg_field_179, align 8
  call void @tl_mem_unregister(ptr %field_val80)
  %unreg_field_4 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 4
  %field_val81 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val81)
  %unreg_field_082 = getelementptr inbounds %Linear, ptr %field_val81, i32 0, i32 0
  %field_val83 = load ptr, ptr %unreg_field_082, align 8
  call void @tl_mem_unregister(ptr %field_val83)
  %unreg_field_184 = getelementptr inbounds %Linear, ptr %field_val81, i32 0, i32 1
  %field_val85 = load ptr, ptr %unreg_field_184, align 8
  call void @tl_mem_unregister(ptr %field_val85)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_GPT_forward(ptr %self, ptr %i) {
entry:
  %pos_emb = alloca ptr, align 16
  %tok_emb = alloca ptr, align 16
  %pos = alloca ptr, align 16
  %pos_data = alloca ptr, align 16
  %i2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store ptr %i, ptr %i2, align 8
  %temp_data_alloc = call ptr @tl_alloc_tmp(i64 48)
  %temp_shape_alloc = call ptr @tl_alloc_tmp(i64 8)
  %data_elem = getelementptr inbounds float, ptr %temp_data_alloc, i64 0
  store float 0.000000e+00, ptr %data_elem, align 4
  %data_elem3 = getelementptr inbounds float, ptr %temp_data_alloc, i64 1
  store float 1.000000e+00, ptr %data_elem3, align 4
  %data_elem4 = getelementptr inbounds float, ptr %temp_data_alloc, i64 2
  store float 2.000000e+00, ptr %data_elem4, align 4
  %data_elem5 = getelementptr inbounds float, ptr %temp_data_alloc, i64 3
  store float 3.000000e+00, ptr %data_elem5, align 4
  %data_elem6 = getelementptr inbounds float, ptr %temp_data_alloc, i64 4
  store float 4.000000e+00, ptr %data_elem6, align 4
  %data_elem7 = getelementptr inbounds float, ptr %temp_data_alloc, i64 5
  store float 5.000000e+00, ptr %data_elem7, align 4
  %data_elem8 = getelementptr inbounds float, ptr %temp_data_alloc, i64 6
  store float 6.000000e+00, ptr %data_elem8, align 4
  %data_elem9 = getelementptr inbounds float, ptr %temp_data_alloc, i64 7
  store float 7.000000e+00, ptr %data_elem9, align 4
  %data_elem10 = getelementptr inbounds float, ptr %temp_data_alloc, i64 8
  store float 8.000000e+00, ptr %data_elem10, align 4
  %data_elem11 = getelementptr inbounds float, ptr %temp_data_alloc, i64 9
  store float 9.000000e+00, ptr %data_elem11, align 4
  %data_elem12 = getelementptr inbounds float, ptr %temp_data_alloc, i64 10
  store float 1.000000e+01, ptr %data_elem12, align 4
  %data_elem13 = getelementptr inbounds float, ptr %temp_data_alloc, i64 11
  store float 1.100000e+01, ptr %data_elem13, align 4
  %shape_elem = getelementptr inbounds i64, ptr %temp_shape_alloc, i64 0
  store i64 12, ptr %shape_elem, align 8
  %new_const_tensor = call ptr @tl_tensor_new(ptr %temp_data_alloc, i64 1, ptr %temp_shape_alloc)
  call void @tl_free_tmp(ptr %temp_data_alloc)
  call void @tl_free_tmp(ptr %temp_shape_alloc)
  store ptr %new_const_tensor, ptr %pos_data, align 8
  %pos_data14 = load ptr, ptr %pos_data, align 8
  %dims_alloca = alloca [2 x i64], align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr, align 8
  %dim_ptr15 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr15, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %pos_data14, ptr %dims_ptr, i64 2)
  store ptr %reshape_dims_res, ptr %pos, align 8
  %self16 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds %GPT, ptr %self16, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %i17 = load ptr, ptr %i2, align 8
  %call_method = call ptr @tl_Embedding_forward(ptr %w, ptr %i17)
  call void @tl_mem_register_tensor(ptr %call_method)
  store ptr %call_method, ptr %tok_emb, align 8
  %self18 = load ptr, ptr %self1, align 8
  %ptr_wp = getelementptr inbounds %GPT, ptr %self18, i32 0, i32 1
  %wp = load ptr, ptr %ptr_wp, align 8
  %pos19 = load ptr, ptr %pos, align 8
  %call_method20 = call ptr @tl_Embedding_forward(ptr %wp, ptr %pos19)
  call void @tl_mem_register_tensor(ptr %call_method20)
  store ptr %call_method20, ptr %pos_emb, align 8
  %self21 = load ptr, ptr %self1, align 8
  %ptr_h = getelementptr inbounds %GPT, ptr %self21, i32 0, i32 4
  %h = load ptr, ptr %ptr_h, align 8
  %self22 = load ptr, ptr %self1, align 8
  %ptr_l = getelementptr inbounds %GPT, ptr %self22, i32 0, i32 3
  %l = load ptr, ptr %ptr_l, align 8
  %self23 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds %GPT, ptr %self23, i32 0, i32 2
  %b = load ptr, ptr %ptr_b, align 8
  %tok_emb24 = load ptr, ptr %tok_emb, align 8
  %pos_emb25 = load ptr, ptr %pos_emb, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %tok_emb24, ptr %pos_emb25)
  %call_method26 = call ptr @tl_Block_forward(ptr %b, ptr %binop_res)
  call void @tl_mem_register_tensor(ptr %call_method26)
  %call_method27 = call ptr @tl_LayerNorm_forward(ptr %l, ptr %call_method26)
  call void @tl_mem_register_tensor(ptr %call_method27)
  %call_method28 = call ptr @tl_Linear_forward(ptr %h, ptr %call_method27)
  call void @tl_mem_register_tensor(ptr %call_method28)
  call void @tl_mem_unregister(ptr %call_method28)
  call void @tl_mem_exit_scope()
  ret ptr %call_method28
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
  store ptr %call_method, ptr %ptr_w, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s8 = load ptr, ptr %s, align 8
  %ptr_wp = getelementptr inbounds %GPT, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_wp10 = getelementptr inbounds %GPT, ptr %s9, i32 0, i32 1
  %wp = load ptr, ptr %ptr_wp10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_Embedding_step(ptr %wp, float %lr11)
  store ptr %call_method12, ptr %ptr_wp, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s13 = load ptr, ptr %s, align 8
  %ptr_b = getelementptr inbounds %GPT, ptr %s13, i32 0, i32 2
  %s14 = load ptr, ptr %s, align 8
  %ptr_b15 = getelementptr inbounds %GPT, ptr %s14, i32 0, i32 2
  %b = load ptr, ptr %ptr_b15, align 8
  %lr16 = load float, ptr %lr2, align 4
  %call_method17 = call ptr @tl_Block_step(ptr %b, float %lr16)
  store ptr %call_method17, ptr %ptr_b, align 8
  call void @tl_mem_unregister(ptr %call_method17)
  %s18 = load ptr, ptr %s, align 8
  %ptr_l = getelementptr inbounds %GPT, ptr %s18, i32 0, i32 3
  %s19 = load ptr, ptr %s, align 8
  %ptr_l20 = getelementptr inbounds %GPT, ptr %s19, i32 0, i32 3
  %l = load ptr, ptr %ptr_l20, align 8
  %lr21 = load float, ptr %lr2, align 4
  %call_method22 = call ptr @tl_LayerNorm_step(ptr %l, float %lr21)
  store ptr %call_method22, ptr %ptr_l, align 8
  call void @tl_mem_unregister(ptr %call_method22)
  %s23 = load ptr, ptr %s, align 8
  %ptr_h = getelementptr inbounds %GPT, ptr %s23, i32 0, i32 4
  %s24 = load ptr, ptr %s, align 8
  %ptr_h25 = getelementptr inbounds %GPT, ptr %s24, i32 0, i32 4
  %h = load ptr, ptr %ptr_h25, align 8
  %lr26 = load float, ptr %lr2, align 4
  %call_method27 = call ptr @tl_Linear_step(ptr %h, float %lr26)
  store ptr %call_method27, ptr %ptr_h, align 8
  call void @tl_mem_unregister(ptr %call_method27)
  %s28 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s28)
  %unreg_field_0 = getelementptr inbounds %GPT, ptr %s28, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_029 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val30 = load ptr, ptr %unreg_field_029, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  %unreg_field_1 = getelementptr inbounds %GPT, ptr %s28, i32 0, i32 1
  %field_val31 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val31)
  %unreg_field_032 = getelementptr inbounds %Embedding, ptr %field_val31, i32 0, i32 0
  %field_val33 = load ptr, ptr %unreg_field_032, align 8
  call void @tl_mem_unregister(ptr %field_val33)
  %unreg_field_2 = getelementptr inbounds %GPT, ptr %s28, i32 0, i32 2
  %field_val34 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  %unreg_field_035 = getelementptr inbounds %Block, ptr %field_val34, i32 0, i32 0
  %field_val36 = load ptr, ptr %unreg_field_035, align 8
  call void @tl_mem_unregister(ptr %field_val36)
  %unreg_field_037 = getelementptr inbounds %LayerNorm, ptr %field_val36, i32 0, i32 0
  %field_val38 = load ptr, ptr %unreg_field_037, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_139 = getelementptr inbounds %LayerNorm, ptr %field_val36, i32 0, i32 1
  %field_val40 = load ptr, ptr %unreg_field_139, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_141 = getelementptr inbounds %Block, ptr %field_val34, i32 0, i32 1
  %field_val42 = load ptr, ptr %unreg_field_141, align 8
  call void @tl_mem_unregister(ptr %field_val42)
  %unreg_field_043 = getelementptr inbounds %CausalSelfAttention, ptr %field_val42, i32 0, i32 0
  %field_val44 = load ptr, ptr %unreg_field_043, align 8
  call void @tl_mem_unregister(ptr %field_val44)
  %unreg_field_045 = getelementptr inbounds %Linear, ptr %field_val44, i32 0, i32 0
  %field_val46 = load ptr, ptr %unreg_field_045, align 8
  call void @tl_mem_unregister(ptr %field_val46)
  %unreg_field_147 = getelementptr inbounds %Linear, ptr %field_val44, i32 0, i32 1
  %field_val48 = load ptr, ptr %unreg_field_147, align 8
  call void @tl_mem_unregister(ptr %field_val48)
  %unreg_field_149 = getelementptr inbounds %CausalSelfAttention, ptr %field_val42, i32 0, i32 1
  %field_val50 = load ptr, ptr %unreg_field_149, align 8
  call void @tl_mem_unregister(ptr %field_val50)
  %unreg_field_051 = getelementptr inbounds %Linear, ptr %field_val50, i32 0, i32 0
  %field_val52 = load ptr, ptr %unreg_field_051, align 8
  call void @tl_mem_unregister(ptr %field_val52)
  %unreg_field_153 = getelementptr inbounds %Linear, ptr %field_val50, i32 0, i32 1
  %field_val54 = load ptr, ptr %unreg_field_153, align 8
  call void @tl_mem_unregister(ptr %field_val54)
  %unreg_field_255 = getelementptr inbounds %CausalSelfAttention, ptr %field_val42, i32 0, i32 2
  %field_val56 = load ptr, ptr %unreg_field_255, align 8
  call void @tl_mem_unregister(ptr %field_val56)
  %unreg_field_057 = getelementptr inbounds %Linear, ptr %field_val56, i32 0, i32 0
  %field_val58 = load ptr, ptr %unreg_field_057, align 8
  call void @tl_mem_unregister(ptr %field_val58)
  %unreg_field_159 = getelementptr inbounds %Linear, ptr %field_val56, i32 0, i32 1
  %field_val60 = load ptr, ptr %unreg_field_159, align 8
  call void @tl_mem_unregister(ptr %field_val60)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val42, i32 0, i32 3
  %field_val61 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val61)
  %unreg_field_062 = getelementptr inbounds %Linear, ptr %field_val61, i32 0, i32 0
  %field_val63 = load ptr, ptr %unreg_field_062, align 8
  call void @tl_mem_unregister(ptr %field_val63)
  %unreg_field_164 = getelementptr inbounds %Linear, ptr %field_val61, i32 0, i32 1
  %field_val65 = load ptr, ptr %unreg_field_164, align 8
  call void @tl_mem_unregister(ptr %field_val65)
  %unreg_field_266 = getelementptr inbounds %Block, ptr %field_val34, i32 0, i32 2
  %field_val67 = load ptr, ptr %unreg_field_266, align 8
  call void @tl_mem_unregister(ptr %field_val67)
  %unreg_field_068 = getelementptr inbounds %LayerNorm, ptr %field_val67, i32 0, i32 0
  %field_val69 = load ptr, ptr %unreg_field_068, align 8
  call void @tl_mem_unregister(ptr %field_val69)
  %unreg_field_170 = getelementptr inbounds %LayerNorm, ptr %field_val67, i32 0, i32 1
  %field_val71 = load ptr, ptr %unreg_field_170, align 8
  call void @tl_mem_unregister(ptr %field_val71)
  %unreg_field_372 = getelementptr inbounds %Block, ptr %field_val34, i32 0, i32 3
  %field_val73 = load ptr, ptr %unreg_field_372, align 8
  call void @tl_mem_unregister(ptr %field_val73)
  %unreg_field_074 = getelementptr inbounds %MLP, ptr %field_val73, i32 0, i32 0
  %field_val75 = load ptr, ptr %unreg_field_074, align 8
  call void @tl_mem_unregister(ptr %field_val75)
  %unreg_field_076 = getelementptr inbounds %Linear, ptr %field_val75, i32 0, i32 0
  %field_val77 = load ptr, ptr %unreg_field_076, align 8
  call void @tl_mem_unregister(ptr %field_val77)
  %unreg_field_178 = getelementptr inbounds %Linear, ptr %field_val75, i32 0, i32 1
  %field_val79 = load ptr, ptr %unreg_field_178, align 8
  call void @tl_mem_unregister(ptr %field_val79)
  %unreg_field_180 = getelementptr inbounds %MLP, ptr %field_val73, i32 0, i32 1
  %field_val81 = load ptr, ptr %unreg_field_180, align 8
  call void @tl_mem_unregister(ptr %field_val81)
  %unreg_field_082 = getelementptr inbounds %Linear, ptr %field_val81, i32 0, i32 0
  %field_val83 = load ptr, ptr %unreg_field_082, align 8
  call void @tl_mem_unregister(ptr %field_val83)
  %unreg_field_184 = getelementptr inbounds %Linear, ptr %field_val81, i32 0, i32 1
  %field_val85 = load ptr, ptr %unreg_field_184, align 8
  call void @tl_mem_unregister(ptr %field_val85)
  %unreg_field_386 = getelementptr inbounds %GPT, ptr %s28, i32 0, i32 3
  %field_val87 = load ptr, ptr %unreg_field_386, align 8
  call void @tl_mem_unregister(ptr %field_val87)
  %unreg_field_088 = getelementptr inbounds %LayerNorm, ptr %field_val87, i32 0, i32 0
  %field_val89 = load ptr, ptr %unreg_field_088, align 8
  call void @tl_mem_unregister(ptr %field_val89)
  %unreg_field_190 = getelementptr inbounds %LayerNorm, ptr %field_val87, i32 0, i32 1
  %field_val91 = load ptr, ptr %unreg_field_190, align 8
  call void @tl_mem_unregister(ptr %field_val91)
  %unreg_field_4 = getelementptr inbounds %GPT, ptr %s28, i32 0, i32 4
  %field_val92 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val92)
  %unreg_field_093 = getelementptr inbounds %Linear, ptr %field_val92, i32 0, i32 0
  %field_val94 = load ptr, ptr %unreg_field_093, align 8
  call void @tl_mem_unregister(ptr %field_val94)
  %unreg_field_195 = getelementptr inbounds %Linear, ptr %field_val92, i32 0, i32 1
  %field_val96 = load ptr, ptr %unreg_field_195, align 8
  call void @tl_mem_unregister(ptr %field_val96)
  call void @tl_mem_exit_scope()
  ret ptr %s28
}

define i64 @argmax(ptr %t) {
entry:
  %v = alloca float, align 16
  %k = alloca i64, align 16
  %idx = alloca i64, align 16
  %max_val = alloca float, align 16
  %t1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %t, ptr %t1, align 8
  store float -1.000000e+06, ptr %max_val, align 4
  store i64 0, ptr %idx, align 8
  br label %for_header

for_header:                                       ; preds = %merge, %entry
  %for_idx = phi i64 [ 0, %entry ], [ %next_idx, %merge ]
  %for_cond = icmp slt i64 %for_idx, 13
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %k, align 8
  %t2 = load ptr, ptr %t1, align 8
  %k3 = load i64, ptr %k, align 8
  %get_res = call float @tl_tensor_get(ptr %t2, i64 %k3)
  store float %get_res, ptr %v, align 4
  %v4 = load float, ptr %v, align 4
  %max_val5 = load float, ptr %max_val, align 4
  %fgttmp = fcmp ogt float %v4, %max_val5
  br i1 %fgttmp, label %then, label %else

for_end:                                          ; preds = %for_header
  %idx8 = load i64, ptr %idx, align 8
  call void @tl_mem_exit_scope()
  ret i64 %idx8

then:                                             ; preds = %for_body
  call void @tl_mem_enter_scope()
  %v6 = load float, ptr %v, align 4
  store float %v6, ptr %max_val, align 4
  %k7 = load i64, ptr %k, align 8
  store i64 %k7, ptr %idx, align 8
  call void @tl_mem_exit_scope()
  br label %merge

else:                                             ; preds = %for_body
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else, %then
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx, 1
  br label %for_header
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
  %qq3 = alloca i64, align 16
  %ll3_flat = alloca ptr, align 16
  %ll3 = alloca ptr, align 16
  %jj3 = alloca ptr, align 16
  %ee3 = alloca ptr, align 16
  %vqq2 = alloca float, align 16
  %scalar_shape591 = alloca i64, align 16
  %scalar_data590 = alloca float, align 16
  %scalar_shape588 = alloca i64, align 16
  %scalar_data586 = alloca float, align 16
  %qq2 = alloca i64, align 16
  %ll2_flat = alloca ptr, align 16
  %ll2 = alloca ptr, align 16
  %jj2 = alloca ptr, align 16
  %ee2 = alloca ptr, align 16
  %vqq1 = alloca float, align 16
  %scalar_shape539 = alloca i64, align 16
  %scalar_data538 = alloca float, align 16
  %scalar_shape536 = alloca i64, align 16
  %scalar_data534 = alloca float, align 16
  %qq1 = alloca i64, align 16
  %ll1_flat = alloca ptr, align 16
  %ll1 = alloca ptr, align 16
  %jj1 = alloca ptr, align 16
  %ee1 = alloca ptr, align 16
  %w5491 = alloca float, align 16
  %w4490 = alloca float, align 16
  %w3 = alloca float, align 16
  %w2 = alloca float, align 16
  %w1 = alloca float, align 16
  %w0 = alloca float, align 16
  %pp3 = alloca i64, align 16
  %llo3_flat = alloca ptr, align 16
  %llo3 = alloca ptr, align 16
  %ii3 = alloca ptr, align 16
  %dd3 = alloca ptr, align 16
  %vpp2 = alloca float, align 16
  %scalar_shape443 = alloca i64, align 16
  %scalar_data442 = alloca float, align 16
  %scalar_shape440 = alloca i64, align 16
  %scalar_data438 = alloca float, align 16
  %pp2 = alloca i64, align 16
  %llo2_flat = alloca ptr, align 16
  %llo2 = alloca ptr, align 16
  %ii2 = alloca ptr, align 16
  %dd2 = alloca ptr, align 16
  %vpp1 = alloca float, align 16
  %scalar_shape391 = alloca i64, align 16
  %scalar_data390 = alloca float, align 16
  %scalar_shape388 = alloca i64, align 16
  %scalar_data386 = alloca float, align 16
  %pp1 = alloca i64, align 16
  %llo1_flat = alloca ptr, align 16
  %llo1 = alloca ptr, align 16
  %ii1 = alloca ptr, align 16
  %dd1 = alloca ptr, align 16
  %z5 = alloca float, align 16
  %z4 = alloca float, align 16
  %z3 = alloca float, align 16
  %z2 = alloca float, align 16
  %z1 = alloca float, align 16
  %z0 = alloca float, align 16
  %p3 = alloca i64, align 16
  %log3_flat = alloca ptr, align 16
  %log3 = alloca ptr, align 16
  %inp3 = alloca ptr, align 16
  %d3 = alloca ptr, align 16
  %vp2 = alloca float, align 16
  %scalar_shape297 = alloca i64, align 16
  %scalar_data296 = alloca float, align 16
  %scalar_shape294 = alloca i64, align 16
  %scalar_data292 = alloca float, align 16
  %p2 = alloca i64, align 16
  %log2_flat = alloca ptr, align 16
  %log2 = alloca ptr, align 16
  %inp2 = alloca ptr, align 16
  %d2 = alloca ptr, align 16
  %vp1 = alloca float, align 16
  %scalar_shape245 = alloca i64, align 16
  %scalar_data244 = alloca float, align 16
  %scalar_shape242 = alloca i64, align 16
  %scalar_data240 = alloca float, align 16
  %p1 = alloca i64, align 16
  %log1_flat = alloca ptr, align 16
  %log1 = alloca ptr, align 16
  %inp1 = alloca ptr, align 16
  %d1 = alloca ptr, align 16
  %y5 = alloca float, align 16
  %y4 = alloca float, align 16
  %y3 = alloca float, align 16
  %y2 = alloca float, align 16
  %y1 = alloca float, align 16
  %y0 = alloca float, align 16
  %pred3 = alloca i64, align 16
  %next_logits3 = alloca ptr, align 16
  %logits3_flat = alloca ptr, align 16
  %logits3 = alloca ptr, align 16
  %input3 = alloca ptr, align 16
  %data3 = alloca ptr, align 16
  %val_pred2 = alloca float, align 16
  %scalar_shape149 = alloca i64, align 16
  %scalar_data148 = alloca float, align 16
  %scalar_shape146 = alloca i64, align 16
  %scalar_data144 = alloca float, align 16
  %pred2 = alloca i64, align 16
  %next_logits2 = alloca ptr, align 16
  %logits2_flat = alloca ptr, align 16
  %logits2 = alloca ptr, align 16
  %input2 = alloca ptr, align 16
  %data2 = alloca ptr, align 16
  %val_pred1 = alloca float, align 16
  %scalar_shape96 = alloca i64, align 16
  %scalar_data95 = alloca float, align 16
  %scalar_shape = alloca i64, align 16
  %scalar_data = alloca float, align 16
  %pred1 = alloca i64, align 16
  %next_logits1 = alloca ptr, align 16
  %logits1_flat = alloca ptr, align 16
  %logits1 = alloca ptr, align 16
  %input1 = alloca ptr, align 16
  %data1 = alloca ptr, align 16
  %val_pad = alloca float, align 16
  %x5 = alloca float, align 16
  %x4 = alloca float, align 16
  %x3 = alloca float, align 16
  %x2 = alloca float, align 16
  %x1 = alloca float, align 16
  %x0 = alloca float, align 16
  %model = alloca ptr, align 16
  %block_size = alloca i64, align 16
  %d_model = alloca i64, align 16
  %vocab_size = alloca i64, align 16
  call void @tl_mem_enter_scope()
  call void @tl_arena_init(i64 5996544)
  store i64 13, ptr %vocab_size, align 8
  store i64 128, ptr %d_model, align 8
  store i64 12, ptr %block_size, align 8
  call void @tl_print_string(ptr @str_literal)
  %vocab_size1 = load i64, ptr %vocab_size, align 8
  %d_model2 = load i64, ptr %d_model, align 8
  %static_call = call ptr @tl_GPT_new(i64 %vocab_size1, i64 %d_model2)
  store ptr %static_call, ptr %model, align 8
  call void @tl_print_string(ptr @str_literal.104)
  call void @tl_print_string(ptr @str_literal.105)
  %model3 = load ptr, ptr %model, align 8
  %w = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 0
  %sub_ptr = load ptr, ptr %w, align 8
  %w4 = getelementptr inbounds %Embedding, ptr %sub_ptr, i32 0, i32 0
  %w5 = load ptr, ptr %w4, align 8
  call void @tl_add_parameter(ptr @key_str, ptr %w5)
  %wp = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 1
  %sub_ptr6 = load ptr, ptr %wp, align 8
  %w7 = getelementptr inbounds %Embedding, ptr %sub_ptr6, i32 0, i32 0
  %w8 = load ptr, ptr %w7, align 8
  call void @tl_add_parameter(ptr @key_str.106, ptr %w8)
  %b = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 2
  %sub_ptr9 = load ptr, ptr %b, align 8
  %l1 = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 0
  %sub_ptr10 = load ptr, ptr %l1, align 8
  %w11 = getelementptr inbounds %LayerNorm, ptr %sub_ptr10, i32 0, i32 0
  %w12 = load ptr, ptr %w11, align 8
  call void @tl_add_parameter(ptr @key_str.107, ptr %w12)
  %b13 = getelementptr inbounds %LayerNorm, ptr %sub_ptr10, i32 0, i32 1
  %b14 = load ptr, ptr %b13, align 8
  call void @tl_add_parameter(ptr @key_str.108, ptr %b14)
  %a = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 1
  %sub_ptr15 = load ptr, ptr %a, align 8
  %q_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr15, i32 0, i32 0
  %sub_ptr16 = load ptr, ptr %q_proj, align 8
  %W = getelementptr inbounds %Linear, ptr %sub_ptr16, i32 0, i32 0
  %W17 = load ptr, ptr %W, align 8
  call void @tl_add_parameter(ptr @key_str.109, ptr %W17)
  %b18 = getelementptr inbounds %Linear, ptr %sub_ptr16, i32 0, i32 1
  %b19 = load ptr, ptr %b18, align 8
  call void @tl_add_parameter(ptr @key_str.110, ptr %b19)
  %k_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr15, i32 0, i32 1
  %sub_ptr20 = load ptr, ptr %k_proj, align 8
  %W21 = getelementptr inbounds %Linear, ptr %sub_ptr20, i32 0, i32 0
  %W22 = load ptr, ptr %W21, align 8
  call void @tl_add_parameter(ptr @key_str.111, ptr %W22)
  %b23 = getelementptr inbounds %Linear, ptr %sub_ptr20, i32 0, i32 1
  %b24 = load ptr, ptr %b23, align 8
  call void @tl_add_parameter(ptr @key_str.112, ptr %b24)
  %v_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr15, i32 0, i32 2
  %sub_ptr25 = load ptr, ptr %v_proj, align 8
  %W26 = getelementptr inbounds %Linear, ptr %sub_ptr25, i32 0, i32 0
  %W27 = load ptr, ptr %W26, align 8
  call void @tl_add_parameter(ptr @key_str.113, ptr %W27)
  %b28 = getelementptr inbounds %Linear, ptr %sub_ptr25, i32 0, i32 1
  %b29 = load ptr, ptr %b28, align 8
  call void @tl_add_parameter(ptr @key_str.114, ptr %b29)
  %p_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr15, i32 0, i32 3
  %sub_ptr30 = load ptr, ptr %p_proj, align 8
  %W31 = getelementptr inbounds %Linear, ptr %sub_ptr30, i32 0, i32 0
  %W32 = load ptr, ptr %W31, align 8
  call void @tl_add_parameter(ptr @key_str.115, ptr %W32)
  %b33 = getelementptr inbounds %Linear, ptr %sub_ptr30, i32 0, i32 1
  %b34 = load ptr, ptr %b33, align 8
  call void @tl_add_parameter(ptr @key_str.116, ptr %b34)
  %l2 = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 2
  %sub_ptr35 = load ptr, ptr %l2, align 8
  %w36 = getelementptr inbounds %LayerNorm, ptr %sub_ptr35, i32 0, i32 0
  %w37 = load ptr, ptr %w36, align 8
  call void @tl_add_parameter(ptr @key_str.117, ptr %w37)
  %b38 = getelementptr inbounds %LayerNorm, ptr %sub_ptr35, i32 0, i32 1
  %b39 = load ptr, ptr %b38, align 8
  call void @tl_add_parameter(ptr @key_str.118, ptr %b39)
  %m = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 3
  %sub_ptr40 = load ptr, ptr %m, align 8
  %f = getelementptr inbounds %MLP, ptr %sub_ptr40, i32 0, i32 0
  %sub_ptr41 = load ptr, ptr %f, align 8
  %W42 = getelementptr inbounds %Linear, ptr %sub_ptr41, i32 0, i32 0
  %W43 = load ptr, ptr %W42, align 8
  call void @tl_add_parameter(ptr @key_str.119, ptr %W43)
  %b44 = getelementptr inbounds %Linear, ptr %sub_ptr41, i32 0, i32 1
  %b45 = load ptr, ptr %b44, align 8
  call void @tl_add_parameter(ptr @key_str.120, ptr %b45)
  %p = getelementptr inbounds %MLP, ptr %sub_ptr40, i32 0, i32 1
  %sub_ptr46 = load ptr, ptr %p, align 8
  %W47 = getelementptr inbounds %Linear, ptr %sub_ptr46, i32 0, i32 0
  %W48 = load ptr, ptr %W47, align 8
  call void @tl_add_parameter(ptr @key_str.121, ptr %W48)
  %b49 = getelementptr inbounds %Linear, ptr %sub_ptr46, i32 0, i32 1
  %b50 = load ptr, ptr %b49, align 8
  call void @tl_add_parameter(ptr @key_str.122, ptr %b50)
  %l = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 3
  %sub_ptr51 = load ptr, ptr %l, align 8
  %w52 = getelementptr inbounds %LayerNorm, ptr %sub_ptr51, i32 0, i32 0
  %w53 = load ptr, ptr %w52, align 8
  call void @tl_add_parameter(ptr @key_str.123, ptr %w53)
  %b54 = getelementptr inbounds %LayerNorm, ptr %sub_ptr51, i32 0, i32 1
  %b55 = load ptr, ptr %b54, align 8
  call void @tl_add_parameter(ptr @key_str.124, ptr %b55)
  %h = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 4
  %sub_ptr56 = load ptr, ptr %h, align 8
  %W57 = getelementptr inbounds %Linear, ptr %sub_ptr56, i32 0, i32 0
  %W58 = load ptr, ptr %W57, align 8
  call void @tl_add_parameter(ptr @key_str.125, ptr %W58)
  %b59 = getelementptr inbounds %Linear, ptr %sub_ptr56, i32 0, i32 1
  %b60 = load ptr, ptr %b59, align 8
  call void @tl_add_parameter(ptr @key_str.126, ptr %b60)
  call void @tl_load_all_params(ptr @str_literal.127)
  call void @tl_print_string(ptr @str_literal.128)
  call void @tl_print_string(ptr @str_literal.129)
  call void @tl_print_string(ptr @str_literal.130)
  call void @tl_print_string(ptr @str_literal.131)
  call void @tl_print_string(ptr @str_literal.132)
  call void @tl_print_string(ptr @str_literal.133)
  call void @tl_print_string(ptr @str_literal.134)
  call void @tl_print_string(ptr @str_literal.135)
  store float 1.000000e+00, ptr %x0, align 4
  store float 2.000000e+00, ptr %x1, align 4
  store float 1.000000e+01, ptr %x2, align 4
  store float 3.000000e+00, ptr %x3, align 4
  store float 4.000000e+00, ptr %x4, align 4
  store float 1.100000e+01, ptr %x5, align 4
  store float 1.200000e+01, ptr %val_pad, align 4
  %buf_void = call ptr @tl_alloc_tmp(i64 48)
  %x061 = load float, ptr %x0, align 4
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %x061, ptr %elem_ptr, align 4
  %x162 = load float, ptr %x1, align 4
  %elem_ptr63 = getelementptr inbounds float, ptr %buf_void, i64 1
  store float %x162, ptr %elem_ptr63, align 4
  %x264 = load float, ptr %x2, align 4
  %elem_ptr65 = getelementptr inbounds float, ptr %buf_void, i64 2
  store float %x264, ptr %elem_ptr65, align 4
  %x366 = load float, ptr %x3, align 4
  %elem_ptr67 = getelementptr inbounds float, ptr %buf_void, i64 3
  store float %x366, ptr %elem_ptr67, align 4
  %x468 = load float, ptr %x4, align 4
  %elem_ptr69 = getelementptr inbounds float, ptr %buf_void, i64 4
  store float %x468, ptr %elem_ptr69, align 4
  %x570 = load float, ptr %x5, align 4
  %elem_ptr71 = getelementptr inbounds float, ptr %buf_void, i64 5
  store float %x570, ptr %elem_ptr71, align 4
  %val_pad72 = load float, ptr %val_pad, align 4
  %elem_ptr73 = getelementptr inbounds float, ptr %buf_void, i64 6
  store float %val_pad72, ptr %elem_ptr73, align 4
  %val_pad74 = load float, ptr %val_pad, align 4
  %elem_ptr75 = getelementptr inbounds float, ptr %buf_void, i64 7
  store float %val_pad74, ptr %elem_ptr75, align 4
  %val_pad76 = load float, ptr %val_pad, align 4
  %elem_ptr77 = getelementptr inbounds float, ptr %buf_void, i64 8
  store float %val_pad76, ptr %elem_ptr77, align 4
  %elem_ptr78 = getelementptr inbounds float, ptr %buf_void, i64 9
  store float 1.200000e+01, ptr %elem_ptr78, align 4
  %elem_ptr79 = getelementptr inbounds float, ptr %buf_void, i64 10
  store float 1.200000e+01, ptr %elem_ptr79, align 4
  %elem_ptr80 = getelementptr inbounds float, ptr %buf_void, i64 11
  store float 1.200000e+01, ptr %elem_ptr80, align 4
  %shape_alloc = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr = getelementptr inbounds i64, ptr %shape_alloc, i64 0
  store i64 12, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %shape_alloc)
  call void @tl_free_tmp(ptr %buf_void)
  call void @tl_free_tmp(ptr %shape_alloc)
  store ptr %new_tensor, ptr %data1, align 8
  %data181 = load ptr, ptr %data1, align 8
  %dims_alloca = alloca [2 x i64], align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr, align 8
  %dim_ptr82 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr82, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %data181, ptr %dims_ptr, i64 2)
  store ptr %reshape_dims_res, ptr %input1, align 8
  %model83 = load ptr, ptr %model, align 8
  %input184 = load ptr, ptr %input1, align 8
  %call_method = call ptr @tl_GPT_forward(ptr %model83, ptr %input184)
  call void @tl_mem_register_tensor(ptr %call_method)
  store ptr %call_method, ptr %logits1, align 8
  %logits185 = load ptr, ptr %logits1, align 8
  %dims_alloca86 = alloca [2 x i64], align 8
  %dim_ptr87 = getelementptr [2 x i64], ptr %dims_alloca86, i64 0, i64 0
  store i64 12, ptr %dim_ptr87, align 8
  %dim_ptr88 = getelementptr [2 x i64], ptr %dims_alloca86, i64 0, i64 1
  store i64 13, ptr %dim_ptr88, align 8
  %dims_ptr89 = getelementptr [2 x i64], ptr %dims_alloca86, i64 0, i64 0
  %reshape_dims_res90 = call ptr @tl_tensor_reshape_dims(ptr %logits185, ptr %dims_ptr89, i64 2)
  store ptr %reshape_dims_res90, ptr %logits1_flat, align 8
  %logits1_flat91 = load ptr, ptr %logits1_flat, align 8
  %slice_res = call ptr @tl_tensor_slice(ptr %logits1_flat91, i64 5, i64 1)
  store ptr %slice_res, ptr %next_logits1, align 8
  %next_logits192 = load ptr, ptr %next_logits1, align 8
  %call_tmp = call i64 @argmax(ptr %next_logits192)
  store i64 %call_tmp, ptr %pred1, align 8
  %pred193 = load i64, ptr %pred1, align 8
  call void @tl_print_i64(i64 %pred193)
  %pred194 = load i64, ptr %pred1, align 8
  %cast_i64_f32 = sitofp i64 %pred194 to float
  store float %cast_i64_f32, ptr %scalar_data, align 4
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  store float 1.000000e+00, ptr %scalar_data95, align 4
  %scalar_tensor97 = call ptr @tl_tensor_new(ptr %scalar_data95, i64 0, ptr %scalar_shape96)
  %pow_res = call ptr @tl_tensor_pow(ptr %scalar_tensor, ptr %scalar_tensor97)
  %get_res = call float @tl_tensor_get(ptr %pow_res, i64 0)
  store float %get_res, ptr %val_pred1, align 4
  %buf_void98 = call ptr @tl_alloc_tmp(i64 48)
  %x099 = load float, ptr %x0, align 4
  %elem_ptr100 = getelementptr inbounds float, ptr %buf_void98, i64 0
  store float %x099, ptr %elem_ptr100, align 4
  %x1101 = load float, ptr %x1, align 4
  %elem_ptr102 = getelementptr inbounds float, ptr %buf_void98, i64 1
  store float %x1101, ptr %elem_ptr102, align 4
  %x2103 = load float, ptr %x2, align 4
  %elem_ptr104 = getelementptr inbounds float, ptr %buf_void98, i64 2
  store float %x2103, ptr %elem_ptr104, align 4
  %x3105 = load float, ptr %x3, align 4
  %elem_ptr106 = getelementptr inbounds float, ptr %buf_void98, i64 3
  store float %x3105, ptr %elem_ptr106, align 4
  %x4107 = load float, ptr %x4, align 4
  %elem_ptr108 = getelementptr inbounds float, ptr %buf_void98, i64 4
  store float %x4107, ptr %elem_ptr108, align 4
  %x5109 = load float, ptr %x5, align 4
  %elem_ptr110 = getelementptr inbounds float, ptr %buf_void98, i64 5
  store float %x5109, ptr %elem_ptr110, align 4
  %val_pred1111 = load float, ptr %val_pred1, align 4
  %elem_ptr112 = getelementptr inbounds float, ptr %buf_void98, i64 6
  store float %val_pred1111, ptr %elem_ptr112, align 4
  %val_pad113 = load float, ptr %val_pad, align 4
  %elem_ptr114 = getelementptr inbounds float, ptr %buf_void98, i64 7
  store float %val_pad113, ptr %elem_ptr114, align 4
  %val_pad115 = load float, ptr %val_pad, align 4
  %elem_ptr116 = getelementptr inbounds float, ptr %buf_void98, i64 8
  store float %val_pad115, ptr %elem_ptr116, align 4
  %elem_ptr117 = getelementptr inbounds float, ptr %buf_void98, i64 9
  store float 1.200000e+01, ptr %elem_ptr117, align 4
  %elem_ptr118 = getelementptr inbounds float, ptr %buf_void98, i64 10
  store float 1.200000e+01, ptr %elem_ptr118, align 4
  %elem_ptr119 = getelementptr inbounds float, ptr %buf_void98, i64 11
  store float 1.200000e+01, ptr %elem_ptr119, align 4
  %shape_alloc120 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr121 = getelementptr inbounds i64, ptr %shape_alloc120, i64 0
  store i64 12, ptr %shape_ptr121, align 8
  %new_tensor122 = call ptr @tl_tensor_new(ptr %buf_void98, i64 1, ptr %shape_alloc120)
  call void @tl_free_tmp(ptr %buf_void98)
  call void @tl_free_tmp(ptr %shape_alloc120)
  store ptr %new_tensor122, ptr %data2, align 8
  %data2123 = load ptr, ptr %data2, align 8
  %dims_alloca124 = alloca [2 x i64], align 8
  %dim_ptr125 = getelementptr [2 x i64], ptr %dims_alloca124, i64 0, i64 0
  store i64 1, ptr %dim_ptr125, align 8
  %dim_ptr126 = getelementptr [2 x i64], ptr %dims_alloca124, i64 0, i64 1
  store i64 12, ptr %dim_ptr126, align 8
  %dims_ptr127 = getelementptr [2 x i64], ptr %dims_alloca124, i64 0, i64 0
  %reshape_dims_res128 = call ptr @tl_tensor_reshape_dims(ptr %data2123, ptr %dims_ptr127, i64 2)
  store ptr %reshape_dims_res128, ptr %input2, align 8
  %model129 = load ptr, ptr %model, align 8
  %input2130 = load ptr, ptr %input2, align 8
  %call_method131 = call ptr @tl_GPT_forward(ptr %model129, ptr %input2130)
  call void @tl_mem_register_tensor(ptr %call_method131)
  store ptr %call_method131, ptr %logits2, align 8
  %logits2132 = load ptr, ptr %logits2, align 8
  %dims_alloca133 = alloca [2 x i64], align 8
  %dim_ptr134 = getelementptr [2 x i64], ptr %dims_alloca133, i64 0, i64 0
  store i64 12, ptr %dim_ptr134, align 8
  %dim_ptr135 = getelementptr [2 x i64], ptr %dims_alloca133, i64 0, i64 1
  store i64 13, ptr %dim_ptr135, align 8
  %dims_ptr136 = getelementptr [2 x i64], ptr %dims_alloca133, i64 0, i64 0
  %reshape_dims_res137 = call ptr @tl_tensor_reshape_dims(ptr %logits2132, ptr %dims_ptr136, i64 2)
  store ptr %reshape_dims_res137, ptr %logits2_flat, align 8
  %logits2_flat138 = load ptr, ptr %logits2_flat, align 8
  %slice_res139 = call ptr @tl_tensor_slice(ptr %logits2_flat138, i64 6, i64 1)
  store ptr %slice_res139, ptr %next_logits2, align 8
  %next_logits2140 = load ptr, ptr %next_logits2, align 8
  %call_tmp141 = call i64 @argmax(ptr %next_logits2140)
  store i64 %call_tmp141, ptr %pred2, align 8
  %pred2142 = load i64, ptr %pred2, align 8
  call void @tl_print_i64(i64 %pred2142)
  %pred2143 = load i64, ptr %pred2, align 8
  %cast_i64_f32145 = sitofp i64 %pred2143 to float
  store float %cast_i64_f32145, ptr %scalar_data144, align 4
  %scalar_tensor147 = call ptr @tl_tensor_new(ptr %scalar_data144, i64 0, ptr %scalar_shape146)
  store float 1.000000e+00, ptr %scalar_data148, align 4
  %scalar_tensor150 = call ptr @tl_tensor_new(ptr %scalar_data148, i64 0, ptr %scalar_shape149)
  %pow_res151 = call ptr @tl_tensor_pow(ptr %scalar_tensor147, ptr %scalar_tensor150)
  %get_res152 = call float @tl_tensor_get(ptr %pow_res151, i64 0)
  store float %get_res152, ptr %val_pred2, align 4
  %buf_void153 = call ptr @tl_alloc_tmp(i64 48)
  %x0154 = load float, ptr %x0, align 4
  %elem_ptr155 = getelementptr inbounds float, ptr %buf_void153, i64 0
  store float %x0154, ptr %elem_ptr155, align 4
  %x1156 = load float, ptr %x1, align 4
  %elem_ptr157 = getelementptr inbounds float, ptr %buf_void153, i64 1
  store float %x1156, ptr %elem_ptr157, align 4
  %x2158 = load float, ptr %x2, align 4
  %elem_ptr159 = getelementptr inbounds float, ptr %buf_void153, i64 2
  store float %x2158, ptr %elem_ptr159, align 4
  %x3160 = load float, ptr %x3, align 4
  %elem_ptr161 = getelementptr inbounds float, ptr %buf_void153, i64 3
  store float %x3160, ptr %elem_ptr161, align 4
  %x4162 = load float, ptr %x4, align 4
  %elem_ptr163 = getelementptr inbounds float, ptr %buf_void153, i64 4
  store float %x4162, ptr %elem_ptr163, align 4
  %x5164 = load float, ptr %x5, align 4
  %elem_ptr165 = getelementptr inbounds float, ptr %buf_void153, i64 5
  store float %x5164, ptr %elem_ptr165, align 4
  %val_pred1166 = load float, ptr %val_pred1, align 4
  %elem_ptr167 = getelementptr inbounds float, ptr %buf_void153, i64 6
  store float %val_pred1166, ptr %elem_ptr167, align 4
  %val_pred2168 = load float, ptr %val_pred2, align 4
  %elem_ptr169 = getelementptr inbounds float, ptr %buf_void153, i64 7
  store float %val_pred2168, ptr %elem_ptr169, align 4
  %val_pad170 = load float, ptr %val_pad, align 4
  %elem_ptr171 = getelementptr inbounds float, ptr %buf_void153, i64 8
  store float %val_pad170, ptr %elem_ptr171, align 4
  %elem_ptr172 = getelementptr inbounds float, ptr %buf_void153, i64 9
  store float 1.200000e+01, ptr %elem_ptr172, align 4
  %elem_ptr173 = getelementptr inbounds float, ptr %buf_void153, i64 10
  store float 1.200000e+01, ptr %elem_ptr173, align 4
  %elem_ptr174 = getelementptr inbounds float, ptr %buf_void153, i64 11
  store float 1.200000e+01, ptr %elem_ptr174, align 4
  %shape_alloc175 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr176 = getelementptr inbounds i64, ptr %shape_alloc175, i64 0
  store i64 12, ptr %shape_ptr176, align 8
  %new_tensor177 = call ptr @tl_tensor_new(ptr %buf_void153, i64 1, ptr %shape_alloc175)
  call void @tl_free_tmp(ptr %buf_void153)
  call void @tl_free_tmp(ptr %shape_alloc175)
  store ptr %new_tensor177, ptr %data3, align 8
  %data3178 = load ptr, ptr %data3, align 8
  %dims_alloca179 = alloca [2 x i64], align 8
  %dim_ptr180 = getelementptr [2 x i64], ptr %dims_alloca179, i64 0, i64 0
  store i64 1, ptr %dim_ptr180, align 8
  %dim_ptr181 = getelementptr [2 x i64], ptr %dims_alloca179, i64 0, i64 1
  store i64 12, ptr %dim_ptr181, align 8
  %dims_ptr182 = getelementptr [2 x i64], ptr %dims_alloca179, i64 0, i64 0
  %reshape_dims_res183 = call ptr @tl_tensor_reshape_dims(ptr %data3178, ptr %dims_ptr182, i64 2)
  store ptr %reshape_dims_res183, ptr %input3, align 8
  %model184 = load ptr, ptr %model, align 8
  %input3185 = load ptr, ptr %input3, align 8
  %call_method186 = call ptr @tl_GPT_forward(ptr %model184, ptr %input3185)
  call void @tl_mem_register_tensor(ptr %call_method186)
  store ptr %call_method186, ptr %logits3, align 8
  %logits3187 = load ptr, ptr %logits3, align 8
  %dims_alloca188 = alloca [2 x i64], align 8
  %dim_ptr189 = getelementptr [2 x i64], ptr %dims_alloca188, i64 0, i64 0
  store i64 12, ptr %dim_ptr189, align 8
  %dim_ptr190 = getelementptr [2 x i64], ptr %dims_alloca188, i64 0, i64 1
  store i64 13, ptr %dim_ptr190, align 8
  %dims_ptr191 = getelementptr [2 x i64], ptr %dims_alloca188, i64 0, i64 0
  %reshape_dims_res192 = call ptr @tl_tensor_reshape_dims(ptr %logits3187, ptr %dims_ptr191, i64 2)
  store ptr %reshape_dims_res192, ptr %logits3_flat, align 8
  %logits3_flat193 = load ptr, ptr %logits3_flat, align 8
  %slice_res194 = call ptr @tl_tensor_slice(ptr %logits3_flat193, i64 7, i64 1)
  store ptr %slice_res194, ptr %next_logits3, align 8
  %next_logits3195 = load ptr, ptr %next_logits3, align 8
  %call_tmp196 = call i64 @argmax(ptr %next_logits3195)
  store i64 %call_tmp196, ptr %pred3, align 8
  %pred3197 = load i64, ptr %pred3, align 8
  call void @tl_print_i64(i64 %pred3197)
  call void @tl_print_string(ptr @str_literal.136)
  call void @tl_print_string(ptr @str_literal.137)
  call void @tl_print_string(ptr @str_literal.138)
  call void @tl_print_string(ptr @str_literal.139)
  call void @tl_print_string(ptr @str_literal.140)
  store float 9.000000e+00, ptr %y0, align 4
  store float 9.000000e+00, ptr %y1, align 4
  store float 1.000000e+01, ptr %y2, align 4
  store float 0.000000e+00, ptr %y3, align 4
  store float 1.000000e+00, ptr %y4, align 4
  store float 1.100000e+01, ptr %y5, align 4
  %buf_void198 = call ptr @tl_alloc_tmp(i64 48)
  %y0199 = load float, ptr %y0, align 4
  %elem_ptr200 = getelementptr inbounds float, ptr %buf_void198, i64 0
  store float %y0199, ptr %elem_ptr200, align 4
  %y1201 = load float, ptr %y1, align 4
  %elem_ptr202 = getelementptr inbounds float, ptr %buf_void198, i64 1
  store float %y1201, ptr %elem_ptr202, align 4
  %y2203 = load float, ptr %y2, align 4
  %elem_ptr204 = getelementptr inbounds float, ptr %buf_void198, i64 2
  store float %y2203, ptr %elem_ptr204, align 4
  %y3205 = load float, ptr %y3, align 4
  %elem_ptr206 = getelementptr inbounds float, ptr %buf_void198, i64 3
  store float %y3205, ptr %elem_ptr206, align 4
  %y4207 = load float, ptr %y4, align 4
  %elem_ptr208 = getelementptr inbounds float, ptr %buf_void198, i64 4
  store float %y4207, ptr %elem_ptr208, align 4
  %y5209 = load float, ptr %y5, align 4
  %elem_ptr210 = getelementptr inbounds float, ptr %buf_void198, i64 5
  store float %y5209, ptr %elem_ptr210, align 4
  %elem_ptr211 = getelementptr inbounds float, ptr %buf_void198, i64 6
  store float 1.200000e+01, ptr %elem_ptr211, align 4
  %elem_ptr212 = getelementptr inbounds float, ptr %buf_void198, i64 7
  store float 1.200000e+01, ptr %elem_ptr212, align 4
  %elem_ptr213 = getelementptr inbounds float, ptr %buf_void198, i64 8
  store float 1.200000e+01, ptr %elem_ptr213, align 4
  %elem_ptr214 = getelementptr inbounds float, ptr %buf_void198, i64 9
  store float 1.200000e+01, ptr %elem_ptr214, align 4
  %elem_ptr215 = getelementptr inbounds float, ptr %buf_void198, i64 10
  store float 1.200000e+01, ptr %elem_ptr215, align 4
  %elem_ptr216 = getelementptr inbounds float, ptr %buf_void198, i64 11
  store float 1.200000e+01, ptr %elem_ptr216, align 4
  %shape_alloc217 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr218 = getelementptr inbounds i64, ptr %shape_alloc217, i64 0
  store i64 12, ptr %shape_ptr218, align 8
  %new_tensor219 = call ptr @tl_tensor_new(ptr %buf_void198, i64 1, ptr %shape_alloc217)
  call void @tl_free_tmp(ptr %buf_void198)
  call void @tl_free_tmp(ptr %shape_alloc217)
  store ptr %new_tensor219, ptr %d1, align 8
  %d1220 = load ptr, ptr %d1, align 8
  %dims_alloca221 = alloca [2 x i64], align 8
  %dim_ptr222 = getelementptr [2 x i64], ptr %dims_alloca221, i64 0, i64 0
  store i64 1, ptr %dim_ptr222, align 8
  %dim_ptr223 = getelementptr [2 x i64], ptr %dims_alloca221, i64 0, i64 1
  store i64 12, ptr %dim_ptr223, align 8
  %dims_ptr224 = getelementptr [2 x i64], ptr %dims_alloca221, i64 0, i64 0
  %reshape_dims_res225 = call ptr @tl_tensor_reshape_dims(ptr %d1220, ptr %dims_ptr224, i64 2)
  store ptr %reshape_dims_res225, ptr %inp1, align 8
  %model226 = load ptr, ptr %model, align 8
  %inp1227 = load ptr, ptr %inp1, align 8
  %call_method228 = call ptr @tl_GPT_forward(ptr %model226, ptr %inp1227)
  call void @tl_mem_register_tensor(ptr %call_method228)
  store ptr %call_method228, ptr %log1, align 8
  %log1229 = load ptr, ptr %log1, align 8
  %dims_alloca230 = alloca [2 x i64], align 8
  %dim_ptr231 = getelementptr [2 x i64], ptr %dims_alloca230, i64 0, i64 0
  store i64 12, ptr %dim_ptr231, align 8
  %dim_ptr232 = getelementptr [2 x i64], ptr %dims_alloca230, i64 0, i64 1
  store i64 13, ptr %dim_ptr232, align 8
  %dims_ptr233 = getelementptr [2 x i64], ptr %dims_alloca230, i64 0, i64 0
  %reshape_dims_res234 = call ptr @tl_tensor_reshape_dims(ptr %log1229, ptr %dims_ptr233, i64 2)
  store ptr %reshape_dims_res234, ptr %log1_flat, align 8
  %log1_flat235 = load ptr, ptr %log1_flat, align 8
  %slice_res236 = call ptr @tl_tensor_slice(ptr %log1_flat235, i64 5, i64 1)
  %call_tmp237 = call i64 @argmax(ptr %slice_res236)
  store i64 %call_tmp237, ptr %p1, align 8
  %p1238 = load i64, ptr %p1, align 8
  call void @tl_print_i64(i64 %p1238)
  %p1239 = load i64, ptr %p1, align 8
  %cast_i64_f32241 = sitofp i64 %p1239 to float
  store float %cast_i64_f32241, ptr %scalar_data240, align 4
  %scalar_tensor243 = call ptr @tl_tensor_new(ptr %scalar_data240, i64 0, ptr %scalar_shape242)
  store float 1.000000e+00, ptr %scalar_data244, align 4
  %scalar_tensor246 = call ptr @tl_tensor_new(ptr %scalar_data244, i64 0, ptr %scalar_shape245)
  %pow_res247 = call ptr @tl_tensor_pow(ptr %scalar_tensor243, ptr %scalar_tensor246)
  %get_res248 = call float @tl_tensor_get(ptr %pow_res247, i64 0)
  store float %get_res248, ptr %vp1, align 4
  %buf_void249 = call ptr @tl_alloc_tmp(i64 48)
  %y0250 = load float, ptr %y0, align 4
  %elem_ptr251 = getelementptr inbounds float, ptr %buf_void249, i64 0
  store float %y0250, ptr %elem_ptr251, align 4
  %y1252 = load float, ptr %y1, align 4
  %elem_ptr253 = getelementptr inbounds float, ptr %buf_void249, i64 1
  store float %y1252, ptr %elem_ptr253, align 4
  %y2254 = load float, ptr %y2, align 4
  %elem_ptr255 = getelementptr inbounds float, ptr %buf_void249, i64 2
  store float %y2254, ptr %elem_ptr255, align 4
  %y3256 = load float, ptr %y3, align 4
  %elem_ptr257 = getelementptr inbounds float, ptr %buf_void249, i64 3
  store float %y3256, ptr %elem_ptr257, align 4
  %y4258 = load float, ptr %y4, align 4
  %elem_ptr259 = getelementptr inbounds float, ptr %buf_void249, i64 4
  store float %y4258, ptr %elem_ptr259, align 4
  %y5260 = load float, ptr %y5, align 4
  %elem_ptr261 = getelementptr inbounds float, ptr %buf_void249, i64 5
  store float %y5260, ptr %elem_ptr261, align 4
  %vp1262 = load float, ptr %vp1, align 4
  %elem_ptr263 = getelementptr inbounds float, ptr %buf_void249, i64 6
  store float %vp1262, ptr %elem_ptr263, align 4
  %elem_ptr264 = getelementptr inbounds float, ptr %buf_void249, i64 7
  store float 1.200000e+01, ptr %elem_ptr264, align 4
  %elem_ptr265 = getelementptr inbounds float, ptr %buf_void249, i64 8
  store float 1.200000e+01, ptr %elem_ptr265, align 4
  %elem_ptr266 = getelementptr inbounds float, ptr %buf_void249, i64 9
  store float 1.200000e+01, ptr %elem_ptr266, align 4
  %elem_ptr267 = getelementptr inbounds float, ptr %buf_void249, i64 10
  store float 1.200000e+01, ptr %elem_ptr267, align 4
  %elem_ptr268 = getelementptr inbounds float, ptr %buf_void249, i64 11
  store float 1.200000e+01, ptr %elem_ptr268, align 4
  %shape_alloc269 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr270 = getelementptr inbounds i64, ptr %shape_alloc269, i64 0
  store i64 12, ptr %shape_ptr270, align 8
  %new_tensor271 = call ptr @tl_tensor_new(ptr %buf_void249, i64 1, ptr %shape_alloc269)
  call void @tl_free_tmp(ptr %buf_void249)
  call void @tl_free_tmp(ptr %shape_alloc269)
  store ptr %new_tensor271, ptr %d2, align 8
  %d2272 = load ptr, ptr %d2, align 8
  %dims_alloca273 = alloca [2 x i64], align 8
  %dim_ptr274 = getelementptr [2 x i64], ptr %dims_alloca273, i64 0, i64 0
  store i64 1, ptr %dim_ptr274, align 8
  %dim_ptr275 = getelementptr [2 x i64], ptr %dims_alloca273, i64 0, i64 1
  store i64 12, ptr %dim_ptr275, align 8
  %dims_ptr276 = getelementptr [2 x i64], ptr %dims_alloca273, i64 0, i64 0
  %reshape_dims_res277 = call ptr @tl_tensor_reshape_dims(ptr %d2272, ptr %dims_ptr276, i64 2)
  store ptr %reshape_dims_res277, ptr %inp2, align 8
  %model278 = load ptr, ptr %model, align 8
  %inp2279 = load ptr, ptr %inp2, align 8
  %call_method280 = call ptr @tl_GPT_forward(ptr %model278, ptr %inp2279)
  call void @tl_mem_register_tensor(ptr %call_method280)
  store ptr %call_method280, ptr %log2, align 8
  %log2281 = load ptr, ptr %log2, align 8
  %dims_alloca282 = alloca [2 x i64], align 8
  %dim_ptr283 = getelementptr [2 x i64], ptr %dims_alloca282, i64 0, i64 0
  store i64 12, ptr %dim_ptr283, align 8
  %dim_ptr284 = getelementptr [2 x i64], ptr %dims_alloca282, i64 0, i64 1
  store i64 13, ptr %dim_ptr284, align 8
  %dims_ptr285 = getelementptr [2 x i64], ptr %dims_alloca282, i64 0, i64 0
  %reshape_dims_res286 = call ptr @tl_tensor_reshape_dims(ptr %log2281, ptr %dims_ptr285, i64 2)
  store ptr %reshape_dims_res286, ptr %log2_flat, align 8
  %log2_flat287 = load ptr, ptr %log2_flat, align 8
  %slice_res288 = call ptr @tl_tensor_slice(ptr %log2_flat287, i64 6, i64 1)
  %call_tmp289 = call i64 @argmax(ptr %slice_res288)
  store i64 %call_tmp289, ptr %p2, align 8
  %p2290 = load i64, ptr %p2, align 8
  call void @tl_print_i64(i64 %p2290)
  %p2291 = load i64, ptr %p2, align 8
  %cast_i64_f32293 = sitofp i64 %p2291 to float
  store float %cast_i64_f32293, ptr %scalar_data292, align 4
  %scalar_tensor295 = call ptr @tl_tensor_new(ptr %scalar_data292, i64 0, ptr %scalar_shape294)
  store float 1.000000e+00, ptr %scalar_data296, align 4
  %scalar_tensor298 = call ptr @tl_tensor_new(ptr %scalar_data296, i64 0, ptr %scalar_shape297)
  %pow_res299 = call ptr @tl_tensor_pow(ptr %scalar_tensor295, ptr %scalar_tensor298)
  %get_res300 = call float @tl_tensor_get(ptr %pow_res299, i64 0)
  store float %get_res300, ptr %vp2, align 4
  %buf_void301 = call ptr @tl_alloc_tmp(i64 48)
  %y0302 = load float, ptr %y0, align 4
  %elem_ptr303 = getelementptr inbounds float, ptr %buf_void301, i64 0
  store float %y0302, ptr %elem_ptr303, align 4
  %y1304 = load float, ptr %y1, align 4
  %elem_ptr305 = getelementptr inbounds float, ptr %buf_void301, i64 1
  store float %y1304, ptr %elem_ptr305, align 4
  %y2306 = load float, ptr %y2, align 4
  %elem_ptr307 = getelementptr inbounds float, ptr %buf_void301, i64 2
  store float %y2306, ptr %elem_ptr307, align 4
  %y3308 = load float, ptr %y3, align 4
  %elem_ptr309 = getelementptr inbounds float, ptr %buf_void301, i64 3
  store float %y3308, ptr %elem_ptr309, align 4
  %y4310 = load float, ptr %y4, align 4
  %elem_ptr311 = getelementptr inbounds float, ptr %buf_void301, i64 4
  store float %y4310, ptr %elem_ptr311, align 4
  %y5312 = load float, ptr %y5, align 4
  %elem_ptr313 = getelementptr inbounds float, ptr %buf_void301, i64 5
  store float %y5312, ptr %elem_ptr313, align 4
  %vp1314 = load float, ptr %vp1, align 4
  %elem_ptr315 = getelementptr inbounds float, ptr %buf_void301, i64 6
  store float %vp1314, ptr %elem_ptr315, align 4
  %vp2316 = load float, ptr %vp2, align 4
  %elem_ptr317 = getelementptr inbounds float, ptr %buf_void301, i64 7
  store float %vp2316, ptr %elem_ptr317, align 4
  %elem_ptr318 = getelementptr inbounds float, ptr %buf_void301, i64 8
  store float 1.200000e+01, ptr %elem_ptr318, align 4
  %elem_ptr319 = getelementptr inbounds float, ptr %buf_void301, i64 9
  store float 1.200000e+01, ptr %elem_ptr319, align 4
  %elem_ptr320 = getelementptr inbounds float, ptr %buf_void301, i64 10
  store float 1.200000e+01, ptr %elem_ptr320, align 4
  %elem_ptr321 = getelementptr inbounds float, ptr %buf_void301, i64 11
  store float 1.200000e+01, ptr %elem_ptr321, align 4
  %shape_alloc322 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr323 = getelementptr inbounds i64, ptr %shape_alloc322, i64 0
  store i64 12, ptr %shape_ptr323, align 8
  %new_tensor324 = call ptr @tl_tensor_new(ptr %buf_void301, i64 1, ptr %shape_alloc322)
  call void @tl_free_tmp(ptr %buf_void301)
  call void @tl_free_tmp(ptr %shape_alloc322)
  store ptr %new_tensor324, ptr %d3, align 8
  %d3325 = load ptr, ptr %d3, align 8
  %dims_alloca326 = alloca [2 x i64], align 8
  %dim_ptr327 = getelementptr [2 x i64], ptr %dims_alloca326, i64 0, i64 0
  store i64 1, ptr %dim_ptr327, align 8
  %dim_ptr328 = getelementptr [2 x i64], ptr %dims_alloca326, i64 0, i64 1
  store i64 12, ptr %dim_ptr328, align 8
  %dims_ptr329 = getelementptr [2 x i64], ptr %dims_alloca326, i64 0, i64 0
  %reshape_dims_res330 = call ptr @tl_tensor_reshape_dims(ptr %d3325, ptr %dims_ptr329, i64 2)
  store ptr %reshape_dims_res330, ptr %inp3, align 8
  %model331 = load ptr, ptr %model, align 8
  %inp3332 = load ptr, ptr %inp3, align 8
  %call_method333 = call ptr @tl_GPT_forward(ptr %model331, ptr %inp3332)
  call void @tl_mem_register_tensor(ptr %call_method333)
  store ptr %call_method333, ptr %log3, align 8
  %log3334 = load ptr, ptr %log3, align 8
  %dims_alloca335 = alloca [2 x i64], align 8
  %dim_ptr336 = getelementptr [2 x i64], ptr %dims_alloca335, i64 0, i64 0
  store i64 12, ptr %dim_ptr336, align 8
  %dim_ptr337 = getelementptr [2 x i64], ptr %dims_alloca335, i64 0, i64 1
  store i64 13, ptr %dim_ptr337, align 8
  %dims_ptr338 = getelementptr [2 x i64], ptr %dims_alloca335, i64 0, i64 0
  %reshape_dims_res339 = call ptr @tl_tensor_reshape_dims(ptr %log3334, ptr %dims_ptr338, i64 2)
  store ptr %reshape_dims_res339, ptr %log3_flat, align 8
  %log3_flat340 = load ptr, ptr %log3_flat, align 8
  %slice_res341 = call ptr @tl_tensor_slice(ptr %log3_flat340, i64 7, i64 1)
  %call_tmp342 = call i64 @argmax(ptr %slice_res341)
  store i64 %call_tmp342, ptr %p3, align 8
  %p3343 = load i64, ptr %p3, align 8
  call void @tl_print_i64(i64 %p3343)
  call void @tl_print_string(ptr @str_literal.141)
  call void @tl_print_string(ptr @str_literal.142)
  call void @tl_print_string(ptr @str_literal.143)
  call void @tl_print_string(ptr @str_literal.144)
  call void @tl_print_string(ptr @str_literal.145)
  store float 0.000000e+00, ptr %z0, align 4
  store float 5.000000e+00, ptr %z1, align 4
  store float 1.000000e+01, ptr %z2, align 4
  store float 0.000000e+00, ptr %z3, align 4
  store float 5.000000e+00, ptr %z4, align 4
  store float 1.100000e+01, ptr %z5, align 4
  %buf_void344 = call ptr @tl_alloc_tmp(i64 48)
  %z0345 = load float, ptr %z0, align 4
  %elem_ptr346 = getelementptr inbounds float, ptr %buf_void344, i64 0
  store float %z0345, ptr %elem_ptr346, align 4
  %z1347 = load float, ptr %z1, align 4
  %elem_ptr348 = getelementptr inbounds float, ptr %buf_void344, i64 1
  store float %z1347, ptr %elem_ptr348, align 4
  %z2349 = load float, ptr %z2, align 4
  %elem_ptr350 = getelementptr inbounds float, ptr %buf_void344, i64 2
  store float %z2349, ptr %elem_ptr350, align 4
  %z3351 = load float, ptr %z3, align 4
  %elem_ptr352 = getelementptr inbounds float, ptr %buf_void344, i64 3
  store float %z3351, ptr %elem_ptr352, align 4
  %z4353 = load float, ptr %z4, align 4
  %elem_ptr354 = getelementptr inbounds float, ptr %buf_void344, i64 4
  store float %z4353, ptr %elem_ptr354, align 4
  %z5355 = load float, ptr %z5, align 4
  %elem_ptr356 = getelementptr inbounds float, ptr %buf_void344, i64 5
  store float %z5355, ptr %elem_ptr356, align 4
  %elem_ptr357 = getelementptr inbounds float, ptr %buf_void344, i64 6
  store float 1.200000e+01, ptr %elem_ptr357, align 4
  %elem_ptr358 = getelementptr inbounds float, ptr %buf_void344, i64 7
  store float 1.200000e+01, ptr %elem_ptr358, align 4
  %elem_ptr359 = getelementptr inbounds float, ptr %buf_void344, i64 8
  store float 1.200000e+01, ptr %elem_ptr359, align 4
  %elem_ptr360 = getelementptr inbounds float, ptr %buf_void344, i64 9
  store float 1.200000e+01, ptr %elem_ptr360, align 4
  %elem_ptr361 = getelementptr inbounds float, ptr %buf_void344, i64 10
  store float 1.200000e+01, ptr %elem_ptr361, align 4
  %elem_ptr362 = getelementptr inbounds float, ptr %buf_void344, i64 11
  store float 1.200000e+01, ptr %elem_ptr362, align 4
  %shape_alloc363 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr364 = getelementptr inbounds i64, ptr %shape_alloc363, i64 0
  store i64 12, ptr %shape_ptr364, align 8
  %new_tensor365 = call ptr @tl_tensor_new(ptr %buf_void344, i64 1, ptr %shape_alloc363)
  call void @tl_free_tmp(ptr %buf_void344)
  call void @tl_free_tmp(ptr %shape_alloc363)
  store ptr %new_tensor365, ptr %dd1, align 8
  %dd1366 = load ptr, ptr %dd1, align 8
  %dims_alloca367 = alloca [2 x i64], align 8
  %dim_ptr368 = getelementptr [2 x i64], ptr %dims_alloca367, i64 0, i64 0
  store i64 1, ptr %dim_ptr368, align 8
  %dim_ptr369 = getelementptr [2 x i64], ptr %dims_alloca367, i64 0, i64 1
  store i64 12, ptr %dim_ptr369, align 8
  %dims_ptr370 = getelementptr [2 x i64], ptr %dims_alloca367, i64 0, i64 0
  %reshape_dims_res371 = call ptr @tl_tensor_reshape_dims(ptr %dd1366, ptr %dims_ptr370, i64 2)
  store ptr %reshape_dims_res371, ptr %ii1, align 8
  %model372 = load ptr, ptr %model, align 8
  %ii1373 = load ptr, ptr %ii1, align 8
  %call_method374 = call ptr @tl_GPT_forward(ptr %model372, ptr %ii1373)
  call void @tl_mem_register_tensor(ptr %call_method374)
  store ptr %call_method374, ptr %llo1, align 8
  %llo1375 = load ptr, ptr %llo1, align 8
  %dims_alloca376 = alloca [2 x i64], align 8
  %dim_ptr377 = getelementptr [2 x i64], ptr %dims_alloca376, i64 0, i64 0
  store i64 12, ptr %dim_ptr377, align 8
  %dim_ptr378 = getelementptr [2 x i64], ptr %dims_alloca376, i64 0, i64 1
  store i64 13, ptr %dim_ptr378, align 8
  %dims_ptr379 = getelementptr [2 x i64], ptr %dims_alloca376, i64 0, i64 0
  %reshape_dims_res380 = call ptr @tl_tensor_reshape_dims(ptr %llo1375, ptr %dims_ptr379, i64 2)
  store ptr %reshape_dims_res380, ptr %llo1_flat, align 8
  %llo1_flat381 = load ptr, ptr %llo1_flat, align 8
  %slice_res382 = call ptr @tl_tensor_slice(ptr %llo1_flat381, i64 5, i64 1)
  %call_tmp383 = call i64 @argmax(ptr %slice_res382)
  store i64 %call_tmp383, ptr %pp1, align 8
  %pp1384 = load i64, ptr %pp1, align 8
  call void @tl_print_i64(i64 %pp1384)
  %pp1385 = load i64, ptr %pp1, align 8
  %cast_i64_f32387 = sitofp i64 %pp1385 to float
  store float %cast_i64_f32387, ptr %scalar_data386, align 4
  %scalar_tensor389 = call ptr @tl_tensor_new(ptr %scalar_data386, i64 0, ptr %scalar_shape388)
  store float 1.000000e+00, ptr %scalar_data390, align 4
  %scalar_tensor392 = call ptr @tl_tensor_new(ptr %scalar_data390, i64 0, ptr %scalar_shape391)
  %pow_res393 = call ptr @tl_tensor_pow(ptr %scalar_tensor389, ptr %scalar_tensor392)
  %get_res394 = call float @tl_tensor_get(ptr %pow_res393, i64 0)
  store float %get_res394, ptr %vpp1, align 4
  %buf_void395 = call ptr @tl_alloc_tmp(i64 48)
  %z0396 = load float, ptr %z0, align 4
  %elem_ptr397 = getelementptr inbounds float, ptr %buf_void395, i64 0
  store float %z0396, ptr %elem_ptr397, align 4
  %z1398 = load float, ptr %z1, align 4
  %elem_ptr399 = getelementptr inbounds float, ptr %buf_void395, i64 1
  store float %z1398, ptr %elem_ptr399, align 4
  %z2400 = load float, ptr %z2, align 4
  %elem_ptr401 = getelementptr inbounds float, ptr %buf_void395, i64 2
  store float %z2400, ptr %elem_ptr401, align 4
  %z3402 = load float, ptr %z3, align 4
  %elem_ptr403 = getelementptr inbounds float, ptr %buf_void395, i64 3
  store float %z3402, ptr %elem_ptr403, align 4
  %z4404 = load float, ptr %z4, align 4
  %elem_ptr405 = getelementptr inbounds float, ptr %buf_void395, i64 4
  store float %z4404, ptr %elem_ptr405, align 4
  %z5406 = load float, ptr %z5, align 4
  %elem_ptr407 = getelementptr inbounds float, ptr %buf_void395, i64 5
  store float %z5406, ptr %elem_ptr407, align 4
  %vpp1408 = load float, ptr %vpp1, align 4
  %elem_ptr409 = getelementptr inbounds float, ptr %buf_void395, i64 6
  store float %vpp1408, ptr %elem_ptr409, align 4
  %elem_ptr410 = getelementptr inbounds float, ptr %buf_void395, i64 7
  store float 1.200000e+01, ptr %elem_ptr410, align 4
  %elem_ptr411 = getelementptr inbounds float, ptr %buf_void395, i64 8
  store float 1.200000e+01, ptr %elem_ptr411, align 4
  %elem_ptr412 = getelementptr inbounds float, ptr %buf_void395, i64 9
  store float 1.200000e+01, ptr %elem_ptr412, align 4
  %elem_ptr413 = getelementptr inbounds float, ptr %buf_void395, i64 10
  store float 1.200000e+01, ptr %elem_ptr413, align 4
  %elem_ptr414 = getelementptr inbounds float, ptr %buf_void395, i64 11
  store float 1.200000e+01, ptr %elem_ptr414, align 4
  %shape_alloc415 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr416 = getelementptr inbounds i64, ptr %shape_alloc415, i64 0
  store i64 12, ptr %shape_ptr416, align 8
  %new_tensor417 = call ptr @tl_tensor_new(ptr %buf_void395, i64 1, ptr %shape_alloc415)
  call void @tl_free_tmp(ptr %buf_void395)
  call void @tl_free_tmp(ptr %shape_alloc415)
  store ptr %new_tensor417, ptr %dd2, align 8
  %dd2418 = load ptr, ptr %dd2, align 8
  %dims_alloca419 = alloca [2 x i64], align 8
  %dim_ptr420 = getelementptr [2 x i64], ptr %dims_alloca419, i64 0, i64 0
  store i64 1, ptr %dim_ptr420, align 8
  %dim_ptr421 = getelementptr [2 x i64], ptr %dims_alloca419, i64 0, i64 1
  store i64 12, ptr %dim_ptr421, align 8
  %dims_ptr422 = getelementptr [2 x i64], ptr %dims_alloca419, i64 0, i64 0
  %reshape_dims_res423 = call ptr @tl_tensor_reshape_dims(ptr %dd2418, ptr %dims_ptr422, i64 2)
  store ptr %reshape_dims_res423, ptr %ii2, align 8
  %model424 = load ptr, ptr %model, align 8
  %ii2425 = load ptr, ptr %ii2, align 8
  %call_method426 = call ptr @tl_GPT_forward(ptr %model424, ptr %ii2425)
  call void @tl_mem_register_tensor(ptr %call_method426)
  store ptr %call_method426, ptr %llo2, align 8
  %llo2427 = load ptr, ptr %llo2, align 8
  %dims_alloca428 = alloca [2 x i64], align 8
  %dim_ptr429 = getelementptr [2 x i64], ptr %dims_alloca428, i64 0, i64 0
  store i64 12, ptr %dim_ptr429, align 8
  %dim_ptr430 = getelementptr [2 x i64], ptr %dims_alloca428, i64 0, i64 1
  store i64 13, ptr %dim_ptr430, align 8
  %dims_ptr431 = getelementptr [2 x i64], ptr %dims_alloca428, i64 0, i64 0
  %reshape_dims_res432 = call ptr @tl_tensor_reshape_dims(ptr %llo2427, ptr %dims_ptr431, i64 2)
  store ptr %reshape_dims_res432, ptr %llo2_flat, align 8
  %llo2_flat433 = load ptr, ptr %llo2_flat, align 8
  %slice_res434 = call ptr @tl_tensor_slice(ptr %llo2_flat433, i64 6, i64 1)
  %call_tmp435 = call i64 @argmax(ptr %slice_res434)
  store i64 %call_tmp435, ptr %pp2, align 8
  %pp2436 = load i64, ptr %pp2, align 8
  call void @tl_print_i64(i64 %pp2436)
  %pp2437 = load i64, ptr %pp2, align 8
  %cast_i64_f32439 = sitofp i64 %pp2437 to float
  store float %cast_i64_f32439, ptr %scalar_data438, align 4
  %scalar_tensor441 = call ptr @tl_tensor_new(ptr %scalar_data438, i64 0, ptr %scalar_shape440)
  store float 1.000000e+00, ptr %scalar_data442, align 4
  %scalar_tensor444 = call ptr @tl_tensor_new(ptr %scalar_data442, i64 0, ptr %scalar_shape443)
  %pow_res445 = call ptr @tl_tensor_pow(ptr %scalar_tensor441, ptr %scalar_tensor444)
  %get_res446 = call float @tl_tensor_get(ptr %pow_res445, i64 0)
  store float %get_res446, ptr %vpp2, align 4
  %buf_void447 = call ptr @tl_alloc_tmp(i64 48)
  %z0448 = load float, ptr %z0, align 4
  %elem_ptr449 = getelementptr inbounds float, ptr %buf_void447, i64 0
  store float %z0448, ptr %elem_ptr449, align 4
  %z1450 = load float, ptr %z1, align 4
  %elem_ptr451 = getelementptr inbounds float, ptr %buf_void447, i64 1
  store float %z1450, ptr %elem_ptr451, align 4
  %z2452 = load float, ptr %z2, align 4
  %elem_ptr453 = getelementptr inbounds float, ptr %buf_void447, i64 2
  store float %z2452, ptr %elem_ptr453, align 4
  %z3454 = load float, ptr %z3, align 4
  %elem_ptr455 = getelementptr inbounds float, ptr %buf_void447, i64 3
  store float %z3454, ptr %elem_ptr455, align 4
  %z4456 = load float, ptr %z4, align 4
  %elem_ptr457 = getelementptr inbounds float, ptr %buf_void447, i64 4
  store float %z4456, ptr %elem_ptr457, align 4
  %z5458 = load float, ptr %z5, align 4
  %elem_ptr459 = getelementptr inbounds float, ptr %buf_void447, i64 5
  store float %z5458, ptr %elem_ptr459, align 4
  %vpp1460 = load float, ptr %vpp1, align 4
  %elem_ptr461 = getelementptr inbounds float, ptr %buf_void447, i64 6
  store float %vpp1460, ptr %elem_ptr461, align 4
  %vpp2462 = load float, ptr %vpp2, align 4
  %elem_ptr463 = getelementptr inbounds float, ptr %buf_void447, i64 7
  store float %vpp2462, ptr %elem_ptr463, align 4
  %elem_ptr464 = getelementptr inbounds float, ptr %buf_void447, i64 8
  store float 1.200000e+01, ptr %elem_ptr464, align 4
  %elem_ptr465 = getelementptr inbounds float, ptr %buf_void447, i64 9
  store float 1.200000e+01, ptr %elem_ptr465, align 4
  %elem_ptr466 = getelementptr inbounds float, ptr %buf_void447, i64 10
  store float 1.200000e+01, ptr %elem_ptr466, align 4
  %elem_ptr467 = getelementptr inbounds float, ptr %buf_void447, i64 11
  store float 1.200000e+01, ptr %elem_ptr467, align 4
  %shape_alloc468 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr469 = getelementptr inbounds i64, ptr %shape_alloc468, i64 0
  store i64 12, ptr %shape_ptr469, align 8
  %new_tensor470 = call ptr @tl_tensor_new(ptr %buf_void447, i64 1, ptr %shape_alloc468)
  call void @tl_free_tmp(ptr %buf_void447)
  call void @tl_free_tmp(ptr %shape_alloc468)
  store ptr %new_tensor470, ptr %dd3, align 8
  %dd3471 = load ptr, ptr %dd3, align 8
  %dims_alloca472 = alloca [2 x i64], align 8
  %dim_ptr473 = getelementptr [2 x i64], ptr %dims_alloca472, i64 0, i64 0
  store i64 1, ptr %dim_ptr473, align 8
  %dim_ptr474 = getelementptr [2 x i64], ptr %dims_alloca472, i64 0, i64 1
  store i64 12, ptr %dim_ptr474, align 8
  %dims_ptr475 = getelementptr [2 x i64], ptr %dims_alloca472, i64 0, i64 0
  %reshape_dims_res476 = call ptr @tl_tensor_reshape_dims(ptr %dd3471, ptr %dims_ptr475, i64 2)
  store ptr %reshape_dims_res476, ptr %ii3, align 8
  %model477 = load ptr, ptr %model, align 8
  %ii3478 = load ptr, ptr %ii3, align 8
  %call_method479 = call ptr @tl_GPT_forward(ptr %model477, ptr %ii3478)
  call void @tl_mem_register_tensor(ptr %call_method479)
  store ptr %call_method479, ptr %llo3, align 8
  %llo3480 = load ptr, ptr %llo3, align 8
  %dims_alloca481 = alloca [2 x i64], align 8
  %dim_ptr482 = getelementptr [2 x i64], ptr %dims_alloca481, i64 0, i64 0
  store i64 12, ptr %dim_ptr482, align 8
  %dim_ptr483 = getelementptr [2 x i64], ptr %dims_alloca481, i64 0, i64 1
  store i64 13, ptr %dim_ptr483, align 8
  %dims_ptr484 = getelementptr [2 x i64], ptr %dims_alloca481, i64 0, i64 0
  %reshape_dims_res485 = call ptr @tl_tensor_reshape_dims(ptr %llo3480, ptr %dims_ptr484, i64 2)
  store ptr %reshape_dims_res485, ptr %llo3_flat, align 8
  %llo3_flat486 = load ptr, ptr %llo3_flat, align 8
  %slice_res487 = call ptr @tl_tensor_slice(ptr %llo3_flat486, i64 7, i64 1)
  %call_tmp488 = call i64 @argmax(ptr %slice_res487)
  store i64 %call_tmp488, ptr %pp3, align 8
  %pp3489 = load i64, ptr %pp3, align 8
  call void @tl_print_i64(i64 %pp3489)
  call void @tl_print_string(ptr @str_literal.146)
  call void @tl_print_string(ptr @str_literal.147)
  call void @tl_print_string(ptr @str_literal.148)
  call void @tl_print_string(ptr @str_literal.149)
  call void @tl_print_string(ptr @str_literal.150)
  store float 8.000000e+00, ptr %w0, align 4
  store float 8.000000e+00, ptr %w1, align 4
  store float 1.000000e+01, ptr %w2, align 4
  store float 9.000000e+00, ptr %w3, align 4
  store float 9.000000e+00, ptr %w4490, align 4
  store float 1.100000e+01, ptr %w5491, align 4
  %buf_void492 = call ptr @tl_alloc_tmp(i64 48)
  %w0493 = load float, ptr %w0, align 4
  %elem_ptr494 = getelementptr inbounds float, ptr %buf_void492, i64 0
  store float %w0493, ptr %elem_ptr494, align 4
  %w1495 = load float, ptr %w1, align 4
  %elem_ptr496 = getelementptr inbounds float, ptr %buf_void492, i64 1
  store float %w1495, ptr %elem_ptr496, align 4
  %w2497 = load float, ptr %w2, align 4
  %elem_ptr498 = getelementptr inbounds float, ptr %buf_void492, i64 2
  store float %w2497, ptr %elem_ptr498, align 4
  %w3499 = load float, ptr %w3, align 4
  %elem_ptr500 = getelementptr inbounds float, ptr %buf_void492, i64 3
  store float %w3499, ptr %elem_ptr500, align 4
  %w4501 = load float, ptr %w4490, align 4
  %elem_ptr502 = getelementptr inbounds float, ptr %buf_void492, i64 4
  store float %w4501, ptr %elem_ptr502, align 4
  %w5503 = load float, ptr %w5491, align 4
  %elem_ptr504 = getelementptr inbounds float, ptr %buf_void492, i64 5
  store float %w5503, ptr %elem_ptr504, align 4
  %elem_ptr505 = getelementptr inbounds float, ptr %buf_void492, i64 6
  store float 1.200000e+01, ptr %elem_ptr505, align 4
  %elem_ptr506 = getelementptr inbounds float, ptr %buf_void492, i64 7
  store float 1.200000e+01, ptr %elem_ptr506, align 4
  %elem_ptr507 = getelementptr inbounds float, ptr %buf_void492, i64 8
  store float 1.200000e+01, ptr %elem_ptr507, align 4
  %elem_ptr508 = getelementptr inbounds float, ptr %buf_void492, i64 9
  store float 1.200000e+01, ptr %elem_ptr508, align 4
  %elem_ptr509 = getelementptr inbounds float, ptr %buf_void492, i64 10
  store float 1.200000e+01, ptr %elem_ptr509, align 4
  %elem_ptr510 = getelementptr inbounds float, ptr %buf_void492, i64 11
  store float 1.200000e+01, ptr %elem_ptr510, align 4
  %shape_alloc511 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr512 = getelementptr inbounds i64, ptr %shape_alloc511, i64 0
  store i64 12, ptr %shape_ptr512, align 8
  %new_tensor513 = call ptr @tl_tensor_new(ptr %buf_void492, i64 1, ptr %shape_alloc511)
  call void @tl_free_tmp(ptr %buf_void492)
  call void @tl_free_tmp(ptr %shape_alloc511)
  store ptr %new_tensor513, ptr %ee1, align 8
  %ee1514 = load ptr, ptr %ee1, align 8
  %dims_alloca515 = alloca [2 x i64], align 8
  %dim_ptr516 = getelementptr [2 x i64], ptr %dims_alloca515, i64 0, i64 0
  store i64 1, ptr %dim_ptr516, align 8
  %dim_ptr517 = getelementptr [2 x i64], ptr %dims_alloca515, i64 0, i64 1
  store i64 12, ptr %dim_ptr517, align 8
  %dims_ptr518 = getelementptr [2 x i64], ptr %dims_alloca515, i64 0, i64 0
  %reshape_dims_res519 = call ptr @tl_tensor_reshape_dims(ptr %ee1514, ptr %dims_ptr518, i64 2)
  store ptr %reshape_dims_res519, ptr %jj1, align 8
  %model520 = load ptr, ptr %model, align 8
  %jj1521 = load ptr, ptr %jj1, align 8
  %call_method522 = call ptr @tl_GPT_forward(ptr %model520, ptr %jj1521)
  call void @tl_mem_register_tensor(ptr %call_method522)
  store ptr %call_method522, ptr %ll1, align 8
  %ll1523 = load ptr, ptr %ll1, align 8
  %dims_alloca524 = alloca [2 x i64], align 8
  %dim_ptr525 = getelementptr [2 x i64], ptr %dims_alloca524, i64 0, i64 0
  store i64 12, ptr %dim_ptr525, align 8
  %dim_ptr526 = getelementptr [2 x i64], ptr %dims_alloca524, i64 0, i64 1
  store i64 13, ptr %dim_ptr526, align 8
  %dims_ptr527 = getelementptr [2 x i64], ptr %dims_alloca524, i64 0, i64 0
  %reshape_dims_res528 = call ptr @tl_tensor_reshape_dims(ptr %ll1523, ptr %dims_ptr527, i64 2)
  store ptr %reshape_dims_res528, ptr %ll1_flat, align 8
  %ll1_flat529 = load ptr, ptr %ll1_flat, align 8
  %slice_res530 = call ptr @tl_tensor_slice(ptr %ll1_flat529, i64 5, i64 1)
  %call_tmp531 = call i64 @argmax(ptr %slice_res530)
  store i64 %call_tmp531, ptr %qq1, align 8
  %qq1532 = load i64, ptr %qq1, align 8
  call void @tl_print_i64(i64 %qq1532)
  %qq1533 = load i64, ptr %qq1, align 8
  %cast_i64_f32535 = sitofp i64 %qq1533 to float
  store float %cast_i64_f32535, ptr %scalar_data534, align 4
  %scalar_tensor537 = call ptr @tl_tensor_new(ptr %scalar_data534, i64 0, ptr %scalar_shape536)
  store float 1.000000e+00, ptr %scalar_data538, align 4
  %scalar_tensor540 = call ptr @tl_tensor_new(ptr %scalar_data538, i64 0, ptr %scalar_shape539)
  %pow_res541 = call ptr @tl_tensor_pow(ptr %scalar_tensor537, ptr %scalar_tensor540)
  %get_res542 = call float @tl_tensor_get(ptr %pow_res541, i64 0)
  store float %get_res542, ptr %vqq1, align 4
  %buf_void543 = call ptr @tl_alloc_tmp(i64 48)
  %w0544 = load float, ptr %w0, align 4
  %elem_ptr545 = getelementptr inbounds float, ptr %buf_void543, i64 0
  store float %w0544, ptr %elem_ptr545, align 4
  %w1546 = load float, ptr %w1, align 4
  %elem_ptr547 = getelementptr inbounds float, ptr %buf_void543, i64 1
  store float %w1546, ptr %elem_ptr547, align 4
  %w2548 = load float, ptr %w2, align 4
  %elem_ptr549 = getelementptr inbounds float, ptr %buf_void543, i64 2
  store float %w2548, ptr %elem_ptr549, align 4
  %w3550 = load float, ptr %w3, align 4
  %elem_ptr551 = getelementptr inbounds float, ptr %buf_void543, i64 3
  store float %w3550, ptr %elem_ptr551, align 4
  %w4552 = load float, ptr %w4490, align 4
  %elem_ptr553 = getelementptr inbounds float, ptr %buf_void543, i64 4
  store float %w4552, ptr %elem_ptr553, align 4
  %w5554 = load float, ptr %w5491, align 4
  %elem_ptr555 = getelementptr inbounds float, ptr %buf_void543, i64 5
  store float %w5554, ptr %elem_ptr555, align 4
  %vqq1556 = load float, ptr %vqq1, align 4
  %elem_ptr557 = getelementptr inbounds float, ptr %buf_void543, i64 6
  store float %vqq1556, ptr %elem_ptr557, align 4
  %elem_ptr558 = getelementptr inbounds float, ptr %buf_void543, i64 7
  store float 1.200000e+01, ptr %elem_ptr558, align 4
  %elem_ptr559 = getelementptr inbounds float, ptr %buf_void543, i64 8
  store float 1.200000e+01, ptr %elem_ptr559, align 4
  %elem_ptr560 = getelementptr inbounds float, ptr %buf_void543, i64 9
  store float 1.200000e+01, ptr %elem_ptr560, align 4
  %elem_ptr561 = getelementptr inbounds float, ptr %buf_void543, i64 10
  store float 1.200000e+01, ptr %elem_ptr561, align 4
  %elem_ptr562 = getelementptr inbounds float, ptr %buf_void543, i64 11
  store float 1.200000e+01, ptr %elem_ptr562, align 4
  %shape_alloc563 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr564 = getelementptr inbounds i64, ptr %shape_alloc563, i64 0
  store i64 12, ptr %shape_ptr564, align 8
  %new_tensor565 = call ptr @tl_tensor_new(ptr %buf_void543, i64 1, ptr %shape_alloc563)
  call void @tl_free_tmp(ptr %buf_void543)
  call void @tl_free_tmp(ptr %shape_alloc563)
  store ptr %new_tensor565, ptr %ee2, align 8
  %ee2566 = load ptr, ptr %ee2, align 8
  %dims_alloca567 = alloca [2 x i64], align 8
  %dim_ptr568 = getelementptr [2 x i64], ptr %dims_alloca567, i64 0, i64 0
  store i64 1, ptr %dim_ptr568, align 8
  %dim_ptr569 = getelementptr [2 x i64], ptr %dims_alloca567, i64 0, i64 1
  store i64 12, ptr %dim_ptr569, align 8
  %dims_ptr570 = getelementptr [2 x i64], ptr %dims_alloca567, i64 0, i64 0
  %reshape_dims_res571 = call ptr @tl_tensor_reshape_dims(ptr %ee2566, ptr %dims_ptr570, i64 2)
  store ptr %reshape_dims_res571, ptr %jj2, align 8
  %model572 = load ptr, ptr %model, align 8
  %jj2573 = load ptr, ptr %jj2, align 8
  %call_method574 = call ptr @tl_GPT_forward(ptr %model572, ptr %jj2573)
  call void @tl_mem_register_tensor(ptr %call_method574)
  store ptr %call_method574, ptr %ll2, align 8
  %ll2575 = load ptr, ptr %ll2, align 8
  %dims_alloca576 = alloca [2 x i64], align 8
  %dim_ptr577 = getelementptr [2 x i64], ptr %dims_alloca576, i64 0, i64 0
  store i64 12, ptr %dim_ptr577, align 8
  %dim_ptr578 = getelementptr [2 x i64], ptr %dims_alloca576, i64 0, i64 1
  store i64 13, ptr %dim_ptr578, align 8
  %dims_ptr579 = getelementptr [2 x i64], ptr %dims_alloca576, i64 0, i64 0
  %reshape_dims_res580 = call ptr @tl_tensor_reshape_dims(ptr %ll2575, ptr %dims_ptr579, i64 2)
  store ptr %reshape_dims_res580, ptr %ll2_flat, align 8
  %ll2_flat581 = load ptr, ptr %ll2_flat, align 8
  %slice_res582 = call ptr @tl_tensor_slice(ptr %ll2_flat581, i64 6, i64 1)
  %call_tmp583 = call i64 @argmax(ptr %slice_res582)
  store i64 %call_tmp583, ptr %qq2, align 8
  %qq2584 = load i64, ptr %qq2, align 8
  call void @tl_print_i64(i64 %qq2584)
  %qq2585 = load i64, ptr %qq2, align 8
  %cast_i64_f32587 = sitofp i64 %qq2585 to float
  store float %cast_i64_f32587, ptr %scalar_data586, align 4
  %scalar_tensor589 = call ptr @tl_tensor_new(ptr %scalar_data586, i64 0, ptr %scalar_shape588)
  store float 1.000000e+00, ptr %scalar_data590, align 4
  %scalar_tensor592 = call ptr @tl_tensor_new(ptr %scalar_data590, i64 0, ptr %scalar_shape591)
  %pow_res593 = call ptr @tl_tensor_pow(ptr %scalar_tensor589, ptr %scalar_tensor592)
  %get_res594 = call float @tl_tensor_get(ptr %pow_res593, i64 0)
  store float %get_res594, ptr %vqq2, align 4
  %buf_void595 = call ptr @tl_alloc_tmp(i64 48)
  %w0596 = load float, ptr %w0, align 4
  %elem_ptr597 = getelementptr inbounds float, ptr %buf_void595, i64 0
  store float %w0596, ptr %elem_ptr597, align 4
  %w1598 = load float, ptr %w1, align 4
  %elem_ptr599 = getelementptr inbounds float, ptr %buf_void595, i64 1
  store float %w1598, ptr %elem_ptr599, align 4
  %w2600 = load float, ptr %w2, align 4
  %elem_ptr601 = getelementptr inbounds float, ptr %buf_void595, i64 2
  store float %w2600, ptr %elem_ptr601, align 4
  %w3602 = load float, ptr %w3, align 4
  %elem_ptr603 = getelementptr inbounds float, ptr %buf_void595, i64 3
  store float %w3602, ptr %elem_ptr603, align 4
  %w4604 = load float, ptr %w4490, align 4
  %elem_ptr605 = getelementptr inbounds float, ptr %buf_void595, i64 4
  store float %w4604, ptr %elem_ptr605, align 4
  %w5606 = load float, ptr %w5491, align 4
  %elem_ptr607 = getelementptr inbounds float, ptr %buf_void595, i64 5
  store float %w5606, ptr %elem_ptr607, align 4
  %vqq1608 = load float, ptr %vqq1, align 4
  %elem_ptr609 = getelementptr inbounds float, ptr %buf_void595, i64 6
  store float %vqq1608, ptr %elem_ptr609, align 4
  %vqq2610 = load float, ptr %vqq2, align 4
  %elem_ptr611 = getelementptr inbounds float, ptr %buf_void595, i64 7
  store float %vqq2610, ptr %elem_ptr611, align 4
  %elem_ptr612 = getelementptr inbounds float, ptr %buf_void595, i64 8
  store float 1.200000e+01, ptr %elem_ptr612, align 4
  %elem_ptr613 = getelementptr inbounds float, ptr %buf_void595, i64 9
  store float 1.200000e+01, ptr %elem_ptr613, align 4
  %elem_ptr614 = getelementptr inbounds float, ptr %buf_void595, i64 10
  store float 1.200000e+01, ptr %elem_ptr614, align 4
  %elem_ptr615 = getelementptr inbounds float, ptr %buf_void595, i64 11
  store float 1.200000e+01, ptr %elem_ptr615, align 4
  %shape_alloc616 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr617 = getelementptr inbounds i64, ptr %shape_alloc616, i64 0
  store i64 12, ptr %shape_ptr617, align 8
  %new_tensor618 = call ptr @tl_tensor_new(ptr %buf_void595, i64 1, ptr %shape_alloc616)
  call void @tl_free_tmp(ptr %buf_void595)
  call void @tl_free_tmp(ptr %shape_alloc616)
  store ptr %new_tensor618, ptr %ee3, align 8
  %ee3619 = load ptr, ptr %ee3, align 8
  %dims_alloca620 = alloca [2 x i64], align 8
  %dim_ptr621 = getelementptr [2 x i64], ptr %dims_alloca620, i64 0, i64 0
  store i64 1, ptr %dim_ptr621, align 8
  %dim_ptr622 = getelementptr [2 x i64], ptr %dims_alloca620, i64 0, i64 1
  store i64 12, ptr %dim_ptr622, align 8
  %dims_ptr623 = getelementptr [2 x i64], ptr %dims_alloca620, i64 0, i64 0
  %reshape_dims_res624 = call ptr @tl_tensor_reshape_dims(ptr %ee3619, ptr %dims_ptr623, i64 2)
  store ptr %reshape_dims_res624, ptr %jj3, align 8
  %model625 = load ptr, ptr %model, align 8
  %jj3626 = load ptr, ptr %jj3, align 8
  %call_method627 = call ptr @tl_GPT_forward(ptr %model625, ptr %jj3626)
  call void @tl_mem_register_tensor(ptr %call_method627)
  store ptr %call_method627, ptr %ll3, align 8
  %ll3628 = load ptr, ptr %ll3, align 8
  %dims_alloca629 = alloca [2 x i64], align 8
  %dim_ptr630 = getelementptr [2 x i64], ptr %dims_alloca629, i64 0, i64 0
  store i64 12, ptr %dim_ptr630, align 8
  %dim_ptr631 = getelementptr [2 x i64], ptr %dims_alloca629, i64 0, i64 1
  store i64 13, ptr %dim_ptr631, align 8
  %dims_ptr632 = getelementptr [2 x i64], ptr %dims_alloca629, i64 0, i64 0
  %reshape_dims_res633 = call ptr @tl_tensor_reshape_dims(ptr %ll3628, ptr %dims_ptr632, i64 2)
  store ptr %reshape_dims_res633, ptr %ll3_flat, align 8
  %ll3_flat634 = load ptr, ptr %ll3_flat, align 8
  %slice_res635 = call ptr @tl_tensor_slice(ptr %ll3_flat634, i64 7, i64 1)
  %call_tmp636 = call i64 @argmax(ptr %slice_res635)
  store i64 %call_tmp636, ptr %qq3, align 8
  %qq3637 = load i64, ptr %qq3, align 8
  call void @tl_print_i64(i64 %qq3637)
  call void @tl_print_string(ptr @str_literal.151)
  call void @tl_mem_exit_scope()
  ret void
}
