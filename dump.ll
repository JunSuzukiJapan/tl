; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

%Linear = type { ptr, ptr }
%Embedding = type { ptr }
%LayerNorm = type { ptr, ptr }
%CausalSelfAttention = type { ptr, ptr, ptr, ptr }
%MLP = type { ptr, ptr }
%Block = type { ptr, ptr, ptr, ptr }
%GPT = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr }

@str_literal = private unnamed_addr constant [15 x i8] c"Argmax result:\00", align 1
@str_literal.104 = private unnamed_addr constant [5 x i8] c"Val:\00", align 1
@str_literal.105 = private unnamed_addr constant [36 x i8] c"Initializing Model for Inference...\00", align 1
@str_literal.106 = private unnamed_addr constant [46 x i8] c"Initializing Runtime: Metal backend selected.\00", align 1
@str_literal.107 = private unnamed_addr constant [22 x i8] c"Loading Parameters...\00", align 1
@key_str = private unnamed_addr constant [4 x i8] c"w.w\00", align 1
@key_str.108 = private unnamed_addr constant [5 x i8] c"wp.w\00", align 1
@key_str.109 = private unnamed_addr constant [8 x i8] c"b1.l1.w\00", align 1
@key_str.110 = private unnamed_addr constant [8 x i8] c"b1.l1.b\00", align 1
@key_str.111 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.W\00", align 1
@key_str.112 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.b\00", align 1
@key_str.113 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.W\00", align 1
@key_str.114 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.b\00", align 1
@key_str.115 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.W\00", align 1
@key_str.116 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.b\00", align 1
@key_str.117 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.W\00", align 1
@key_str.118 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.b\00", align 1
@key_str.119 = private unnamed_addr constant [8 x i8] c"b1.l2.w\00", align 1
@key_str.120 = private unnamed_addr constant [8 x i8] c"b1.l2.b\00", align 1
@key_str.121 = private unnamed_addr constant [9 x i8] c"b1.m.f.W\00", align 1
@key_str.122 = private unnamed_addr constant [9 x i8] c"b1.m.f.b\00", align 1
@key_str.123 = private unnamed_addr constant [9 x i8] c"b1.m.p.W\00", align 1
@key_str.124 = private unnamed_addr constant [9 x i8] c"b1.m.p.b\00", align 1
@key_str.125 = private unnamed_addr constant [8 x i8] c"b2.l1.w\00", align 1
@key_str.126 = private unnamed_addr constant [8 x i8] c"b2.l1.b\00", align 1
@key_str.127 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.W\00", align 1
@key_str.128 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.b\00", align 1
@key_str.129 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.W\00", align 1
@key_str.130 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.b\00", align 1
@key_str.131 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.W\00", align 1
@key_str.132 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.b\00", align 1
@key_str.133 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.W\00", align 1
@key_str.134 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.b\00", align 1
@key_str.135 = private unnamed_addr constant [8 x i8] c"b2.l2.w\00", align 1
@key_str.136 = private unnamed_addr constant [8 x i8] c"b2.l2.b\00", align 1
@key_str.137 = private unnamed_addr constant [9 x i8] c"b2.m.f.W\00", align 1
@key_str.138 = private unnamed_addr constant [9 x i8] c"b2.m.f.b\00", align 1
@key_str.139 = private unnamed_addr constant [9 x i8] c"b2.m.p.W\00", align 1
@key_str.140 = private unnamed_addr constant [9 x i8] c"b2.m.p.b\00", align 1
@key_str.141 = private unnamed_addr constant [8 x i8] c"b3.l1.w\00", align 1
@key_str.142 = private unnamed_addr constant [8 x i8] c"b3.l1.b\00", align 1
@key_str.143 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.W\00", align 1
@key_str.144 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.b\00", align 1
@key_str.145 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.W\00", align 1
@key_str.146 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.b\00", align 1
@key_str.147 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.W\00", align 1
@key_str.148 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.b\00", align 1
@key_str.149 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.W\00", align 1
@key_str.150 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.b\00", align 1
@key_str.151 = private unnamed_addr constant [8 x i8] c"b3.l2.w\00", align 1
@key_str.152 = private unnamed_addr constant [8 x i8] c"b3.l2.b\00", align 1
@key_str.153 = private unnamed_addr constant [9 x i8] c"b3.m.f.W\00", align 1
@key_str.154 = private unnamed_addr constant [9 x i8] c"b3.m.f.b\00", align 1
@key_str.155 = private unnamed_addr constant [9 x i8] c"b3.m.p.W\00", align 1
@key_str.156 = private unnamed_addr constant [9 x i8] c"b3.m.p.b\00", align 1
@key_str.157 = private unnamed_addr constant [4 x i8] c"l.w\00", align 1
@key_str.158 = private unnamed_addr constant [4 x i8] c"l.b\00", align 1
@key_str.159 = private unnamed_addr constant [4 x i8] c"h.W\00", align 1
@key_str.160 = private unnamed_addr constant [4 x i8] c"h.b\00", align 1
@str_literal.161 = private unnamed_addr constant [29 x i8] c"model_2digit_rev.safetensors\00", align 1
@str_literal.162 = private unnamed_addr constant [19 x i8] c"Parameters Loaded.\00", align 1
@str_literal.163 = private unnamed_addr constant [14 x i8] c"Test: 12 + 34\00", align 1
@str_literal.164 = private unnamed_addr constant [21 x i8] c"Predicted (Reverse):\00", align 1
@str_literal.165 = private unnamed_addr constant [13 x i8] c"Test: 99 + 1\00", align 1
@str_literal.166 = private unnamed_addr constant [21 x i8] c"Predicted (Reverse):\00", align 1
@str_literal.167 = private unnamed_addr constant [33 x i8] c"Inference Verification Complete.\00", align 1

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
  call void @tl_tensor_free(ptr %static_call)
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
  call void @tl_tensor_free(ptr %static_call14)
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
  call void @tl_tensor_free(ptr %matmul_res)
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
  call void @tl_mem_unregister(ptr %grad_res)
  store ptr %grad_res, ptr %gW, align 8
  %s5 = load ptr, ptr %s, align 8
  %ptr_b = getelementptr inbounds %Linear, ptr %s5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res6 = call ptr @tl_tensor_grad(ptr %b)
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
  %binop_res14 = call ptr @tl_tensor_sub(ptr %W11, ptr %binop_res)
  call void @tl_tensor_free(ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res14, i1 true)
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
  %binop_res26 = call ptr @tl_tensor_sub(ptr %b19, ptr %binop_res25)
  call void @tl_tensor_free(ptr %binop_res25)
  %detach_res27 = call ptr @tl_tensor_detach(ptr %binop_res26, i1 true)
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
  call void @tl_mem_exit_scope()
  ret ptr %s32
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
  call void @tl_tensor_free(ptr %static_call)
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
  %binop_res12 = call ptr @tl_tensor_sub(ptr %w9, ptr %binop_res)
  call void @tl_tensor_free(ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res12, i1 true)
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
  call void @tl_tensor_free(ptr %static_call)
  store float 1.000000e+00, ptr %scalar_data_rhs3, align 4
  %scalar_tensor_rhs5 = call ptr @tl_tensor_new(ptr %scalar_data_rhs3, i64 0, ptr %scalar_shape_rhs4)
  %binop_res6 = call ptr @tl_tensor_add(ptr %binop_res, ptr %scalar_tensor_rhs5)
  call void @tl_tensor_free(ptr %binop_res)
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
  call void @tl_tensor_free(ptr %static_call14)
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
  %binop_res12 = call ptr @tl_tensor_sub(ptr %b9, ptr %binop_res)
  call void @tl_tensor_free(ptr %binop_res)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res12, i1 true)
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
  store float 0x3FB6A0BA20000000, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %matmul_res, ptr %scalar_tensor_rhs)
  call void @tl_tensor_free(ptr %matmul_res)
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
  %is_not_null = icmp ne ptr %old_field_val, null
  br i1 %is_not_null, label %recursive_free_struct, label %continue_after_recursive_free

skip_free:                                        ; preds = %continue_after_recursive_free, %entry
  store ptr %call_method, ptr %ptr_q_proj, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s9 = load ptr, ptr %s, align 8
  %ptr_k_proj = getelementptr inbounds %CausalSelfAttention, ptr %s9, i32 0, i32 1
  %s10 = load ptr, ptr %s, align 8
  %ptr_k_proj11 = getelementptr inbounds %CausalSelfAttention, ptr %s10, i32 0, i32 1
  %k_proj = load ptr, ptr %ptr_k_proj11, align 8
  %lr12 = load float, ptr %lr2, align 4
  %call_method13 = call ptr @tl_Linear_step(ptr %k_proj, float %lr12)
  call void @tl_mem_register_struct(ptr %call_method13)
  %old_field_val14 = load ptr, ptr %ptr_k_proj, align 8
  %cnt_free_diff15 = icmp ne ptr %old_field_val14, %call_method13
  br i1 %cnt_free_diff15, label %free_old_val16, label %skip_free17

recursive_free_struct:                            ; preds = %free_old_val
  %free_field_0 = getelementptr inbounds %Linear, ptr %old_field_val, i32 0, i32 0
  %field_val_to_free = load ptr, ptr %free_field_0, align 8
  call void @tl_tensor_free(ptr %field_val_to_free)
  %free_field_1 = getelementptr inbounds %Linear, ptr %old_field_val, i32 0, i32 1
  %field_val_to_free8 = load ptr, ptr %free_field_1, align 8
  call void @tl_tensor_free(ptr %field_val_to_free8)
  call void @free(ptr %old_field_val)
  br label %continue_after_recursive_free

continue_after_recursive_free:                    ; preds = %recursive_free_struct, %free_old_val
  br label %skip_free

free_old_val16:                                   ; preds = %skip_free
  %is_not_null18 = icmp ne ptr %old_field_val14, null
  br i1 %is_not_null18, label %recursive_free_struct19, label %continue_after_recursive_free20

skip_free17:                                      ; preds = %continue_after_recursive_free20, %skip_free
  store ptr %call_method13, ptr %ptr_k_proj, align 8
  call void @tl_mem_unregister(ptr %call_method13)
  %s25 = load ptr, ptr %s, align 8
  %ptr_v_proj = getelementptr inbounds %CausalSelfAttention, ptr %s25, i32 0, i32 2
  %s26 = load ptr, ptr %s, align 8
  %ptr_v_proj27 = getelementptr inbounds %CausalSelfAttention, ptr %s26, i32 0, i32 2
  %v_proj = load ptr, ptr %ptr_v_proj27, align 8
  %lr28 = load float, ptr %lr2, align 4
  %call_method29 = call ptr @tl_Linear_step(ptr %v_proj, float %lr28)
  call void @tl_mem_register_struct(ptr %call_method29)
  %old_field_val30 = load ptr, ptr %ptr_v_proj, align 8
  %cnt_free_diff31 = icmp ne ptr %old_field_val30, %call_method29
  br i1 %cnt_free_diff31, label %free_old_val32, label %skip_free33

recursive_free_struct19:                          ; preds = %free_old_val16
  %free_field_021 = getelementptr inbounds %Linear, ptr %old_field_val14, i32 0, i32 0
  %field_val_to_free22 = load ptr, ptr %free_field_021, align 8
  call void @tl_tensor_free(ptr %field_val_to_free22)
  %free_field_123 = getelementptr inbounds %Linear, ptr %old_field_val14, i32 0, i32 1
  %field_val_to_free24 = load ptr, ptr %free_field_123, align 8
  call void @tl_tensor_free(ptr %field_val_to_free24)
  call void @free(ptr %old_field_val14)
  br label %continue_after_recursive_free20

continue_after_recursive_free20:                  ; preds = %recursive_free_struct19, %free_old_val16
  br label %skip_free17

free_old_val32:                                   ; preds = %skip_free17
  %is_not_null34 = icmp ne ptr %old_field_val30, null
  br i1 %is_not_null34, label %recursive_free_struct35, label %continue_after_recursive_free36

skip_free33:                                      ; preds = %continue_after_recursive_free36, %skip_free17
  store ptr %call_method29, ptr %ptr_v_proj, align 8
  call void @tl_mem_unregister(ptr %call_method29)
  %s41 = load ptr, ptr %s, align 8
  %ptr_p_proj = getelementptr inbounds %CausalSelfAttention, ptr %s41, i32 0, i32 3
  %s42 = load ptr, ptr %s, align 8
  %ptr_p_proj43 = getelementptr inbounds %CausalSelfAttention, ptr %s42, i32 0, i32 3
  %p_proj = load ptr, ptr %ptr_p_proj43, align 8
  %lr44 = load float, ptr %lr2, align 4
  %call_method45 = call ptr @tl_Linear_step(ptr %p_proj, float %lr44)
  call void @tl_mem_register_struct(ptr %call_method45)
  %old_field_val46 = load ptr, ptr %ptr_p_proj, align 8
  %cnt_free_diff47 = icmp ne ptr %old_field_val46, %call_method45
  br i1 %cnt_free_diff47, label %free_old_val48, label %skip_free49

recursive_free_struct35:                          ; preds = %free_old_val32
  %free_field_037 = getelementptr inbounds %Linear, ptr %old_field_val30, i32 0, i32 0
  %field_val_to_free38 = load ptr, ptr %free_field_037, align 8
  call void @tl_tensor_free(ptr %field_val_to_free38)
  %free_field_139 = getelementptr inbounds %Linear, ptr %old_field_val30, i32 0, i32 1
  %field_val_to_free40 = load ptr, ptr %free_field_139, align 8
  call void @tl_tensor_free(ptr %field_val_to_free40)
  call void @free(ptr %old_field_val30)
  br label %continue_after_recursive_free36

continue_after_recursive_free36:                  ; preds = %recursive_free_struct35, %free_old_val32
  br label %skip_free33

free_old_val48:                                   ; preds = %skip_free33
  %is_not_null50 = icmp ne ptr %old_field_val46, null
  br i1 %is_not_null50, label %recursive_free_struct51, label %continue_after_recursive_free52

skip_free49:                                      ; preds = %continue_after_recursive_free52, %skip_free33
  store ptr %call_method45, ptr %ptr_p_proj, align 8
  call void @tl_mem_unregister(ptr %call_method45)
  %s57 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s57)
  %unreg_field_0 = getelementptr inbounds %CausalSelfAttention, ptr %s57, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_058 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 0
  %field_val59 = load ptr, ptr %unreg_field_058, align 8
  call void @tl_mem_unregister(ptr %field_val59)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 1
  %field_val60 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val60)
  %unreg_field_161 = getelementptr inbounds %CausalSelfAttention, ptr %s57, i32 0, i32 1
  %field_val62 = load ptr, ptr %unreg_field_161, align 8
  call void @tl_mem_unregister(ptr %field_val62)
  %unreg_field_063 = getelementptr inbounds %Linear, ptr %field_val62, i32 0, i32 0
  %field_val64 = load ptr, ptr %unreg_field_063, align 8
  call void @tl_mem_unregister(ptr %field_val64)
  %unreg_field_165 = getelementptr inbounds %Linear, ptr %field_val62, i32 0, i32 1
  %field_val66 = load ptr, ptr %unreg_field_165, align 8
  call void @tl_mem_unregister(ptr %field_val66)
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %s57, i32 0, i32 2
  %field_val67 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val67)
  %unreg_field_068 = getelementptr inbounds %Linear, ptr %field_val67, i32 0, i32 0
  %field_val69 = load ptr, ptr %unreg_field_068, align 8
  call void @tl_mem_unregister(ptr %field_val69)
  %unreg_field_170 = getelementptr inbounds %Linear, ptr %field_val67, i32 0, i32 1
  %field_val71 = load ptr, ptr %unreg_field_170, align 8
  call void @tl_mem_unregister(ptr %field_val71)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %s57, i32 0, i32 3
  %field_val72 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val72)
  %unreg_field_073 = getelementptr inbounds %Linear, ptr %field_val72, i32 0, i32 0
  %field_val74 = load ptr, ptr %unreg_field_073, align 8
  call void @tl_mem_unregister(ptr %field_val74)
  %unreg_field_175 = getelementptr inbounds %Linear, ptr %field_val72, i32 0, i32 1
  %field_val76 = load ptr, ptr %unreg_field_175, align 8
  call void @tl_mem_unregister(ptr %field_val76)
  call void @tl_mem_exit_scope()
  ret ptr %s57

recursive_free_struct51:                          ; preds = %free_old_val48
  %free_field_053 = getelementptr inbounds %Linear, ptr %old_field_val46, i32 0, i32 0
  %field_val_to_free54 = load ptr, ptr %free_field_053, align 8
  call void @tl_tensor_free(ptr %field_val_to_free54)
  %free_field_155 = getelementptr inbounds %Linear, ptr %old_field_val46, i32 0, i32 1
  %field_val_to_free56 = load ptr, ptr %free_field_155, align 8
  call void @tl_tensor_free(ptr %field_val_to_free56)
  call void @free(ptr %old_field_val46)
  br label %continue_after_recursive_free52

continue_after_recursive_free52:                  ; preds = %recursive_free_struct51, %free_old_val48
  br label %skip_free49
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
  %is_not_null = icmp ne ptr %old_field_val, null
  br i1 %is_not_null, label %recursive_free_struct, label %continue_after_recursive_free

skip_free:                                        ; preds = %continue_after_recursive_free, %entry
  store ptr %call_method, ptr %ptr_f, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s9 = load ptr, ptr %s, align 8
  %ptr_p = getelementptr inbounds %MLP, ptr %s9, i32 0, i32 1
  %s10 = load ptr, ptr %s, align 8
  %ptr_p11 = getelementptr inbounds %MLP, ptr %s10, i32 0, i32 1
  %p = load ptr, ptr %ptr_p11, align 8
  %lr12 = load float, ptr %lr2, align 4
  %call_method13 = call ptr @tl_Linear_step(ptr %p, float %lr12)
  call void @tl_mem_register_struct(ptr %call_method13)
  %old_field_val14 = load ptr, ptr %ptr_p, align 8
  %cnt_free_diff15 = icmp ne ptr %old_field_val14, %call_method13
  br i1 %cnt_free_diff15, label %free_old_val16, label %skip_free17

recursive_free_struct:                            ; preds = %free_old_val
  %free_field_0 = getelementptr inbounds %Linear, ptr %old_field_val, i32 0, i32 0
  %field_val_to_free = load ptr, ptr %free_field_0, align 8
  call void @tl_tensor_free(ptr %field_val_to_free)
  %free_field_1 = getelementptr inbounds %Linear, ptr %old_field_val, i32 0, i32 1
  %field_val_to_free8 = load ptr, ptr %free_field_1, align 8
  call void @tl_tensor_free(ptr %field_val_to_free8)
  call void @free(ptr %old_field_val)
  br label %continue_after_recursive_free

continue_after_recursive_free:                    ; preds = %recursive_free_struct, %free_old_val
  br label %skip_free

free_old_val16:                                   ; preds = %skip_free
  %is_not_null18 = icmp ne ptr %old_field_val14, null
  br i1 %is_not_null18, label %recursive_free_struct19, label %continue_after_recursive_free20

skip_free17:                                      ; preds = %continue_after_recursive_free20, %skip_free
  store ptr %call_method13, ptr %ptr_p, align 8
  call void @tl_mem_unregister(ptr %call_method13)
  %s25 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s25)
  %unreg_field_0 = getelementptr inbounds %MLP, ptr %s25, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_026 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 0
  %field_val27 = load ptr, ptr %unreg_field_026, align 8
  call void @tl_mem_unregister(ptr %field_val27)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 1
  %field_val28 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_129 = getelementptr inbounds %MLP, ptr %s25, i32 0, i32 1
  %field_val30 = load ptr, ptr %unreg_field_129, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  %unreg_field_031 = getelementptr inbounds %Linear, ptr %field_val30, i32 0, i32 0
  %field_val32 = load ptr, ptr %unreg_field_031, align 8
  call void @tl_mem_unregister(ptr %field_val32)
  %unreg_field_133 = getelementptr inbounds %Linear, ptr %field_val30, i32 0, i32 1
  %field_val34 = load ptr, ptr %unreg_field_133, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  call void @tl_mem_exit_scope()
  ret ptr %s25

recursive_free_struct19:                          ; preds = %free_old_val16
  %free_field_021 = getelementptr inbounds %Linear, ptr %old_field_val14, i32 0, i32 0
  %field_val_to_free22 = load ptr, ptr %free_field_021, align 8
  call void @tl_tensor_free(ptr %field_val_to_free22)
  %free_field_123 = getelementptr inbounds %Linear, ptr %old_field_val14, i32 0, i32 1
  %field_val_to_free24 = load ptr, ptr %free_field_123, align 8
  call void @tl_tensor_free(ptr %field_val_to_free24)
  call void @free(ptr %old_field_val14)
  br label %continue_after_recursive_free20

continue_after_recursive_free20:                  ; preds = %recursive_free_struct19, %free_old_val16
  br label %skip_free17
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
  call void @tl_tensor_free(ptr %call_method7)
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
  call void @tl_tensor_free(ptr %call_method14)
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
  call void @tl_mem_register_struct(ptr %call_method)
  %old_field_val = load ptr, ptr %ptr_l1, align 8
  %cnt_free_diff = icmp ne ptr %old_field_val, %call_method
  br i1 %cnt_free_diff, label %free_old_val, label %skip_free

free_old_val:                                     ; preds = %entry
  %is_not_null = icmp ne ptr %old_field_val, null
  br i1 %is_not_null, label %recursive_free_struct, label %continue_after_recursive_free

skip_free:                                        ; preds = %continue_after_recursive_free, %entry
  store ptr %call_method, ptr %ptr_l1, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s9 = load ptr, ptr %s, align 8
  %ptr_a = getelementptr inbounds %Block, ptr %s9, i32 0, i32 1
  %s10 = load ptr, ptr %s, align 8
  %ptr_a11 = getelementptr inbounds %Block, ptr %s10, i32 0, i32 1
  %a = load ptr, ptr %ptr_a11, align 8
  %lr12 = load float, ptr %lr2, align 4
  %call_method13 = call ptr @tl_CausalSelfAttention_step(ptr %a, float %lr12)
  call void @tl_mem_register_struct(ptr %call_method13)
  %old_field_val14 = load ptr, ptr %ptr_a, align 8
  %cnt_free_diff15 = icmp ne ptr %old_field_val14, %call_method13
  br i1 %cnt_free_diff15, label %free_old_val16, label %skip_free17

recursive_free_struct:                            ; preds = %free_old_val
  %free_field_0 = getelementptr inbounds %LayerNorm, ptr %old_field_val, i32 0, i32 0
  %field_val_to_free = load ptr, ptr %free_field_0, align 8
  call void @tl_tensor_free(ptr %field_val_to_free)
  %free_field_1 = getelementptr inbounds %LayerNorm, ptr %old_field_val, i32 0, i32 1
  %field_val_to_free8 = load ptr, ptr %free_field_1, align 8
  call void @tl_tensor_free(ptr %field_val_to_free8)
  call void @free(ptr %old_field_val)
  br label %continue_after_recursive_free

continue_after_recursive_free:                    ; preds = %recursive_free_struct, %free_old_val
  br label %skip_free

free_old_val16:                                   ; preds = %skip_free
  %is_not_null18 = icmp ne ptr %old_field_val14, null
  br i1 %is_not_null18, label %recursive_free_struct19, label %continue_after_recursive_free20

skip_free17:                                      ; preds = %continue_after_recursive_free20, %skip_free
  store ptr %call_method13, ptr %ptr_a, align 8
  call void @tl_mem_unregister(ptr %call_method13)
  %s55 = load ptr, ptr %s, align 8
  %ptr_l2 = getelementptr inbounds %Block, ptr %s55, i32 0, i32 2
  %s56 = load ptr, ptr %s, align 8
  %ptr_l257 = getelementptr inbounds %Block, ptr %s56, i32 0, i32 2
  %l2 = load ptr, ptr %ptr_l257, align 8
  %lr58 = load float, ptr %lr2, align 4
  %call_method59 = call ptr @tl_LayerNorm_step(ptr %l2, float %lr58)
  call void @tl_mem_register_struct(ptr %call_method59)
  %old_field_val60 = load ptr, ptr %ptr_l2, align 8
  %cnt_free_diff61 = icmp ne ptr %old_field_val60, %call_method59
  br i1 %cnt_free_diff61, label %free_old_val62, label %skip_free63

recursive_free_struct19:                          ; preds = %free_old_val16
  %free_field_021 = getelementptr inbounds %CausalSelfAttention, ptr %old_field_val14, i32 0, i32 0
  %field_val_to_free22 = load ptr, ptr %free_field_021, align 8
  %is_not_null23 = icmp ne ptr %field_val_to_free22, null
  br i1 %is_not_null23, label %recursive_free_struct24, label %continue_after_recursive_free25

continue_after_recursive_free20:                  ; preds = %continue_after_recursive_free50, %free_old_val16
  br label %skip_free17

recursive_free_struct24:                          ; preds = %recursive_free_struct19
  %free_field_026 = getelementptr inbounds %Linear, ptr %field_val_to_free22, i32 0, i32 0
  %field_val_to_free27 = load ptr, ptr %free_field_026, align 8
  call void @tl_tensor_free(ptr %field_val_to_free27)
  %free_field_128 = getelementptr inbounds %Linear, ptr %field_val_to_free22, i32 0, i32 1
  %field_val_to_free29 = load ptr, ptr %free_field_128, align 8
  call void @tl_tensor_free(ptr %field_val_to_free29)
  call void @free(ptr %field_val_to_free22)
  br label %continue_after_recursive_free25

continue_after_recursive_free25:                  ; preds = %recursive_free_struct24, %recursive_free_struct19
  %free_field_130 = getelementptr inbounds %CausalSelfAttention, ptr %old_field_val14, i32 0, i32 1
  %field_val_to_free31 = load ptr, ptr %free_field_130, align 8
  %is_not_null32 = icmp ne ptr %field_val_to_free31, null
  br i1 %is_not_null32, label %recursive_free_struct33, label %continue_after_recursive_free34

recursive_free_struct33:                          ; preds = %continue_after_recursive_free25
  %free_field_035 = getelementptr inbounds %Linear, ptr %field_val_to_free31, i32 0, i32 0
  %field_val_to_free36 = load ptr, ptr %free_field_035, align 8
  call void @tl_tensor_free(ptr %field_val_to_free36)
  %free_field_137 = getelementptr inbounds %Linear, ptr %field_val_to_free31, i32 0, i32 1
  %field_val_to_free38 = load ptr, ptr %free_field_137, align 8
  call void @tl_tensor_free(ptr %field_val_to_free38)
  call void @free(ptr %field_val_to_free31)
  br label %continue_after_recursive_free34

continue_after_recursive_free34:                  ; preds = %recursive_free_struct33, %continue_after_recursive_free25
  %free_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %old_field_val14, i32 0, i32 2
  %field_val_to_free39 = load ptr, ptr %free_field_2, align 8
  %is_not_null40 = icmp ne ptr %field_val_to_free39, null
  br i1 %is_not_null40, label %recursive_free_struct41, label %continue_after_recursive_free42

recursive_free_struct41:                          ; preds = %continue_after_recursive_free34
  %free_field_043 = getelementptr inbounds %Linear, ptr %field_val_to_free39, i32 0, i32 0
  %field_val_to_free44 = load ptr, ptr %free_field_043, align 8
  call void @tl_tensor_free(ptr %field_val_to_free44)
  %free_field_145 = getelementptr inbounds %Linear, ptr %field_val_to_free39, i32 0, i32 1
  %field_val_to_free46 = load ptr, ptr %free_field_145, align 8
  call void @tl_tensor_free(ptr %field_val_to_free46)
  call void @free(ptr %field_val_to_free39)
  br label %continue_after_recursive_free42

continue_after_recursive_free42:                  ; preds = %recursive_free_struct41, %continue_after_recursive_free34
  %free_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %old_field_val14, i32 0, i32 3
  %field_val_to_free47 = load ptr, ptr %free_field_3, align 8
  %is_not_null48 = icmp ne ptr %field_val_to_free47, null
  br i1 %is_not_null48, label %recursive_free_struct49, label %continue_after_recursive_free50

recursive_free_struct49:                          ; preds = %continue_after_recursive_free42
  %free_field_051 = getelementptr inbounds %Linear, ptr %field_val_to_free47, i32 0, i32 0
  %field_val_to_free52 = load ptr, ptr %free_field_051, align 8
  call void @tl_tensor_free(ptr %field_val_to_free52)
  %free_field_153 = getelementptr inbounds %Linear, ptr %field_val_to_free47, i32 0, i32 1
  %field_val_to_free54 = load ptr, ptr %free_field_153, align 8
  call void @tl_tensor_free(ptr %field_val_to_free54)
  call void @free(ptr %field_val_to_free47)
  br label %continue_after_recursive_free50

continue_after_recursive_free50:                  ; preds = %recursive_free_struct49, %continue_after_recursive_free42
  call void @free(ptr %old_field_val14)
  br label %continue_after_recursive_free20

free_old_val62:                                   ; preds = %skip_free17
  %is_not_null64 = icmp ne ptr %old_field_val60, null
  br i1 %is_not_null64, label %recursive_free_struct65, label %continue_after_recursive_free66

skip_free63:                                      ; preds = %continue_after_recursive_free66, %skip_free17
  store ptr %call_method59, ptr %ptr_l2, align 8
  call void @tl_mem_unregister(ptr %call_method59)
  %s71 = load ptr, ptr %s, align 8
  %ptr_m = getelementptr inbounds %Block, ptr %s71, i32 0, i32 3
  %s72 = load ptr, ptr %s, align 8
  %ptr_m73 = getelementptr inbounds %Block, ptr %s72, i32 0, i32 3
  %m = load ptr, ptr %ptr_m73, align 8
  %lr74 = load float, ptr %lr2, align 4
  %call_method75 = call ptr @tl_MLP_step(ptr %m, float %lr74)
  call void @tl_mem_register_struct(ptr %call_method75)
  %old_field_val76 = load ptr, ptr %ptr_m, align 8
  %cnt_free_diff77 = icmp ne ptr %old_field_val76, %call_method75
  br i1 %cnt_free_diff77, label %free_old_val78, label %skip_free79

recursive_free_struct65:                          ; preds = %free_old_val62
  %free_field_067 = getelementptr inbounds %LayerNorm, ptr %old_field_val60, i32 0, i32 0
  %field_val_to_free68 = load ptr, ptr %free_field_067, align 8
  call void @tl_tensor_free(ptr %field_val_to_free68)
  %free_field_169 = getelementptr inbounds %LayerNorm, ptr %old_field_val60, i32 0, i32 1
  %field_val_to_free70 = load ptr, ptr %free_field_169, align 8
  call void @tl_tensor_free(ptr %field_val_to_free70)
  call void @free(ptr %old_field_val60)
  br label %continue_after_recursive_free66

continue_after_recursive_free66:                  ; preds = %recursive_free_struct65, %free_old_val62
  br label %skip_free63

free_old_val78:                                   ; preds = %skip_free63
  %is_not_null80 = icmp ne ptr %old_field_val76, null
  br i1 %is_not_null80, label %recursive_free_struct81, label %continue_after_recursive_free82

skip_free79:                                      ; preds = %continue_after_recursive_free82, %skip_free63
  store ptr %call_method75, ptr %ptr_m, align 8
  call void @tl_mem_unregister(ptr %call_method75)
  %s101 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s101)
  %unreg_field_0 = getelementptr inbounds %Block, ptr %s101, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_0102 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 0
  %field_val103 = load ptr, ptr %unreg_field_0102, align 8
  call void @tl_mem_unregister(ptr %field_val103)
  %unreg_field_1 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 1
  %field_val104 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val104)
  %unreg_field_1105 = getelementptr inbounds %Block, ptr %s101, i32 0, i32 1
  %field_val106 = load ptr, ptr %unreg_field_1105, align 8
  call void @tl_mem_unregister(ptr %field_val106)
  %unreg_field_0107 = getelementptr inbounds %CausalSelfAttention, ptr %field_val106, i32 0, i32 0
  %field_val108 = load ptr, ptr %unreg_field_0107, align 8
  call void @tl_mem_unregister(ptr %field_val108)
  %unreg_field_0109 = getelementptr inbounds %Linear, ptr %field_val108, i32 0, i32 0
  %field_val110 = load ptr, ptr %unreg_field_0109, align 8
  call void @tl_mem_unregister(ptr %field_val110)
  %unreg_field_1111 = getelementptr inbounds %Linear, ptr %field_val108, i32 0, i32 1
  %field_val112 = load ptr, ptr %unreg_field_1111, align 8
  call void @tl_mem_unregister(ptr %field_val112)
  %unreg_field_1113 = getelementptr inbounds %CausalSelfAttention, ptr %field_val106, i32 0, i32 1
  %field_val114 = load ptr, ptr %unreg_field_1113, align 8
  call void @tl_mem_unregister(ptr %field_val114)
  %unreg_field_0115 = getelementptr inbounds %Linear, ptr %field_val114, i32 0, i32 0
  %field_val116 = load ptr, ptr %unreg_field_0115, align 8
  call void @tl_mem_unregister(ptr %field_val116)
  %unreg_field_1117 = getelementptr inbounds %Linear, ptr %field_val114, i32 0, i32 1
  %field_val118 = load ptr, ptr %unreg_field_1117, align 8
  call void @tl_mem_unregister(ptr %field_val118)
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %field_val106, i32 0, i32 2
  %field_val119 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val119)
  %unreg_field_0120 = getelementptr inbounds %Linear, ptr %field_val119, i32 0, i32 0
  %field_val121 = load ptr, ptr %unreg_field_0120, align 8
  call void @tl_mem_unregister(ptr %field_val121)
  %unreg_field_1122 = getelementptr inbounds %Linear, ptr %field_val119, i32 0, i32 1
  %field_val123 = load ptr, ptr %unreg_field_1122, align 8
  call void @tl_mem_unregister(ptr %field_val123)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val106, i32 0, i32 3
  %field_val124 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val124)
  %unreg_field_0125 = getelementptr inbounds %Linear, ptr %field_val124, i32 0, i32 0
  %field_val126 = load ptr, ptr %unreg_field_0125, align 8
  call void @tl_mem_unregister(ptr %field_val126)
  %unreg_field_1127 = getelementptr inbounds %Linear, ptr %field_val124, i32 0, i32 1
  %field_val128 = load ptr, ptr %unreg_field_1127, align 8
  call void @tl_mem_unregister(ptr %field_val128)
  %unreg_field_2129 = getelementptr inbounds %Block, ptr %s101, i32 0, i32 2
  %field_val130 = load ptr, ptr %unreg_field_2129, align 8
  call void @tl_mem_unregister(ptr %field_val130)
  %unreg_field_0131 = getelementptr inbounds %LayerNorm, ptr %field_val130, i32 0, i32 0
  %field_val132 = load ptr, ptr %unreg_field_0131, align 8
  call void @tl_mem_unregister(ptr %field_val132)
  %unreg_field_1133 = getelementptr inbounds %LayerNorm, ptr %field_val130, i32 0, i32 1
  %field_val134 = load ptr, ptr %unreg_field_1133, align 8
  call void @tl_mem_unregister(ptr %field_val134)
  %unreg_field_3135 = getelementptr inbounds %Block, ptr %s101, i32 0, i32 3
  %field_val136 = load ptr, ptr %unreg_field_3135, align 8
  call void @tl_mem_unregister(ptr %field_val136)
  %unreg_field_0137 = getelementptr inbounds %MLP, ptr %field_val136, i32 0, i32 0
  %field_val138 = load ptr, ptr %unreg_field_0137, align 8
  call void @tl_mem_unregister(ptr %field_val138)
  %unreg_field_0139 = getelementptr inbounds %Linear, ptr %field_val138, i32 0, i32 0
  %field_val140 = load ptr, ptr %unreg_field_0139, align 8
  call void @tl_mem_unregister(ptr %field_val140)
  %unreg_field_1141 = getelementptr inbounds %Linear, ptr %field_val138, i32 0, i32 1
  %field_val142 = load ptr, ptr %unreg_field_1141, align 8
  call void @tl_mem_unregister(ptr %field_val142)
  %unreg_field_1143 = getelementptr inbounds %MLP, ptr %field_val136, i32 0, i32 1
  %field_val144 = load ptr, ptr %unreg_field_1143, align 8
  call void @tl_mem_unregister(ptr %field_val144)
  %unreg_field_0145 = getelementptr inbounds %Linear, ptr %field_val144, i32 0, i32 0
  %field_val146 = load ptr, ptr %unreg_field_0145, align 8
  call void @tl_mem_unregister(ptr %field_val146)
  %unreg_field_1147 = getelementptr inbounds %Linear, ptr %field_val144, i32 0, i32 1
  %field_val148 = load ptr, ptr %unreg_field_1147, align 8
  call void @tl_mem_unregister(ptr %field_val148)
  call void @tl_mem_exit_scope()
  ret ptr %s101

recursive_free_struct81:                          ; preds = %free_old_val78
  %free_field_083 = getelementptr inbounds %MLP, ptr %old_field_val76, i32 0, i32 0
  %field_val_to_free84 = load ptr, ptr %free_field_083, align 8
  %is_not_null85 = icmp ne ptr %field_val_to_free84, null
  br i1 %is_not_null85, label %recursive_free_struct86, label %continue_after_recursive_free87

continue_after_recursive_free82:                  ; preds = %continue_after_recursive_free96, %free_old_val78
  br label %skip_free79

recursive_free_struct86:                          ; preds = %recursive_free_struct81
  %free_field_088 = getelementptr inbounds %Linear, ptr %field_val_to_free84, i32 0, i32 0
  %field_val_to_free89 = load ptr, ptr %free_field_088, align 8
  call void @tl_tensor_free(ptr %field_val_to_free89)
  %free_field_190 = getelementptr inbounds %Linear, ptr %field_val_to_free84, i32 0, i32 1
  %field_val_to_free91 = load ptr, ptr %free_field_190, align 8
  call void @tl_tensor_free(ptr %field_val_to_free91)
  call void @free(ptr %field_val_to_free84)
  br label %continue_after_recursive_free87

continue_after_recursive_free87:                  ; preds = %recursive_free_struct86, %recursive_free_struct81
  %free_field_192 = getelementptr inbounds %MLP, ptr %old_field_val76, i32 0, i32 1
  %field_val_to_free93 = load ptr, ptr %free_field_192, align 8
  %is_not_null94 = icmp ne ptr %field_val_to_free93, null
  br i1 %is_not_null94, label %recursive_free_struct95, label %continue_after_recursive_free96

recursive_free_struct95:                          ; preds = %continue_after_recursive_free87
  %free_field_097 = getelementptr inbounds %Linear, ptr %field_val_to_free93, i32 0, i32 0
  %field_val_to_free98 = load ptr, ptr %free_field_097, align 8
  call void @tl_tensor_free(ptr %field_val_to_free98)
  %free_field_199 = getelementptr inbounds %Linear, ptr %field_val_to_free93, i32 0, i32 1
  %field_val_to_free100 = load ptr, ptr %free_field_199, align 8
  call void @tl_tensor_free(ptr %field_val_to_free100)
  call void @free(ptr %field_val_to_free93)
  br label %continue_after_recursive_free96

continue_after_recursive_free96:                  ; preds = %recursive_free_struct95, %continue_after_recursive_free87
  call void @free(ptr %old_field_val76)
  br label %continue_after_recursive_free82
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
  %static_call12 = call ptr @tl_Block_new(i64 %d11)
  %init_field13 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 3
  store ptr %static_call12, ptr %init_field13, align 8
  %d14 = load i64, ptr %d2, align 8
  %static_call15 = call ptr @tl_Block_new(i64 %d14)
  %init_field16 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 4
  store ptr %static_call15, ptr %init_field16, align 8
  %d17 = load i64, ptr %d2, align 8
  %static_call18 = call ptr @tl_LayerNorm_new(i64 %d17)
  %init_field19 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 5
  store ptr %static_call18, ptr %init_field19, align 8
  %d20 = load i64, ptr %d2, align 8
  %v21 = load i64, ptr %v1, align 8
  %static_call22 = call ptr @tl_Linear_new(i64 %d20, i64 %v21)
  %init_field23 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 6
  store ptr %static_call22, ptr %init_field23, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_024 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val25 = load ptr, ptr %unreg_field_024, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_1 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 1
  %field_val26 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  %unreg_field_027 = getelementptr inbounds %Embedding, ptr %field_val26, i32 0, i32 0
  %field_val28 = load ptr, ptr %unreg_field_027, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_2 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 2
  %field_val29 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val29)
  %unreg_field_030 = getelementptr inbounds %Block, ptr %field_val29, i32 0, i32 0
  %field_val31 = load ptr, ptr %unreg_field_030, align 8
  call void @tl_mem_unregister(ptr %field_val31)
  %unreg_field_032 = getelementptr inbounds %LayerNorm, ptr %field_val31, i32 0, i32 0
  %field_val33 = load ptr, ptr %unreg_field_032, align 8
  call void @tl_mem_unregister(ptr %field_val33)
  %unreg_field_134 = getelementptr inbounds %LayerNorm, ptr %field_val31, i32 0, i32 1
  %field_val35 = load ptr, ptr %unreg_field_134, align 8
  call void @tl_mem_unregister(ptr %field_val35)
  %unreg_field_136 = getelementptr inbounds %Block, ptr %field_val29, i32 0, i32 1
  %field_val37 = load ptr, ptr %unreg_field_136, align 8
  call void @tl_mem_unregister(ptr %field_val37)
  %unreg_field_038 = getelementptr inbounds %CausalSelfAttention, ptr %field_val37, i32 0, i32 0
  %field_val39 = load ptr, ptr %unreg_field_038, align 8
  call void @tl_mem_unregister(ptr %field_val39)
  %unreg_field_040 = getelementptr inbounds %Linear, ptr %field_val39, i32 0, i32 0
  %field_val41 = load ptr, ptr %unreg_field_040, align 8
  call void @tl_mem_unregister(ptr %field_val41)
  %unreg_field_142 = getelementptr inbounds %Linear, ptr %field_val39, i32 0, i32 1
  %field_val43 = load ptr, ptr %unreg_field_142, align 8
  call void @tl_mem_unregister(ptr %field_val43)
  %unreg_field_144 = getelementptr inbounds %CausalSelfAttention, ptr %field_val37, i32 0, i32 1
  %field_val45 = load ptr, ptr %unreg_field_144, align 8
  call void @tl_mem_unregister(ptr %field_val45)
  %unreg_field_046 = getelementptr inbounds %Linear, ptr %field_val45, i32 0, i32 0
  %field_val47 = load ptr, ptr %unreg_field_046, align 8
  call void @tl_mem_unregister(ptr %field_val47)
  %unreg_field_148 = getelementptr inbounds %Linear, ptr %field_val45, i32 0, i32 1
  %field_val49 = load ptr, ptr %unreg_field_148, align 8
  call void @tl_mem_unregister(ptr %field_val49)
  %unreg_field_250 = getelementptr inbounds %CausalSelfAttention, ptr %field_val37, i32 0, i32 2
  %field_val51 = load ptr, ptr %unreg_field_250, align 8
  call void @tl_mem_unregister(ptr %field_val51)
  %unreg_field_052 = getelementptr inbounds %Linear, ptr %field_val51, i32 0, i32 0
  %field_val53 = load ptr, ptr %unreg_field_052, align 8
  call void @tl_mem_unregister(ptr %field_val53)
  %unreg_field_154 = getelementptr inbounds %Linear, ptr %field_val51, i32 0, i32 1
  %field_val55 = load ptr, ptr %unreg_field_154, align 8
  call void @tl_mem_unregister(ptr %field_val55)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val37, i32 0, i32 3
  %field_val56 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val56)
  %unreg_field_057 = getelementptr inbounds %Linear, ptr %field_val56, i32 0, i32 0
  %field_val58 = load ptr, ptr %unreg_field_057, align 8
  call void @tl_mem_unregister(ptr %field_val58)
  %unreg_field_159 = getelementptr inbounds %Linear, ptr %field_val56, i32 0, i32 1
  %field_val60 = load ptr, ptr %unreg_field_159, align 8
  call void @tl_mem_unregister(ptr %field_val60)
  %unreg_field_261 = getelementptr inbounds %Block, ptr %field_val29, i32 0, i32 2
  %field_val62 = load ptr, ptr %unreg_field_261, align 8
  call void @tl_mem_unregister(ptr %field_val62)
  %unreg_field_063 = getelementptr inbounds %LayerNorm, ptr %field_val62, i32 0, i32 0
  %field_val64 = load ptr, ptr %unreg_field_063, align 8
  call void @tl_mem_unregister(ptr %field_val64)
  %unreg_field_165 = getelementptr inbounds %LayerNorm, ptr %field_val62, i32 0, i32 1
  %field_val66 = load ptr, ptr %unreg_field_165, align 8
  call void @tl_mem_unregister(ptr %field_val66)
  %unreg_field_367 = getelementptr inbounds %Block, ptr %field_val29, i32 0, i32 3
  %field_val68 = load ptr, ptr %unreg_field_367, align 8
  call void @tl_mem_unregister(ptr %field_val68)
  %unreg_field_069 = getelementptr inbounds %MLP, ptr %field_val68, i32 0, i32 0
  %field_val70 = load ptr, ptr %unreg_field_069, align 8
  call void @tl_mem_unregister(ptr %field_val70)
  %unreg_field_071 = getelementptr inbounds %Linear, ptr %field_val70, i32 0, i32 0
  %field_val72 = load ptr, ptr %unreg_field_071, align 8
  call void @tl_mem_unregister(ptr %field_val72)
  %unreg_field_173 = getelementptr inbounds %Linear, ptr %field_val70, i32 0, i32 1
  %field_val74 = load ptr, ptr %unreg_field_173, align 8
  call void @tl_mem_unregister(ptr %field_val74)
  %unreg_field_175 = getelementptr inbounds %MLP, ptr %field_val68, i32 0, i32 1
  %field_val76 = load ptr, ptr %unreg_field_175, align 8
  call void @tl_mem_unregister(ptr %field_val76)
  %unreg_field_077 = getelementptr inbounds %Linear, ptr %field_val76, i32 0, i32 0
  %field_val78 = load ptr, ptr %unreg_field_077, align 8
  call void @tl_mem_unregister(ptr %field_val78)
  %unreg_field_179 = getelementptr inbounds %Linear, ptr %field_val76, i32 0, i32 1
  %field_val80 = load ptr, ptr %unreg_field_179, align 8
  call void @tl_mem_unregister(ptr %field_val80)
  %unreg_field_381 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 3
  %field_val82 = load ptr, ptr %unreg_field_381, align 8
  call void @tl_mem_unregister(ptr %field_val82)
  %unreg_field_083 = getelementptr inbounds %Block, ptr %field_val82, i32 0, i32 0
  %field_val84 = load ptr, ptr %unreg_field_083, align 8
  call void @tl_mem_unregister(ptr %field_val84)
  %unreg_field_085 = getelementptr inbounds %LayerNorm, ptr %field_val84, i32 0, i32 0
  %field_val86 = load ptr, ptr %unreg_field_085, align 8
  call void @tl_mem_unregister(ptr %field_val86)
  %unreg_field_187 = getelementptr inbounds %LayerNorm, ptr %field_val84, i32 0, i32 1
  %field_val88 = load ptr, ptr %unreg_field_187, align 8
  call void @tl_mem_unregister(ptr %field_val88)
  %unreg_field_189 = getelementptr inbounds %Block, ptr %field_val82, i32 0, i32 1
  %field_val90 = load ptr, ptr %unreg_field_189, align 8
  call void @tl_mem_unregister(ptr %field_val90)
  %unreg_field_091 = getelementptr inbounds %CausalSelfAttention, ptr %field_val90, i32 0, i32 0
  %field_val92 = load ptr, ptr %unreg_field_091, align 8
  call void @tl_mem_unregister(ptr %field_val92)
  %unreg_field_093 = getelementptr inbounds %Linear, ptr %field_val92, i32 0, i32 0
  %field_val94 = load ptr, ptr %unreg_field_093, align 8
  call void @tl_mem_unregister(ptr %field_val94)
  %unreg_field_195 = getelementptr inbounds %Linear, ptr %field_val92, i32 0, i32 1
  %field_val96 = load ptr, ptr %unreg_field_195, align 8
  call void @tl_mem_unregister(ptr %field_val96)
  %unreg_field_197 = getelementptr inbounds %CausalSelfAttention, ptr %field_val90, i32 0, i32 1
  %field_val98 = load ptr, ptr %unreg_field_197, align 8
  call void @tl_mem_unregister(ptr %field_val98)
  %unreg_field_099 = getelementptr inbounds %Linear, ptr %field_val98, i32 0, i32 0
  %field_val100 = load ptr, ptr %unreg_field_099, align 8
  call void @tl_mem_unregister(ptr %field_val100)
  %unreg_field_1101 = getelementptr inbounds %Linear, ptr %field_val98, i32 0, i32 1
  %field_val102 = load ptr, ptr %unreg_field_1101, align 8
  call void @tl_mem_unregister(ptr %field_val102)
  %unreg_field_2103 = getelementptr inbounds %CausalSelfAttention, ptr %field_val90, i32 0, i32 2
  %field_val104 = load ptr, ptr %unreg_field_2103, align 8
  call void @tl_mem_unregister(ptr %field_val104)
  %unreg_field_0105 = getelementptr inbounds %Linear, ptr %field_val104, i32 0, i32 0
  %field_val106 = load ptr, ptr %unreg_field_0105, align 8
  call void @tl_mem_unregister(ptr %field_val106)
  %unreg_field_1107 = getelementptr inbounds %Linear, ptr %field_val104, i32 0, i32 1
  %field_val108 = load ptr, ptr %unreg_field_1107, align 8
  call void @tl_mem_unregister(ptr %field_val108)
  %unreg_field_3109 = getelementptr inbounds %CausalSelfAttention, ptr %field_val90, i32 0, i32 3
  %field_val110 = load ptr, ptr %unreg_field_3109, align 8
  call void @tl_mem_unregister(ptr %field_val110)
  %unreg_field_0111 = getelementptr inbounds %Linear, ptr %field_val110, i32 0, i32 0
  %field_val112 = load ptr, ptr %unreg_field_0111, align 8
  call void @tl_mem_unregister(ptr %field_val112)
  %unreg_field_1113 = getelementptr inbounds %Linear, ptr %field_val110, i32 0, i32 1
  %field_val114 = load ptr, ptr %unreg_field_1113, align 8
  call void @tl_mem_unregister(ptr %field_val114)
  %unreg_field_2115 = getelementptr inbounds %Block, ptr %field_val82, i32 0, i32 2
  %field_val116 = load ptr, ptr %unreg_field_2115, align 8
  call void @tl_mem_unregister(ptr %field_val116)
  %unreg_field_0117 = getelementptr inbounds %LayerNorm, ptr %field_val116, i32 0, i32 0
  %field_val118 = load ptr, ptr %unreg_field_0117, align 8
  call void @tl_mem_unregister(ptr %field_val118)
  %unreg_field_1119 = getelementptr inbounds %LayerNorm, ptr %field_val116, i32 0, i32 1
  %field_val120 = load ptr, ptr %unreg_field_1119, align 8
  call void @tl_mem_unregister(ptr %field_val120)
  %unreg_field_3121 = getelementptr inbounds %Block, ptr %field_val82, i32 0, i32 3
  %field_val122 = load ptr, ptr %unreg_field_3121, align 8
  call void @tl_mem_unregister(ptr %field_val122)
  %unreg_field_0123 = getelementptr inbounds %MLP, ptr %field_val122, i32 0, i32 0
  %field_val124 = load ptr, ptr %unreg_field_0123, align 8
  call void @tl_mem_unregister(ptr %field_val124)
  %unreg_field_0125 = getelementptr inbounds %Linear, ptr %field_val124, i32 0, i32 0
  %field_val126 = load ptr, ptr %unreg_field_0125, align 8
  call void @tl_mem_unregister(ptr %field_val126)
  %unreg_field_1127 = getelementptr inbounds %Linear, ptr %field_val124, i32 0, i32 1
  %field_val128 = load ptr, ptr %unreg_field_1127, align 8
  call void @tl_mem_unregister(ptr %field_val128)
  %unreg_field_1129 = getelementptr inbounds %MLP, ptr %field_val122, i32 0, i32 1
  %field_val130 = load ptr, ptr %unreg_field_1129, align 8
  call void @tl_mem_unregister(ptr %field_val130)
  %unreg_field_0131 = getelementptr inbounds %Linear, ptr %field_val130, i32 0, i32 0
  %field_val132 = load ptr, ptr %unreg_field_0131, align 8
  call void @tl_mem_unregister(ptr %field_val132)
  %unreg_field_1133 = getelementptr inbounds %Linear, ptr %field_val130, i32 0, i32 1
  %field_val134 = load ptr, ptr %unreg_field_1133, align 8
  call void @tl_mem_unregister(ptr %field_val134)
  %unreg_field_4 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 4
  %field_val135 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val135)
  %unreg_field_0136 = getelementptr inbounds %Block, ptr %field_val135, i32 0, i32 0
  %field_val137 = load ptr, ptr %unreg_field_0136, align 8
  call void @tl_mem_unregister(ptr %field_val137)
  %unreg_field_0138 = getelementptr inbounds %LayerNorm, ptr %field_val137, i32 0, i32 0
  %field_val139 = load ptr, ptr %unreg_field_0138, align 8
  call void @tl_mem_unregister(ptr %field_val139)
  %unreg_field_1140 = getelementptr inbounds %LayerNorm, ptr %field_val137, i32 0, i32 1
  %field_val141 = load ptr, ptr %unreg_field_1140, align 8
  call void @tl_mem_unregister(ptr %field_val141)
  %unreg_field_1142 = getelementptr inbounds %Block, ptr %field_val135, i32 0, i32 1
  %field_val143 = load ptr, ptr %unreg_field_1142, align 8
  call void @tl_mem_unregister(ptr %field_val143)
  %unreg_field_0144 = getelementptr inbounds %CausalSelfAttention, ptr %field_val143, i32 0, i32 0
  %field_val145 = load ptr, ptr %unreg_field_0144, align 8
  call void @tl_mem_unregister(ptr %field_val145)
  %unreg_field_0146 = getelementptr inbounds %Linear, ptr %field_val145, i32 0, i32 0
  %field_val147 = load ptr, ptr %unreg_field_0146, align 8
  call void @tl_mem_unregister(ptr %field_val147)
  %unreg_field_1148 = getelementptr inbounds %Linear, ptr %field_val145, i32 0, i32 1
  %field_val149 = load ptr, ptr %unreg_field_1148, align 8
  call void @tl_mem_unregister(ptr %field_val149)
  %unreg_field_1150 = getelementptr inbounds %CausalSelfAttention, ptr %field_val143, i32 0, i32 1
  %field_val151 = load ptr, ptr %unreg_field_1150, align 8
  call void @tl_mem_unregister(ptr %field_val151)
  %unreg_field_0152 = getelementptr inbounds %Linear, ptr %field_val151, i32 0, i32 0
  %field_val153 = load ptr, ptr %unreg_field_0152, align 8
  call void @tl_mem_unregister(ptr %field_val153)
  %unreg_field_1154 = getelementptr inbounds %Linear, ptr %field_val151, i32 0, i32 1
  %field_val155 = load ptr, ptr %unreg_field_1154, align 8
  call void @tl_mem_unregister(ptr %field_val155)
  %unreg_field_2156 = getelementptr inbounds %CausalSelfAttention, ptr %field_val143, i32 0, i32 2
  %field_val157 = load ptr, ptr %unreg_field_2156, align 8
  call void @tl_mem_unregister(ptr %field_val157)
  %unreg_field_0158 = getelementptr inbounds %Linear, ptr %field_val157, i32 0, i32 0
  %field_val159 = load ptr, ptr %unreg_field_0158, align 8
  call void @tl_mem_unregister(ptr %field_val159)
  %unreg_field_1160 = getelementptr inbounds %Linear, ptr %field_val157, i32 0, i32 1
  %field_val161 = load ptr, ptr %unreg_field_1160, align 8
  call void @tl_mem_unregister(ptr %field_val161)
  %unreg_field_3162 = getelementptr inbounds %CausalSelfAttention, ptr %field_val143, i32 0, i32 3
  %field_val163 = load ptr, ptr %unreg_field_3162, align 8
  call void @tl_mem_unregister(ptr %field_val163)
  %unreg_field_0164 = getelementptr inbounds %Linear, ptr %field_val163, i32 0, i32 0
  %field_val165 = load ptr, ptr %unreg_field_0164, align 8
  call void @tl_mem_unregister(ptr %field_val165)
  %unreg_field_1166 = getelementptr inbounds %Linear, ptr %field_val163, i32 0, i32 1
  %field_val167 = load ptr, ptr %unreg_field_1166, align 8
  call void @tl_mem_unregister(ptr %field_val167)
  %unreg_field_2168 = getelementptr inbounds %Block, ptr %field_val135, i32 0, i32 2
  %field_val169 = load ptr, ptr %unreg_field_2168, align 8
  call void @tl_mem_unregister(ptr %field_val169)
  %unreg_field_0170 = getelementptr inbounds %LayerNorm, ptr %field_val169, i32 0, i32 0
  %field_val171 = load ptr, ptr %unreg_field_0170, align 8
  call void @tl_mem_unregister(ptr %field_val171)
  %unreg_field_1172 = getelementptr inbounds %LayerNorm, ptr %field_val169, i32 0, i32 1
  %field_val173 = load ptr, ptr %unreg_field_1172, align 8
  call void @tl_mem_unregister(ptr %field_val173)
  %unreg_field_3174 = getelementptr inbounds %Block, ptr %field_val135, i32 0, i32 3
  %field_val175 = load ptr, ptr %unreg_field_3174, align 8
  call void @tl_mem_unregister(ptr %field_val175)
  %unreg_field_0176 = getelementptr inbounds %MLP, ptr %field_val175, i32 0, i32 0
  %field_val177 = load ptr, ptr %unreg_field_0176, align 8
  call void @tl_mem_unregister(ptr %field_val177)
  %unreg_field_0178 = getelementptr inbounds %Linear, ptr %field_val177, i32 0, i32 0
  %field_val179 = load ptr, ptr %unreg_field_0178, align 8
  call void @tl_mem_unregister(ptr %field_val179)
  %unreg_field_1180 = getelementptr inbounds %Linear, ptr %field_val177, i32 0, i32 1
  %field_val181 = load ptr, ptr %unreg_field_1180, align 8
  call void @tl_mem_unregister(ptr %field_val181)
  %unreg_field_1182 = getelementptr inbounds %MLP, ptr %field_val175, i32 0, i32 1
  %field_val183 = load ptr, ptr %unreg_field_1182, align 8
  call void @tl_mem_unregister(ptr %field_val183)
  %unreg_field_0184 = getelementptr inbounds %Linear, ptr %field_val183, i32 0, i32 0
  %field_val185 = load ptr, ptr %unreg_field_0184, align 8
  call void @tl_mem_unregister(ptr %field_val185)
  %unreg_field_1186 = getelementptr inbounds %Linear, ptr %field_val183, i32 0, i32 1
  %field_val187 = load ptr, ptr %unreg_field_1186, align 8
  call void @tl_mem_unregister(ptr %field_val187)
  %unreg_field_5 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 5
  %field_val188 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val188)
  %unreg_field_0189 = getelementptr inbounds %LayerNorm, ptr %field_val188, i32 0, i32 0
  %field_val190 = load ptr, ptr %unreg_field_0189, align 8
  call void @tl_mem_unregister(ptr %field_val190)
  %unreg_field_1191 = getelementptr inbounds %LayerNorm, ptr %field_val188, i32 0, i32 1
  %field_val192 = load ptr, ptr %unreg_field_1191, align 8
  call void @tl_mem_unregister(ptr %field_val192)
  %unreg_field_6 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 6
  %field_val193 = load ptr, ptr %unreg_field_6, align 8
  call void @tl_mem_unregister(ptr %field_val193)
  %unreg_field_0194 = getelementptr inbounds %Linear, ptr %field_val193, i32 0, i32 0
  %field_val195 = load ptr, ptr %unreg_field_0194, align 8
  call void @tl_mem_unregister(ptr %field_val195)
  %unreg_field_1196 = getelementptr inbounds %Linear, ptr %field_val193, i32 0, i32 1
  %field_val197 = load ptr, ptr %unreg_field_1196, align 8
  call void @tl_mem_unregister(ptr %field_val197)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_GPT_forward(ptr %self, ptr %i) {
entry:
  %x35 = alloca ptr, align 16
  %x30 = alloca ptr, align 16
  %x25 = alloca ptr, align 16
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
  call void @tl_mem_unregister(ptr %new_const_tensor)
  store ptr %new_const_tensor, ptr %pos_data, align 8
  %pos_data14 = load ptr, ptr %pos_data, align 8
  %dim_ptr_0 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0, align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %pos_data14, ptr %dims_ptr, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res)
  store ptr %reshape_dims_res, ptr %pos, align 8
  %self15 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds %GPT, ptr %self15, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %i16 = load ptr, ptr %i2, align 8
  %call_method = call ptr @tl_Embedding_forward(ptr %w, ptr %i16)
  call void @tl_mem_register_tensor(ptr %call_method)
  call void @tl_mem_unregister(ptr %call_method)
  store ptr %call_method, ptr %tok_emb, align 8
  %self17 = load ptr, ptr %self1, align 8
  %ptr_wp = getelementptr inbounds %GPT, ptr %self17, i32 0, i32 1
  %wp = load ptr, ptr %ptr_wp, align 8
  %pos18 = load ptr, ptr %pos, align 8
  %call_method19 = call ptr @tl_Embedding_forward(ptr %wp, ptr %pos18)
  call void @tl_mem_register_tensor(ptr %call_method19)
  call void @tl_mem_unregister(ptr %call_method19)
  store ptr %call_method19, ptr %pos_emb, align 8
  %tok_emb20 = load ptr, ptr %tok_emb, align 8
  %pos_emb21 = load ptr, ptr %pos_emb, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %tok_emb20, ptr %pos_emb21)
  call void @tl_mem_unregister(ptr %binop_res)
  store ptr %binop_res, ptr %x, align 8
  %self22 = load ptr, ptr %self1, align 8
  %ptr_b1 = getelementptr inbounds %GPT, ptr %self22, i32 0, i32 2
  %b1 = load ptr, ptr %ptr_b1, align 8
  %x23 = load ptr, ptr %x, align 8
  %call_method24 = call ptr @tl_Block_forward(ptr %b1, ptr %x23)
  call void @tl_mem_register_tensor(ptr %call_method24)
  call void @tl_mem_unregister(ptr %call_method24)
  %old_shadowed = load ptr, ptr %x, align 8
  call void @tl_mem_unregister(ptr %old_shadowed)
  store ptr %call_method24, ptr %x25, align 8
  %self26 = load ptr, ptr %self1, align 8
  %ptr_b2 = getelementptr inbounds %GPT, ptr %self26, i32 0, i32 3
  %b2 = load ptr, ptr %ptr_b2, align 8
  %x27 = load ptr, ptr %x25, align 8
  %call_method28 = call ptr @tl_Block_forward(ptr %b2, ptr %x27)
  call void @tl_mem_register_tensor(ptr %call_method28)
  call void @tl_mem_unregister(ptr %call_method28)
  %old_shadowed29 = load ptr, ptr %x25, align 8
  call void @tl_mem_unregister(ptr %old_shadowed29)
  store ptr %call_method28, ptr %x30, align 8
  %self31 = load ptr, ptr %self1, align 8
  %ptr_b3 = getelementptr inbounds %GPT, ptr %self31, i32 0, i32 4
  %b3 = load ptr, ptr %ptr_b3, align 8
  %x32 = load ptr, ptr %x30, align 8
  %call_method33 = call ptr @tl_Block_forward(ptr %b3, ptr %x32)
  call void @tl_mem_register_tensor(ptr %call_method33)
  call void @tl_mem_unregister(ptr %call_method33)
  %old_shadowed34 = load ptr, ptr %x30, align 8
  call void @tl_mem_unregister(ptr %old_shadowed34)
  store ptr %call_method33, ptr %x35, align 8
  %self36 = load ptr, ptr %self1, align 8
  %ptr_h = getelementptr inbounds %GPT, ptr %self36, i32 0, i32 6
  %h = load ptr, ptr %ptr_h, align 8
  %self37 = load ptr, ptr %self1, align 8
  %ptr_l = getelementptr inbounds %GPT, ptr %self37, i32 0, i32 5
  %l = load ptr, ptr %ptr_l, align 8
  %x38 = load ptr, ptr %x35, align 8
  %call_method39 = call ptr @tl_LayerNorm_forward(ptr %l, ptr %x38)
  call void @tl_mem_register_tensor(ptr %call_method39)
  %call_method40 = call ptr @tl_Linear_forward(ptr %h, ptr %call_method39)
  call void @tl_tensor_free(ptr %call_method39)
  call void @tl_mem_register_tensor(ptr %call_method40)
  call void @tl_mem_unregister(ptr %call_method40)
  call void @tl_mem_exit_scope()
  ret ptr %call_method40
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
  %is_not_null = icmp ne ptr %old_field_val, null
  br i1 %is_not_null, label %recursive_free_struct, label %continue_after_recursive_free

skip_free:                                        ; preds = %continue_after_recursive_free, %entry
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

recursive_free_struct:                            ; preds = %free_old_val
  %free_field_0 = getelementptr inbounds %Embedding, ptr %old_field_val, i32 0, i32 0
  %field_val_to_free = load ptr, ptr %free_field_0, align 8
  call void @tl_tensor_free(ptr %field_val_to_free)
  call void @free(ptr %old_field_val)
  br label %continue_after_recursive_free

continue_after_recursive_free:                    ; preds = %recursive_free_struct, %free_old_val
  br label %skip_free

free_old_val15:                                   ; preds = %skip_free
  %is_not_null17 = icmp ne ptr %old_field_val13, null
  br i1 %is_not_null17, label %recursive_free_struct18, label %continue_after_recursive_free19

skip_free16:                                      ; preds = %continue_after_recursive_free19, %skip_free
  store ptr %call_method12, ptr %ptr_wp, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s22 = load ptr, ptr %s, align 8
  %ptr_b1 = getelementptr inbounds %GPT, ptr %s22, i32 0, i32 2
  %s23 = load ptr, ptr %s, align 8
  %ptr_b124 = getelementptr inbounds %GPT, ptr %s23, i32 0, i32 2
  %b1 = load ptr, ptr %ptr_b124, align 8
  %lr25 = load float, ptr %lr2, align 4
  %call_method26 = call ptr @tl_Block_step(ptr %b1, float %lr25)
  call void @tl_mem_register_struct(ptr %call_method26)
  %old_field_val27 = load ptr, ptr %ptr_b1, align 8
  %cnt_free_diff28 = icmp ne ptr %old_field_val27, %call_method26
  br i1 %cnt_free_diff28, label %free_old_val29, label %skip_free30

recursive_free_struct18:                          ; preds = %free_old_val15
  %free_field_020 = getelementptr inbounds %Embedding, ptr %old_field_val13, i32 0, i32 0
  %field_val_to_free21 = load ptr, ptr %free_field_020, align 8
  call void @tl_tensor_free(ptr %field_val_to_free21)
  call void @free(ptr %old_field_val13)
  br label %continue_after_recursive_free19

continue_after_recursive_free19:                  ; preds = %recursive_free_struct18, %free_old_val15
  br label %skip_free16

free_old_val29:                                   ; preds = %skip_free16
  %is_not_null31 = icmp ne ptr %old_field_val27, null
  br i1 %is_not_null31, label %recursive_free_struct32, label %continue_after_recursive_free33

skip_free30:                                      ; preds = %continue_after_recursive_free33, %skip_free16
  store ptr %call_method26, ptr %ptr_b1, align 8
  call void @tl_mem_unregister(ptr %call_method26)
  %s113 = load ptr, ptr %s, align 8
  %ptr_b2 = getelementptr inbounds %GPT, ptr %s113, i32 0, i32 3
  %s114 = load ptr, ptr %s, align 8
  %ptr_b2115 = getelementptr inbounds %GPT, ptr %s114, i32 0, i32 3
  %b2 = load ptr, ptr %ptr_b2115, align 8
  %lr116 = load float, ptr %lr2, align 4
  %call_method117 = call ptr @tl_Block_step(ptr %b2, float %lr116)
  call void @tl_mem_register_struct(ptr %call_method117)
  %old_field_val118 = load ptr, ptr %ptr_b2, align 8
  %cnt_free_diff119 = icmp ne ptr %old_field_val118, %call_method117
  br i1 %cnt_free_diff119, label %free_old_val120, label %skip_free121

recursive_free_struct32:                          ; preds = %free_old_val29
  %free_field_034 = getelementptr inbounds %Block, ptr %old_field_val27, i32 0, i32 0
  %field_val_to_free35 = load ptr, ptr %free_field_034, align 8
  %is_not_null36 = icmp ne ptr %field_val_to_free35, null
  br i1 %is_not_null36, label %recursive_free_struct37, label %continue_after_recursive_free38

continue_after_recursive_free33:                  ; preds = %continue_after_recursive_free94, %free_old_val29
  br label %skip_free30

recursive_free_struct37:                          ; preds = %recursive_free_struct32
  %free_field_039 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free35, i32 0, i32 0
  %field_val_to_free40 = load ptr, ptr %free_field_039, align 8
  call void @tl_tensor_free(ptr %field_val_to_free40)
  %free_field_1 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free35, i32 0, i32 1
  %field_val_to_free41 = load ptr, ptr %free_field_1, align 8
  call void @tl_tensor_free(ptr %field_val_to_free41)
  call void @free(ptr %field_val_to_free35)
  br label %continue_after_recursive_free38

continue_after_recursive_free38:                  ; preds = %recursive_free_struct37, %recursive_free_struct32
  %free_field_142 = getelementptr inbounds %Block, ptr %old_field_val27, i32 0, i32 1
  %field_val_to_free43 = load ptr, ptr %free_field_142, align 8
  %is_not_null44 = icmp ne ptr %field_val_to_free43, null
  br i1 %is_not_null44, label %recursive_free_struct45, label %continue_after_recursive_free46

recursive_free_struct45:                          ; preds = %continue_after_recursive_free38
  %free_field_047 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free43, i32 0, i32 0
  %field_val_to_free48 = load ptr, ptr %free_field_047, align 8
  %is_not_null49 = icmp ne ptr %field_val_to_free48, null
  br i1 %is_not_null49, label %recursive_free_struct50, label %continue_after_recursive_free51

continue_after_recursive_free46:                  ; preds = %continue_after_recursive_free76, %continue_after_recursive_free38
  %free_field_281 = getelementptr inbounds %Block, ptr %old_field_val27, i32 0, i32 2
  %field_val_to_free82 = load ptr, ptr %free_field_281, align 8
  %is_not_null83 = icmp ne ptr %field_val_to_free82, null
  br i1 %is_not_null83, label %recursive_free_struct84, label %continue_after_recursive_free85

recursive_free_struct50:                          ; preds = %recursive_free_struct45
  %free_field_052 = getelementptr inbounds %Linear, ptr %field_val_to_free48, i32 0, i32 0
  %field_val_to_free53 = load ptr, ptr %free_field_052, align 8
  call void @tl_tensor_free(ptr %field_val_to_free53)
  %free_field_154 = getelementptr inbounds %Linear, ptr %field_val_to_free48, i32 0, i32 1
  %field_val_to_free55 = load ptr, ptr %free_field_154, align 8
  call void @tl_tensor_free(ptr %field_val_to_free55)
  call void @free(ptr %field_val_to_free48)
  br label %continue_after_recursive_free51

continue_after_recursive_free51:                  ; preds = %recursive_free_struct50, %recursive_free_struct45
  %free_field_156 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free43, i32 0, i32 1
  %field_val_to_free57 = load ptr, ptr %free_field_156, align 8
  %is_not_null58 = icmp ne ptr %field_val_to_free57, null
  br i1 %is_not_null58, label %recursive_free_struct59, label %continue_after_recursive_free60

recursive_free_struct59:                          ; preds = %continue_after_recursive_free51
  %free_field_061 = getelementptr inbounds %Linear, ptr %field_val_to_free57, i32 0, i32 0
  %field_val_to_free62 = load ptr, ptr %free_field_061, align 8
  call void @tl_tensor_free(ptr %field_val_to_free62)
  %free_field_163 = getelementptr inbounds %Linear, ptr %field_val_to_free57, i32 0, i32 1
  %field_val_to_free64 = load ptr, ptr %free_field_163, align 8
  call void @tl_tensor_free(ptr %field_val_to_free64)
  call void @free(ptr %field_val_to_free57)
  br label %continue_after_recursive_free60

continue_after_recursive_free60:                  ; preds = %recursive_free_struct59, %continue_after_recursive_free51
  %free_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free43, i32 0, i32 2
  %field_val_to_free65 = load ptr, ptr %free_field_2, align 8
  %is_not_null66 = icmp ne ptr %field_val_to_free65, null
  br i1 %is_not_null66, label %recursive_free_struct67, label %continue_after_recursive_free68

recursive_free_struct67:                          ; preds = %continue_after_recursive_free60
  %free_field_069 = getelementptr inbounds %Linear, ptr %field_val_to_free65, i32 0, i32 0
  %field_val_to_free70 = load ptr, ptr %free_field_069, align 8
  call void @tl_tensor_free(ptr %field_val_to_free70)
  %free_field_171 = getelementptr inbounds %Linear, ptr %field_val_to_free65, i32 0, i32 1
  %field_val_to_free72 = load ptr, ptr %free_field_171, align 8
  call void @tl_tensor_free(ptr %field_val_to_free72)
  call void @free(ptr %field_val_to_free65)
  br label %continue_after_recursive_free68

continue_after_recursive_free68:                  ; preds = %recursive_free_struct67, %continue_after_recursive_free60
  %free_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free43, i32 0, i32 3
  %field_val_to_free73 = load ptr, ptr %free_field_3, align 8
  %is_not_null74 = icmp ne ptr %field_val_to_free73, null
  br i1 %is_not_null74, label %recursive_free_struct75, label %continue_after_recursive_free76

recursive_free_struct75:                          ; preds = %continue_after_recursive_free68
  %free_field_077 = getelementptr inbounds %Linear, ptr %field_val_to_free73, i32 0, i32 0
  %field_val_to_free78 = load ptr, ptr %free_field_077, align 8
  call void @tl_tensor_free(ptr %field_val_to_free78)
  %free_field_179 = getelementptr inbounds %Linear, ptr %field_val_to_free73, i32 0, i32 1
  %field_val_to_free80 = load ptr, ptr %free_field_179, align 8
  call void @tl_tensor_free(ptr %field_val_to_free80)
  call void @free(ptr %field_val_to_free73)
  br label %continue_after_recursive_free76

continue_after_recursive_free76:                  ; preds = %recursive_free_struct75, %continue_after_recursive_free68
  call void @free(ptr %field_val_to_free43)
  br label %continue_after_recursive_free46

recursive_free_struct84:                          ; preds = %continue_after_recursive_free46
  %free_field_086 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free82, i32 0, i32 0
  %field_val_to_free87 = load ptr, ptr %free_field_086, align 8
  call void @tl_tensor_free(ptr %field_val_to_free87)
  %free_field_188 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free82, i32 0, i32 1
  %field_val_to_free89 = load ptr, ptr %free_field_188, align 8
  call void @tl_tensor_free(ptr %field_val_to_free89)
  call void @free(ptr %field_val_to_free82)
  br label %continue_after_recursive_free85

continue_after_recursive_free85:                  ; preds = %recursive_free_struct84, %continue_after_recursive_free46
  %free_field_390 = getelementptr inbounds %Block, ptr %old_field_val27, i32 0, i32 3
  %field_val_to_free91 = load ptr, ptr %free_field_390, align 8
  %is_not_null92 = icmp ne ptr %field_val_to_free91, null
  br i1 %is_not_null92, label %recursive_free_struct93, label %continue_after_recursive_free94

recursive_free_struct93:                          ; preds = %continue_after_recursive_free85
  %free_field_095 = getelementptr inbounds %MLP, ptr %field_val_to_free91, i32 0, i32 0
  %field_val_to_free96 = load ptr, ptr %free_field_095, align 8
  %is_not_null97 = icmp ne ptr %field_val_to_free96, null
  br i1 %is_not_null97, label %recursive_free_struct98, label %continue_after_recursive_free99

continue_after_recursive_free94:                  ; preds = %continue_after_recursive_free108, %continue_after_recursive_free85
  call void @free(ptr %old_field_val27)
  br label %continue_after_recursive_free33

recursive_free_struct98:                          ; preds = %recursive_free_struct93
  %free_field_0100 = getelementptr inbounds %Linear, ptr %field_val_to_free96, i32 0, i32 0
  %field_val_to_free101 = load ptr, ptr %free_field_0100, align 8
  call void @tl_tensor_free(ptr %field_val_to_free101)
  %free_field_1102 = getelementptr inbounds %Linear, ptr %field_val_to_free96, i32 0, i32 1
  %field_val_to_free103 = load ptr, ptr %free_field_1102, align 8
  call void @tl_tensor_free(ptr %field_val_to_free103)
  call void @free(ptr %field_val_to_free96)
  br label %continue_after_recursive_free99

continue_after_recursive_free99:                  ; preds = %recursive_free_struct98, %recursive_free_struct93
  %free_field_1104 = getelementptr inbounds %MLP, ptr %field_val_to_free91, i32 0, i32 1
  %field_val_to_free105 = load ptr, ptr %free_field_1104, align 8
  %is_not_null106 = icmp ne ptr %field_val_to_free105, null
  br i1 %is_not_null106, label %recursive_free_struct107, label %continue_after_recursive_free108

recursive_free_struct107:                         ; preds = %continue_after_recursive_free99
  %free_field_0109 = getelementptr inbounds %Linear, ptr %field_val_to_free105, i32 0, i32 0
  %field_val_to_free110 = load ptr, ptr %free_field_0109, align 8
  call void @tl_tensor_free(ptr %field_val_to_free110)
  %free_field_1111 = getelementptr inbounds %Linear, ptr %field_val_to_free105, i32 0, i32 1
  %field_val_to_free112 = load ptr, ptr %free_field_1111, align 8
  call void @tl_tensor_free(ptr %field_val_to_free112)
  call void @free(ptr %field_val_to_free105)
  br label %continue_after_recursive_free108

continue_after_recursive_free108:                 ; preds = %recursive_free_struct107, %continue_after_recursive_free99
  call void @free(ptr %field_val_to_free91)
  br label %continue_after_recursive_free94

free_old_val120:                                  ; preds = %skip_free30
  %is_not_null122 = icmp ne ptr %old_field_val118, null
  br i1 %is_not_null122, label %recursive_free_struct123, label %continue_after_recursive_free124

skip_free121:                                     ; preds = %continue_after_recursive_free124, %skip_free30
  store ptr %call_method117, ptr %ptr_b2, align 8
  call void @tl_mem_unregister(ptr %call_method117)
  %s207 = load ptr, ptr %s, align 8
  %ptr_b3 = getelementptr inbounds %GPT, ptr %s207, i32 0, i32 4
  %s208 = load ptr, ptr %s, align 8
  %ptr_b3209 = getelementptr inbounds %GPT, ptr %s208, i32 0, i32 4
  %b3 = load ptr, ptr %ptr_b3209, align 8
  %lr210 = load float, ptr %lr2, align 4
  %call_method211 = call ptr @tl_Block_step(ptr %b3, float %lr210)
  call void @tl_mem_register_struct(ptr %call_method211)
  %old_field_val212 = load ptr, ptr %ptr_b3, align 8
  %cnt_free_diff213 = icmp ne ptr %old_field_val212, %call_method211
  br i1 %cnt_free_diff213, label %free_old_val214, label %skip_free215

recursive_free_struct123:                         ; preds = %free_old_val120
  %free_field_0125 = getelementptr inbounds %Block, ptr %old_field_val118, i32 0, i32 0
  %field_val_to_free126 = load ptr, ptr %free_field_0125, align 8
  %is_not_null127 = icmp ne ptr %field_val_to_free126, null
  br i1 %is_not_null127, label %recursive_free_struct128, label %continue_after_recursive_free129

continue_after_recursive_free124:                 ; preds = %continue_after_recursive_free188, %free_old_val120
  br label %skip_free121

recursive_free_struct128:                         ; preds = %recursive_free_struct123
  %free_field_0130 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free126, i32 0, i32 0
  %field_val_to_free131 = load ptr, ptr %free_field_0130, align 8
  call void @tl_tensor_free(ptr %field_val_to_free131)
  %free_field_1132 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free126, i32 0, i32 1
  %field_val_to_free133 = load ptr, ptr %free_field_1132, align 8
  call void @tl_tensor_free(ptr %field_val_to_free133)
  call void @free(ptr %field_val_to_free126)
  br label %continue_after_recursive_free129

continue_after_recursive_free129:                 ; preds = %recursive_free_struct128, %recursive_free_struct123
  %free_field_1134 = getelementptr inbounds %Block, ptr %old_field_val118, i32 0, i32 1
  %field_val_to_free135 = load ptr, ptr %free_field_1134, align 8
  %is_not_null136 = icmp ne ptr %field_val_to_free135, null
  br i1 %is_not_null136, label %recursive_free_struct137, label %continue_after_recursive_free138

recursive_free_struct137:                         ; preds = %continue_after_recursive_free129
  %free_field_0139 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free135, i32 0, i32 0
  %field_val_to_free140 = load ptr, ptr %free_field_0139, align 8
  %is_not_null141 = icmp ne ptr %field_val_to_free140, null
  br i1 %is_not_null141, label %recursive_free_struct142, label %continue_after_recursive_free143

continue_after_recursive_free138:                 ; preds = %continue_after_recursive_free170, %continue_after_recursive_free129
  %free_field_2175 = getelementptr inbounds %Block, ptr %old_field_val118, i32 0, i32 2
  %field_val_to_free176 = load ptr, ptr %free_field_2175, align 8
  %is_not_null177 = icmp ne ptr %field_val_to_free176, null
  br i1 %is_not_null177, label %recursive_free_struct178, label %continue_after_recursive_free179

recursive_free_struct142:                         ; preds = %recursive_free_struct137
  %free_field_0144 = getelementptr inbounds %Linear, ptr %field_val_to_free140, i32 0, i32 0
  %field_val_to_free145 = load ptr, ptr %free_field_0144, align 8
  call void @tl_tensor_free(ptr %field_val_to_free145)
  %free_field_1146 = getelementptr inbounds %Linear, ptr %field_val_to_free140, i32 0, i32 1
  %field_val_to_free147 = load ptr, ptr %free_field_1146, align 8
  call void @tl_tensor_free(ptr %field_val_to_free147)
  call void @free(ptr %field_val_to_free140)
  br label %continue_after_recursive_free143

continue_after_recursive_free143:                 ; preds = %recursive_free_struct142, %recursive_free_struct137
  %free_field_1148 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free135, i32 0, i32 1
  %field_val_to_free149 = load ptr, ptr %free_field_1148, align 8
  %is_not_null150 = icmp ne ptr %field_val_to_free149, null
  br i1 %is_not_null150, label %recursive_free_struct151, label %continue_after_recursive_free152

recursive_free_struct151:                         ; preds = %continue_after_recursive_free143
  %free_field_0153 = getelementptr inbounds %Linear, ptr %field_val_to_free149, i32 0, i32 0
  %field_val_to_free154 = load ptr, ptr %free_field_0153, align 8
  call void @tl_tensor_free(ptr %field_val_to_free154)
  %free_field_1155 = getelementptr inbounds %Linear, ptr %field_val_to_free149, i32 0, i32 1
  %field_val_to_free156 = load ptr, ptr %free_field_1155, align 8
  call void @tl_tensor_free(ptr %field_val_to_free156)
  call void @free(ptr %field_val_to_free149)
  br label %continue_after_recursive_free152

continue_after_recursive_free152:                 ; preds = %recursive_free_struct151, %continue_after_recursive_free143
  %free_field_2157 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free135, i32 0, i32 2
  %field_val_to_free158 = load ptr, ptr %free_field_2157, align 8
  %is_not_null159 = icmp ne ptr %field_val_to_free158, null
  br i1 %is_not_null159, label %recursive_free_struct160, label %continue_after_recursive_free161

recursive_free_struct160:                         ; preds = %continue_after_recursive_free152
  %free_field_0162 = getelementptr inbounds %Linear, ptr %field_val_to_free158, i32 0, i32 0
  %field_val_to_free163 = load ptr, ptr %free_field_0162, align 8
  call void @tl_tensor_free(ptr %field_val_to_free163)
  %free_field_1164 = getelementptr inbounds %Linear, ptr %field_val_to_free158, i32 0, i32 1
  %field_val_to_free165 = load ptr, ptr %free_field_1164, align 8
  call void @tl_tensor_free(ptr %field_val_to_free165)
  call void @free(ptr %field_val_to_free158)
  br label %continue_after_recursive_free161

continue_after_recursive_free161:                 ; preds = %recursive_free_struct160, %continue_after_recursive_free152
  %free_field_3166 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free135, i32 0, i32 3
  %field_val_to_free167 = load ptr, ptr %free_field_3166, align 8
  %is_not_null168 = icmp ne ptr %field_val_to_free167, null
  br i1 %is_not_null168, label %recursive_free_struct169, label %continue_after_recursive_free170

recursive_free_struct169:                         ; preds = %continue_after_recursive_free161
  %free_field_0171 = getelementptr inbounds %Linear, ptr %field_val_to_free167, i32 0, i32 0
  %field_val_to_free172 = load ptr, ptr %free_field_0171, align 8
  call void @tl_tensor_free(ptr %field_val_to_free172)
  %free_field_1173 = getelementptr inbounds %Linear, ptr %field_val_to_free167, i32 0, i32 1
  %field_val_to_free174 = load ptr, ptr %free_field_1173, align 8
  call void @tl_tensor_free(ptr %field_val_to_free174)
  call void @free(ptr %field_val_to_free167)
  br label %continue_after_recursive_free170

continue_after_recursive_free170:                 ; preds = %recursive_free_struct169, %continue_after_recursive_free161
  call void @free(ptr %field_val_to_free135)
  br label %continue_after_recursive_free138

recursive_free_struct178:                         ; preds = %continue_after_recursive_free138
  %free_field_0180 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free176, i32 0, i32 0
  %field_val_to_free181 = load ptr, ptr %free_field_0180, align 8
  call void @tl_tensor_free(ptr %field_val_to_free181)
  %free_field_1182 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free176, i32 0, i32 1
  %field_val_to_free183 = load ptr, ptr %free_field_1182, align 8
  call void @tl_tensor_free(ptr %field_val_to_free183)
  call void @free(ptr %field_val_to_free176)
  br label %continue_after_recursive_free179

continue_after_recursive_free179:                 ; preds = %recursive_free_struct178, %continue_after_recursive_free138
  %free_field_3184 = getelementptr inbounds %Block, ptr %old_field_val118, i32 0, i32 3
  %field_val_to_free185 = load ptr, ptr %free_field_3184, align 8
  %is_not_null186 = icmp ne ptr %field_val_to_free185, null
  br i1 %is_not_null186, label %recursive_free_struct187, label %continue_after_recursive_free188

recursive_free_struct187:                         ; preds = %continue_after_recursive_free179
  %free_field_0189 = getelementptr inbounds %MLP, ptr %field_val_to_free185, i32 0, i32 0
  %field_val_to_free190 = load ptr, ptr %free_field_0189, align 8
  %is_not_null191 = icmp ne ptr %field_val_to_free190, null
  br i1 %is_not_null191, label %recursive_free_struct192, label %continue_after_recursive_free193

continue_after_recursive_free188:                 ; preds = %continue_after_recursive_free202, %continue_after_recursive_free179
  call void @free(ptr %old_field_val118)
  br label %continue_after_recursive_free124

recursive_free_struct192:                         ; preds = %recursive_free_struct187
  %free_field_0194 = getelementptr inbounds %Linear, ptr %field_val_to_free190, i32 0, i32 0
  %field_val_to_free195 = load ptr, ptr %free_field_0194, align 8
  call void @tl_tensor_free(ptr %field_val_to_free195)
  %free_field_1196 = getelementptr inbounds %Linear, ptr %field_val_to_free190, i32 0, i32 1
  %field_val_to_free197 = load ptr, ptr %free_field_1196, align 8
  call void @tl_tensor_free(ptr %field_val_to_free197)
  call void @free(ptr %field_val_to_free190)
  br label %continue_after_recursive_free193

continue_after_recursive_free193:                 ; preds = %recursive_free_struct192, %recursive_free_struct187
  %free_field_1198 = getelementptr inbounds %MLP, ptr %field_val_to_free185, i32 0, i32 1
  %field_val_to_free199 = load ptr, ptr %free_field_1198, align 8
  %is_not_null200 = icmp ne ptr %field_val_to_free199, null
  br i1 %is_not_null200, label %recursive_free_struct201, label %continue_after_recursive_free202

recursive_free_struct201:                         ; preds = %continue_after_recursive_free193
  %free_field_0203 = getelementptr inbounds %Linear, ptr %field_val_to_free199, i32 0, i32 0
  %field_val_to_free204 = load ptr, ptr %free_field_0203, align 8
  call void @tl_tensor_free(ptr %field_val_to_free204)
  %free_field_1205 = getelementptr inbounds %Linear, ptr %field_val_to_free199, i32 0, i32 1
  %field_val_to_free206 = load ptr, ptr %free_field_1205, align 8
  call void @tl_tensor_free(ptr %field_val_to_free206)
  call void @free(ptr %field_val_to_free199)
  br label %continue_after_recursive_free202

continue_after_recursive_free202:                 ; preds = %recursive_free_struct201, %continue_after_recursive_free193
  call void @free(ptr %field_val_to_free185)
  br label %continue_after_recursive_free188

free_old_val214:                                  ; preds = %skip_free121
  %is_not_null216 = icmp ne ptr %old_field_val212, null
  br i1 %is_not_null216, label %recursive_free_struct217, label %continue_after_recursive_free218

skip_free215:                                     ; preds = %continue_after_recursive_free218, %skip_free121
  store ptr %call_method211, ptr %ptr_b3, align 8
  call void @tl_mem_unregister(ptr %call_method211)
  %s301 = load ptr, ptr %s, align 8
  %ptr_l = getelementptr inbounds %GPT, ptr %s301, i32 0, i32 5
  %s302 = load ptr, ptr %s, align 8
  %ptr_l303 = getelementptr inbounds %GPT, ptr %s302, i32 0, i32 5
  %l = load ptr, ptr %ptr_l303, align 8
  %lr304 = load float, ptr %lr2, align 4
  %call_method305 = call ptr @tl_LayerNorm_step(ptr %l, float %lr304)
  call void @tl_mem_register_struct(ptr %call_method305)
  %old_field_val306 = load ptr, ptr %ptr_l, align 8
  %cnt_free_diff307 = icmp ne ptr %old_field_val306, %call_method305
  br i1 %cnt_free_diff307, label %free_old_val308, label %skip_free309

recursive_free_struct217:                         ; preds = %free_old_val214
  %free_field_0219 = getelementptr inbounds %Block, ptr %old_field_val212, i32 0, i32 0
  %field_val_to_free220 = load ptr, ptr %free_field_0219, align 8
  %is_not_null221 = icmp ne ptr %field_val_to_free220, null
  br i1 %is_not_null221, label %recursive_free_struct222, label %continue_after_recursive_free223

continue_after_recursive_free218:                 ; preds = %continue_after_recursive_free282, %free_old_val214
  br label %skip_free215

recursive_free_struct222:                         ; preds = %recursive_free_struct217
  %free_field_0224 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free220, i32 0, i32 0
  %field_val_to_free225 = load ptr, ptr %free_field_0224, align 8
  call void @tl_tensor_free(ptr %field_val_to_free225)
  %free_field_1226 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free220, i32 0, i32 1
  %field_val_to_free227 = load ptr, ptr %free_field_1226, align 8
  call void @tl_tensor_free(ptr %field_val_to_free227)
  call void @free(ptr %field_val_to_free220)
  br label %continue_after_recursive_free223

continue_after_recursive_free223:                 ; preds = %recursive_free_struct222, %recursive_free_struct217
  %free_field_1228 = getelementptr inbounds %Block, ptr %old_field_val212, i32 0, i32 1
  %field_val_to_free229 = load ptr, ptr %free_field_1228, align 8
  %is_not_null230 = icmp ne ptr %field_val_to_free229, null
  br i1 %is_not_null230, label %recursive_free_struct231, label %continue_after_recursive_free232

recursive_free_struct231:                         ; preds = %continue_after_recursive_free223
  %free_field_0233 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free229, i32 0, i32 0
  %field_val_to_free234 = load ptr, ptr %free_field_0233, align 8
  %is_not_null235 = icmp ne ptr %field_val_to_free234, null
  br i1 %is_not_null235, label %recursive_free_struct236, label %continue_after_recursive_free237

continue_after_recursive_free232:                 ; preds = %continue_after_recursive_free264, %continue_after_recursive_free223
  %free_field_2269 = getelementptr inbounds %Block, ptr %old_field_val212, i32 0, i32 2
  %field_val_to_free270 = load ptr, ptr %free_field_2269, align 8
  %is_not_null271 = icmp ne ptr %field_val_to_free270, null
  br i1 %is_not_null271, label %recursive_free_struct272, label %continue_after_recursive_free273

recursive_free_struct236:                         ; preds = %recursive_free_struct231
  %free_field_0238 = getelementptr inbounds %Linear, ptr %field_val_to_free234, i32 0, i32 0
  %field_val_to_free239 = load ptr, ptr %free_field_0238, align 8
  call void @tl_tensor_free(ptr %field_val_to_free239)
  %free_field_1240 = getelementptr inbounds %Linear, ptr %field_val_to_free234, i32 0, i32 1
  %field_val_to_free241 = load ptr, ptr %free_field_1240, align 8
  call void @tl_tensor_free(ptr %field_val_to_free241)
  call void @free(ptr %field_val_to_free234)
  br label %continue_after_recursive_free237

continue_after_recursive_free237:                 ; preds = %recursive_free_struct236, %recursive_free_struct231
  %free_field_1242 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free229, i32 0, i32 1
  %field_val_to_free243 = load ptr, ptr %free_field_1242, align 8
  %is_not_null244 = icmp ne ptr %field_val_to_free243, null
  br i1 %is_not_null244, label %recursive_free_struct245, label %continue_after_recursive_free246

recursive_free_struct245:                         ; preds = %continue_after_recursive_free237
  %free_field_0247 = getelementptr inbounds %Linear, ptr %field_val_to_free243, i32 0, i32 0
  %field_val_to_free248 = load ptr, ptr %free_field_0247, align 8
  call void @tl_tensor_free(ptr %field_val_to_free248)
  %free_field_1249 = getelementptr inbounds %Linear, ptr %field_val_to_free243, i32 0, i32 1
  %field_val_to_free250 = load ptr, ptr %free_field_1249, align 8
  call void @tl_tensor_free(ptr %field_val_to_free250)
  call void @free(ptr %field_val_to_free243)
  br label %continue_after_recursive_free246

continue_after_recursive_free246:                 ; preds = %recursive_free_struct245, %continue_after_recursive_free237
  %free_field_2251 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free229, i32 0, i32 2
  %field_val_to_free252 = load ptr, ptr %free_field_2251, align 8
  %is_not_null253 = icmp ne ptr %field_val_to_free252, null
  br i1 %is_not_null253, label %recursive_free_struct254, label %continue_after_recursive_free255

recursive_free_struct254:                         ; preds = %continue_after_recursive_free246
  %free_field_0256 = getelementptr inbounds %Linear, ptr %field_val_to_free252, i32 0, i32 0
  %field_val_to_free257 = load ptr, ptr %free_field_0256, align 8
  call void @tl_tensor_free(ptr %field_val_to_free257)
  %free_field_1258 = getelementptr inbounds %Linear, ptr %field_val_to_free252, i32 0, i32 1
  %field_val_to_free259 = load ptr, ptr %free_field_1258, align 8
  call void @tl_tensor_free(ptr %field_val_to_free259)
  call void @free(ptr %field_val_to_free252)
  br label %continue_after_recursive_free255

continue_after_recursive_free255:                 ; preds = %recursive_free_struct254, %continue_after_recursive_free246
  %free_field_3260 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free229, i32 0, i32 3
  %field_val_to_free261 = load ptr, ptr %free_field_3260, align 8
  %is_not_null262 = icmp ne ptr %field_val_to_free261, null
  br i1 %is_not_null262, label %recursive_free_struct263, label %continue_after_recursive_free264

recursive_free_struct263:                         ; preds = %continue_after_recursive_free255
  %free_field_0265 = getelementptr inbounds %Linear, ptr %field_val_to_free261, i32 0, i32 0
  %field_val_to_free266 = load ptr, ptr %free_field_0265, align 8
  call void @tl_tensor_free(ptr %field_val_to_free266)
  %free_field_1267 = getelementptr inbounds %Linear, ptr %field_val_to_free261, i32 0, i32 1
  %field_val_to_free268 = load ptr, ptr %free_field_1267, align 8
  call void @tl_tensor_free(ptr %field_val_to_free268)
  call void @free(ptr %field_val_to_free261)
  br label %continue_after_recursive_free264

continue_after_recursive_free264:                 ; preds = %recursive_free_struct263, %continue_after_recursive_free255
  call void @free(ptr %field_val_to_free229)
  br label %continue_after_recursive_free232

recursive_free_struct272:                         ; preds = %continue_after_recursive_free232
  %free_field_0274 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free270, i32 0, i32 0
  %field_val_to_free275 = load ptr, ptr %free_field_0274, align 8
  call void @tl_tensor_free(ptr %field_val_to_free275)
  %free_field_1276 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free270, i32 0, i32 1
  %field_val_to_free277 = load ptr, ptr %free_field_1276, align 8
  call void @tl_tensor_free(ptr %field_val_to_free277)
  call void @free(ptr %field_val_to_free270)
  br label %continue_after_recursive_free273

continue_after_recursive_free273:                 ; preds = %recursive_free_struct272, %continue_after_recursive_free232
  %free_field_3278 = getelementptr inbounds %Block, ptr %old_field_val212, i32 0, i32 3
  %field_val_to_free279 = load ptr, ptr %free_field_3278, align 8
  %is_not_null280 = icmp ne ptr %field_val_to_free279, null
  br i1 %is_not_null280, label %recursive_free_struct281, label %continue_after_recursive_free282

recursive_free_struct281:                         ; preds = %continue_after_recursive_free273
  %free_field_0283 = getelementptr inbounds %MLP, ptr %field_val_to_free279, i32 0, i32 0
  %field_val_to_free284 = load ptr, ptr %free_field_0283, align 8
  %is_not_null285 = icmp ne ptr %field_val_to_free284, null
  br i1 %is_not_null285, label %recursive_free_struct286, label %continue_after_recursive_free287

continue_after_recursive_free282:                 ; preds = %continue_after_recursive_free296, %continue_after_recursive_free273
  call void @free(ptr %old_field_val212)
  br label %continue_after_recursive_free218

recursive_free_struct286:                         ; preds = %recursive_free_struct281
  %free_field_0288 = getelementptr inbounds %Linear, ptr %field_val_to_free284, i32 0, i32 0
  %field_val_to_free289 = load ptr, ptr %free_field_0288, align 8
  call void @tl_tensor_free(ptr %field_val_to_free289)
  %free_field_1290 = getelementptr inbounds %Linear, ptr %field_val_to_free284, i32 0, i32 1
  %field_val_to_free291 = load ptr, ptr %free_field_1290, align 8
  call void @tl_tensor_free(ptr %field_val_to_free291)
  call void @free(ptr %field_val_to_free284)
  br label %continue_after_recursive_free287

continue_after_recursive_free287:                 ; preds = %recursive_free_struct286, %recursive_free_struct281
  %free_field_1292 = getelementptr inbounds %MLP, ptr %field_val_to_free279, i32 0, i32 1
  %field_val_to_free293 = load ptr, ptr %free_field_1292, align 8
  %is_not_null294 = icmp ne ptr %field_val_to_free293, null
  br i1 %is_not_null294, label %recursive_free_struct295, label %continue_after_recursive_free296

recursive_free_struct295:                         ; preds = %continue_after_recursive_free287
  %free_field_0297 = getelementptr inbounds %Linear, ptr %field_val_to_free293, i32 0, i32 0
  %field_val_to_free298 = load ptr, ptr %free_field_0297, align 8
  call void @tl_tensor_free(ptr %field_val_to_free298)
  %free_field_1299 = getelementptr inbounds %Linear, ptr %field_val_to_free293, i32 0, i32 1
  %field_val_to_free300 = load ptr, ptr %free_field_1299, align 8
  call void @tl_tensor_free(ptr %field_val_to_free300)
  call void @free(ptr %field_val_to_free293)
  br label %continue_after_recursive_free296

continue_after_recursive_free296:                 ; preds = %recursive_free_struct295, %continue_after_recursive_free287
  call void @free(ptr %field_val_to_free279)
  br label %continue_after_recursive_free282

free_old_val308:                                  ; preds = %skip_free215
  %is_not_null310 = icmp ne ptr %old_field_val306, null
  br i1 %is_not_null310, label %recursive_free_struct311, label %continue_after_recursive_free312

skip_free309:                                     ; preds = %continue_after_recursive_free312, %skip_free215
  store ptr %call_method305, ptr %ptr_l, align 8
  call void @tl_mem_unregister(ptr %call_method305)
  %s317 = load ptr, ptr %s, align 8
  %ptr_h = getelementptr inbounds %GPT, ptr %s317, i32 0, i32 6
  %s318 = load ptr, ptr %s, align 8
  %ptr_h319 = getelementptr inbounds %GPT, ptr %s318, i32 0, i32 6
  %h = load ptr, ptr %ptr_h319, align 8
  %lr320 = load float, ptr %lr2, align 4
  %call_method321 = call ptr @tl_Linear_step(ptr %h, float %lr320)
  call void @tl_mem_register_struct(ptr %call_method321)
  %old_field_val322 = load ptr, ptr %ptr_h, align 8
  %cnt_free_diff323 = icmp ne ptr %old_field_val322, %call_method321
  br i1 %cnt_free_diff323, label %free_old_val324, label %skip_free325

recursive_free_struct311:                         ; preds = %free_old_val308
  %free_field_0313 = getelementptr inbounds %LayerNorm, ptr %old_field_val306, i32 0, i32 0
  %field_val_to_free314 = load ptr, ptr %free_field_0313, align 8
  call void @tl_tensor_free(ptr %field_val_to_free314)
  %free_field_1315 = getelementptr inbounds %LayerNorm, ptr %old_field_val306, i32 0, i32 1
  %field_val_to_free316 = load ptr, ptr %free_field_1315, align 8
  call void @tl_tensor_free(ptr %field_val_to_free316)
  call void @free(ptr %old_field_val306)
  br label %continue_after_recursive_free312

continue_after_recursive_free312:                 ; preds = %recursive_free_struct311, %free_old_val308
  br label %skip_free309

free_old_val324:                                  ; preds = %skip_free309
  %is_not_null326 = icmp ne ptr %old_field_val322, null
  br i1 %is_not_null326, label %recursive_free_struct327, label %continue_after_recursive_free328

skip_free325:                                     ; preds = %continue_after_recursive_free328, %skip_free309
  store ptr %call_method321, ptr %ptr_h, align 8
  call void @tl_mem_unregister(ptr %call_method321)
  %s333 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s333)
  %unreg_field_0 = getelementptr inbounds %GPT, ptr %s333, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_0334 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val335 = load ptr, ptr %unreg_field_0334, align 8
  call void @tl_mem_unregister(ptr %field_val335)
  %unreg_field_1 = getelementptr inbounds %GPT, ptr %s333, i32 0, i32 1
  %field_val336 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val336)
  %unreg_field_0337 = getelementptr inbounds %Embedding, ptr %field_val336, i32 0, i32 0
  %field_val338 = load ptr, ptr %unreg_field_0337, align 8
  call void @tl_mem_unregister(ptr %field_val338)
  %unreg_field_2 = getelementptr inbounds %GPT, ptr %s333, i32 0, i32 2
  %field_val339 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val339)
  %unreg_field_0340 = getelementptr inbounds %Block, ptr %field_val339, i32 0, i32 0
  %field_val341 = load ptr, ptr %unreg_field_0340, align 8
  call void @tl_mem_unregister(ptr %field_val341)
  %unreg_field_0342 = getelementptr inbounds %LayerNorm, ptr %field_val341, i32 0, i32 0
  %field_val343 = load ptr, ptr %unreg_field_0342, align 8
  call void @tl_mem_unregister(ptr %field_val343)
  %unreg_field_1344 = getelementptr inbounds %LayerNorm, ptr %field_val341, i32 0, i32 1
  %field_val345 = load ptr, ptr %unreg_field_1344, align 8
  call void @tl_mem_unregister(ptr %field_val345)
  %unreg_field_1346 = getelementptr inbounds %Block, ptr %field_val339, i32 0, i32 1
  %field_val347 = load ptr, ptr %unreg_field_1346, align 8
  call void @tl_mem_unregister(ptr %field_val347)
  %unreg_field_0348 = getelementptr inbounds %CausalSelfAttention, ptr %field_val347, i32 0, i32 0
  %field_val349 = load ptr, ptr %unreg_field_0348, align 8
  call void @tl_mem_unregister(ptr %field_val349)
  %unreg_field_0350 = getelementptr inbounds %Linear, ptr %field_val349, i32 0, i32 0
  %field_val351 = load ptr, ptr %unreg_field_0350, align 8
  call void @tl_mem_unregister(ptr %field_val351)
  %unreg_field_1352 = getelementptr inbounds %Linear, ptr %field_val349, i32 0, i32 1
  %field_val353 = load ptr, ptr %unreg_field_1352, align 8
  call void @tl_mem_unregister(ptr %field_val353)
  %unreg_field_1354 = getelementptr inbounds %CausalSelfAttention, ptr %field_val347, i32 0, i32 1
  %field_val355 = load ptr, ptr %unreg_field_1354, align 8
  call void @tl_mem_unregister(ptr %field_val355)
  %unreg_field_0356 = getelementptr inbounds %Linear, ptr %field_val355, i32 0, i32 0
  %field_val357 = load ptr, ptr %unreg_field_0356, align 8
  call void @tl_mem_unregister(ptr %field_val357)
  %unreg_field_1358 = getelementptr inbounds %Linear, ptr %field_val355, i32 0, i32 1
  %field_val359 = load ptr, ptr %unreg_field_1358, align 8
  call void @tl_mem_unregister(ptr %field_val359)
  %unreg_field_2360 = getelementptr inbounds %CausalSelfAttention, ptr %field_val347, i32 0, i32 2
  %field_val361 = load ptr, ptr %unreg_field_2360, align 8
  call void @tl_mem_unregister(ptr %field_val361)
  %unreg_field_0362 = getelementptr inbounds %Linear, ptr %field_val361, i32 0, i32 0
  %field_val363 = load ptr, ptr %unreg_field_0362, align 8
  call void @tl_mem_unregister(ptr %field_val363)
  %unreg_field_1364 = getelementptr inbounds %Linear, ptr %field_val361, i32 0, i32 1
  %field_val365 = load ptr, ptr %unreg_field_1364, align 8
  call void @tl_mem_unregister(ptr %field_val365)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val347, i32 0, i32 3
  %field_val366 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val366)
  %unreg_field_0367 = getelementptr inbounds %Linear, ptr %field_val366, i32 0, i32 0
  %field_val368 = load ptr, ptr %unreg_field_0367, align 8
  call void @tl_mem_unregister(ptr %field_val368)
  %unreg_field_1369 = getelementptr inbounds %Linear, ptr %field_val366, i32 0, i32 1
  %field_val370 = load ptr, ptr %unreg_field_1369, align 8
  call void @tl_mem_unregister(ptr %field_val370)
  %unreg_field_2371 = getelementptr inbounds %Block, ptr %field_val339, i32 0, i32 2
  %field_val372 = load ptr, ptr %unreg_field_2371, align 8
  call void @tl_mem_unregister(ptr %field_val372)
  %unreg_field_0373 = getelementptr inbounds %LayerNorm, ptr %field_val372, i32 0, i32 0
  %field_val374 = load ptr, ptr %unreg_field_0373, align 8
  call void @tl_mem_unregister(ptr %field_val374)
  %unreg_field_1375 = getelementptr inbounds %LayerNorm, ptr %field_val372, i32 0, i32 1
  %field_val376 = load ptr, ptr %unreg_field_1375, align 8
  call void @tl_mem_unregister(ptr %field_val376)
  %unreg_field_3377 = getelementptr inbounds %Block, ptr %field_val339, i32 0, i32 3
  %field_val378 = load ptr, ptr %unreg_field_3377, align 8
  call void @tl_mem_unregister(ptr %field_val378)
  %unreg_field_0379 = getelementptr inbounds %MLP, ptr %field_val378, i32 0, i32 0
  %field_val380 = load ptr, ptr %unreg_field_0379, align 8
  call void @tl_mem_unregister(ptr %field_val380)
  %unreg_field_0381 = getelementptr inbounds %Linear, ptr %field_val380, i32 0, i32 0
  %field_val382 = load ptr, ptr %unreg_field_0381, align 8
  call void @tl_mem_unregister(ptr %field_val382)
  %unreg_field_1383 = getelementptr inbounds %Linear, ptr %field_val380, i32 0, i32 1
  %field_val384 = load ptr, ptr %unreg_field_1383, align 8
  call void @tl_mem_unregister(ptr %field_val384)
  %unreg_field_1385 = getelementptr inbounds %MLP, ptr %field_val378, i32 0, i32 1
  %field_val386 = load ptr, ptr %unreg_field_1385, align 8
  call void @tl_mem_unregister(ptr %field_val386)
  %unreg_field_0387 = getelementptr inbounds %Linear, ptr %field_val386, i32 0, i32 0
  %field_val388 = load ptr, ptr %unreg_field_0387, align 8
  call void @tl_mem_unregister(ptr %field_val388)
  %unreg_field_1389 = getelementptr inbounds %Linear, ptr %field_val386, i32 0, i32 1
  %field_val390 = load ptr, ptr %unreg_field_1389, align 8
  call void @tl_mem_unregister(ptr %field_val390)
  %unreg_field_3391 = getelementptr inbounds %GPT, ptr %s333, i32 0, i32 3
  %field_val392 = load ptr, ptr %unreg_field_3391, align 8
  call void @tl_mem_unregister(ptr %field_val392)
  %unreg_field_0393 = getelementptr inbounds %Block, ptr %field_val392, i32 0, i32 0
  %field_val394 = load ptr, ptr %unreg_field_0393, align 8
  call void @tl_mem_unregister(ptr %field_val394)
  %unreg_field_0395 = getelementptr inbounds %LayerNorm, ptr %field_val394, i32 0, i32 0
  %field_val396 = load ptr, ptr %unreg_field_0395, align 8
  call void @tl_mem_unregister(ptr %field_val396)
  %unreg_field_1397 = getelementptr inbounds %LayerNorm, ptr %field_val394, i32 0, i32 1
  %field_val398 = load ptr, ptr %unreg_field_1397, align 8
  call void @tl_mem_unregister(ptr %field_val398)
  %unreg_field_1399 = getelementptr inbounds %Block, ptr %field_val392, i32 0, i32 1
  %field_val400 = load ptr, ptr %unreg_field_1399, align 8
  call void @tl_mem_unregister(ptr %field_val400)
  %unreg_field_0401 = getelementptr inbounds %CausalSelfAttention, ptr %field_val400, i32 0, i32 0
  %field_val402 = load ptr, ptr %unreg_field_0401, align 8
  call void @tl_mem_unregister(ptr %field_val402)
  %unreg_field_0403 = getelementptr inbounds %Linear, ptr %field_val402, i32 0, i32 0
  %field_val404 = load ptr, ptr %unreg_field_0403, align 8
  call void @tl_mem_unregister(ptr %field_val404)
  %unreg_field_1405 = getelementptr inbounds %Linear, ptr %field_val402, i32 0, i32 1
  %field_val406 = load ptr, ptr %unreg_field_1405, align 8
  call void @tl_mem_unregister(ptr %field_val406)
  %unreg_field_1407 = getelementptr inbounds %CausalSelfAttention, ptr %field_val400, i32 0, i32 1
  %field_val408 = load ptr, ptr %unreg_field_1407, align 8
  call void @tl_mem_unregister(ptr %field_val408)
  %unreg_field_0409 = getelementptr inbounds %Linear, ptr %field_val408, i32 0, i32 0
  %field_val410 = load ptr, ptr %unreg_field_0409, align 8
  call void @tl_mem_unregister(ptr %field_val410)
  %unreg_field_1411 = getelementptr inbounds %Linear, ptr %field_val408, i32 0, i32 1
  %field_val412 = load ptr, ptr %unreg_field_1411, align 8
  call void @tl_mem_unregister(ptr %field_val412)
  %unreg_field_2413 = getelementptr inbounds %CausalSelfAttention, ptr %field_val400, i32 0, i32 2
  %field_val414 = load ptr, ptr %unreg_field_2413, align 8
  call void @tl_mem_unregister(ptr %field_val414)
  %unreg_field_0415 = getelementptr inbounds %Linear, ptr %field_val414, i32 0, i32 0
  %field_val416 = load ptr, ptr %unreg_field_0415, align 8
  call void @tl_mem_unregister(ptr %field_val416)
  %unreg_field_1417 = getelementptr inbounds %Linear, ptr %field_val414, i32 0, i32 1
  %field_val418 = load ptr, ptr %unreg_field_1417, align 8
  call void @tl_mem_unregister(ptr %field_val418)
  %unreg_field_3419 = getelementptr inbounds %CausalSelfAttention, ptr %field_val400, i32 0, i32 3
  %field_val420 = load ptr, ptr %unreg_field_3419, align 8
  call void @tl_mem_unregister(ptr %field_val420)
  %unreg_field_0421 = getelementptr inbounds %Linear, ptr %field_val420, i32 0, i32 0
  %field_val422 = load ptr, ptr %unreg_field_0421, align 8
  call void @tl_mem_unregister(ptr %field_val422)
  %unreg_field_1423 = getelementptr inbounds %Linear, ptr %field_val420, i32 0, i32 1
  %field_val424 = load ptr, ptr %unreg_field_1423, align 8
  call void @tl_mem_unregister(ptr %field_val424)
  %unreg_field_2425 = getelementptr inbounds %Block, ptr %field_val392, i32 0, i32 2
  %field_val426 = load ptr, ptr %unreg_field_2425, align 8
  call void @tl_mem_unregister(ptr %field_val426)
  %unreg_field_0427 = getelementptr inbounds %LayerNorm, ptr %field_val426, i32 0, i32 0
  %field_val428 = load ptr, ptr %unreg_field_0427, align 8
  call void @tl_mem_unregister(ptr %field_val428)
  %unreg_field_1429 = getelementptr inbounds %LayerNorm, ptr %field_val426, i32 0, i32 1
  %field_val430 = load ptr, ptr %unreg_field_1429, align 8
  call void @tl_mem_unregister(ptr %field_val430)
  %unreg_field_3431 = getelementptr inbounds %Block, ptr %field_val392, i32 0, i32 3
  %field_val432 = load ptr, ptr %unreg_field_3431, align 8
  call void @tl_mem_unregister(ptr %field_val432)
  %unreg_field_0433 = getelementptr inbounds %MLP, ptr %field_val432, i32 0, i32 0
  %field_val434 = load ptr, ptr %unreg_field_0433, align 8
  call void @tl_mem_unregister(ptr %field_val434)
  %unreg_field_0435 = getelementptr inbounds %Linear, ptr %field_val434, i32 0, i32 0
  %field_val436 = load ptr, ptr %unreg_field_0435, align 8
  call void @tl_mem_unregister(ptr %field_val436)
  %unreg_field_1437 = getelementptr inbounds %Linear, ptr %field_val434, i32 0, i32 1
  %field_val438 = load ptr, ptr %unreg_field_1437, align 8
  call void @tl_mem_unregister(ptr %field_val438)
  %unreg_field_1439 = getelementptr inbounds %MLP, ptr %field_val432, i32 0, i32 1
  %field_val440 = load ptr, ptr %unreg_field_1439, align 8
  call void @tl_mem_unregister(ptr %field_val440)
  %unreg_field_0441 = getelementptr inbounds %Linear, ptr %field_val440, i32 0, i32 0
  %field_val442 = load ptr, ptr %unreg_field_0441, align 8
  call void @tl_mem_unregister(ptr %field_val442)
  %unreg_field_1443 = getelementptr inbounds %Linear, ptr %field_val440, i32 0, i32 1
  %field_val444 = load ptr, ptr %unreg_field_1443, align 8
  call void @tl_mem_unregister(ptr %field_val444)
  %unreg_field_4 = getelementptr inbounds %GPT, ptr %s333, i32 0, i32 4
  %field_val445 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val445)
  %unreg_field_0446 = getelementptr inbounds %Block, ptr %field_val445, i32 0, i32 0
  %field_val447 = load ptr, ptr %unreg_field_0446, align 8
  call void @tl_mem_unregister(ptr %field_val447)
  %unreg_field_0448 = getelementptr inbounds %LayerNorm, ptr %field_val447, i32 0, i32 0
  %field_val449 = load ptr, ptr %unreg_field_0448, align 8
  call void @tl_mem_unregister(ptr %field_val449)
  %unreg_field_1450 = getelementptr inbounds %LayerNorm, ptr %field_val447, i32 0, i32 1
  %field_val451 = load ptr, ptr %unreg_field_1450, align 8
  call void @tl_mem_unregister(ptr %field_val451)
  %unreg_field_1452 = getelementptr inbounds %Block, ptr %field_val445, i32 0, i32 1
  %field_val453 = load ptr, ptr %unreg_field_1452, align 8
  call void @tl_mem_unregister(ptr %field_val453)
  %unreg_field_0454 = getelementptr inbounds %CausalSelfAttention, ptr %field_val453, i32 0, i32 0
  %field_val455 = load ptr, ptr %unreg_field_0454, align 8
  call void @tl_mem_unregister(ptr %field_val455)
  %unreg_field_0456 = getelementptr inbounds %Linear, ptr %field_val455, i32 0, i32 0
  %field_val457 = load ptr, ptr %unreg_field_0456, align 8
  call void @tl_mem_unregister(ptr %field_val457)
  %unreg_field_1458 = getelementptr inbounds %Linear, ptr %field_val455, i32 0, i32 1
  %field_val459 = load ptr, ptr %unreg_field_1458, align 8
  call void @tl_mem_unregister(ptr %field_val459)
  %unreg_field_1460 = getelementptr inbounds %CausalSelfAttention, ptr %field_val453, i32 0, i32 1
  %field_val461 = load ptr, ptr %unreg_field_1460, align 8
  call void @tl_mem_unregister(ptr %field_val461)
  %unreg_field_0462 = getelementptr inbounds %Linear, ptr %field_val461, i32 0, i32 0
  %field_val463 = load ptr, ptr %unreg_field_0462, align 8
  call void @tl_mem_unregister(ptr %field_val463)
  %unreg_field_1464 = getelementptr inbounds %Linear, ptr %field_val461, i32 0, i32 1
  %field_val465 = load ptr, ptr %unreg_field_1464, align 8
  call void @tl_mem_unregister(ptr %field_val465)
  %unreg_field_2466 = getelementptr inbounds %CausalSelfAttention, ptr %field_val453, i32 0, i32 2
  %field_val467 = load ptr, ptr %unreg_field_2466, align 8
  call void @tl_mem_unregister(ptr %field_val467)
  %unreg_field_0468 = getelementptr inbounds %Linear, ptr %field_val467, i32 0, i32 0
  %field_val469 = load ptr, ptr %unreg_field_0468, align 8
  call void @tl_mem_unregister(ptr %field_val469)
  %unreg_field_1470 = getelementptr inbounds %Linear, ptr %field_val467, i32 0, i32 1
  %field_val471 = load ptr, ptr %unreg_field_1470, align 8
  call void @tl_mem_unregister(ptr %field_val471)
  %unreg_field_3472 = getelementptr inbounds %CausalSelfAttention, ptr %field_val453, i32 0, i32 3
  %field_val473 = load ptr, ptr %unreg_field_3472, align 8
  call void @tl_mem_unregister(ptr %field_val473)
  %unreg_field_0474 = getelementptr inbounds %Linear, ptr %field_val473, i32 0, i32 0
  %field_val475 = load ptr, ptr %unreg_field_0474, align 8
  call void @tl_mem_unregister(ptr %field_val475)
  %unreg_field_1476 = getelementptr inbounds %Linear, ptr %field_val473, i32 0, i32 1
  %field_val477 = load ptr, ptr %unreg_field_1476, align 8
  call void @tl_mem_unregister(ptr %field_val477)
  %unreg_field_2478 = getelementptr inbounds %Block, ptr %field_val445, i32 0, i32 2
  %field_val479 = load ptr, ptr %unreg_field_2478, align 8
  call void @tl_mem_unregister(ptr %field_val479)
  %unreg_field_0480 = getelementptr inbounds %LayerNorm, ptr %field_val479, i32 0, i32 0
  %field_val481 = load ptr, ptr %unreg_field_0480, align 8
  call void @tl_mem_unregister(ptr %field_val481)
  %unreg_field_1482 = getelementptr inbounds %LayerNorm, ptr %field_val479, i32 0, i32 1
  %field_val483 = load ptr, ptr %unreg_field_1482, align 8
  call void @tl_mem_unregister(ptr %field_val483)
  %unreg_field_3484 = getelementptr inbounds %Block, ptr %field_val445, i32 0, i32 3
  %field_val485 = load ptr, ptr %unreg_field_3484, align 8
  call void @tl_mem_unregister(ptr %field_val485)
  %unreg_field_0486 = getelementptr inbounds %MLP, ptr %field_val485, i32 0, i32 0
  %field_val487 = load ptr, ptr %unreg_field_0486, align 8
  call void @tl_mem_unregister(ptr %field_val487)
  %unreg_field_0488 = getelementptr inbounds %Linear, ptr %field_val487, i32 0, i32 0
  %field_val489 = load ptr, ptr %unreg_field_0488, align 8
  call void @tl_mem_unregister(ptr %field_val489)
  %unreg_field_1490 = getelementptr inbounds %Linear, ptr %field_val487, i32 0, i32 1
  %field_val491 = load ptr, ptr %unreg_field_1490, align 8
  call void @tl_mem_unregister(ptr %field_val491)
  %unreg_field_1492 = getelementptr inbounds %MLP, ptr %field_val485, i32 0, i32 1
  %field_val493 = load ptr, ptr %unreg_field_1492, align 8
  call void @tl_mem_unregister(ptr %field_val493)
  %unreg_field_0494 = getelementptr inbounds %Linear, ptr %field_val493, i32 0, i32 0
  %field_val495 = load ptr, ptr %unreg_field_0494, align 8
  call void @tl_mem_unregister(ptr %field_val495)
  %unreg_field_1496 = getelementptr inbounds %Linear, ptr %field_val493, i32 0, i32 1
  %field_val497 = load ptr, ptr %unreg_field_1496, align 8
  call void @tl_mem_unregister(ptr %field_val497)
  %unreg_field_5 = getelementptr inbounds %GPT, ptr %s333, i32 0, i32 5
  %field_val498 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val498)
  %unreg_field_0499 = getelementptr inbounds %LayerNorm, ptr %field_val498, i32 0, i32 0
  %field_val500 = load ptr, ptr %unreg_field_0499, align 8
  call void @tl_mem_unregister(ptr %field_val500)
  %unreg_field_1501 = getelementptr inbounds %LayerNorm, ptr %field_val498, i32 0, i32 1
  %field_val502 = load ptr, ptr %unreg_field_1501, align 8
  call void @tl_mem_unregister(ptr %field_val502)
  %unreg_field_6 = getelementptr inbounds %GPT, ptr %s333, i32 0, i32 6
  %field_val503 = load ptr, ptr %unreg_field_6, align 8
  call void @tl_mem_unregister(ptr %field_val503)
  %unreg_field_0504 = getelementptr inbounds %Linear, ptr %field_val503, i32 0, i32 0
  %field_val505 = load ptr, ptr %unreg_field_0504, align 8
  call void @tl_mem_unregister(ptr %field_val505)
  %unreg_field_1506 = getelementptr inbounds %Linear, ptr %field_val503, i32 0, i32 1
  %field_val507 = load ptr, ptr %unreg_field_1506, align 8
  call void @tl_mem_unregister(ptr %field_val507)
  call void @tl_mem_exit_scope()
  ret ptr %s333

recursive_free_struct327:                         ; preds = %free_old_val324
  %free_field_0329 = getelementptr inbounds %Linear, ptr %old_field_val322, i32 0, i32 0
  %field_val_to_free330 = load ptr, ptr %free_field_0329, align 8
  call void @tl_tensor_free(ptr %field_val_to_free330)
  %free_field_1331 = getelementptr inbounds %Linear, ptr %old_field_val322, i32 0, i32 1
  %field_val_to_free332 = load ptr, ptr %free_field_1331, align 8
  call void @tl_tensor_free(ptr %field_val_to_free332)
  call void @free(ptr %old_field_val322)
  br label %continue_after_recursive_free328

continue_after_recursive_free328:                 ; preds = %recursive_free_struct327, %free_old_val324
  br label %skip_free325
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
  %for_idx = phi i64 [ %next_idx, %merge ], [ 0, %entry ]
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
  call void @tl_print_string(ptr @str_literal)
  %idx8 = load i64, ptr %idx, align 8
  call void @tl_print_i64(i64 %idx8)
  call void @tl_print_string(ptr @str_literal.104)
  %max_val9 = load float, ptr %max_val, align 4
  call void @tl_print_f32(float %max_val9)
  %idx10 = load i64, ptr %idx, align 8
  call void @tl_mem_exit_scope()
  ret i64 %idx10

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
  %p3 = alloca i64, align 16
  %log3_flat = alloca ptr, align 16
  %dims_alloca434 = alloca [2 x i64], align 8
  %log3 = alloca ptr, align 16
  %inp3 = alloca ptr, align 16
  %dims_alloca425 = alloca [2 x i64], align 8
  %d3 = alloca ptr, align 16
  %vp2 = alloca float, align 16
  %scalar_shape396 = alloca i64, align 16
  %scalar_data395 = alloca float, align 16
  %scalar_shape393 = alloca i64, align 16
  %scalar_data391 = alloca float, align 16
  %p2 = alloca i64, align 16
  %log2_flat = alloca ptr, align 16
  %dims_alloca382 = alloca [2 x i64], align 8
  %log2 = alloca ptr, align 16
  %inp2 = alloca ptr, align 16
  %dims_alloca373 = alloca [2 x i64], align 8
  %d2 = alloca ptr, align 16
  %vp1 = alloca float, align 16
  %scalar_shape345 = alloca i64, align 16
  %scalar_data344 = alloca float, align 16
  %scalar_shape342 = alloca i64, align 16
  %scalar_data340 = alloca float, align 16
  %p1 = alloca i64, align 16
  %log1_flat = alloca ptr, align 16
  %dims_alloca331 = alloca [2 x i64], align 8
  %log1 = alloca ptr, align 16
  %inp1 = alloca ptr, align 16
  %dims_alloca322 = alloca [2 x i64], align 8
  %d1 = alloca ptr, align 16
  %y5 = alloca float, align 16
  %y4 = alloca float, align 16
  %y3 = alloca float, align 16
  %y2 = alloca float, align 16
  %y1 = alloca float, align 16
  %y0 = alloca float, align 16
  %pred3 = alloca i64, align 16
  %logits3_flat = alloca ptr, align 16
  %dims_alloca288 = alloca [2 x i64], align 8
  %logits3 = alloca ptr, align 16
  %input3 = alloca ptr, align 16
  %dims_alloca279 = alloca [2 x i64], align 8
  %data3 = alloca ptr, align 16
  %val_pred2 = alloca float, align 16
  %scalar_shape249 = alloca i64, align 16
  %scalar_data248 = alloca float, align 16
  %scalar_shape246 = alloca i64, align 16
  %scalar_data244 = alloca float, align 16
  %pred2 = alloca i64, align 16
  %logits2_flat = alloca ptr, align 16
  %dims_alloca235 = alloca [2 x i64], align 8
  %logits2 = alloca ptr, align 16
  %input2 = alloca ptr, align 16
  %dims_alloca226 = alloca [2 x i64], align 8
  %data2 = alloca ptr, align 16
  %val_pred1 = alloca float, align 16
  %scalar_shape198 = alloca i64, align 16
  %scalar_data197 = alloca float, align 16
  %scalar_shape = alloca i64, align 16
  %scalar_data = alloca float, align 16
  %pred1 = alloca i64, align 16
  %logits1_flat = alloca ptr, align 16
  %dims_alloca190 = alloca [2 x i64], align 8
  %logits1 = alloca ptr, align 16
  %input1 = alloca ptr, align 16
  %dims_alloca = alloca [2 x i64], align 8
  %data1 = alloca ptr, align 16
  %val_pad = alloca float, align 16
  %x5 = alloca float, align 16
  %x4 = alloca float, align 16
  %x3 = alloca float, align 16
  %x2 = alloca float, align 16
  %x1 = alloca float, align 16
  %x0 = alloca float, align 16
  %model = alloca ptr, align 16
  %d_model = alloca i64, align 16
  %vocab_size = alloca i64, align 16
  call void @tl_mem_enter_scope()
  call void @tl_arena_init(i64 2819072)
  store i64 13, ptr %vocab_size, align 8
  store i64 128, ptr %d_model, align 8
  call void @tl_print_string(ptr @str_literal.105)
  %vocab_size1 = load i64, ptr %vocab_size, align 8
  %d_model2 = load i64, ptr %d_model, align 8
  %static_call = call ptr @tl_GPT_new(i64 %vocab_size1, i64 %d_model2)
  call void @tl_mem_unregister(ptr %static_call)
  store ptr %static_call, ptr %model, align 8
  call void @tl_print_string(ptr @str_literal.106)
  call void @tl_print_string(ptr @str_literal.107)
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
  call void @tl_add_parameter(ptr @key_str.108, ptr %w8)
  %b1 = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 2
  %sub_ptr9 = load ptr, ptr %b1, align 8
  %l1 = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 0
  %sub_ptr10 = load ptr, ptr %l1, align 8
  %w11 = getelementptr inbounds %LayerNorm, ptr %sub_ptr10, i32 0, i32 0
  %w12 = load ptr, ptr %w11, align 8
  call void @tl_add_parameter(ptr @key_str.109, ptr %w12)
  %b = getelementptr inbounds %LayerNorm, ptr %sub_ptr10, i32 0, i32 1
  %b13 = load ptr, ptr %b, align 8
  call void @tl_add_parameter(ptr @key_str.110, ptr %b13)
  %a = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 1
  %sub_ptr14 = load ptr, ptr %a, align 8
  %q_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 0
  %sub_ptr15 = load ptr, ptr %q_proj, align 8
  %W = getelementptr inbounds %Linear, ptr %sub_ptr15, i32 0, i32 0
  %W16 = load ptr, ptr %W, align 8
  call void @tl_add_parameter(ptr @key_str.111, ptr %W16)
  %b17 = getelementptr inbounds %Linear, ptr %sub_ptr15, i32 0, i32 1
  %b18 = load ptr, ptr %b17, align 8
  call void @tl_add_parameter(ptr @key_str.112, ptr %b18)
  %k_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 1
  %sub_ptr19 = load ptr, ptr %k_proj, align 8
  %W20 = getelementptr inbounds %Linear, ptr %sub_ptr19, i32 0, i32 0
  %W21 = load ptr, ptr %W20, align 8
  call void @tl_add_parameter(ptr @key_str.113, ptr %W21)
  %b22 = getelementptr inbounds %Linear, ptr %sub_ptr19, i32 0, i32 1
  %b23 = load ptr, ptr %b22, align 8
  call void @tl_add_parameter(ptr @key_str.114, ptr %b23)
  %v_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 2
  %sub_ptr24 = load ptr, ptr %v_proj, align 8
  %W25 = getelementptr inbounds %Linear, ptr %sub_ptr24, i32 0, i32 0
  %W26 = load ptr, ptr %W25, align 8
  call void @tl_add_parameter(ptr @key_str.115, ptr %W26)
  %b27 = getelementptr inbounds %Linear, ptr %sub_ptr24, i32 0, i32 1
  %b28 = load ptr, ptr %b27, align 8
  call void @tl_add_parameter(ptr @key_str.116, ptr %b28)
  %p_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 3
  %sub_ptr29 = load ptr, ptr %p_proj, align 8
  %W30 = getelementptr inbounds %Linear, ptr %sub_ptr29, i32 0, i32 0
  %W31 = load ptr, ptr %W30, align 8
  call void @tl_add_parameter(ptr @key_str.117, ptr %W31)
  %b32 = getelementptr inbounds %Linear, ptr %sub_ptr29, i32 0, i32 1
  %b33 = load ptr, ptr %b32, align 8
  call void @tl_add_parameter(ptr @key_str.118, ptr %b33)
  %l2 = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 2
  %sub_ptr34 = load ptr, ptr %l2, align 8
  %w35 = getelementptr inbounds %LayerNorm, ptr %sub_ptr34, i32 0, i32 0
  %w36 = load ptr, ptr %w35, align 8
  call void @tl_add_parameter(ptr @key_str.119, ptr %w36)
  %b37 = getelementptr inbounds %LayerNorm, ptr %sub_ptr34, i32 0, i32 1
  %b38 = load ptr, ptr %b37, align 8
  call void @tl_add_parameter(ptr @key_str.120, ptr %b38)
  %m = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 3
  %sub_ptr39 = load ptr, ptr %m, align 8
  %f = getelementptr inbounds %MLP, ptr %sub_ptr39, i32 0, i32 0
  %sub_ptr40 = load ptr, ptr %f, align 8
  %W41 = getelementptr inbounds %Linear, ptr %sub_ptr40, i32 0, i32 0
  %W42 = load ptr, ptr %W41, align 8
  call void @tl_add_parameter(ptr @key_str.121, ptr %W42)
  %b43 = getelementptr inbounds %Linear, ptr %sub_ptr40, i32 0, i32 1
  %b44 = load ptr, ptr %b43, align 8
  call void @tl_add_parameter(ptr @key_str.122, ptr %b44)
  %p = getelementptr inbounds %MLP, ptr %sub_ptr39, i32 0, i32 1
  %sub_ptr45 = load ptr, ptr %p, align 8
  %W46 = getelementptr inbounds %Linear, ptr %sub_ptr45, i32 0, i32 0
  %W47 = load ptr, ptr %W46, align 8
  call void @tl_add_parameter(ptr @key_str.123, ptr %W47)
  %b48 = getelementptr inbounds %Linear, ptr %sub_ptr45, i32 0, i32 1
  %b49 = load ptr, ptr %b48, align 8
  call void @tl_add_parameter(ptr @key_str.124, ptr %b49)
  %b2 = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 3
  %sub_ptr50 = load ptr, ptr %b2, align 8
  %l151 = getelementptr inbounds %Block, ptr %sub_ptr50, i32 0, i32 0
  %sub_ptr52 = load ptr, ptr %l151, align 8
  %w53 = getelementptr inbounds %LayerNorm, ptr %sub_ptr52, i32 0, i32 0
  %w54 = load ptr, ptr %w53, align 8
  call void @tl_add_parameter(ptr @key_str.125, ptr %w54)
  %b55 = getelementptr inbounds %LayerNorm, ptr %sub_ptr52, i32 0, i32 1
  %b56 = load ptr, ptr %b55, align 8
  call void @tl_add_parameter(ptr @key_str.126, ptr %b56)
  %a57 = getelementptr inbounds %Block, ptr %sub_ptr50, i32 0, i32 1
  %sub_ptr58 = load ptr, ptr %a57, align 8
  %q_proj59 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr58, i32 0, i32 0
  %sub_ptr60 = load ptr, ptr %q_proj59, align 8
  %W61 = getelementptr inbounds %Linear, ptr %sub_ptr60, i32 0, i32 0
  %W62 = load ptr, ptr %W61, align 8
  call void @tl_add_parameter(ptr @key_str.127, ptr %W62)
  %b63 = getelementptr inbounds %Linear, ptr %sub_ptr60, i32 0, i32 1
  %b64 = load ptr, ptr %b63, align 8
  call void @tl_add_parameter(ptr @key_str.128, ptr %b64)
  %k_proj65 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr58, i32 0, i32 1
  %sub_ptr66 = load ptr, ptr %k_proj65, align 8
  %W67 = getelementptr inbounds %Linear, ptr %sub_ptr66, i32 0, i32 0
  %W68 = load ptr, ptr %W67, align 8
  call void @tl_add_parameter(ptr @key_str.129, ptr %W68)
  %b69 = getelementptr inbounds %Linear, ptr %sub_ptr66, i32 0, i32 1
  %b70 = load ptr, ptr %b69, align 8
  call void @tl_add_parameter(ptr @key_str.130, ptr %b70)
  %v_proj71 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr58, i32 0, i32 2
  %sub_ptr72 = load ptr, ptr %v_proj71, align 8
  %W73 = getelementptr inbounds %Linear, ptr %sub_ptr72, i32 0, i32 0
  %W74 = load ptr, ptr %W73, align 8
  call void @tl_add_parameter(ptr @key_str.131, ptr %W74)
  %b75 = getelementptr inbounds %Linear, ptr %sub_ptr72, i32 0, i32 1
  %b76 = load ptr, ptr %b75, align 8
  call void @tl_add_parameter(ptr @key_str.132, ptr %b76)
  %p_proj77 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr58, i32 0, i32 3
  %sub_ptr78 = load ptr, ptr %p_proj77, align 8
  %W79 = getelementptr inbounds %Linear, ptr %sub_ptr78, i32 0, i32 0
  %W80 = load ptr, ptr %W79, align 8
  call void @tl_add_parameter(ptr @key_str.133, ptr %W80)
  %b81 = getelementptr inbounds %Linear, ptr %sub_ptr78, i32 0, i32 1
  %b82 = load ptr, ptr %b81, align 8
  call void @tl_add_parameter(ptr @key_str.134, ptr %b82)
  %l283 = getelementptr inbounds %Block, ptr %sub_ptr50, i32 0, i32 2
  %sub_ptr84 = load ptr, ptr %l283, align 8
  %w85 = getelementptr inbounds %LayerNorm, ptr %sub_ptr84, i32 0, i32 0
  %w86 = load ptr, ptr %w85, align 8
  call void @tl_add_parameter(ptr @key_str.135, ptr %w86)
  %b87 = getelementptr inbounds %LayerNorm, ptr %sub_ptr84, i32 0, i32 1
  %b88 = load ptr, ptr %b87, align 8
  call void @tl_add_parameter(ptr @key_str.136, ptr %b88)
  %m89 = getelementptr inbounds %Block, ptr %sub_ptr50, i32 0, i32 3
  %sub_ptr90 = load ptr, ptr %m89, align 8
  %f91 = getelementptr inbounds %MLP, ptr %sub_ptr90, i32 0, i32 0
  %sub_ptr92 = load ptr, ptr %f91, align 8
  %W93 = getelementptr inbounds %Linear, ptr %sub_ptr92, i32 0, i32 0
  %W94 = load ptr, ptr %W93, align 8
  call void @tl_add_parameter(ptr @key_str.137, ptr %W94)
  %b95 = getelementptr inbounds %Linear, ptr %sub_ptr92, i32 0, i32 1
  %b96 = load ptr, ptr %b95, align 8
  call void @tl_add_parameter(ptr @key_str.138, ptr %b96)
  %p97 = getelementptr inbounds %MLP, ptr %sub_ptr90, i32 0, i32 1
  %sub_ptr98 = load ptr, ptr %p97, align 8
  %W99 = getelementptr inbounds %Linear, ptr %sub_ptr98, i32 0, i32 0
  %W100 = load ptr, ptr %W99, align 8
  call void @tl_add_parameter(ptr @key_str.139, ptr %W100)
  %b101 = getelementptr inbounds %Linear, ptr %sub_ptr98, i32 0, i32 1
  %b102 = load ptr, ptr %b101, align 8
  call void @tl_add_parameter(ptr @key_str.140, ptr %b102)
  %b3 = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 4
  %sub_ptr103 = load ptr, ptr %b3, align 8
  %l1104 = getelementptr inbounds %Block, ptr %sub_ptr103, i32 0, i32 0
  %sub_ptr105 = load ptr, ptr %l1104, align 8
  %w106 = getelementptr inbounds %LayerNorm, ptr %sub_ptr105, i32 0, i32 0
  %w107 = load ptr, ptr %w106, align 8
  call void @tl_add_parameter(ptr @key_str.141, ptr %w107)
  %b108 = getelementptr inbounds %LayerNorm, ptr %sub_ptr105, i32 0, i32 1
  %b109 = load ptr, ptr %b108, align 8
  call void @tl_add_parameter(ptr @key_str.142, ptr %b109)
  %a110 = getelementptr inbounds %Block, ptr %sub_ptr103, i32 0, i32 1
  %sub_ptr111 = load ptr, ptr %a110, align 8
  %q_proj112 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr111, i32 0, i32 0
  %sub_ptr113 = load ptr, ptr %q_proj112, align 8
  %W114 = getelementptr inbounds %Linear, ptr %sub_ptr113, i32 0, i32 0
  %W115 = load ptr, ptr %W114, align 8
  call void @tl_add_parameter(ptr @key_str.143, ptr %W115)
  %b116 = getelementptr inbounds %Linear, ptr %sub_ptr113, i32 0, i32 1
  %b117 = load ptr, ptr %b116, align 8
  call void @tl_add_parameter(ptr @key_str.144, ptr %b117)
  %k_proj118 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr111, i32 0, i32 1
  %sub_ptr119 = load ptr, ptr %k_proj118, align 8
  %W120 = getelementptr inbounds %Linear, ptr %sub_ptr119, i32 0, i32 0
  %W121 = load ptr, ptr %W120, align 8
  call void @tl_add_parameter(ptr @key_str.145, ptr %W121)
  %b122 = getelementptr inbounds %Linear, ptr %sub_ptr119, i32 0, i32 1
  %b123 = load ptr, ptr %b122, align 8
  call void @tl_add_parameter(ptr @key_str.146, ptr %b123)
  %v_proj124 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr111, i32 0, i32 2
  %sub_ptr125 = load ptr, ptr %v_proj124, align 8
  %W126 = getelementptr inbounds %Linear, ptr %sub_ptr125, i32 0, i32 0
  %W127 = load ptr, ptr %W126, align 8
  call void @tl_add_parameter(ptr @key_str.147, ptr %W127)
  %b128 = getelementptr inbounds %Linear, ptr %sub_ptr125, i32 0, i32 1
  %b129 = load ptr, ptr %b128, align 8
  call void @tl_add_parameter(ptr @key_str.148, ptr %b129)
  %p_proj130 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr111, i32 0, i32 3
  %sub_ptr131 = load ptr, ptr %p_proj130, align 8
  %W132 = getelementptr inbounds %Linear, ptr %sub_ptr131, i32 0, i32 0
  %W133 = load ptr, ptr %W132, align 8
  call void @tl_add_parameter(ptr @key_str.149, ptr %W133)
  %b134 = getelementptr inbounds %Linear, ptr %sub_ptr131, i32 0, i32 1
  %b135 = load ptr, ptr %b134, align 8
  call void @tl_add_parameter(ptr @key_str.150, ptr %b135)
  %l2136 = getelementptr inbounds %Block, ptr %sub_ptr103, i32 0, i32 2
  %sub_ptr137 = load ptr, ptr %l2136, align 8
  %w138 = getelementptr inbounds %LayerNorm, ptr %sub_ptr137, i32 0, i32 0
  %w139 = load ptr, ptr %w138, align 8
  call void @tl_add_parameter(ptr @key_str.151, ptr %w139)
  %b140 = getelementptr inbounds %LayerNorm, ptr %sub_ptr137, i32 0, i32 1
  %b141 = load ptr, ptr %b140, align 8
  call void @tl_add_parameter(ptr @key_str.152, ptr %b141)
  %m142 = getelementptr inbounds %Block, ptr %sub_ptr103, i32 0, i32 3
  %sub_ptr143 = load ptr, ptr %m142, align 8
  %f144 = getelementptr inbounds %MLP, ptr %sub_ptr143, i32 0, i32 0
  %sub_ptr145 = load ptr, ptr %f144, align 8
  %W146 = getelementptr inbounds %Linear, ptr %sub_ptr145, i32 0, i32 0
  %W147 = load ptr, ptr %W146, align 8
  call void @tl_add_parameter(ptr @key_str.153, ptr %W147)
  %b148 = getelementptr inbounds %Linear, ptr %sub_ptr145, i32 0, i32 1
  %b149 = load ptr, ptr %b148, align 8
  call void @tl_add_parameter(ptr @key_str.154, ptr %b149)
  %p150 = getelementptr inbounds %MLP, ptr %sub_ptr143, i32 0, i32 1
  %sub_ptr151 = load ptr, ptr %p150, align 8
  %W152 = getelementptr inbounds %Linear, ptr %sub_ptr151, i32 0, i32 0
  %W153 = load ptr, ptr %W152, align 8
  call void @tl_add_parameter(ptr @key_str.155, ptr %W153)
  %b154 = getelementptr inbounds %Linear, ptr %sub_ptr151, i32 0, i32 1
  %b155 = load ptr, ptr %b154, align 8
  call void @tl_add_parameter(ptr @key_str.156, ptr %b155)
  %l = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 5
  %sub_ptr156 = load ptr, ptr %l, align 8
  %w157 = getelementptr inbounds %LayerNorm, ptr %sub_ptr156, i32 0, i32 0
  %w158 = load ptr, ptr %w157, align 8
  call void @tl_add_parameter(ptr @key_str.157, ptr %w158)
  %b159 = getelementptr inbounds %LayerNorm, ptr %sub_ptr156, i32 0, i32 1
  %b160 = load ptr, ptr %b159, align 8
  call void @tl_add_parameter(ptr @key_str.158, ptr %b160)
  %h = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 6
  %sub_ptr161 = load ptr, ptr %h, align 8
  %W162 = getelementptr inbounds %Linear, ptr %sub_ptr161, i32 0, i32 0
  %W163 = load ptr, ptr %W162, align 8
  call void @tl_add_parameter(ptr @key_str.159, ptr %W163)
  %b164 = getelementptr inbounds %Linear, ptr %sub_ptr161, i32 0, i32 1
  %b165 = load ptr, ptr %b164, align 8
  call void @tl_add_parameter(ptr @key_str.160, ptr %b165)
  call void @tl_load_all_params(ptr @str_literal.161)
  call void @tl_print_string(ptr @str_literal.162)
  call void @tl_print_string(ptr @str_literal.163)
  store float 2.000000e+00, ptr %x0, align 4
  store float 1.000000e+00, ptr %x1, align 4
  store float 1.000000e+01, ptr %x2, align 4
  store float 4.000000e+00, ptr %x3, align 4
  store float 3.000000e+00, ptr %x4, align 4
  store float 1.100000e+01, ptr %x5, align 4
  store float 1.200000e+01, ptr %val_pad, align 4
  %buf_void = call ptr @tl_alloc_tmp(i64 48)
  %x0166 = load float, ptr %x0, align 4
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %x0166, ptr %elem_ptr, align 4
  %x1167 = load float, ptr %x1, align 4
  %elem_ptr168 = getelementptr inbounds float, ptr %buf_void, i64 1
  store float %x1167, ptr %elem_ptr168, align 4
  %x2169 = load float, ptr %x2, align 4
  %elem_ptr170 = getelementptr inbounds float, ptr %buf_void, i64 2
  store float %x2169, ptr %elem_ptr170, align 4
  %x3171 = load float, ptr %x3, align 4
  %elem_ptr172 = getelementptr inbounds float, ptr %buf_void, i64 3
  store float %x3171, ptr %elem_ptr172, align 4
  %x4173 = load float, ptr %x4, align 4
  %elem_ptr174 = getelementptr inbounds float, ptr %buf_void, i64 4
  store float %x4173, ptr %elem_ptr174, align 4
  %x5175 = load float, ptr %x5, align 4
  %elem_ptr176 = getelementptr inbounds float, ptr %buf_void, i64 5
  store float %x5175, ptr %elem_ptr176, align 4
  %val_pad177 = load float, ptr %val_pad, align 4
  %elem_ptr178 = getelementptr inbounds float, ptr %buf_void, i64 6
  store float %val_pad177, ptr %elem_ptr178, align 4
  %val_pad179 = load float, ptr %val_pad, align 4
  %elem_ptr180 = getelementptr inbounds float, ptr %buf_void, i64 7
  store float %val_pad179, ptr %elem_ptr180, align 4
  %val_pad181 = load float, ptr %val_pad, align 4
  %elem_ptr182 = getelementptr inbounds float, ptr %buf_void, i64 8
  store float %val_pad181, ptr %elem_ptr182, align 4
  %elem_ptr183 = getelementptr inbounds float, ptr %buf_void, i64 9
  store float 1.200000e+01, ptr %elem_ptr183, align 4
  %elem_ptr184 = getelementptr inbounds float, ptr %buf_void, i64 10
  store float 1.200000e+01, ptr %elem_ptr184, align 4
  %elem_ptr185 = getelementptr inbounds float, ptr %buf_void, i64 11
  store float 1.200000e+01, ptr %elem_ptr185, align 4
  %shape_alloc = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr = getelementptr inbounds i64, ptr %shape_alloc, i64 0
  store i64 12, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %shape_alloc)
  call void @tl_free_tmp(ptr %buf_void)
  call void @tl_free_tmp(ptr %shape_alloc)
  call void @tl_mem_unregister(ptr %new_tensor)
  store ptr %new_tensor, ptr %data1, align 8
  %data1186 = load ptr, ptr %data1, align 8
  %dim_ptr_0 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0, align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %data1186, ptr %dims_ptr, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res)
  store ptr %reshape_dims_res, ptr %input1, align 8
  %model187 = load ptr, ptr %model, align 8
  %input1188 = load ptr, ptr %input1, align 8
  %call_method = call ptr @tl_GPT_forward(ptr %model187, ptr %input1188)
  call void @tl_mem_register_tensor(ptr %call_method)
  call void @tl_mem_unregister(ptr %call_method)
  store ptr %call_method, ptr %logits1, align 8
  %logits1189 = load ptr, ptr %logits1, align 8
  %dim_ptr_0191 = getelementptr [2 x i64], ptr %dims_alloca190, i64 0, i64 0
  store i64 12, ptr %dim_ptr_0191, align 8
  %dim_ptr192 = getelementptr [2 x i64], ptr %dims_alloca190, i64 0, i64 1
  store i64 13, ptr %dim_ptr192, align 8
  %dims_ptr193 = getelementptr [2 x i64], ptr %dims_alloca190, i64 0, i64 0
  %reshape_dims_res194 = call ptr @tl_tensor_reshape_dims(ptr %logits1189, ptr %dims_ptr193, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res194)
  store ptr %reshape_dims_res194, ptr %logits1_flat, align 8
  %logits1_flat195 = load ptr, ptr %logits1_flat, align 8
  %slice_res = call ptr @tl_tensor_slice(ptr %logits1_flat195, i64 5, i64 1)
  %call_tmp = call i64 @argmax(ptr %slice_res)
  call void @tl_tensor_free(ptr %slice_res)
  store i64 %call_tmp, ptr %pred1, align 8
  %pred1196 = load i64, ptr %pred1, align 8
  %cast_i64_f32 = sitofp i64 %pred1196 to float
  store float %cast_i64_f32, ptr %scalar_data, align 4
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  store float 1.000000e+00, ptr %scalar_data197, align 4
  %scalar_tensor199 = call ptr @tl_tensor_new(ptr %scalar_data197, i64 0, ptr %scalar_shape198)
  %pow_res = call ptr @tl_tensor_pow(ptr %scalar_tensor, ptr %scalar_tensor199)
  call void @tl_tensor_free(ptr %scalar_tensor199)
  %get_res = call float @tl_tensor_get(ptr %pow_res, i64 0)
  call void @tl_tensor_free(ptr %pow_res)
  store float %get_res, ptr %val_pred1, align 4
  %buf_void200 = call ptr @tl_alloc_tmp(i64 48)
  %x0201 = load float, ptr %x0, align 4
  %elem_ptr202 = getelementptr inbounds float, ptr %buf_void200, i64 0
  store float %x0201, ptr %elem_ptr202, align 4
  %x1203 = load float, ptr %x1, align 4
  %elem_ptr204 = getelementptr inbounds float, ptr %buf_void200, i64 1
  store float %x1203, ptr %elem_ptr204, align 4
  %x2205 = load float, ptr %x2, align 4
  %elem_ptr206 = getelementptr inbounds float, ptr %buf_void200, i64 2
  store float %x2205, ptr %elem_ptr206, align 4
  %x3207 = load float, ptr %x3, align 4
  %elem_ptr208 = getelementptr inbounds float, ptr %buf_void200, i64 3
  store float %x3207, ptr %elem_ptr208, align 4
  %x4209 = load float, ptr %x4, align 4
  %elem_ptr210 = getelementptr inbounds float, ptr %buf_void200, i64 4
  store float %x4209, ptr %elem_ptr210, align 4
  %x5211 = load float, ptr %x5, align 4
  %elem_ptr212 = getelementptr inbounds float, ptr %buf_void200, i64 5
  store float %x5211, ptr %elem_ptr212, align 4
  %val_pred1213 = load float, ptr %val_pred1, align 4
  %elem_ptr214 = getelementptr inbounds float, ptr %buf_void200, i64 6
  store float %val_pred1213, ptr %elem_ptr214, align 4
  %val_pad215 = load float, ptr %val_pad, align 4
  %elem_ptr216 = getelementptr inbounds float, ptr %buf_void200, i64 7
  store float %val_pad215, ptr %elem_ptr216, align 4
  %val_pad217 = load float, ptr %val_pad, align 4
  %elem_ptr218 = getelementptr inbounds float, ptr %buf_void200, i64 8
  store float %val_pad217, ptr %elem_ptr218, align 4
  %elem_ptr219 = getelementptr inbounds float, ptr %buf_void200, i64 9
  store float 1.200000e+01, ptr %elem_ptr219, align 4
  %elem_ptr220 = getelementptr inbounds float, ptr %buf_void200, i64 10
  store float 1.200000e+01, ptr %elem_ptr220, align 4
  %elem_ptr221 = getelementptr inbounds float, ptr %buf_void200, i64 11
  store float 1.200000e+01, ptr %elem_ptr221, align 4
  %shape_alloc222 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr223 = getelementptr inbounds i64, ptr %shape_alloc222, i64 0
  store i64 12, ptr %shape_ptr223, align 8
  %new_tensor224 = call ptr @tl_tensor_new(ptr %buf_void200, i64 1, ptr %shape_alloc222)
  call void @tl_free_tmp(ptr %buf_void200)
  call void @tl_free_tmp(ptr %shape_alloc222)
  call void @tl_mem_unregister(ptr %new_tensor224)
  store ptr %new_tensor224, ptr %data2, align 8
  %data2225 = load ptr, ptr %data2, align 8
  %dim_ptr_0227 = getelementptr [2 x i64], ptr %dims_alloca226, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0227, align 8
  %dim_ptr228 = getelementptr [2 x i64], ptr %dims_alloca226, i64 0, i64 1
  store i64 12, ptr %dim_ptr228, align 8
  %dims_ptr229 = getelementptr [2 x i64], ptr %dims_alloca226, i64 0, i64 0
  %reshape_dims_res230 = call ptr @tl_tensor_reshape_dims(ptr %data2225, ptr %dims_ptr229, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res230)
  store ptr %reshape_dims_res230, ptr %input2, align 8
  %model231 = load ptr, ptr %model, align 8
  %input2232 = load ptr, ptr %input2, align 8
  %call_method233 = call ptr @tl_GPT_forward(ptr %model231, ptr %input2232)
  call void @tl_mem_register_tensor(ptr %call_method233)
  call void @tl_mem_unregister(ptr %call_method233)
  store ptr %call_method233, ptr %logits2, align 8
  %logits2234 = load ptr, ptr %logits2, align 8
  %dim_ptr_0236 = getelementptr [2 x i64], ptr %dims_alloca235, i64 0, i64 0
  store i64 12, ptr %dim_ptr_0236, align 8
  %dim_ptr237 = getelementptr [2 x i64], ptr %dims_alloca235, i64 0, i64 1
  store i64 13, ptr %dim_ptr237, align 8
  %dims_ptr238 = getelementptr [2 x i64], ptr %dims_alloca235, i64 0, i64 0
  %reshape_dims_res239 = call ptr @tl_tensor_reshape_dims(ptr %logits2234, ptr %dims_ptr238, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res239)
  store ptr %reshape_dims_res239, ptr %logits2_flat, align 8
  %logits2_flat240 = load ptr, ptr %logits2_flat, align 8
  %slice_res241 = call ptr @tl_tensor_slice(ptr %logits2_flat240, i64 6, i64 1)
  %call_tmp242 = call i64 @argmax(ptr %slice_res241)
  call void @tl_tensor_free(ptr %slice_res241)
  store i64 %call_tmp242, ptr %pred2, align 8
  %pred2243 = load i64, ptr %pred2, align 8
  %cast_i64_f32245 = sitofp i64 %pred2243 to float
  store float %cast_i64_f32245, ptr %scalar_data244, align 4
  %scalar_tensor247 = call ptr @tl_tensor_new(ptr %scalar_data244, i64 0, ptr %scalar_shape246)
  store float 1.000000e+00, ptr %scalar_data248, align 4
  %scalar_tensor250 = call ptr @tl_tensor_new(ptr %scalar_data248, i64 0, ptr %scalar_shape249)
  %pow_res251 = call ptr @tl_tensor_pow(ptr %scalar_tensor247, ptr %scalar_tensor250)
  call void @tl_tensor_free(ptr %scalar_tensor250)
  %get_res252 = call float @tl_tensor_get(ptr %pow_res251, i64 0)
  call void @tl_tensor_free(ptr %pow_res251)
  store float %get_res252, ptr %val_pred2, align 4
  %buf_void253 = call ptr @tl_alloc_tmp(i64 48)
  %x0254 = load float, ptr %x0, align 4
  %elem_ptr255 = getelementptr inbounds float, ptr %buf_void253, i64 0
  store float %x0254, ptr %elem_ptr255, align 4
  %x1256 = load float, ptr %x1, align 4
  %elem_ptr257 = getelementptr inbounds float, ptr %buf_void253, i64 1
  store float %x1256, ptr %elem_ptr257, align 4
  %x2258 = load float, ptr %x2, align 4
  %elem_ptr259 = getelementptr inbounds float, ptr %buf_void253, i64 2
  store float %x2258, ptr %elem_ptr259, align 4
  %x3260 = load float, ptr %x3, align 4
  %elem_ptr261 = getelementptr inbounds float, ptr %buf_void253, i64 3
  store float %x3260, ptr %elem_ptr261, align 4
  %x4262 = load float, ptr %x4, align 4
  %elem_ptr263 = getelementptr inbounds float, ptr %buf_void253, i64 4
  store float %x4262, ptr %elem_ptr263, align 4
  %x5264 = load float, ptr %x5, align 4
  %elem_ptr265 = getelementptr inbounds float, ptr %buf_void253, i64 5
  store float %x5264, ptr %elem_ptr265, align 4
  %val_pred1266 = load float, ptr %val_pred1, align 4
  %elem_ptr267 = getelementptr inbounds float, ptr %buf_void253, i64 6
  store float %val_pred1266, ptr %elem_ptr267, align 4
  %val_pred2268 = load float, ptr %val_pred2, align 4
  %elem_ptr269 = getelementptr inbounds float, ptr %buf_void253, i64 7
  store float %val_pred2268, ptr %elem_ptr269, align 4
  %val_pad270 = load float, ptr %val_pad, align 4
  %elem_ptr271 = getelementptr inbounds float, ptr %buf_void253, i64 8
  store float %val_pad270, ptr %elem_ptr271, align 4
  %elem_ptr272 = getelementptr inbounds float, ptr %buf_void253, i64 9
  store float 1.200000e+01, ptr %elem_ptr272, align 4
  %elem_ptr273 = getelementptr inbounds float, ptr %buf_void253, i64 10
  store float 1.200000e+01, ptr %elem_ptr273, align 4
  %elem_ptr274 = getelementptr inbounds float, ptr %buf_void253, i64 11
  store float 1.200000e+01, ptr %elem_ptr274, align 4
  %shape_alloc275 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr276 = getelementptr inbounds i64, ptr %shape_alloc275, i64 0
  store i64 12, ptr %shape_ptr276, align 8
  %new_tensor277 = call ptr @tl_tensor_new(ptr %buf_void253, i64 1, ptr %shape_alloc275)
  call void @tl_free_tmp(ptr %buf_void253)
  call void @tl_free_tmp(ptr %shape_alloc275)
  call void @tl_mem_unregister(ptr %new_tensor277)
  store ptr %new_tensor277, ptr %data3, align 8
  %data3278 = load ptr, ptr %data3, align 8
  %dim_ptr_0280 = getelementptr [2 x i64], ptr %dims_alloca279, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0280, align 8
  %dim_ptr281 = getelementptr [2 x i64], ptr %dims_alloca279, i64 0, i64 1
  store i64 12, ptr %dim_ptr281, align 8
  %dims_ptr282 = getelementptr [2 x i64], ptr %dims_alloca279, i64 0, i64 0
  %reshape_dims_res283 = call ptr @tl_tensor_reshape_dims(ptr %data3278, ptr %dims_ptr282, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res283)
  store ptr %reshape_dims_res283, ptr %input3, align 8
  %model284 = load ptr, ptr %model, align 8
  %input3285 = load ptr, ptr %input3, align 8
  %call_method286 = call ptr @tl_GPT_forward(ptr %model284, ptr %input3285)
  call void @tl_mem_register_tensor(ptr %call_method286)
  call void @tl_mem_unregister(ptr %call_method286)
  store ptr %call_method286, ptr %logits3, align 8
  %logits3287 = load ptr, ptr %logits3, align 8
  %dim_ptr_0289 = getelementptr [2 x i64], ptr %dims_alloca288, i64 0, i64 0
  store i64 12, ptr %dim_ptr_0289, align 8
  %dim_ptr290 = getelementptr [2 x i64], ptr %dims_alloca288, i64 0, i64 1
  store i64 13, ptr %dim_ptr290, align 8
  %dims_ptr291 = getelementptr [2 x i64], ptr %dims_alloca288, i64 0, i64 0
  %reshape_dims_res292 = call ptr @tl_tensor_reshape_dims(ptr %logits3287, ptr %dims_ptr291, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res292)
  store ptr %reshape_dims_res292, ptr %logits3_flat, align 8
  %logits3_flat293 = load ptr, ptr %logits3_flat, align 8
  %slice_res294 = call ptr @tl_tensor_slice(ptr %logits3_flat293, i64 7, i64 1)
  %call_tmp295 = call i64 @argmax(ptr %slice_res294)
  call void @tl_tensor_free(ptr %slice_res294)
  store i64 %call_tmp295, ptr %pred3, align 8
  call void @tl_print_string(ptr @str_literal.164)
  %pred1296 = load i64, ptr %pred1, align 8
  call void @tl_print_i64(i64 %pred1296)
  %pred2297 = load i64, ptr %pred2, align 8
  call void @tl_print_i64(i64 %pred2297)
  %pred3298 = load i64, ptr %pred3, align 8
  call void @tl_print_i64(i64 %pred3298)
  call void @tl_print_string(ptr @str_literal.165)
  store float 9.000000e+00, ptr %y0, align 4
  store float 9.000000e+00, ptr %y1, align 4
  store float 1.000000e+01, ptr %y2, align 4
  store float 1.000000e+00, ptr %y3, align 4
  store float 0.000000e+00, ptr %y4, align 4
  store float 1.100000e+01, ptr %y5, align 4
  %buf_void299 = call ptr @tl_alloc_tmp(i64 48)
  %y0300 = load float, ptr %y0, align 4
  %elem_ptr301 = getelementptr inbounds float, ptr %buf_void299, i64 0
  store float %y0300, ptr %elem_ptr301, align 4
  %y1302 = load float, ptr %y1, align 4
  %elem_ptr303 = getelementptr inbounds float, ptr %buf_void299, i64 1
  store float %y1302, ptr %elem_ptr303, align 4
  %y2304 = load float, ptr %y2, align 4
  %elem_ptr305 = getelementptr inbounds float, ptr %buf_void299, i64 2
  store float %y2304, ptr %elem_ptr305, align 4
  %y3306 = load float, ptr %y3, align 4
  %elem_ptr307 = getelementptr inbounds float, ptr %buf_void299, i64 3
  store float %y3306, ptr %elem_ptr307, align 4
  %y4308 = load float, ptr %y4, align 4
  %elem_ptr309 = getelementptr inbounds float, ptr %buf_void299, i64 4
  store float %y4308, ptr %elem_ptr309, align 4
  %y5310 = load float, ptr %y5, align 4
  %elem_ptr311 = getelementptr inbounds float, ptr %buf_void299, i64 5
  store float %y5310, ptr %elem_ptr311, align 4
  %elem_ptr312 = getelementptr inbounds float, ptr %buf_void299, i64 6
  store float 1.200000e+01, ptr %elem_ptr312, align 4
  %elem_ptr313 = getelementptr inbounds float, ptr %buf_void299, i64 7
  store float 1.200000e+01, ptr %elem_ptr313, align 4
  %elem_ptr314 = getelementptr inbounds float, ptr %buf_void299, i64 8
  store float 1.200000e+01, ptr %elem_ptr314, align 4
  %elem_ptr315 = getelementptr inbounds float, ptr %buf_void299, i64 9
  store float 1.200000e+01, ptr %elem_ptr315, align 4
  %elem_ptr316 = getelementptr inbounds float, ptr %buf_void299, i64 10
  store float 1.200000e+01, ptr %elem_ptr316, align 4
  %elem_ptr317 = getelementptr inbounds float, ptr %buf_void299, i64 11
  store float 1.200000e+01, ptr %elem_ptr317, align 4
  %shape_alloc318 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr319 = getelementptr inbounds i64, ptr %shape_alloc318, i64 0
  store i64 12, ptr %shape_ptr319, align 8
  %new_tensor320 = call ptr @tl_tensor_new(ptr %buf_void299, i64 1, ptr %shape_alloc318)
  call void @tl_free_tmp(ptr %buf_void299)
  call void @tl_free_tmp(ptr %shape_alloc318)
  call void @tl_mem_unregister(ptr %new_tensor320)
  store ptr %new_tensor320, ptr %d1, align 8
  %d1321 = load ptr, ptr %d1, align 8
  %dim_ptr_0323 = getelementptr [2 x i64], ptr %dims_alloca322, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0323, align 8
  %dim_ptr324 = getelementptr [2 x i64], ptr %dims_alloca322, i64 0, i64 1
  store i64 12, ptr %dim_ptr324, align 8
  %dims_ptr325 = getelementptr [2 x i64], ptr %dims_alloca322, i64 0, i64 0
  %reshape_dims_res326 = call ptr @tl_tensor_reshape_dims(ptr %d1321, ptr %dims_ptr325, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res326)
  store ptr %reshape_dims_res326, ptr %inp1, align 8
  %model327 = load ptr, ptr %model, align 8
  %inp1328 = load ptr, ptr %inp1, align 8
  %call_method329 = call ptr @tl_GPT_forward(ptr %model327, ptr %inp1328)
  call void @tl_mem_register_tensor(ptr %call_method329)
  call void @tl_mem_unregister(ptr %call_method329)
  store ptr %call_method329, ptr %log1, align 8
  %log1330 = load ptr, ptr %log1, align 8
  %dim_ptr_0332 = getelementptr [2 x i64], ptr %dims_alloca331, i64 0, i64 0
  store i64 12, ptr %dim_ptr_0332, align 8
  %dim_ptr333 = getelementptr [2 x i64], ptr %dims_alloca331, i64 0, i64 1
  store i64 13, ptr %dim_ptr333, align 8
  %dims_ptr334 = getelementptr [2 x i64], ptr %dims_alloca331, i64 0, i64 0
  %reshape_dims_res335 = call ptr @tl_tensor_reshape_dims(ptr %log1330, ptr %dims_ptr334, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res335)
  store ptr %reshape_dims_res335, ptr %log1_flat, align 8
  %log1_flat336 = load ptr, ptr %log1_flat, align 8
  %slice_res337 = call ptr @tl_tensor_slice(ptr %log1_flat336, i64 5, i64 1)
  %call_tmp338 = call i64 @argmax(ptr %slice_res337)
  call void @tl_tensor_free(ptr %slice_res337)
  store i64 %call_tmp338, ptr %p1, align 8
  %p1339 = load i64, ptr %p1, align 8
  %cast_i64_f32341 = sitofp i64 %p1339 to float
  store float %cast_i64_f32341, ptr %scalar_data340, align 4
  %scalar_tensor343 = call ptr @tl_tensor_new(ptr %scalar_data340, i64 0, ptr %scalar_shape342)
  store float 1.000000e+00, ptr %scalar_data344, align 4
  %scalar_tensor346 = call ptr @tl_tensor_new(ptr %scalar_data344, i64 0, ptr %scalar_shape345)
  %pow_res347 = call ptr @tl_tensor_pow(ptr %scalar_tensor343, ptr %scalar_tensor346)
  call void @tl_tensor_free(ptr %scalar_tensor346)
  %get_res348 = call float @tl_tensor_get(ptr %pow_res347, i64 0)
  call void @tl_tensor_free(ptr %pow_res347)
  store float %get_res348, ptr %vp1, align 4
  %buf_void349 = call ptr @tl_alloc_tmp(i64 48)
  %y0350 = load float, ptr %y0, align 4
  %elem_ptr351 = getelementptr inbounds float, ptr %buf_void349, i64 0
  store float %y0350, ptr %elem_ptr351, align 4
  %y1352 = load float, ptr %y1, align 4
  %elem_ptr353 = getelementptr inbounds float, ptr %buf_void349, i64 1
  store float %y1352, ptr %elem_ptr353, align 4
  %y2354 = load float, ptr %y2, align 4
  %elem_ptr355 = getelementptr inbounds float, ptr %buf_void349, i64 2
  store float %y2354, ptr %elem_ptr355, align 4
  %y3356 = load float, ptr %y3, align 4
  %elem_ptr357 = getelementptr inbounds float, ptr %buf_void349, i64 3
  store float %y3356, ptr %elem_ptr357, align 4
  %y4358 = load float, ptr %y4, align 4
  %elem_ptr359 = getelementptr inbounds float, ptr %buf_void349, i64 4
  store float %y4358, ptr %elem_ptr359, align 4
  %y5360 = load float, ptr %y5, align 4
  %elem_ptr361 = getelementptr inbounds float, ptr %buf_void349, i64 5
  store float %y5360, ptr %elem_ptr361, align 4
  %vp1362 = load float, ptr %vp1, align 4
  %elem_ptr363 = getelementptr inbounds float, ptr %buf_void349, i64 6
  store float %vp1362, ptr %elem_ptr363, align 4
  %elem_ptr364 = getelementptr inbounds float, ptr %buf_void349, i64 7
  store float 1.200000e+01, ptr %elem_ptr364, align 4
  %elem_ptr365 = getelementptr inbounds float, ptr %buf_void349, i64 8
  store float 1.200000e+01, ptr %elem_ptr365, align 4
  %elem_ptr366 = getelementptr inbounds float, ptr %buf_void349, i64 9
  store float 1.200000e+01, ptr %elem_ptr366, align 4
  %elem_ptr367 = getelementptr inbounds float, ptr %buf_void349, i64 10
  store float 1.200000e+01, ptr %elem_ptr367, align 4
  %elem_ptr368 = getelementptr inbounds float, ptr %buf_void349, i64 11
  store float 1.200000e+01, ptr %elem_ptr368, align 4
  %shape_alloc369 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr370 = getelementptr inbounds i64, ptr %shape_alloc369, i64 0
  store i64 12, ptr %shape_ptr370, align 8
  %new_tensor371 = call ptr @tl_tensor_new(ptr %buf_void349, i64 1, ptr %shape_alloc369)
  call void @tl_free_tmp(ptr %buf_void349)
  call void @tl_free_tmp(ptr %shape_alloc369)
  call void @tl_mem_unregister(ptr %new_tensor371)
  store ptr %new_tensor371, ptr %d2, align 8
  %d2372 = load ptr, ptr %d2, align 8
  %dim_ptr_0374 = getelementptr [2 x i64], ptr %dims_alloca373, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0374, align 8
  %dim_ptr375 = getelementptr [2 x i64], ptr %dims_alloca373, i64 0, i64 1
  store i64 12, ptr %dim_ptr375, align 8
  %dims_ptr376 = getelementptr [2 x i64], ptr %dims_alloca373, i64 0, i64 0
  %reshape_dims_res377 = call ptr @tl_tensor_reshape_dims(ptr %d2372, ptr %dims_ptr376, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res377)
  store ptr %reshape_dims_res377, ptr %inp2, align 8
  %model378 = load ptr, ptr %model, align 8
  %inp2379 = load ptr, ptr %inp2, align 8
  %call_method380 = call ptr @tl_GPT_forward(ptr %model378, ptr %inp2379)
  call void @tl_mem_register_tensor(ptr %call_method380)
  call void @tl_mem_unregister(ptr %call_method380)
  store ptr %call_method380, ptr %log2, align 8
  %log2381 = load ptr, ptr %log2, align 8
  %dim_ptr_0383 = getelementptr [2 x i64], ptr %dims_alloca382, i64 0, i64 0
  store i64 12, ptr %dim_ptr_0383, align 8
  %dim_ptr384 = getelementptr [2 x i64], ptr %dims_alloca382, i64 0, i64 1
  store i64 13, ptr %dim_ptr384, align 8
  %dims_ptr385 = getelementptr [2 x i64], ptr %dims_alloca382, i64 0, i64 0
  %reshape_dims_res386 = call ptr @tl_tensor_reshape_dims(ptr %log2381, ptr %dims_ptr385, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res386)
  store ptr %reshape_dims_res386, ptr %log2_flat, align 8
  %log2_flat387 = load ptr, ptr %log2_flat, align 8
  %slice_res388 = call ptr @tl_tensor_slice(ptr %log2_flat387, i64 6, i64 1)
  %call_tmp389 = call i64 @argmax(ptr %slice_res388)
  call void @tl_tensor_free(ptr %slice_res388)
  store i64 %call_tmp389, ptr %p2, align 8
  %p2390 = load i64, ptr %p2, align 8
  %cast_i64_f32392 = sitofp i64 %p2390 to float
  store float %cast_i64_f32392, ptr %scalar_data391, align 4
  %scalar_tensor394 = call ptr @tl_tensor_new(ptr %scalar_data391, i64 0, ptr %scalar_shape393)
  store float 1.000000e+00, ptr %scalar_data395, align 4
  %scalar_tensor397 = call ptr @tl_tensor_new(ptr %scalar_data395, i64 0, ptr %scalar_shape396)
  %pow_res398 = call ptr @tl_tensor_pow(ptr %scalar_tensor394, ptr %scalar_tensor397)
  call void @tl_tensor_free(ptr %scalar_tensor397)
  %get_res399 = call float @tl_tensor_get(ptr %pow_res398, i64 0)
  call void @tl_tensor_free(ptr %pow_res398)
  store float %get_res399, ptr %vp2, align 4
  %buf_void400 = call ptr @tl_alloc_tmp(i64 48)
  %y0401 = load float, ptr %y0, align 4
  %elem_ptr402 = getelementptr inbounds float, ptr %buf_void400, i64 0
  store float %y0401, ptr %elem_ptr402, align 4
  %y1403 = load float, ptr %y1, align 4
  %elem_ptr404 = getelementptr inbounds float, ptr %buf_void400, i64 1
  store float %y1403, ptr %elem_ptr404, align 4
  %y2405 = load float, ptr %y2, align 4
  %elem_ptr406 = getelementptr inbounds float, ptr %buf_void400, i64 2
  store float %y2405, ptr %elem_ptr406, align 4
  %y3407 = load float, ptr %y3, align 4
  %elem_ptr408 = getelementptr inbounds float, ptr %buf_void400, i64 3
  store float %y3407, ptr %elem_ptr408, align 4
  %y4409 = load float, ptr %y4, align 4
  %elem_ptr410 = getelementptr inbounds float, ptr %buf_void400, i64 4
  store float %y4409, ptr %elem_ptr410, align 4
  %y5411 = load float, ptr %y5, align 4
  %elem_ptr412 = getelementptr inbounds float, ptr %buf_void400, i64 5
  store float %y5411, ptr %elem_ptr412, align 4
  %vp1413 = load float, ptr %vp1, align 4
  %elem_ptr414 = getelementptr inbounds float, ptr %buf_void400, i64 6
  store float %vp1413, ptr %elem_ptr414, align 4
  %vp2415 = load float, ptr %vp2, align 4
  %elem_ptr416 = getelementptr inbounds float, ptr %buf_void400, i64 7
  store float %vp2415, ptr %elem_ptr416, align 4
  %elem_ptr417 = getelementptr inbounds float, ptr %buf_void400, i64 8
  store float 1.200000e+01, ptr %elem_ptr417, align 4
  %elem_ptr418 = getelementptr inbounds float, ptr %buf_void400, i64 9
  store float 1.200000e+01, ptr %elem_ptr418, align 4
  %elem_ptr419 = getelementptr inbounds float, ptr %buf_void400, i64 10
  store float 1.200000e+01, ptr %elem_ptr419, align 4
  %elem_ptr420 = getelementptr inbounds float, ptr %buf_void400, i64 11
  store float 1.200000e+01, ptr %elem_ptr420, align 4
  %shape_alloc421 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr422 = getelementptr inbounds i64, ptr %shape_alloc421, i64 0
  store i64 12, ptr %shape_ptr422, align 8
  %new_tensor423 = call ptr @tl_tensor_new(ptr %buf_void400, i64 1, ptr %shape_alloc421)
  call void @tl_free_tmp(ptr %buf_void400)
  call void @tl_free_tmp(ptr %shape_alloc421)
  call void @tl_mem_unregister(ptr %new_tensor423)
  store ptr %new_tensor423, ptr %d3, align 8
  %d3424 = load ptr, ptr %d3, align 8
  %dim_ptr_0426 = getelementptr [2 x i64], ptr %dims_alloca425, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0426, align 8
  %dim_ptr427 = getelementptr [2 x i64], ptr %dims_alloca425, i64 0, i64 1
  store i64 12, ptr %dim_ptr427, align 8
  %dims_ptr428 = getelementptr [2 x i64], ptr %dims_alloca425, i64 0, i64 0
  %reshape_dims_res429 = call ptr @tl_tensor_reshape_dims(ptr %d3424, ptr %dims_ptr428, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res429)
  store ptr %reshape_dims_res429, ptr %inp3, align 8
  %model430 = load ptr, ptr %model, align 8
  %inp3431 = load ptr, ptr %inp3, align 8
  %call_method432 = call ptr @tl_GPT_forward(ptr %model430, ptr %inp3431)
  call void @tl_mem_register_tensor(ptr %call_method432)
  call void @tl_mem_unregister(ptr %call_method432)
  store ptr %call_method432, ptr %log3, align 8
  %log3433 = load ptr, ptr %log3, align 8
  %dim_ptr_0435 = getelementptr [2 x i64], ptr %dims_alloca434, i64 0, i64 0
  store i64 12, ptr %dim_ptr_0435, align 8
  %dim_ptr436 = getelementptr [2 x i64], ptr %dims_alloca434, i64 0, i64 1
  store i64 13, ptr %dim_ptr436, align 8
  %dims_ptr437 = getelementptr [2 x i64], ptr %dims_alloca434, i64 0, i64 0
  %reshape_dims_res438 = call ptr @tl_tensor_reshape_dims(ptr %log3433, ptr %dims_ptr437, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res438)
  store ptr %reshape_dims_res438, ptr %log3_flat, align 8
  %log3_flat439 = load ptr, ptr %log3_flat, align 8
  %slice_res440 = call ptr @tl_tensor_slice(ptr %log3_flat439, i64 7, i64 1)
  %call_tmp441 = call i64 @argmax(ptr %slice_res440)
  call void @tl_tensor_free(ptr %slice_res440)
  store i64 %call_tmp441, ptr %p3, align 8
  call void @tl_print_string(ptr @str_literal.166)
  %p1442 = load i64, ptr %p1, align 8
  call void @tl_print_i64(i64 %p1442)
  %p2443 = load i64, ptr %p2, align 8
  call void @tl_print_i64(i64 %p2443)
  %p3444 = load i64, ptr %p3, align 8
  call void @tl_print_i64(i64 %p3444)
  call void @tl_print_string(ptr @str_literal.167)
  %struct_to_free = load ptr, ptr %model, align 8
  %is_not_null = icmp ne ptr %struct_to_free, null
  br i1 %is_not_null, label %recursive_free_struct, label %continue_after_recursive_free

recursive_free_struct:                            ; preds = %entry
  %free_field_0 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 0
  %field_val_to_free = load ptr, ptr %free_field_0, align 8
  %is_not_null445 = icmp ne ptr %field_val_to_free, null
  br i1 %is_not_null445, label %recursive_free_struct446, label %continue_after_recursive_free447

continue_after_recursive_free:                    ; preds = %continue_after_recursive_free725, %entry
  call void @tl_mem_unregister(ptr %struct_to_free)
  call void @tl_mem_exit_scope()
  ret void

recursive_free_struct446:                         ; preds = %recursive_free_struct
  %free_field_0448 = getelementptr inbounds %Embedding, ptr %field_val_to_free, i32 0, i32 0
  %field_val_to_free449 = load ptr, ptr %free_field_0448, align 8
  call void @tl_tensor_free(ptr %field_val_to_free449)
  call void @free(ptr %field_val_to_free)
  br label %continue_after_recursive_free447

continue_after_recursive_free447:                 ; preds = %recursive_free_struct446, %recursive_free_struct
  %free_field_1 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 1
  %field_val_to_free450 = load ptr, ptr %free_field_1, align 8
  %is_not_null451 = icmp ne ptr %field_val_to_free450, null
  br i1 %is_not_null451, label %recursive_free_struct452, label %continue_after_recursive_free453

recursive_free_struct452:                         ; preds = %continue_after_recursive_free447
  %free_field_0454 = getelementptr inbounds %Embedding, ptr %field_val_to_free450, i32 0, i32 0
  %field_val_to_free455 = load ptr, ptr %free_field_0454, align 8
  call void @tl_tensor_free(ptr %field_val_to_free455)
  call void @free(ptr %field_val_to_free450)
  br label %continue_after_recursive_free453

continue_after_recursive_free453:                 ; preds = %recursive_free_struct452, %continue_after_recursive_free447
  %free_field_2 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 2
  %field_val_to_free456 = load ptr, ptr %free_field_2, align 8
  %is_not_null457 = icmp ne ptr %field_val_to_free456, null
  br i1 %is_not_null457, label %recursive_free_struct458, label %continue_after_recursive_free459

recursive_free_struct458:                         ; preds = %continue_after_recursive_free453
  %free_field_0460 = getelementptr inbounds %Block, ptr %field_val_to_free456, i32 0, i32 0
  %field_val_to_free461 = load ptr, ptr %free_field_0460, align 8
  %is_not_null462 = icmp ne ptr %field_val_to_free461, null
  br i1 %is_not_null462, label %recursive_free_struct463, label %continue_after_recursive_free464

continue_after_recursive_free459:                 ; preds = %continue_after_recursive_free522, %continue_after_recursive_free453
  %free_field_3541 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 3
  %field_val_to_free542 = load ptr, ptr %free_field_3541, align 8
  %is_not_null543 = icmp ne ptr %field_val_to_free542, null
  br i1 %is_not_null543, label %recursive_free_struct544, label %continue_after_recursive_free545

recursive_free_struct463:                         ; preds = %recursive_free_struct458
  %free_field_0465 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free461, i32 0, i32 0
  %field_val_to_free466 = load ptr, ptr %free_field_0465, align 8
  call void @tl_tensor_free(ptr %field_val_to_free466)
  %free_field_1467 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free461, i32 0, i32 1
  %field_val_to_free468 = load ptr, ptr %free_field_1467, align 8
  call void @tl_tensor_free(ptr %field_val_to_free468)
  call void @free(ptr %field_val_to_free461)
  br label %continue_after_recursive_free464

continue_after_recursive_free464:                 ; preds = %recursive_free_struct463, %recursive_free_struct458
  %free_field_1469 = getelementptr inbounds %Block, ptr %field_val_to_free456, i32 0, i32 1
  %field_val_to_free470 = load ptr, ptr %free_field_1469, align 8
  %is_not_null471 = icmp ne ptr %field_val_to_free470, null
  br i1 %is_not_null471, label %recursive_free_struct472, label %continue_after_recursive_free473

recursive_free_struct472:                         ; preds = %continue_after_recursive_free464
  %free_field_0474 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free470, i32 0, i32 0
  %field_val_to_free475 = load ptr, ptr %free_field_0474, align 8
  %is_not_null476 = icmp ne ptr %field_val_to_free475, null
  br i1 %is_not_null476, label %recursive_free_struct477, label %continue_after_recursive_free478

continue_after_recursive_free473:                 ; preds = %continue_after_recursive_free504, %continue_after_recursive_free464
  %free_field_2509 = getelementptr inbounds %Block, ptr %field_val_to_free456, i32 0, i32 2
  %field_val_to_free510 = load ptr, ptr %free_field_2509, align 8
  %is_not_null511 = icmp ne ptr %field_val_to_free510, null
  br i1 %is_not_null511, label %recursive_free_struct512, label %continue_after_recursive_free513

recursive_free_struct477:                         ; preds = %recursive_free_struct472
  %free_field_0479 = getelementptr inbounds %Linear, ptr %field_val_to_free475, i32 0, i32 0
  %field_val_to_free480 = load ptr, ptr %free_field_0479, align 8
  call void @tl_tensor_free(ptr %field_val_to_free480)
  %free_field_1481 = getelementptr inbounds %Linear, ptr %field_val_to_free475, i32 0, i32 1
  %field_val_to_free482 = load ptr, ptr %free_field_1481, align 8
  call void @tl_tensor_free(ptr %field_val_to_free482)
  call void @free(ptr %field_val_to_free475)
  br label %continue_after_recursive_free478

continue_after_recursive_free478:                 ; preds = %recursive_free_struct477, %recursive_free_struct472
  %free_field_1483 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free470, i32 0, i32 1
  %field_val_to_free484 = load ptr, ptr %free_field_1483, align 8
  %is_not_null485 = icmp ne ptr %field_val_to_free484, null
  br i1 %is_not_null485, label %recursive_free_struct486, label %continue_after_recursive_free487

recursive_free_struct486:                         ; preds = %continue_after_recursive_free478
  %free_field_0488 = getelementptr inbounds %Linear, ptr %field_val_to_free484, i32 0, i32 0
  %field_val_to_free489 = load ptr, ptr %free_field_0488, align 8
  call void @tl_tensor_free(ptr %field_val_to_free489)
  %free_field_1490 = getelementptr inbounds %Linear, ptr %field_val_to_free484, i32 0, i32 1
  %field_val_to_free491 = load ptr, ptr %free_field_1490, align 8
  call void @tl_tensor_free(ptr %field_val_to_free491)
  call void @free(ptr %field_val_to_free484)
  br label %continue_after_recursive_free487

continue_after_recursive_free487:                 ; preds = %recursive_free_struct486, %continue_after_recursive_free478
  %free_field_2492 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free470, i32 0, i32 2
  %field_val_to_free493 = load ptr, ptr %free_field_2492, align 8
  %is_not_null494 = icmp ne ptr %field_val_to_free493, null
  br i1 %is_not_null494, label %recursive_free_struct495, label %continue_after_recursive_free496

recursive_free_struct495:                         ; preds = %continue_after_recursive_free487
  %free_field_0497 = getelementptr inbounds %Linear, ptr %field_val_to_free493, i32 0, i32 0
  %field_val_to_free498 = load ptr, ptr %free_field_0497, align 8
  call void @tl_tensor_free(ptr %field_val_to_free498)
  %free_field_1499 = getelementptr inbounds %Linear, ptr %field_val_to_free493, i32 0, i32 1
  %field_val_to_free500 = load ptr, ptr %free_field_1499, align 8
  call void @tl_tensor_free(ptr %field_val_to_free500)
  call void @free(ptr %field_val_to_free493)
  br label %continue_after_recursive_free496

continue_after_recursive_free496:                 ; preds = %recursive_free_struct495, %continue_after_recursive_free487
  %free_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free470, i32 0, i32 3
  %field_val_to_free501 = load ptr, ptr %free_field_3, align 8
  %is_not_null502 = icmp ne ptr %field_val_to_free501, null
  br i1 %is_not_null502, label %recursive_free_struct503, label %continue_after_recursive_free504

recursive_free_struct503:                         ; preds = %continue_after_recursive_free496
  %free_field_0505 = getelementptr inbounds %Linear, ptr %field_val_to_free501, i32 0, i32 0
  %field_val_to_free506 = load ptr, ptr %free_field_0505, align 8
  call void @tl_tensor_free(ptr %field_val_to_free506)
  %free_field_1507 = getelementptr inbounds %Linear, ptr %field_val_to_free501, i32 0, i32 1
  %field_val_to_free508 = load ptr, ptr %free_field_1507, align 8
  call void @tl_tensor_free(ptr %field_val_to_free508)
  call void @free(ptr %field_val_to_free501)
  br label %continue_after_recursive_free504

continue_after_recursive_free504:                 ; preds = %recursive_free_struct503, %continue_after_recursive_free496
  call void @free(ptr %field_val_to_free470)
  br label %continue_after_recursive_free473

recursive_free_struct512:                         ; preds = %continue_after_recursive_free473
  %free_field_0514 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free510, i32 0, i32 0
  %field_val_to_free515 = load ptr, ptr %free_field_0514, align 8
  call void @tl_tensor_free(ptr %field_val_to_free515)
  %free_field_1516 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free510, i32 0, i32 1
  %field_val_to_free517 = load ptr, ptr %free_field_1516, align 8
  call void @tl_tensor_free(ptr %field_val_to_free517)
  call void @free(ptr %field_val_to_free510)
  br label %continue_after_recursive_free513

continue_after_recursive_free513:                 ; preds = %recursive_free_struct512, %continue_after_recursive_free473
  %free_field_3518 = getelementptr inbounds %Block, ptr %field_val_to_free456, i32 0, i32 3
  %field_val_to_free519 = load ptr, ptr %free_field_3518, align 8
  %is_not_null520 = icmp ne ptr %field_val_to_free519, null
  br i1 %is_not_null520, label %recursive_free_struct521, label %continue_after_recursive_free522

recursive_free_struct521:                         ; preds = %continue_after_recursive_free513
  %free_field_0523 = getelementptr inbounds %MLP, ptr %field_val_to_free519, i32 0, i32 0
  %field_val_to_free524 = load ptr, ptr %free_field_0523, align 8
  %is_not_null525 = icmp ne ptr %field_val_to_free524, null
  br i1 %is_not_null525, label %recursive_free_struct526, label %continue_after_recursive_free527

continue_after_recursive_free522:                 ; preds = %continue_after_recursive_free536, %continue_after_recursive_free513
  call void @free(ptr %field_val_to_free456)
  br label %continue_after_recursive_free459

recursive_free_struct526:                         ; preds = %recursive_free_struct521
  %free_field_0528 = getelementptr inbounds %Linear, ptr %field_val_to_free524, i32 0, i32 0
  %field_val_to_free529 = load ptr, ptr %free_field_0528, align 8
  call void @tl_tensor_free(ptr %field_val_to_free529)
  %free_field_1530 = getelementptr inbounds %Linear, ptr %field_val_to_free524, i32 0, i32 1
  %field_val_to_free531 = load ptr, ptr %free_field_1530, align 8
  call void @tl_tensor_free(ptr %field_val_to_free531)
  call void @free(ptr %field_val_to_free524)
  br label %continue_after_recursive_free527

continue_after_recursive_free527:                 ; preds = %recursive_free_struct526, %recursive_free_struct521
  %free_field_1532 = getelementptr inbounds %MLP, ptr %field_val_to_free519, i32 0, i32 1
  %field_val_to_free533 = load ptr, ptr %free_field_1532, align 8
  %is_not_null534 = icmp ne ptr %field_val_to_free533, null
  br i1 %is_not_null534, label %recursive_free_struct535, label %continue_after_recursive_free536

recursive_free_struct535:                         ; preds = %continue_after_recursive_free527
  %free_field_0537 = getelementptr inbounds %Linear, ptr %field_val_to_free533, i32 0, i32 0
  %field_val_to_free538 = load ptr, ptr %free_field_0537, align 8
  call void @tl_tensor_free(ptr %field_val_to_free538)
  %free_field_1539 = getelementptr inbounds %Linear, ptr %field_val_to_free533, i32 0, i32 1
  %field_val_to_free540 = load ptr, ptr %free_field_1539, align 8
  call void @tl_tensor_free(ptr %field_val_to_free540)
  call void @free(ptr %field_val_to_free533)
  br label %continue_after_recursive_free536

continue_after_recursive_free536:                 ; preds = %recursive_free_struct535, %continue_after_recursive_free527
  call void @free(ptr %field_val_to_free519)
  br label %continue_after_recursive_free522

recursive_free_struct544:                         ; preds = %continue_after_recursive_free459
  %free_field_0546 = getelementptr inbounds %Block, ptr %field_val_to_free542, i32 0, i32 0
  %field_val_to_free547 = load ptr, ptr %free_field_0546, align 8
  %is_not_null548 = icmp ne ptr %field_val_to_free547, null
  br i1 %is_not_null548, label %recursive_free_struct549, label %continue_after_recursive_free550

continue_after_recursive_free545:                 ; preds = %continue_after_recursive_free609, %continue_after_recursive_free459
  %free_field_4 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 4
  %field_val_to_free628 = load ptr, ptr %free_field_4, align 8
  %is_not_null629 = icmp ne ptr %field_val_to_free628, null
  br i1 %is_not_null629, label %recursive_free_struct630, label %continue_after_recursive_free631

recursive_free_struct549:                         ; preds = %recursive_free_struct544
  %free_field_0551 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free547, i32 0, i32 0
  %field_val_to_free552 = load ptr, ptr %free_field_0551, align 8
  call void @tl_tensor_free(ptr %field_val_to_free552)
  %free_field_1553 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free547, i32 0, i32 1
  %field_val_to_free554 = load ptr, ptr %free_field_1553, align 8
  call void @tl_tensor_free(ptr %field_val_to_free554)
  call void @free(ptr %field_val_to_free547)
  br label %continue_after_recursive_free550

continue_after_recursive_free550:                 ; preds = %recursive_free_struct549, %recursive_free_struct544
  %free_field_1555 = getelementptr inbounds %Block, ptr %field_val_to_free542, i32 0, i32 1
  %field_val_to_free556 = load ptr, ptr %free_field_1555, align 8
  %is_not_null557 = icmp ne ptr %field_val_to_free556, null
  br i1 %is_not_null557, label %recursive_free_struct558, label %continue_after_recursive_free559

recursive_free_struct558:                         ; preds = %continue_after_recursive_free550
  %free_field_0560 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free556, i32 0, i32 0
  %field_val_to_free561 = load ptr, ptr %free_field_0560, align 8
  %is_not_null562 = icmp ne ptr %field_val_to_free561, null
  br i1 %is_not_null562, label %recursive_free_struct563, label %continue_after_recursive_free564

continue_after_recursive_free559:                 ; preds = %continue_after_recursive_free591, %continue_after_recursive_free550
  %free_field_2596 = getelementptr inbounds %Block, ptr %field_val_to_free542, i32 0, i32 2
  %field_val_to_free597 = load ptr, ptr %free_field_2596, align 8
  %is_not_null598 = icmp ne ptr %field_val_to_free597, null
  br i1 %is_not_null598, label %recursive_free_struct599, label %continue_after_recursive_free600

recursive_free_struct563:                         ; preds = %recursive_free_struct558
  %free_field_0565 = getelementptr inbounds %Linear, ptr %field_val_to_free561, i32 0, i32 0
  %field_val_to_free566 = load ptr, ptr %free_field_0565, align 8
  call void @tl_tensor_free(ptr %field_val_to_free566)
  %free_field_1567 = getelementptr inbounds %Linear, ptr %field_val_to_free561, i32 0, i32 1
  %field_val_to_free568 = load ptr, ptr %free_field_1567, align 8
  call void @tl_tensor_free(ptr %field_val_to_free568)
  call void @free(ptr %field_val_to_free561)
  br label %continue_after_recursive_free564

continue_after_recursive_free564:                 ; preds = %recursive_free_struct563, %recursive_free_struct558
  %free_field_1569 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free556, i32 0, i32 1
  %field_val_to_free570 = load ptr, ptr %free_field_1569, align 8
  %is_not_null571 = icmp ne ptr %field_val_to_free570, null
  br i1 %is_not_null571, label %recursive_free_struct572, label %continue_after_recursive_free573

recursive_free_struct572:                         ; preds = %continue_after_recursive_free564
  %free_field_0574 = getelementptr inbounds %Linear, ptr %field_val_to_free570, i32 0, i32 0
  %field_val_to_free575 = load ptr, ptr %free_field_0574, align 8
  call void @tl_tensor_free(ptr %field_val_to_free575)
  %free_field_1576 = getelementptr inbounds %Linear, ptr %field_val_to_free570, i32 0, i32 1
  %field_val_to_free577 = load ptr, ptr %free_field_1576, align 8
  call void @tl_tensor_free(ptr %field_val_to_free577)
  call void @free(ptr %field_val_to_free570)
  br label %continue_after_recursive_free573

continue_after_recursive_free573:                 ; preds = %recursive_free_struct572, %continue_after_recursive_free564
  %free_field_2578 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free556, i32 0, i32 2
  %field_val_to_free579 = load ptr, ptr %free_field_2578, align 8
  %is_not_null580 = icmp ne ptr %field_val_to_free579, null
  br i1 %is_not_null580, label %recursive_free_struct581, label %continue_after_recursive_free582

recursive_free_struct581:                         ; preds = %continue_after_recursive_free573
  %free_field_0583 = getelementptr inbounds %Linear, ptr %field_val_to_free579, i32 0, i32 0
  %field_val_to_free584 = load ptr, ptr %free_field_0583, align 8
  call void @tl_tensor_free(ptr %field_val_to_free584)
  %free_field_1585 = getelementptr inbounds %Linear, ptr %field_val_to_free579, i32 0, i32 1
  %field_val_to_free586 = load ptr, ptr %free_field_1585, align 8
  call void @tl_tensor_free(ptr %field_val_to_free586)
  call void @free(ptr %field_val_to_free579)
  br label %continue_after_recursive_free582

continue_after_recursive_free582:                 ; preds = %recursive_free_struct581, %continue_after_recursive_free573
  %free_field_3587 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free556, i32 0, i32 3
  %field_val_to_free588 = load ptr, ptr %free_field_3587, align 8
  %is_not_null589 = icmp ne ptr %field_val_to_free588, null
  br i1 %is_not_null589, label %recursive_free_struct590, label %continue_after_recursive_free591

recursive_free_struct590:                         ; preds = %continue_after_recursive_free582
  %free_field_0592 = getelementptr inbounds %Linear, ptr %field_val_to_free588, i32 0, i32 0
  %field_val_to_free593 = load ptr, ptr %free_field_0592, align 8
  call void @tl_tensor_free(ptr %field_val_to_free593)
  %free_field_1594 = getelementptr inbounds %Linear, ptr %field_val_to_free588, i32 0, i32 1
  %field_val_to_free595 = load ptr, ptr %free_field_1594, align 8
  call void @tl_tensor_free(ptr %field_val_to_free595)
  call void @free(ptr %field_val_to_free588)
  br label %continue_after_recursive_free591

continue_after_recursive_free591:                 ; preds = %recursive_free_struct590, %continue_after_recursive_free582
  call void @free(ptr %field_val_to_free556)
  br label %continue_after_recursive_free559

recursive_free_struct599:                         ; preds = %continue_after_recursive_free559
  %free_field_0601 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free597, i32 0, i32 0
  %field_val_to_free602 = load ptr, ptr %free_field_0601, align 8
  call void @tl_tensor_free(ptr %field_val_to_free602)
  %free_field_1603 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free597, i32 0, i32 1
  %field_val_to_free604 = load ptr, ptr %free_field_1603, align 8
  call void @tl_tensor_free(ptr %field_val_to_free604)
  call void @free(ptr %field_val_to_free597)
  br label %continue_after_recursive_free600

continue_after_recursive_free600:                 ; preds = %recursive_free_struct599, %continue_after_recursive_free559
  %free_field_3605 = getelementptr inbounds %Block, ptr %field_val_to_free542, i32 0, i32 3
  %field_val_to_free606 = load ptr, ptr %free_field_3605, align 8
  %is_not_null607 = icmp ne ptr %field_val_to_free606, null
  br i1 %is_not_null607, label %recursive_free_struct608, label %continue_after_recursive_free609

recursive_free_struct608:                         ; preds = %continue_after_recursive_free600
  %free_field_0610 = getelementptr inbounds %MLP, ptr %field_val_to_free606, i32 0, i32 0
  %field_val_to_free611 = load ptr, ptr %free_field_0610, align 8
  %is_not_null612 = icmp ne ptr %field_val_to_free611, null
  br i1 %is_not_null612, label %recursive_free_struct613, label %continue_after_recursive_free614

continue_after_recursive_free609:                 ; preds = %continue_after_recursive_free623, %continue_after_recursive_free600
  call void @free(ptr %field_val_to_free542)
  br label %continue_after_recursive_free545

recursive_free_struct613:                         ; preds = %recursive_free_struct608
  %free_field_0615 = getelementptr inbounds %Linear, ptr %field_val_to_free611, i32 0, i32 0
  %field_val_to_free616 = load ptr, ptr %free_field_0615, align 8
  call void @tl_tensor_free(ptr %field_val_to_free616)
  %free_field_1617 = getelementptr inbounds %Linear, ptr %field_val_to_free611, i32 0, i32 1
  %field_val_to_free618 = load ptr, ptr %free_field_1617, align 8
  call void @tl_tensor_free(ptr %field_val_to_free618)
  call void @free(ptr %field_val_to_free611)
  br label %continue_after_recursive_free614

continue_after_recursive_free614:                 ; preds = %recursive_free_struct613, %recursive_free_struct608
  %free_field_1619 = getelementptr inbounds %MLP, ptr %field_val_to_free606, i32 0, i32 1
  %field_val_to_free620 = load ptr, ptr %free_field_1619, align 8
  %is_not_null621 = icmp ne ptr %field_val_to_free620, null
  br i1 %is_not_null621, label %recursive_free_struct622, label %continue_after_recursive_free623

recursive_free_struct622:                         ; preds = %continue_after_recursive_free614
  %free_field_0624 = getelementptr inbounds %Linear, ptr %field_val_to_free620, i32 0, i32 0
  %field_val_to_free625 = load ptr, ptr %free_field_0624, align 8
  call void @tl_tensor_free(ptr %field_val_to_free625)
  %free_field_1626 = getelementptr inbounds %Linear, ptr %field_val_to_free620, i32 0, i32 1
  %field_val_to_free627 = load ptr, ptr %free_field_1626, align 8
  call void @tl_tensor_free(ptr %field_val_to_free627)
  call void @free(ptr %field_val_to_free620)
  br label %continue_after_recursive_free623

continue_after_recursive_free623:                 ; preds = %recursive_free_struct622, %continue_after_recursive_free614
  call void @free(ptr %field_val_to_free606)
  br label %continue_after_recursive_free609

recursive_free_struct630:                         ; preds = %continue_after_recursive_free545
  %free_field_0632 = getelementptr inbounds %Block, ptr %field_val_to_free628, i32 0, i32 0
  %field_val_to_free633 = load ptr, ptr %free_field_0632, align 8
  %is_not_null634 = icmp ne ptr %field_val_to_free633, null
  br i1 %is_not_null634, label %recursive_free_struct635, label %continue_after_recursive_free636

continue_after_recursive_free631:                 ; preds = %continue_after_recursive_free695, %continue_after_recursive_free545
  %free_field_5 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 5
  %field_val_to_free714 = load ptr, ptr %free_field_5, align 8
  %is_not_null715 = icmp ne ptr %field_val_to_free714, null
  br i1 %is_not_null715, label %recursive_free_struct716, label %continue_after_recursive_free717

recursive_free_struct635:                         ; preds = %recursive_free_struct630
  %free_field_0637 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free633, i32 0, i32 0
  %field_val_to_free638 = load ptr, ptr %free_field_0637, align 8
  call void @tl_tensor_free(ptr %field_val_to_free638)
  %free_field_1639 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free633, i32 0, i32 1
  %field_val_to_free640 = load ptr, ptr %free_field_1639, align 8
  call void @tl_tensor_free(ptr %field_val_to_free640)
  call void @free(ptr %field_val_to_free633)
  br label %continue_after_recursive_free636

continue_after_recursive_free636:                 ; preds = %recursive_free_struct635, %recursive_free_struct630
  %free_field_1641 = getelementptr inbounds %Block, ptr %field_val_to_free628, i32 0, i32 1
  %field_val_to_free642 = load ptr, ptr %free_field_1641, align 8
  %is_not_null643 = icmp ne ptr %field_val_to_free642, null
  br i1 %is_not_null643, label %recursive_free_struct644, label %continue_after_recursive_free645

recursive_free_struct644:                         ; preds = %continue_after_recursive_free636
  %free_field_0646 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free642, i32 0, i32 0
  %field_val_to_free647 = load ptr, ptr %free_field_0646, align 8
  %is_not_null648 = icmp ne ptr %field_val_to_free647, null
  br i1 %is_not_null648, label %recursive_free_struct649, label %continue_after_recursive_free650

continue_after_recursive_free645:                 ; preds = %continue_after_recursive_free677, %continue_after_recursive_free636
  %free_field_2682 = getelementptr inbounds %Block, ptr %field_val_to_free628, i32 0, i32 2
  %field_val_to_free683 = load ptr, ptr %free_field_2682, align 8
  %is_not_null684 = icmp ne ptr %field_val_to_free683, null
  br i1 %is_not_null684, label %recursive_free_struct685, label %continue_after_recursive_free686

recursive_free_struct649:                         ; preds = %recursive_free_struct644
  %free_field_0651 = getelementptr inbounds %Linear, ptr %field_val_to_free647, i32 0, i32 0
  %field_val_to_free652 = load ptr, ptr %free_field_0651, align 8
  call void @tl_tensor_free(ptr %field_val_to_free652)
  %free_field_1653 = getelementptr inbounds %Linear, ptr %field_val_to_free647, i32 0, i32 1
  %field_val_to_free654 = load ptr, ptr %free_field_1653, align 8
  call void @tl_tensor_free(ptr %field_val_to_free654)
  call void @free(ptr %field_val_to_free647)
  br label %continue_after_recursive_free650

continue_after_recursive_free650:                 ; preds = %recursive_free_struct649, %recursive_free_struct644
  %free_field_1655 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free642, i32 0, i32 1
  %field_val_to_free656 = load ptr, ptr %free_field_1655, align 8
  %is_not_null657 = icmp ne ptr %field_val_to_free656, null
  br i1 %is_not_null657, label %recursive_free_struct658, label %continue_after_recursive_free659

recursive_free_struct658:                         ; preds = %continue_after_recursive_free650
  %free_field_0660 = getelementptr inbounds %Linear, ptr %field_val_to_free656, i32 0, i32 0
  %field_val_to_free661 = load ptr, ptr %free_field_0660, align 8
  call void @tl_tensor_free(ptr %field_val_to_free661)
  %free_field_1662 = getelementptr inbounds %Linear, ptr %field_val_to_free656, i32 0, i32 1
  %field_val_to_free663 = load ptr, ptr %free_field_1662, align 8
  call void @tl_tensor_free(ptr %field_val_to_free663)
  call void @free(ptr %field_val_to_free656)
  br label %continue_after_recursive_free659

continue_after_recursive_free659:                 ; preds = %recursive_free_struct658, %continue_after_recursive_free650
  %free_field_2664 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free642, i32 0, i32 2
  %field_val_to_free665 = load ptr, ptr %free_field_2664, align 8
  %is_not_null666 = icmp ne ptr %field_val_to_free665, null
  br i1 %is_not_null666, label %recursive_free_struct667, label %continue_after_recursive_free668

recursive_free_struct667:                         ; preds = %continue_after_recursive_free659
  %free_field_0669 = getelementptr inbounds %Linear, ptr %field_val_to_free665, i32 0, i32 0
  %field_val_to_free670 = load ptr, ptr %free_field_0669, align 8
  call void @tl_tensor_free(ptr %field_val_to_free670)
  %free_field_1671 = getelementptr inbounds %Linear, ptr %field_val_to_free665, i32 0, i32 1
  %field_val_to_free672 = load ptr, ptr %free_field_1671, align 8
  call void @tl_tensor_free(ptr %field_val_to_free672)
  call void @free(ptr %field_val_to_free665)
  br label %continue_after_recursive_free668

continue_after_recursive_free668:                 ; preds = %recursive_free_struct667, %continue_after_recursive_free659
  %free_field_3673 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free642, i32 0, i32 3
  %field_val_to_free674 = load ptr, ptr %free_field_3673, align 8
  %is_not_null675 = icmp ne ptr %field_val_to_free674, null
  br i1 %is_not_null675, label %recursive_free_struct676, label %continue_after_recursive_free677

recursive_free_struct676:                         ; preds = %continue_after_recursive_free668
  %free_field_0678 = getelementptr inbounds %Linear, ptr %field_val_to_free674, i32 0, i32 0
  %field_val_to_free679 = load ptr, ptr %free_field_0678, align 8
  call void @tl_tensor_free(ptr %field_val_to_free679)
  %free_field_1680 = getelementptr inbounds %Linear, ptr %field_val_to_free674, i32 0, i32 1
  %field_val_to_free681 = load ptr, ptr %free_field_1680, align 8
  call void @tl_tensor_free(ptr %field_val_to_free681)
  call void @free(ptr %field_val_to_free674)
  br label %continue_after_recursive_free677

continue_after_recursive_free677:                 ; preds = %recursive_free_struct676, %continue_after_recursive_free668
  call void @free(ptr %field_val_to_free642)
  br label %continue_after_recursive_free645

recursive_free_struct685:                         ; preds = %continue_after_recursive_free645
  %free_field_0687 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free683, i32 0, i32 0
  %field_val_to_free688 = load ptr, ptr %free_field_0687, align 8
  call void @tl_tensor_free(ptr %field_val_to_free688)
  %free_field_1689 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free683, i32 0, i32 1
  %field_val_to_free690 = load ptr, ptr %free_field_1689, align 8
  call void @tl_tensor_free(ptr %field_val_to_free690)
  call void @free(ptr %field_val_to_free683)
  br label %continue_after_recursive_free686

continue_after_recursive_free686:                 ; preds = %recursive_free_struct685, %continue_after_recursive_free645
  %free_field_3691 = getelementptr inbounds %Block, ptr %field_val_to_free628, i32 0, i32 3
  %field_val_to_free692 = load ptr, ptr %free_field_3691, align 8
  %is_not_null693 = icmp ne ptr %field_val_to_free692, null
  br i1 %is_not_null693, label %recursive_free_struct694, label %continue_after_recursive_free695

recursive_free_struct694:                         ; preds = %continue_after_recursive_free686
  %free_field_0696 = getelementptr inbounds %MLP, ptr %field_val_to_free692, i32 0, i32 0
  %field_val_to_free697 = load ptr, ptr %free_field_0696, align 8
  %is_not_null698 = icmp ne ptr %field_val_to_free697, null
  br i1 %is_not_null698, label %recursive_free_struct699, label %continue_after_recursive_free700

continue_after_recursive_free695:                 ; preds = %continue_after_recursive_free709, %continue_after_recursive_free686
  call void @free(ptr %field_val_to_free628)
  br label %continue_after_recursive_free631

recursive_free_struct699:                         ; preds = %recursive_free_struct694
  %free_field_0701 = getelementptr inbounds %Linear, ptr %field_val_to_free697, i32 0, i32 0
  %field_val_to_free702 = load ptr, ptr %free_field_0701, align 8
  call void @tl_tensor_free(ptr %field_val_to_free702)
  %free_field_1703 = getelementptr inbounds %Linear, ptr %field_val_to_free697, i32 0, i32 1
  %field_val_to_free704 = load ptr, ptr %free_field_1703, align 8
  call void @tl_tensor_free(ptr %field_val_to_free704)
  call void @free(ptr %field_val_to_free697)
  br label %continue_after_recursive_free700

continue_after_recursive_free700:                 ; preds = %recursive_free_struct699, %recursive_free_struct694
  %free_field_1705 = getelementptr inbounds %MLP, ptr %field_val_to_free692, i32 0, i32 1
  %field_val_to_free706 = load ptr, ptr %free_field_1705, align 8
  %is_not_null707 = icmp ne ptr %field_val_to_free706, null
  br i1 %is_not_null707, label %recursive_free_struct708, label %continue_after_recursive_free709

recursive_free_struct708:                         ; preds = %continue_after_recursive_free700
  %free_field_0710 = getelementptr inbounds %Linear, ptr %field_val_to_free706, i32 0, i32 0
  %field_val_to_free711 = load ptr, ptr %free_field_0710, align 8
  call void @tl_tensor_free(ptr %field_val_to_free711)
  %free_field_1712 = getelementptr inbounds %Linear, ptr %field_val_to_free706, i32 0, i32 1
  %field_val_to_free713 = load ptr, ptr %free_field_1712, align 8
  call void @tl_tensor_free(ptr %field_val_to_free713)
  call void @free(ptr %field_val_to_free706)
  br label %continue_after_recursive_free709

continue_after_recursive_free709:                 ; preds = %recursive_free_struct708, %continue_after_recursive_free700
  call void @free(ptr %field_val_to_free692)
  br label %continue_after_recursive_free695

recursive_free_struct716:                         ; preds = %continue_after_recursive_free631
  %free_field_0718 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free714, i32 0, i32 0
  %field_val_to_free719 = load ptr, ptr %free_field_0718, align 8
  call void @tl_tensor_free(ptr %field_val_to_free719)
  %free_field_1720 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free714, i32 0, i32 1
  %field_val_to_free721 = load ptr, ptr %free_field_1720, align 8
  call void @tl_tensor_free(ptr %field_val_to_free721)
  call void @free(ptr %field_val_to_free714)
  br label %continue_after_recursive_free717

continue_after_recursive_free717:                 ; preds = %recursive_free_struct716, %continue_after_recursive_free631
  %free_field_6 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 6
  %field_val_to_free722 = load ptr, ptr %free_field_6, align 8
  %is_not_null723 = icmp ne ptr %field_val_to_free722, null
  br i1 %is_not_null723, label %recursive_free_struct724, label %continue_after_recursive_free725

recursive_free_struct724:                         ; preds = %continue_after_recursive_free717
  %free_field_0726 = getelementptr inbounds %Linear, ptr %field_val_to_free722, i32 0, i32 0
  %field_val_to_free727 = load ptr, ptr %free_field_0726, align 8
  call void @tl_tensor_free(ptr %field_val_to_free727)
  %free_field_1728 = getelementptr inbounds %Linear, ptr %field_val_to_free722, i32 0, i32 1
  %field_val_to_free729 = load ptr, ptr %free_field_1728, align 8
  call void @tl_tensor_free(ptr %field_val_to_free729)
  call void @free(ptr %field_val_to_free722)
  br label %continue_after_recursive_free725

continue_after_recursive_free725:                 ; preds = %recursive_free_struct724, %continue_after_recursive_free717
  call void @free(ptr %struct_to_free)
  br label %continue_after_recursive_free
}
