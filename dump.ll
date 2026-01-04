; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

%Linear = type { ptr, ptr }
%Embedding = type { ptr }
%LayerNorm = type { ptr, ptr }
%CausalSelfAttention = type { ptr, ptr, ptr, ptr }
%MLP = type { ptr, ptr }
%Block = type { ptr, ptr, ptr, ptr }
%PositionalEmbedding = type { ptr }
%SeqModel = type { ptr, ptr, ptr, ptr, ptr, ptr }

@str_literal = private unnamed_addr constant [20 x i8] c"Full Context Input:\00", align 1
@str_literal.104 = private unnamed_addr constant [24 x i8] c"Checking predictions...\00", align 1
@str_literal.105 = private unnamed_addr constant [30 x i8] c"Step 1 (Pos 4) -> Expect '4':\00", align 1
@str_literal.106 = private unnamed_addr constant [30 x i8] c"Step 2 (Pos 5) -> Expect '3':\00", align 1
@str_literal.107 = private unnamed_addr constant [30 x i8] c"Step 3 (Pos 6) -> Expect '2':\00", align 1
@str_literal.108 = private unnamed_addr constant [30 x i8] c"Step 4 (Pos 7) -> Expect '1':\00", align 1
@str_literal.109 = private unnamed_addr constant [19 x i8] c"Loading weights...\00", align 1
@str_literal.110 = private unnamed_addr constant [26 x i8] c"reverse_model.safetensors\00", align 1
@key_str = private unnamed_addr constant [4 x i8] c"w.w\00", align 1
@key_str.111 = private unnamed_addr constant [4 x i8] c"p.w\00", align 1
@key_str.112 = private unnamed_addr constant [8 x i8] c"b1.l1.w\00", align 1
@key_str.113 = private unnamed_addr constant [8 x i8] c"b1.l1.b\00", align 1
@key_str.114 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.W\00", align 1
@key_str.115 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.b\00", align 1
@key_str.116 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.W\00", align 1
@key_str.117 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.b\00", align 1
@key_str.118 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.W\00", align 1
@key_str.119 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.b\00", align 1
@key_str.120 = private unnamed_addr constant [14 x i8] c"b1.a.o_proj.W\00", align 1
@key_str.121 = private unnamed_addr constant [14 x i8] c"b1.a.o_proj.b\00", align 1
@key_str.122 = private unnamed_addr constant [8 x i8] c"b1.l2.w\00", align 1
@key_str.123 = private unnamed_addr constant [8 x i8] c"b1.l2.b\00", align 1
@key_str.124 = private unnamed_addr constant [9 x i8] c"b1.m.f.W\00", align 1
@key_str.125 = private unnamed_addr constant [9 x i8] c"b1.m.f.b\00", align 1
@key_str.126 = private unnamed_addr constant [9 x i8] c"b1.m.p.W\00", align 1
@key_str.127 = private unnamed_addr constant [9 x i8] c"b1.m.p.b\00", align 1
@key_str.128 = private unnamed_addr constant [8 x i8] c"b2.l1.w\00", align 1
@key_str.129 = private unnamed_addr constant [8 x i8] c"b2.l1.b\00", align 1
@key_str.130 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.W\00", align 1
@key_str.131 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.b\00", align 1
@key_str.132 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.W\00", align 1
@key_str.133 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.b\00", align 1
@key_str.134 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.W\00", align 1
@key_str.135 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.b\00", align 1
@key_str.136 = private unnamed_addr constant [14 x i8] c"b2.a.o_proj.W\00", align 1
@key_str.137 = private unnamed_addr constant [14 x i8] c"b2.a.o_proj.b\00", align 1
@key_str.138 = private unnamed_addr constant [8 x i8] c"b2.l2.w\00", align 1
@key_str.139 = private unnamed_addr constant [8 x i8] c"b2.l2.b\00", align 1
@key_str.140 = private unnamed_addr constant [9 x i8] c"b2.m.f.W\00", align 1
@key_str.141 = private unnamed_addr constant [9 x i8] c"b2.m.f.b\00", align 1
@key_str.142 = private unnamed_addr constant [9 x i8] c"b2.m.p.W\00", align 1
@key_str.143 = private unnamed_addr constant [9 x i8] c"b2.m.p.b\00", align 1
@key_str.144 = private unnamed_addr constant [4 x i8] c"l.w\00", align 1
@key_str.145 = private unnamed_addr constant [4 x i8] c"l.b\00", align 1
@key_str.146 = private unnamed_addr constant [4 x i8] c"h.W\00", align 1
@key_str.147 = private unnamed_addr constant [4 x i8] c"h.b\00", align 1
@str_literal.148 = private unnamed_addr constant [16 x i8] c"Weights loaded.\00", align 1

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
  %init_field = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 0
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
  %init_field20 = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 1
  store ptr %detach_res19, ptr %init_field20, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 1
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
  %init_field = getelementptr inbounds nuw %LayerNorm, ptr %struct_malloc, i32 0, i32 0
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
  %init_field20 = getelementptr inbounds nuw %LayerNorm, ptr %struct_malloc, i32 0, i32 1
  store ptr %detach_res19, ptr %init_field20, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds nuw %LayerNorm, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds nuw %LayerNorm, ptr %struct_malloc, i32 0, i32 1
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
  %ptr_w = getelementptr inbounds nuw %LayerNorm, ptr %self4, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %binop_res = call ptr @tl_tensor_mul(ptr %x3, ptr %w)
  %self5 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds nuw %LayerNorm, ptr %self5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %binop_res6 = call ptr @tl_tensor_add(ptr %binop_res, ptr %b)
  call void @tl_mem_unregister(ptr %binop_res6)
  call void @tl_mem_exit_scope()
  ret ptr %binop_res6
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
  %init_field = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d4 = load i64, ptr %d1, align 8
  %d5 = load i64, ptr %d1, align 8
  %static_call6 = call ptr @tl_Linear_new(i64 %d4, i64 %d5)
  %init_field7 = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call6, ptr %init_field7, align 8
  %d8 = load i64, ptr %d1, align 8
  %d9 = load i64, ptr %d1, align 8
  %static_call10 = call ptr @tl_Linear_new(i64 %d8, i64 %d9)
  %init_field11 = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 2
  store ptr %static_call10, ptr %init_field11, align 8
  %d12 = load i64, ptr %d1, align 8
  %d13 = load i64, ptr %d1, align 8
  %static_call14 = call ptr @tl_Linear_new(i64 %d12, i64 %d13)
  %init_field15 = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 3
  store ptr %static_call14, ptr %init_field15, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_016 = getelementptr inbounds nuw %Linear, ptr %field_val, i32 0, i32 0
  %field_val17 = load ptr, ptr %unreg_field_016, align 8
  call void @tl_mem_unregister(ptr %field_val17)
  %unreg_field_1 = getelementptr inbounds nuw %Linear, ptr %field_val, i32 0, i32 1
  %field_val18 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val18)
  %unreg_field_119 = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 1
  %field_val20 = load ptr, ptr %unreg_field_119, align 8
  call void @tl_mem_unregister(ptr %field_val20)
  %unreg_field_021 = getelementptr inbounds nuw %Linear, ptr %field_val20, i32 0, i32 0
  %field_val22 = load ptr, ptr %unreg_field_021, align 8
  call void @tl_mem_unregister(ptr %field_val22)
  %unreg_field_123 = getelementptr inbounds nuw %Linear, ptr %field_val20, i32 0, i32 1
  %field_val24 = load ptr, ptr %unreg_field_123, align 8
  call void @tl_mem_unregister(ptr %field_val24)
  %unreg_field_2 = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 2
  %field_val25 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_026 = getelementptr inbounds nuw %Linear, ptr %field_val25, i32 0, i32 0
  %field_val27 = load ptr, ptr %unreg_field_026, align 8
  call void @tl_mem_unregister(ptr %field_val27)
  %unreg_field_128 = getelementptr inbounds nuw %Linear, ptr %field_val25, i32 0, i32 1
  %field_val29 = load ptr, ptr %unreg_field_128, align 8
  call void @tl_mem_unregister(ptr %field_val29)
  %unreg_field_3 = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 3
  %field_val30 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  %unreg_field_031 = getelementptr inbounds nuw %Linear, ptr %field_val30, i32 0, i32 0
  %field_val32 = load ptr, ptr %unreg_field_031, align 8
  call void @tl_mem_unregister(ptr %field_val32)
  %unreg_field_133 = getelementptr inbounds nuw %Linear, ptr %field_val30, i32 0, i32 1
  %field_val34 = load ptr, ptr %unreg_field_133, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_CausalSelfAttention_forward(ptr %self, ptr %x) {
entry:
  %y_out = alloca ptr, align 16
  %y_trans = alloca ptr, align 16
  %y_heads = alloca ptr, align 16
  %V_cont = alloca ptr, align 16
  %probs_cont = alloca ptr, align 16
  %probs = alloca ptr, align 16
  %masked = alloca ptr, align 16
  %logits = alloca ptr, align 16
  %Q_cont = alloca ptr, align 16
  %K_scaled = alloca ptr, align 16
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %K_heads_T = alloca ptr, align 16
  %V_heads = alloca ptr, align 16
  %V_split = alloca ptr, align 16
  %K_heads = alloca ptr, align 16
  %K_split = alloca ptr, align 16
  %Q_heads = alloca ptr, align 16
  %Q_split = alloca ptr, align 16
  %V = alloca ptr, align 16
  %K = alloca ptr, align 16
  %Q = alloca ptr, align 16
  %x2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_q_proj = getelementptr inbounds nuw %CausalSelfAttention, ptr %self3, i32 0, i32 0
  %q_proj = load ptr, ptr %ptr_q_proj, align 8
  %x4 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_Linear_forward(ptr %q_proj, ptr %x4)
  call void @tl_mem_register_tensor(ptr %call_method)
  store ptr %call_method, ptr %Q, align 8
  %self5 = load ptr, ptr %self1, align 8
  %ptr_k_proj = getelementptr inbounds nuw %CausalSelfAttention, ptr %self5, i32 0, i32 1
  %k_proj = load ptr, ptr %ptr_k_proj, align 8
  %x6 = load ptr, ptr %x2, align 8
  %call_method7 = call ptr @tl_Linear_forward(ptr %k_proj, ptr %x6)
  call void @tl_mem_register_tensor(ptr %call_method7)
  store ptr %call_method7, ptr %K, align 8
  %self8 = load ptr, ptr %self1, align 8
  %ptr_v_proj = getelementptr inbounds nuw %CausalSelfAttention, ptr %self8, i32 0, i32 2
  %v_proj = load ptr, ptr %ptr_v_proj, align 8
  %x9 = load ptr, ptr %x2, align 8
  %call_method10 = call ptr @tl_Linear_forward(ptr %v_proj, ptr %x9)
  call void @tl_mem_register_tensor(ptr %call_method10)
  store ptr %call_method10, ptr %V, align 8
  %Q11 = load ptr, ptr %Q, align 8
  %dims_alloca = alloca [3 x i64], align 8
  %dim_ptr = getelementptr [3 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 10, ptr %dim_ptr, align 8
  %dim_ptr12 = getelementptr [3 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 4, ptr %dim_ptr12, align 8
  %dim_ptr13 = getelementptr [3 x i64], ptr %dims_alloca, i64 0, i64 2
  store i64 32, ptr %dim_ptr13, align 8
  %dims_ptr = getelementptr [3 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %Q11, ptr %dims_ptr, i64 3)
  store ptr %reshape_dims_res, ptr %Q_split, align 8
  %Q_split14 = load ptr, ptr %Q_split, align 8
  %transpose_res = call ptr @tl_tensor_transpose(ptr %Q_split14, i64 0, i64 1)
  store ptr %transpose_res, ptr %Q_heads, align 8
  %K15 = load ptr, ptr %K, align 8
  %dims_alloca16 = alloca [3 x i64], align 8
  %dim_ptr17 = getelementptr [3 x i64], ptr %dims_alloca16, i64 0, i64 0
  store i64 10, ptr %dim_ptr17, align 8
  %dim_ptr18 = getelementptr [3 x i64], ptr %dims_alloca16, i64 0, i64 1
  store i64 4, ptr %dim_ptr18, align 8
  %dim_ptr19 = getelementptr [3 x i64], ptr %dims_alloca16, i64 0, i64 2
  store i64 32, ptr %dim_ptr19, align 8
  %dims_ptr20 = getelementptr [3 x i64], ptr %dims_alloca16, i64 0, i64 0
  %reshape_dims_res21 = call ptr @tl_tensor_reshape_dims(ptr %K15, ptr %dims_ptr20, i64 3)
  store ptr %reshape_dims_res21, ptr %K_split, align 8
  %K_split22 = load ptr, ptr %K_split, align 8
  %transpose_res23 = call ptr @tl_tensor_transpose(ptr %K_split22, i64 0, i64 1)
  store ptr %transpose_res23, ptr %K_heads, align 8
  %V24 = load ptr, ptr %V, align 8
  %dims_alloca25 = alloca [3 x i64], align 8
  %dim_ptr26 = getelementptr [3 x i64], ptr %dims_alloca25, i64 0, i64 0
  store i64 10, ptr %dim_ptr26, align 8
  %dim_ptr27 = getelementptr [3 x i64], ptr %dims_alloca25, i64 0, i64 1
  store i64 4, ptr %dim_ptr27, align 8
  %dim_ptr28 = getelementptr [3 x i64], ptr %dims_alloca25, i64 0, i64 2
  store i64 32, ptr %dim_ptr28, align 8
  %dims_ptr29 = getelementptr [3 x i64], ptr %dims_alloca25, i64 0, i64 0
  %reshape_dims_res30 = call ptr @tl_tensor_reshape_dims(ptr %V24, ptr %dims_ptr29, i64 3)
  store ptr %reshape_dims_res30, ptr %V_split, align 8
  %V_split31 = load ptr, ptr %V_split, align 8
  %transpose_res32 = call ptr @tl_tensor_transpose(ptr %V_split31, i64 0, i64 1)
  store ptr %transpose_res32, ptr %V_heads, align 8
  %K_heads33 = load ptr, ptr %K_heads, align 8
  %transpose_res34 = call ptr @tl_tensor_transpose(ptr %K_heads33, i64 1, i64 2)
  store ptr %transpose_res34, ptr %K_heads_T, align 8
  %K_heads_T35 = load ptr, ptr %K_heads_T, align 8
  store float 0x3FC6872B00000000, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %K_heads_T35, ptr %scalar_tensor_rhs)
  %contiguous_res = call ptr @tl_tensor_contiguous(ptr %binop_res)
  store ptr %contiguous_res, ptr %K_scaled, align 8
  %Q_heads36 = load ptr, ptr %Q_heads, align 8
  %contiguous_res37 = call ptr @tl_tensor_contiguous(ptr %Q_heads36)
  store ptr %contiguous_res37, ptr %Q_cont, align 8
  %Q_cont38 = load ptr, ptr %Q_cont, align 8
  %K_scaled39 = load ptr, ptr %K_scaled, align 8
  %matmul_res = call ptr @tl_tensor_matmul(ptr %Q_cont38, ptr %K_scaled39)
  store ptr %matmul_res, ptr %logits, align 8
  %logits40 = load ptr, ptr %logits, align 8
  %tril_res = call ptr @tl_tensor_tril(ptr %logits40, i32 0)
  store ptr %tril_res, ptr %masked, align 8
  %masked41 = load ptr, ptr %masked, align 8
  %softmax_res = call ptr @tl_tensor_softmax(ptr %masked41, i64 2)
  store ptr %softmax_res, ptr %probs, align 8
  %probs42 = load ptr, ptr %probs, align 8
  %contiguous_res43 = call ptr @tl_tensor_contiguous(ptr %probs42)
  store ptr %contiguous_res43, ptr %probs_cont, align 8
  %V_heads44 = load ptr, ptr %V_heads, align 8
  %contiguous_res45 = call ptr @tl_tensor_contiguous(ptr %V_heads44)
  store ptr %contiguous_res45, ptr %V_cont, align 8
  %probs_cont46 = load ptr, ptr %probs_cont, align 8
  %V_cont47 = load ptr, ptr %V_cont, align 8
  %matmul_res48 = call ptr @tl_tensor_matmul(ptr %probs_cont46, ptr %V_cont47)
  store ptr %matmul_res48, ptr %y_heads, align 8
  %y_heads49 = load ptr, ptr %y_heads, align 8
  %transpose_res50 = call ptr @tl_tensor_transpose(ptr %y_heads49, i64 0, i64 1)
  store ptr %transpose_res50, ptr %y_trans, align 8
  %y_trans51 = load ptr, ptr %y_trans, align 8
  %dims_alloca52 = alloca [3 x i64], align 8
  %dim_ptr53 = getelementptr [3 x i64], ptr %dims_alloca52, i64 0, i64 0
  store i64 1, ptr %dim_ptr53, align 8
  %dim_ptr54 = getelementptr [3 x i64], ptr %dims_alloca52, i64 0, i64 1
  store i64 10, ptr %dim_ptr54, align 8
  %dim_ptr55 = getelementptr [3 x i64], ptr %dims_alloca52, i64 0, i64 2
  store i64 128, ptr %dim_ptr55, align 8
  %dims_ptr56 = getelementptr [3 x i64], ptr %dims_alloca52, i64 0, i64 0
  %reshape_dims_res57 = call ptr @tl_tensor_reshape_dims(ptr %y_trans51, ptr %dims_ptr56, i64 3)
  store ptr %reshape_dims_res57, ptr %y_out, align 8
  %self58 = load ptr, ptr %self1, align 8
  %ptr_o_proj = getelementptr inbounds nuw %CausalSelfAttention, ptr %self58, i32 0, i32 3
  %o_proj = load ptr, ptr %ptr_o_proj, align 8
  %y_out59 = load ptr, ptr %y_out, align 8
  %call_method60 = call ptr @tl_Linear_forward(ptr %o_proj, ptr %y_out59)
  call void @tl_mem_register_tensor(ptr %call_method60)
  call void @tl_mem_unregister(ptr %call_method60)
  call void @tl_mem_exit_scope()
  ret ptr %call_method60
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
  %unreg_field_2 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val16, i32 0, i32 2
  %field_val29 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val29)
  %unreg_field_030 = getelementptr inbounds nuw %Linear, ptr %field_val29, i32 0, i32 0
  %field_val31 = load ptr, ptr %unreg_field_030, align 8
  call void @tl_mem_unregister(ptr %field_val31)
  %unreg_field_132 = getelementptr inbounds nuw %Linear, ptr %field_val29, i32 0, i32 1
  %field_val33 = load ptr, ptr %unreg_field_132, align 8
  call void @tl_mem_unregister(ptr %field_val33)
  %unreg_field_3 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val16, i32 0, i32 3
  %field_val34 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  %unreg_field_035 = getelementptr inbounds nuw %Linear, ptr %field_val34, i32 0, i32 0
  %field_val36 = load ptr, ptr %unreg_field_035, align 8
  call void @tl_mem_unregister(ptr %field_val36)
  %unreg_field_137 = getelementptr inbounds nuw %Linear, ptr %field_val34, i32 0, i32 1
  %field_val38 = load ptr, ptr %unreg_field_137, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_239 = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 2
  %field_val40 = load ptr, ptr %unreg_field_239, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_041 = getelementptr inbounds nuw %LayerNorm, ptr %field_val40, i32 0, i32 0
  %field_val42 = load ptr, ptr %unreg_field_041, align 8
  call void @tl_mem_unregister(ptr %field_val42)
  %unreg_field_143 = getelementptr inbounds nuw %LayerNorm, ptr %field_val40, i32 0, i32 1
  %field_val44 = load ptr, ptr %unreg_field_143, align 8
  call void @tl_mem_unregister(ptr %field_val44)
  %unreg_field_345 = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 3
  %field_val46 = load ptr, ptr %unreg_field_345, align 8
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

define ptr @tl_PositionalEmbedding_new(i64 %max_len, i64 %d) {
entry:
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %d2 = alloca i64, align 16
  %max_len1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %max_len, ptr %max_len1, align 8
  store i64 %d, ptr %d2, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%PositionalEmbedding, ptr null, i32 1) to i64))
  %buf_void = call ptr @tl_alloc_tmp(i64 8)
  %max_len3 = load i64, ptr %max_len1, align 8
  %i2f = sitofp i64 %max_len3 to float
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
  store float 0x3F947AE140000000, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %static_call, ptr %scalar_tensor_rhs)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res, i1 true)
  %init_field = getelementptr inbounds nuw %PositionalEmbedding, ptr %struct_malloc, i32 0, i32 0
  store ptr %detach_res, ptr %init_field, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds nuw %PositionalEmbedding, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_PositionalEmbedding_forward(ptr %self, ptr %p) {
entry:
  %p2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store ptr %p, ptr %p2, align 8
  %p3 = load ptr, ptr %p2, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds nuw %PositionalEmbedding, ptr %self4, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %emb_res = call ptr @tl_tensor_embedding(ptr %p3, ptr %w)
  call void @tl_mem_unregister(ptr %emb_res)
  call void @tl_mem_exit_scope()
  ret ptr %emb_res
}

define ptr @tl_SeqModel_new(i64 %v, i64 %max_len, i64 %d) {
entry:
  %d3 = alloca i64, align 16
  %max_len2 = alloca i64, align 16
  %v1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %v, ptr %v1, align 8
  store i64 %max_len, ptr %max_len2, align 8
  store i64 %d, ptr %d3, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%SeqModel, ptr null, i32 1) to i64))
  %v4 = load i64, ptr %v1, align 8
  %d5 = load i64, ptr %d3, align 8
  %static_call = call ptr @tl_Embedding_new(i64 %v4, i64 %d5)
  %init_field = getelementptr inbounds nuw %SeqModel, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %max_len6 = load i64, ptr %max_len2, align 8
  %d7 = load i64, ptr %d3, align 8
  %static_call8 = call ptr @tl_PositionalEmbedding_new(i64 %max_len6, i64 %d7)
  %init_field9 = getelementptr inbounds nuw %SeqModel, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call8, ptr %init_field9, align 8
  %d10 = load i64, ptr %d3, align 8
  %static_call11 = call ptr @tl_Block_new(i64 %d10)
  %init_field12 = getelementptr inbounds nuw %SeqModel, ptr %struct_malloc, i32 0, i32 2
  store ptr %static_call11, ptr %init_field12, align 8
  %d13 = load i64, ptr %d3, align 8
  %static_call14 = call ptr @tl_Block_new(i64 %d13)
  %init_field15 = getelementptr inbounds nuw %SeqModel, ptr %struct_malloc, i32 0, i32 3
  store ptr %static_call14, ptr %init_field15, align 8
  %d16 = load i64, ptr %d3, align 8
  %static_call17 = call ptr @tl_LayerNorm_new(i64 %d16)
  %init_field18 = getelementptr inbounds nuw %SeqModel, ptr %struct_malloc, i32 0, i32 4
  store ptr %static_call17, ptr %init_field18, align 8
  %d19 = load i64, ptr %d3, align 8
  %v20 = load i64, ptr %v1, align 8
  %static_call21 = call ptr @tl_Linear_new(i64 %d19, i64 %v20)
  %init_field22 = getelementptr inbounds nuw %SeqModel, ptr %struct_malloc, i32 0, i32 5
  store ptr %static_call21, ptr %init_field22, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds nuw %SeqModel, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_023 = getelementptr inbounds nuw %Embedding, ptr %field_val, i32 0, i32 0
  %field_val24 = load ptr, ptr %unreg_field_023, align 8
  call void @tl_mem_unregister(ptr %field_val24)
  %unreg_field_1 = getelementptr inbounds nuw %SeqModel, ptr %struct_malloc, i32 0, i32 1
  %field_val25 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_026 = getelementptr inbounds nuw %PositionalEmbedding, ptr %field_val25, i32 0, i32 0
  %field_val27 = load ptr, ptr %unreg_field_026, align 8
  call void @tl_mem_unregister(ptr %field_val27)
  %unreg_field_2 = getelementptr inbounds nuw %SeqModel, ptr %struct_malloc, i32 0, i32 2
  %field_val28 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_029 = getelementptr inbounds nuw %Block, ptr %field_val28, i32 0, i32 0
  %field_val30 = load ptr, ptr %unreg_field_029, align 8
  call void @tl_mem_unregister(ptr %field_val30)
  %unreg_field_031 = getelementptr inbounds nuw %LayerNorm, ptr %field_val30, i32 0, i32 0
  %field_val32 = load ptr, ptr %unreg_field_031, align 8
  call void @tl_mem_unregister(ptr %field_val32)
  %unreg_field_133 = getelementptr inbounds nuw %LayerNorm, ptr %field_val30, i32 0, i32 1
  %field_val34 = load ptr, ptr %unreg_field_133, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  %unreg_field_135 = getelementptr inbounds nuw %Block, ptr %field_val28, i32 0, i32 1
  %field_val36 = load ptr, ptr %unreg_field_135, align 8
  call void @tl_mem_unregister(ptr %field_val36)
  %unreg_field_037 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val36, i32 0, i32 0
  %field_val38 = load ptr, ptr %unreg_field_037, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_039 = getelementptr inbounds nuw %Linear, ptr %field_val38, i32 0, i32 0
  %field_val40 = load ptr, ptr %unreg_field_039, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_141 = getelementptr inbounds nuw %Linear, ptr %field_val38, i32 0, i32 1
  %field_val42 = load ptr, ptr %unreg_field_141, align 8
  call void @tl_mem_unregister(ptr %field_val42)
  %unreg_field_143 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val36, i32 0, i32 1
  %field_val44 = load ptr, ptr %unreg_field_143, align 8
  call void @tl_mem_unregister(ptr %field_val44)
  %unreg_field_045 = getelementptr inbounds nuw %Linear, ptr %field_val44, i32 0, i32 0
  %field_val46 = load ptr, ptr %unreg_field_045, align 8
  call void @tl_mem_unregister(ptr %field_val46)
  %unreg_field_147 = getelementptr inbounds nuw %Linear, ptr %field_val44, i32 0, i32 1
  %field_val48 = load ptr, ptr %unreg_field_147, align 8
  call void @tl_mem_unregister(ptr %field_val48)
  %unreg_field_249 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val36, i32 0, i32 2
  %field_val50 = load ptr, ptr %unreg_field_249, align 8
  call void @tl_mem_unregister(ptr %field_val50)
  %unreg_field_051 = getelementptr inbounds nuw %Linear, ptr %field_val50, i32 0, i32 0
  %field_val52 = load ptr, ptr %unreg_field_051, align 8
  call void @tl_mem_unregister(ptr %field_val52)
  %unreg_field_153 = getelementptr inbounds nuw %Linear, ptr %field_val50, i32 0, i32 1
  %field_val54 = load ptr, ptr %unreg_field_153, align 8
  call void @tl_mem_unregister(ptr %field_val54)
  %unreg_field_3 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val36, i32 0, i32 3
  %field_val55 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val55)
  %unreg_field_056 = getelementptr inbounds nuw %Linear, ptr %field_val55, i32 0, i32 0
  %field_val57 = load ptr, ptr %unreg_field_056, align 8
  call void @tl_mem_unregister(ptr %field_val57)
  %unreg_field_158 = getelementptr inbounds nuw %Linear, ptr %field_val55, i32 0, i32 1
  %field_val59 = load ptr, ptr %unreg_field_158, align 8
  call void @tl_mem_unregister(ptr %field_val59)
  %unreg_field_260 = getelementptr inbounds nuw %Block, ptr %field_val28, i32 0, i32 2
  %field_val61 = load ptr, ptr %unreg_field_260, align 8
  call void @tl_mem_unregister(ptr %field_val61)
  %unreg_field_062 = getelementptr inbounds nuw %LayerNorm, ptr %field_val61, i32 0, i32 0
  %field_val63 = load ptr, ptr %unreg_field_062, align 8
  call void @tl_mem_unregister(ptr %field_val63)
  %unreg_field_164 = getelementptr inbounds nuw %LayerNorm, ptr %field_val61, i32 0, i32 1
  %field_val65 = load ptr, ptr %unreg_field_164, align 8
  call void @tl_mem_unregister(ptr %field_val65)
  %unreg_field_366 = getelementptr inbounds nuw %Block, ptr %field_val28, i32 0, i32 3
  %field_val67 = load ptr, ptr %unreg_field_366, align 8
  call void @tl_mem_unregister(ptr %field_val67)
  %unreg_field_068 = getelementptr inbounds nuw %MLP, ptr %field_val67, i32 0, i32 0
  %field_val69 = load ptr, ptr %unreg_field_068, align 8
  call void @tl_mem_unregister(ptr %field_val69)
  %unreg_field_070 = getelementptr inbounds nuw %Linear, ptr %field_val69, i32 0, i32 0
  %field_val71 = load ptr, ptr %unreg_field_070, align 8
  call void @tl_mem_unregister(ptr %field_val71)
  %unreg_field_172 = getelementptr inbounds nuw %Linear, ptr %field_val69, i32 0, i32 1
  %field_val73 = load ptr, ptr %unreg_field_172, align 8
  call void @tl_mem_unregister(ptr %field_val73)
  %unreg_field_174 = getelementptr inbounds nuw %MLP, ptr %field_val67, i32 0, i32 1
  %field_val75 = load ptr, ptr %unreg_field_174, align 8
  call void @tl_mem_unregister(ptr %field_val75)
  %unreg_field_076 = getelementptr inbounds nuw %Linear, ptr %field_val75, i32 0, i32 0
  %field_val77 = load ptr, ptr %unreg_field_076, align 8
  call void @tl_mem_unregister(ptr %field_val77)
  %unreg_field_178 = getelementptr inbounds nuw %Linear, ptr %field_val75, i32 0, i32 1
  %field_val79 = load ptr, ptr %unreg_field_178, align 8
  call void @tl_mem_unregister(ptr %field_val79)
  %unreg_field_380 = getelementptr inbounds nuw %SeqModel, ptr %struct_malloc, i32 0, i32 3
  %field_val81 = load ptr, ptr %unreg_field_380, align 8
  call void @tl_mem_unregister(ptr %field_val81)
  %unreg_field_082 = getelementptr inbounds nuw %Block, ptr %field_val81, i32 0, i32 0
  %field_val83 = load ptr, ptr %unreg_field_082, align 8
  call void @tl_mem_unregister(ptr %field_val83)
  %unreg_field_084 = getelementptr inbounds nuw %LayerNorm, ptr %field_val83, i32 0, i32 0
  %field_val85 = load ptr, ptr %unreg_field_084, align 8
  call void @tl_mem_unregister(ptr %field_val85)
  %unreg_field_186 = getelementptr inbounds nuw %LayerNorm, ptr %field_val83, i32 0, i32 1
  %field_val87 = load ptr, ptr %unreg_field_186, align 8
  call void @tl_mem_unregister(ptr %field_val87)
  %unreg_field_188 = getelementptr inbounds nuw %Block, ptr %field_val81, i32 0, i32 1
  %field_val89 = load ptr, ptr %unreg_field_188, align 8
  call void @tl_mem_unregister(ptr %field_val89)
  %unreg_field_090 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val89, i32 0, i32 0
  %field_val91 = load ptr, ptr %unreg_field_090, align 8
  call void @tl_mem_unregister(ptr %field_val91)
  %unreg_field_092 = getelementptr inbounds nuw %Linear, ptr %field_val91, i32 0, i32 0
  %field_val93 = load ptr, ptr %unreg_field_092, align 8
  call void @tl_mem_unregister(ptr %field_val93)
  %unreg_field_194 = getelementptr inbounds nuw %Linear, ptr %field_val91, i32 0, i32 1
  %field_val95 = load ptr, ptr %unreg_field_194, align 8
  call void @tl_mem_unregister(ptr %field_val95)
  %unreg_field_196 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val89, i32 0, i32 1
  %field_val97 = load ptr, ptr %unreg_field_196, align 8
  call void @tl_mem_unregister(ptr %field_val97)
  %unreg_field_098 = getelementptr inbounds nuw %Linear, ptr %field_val97, i32 0, i32 0
  %field_val99 = load ptr, ptr %unreg_field_098, align 8
  call void @tl_mem_unregister(ptr %field_val99)
  %unreg_field_1100 = getelementptr inbounds nuw %Linear, ptr %field_val97, i32 0, i32 1
  %field_val101 = load ptr, ptr %unreg_field_1100, align 8
  call void @tl_mem_unregister(ptr %field_val101)
  %unreg_field_2102 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val89, i32 0, i32 2
  %field_val103 = load ptr, ptr %unreg_field_2102, align 8
  call void @tl_mem_unregister(ptr %field_val103)
  %unreg_field_0104 = getelementptr inbounds nuw %Linear, ptr %field_val103, i32 0, i32 0
  %field_val105 = load ptr, ptr %unreg_field_0104, align 8
  call void @tl_mem_unregister(ptr %field_val105)
  %unreg_field_1106 = getelementptr inbounds nuw %Linear, ptr %field_val103, i32 0, i32 1
  %field_val107 = load ptr, ptr %unreg_field_1106, align 8
  call void @tl_mem_unregister(ptr %field_val107)
  %unreg_field_3108 = getelementptr inbounds nuw %CausalSelfAttention, ptr %field_val89, i32 0, i32 3
  %field_val109 = load ptr, ptr %unreg_field_3108, align 8
  call void @tl_mem_unregister(ptr %field_val109)
  %unreg_field_0110 = getelementptr inbounds nuw %Linear, ptr %field_val109, i32 0, i32 0
  %field_val111 = load ptr, ptr %unreg_field_0110, align 8
  call void @tl_mem_unregister(ptr %field_val111)
  %unreg_field_1112 = getelementptr inbounds nuw %Linear, ptr %field_val109, i32 0, i32 1
  %field_val113 = load ptr, ptr %unreg_field_1112, align 8
  call void @tl_mem_unregister(ptr %field_val113)
  %unreg_field_2114 = getelementptr inbounds nuw %Block, ptr %field_val81, i32 0, i32 2
  %field_val115 = load ptr, ptr %unreg_field_2114, align 8
  call void @tl_mem_unregister(ptr %field_val115)
  %unreg_field_0116 = getelementptr inbounds nuw %LayerNorm, ptr %field_val115, i32 0, i32 0
  %field_val117 = load ptr, ptr %unreg_field_0116, align 8
  call void @tl_mem_unregister(ptr %field_val117)
  %unreg_field_1118 = getelementptr inbounds nuw %LayerNorm, ptr %field_val115, i32 0, i32 1
  %field_val119 = load ptr, ptr %unreg_field_1118, align 8
  call void @tl_mem_unregister(ptr %field_val119)
  %unreg_field_3120 = getelementptr inbounds nuw %Block, ptr %field_val81, i32 0, i32 3
  %field_val121 = load ptr, ptr %unreg_field_3120, align 8
  call void @tl_mem_unregister(ptr %field_val121)
  %unreg_field_0122 = getelementptr inbounds nuw %MLP, ptr %field_val121, i32 0, i32 0
  %field_val123 = load ptr, ptr %unreg_field_0122, align 8
  call void @tl_mem_unregister(ptr %field_val123)
  %unreg_field_0124 = getelementptr inbounds nuw %Linear, ptr %field_val123, i32 0, i32 0
  %field_val125 = load ptr, ptr %unreg_field_0124, align 8
  call void @tl_mem_unregister(ptr %field_val125)
  %unreg_field_1126 = getelementptr inbounds nuw %Linear, ptr %field_val123, i32 0, i32 1
  %field_val127 = load ptr, ptr %unreg_field_1126, align 8
  call void @tl_mem_unregister(ptr %field_val127)
  %unreg_field_1128 = getelementptr inbounds nuw %MLP, ptr %field_val121, i32 0, i32 1
  %field_val129 = load ptr, ptr %unreg_field_1128, align 8
  call void @tl_mem_unregister(ptr %field_val129)
  %unreg_field_0130 = getelementptr inbounds nuw %Linear, ptr %field_val129, i32 0, i32 0
  %field_val131 = load ptr, ptr %unreg_field_0130, align 8
  call void @tl_mem_unregister(ptr %field_val131)
  %unreg_field_1132 = getelementptr inbounds nuw %Linear, ptr %field_val129, i32 0, i32 1
  %field_val133 = load ptr, ptr %unreg_field_1132, align 8
  call void @tl_mem_unregister(ptr %field_val133)
  %unreg_field_4 = getelementptr inbounds nuw %SeqModel, ptr %struct_malloc, i32 0, i32 4
  %field_val134 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val134)
  %unreg_field_0135 = getelementptr inbounds nuw %LayerNorm, ptr %field_val134, i32 0, i32 0
  %field_val136 = load ptr, ptr %unreg_field_0135, align 8
  call void @tl_mem_unregister(ptr %field_val136)
  %unreg_field_1137 = getelementptr inbounds nuw %LayerNorm, ptr %field_val134, i32 0, i32 1
  %field_val138 = load ptr, ptr %unreg_field_1137, align 8
  call void @tl_mem_unregister(ptr %field_val138)
  %unreg_field_5 = getelementptr inbounds nuw %SeqModel, ptr %struct_malloc, i32 0, i32 5
  %field_val139 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val139)
  %unreg_field_0140 = getelementptr inbounds nuw %Linear, ptr %field_val139, i32 0, i32 0
  %field_val141 = load ptr, ptr %unreg_field_0140, align 8
  call void @tl_mem_unregister(ptr %field_val141)
  %unreg_field_1142 = getelementptr inbounds nuw %Linear, ptr %field_val139, i32 0, i32 1
  %field_val143 = load ptr, ptr %unreg_field_1142, align 8
  call void @tl_mem_unregister(ptr %field_val143)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_SeqModel_forward(ptr %self, ptr %i, ptr %p) {
entry:
  %x20 = alloca ptr, align 16
  %x15 = alloca ptr, align 16
  %x = alloca ptr, align 16
  %pos_emb = alloca ptr, align 16
  %tok_emb = alloca ptr, align 16
  %p3 = alloca ptr, align 16
  %i2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store ptr %i, ptr %i2, align 8
  store ptr %p, ptr %p3, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds nuw %SeqModel, ptr %self4, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %i5 = load ptr, ptr %i2, align 8
  %call_method = call ptr @tl_Embedding_forward(ptr %w, ptr %i5)
  call void @tl_mem_register_tensor(ptr %call_method)
  store ptr %call_method, ptr %tok_emb, align 8
  %self6 = load ptr, ptr %self1, align 8
  %ptr_p = getelementptr inbounds nuw %SeqModel, ptr %self6, i32 0, i32 1
  %p7 = load ptr, ptr %ptr_p, align 8
  %p8 = load ptr, ptr %p3, align 8
  %call_method9 = call ptr @tl_PositionalEmbedding_forward(ptr %p7, ptr %p8)
  call void @tl_mem_register_tensor(ptr %call_method9)
  store ptr %call_method9, ptr %pos_emb, align 8
  %tok_emb10 = load ptr, ptr %tok_emb, align 8
  %pos_emb11 = load ptr, ptr %pos_emb, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %tok_emb10, ptr %pos_emb11)
  store ptr %binop_res, ptr %x, align 8
  %self12 = load ptr, ptr %self1, align 8
  %ptr_b1 = getelementptr inbounds nuw %SeqModel, ptr %self12, i32 0, i32 2
  %b1 = load ptr, ptr %ptr_b1, align 8
  %x13 = load ptr, ptr %x, align 8
  %call_method14 = call ptr @tl_Block_forward(ptr %b1, ptr %x13)
  call void @tl_mem_register_tensor(ptr %call_method14)
  %old_shadowed = load ptr, ptr %x, align 8
  call void @tl_mem_unregister(ptr %old_shadowed)
  store ptr %call_method14, ptr %x15, align 8
  %self16 = load ptr, ptr %self1, align 8
  %ptr_b2 = getelementptr inbounds nuw %SeqModel, ptr %self16, i32 0, i32 3
  %b2 = load ptr, ptr %ptr_b2, align 8
  %x17 = load ptr, ptr %x15, align 8
  %call_method18 = call ptr @tl_Block_forward(ptr %b2, ptr %x17)
  call void @tl_mem_register_tensor(ptr %call_method18)
  %old_shadowed19 = load ptr, ptr %x15, align 8
  call void @tl_mem_unregister(ptr %old_shadowed19)
  store ptr %call_method18, ptr %x20, align 8
  %self21 = load ptr, ptr %self1, align 8
  %ptr_h = getelementptr inbounds nuw %SeqModel, ptr %self21, i32 0, i32 5
  %h = load ptr, ptr %ptr_h, align 8
  %self22 = load ptr, ptr %self1, align 8
  %ptr_l = getelementptr inbounds nuw %SeqModel, ptr %self22, i32 0, i32 4
  %l = load ptr, ptr %ptr_l, align 8
  %x23 = load ptr, ptr %x20, align 8
  %call_method24 = call ptr @tl_LayerNorm_forward(ptr %l, ptr %x23)
  call void @tl_mem_register_tensor(ptr %call_method24)
  %call_method25 = call ptr @tl_Linear_forward(ptr %h, ptr %call_method24)
  call void @tl_mem_register_tensor(ptr %call_method25)
  call void @tl_mem_unregister(ptr %call_method25)
  call void @tl_mem_exit_scope()
  ret ptr %call_method25
}

define void @inference(ptr %model) {
entry:
  %l_step4 = alloca ptr, align 16
  %l_step3 = alloca ptr, align 16
  %l_step2 = alloca ptr, align 16
  %l_step1 = alloca ptr, align 16
  %logits_flat = alloca ptr, align 16
  %logits = alloca ptr, align 16
  %X = alloca ptr, align 16
  %P_t = alloca ptr, align 16
  %P_arr = alloca ptr, align 16
  %p9 = alloca float, align 16
  %p8 = alloca float, align 16
  %p7 = alloca float, align 16
  %p6 = alloca float, align 16
  %p5 = alloca float, align 16
  %p4 = alloca float, align 16
  %p3 = alloca float, align 16
  %p2 = alloca float, align 16
  %p1 = alloca float, align 16
  %p0 = alloca float, align 16
  %input = alloca ptr, align 16
  %d4 = alloca float, align 16
  %d3 = alloca float, align 16
  %d2 = alloca float, align 16
  %d1 = alloca float, align 16
  %PAD = alloca float, align 16
  %SEP = alloca float, align 16
  %model1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %model, ptr %model1, align 8
  store float 1.000000e+01, ptr %SEP, align 4
  store float 1.100000e+01, ptr %PAD, align 4
  store float 1.000000e+00, ptr %d1, align 4
  store float 2.000000e+00, ptr %d2, align 4
  store float 3.000000e+00, ptr %d3, align 4
  store float 4.000000e+00, ptr %d4, align 4
  %buf_void = call ptr @tl_alloc_tmp(i64 40)
  %d12 = load float, ptr %d1, align 4
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %d12, ptr %elem_ptr, align 4
  %d23 = load float, ptr %d2, align 4
  %elem_ptr4 = getelementptr inbounds float, ptr %buf_void, i64 1
  store float %d23, ptr %elem_ptr4, align 4
  %d35 = load float, ptr %d3, align 4
  %elem_ptr6 = getelementptr inbounds float, ptr %buf_void, i64 2
  store float %d35, ptr %elem_ptr6, align 4
  %d47 = load float, ptr %d4, align 4
  %elem_ptr8 = getelementptr inbounds float, ptr %buf_void, i64 3
  store float %d47, ptr %elem_ptr8, align 4
  %SEP9 = load float, ptr %SEP, align 4
  %elem_ptr10 = getelementptr inbounds float, ptr %buf_void, i64 4
  store float %SEP9, ptr %elem_ptr10, align 4
  %d411 = load float, ptr %d4, align 4
  %elem_ptr12 = getelementptr inbounds float, ptr %buf_void, i64 5
  store float %d411, ptr %elem_ptr12, align 4
  %d313 = load float, ptr %d3, align 4
  %elem_ptr14 = getelementptr inbounds float, ptr %buf_void, i64 6
  store float %d313, ptr %elem_ptr14, align 4
  %d215 = load float, ptr %d2, align 4
  %elem_ptr16 = getelementptr inbounds float, ptr %buf_void, i64 7
  store float %d215, ptr %elem_ptr16, align 4
  %d117 = load float, ptr %d1, align 4
  %elem_ptr18 = getelementptr inbounds float, ptr %buf_void, i64 8
  store float %d117, ptr %elem_ptr18, align 4
  %PAD19 = load float, ptr %PAD, align 4
  %elem_ptr20 = getelementptr inbounds float, ptr %buf_void, i64 9
  store float %PAD19, ptr %elem_ptr20, align 4
  %shape_alloc = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr = getelementptr inbounds i64, ptr %shape_alloc, i64 0
  store i64 10, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %shape_alloc)
  call void @tl_free_tmp(ptr %buf_void)
  call void @tl_free_tmp(ptr %shape_alloc)
  store ptr %new_tensor, ptr %input, align 8
  call void @tl_print_string(ptr @str_literal)
  %input21 = load ptr, ptr %input, align 8
  call void @tl_tensor_print(ptr %input21)
  store float 0.000000e+00, ptr %p0, align 4
  store float 1.000000e+00, ptr %p1, align 4
  store float 2.000000e+00, ptr %p2, align 4
  store float 3.000000e+00, ptr %p3, align 4
  store float 4.000000e+00, ptr %p4, align 4
  store float 5.000000e+00, ptr %p5, align 4
  store float 6.000000e+00, ptr %p6, align 4
  store float 7.000000e+00, ptr %p7, align 4
  store float 8.000000e+00, ptr %p8, align 4
  store float 9.000000e+00, ptr %p9, align 4
  %buf_void22 = call ptr @tl_alloc_tmp(i64 40)
  %p023 = load float, ptr %p0, align 4
  %elem_ptr24 = getelementptr inbounds float, ptr %buf_void22, i64 0
  store float %p023, ptr %elem_ptr24, align 4
  %p125 = load float, ptr %p1, align 4
  %elem_ptr26 = getelementptr inbounds float, ptr %buf_void22, i64 1
  store float %p125, ptr %elem_ptr26, align 4
  %p227 = load float, ptr %p2, align 4
  %elem_ptr28 = getelementptr inbounds float, ptr %buf_void22, i64 2
  store float %p227, ptr %elem_ptr28, align 4
  %p329 = load float, ptr %p3, align 4
  %elem_ptr30 = getelementptr inbounds float, ptr %buf_void22, i64 3
  store float %p329, ptr %elem_ptr30, align 4
  %p431 = load float, ptr %p4, align 4
  %elem_ptr32 = getelementptr inbounds float, ptr %buf_void22, i64 4
  store float %p431, ptr %elem_ptr32, align 4
  %p533 = load float, ptr %p5, align 4
  %elem_ptr34 = getelementptr inbounds float, ptr %buf_void22, i64 5
  store float %p533, ptr %elem_ptr34, align 4
  %p635 = load float, ptr %p6, align 4
  %elem_ptr36 = getelementptr inbounds float, ptr %buf_void22, i64 6
  store float %p635, ptr %elem_ptr36, align 4
  %p737 = load float, ptr %p7, align 4
  %elem_ptr38 = getelementptr inbounds float, ptr %buf_void22, i64 7
  store float %p737, ptr %elem_ptr38, align 4
  %p839 = load float, ptr %p8, align 4
  %elem_ptr40 = getelementptr inbounds float, ptr %buf_void22, i64 8
  store float %p839, ptr %elem_ptr40, align 4
  %p941 = load float, ptr %p9, align 4
  %elem_ptr42 = getelementptr inbounds float, ptr %buf_void22, i64 9
  store float %p941, ptr %elem_ptr42, align 4
  %shape_alloc43 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr44 = getelementptr inbounds i64, ptr %shape_alloc43, i64 0
  store i64 10, ptr %shape_ptr44, align 8
  %new_tensor45 = call ptr @tl_tensor_new(ptr %buf_void22, i64 1, ptr %shape_alloc43)
  call void @tl_free_tmp(ptr %buf_void22)
  call void @tl_free_tmp(ptr %shape_alloc43)
  store ptr %new_tensor45, ptr %P_arr, align 8
  %P_arr46 = load ptr, ptr %P_arr, align 8
  %dims_alloca = alloca [2 x i64], align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr, align 8
  %dim_ptr47 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 10, ptr %dim_ptr47, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %P_arr46, ptr %dims_ptr, i64 2)
  store ptr %reshape_dims_res, ptr %P_t, align 8
  %input48 = load ptr, ptr %input, align 8
  %dims_alloca49 = alloca [2 x i64], align 8
  %dim_ptr50 = getelementptr [2 x i64], ptr %dims_alloca49, i64 0, i64 0
  store i64 1, ptr %dim_ptr50, align 8
  %dim_ptr51 = getelementptr [2 x i64], ptr %dims_alloca49, i64 0, i64 1
  store i64 10, ptr %dim_ptr51, align 8
  %dims_ptr52 = getelementptr [2 x i64], ptr %dims_alloca49, i64 0, i64 0
  %reshape_dims_res53 = call ptr @tl_tensor_reshape_dims(ptr %input48, ptr %dims_ptr52, i64 2)
  store ptr %reshape_dims_res53, ptr %X, align 8
  %model54 = load ptr, ptr %model1, align 8
  %X55 = load ptr, ptr %X, align 8
  %P_t56 = load ptr, ptr %P_t, align 8
  %call_method = call ptr @tl_SeqModel_forward(ptr %model54, ptr %X55, ptr %P_t56)
  call void @tl_mem_register_tensor(ptr %call_method)
  store ptr %call_method, ptr %logits, align 8
  %logits57 = load ptr, ptr %logits, align 8
  %dims_alloca58 = alloca [2 x i64], align 8
  %dim_ptr59 = getelementptr [2 x i64], ptr %dims_alloca58, i64 0, i64 0
  store i64 10, ptr %dim_ptr59, align 8
  %dim_ptr60 = getelementptr [2 x i64], ptr %dims_alloca58, i64 0, i64 1
  store i64 12, ptr %dim_ptr60, align 8
  %dims_ptr61 = getelementptr [2 x i64], ptr %dims_alloca58, i64 0, i64 0
  %reshape_dims_res62 = call ptr @tl_tensor_reshape_dims(ptr %logits57, ptr %dims_ptr61, i64 2)
  store ptr %reshape_dims_res62, ptr %logits_flat, align 8
  call void @tl_print_string(ptr @str_literal.104)
  %logits_flat63 = load ptr, ptr %logits_flat, align 8
  %call_tmp = call ptr @tl_tensor_slice(ptr %logits_flat63, i64 4, i64 1)
  call void @tl_mem_register_tensor(ptr %call_tmp)
  store ptr %call_tmp, ptr %l_step1, align 8
  call void @tl_print_string(ptr @str_literal.105)
  %l_step164 = load ptr, ptr %l_step1, align 8
  call void @tl_tensor_print(ptr %l_step164)
  %logits_flat65 = load ptr, ptr %logits_flat, align 8
  %call_tmp66 = call ptr @tl_tensor_slice(ptr %logits_flat65, i64 5, i64 1)
  call void @tl_mem_register_tensor(ptr %call_tmp66)
  store ptr %call_tmp66, ptr %l_step2, align 8
  call void @tl_print_string(ptr @str_literal.106)
  %l_step267 = load ptr, ptr %l_step2, align 8
  call void @tl_tensor_print(ptr %l_step267)
  %logits_flat68 = load ptr, ptr %logits_flat, align 8
  %call_tmp69 = call ptr @tl_tensor_slice(ptr %logits_flat68, i64 6, i64 1)
  call void @tl_mem_register_tensor(ptr %call_tmp69)
  store ptr %call_tmp69, ptr %l_step3, align 8
  call void @tl_print_string(ptr @str_literal.107)
  %l_step370 = load ptr, ptr %l_step3, align 8
  call void @tl_tensor_print(ptr %l_step370)
  %logits_flat71 = load ptr, ptr %logits_flat, align 8
  %call_tmp72 = call ptr @tl_tensor_slice(ptr %logits_flat71, i64 7, i64 1)
  call void @tl_mem_register_tensor(ptr %call_tmp72)
  store ptr %call_tmp72, ptr %l_step4, align 8
  call void @tl_print_string(ptr @str_literal.108)
  %l_step473 = load ptr, ptr %l_step4, align 8
  call void @tl_tensor_print(ptr %l_step473)
  call void @tl_mem_exit_scope()
  ret void
}

define void @main() {
entry:
  %model = alloca ptr, align 16
  %max_len = alloca i64, align 16
  %d_model = alloca i64, align 16
  %vocab_size = alloca i64, align 16
  call void @tl_mem_enter_scope()
  call void @tl_arena_init(i64 256000)
  store i64 12, ptr %vocab_size, align 8
  store i64 128, ptr %d_model, align 8
  store i64 16, ptr %max_len, align 8
  %vocab_size1 = load i64, ptr %vocab_size, align 8
  %max_len2 = load i64, ptr %max_len, align 8
  %d_model3 = load i64, ptr %d_model, align 8
  %static_call = call ptr @tl_SeqModel_new(i64 %vocab_size1, i64 %max_len2, i64 %d_model3)
  store ptr %static_call, ptr %model, align 8
  call void @tl_print_string(ptr @str_literal.109)
  %model4 = load ptr, ptr %model, align 8
  %map = call ptr @tl_tensor_map_load(ptr @str_literal.110)
  %w = getelementptr inbounds nuw %SeqModel, ptr %model4, i32 0, i32 0
  %sub_ptr = load ptr, ptr %w, align 8
  %w5 = getelementptr inbounds nuw %Embedding, ptr %sub_ptr, i32 0, i32 0
  %t_val = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str)
  store ptr %t_val, ptr %w5, align 8
  %p = getelementptr inbounds nuw %SeqModel, ptr %model4, i32 0, i32 1
  %sub_ptr6 = load ptr, ptr %p, align 8
  %w7 = getelementptr inbounds nuw %PositionalEmbedding, ptr %sub_ptr6, i32 0, i32 0
  %t_val8 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.111)
  store ptr %t_val8, ptr %w7, align 8
  %b1 = getelementptr inbounds nuw %SeqModel, ptr %model4, i32 0, i32 2
  %sub_ptr9 = load ptr, ptr %b1, align 8
  %l1 = getelementptr inbounds nuw %Block, ptr %sub_ptr9, i32 0, i32 0
  %sub_ptr10 = load ptr, ptr %l1, align 8
  %w11 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr10, i32 0, i32 0
  %t_val12 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.112)
  store ptr %t_val12, ptr %w11, align 8
  %b = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr10, i32 0, i32 1
  %t_val13 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.113)
  store ptr %t_val13, ptr %b, align 8
  %a = getelementptr inbounds nuw %Block, ptr %sub_ptr9, i32 0, i32 1
  %sub_ptr14 = load ptr, ptr %a, align 8
  %q_proj = getelementptr inbounds nuw %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 0
  %sub_ptr15 = load ptr, ptr %q_proj, align 8
  %W = getelementptr inbounds nuw %Linear, ptr %sub_ptr15, i32 0, i32 0
  %t_val16 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.114)
  store ptr %t_val16, ptr %W, align 8
  %b17 = getelementptr inbounds nuw %Linear, ptr %sub_ptr15, i32 0, i32 1
  %t_val18 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.115)
  store ptr %t_val18, ptr %b17, align 8
  %k_proj = getelementptr inbounds nuw %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 1
  %sub_ptr19 = load ptr, ptr %k_proj, align 8
  %W20 = getelementptr inbounds nuw %Linear, ptr %sub_ptr19, i32 0, i32 0
  %t_val21 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.116)
  store ptr %t_val21, ptr %W20, align 8
  %b22 = getelementptr inbounds nuw %Linear, ptr %sub_ptr19, i32 0, i32 1
  %t_val23 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.117)
  store ptr %t_val23, ptr %b22, align 8
  %v_proj = getelementptr inbounds nuw %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 2
  %sub_ptr24 = load ptr, ptr %v_proj, align 8
  %W25 = getelementptr inbounds nuw %Linear, ptr %sub_ptr24, i32 0, i32 0
  %t_val26 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.118)
  store ptr %t_val26, ptr %W25, align 8
  %b27 = getelementptr inbounds nuw %Linear, ptr %sub_ptr24, i32 0, i32 1
  %t_val28 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.119)
  store ptr %t_val28, ptr %b27, align 8
  %o_proj = getelementptr inbounds nuw %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 3
  %sub_ptr29 = load ptr, ptr %o_proj, align 8
  %W30 = getelementptr inbounds nuw %Linear, ptr %sub_ptr29, i32 0, i32 0
  %t_val31 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.120)
  store ptr %t_val31, ptr %W30, align 8
  %b32 = getelementptr inbounds nuw %Linear, ptr %sub_ptr29, i32 0, i32 1
  %t_val33 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.121)
  store ptr %t_val33, ptr %b32, align 8
  %l2 = getelementptr inbounds nuw %Block, ptr %sub_ptr9, i32 0, i32 2
  %sub_ptr34 = load ptr, ptr %l2, align 8
  %w35 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr34, i32 0, i32 0
  %t_val36 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.122)
  store ptr %t_val36, ptr %w35, align 8
  %b37 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr34, i32 0, i32 1
  %t_val38 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.123)
  store ptr %t_val38, ptr %b37, align 8
  %m = getelementptr inbounds nuw %Block, ptr %sub_ptr9, i32 0, i32 3
  %sub_ptr39 = load ptr, ptr %m, align 8
  %f = getelementptr inbounds nuw %MLP, ptr %sub_ptr39, i32 0, i32 0
  %sub_ptr40 = load ptr, ptr %f, align 8
  %W41 = getelementptr inbounds nuw %Linear, ptr %sub_ptr40, i32 0, i32 0
  %t_val42 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.124)
  store ptr %t_val42, ptr %W41, align 8
  %b43 = getelementptr inbounds nuw %Linear, ptr %sub_ptr40, i32 0, i32 1
  %t_val44 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.125)
  store ptr %t_val44, ptr %b43, align 8
  %p45 = getelementptr inbounds nuw %MLP, ptr %sub_ptr39, i32 0, i32 1
  %sub_ptr46 = load ptr, ptr %p45, align 8
  %W47 = getelementptr inbounds nuw %Linear, ptr %sub_ptr46, i32 0, i32 0
  %t_val48 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.126)
  store ptr %t_val48, ptr %W47, align 8
  %b49 = getelementptr inbounds nuw %Linear, ptr %sub_ptr46, i32 0, i32 1
  %t_val50 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.127)
  store ptr %t_val50, ptr %b49, align 8
  %b2 = getelementptr inbounds nuw %SeqModel, ptr %model4, i32 0, i32 3
  %sub_ptr51 = load ptr, ptr %b2, align 8
  %l152 = getelementptr inbounds nuw %Block, ptr %sub_ptr51, i32 0, i32 0
  %sub_ptr53 = load ptr, ptr %l152, align 8
  %w54 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr53, i32 0, i32 0
  %t_val55 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.128)
  store ptr %t_val55, ptr %w54, align 8
  %b56 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr53, i32 0, i32 1
  %t_val57 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.129)
  store ptr %t_val57, ptr %b56, align 8
  %a58 = getelementptr inbounds nuw %Block, ptr %sub_ptr51, i32 0, i32 1
  %sub_ptr59 = load ptr, ptr %a58, align 8
  %q_proj60 = getelementptr inbounds nuw %CausalSelfAttention, ptr %sub_ptr59, i32 0, i32 0
  %sub_ptr61 = load ptr, ptr %q_proj60, align 8
  %W62 = getelementptr inbounds nuw %Linear, ptr %sub_ptr61, i32 0, i32 0
  %t_val63 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.130)
  store ptr %t_val63, ptr %W62, align 8
  %b64 = getelementptr inbounds nuw %Linear, ptr %sub_ptr61, i32 0, i32 1
  %t_val65 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.131)
  store ptr %t_val65, ptr %b64, align 8
  %k_proj66 = getelementptr inbounds nuw %CausalSelfAttention, ptr %sub_ptr59, i32 0, i32 1
  %sub_ptr67 = load ptr, ptr %k_proj66, align 8
  %W68 = getelementptr inbounds nuw %Linear, ptr %sub_ptr67, i32 0, i32 0
  %t_val69 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.132)
  store ptr %t_val69, ptr %W68, align 8
  %b70 = getelementptr inbounds nuw %Linear, ptr %sub_ptr67, i32 0, i32 1
  %t_val71 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.133)
  store ptr %t_val71, ptr %b70, align 8
  %v_proj72 = getelementptr inbounds nuw %CausalSelfAttention, ptr %sub_ptr59, i32 0, i32 2
  %sub_ptr73 = load ptr, ptr %v_proj72, align 8
  %W74 = getelementptr inbounds nuw %Linear, ptr %sub_ptr73, i32 0, i32 0
  %t_val75 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.134)
  store ptr %t_val75, ptr %W74, align 8
  %b76 = getelementptr inbounds nuw %Linear, ptr %sub_ptr73, i32 0, i32 1
  %t_val77 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.135)
  store ptr %t_val77, ptr %b76, align 8
  %o_proj78 = getelementptr inbounds nuw %CausalSelfAttention, ptr %sub_ptr59, i32 0, i32 3
  %sub_ptr79 = load ptr, ptr %o_proj78, align 8
  %W80 = getelementptr inbounds nuw %Linear, ptr %sub_ptr79, i32 0, i32 0
  %t_val81 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.136)
  store ptr %t_val81, ptr %W80, align 8
  %b82 = getelementptr inbounds nuw %Linear, ptr %sub_ptr79, i32 0, i32 1
  %t_val83 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.137)
  store ptr %t_val83, ptr %b82, align 8
  %l284 = getelementptr inbounds nuw %Block, ptr %sub_ptr51, i32 0, i32 2
  %sub_ptr85 = load ptr, ptr %l284, align 8
  %w86 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr85, i32 0, i32 0
  %t_val87 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.138)
  store ptr %t_val87, ptr %w86, align 8
  %b88 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr85, i32 0, i32 1
  %t_val89 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.139)
  store ptr %t_val89, ptr %b88, align 8
  %m90 = getelementptr inbounds nuw %Block, ptr %sub_ptr51, i32 0, i32 3
  %sub_ptr91 = load ptr, ptr %m90, align 8
  %f92 = getelementptr inbounds nuw %MLP, ptr %sub_ptr91, i32 0, i32 0
  %sub_ptr93 = load ptr, ptr %f92, align 8
  %W94 = getelementptr inbounds nuw %Linear, ptr %sub_ptr93, i32 0, i32 0
  %t_val95 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.140)
  store ptr %t_val95, ptr %W94, align 8
  %b96 = getelementptr inbounds nuw %Linear, ptr %sub_ptr93, i32 0, i32 1
  %t_val97 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.141)
  store ptr %t_val97, ptr %b96, align 8
  %p98 = getelementptr inbounds nuw %MLP, ptr %sub_ptr91, i32 0, i32 1
  %sub_ptr99 = load ptr, ptr %p98, align 8
  %W100 = getelementptr inbounds nuw %Linear, ptr %sub_ptr99, i32 0, i32 0
  %t_val101 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.142)
  store ptr %t_val101, ptr %W100, align 8
  %b102 = getelementptr inbounds nuw %Linear, ptr %sub_ptr99, i32 0, i32 1
  %t_val103 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.143)
  store ptr %t_val103, ptr %b102, align 8
  %l = getelementptr inbounds nuw %SeqModel, ptr %model4, i32 0, i32 4
  %sub_ptr104 = load ptr, ptr %l, align 8
  %w105 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr104, i32 0, i32 0
  %t_val106 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.144)
  store ptr %t_val106, ptr %w105, align 8
  %b107 = getelementptr inbounds nuw %LayerNorm, ptr %sub_ptr104, i32 0, i32 1
  %t_val108 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.145)
  store ptr %t_val108, ptr %b107, align 8
  %h = getelementptr inbounds nuw %SeqModel, ptr %model4, i32 0, i32 5
  %sub_ptr109 = load ptr, ptr %h, align 8
  %W110 = getelementptr inbounds nuw %Linear, ptr %sub_ptr109, i32 0, i32 0
  %t_val111 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.146)
  store ptr %t_val111, ptr %W110, align 8
  %b112 = getelementptr inbounds nuw %Linear, ptr %sub_ptr109, i32 0, i32 1
  %t_val113 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.147)
  store ptr %t_val113, ptr %b112, align 8
  call void @tl_tensor_map_free(ptr %map)
  call void @tl_print_string(ptr @str_literal.148)
  %model114 = load ptr, ptr %model, align 8
  call void @inference(ptr %model114)
  call void @tl_mem_exit_scope()
  ret void
}
