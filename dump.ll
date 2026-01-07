; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

%Linear = type { ptr, ptr }
%Embedding = type { ptr }
%LayerNorm = type { ptr, ptr }
%CausalSelfAttention = type { ptr, ptr, ptr, ptr, ptr }
%MLP = type { ptr, ptr }
%Block = type { ptr, ptr, ptr, ptr }
%GPT = type { ptr, ptr, ptr, ptr, ptr }

@str_literal = private unnamed_addr constant [65 x i8] c"Loading trained model weights from recall_weights.safetensors...\00", align 1
@str_literal.104 = private unnamed_addr constant [27 x i8] c"recall_weights.safetensors\00", align 1
@key_str = private unnamed_addr constant [6 x i8] c"wte.w\00", align 1
@key_str.105 = private unnamed_addr constant [6 x i8] c"wpe.w\00", align 1
@key_str.106 = private unnamed_addr constant [9 x i8] c"bg.ln1.w\00", align 1
@key_str.107 = private unnamed_addr constant [9 x i8] c"bg.ln1.b\00", align 1
@key_str.108 = private unnamed_addr constant [17 x i8] c"bg.attn.c_attn.W\00", align 1
@key_str.109 = private unnamed_addr constant [17 x i8] c"bg.attn.c_attn.b\00", align 1
@key_str.110 = private unnamed_addr constant [17 x i8] c"bg.attn.q_proj.W\00", align 1
@key_str.111 = private unnamed_addr constant [17 x i8] c"bg.attn.q_proj.b\00", align 1
@key_str.112 = private unnamed_addr constant [17 x i8] c"bg.attn.k_proj.W\00", align 1
@key_str.113 = private unnamed_addr constant [17 x i8] c"bg.attn.k_proj.b\00", align 1
@key_str.114 = private unnamed_addr constant [17 x i8] c"bg.attn.v_proj.W\00", align 1
@key_str.115 = private unnamed_addr constant [17 x i8] c"bg.attn.v_proj.b\00", align 1
@key_str.116 = private unnamed_addr constant [17 x i8] c"bg.attn.c_proj.W\00", align 1
@key_str.117 = private unnamed_addr constant [17 x i8] c"bg.attn.c_proj.b\00", align 1
@key_str.118 = private unnamed_addr constant [9 x i8] c"bg.ln2.w\00", align 1
@key_str.119 = private unnamed_addr constant [9 x i8] c"bg.ln2.b\00", align 1
@key_str.120 = private unnamed_addr constant [14 x i8] c"bg.mlp.c_fc.W\00", align 1
@key_str.121 = private unnamed_addr constant [14 x i8] c"bg.mlp.c_fc.b\00", align 1
@key_str.122 = private unnamed_addr constant [16 x i8] c"bg.mlp.c_proj.W\00", align 1
@key_str.123 = private unnamed_addr constant [16 x i8] c"bg.mlp.c_proj.b\00", align 1
@key_str.124 = private unnamed_addr constant [7 x i8] c"ln_f.w\00", align 1
@key_str.125 = private unnamed_addr constant [7 x i8] c"ln_f.b\00", align 1
@key_str.126 = private unnamed_addr constant [7 x i8] c"head.W\00", align 1
@key_str.127 = private unnamed_addr constant [7 x i8] c"head.b\00", align 1
@str_literal.128 = private unnamed_addr constant [27 x i8] c"Model loaded successfully.\00", align 1
@str_literal.129 = private unnamed_addr constant [39 x i8] c"Running Inference on Validation Set...\00", align 1
@str_literal.130 = private unnamed_addr constant [32 x i8] c"Format: Pred / Target -> Result\00", align 1
@str_literal.131 = private unnamed_addr constant [8 x i8] c"Sample:\00", align 1
@str_literal.132 = private unnamed_addr constant [6 x i8] c"Pred:\00", align 1
@str_literal.133 = private unnamed_addr constant [8 x i8] c"Target:\00", align 1
@str_literal.134 = private unnamed_addr constant [16 x i8] c"Result: Correct\00", align 1
@str_literal.135 = private unnamed_addr constant [14 x i8] c"Result: Wrong\00", align 1
@str_literal.136 = private unnamed_addr constant [20 x i8] c"Inference Complete.\00", align 1
@str_literal.137 = private unnamed_addr constant [10 x i8] c"Accuracy:\00", align 1
@str_literal.138 = private unnamed_addr constant [15 x i8] c"Correct out of\00", align 1

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

declare ptr @tl_tensor_to_f32(ptr)

declare ptr @tl_tensor_to_i64(ptr)

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
  %gbg = alloca ptr, align 16
  %gWg = alloca ptr, align 16
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
  store ptr %grad_res, ptr %gWg, align 8
  %s5 = load ptr, ptr %s, align 8
  %ptr_b = getelementptr inbounds %Linear, ptr %s5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res6 = call ptr @tl_tensor_grad(ptr %b)
  call void @tl_mem_register_tensor(ptr %grad_res6)
  call void @tl_mem_unregister(ptr %grad_res6)
  store ptr %grad_res6, ptr %gbg, align 8
  %s7 = load ptr, ptr %s, align 8
  %ptr_W8 = getelementptr inbounds %Linear, ptr %s7, i32 0, i32 0
  %s9 = load ptr, ptr %s, align 8
  %ptr_W10 = getelementptr inbounds %Linear, ptr %s9, i32 0, i32 0
  %W11 = load ptr, ptr %ptr_W10, align 8
  %gWg12 = load ptr, ptr %gWg, align 8
  %lr13 = load float, ptr %lr2, align 4
  store float %lr13, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %gWg12, ptr %scalar_tensor_rhs)
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
  %gbg20 = load ptr, ptr %gbg, align 8
  %lr21 = load float, ptr %lr2, align 4
  store float %lr21, ptr %scalar_data_rhs22, align 4
  %scalar_tensor_rhs24 = call ptr @tl_tensor_new(ptr %scalar_data_rhs22, i64 0, ptr %scalar_shape_rhs23)
  %binop_res25 = call ptr @tl_tensor_mul(ptr %gbg20, ptr %scalar_tensor_rhs24)
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
  %tensor_to_free = load ptr, ptr %gbg, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free34 = load ptr, ptr %gWg, align 8
  call void @tl_tensor_free(ptr %tensor_to_free34)
  call void @tl_mem_exit_scope()
  ret ptr %s32
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
  %ptr_w = getelementptr inbounds %LayerNorm, ptr %self4, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %binop_res = call ptr @tl_tensor_mul(ptr %x3, ptr %w)
  call void @tl_mem_register_tensor(ptr %binop_res)
  %self5 = load ptr, ptr %self1, align 8
  %ptr_b = getelementptr inbounds %LayerNorm, ptr %self5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %binop_res6 = call ptr @tl_tensor_add(ptr %binop_res, ptr %b)
  call void @tl_mem_register_tensor(ptr %binop_res6)
  call void @tl_mem_unregister(ptr %binop_res6)
  call void @tl_mem_exit_scope()
  ret ptr %binop_res6
}

define ptr @tl_LayerNorm_step(ptr %self, float %lr) {
entry:
  %scalar_shape_rhs23 = alloca i64, align 16
  %scalar_data_rhs22 = alloca float, align 16
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %gb = alloca ptr, align 16
  %gw = alloca ptr, align 16
  %s = alloca ptr, align 16
  %lr2 = alloca float, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store float %lr, ptr %lr2, align 4
  %self3 = load ptr, ptr %self1, align 8
  store ptr %self3, ptr %s, align 8
  %s4 = load ptr, ptr %s, align 8
  %ptr_w = getelementptr inbounds %LayerNorm, ptr %s4, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %w)
  call void @tl_mem_register_tensor(ptr %grad_res)
  call void @tl_mem_unregister(ptr %grad_res)
  store ptr %grad_res, ptr %gw, align 8
  %s5 = load ptr, ptr %s, align 8
  %ptr_b = getelementptr inbounds %LayerNorm, ptr %s5, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res6 = call ptr @tl_tensor_grad(ptr %b)
  call void @tl_mem_register_tensor(ptr %grad_res6)
  call void @tl_mem_unregister(ptr %grad_res6)
  store ptr %grad_res6, ptr %gb, align 8
  %s7 = load ptr, ptr %s, align 8
  %ptr_w8 = getelementptr inbounds %LayerNorm, ptr %s7, i32 0, i32 0
  %s9 = load ptr, ptr %s, align 8
  %ptr_w10 = getelementptr inbounds %LayerNorm, ptr %s9, i32 0, i32 0
  %w11 = load ptr, ptr %ptr_w10, align 8
  %gw12 = load ptr, ptr %gw, align 8
  %lr13 = load float, ptr %lr2, align 4
  store float %lr13, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %gw12, ptr %scalar_tensor_rhs)
  call void @tl_mem_register_tensor(ptr %binop_res)
  %binop_res14 = call ptr @tl_tensor_sub(ptr %w11, ptr %binop_res)
  call void @tl_mem_register_tensor(ptr %binop_res14)
  %detach_res = call ptr @tl_tensor_detach(ptr %binop_res14, i1 true)
  call void @tl_mem_register_tensor(ptr %detach_res)
  %old_field_val = load ptr, ptr %ptr_w8, align 8
  %cnt_free_diff = icmp ne ptr %old_field_val, %detach_res
  br i1 %cnt_free_diff, label %free_old_val, label %skip_free

free_old_val:                                     ; preds = %entry
  call void @tl_tensor_free(ptr %old_field_val)
  br label %skip_free

skip_free:                                        ; preds = %free_old_val, %entry
  store ptr %detach_res, ptr %ptr_w8, align 8
  call void @tl_mem_unregister(ptr %detach_res)
  %s15 = load ptr, ptr %s, align 8
  %ptr_b16 = getelementptr inbounds %LayerNorm, ptr %s15, i32 0, i32 1
  %s17 = load ptr, ptr %s, align 8
  %ptr_b18 = getelementptr inbounds %LayerNorm, ptr %s17, i32 0, i32 1
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
  %unreg_field_0 = getelementptr inbounds %LayerNorm, ptr %s32, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_1 = getelementptr inbounds %LayerNorm, ptr %s32, i32 0, i32 1
  %field_val33 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val33)
  %tensor_to_free = load ptr, ptr %gb, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free34 = load ptr, ptr %gw, align 8
  call void @tl_tensor_free(ptr %tensor_to_free34)
  call void @tl_mem_exit_scope()
  ret ptr %s32
}

define ptr @tl_CausalSelfAttention_new(i64 %d) {
entry:
  %d1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %d, ptr %d1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%CausalSelfAttention, ptr null, i32 1) to i64))
  %static_call = call ptr @tl_Linear_new(i64 1, i64 1)
  %init_field = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d2 = load i64, ptr %d1, align 8
  %d3 = load i64, ptr %d1, align 8
  %static_call4 = call ptr @tl_Linear_new(i64 %d2, i64 %d3)
  %init_field5 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call4, ptr %init_field5, align 8
  %d6 = load i64, ptr %d1, align 8
  %d7 = load i64, ptr %d1, align 8
  %static_call8 = call ptr @tl_Linear_new(i64 %d6, i64 %d7)
  %init_field9 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 2
  store ptr %static_call8, ptr %init_field9, align 8
  %d10 = load i64, ptr %d1, align 8
  %d11 = load i64, ptr %d1, align 8
  %static_call12 = call ptr @tl_Linear_new(i64 %d10, i64 %d11)
  %init_field13 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 3
  store ptr %static_call12, ptr %init_field13, align 8
  %d14 = load i64, ptr %d1, align 8
  %d15 = load i64, ptr %d1, align 8
  %static_call16 = call ptr @tl_Linear_new(i64 %d14, i64 %d15)
  %init_field17 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 4
  store ptr %static_call16, ptr %init_field17, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_018 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 0
  %field_val19 = load ptr, ptr %unreg_field_018, align 8
  call void @tl_mem_unregister(ptr %field_val19)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 1
  %field_val20 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val20)
  %unreg_field_121 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 1
  %field_val22 = load ptr, ptr %unreg_field_121, align 8
  call void @tl_mem_unregister(ptr %field_val22)
  %unreg_field_023 = getelementptr inbounds %Linear, ptr %field_val22, i32 0, i32 0
  %field_val24 = load ptr, ptr %unreg_field_023, align 8
  call void @tl_mem_unregister(ptr %field_val24)
  %unreg_field_125 = getelementptr inbounds %Linear, ptr %field_val22, i32 0, i32 1
  %field_val26 = load ptr, ptr %unreg_field_125, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 2
  %field_val27 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val27)
  %unreg_field_028 = getelementptr inbounds %Linear, ptr %field_val27, i32 0, i32 0
  %field_val29 = load ptr, ptr %unreg_field_028, align 8
  call void @tl_mem_unregister(ptr %field_val29)
  %unreg_field_130 = getelementptr inbounds %Linear, ptr %field_val27, i32 0, i32 1
  %field_val31 = load ptr, ptr %unreg_field_130, align 8
  call void @tl_mem_unregister(ptr %field_val31)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 3
  %field_val32 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val32)
  %unreg_field_033 = getelementptr inbounds %Linear, ptr %field_val32, i32 0, i32 0
  %field_val34 = load ptr, ptr %unreg_field_033, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  %unreg_field_135 = getelementptr inbounds %Linear, ptr %field_val32, i32 0, i32 1
  %field_val36 = load ptr, ptr %unreg_field_135, align 8
  call void @tl_mem_unregister(ptr %field_val36)
  %unreg_field_4 = getelementptr inbounds %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 4
  %field_val37 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val37)
  %unreg_field_038 = getelementptr inbounds %Linear, ptr %field_val37, i32 0, i32 0
  %field_val39 = load ptr, ptr %unreg_field_038, align 8
  call void @tl_mem_unregister(ptr %field_val39)
  %unreg_field_140 = getelementptr inbounds %Linear, ptr %field_val37, i32 0, i32 1
  %field_val41 = load ptr, ptr %unreg_field_140, align 8
  call void @tl_mem_unregister(ptr %field_val41)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_CausalSelfAttention_forward(ptr %self, ptr %x) {
entry:
  %y = alloca ptr, align 16
  %att18 = alloca ptr, align 16
  %att15 = alloca ptr, align 16
  %att = alloca ptr, align 16
  %scalar_shape_rhs = alloca i64, align 16
  %scalar_data_rhs = alloca float, align 16
  %k_t = alloca ptr, align 16
  %v = alloca ptr, align 16
  %k = alloca ptr, align 16
  %q = alloca ptr, align 16
  %x2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_q_proj = getelementptr inbounds %CausalSelfAttention, ptr %self3, i32 0, i32 1
  %q_proj = load ptr, ptr %ptr_q_proj, align 8
  %x4 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_Linear_forward(ptr %q_proj, ptr %x4)
  call void @tl_mem_register_tensor(ptr %call_method)
  call void @tl_mem_unregister(ptr %call_method)
  store ptr %call_method, ptr %q, align 8
  %self5 = load ptr, ptr %self1, align 8
  %ptr_k_proj = getelementptr inbounds %CausalSelfAttention, ptr %self5, i32 0, i32 2
  %k_proj = load ptr, ptr %ptr_k_proj, align 8
  %x6 = load ptr, ptr %x2, align 8
  %call_method7 = call ptr @tl_Linear_forward(ptr %k_proj, ptr %x6)
  call void @tl_mem_register_tensor(ptr %call_method7)
  call void @tl_mem_unregister(ptr %call_method7)
  store ptr %call_method7, ptr %k, align 8
  %self8 = load ptr, ptr %self1, align 8
  %ptr_v_proj = getelementptr inbounds %CausalSelfAttention, ptr %self8, i32 0, i32 3
  %v_proj = load ptr, ptr %ptr_v_proj, align 8
  %x9 = load ptr, ptr %x2, align 8
  %call_method10 = call ptr @tl_Linear_forward(ptr %v_proj, ptr %x9)
  call void @tl_mem_register_tensor(ptr %call_method10)
  call void @tl_mem_unregister(ptr %call_method10)
  store ptr %call_method10, ptr %v, align 8
  %k11 = load ptr, ptr %k, align 8
  %transpose_res = call ptr @tl_tensor_transpose(ptr %k11, i64 1, i64 2)
  call void @tl_mem_unregister(ptr %transpose_res)
  store ptr %transpose_res, ptr %k_t, align 8
  %q12 = load ptr, ptr %q, align 8
  %k_t13 = load ptr, ptr %k_t, align 8
  %matmul_res = call ptr @tl_tensor_matmul(ptr %q12, ptr %k_t13)
  store float 1.250000e-01, ptr %scalar_data_rhs, align 4
  %scalar_tensor_rhs = call ptr @tl_tensor_new(ptr %scalar_data_rhs, i64 0, ptr %scalar_shape_rhs)
  %binop_res = call ptr @tl_tensor_mul(ptr %matmul_res, ptr %scalar_tensor_rhs)
  call void @tl_mem_register_tensor(ptr %binop_res)
  call void @tl_mem_unregister(ptr %binop_res)
  store ptr %binop_res, ptr %att, align 8
  %att14 = load ptr, ptr %att, align 8
  %tril_res = call ptr @tl_tensor_tril(ptr %att14, i32 0)
  call void @tl_mem_unregister(ptr %tril_res)
  %old_shadowed = load ptr, ptr %att, align 8
  call void @tl_mem_unregister(ptr %old_shadowed)
  store ptr %tril_res, ptr %att15, align 8
  %att16 = load ptr, ptr %att15, align 8
  %softmax_res = call ptr @tl_tensor_softmax(ptr %att16, i64 2)
  call void @tl_mem_unregister(ptr %softmax_res)
  %old_shadowed17 = load ptr, ptr %att15, align 8
  call void @tl_mem_unregister(ptr %old_shadowed17)
  store ptr %softmax_res, ptr %att18, align 8
  %att19 = load ptr, ptr %att18, align 8
  %v20 = load ptr, ptr %v, align 8
  %matmul_res21 = call ptr @tl_tensor_matmul(ptr %att19, ptr %v20)
  call void @tl_mem_unregister(ptr %matmul_res21)
  store ptr %matmul_res21, ptr %y, align 8
  %self22 = load ptr, ptr %self1, align 8
  %ptr_c_proj = getelementptr inbounds %CausalSelfAttention, ptr %self22, i32 0, i32 4
  %c_proj = load ptr, ptr %ptr_c_proj, align 8
  %y23 = load ptr, ptr %y, align 8
  %call_method24 = call ptr @tl_Linear_forward(ptr %c_proj, ptr %y23)
  call void @tl_mem_register_tensor(ptr %call_method24)
  call void @tl_mem_unregister(ptr %call_method24)
  %tensor_to_free = load ptr, ptr %k, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free25 = load ptr, ptr %v, align 8
  call void @tl_tensor_free(ptr %tensor_to_free25)
  %tensor_to_free26 = load ptr, ptr %k_t, align 8
  call void @tl_tensor_free(ptr %tensor_to_free26)
  %tensor_to_free27 = load ptr, ptr %att18, align 8
  call void @tl_tensor_free(ptr %tensor_to_free27)
  %tensor_to_free28 = load ptr, ptr %q, align 8
  call void @tl_tensor_free(ptr %tensor_to_free28)
  %tensor_to_free29 = load ptr, ptr %y, align 8
  call void @tl_tensor_free(ptr %tensor_to_free29)
  call void @tl_mem_exit_scope()
  ret ptr %call_method24
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
  %ptr_q_proj = getelementptr inbounds %CausalSelfAttention, ptr %s4, i32 0, i32 1
  %s5 = load ptr, ptr %s, align 8
  %ptr_q_proj6 = getelementptr inbounds %CausalSelfAttention, ptr %s5, i32 0, i32 1
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
  %ptr_k_proj = getelementptr inbounds %CausalSelfAttention, ptr %s10, i32 0, i32 2
  %s11 = load ptr, ptr %s, align 8
  %ptr_k_proj12 = getelementptr inbounds %CausalSelfAttention, ptr %s11, i32 0, i32 2
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
  %ptr_v_proj = getelementptr inbounds %CausalSelfAttention, ptr %s23, i32 0, i32 3
  %s24 = load ptr, ptr %s, align 8
  %ptr_v_proj25 = getelementptr inbounds %CausalSelfAttention, ptr %s24, i32 0, i32 3
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
  %ptr_c_proj = getelementptr inbounds %CausalSelfAttention, ptr %s36, i32 0, i32 4
  %s37 = load ptr, ptr %s, align 8
  %ptr_c_proj38 = getelementptr inbounds %CausalSelfAttention, ptr %s37, i32 0, i32 4
  %c_proj = load ptr, ptr %ptr_c_proj38, align 8
  %lr39 = load float, ptr %lr2, align 4
  %call_method40 = call ptr @tl_Linear_step(ptr %c_proj, float %lr39)
  call void @tl_mem_register_struct(ptr %call_method40)
  %old_field_val41 = load ptr, ptr %ptr_c_proj, align 8
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
  store ptr %call_method40, ptr %ptr_c_proj, align 8
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
  %unreg_field_4 = getelementptr inbounds %CausalSelfAttention, ptr %s49, i32 0, i32 4
  %field_val69 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val69)
  %unreg_field_070 = getelementptr inbounds %Linear, ptr %field_val69, i32 0, i32 0
  %field_val71 = load ptr, ptr %unreg_field_070, align 8
  call void @tl_mem_unregister(ptr %field_val71)
  %unreg_field_172 = getelementptr inbounds %Linear, ptr %field_val69, i32 0, i32 1
  %field_val73 = load ptr, ptr %unreg_field_172, align 8
  call void @tl_mem_unregister(ptr %field_val73)
  call void @tl_mem_exit_scope()
  ret ptr %s49
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
  %ptr_c_proj = getelementptr inbounds %MLP, ptr %self3, i32 0, i32 1
  %c_proj = load ptr, ptr %ptr_c_proj, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_c_fc = getelementptr inbounds %MLP, ptr %self4, i32 0, i32 0
  %c_fc = load ptr, ptr %ptr_c_fc, align 8
  %x5 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_Linear_forward(ptr %c_fc, ptr %x5)
  call void @tl_mem_register_tensor(ptr %call_method)
  %relu_res = call ptr @tl_tensor_relu(ptr %call_method)
  %call_method6 = call ptr @tl_Linear_forward(ptr %c_proj, ptr %relu_res)
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
  %ptr_c_fc = getelementptr inbounds %MLP, ptr %s4, i32 0, i32 0
  %s5 = load ptr, ptr %s, align 8
  %ptr_c_fc6 = getelementptr inbounds %MLP, ptr %s5, i32 0, i32 0
  %c_fc = load ptr, ptr %ptr_c_fc6, align 8
  %lr7 = load float, ptr %lr2, align 4
  %call_method = call ptr @tl_Linear_step(ptr %c_fc, float %lr7)
  call void @tl_mem_register_struct(ptr %call_method)
  %old_field_val = load ptr, ptr %ptr_c_fc, align 8
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
  store ptr %call_method, ptr %ptr_c_fc, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s10 = load ptr, ptr %s, align 8
  %ptr_c_proj = getelementptr inbounds %MLP, ptr %s10, i32 0, i32 1
  %s11 = load ptr, ptr %s, align 8
  %ptr_c_proj12 = getelementptr inbounds %MLP, ptr %s11, i32 0, i32 1
  %c_proj = load ptr, ptr %ptr_c_proj12, align 8
  %lr13 = load float, ptr %lr2, align 4
  %call_method14 = call ptr @tl_Linear_step(ptr %c_proj, float %lr13)
  call void @tl_mem_register_struct(ptr %call_method14)
  %old_field_val15 = load ptr, ptr %ptr_c_proj, align 8
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
  store ptr %call_method14, ptr %ptr_c_proj, align 8
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
  %unreg_field_4 = getelementptr inbounds %CausalSelfAttention, ptr %field_val16, i32 0, i32 4
  %field_val39 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val39)
  %unreg_field_040 = getelementptr inbounds %Linear, ptr %field_val39, i32 0, i32 0
  %field_val41 = load ptr, ptr %unreg_field_040, align 8
  call void @tl_mem_unregister(ptr %field_val41)
  %unreg_field_142 = getelementptr inbounds %Linear, ptr %field_val39, i32 0, i32 1
  %field_val43 = load ptr, ptr %unreg_field_142, align 8
  call void @tl_mem_unregister(ptr %field_val43)
  %unreg_field_244 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 2
  %field_val45 = load ptr, ptr %unreg_field_244, align 8
  call void @tl_mem_unregister(ptr %field_val45)
  %unreg_field_046 = getelementptr inbounds %LayerNorm, ptr %field_val45, i32 0, i32 0
  %field_val47 = load ptr, ptr %unreg_field_046, align 8
  call void @tl_mem_unregister(ptr %field_val47)
  %unreg_field_148 = getelementptr inbounds %LayerNorm, ptr %field_val45, i32 0, i32 1
  %field_val49 = load ptr, ptr %unreg_field_148, align 8
  call void @tl_mem_unregister(ptr %field_val49)
  %unreg_field_350 = getelementptr inbounds %Block, ptr %struct_malloc, i32 0, i32 3
  %field_val51 = load ptr, ptr %unreg_field_350, align 8
  call void @tl_mem_unregister(ptr %field_val51)
  %unreg_field_052 = getelementptr inbounds %MLP, ptr %field_val51, i32 0, i32 0
  %field_val53 = load ptr, ptr %unreg_field_052, align 8
  call void @tl_mem_unregister(ptr %field_val53)
  %unreg_field_054 = getelementptr inbounds %Linear, ptr %field_val53, i32 0, i32 0
  %field_val55 = load ptr, ptr %unreg_field_054, align 8
  call void @tl_mem_unregister(ptr %field_val55)
  %unreg_field_156 = getelementptr inbounds %Linear, ptr %field_val53, i32 0, i32 1
  %field_val57 = load ptr, ptr %unreg_field_156, align 8
  call void @tl_mem_unregister(ptr %field_val57)
  %unreg_field_158 = getelementptr inbounds %MLP, ptr %field_val51, i32 0, i32 1
  %field_val59 = load ptr, ptr %unreg_field_158, align 8
  call void @tl_mem_unregister(ptr %field_val59)
  %unreg_field_060 = getelementptr inbounds %Linear, ptr %field_val59, i32 0, i32 0
  %field_val61 = load ptr, ptr %unreg_field_060, align 8
  call void @tl_mem_unregister(ptr %field_val61)
  %unreg_field_162 = getelementptr inbounds %Linear, ptr %field_val59, i32 0, i32 1
  %field_val63 = load ptr, ptr %unreg_field_162, align 8
  call void @tl_mem_unregister(ptr %field_val63)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_Block_forward(ptr %self, ptr %x) {
entry:
  %x16 = alloca ptr, align 16
  %x8 = alloca ptr, align 16
  %x2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %x3 = load ptr, ptr %x2, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_attn = getelementptr inbounds %Block, ptr %self4, i32 0, i32 1
  %attn = load ptr, ptr %ptr_attn, align 8
  %self5 = load ptr, ptr %self1, align 8
  %ptr_ln1 = getelementptr inbounds %Block, ptr %self5, i32 0, i32 0
  %ln1 = load ptr, ptr %ptr_ln1, align 8
  %x6 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_LayerNorm_forward(ptr %ln1, ptr %x6)
  call void @tl_mem_register_tensor(ptr %call_method)
  %call_method7 = call ptr @tl_CausalSelfAttention_forward(ptr %attn, ptr %call_method)
  call void @tl_tensor_free(ptr %call_method)
  call void @tl_mem_register_tensor(ptr %call_method7)
  %binop_res = call ptr @tl_tensor_add(ptr %x3, ptr %call_method7)
  call void @tl_mem_register_tensor(ptr %binop_res)
  call void @tl_mem_unregister(ptr %binop_res)
  store ptr %binop_res, ptr %x8, align 8
  %x9 = load ptr, ptr %x8, align 8
  %self10 = load ptr, ptr %self1, align 8
  %ptr_mlp = getelementptr inbounds %Block, ptr %self10, i32 0, i32 3
  %mlp = load ptr, ptr %ptr_mlp, align 8
  %self11 = load ptr, ptr %self1, align 8
  %ptr_ln2 = getelementptr inbounds %Block, ptr %self11, i32 0, i32 2
  %ln2 = load ptr, ptr %ptr_ln2, align 8
  %x12 = load ptr, ptr %x8, align 8
  %call_method13 = call ptr @tl_LayerNorm_forward(ptr %ln2, ptr %x12)
  call void @tl_mem_register_tensor(ptr %call_method13)
  %call_method14 = call ptr @tl_MLP_forward(ptr %mlp, ptr %call_method13)
  call void @tl_tensor_free(ptr %call_method13)
  call void @tl_mem_register_tensor(ptr %call_method14)
  %binop_res15 = call ptr @tl_tensor_add(ptr %x9, ptr %call_method14)
  call void @tl_mem_register_tensor(ptr %binop_res15)
  call void @tl_mem_unregister(ptr %binop_res15)
  %old_shadowed = load ptr, ptr %x8, align 8
  call void @tl_mem_unregister(ptr %old_shadowed)
  store ptr %binop_res15, ptr %x16, align 8
  %x17 = load ptr, ptr %x16, align 8
  call void @tl_mem_unregister(ptr %x17)
  call void @tl_mem_exit_scope()
  ret ptr %x17
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
  %ptr_ln1 = getelementptr inbounds %Block, ptr %s4, i32 0, i32 0
  %s5 = load ptr, ptr %s, align 8
  %ptr_ln16 = getelementptr inbounds %Block, ptr %s5, i32 0, i32 0
  %ln1 = load ptr, ptr %ptr_ln16, align 8
  %lr7 = load float, ptr %lr2, align 4
  %call_method = call ptr @tl_LayerNorm_step(ptr %ln1, float %lr7)
  call void @tl_mem_register_struct(ptr %call_method)
  %old_field_val = load ptr, ptr %ptr_ln1, align 8
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
  store ptr %call_method, ptr %ptr_ln1, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s10 = load ptr, ptr %s, align 8
  %ptr_attn = getelementptr inbounds %Block, ptr %s10, i32 0, i32 1
  %s11 = load ptr, ptr %s, align 8
  %ptr_attn12 = getelementptr inbounds %Block, ptr %s11, i32 0, i32 1
  %attn = load ptr, ptr %ptr_attn12, align 8
  %lr13 = load float, ptr %lr2, align 4
  %call_method14 = call ptr @tl_CausalSelfAttention_step(ptr %attn, float %lr13)
  call void @tl_mem_register_struct(ptr %call_method14)
  %old_field_val15 = load ptr, ptr %ptr_attn, align 8
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
  %field_gep43 = getelementptr inbounds %CausalSelfAttention, ptr %old_field_val15, i32 0, i32 4
  %field_load44 = load ptr, ptr %field_gep43, align 8
  %field_gep45 = getelementptr inbounds %Linear, ptr %field_load44, i32 0, i32 0
  %field_load46 = load ptr, ptr %field_gep45, align 8
  call void @tl_tensor_free(ptr %field_load46)
  %field_gep47 = getelementptr inbounds %Linear, ptr %field_load44, i32 0, i32 1
  %field_load48 = load ptr, ptr %field_gep47, align 8
  call void @tl_tensor_free(ptr %field_load48)
  br label %skip_free18

skip_free18:                                      ; preds = %free_old_val17, %skip_free
  store ptr %call_method14, ptr %ptr_attn, align 8
  call void @tl_mem_unregister(ptr %call_method14)
  %s49 = load ptr, ptr %s, align 8
  %ptr_ln2 = getelementptr inbounds %Block, ptr %s49, i32 0, i32 2
  %s50 = load ptr, ptr %s, align 8
  %ptr_ln251 = getelementptr inbounds %Block, ptr %s50, i32 0, i32 2
  %ln2 = load ptr, ptr %ptr_ln251, align 8
  %lr52 = load float, ptr %lr2, align 4
  %call_method53 = call ptr @tl_LayerNorm_step(ptr %ln2, float %lr52)
  call void @tl_mem_register_struct(ptr %call_method53)
  %old_field_val54 = load ptr, ptr %ptr_ln2, align 8
  %cnt_free_diff55 = icmp ne ptr %old_field_val54, %call_method53
  br i1 %cnt_free_diff55, label %free_old_val56, label %skip_free57

free_old_val56:                                   ; preds = %skip_free18
  %field_gep58 = getelementptr inbounds %LayerNorm, ptr %old_field_val54, i32 0, i32 0
  %field_load59 = load ptr, ptr %field_gep58, align 8
  call void @tl_tensor_free(ptr %field_load59)
  %field_gep60 = getelementptr inbounds %LayerNorm, ptr %old_field_val54, i32 0, i32 1
  %field_load61 = load ptr, ptr %field_gep60, align 8
  call void @tl_tensor_free(ptr %field_load61)
  br label %skip_free57

skip_free57:                                      ; preds = %free_old_val56, %skip_free18
  store ptr %call_method53, ptr %ptr_ln2, align 8
  call void @tl_mem_unregister(ptr %call_method53)
  %s62 = load ptr, ptr %s, align 8
  %ptr_mlp = getelementptr inbounds %Block, ptr %s62, i32 0, i32 3
  %s63 = load ptr, ptr %s, align 8
  %ptr_mlp64 = getelementptr inbounds %Block, ptr %s63, i32 0, i32 3
  %mlp = load ptr, ptr %ptr_mlp64, align 8
  %lr65 = load float, ptr %lr2, align 4
  %call_method66 = call ptr @tl_MLP_step(ptr %mlp, float %lr65)
  call void @tl_mem_register_struct(ptr %call_method66)
  %old_field_val67 = load ptr, ptr %ptr_mlp, align 8
  %cnt_free_diff68 = icmp ne ptr %old_field_val67, %call_method66
  br i1 %cnt_free_diff68, label %free_old_val69, label %skip_free70

free_old_val69:                                   ; preds = %skip_free57
  %field_gep71 = getelementptr inbounds %MLP, ptr %old_field_val67, i32 0, i32 0
  %field_load72 = load ptr, ptr %field_gep71, align 8
  %field_gep73 = getelementptr inbounds %Linear, ptr %field_load72, i32 0, i32 0
  %field_load74 = load ptr, ptr %field_gep73, align 8
  call void @tl_tensor_free(ptr %field_load74)
  %field_gep75 = getelementptr inbounds %Linear, ptr %field_load72, i32 0, i32 1
  %field_load76 = load ptr, ptr %field_gep75, align 8
  call void @tl_tensor_free(ptr %field_load76)
  %field_gep77 = getelementptr inbounds %MLP, ptr %old_field_val67, i32 0, i32 1
  %field_load78 = load ptr, ptr %field_gep77, align 8
  %field_gep79 = getelementptr inbounds %Linear, ptr %field_load78, i32 0, i32 0
  %field_load80 = load ptr, ptr %field_gep79, align 8
  call void @tl_tensor_free(ptr %field_load80)
  %field_gep81 = getelementptr inbounds %Linear, ptr %field_load78, i32 0, i32 1
  %field_load82 = load ptr, ptr %field_gep81, align 8
  call void @tl_tensor_free(ptr %field_load82)
  br label %skip_free70

skip_free70:                                      ; preds = %free_old_val69, %skip_free57
  store ptr %call_method66, ptr %ptr_mlp, align 8
  call void @tl_mem_unregister(ptr %call_method66)
  %s83 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s83)
  %unreg_field_0 = getelementptr inbounds %Block, ptr %s83, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_084 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 0
  %field_val85 = load ptr, ptr %unreg_field_084, align 8
  call void @tl_mem_unregister(ptr %field_val85)
  %unreg_field_1 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 1
  %field_val86 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val86)
  %unreg_field_187 = getelementptr inbounds %Block, ptr %s83, i32 0, i32 1
  %field_val88 = load ptr, ptr %unreg_field_187, align 8
  call void @tl_mem_unregister(ptr %field_val88)
  %unreg_field_089 = getelementptr inbounds %CausalSelfAttention, ptr %field_val88, i32 0, i32 0
  %field_val90 = load ptr, ptr %unreg_field_089, align 8
  call void @tl_mem_unregister(ptr %field_val90)
  %unreg_field_091 = getelementptr inbounds %Linear, ptr %field_val90, i32 0, i32 0
  %field_val92 = load ptr, ptr %unreg_field_091, align 8
  call void @tl_mem_unregister(ptr %field_val92)
  %unreg_field_193 = getelementptr inbounds %Linear, ptr %field_val90, i32 0, i32 1
  %field_val94 = load ptr, ptr %unreg_field_193, align 8
  call void @tl_mem_unregister(ptr %field_val94)
  %unreg_field_195 = getelementptr inbounds %CausalSelfAttention, ptr %field_val88, i32 0, i32 1
  %field_val96 = load ptr, ptr %unreg_field_195, align 8
  call void @tl_mem_unregister(ptr %field_val96)
  %unreg_field_097 = getelementptr inbounds %Linear, ptr %field_val96, i32 0, i32 0
  %field_val98 = load ptr, ptr %unreg_field_097, align 8
  call void @tl_mem_unregister(ptr %field_val98)
  %unreg_field_199 = getelementptr inbounds %Linear, ptr %field_val96, i32 0, i32 1
  %field_val100 = load ptr, ptr %unreg_field_199, align 8
  call void @tl_mem_unregister(ptr %field_val100)
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %field_val88, i32 0, i32 2
  %field_val101 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val101)
  %unreg_field_0102 = getelementptr inbounds %Linear, ptr %field_val101, i32 0, i32 0
  %field_val103 = load ptr, ptr %unreg_field_0102, align 8
  call void @tl_mem_unregister(ptr %field_val103)
  %unreg_field_1104 = getelementptr inbounds %Linear, ptr %field_val101, i32 0, i32 1
  %field_val105 = load ptr, ptr %unreg_field_1104, align 8
  call void @tl_mem_unregister(ptr %field_val105)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val88, i32 0, i32 3
  %field_val106 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val106)
  %unreg_field_0107 = getelementptr inbounds %Linear, ptr %field_val106, i32 0, i32 0
  %field_val108 = load ptr, ptr %unreg_field_0107, align 8
  call void @tl_mem_unregister(ptr %field_val108)
  %unreg_field_1109 = getelementptr inbounds %Linear, ptr %field_val106, i32 0, i32 1
  %field_val110 = load ptr, ptr %unreg_field_1109, align 8
  call void @tl_mem_unregister(ptr %field_val110)
  %unreg_field_4 = getelementptr inbounds %CausalSelfAttention, ptr %field_val88, i32 0, i32 4
  %field_val111 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val111)
  %unreg_field_0112 = getelementptr inbounds %Linear, ptr %field_val111, i32 0, i32 0
  %field_val113 = load ptr, ptr %unreg_field_0112, align 8
  call void @tl_mem_unregister(ptr %field_val113)
  %unreg_field_1114 = getelementptr inbounds %Linear, ptr %field_val111, i32 0, i32 1
  %field_val115 = load ptr, ptr %unreg_field_1114, align 8
  call void @tl_mem_unregister(ptr %field_val115)
  %unreg_field_2116 = getelementptr inbounds %Block, ptr %s83, i32 0, i32 2
  %field_val117 = load ptr, ptr %unreg_field_2116, align 8
  call void @tl_mem_unregister(ptr %field_val117)
  %unreg_field_0118 = getelementptr inbounds %LayerNorm, ptr %field_val117, i32 0, i32 0
  %field_val119 = load ptr, ptr %unreg_field_0118, align 8
  call void @tl_mem_unregister(ptr %field_val119)
  %unreg_field_1120 = getelementptr inbounds %LayerNorm, ptr %field_val117, i32 0, i32 1
  %field_val121 = load ptr, ptr %unreg_field_1120, align 8
  call void @tl_mem_unregister(ptr %field_val121)
  %unreg_field_3122 = getelementptr inbounds %Block, ptr %s83, i32 0, i32 3
  %field_val123 = load ptr, ptr %unreg_field_3122, align 8
  call void @tl_mem_unregister(ptr %field_val123)
  %unreg_field_0124 = getelementptr inbounds %MLP, ptr %field_val123, i32 0, i32 0
  %field_val125 = load ptr, ptr %unreg_field_0124, align 8
  call void @tl_mem_unregister(ptr %field_val125)
  %unreg_field_0126 = getelementptr inbounds %Linear, ptr %field_val125, i32 0, i32 0
  %field_val127 = load ptr, ptr %unreg_field_0126, align 8
  call void @tl_mem_unregister(ptr %field_val127)
  %unreg_field_1128 = getelementptr inbounds %Linear, ptr %field_val125, i32 0, i32 1
  %field_val129 = load ptr, ptr %unreg_field_1128, align 8
  call void @tl_mem_unregister(ptr %field_val129)
  %unreg_field_1130 = getelementptr inbounds %MLP, ptr %field_val123, i32 0, i32 1
  %field_val131 = load ptr, ptr %unreg_field_1130, align 8
  call void @tl_mem_unregister(ptr %field_val131)
  %unreg_field_0132 = getelementptr inbounds %Linear, ptr %field_val131, i32 0, i32 0
  %field_val133 = load ptr, ptr %unreg_field_0132, align 8
  call void @tl_mem_unregister(ptr %field_val133)
  %unreg_field_1134 = getelementptr inbounds %Linear, ptr %field_val131, i32 0, i32 1
  %field_val135 = load ptr, ptr %unreg_field_1134, align 8
  call void @tl_mem_unregister(ptr %field_val135)
  call void @tl_mem_exit_scope()
  ret ptr %s83
}

define ptr @tl_GPT_new(i64 %vocab_size, i64 %d_model, i64 %max_len) {
entry:
  %max_len3 = alloca i64, align 16
  %d_model2 = alloca i64, align 16
  %vocab_size1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %vocab_size, ptr %vocab_size1, align 8
  store i64 %d_model, ptr %d_model2, align 8
  store i64 %max_len, ptr %max_len3, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%GPT, ptr null, i32 1) to i64))
  %vocab_size4 = load i64, ptr %vocab_size1, align 8
  %d_model5 = load i64, ptr %d_model2, align 8
  %static_call = call ptr @tl_Embedding_new(i64 %vocab_size4, i64 %d_model5)
  %init_field = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %max_len6 = load i64, ptr %max_len3, align 8
  %d_model7 = load i64, ptr %d_model2, align 8
  %static_call8 = call ptr @tl_Embedding_new(i64 %max_len6, i64 %d_model7)
  %init_field9 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call8, ptr %init_field9, align 8
  %d_model10 = load i64, ptr %d_model2, align 8
  %static_call11 = call ptr @tl_Block_new(i64 %d_model10)
  %init_field12 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 2
  store ptr %static_call11, ptr %init_field12, align 8
  %d_model13 = load i64, ptr %d_model2, align 8
  %static_call14 = call ptr @tl_LayerNorm_new(i64 %d_model13)
  %init_field15 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 3
  store ptr %static_call14, ptr %init_field15, align 8
  %d_model16 = load i64, ptr %d_model2, align 8
  %vocab_size17 = load i64, ptr %vocab_size1, align 8
  %static_call18 = call ptr @tl_Linear_new(i64 %d_model16, i64 %vocab_size17)
  %init_field19 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 4
  store ptr %static_call18, ptr %init_field19, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_020 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val21 = load ptr, ptr %unreg_field_020, align 8
  call void @tl_mem_unregister(ptr %field_val21)
  %unreg_field_1 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 1
  %field_val22 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val22)
  %unreg_field_023 = getelementptr inbounds %Embedding, ptr %field_val22, i32 0, i32 0
  %field_val24 = load ptr, ptr %unreg_field_023, align 8
  call void @tl_mem_unregister(ptr %field_val24)
  %unreg_field_2 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 2
  %field_val25 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_026 = getelementptr inbounds %Block, ptr %field_val25, i32 0, i32 0
  %field_val27 = load ptr, ptr %unreg_field_026, align 8
  call void @tl_mem_unregister(ptr %field_val27)
  %unreg_field_028 = getelementptr inbounds %LayerNorm, ptr %field_val27, i32 0, i32 0
  %field_val29 = load ptr, ptr %unreg_field_028, align 8
  call void @tl_mem_unregister(ptr %field_val29)
  %unreg_field_130 = getelementptr inbounds %LayerNorm, ptr %field_val27, i32 0, i32 1
  %field_val31 = load ptr, ptr %unreg_field_130, align 8
  call void @tl_mem_unregister(ptr %field_val31)
  %unreg_field_132 = getelementptr inbounds %Block, ptr %field_val25, i32 0, i32 1
  %field_val33 = load ptr, ptr %unreg_field_132, align 8
  call void @tl_mem_unregister(ptr %field_val33)
  %unreg_field_034 = getelementptr inbounds %CausalSelfAttention, ptr %field_val33, i32 0, i32 0
  %field_val35 = load ptr, ptr %unreg_field_034, align 8
  call void @tl_mem_unregister(ptr %field_val35)
  %unreg_field_036 = getelementptr inbounds %Linear, ptr %field_val35, i32 0, i32 0
  %field_val37 = load ptr, ptr %unreg_field_036, align 8
  call void @tl_mem_unregister(ptr %field_val37)
  %unreg_field_138 = getelementptr inbounds %Linear, ptr %field_val35, i32 0, i32 1
  %field_val39 = load ptr, ptr %unreg_field_138, align 8
  call void @tl_mem_unregister(ptr %field_val39)
  %unreg_field_140 = getelementptr inbounds %CausalSelfAttention, ptr %field_val33, i32 0, i32 1
  %field_val41 = load ptr, ptr %unreg_field_140, align 8
  call void @tl_mem_unregister(ptr %field_val41)
  %unreg_field_042 = getelementptr inbounds %Linear, ptr %field_val41, i32 0, i32 0
  %field_val43 = load ptr, ptr %unreg_field_042, align 8
  call void @tl_mem_unregister(ptr %field_val43)
  %unreg_field_144 = getelementptr inbounds %Linear, ptr %field_val41, i32 0, i32 1
  %field_val45 = load ptr, ptr %unreg_field_144, align 8
  call void @tl_mem_unregister(ptr %field_val45)
  %unreg_field_246 = getelementptr inbounds %CausalSelfAttention, ptr %field_val33, i32 0, i32 2
  %field_val47 = load ptr, ptr %unreg_field_246, align 8
  call void @tl_mem_unregister(ptr %field_val47)
  %unreg_field_048 = getelementptr inbounds %Linear, ptr %field_val47, i32 0, i32 0
  %field_val49 = load ptr, ptr %unreg_field_048, align 8
  call void @tl_mem_unregister(ptr %field_val49)
  %unreg_field_150 = getelementptr inbounds %Linear, ptr %field_val47, i32 0, i32 1
  %field_val51 = load ptr, ptr %unreg_field_150, align 8
  call void @tl_mem_unregister(ptr %field_val51)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val33, i32 0, i32 3
  %field_val52 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val52)
  %unreg_field_053 = getelementptr inbounds %Linear, ptr %field_val52, i32 0, i32 0
  %field_val54 = load ptr, ptr %unreg_field_053, align 8
  call void @tl_mem_unregister(ptr %field_val54)
  %unreg_field_155 = getelementptr inbounds %Linear, ptr %field_val52, i32 0, i32 1
  %field_val56 = load ptr, ptr %unreg_field_155, align 8
  call void @tl_mem_unregister(ptr %field_val56)
  %unreg_field_4 = getelementptr inbounds %CausalSelfAttention, ptr %field_val33, i32 0, i32 4
  %field_val57 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val57)
  %unreg_field_058 = getelementptr inbounds %Linear, ptr %field_val57, i32 0, i32 0
  %field_val59 = load ptr, ptr %unreg_field_058, align 8
  call void @tl_mem_unregister(ptr %field_val59)
  %unreg_field_160 = getelementptr inbounds %Linear, ptr %field_val57, i32 0, i32 1
  %field_val61 = load ptr, ptr %unreg_field_160, align 8
  call void @tl_mem_unregister(ptr %field_val61)
  %unreg_field_262 = getelementptr inbounds %Block, ptr %field_val25, i32 0, i32 2
  %field_val63 = load ptr, ptr %unreg_field_262, align 8
  call void @tl_mem_unregister(ptr %field_val63)
  %unreg_field_064 = getelementptr inbounds %LayerNorm, ptr %field_val63, i32 0, i32 0
  %field_val65 = load ptr, ptr %unreg_field_064, align 8
  call void @tl_mem_unregister(ptr %field_val65)
  %unreg_field_166 = getelementptr inbounds %LayerNorm, ptr %field_val63, i32 0, i32 1
  %field_val67 = load ptr, ptr %unreg_field_166, align 8
  call void @tl_mem_unregister(ptr %field_val67)
  %unreg_field_368 = getelementptr inbounds %Block, ptr %field_val25, i32 0, i32 3
  %field_val69 = load ptr, ptr %unreg_field_368, align 8
  call void @tl_mem_unregister(ptr %field_val69)
  %unreg_field_070 = getelementptr inbounds %MLP, ptr %field_val69, i32 0, i32 0
  %field_val71 = load ptr, ptr %unreg_field_070, align 8
  call void @tl_mem_unregister(ptr %field_val71)
  %unreg_field_072 = getelementptr inbounds %Linear, ptr %field_val71, i32 0, i32 0
  %field_val73 = load ptr, ptr %unreg_field_072, align 8
  call void @tl_mem_unregister(ptr %field_val73)
  %unreg_field_174 = getelementptr inbounds %Linear, ptr %field_val71, i32 0, i32 1
  %field_val75 = load ptr, ptr %unreg_field_174, align 8
  call void @tl_mem_unregister(ptr %field_val75)
  %unreg_field_176 = getelementptr inbounds %MLP, ptr %field_val69, i32 0, i32 1
  %field_val77 = load ptr, ptr %unreg_field_176, align 8
  call void @tl_mem_unregister(ptr %field_val77)
  %unreg_field_078 = getelementptr inbounds %Linear, ptr %field_val77, i32 0, i32 0
  %field_val79 = load ptr, ptr %unreg_field_078, align 8
  call void @tl_mem_unregister(ptr %field_val79)
  %unreg_field_180 = getelementptr inbounds %Linear, ptr %field_val77, i32 0, i32 1
  %field_val81 = load ptr, ptr %unreg_field_180, align 8
  call void @tl_mem_unregister(ptr %field_val81)
  %unreg_field_382 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 3
  %field_val83 = load ptr, ptr %unreg_field_382, align 8
  call void @tl_mem_unregister(ptr %field_val83)
  %unreg_field_084 = getelementptr inbounds %LayerNorm, ptr %field_val83, i32 0, i32 0
  %field_val85 = load ptr, ptr %unreg_field_084, align 8
  call void @tl_mem_unregister(ptr %field_val85)
  %unreg_field_186 = getelementptr inbounds %LayerNorm, ptr %field_val83, i32 0, i32 1
  %field_val87 = load ptr, ptr %unreg_field_186, align 8
  call void @tl_mem_unregister(ptr %field_val87)
  %unreg_field_488 = getelementptr inbounds %GPT, ptr %struct_malloc, i32 0, i32 4
  %field_val89 = load ptr, ptr %unreg_field_488, align 8
  call void @tl_mem_unregister(ptr %field_val89)
  %unreg_field_090 = getelementptr inbounds %Linear, ptr %field_val89, i32 0, i32 0
  %field_val91 = load ptr, ptr %unreg_field_090, align 8
  call void @tl_mem_unregister(ptr %field_val91)
  %unreg_field_192 = getelementptr inbounds %Linear, ptr %field_val89, i32 0, i32 1
  %field_val93 = load ptr, ptr %unreg_field_192, align 8
  call void @tl_mem_unregister(ptr %field_val93)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_GPT_forward(ptr %self, ptr %idx) {
entry:
  %logits = alloca ptr, align 16
  %x31 = alloca ptr, align 16
  %x26 = alloca ptr, align 16
  %x = alloca ptr, align 16
  %pos_emb = alloca ptr, align 16
  %tok_emb = alloca ptr, align 16
  %pos = alloca ptr, align 16
  %dims_alloca = alloca [2 x i64], align 8
  %pos_arr = alloca ptr, align 16
  %T = alloca i64, align 16
  %idx2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store ptr %idx, ptr %idx2, align 8
  store i64 13, ptr %T, align 8
  %temp_data_alloc = call ptr @tl_alloc_tmp(i64 52)
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
  %data_elem14 = getelementptr inbounds float, ptr %temp_data_alloc, i64 12
  store float 1.200000e+01, ptr %data_elem14, align 4
  %shape_elem = getelementptr inbounds i64, ptr %temp_shape_alloc, i64 0
  store i64 13, ptr %shape_elem, align 8
  %new_const_tensor = call ptr @tl_tensor_new(ptr %temp_data_alloc, i64 1, ptr %temp_shape_alloc)
  call void @tl_free_tmp(ptr %temp_data_alloc)
  call void @tl_free_tmp(ptr %temp_shape_alloc)
  call void @tl_mem_unregister(ptr %new_const_tensor)
  store ptr %new_const_tensor, ptr %pos_arr, align 8
  %pos_arr15 = load ptr, ptr %pos_arr, align 8
  %dim_ptr_0 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0, align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 13, ptr %dim_ptr, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %pos_arr15, ptr %dims_ptr, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res)
  store ptr %reshape_dims_res, ptr %pos, align 8
  %self16 = load ptr, ptr %self1, align 8
  %ptr_wte = getelementptr inbounds %GPT, ptr %self16, i32 0, i32 0
  %wte = load ptr, ptr %ptr_wte, align 8
  %idx17 = load ptr, ptr %idx2, align 8
  %call_method = call ptr @tl_Embedding_forward(ptr %wte, ptr %idx17)
  call void @tl_mem_register_tensor(ptr %call_method)
  call void @tl_mem_unregister(ptr %call_method)
  store ptr %call_method, ptr %tok_emb, align 8
  %self18 = load ptr, ptr %self1, align 8
  %ptr_wpe = getelementptr inbounds %GPT, ptr %self18, i32 0, i32 1
  %wpe = load ptr, ptr %ptr_wpe, align 8
  %pos19 = load ptr, ptr %pos, align 8
  %call_method20 = call ptr @tl_Embedding_forward(ptr %wpe, ptr %pos19)
  call void @tl_mem_register_tensor(ptr %call_method20)
  call void @tl_mem_unregister(ptr %call_method20)
  store ptr %call_method20, ptr %pos_emb, align 8
  %tok_emb21 = load ptr, ptr %tok_emb, align 8
  %pos_emb22 = load ptr, ptr %pos_emb, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %tok_emb21, ptr %pos_emb22)
  call void @tl_mem_register_tensor(ptr %binop_res)
  call void @tl_mem_unregister(ptr %binop_res)
  store ptr %binop_res, ptr %x, align 8
  %self23 = load ptr, ptr %self1, align 8
  %ptr_bg = getelementptr inbounds %GPT, ptr %self23, i32 0, i32 2
  %bg = load ptr, ptr %ptr_bg, align 8
  %x24 = load ptr, ptr %x, align 8
  %call_method25 = call ptr @tl_Block_forward(ptr %bg, ptr %x24)
  call void @tl_mem_register_tensor(ptr %call_method25)
  call void @tl_mem_unregister(ptr %call_method25)
  %old_shadowed = load ptr, ptr %x, align 8
  call void @tl_mem_unregister(ptr %old_shadowed)
  store ptr %call_method25, ptr %x26, align 8
  %self27 = load ptr, ptr %self1, align 8
  %ptr_ln_f = getelementptr inbounds %GPT, ptr %self27, i32 0, i32 3
  %ln_f = load ptr, ptr %ptr_ln_f, align 8
  %x28 = load ptr, ptr %x26, align 8
  %call_method29 = call ptr @tl_LayerNorm_forward(ptr %ln_f, ptr %x28)
  call void @tl_mem_register_tensor(ptr %call_method29)
  call void @tl_mem_unregister(ptr %call_method29)
  %old_shadowed30 = load ptr, ptr %x26, align 8
  call void @tl_mem_unregister(ptr %old_shadowed30)
  store ptr %call_method29, ptr %x31, align 8
  %self32 = load ptr, ptr %self1, align 8
  %ptr_head = getelementptr inbounds %GPT, ptr %self32, i32 0, i32 4
  %head = load ptr, ptr %ptr_head, align 8
  %x33 = load ptr, ptr %x31, align 8
  %call_method34 = call ptr @tl_Linear_forward(ptr %head, ptr %x33)
  call void @tl_mem_register_tensor(ptr %call_method34)
  call void @tl_mem_unregister(ptr %call_method34)
  store ptr %call_method34, ptr %logits, align 8
  %logits35 = load ptr, ptr %logits, align 8
  call void @tl_mem_unregister(ptr %logits35)
  %tensor_to_free = load ptr, ptr %x31, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free36 = load ptr, ptr %tok_emb, align 8
  call void @tl_tensor_free(ptr %tensor_to_free36)
  %tensor_to_free37 = load ptr, ptr %pos_arr, align 8
  call void @tl_tensor_free(ptr %tensor_to_free37)
  %tensor_to_free38 = load ptr, ptr %pos, align 8
  call void @tl_tensor_free(ptr %tensor_to_free38)
  %tensor_to_free39 = load ptr, ptr %pos_emb, align 8
  call void @tl_tensor_free(ptr %tensor_to_free39)
  call void @tl_mem_exit_scope()
  ret ptr %logits35
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
  %ptr_wte = getelementptr inbounds %GPT, ptr %s4, i32 0, i32 0
  %s5 = load ptr, ptr %s, align 8
  %ptr_wte6 = getelementptr inbounds %GPT, ptr %s5, i32 0, i32 0
  %wte = load ptr, ptr %ptr_wte6, align 8
  %lr7 = load float, ptr %lr2, align 4
  %call_method = call ptr @tl_Embedding_step(ptr %wte, float %lr7)
  call void @tl_mem_register_struct(ptr %call_method)
  %old_field_val = load ptr, ptr %ptr_wte, align 8
  %cnt_free_diff = icmp ne ptr %old_field_val, %call_method
  br i1 %cnt_free_diff, label %free_old_val, label %skip_free

free_old_val:                                     ; preds = %entry
  %field_gep = getelementptr inbounds %Embedding, ptr %old_field_val, i32 0, i32 0
  %field_load = load ptr, ptr %field_gep, align 8
  call void @tl_tensor_free(ptr %field_load)
  br label %skip_free

skip_free:                                        ; preds = %free_old_val, %entry
  store ptr %call_method, ptr %ptr_wte, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s8 = load ptr, ptr %s, align 8
  %ptr_wpe = getelementptr inbounds %GPT, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_wpe10 = getelementptr inbounds %GPT, ptr %s9, i32 0, i32 1
  %wpe = load ptr, ptr %ptr_wpe10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_Embedding_step(ptr %wpe, float %lr11)
  call void @tl_mem_register_struct(ptr %call_method12)
  %old_field_val13 = load ptr, ptr %ptr_wpe, align 8
  %cnt_free_diff14 = icmp ne ptr %old_field_val13, %call_method12
  br i1 %cnt_free_diff14, label %free_old_val15, label %skip_free16

free_old_val15:                                   ; preds = %skip_free
  %field_gep17 = getelementptr inbounds %Embedding, ptr %old_field_val13, i32 0, i32 0
  %field_load18 = load ptr, ptr %field_gep17, align 8
  call void @tl_tensor_free(ptr %field_load18)
  br label %skip_free16

skip_free16:                                      ; preds = %free_old_val15, %skip_free
  store ptr %call_method12, ptr %ptr_wpe, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s19 = load ptr, ptr %s, align 8
  %ptr_bg = getelementptr inbounds %GPT, ptr %s19, i32 0, i32 2
  %s20 = load ptr, ptr %s, align 8
  %ptr_bg21 = getelementptr inbounds %GPT, ptr %s20, i32 0, i32 2
  %bg = load ptr, ptr %ptr_bg21, align 8
  %lr22 = load float, ptr %lr2, align 4
  %call_method23 = call ptr @tl_Block_step(ptr %bg, float %lr22)
  call void @tl_mem_register_struct(ptr %call_method23)
  %old_field_val24 = load ptr, ptr %ptr_bg, align 8
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
  %field_gep60 = getelementptr inbounds %CausalSelfAttention, ptr %field_load35, i32 0, i32 4
  %field_load61 = load ptr, ptr %field_gep60, align 8
  %field_gep62 = getelementptr inbounds %Linear, ptr %field_load61, i32 0, i32 0
  %field_load63 = load ptr, ptr %field_gep62, align 8
  call void @tl_tensor_free(ptr %field_load63)
  %field_gep64 = getelementptr inbounds %Linear, ptr %field_load61, i32 0, i32 1
  %field_load65 = load ptr, ptr %field_gep64, align 8
  call void @tl_tensor_free(ptr %field_load65)
  %field_gep66 = getelementptr inbounds %Block, ptr %old_field_val24, i32 0, i32 2
  %field_load67 = load ptr, ptr %field_gep66, align 8
  %field_gep68 = getelementptr inbounds %LayerNorm, ptr %field_load67, i32 0, i32 0
  %field_load69 = load ptr, ptr %field_gep68, align 8
  call void @tl_tensor_free(ptr %field_load69)
  %field_gep70 = getelementptr inbounds %LayerNorm, ptr %field_load67, i32 0, i32 1
  %field_load71 = load ptr, ptr %field_gep70, align 8
  call void @tl_tensor_free(ptr %field_load71)
  %field_gep72 = getelementptr inbounds %Block, ptr %old_field_val24, i32 0, i32 3
  %field_load73 = load ptr, ptr %field_gep72, align 8
  %field_gep74 = getelementptr inbounds %MLP, ptr %field_load73, i32 0, i32 0
  %field_load75 = load ptr, ptr %field_gep74, align 8
  %field_gep76 = getelementptr inbounds %Linear, ptr %field_load75, i32 0, i32 0
  %field_load77 = load ptr, ptr %field_gep76, align 8
  call void @tl_tensor_free(ptr %field_load77)
  %field_gep78 = getelementptr inbounds %Linear, ptr %field_load75, i32 0, i32 1
  %field_load79 = load ptr, ptr %field_gep78, align 8
  call void @tl_tensor_free(ptr %field_load79)
  %field_gep80 = getelementptr inbounds %MLP, ptr %field_load73, i32 0, i32 1
  %field_load81 = load ptr, ptr %field_gep80, align 8
  %field_gep82 = getelementptr inbounds %Linear, ptr %field_load81, i32 0, i32 0
  %field_load83 = load ptr, ptr %field_gep82, align 8
  call void @tl_tensor_free(ptr %field_load83)
  %field_gep84 = getelementptr inbounds %Linear, ptr %field_load81, i32 0, i32 1
  %field_load85 = load ptr, ptr %field_gep84, align 8
  call void @tl_tensor_free(ptr %field_load85)
  br label %skip_free27

skip_free27:                                      ; preds = %free_old_val26, %skip_free16
  store ptr %call_method23, ptr %ptr_bg, align 8
  call void @tl_mem_unregister(ptr %call_method23)
  %s86 = load ptr, ptr %s, align 8
  %ptr_ln_f = getelementptr inbounds %GPT, ptr %s86, i32 0, i32 3
  %s87 = load ptr, ptr %s, align 8
  %ptr_ln_f88 = getelementptr inbounds %GPT, ptr %s87, i32 0, i32 3
  %ln_f = load ptr, ptr %ptr_ln_f88, align 8
  %lr89 = load float, ptr %lr2, align 4
  %call_method90 = call ptr @tl_LayerNorm_step(ptr %ln_f, float %lr89)
  call void @tl_mem_register_struct(ptr %call_method90)
  %old_field_val91 = load ptr, ptr %ptr_ln_f, align 8
  %cnt_free_diff92 = icmp ne ptr %old_field_val91, %call_method90
  br i1 %cnt_free_diff92, label %free_old_val93, label %skip_free94

free_old_val93:                                   ; preds = %skip_free27
  %field_gep95 = getelementptr inbounds %LayerNorm, ptr %old_field_val91, i32 0, i32 0
  %field_load96 = load ptr, ptr %field_gep95, align 8
  call void @tl_tensor_free(ptr %field_load96)
  %field_gep97 = getelementptr inbounds %LayerNorm, ptr %old_field_val91, i32 0, i32 1
  %field_load98 = load ptr, ptr %field_gep97, align 8
  call void @tl_tensor_free(ptr %field_load98)
  br label %skip_free94

skip_free94:                                      ; preds = %free_old_val93, %skip_free27
  store ptr %call_method90, ptr %ptr_ln_f, align 8
  call void @tl_mem_unregister(ptr %call_method90)
  %s99 = load ptr, ptr %s, align 8
  %ptr_head = getelementptr inbounds %GPT, ptr %s99, i32 0, i32 4
  %s100 = load ptr, ptr %s, align 8
  %ptr_head101 = getelementptr inbounds %GPT, ptr %s100, i32 0, i32 4
  %head = load ptr, ptr %ptr_head101, align 8
  %lr102 = load float, ptr %lr2, align 4
  %call_method103 = call ptr @tl_Linear_step(ptr %head, float %lr102)
  call void @tl_mem_register_struct(ptr %call_method103)
  %old_field_val104 = load ptr, ptr %ptr_head, align 8
  %cnt_free_diff105 = icmp ne ptr %old_field_val104, %call_method103
  br i1 %cnt_free_diff105, label %free_old_val106, label %skip_free107

free_old_val106:                                  ; preds = %skip_free94
  %field_gep108 = getelementptr inbounds %Linear, ptr %old_field_val104, i32 0, i32 0
  %field_load109 = load ptr, ptr %field_gep108, align 8
  call void @tl_tensor_free(ptr %field_load109)
  %field_gep110 = getelementptr inbounds %Linear, ptr %old_field_val104, i32 0, i32 1
  %field_load111 = load ptr, ptr %field_gep110, align 8
  call void @tl_tensor_free(ptr %field_load111)
  br label %skip_free107

skip_free107:                                     ; preds = %free_old_val106, %skip_free94
  store ptr %call_method103, ptr %ptr_head, align 8
  call void @tl_mem_unregister(ptr %call_method103)
  %s112 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s112)
  %unreg_field_0 = getelementptr inbounds %GPT, ptr %s112, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_0113 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val114 = load ptr, ptr %unreg_field_0113, align 8
  call void @tl_mem_unregister(ptr %field_val114)
  %unreg_field_1 = getelementptr inbounds %GPT, ptr %s112, i32 0, i32 1
  %field_val115 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val115)
  %unreg_field_0116 = getelementptr inbounds %Embedding, ptr %field_val115, i32 0, i32 0
  %field_val117 = load ptr, ptr %unreg_field_0116, align 8
  call void @tl_mem_unregister(ptr %field_val117)
  %unreg_field_2 = getelementptr inbounds %GPT, ptr %s112, i32 0, i32 2
  %field_val118 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val118)
  %unreg_field_0119 = getelementptr inbounds %Block, ptr %field_val118, i32 0, i32 0
  %field_val120 = load ptr, ptr %unreg_field_0119, align 8
  call void @tl_mem_unregister(ptr %field_val120)
  %unreg_field_0121 = getelementptr inbounds %LayerNorm, ptr %field_val120, i32 0, i32 0
  %field_val122 = load ptr, ptr %unreg_field_0121, align 8
  call void @tl_mem_unregister(ptr %field_val122)
  %unreg_field_1123 = getelementptr inbounds %LayerNorm, ptr %field_val120, i32 0, i32 1
  %field_val124 = load ptr, ptr %unreg_field_1123, align 8
  call void @tl_mem_unregister(ptr %field_val124)
  %unreg_field_1125 = getelementptr inbounds %Block, ptr %field_val118, i32 0, i32 1
  %field_val126 = load ptr, ptr %unreg_field_1125, align 8
  call void @tl_mem_unregister(ptr %field_val126)
  %unreg_field_0127 = getelementptr inbounds %CausalSelfAttention, ptr %field_val126, i32 0, i32 0
  %field_val128 = load ptr, ptr %unreg_field_0127, align 8
  call void @tl_mem_unregister(ptr %field_val128)
  %unreg_field_0129 = getelementptr inbounds %Linear, ptr %field_val128, i32 0, i32 0
  %field_val130 = load ptr, ptr %unreg_field_0129, align 8
  call void @tl_mem_unregister(ptr %field_val130)
  %unreg_field_1131 = getelementptr inbounds %Linear, ptr %field_val128, i32 0, i32 1
  %field_val132 = load ptr, ptr %unreg_field_1131, align 8
  call void @tl_mem_unregister(ptr %field_val132)
  %unreg_field_1133 = getelementptr inbounds %CausalSelfAttention, ptr %field_val126, i32 0, i32 1
  %field_val134 = load ptr, ptr %unreg_field_1133, align 8
  call void @tl_mem_unregister(ptr %field_val134)
  %unreg_field_0135 = getelementptr inbounds %Linear, ptr %field_val134, i32 0, i32 0
  %field_val136 = load ptr, ptr %unreg_field_0135, align 8
  call void @tl_mem_unregister(ptr %field_val136)
  %unreg_field_1137 = getelementptr inbounds %Linear, ptr %field_val134, i32 0, i32 1
  %field_val138 = load ptr, ptr %unreg_field_1137, align 8
  call void @tl_mem_unregister(ptr %field_val138)
  %unreg_field_2139 = getelementptr inbounds %CausalSelfAttention, ptr %field_val126, i32 0, i32 2
  %field_val140 = load ptr, ptr %unreg_field_2139, align 8
  call void @tl_mem_unregister(ptr %field_val140)
  %unreg_field_0141 = getelementptr inbounds %Linear, ptr %field_val140, i32 0, i32 0
  %field_val142 = load ptr, ptr %unreg_field_0141, align 8
  call void @tl_mem_unregister(ptr %field_val142)
  %unreg_field_1143 = getelementptr inbounds %Linear, ptr %field_val140, i32 0, i32 1
  %field_val144 = load ptr, ptr %unreg_field_1143, align 8
  call void @tl_mem_unregister(ptr %field_val144)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val126, i32 0, i32 3
  %field_val145 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val145)
  %unreg_field_0146 = getelementptr inbounds %Linear, ptr %field_val145, i32 0, i32 0
  %field_val147 = load ptr, ptr %unreg_field_0146, align 8
  call void @tl_mem_unregister(ptr %field_val147)
  %unreg_field_1148 = getelementptr inbounds %Linear, ptr %field_val145, i32 0, i32 1
  %field_val149 = load ptr, ptr %unreg_field_1148, align 8
  call void @tl_mem_unregister(ptr %field_val149)
  %unreg_field_4 = getelementptr inbounds %CausalSelfAttention, ptr %field_val126, i32 0, i32 4
  %field_val150 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val150)
  %unreg_field_0151 = getelementptr inbounds %Linear, ptr %field_val150, i32 0, i32 0
  %field_val152 = load ptr, ptr %unreg_field_0151, align 8
  call void @tl_mem_unregister(ptr %field_val152)
  %unreg_field_1153 = getelementptr inbounds %Linear, ptr %field_val150, i32 0, i32 1
  %field_val154 = load ptr, ptr %unreg_field_1153, align 8
  call void @tl_mem_unregister(ptr %field_val154)
  %unreg_field_2155 = getelementptr inbounds %Block, ptr %field_val118, i32 0, i32 2
  %field_val156 = load ptr, ptr %unreg_field_2155, align 8
  call void @tl_mem_unregister(ptr %field_val156)
  %unreg_field_0157 = getelementptr inbounds %LayerNorm, ptr %field_val156, i32 0, i32 0
  %field_val158 = load ptr, ptr %unreg_field_0157, align 8
  call void @tl_mem_unregister(ptr %field_val158)
  %unreg_field_1159 = getelementptr inbounds %LayerNorm, ptr %field_val156, i32 0, i32 1
  %field_val160 = load ptr, ptr %unreg_field_1159, align 8
  call void @tl_mem_unregister(ptr %field_val160)
  %unreg_field_3161 = getelementptr inbounds %Block, ptr %field_val118, i32 0, i32 3
  %field_val162 = load ptr, ptr %unreg_field_3161, align 8
  call void @tl_mem_unregister(ptr %field_val162)
  %unreg_field_0163 = getelementptr inbounds %MLP, ptr %field_val162, i32 0, i32 0
  %field_val164 = load ptr, ptr %unreg_field_0163, align 8
  call void @tl_mem_unregister(ptr %field_val164)
  %unreg_field_0165 = getelementptr inbounds %Linear, ptr %field_val164, i32 0, i32 0
  %field_val166 = load ptr, ptr %unreg_field_0165, align 8
  call void @tl_mem_unregister(ptr %field_val166)
  %unreg_field_1167 = getelementptr inbounds %Linear, ptr %field_val164, i32 0, i32 1
  %field_val168 = load ptr, ptr %unreg_field_1167, align 8
  call void @tl_mem_unregister(ptr %field_val168)
  %unreg_field_1169 = getelementptr inbounds %MLP, ptr %field_val162, i32 0, i32 1
  %field_val170 = load ptr, ptr %unreg_field_1169, align 8
  call void @tl_mem_unregister(ptr %field_val170)
  %unreg_field_0171 = getelementptr inbounds %Linear, ptr %field_val170, i32 0, i32 0
  %field_val172 = load ptr, ptr %unreg_field_0171, align 8
  call void @tl_mem_unregister(ptr %field_val172)
  %unreg_field_1173 = getelementptr inbounds %Linear, ptr %field_val170, i32 0, i32 1
  %field_val174 = load ptr, ptr %unreg_field_1173, align 8
  call void @tl_mem_unregister(ptr %field_val174)
  %unreg_field_3175 = getelementptr inbounds %GPT, ptr %s112, i32 0, i32 3
  %field_val176 = load ptr, ptr %unreg_field_3175, align 8
  call void @tl_mem_unregister(ptr %field_val176)
  %unreg_field_0177 = getelementptr inbounds %LayerNorm, ptr %field_val176, i32 0, i32 0
  %field_val178 = load ptr, ptr %unreg_field_0177, align 8
  call void @tl_mem_unregister(ptr %field_val178)
  %unreg_field_1179 = getelementptr inbounds %LayerNorm, ptr %field_val176, i32 0, i32 1
  %field_val180 = load ptr, ptr %unreg_field_1179, align 8
  call void @tl_mem_unregister(ptr %field_val180)
  %unreg_field_4181 = getelementptr inbounds %GPT, ptr %s112, i32 0, i32 4
  %field_val182 = load ptr, ptr %unreg_field_4181, align 8
  call void @tl_mem_unregister(ptr %field_val182)
  %unreg_field_0183 = getelementptr inbounds %Linear, ptr %field_val182, i32 0, i32 0
  %field_val184 = load ptr, ptr %unreg_field_0183, align 8
  call void @tl_mem_unregister(ptr %field_val184)
  %unreg_field_1185 = getelementptr inbounds %Linear, ptr %field_val182, i32 0, i32 1
  %field_val186 = load ptr, ptr %unreg_field_1185, align 8
  call void @tl_mem_unregister(ptr %field_val186)
  call void @tl_mem_exit_scope()
  ret ptr %s112
}

define ptr @gen_data(i64 %step) {
entry:
  %dims_alloca = alloca [2 x i64], align 8
  %arr = alloca ptr, align 16
  %fqa = alloca float, align 16
  %fqk = alloca float, align 16
  %fv6 = alloca float, align 16
  %fk6 = alloca float, align 16
  %fv5 = alloca float, align 16
  %fk5 = alloca float, align 16
  %fv4 = alloca float, align 16
  %fk4 = alloca float, align 16
  %fv3 = alloca float, align 16
  %fk3 = alloca float, align 16
  %fv2 = alloca float, align 16
  %fk2 = alloca float, align 16
  %fv1 = alloca float, align 16
  %fk1 = alloca float, align 16
  %q_ans = alloca i64, align 16
  %q_k = alloca i64, align 16
  %pick = alloca i64, align 16
  %v6 = alloca i64, align 16
  %k6 = alloca i64, align 16
  %v5 = alloca i64, align 16
  %k5 = alloca i64, align 16
  %v4 = alloca i64, align 16
  %k4 = alloca i64, align 16
  %v3 = alloca i64, align 16
  %k3 = alloca i64, align 16
  %v2 = alloca i64, align 16
  %k2 = alloca i64, align 16
  %v1 = alloca i64, align 16
  %k1 = alloca i64, align 16
  %seed = alloca i64, align 16
  %step1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %step, ptr %step1, align 8
  %step2 = load i64, ptr %step1, align 8
  %multmp = mul i64 %step2, 12345
  %addtmp = add i64 %multmp, 6789
  store i64 %addtmp, ptr %seed, align 8
  %seed3 = load i64, ptr %seed, align 8
  %addtmp4 = add i64 %seed3, 1
  %multmp5 = mul i64 %addtmp4, 7
  %seed6 = load i64, ptr %seed, align 8
  %addtmp7 = add i64 %seed6, 1
  %multmp8 = mul i64 %addtmp7, 7
  %divtmp = sdiv i64 %multmp8, 20
  %multmp9 = mul i64 %divtmp, 20
  %subtmp = sub i64 %multmp5, %multmp9
  %addtmp10 = add i64 %subtmp, 10
  store i64 %addtmp10, ptr %k1, align 8
  %seed11 = load i64, ptr %seed, align 8
  %addtmp12 = add i64 %seed11, 2
  %multmp13 = mul i64 %addtmp12, 3
  %seed14 = load i64, ptr %seed, align 8
  %addtmp15 = add i64 %seed14, 2
  %multmp16 = mul i64 %addtmp15, 3
  %divtmp17 = sdiv i64 %multmp16, 10
  %multmp18 = mul i64 %divtmp17, 10
  %subtmp19 = sub i64 %multmp13, %multmp18
  store i64 %subtmp19, ptr %v1, align 8
  %seed20 = load i64, ptr %seed, align 8
  %addtmp21 = add i64 %seed20, 3
  %multmp22 = mul i64 %addtmp21, 7
  %seed23 = load i64, ptr %seed, align 8
  %addtmp24 = add i64 %seed23, 3
  %multmp25 = mul i64 %addtmp24, 7
  %divtmp26 = sdiv i64 %multmp25, 20
  %multmp27 = mul i64 %divtmp26, 20
  %subtmp28 = sub i64 %multmp22, %multmp27
  %addtmp29 = add i64 %subtmp28, 10
  store i64 %addtmp29, ptr %k2, align 8
  %seed30 = load i64, ptr %seed, align 8
  %addtmp31 = add i64 %seed30, 4
  %multmp32 = mul i64 %addtmp31, 3
  %seed33 = load i64, ptr %seed, align 8
  %addtmp34 = add i64 %seed33, 4
  %multmp35 = mul i64 %addtmp34, 3
  %divtmp36 = sdiv i64 %multmp35, 10
  %multmp37 = mul i64 %divtmp36, 10
  %subtmp38 = sub i64 %multmp32, %multmp37
  store i64 %subtmp38, ptr %v2, align 8
  %seed39 = load i64, ptr %seed, align 8
  %addtmp40 = add i64 %seed39, 5
  %multmp41 = mul i64 %addtmp40, 7
  %seed42 = load i64, ptr %seed, align 8
  %addtmp43 = add i64 %seed42, 5
  %multmp44 = mul i64 %addtmp43, 7
  %divtmp45 = sdiv i64 %multmp44, 20
  %multmp46 = mul i64 %divtmp45, 20
  %subtmp47 = sub i64 %multmp41, %multmp46
  %addtmp48 = add i64 %subtmp47, 10
  store i64 %addtmp48, ptr %k3, align 8
  %seed49 = load i64, ptr %seed, align 8
  %addtmp50 = add i64 %seed49, 6
  %multmp51 = mul i64 %addtmp50, 3
  %seed52 = load i64, ptr %seed, align 8
  %addtmp53 = add i64 %seed52, 6
  %multmp54 = mul i64 %addtmp53, 3
  %divtmp55 = sdiv i64 %multmp54, 10
  %multmp56 = mul i64 %divtmp55, 10
  %subtmp57 = sub i64 %multmp51, %multmp56
  store i64 %subtmp57, ptr %v3, align 8
  %seed58 = load i64, ptr %seed, align 8
  %addtmp59 = add i64 %seed58, 7
  %multmp60 = mul i64 %addtmp59, 7
  %seed61 = load i64, ptr %seed, align 8
  %addtmp62 = add i64 %seed61, 7
  %multmp63 = mul i64 %addtmp62, 7
  %divtmp64 = sdiv i64 %multmp63, 20
  %multmp65 = mul i64 %divtmp64, 20
  %subtmp66 = sub i64 %multmp60, %multmp65
  %addtmp67 = add i64 %subtmp66, 10
  store i64 %addtmp67, ptr %k4, align 8
  %seed68 = load i64, ptr %seed, align 8
  %addtmp69 = add i64 %seed68, 8
  %multmp70 = mul i64 %addtmp69, 3
  %seed71 = load i64, ptr %seed, align 8
  %addtmp72 = add i64 %seed71, 8
  %multmp73 = mul i64 %addtmp72, 3
  %divtmp74 = sdiv i64 %multmp73, 10
  %multmp75 = mul i64 %divtmp74, 10
  %subtmp76 = sub i64 %multmp70, %multmp75
  store i64 %subtmp76, ptr %v4, align 8
  %seed77 = load i64, ptr %seed, align 8
  %addtmp78 = add i64 %seed77, 9
  %multmp79 = mul i64 %addtmp78, 7
  %seed80 = load i64, ptr %seed, align 8
  %addtmp81 = add i64 %seed80, 9
  %multmp82 = mul i64 %addtmp81, 7
  %divtmp83 = sdiv i64 %multmp82, 20
  %multmp84 = mul i64 %divtmp83, 20
  %subtmp85 = sub i64 %multmp79, %multmp84
  %addtmp86 = add i64 %subtmp85, 10
  store i64 %addtmp86, ptr %k5, align 8
  %seed87 = load i64, ptr %seed, align 8
  %addtmp88 = add i64 %seed87, 10
  %multmp89 = mul i64 %addtmp88, 3
  %seed90 = load i64, ptr %seed, align 8
  %addtmp91 = add i64 %seed90, 10
  %multmp92 = mul i64 %addtmp91, 3
  %divtmp93 = sdiv i64 %multmp92, 10
  %multmp94 = mul i64 %divtmp93, 10
  %subtmp95 = sub i64 %multmp89, %multmp94
  store i64 %subtmp95, ptr %v5, align 8
  %seed96 = load i64, ptr %seed, align 8
  %addtmp97 = add i64 %seed96, 11
  %multmp98 = mul i64 %addtmp97, 7
  %seed99 = load i64, ptr %seed, align 8
  %addtmp100 = add i64 %seed99, 11
  %multmp101 = mul i64 %addtmp100, 7
  %divtmp102 = sdiv i64 %multmp101, 20
  %multmp103 = mul i64 %divtmp102, 20
  %subtmp104 = sub i64 %multmp98, %multmp103
  %addtmp105 = add i64 %subtmp104, 10
  store i64 %addtmp105, ptr %k6, align 8
  %seed106 = load i64, ptr %seed, align 8
  %addtmp107 = add i64 %seed106, 12
  %multmp108 = mul i64 %addtmp107, 3
  %seed109 = load i64, ptr %seed, align 8
  %addtmp110 = add i64 %seed109, 12
  %multmp111 = mul i64 %addtmp110, 3
  %divtmp112 = sdiv i64 %multmp111, 10
  %multmp113 = mul i64 %divtmp112, 10
  %subtmp114 = sub i64 %multmp108, %multmp113
  store i64 %subtmp114, ptr %v6, align 8
  %seed115 = load i64, ptr %seed, align 8
  %seed116 = load i64, ptr %seed, align 8
  %divtmp117 = sdiv i64 %seed116, 6
  %multmp118 = mul i64 %divtmp117, 6
  %subtmp119 = sub i64 %seed115, %multmp118
  store i64 %subtmp119, ptr %pick, align 8
  store i64 0, ptr %q_k, align 8
  store i64 0, ptr %q_ans, align 8
  %pick120 = load i64, ptr %pick, align 8
  %eqtmp = icmp eq i64 %pick120, 0
  br i1 %eqtmp, label %then, label %else

then:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
  %k1121 = load i64, ptr %k1, align 8
  store i64 %k1121, ptr %q_k, align 8
  %v1122 = load i64, ptr %v1, align 8
  store i64 %v1122, ptr %q_ans, align 8
  call void @tl_mem_exit_scope()
  br label %merge

else:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else, %then
  %pick123 = load i64, ptr %pick, align 8
  %eqtmp124 = icmp eq i64 %pick123, 1
  br i1 %eqtmp124, label %then125, label %else126

then125:                                          ; preds = %merge
  call void @tl_mem_enter_scope()
  %k2128 = load i64, ptr %k2, align 8
  store i64 %k2128, ptr %q_k, align 8
  %v2129 = load i64, ptr %v2, align 8
  store i64 %v2129, ptr %q_ans, align 8
  call void @tl_mem_exit_scope()
  br label %merge127

else126:                                          ; preds = %merge
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge127

merge127:                                         ; preds = %else126, %then125
  %pick130 = load i64, ptr %pick, align 8
  %eqtmp131 = icmp eq i64 %pick130, 2
  br i1 %eqtmp131, label %then132, label %else133

then132:                                          ; preds = %merge127
  call void @tl_mem_enter_scope()
  %k3135 = load i64, ptr %k3, align 8
  store i64 %k3135, ptr %q_k, align 8
  %v3136 = load i64, ptr %v3, align 8
  store i64 %v3136, ptr %q_ans, align 8
  call void @tl_mem_exit_scope()
  br label %merge134

else133:                                          ; preds = %merge127
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge134

merge134:                                         ; preds = %else133, %then132
  %pick137 = load i64, ptr %pick, align 8
  %eqtmp138 = icmp eq i64 %pick137, 3
  br i1 %eqtmp138, label %then139, label %else140

then139:                                          ; preds = %merge134
  call void @tl_mem_enter_scope()
  %k4142 = load i64, ptr %k4, align 8
  store i64 %k4142, ptr %q_k, align 8
  %v4143 = load i64, ptr %v4, align 8
  store i64 %v4143, ptr %q_ans, align 8
  call void @tl_mem_exit_scope()
  br label %merge141

else140:                                          ; preds = %merge134
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge141

merge141:                                         ; preds = %else140, %then139
  %pick144 = load i64, ptr %pick, align 8
  %eqtmp145 = icmp eq i64 %pick144, 4
  br i1 %eqtmp145, label %then146, label %else147

then146:                                          ; preds = %merge141
  call void @tl_mem_enter_scope()
  %k5149 = load i64, ptr %k5, align 8
  store i64 %k5149, ptr %q_k, align 8
  %v5150 = load i64, ptr %v5, align 8
  store i64 %v5150, ptr %q_ans, align 8
  call void @tl_mem_exit_scope()
  br label %merge148

else147:                                          ; preds = %merge141
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge148

merge148:                                         ; preds = %else147, %then146
  %pick151 = load i64, ptr %pick, align 8
  %eqtmp152 = icmp eq i64 %pick151, 5
  br i1 %eqtmp152, label %then153, label %else154

then153:                                          ; preds = %merge148
  call void @tl_mem_enter_scope()
  %k6156 = load i64, ptr %k6, align 8
  store i64 %k6156, ptr %q_k, align 8
  %v6157 = load i64, ptr %v6, align 8
  store i64 %v6157, ptr %q_ans, align 8
  call void @tl_mem_exit_scope()
  br label %merge155

else154:                                          ; preds = %merge148
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge155

merge155:                                         ; preds = %else154, %then153
  %k1158 = load i64, ptr %k1, align 8
  %cast = sitofp i64 %k1158 to float
  store float %cast, ptr %fk1, align 4
  %v1159 = load i64, ptr %v1, align 8
  %cast160 = sitofp i64 %v1159 to float
  store float %cast160, ptr %fv1, align 4
  %k2161 = load i64, ptr %k2, align 8
  %cast162 = sitofp i64 %k2161 to float
  store float %cast162, ptr %fk2, align 4
  %v2163 = load i64, ptr %v2, align 8
  %cast164 = sitofp i64 %v2163 to float
  store float %cast164, ptr %fv2, align 4
  %k3165 = load i64, ptr %k3, align 8
  %cast166 = sitofp i64 %k3165 to float
  store float %cast166, ptr %fk3, align 4
  %v3167 = load i64, ptr %v3, align 8
  %cast168 = sitofp i64 %v3167 to float
  store float %cast168, ptr %fv3, align 4
  %k4169 = load i64, ptr %k4, align 8
  %cast170 = sitofp i64 %k4169 to float
  store float %cast170, ptr %fk4, align 4
  %v4171 = load i64, ptr %v4, align 8
  %cast172 = sitofp i64 %v4171 to float
  store float %cast172, ptr %fv4, align 4
  %k5173 = load i64, ptr %k5, align 8
  %cast174 = sitofp i64 %k5173 to float
  store float %cast174, ptr %fk5, align 4
  %v5175 = load i64, ptr %v5, align 8
  %cast176 = sitofp i64 %v5175 to float
  store float %cast176, ptr %fv5, align 4
  %k6177 = load i64, ptr %k6, align 8
  %cast178 = sitofp i64 %k6177 to float
  store float %cast178, ptr %fk6, align 4
  %v6179 = load i64, ptr %v6, align 8
  %cast180 = sitofp i64 %v6179 to float
  store float %cast180, ptr %fv6, align 4
  %q_k181 = load i64, ptr %q_k, align 8
  %cast182 = sitofp i64 %q_k181 to float
  store float %cast182, ptr %fqk, align 4
  %q_ans183 = load i64, ptr %q_ans, align 8
  %cast184 = sitofp i64 %q_ans183 to float
  store float %cast184, ptr %fqa, align 4
  %buf_void = call ptr @tl_alloc_tmp(i64 56)
  %fk1185 = load float, ptr %fk1, align 4
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %fk1185, ptr %elem_ptr, align 4
  %fv1186 = load float, ptr %fv1, align 4
  %elem_ptr187 = getelementptr inbounds float, ptr %buf_void, i64 1
  store float %fv1186, ptr %elem_ptr187, align 4
  %fk2188 = load float, ptr %fk2, align 4
  %elem_ptr189 = getelementptr inbounds float, ptr %buf_void, i64 2
  store float %fk2188, ptr %elem_ptr189, align 4
  %fv2190 = load float, ptr %fv2, align 4
  %elem_ptr191 = getelementptr inbounds float, ptr %buf_void, i64 3
  store float %fv2190, ptr %elem_ptr191, align 4
  %fk3192 = load float, ptr %fk3, align 4
  %elem_ptr193 = getelementptr inbounds float, ptr %buf_void, i64 4
  store float %fk3192, ptr %elem_ptr193, align 4
  %fv3194 = load float, ptr %fv3, align 4
  %elem_ptr195 = getelementptr inbounds float, ptr %buf_void, i64 5
  store float %fv3194, ptr %elem_ptr195, align 4
  %fk4196 = load float, ptr %fk4, align 4
  %elem_ptr197 = getelementptr inbounds float, ptr %buf_void, i64 6
  store float %fk4196, ptr %elem_ptr197, align 4
  %fv4198 = load float, ptr %fv4, align 4
  %elem_ptr199 = getelementptr inbounds float, ptr %buf_void, i64 7
  store float %fv4198, ptr %elem_ptr199, align 4
  %fk5200 = load float, ptr %fk5, align 4
  %elem_ptr201 = getelementptr inbounds float, ptr %buf_void, i64 8
  store float %fk5200, ptr %elem_ptr201, align 4
  %fv5202 = load float, ptr %fv5, align 4
  %elem_ptr203 = getelementptr inbounds float, ptr %buf_void, i64 9
  store float %fv5202, ptr %elem_ptr203, align 4
  %fk6204 = load float, ptr %fk6, align 4
  %elem_ptr205 = getelementptr inbounds float, ptr %buf_void, i64 10
  store float %fk6204, ptr %elem_ptr205, align 4
  %fv6206 = load float, ptr %fv6, align 4
  %elem_ptr207 = getelementptr inbounds float, ptr %buf_void, i64 11
  store float %fv6206, ptr %elem_ptr207, align 4
  %fqk208 = load float, ptr %fqk, align 4
  %elem_ptr209 = getelementptr inbounds float, ptr %buf_void, i64 12
  store float %fqk208, ptr %elem_ptr209, align 4
  %fqa210 = load float, ptr %fqa, align 4
  %elem_ptr211 = getelementptr inbounds float, ptr %buf_void, i64 13
  store float %fqa210, ptr %elem_ptr211, align 4
  %shape_alloc = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr = getelementptr inbounds i64, ptr %shape_alloc, i64 0
  store i64 14, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %shape_alloc)
  call void @tl_free_tmp(ptr %buf_void)
  call void @tl_free_tmp(ptr %shape_alloc)
  call void @tl_mem_unregister(ptr %new_tensor)
  store ptr %new_tensor, ptr %arr, align 8
  %arr212 = load ptr, ptr %arr, align 8
  %dim_ptr_0 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0, align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 14, ptr %dim_ptr, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %arr212, ptr %dims_ptr, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res)
  %tensor_to_free = load ptr, ptr %arr, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  call void @tl_mem_exit_scope()
  ret ptr %reshape_dims_res
}

define void @main() {
entry:
  %target = alloca i64, align 16
  %pred = alloca i64, align 16
  %pred_t_f32 = alloca ptr, align 16
  %pred_t_i64 = alloca ptr, align 16
  %pred_t = alloca ptr, align 16
  %last_row = alloca ptr, align 16
  %val_logits_1d = alloca ptr, align 16
  %dims_alloca116 = alloca [1 x i64], align 8
  %val_logits_flat = alloca ptr, align 16
  %dims_alloca110 = alloca [2 x i64], align 8
  %val_logits = alloca ptr, align 16
  %val_X = alloca ptr, align 16
  %dims_alloca = alloca [2 x i64], align 8
  %val_in = alloca ptr, align 16
  %val_data = alloca ptr, align 16
  %i = alloca i64, align 16
  %start_idx = alloca i64, align 16
  %total_count = alloca i64, align 16
  %correct_count = alloca i64, align 16
  %model = alloca ptr, align 16
  %max_len = alloca i64, align 16
  %d_model = alloca i64, align 16
  %vocab_size = alloca i64, align 16
  call void @tl_mem_enter_scope()
  call void @tl_arena_init(i64 563200)
  store i64 30, ptr %vocab_size, align 8
  store i64 64, ptr %d_model, align 8
  store i64 32, ptr %max_len, align 8
  %vocab_size1 = load i64, ptr %vocab_size, align 8
  %d_model2 = load i64, ptr %d_model, align 8
  %max_len3 = load i64, ptr %max_len, align 8
  %static_call = call ptr @tl_GPT_new(i64 %vocab_size1, i64 %d_model2, i64 %max_len3)
  call void @tl_mem_unregister(ptr %static_call)
  store ptr %static_call, ptr %model, align 8
  call void @tl_print_string(ptr @str_literal)
  %model4 = load ptr, ptr %model, align 8
  %map = call ptr @tl_tensor_map_load(ptr @str_literal.104)
  %wte = getelementptr inbounds %GPT, ptr %model4, i32 0, i32 0
  %sub_ptr = load ptr, ptr %wte, align 8
  %w = getelementptr inbounds %Embedding, ptr %sub_ptr, i32 0, i32 0
  %t_val = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str)
  store ptr %t_val, ptr %w, align 8
  %wpe = getelementptr inbounds %GPT, ptr %model4, i32 0, i32 1
  %sub_ptr5 = load ptr, ptr %wpe, align 8
  %w6 = getelementptr inbounds %Embedding, ptr %sub_ptr5, i32 0, i32 0
  %t_val7 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.105)
  store ptr %t_val7, ptr %w6, align 8
  %bg = getelementptr inbounds %GPT, ptr %model4, i32 0, i32 2
  %sub_ptr8 = load ptr, ptr %bg, align 8
  %ln1 = getelementptr inbounds %Block, ptr %sub_ptr8, i32 0, i32 0
  %sub_ptr9 = load ptr, ptr %ln1, align 8
  %w10 = getelementptr inbounds %LayerNorm, ptr %sub_ptr9, i32 0, i32 0
  %t_val11 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.106)
  store ptr %t_val11, ptr %w10, align 8
  %b = getelementptr inbounds %LayerNorm, ptr %sub_ptr9, i32 0, i32 1
  %t_val12 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.107)
  store ptr %t_val12, ptr %b, align 8
  %attn = getelementptr inbounds %Block, ptr %sub_ptr8, i32 0, i32 1
  %sub_ptr13 = load ptr, ptr %attn, align 8
  %c_attn = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr13, i32 0, i32 0
  %sub_ptr14 = load ptr, ptr %c_attn, align 8
  %W = getelementptr inbounds %Linear, ptr %sub_ptr14, i32 0, i32 0
  %t_val15 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.108)
  store ptr %t_val15, ptr %W, align 8
  %b16 = getelementptr inbounds %Linear, ptr %sub_ptr14, i32 0, i32 1
  %t_val17 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.109)
  store ptr %t_val17, ptr %b16, align 8
  %q_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr13, i32 0, i32 1
  %sub_ptr18 = load ptr, ptr %q_proj, align 8
  %W19 = getelementptr inbounds %Linear, ptr %sub_ptr18, i32 0, i32 0
  %t_val20 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.110)
  store ptr %t_val20, ptr %W19, align 8
  %b21 = getelementptr inbounds %Linear, ptr %sub_ptr18, i32 0, i32 1
  %t_val22 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.111)
  store ptr %t_val22, ptr %b21, align 8
  %k_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr13, i32 0, i32 2
  %sub_ptr23 = load ptr, ptr %k_proj, align 8
  %W24 = getelementptr inbounds %Linear, ptr %sub_ptr23, i32 0, i32 0
  %t_val25 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.112)
  store ptr %t_val25, ptr %W24, align 8
  %b26 = getelementptr inbounds %Linear, ptr %sub_ptr23, i32 0, i32 1
  %t_val27 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.113)
  store ptr %t_val27, ptr %b26, align 8
  %v_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr13, i32 0, i32 3
  %sub_ptr28 = load ptr, ptr %v_proj, align 8
  %W29 = getelementptr inbounds %Linear, ptr %sub_ptr28, i32 0, i32 0
  %t_val30 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.114)
  store ptr %t_val30, ptr %W29, align 8
  %b31 = getelementptr inbounds %Linear, ptr %sub_ptr28, i32 0, i32 1
  %t_val32 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.115)
  store ptr %t_val32, ptr %b31, align 8
  %c_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr13, i32 0, i32 4
  %sub_ptr33 = load ptr, ptr %c_proj, align 8
  %W34 = getelementptr inbounds %Linear, ptr %sub_ptr33, i32 0, i32 0
  %t_val35 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.116)
  store ptr %t_val35, ptr %W34, align 8
  %b36 = getelementptr inbounds %Linear, ptr %sub_ptr33, i32 0, i32 1
  %t_val37 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.117)
  store ptr %t_val37, ptr %b36, align 8
  %ln2 = getelementptr inbounds %Block, ptr %sub_ptr8, i32 0, i32 2
  %sub_ptr38 = load ptr, ptr %ln2, align 8
  %w39 = getelementptr inbounds %LayerNorm, ptr %sub_ptr38, i32 0, i32 0
  %t_val40 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.118)
  store ptr %t_val40, ptr %w39, align 8
  %b41 = getelementptr inbounds %LayerNorm, ptr %sub_ptr38, i32 0, i32 1
  %t_val42 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.119)
  store ptr %t_val42, ptr %b41, align 8
  %mlp = getelementptr inbounds %Block, ptr %sub_ptr8, i32 0, i32 3
  %sub_ptr43 = load ptr, ptr %mlp, align 8
  %c_fc = getelementptr inbounds %MLP, ptr %sub_ptr43, i32 0, i32 0
  %sub_ptr44 = load ptr, ptr %c_fc, align 8
  %W45 = getelementptr inbounds %Linear, ptr %sub_ptr44, i32 0, i32 0
  %t_val46 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.120)
  store ptr %t_val46, ptr %W45, align 8
  %b47 = getelementptr inbounds %Linear, ptr %sub_ptr44, i32 0, i32 1
  %t_val48 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.121)
  store ptr %t_val48, ptr %b47, align 8
  %c_proj49 = getelementptr inbounds %MLP, ptr %sub_ptr43, i32 0, i32 1
  %sub_ptr50 = load ptr, ptr %c_proj49, align 8
  %W51 = getelementptr inbounds %Linear, ptr %sub_ptr50, i32 0, i32 0
  %t_val52 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.122)
  store ptr %t_val52, ptr %W51, align 8
  %b53 = getelementptr inbounds %Linear, ptr %sub_ptr50, i32 0, i32 1
  %t_val54 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.123)
  store ptr %t_val54, ptr %b53, align 8
  %ln_f = getelementptr inbounds %GPT, ptr %model4, i32 0, i32 3
  %sub_ptr55 = load ptr, ptr %ln_f, align 8
  %w56 = getelementptr inbounds %LayerNorm, ptr %sub_ptr55, i32 0, i32 0
  %t_val57 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.124)
  store ptr %t_val57, ptr %w56, align 8
  %b58 = getelementptr inbounds %LayerNorm, ptr %sub_ptr55, i32 0, i32 1
  %t_val59 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.125)
  store ptr %t_val59, ptr %b58, align 8
  %head = getelementptr inbounds %GPT, ptr %model4, i32 0, i32 4
  %sub_ptr60 = load ptr, ptr %head, align 8
  %W61 = getelementptr inbounds %Linear, ptr %sub_ptr60, i32 0, i32 0
  %t_val62 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.126)
  store ptr %t_val62, ptr %W61, align 8
  %b63 = getelementptr inbounds %Linear, ptr %sub_ptr60, i32 0, i32 1
  %t_val64 = call ptr @tl_tensor_map_get(ptr %map, ptr @key_str.127)
  store ptr %t_val64, ptr %b63, align 8
  call void @tl_tensor_map_free(ptr %map)
  call void @tl_print_string(ptr @str_literal.128)
  call void @tl_print_string(ptr @str_literal.129)
  call void @tl_print_string(ptr @str_literal.130)
  store i64 0, ptr %correct_count, align 8
  store i64 20, ptr %total_count, align 8
  store i64 10000, ptr %start_idx, align 8
  %start_idx65 = load i64, ptr %start_idx, align 8
  %start_idx66 = load i64, ptr %start_idx, align 8
  %total_count67 = load i64, ptr %total_count, align 8
  %addtmp = add i64 %start_idx66, %total_count67
  br label %for_header

for_header:                                       ; preds = %merge, %entry
  %for_idx = phi i64 [ %next_idx, %merge ], [ %start_idx65, %entry ]
  %for_cond = icmp slt i64 %for_idx, %addtmp
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %i, align 8
  %i68 = load i64, ptr %i, align 8
  %call_tmp = call ptr @gen_data(i64 %i68)
  call void @tl_mem_register_tensor(ptr %call_tmp)
  call void @tl_mem_unregister(ptr %call_tmp)
  store ptr %call_tmp, ptr %val_data, align 8
  %buf_void = call ptr @tl_alloc_tmp(i64 52)
  %val_data69 = load ptr, ptr %val_data, align 8
  %get_res = call float @tl_tensor_get(ptr %val_data69, i64 0)
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %get_res, ptr %elem_ptr, align 4
  %val_data70 = load ptr, ptr %val_data, align 8
  %get_res71 = call float @tl_tensor_get(ptr %val_data70, i64 1)
  %elem_ptr72 = getelementptr inbounds float, ptr %buf_void, i64 1
  store float %get_res71, ptr %elem_ptr72, align 4
  %val_data73 = load ptr, ptr %val_data, align 8
  %get_res74 = call float @tl_tensor_get(ptr %val_data73, i64 2)
  %elem_ptr75 = getelementptr inbounds float, ptr %buf_void, i64 2
  store float %get_res74, ptr %elem_ptr75, align 4
  %val_data76 = load ptr, ptr %val_data, align 8
  %get_res77 = call float @tl_tensor_get(ptr %val_data76, i64 3)
  %elem_ptr78 = getelementptr inbounds float, ptr %buf_void, i64 3
  store float %get_res77, ptr %elem_ptr78, align 4
  %val_data79 = load ptr, ptr %val_data, align 8
  %get_res80 = call float @tl_tensor_get(ptr %val_data79, i64 4)
  %elem_ptr81 = getelementptr inbounds float, ptr %buf_void, i64 4
  store float %get_res80, ptr %elem_ptr81, align 4
  %val_data82 = load ptr, ptr %val_data, align 8
  %get_res83 = call float @tl_tensor_get(ptr %val_data82, i64 5)
  %elem_ptr84 = getelementptr inbounds float, ptr %buf_void, i64 5
  store float %get_res83, ptr %elem_ptr84, align 4
  %val_data85 = load ptr, ptr %val_data, align 8
  %get_res86 = call float @tl_tensor_get(ptr %val_data85, i64 6)
  %elem_ptr87 = getelementptr inbounds float, ptr %buf_void, i64 6
  store float %get_res86, ptr %elem_ptr87, align 4
  %val_data88 = load ptr, ptr %val_data, align 8
  %get_res89 = call float @tl_tensor_get(ptr %val_data88, i64 7)
  %elem_ptr90 = getelementptr inbounds float, ptr %buf_void, i64 7
  store float %get_res89, ptr %elem_ptr90, align 4
  %val_data91 = load ptr, ptr %val_data, align 8
  %get_res92 = call float @tl_tensor_get(ptr %val_data91, i64 8)
  %elem_ptr93 = getelementptr inbounds float, ptr %buf_void, i64 8
  store float %get_res92, ptr %elem_ptr93, align 4
  %val_data94 = load ptr, ptr %val_data, align 8
  %get_res95 = call float @tl_tensor_get(ptr %val_data94, i64 9)
  %elem_ptr96 = getelementptr inbounds float, ptr %buf_void, i64 9
  store float %get_res95, ptr %elem_ptr96, align 4
  %val_data97 = load ptr, ptr %val_data, align 8
  %get_res98 = call float @tl_tensor_get(ptr %val_data97, i64 10)
  %elem_ptr99 = getelementptr inbounds float, ptr %buf_void, i64 10
  store float %get_res98, ptr %elem_ptr99, align 4
  %val_data100 = load ptr, ptr %val_data, align 8
  %get_res101 = call float @tl_tensor_get(ptr %val_data100, i64 11)
  %elem_ptr102 = getelementptr inbounds float, ptr %buf_void, i64 11
  store float %get_res101, ptr %elem_ptr102, align 4
  %val_data103 = load ptr, ptr %val_data, align 8
  %get_res104 = call float @tl_tensor_get(ptr %val_data103, i64 12)
  %elem_ptr105 = getelementptr inbounds float, ptr %buf_void, i64 12
  store float %get_res104, ptr %elem_ptr105, align 4
  %shape_alloc = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr = getelementptr inbounds i64, ptr %shape_alloc, i64 0
  store i64 13, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %shape_alloc)
  call void @tl_free_tmp(ptr %buf_void)
  call void @tl_free_tmp(ptr %shape_alloc)
  call void @tl_mem_unregister(ptr %new_tensor)
  store ptr %new_tensor, ptr %val_in, align 8
  %val_in106 = load ptr, ptr %val_in, align 8
  %dim_ptr_0 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0, align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 13, ptr %dim_ptr, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %val_in106, ptr %dims_ptr, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res)
  store ptr %reshape_dims_res, ptr %val_X, align 8
  %model107 = load ptr, ptr %model, align 8
  %val_X108 = load ptr, ptr %val_X, align 8
  %call_method = call ptr @tl_GPT_forward(ptr %model107, ptr %val_X108)
  call void @tl_mem_register_tensor(ptr %call_method)
  call void @tl_mem_unregister(ptr %call_method)
  store ptr %call_method, ptr %val_logits, align 8
  %val_logits109 = load ptr, ptr %val_logits, align 8
  %dim_ptr_0111 = getelementptr [2 x i64], ptr %dims_alloca110, i64 0, i64 0
  store i64 13, ptr %dim_ptr_0111, align 8
  %dim_ptr112 = getelementptr [2 x i64], ptr %dims_alloca110, i64 0, i64 1
  store i64 30, ptr %dim_ptr112, align 8
  %dims_ptr113 = getelementptr [2 x i64], ptr %dims_alloca110, i64 0, i64 0
  %reshape_dims_res114 = call ptr @tl_tensor_reshape_dims(ptr %val_logits109, ptr %dims_ptr113, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res114)
  store ptr %reshape_dims_res114, ptr %val_logits_flat, align 8
  %val_logits_flat115 = load ptr, ptr %val_logits_flat, align 8
  %dim_ptr_0117 = getelementptr [1 x i64], ptr %dims_alloca116, i64 0, i64 0
  store i64 390, ptr %dim_ptr_0117, align 8
  %dims_ptr118 = getelementptr [1 x i64], ptr %dims_alloca116, i64 0, i64 0
  %reshape_dims_res119 = call ptr @tl_tensor_reshape_dims(ptr %val_logits_flat115, ptr %dims_ptr118, i64 1)
  call void @tl_mem_unregister(ptr %reshape_dims_res119)
  store ptr %reshape_dims_res119, ptr %val_logits_1d, align 8
  %val_logits_1d120 = load ptr, ptr %val_logits_1d, align 8
  %call_tmp121 = call ptr @tl_tensor_slice(ptr %val_logits_1d120, i64 360, i64 30)
  call void @tl_mem_register_tensor(ptr %call_tmp121)
  call void @tl_mem_unregister(ptr %call_tmp121)
  store ptr %call_tmp121, ptr %last_row, align 8
  %last_row122 = load ptr, ptr %last_row, align 8
  %argmax_res = call ptr @tl_tensor_argmax(ptr %last_row122, i64 0, i1 false)
  call void @tl_mem_register_tensor(ptr %argmax_res)
  call void @tl_mem_unregister(ptr %argmax_res)
  store ptr %argmax_res, ptr %pred_t, align 8
  %pred_t123 = load ptr, ptr %pred_t, align 8
  %cast_t = call ptr @tl_tensor_to_i64(ptr %pred_t123)
  call void @tl_mem_register_tensor(ptr %cast_t)
  call void @tl_mem_unregister(ptr %cast_t)
  store ptr %cast_t, ptr %pred_t_i64, align 8
  %pred_t_i64124 = load ptr, ptr %pred_t_i64, align 8
  %cast_t125 = call ptr @tl_tensor_to_f32(ptr %pred_t_i64124)
  call void @tl_mem_register_tensor(ptr %cast_t125)
  call void @tl_mem_unregister(ptr %cast_t125)
  store ptr %cast_t125, ptr %pred_t_f32, align 8
  %pred_t_f32126 = load ptr, ptr %pred_t_f32, align 8
  %get_res127 = call float @tl_tensor_get(ptr %pred_t_f32126, i64 0)
  %cast = fptosi float %get_res127 to i64
  store i64 %cast, ptr %pred, align 8
  %val_data128 = load ptr, ptr %val_data, align 8
  %get_res129 = call float @tl_tensor_get(ptr %val_data128, i64 13)
  %cast130 = fptosi float %get_res129 to i64
  store i64 %cast130, ptr %target, align 8
  call void @tl_print_string(ptr @str_literal.131)
  %i131 = load i64, ptr %i, align 8
  call void @tl_print_i64(i64 %i131)
  call void @tl_print_string(ptr @str_literal.132)
  %pred132 = load i64, ptr %pred, align 8
  call void @tl_print_i64(i64 %pred132)
  call void @tl_print_string(ptr @str_literal.133)
  %target133 = load i64, ptr %target, align 8
  call void @tl_print_i64(i64 %target133)
  %pred134 = load i64, ptr %pred, align 8
  %target135 = load i64, ptr %target, align 8
  %eqtmp = icmp eq i64 %pred134, %target135
  br i1 %eqtmp, label %then, label %else

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal.136)
  call void @tl_print_string(ptr @str_literal.137)
  %correct_count147 = load i64, ptr %correct_count, align 8
  call void @tl_print_i64(i64 %correct_count147)
  call void @tl_print_string(ptr @str_literal.138)
  %total_count148 = load i64, ptr %total_count, align 8
  call void @tl_print_i64(i64 %total_count148)
  %struct_to_free = load ptr, ptr %model, align 8
  %field_gep = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 0
  %field_load = load ptr, ptr %field_gep, align 8
  %field_gep149 = getelementptr inbounds %Embedding, ptr %field_load, i32 0, i32 0
  %field_load150 = load ptr, ptr %field_gep149, align 8
  call void @tl_tensor_free(ptr %field_load150)
  %field_gep151 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 1
  %field_load152 = load ptr, ptr %field_gep151, align 8
  %field_gep153 = getelementptr inbounds %Embedding, ptr %field_load152, i32 0, i32 0
  %field_load154 = load ptr, ptr %field_gep153, align 8
  call void @tl_tensor_free(ptr %field_load154)
  %field_gep155 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 2
  %field_load156 = load ptr, ptr %field_gep155, align 8
  %field_gep157 = getelementptr inbounds %Block, ptr %field_load156, i32 0, i32 0
  %field_load158 = load ptr, ptr %field_gep157, align 8
  %field_gep159 = getelementptr inbounds %LayerNorm, ptr %field_load158, i32 0, i32 0
  %field_load160 = load ptr, ptr %field_gep159, align 8
  call void @tl_tensor_free(ptr %field_load160)
  %field_gep161 = getelementptr inbounds %LayerNorm, ptr %field_load158, i32 0, i32 1
  %field_load162 = load ptr, ptr %field_gep161, align 8
  call void @tl_tensor_free(ptr %field_load162)
  %field_gep163 = getelementptr inbounds %Block, ptr %field_load156, i32 0, i32 1
  %field_load164 = load ptr, ptr %field_gep163, align 8
  %field_gep165 = getelementptr inbounds %CausalSelfAttention, ptr %field_load164, i32 0, i32 0
  %field_load166 = load ptr, ptr %field_gep165, align 8
  %field_gep167 = getelementptr inbounds %Linear, ptr %field_load166, i32 0, i32 0
  %field_load168 = load ptr, ptr %field_gep167, align 8
  call void @tl_tensor_free(ptr %field_load168)
  %field_gep169 = getelementptr inbounds %Linear, ptr %field_load166, i32 0, i32 1
  %field_load170 = load ptr, ptr %field_gep169, align 8
  call void @tl_tensor_free(ptr %field_load170)
  %field_gep171 = getelementptr inbounds %CausalSelfAttention, ptr %field_load164, i32 0, i32 1
  %field_load172 = load ptr, ptr %field_gep171, align 8
  %field_gep173 = getelementptr inbounds %Linear, ptr %field_load172, i32 0, i32 0
  %field_load174 = load ptr, ptr %field_gep173, align 8
  call void @tl_tensor_free(ptr %field_load174)
  %field_gep175 = getelementptr inbounds %Linear, ptr %field_load172, i32 0, i32 1
  %field_load176 = load ptr, ptr %field_gep175, align 8
  call void @tl_tensor_free(ptr %field_load176)
  %field_gep177 = getelementptr inbounds %CausalSelfAttention, ptr %field_load164, i32 0, i32 2
  %field_load178 = load ptr, ptr %field_gep177, align 8
  %field_gep179 = getelementptr inbounds %Linear, ptr %field_load178, i32 0, i32 0
  %field_load180 = load ptr, ptr %field_gep179, align 8
  call void @tl_tensor_free(ptr %field_load180)
  %field_gep181 = getelementptr inbounds %Linear, ptr %field_load178, i32 0, i32 1
  %field_load182 = load ptr, ptr %field_gep181, align 8
  call void @tl_tensor_free(ptr %field_load182)
  %field_gep183 = getelementptr inbounds %CausalSelfAttention, ptr %field_load164, i32 0, i32 3
  %field_load184 = load ptr, ptr %field_gep183, align 8
  %field_gep185 = getelementptr inbounds %Linear, ptr %field_load184, i32 0, i32 0
  %field_load186 = load ptr, ptr %field_gep185, align 8
  call void @tl_tensor_free(ptr %field_load186)
  %field_gep187 = getelementptr inbounds %Linear, ptr %field_load184, i32 0, i32 1
  %field_load188 = load ptr, ptr %field_gep187, align 8
  call void @tl_tensor_free(ptr %field_load188)
  %field_gep189 = getelementptr inbounds %CausalSelfAttention, ptr %field_load164, i32 0, i32 4
  %field_load190 = load ptr, ptr %field_gep189, align 8
  %field_gep191 = getelementptr inbounds %Linear, ptr %field_load190, i32 0, i32 0
  %field_load192 = load ptr, ptr %field_gep191, align 8
  call void @tl_tensor_free(ptr %field_load192)
  %field_gep193 = getelementptr inbounds %Linear, ptr %field_load190, i32 0, i32 1
  %field_load194 = load ptr, ptr %field_gep193, align 8
  call void @tl_tensor_free(ptr %field_load194)
  %field_gep195 = getelementptr inbounds %Block, ptr %field_load156, i32 0, i32 2
  %field_load196 = load ptr, ptr %field_gep195, align 8
  %field_gep197 = getelementptr inbounds %LayerNorm, ptr %field_load196, i32 0, i32 0
  %field_load198 = load ptr, ptr %field_gep197, align 8
  call void @tl_tensor_free(ptr %field_load198)
  %field_gep199 = getelementptr inbounds %LayerNorm, ptr %field_load196, i32 0, i32 1
  %field_load200 = load ptr, ptr %field_gep199, align 8
  call void @tl_tensor_free(ptr %field_load200)
  %field_gep201 = getelementptr inbounds %Block, ptr %field_load156, i32 0, i32 3
  %field_load202 = load ptr, ptr %field_gep201, align 8
  %field_gep203 = getelementptr inbounds %MLP, ptr %field_load202, i32 0, i32 0
  %field_load204 = load ptr, ptr %field_gep203, align 8
  %field_gep205 = getelementptr inbounds %Linear, ptr %field_load204, i32 0, i32 0
  %field_load206 = load ptr, ptr %field_gep205, align 8
  call void @tl_tensor_free(ptr %field_load206)
  %field_gep207 = getelementptr inbounds %Linear, ptr %field_load204, i32 0, i32 1
  %field_load208 = load ptr, ptr %field_gep207, align 8
  call void @tl_tensor_free(ptr %field_load208)
  %field_gep209 = getelementptr inbounds %MLP, ptr %field_load202, i32 0, i32 1
  %field_load210 = load ptr, ptr %field_gep209, align 8
  %field_gep211 = getelementptr inbounds %Linear, ptr %field_load210, i32 0, i32 0
  %field_load212 = load ptr, ptr %field_gep211, align 8
  call void @tl_tensor_free(ptr %field_load212)
  %field_gep213 = getelementptr inbounds %Linear, ptr %field_load210, i32 0, i32 1
  %field_load214 = load ptr, ptr %field_gep213, align 8
  call void @tl_tensor_free(ptr %field_load214)
  %field_gep215 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 3
  %field_load216 = load ptr, ptr %field_gep215, align 8
  %field_gep217 = getelementptr inbounds %LayerNorm, ptr %field_load216, i32 0, i32 0
  %field_load218 = load ptr, ptr %field_gep217, align 8
  call void @tl_tensor_free(ptr %field_load218)
  %field_gep219 = getelementptr inbounds %LayerNorm, ptr %field_load216, i32 0, i32 1
  %field_load220 = load ptr, ptr %field_gep219, align 8
  call void @tl_tensor_free(ptr %field_load220)
  %field_gep221 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 4
  %field_load222 = load ptr, ptr %field_gep221, align 8
  %field_gep223 = getelementptr inbounds %Linear, ptr %field_load222, i32 0, i32 0
  %field_load224 = load ptr, ptr %field_gep223, align 8
  call void @tl_tensor_free(ptr %field_load224)
  %field_gep225 = getelementptr inbounds %Linear, ptr %field_load222, i32 0, i32 1
  %field_load226 = load ptr, ptr %field_gep225, align 8
  call void @tl_tensor_free(ptr %field_load226)
  call void @tl_mem_unregister(ptr %struct_to_free)
  call void @tl_mem_exit_scope()
  ret void

then:                                             ; preds = %for_body
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.134)
  %correct_count136 = load i64, ptr %correct_count, align 8
  %addtmp137 = add i64 %correct_count136, 1
  store i64 %addtmp137, ptr %correct_count, align 8
  call void @tl_mem_exit_scope()
  br label %merge

else:                                             ; preds = %for_body
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.135)
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else, %then
  %tensor_to_free = load ptr, ptr %val_X, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free138 = load ptr, ptr %last_row, align 8
  call void @tl_tensor_free(ptr %tensor_to_free138)
  %tensor_to_free139 = load ptr, ptr %val_data, align 8
  call void @tl_tensor_free(ptr %tensor_to_free139)
  %tensor_to_free140 = load ptr, ptr %val_logits, align 8
  call void @tl_tensor_free(ptr %tensor_to_free140)
  %tensor_to_free141 = load ptr, ptr %val_in, align 8
  call void @tl_tensor_free(ptr %tensor_to_free141)
  %tensor_to_free142 = load ptr, ptr %pred_t_f32, align 8
  call void @tl_tensor_free(ptr %tensor_to_free142)
  %tensor_to_free143 = load ptr, ptr %val_logits_flat, align 8
  call void @tl_tensor_free(ptr %tensor_to_free143)
  %tensor_to_free144 = load ptr, ptr %pred_t, align 8
  call void @tl_tensor_free(ptr %tensor_to_free144)
  %tensor_to_free145 = load ptr, ptr %pred_t_i64, align 8
  call void @tl_tensor_free(ptr %tensor_to_free145)
  %tensor_to_free146 = load ptr, ptr %val_logits_1d, align 8
  call void @tl_tensor_free(ptr %tensor_to_free146)
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx, 1
  br label %for_header
}
