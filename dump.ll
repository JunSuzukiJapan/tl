; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

%Linear = type { ptr, ptr }
%SimpleModel = type { ptr }

@str_literal = private unnamed_addr constant [18 x i8] c"Creating model...\00", align 1
@str_literal.104 = private unnamed_addr constant [14 x i8] c"Model created\00", align 1
@str_literal.105 = private unnamed_addr constant [18 x i8] c"Creating input...\00", align 1
@str_literal.106 = private unnamed_addr constant [14 x i8] c"Input created\00", align 1
@str_literal.107 = private unnamed_addr constant [16 x i8] c"Forward pass...\00", align 1
@str_literal.108 = private unnamed_addr constant [13 x i8] c"Forward done\00", align 1
@str_literal.109 = private unnamed_addr constant [13 x i8] c"Reshaping...\00", align 1
@str_literal.110 = private unnamed_addr constant [9 x i8] c"Reshaped\00", align 1
@str_literal.111 = private unnamed_addr constant [11 x i8] c"Slicing...\00", align 1
@str_literal.112 = private unnamed_addr constant [9 x i8] c"Slice 1:\00", align 1
@str_literal.113 = private unnamed_addr constant [9 x i8] c"Slice 2:\00", align 1
@str_literal.114 = private unnamed_addr constant [9 x i8] c"Slice 3:\00", align 1
@str_literal.115 = private unnamed_addr constant [9 x i8] c"Slice 4:\00", align 1
@str_literal.116 = private unnamed_addr constant [16 x i8] c"All slices done\00", align 1
@str_literal.117 = private unnamed_addr constant [17 x i8] c"Starting test...\00", align 1
@str_literal.118 = private unnamed_addr constant [29 x i8] c"Test finished - no segfault!\00", align 1

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

define ptr @tl_SimpleModel_new() {
entry:
  call void @tl_mem_enter_scope()
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%SimpleModel, ptr null, i32 1) to i64))
  %static_call = call ptr @tl_Linear_new(i64 128, i64 12)
  %init_field = getelementptr inbounds nuw %SimpleModel, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds nuw %SimpleModel, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_01 = getelementptr inbounds nuw %Linear, ptr %field_val, i32 0, i32 0
  %field_val2 = load ptr, ptr %unreg_field_01, align 8
  call void @tl_mem_unregister(ptr %field_val2)
  %unreg_field_1 = getelementptr inbounds nuw %Linear, ptr %field_val, i32 0, i32 1
  %field_val3 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val3)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_SimpleModel_forward(ptr %self, ptr %x) {
entry:
  %x2 = alloca ptr, align 16
  %self1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_linear = getelementptr inbounds nuw %SimpleModel, ptr %self3, i32 0, i32 0
  %linear = load ptr, ptr %ptr_linear, align 8
  %x4 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_Linear_forward(ptr %linear, ptr %x4)
  call void @tl_mem_register_tensor(ptr %call_method)
  call void @tl_mem_unregister(ptr %call_method)
  call void @tl_mem_exit_scope()
  ret ptr %call_method
}

define void @test_with_slice() {
entry:
  %s4 = alloca ptr, align 16
  %s3 = alloca ptr, align 16
  %s2 = alloca ptr, align 16
  %s1 = alloca ptr, align 16
  %y_flat = alloca ptr, align 16
  %y = alloca ptr, align 16
  %x = alloca ptr, align 16
  %conv_buf = alloca [3 x float], align 4
  %model = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal)
  %static_call = call ptr @tl_SimpleModel_new()
  store ptr %static_call, ptr %model, align 8
  call void @tl_print_string(ptr @str_literal.104)
  call void @tl_print_string(ptr @str_literal.105)
  %malloc_size = mul i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), 3
  %arr_malloc = call ptr @malloc(i64 %malloc_size)
  call void @tl_mem_register_struct(ptr %arr_malloc)
  %elem_ptr = getelementptr inbounds i64, ptr %arr_malloc, i64 0
  store i64 1, ptr %elem_ptr, align 8
  %elem_ptr1 = getelementptr inbounds i64, ptr %arr_malloc, i64 1
  store i64 10, ptr %elem_ptr1, align 8
  %elem_ptr2 = getelementptr inbounds i64, ptr %arr_malloc, i64 2
  store i64 128, ptr %elem_ptr2, align 8
  %src = getelementptr inbounds i64, ptr %arr_malloc, i64 0
  %l = load i64, ptr %src, align 8
  %c = sitofp i64 %l to float
  %dst = getelementptr inbounds float, ptr %conv_buf, i64 0
  store float %c, ptr %dst, align 4
  %src3 = getelementptr inbounds i64, ptr %arr_malloc, i64 1
  %l4 = load i64, ptr %src3, align 8
  %c5 = sitofp i64 %l4 to float
  %dst6 = getelementptr inbounds float, ptr %conv_buf, i64 1
  store float %c5, ptr %dst6, align 4
  %src7 = getelementptr inbounds i64, ptr %arr_malloc, i64 2
  %l8 = load i64, ptr %src7, align 8
  %c9 = sitofp i64 %l8 to float
  %dst10 = getelementptr inbounds float, ptr %conv_buf, i64 2
  store float %c9, ptr %dst10, align 4
  %shape_arr = alloca [1 x i64], align 8
  %shape_ptr = getelementptr inbounds [1 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 3, ptr %shape_ptr, align 8
  %converted_tensor = call ptr @tl_tensor_new(ptr %conv_buf, i64 1, ptr %shape_arr)
  %static_call11 = call ptr @tl_tensor_randn(ptr %converted_tensor, i1 false)
  store ptr %static_call11, ptr %x, align 8
  call void @tl_print_string(ptr @str_literal.106)
  call void @tl_print_string(ptr @str_literal.107)
  %model12 = load ptr, ptr %model, align 8
  %x13 = load ptr, ptr %x, align 8
  %call_method = call ptr @tl_SimpleModel_forward(ptr %model12, ptr %x13)
  call void @tl_mem_register_tensor(ptr %call_method)
  store ptr %call_method, ptr %y, align 8
  call void @tl_print_string(ptr @str_literal.108)
  call void @tl_print_string(ptr @str_literal.109)
  %y14 = load ptr, ptr %y, align 8
  %dims_alloca = alloca [2 x i64], align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 10, ptr %dim_ptr, align 8
  %dim_ptr15 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr15, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %y14, ptr %dims_ptr, i64 2)
  store ptr %reshape_dims_res, ptr %y_flat, align 8
  call void @tl_print_string(ptr @str_literal.110)
  call void @tl_print_string(ptr @str_literal.111)
  %y_flat16 = load ptr, ptr %y_flat, align 8
  %call_tmp = call ptr @tl_tensor_slice(ptr %y_flat16, i64 0, i64 1)
  call void @tl_mem_register_tensor(ptr %call_tmp)
  store ptr %call_tmp, ptr %s1, align 8
  call void @tl_print_string(ptr @str_literal.112)
  %s117 = load ptr, ptr %s1, align 8
  call void @tl_tensor_print(ptr %s117)
  %y_flat18 = load ptr, ptr %y_flat, align 8
  %call_tmp19 = call ptr @tl_tensor_slice(ptr %y_flat18, i64 1, i64 1)
  call void @tl_mem_register_tensor(ptr %call_tmp19)
  store ptr %call_tmp19, ptr %s2, align 8
  call void @tl_print_string(ptr @str_literal.113)
  %s220 = load ptr, ptr %s2, align 8
  call void @tl_tensor_print(ptr %s220)
  %y_flat21 = load ptr, ptr %y_flat, align 8
  %call_tmp22 = call ptr @tl_tensor_slice(ptr %y_flat21, i64 2, i64 1)
  call void @tl_mem_register_tensor(ptr %call_tmp22)
  store ptr %call_tmp22, ptr %s3, align 8
  call void @tl_print_string(ptr @str_literal.114)
  %s323 = load ptr, ptr %s3, align 8
  call void @tl_tensor_print(ptr %s323)
  %y_flat24 = load ptr, ptr %y_flat, align 8
  %call_tmp25 = call ptr @tl_tensor_slice(ptr %y_flat24, i64 3, i64 1)
  call void @tl_mem_register_tensor(ptr %call_tmp25)
  store ptr %call_tmp25, ptr %s4, align 8
  call void @tl_print_string(ptr @str_literal.115)
  %s426 = load ptr, ptr %s4, align 8
  call void @tl_tensor_print(ptr %s426)
  call void @tl_print_string(ptr @str_literal.116)
  call void @tl_mem_exit_scope()
  ret void
}

define void @main() {
entry:
  call void @tl_mem_enter_scope()
  call void @tl_arena_init(i64 153600)
  call void @tl_print_string(ptr @str_literal.117)
  call void @test_with_slice()
  call void @tl_print_string(ptr @str_literal.118)
  call void @tl_mem_exit_scope()
  ret void
}
