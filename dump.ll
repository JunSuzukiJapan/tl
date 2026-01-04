; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@str_literal = private unnamed_addr constant [34 x i8] c"Arena offset mismatch! Expected: \00", align 1
@str_literal.104 = private unnamed_addr constant [8 x i8] c", Got: \00", align 1
@str_literal.105 = private unnamed_addr constant [39 x i8] c"Testing function level early return...\00", align 1
@str_literal.106 = private unnamed_addr constant [53 x i8] c"FAILED: Early return did not restore offset. Start: \00", align 1
@str_literal.107 = private unnamed_addr constant [8 x i8] c", End: \00", align 1
@str_literal.108 = private unnamed_addr constant [37 x i8] c"PASSED: Early return restored offset\00", align 1
@str_literal.109 = private unnamed_addr constant [54 x i8] c"FAILED: Normal return did not restore offset. Start: \00", align 1
@str_literal.110 = private unnamed_addr constant [8 x i8] c", End: \00", align 1
@str_literal.111 = private unnamed_addr constant [38 x i8] c"PASSED: Normal return restored offset\00", align 1
@str_literal.112 = private unnamed_addr constant [37 x i8] c"Testing loop allocation stability...\00", align 1
@str_literal.113 = private unnamed_addr constant [56 x i8] c"FAILED: Loop iterations did not restore offset. Start: \00", align 1
@str_literal.114 = private unnamed_addr constant [8 x i8] c", End: \00", align 1
@str_literal.115 = private unnamed_addr constant [40 x i8] c"PASSED: Loop iterations restored offset\00", align 1
@str_literal.116 = private unnamed_addr constant [37 x i8] c"Testing nested block scope resets...\00", align 1
@str_literal.117 = private unnamed_addr constant [46 x i8] c"FAILED: Outer block did not allocate in arena\00", align 1
@str_literal.118 = private unnamed_addr constant [46 x i8] c"FAILED: Inner block did not allocate in arena\00", align 1
@str_literal.119 = private unnamed_addr constant [53 x i8] c"FAILED: Inner block did not restore offset. Middle: \00", align 1
@str_literal.120 = private unnamed_addr constant [16 x i8] c", After inner: \00", align 1
@str_literal.121 = private unnamed_addr constant [36 x i8] c"PASSED: Inner block restored offset\00", align 1
@str_literal.122 = private unnamed_addr constant [54 x i8] c"FAILED: Nested blocks did not restore offset. Start: \00", align 1
@str_literal.123 = private unnamed_addr constant [8 x i8] c", End: \00", align 1
@str_literal.124 = private unnamed_addr constant [38 x i8] c"PASSED: Nested blocks restored offset\00", align 1
@str_literal.125 = private unnamed_addr constant [43 x i8] c"Starting Arena Allocator Integration Tests\00", align 1
@str_literal.126 = private unnamed_addr constant [17 x i8] c"Arena is active.\00", align 1
@str_literal.127 = private unnamed_addr constant [58 x i8] c"Arena is NOT active. Creating a tensor to trigger init...\00", align 1
@str_literal.128 = private unnamed_addr constant [21 x i8] c"Arena is now active.\00", align 1
@str_literal.129 = private unnamed_addr constant [39 x i8] c"Arena still NOT active. Manual init...\00", align 1
@str_literal.130 = private unnamed_addr constant [34 x i8] c"Arena Integration Tests Completed\00", align 1

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

define void @verify_offset(i64 %expected) {
entry:
  %actual = alloca i64, align 16
  %expected1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %expected, ptr %expected1, align 8
  %call_tmp = call i64 @tl_arena_get_offset()
  store i64 %call_tmp, ptr %actual, align 8
  %actual2 = load i64, ptr %actual, align 8
  %expected3 = load i64, ptr %expected1, align 8
  %neqtmp = icmp ne i64 %actual2, %expected3
  br i1 %neqtmp, label %then, label %else

then:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal)
  %expected4 = load i64, ptr %expected1, align 8
  call void @tl_print_i64(i64 %expected4)
  call void @tl_print_string(ptr @str_literal.104)
  %actual5 = load i64, ptr %actual, align 8
  call void @tl_print_i64(i64 %actual5)
  call void @tl_mem_exit_scope()
  br label %merge

else:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else, %then
  call void @tl_mem_exit_scope()
  ret void
}

define ptr @test_early_return(i1 %cond) {
entry:
  %ptr2 = alloca i64, align 16
  %large1 = alloca ptr, align 16
  %conv_buf = alloca [1 x float], align 4
  %ptr = alloca i64, align 16
  %cond1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i1 %cond, ptr %cond1, align 1
  %call_tmp = call ptr @tl_arena_alloc(i64 128)
  store ptr %call_tmp, ptr %ptr, align 8
  %arr_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64))
  call void @tl_mem_register_struct(ptr %arr_malloc)
  %elem_ptr = getelementptr inbounds i64, ptr %arr_malloc, i64 0
  store i64 10, ptr %elem_ptr, align 8
  %src = getelementptr inbounds i64, ptr %arr_malloc, i64 0
  %l = load i64, ptr %src, align 8
  %c = sitofp i64 %l to float
  %dst = getelementptr inbounds float, ptr %conv_buf, i64 0
  store float %c, ptr %dst, align 4
  %shape_arr = alloca [1 x i64], align 8
  %shape_ptr = getelementptr inbounds [1 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 1, ptr %shape_ptr, align 8
  %converted_tensor = call ptr @tl_tensor_new(ptr %conv_buf, i64 1, ptr %shape_arr)
  %static_call = call ptr @tl_tensor_randn(ptr %converted_tensor, i1 false)
  store ptr %static_call, ptr %large1, align 8
  %cond2 = load i1, ptr %cond1, align 1
  br i1 %cond2, label %then, label %else

then:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
  %large13 = load ptr, ptr %large1, align 8
  call void @tl_mem_unregister(ptr %large13)
  call void @tl_mem_exit_scope()
  call void @tl_mem_exit_scope()
  ret ptr %large13

else:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else
  %call_tmp4 = call ptr @tl_arena_alloc(i64 256)
  store ptr %call_tmp4, ptr %ptr2, align 8
  %large15 = load ptr, ptr %large1, align 8
  call void @tl_mem_unregister(ptr %large15)
  call void @tl_mem_exit_scope()
  ret ptr %large15
}

define void @run_func_test() {
entry:
  %end_offset2 = alloca i64, align 16
  %r2 = alloca ptr, align 16
  %end_offset1 = alloca i64, align 16
  %r1 = alloca ptr, align 16
  %start_offset = alloca i64, align 16
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.105)
  %call_tmp = call i64 @tl_arena_get_offset()
  store i64 %call_tmp, ptr %start_offset, align 8
  %call_tmp1 = call ptr @test_early_return(i1 true)
  call void @tl_mem_register_tensor(ptr %call_tmp1)
  store ptr %call_tmp1, ptr %r1, align 8
  %call_tmp2 = call i64 @tl_arena_get_offset()
  store i64 %call_tmp2, ptr %end_offset1, align 8
  %end_offset13 = load i64, ptr %end_offset1, align 8
  %start_offset4 = load i64, ptr %start_offset, align 8
  %neqtmp = icmp ne i64 %end_offset13, %start_offset4
  br i1 %neqtmp, label %then, label %else

then:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.106)
  %start_offset5 = load i64, ptr %start_offset, align 8
  call void @tl_print_i64(i64 %start_offset5)
  call void @tl_print_string(ptr @str_literal.107)
  %end_offset16 = load i64, ptr %end_offset1, align 8
  call void @tl_print_i64(i64 %end_offset16)
  call void @tl_mem_exit_scope()
  br label %merge

else:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.108)
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else, %then
  %call_tmp7 = call ptr @test_early_return(i1 false)
  call void @tl_mem_register_tensor(ptr %call_tmp7)
  store ptr %call_tmp7, ptr %r2, align 8
  %call_tmp8 = call i64 @tl_arena_get_offset()
  store i64 %call_tmp8, ptr %end_offset2, align 8
  %end_offset29 = load i64, ptr %end_offset2, align 8
  %start_offset10 = load i64, ptr %start_offset, align 8
  %neqtmp11 = icmp ne i64 %end_offset29, %start_offset10
  br i1 %neqtmp11, label %then12, label %else13

then12:                                           ; preds = %merge
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.109)
  %start_offset15 = load i64, ptr %start_offset, align 8
  call void @tl_print_i64(i64 %start_offset15)
  call void @tl_print_string(ptr @str_literal.110)
  %end_offset216 = load i64, ptr %end_offset2, align 8
  call void @tl_print_i64(i64 %end_offset216)
  call void @tl_mem_exit_scope()
  br label %merge14

else13:                                           ; preds = %merge
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.111)
  call void @tl_mem_exit_scope()
  br label %merge14

merge14:                                          ; preds = %else13, %then12
  call void @tl_mem_exit_scope()
  ret void
}

define void @run_loop_test() {
entry:
  %end_offset = alloca i64, align 16
  %ptr = alloca i64, align 16
  %i = alloca i64, align 16
  %start_offset = alloca i64, align 16
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.112)
  %call_tmp = call i64 @tl_arena_get_offset()
  store i64 %call_tmp, ptr %start_offset, align 8
  br label %for_header

for_header:                                       ; preds = %for_body, %entry
  %for_idx = phi i64 [ 0, %entry ], [ %next_idx, %for_body ]
  %for_cond = icmp slt i64 %for_idx, 5
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %i, align 8
  %call_tmp1 = call ptr @tl_arena_alloc(i64 64)
  store ptr %call_tmp1, ptr %ptr, align 8
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx, 1
  br label %for_header

for_end:                                          ; preds = %for_header
  %call_tmp2 = call i64 @tl_arena_get_offset()
  store i64 %call_tmp2, ptr %end_offset, align 8
  %end_offset3 = load i64, ptr %end_offset, align 8
  %start_offset4 = load i64, ptr %start_offset, align 8
  %neqtmp = icmp ne i64 %end_offset3, %start_offset4
  br i1 %neqtmp, label %then, label %else

then:                                             ; preds = %for_end
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.113)
  %start_offset5 = load i64, ptr %start_offset, align 8
  call void @tl_print_i64(i64 %start_offset5)
  call void @tl_print_string(ptr @str_literal.114)
  %end_offset6 = load i64, ptr %end_offset, align 8
  call void @tl_print_i64(i64 %end_offset6)
  call void @tl_mem_exit_scope()
  br label %merge

else:                                             ; preds = %for_end
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.115)
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else, %then
  call void @tl_mem_exit_scope()
  ret void
}

define void @run_nested_test() {
entry:
  %end_offset = alloca i64, align 16
  %inner_offset = alloca i64, align 16
  %ptr2 = alloca i64, align 16
  %middle_offset = alloca i64, align 16
  %ptr1 = alloca i64, align 16
  %start_offset = alloca i64, align 16
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.116)
  %call_tmp = call i64 @tl_arena_get_offset()
  store i64 %call_tmp, ptr %start_offset, align 8
  br i1 true, label %then, label %else

then:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
  %call_tmp1 = call ptr @tl_arena_alloc(i64 128)
  store ptr %call_tmp1, ptr %ptr1, align 8
  %call_tmp2 = call i64 @tl_arena_get_offset()
  store i64 %call_tmp2, ptr %middle_offset, align 8
  %middle_offset3 = load i64, ptr %middle_offset, align 8
  %start_offset4 = load i64, ptr %start_offset, align 8
  %letmp = icmp sle i64 %middle_offset3, %start_offset4
  br i1 %letmp, label %then5, label %else6

else:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else, %merge23
  %call_tmp26 = call i64 @tl_arena_get_offset()
  store i64 %call_tmp26, ptr %end_offset, align 8
  %end_offset27 = load i64, ptr %end_offset, align 8
  %start_offset28 = load i64, ptr %start_offset, align 8
  %neqtmp29 = icmp ne i64 %end_offset27, %start_offset28
  br i1 %neqtmp29, label %then30, label %else31

then5:                                            ; preds = %then
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.117)
  call void @tl_mem_exit_scope()
  br label %merge7

else6:                                            ; preds = %then
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge7

merge7:                                           ; preds = %else6, %then5
  br i1 true, label %then8, label %else9

then8:                                            ; preds = %merge7
  call void @tl_mem_enter_scope()
  %call_tmp11 = call ptr @tl_arena_alloc(i64 256)
  store ptr %call_tmp11, ptr %ptr2, align 8
  %call_tmp12 = call i64 @tl_arena_get_offset()
  store i64 %call_tmp12, ptr %inner_offset, align 8
  %inner_offset13 = load i64, ptr %inner_offset, align 8
  %middle_offset14 = load i64, ptr %middle_offset, align 8
  %letmp15 = icmp sle i64 %inner_offset13, %middle_offset14
  br i1 %letmp15, label %then16, label %else17

else9:                                            ; preds = %merge7
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge10

merge10:                                          ; preds = %else9, %merge18
  %call_tmp19 = call i64 @tl_arena_get_offset()
  %middle_offset20 = load i64, ptr %middle_offset, align 8
  %neqtmp = icmp ne i64 %call_tmp19, %middle_offset20
  br i1 %neqtmp, label %then21, label %else22

then16:                                           ; preds = %then8
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.118)
  call void @tl_mem_exit_scope()
  br label %merge18

else17:                                           ; preds = %then8
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge18

merge18:                                          ; preds = %else17, %then16
  call void @tl_mem_exit_scope()
  br label %merge10

then21:                                           ; preds = %merge10
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.119)
  %middle_offset24 = load i64, ptr %middle_offset, align 8
  call void @tl_print_i64(i64 %middle_offset24)
  call void @tl_print_string(ptr @str_literal.120)
  %call_tmp25 = call i64 @tl_arena_get_offset()
  call void @tl_print_i64(i64 %call_tmp25)
  call void @tl_mem_exit_scope()
  br label %merge23

else22:                                           ; preds = %merge10
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.121)
  call void @tl_mem_exit_scope()
  br label %merge23

merge23:                                          ; preds = %else22, %then21
  call void @tl_mem_exit_scope()
  br label %merge

then30:                                           ; preds = %merge
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.122)
  %start_offset33 = load i64, ptr %start_offset, align 8
  call void @tl_print_i64(i64 %start_offset33)
  call void @tl_print_string(ptr @str_literal.123)
  %end_offset34 = load i64, ptr %end_offset, align 8
  call void @tl_print_i64(i64 %end_offset34)
  call void @tl_mem_exit_scope()
  br label %merge32

else31:                                           ; preds = %merge
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.124)
  call void @tl_mem_exit_scope()
  br label %merge32

merge32:                                          ; preds = %else31, %then30
  call void @tl_mem_exit_scope()
  ret void
}

define void @main() {
entry:
  %active2 = alloca i64, align 16
  %dummy = alloca ptr, align 16
  %conv_buf = alloca [1 x float], align 4
  %active = alloca i64, align 16
  call void @tl_mem_enter_scope()
  call void @tl_arena_init(i64 666112)
  call void @tl_print_string(ptr @str_literal.125)
  %call_tmp = call i1 @tl_arena_is_active()
  store i1 %call_tmp, ptr %active, align 1
  %active1 = load i1, ptr %active, align 1
  br i1 %active1, label %then, label %else

then:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.126)
  call void @tl_mem_exit_scope()
  br label %merge

else:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.127)
  %arr_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64))
  call void @tl_mem_register_struct(ptr %arr_malloc)
  %elem_ptr = getelementptr inbounds i64, ptr %arr_malloc, i64 0
  store i64 10, ptr %elem_ptr, align 8
  %src = getelementptr inbounds i64, ptr %arr_malloc, i64 0
  %l = load i64, ptr %src, align 8
  %c = sitofp i64 %l to float
  %dst = getelementptr inbounds float, ptr %conv_buf, i64 0
  store float %c, ptr %dst, align 4
  %shape_arr = alloca [1 x i64], align 8
  %shape_ptr = getelementptr inbounds [1 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 1, ptr %shape_ptr, align 8
  %converted_tensor = call ptr @tl_tensor_new(ptr %conv_buf, i64 1, ptr %shape_arr)
  %static_call = call ptr @tl_tensor_randn(ptr %converted_tensor, i1 false)
  store ptr %static_call, ptr %dummy, align 8
  %call_tmp2 = call i1 @tl_arena_is_active()
  store i1 %call_tmp2, ptr %active2, align 1
  %active23 = load i1, ptr %active2, align 1
  br i1 %active23, label %then4, label %else5

merge:                                            ; preds = %merge6, %then
  call void @run_func_test()
  call void @run_loop_test()
  call void @run_nested_test()
  call void @tl_print_string(ptr @str_literal.130)
  call void @tl_mem_exit_scope()
  ret void

then4:                                            ; preds = %else
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.128)
  call void @tl_mem_exit_scope()
  br label %merge6

else5:                                            ; preds = %else
  call void @tl_mem_enter_scope()
  call void @tl_print_string(ptr @str_literal.129)
  call void @tl_arena_init(i64 65536)
  call void @tl_mem_exit_scope()
  br label %merge6

merge6:                                           ; preds = %else5, %then4
  call void @tl_mem_exit_scope()
  br label %merge
}
