; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

%Linear = type { ptr, ptr }
%Embedding = type { ptr }
%LayerNorm = type { ptr, ptr }
%CausalSelfAttention = type { ptr, ptr }
%MLP = type { ptr, ptr }
%Block = type { ptr, ptr, ptr, ptr }

@tensor_name = private unnamed_addr constant [4 x i8] c"qkv\00", align 1
@tensor_name.66 = private unnamed_addr constant [2 x i8] c"Q\00", align 1
@tensor_name.67 = private unnamed_addr constant [2 x i8] c"K\00", align 1
@tensor_name.68 = private unnamed_addr constant [2 x i8] c"V\00", align 1
@tensor_name.69 = private unnamed_addr constant [4 x i8] c"K_T\00", align 1
@tensor_name.70 = private unnamed_addr constant [4 x i8] c"att\00", align 1
@tensor_name.71 = private unnamed_addr constant [4 x i8] c"att\00", align 1
@tensor_name.72 = private unnamed_addr constant [11 x i8] c"masked_att\00", align 1
@tensor_name.73 = private unnamed_addr constant [5 x i8] c"prob\00", align 1
@tensor_name.74 = private unnamed_addr constant [2 x i8] c"y\00", align 1
@tensor_name.75 = private unnamed_addr constant [4 x i8] c"out\00", align 1
@tensor_name.76 = private unnamed_addr constant [2 x i8] c"h\00", align 1
@tensor_name.77 = private unnamed_addr constant [2 x i8] c"h\00", align 1
@tensor_name.78 = private unnamed_addr constant [4 x i8] c"ln1\00", align 1
@tensor_name.79 = private unnamed_addr constant [5 x i8] c"attn\00", align 1
@tensor_name.80 = private unnamed_addr constant [2 x i8] c"x\00", align 1
@tensor_name.81 = private unnamed_addr constant [4 x i8] c"ln2\00", align 1
@tensor_name.82 = private unnamed_addr constant [4 x i8] c"mlp\00", align 1
@tensor_name.83 = private unnamed_addr constant [2 x i8] c"x\00", align 1
@str_literal = private unnamed_addr constant [39 x i8] c"Initializing Transformer components...\00", align 1
@tensor_name.84 = private unnamed_addr constant [4 x i8] c"idx\00", align 1
@str_literal.85 = private unnamed_addr constant [21 x i8] c"Running Embedding...\00", align 1
@tensor_name.86 = private unnamed_addr constant [2 x i8] c"x\00", align 1
@str_literal.87 = private unnamed_addr constant [41 x i8] c"Running Positional Encoding (sin/cos)...\00", align 1
@tensor_name.88 = private unnamed_addr constant [8 x i8] c"pos_enc\00", align 1
@tensor_name.89 = private unnamed_addr constant [2 x i8] c"x\00", align 1
@str_literal.90 = private unnamed_addr constant [29 x i8] c"Running Transformer Block...\00", align 1
@tensor_name.91 = private unnamed_addr constant [4 x i8] c"out\00", align 1
@str_literal.92 = private unnamed_addr constant [48 x i8] c"Output shape verification (by printing tensor):\00", align 1
@str_literal.93 = private unnamed_addr constant [18 x i8] c"Verifying tril...\00", align 1
@tensor_name.94 = private unnamed_addr constant [2 x i8] c"t\00", align 1
@tensor_name.95 = private unnamed_addr constant [7 x i8] c"t_tril\00", align 1
@str_literal.96 = private unnamed_addr constant [22 x i8] c"Verifying sum(dim)...\00", align 1
@tensor_name.97 = private unnamed_addr constant [2 x i8] c"s\00", align 1
@str_literal.98 = private unnamed_addr constant [27 x i8] c"Transformer Test Complete.\00", align 1

declare void @tl_print_i64(i64)

declare void @tl_print_f32(float)

declare void @tl_print_string(ptr)

declare ptr @malloc(i64)

declare ptr @calloc(i64, i64)

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

declare ptr @tl_tensor_pow(ptr, float)

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

declare ptr @tl_tensor_randn(i64, ptr, i1)

declare void @tl_tensor_backward(ptr)

declare ptr @tl_tensor_grad(ptr)

declare ptr @tl_tensor_detach(ptr, i1)

declare ptr @tl_tensor_softmax(ptr, i64)

declare ptr @tl_tensor_cross_entropy(ptr, ptr)

declare void @tl_tensor_sub_assign.1(ptr, ptr)

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

declare void @tl_print_i64.2(i64)

declare void @tl_print_f32.3(float)

declare void @tl_print_string.4(ptr)

declare ptr @malloc.5(i64)

declare ptr @calloc.6(i64, i64)

declare i64 @tl_tensor_dim.7(ptr, i64)

declare float @tl_tensor_get_f32_md.8(ptr, ptr, i64)

declare ptr @tl_tensor_new.9(ptr, i64, ptr)

declare ptr @tl_tensor_sub.10(ptr, ptr)

declare void @tl_tensor_free.11(ptr)

declare ptr @tl_tensor_clone.12(ptr)

declare ptr @tl_tensor_add.13(ptr, ptr)

declare ptr @tl_tensor_mul.14(ptr, ptr)

declare void @tl_tensor_print.15(ptr)

declare float @tl_tensor_get.16(ptr, i64)

declare ptr @tl_tensor_slice.17(ptr, i64, i64)

declare i64 @tl_tensor_len.18(ptr)

declare ptr @tl_tensor_neg.19(ptr)

declare ptr @tl_tensor_transpose.20(ptr, i64, i64)

declare ptr @tl_tensor_pow.21(ptr, float)

declare ptr @tl_tensor_sqrt.22(ptr)

declare ptr @tl_tensor_sin.23(ptr)

declare ptr @tl_tensor_cos.24(ptr)

declare ptr @tl_tensor_relu.25(ptr)

declare ptr @tl_tensor_gelu.26(ptr)

declare ptr @tl_tensor_tril.27(ptr, i32)

declare ptr @tl_tensor_sum_dim.28(ptr, i64, i1)

declare ptr @tl_tensor_embedding.29(ptr, ptr)

declare ptr @tl_tensor_sum.30(ptr)

declare ptr @tl_tensor_div.31(ptr, ptr)

declare ptr @tl_tensor_matmul.32(ptr, ptr)

declare ptr @tl_tensor_exp.33(ptr)

declare ptr @tl_tensor_log.34(ptr)

declare void @tl_tensor_add_assign.35(ptr, ptr)

declare void @tl_tensor_sub_assign.36(ptr, ptr)

declare void @tl_tensor_mul_assign.37(ptr, ptr)

declare void @tl_tensor_div_assign.38(ptr, ptr)

declare void @tl_register_tensor.39(ptr, ptr)

declare i32 @strcmp.40(ptr, ptr)

declare ptr @tl_tensor_randn.41(i64, ptr, i1)

declare void @tl_tensor_backward.42(ptr)

declare ptr @tl_tensor_grad.43(ptr)

declare ptr @tl_tensor_detach.44(ptr, i1)

declare ptr @tl_tensor_softmax.45(ptr, i64)

declare ptr @tl_tensor_cross_entropy.46(ptr, ptr)

declare void @tl_tensor_sub_assign.47(ptr, ptr)

declare ptr @tl_string_concat.48(ptr, ptr)

declare ptr @tl_file_open.49(ptr, ptr)

declare ptr @tl_file_read_string.50(ptr)

declare void @tl_file_write_string.51(ptr, ptr)

declare void @tl_file_close.52(ptr)

declare ptr @tl_path_new.53(ptr)

declare ptr @tl_path_join.54(ptr, ptr)

declare i1 @tl_path_exists.55(ptr)

declare i1 @tl_path_is_dir.56(ptr)

declare i1 @tl_path_is_file.57(ptr)

declare ptr @tl_path_to_string.58(ptr)

declare void @tl_path_free.59(ptr)

declare i1 @tl_http_download.60(ptr, ptr)

declare ptr @tl_http_get.61(ptr)

declare ptr @tl_env_get.62(ptr)

declare void @tl_env_set.63(ptr, ptr)

declare float @tl_system_time.64()

declare void @tl_system_sleep.65(float)

define ptr @tl_Linear_new(i64 %in_dim, i64 %out_dim) {
entry:
  %out_dim2 = alloca i64, align 8
  %in_dim1 = alloca i64, align 8
  store i64 %in_dim, ptr %in_dim1, align 8
  store i64 %out_dim, ptr %out_dim2, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Linear, ptr null, i32 1) to i64))
  %in_dim3 = load i64, ptr %in_dim1, align 8
  %out_dim4 = load i64, ptr %out_dim2, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %in_dim3, ptr %shape_ptr_in, align 8
  %shape_ptr_in5 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 %out_dim4, ptr %shape_ptr_in5, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 true)
  %init_field = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 0
  store ptr %randn_res, ptr %init_field, align 8
  %out_dim6 = load i64, ptr %out_dim2, align 8
  %shape_arr7 = alloca [1 x i64], align 8
  %shape_ptr_in8 = getelementptr inbounds [1 x i64], ptr %shape_arr7, i64 0, i64 0
  store i64 %out_dim6, ptr %shape_ptr_in8, align 8
  %randn_res9 = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr7, i1 true)
  %init_field10 = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 1
  store ptr %randn_res9, ptr %init_field10, align 8
  ret ptr %struct_malloc
}

define ptr @tl_Linear_forward(ptr %self, ptr %x) {
entry:
  %x2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
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
  ret ptr %binop_res
}

define ptr @tl_Embedding_new(i64 %vocab, i64 %d_model) {
entry:
  %d_model2 = alloca i64, align 8
  %vocab1 = alloca i64, align 8
  store i64 %vocab, ptr %vocab1, align 8
  store i64 %d_model, ptr %d_model2, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Embedding, ptr null, i32 1) to i64))
  %vocab3 = load i64, ptr %vocab1, align 8
  %d_model4 = load i64, ptr %d_model2, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %vocab3, ptr %shape_ptr_in, align 8
  %shape_ptr_in5 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 %d_model4, ptr %shape_ptr_in5, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 true)
  %init_field = getelementptr inbounds nuw %Embedding, ptr %struct_malloc, i32 0, i32 0
  store ptr %randn_res, ptr %init_field, align 8
  ret ptr %struct_malloc
}

define ptr @tl_Embedding_forward(ptr %self, ptr %idx) {
entry:
  %idx2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %idx, ptr %idx2, align 8
  %idx3 = load ptr, ptr %idx2, align 8
  %self4 = load ptr, ptr %self1, align 8
  %ptr_weight = getelementptr inbounds nuw %Embedding, ptr %self4, i32 0, i32 0
  %weight = load ptr, ptr %ptr_weight, align 8
  %emb_res = call ptr @tl_tensor_embedding(ptr %idx3, ptr %weight)
  ret ptr %emb_res
}

define ptr @tl_LayerNorm_new(i64 %d_model) {
entry:
  %d_model1 = alloca i64, align 8
  store i64 %d_model, ptr %d_model1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%LayerNorm, ptr null, i32 1) to i64))
  %d_model2 = load i64, ptr %d_model1, align 8
  %shape_arr = alloca [1 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [1 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %d_model2, ptr %shape_ptr_in, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr, i1 true)
  %init_field = getelementptr inbounds nuw %LayerNorm, ptr %struct_malloc, i32 0, i32 0
  store ptr %randn_res, ptr %init_field, align 8
  %d_model3 = load i64, ptr %d_model1, align 8
  %shape_arr4 = alloca [1 x i64], align 8
  %shape_ptr_in5 = getelementptr inbounds [1 x i64], ptr %shape_arr4, i64 0, i64 0
  store i64 %d_model3, ptr %shape_ptr_in5, align 8
  %randn_res6 = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr4, i1 true)
  %scalar_data = alloca float, align 4
  store float 0.000000e+00, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res6, ptr %scalar_tensor)
  %init_field7 = getelementptr inbounds nuw %LayerNorm, ptr %struct_malloc, i32 0, i32 1
  store ptr %binop_res, ptr %init_field7, align 8
  ret ptr %struct_malloc
}

define ptr @tl_LayerNorm_forward(ptr %self, ptr %x) {
entry:
  %x2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %x3 = load ptr, ptr %x2, align 8
  %scalar_data = alloca float, align 4
  store float 0x3FB99999A0000000, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_add(ptr %x3, ptr %scalar_tensor)
  ret ptr %binop_res
}

define ptr @tl_CausalSelfAttention_new(i64 %d_model) {
entry:
  %d_model1 = alloca i64, align 8
  store i64 %d_model, ptr %d_model1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%CausalSelfAttention, ptr null, i32 1) to i64))
  %d_model2 = load i64, ptr %d_model1, align 8
  %d_model3 = load i64, ptr %d_model1, align 8
  %multmp = mul i64 %d_model3, 3
  %static_call = call ptr @tl_Linear_new(i64 %d_model2, i64 %multmp)
  %init_field = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d_model4 = load i64, ptr %d_model1, align 8
  %multmp5 = mul i64 %d_model4, 3
  %d_model6 = load i64, ptr %d_model1, align 8
  %static_call7 = call ptr @tl_Linear_new(i64 %multmp5, i64 %d_model6)
  %init_field8 = getelementptr inbounds nuw %CausalSelfAttention, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call7, ptr %init_field8, align 8
  ret ptr %struct_malloc
}

define ptr @tl_CausalSelfAttention_forward(ptr %self, ptr %x) {
entry:
  %out = alloca ptr, align 8
  %y = alloca ptr, align 8
  %prob = alloca ptr, align 8
  %masked_att = alloca ptr, align 8
  %att14 = alloca ptr, align 8
  %att = alloca ptr, align 8
  %K_T = alloca ptr, align 8
  %V = alloca ptr, align 8
  %K = alloca ptr, align 8
  %Q = alloca ptr, align 8
  %qkv = alloca ptr, align 8
  %x2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_c_attn = getelementptr inbounds nuw %CausalSelfAttention, ptr %self3, i32 0, i32 0
  %c_attn = load ptr, ptr %ptr_c_attn, align 8
  %x4 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_Linear_forward(ptr %c_attn, ptr %x4)
  store ptr %call_method, ptr %qkv, align 8
  call void @tl_register_tensor(ptr @tensor_name, ptr %call_method)
  %qkv5 = load ptr, ptr %qkv, align 8
  %cloned = call ptr @tl_tensor_clone(ptr %qkv5)
  store ptr %cloned, ptr %Q, align 8
  call void @tl_register_tensor(ptr @tensor_name.66, ptr %cloned)
  %qkv6 = load ptr, ptr %qkv, align 8
  %cloned7 = call ptr @tl_tensor_clone(ptr %qkv6)
  store ptr %cloned7, ptr %K, align 8
  call void @tl_register_tensor(ptr @tensor_name.67, ptr %cloned7)
  %qkv8 = load ptr, ptr %qkv, align 8
  %cloned9 = call ptr @tl_tensor_clone(ptr %qkv8)
  store ptr %cloned9, ptr %V, align 8
  call void @tl_register_tensor(ptr @tensor_name.68, ptr %cloned9)
  %K10 = load ptr, ptr %K, align 8
  %transpose_res = call ptr @tl_tensor_transpose(ptr %K10, i64 1, i64 2)
  store ptr %transpose_res, ptr %K_T, align 8
  call void @tl_register_tensor(ptr @tensor_name.69, ptr %transpose_res)
  %Q11 = load ptr, ptr %Q, align 8
  %K_T12 = load ptr, ptr %K_T, align 8
  %matmul_res = call ptr @tl_tensor_matmul(ptr %Q11, ptr %K_T12)
  store ptr %matmul_res, ptr %att, align 8
  call void @tl_register_tensor(ptr @tensor_name.70, ptr %matmul_res)
  %att13 = load ptr, ptr %att, align 8
  %scalar_data = alloca float, align 4
  store float 1.250000e-01, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %att13, ptr %scalar_tensor)
  %old_shadow_val = load ptr, ptr %att, align 8
  call void @tl_tensor_free(ptr %old_shadow_val)
  store ptr %binop_res, ptr %att14, align 8
  call void @tl_register_tensor(ptr @tensor_name.71, ptr %binop_res)
  %att15 = load ptr, ptr %att14, align 8
  %tril_res = call ptr @tl_tensor_tril(ptr %att15, i32 0)
  store ptr %tril_res, ptr %masked_att, align 8
  call void @tl_register_tensor(ptr @tensor_name.72, ptr %tril_res)
  %masked_att16 = load ptr, ptr %masked_att, align 8
  %softmax_res = call ptr @tl_tensor_softmax(ptr %masked_att16, i64 2)
  store ptr %softmax_res, ptr %prob, align 8
  call void @tl_register_tensor(ptr @tensor_name.73, ptr %softmax_res)
  %prob17 = load ptr, ptr %prob, align 8
  %V18 = load ptr, ptr %V, align 8
  %matmul_res19 = call ptr @tl_tensor_matmul(ptr %prob17, ptr %V18)
  store ptr %matmul_res19, ptr %y, align 8
  call void @tl_register_tensor(ptr @tensor_name.74, ptr %matmul_res19)
  %self20 = load ptr, ptr %self1, align 8
  %ptr_c_proj = getelementptr inbounds nuw %CausalSelfAttention, ptr %self20, i32 0, i32 1
  %c_proj = load ptr, ptr %ptr_c_proj, align 8
  %y21 = load ptr, ptr %y, align 8
  %call_method22 = call ptr @tl_Linear_forward(ptr %c_proj, ptr %y21)
  store ptr %call_method22, ptr %out, align 8
  call void @tl_register_tensor(ptr @tensor_name.75, ptr %call_method22)
  %out23 = load ptr, ptr %out, align 8
  %load_for_free = load ptr, ptr %y, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free24 = load ptr, ptr %masked_att, align 8
  call void @tl_tensor_free(ptr %load_for_free24)
  %load_for_free25 = load ptr, ptr %att14, align 8
  call void @tl_tensor_free(ptr %load_for_free25)
  %load_for_free26 = load ptr, ptr %K, align 8
  call void @tl_tensor_free(ptr %load_for_free26)
  %load_for_free27 = load ptr, ptr %Q, align 8
  call void @tl_tensor_free(ptr %load_for_free27)
  %load_for_free28 = load ptr, ptr %V, align 8
  call void @tl_tensor_free(ptr %load_for_free28)
  %load_for_free29 = load ptr, ptr %K_T, align 8
  call void @tl_tensor_free(ptr %load_for_free29)
  %load_for_free30 = load ptr, ptr %prob, align 8
  call void @tl_tensor_free(ptr %load_for_free30)
  %load_for_free31 = load ptr, ptr %qkv, align 8
  call void @tl_tensor_free(ptr %load_for_free31)
  ret ptr %out23
}

define ptr @tl_MLP_new(i64 %d_model) {
entry:
  %d_model1 = alloca i64, align 8
  store i64 %d_model, ptr %d_model1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%MLP, ptr null, i32 1) to i64))
  %d_model2 = load i64, ptr %d_model1, align 8
  %d_model3 = load i64, ptr %d_model1, align 8
  %multmp = mul i64 %d_model3, 4
  %static_call = call ptr @tl_Linear_new(i64 %d_model2, i64 %multmp)
  %init_field = getelementptr inbounds nuw %MLP, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d_model4 = load i64, ptr %d_model1, align 8
  %multmp5 = mul i64 %d_model4, 4
  %d_model6 = load i64, ptr %d_model1, align 8
  %static_call7 = call ptr @tl_Linear_new(i64 %multmp5, i64 %d_model6)
  %init_field8 = getelementptr inbounds nuw %MLP, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call7, ptr %init_field8, align 8
  ret ptr %struct_malloc
}

define ptr @tl_MLP_forward(ptr %self, ptr %x) {
entry:
  %h6 = alloca ptr, align 8
  %h = alloca ptr, align 8
  %x2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_c_fc = getelementptr inbounds nuw %MLP, ptr %self3, i32 0, i32 0
  %c_fc = load ptr, ptr %ptr_c_fc, align 8
  %x4 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_Linear_forward(ptr %c_fc, ptr %x4)
  store ptr %call_method, ptr %h, align 8
  call void @tl_register_tensor(ptr @tensor_name.76, ptr %call_method)
  %h5 = load ptr, ptr %h, align 8
  %gelu_res = call ptr @tl_tensor_gelu(ptr %h5)
  %old_shadow_val = load ptr, ptr %h, align 8
  call void @tl_tensor_free(ptr %old_shadow_val)
  store ptr %gelu_res, ptr %h6, align 8
  call void @tl_register_tensor(ptr @tensor_name.77, ptr %gelu_res)
  %self7 = load ptr, ptr %self1, align 8
  %ptr_c_proj = getelementptr inbounds nuw %MLP, ptr %self7, i32 0, i32 1
  %c_proj = load ptr, ptr %ptr_c_proj, align 8
  %h8 = load ptr, ptr %h6, align 8
  %call_method9 = call ptr @tl_Linear_forward(ptr %c_proj, ptr %h8)
  %load_for_free = load ptr, ptr %h6, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  ret ptr %call_method9
}

define ptr @tl_Block_new(i64 %d_model) {
entry:
  %d_model1 = alloca i64, align 8
  store i64 %d_model, ptr %d_model1, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Block, ptr null, i32 1) to i64))
  %d_model2 = load i64, ptr %d_model1, align 8
  %static_call = call ptr @tl_LayerNorm_new(i64 %d_model2)
  %init_field = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d_model3 = load i64, ptr %d_model1, align 8
  %static_call4 = call ptr @tl_CausalSelfAttention_new(i64 %d_model3)
  %init_field5 = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call4, ptr %init_field5, align 8
  %d_model6 = load i64, ptr %d_model1, align 8
  %static_call7 = call ptr @tl_LayerNorm_new(i64 %d_model6)
  %init_field8 = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 2
  store ptr %static_call7, ptr %init_field8, align 8
  %d_model9 = load i64, ptr %d_model1, align 8
  %static_call10 = call ptr @tl_MLP_new(i64 %d_model9)
  %init_field11 = getelementptr inbounds nuw %Block, ptr %struct_malloc, i32 0, i32 3
  store ptr %static_call10, ptr %init_field11, align 8
  ret ptr %struct_malloc
}

define ptr @tl_Block_forward(ptr %self, ptr %x) {
entry:
  %x24 = alloca ptr, align 8
  %mlp20 = alloca ptr, align 8
  %ln216 = alloca ptr, align 8
  %x12 = alloca ptr, align 8
  %attn9 = alloca ptr, align 8
  %ln15 = alloca ptr, align 8
  %x2 = alloca ptr, align 8
  %self1 = alloca ptr, align 8
  store ptr %self, ptr %self1, align 8
  store ptr %x, ptr %x2, align 8
  %self3 = load ptr, ptr %self1, align 8
  %ptr_ln1 = getelementptr inbounds nuw %Block, ptr %self3, i32 0, i32 0
  %ln1 = load ptr, ptr %ptr_ln1, align 8
  %x4 = load ptr, ptr %x2, align 8
  %call_method = call ptr @tl_LayerNorm_forward(ptr %ln1, ptr %x4)
  store ptr %call_method, ptr %ln15, align 8
  call void @tl_register_tensor(ptr @tensor_name.78, ptr %call_method)
  %self6 = load ptr, ptr %self1, align 8
  %ptr_attn = getelementptr inbounds nuw %Block, ptr %self6, i32 0, i32 1
  %attn = load ptr, ptr %ptr_attn, align 8
  %ln17 = load ptr, ptr %ln15, align 8
  %call_method8 = call ptr @tl_CausalSelfAttention_forward(ptr %attn, ptr %ln17)
  store ptr %call_method8, ptr %attn9, align 8
  call void @tl_register_tensor(ptr @tensor_name.79, ptr %call_method8)
  %x10 = load ptr, ptr %x2, align 8
  %attn11 = load ptr, ptr %attn9, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %x10, ptr %attn11)
  store ptr %binop_res, ptr %x12, align 8
  call void @tl_register_tensor(ptr @tensor_name.80, ptr %binop_res)
  %self13 = load ptr, ptr %self1, align 8
  %ptr_ln2 = getelementptr inbounds nuw %Block, ptr %self13, i32 0, i32 2
  %ln2 = load ptr, ptr %ptr_ln2, align 8
  %x14 = load ptr, ptr %x12, align 8
  %call_method15 = call ptr @tl_LayerNorm_forward(ptr %ln2, ptr %x14)
  store ptr %call_method15, ptr %ln216, align 8
  call void @tl_register_tensor(ptr @tensor_name.81, ptr %call_method15)
  %self17 = load ptr, ptr %self1, align 8
  %ptr_mlp = getelementptr inbounds nuw %Block, ptr %self17, i32 0, i32 3
  %mlp = load ptr, ptr %ptr_mlp, align 8
  %ln218 = load ptr, ptr %ln216, align 8
  %call_method19 = call ptr @tl_MLP_forward(ptr %mlp, ptr %ln218)
  store ptr %call_method19, ptr %mlp20, align 8
  call void @tl_register_tensor(ptr @tensor_name.82, ptr %call_method19)
  %x21 = load ptr, ptr %x12, align 8
  %mlp22 = load ptr, ptr %mlp20, align 8
  %binop_res23 = call ptr @tl_tensor_add(ptr %x21, ptr %mlp22)
  %old_shadow_val = load ptr, ptr %x12, align 8
  call void @tl_tensor_free(ptr %old_shadow_val)
  store ptr %binop_res23, ptr %x24, align 8
  call void @tl_register_tensor(ptr @tensor_name.83, ptr %binop_res23)
  %x25 = load ptr, ptr %x24, align 8
  %load_for_free = load ptr, ptr %mlp20, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free26 = load ptr, ptr %ln216, align 8
  call void @tl_tensor_free(ptr %load_for_free26)
  %load_for_free27 = load ptr, ptr %ln15, align 8
  call void @tl_tensor_free(ptr %load_for_free27)
  %load_for_free28 = load ptr, ptr %attn9, align 8
  call void @tl_tensor_free(ptr %load_for_free28)
  ret ptr %x25
}

define void @main() {
entry:
  %s = alloca ptr, align 8
  %t_tril = alloca ptr, align 8
  %t = alloca ptr, align 8
  %out = alloca ptr, align 8
  %block = alloca ptr, align 8
  %x13 = alloca ptr, align 8
  %pos_enc = alloca ptr, align 8
  %x = alloca ptr, align 8
  %idx = alloca ptr, align 8
  %tok_emb = alloca ptr, align 8
  %vocab_size = alloca i64, align 8
  %d_model = alloca i64, align 8
  call void @tl_print_string(ptr @str_literal)
  store i64 16, ptr %d_model, align 8
  store i64 100, ptr %vocab_size, align 8
  %vocab_size1 = load i64, ptr %vocab_size, align 8
  %d_model2 = load i64, ptr %d_model, align 8
  %static_call = call ptr @tl_Embedding_new(i64 %vocab_size1, i64 %d_model2)
  store ptr %static_call, ptr %tok_emb, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 2, ptr %shape_ptr_in, align 8
  %shape_ptr_in3 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 4, ptr %shape_ptr_in3, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 false)
  %scalar_data = alloca float, align 4
  store float 0.000000e+00, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor)
  store ptr %binop_res, ptr %idx, align 8
  call void @tl_register_tensor(ptr @tensor_name.84, ptr %binop_res)
  call void @tl_print_string(ptr @str_literal.85)
  %tok_emb4 = load ptr, ptr %tok_emb, align 8
  %idx5 = load ptr, ptr %idx, align 8
  %call_method = call ptr @tl_Embedding_forward(ptr %tok_emb4, ptr %idx5)
  store ptr %call_method, ptr %x, align 8
  call void @tl_register_tensor(ptr @tensor_name.86, ptr %call_method)
  %x6 = load ptr, ptr %x, align 8
  call void @tl_tensor_print(ptr %x6)
  call void @tl_print_string(ptr @str_literal.87)
  %x7 = load ptr, ptr %x, align 8
  %sin_res = call ptr @tl_tensor_sin(ptr %x7)
  %x8 = load ptr, ptr %x, align 8
  %cos_res = call ptr @tl_tensor_cos(ptr %x8)
  %binop_res9 = call ptr @tl_tensor_add(ptr %sin_res, ptr %cos_res)
  store ptr %binop_res9, ptr %pos_enc, align 8
  call void @tl_register_tensor(ptr @tensor_name.88, ptr %binop_res9)
  %x10 = load ptr, ptr %x, align 8
  %pos_enc11 = load ptr, ptr %pos_enc, align 8
  %binop_res12 = call ptr @tl_tensor_add(ptr %x10, ptr %pos_enc11)
  %old_shadow_val = load ptr, ptr %x, align 8
  call void @tl_tensor_free(ptr %old_shadow_val)
  store ptr %binop_res12, ptr %x13, align 8
  call void @tl_register_tensor(ptr @tensor_name.89, ptr %binop_res12)
  call void @tl_print_string(ptr @str_literal.90)
  %d_model14 = load i64, ptr %d_model, align 8
  %static_call15 = call ptr @tl_Block_new(i64 %d_model14)
  store ptr %static_call15, ptr %block, align 8
  %block16 = load ptr, ptr %block, align 8
  %x17 = load ptr, ptr %x13, align 8
  %call_method18 = call ptr @tl_Block_forward(ptr %block16, ptr %x17)
  store ptr %call_method18, ptr %out, align 8
  call void @tl_register_tensor(ptr @tensor_name.91, ptr %call_method18)
  call void @tl_print_string(ptr @str_literal.92)
  %out19 = load ptr, ptr %out, align 8
  call void @tl_tensor_print(ptr %out19)
  call void @tl_print_string(ptr @str_literal.93)
  %shape_arr20 = alloca [2 x i64], align 8
  %shape_ptr_in21 = getelementptr inbounds [2 x i64], ptr %shape_arr20, i64 0, i64 0
  store i64 3, ptr %shape_ptr_in21, align 8
  %shape_ptr_in22 = getelementptr inbounds [2 x i64], ptr %shape_arr20, i64 0, i64 1
  store i64 3, ptr %shape_ptr_in22, align 8
  %randn_res23 = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr20, i1 false)
  store ptr %randn_res23, ptr %t, align 8
  call void @tl_register_tensor(ptr @tensor_name.94, ptr %randn_res23)
  %t24 = load ptr, ptr %t, align 8
  %tril_res = call ptr @tl_tensor_tril(ptr %t24, i32 0)
  store ptr %tril_res, ptr %t_tril, align 8
  call void @tl_register_tensor(ptr @tensor_name.95, ptr %tril_res)
  %t_tril25 = load ptr, ptr %t_tril, align 8
  call void @tl_tensor_print(ptr %t_tril25)
  call void @tl_print_string(ptr @str_literal.96)
  %out26 = load ptr, ptr %out, align 8
  %sum_dim_res = call ptr @tl_tensor_sum_dim(ptr %out26, i64 1, i1 false)
  store ptr %sum_dim_res, ptr %s, align 8
  call void @tl_register_tensor(ptr @tensor_name.97, ptr %sum_dim_res)
  %s27 = load ptr, ptr %s, align 8
  call void @tl_tensor_print(ptr %s27)
  call void @tl_print_string(ptr @str_literal.98)
  %load_for_free = load ptr, ptr %x13, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free28 = load ptr, ptr %t_tril, align 8
  call void @tl_tensor_free(ptr %load_for_free28)
  %load_for_free29 = load ptr, ptr %t, align 8
  call void @tl_tensor_free(ptr %load_for_free29)
  %load_for_free30 = load ptr, ptr %pos_enc, align 8
  call void @tl_tensor_free(ptr %load_for_free30)
  %load_for_free31 = load ptr, ptr %idx, align 8
  call void @tl_tensor_free(ptr %load_for_free31)
  %load_for_free32 = load ptr, ptr %out, align 8
  call void @tl_tensor_free(ptr %load_for_free32)
  %load_for_free33 = load ptr, ptr %s, align 8
  call void @tl_tensor_free(ptr %load_for_free33)
  ret void
}
