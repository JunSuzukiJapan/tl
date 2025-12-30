; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

%Linear = type { ptr, ptr }
%Adam = type { float, float, float, float, i64, ptr, ptr, ptr, ptr }

@tensor_name = private unnamed_addr constant [2 x i8] c"g\00", align 1
@tensor_name.42 = private unnamed_addr constant [8 x i8] c"term2_m\00", align 1
@tensor_name.43 = private unnamed_addr constant [3 x i8] c"g2\00", align 1
@tensor_name.44 = private unnamed_addr constant [8 x i8] c"term2_v\00", align 1
@tensor_name.45 = private unnamed_addr constant [7 x i8] c"v_sqrt\00", align 1
@tensor_name.46 = private unnamed_addr constant [6 x i8] c"denom\00", align 1
@tensor_name.47 = private unnamed_addr constant [10 x i8] c"step_size\00", align 1
@tensor_name.48 = private unnamed_addr constant [7 x i8] c"update\00", align 1
@tensor_name.49 = private unnamed_addr constant [7 x i8] c"w_next\00", align 1
@tensor_name.50 = private unnamed_addr constant [7 x i8] c"w_leaf\00", align 1
@tensor_name.51 = private unnamed_addr constant [3 x i8] c"gb\00", align 1
@tensor_name.52 = private unnamed_addr constant [9 x i8] c"term2_mb\00", align 1
@tensor_name.53 = private unnamed_addr constant [4 x i8] c"gb2\00", align 1
@tensor_name.54 = private unnamed_addr constant [9 x i8] c"term2_vb\00", align 1
@tensor_name.55 = private unnamed_addr constant [8 x i8] c"vb_sqrt\00", align 1
@tensor_name.56 = private unnamed_addr constant [7 x i8] c"denomb\00", align 1
@tensor_name.57 = private unnamed_addr constant [11 x i8] c"step_sizeb\00", align 1
@tensor_name.58 = private unnamed_addr constant [8 x i8] c"updateb\00", align 1
@tensor_name.59 = private unnamed_addr constant [7 x i8] c"b_next\00", align 1
@tensor_name.60 = private unnamed_addr constant [7 x i8] c"b_leaf\00", align 1
@tensor_name.61 = private unnamed_addr constant [2 x i8] c"W\00", align 1
@tensor_name.62 = private unnamed_addr constant [2 x i8] c"b\00", align 1
@tensor_name.63 = private unnamed_addr constant [4 x i8] c"m_W\00", align 1
@tensor_name.64 = private unnamed_addr constant [4 x i8] c"v_W\00", align 1
@tensor_name.65 = private unnamed_addr constant [4 x i8] c"m_b\00", align 1
@tensor_name.66 = private unnamed_addr constant [4 x i8] c"v_b\00", align 1
@tensor_name.67 = private unnamed_addr constant [2 x i8] c"X\00", align 1
@tensor_name.68 = private unnamed_addr constant [7 x i8] c"y_pred\00", align 1
@tensor_name.69 = private unnamed_addr constant [5 x i8] c"loss\00", align 1
@tensor_name.70 = private unnamed_addr constant [7 x i8] c"y_pred\00", align 1
@tensor_name.71 = private unnamed_addr constant [5 x i8] c"loss\00", align 1
@tensor_name.72 = private unnamed_addr constant [7 x i8] c"y_pred\00", align 1
@tensor_name.73 = private unnamed_addr constant [5 x i8] c"loss\00", align 1

declare void @tl_print_i64(i64)

declare void @tl_print_f32(float)

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

declare ptr @tl_tensor_reshape(ptr, ptr)

declare ptr @tl_tensor_sum(ptr)

declare ptr @tl_tensor_div(ptr, ptr)

declare ptr @tl_tensor_sub.1(ptr, ptr)

declare ptr @tl_tensor_matmul(ptr, ptr)

declare ptr @tl_tensor_exp(ptr)

declare ptr @tl_tensor_log(ptr)

declare ptr @tl_tensor_sqrt(ptr)

declare ptr @tl_tensor_pow(ptr, ptr)

declare void @tl_tensor_add_assign(ptr, ptr)

declare void @tl_tensor_sub_assign(ptr, ptr)

declare void @tl_tensor_mul_assign(ptr, ptr)

declare void @tl_tensor_div_assign(ptr, ptr)

declare void @tl_register_tensor(ptr, ptr)

declare ptr @tl_tensor_randn(i64, ptr, i1)

declare void @tl_tensor_backward(ptr)

declare ptr @tl_tensor_grad(ptr)

declare ptr @tl_tensor_detach(ptr, i1)

declare ptr @tl_tensor_softmax(ptr, i64)

declare ptr @tl_tensor_cross_entropy(ptr, ptr)

declare void @tl_tensor_sub_assign.2(ptr, ptr)

declare void @tl_print_i64.3(i64)

declare void @tl_print_f32.4(float)

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

declare ptr @tl_tensor_reshape.21(ptr, ptr)

declare ptr @tl_tensor_sum.22(ptr)

declare ptr @tl_tensor_div.23(ptr, ptr)

declare ptr @tl_tensor_sub.24(ptr, ptr)

declare ptr @tl_tensor_matmul.25(ptr, ptr)

declare ptr @tl_tensor_exp.26(ptr)

declare ptr @tl_tensor_log.27(ptr)

declare ptr @tl_tensor_sqrt.28(ptr)

declare ptr @tl_tensor_pow.29(ptr, ptr)

declare void @tl_tensor_add_assign.30(ptr, ptr)

declare void @tl_tensor_sub_assign.31(ptr, ptr)

declare void @tl_tensor_mul_assign.32(ptr, ptr)

declare void @tl_tensor_div_assign.33(ptr, ptr)

declare void @tl_register_tensor.34(ptr, ptr)

declare ptr @tl_tensor_randn.35(i64, ptr, i1)

declare void @tl_tensor_backward.36(ptr)

declare ptr @tl_tensor_grad.37(ptr)

declare ptr @tl_tensor_detach.38(ptr, i1)

declare ptr @tl_tensor_softmax.39(ptr, i64)

declare ptr @tl_tensor_cross_entropy.40(ptr, ptr)

declare void @tl_tensor_sub_assign.41(ptr, ptr)

define void @Adam_step(ptr %0, ptr %1) {
entry:
  %b_leaf = alloca ptr, align 8
  %b_next = alloca ptr, align 8
  %updateb = alloca ptr, align 8
  %step_sizeb = alloca ptr, align 8
  %denomb = alloca ptr, align 8
  %vb_sqrt = alloca ptr, align 8
  %term2_vb = alloca ptr, align 8
  %gb2 = alloca ptr, align 8
  %term2_mb = alloca ptr, align 8
  %gb = alloca ptr, align 8
  %w_leaf = alloca ptr, align 8
  %w_next = alloca ptr, align 8
  %update = alloca ptr, align 8
  %step_size = alloca ptr, align 8
  %denom = alloca ptr, align 8
  %v_sqrt = alloca ptr, align 8
  %term2_v = alloca ptr, align 8
  %one_minus_beta2 = alloca float, align 4
  %g2 = alloca ptr, align 8
  %term2_m = alloca ptr, align 8
  %one_minus_beta1 = alloca float, align 4
  %g = alloca ptr, align 8
  %model = alloca ptr, align 8
  %self = alloca ptr, align 8
  store ptr %0, ptr %self, align 8
  store ptr %1, ptr %model, align 8
  %model1 = load ptr, ptr %model, align 8
  %ptr_W = getelementptr inbounds nuw %Linear, ptr %model1, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  %grad_res = call ptr @tl_tensor_grad(ptr %W)
  store ptr %grad_res, ptr %g, align 8
  call void @tl_register_tensor(ptr @tensor_name, ptr %grad_res)
  %self2 = load ptr, ptr %self, align 8
  %ptr_m_W = getelementptr inbounds nuw %Adam, ptr %self2, i32 0, i32 5
  %m_W = load ptr, ptr %ptr_m_W, align 8
  %self3 = load ptr, ptr %self, align 8
  %ptr_beta1 = getelementptr inbounds nuw %Adam, ptr %self3, i32 0, i32 1
  %beta1 = load float, ptr %ptr_beta1, align 4
  %scalar_data = alloca float, align 4
  store float %beta1, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  call void @tl_tensor_mul_assign(ptr %m_W, ptr %scalar_tensor)
  %self4 = load ptr, ptr %self, align 8
  %ptr_beta15 = getelementptr inbounds nuw %Adam, ptr %self4, i32 0, i32 1
  %beta16 = load float, ptr %ptr_beta15, align 4
  %fsubtmp = fsub float 1.000000e+00, %beta16
  store float %fsubtmp, ptr %one_minus_beta1, align 4
  %g7 = load ptr, ptr %g, align 8
  %one_minus_beta18 = load float, ptr %one_minus_beta1, align 4
  %scalar_data9 = alloca float, align 4
  store float %one_minus_beta18, ptr %scalar_data9, align 4
  %scalar_shape10 = alloca i64, i64 0, align 8
  %scalar_tensor11 = call ptr @tl_tensor_new(ptr %scalar_data9, i64 0, ptr %scalar_shape10)
  %binop_res = call ptr @tl_tensor_mul(ptr %g7, ptr %scalar_tensor11)
  store ptr %binop_res, ptr %term2_m, align 8
  call void @tl_register_tensor(ptr @tensor_name.42, ptr %binop_res)
  %self12 = load ptr, ptr %self, align 8
  %ptr_m_W13 = getelementptr inbounds nuw %Adam, ptr %self12, i32 0, i32 5
  %m_W14 = load ptr, ptr %ptr_m_W13, align 8
  %term2_m15 = load ptr, ptr %term2_m, align 8
  call void @tl_tensor_add_assign(ptr %m_W14, ptr %term2_m15)
  %self16 = load ptr, ptr %self, align 8
  %ptr_v_W = getelementptr inbounds nuw %Adam, ptr %self16, i32 0, i32 6
  %v_W = load ptr, ptr %ptr_v_W, align 8
  %self17 = load ptr, ptr %self, align 8
  %ptr_beta2 = getelementptr inbounds nuw %Adam, ptr %self17, i32 0, i32 2
  %beta2 = load float, ptr %ptr_beta2, align 4
  %scalar_data18 = alloca float, align 4
  store float %beta2, ptr %scalar_data18, align 4
  %scalar_shape19 = alloca i64, align 8
  %scalar_tensor20 = call ptr @tl_tensor_new(ptr %scalar_data18, i64 0, ptr %scalar_shape19)
  call void @tl_tensor_mul_assign(ptr %v_W, ptr %scalar_tensor20)
  %g21 = load ptr, ptr %g, align 8
  %g22 = load ptr, ptr %g, align 8
  %binop_res23 = call ptr @tl_tensor_mul(ptr %g21, ptr %g22)
  store ptr %binop_res23, ptr %g2, align 8
  call void @tl_register_tensor(ptr @tensor_name.43, ptr %binop_res23)
  %self24 = load ptr, ptr %self, align 8
  %ptr_beta225 = getelementptr inbounds nuw %Adam, ptr %self24, i32 0, i32 2
  %beta226 = load float, ptr %ptr_beta225, align 4
  %fsubtmp27 = fsub float 1.000000e+00, %beta226
  store float %fsubtmp27, ptr %one_minus_beta2, align 4
  %g228 = load ptr, ptr %g2, align 8
  %one_minus_beta229 = load float, ptr %one_minus_beta2, align 4
  %scalar_data30 = alloca float, align 4
  store float %one_minus_beta229, ptr %scalar_data30, align 4
  %scalar_shape31 = alloca i64, i64 0, align 8
  %scalar_tensor32 = call ptr @tl_tensor_new(ptr %scalar_data30, i64 0, ptr %scalar_shape31)
  %binop_res33 = call ptr @tl_tensor_mul(ptr %g228, ptr %scalar_tensor32)
  store ptr %binop_res33, ptr %term2_v, align 8
  call void @tl_register_tensor(ptr @tensor_name.44, ptr %binop_res33)
  %self34 = load ptr, ptr %self, align 8
  %ptr_v_W35 = getelementptr inbounds nuw %Adam, ptr %self34, i32 0, i32 6
  %v_W36 = load ptr, ptr %ptr_v_W35, align 8
  %term2_v37 = load ptr, ptr %term2_v, align 8
  call void @tl_tensor_add_assign(ptr %v_W36, ptr %term2_v37)
  %self38 = load ptr, ptr %self, align 8
  %ptr_v_W39 = getelementptr inbounds nuw %Adam, ptr %self38, i32 0, i32 6
  %v_W40 = load ptr, ptr %ptr_v_W39, align 8
  %sqrt_res = call ptr @tl_tensor_sqrt(ptr %v_W40)
  store ptr %sqrt_res, ptr %v_sqrt, align 8
  call void @tl_register_tensor(ptr @tensor_name.45, ptr %sqrt_res)
  %v_sqrt41 = load ptr, ptr %v_sqrt, align 8
  %self42 = load ptr, ptr %self, align 8
  %ptr_eps = getelementptr inbounds nuw %Adam, ptr %self42, i32 0, i32 3
  %eps = load float, ptr %ptr_eps, align 4
  %scalar_data43 = alloca float, align 4
  store float %eps, ptr %scalar_data43, align 4
  %scalar_shape44 = alloca i64, i64 0, align 8
  %scalar_tensor45 = call ptr @tl_tensor_new(ptr %scalar_data43, i64 0, ptr %scalar_shape44)
  %binop_res46 = call ptr @tl_tensor_add(ptr %v_sqrt41, ptr %scalar_tensor45)
  store ptr %binop_res46, ptr %denom, align 8
  call void @tl_register_tensor(ptr @tensor_name.46, ptr %binop_res46)
  %self47 = load ptr, ptr %self, align 8
  %ptr_m_W48 = getelementptr inbounds nuw %Adam, ptr %self47, i32 0, i32 5
  %m_W49 = load ptr, ptr %ptr_m_W48, align 8
  %denom50 = load ptr, ptr %denom, align 8
  %binop_res51 = call ptr @tl_tensor_div(ptr %m_W49, ptr %denom50)
  store ptr %binop_res51, ptr %step_size, align 8
  call void @tl_register_tensor(ptr @tensor_name.47, ptr %binop_res51)
  %step_size52 = load ptr, ptr %step_size, align 8
  %self53 = load ptr, ptr %self, align 8
  %ptr_lr = getelementptr inbounds nuw %Adam, ptr %self53, i32 0, i32 0
  %lr = load float, ptr %ptr_lr, align 4
  %scalar_data54 = alloca float, align 4
  store float %lr, ptr %scalar_data54, align 4
  %scalar_shape55 = alloca i64, i64 0, align 8
  %scalar_tensor56 = call ptr @tl_tensor_new(ptr %scalar_data54, i64 0, ptr %scalar_shape55)
  %binop_res57 = call ptr @tl_tensor_mul(ptr %step_size52, ptr %scalar_tensor56)
  store ptr %binop_res57, ptr %update, align 8
  call void @tl_register_tensor(ptr @tensor_name.48, ptr %binop_res57)
  %model58 = load ptr, ptr %model, align 8
  %ptr_W59 = getelementptr inbounds nuw %Linear, ptr %model58, i32 0, i32 0
  %W60 = load ptr, ptr %ptr_W59, align 8
  %update61 = load ptr, ptr %update, align 8
  %binop_res62 = call ptr @tl_tensor_sub(ptr %W60, ptr %update61)
  store ptr %binop_res62, ptr %w_next, align 8
  call void @tl_register_tensor(ptr @tensor_name.49, ptr %binop_res62)
  %w_next63 = load ptr, ptr %w_next, align 8
  %detach_res = call ptr @tl_tensor_detach(ptr %w_next63, i1 true)
  store ptr %detach_res, ptr %w_leaf, align 8
  call void @tl_register_tensor(ptr @tensor_name.50, ptr %detach_res)
  %model64 = load ptr, ptr %model, align 8
  %ptr_W65 = getelementptr inbounds nuw %Linear, ptr %model64, i32 0, i32 0
  %w_leaf66 = load ptr, ptr %w_leaf, align 8
  %old_field_val = load ptr, ptr %ptr_W65, align 8
  call void @tl_tensor_free(ptr %old_field_val)
  store ptr %w_leaf66, ptr %ptr_W65, align 8
  %model67 = load ptr, ptr %model, align 8
  %ptr_b = getelementptr inbounds nuw %Linear, ptr %model67, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %grad_res68 = call ptr @tl_tensor_grad(ptr %b)
  store ptr %grad_res68, ptr %gb, align 8
  call void @tl_register_tensor(ptr @tensor_name.51, ptr %grad_res68)
  %self69 = load ptr, ptr %self, align 8
  %ptr_m_b = getelementptr inbounds nuw %Adam, ptr %self69, i32 0, i32 7
  %m_b = load ptr, ptr %ptr_m_b, align 8
  %self70 = load ptr, ptr %self, align 8
  %ptr_beta171 = getelementptr inbounds nuw %Adam, ptr %self70, i32 0, i32 1
  %beta172 = load float, ptr %ptr_beta171, align 4
  %scalar_data73 = alloca float, align 4
  store float %beta172, ptr %scalar_data73, align 4
  %scalar_shape74 = alloca i64, align 8
  %scalar_tensor75 = call ptr @tl_tensor_new(ptr %scalar_data73, i64 0, ptr %scalar_shape74)
  call void @tl_tensor_mul_assign(ptr %m_b, ptr %scalar_tensor75)
  %gb76 = load ptr, ptr %gb, align 8
  %one_minus_beta177 = load float, ptr %one_minus_beta1, align 4
  %scalar_data78 = alloca float, align 4
  store float %one_minus_beta177, ptr %scalar_data78, align 4
  %scalar_shape79 = alloca i64, i64 0, align 8
  %scalar_tensor80 = call ptr @tl_tensor_new(ptr %scalar_data78, i64 0, ptr %scalar_shape79)
  %binop_res81 = call ptr @tl_tensor_mul(ptr %gb76, ptr %scalar_tensor80)
  store ptr %binop_res81, ptr %term2_mb, align 8
  call void @tl_register_tensor(ptr @tensor_name.52, ptr %binop_res81)
  %self82 = load ptr, ptr %self, align 8
  %ptr_m_b83 = getelementptr inbounds nuw %Adam, ptr %self82, i32 0, i32 7
  %m_b84 = load ptr, ptr %ptr_m_b83, align 8
  %term2_mb85 = load ptr, ptr %term2_mb, align 8
  call void @tl_tensor_add_assign(ptr %m_b84, ptr %term2_mb85)
  %self86 = load ptr, ptr %self, align 8
  %ptr_v_b = getelementptr inbounds nuw %Adam, ptr %self86, i32 0, i32 8
  %v_b = load ptr, ptr %ptr_v_b, align 8
  %self87 = load ptr, ptr %self, align 8
  %ptr_beta288 = getelementptr inbounds nuw %Adam, ptr %self87, i32 0, i32 2
  %beta289 = load float, ptr %ptr_beta288, align 4
  %scalar_data90 = alloca float, align 4
  store float %beta289, ptr %scalar_data90, align 4
  %scalar_shape91 = alloca i64, align 8
  %scalar_tensor92 = call ptr @tl_tensor_new(ptr %scalar_data90, i64 0, ptr %scalar_shape91)
  call void @tl_tensor_mul_assign(ptr %v_b, ptr %scalar_tensor92)
  %gb93 = load ptr, ptr %gb, align 8
  %gb94 = load ptr, ptr %gb, align 8
  %binop_res95 = call ptr @tl_tensor_mul(ptr %gb93, ptr %gb94)
  store ptr %binop_res95, ptr %gb2, align 8
  call void @tl_register_tensor(ptr @tensor_name.53, ptr %binop_res95)
  %gb296 = load ptr, ptr %gb2, align 8
  %one_minus_beta297 = load float, ptr %one_minus_beta2, align 4
  %scalar_data98 = alloca float, align 4
  store float %one_minus_beta297, ptr %scalar_data98, align 4
  %scalar_shape99 = alloca i64, i64 0, align 8
  %scalar_tensor100 = call ptr @tl_tensor_new(ptr %scalar_data98, i64 0, ptr %scalar_shape99)
  %binop_res101 = call ptr @tl_tensor_mul(ptr %gb296, ptr %scalar_tensor100)
  store ptr %binop_res101, ptr %term2_vb, align 8
  call void @tl_register_tensor(ptr @tensor_name.54, ptr %binop_res101)
  %self102 = load ptr, ptr %self, align 8
  %ptr_v_b103 = getelementptr inbounds nuw %Adam, ptr %self102, i32 0, i32 8
  %v_b104 = load ptr, ptr %ptr_v_b103, align 8
  %term2_vb105 = load ptr, ptr %term2_vb, align 8
  call void @tl_tensor_add_assign(ptr %v_b104, ptr %term2_vb105)
  %self106 = load ptr, ptr %self, align 8
  %ptr_v_b107 = getelementptr inbounds nuw %Adam, ptr %self106, i32 0, i32 8
  %v_b108 = load ptr, ptr %ptr_v_b107, align 8
  %sqrt_res109 = call ptr @tl_tensor_sqrt(ptr %v_b108)
  store ptr %sqrt_res109, ptr %vb_sqrt, align 8
  call void @tl_register_tensor(ptr @tensor_name.55, ptr %sqrt_res109)
  %vb_sqrt110 = load ptr, ptr %vb_sqrt, align 8
  %self111 = load ptr, ptr %self, align 8
  %ptr_eps112 = getelementptr inbounds nuw %Adam, ptr %self111, i32 0, i32 3
  %eps113 = load float, ptr %ptr_eps112, align 4
  %scalar_data114 = alloca float, align 4
  store float %eps113, ptr %scalar_data114, align 4
  %scalar_shape115 = alloca i64, i64 0, align 8
  %scalar_tensor116 = call ptr @tl_tensor_new(ptr %scalar_data114, i64 0, ptr %scalar_shape115)
  %binop_res117 = call ptr @tl_tensor_add(ptr %vb_sqrt110, ptr %scalar_tensor116)
  store ptr %binop_res117, ptr %denomb, align 8
  call void @tl_register_tensor(ptr @tensor_name.56, ptr %binop_res117)
  %self118 = load ptr, ptr %self, align 8
  %ptr_m_b119 = getelementptr inbounds nuw %Adam, ptr %self118, i32 0, i32 7
  %m_b120 = load ptr, ptr %ptr_m_b119, align 8
  %denomb121 = load ptr, ptr %denomb, align 8
  %binop_res122 = call ptr @tl_tensor_div(ptr %m_b120, ptr %denomb121)
  store ptr %binop_res122, ptr %step_sizeb, align 8
  call void @tl_register_tensor(ptr @tensor_name.57, ptr %binop_res122)
  %step_sizeb123 = load ptr, ptr %step_sizeb, align 8
  %self124 = load ptr, ptr %self, align 8
  %ptr_lr125 = getelementptr inbounds nuw %Adam, ptr %self124, i32 0, i32 0
  %lr126 = load float, ptr %ptr_lr125, align 4
  %scalar_data127 = alloca float, align 4
  store float %lr126, ptr %scalar_data127, align 4
  %scalar_shape128 = alloca i64, i64 0, align 8
  %scalar_tensor129 = call ptr @tl_tensor_new(ptr %scalar_data127, i64 0, ptr %scalar_shape128)
  %binop_res130 = call ptr @tl_tensor_mul(ptr %step_sizeb123, ptr %scalar_tensor129)
  store ptr %binop_res130, ptr %updateb, align 8
  call void @tl_register_tensor(ptr @tensor_name.58, ptr %binop_res130)
  %model131 = load ptr, ptr %model, align 8
  %ptr_b132 = getelementptr inbounds nuw %Linear, ptr %model131, i32 0, i32 1
  %b133 = load ptr, ptr %ptr_b132, align 8
  %updateb134 = load ptr, ptr %updateb, align 8
  %binop_res135 = call ptr @tl_tensor_sub(ptr %b133, ptr %updateb134)
  store ptr %binop_res135, ptr %b_next, align 8
  call void @tl_register_tensor(ptr @tensor_name.59, ptr %binop_res135)
  %b_next136 = load ptr, ptr %b_next, align 8
  %detach_res137 = call ptr @tl_tensor_detach(ptr %b_next136, i1 true)
  store ptr %detach_res137, ptr %b_leaf, align 8
  call void @tl_register_tensor(ptr @tensor_name.60, ptr %detach_res137)
  %model138 = load ptr, ptr %model, align 8
  %ptr_b139 = getelementptr inbounds nuw %Linear, ptr %model138, i32 0, i32 1
  %b_leaf140 = load ptr, ptr %b_leaf, align 8
  %old_field_val141 = load ptr, ptr %ptr_b139, align 8
  call void @tl_tensor_free(ptr %old_field_val141)
  store ptr %b_leaf140, ptr %ptr_b139, align 8
  ret void
}

define ptr @Linear_new(i64 %in_dim, i64 %out_dim) {
entry:
  %b = alloca ptr, align 8
  %W = alloca ptr, align 8
  %out_dim2 = alloca i64, align 8
  %in_dim1 = alloca i64, align 8
  store i64 %in_dim, ptr %in_dim1, align 8
  store i64 %out_dim, ptr %out_dim2, align 8
  %in_dim3 = load i64, ptr %in_dim1, align 8
  %out_dim4 = load i64, ptr %out_dim2, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 %in_dim3, ptr %shape_ptr_in, align 8
  %shape_ptr_in5 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 %out_dim4, ptr %shape_ptr_in5, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 true)
  store ptr %randn_res, ptr %W, align 8
  call void @tl_register_tensor(ptr @tensor_name.61, ptr %randn_res)
  %out_dim6 = load i64, ptr %out_dim2, align 8
  %shape_arr7 = alloca [1 x i64], align 8
  %shape_ptr_in8 = getelementptr inbounds [1 x i64], ptr %shape_arr7, i64 0, i64 0
  store i64 %out_dim6, ptr %shape_ptr_in8, align 8
  %randn_res9 = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr7, i1 true)
  store ptr %randn_res9, ptr %b, align 8
  call void @tl_register_tensor(ptr @tensor_name.62, ptr %randn_res9)
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Linear, ptr null, i32 1) to i64))
  %W10 = load ptr, ptr %W, align 8
  %clone_res = call ptr @tl_tensor_clone(ptr %W10)
  %init_field = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 0
  store ptr %clone_res, ptr %init_field, align 8
  %b11 = load ptr, ptr %b, align 8
  %clone_res12 = call ptr @tl_tensor_clone(ptr %b11)
  %init_field13 = getelementptr inbounds nuw %Linear, ptr %struct_malloc, i32 0, i32 1
  store ptr %clone_res12, ptr %init_field13, align 8
  %load_for_free = load ptr, ptr %b, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free14 = load ptr, ptr %W, align 8
  call void @tl_tensor_free(ptr %load_for_free14)
  ret ptr %struct_malloc
}

define void @main() {
entry:
  %loss87 = alloca ptr, align 8
  %y_pred82 = alloca ptr, align 8
  %loss68 = alloca ptr, align 8
  %y_pred63 = alloca ptr, align 8
  %loss = alloca ptr, align 8
  %y_pred = alloca ptr, align 8
  %X = alloca ptr, align 8
  %optim = alloca ptr, align 8
  %v_b = alloca ptr, align 8
  %m_b = alloca ptr, align 8
  %v_W = alloca ptr, align 8
  %m_W = alloca ptr, align 8
  %model = alloca ptr, align 8
  %call_tmp = call ptr @Linear_new(i64 2, i64 1)
  store ptr %call_tmp, ptr %model, align 8
  %shape_arr = alloca [2 x i64], align 8
  %shape_ptr_in = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 0
  store i64 2, ptr %shape_ptr_in, align 8
  %shape_ptr_in1 = getelementptr inbounds [2 x i64], ptr %shape_arr, i64 0, i64 1
  store i64 1, ptr %shape_ptr_in1, align 8
  %randn_res = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr, i1 false)
  %scalar_data = alloca float, align 4
  store float 0.000000e+00, ptr %scalar_data, align 4
  %scalar_shape = alloca i64, i64 0, align 8
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  %binop_res = call ptr @tl_tensor_mul(ptr %randn_res, ptr %scalar_tensor)
  store ptr %binop_res, ptr %m_W, align 8
  call void @tl_register_tensor(ptr @tensor_name.63, ptr %binop_res)
  %shape_arr2 = alloca [2 x i64], align 8
  %shape_ptr_in3 = getelementptr inbounds [2 x i64], ptr %shape_arr2, i64 0, i64 0
  store i64 2, ptr %shape_ptr_in3, align 8
  %shape_ptr_in4 = getelementptr inbounds [2 x i64], ptr %shape_arr2, i64 0, i64 1
  store i64 1, ptr %shape_ptr_in4, align 8
  %randn_res5 = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr2, i1 false)
  %scalar_data6 = alloca float, align 4
  store float 0.000000e+00, ptr %scalar_data6, align 4
  %scalar_shape7 = alloca i64, i64 0, align 8
  %scalar_tensor8 = call ptr @tl_tensor_new(ptr %scalar_data6, i64 0, ptr %scalar_shape7)
  %binop_res9 = call ptr @tl_tensor_mul(ptr %randn_res5, ptr %scalar_tensor8)
  store ptr %binop_res9, ptr %v_W, align 8
  call void @tl_register_tensor(ptr @tensor_name.64, ptr %binop_res9)
  %shape_arr10 = alloca [1 x i64], align 8
  %shape_ptr_in11 = getelementptr inbounds [1 x i64], ptr %shape_arr10, i64 0, i64 0
  store i64 1, ptr %shape_ptr_in11, align 8
  %randn_res12 = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr10, i1 false)
  %scalar_data13 = alloca float, align 4
  store float 0.000000e+00, ptr %scalar_data13, align 4
  %scalar_shape14 = alloca i64, i64 0, align 8
  %scalar_tensor15 = call ptr @tl_tensor_new(ptr %scalar_data13, i64 0, ptr %scalar_shape14)
  %binop_res16 = call ptr @tl_tensor_mul(ptr %randn_res12, ptr %scalar_tensor15)
  store ptr %binop_res16, ptr %m_b, align 8
  call void @tl_register_tensor(ptr @tensor_name.65, ptr %binop_res16)
  %shape_arr17 = alloca [1 x i64], align 8
  %shape_ptr_in18 = getelementptr inbounds [1 x i64], ptr %shape_arr17, i64 0, i64 0
  store i64 1, ptr %shape_ptr_in18, align 8
  %randn_res19 = call ptr @tl_tensor_randn(i64 1, ptr %shape_arr17, i1 false)
  %scalar_data20 = alloca float, align 4
  store float 0.000000e+00, ptr %scalar_data20, align 4
  %scalar_shape21 = alloca i64, i64 0, align 8
  %scalar_tensor22 = call ptr @tl_tensor_new(ptr %scalar_data20, i64 0, ptr %scalar_shape21)
  %binop_res23 = call ptr @tl_tensor_mul(ptr %randn_res19, ptr %scalar_tensor22)
  store ptr %binop_res23, ptr %v_b, align 8
  call void @tl_register_tensor(ptr @tensor_name.66, ptr %binop_res23)
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%Adam, ptr null, i32 1) to i64))
  %init_field = getelementptr inbounds nuw %Adam, ptr %struct_malloc, i32 0, i32 0
  store float 0x3F847AE140000000, ptr %init_field, align 4
  %init_field24 = getelementptr inbounds nuw %Adam, ptr %struct_malloc, i32 0, i32 1
  store float 0x3FECCCCCC0000000, ptr %init_field24, align 4
  %init_field25 = getelementptr inbounds nuw %Adam, ptr %struct_malloc, i32 0, i32 2
  store float 0x3FEFF7CEE0000000, ptr %init_field25, align 4
  %init_field26 = getelementptr inbounds nuw %Adam, ptr %struct_malloc, i32 0, i32 3
  store float 0x3E45798EE0000000, ptr %init_field26, align 4
  %init_field27 = getelementptr inbounds nuw %Adam, ptr %struct_malloc, i32 0, i32 4
  store i64 1, ptr %init_field27, align 8
  %m_W28 = load ptr, ptr %m_W, align 8
  %init_field29 = getelementptr inbounds nuw %Adam, ptr %struct_malloc, i32 0, i32 5
  store ptr %m_W28, ptr %init_field29, align 8
  %v_W30 = load ptr, ptr %v_W, align 8
  %init_field31 = getelementptr inbounds nuw %Adam, ptr %struct_malloc, i32 0, i32 6
  store ptr %v_W30, ptr %init_field31, align 8
  %m_b32 = load ptr, ptr %m_b, align 8
  %init_field33 = getelementptr inbounds nuw %Adam, ptr %struct_malloc, i32 0, i32 7
  store ptr %m_b32, ptr %init_field33, align 8
  %v_b34 = load ptr, ptr %v_b, align 8
  %init_field35 = getelementptr inbounds nuw %Adam, ptr %struct_malloc, i32 0, i32 8
  store ptr %v_b34, ptr %init_field35, align 8
  store ptr %struct_malloc, ptr %optim, align 8
  %shape_arr36 = alloca [2 x i64], align 8
  %shape_ptr_in37 = getelementptr inbounds [2 x i64], ptr %shape_arr36, i64 0, i64 0
  store i64 16, ptr %shape_ptr_in37, align 8
  %shape_ptr_in38 = getelementptr inbounds [2 x i64], ptr %shape_arr36, i64 0, i64 1
  store i64 2, ptr %shape_ptr_in38, align 8
  %randn_res39 = call ptr @tl_tensor_randn(i64 2, ptr %shape_arr36, i1 false)
  store ptr %randn_res39, ptr %X, align 8
  call void @tl_register_tensor(ptr @tensor_name.67, ptr %randn_res39)
  %model40 = load ptr, ptr %model, align 8
  %ptr_W = getelementptr inbounds nuw %Linear, ptr %model40, i32 0, i32 0
  %W = load ptr, ptr %ptr_W, align 8
  call void @tl_tensor_print(ptr %W)
  %X41 = load ptr, ptr %X, align 8
  %model42 = load ptr, ptr %model, align 8
  %ptr_W43 = getelementptr inbounds nuw %Linear, ptr %model42, i32 0, i32 0
  %W44 = load ptr, ptr %ptr_W43, align 8
  %matmul_res = call ptr @tl_tensor_matmul(ptr %X41, ptr %W44)
  %model45 = load ptr, ptr %model, align 8
  %ptr_b = getelementptr inbounds nuw %Linear, ptr %model45, i32 0, i32 1
  %b = load ptr, ptr %ptr_b, align 8
  %binop_res46 = call ptr @tl_tensor_add(ptr %matmul_res, ptr %b)
  store ptr %binop_res46, ptr %y_pred, align 8
  call void @tl_register_tensor(ptr @tensor_name.68, ptr %binop_res46)
  %y_pred47 = load ptr, ptr %y_pred, align 8
  %y_pred48 = load ptr, ptr %y_pred, align 8
  %binop_res49 = call ptr @tl_tensor_mul(ptr %y_pred47, ptr %y_pred48)
  %sum_res = call ptr @tl_tensor_sum(ptr %binop_res49)
  store ptr %sum_res, ptr %loss, align 8
  call void @tl_register_tensor(ptr @tensor_name.69, ptr %sum_res)
  %loss50 = load ptr, ptr %loss, align 8
  call void @tl_tensor_print(ptr %loss50)
  %loss51 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss51)
  %optim52 = load ptr, ptr %optim, align 8
  %model53 = load ptr, ptr %model, align 8
  call void @Adam_step(ptr %optim52, ptr %model53)
  %X54 = load ptr, ptr %X, align 8
  %model55 = load ptr, ptr %model, align 8
  %ptr_W56 = getelementptr inbounds nuw %Linear, ptr %model55, i32 0, i32 0
  %W57 = load ptr, ptr %ptr_W56, align 8
  %matmul_res58 = call ptr @tl_tensor_matmul(ptr %X54, ptr %W57)
  %model59 = load ptr, ptr %model, align 8
  %ptr_b60 = getelementptr inbounds nuw %Linear, ptr %model59, i32 0, i32 1
  %b61 = load ptr, ptr %ptr_b60, align 8
  %binop_res62 = call ptr @tl_tensor_add(ptr %matmul_res58, ptr %b61)
  store ptr %binop_res62, ptr %y_pred63, align 8
  call void @tl_register_tensor(ptr @tensor_name.70, ptr %binop_res62)
  %y_pred64 = load ptr, ptr %y_pred63, align 8
  %y_pred65 = load ptr, ptr %y_pred63, align 8
  %binop_res66 = call ptr @tl_tensor_mul(ptr %y_pred64, ptr %y_pred65)
  %sum_res67 = call ptr @tl_tensor_sum(ptr %binop_res66)
  store ptr %sum_res67, ptr %loss68, align 8
  call void @tl_register_tensor(ptr @tensor_name.71, ptr %sum_res67)
  %loss69 = load ptr, ptr %loss68, align 8
  call void @tl_tensor_print(ptr %loss69)
  %loss70 = load ptr, ptr %loss68, align 8
  call void @tl_tensor_backward(ptr %loss70)
  %optim71 = load ptr, ptr %optim, align 8
  %model72 = load ptr, ptr %model, align 8
  call void @Adam_step(ptr %optim71, ptr %model72)
  %X73 = load ptr, ptr %X, align 8
  %model74 = load ptr, ptr %model, align 8
  %ptr_W75 = getelementptr inbounds nuw %Linear, ptr %model74, i32 0, i32 0
  %W76 = load ptr, ptr %ptr_W75, align 8
  %matmul_res77 = call ptr @tl_tensor_matmul(ptr %X73, ptr %W76)
  %model78 = load ptr, ptr %model, align 8
  %ptr_b79 = getelementptr inbounds nuw %Linear, ptr %model78, i32 0, i32 1
  %b80 = load ptr, ptr %ptr_b79, align 8
  %binop_res81 = call ptr @tl_tensor_add(ptr %matmul_res77, ptr %b80)
  store ptr %binop_res81, ptr %y_pred82, align 8
  call void @tl_register_tensor(ptr @tensor_name.72, ptr %binop_res81)
  %y_pred83 = load ptr, ptr %y_pred82, align 8
  %y_pred84 = load ptr, ptr %y_pred82, align 8
  %binop_res85 = call ptr @tl_tensor_mul(ptr %y_pred83, ptr %y_pred84)
  %sum_res86 = call ptr @tl_tensor_sum(ptr %binop_res85)
  store ptr %sum_res86, ptr %loss87, align 8
  call void @tl_register_tensor(ptr @tensor_name.73, ptr %sum_res86)
  %loss88 = load ptr, ptr %loss87, align 8
  call void @tl_tensor_print(ptr %loss88)
  %loss89 = load ptr, ptr %loss87, align 8
  call void @tl_tensor_backward(ptr %loss89)
  %optim90 = load ptr, ptr %optim, align 8
  %model91 = load ptr, ptr %model, align 8
  call void @Adam_step(ptr %optim90, ptr %model91)
  %model92 = load ptr, ptr %model, align 8
  %ptr_W93 = getelementptr inbounds nuw %Linear, ptr %model92, i32 0, i32 0
  %W94 = load ptr, ptr %ptr_W93, align 8
  call void @tl_tensor_print(ptr %W94)
  %load_for_free = load ptr, ptr %y_pred82, align 8
  call void @tl_tensor_free(ptr %load_for_free)
  %load_for_free95 = load ptr, ptr %loss87, align 8
  call void @tl_tensor_free(ptr %load_for_free95)
  %load_for_free96 = load ptr, ptr %v_b, align 8
  call void @tl_tensor_free(ptr %load_for_free96)
  %load_for_free97 = load ptr, ptr %m_b, align 8
  call void @tl_tensor_free(ptr %load_for_free97)
  %load_for_free98 = load ptr, ptr %m_W, align 8
  call void @tl_tensor_free(ptr %load_for_free98)
  %load_for_free99 = load ptr, ptr %v_W, align 8
  call void @tl_tensor_free(ptr %load_for_free99)
  %load_for_free100 = load ptr, ptr %X, align 8
  call void @tl_tensor_free(ptr %load_for_free100)
  ret void
}
