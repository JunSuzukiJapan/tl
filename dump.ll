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

@str_literal = private unnamed_addr constant [6 x i8] c"Loss:\00", align 1
@str_literal.104 = private unnamed_addr constant [12 x i8] c"Memory(MB):\00", align 1
@str_literal.105 = private unnamed_addr constant [55 x i8] c"Resuming training from model_2digit_rev.safetensors...\00", align 1
@key_str = private unnamed_addr constant [4 x i8] c"w.w\00", align 1
@key_str.106 = private unnamed_addr constant [5 x i8] c"wp.w\00", align 1
@key_str.107 = private unnamed_addr constant [8 x i8] c"b1.l1.w\00", align 1
@key_str.108 = private unnamed_addr constant [8 x i8] c"b1.l1.b\00", align 1
@key_str.109 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.W\00", align 1
@key_str.110 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.b\00", align 1
@key_str.111 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.W\00", align 1
@key_str.112 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.b\00", align 1
@key_str.113 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.W\00", align 1
@key_str.114 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.b\00", align 1
@key_str.115 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.W\00", align 1
@key_str.116 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.b\00", align 1
@key_str.117 = private unnamed_addr constant [8 x i8] c"b1.l2.w\00", align 1
@key_str.118 = private unnamed_addr constant [8 x i8] c"b1.l2.b\00", align 1
@key_str.119 = private unnamed_addr constant [9 x i8] c"b1.m.f.W\00", align 1
@key_str.120 = private unnamed_addr constant [9 x i8] c"b1.m.f.b\00", align 1
@key_str.121 = private unnamed_addr constant [9 x i8] c"b1.m.p.W\00", align 1
@key_str.122 = private unnamed_addr constant [9 x i8] c"b1.m.p.b\00", align 1
@key_str.123 = private unnamed_addr constant [8 x i8] c"b2.l1.w\00", align 1
@key_str.124 = private unnamed_addr constant [8 x i8] c"b2.l1.b\00", align 1
@key_str.125 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.W\00", align 1
@key_str.126 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.b\00", align 1
@key_str.127 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.W\00", align 1
@key_str.128 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.b\00", align 1
@key_str.129 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.W\00", align 1
@key_str.130 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.b\00", align 1
@key_str.131 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.W\00", align 1
@key_str.132 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.b\00", align 1
@key_str.133 = private unnamed_addr constant [8 x i8] c"b2.l2.w\00", align 1
@key_str.134 = private unnamed_addr constant [8 x i8] c"b2.l2.b\00", align 1
@key_str.135 = private unnamed_addr constant [9 x i8] c"b2.m.f.W\00", align 1
@key_str.136 = private unnamed_addr constant [9 x i8] c"b2.m.f.b\00", align 1
@key_str.137 = private unnamed_addr constant [9 x i8] c"b2.m.p.W\00", align 1
@key_str.138 = private unnamed_addr constant [9 x i8] c"b2.m.p.b\00", align 1
@key_str.139 = private unnamed_addr constant [8 x i8] c"b3.l1.w\00", align 1
@key_str.140 = private unnamed_addr constant [8 x i8] c"b3.l1.b\00", align 1
@key_str.141 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.W\00", align 1
@key_str.142 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.b\00", align 1
@key_str.143 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.W\00", align 1
@key_str.144 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.b\00", align 1
@key_str.145 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.W\00", align 1
@key_str.146 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.b\00", align 1
@key_str.147 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.W\00", align 1
@key_str.148 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.b\00", align 1
@key_str.149 = private unnamed_addr constant [8 x i8] c"b3.l2.w\00", align 1
@key_str.150 = private unnamed_addr constant [8 x i8] c"b3.l2.b\00", align 1
@key_str.151 = private unnamed_addr constant [9 x i8] c"b3.m.f.W\00", align 1
@key_str.152 = private unnamed_addr constant [9 x i8] c"b3.m.f.b\00", align 1
@key_str.153 = private unnamed_addr constant [9 x i8] c"b3.m.p.W\00", align 1
@key_str.154 = private unnamed_addr constant [9 x i8] c"b3.m.p.b\00", align 1
@key_str.155 = private unnamed_addr constant [4 x i8] c"l.w\00", align 1
@key_str.156 = private unnamed_addr constant [4 x i8] c"l.b\00", align 1
@key_str.157 = private unnamed_addr constant [4 x i8] c"h.W\00", align 1
@key_str.158 = private unnamed_addr constant [4 x i8] c"h.b\00", align 1
@str_literal.159 = private unnamed_addr constant [29 x i8] c"model_2digit_rev.safetensors\00", align 1
@str_literal.160 = private unnamed_addr constant [61 x i8] c"Training 2-digit addition (Reverse Digits, 3-Layer) - RESUME\00", align 1
@str_literal.161 = private unnamed_addr constant [7 x i8] c"Epoch:\00", align 1
@key_str.162 = private unnamed_addr constant [4 x i8] c"w.w\00", align 1
@key_str.163 = private unnamed_addr constant [5 x i8] c"wp.w\00", align 1
@key_str.164 = private unnamed_addr constant [8 x i8] c"b1.l1.w\00", align 1
@key_str.165 = private unnamed_addr constant [8 x i8] c"b1.l1.b\00", align 1
@key_str.166 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.W\00", align 1
@key_str.167 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.b\00", align 1
@key_str.168 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.W\00", align 1
@key_str.169 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.b\00", align 1
@key_str.170 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.W\00", align 1
@key_str.171 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.b\00", align 1
@key_str.172 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.W\00", align 1
@key_str.173 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.b\00", align 1
@key_str.174 = private unnamed_addr constant [8 x i8] c"b1.l2.w\00", align 1
@key_str.175 = private unnamed_addr constant [8 x i8] c"b1.l2.b\00", align 1
@key_str.176 = private unnamed_addr constant [9 x i8] c"b1.m.f.W\00", align 1
@key_str.177 = private unnamed_addr constant [9 x i8] c"b1.m.f.b\00", align 1
@key_str.178 = private unnamed_addr constant [9 x i8] c"b1.m.p.W\00", align 1
@key_str.179 = private unnamed_addr constant [9 x i8] c"b1.m.p.b\00", align 1
@key_str.180 = private unnamed_addr constant [8 x i8] c"b2.l1.w\00", align 1
@key_str.181 = private unnamed_addr constant [8 x i8] c"b2.l1.b\00", align 1
@key_str.182 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.W\00", align 1
@key_str.183 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.b\00", align 1
@key_str.184 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.W\00", align 1
@key_str.185 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.b\00", align 1
@key_str.186 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.W\00", align 1
@key_str.187 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.b\00", align 1
@key_str.188 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.W\00", align 1
@key_str.189 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.b\00", align 1
@key_str.190 = private unnamed_addr constant [8 x i8] c"b2.l2.w\00", align 1
@key_str.191 = private unnamed_addr constant [8 x i8] c"b2.l2.b\00", align 1
@key_str.192 = private unnamed_addr constant [9 x i8] c"b2.m.f.W\00", align 1
@key_str.193 = private unnamed_addr constant [9 x i8] c"b2.m.f.b\00", align 1
@key_str.194 = private unnamed_addr constant [9 x i8] c"b2.m.p.W\00", align 1
@key_str.195 = private unnamed_addr constant [9 x i8] c"b2.m.p.b\00", align 1
@key_str.196 = private unnamed_addr constant [8 x i8] c"b3.l1.w\00", align 1
@key_str.197 = private unnamed_addr constant [8 x i8] c"b3.l1.b\00", align 1
@key_str.198 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.W\00", align 1
@key_str.199 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.b\00", align 1
@key_str.200 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.W\00", align 1
@key_str.201 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.b\00", align 1
@key_str.202 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.W\00", align 1
@key_str.203 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.b\00", align 1
@key_str.204 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.W\00", align 1
@key_str.205 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.b\00", align 1
@key_str.206 = private unnamed_addr constant [8 x i8] c"b3.l2.w\00", align 1
@key_str.207 = private unnamed_addr constant [8 x i8] c"b3.l2.b\00", align 1
@key_str.208 = private unnamed_addr constant [9 x i8] c"b3.m.f.W\00", align 1
@key_str.209 = private unnamed_addr constant [9 x i8] c"b3.m.f.b\00", align 1
@key_str.210 = private unnamed_addr constant [9 x i8] c"b3.m.p.W\00", align 1
@key_str.211 = private unnamed_addr constant [9 x i8] c"b3.m.p.b\00", align 1
@key_str.212 = private unnamed_addr constant [4 x i8] c"l.w\00", align 1
@key_str.213 = private unnamed_addr constant [4 x i8] c"l.b\00", align 1
@key_str.214 = private unnamed_addr constant [4 x i8] c"h.W\00", align 1
@key_str.215 = private unnamed_addr constant [4 x i8] c"h.b\00", align 1
@str_literal.216 = private unnamed_addr constant [29 x i8] c"model_2digit_rev.safetensors\00", align 1
@str_literal.217 = private unnamed_addr constant [19 x i8] c"Training Complete!\00", align 1
@key_str.218 = private unnamed_addr constant [4 x i8] c"w.w\00", align 1
@key_str.219 = private unnamed_addr constant [5 x i8] c"wp.w\00", align 1
@key_str.220 = private unnamed_addr constant [8 x i8] c"b1.l1.w\00", align 1
@key_str.221 = private unnamed_addr constant [8 x i8] c"b1.l1.b\00", align 1
@key_str.222 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.W\00", align 1
@key_str.223 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.b\00", align 1
@key_str.224 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.W\00", align 1
@key_str.225 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.b\00", align 1
@key_str.226 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.W\00", align 1
@key_str.227 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.b\00", align 1
@key_str.228 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.W\00", align 1
@key_str.229 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.b\00", align 1
@key_str.230 = private unnamed_addr constant [8 x i8] c"b1.l2.w\00", align 1
@key_str.231 = private unnamed_addr constant [8 x i8] c"b1.l2.b\00", align 1
@key_str.232 = private unnamed_addr constant [9 x i8] c"b1.m.f.W\00", align 1
@key_str.233 = private unnamed_addr constant [9 x i8] c"b1.m.f.b\00", align 1
@key_str.234 = private unnamed_addr constant [9 x i8] c"b1.m.p.W\00", align 1
@key_str.235 = private unnamed_addr constant [9 x i8] c"b1.m.p.b\00", align 1
@key_str.236 = private unnamed_addr constant [8 x i8] c"b2.l1.w\00", align 1
@key_str.237 = private unnamed_addr constant [8 x i8] c"b2.l1.b\00", align 1
@key_str.238 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.W\00", align 1
@key_str.239 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.b\00", align 1
@key_str.240 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.W\00", align 1
@key_str.241 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.b\00", align 1
@key_str.242 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.W\00", align 1
@key_str.243 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.b\00", align 1
@key_str.244 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.W\00", align 1
@key_str.245 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.b\00", align 1
@key_str.246 = private unnamed_addr constant [8 x i8] c"b2.l2.w\00", align 1
@key_str.247 = private unnamed_addr constant [8 x i8] c"b2.l2.b\00", align 1
@key_str.248 = private unnamed_addr constant [9 x i8] c"b2.m.f.W\00", align 1
@key_str.249 = private unnamed_addr constant [9 x i8] c"b2.m.f.b\00", align 1
@key_str.250 = private unnamed_addr constant [9 x i8] c"b2.m.p.W\00", align 1
@key_str.251 = private unnamed_addr constant [9 x i8] c"b2.m.p.b\00", align 1
@key_str.252 = private unnamed_addr constant [8 x i8] c"b3.l1.w\00", align 1
@key_str.253 = private unnamed_addr constant [8 x i8] c"b3.l1.b\00", align 1
@key_str.254 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.W\00", align 1
@key_str.255 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.b\00", align 1
@key_str.256 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.W\00", align 1
@key_str.257 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.b\00", align 1
@key_str.258 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.W\00", align 1
@key_str.259 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.b\00", align 1
@key_str.260 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.W\00", align 1
@key_str.261 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.b\00", align 1
@key_str.262 = private unnamed_addr constant [8 x i8] c"b3.l2.w\00", align 1
@key_str.263 = private unnamed_addr constant [8 x i8] c"b3.l2.b\00", align 1
@key_str.264 = private unnamed_addr constant [9 x i8] c"b3.m.f.W\00", align 1
@key_str.265 = private unnamed_addr constant [9 x i8] c"b3.m.f.b\00", align 1
@key_str.266 = private unnamed_addr constant [9 x i8] c"b3.m.p.W\00", align 1
@key_str.267 = private unnamed_addr constant [9 x i8] c"b3.m.p.b\00", align 1
@key_str.268 = private unnamed_addr constant [4 x i8] c"l.w\00", align 1
@key_str.269 = private unnamed_addr constant [4 x i8] c"l.b\00", align 1
@key_str.270 = private unnamed_addr constant [4 x i8] c"h.W\00", align 1
@key_str.271 = private unnamed_addr constant [4 x i8] c"h.b\00", align 1
@str_literal.272 = private unnamed_addr constant [29 x i8] c"model_2digit_rev.safetensors\00", align 1

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
  call void @tl_mem_unregister(ptr %self3)
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
  call void @tl_mem_unregister(ptr %self3)
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
  call void @tl_mem_unregister(ptr %self3)
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
  %tril_res = call ptr @tl_tensor_tril(ptr %binop_res, i32 0)
  %softmax_res = call ptr @tl_tensor_softmax(ptr %tril_res, i64 2)
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
  call void @tl_mem_unregister(ptr %self3)
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
  call void @tl_mem_unregister(ptr %self3)
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
  call void @tl_mem_register_tensor(ptr %call_method7)
  %binop_res = call ptr @tl_tensor_add(ptr %x3, ptr %call_method7)
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
  call void @tl_mem_unregister(ptr %self3)
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
  %x36 = alloca ptr, align 16
  %x31 = alloca ptr, align 16
  %x26 = alloca ptr, align 16
  %x = alloca ptr, align 16
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
  call void @tl_mem_unregister(ptr %new_const_tensor)
  store ptr %new_const_tensor, ptr %pos_data, align 8
  %pos_data14 = load ptr, ptr %pos_data, align 8
  %dims_alloca = alloca [2 x i64], align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr, align 8
  %dim_ptr15 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr15, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %pos_data14, ptr %dims_ptr, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res)
  store ptr %reshape_dims_res, ptr %pos, align 8
  %self16 = load ptr, ptr %self1, align 8
  %ptr_w = getelementptr inbounds %GPT, ptr %self16, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %i17 = load ptr, ptr %i2, align 8
  %call_method = call ptr @tl_Embedding_forward(ptr %w, ptr %i17)
  call void @tl_mem_register_tensor(ptr %call_method)
  call void @tl_mem_unregister(ptr %call_method)
  store ptr %call_method, ptr %tok_emb, align 8
  %self18 = load ptr, ptr %self1, align 8
  %ptr_wp = getelementptr inbounds %GPT, ptr %self18, i32 0, i32 1
  %wp = load ptr, ptr %ptr_wp, align 8
  %pos19 = load ptr, ptr %pos, align 8
  %call_method20 = call ptr @tl_Embedding_forward(ptr %wp, ptr %pos19)
  call void @tl_mem_register_tensor(ptr %call_method20)
  call void @tl_mem_unregister(ptr %call_method20)
  store ptr %call_method20, ptr %pos_emb, align 8
  %tok_emb21 = load ptr, ptr %tok_emb, align 8
  %pos_emb22 = load ptr, ptr %pos_emb, align 8
  %binop_res = call ptr @tl_tensor_add(ptr %tok_emb21, ptr %pos_emb22)
  call void @tl_mem_unregister(ptr %binop_res)
  store ptr %binop_res, ptr %x, align 8
  %self23 = load ptr, ptr %self1, align 8
  %ptr_b1 = getelementptr inbounds %GPT, ptr %self23, i32 0, i32 2
  %b1 = load ptr, ptr %ptr_b1, align 8
  %x24 = load ptr, ptr %x, align 8
  %call_method25 = call ptr @tl_Block_forward(ptr %b1, ptr %x24)
  call void @tl_mem_register_tensor(ptr %call_method25)
  call void @tl_mem_unregister(ptr %call_method25)
  %old_shadowed = load ptr, ptr %x, align 8
  call void @tl_mem_unregister(ptr %old_shadowed)
  store ptr %call_method25, ptr %x26, align 8
  %self27 = load ptr, ptr %self1, align 8
  %ptr_b2 = getelementptr inbounds %GPT, ptr %self27, i32 0, i32 3
  %b2 = load ptr, ptr %ptr_b2, align 8
  %x28 = load ptr, ptr %x26, align 8
  %call_method29 = call ptr @tl_Block_forward(ptr %b2, ptr %x28)
  call void @tl_mem_register_tensor(ptr %call_method29)
  call void @tl_mem_unregister(ptr %call_method29)
  %old_shadowed30 = load ptr, ptr %x26, align 8
  call void @tl_mem_unregister(ptr %old_shadowed30)
  store ptr %call_method29, ptr %x31, align 8
  %self32 = load ptr, ptr %self1, align 8
  %ptr_b3 = getelementptr inbounds %GPT, ptr %self32, i32 0, i32 4
  %b3 = load ptr, ptr %ptr_b3, align 8
  %x33 = load ptr, ptr %x31, align 8
  %call_method34 = call ptr @tl_Block_forward(ptr %b3, ptr %x33)
  call void @tl_mem_register_tensor(ptr %call_method34)
  call void @tl_mem_unregister(ptr %call_method34)
  %old_shadowed35 = load ptr, ptr %x31, align 8
  call void @tl_mem_unregister(ptr %old_shadowed35)
  store ptr %call_method34, ptr %x36, align 8
  %self37 = load ptr, ptr %self1, align 8
  %ptr_h = getelementptr inbounds %GPT, ptr %self37, i32 0, i32 6
  %h = load ptr, ptr %ptr_h, align 8
  %self38 = load ptr, ptr %self1, align 8
  %ptr_l = getelementptr inbounds %GPT, ptr %self38, i32 0, i32 5
  %l = load ptr, ptr %ptr_l, align 8
  %x39 = load ptr, ptr %x36, align 8
  %call_method40 = call ptr @tl_LayerNorm_forward(ptr %l, ptr %x39)
  call void @tl_mem_register_tensor(ptr %call_method40)
  %call_method41 = call ptr @tl_Linear_forward(ptr %h, ptr %call_method40)
  call void @tl_mem_register_tensor(ptr %call_method41)
  call void @tl_mem_unregister(ptr %call_method41)
  call void @tl_mem_exit_scope()
  ret ptr %call_method41
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
  call void @tl_mem_unregister(ptr %self3)
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

define i64 @get_memory() {
entry:
  call void @tl_mem_enter_scope()
  %call_tmp = call i64 @tl_get_memory_mb()
  call void @tl_mem_exit_scope()
  ret i64 %call_tmp
}

define void @train_epoch(ptr %model, float %lr, i64 %epoch) {
entry:
  %mem_mb = alloca i64, align 16
  %loss = alloca ptr, align 16
  %Y_flat = alloca ptr, align 16
  %logits_flat = alloca ptr, align 16
  %logits = alloca ptr, align 16
  %Y = alloca ptr, align 16
  %X = alloca ptr, align 16
  %target = alloca ptr, align 16
  %data = alloca ptr, align 16
  %v_s_d100 = alloca float, align 16
  %scalar_shape114 = alloca i64, align 16
  %scalar_data113 = alloca float, align 16
  %scalar_shape111 = alloca i64, align 16
  %scalar_data109 = alloca float, align 16
  %v_s_d10 = alloca float, align 16
  %scalar_shape104 = alloca i64, align 16
  %scalar_data103 = alloca float, align 16
  %scalar_shape101 = alloca i64, align 16
  %scalar_data99 = alloca float, align 16
  %v_s_d1 = alloca float, align 16
  %scalar_shape94 = alloca i64, align 16
  %scalar_data93 = alloca float, align 16
  %scalar_shape91 = alloca i64, align 16
  %scalar_data89 = alloca float, align 16
  %v_j_d10 = alloca float, align 16
  %scalar_shape84 = alloca i64, align 16
  %scalar_data83 = alloca float, align 16
  %scalar_shape81 = alloca i64, align 16
  %scalar_data79 = alloca float, align 16
  %v_j_d1 = alloca float, align 16
  %scalar_shape74 = alloca i64, align 16
  %scalar_data73 = alloca float, align 16
  %scalar_shape71 = alloca i64, align 16
  %scalar_data69 = alloca float, align 16
  %v_i_d10 = alloca float, align 16
  %scalar_shape64 = alloca i64, align 16
  %scalar_data63 = alloca float, align 16
  %scalar_shape61 = alloca i64, align 16
  %scalar_data59 = alloca float, align 16
  %v_i_d1 = alloca float, align 16
  %scalar_shape55 = alloca i64, align 16
  %scalar_data54 = alloca float, align 16
  %scalar_shape52 = alloca i64, align 16
  %scalar_data51 = alloca float, align 16
  %s_d1 = alloca i64, align 16
  %s_d10 = alloca i64, align 16
  %rem = alloca i64, align 16
  %s_d100 = alloca i64, align 16
  %j_d1 = alloca i64, align 16
  %j_d10 = alloca i64, align 16
  %i_d1 = alloca i64, align 16
  %i_d10 = alloca i64, align 16
  %sum = alloca i64, align 16
  %j = alloca i64, align 16
  %i = alloca i64, align 16
  %idx = alloca i64, align 16
  %raw = alloca i64, align 16
  %s = alloca i64, align 16
  %offset = alloca i64, align 16
  %stride = alloca i64, align 16
  %total_steps = alloca i64, align 16
  %total_loss = alloca ptr, align 16
  %scalar_shape5 = alloca i64, align 16
  %scalar_data4 = alloca float, align 16
  %scalar_shape = alloca i64, align 16
  %scalar_data = alloca float, align 16
  %epoch3 = alloca i64, align 16
  %lr2 = alloca float, align 16
  %model1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %model, ptr %model1, align 8
  store float %lr, ptr %lr2, align 4
  store i64 %epoch, ptr %epoch3, align 8
  store float 0.000000e+00, ptr %scalar_data, align 4
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  store float 1.000000e+00, ptr %scalar_data4, align 4
  %scalar_tensor6 = call ptr @tl_tensor_new(ptr %scalar_data4, i64 0, ptr %scalar_shape5)
  %pow_res = call ptr @tl_tensor_pow(ptr %scalar_tensor, ptr %scalar_tensor6)
  call void @tl_mem_unregister(ptr %pow_res)
  store ptr %pow_res, ptr %total_loss, align 8
  store i64 1000, ptr %total_steps, align 8
  store i64 137, ptr %stride, align 8
  %epoch7 = load i64, ptr %epoch3, align 8
  %multmp = mul i64 %epoch7, 79
  store i64 %multmp, ptr %offset, align 8
  %total_steps8 = load i64, ptr %total_steps, align 8
  br label %for_header

for_header:                                       ; preds = %continue_block, %entry
  %for_idx = phi i64 [ 0, %entry ], [ %next_idx, %continue_block ]
  %for_cond = icmp slt i64 %for_idx, %total_steps8
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %s, align 8
  %s9 = load i64, ptr %s, align 8
  %stride10 = load i64, ptr %stride, align 8
  %multmp11 = mul i64 %s9, %stride10
  %offset12 = load i64, ptr %offset, align 8
  %addtmp = add i64 %multmp11, %offset12
  store i64 %addtmp, ptr %raw, align 8
  %raw13 = load i64, ptr %raw, align 8
  %raw14 = load i64, ptr %raw, align 8
  %divtmp = sdiv i64 %raw14, 10000
  %multmp15 = mul i64 %divtmp, 10000
  %subtmp = sub i64 %raw13, %multmp15
  store i64 %subtmp, ptr %idx, align 8
  %idx16 = load i64, ptr %idx, align 8
  %divtmp17 = sdiv i64 %idx16, 100
  store i64 %divtmp17, ptr %i, align 8
  %idx18 = load i64, ptr %idx, align 8
  %idx19 = load i64, ptr %idx, align 8
  %divtmp20 = sdiv i64 %idx19, 100
  %multmp21 = mul i64 %divtmp20, 100
  %subtmp22 = sub i64 %idx18, %multmp21
  store i64 %subtmp22, ptr %j, align 8
  %i23 = load i64, ptr %i, align 8
  %j24 = load i64, ptr %j, align 8
  %addtmp25 = add i64 %i23, %j24
  store i64 %addtmp25, ptr %sum, align 8
  %i26 = load i64, ptr %i, align 8
  %divtmp27 = sdiv i64 %i26, 10
  store i64 %divtmp27, ptr %i_d10, align 8
  %i28 = load i64, ptr %i, align 8
  %i_d1029 = load i64, ptr %i_d10, align 8
  %multmp30 = mul i64 %i_d1029, 10
  %subtmp31 = sub i64 %i28, %multmp30
  store i64 %subtmp31, ptr %i_d1, align 8
  %j32 = load i64, ptr %j, align 8
  %divtmp33 = sdiv i64 %j32, 10
  store i64 %divtmp33, ptr %j_d10, align 8
  %j34 = load i64, ptr %j, align 8
  %j_d1035 = load i64, ptr %j_d10, align 8
  %multmp36 = mul i64 %j_d1035, 10
  %subtmp37 = sub i64 %j34, %multmp36
  store i64 %subtmp37, ptr %j_d1, align 8
  %sum38 = load i64, ptr %sum, align 8
  %divtmp39 = sdiv i64 %sum38, 100
  store i64 %divtmp39, ptr %s_d100, align 8
  %sum40 = load i64, ptr %sum, align 8
  %s_d10041 = load i64, ptr %s_d100, align 8
  %multmp42 = mul i64 %s_d10041, 100
  %subtmp43 = sub i64 %sum40, %multmp42
  store i64 %subtmp43, ptr %rem, align 8
  %rem44 = load i64, ptr %rem, align 8
  %divtmp45 = sdiv i64 %rem44, 10
  store i64 %divtmp45, ptr %s_d10, align 8
  %rem46 = load i64, ptr %rem, align 8
  %s_d1047 = load i64, ptr %s_d10, align 8
  %multmp48 = mul i64 %s_d1047, 10
  %subtmp49 = sub i64 %rem46, %multmp48
  store i64 %subtmp49, ptr %s_d1, align 8
  %i_d150 = load i64, ptr %i_d1, align 8
  %cast_i64_f32 = sitofp i64 %i_d150 to float
  store float %cast_i64_f32, ptr %scalar_data51, align 4
  %scalar_tensor53 = call ptr @tl_tensor_new(ptr %scalar_data51, i64 0, ptr %scalar_shape52)
  store float 1.000000e+00, ptr %scalar_data54, align 4
  %scalar_tensor56 = call ptr @tl_tensor_new(ptr %scalar_data54, i64 0, ptr %scalar_shape55)
  %pow_res57 = call ptr @tl_tensor_pow(ptr %scalar_tensor53, ptr %scalar_tensor56)
  %get_res = call float @tl_tensor_get(ptr %pow_res57, i64 0)
  store float %get_res, ptr %v_i_d1, align 4
  %i_d1058 = load i64, ptr %i_d10, align 8
  %cast_i64_f3260 = sitofp i64 %i_d1058 to float
  store float %cast_i64_f3260, ptr %scalar_data59, align 4
  %scalar_tensor62 = call ptr @tl_tensor_new(ptr %scalar_data59, i64 0, ptr %scalar_shape61)
  store float 1.000000e+00, ptr %scalar_data63, align 4
  %scalar_tensor65 = call ptr @tl_tensor_new(ptr %scalar_data63, i64 0, ptr %scalar_shape64)
  %pow_res66 = call ptr @tl_tensor_pow(ptr %scalar_tensor62, ptr %scalar_tensor65)
  %get_res67 = call float @tl_tensor_get(ptr %pow_res66, i64 0)
  store float %get_res67, ptr %v_i_d10, align 4
  %j_d168 = load i64, ptr %j_d1, align 8
  %cast_i64_f3270 = sitofp i64 %j_d168 to float
  store float %cast_i64_f3270, ptr %scalar_data69, align 4
  %scalar_tensor72 = call ptr @tl_tensor_new(ptr %scalar_data69, i64 0, ptr %scalar_shape71)
  store float 1.000000e+00, ptr %scalar_data73, align 4
  %scalar_tensor75 = call ptr @tl_tensor_new(ptr %scalar_data73, i64 0, ptr %scalar_shape74)
  %pow_res76 = call ptr @tl_tensor_pow(ptr %scalar_tensor72, ptr %scalar_tensor75)
  %get_res77 = call float @tl_tensor_get(ptr %pow_res76, i64 0)
  store float %get_res77, ptr %v_j_d1, align 4
  %j_d1078 = load i64, ptr %j_d10, align 8
  %cast_i64_f3280 = sitofp i64 %j_d1078 to float
  store float %cast_i64_f3280, ptr %scalar_data79, align 4
  %scalar_tensor82 = call ptr @tl_tensor_new(ptr %scalar_data79, i64 0, ptr %scalar_shape81)
  store float 1.000000e+00, ptr %scalar_data83, align 4
  %scalar_tensor85 = call ptr @tl_tensor_new(ptr %scalar_data83, i64 0, ptr %scalar_shape84)
  %pow_res86 = call ptr @tl_tensor_pow(ptr %scalar_tensor82, ptr %scalar_tensor85)
  %get_res87 = call float @tl_tensor_get(ptr %pow_res86, i64 0)
  store float %get_res87, ptr %v_j_d10, align 4
  %s_d188 = load i64, ptr %s_d1, align 8
  %cast_i64_f3290 = sitofp i64 %s_d188 to float
  store float %cast_i64_f3290, ptr %scalar_data89, align 4
  %scalar_tensor92 = call ptr @tl_tensor_new(ptr %scalar_data89, i64 0, ptr %scalar_shape91)
  store float 1.000000e+00, ptr %scalar_data93, align 4
  %scalar_tensor95 = call ptr @tl_tensor_new(ptr %scalar_data93, i64 0, ptr %scalar_shape94)
  %pow_res96 = call ptr @tl_tensor_pow(ptr %scalar_tensor92, ptr %scalar_tensor95)
  %get_res97 = call float @tl_tensor_get(ptr %pow_res96, i64 0)
  store float %get_res97, ptr %v_s_d1, align 4
  %s_d1098 = load i64, ptr %s_d10, align 8
  %cast_i64_f32100 = sitofp i64 %s_d1098 to float
  store float %cast_i64_f32100, ptr %scalar_data99, align 4
  %scalar_tensor102 = call ptr @tl_tensor_new(ptr %scalar_data99, i64 0, ptr %scalar_shape101)
  store float 1.000000e+00, ptr %scalar_data103, align 4
  %scalar_tensor105 = call ptr @tl_tensor_new(ptr %scalar_data103, i64 0, ptr %scalar_shape104)
  %pow_res106 = call ptr @tl_tensor_pow(ptr %scalar_tensor102, ptr %scalar_tensor105)
  %get_res107 = call float @tl_tensor_get(ptr %pow_res106, i64 0)
  store float %get_res107, ptr %v_s_d10, align 4
  %s_d100108 = load i64, ptr %s_d100, align 8
  %cast_i64_f32110 = sitofp i64 %s_d100108 to float
  store float %cast_i64_f32110, ptr %scalar_data109, align 4
  %scalar_tensor112 = call ptr @tl_tensor_new(ptr %scalar_data109, i64 0, ptr %scalar_shape111)
  store float 1.000000e+00, ptr %scalar_data113, align 4
  %scalar_tensor115 = call ptr @tl_tensor_new(ptr %scalar_data113, i64 0, ptr %scalar_shape114)
  %pow_res116 = call ptr @tl_tensor_pow(ptr %scalar_tensor112, ptr %scalar_tensor115)
  %get_res117 = call float @tl_tensor_get(ptr %pow_res116, i64 0)
  store float %get_res117, ptr %v_s_d100, align 4
  %buf_void = call ptr @tl_alloc_tmp(i64 48)
  %v_i_d1118 = load float, ptr %v_i_d1, align 4
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %v_i_d1118, ptr %elem_ptr, align 4
  %v_i_d10119 = load float, ptr %v_i_d10, align 4
  %elem_ptr120 = getelementptr inbounds float, ptr %buf_void, i64 1
  store float %v_i_d10119, ptr %elem_ptr120, align 4
  %elem_ptr121 = getelementptr inbounds float, ptr %buf_void, i64 2
  store float 1.000000e+01, ptr %elem_ptr121, align 4
  %v_j_d1122 = load float, ptr %v_j_d1, align 4
  %elem_ptr123 = getelementptr inbounds float, ptr %buf_void, i64 3
  store float %v_j_d1122, ptr %elem_ptr123, align 4
  %v_j_d10124 = load float, ptr %v_j_d10, align 4
  %elem_ptr125 = getelementptr inbounds float, ptr %buf_void, i64 4
  store float %v_j_d10124, ptr %elem_ptr125, align 4
  %elem_ptr126 = getelementptr inbounds float, ptr %buf_void, i64 5
  store float 1.100000e+01, ptr %elem_ptr126, align 4
  %v_s_d1127 = load float, ptr %v_s_d1, align 4
  %elem_ptr128 = getelementptr inbounds float, ptr %buf_void, i64 6
  store float %v_s_d1127, ptr %elem_ptr128, align 4
  %v_s_d10129 = load float, ptr %v_s_d10, align 4
  %elem_ptr130 = getelementptr inbounds float, ptr %buf_void, i64 7
  store float %v_s_d10129, ptr %elem_ptr130, align 4
  %v_s_d100131 = load float, ptr %v_s_d100, align 4
  %elem_ptr132 = getelementptr inbounds float, ptr %buf_void, i64 8
  store float %v_s_d100131, ptr %elem_ptr132, align 4
  %elem_ptr133 = getelementptr inbounds float, ptr %buf_void, i64 9
  store float 1.200000e+01, ptr %elem_ptr133, align 4
  %elem_ptr134 = getelementptr inbounds float, ptr %buf_void, i64 10
  store float 1.200000e+01, ptr %elem_ptr134, align 4
  %elem_ptr135 = getelementptr inbounds float, ptr %buf_void, i64 11
  store float 1.200000e+01, ptr %elem_ptr135, align 4
  %shape_alloc = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr = getelementptr inbounds i64, ptr %shape_alloc, i64 0
  store i64 12, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %shape_alloc)
  call void @tl_free_tmp(ptr %buf_void)
  call void @tl_free_tmp(ptr %shape_alloc)
  call void @tl_mem_unregister(ptr %new_tensor)
  store ptr %new_tensor, ptr %data, align 8
  %buf_void136 = call ptr @tl_alloc_tmp(i64 48)
  %v_i_d10137 = load float, ptr %v_i_d10, align 4
  %elem_ptr138 = getelementptr inbounds float, ptr %buf_void136, i64 0
  store float %v_i_d10137, ptr %elem_ptr138, align 4
  %elem_ptr139 = getelementptr inbounds float, ptr %buf_void136, i64 1
  store float 1.000000e+01, ptr %elem_ptr139, align 4
  %v_j_d1140 = load float, ptr %v_j_d1, align 4
  %elem_ptr141 = getelementptr inbounds float, ptr %buf_void136, i64 2
  store float %v_j_d1140, ptr %elem_ptr141, align 4
  %v_j_d10142 = load float, ptr %v_j_d10, align 4
  %elem_ptr143 = getelementptr inbounds float, ptr %buf_void136, i64 3
  store float %v_j_d10142, ptr %elem_ptr143, align 4
  %elem_ptr144 = getelementptr inbounds float, ptr %buf_void136, i64 4
  store float 1.100000e+01, ptr %elem_ptr144, align 4
  %v_s_d1145 = load float, ptr %v_s_d1, align 4
  %elem_ptr146 = getelementptr inbounds float, ptr %buf_void136, i64 5
  store float %v_s_d1145, ptr %elem_ptr146, align 4
  %v_s_d10147 = load float, ptr %v_s_d10, align 4
  %elem_ptr148 = getelementptr inbounds float, ptr %buf_void136, i64 6
  store float %v_s_d10147, ptr %elem_ptr148, align 4
  %v_s_d100149 = load float, ptr %v_s_d100, align 4
  %elem_ptr150 = getelementptr inbounds float, ptr %buf_void136, i64 7
  store float %v_s_d100149, ptr %elem_ptr150, align 4
  %elem_ptr151 = getelementptr inbounds float, ptr %buf_void136, i64 8
  store float 1.200000e+01, ptr %elem_ptr151, align 4
  %elem_ptr152 = getelementptr inbounds float, ptr %buf_void136, i64 9
  store float 1.200000e+01, ptr %elem_ptr152, align 4
  %elem_ptr153 = getelementptr inbounds float, ptr %buf_void136, i64 10
  store float 1.200000e+01, ptr %elem_ptr153, align 4
  %elem_ptr154 = getelementptr inbounds float, ptr %buf_void136, i64 11
  store float 1.200000e+01, ptr %elem_ptr154, align 4
  %shape_alloc155 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr156 = getelementptr inbounds i64, ptr %shape_alloc155, i64 0
  store i64 12, ptr %shape_ptr156, align 8
  %new_tensor157 = call ptr @tl_tensor_new(ptr %buf_void136, i64 1, ptr %shape_alloc155)
  call void @tl_free_tmp(ptr %buf_void136)
  call void @tl_free_tmp(ptr %shape_alloc155)
  call void @tl_mem_unregister(ptr %new_tensor157)
  store ptr %new_tensor157, ptr %target, align 8
  %data158 = load ptr, ptr %data, align 8
  %dims_alloca = alloca [2 x i64], align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr, align 8
  %dim_ptr159 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr159, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %data158, ptr %dims_ptr, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res)
  store ptr %reshape_dims_res, ptr %X, align 8
  %target160 = load ptr, ptr %target, align 8
  %dims_alloca161 = alloca [2 x i64], align 8
  %dim_ptr162 = getelementptr [2 x i64], ptr %dims_alloca161, i64 0, i64 0
  store i64 1, ptr %dim_ptr162, align 8
  %dim_ptr163 = getelementptr [2 x i64], ptr %dims_alloca161, i64 0, i64 1
  store i64 12, ptr %dim_ptr163, align 8
  %dims_ptr164 = getelementptr [2 x i64], ptr %dims_alloca161, i64 0, i64 0
  %reshape_dims_res165 = call ptr @tl_tensor_reshape_dims(ptr %target160, ptr %dims_ptr164, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res165)
  store ptr %reshape_dims_res165, ptr %Y, align 8
  %model166 = load ptr, ptr %model1, align 8
  %X167 = load ptr, ptr %X, align 8
  %call_method = call ptr @tl_GPT_forward(ptr %model166, ptr %X167)
  call void @tl_mem_register_tensor(ptr %call_method)
  call void @tl_mem_unregister(ptr %call_method)
  store ptr %call_method, ptr %logits, align 8
  %logits168 = load ptr, ptr %logits, align 8
  %dims_alloca169 = alloca [2 x i64], align 8
  %dim_ptr170 = getelementptr [2 x i64], ptr %dims_alloca169, i64 0, i64 0
  store i64 12, ptr %dim_ptr170, align 8
  %dim_ptr171 = getelementptr [2 x i64], ptr %dims_alloca169, i64 0, i64 1
  store i64 13, ptr %dim_ptr171, align 8
  %dims_ptr172 = getelementptr [2 x i64], ptr %dims_alloca169, i64 0, i64 0
  %reshape_dims_res173 = call ptr @tl_tensor_reshape_dims(ptr %logits168, ptr %dims_ptr172, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res173)
  store ptr %reshape_dims_res173, ptr %logits_flat, align 8
  %Y174 = load ptr, ptr %Y, align 8
  %dims_alloca175 = alloca [1 x i64], align 8
  %dim_ptr176 = getelementptr [1 x i64], ptr %dims_alloca175, i64 0, i64 0
  store i64 12, ptr %dim_ptr176, align 8
  %dims_ptr177 = getelementptr [1 x i64], ptr %dims_alloca175, i64 0, i64 0
  %reshape_dims_res178 = call ptr @tl_tensor_reshape_dims(ptr %Y174, ptr %dims_ptr177, i64 1)
  call void @tl_mem_unregister(ptr %reshape_dims_res178)
  store ptr %reshape_dims_res178, ptr %Y_flat, align 8
  %logits_flat179 = load ptr, ptr %logits_flat, align 8
  %Y_flat180 = load ptr, ptr %Y_flat, align 8
  %ce_res = call ptr @tl_tensor_cross_entropy(ptr %logits_flat179, ptr %Y_flat180)
  call void @tl_mem_unregister(ptr %ce_res)
  store ptr %ce_res, ptr %loss, align 8
  %loss181 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss181)
  %model182 = load ptr, ptr %model1, align 8
  %lr183 = load float, ptr %lr2, align 4
  %call_method184 = call ptr @tl_GPT_step(ptr %model182, float %lr183)
  call void @tl_mem_register_struct(ptr %call_method184)
  call void @tl_mem_unregister(ptr %call_method184)
  %loss185 = load ptr, ptr %loss, align 8
  %detach_res = call ptr @tl_tensor_detach(ptr %loss185, i1 true)
  call void @tl_mem_unregister(ptr %detach_res)
  %old_val = load ptr, ptr %total_loss, align 8
  %is_not_null = icmp ne ptr %old_val, null
  %can_free = and i1 %is_not_null, true
  br i1 %can_free, label %free_block, label %continue_block

for_end:                                          ; preds = %for_header
  %call_tmp = call i64 @get_memory()
  store i64 %call_tmp, ptr %mem_mb, align 8
  call void @tl_print_string(ptr @str_literal)
  %total_loss186 = load ptr, ptr %total_loss, align 8
  call void @tl_tensor_print(ptr %total_loss186)
  call void @tl_print_string(ptr @str_literal.104)
  %mem_mb187 = load i64, ptr %mem_mb, align 8
  call void @tl_print_i64(i64 %mem_mb187)
  call void @tl_mem_exit_scope()
  ret void

free_block:                                       ; preds = %for_body
  call void @tl_tensor_free(ptr %old_val)
  br label %continue_block

continue_block:                                   ; preds = %free_block, %for_body
  call void @tl_mem_unregister(ptr %detach_res)
  store ptr %detach_res, ptr %total_loss, align 8
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx, 1
  br label %for_header
}

define void @main() {
entry:
  %epoch = alloca i64, align 16
  %epochs = alloca i64, align 16
  %lr = alloca float, align 16
  %model = alloca ptr, align 16
  %d_model = alloca i64, align 16
  %vocab_size = alloca i64, align 16
  call void @tl_mem_enter_scope()
  call void @tl_arena_init(i64 307200)
  store i64 13, ptr %vocab_size, align 8
  store i64 128, ptr %d_model, align 8
  %vocab_size1 = load i64, ptr %vocab_size, align 8
  %d_model2 = load i64, ptr %d_model, align 8
  %static_call = call ptr @tl_GPT_new(i64 %vocab_size1, i64 %d_model2)
  call void @tl_mem_unregister(ptr %static_call)
  store ptr %static_call, ptr %model, align 8
  br i1 true, label %then, label %else

then:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
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
  %b1 = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 2
  %sub_ptr9 = load ptr, ptr %b1, align 8
  %l1 = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 0
  %sub_ptr10 = load ptr, ptr %l1, align 8
  %w11 = getelementptr inbounds %LayerNorm, ptr %sub_ptr10, i32 0, i32 0
  %w12 = load ptr, ptr %w11, align 8
  call void @tl_add_parameter(ptr @key_str.107, ptr %w12)
  %b = getelementptr inbounds %LayerNorm, ptr %sub_ptr10, i32 0, i32 1
  %b13 = load ptr, ptr %b, align 8
  call void @tl_add_parameter(ptr @key_str.108, ptr %b13)
  %a = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 1
  %sub_ptr14 = load ptr, ptr %a, align 8
  %q_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 0
  %sub_ptr15 = load ptr, ptr %q_proj, align 8
  %W = getelementptr inbounds %Linear, ptr %sub_ptr15, i32 0, i32 0
  %W16 = load ptr, ptr %W, align 8
  call void @tl_add_parameter(ptr @key_str.109, ptr %W16)
  %b17 = getelementptr inbounds %Linear, ptr %sub_ptr15, i32 0, i32 1
  %b18 = load ptr, ptr %b17, align 8
  call void @tl_add_parameter(ptr @key_str.110, ptr %b18)
  %k_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 1
  %sub_ptr19 = load ptr, ptr %k_proj, align 8
  %W20 = getelementptr inbounds %Linear, ptr %sub_ptr19, i32 0, i32 0
  %W21 = load ptr, ptr %W20, align 8
  call void @tl_add_parameter(ptr @key_str.111, ptr %W21)
  %b22 = getelementptr inbounds %Linear, ptr %sub_ptr19, i32 0, i32 1
  %b23 = load ptr, ptr %b22, align 8
  call void @tl_add_parameter(ptr @key_str.112, ptr %b23)
  %v_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 2
  %sub_ptr24 = load ptr, ptr %v_proj, align 8
  %W25 = getelementptr inbounds %Linear, ptr %sub_ptr24, i32 0, i32 0
  %W26 = load ptr, ptr %W25, align 8
  call void @tl_add_parameter(ptr @key_str.113, ptr %W26)
  %b27 = getelementptr inbounds %Linear, ptr %sub_ptr24, i32 0, i32 1
  %b28 = load ptr, ptr %b27, align 8
  call void @tl_add_parameter(ptr @key_str.114, ptr %b28)
  %p_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 3
  %sub_ptr29 = load ptr, ptr %p_proj, align 8
  %W30 = getelementptr inbounds %Linear, ptr %sub_ptr29, i32 0, i32 0
  %W31 = load ptr, ptr %W30, align 8
  call void @tl_add_parameter(ptr @key_str.115, ptr %W31)
  %b32 = getelementptr inbounds %Linear, ptr %sub_ptr29, i32 0, i32 1
  %b33 = load ptr, ptr %b32, align 8
  call void @tl_add_parameter(ptr @key_str.116, ptr %b33)
  %l2 = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 2
  %sub_ptr34 = load ptr, ptr %l2, align 8
  %w35 = getelementptr inbounds %LayerNorm, ptr %sub_ptr34, i32 0, i32 0
  %w36 = load ptr, ptr %w35, align 8
  call void @tl_add_parameter(ptr @key_str.117, ptr %w36)
  %b37 = getelementptr inbounds %LayerNorm, ptr %sub_ptr34, i32 0, i32 1
  %b38 = load ptr, ptr %b37, align 8
  call void @tl_add_parameter(ptr @key_str.118, ptr %b38)
  %m = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 3
  %sub_ptr39 = load ptr, ptr %m, align 8
  %f = getelementptr inbounds %MLP, ptr %sub_ptr39, i32 0, i32 0
  %sub_ptr40 = load ptr, ptr %f, align 8
  %W41 = getelementptr inbounds %Linear, ptr %sub_ptr40, i32 0, i32 0
  %W42 = load ptr, ptr %W41, align 8
  call void @tl_add_parameter(ptr @key_str.119, ptr %W42)
  %b43 = getelementptr inbounds %Linear, ptr %sub_ptr40, i32 0, i32 1
  %b44 = load ptr, ptr %b43, align 8
  call void @tl_add_parameter(ptr @key_str.120, ptr %b44)
  %p = getelementptr inbounds %MLP, ptr %sub_ptr39, i32 0, i32 1
  %sub_ptr45 = load ptr, ptr %p, align 8
  %W46 = getelementptr inbounds %Linear, ptr %sub_ptr45, i32 0, i32 0
  %W47 = load ptr, ptr %W46, align 8
  call void @tl_add_parameter(ptr @key_str.121, ptr %W47)
  %b48 = getelementptr inbounds %Linear, ptr %sub_ptr45, i32 0, i32 1
  %b49 = load ptr, ptr %b48, align 8
  call void @tl_add_parameter(ptr @key_str.122, ptr %b49)
  %b2 = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 3
  %sub_ptr50 = load ptr, ptr %b2, align 8
  %l151 = getelementptr inbounds %Block, ptr %sub_ptr50, i32 0, i32 0
  %sub_ptr52 = load ptr, ptr %l151, align 8
  %w53 = getelementptr inbounds %LayerNorm, ptr %sub_ptr52, i32 0, i32 0
  %w54 = load ptr, ptr %w53, align 8
  call void @tl_add_parameter(ptr @key_str.123, ptr %w54)
  %b55 = getelementptr inbounds %LayerNorm, ptr %sub_ptr52, i32 0, i32 1
  %b56 = load ptr, ptr %b55, align 8
  call void @tl_add_parameter(ptr @key_str.124, ptr %b56)
  %a57 = getelementptr inbounds %Block, ptr %sub_ptr50, i32 0, i32 1
  %sub_ptr58 = load ptr, ptr %a57, align 8
  %q_proj59 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr58, i32 0, i32 0
  %sub_ptr60 = load ptr, ptr %q_proj59, align 8
  %W61 = getelementptr inbounds %Linear, ptr %sub_ptr60, i32 0, i32 0
  %W62 = load ptr, ptr %W61, align 8
  call void @tl_add_parameter(ptr @key_str.125, ptr %W62)
  %b63 = getelementptr inbounds %Linear, ptr %sub_ptr60, i32 0, i32 1
  %b64 = load ptr, ptr %b63, align 8
  call void @tl_add_parameter(ptr @key_str.126, ptr %b64)
  %k_proj65 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr58, i32 0, i32 1
  %sub_ptr66 = load ptr, ptr %k_proj65, align 8
  %W67 = getelementptr inbounds %Linear, ptr %sub_ptr66, i32 0, i32 0
  %W68 = load ptr, ptr %W67, align 8
  call void @tl_add_parameter(ptr @key_str.127, ptr %W68)
  %b69 = getelementptr inbounds %Linear, ptr %sub_ptr66, i32 0, i32 1
  %b70 = load ptr, ptr %b69, align 8
  call void @tl_add_parameter(ptr @key_str.128, ptr %b70)
  %v_proj71 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr58, i32 0, i32 2
  %sub_ptr72 = load ptr, ptr %v_proj71, align 8
  %W73 = getelementptr inbounds %Linear, ptr %sub_ptr72, i32 0, i32 0
  %W74 = load ptr, ptr %W73, align 8
  call void @tl_add_parameter(ptr @key_str.129, ptr %W74)
  %b75 = getelementptr inbounds %Linear, ptr %sub_ptr72, i32 0, i32 1
  %b76 = load ptr, ptr %b75, align 8
  call void @tl_add_parameter(ptr @key_str.130, ptr %b76)
  %p_proj77 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr58, i32 0, i32 3
  %sub_ptr78 = load ptr, ptr %p_proj77, align 8
  %W79 = getelementptr inbounds %Linear, ptr %sub_ptr78, i32 0, i32 0
  %W80 = load ptr, ptr %W79, align 8
  call void @tl_add_parameter(ptr @key_str.131, ptr %W80)
  %b81 = getelementptr inbounds %Linear, ptr %sub_ptr78, i32 0, i32 1
  %b82 = load ptr, ptr %b81, align 8
  call void @tl_add_parameter(ptr @key_str.132, ptr %b82)
  %l283 = getelementptr inbounds %Block, ptr %sub_ptr50, i32 0, i32 2
  %sub_ptr84 = load ptr, ptr %l283, align 8
  %w85 = getelementptr inbounds %LayerNorm, ptr %sub_ptr84, i32 0, i32 0
  %w86 = load ptr, ptr %w85, align 8
  call void @tl_add_parameter(ptr @key_str.133, ptr %w86)
  %b87 = getelementptr inbounds %LayerNorm, ptr %sub_ptr84, i32 0, i32 1
  %b88 = load ptr, ptr %b87, align 8
  call void @tl_add_parameter(ptr @key_str.134, ptr %b88)
  %m89 = getelementptr inbounds %Block, ptr %sub_ptr50, i32 0, i32 3
  %sub_ptr90 = load ptr, ptr %m89, align 8
  %f91 = getelementptr inbounds %MLP, ptr %sub_ptr90, i32 0, i32 0
  %sub_ptr92 = load ptr, ptr %f91, align 8
  %W93 = getelementptr inbounds %Linear, ptr %sub_ptr92, i32 0, i32 0
  %W94 = load ptr, ptr %W93, align 8
  call void @tl_add_parameter(ptr @key_str.135, ptr %W94)
  %b95 = getelementptr inbounds %Linear, ptr %sub_ptr92, i32 0, i32 1
  %b96 = load ptr, ptr %b95, align 8
  call void @tl_add_parameter(ptr @key_str.136, ptr %b96)
  %p97 = getelementptr inbounds %MLP, ptr %sub_ptr90, i32 0, i32 1
  %sub_ptr98 = load ptr, ptr %p97, align 8
  %W99 = getelementptr inbounds %Linear, ptr %sub_ptr98, i32 0, i32 0
  %W100 = load ptr, ptr %W99, align 8
  call void @tl_add_parameter(ptr @key_str.137, ptr %W100)
  %b101 = getelementptr inbounds %Linear, ptr %sub_ptr98, i32 0, i32 1
  %b102 = load ptr, ptr %b101, align 8
  call void @tl_add_parameter(ptr @key_str.138, ptr %b102)
  %b3 = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 4
  %sub_ptr103 = load ptr, ptr %b3, align 8
  %l1104 = getelementptr inbounds %Block, ptr %sub_ptr103, i32 0, i32 0
  %sub_ptr105 = load ptr, ptr %l1104, align 8
  %w106 = getelementptr inbounds %LayerNorm, ptr %sub_ptr105, i32 0, i32 0
  %w107 = load ptr, ptr %w106, align 8
  call void @tl_add_parameter(ptr @key_str.139, ptr %w107)
  %b108 = getelementptr inbounds %LayerNorm, ptr %sub_ptr105, i32 0, i32 1
  %b109 = load ptr, ptr %b108, align 8
  call void @tl_add_parameter(ptr @key_str.140, ptr %b109)
  %a110 = getelementptr inbounds %Block, ptr %sub_ptr103, i32 0, i32 1
  %sub_ptr111 = load ptr, ptr %a110, align 8
  %q_proj112 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr111, i32 0, i32 0
  %sub_ptr113 = load ptr, ptr %q_proj112, align 8
  %W114 = getelementptr inbounds %Linear, ptr %sub_ptr113, i32 0, i32 0
  %W115 = load ptr, ptr %W114, align 8
  call void @tl_add_parameter(ptr @key_str.141, ptr %W115)
  %b116 = getelementptr inbounds %Linear, ptr %sub_ptr113, i32 0, i32 1
  %b117 = load ptr, ptr %b116, align 8
  call void @tl_add_parameter(ptr @key_str.142, ptr %b117)
  %k_proj118 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr111, i32 0, i32 1
  %sub_ptr119 = load ptr, ptr %k_proj118, align 8
  %W120 = getelementptr inbounds %Linear, ptr %sub_ptr119, i32 0, i32 0
  %W121 = load ptr, ptr %W120, align 8
  call void @tl_add_parameter(ptr @key_str.143, ptr %W121)
  %b122 = getelementptr inbounds %Linear, ptr %sub_ptr119, i32 0, i32 1
  %b123 = load ptr, ptr %b122, align 8
  call void @tl_add_parameter(ptr @key_str.144, ptr %b123)
  %v_proj124 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr111, i32 0, i32 2
  %sub_ptr125 = load ptr, ptr %v_proj124, align 8
  %W126 = getelementptr inbounds %Linear, ptr %sub_ptr125, i32 0, i32 0
  %W127 = load ptr, ptr %W126, align 8
  call void @tl_add_parameter(ptr @key_str.145, ptr %W127)
  %b128 = getelementptr inbounds %Linear, ptr %sub_ptr125, i32 0, i32 1
  %b129 = load ptr, ptr %b128, align 8
  call void @tl_add_parameter(ptr @key_str.146, ptr %b129)
  %p_proj130 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr111, i32 0, i32 3
  %sub_ptr131 = load ptr, ptr %p_proj130, align 8
  %W132 = getelementptr inbounds %Linear, ptr %sub_ptr131, i32 0, i32 0
  %W133 = load ptr, ptr %W132, align 8
  call void @tl_add_parameter(ptr @key_str.147, ptr %W133)
  %b134 = getelementptr inbounds %Linear, ptr %sub_ptr131, i32 0, i32 1
  %b135 = load ptr, ptr %b134, align 8
  call void @tl_add_parameter(ptr @key_str.148, ptr %b135)
  %l2136 = getelementptr inbounds %Block, ptr %sub_ptr103, i32 0, i32 2
  %sub_ptr137 = load ptr, ptr %l2136, align 8
  %w138 = getelementptr inbounds %LayerNorm, ptr %sub_ptr137, i32 0, i32 0
  %w139 = load ptr, ptr %w138, align 8
  call void @tl_add_parameter(ptr @key_str.149, ptr %w139)
  %b140 = getelementptr inbounds %LayerNorm, ptr %sub_ptr137, i32 0, i32 1
  %b141 = load ptr, ptr %b140, align 8
  call void @tl_add_parameter(ptr @key_str.150, ptr %b141)
  %m142 = getelementptr inbounds %Block, ptr %sub_ptr103, i32 0, i32 3
  %sub_ptr143 = load ptr, ptr %m142, align 8
  %f144 = getelementptr inbounds %MLP, ptr %sub_ptr143, i32 0, i32 0
  %sub_ptr145 = load ptr, ptr %f144, align 8
  %W146 = getelementptr inbounds %Linear, ptr %sub_ptr145, i32 0, i32 0
  %W147 = load ptr, ptr %W146, align 8
  call void @tl_add_parameter(ptr @key_str.151, ptr %W147)
  %b148 = getelementptr inbounds %Linear, ptr %sub_ptr145, i32 0, i32 1
  %b149 = load ptr, ptr %b148, align 8
  call void @tl_add_parameter(ptr @key_str.152, ptr %b149)
  %p150 = getelementptr inbounds %MLP, ptr %sub_ptr143, i32 0, i32 1
  %sub_ptr151 = load ptr, ptr %p150, align 8
  %W152 = getelementptr inbounds %Linear, ptr %sub_ptr151, i32 0, i32 0
  %W153 = load ptr, ptr %W152, align 8
  call void @tl_add_parameter(ptr @key_str.153, ptr %W153)
  %b154 = getelementptr inbounds %Linear, ptr %sub_ptr151, i32 0, i32 1
  %b155 = load ptr, ptr %b154, align 8
  call void @tl_add_parameter(ptr @key_str.154, ptr %b155)
  %l = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 5
  %sub_ptr156 = load ptr, ptr %l, align 8
  %w157 = getelementptr inbounds %LayerNorm, ptr %sub_ptr156, i32 0, i32 0
  %w158 = load ptr, ptr %w157, align 8
  call void @tl_add_parameter(ptr @key_str.155, ptr %w158)
  %b159 = getelementptr inbounds %LayerNorm, ptr %sub_ptr156, i32 0, i32 1
  %b160 = load ptr, ptr %b159, align 8
  call void @tl_add_parameter(ptr @key_str.156, ptr %b160)
  %h = getelementptr inbounds %GPT, ptr %model3, i32 0, i32 6
  %sub_ptr161 = load ptr, ptr %h, align 8
  %W162 = getelementptr inbounds %Linear, ptr %sub_ptr161, i32 0, i32 0
  %W163 = load ptr, ptr %W162, align 8
  call void @tl_add_parameter(ptr @key_str.157, ptr %W163)
  %b164 = getelementptr inbounds %Linear, ptr %sub_ptr161, i32 0, i32 1
  %b165 = load ptr, ptr %b164, align 8
  call void @tl_add_parameter(ptr @key_str.158, ptr %b165)
  call void @tl_load_all_params(ptr @str_literal.159)
  call void @tl_mem_exit_scope()
  br label %merge

else:                                             ; preds = %entry
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else, %then
  store float 0x3F40624DE0000000, ptr %lr, align 4
  store i64 20, ptr %epochs, align 8
  call void @tl_print_string(ptr @str_literal.160)
  %epochs166 = load i64, ptr %epochs, align 8
  br label %for_header

for_header:                                       ; preds = %for_body, %merge
  %for_idx = phi i64 [ 0, %merge ], [ %next_idx, %for_body ]
  %for_cond = icmp slt i64 %for_idx, %epochs166
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %epoch, align 8
  call void @tl_print_string(ptr @str_literal.161)
  %epoch167 = load i64, ptr %epoch, align 8
  call void @tl_print_i64(i64 %epoch167)
  %model168 = load ptr, ptr %model, align 8
  %lr169 = load float, ptr %lr, align 4
  %epoch170 = load i64, ptr %epoch, align 8
  call void @train_epoch(ptr %model168, float %lr169, i64 %epoch170)
  %model171 = load ptr, ptr %model, align 8
  %w172 = getelementptr inbounds %GPT, ptr %model171, i32 0, i32 0
  %sub_ptr173 = load ptr, ptr %w172, align 8
  %w174 = getelementptr inbounds %Embedding, ptr %sub_ptr173, i32 0, i32 0
  %w175 = load ptr, ptr %w174, align 8
  call void @tl_add_parameter(ptr @key_str.162, ptr %w175)
  %wp176 = getelementptr inbounds %GPT, ptr %model171, i32 0, i32 1
  %sub_ptr177 = load ptr, ptr %wp176, align 8
  %w178 = getelementptr inbounds %Embedding, ptr %sub_ptr177, i32 0, i32 0
  %w179 = load ptr, ptr %w178, align 8
  call void @tl_add_parameter(ptr @key_str.163, ptr %w179)
  %b1180 = getelementptr inbounds %GPT, ptr %model171, i32 0, i32 2
  %sub_ptr181 = load ptr, ptr %b1180, align 8
  %l1182 = getelementptr inbounds %Block, ptr %sub_ptr181, i32 0, i32 0
  %sub_ptr183 = load ptr, ptr %l1182, align 8
  %w184 = getelementptr inbounds %LayerNorm, ptr %sub_ptr183, i32 0, i32 0
  %w185 = load ptr, ptr %w184, align 8
  call void @tl_add_parameter(ptr @key_str.164, ptr %w185)
  %b186 = getelementptr inbounds %LayerNorm, ptr %sub_ptr183, i32 0, i32 1
  %b187 = load ptr, ptr %b186, align 8
  call void @tl_add_parameter(ptr @key_str.165, ptr %b187)
  %a188 = getelementptr inbounds %Block, ptr %sub_ptr181, i32 0, i32 1
  %sub_ptr189 = load ptr, ptr %a188, align 8
  %q_proj190 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr189, i32 0, i32 0
  %sub_ptr191 = load ptr, ptr %q_proj190, align 8
  %W192 = getelementptr inbounds %Linear, ptr %sub_ptr191, i32 0, i32 0
  %W193 = load ptr, ptr %W192, align 8
  call void @tl_add_parameter(ptr @key_str.166, ptr %W193)
  %b194 = getelementptr inbounds %Linear, ptr %sub_ptr191, i32 0, i32 1
  %b195 = load ptr, ptr %b194, align 8
  call void @tl_add_parameter(ptr @key_str.167, ptr %b195)
  %k_proj196 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr189, i32 0, i32 1
  %sub_ptr197 = load ptr, ptr %k_proj196, align 8
  %W198 = getelementptr inbounds %Linear, ptr %sub_ptr197, i32 0, i32 0
  %W199 = load ptr, ptr %W198, align 8
  call void @tl_add_parameter(ptr @key_str.168, ptr %W199)
  %b200 = getelementptr inbounds %Linear, ptr %sub_ptr197, i32 0, i32 1
  %b201 = load ptr, ptr %b200, align 8
  call void @tl_add_parameter(ptr @key_str.169, ptr %b201)
  %v_proj202 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr189, i32 0, i32 2
  %sub_ptr203 = load ptr, ptr %v_proj202, align 8
  %W204 = getelementptr inbounds %Linear, ptr %sub_ptr203, i32 0, i32 0
  %W205 = load ptr, ptr %W204, align 8
  call void @tl_add_parameter(ptr @key_str.170, ptr %W205)
  %b206 = getelementptr inbounds %Linear, ptr %sub_ptr203, i32 0, i32 1
  %b207 = load ptr, ptr %b206, align 8
  call void @tl_add_parameter(ptr @key_str.171, ptr %b207)
  %p_proj208 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr189, i32 0, i32 3
  %sub_ptr209 = load ptr, ptr %p_proj208, align 8
  %W210 = getelementptr inbounds %Linear, ptr %sub_ptr209, i32 0, i32 0
  %W211 = load ptr, ptr %W210, align 8
  call void @tl_add_parameter(ptr @key_str.172, ptr %W211)
  %b212 = getelementptr inbounds %Linear, ptr %sub_ptr209, i32 0, i32 1
  %b213 = load ptr, ptr %b212, align 8
  call void @tl_add_parameter(ptr @key_str.173, ptr %b213)
  %l2214 = getelementptr inbounds %Block, ptr %sub_ptr181, i32 0, i32 2
  %sub_ptr215 = load ptr, ptr %l2214, align 8
  %w216 = getelementptr inbounds %LayerNorm, ptr %sub_ptr215, i32 0, i32 0
  %w217 = load ptr, ptr %w216, align 8
  call void @tl_add_parameter(ptr @key_str.174, ptr %w217)
  %b218 = getelementptr inbounds %LayerNorm, ptr %sub_ptr215, i32 0, i32 1
  %b219 = load ptr, ptr %b218, align 8
  call void @tl_add_parameter(ptr @key_str.175, ptr %b219)
  %m220 = getelementptr inbounds %Block, ptr %sub_ptr181, i32 0, i32 3
  %sub_ptr221 = load ptr, ptr %m220, align 8
  %f222 = getelementptr inbounds %MLP, ptr %sub_ptr221, i32 0, i32 0
  %sub_ptr223 = load ptr, ptr %f222, align 8
  %W224 = getelementptr inbounds %Linear, ptr %sub_ptr223, i32 0, i32 0
  %W225 = load ptr, ptr %W224, align 8
  call void @tl_add_parameter(ptr @key_str.176, ptr %W225)
  %b226 = getelementptr inbounds %Linear, ptr %sub_ptr223, i32 0, i32 1
  %b227 = load ptr, ptr %b226, align 8
  call void @tl_add_parameter(ptr @key_str.177, ptr %b227)
  %p228 = getelementptr inbounds %MLP, ptr %sub_ptr221, i32 0, i32 1
  %sub_ptr229 = load ptr, ptr %p228, align 8
  %W230 = getelementptr inbounds %Linear, ptr %sub_ptr229, i32 0, i32 0
  %W231 = load ptr, ptr %W230, align 8
  call void @tl_add_parameter(ptr @key_str.178, ptr %W231)
  %b232 = getelementptr inbounds %Linear, ptr %sub_ptr229, i32 0, i32 1
  %b233 = load ptr, ptr %b232, align 8
  call void @tl_add_parameter(ptr @key_str.179, ptr %b233)
  %b2234 = getelementptr inbounds %GPT, ptr %model171, i32 0, i32 3
  %sub_ptr235 = load ptr, ptr %b2234, align 8
  %l1236 = getelementptr inbounds %Block, ptr %sub_ptr235, i32 0, i32 0
  %sub_ptr237 = load ptr, ptr %l1236, align 8
  %w238 = getelementptr inbounds %LayerNorm, ptr %sub_ptr237, i32 0, i32 0
  %w239 = load ptr, ptr %w238, align 8
  call void @tl_add_parameter(ptr @key_str.180, ptr %w239)
  %b240 = getelementptr inbounds %LayerNorm, ptr %sub_ptr237, i32 0, i32 1
  %b241 = load ptr, ptr %b240, align 8
  call void @tl_add_parameter(ptr @key_str.181, ptr %b241)
  %a242 = getelementptr inbounds %Block, ptr %sub_ptr235, i32 0, i32 1
  %sub_ptr243 = load ptr, ptr %a242, align 8
  %q_proj244 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr243, i32 0, i32 0
  %sub_ptr245 = load ptr, ptr %q_proj244, align 8
  %W246 = getelementptr inbounds %Linear, ptr %sub_ptr245, i32 0, i32 0
  %W247 = load ptr, ptr %W246, align 8
  call void @tl_add_parameter(ptr @key_str.182, ptr %W247)
  %b248 = getelementptr inbounds %Linear, ptr %sub_ptr245, i32 0, i32 1
  %b249 = load ptr, ptr %b248, align 8
  call void @tl_add_parameter(ptr @key_str.183, ptr %b249)
  %k_proj250 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr243, i32 0, i32 1
  %sub_ptr251 = load ptr, ptr %k_proj250, align 8
  %W252 = getelementptr inbounds %Linear, ptr %sub_ptr251, i32 0, i32 0
  %W253 = load ptr, ptr %W252, align 8
  call void @tl_add_parameter(ptr @key_str.184, ptr %W253)
  %b254 = getelementptr inbounds %Linear, ptr %sub_ptr251, i32 0, i32 1
  %b255 = load ptr, ptr %b254, align 8
  call void @tl_add_parameter(ptr @key_str.185, ptr %b255)
  %v_proj256 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr243, i32 0, i32 2
  %sub_ptr257 = load ptr, ptr %v_proj256, align 8
  %W258 = getelementptr inbounds %Linear, ptr %sub_ptr257, i32 0, i32 0
  %W259 = load ptr, ptr %W258, align 8
  call void @tl_add_parameter(ptr @key_str.186, ptr %W259)
  %b260 = getelementptr inbounds %Linear, ptr %sub_ptr257, i32 0, i32 1
  %b261 = load ptr, ptr %b260, align 8
  call void @tl_add_parameter(ptr @key_str.187, ptr %b261)
  %p_proj262 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr243, i32 0, i32 3
  %sub_ptr263 = load ptr, ptr %p_proj262, align 8
  %W264 = getelementptr inbounds %Linear, ptr %sub_ptr263, i32 0, i32 0
  %W265 = load ptr, ptr %W264, align 8
  call void @tl_add_parameter(ptr @key_str.188, ptr %W265)
  %b266 = getelementptr inbounds %Linear, ptr %sub_ptr263, i32 0, i32 1
  %b267 = load ptr, ptr %b266, align 8
  call void @tl_add_parameter(ptr @key_str.189, ptr %b267)
  %l2268 = getelementptr inbounds %Block, ptr %sub_ptr235, i32 0, i32 2
  %sub_ptr269 = load ptr, ptr %l2268, align 8
  %w270 = getelementptr inbounds %LayerNorm, ptr %sub_ptr269, i32 0, i32 0
  %w271 = load ptr, ptr %w270, align 8
  call void @tl_add_parameter(ptr @key_str.190, ptr %w271)
  %b272 = getelementptr inbounds %LayerNorm, ptr %sub_ptr269, i32 0, i32 1
  %b273 = load ptr, ptr %b272, align 8
  call void @tl_add_parameter(ptr @key_str.191, ptr %b273)
  %m274 = getelementptr inbounds %Block, ptr %sub_ptr235, i32 0, i32 3
  %sub_ptr275 = load ptr, ptr %m274, align 8
  %f276 = getelementptr inbounds %MLP, ptr %sub_ptr275, i32 0, i32 0
  %sub_ptr277 = load ptr, ptr %f276, align 8
  %W278 = getelementptr inbounds %Linear, ptr %sub_ptr277, i32 0, i32 0
  %W279 = load ptr, ptr %W278, align 8
  call void @tl_add_parameter(ptr @key_str.192, ptr %W279)
  %b280 = getelementptr inbounds %Linear, ptr %sub_ptr277, i32 0, i32 1
  %b281 = load ptr, ptr %b280, align 8
  call void @tl_add_parameter(ptr @key_str.193, ptr %b281)
  %p282 = getelementptr inbounds %MLP, ptr %sub_ptr275, i32 0, i32 1
  %sub_ptr283 = load ptr, ptr %p282, align 8
  %W284 = getelementptr inbounds %Linear, ptr %sub_ptr283, i32 0, i32 0
  %W285 = load ptr, ptr %W284, align 8
  call void @tl_add_parameter(ptr @key_str.194, ptr %W285)
  %b286 = getelementptr inbounds %Linear, ptr %sub_ptr283, i32 0, i32 1
  %b287 = load ptr, ptr %b286, align 8
  call void @tl_add_parameter(ptr @key_str.195, ptr %b287)
  %b3288 = getelementptr inbounds %GPT, ptr %model171, i32 0, i32 4
  %sub_ptr289 = load ptr, ptr %b3288, align 8
  %l1290 = getelementptr inbounds %Block, ptr %sub_ptr289, i32 0, i32 0
  %sub_ptr291 = load ptr, ptr %l1290, align 8
  %w292 = getelementptr inbounds %LayerNorm, ptr %sub_ptr291, i32 0, i32 0
  %w293 = load ptr, ptr %w292, align 8
  call void @tl_add_parameter(ptr @key_str.196, ptr %w293)
  %b294 = getelementptr inbounds %LayerNorm, ptr %sub_ptr291, i32 0, i32 1
  %b295 = load ptr, ptr %b294, align 8
  call void @tl_add_parameter(ptr @key_str.197, ptr %b295)
  %a296 = getelementptr inbounds %Block, ptr %sub_ptr289, i32 0, i32 1
  %sub_ptr297 = load ptr, ptr %a296, align 8
  %q_proj298 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr297, i32 0, i32 0
  %sub_ptr299 = load ptr, ptr %q_proj298, align 8
  %W300 = getelementptr inbounds %Linear, ptr %sub_ptr299, i32 0, i32 0
  %W301 = load ptr, ptr %W300, align 8
  call void @tl_add_parameter(ptr @key_str.198, ptr %W301)
  %b302 = getelementptr inbounds %Linear, ptr %sub_ptr299, i32 0, i32 1
  %b303 = load ptr, ptr %b302, align 8
  call void @tl_add_parameter(ptr @key_str.199, ptr %b303)
  %k_proj304 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr297, i32 0, i32 1
  %sub_ptr305 = load ptr, ptr %k_proj304, align 8
  %W306 = getelementptr inbounds %Linear, ptr %sub_ptr305, i32 0, i32 0
  %W307 = load ptr, ptr %W306, align 8
  call void @tl_add_parameter(ptr @key_str.200, ptr %W307)
  %b308 = getelementptr inbounds %Linear, ptr %sub_ptr305, i32 0, i32 1
  %b309 = load ptr, ptr %b308, align 8
  call void @tl_add_parameter(ptr @key_str.201, ptr %b309)
  %v_proj310 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr297, i32 0, i32 2
  %sub_ptr311 = load ptr, ptr %v_proj310, align 8
  %W312 = getelementptr inbounds %Linear, ptr %sub_ptr311, i32 0, i32 0
  %W313 = load ptr, ptr %W312, align 8
  call void @tl_add_parameter(ptr @key_str.202, ptr %W313)
  %b314 = getelementptr inbounds %Linear, ptr %sub_ptr311, i32 0, i32 1
  %b315 = load ptr, ptr %b314, align 8
  call void @tl_add_parameter(ptr @key_str.203, ptr %b315)
  %p_proj316 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr297, i32 0, i32 3
  %sub_ptr317 = load ptr, ptr %p_proj316, align 8
  %W318 = getelementptr inbounds %Linear, ptr %sub_ptr317, i32 0, i32 0
  %W319 = load ptr, ptr %W318, align 8
  call void @tl_add_parameter(ptr @key_str.204, ptr %W319)
  %b320 = getelementptr inbounds %Linear, ptr %sub_ptr317, i32 0, i32 1
  %b321 = load ptr, ptr %b320, align 8
  call void @tl_add_parameter(ptr @key_str.205, ptr %b321)
  %l2322 = getelementptr inbounds %Block, ptr %sub_ptr289, i32 0, i32 2
  %sub_ptr323 = load ptr, ptr %l2322, align 8
  %w324 = getelementptr inbounds %LayerNorm, ptr %sub_ptr323, i32 0, i32 0
  %w325 = load ptr, ptr %w324, align 8
  call void @tl_add_parameter(ptr @key_str.206, ptr %w325)
  %b326 = getelementptr inbounds %LayerNorm, ptr %sub_ptr323, i32 0, i32 1
  %b327 = load ptr, ptr %b326, align 8
  call void @tl_add_parameter(ptr @key_str.207, ptr %b327)
  %m328 = getelementptr inbounds %Block, ptr %sub_ptr289, i32 0, i32 3
  %sub_ptr329 = load ptr, ptr %m328, align 8
  %f330 = getelementptr inbounds %MLP, ptr %sub_ptr329, i32 0, i32 0
  %sub_ptr331 = load ptr, ptr %f330, align 8
  %W332 = getelementptr inbounds %Linear, ptr %sub_ptr331, i32 0, i32 0
  %W333 = load ptr, ptr %W332, align 8
  call void @tl_add_parameter(ptr @key_str.208, ptr %W333)
  %b334 = getelementptr inbounds %Linear, ptr %sub_ptr331, i32 0, i32 1
  %b335 = load ptr, ptr %b334, align 8
  call void @tl_add_parameter(ptr @key_str.209, ptr %b335)
  %p336 = getelementptr inbounds %MLP, ptr %sub_ptr329, i32 0, i32 1
  %sub_ptr337 = load ptr, ptr %p336, align 8
  %W338 = getelementptr inbounds %Linear, ptr %sub_ptr337, i32 0, i32 0
  %W339 = load ptr, ptr %W338, align 8
  call void @tl_add_parameter(ptr @key_str.210, ptr %W339)
  %b340 = getelementptr inbounds %Linear, ptr %sub_ptr337, i32 0, i32 1
  %b341 = load ptr, ptr %b340, align 8
  call void @tl_add_parameter(ptr @key_str.211, ptr %b341)
  %l342 = getelementptr inbounds %GPT, ptr %model171, i32 0, i32 5
  %sub_ptr343 = load ptr, ptr %l342, align 8
  %w344 = getelementptr inbounds %LayerNorm, ptr %sub_ptr343, i32 0, i32 0
  %w345 = load ptr, ptr %w344, align 8
  call void @tl_add_parameter(ptr @key_str.212, ptr %w345)
  %b346 = getelementptr inbounds %LayerNorm, ptr %sub_ptr343, i32 0, i32 1
  %b347 = load ptr, ptr %b346, align 8
  call void @tl_add_parameter(ptr @key_str.213, ptr %b347)
  %h348 = getelementptr inbounds %GPT, ptr %model171, i32 0, i32 6
  %sub_ptr349 = load ptr, ptr %h348, align 8
  %W350 = getelementptr inbounds %Linear, ptr %sub_ptr349, i32 0, i32 0
  %W351 = load ptr, ptr %W350, align 8
  call void @tl_add_parameter(ptr @key_str.214, ptr %W351)
  %b352 = getelementptr inbounds %Linear, ptr %sub_ptr349, i32 0, i32 1
  %b353 = load ptr, ptr %b352, align 8
  call void @tl_add_parameter(ptr @key_str.215, ptr %b353)
  call void @tl_save_all_params(ptr @str_literal.216)
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx, 1
  br label %for_header

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal.217)
  %model354 = load ptr, ptr %model, align 8
  %w355 = getelementptr inbounds %GPT, ptr %model354, i32 0, i32 0
  %sub_ptr356 = load ptr, ptr %w355, align 8
  %w357 = getelementptr inbounds %Embedding, ptr %sub_ptr356, i32 0, i32 0
  %w358 = load ptr, ptr %w357, align 8
  call void @tl_add_parameter(ptr @key_str.218, ptr %w358)
  %wp359 = getelementptr inbounds %GPT, ptr %model354, i32 0, i32 1
  %sub_ptr360 = load ptr, ptr %wp359, align 8
  %w361 = getelementptr inbounds %Embedding, ptr %sub_ptr360, i32 0, i32 0
  %w362 = load ptr, ptr %w361, align 8
  call void @tl_add_parameter(ptr @key_str.219, ptr %w362)
  %b1363 = getelementptr inbounds %GPT, ptr %model354, i32 0, i32 2
  %sub_ptr364 = load ptr, ptr %b1363, align 8
  %l1365 = getelementptr inbounds %Block, ptr %sub_ptr364, i32 0, i32 0
  %sub_ptr366 = load ptr, ptr %l1365, align 8
  %w367 = getelementptr inbounds %LayerNorm, ptr %sub_ptr366, i32 0, i32 0
  %w368 = load ptr, ptr %w367, align 8
  call void @tl_add_parameter(ptr @key_str.220, ptr %w368)
  %b369 = getelementptr inbounds %LayerNorm, ptr %sub_ptr366, i32 0, i32 1
  %b370 = load ptr, ptr %b369, align 8
  call void @tl_add_parameter(ptr @key_str.221, ptr %b370)
  %a371 = getelementptr inbounds %Block, ptr %sub_ptr364, i32 0, i32 1
  %sub_ptr372 = load ptr, ptr %a371, align 8
  %q_proj373 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr372, i32 0, i32 0
  %sub_ptr374 = load ptr, ptr %q_proj373, align 8
  %W375 = getelementptr inbounds %Linear, ptr %sub_ptr374, i32 0, i32 0
  %W376 = load ptr, ptr %W375, align 8
  call void @tl_add_parameter(ptr @key_str.222, ptr %W376)
  %b377 = getelementptr inbounds %Linear, ptr %sub_ptr374, i32 0, i32 1
  %b378 = load ptr, ptr %b377, align 8
  call void @tl_add_parameter(ptr @key_str.223, ptr %b378)
  %k_proj379 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr372, i32 0, i32 1
  %sub_ptr380 = load ptr, ptr %k_proj379, align 8
  %W381 = getelementptr inbounds %Linear, ptr %sub_ptr380, i32 0, i32 0
  %W382 = load ptr, ptr %W381, align 8
  call void @tl_add_parameter(ptr @key_str.224, ptr %W382)
  %b383 = getelementptr inbounds %Linear, ptr %sub_ptr380, i32 0, i32 1
  %b384 = load ptr, ptr %b383, align 8
  call void @tl_add_parameter(ptr @key_str.225, ptr %b384)
  %v_proj385 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr372, i32 0, i32 2
  %sub_ptr386 = load ptr, ptr %v_proj385, align 8
  %W387 = getelementptr inbounds %Linear, ptr %sub_ptr386, i32 0, i32 0
  %W388 = load ptr, ptr %W387, align 8
  call void @tl_add_parameter(ptr @key_str.226, ptr %W388)
  %b389 = getelementptr inbounds %Linear, ptr %sub_ptr386, i32 0, i32 1
  %b390 = load ptr, ptr %b389, align 8
  call void @tl_add_parameter(ptr @key_str.227, ptr %b390)
  %p_proj391 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr372, i32 0, i32 3
  %sub_ptr392 = load ptr, ptr %p_proj391, align 8
  %W393 = getelementptr inbounds %Linear, ptr %sub_ptr392, i32 0, i32 0
  %W394 = load ptr, ptr %W393, align 8
  call void @tl_add_parameter(ptr @key_str.228, ptr %W394)
  %b395 = getelementptr inbounds %Linear, ptr %sub_ptr392, i32 0, i32 1
  %b396 = load ptr, ptr %b395, align 8
  call void @tl_add_parameter(ptr @key_str.229, ptr %b396)
  %l2397 = getelementptr inbounds %Block, ptr %sub_ptr364, i32 0, i32 2
  %sub_ptr398 = load ptr, ptr %l2397, align 8
  %w399 = getelementptr inbounds %LayerNorm, ptr %sub_ptr398, i32 0, i32 0
  %w400 = load ptr, ptr %w399, align 8
  call void @tl_add_parameter(ptr @key_str.230, ptr %w400)
  %b401 = getelementptr inbounds %LayerNorm, ptr %sub_ptr398, i32 0, i32 1
  %b402 = load ptr, ptr %b401, align 8
  call void @tl_add_parameter(ptr @key_str.231, ptr %b402)
  %m403 = getelementptr inbounds %Block, ptr %sub_ptr364, i32 0, i32 3
  %sub_ptr404 = load ptr, ptr %m403, align 8
  %f405 = getelementptr inbounds %MLP, ptr %sub_ptr404, i32 0, i32 0
  %sub_ptr406 = load ptr, ptr %f405, align 8
  %W407 = getelementptr inbounds %Linear, ptr %sub_ptr406, i32 0, i32 0
  %W408 = load ptr, ptr %W407, align 8
  call void @tl_add_parameter(ptr @key_str.232, ptr %W408)
  %b409 = getelementptr inbounds %Linear, ptr %sub_ptr406, i32 0, i32 1
  %b410 = load ptr, ptr %b409, align 8
  call void @tl_add_parameter(ptr @key_str.233, ptr %b410)
  %p411 = getelementptr inbounds %MLP, ptr %sub_ptr404, i32 0, i32 1
  %sub_ptr412 = load ptr, ptr %p411, align 8
  %W413 = getelementptr inbounds %Linear, ptr %sub_ptr412, i32 0, i32 0
  %W414 = load ptr, ptr %W413, align 8
  call void @tl_add_parameter(ptr @key_str.234, ptr %W414)
  %b415 = getelementptr inbounds %Linear, ptr %sub_ptr412, i32 0, i32 1
  %b416 = load ptr, ptr %b415, align 8
  call void @tl_add_parameter(ptr @key_str.235, ptr %b416)
  %b2417 = getelementptr inbounds %GPT, ptr %model354, i32 0, i32 3
  %sub_ptr418 = load ptr, ptr %b2417, align 8
  %l1419 = getelementptr inbounds %Block, ptr %sub_ptr418, i32 0, i32 0
  %sub_ptr420 = load ptr, ptr %l1419, align 8
  %w421 = getelementptr inbounds %LayerNorm, ptr %sub_ptr420, i32 0, i32 0
  %w422 = load ptr, ptr %w421, align 8
  call void @tl_add_parameter(ptr @key_str.236, ptr %w422)
  %b423 = getelementptr inbounds %LayerNorm, ptr %sub_ptr420, i32 0, i32 1
  %b424 = load ptr, ptr %b423, align 8
  call void @tl_add_parameter(ptr @key_str.237, ptr %b424)
  %a425 = getelementptr inbounds %Block, ptr %sub_ptr418, i32 0, i32 1
  %sub_ptr426 = load ptr, ptr %a425, align 8
  %q_proj427 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr426, i32 0, i32 0
  %sub_ptr428 = load ptr, ptr %q_proj427, align 8
  %W429 = getelementptr inbounds %Linear, ptr %sub_ptr428, i32 0, i32 0
  %W430 = load ptr, ptr %W429, align 8
  call void @tl_add_parameter(ptr @key_str.238, ptr %W430)
  %b431 = getelementptr inbounds %Linear, ptr %sub_ptr428, i32 0, i32 1
  %b432 = load ptr, ptr %b431, align 8
  call void @tl_add_parameter(ptr @key_str.239, ptr %b432)
  %k_proj433 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr426, i32 0, i32 1
  %sub_ptr434 = load ptr, ptr %k_proj433, align 8
  %W435 = getelementptr inbounds %Linear, ptr %sub_ptr434, i32 0, i32 0
  %W436 = load ptr, ptr %W435, align 8
  call void @tl_add_parameter(ptr @key_str.240, ptr %W436)
  %b437 = getelementptr inbounds %Linear, ptr %sub_ptr434, i32 0, i32 1
  %b438 = load ptr, ptr %b437, align 8
  call void @tl_add_parameter(ptr @key_str.241, ptr %b438)
  %v_proj439 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr426, i32 0, i32 2
  %sub_ptr440 = load ptr, ptr %v_proj439, align 8
  %W441 = getelementptr inbounds %Linear, ptr %sub_ptr440, i32 0, i32 0
  %W442 = load ptr, ptr %W441, align 8
  call void @tl_add_parameter(ptr @key_str.242, ptr %W442)
  %b443 = getelementptr inbounds %Linear, ptr %sub_ptr440, i32 0, i32 1
  %b444 = load ptr, ptr %b443, align 8
  call void @tl_add_parameter(ptr @key_str.243, ptr %b444)
  %p_proj445 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr426, i32 0, i32 3
  %sub_ptr446 = load ptr, ptr %p_proj445, align 8
  %W447 = getelementptr inbounds %Linear, ptr %sub_ptr446, i32 0, i32 0
  %W448 = load ptr, ptr %W447, align 8
  call void @tl_add_parameter(ptr @key_str.244, ptr %W448)
  %b449 = getelementptr inbounds %Linear, ptr %sub_ptr446, i32 0, i32 1
  %b450 = load ptr, ptr %b449, align 8
  call void @tl_add_parameter(ptr @key_str.245, ptr %b450)
  %l2451 = getelementptr inbounds %Block, ptr %sub_ptr418, i32 0, i32 2
  %sub_ptr452 = load ptr, ptr %l2451, align 8
  %w453 = getelementptr inbounds %LayerNorm, ptr %sub_ptr452, i32 0, i32 0
  %w454 = load ptr, ptr %w453, align 8
  call void @tl_add_parameter(ptr @key_str.246, ptr %w454)
  %b455 = getelementptr inbounds %LayerNorm, ptr %sub_ptr452, i32 0, i32 1
  %b456 = load ptr, ptr %b455, align 8
  call void @tl_add_parameter(ptr @key_str.247, ptr %b456)
  %m457 = getelementptr inbounds %Block, ptr %sub_ptr418, i32 0, i32 3
  %sub_ptr458 = load ptr, ptr %m457, align 8
  %f459 = getelementptr inbounds %MLP, ptr %sub_ptr458, i32 0, i32 0
  %sub_ptr460 = load ptr, ptr %f459, align 8
  %W461 = getelementptr inbounds %Linear, ptr %sub_ptr460, i32 0, i32 0
  %W462 = load ptr, ptr %W461, align 8
  call void @tl_add_parameter(ptr @key_str.248, ptr %W462)
  %b463 = getelementptr inbounds %Linear, ptr %sub_ptr460, i32 0, i32 1
  %b464 = load ptr, ptr %b463, align 8
  call void @tl_add_parameter(ptr @key_str.249, ptr %b464)
  %p465 = getelementptr inbounds %MLP, ptr %sub_ptr458, i32 0, i32 1
  %sub_ptr466 = load ptr, ptr %p465, align 8
  %W467 = getelementptr inbounds %Linear, ptr %sub_ptr466, i32 0, i32 0
  %W468 = load ptr, ptr %W467, align 8
  call void @tl_add_parameter(ptr @key_str.250, ptr %W468)
  %b469 = getelementptr inbounds %Linear, ptr %sub_ptr466, i32 0, i32 1
  %b470 = load ptr, ptr %b469, align 8
  call void @tl_add_parameter(ptr @key_str.251, ptr %b470)
  %b3471 = getelementptr inbounds %GPT, ptr %model354, i32 0, i32 4
  %sub_ptr472 = load ptr, ptr %b3471, align 8
  %l1473 = getelementptr inbounds %Block, ptr %sub_ptr472, i32 0, i32 0
  %sub_ptr474 = load ptr, ptr %l1473, align 8
  %w475 = getelementptr inbounds %LayerNorm, ptr %sub_ptr474, i32 0, i32 0
  %w476 = load ptr, ptr %w475, align 8
  call void @tl_add_parameter(ptr @key_str.252, ptr %w476)
  %b477 = getelementptr inbounds %LayerNorm, ptr %sub_ptr474, i32 0, i32 1
  %b478 = load ptr, ptr %b477, align 8
  call void @tl_add_parameter(ptr @key_str.253, ptr %b478)
  %a479 = getelementptr inbounds %Block, ptr %sub_ptr472, i32 0, i32 1
  %sub_ptr480 = load ptr, ptr %a479, align 8
  %q_proj481 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr480, i32 0, i32 0
  %sub_ptr482 = load ptr, ptr %q_proj481, align 8
  %W483 = getelementptr inbounds %Linear, ptr %sub_ptr482, i32 0, i32 0
  %W484 = load ptr, ptr %W483, align 8
  call void @tl_add_parameter(ptr @key_str.254, ptr %W484)
  %b485 = getelementptr inbounds %Linear, ptr %sub_ptr482, i32 0, i32 1
  %b486 = load ptr, ptr %b485, align 8
  call void @tl_add_parameter(ptr @key_str.255, ptr %b486)
  %k_proj487 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr480, i32 0, i32 1
  %sub_ptr488 = load ptr, ptr %k_proj487, align 8
  %W489 = getelementptr inbounds %Linear, ptr %sub_ptr488, i32 0, i32 0
  %W490 = load ptr, ptr %W489, align 8
  call void @tl_add_parameter(ptr @key_str.256, ptr %W490)
  %b491 = getelementptr inbounds %Linear, ptr %sub_ptr488, i32 0, i32 1
  %b492 = load ptr, ptr %b491, align 8
  call void @tl_add_parameter(ptr @key_str.257, ptr %b492)
  %v_proj493 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr480, i32 0, i32 2
  %sub_ptr494 = load ptr, ptr %v_proj493, align 8
  %W495 = getelementptr inbounds %Linear, ptr %sub_ptr494, i32 0, i32 0
  %W496 = load ptr, ptr %W495, align 8
  call void @tl_add_parameter(ptr @key_str.258, ptr %W496)
  %b497 = getelementptr inbounds %Linear, ptr %sub_ptr494, i32 0, i32 1
  %b498 = load ptr, ptr %b497, align 8
  call void @tl_add_parameter(ptr @key_str.259, ptr %b498)
  %p_proj499 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr480, i32 0, i32 3
  %sub_ptr500 = load ptr, ptr %p_proj499, align 8
  %W501 = getelementptr inbounds %Linear, ptr %sub_ptr500, i32 0, i32 0
  %W502 = load ptr, ptr %W501, align 8
  call void @tl_add_parameter(ptr @key_str.260, ptr %W502)
  %b503 = getelementptr inbounds %Linear, ptr %sub_ptr500, i32 0, i32 1
  %b504 = load ptr, ptr %b503, align 8
  call void @tl_add_parameter(ptr @key_str.261, ptr %b504)
  %l2505 = getelementptr inbounds %Block, ptr %sub_ptr472, i32 0, i32 2
  %sub_ptr506 = load ptr, ptr %l2505, align 8
  %w507 = getelementptr inbounds %LayerNorm, ptr %sub_ptr506, i32 0, i32 0
  %w508 = load ptr, ptr %w507, align 8
  call void @tl_add_parameter(ptr @key_str.262, ptr %w508)
  %b509 = getelementptr inbounds %LayerNorm, ptr %sub_ptr506, i32 0, i32 1
  %b510 = load ptr, ptr %b509, align 8
  call void @tl_add_parameter(ptr @key_str.263, ptr %b510)
  %m511 = getelementptr inbounds %Block, ptr %sub_ptr472, i32 0, i32 3
  %sub_ptr512 = load ptr, ptr %m511, align 8
  %f513 = getelementptr inbounds %MLP, ptr %sub_ptr512, i32 0, i32 0
  %sub_ptr514 = load ptr, ptr %f513, align 8
  %W515 = getelementptr inbounds %Linear, ptr %sub_ptr514, i32 0, i32 0
  %W516 = load ptr, ptr %W515, align 8
  call void @tl_add_parameter(ptr @key_str.264, ptr %W516)
  %b517 = getelementptr inbounds %Linear, ptr %sub_ptr514, i32 0, i32 1
  %b518 = load ptr, ptr %b517, align 8
  call void @tl_add_parameter(ptr @key_str.265, ptr %b518)
  %p519 = getelementptr inbounds %MLP, ptr %sub_ptr512, i32 0, i32 1
  %sub_ptr520 = load ptr, ptr %p519, align 8
  %W521 = getelementptr inbounds %Linear, ptr %sub_ptr520, i32 0, i32 0
  %W522 = load ptr, ptr %W521, align 8
  call void @tl_add_parameter(ptr @key_str.266, ptr %W522)
  %b523 = getelementptr inbounds %Linear, ptr %sub_ptr520, i32 0, i32 1
  %b524 = load ptr, ptr %b523, align 8
  call void @tl_add_parameter(ptr @key_str.267, ptr %b524)
  %l525 = getelementptr inbounds %GPT, ptr %model354, i32 0, i32 5
  %sub_ptr526 = load ptr, ptr %l525, align 8
  %w527 = getelementptr inbounds %LayerNorm, ptr %sub_ptr526, i32 0, i32 0
  %w528 = load ptr, ptr %w527, align 8
  call void @tl_add_parameter(ptr @key_str.268, ptr %w528)
  %b529 = getelementptr inbounds %LayerNorm, ptr %sub_ptr526, i32 0, i32 1
  %b530 = load ptr, ptr %b529, align 8
  call void @tl_add_parameter(ptr @key_str.269, ptr %b530)
  %h531 = getelementptr inbounds %GPT, ptr %model354, i32 0, i32 6
  %sub_ptr532 = load ptr, ptr %h531, align 8
  %W533 = getelementptr inbounds %Linear, ptr %sub_ptr532, i32 0, i32 0
  %W534 = load ptr, ptr %W533, align 8
  call void @tl_add_parameter(ptr @key_str.270, ptr %W534)
  %b535 = getelementptr inbounds %Linear, ptr %sub_ptr532, i32 0, i32 1
  %b536 = load ptr, ptr %b535, align 8
  call void @tl_add_parameter(ptr @key_str.271, ptr %b536)
  call void @tl_save_all_params(ptr @str_literal.272)
  %struct_to_free = load ptr, ptr %model, align 8
  %is_not_null = icmp ne ptr %struct_to_free, null
  br i1 %is_not_null, label %recursive_free_struct, label %continue_after_recursive_free

recursive_free_struct:                            ; preds = %for_end
  %free_field_0 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 0
  %field_val_to_free = load ptr, ptr %free_field_0, align 8
  %is_not_null537 = icmp ne ptr %field_val_to_free, null
  br i1 %is_not_null537, label %recursive_free_struct538, label %continue_after_recursive_free539

continue_after_recursive_free:                    ; preds = %continue_after_recursive_free817, %for_end
  call void @tl_mem_unregister(ptr %struct_to_free)
  call void @tl_mem_exit_scope()
  ret void

recursive_free_struct538:                         ; preds = %recursive_free_struct
  %free_field_0540 = getelementptr inbounds %Embedding, ptr %field_val_to_free, i32 0, i32 0
  %field_val_to_free541 = load ptr, ptr %free_field_0540, align 8
  call void @tl_tensor_free(ptr %field_val_to_free541)
  call void @free(ptr %field_val_to_free)
  br label %continue_after_recursive_free539

continue_after_recursive_free539:                 ; preds = %recursive_free_struct538, %recursive_free_struct
  %free_field_1 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 1
  %field_val_to_free542 = load ptr, ptr %free_field_1, align 8
  %is_not_null543 = icmp ne ptr %field_val_to_free542, null
  br i1 %is_not_null543, label %recursive_free_struct544, label %continue_after_recursive_free545

recursive_free_struct544:                         ; preds = %continue_after_recursive_free539
  %free_field_0546 = getelementptr inbounds %Embedding, ptr %field_val_to_free542, i32 0, i32 0
  %field_val_to_free547 = load ptr, ptr %free_field_0546, align 8
  call void @tl_tensor_free(ptr %field_val_to_free547)
  call void @free(ptr %field_val_to_free542)
  br label %continue_after_recursive_free545

continue_after_recursive_free545:                 ; preds = %recursive_free_struct544, %continue_after_recursive_free539
  %free_field_2 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 2
  %field_val_to_free548 = load ptr, ptr %free_field_2, align 8
  %is_not_null549 = icmp ne ptr %field_val_to_free548, null
  br i1 %is_not_null549, label %recursive_free_struct550, label %continue_after_recursive_free551

recursive_free_struct550:                         ; preds = %continue_after_recursive_free545
  %free_field_0552 = getelementptr inbounds %Block, ptr %field_val_to_free548, i32 0, i32 0
  %field_val_to_free553 = load ptr, ptr %free_field_0552, align 8
  %is_not_null554 = icmp ne ptr %field_val_to_free553, null
  br i1 %is_not_null554, label %recursive_free_struct555, label %continue_after_recursive_free556

continue_after_recursive_free551:                 ; preds = %continue_after_recursive_free614, %continue_after_recursive_free545
  %free_field_3633 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 3
  %field_val_to_free634 = load ptr, ptr %free_field_3633, align 8
  %is_not_null635 = icmp ne ptr %field_val_to_free634, null
  br i1 %is_not_null635, label %recursive_free_struct636, label %continue_after_recursive_free637

recursive_free_struct555:                         ; preds = %recursive_free_struct550
  %free_field_0557 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free553, i32 0, i32 0
  %field_val_to_free558 = load ptr, ptr %free_field_0557, align 8
  call void @tl_tensor_free(ptr %field_val_to_free558)
  %free_field_1559 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free553, i32 0, i32 1
  %field_val_to_free560 = load ptr, ptr %free_field_1559, align 8
  call void @tl_tensor_free(ptr %field_val_to_free560)
  call void @free(ptr %field_val_to_free553)
  br label %continue_after_recursive_free556

continue_after_recursive_free556:                 ; preds = %recursive_free_struct555, %recursive_free_struct550
  %free_field_1561 = getelementptr inbounds %Block, ptr %field_val_to_free548, i32 0, i32 1
  %field_val_to_free562 = load ptr, ptr %free_field_1561, align 8
  %is_not_null563 = icmp ne ptr %field_val_to_free562, null
  br i1 %is_not_null563, label %recursive_free_struct564, label %continue_after_recursive_free565

recursive_free_struct564:                         ; preds = %continue_after_recursive_free556
  %free_field_0566 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free562, i32 0, i32 0
  %field_val_to_free567 = load ptr, ptr %free_field_0566, align 8
  %is_not_null568 = icmp ne ptr %field_val_to_free567, null
  br i1 %is_not_null568, label %recursive_free_struct569, label %continue_after_recursive_free570

continue_after_recursive_free565:                 ; preds = %continue_after_recursive_free596, %continue_after_recursive_free556
  %free_field_2601 = getelementptr inbounds %Block, ptr %field_val_to_free548, i32 0, i32 2
  %field_val_to_free602 = load ptr, ptr %free_field_2601, align 8
  %is_not_null603 = icmp ne ptr %field_val_to_free602, null
  br i1 %is_not_null603, label %recursive_free_struct604, label %continue_after_recursive_free605

recursive_free_struct569:                         ; preds = %recursive_free_struct564
  %free_field_0571 = getelementptr inbounds %Linear, ptr %field_val_to_free567, i32 0, i32 0
  %field_val_to_free572 = load ptr, ptr %free_field_0571, align 8
  call void @tl_tensor_free(ptr %field_val_to_free572)
  %free_field_1573 = getelementptr inbounds %Linear, ptr %field_val_to_free567, i32 0, i32 1
  %field_val_to_free574 = load ptr, ptr %free_field_1573, align 8
  call void @tl_tensor_free(ptr %field_val_to_free574)
  call void @free(ptr %field_val_to_free567)
  br label %continue_after_recursive_free570

continue_after_recursive_free570:                 ; preds = %recursive_free_struct569, %recursive_free_struct564
  %free_field_1575 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free562, i32 0, i32 1
  %field_val_to_free576 = load ptr, ptr %free_field_1575, align 8
  %is_not_null577 = icmp ne ptr %field_val_to_free576, null
  br i1 %is_not_null577, label %recursive_free_struct578, label %continue_after_recursive_free579

recursive_free_struct578:                         ; preds = %continue_after_recursive_free570
  %free_field_0580 = getelementptr inbounds %Linear, ptr %field_val_to_free576, i32 0, i32 0
  %field_val_to_free581 = load ptr, ptr %free_field_0580, align 8
  call void @tl_tensor_free(ptr %field_val_to_free581)
  %free_field_1582 = getelementptr inbounds %Linear, ptr %field_val_to_free576, i32 0, i32 1
  %field_val_to_free583 = load ptr, ptr %free_field_1582, align 8
  call void @tl_tensor_free(ptr %field_val_to_free583)
  call void @free(ptr %field_val_to_free576)
  br label %continue_after_recursive_free579

continue_after_recursive_free579:                 ; preds = %recursive_free_struct578, %continue_after_recursive_free570
  %free_field_2584 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free562, i32 0, i32 2
  %field_val_to_free585 = load ptr, ptr %free_field_2584, align 8
  %is_not_null586 = icmp ne ptr %field_val_to_free585, null
  br i1 %is_not_null586, label %recursive_free_struct587, label %continue_after_recursive_free588

recursive_free_struct587:                         ; preds = %continue_after_recursive_free579
  %free_field_0589 = getelementptr inbounds %Linear, ptr %field_val_to_free585, i32 0, i32 0
  %field_val_to_free590 = load ptr, ptr %free_field_0589, align 8
  call void @tl_tensor_free(ptr %field_val_to_free590)
  %free_field_1591 = getelementptr inbounds %Linear, ptr %field_val_to_free585, i32 0, i32 1
  %field_val_to_free592 = load ptr, ptr %free_field_1591, align 8
  call void @tl_tensor_free(ptr %field_val_to_free592)
  call void @free(ptr %field_val_to_free585)
  br label %continue_after_recursive_free588

continue_after_recursive_free588:                 ; preds = %recursive_free_struct587, %continue_after_recursive_free579
  %free_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free562, i32 0, i32 3
  %field_val_to_free593 = load ptr, ptr %free_field_3, align 8
  %is_not_null594 = icmp ne ptr %field_val_to_free593, null
  br i1 %is_not_null594, label %recursive_free_struct595, label %continue_after_recursive_free596

recursive_free_struct595:                         ; preds = %continue_after_recursive_free588
  %free_field_0597 = getelementptr inbounds %Linear, ptr %field_val_to_free593, i32 0, i32 0
  %field_val_to_free598 = load ptr, ptr %free_field_0597, align 8
  call void @tl_tensor_free(ptr %field_val_to_free598)
  %free_field_1599 = getelementptr inbounds %Linear, ptr %field_val_to_free593, i32 0, i32 1
  %field_val_to_free600 = load ptr, ptr %free_field_1599, align 8
  call void @tl_tensor_free(ptr %field_val_to_free600)
  call void @free(ptr %field_val_to_free593)
  br label %continue_after_recursive_free596

continue_after_recursive_free596:                 ; preds = %recursive_free_struct595, %continue_after_recursive_free588
  call void @free(ptr %field_val_to_free562)
  br label %continue_after_recursive_free565

recursive_free_struct604:                         ; preds = %continue_after_recursive_free565
  %free_field_0606 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free602, i32 0, i32 0
  %field_val_to_free607 = load ptr, ptr %free_field_0606, align 8
  call void @tl_tensor_free(ptr %field_val_to_free607)
  %free_field_1608 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free602, i32 0, i32 1
  %field_val_to_free609 = load ptr, ptr %free_field_1608, align 8
  call void @tl_tensor_free(ptr %field_val_to_free609)
  call void @free(ptr %field_val_to_free602)
  br label %continue_after_recursive_free605

continue_after_recursive_free605:                 ; preds = %recursive_free_struct604, %continue_after_recursive_free565
  %free_field_3610 = getelementptr inbounds %Block, ptr %field_val_to_free548, i32 0, i32 3
  %field_val_to_free611 = load ptr, ptr %free_field_3610, align 8
  %is_not_null612 = icmp ne ptr %field_val_to_free611, null
  br i1 %is_not_null612, label %recursive_free_struct613, label %continue_after_recursive_free614

recursive_free_struct613:                         ; preds = %continue_after_recursive_free605
  %free_field_0615 = getelementptr inbounds %MLP, ptr %field_val_to_free611, i32 0, i32 0
  %field_val_to_free616 = load ptr, ptr %free_field_0615, align 8
  %is_not_null617 = icmp ne ptr %field_val_to_free616, null
  br i1 %is_not_null617, label %recursive_free_struct618, label %continue_after_recursive_free619

continue_after_recursive_free614:                 ; preds = %continue_after_recursive_free628, %continue_after_recursive_free605
  call void @free(ptr %field_val_to_free548)
  br label %continue_after_recursive_free551

recursive_free_struct618:                         ; preds = %recursive_free_struct613
  %free_field_0620 = getelementptr inbounds %Linear, ptr %field_val_to_free616, i32 0, i32 0
  %field_val_to_free621 = load ptr, ptr %free_field_0620, align 8
  call void @tl_tensor_free(ptr %field_val_to_free621)
  %free_field_1622 = getelementptr inbounds %Linear, ptr %field_val_to_free616, i32 0, i32 1
  %field_val_to_free623 = load ptr, ptr %free_field_1622, align 8
  call void @tl_tensor_free(ptr %field_val_to_free623)
  call void @free(ptr %field_val_to_free616)
  br label %continue_after_recursive_free619

continue_after_recursive_free619:                 ; preds = %recursive_free_struct618, %recursive_free_struct613
  %free_field_1624 = getelementptr inbounds %MLP, ptr %field_val_to_free611, i32 0, i32 1
  %field_val_to_free625 = load ptr, ptr %free_field_1624, align 8
  %is_not_null626 = icmp ne ptr %field_val_to_free625, null
  br i1 %is_not_null626, label %recursive_free_struct627, label %continue_after_recursive_free628

recursive_free_struct627:                         ; preds = %continue_after_recursive_free619
  %free_field_0629 = getelementptr inbounds %Linear, ptr %field_val_to_free625, i32 0, i32 0
  %field_val_to_free630 = load ptr, ptr %free_field_0629, align 8
  call void @tl_tensor_free(ptr %field_val_to_free630)
  %free_field_1631 = getelementptr inbounds %Linear, ptr %field_val_to_free625, i32 0, i32 1
  %field_val_to_free632 = load ptr, ptr %free_field_1631, align 8
  call void @tl_tensor_free(ptr %field_val_to_free632)
  call void @free(ptr %field_val_to_free625)
  br label %continue_after_recursive_free628

continue_after_recursive_free628:                 ; preds = %recursive_free_struct627, %continue_after_recursive_free619
  call void @free(ptr %field_val_to_free611)
  br label %continue_after_recursive_free614

recursive_free_struct636:                         ; preds = %continue_after_recursive_free551
  %free_field_0638 = getelementptr inbounds %Block, ptr %field_val_to_free634, i32 0, i32 0
  %field_val_to_free639 = load ptr, ptr %free_field_0638, align 8
  %is_not_null640 = icmp ne ptr %field_val_to_free639, null
  br i1 %is_not_null640, label %recursive_free_struct641, label %continue_after_recursive_free642

continue_after_recursive_free637:                 ; preds = %continue_after_recursive_free701, %continue_after_recursive_free551
  %free_field_4 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 4
  %field_val_to_free720 = load ptr, ptr %free_field_4, align 8
  %is_not_null721 = icmp ne ptr %field_val_to_free720, null
  br i1 %is_not_null721, label %recursive_free_struct722, label %continue_after_recursive_free723

recursive_free_struct641:                         ; preds = %recursive_free_struct636
  %free_field_0643 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free639, i32 0, i32 0
  %field_val_to_free644 = load ptr, ptr %free_field_0643, align 8
  call void @tl_tensor_free(ptr %field_val_to_free644)
  %free_field_1645 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free639, i32 0, i32 1
  %field_val_to_free646 = load ptr, ptr %free_field_1645, align 8
  call void @tl_tensor_free(ptr %field_val_to_free646)
  call void @free(ptr %field_val_to_free639)
  br label %continue_after_recursive_free642

continue_after_recursive_free642:                 ; preds = %recursive_free_struct641, %recursive_free_struct636
  %free_field_1647 = getelementptr inbounds %Block, ptr %field_val_to_free634, i32 0, i32 1
  %field_val_to_free648 = load ptr, ptr %free_field_1647, align 8
  %is_not_null649 = icmp ne ptr %field_val_to_free648, null
  br i1 %is_not_null649, label %recursive_free_struct650, label %continue_after_recursive_free651

recursive_free_struct650:                         ; preds = %continue_after_recursive_free642
  %free_field_0652 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free648, i32 0, i32 0
  %field_val_to_free653 = load ptr, ptr %free_field_0652, align 8
  %is_not_null654 = icmp ne ptr %field_val_to_free653, null
  br i1 %is_not_null654, label %recursive_free_struct655, label %continue_after_recursive_free656

continue_after_recursive_free651:                 ; preds = %continue_after_recursive_free683, %continue_after_recursive_free642
  %free_field_2688 = getelementptr inbounds %Block, ptr %field_val_to_free634, i32 0, i32 2
  %field_val_to_free689 = load ptr, ptr %free_field_2688, align 8
  %is_not_null690 = icmp ne ptr %field_val_to_free689, null
  br i1 %is_not_null690, label %recursive_free_struct691, label %continue_after_recursive_free692

recursive_free_struct655:                         ; preds = %recursive_free_struct650
  %free_field_0657 = getelementptr inbounds %Linear, ptr %field_val_to_free653, i32 0, i32 0
  %field_val_to_free658 = load ptr, ptr %free_field_0657, align 8
  call void @tl_tensor_free(ptr %field_val_to_free658)
  %free_field_1659 = getelementptr inbounds %Linear, ptr %field_val_to_free653, i32 0, i32 1
  %field_val_to_free660 = load ptr, ptr %free_field_1659, align 8
  call void @tl_tensor_free(ptr %field_val_to_free660)
  call void @free(ptr %field_val_to_free653)
  br label %continue_after_recursive_free656

continue_after_recursive_free656:                 ; preds = %recursive_free_struct655, %recursive_free_struct650
  %free_field_1661 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free648, i32 0, i32 1
  %field_val_to_free662 = load ptr, ptr %free_field_1661, align 8
  %is_not_null663 = icmp ne ptr %field_val_to_free662, null
  br i1 %is_not_null663, label %recursive_free_struct664, label %continue_after_recursive_free665

recursive_free_struct664:                         ; preds = %continue_after_recursive_free656
  %free_field_0666 = getelementptr inbounds %Linear, ptr %field_val_to_free662, i32 0, i32 0
  %field_val_to_free667 = load ptr, ptr %free_field_0666, align 8
  call void @tl_tensor_free(ptr %field_val_to_free667)
  %free_field_1668 = getelementptr inbounds %Linear, ptr %field_val_to_free662, i32 0, i32 1
  %field_val_to_free669 = load ptr, ptr %free_field_1668, align 8
  call void @tl_tensor_free(ptr %field_val_to_free669)
  call void @free(ptr %field_val_to_free662)
  br label %continue_after_recursive_free665

continue_after_recursive_free665:                 ; preds = %recursive_free_struct664, %continue_after_recursive_free656
  %free_field_2670 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free648, i32 0, i32 2
  %field_val_to_free671 = load ptr, ptr %free_field_2670, align 8
  %is_not_null672 = icmp ne ptr %field_val_to_free671, null
  br i1 %is_not_null672, label %recursive_free_struct673, label %continue_after_recursive_free674

recursive_free_struct673:                         ; preds = %continue_after_recursive_free665
  %free_field_0675 = getelementptr inbounds %Linear, ptr %field_val_to_free671, i32 0, i32 0
  %field_val_to_free676 = load ptr, ptr %free_field_0675, align 8
  call void @tl_tensor_free(ptr %field_val_to_free676)
  %free_field_1677 = getelementptr inbounds %Linear, ptr %field_val_to_free671, i32 0, i32 1
  %field_val_to_free678 = load ptr, ptr %free_field_1677, align 8
  call void @tl_tensor_free(ptr %field_val_to_free678)
  call void @free(ptr %field_val_to_free671)
  br label %continue_after_recursive_free674

continue_after_recursive_free674:                 ; preds = %recursive_free_struct673, %continue_after_recursive_free665
  %free_field_3679 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free648, i32 0, i32 3
  %field_val_to_free680 = load ptr, ptr %free_field_3679, align 8
  %is_not_null681 = icmp ne ptr %field_val_to_free680, null
  br i1 %is_not_null681, label %recursive_free_struct682, label %continue_after_recursive_free683

recursive_free_struct682:                         ; preds = %continue_after_recursive_free674
  %free_field_0684 = getelementptr inbounds %Linear, ptr %field_val_to_free680, i32 0, i32 0
  %field_val_to_free685 = load ptr, ptr %free_field_0684, align 8
  call void @tl_tensor_free(ptr %field_val_to_free685)
  %free_field_1686 = getelementptr inbounds %Linear, ptr %field_val_to_free680, i32 0, i32 1
  %field_val_to_free687 = load ptr, ptr %free_field_1686, align 8
  call void @tl_tensor_free(ptr %field_val_to_free687)
  call void @free(ptr %field_val_to_free680)
  br label %continue_after_recursive_free683

continue_after_recursive_free683:                 ; preds = %recursive_free_struct682, %continue_after_recursive_free674
  call void @free(ptr %field_val_to_free648)
  br label %continue_after_recursive_free651

recursive_free_struct691:                         ; preds = %continue_after_recursive_free651
  %free_field_0693 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free689, i32 0, i32 0
  %field_val_to_free694 = load ptr, ptr %free_field_0693, align 8
  call void @tl_tensor_free(ptr %field_val_to_free694)
  %free_field_1695 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free689, i32 0, i32 1
  %field_val_to_free696 = load ptr, ptr %free_field_1695, align 8
  call void @tl_tensor_free(ptr %field_val_to_free696)
  call void @free(ptr %field_val_to_free689)
  br label %continue_after_recursive_free692

continue_after_recursive_free692:                 ; preds = %recursive_free_struct691, %continue_after_recursive_free651
  %free_field_3697 = getelementptr inbounds %Block, ptr %field_val_to_free634, i32 0, i32 3
  %field_val_to_free698 = load ptr, ptr %free_field_3697, align 8
  %is_not_null699 = icmp ne ptr %field_val_to_free698, null
  br i1 %is_not_null699, label %recursive_free_struct700, label %continue_after_recursive_free701

recursive_free_struct700:                         ; preds = %continue_after_recursive_free692
  %free_field_0702 = getelementptr inbounds %MLP, ptr %field_val_to_free698, i32 0, i32 0
  %field_val_to_free703 = load ptr, ptr %free_field_0702, align 8
  %is_not_null704 = icmp ne ptr %field_val_to_free703, null
  br i1 %is_not_null704, label %recursive_free_struct705, label %continue_after_recursive_free706

continue_after_recursive_free701:                 ; preds = %continue_after_recursive_free715, %continue_after_recursive_free692
  call void @free(ptr %field_val_to_free634)
  br label %continue_after_recursive_free637

recursive_free_struct705:                         ; preds = %recursive_free_struct700
  %free_field_0707 = getelementptr inbounds %Linear, ptr %field_val_to_free703, i32 0, i32 0
  %field_val_to_free708 = load ptr, ptr %free_field_0707, align 8
  call void @tl_tensor_free(ptr %field_val_to_free708)
  %free_field_1709 = getelementptr inbounds %Linear, ptr %field_val_to_free703, i32 0, i32 1
  %field_val_to_free710 = load ptr, ptr %free_field_1709, align 8
  call void @tl_tensor_free(ptr %field_val_to_free710)
  call void @free(ptr %field_val_to_free703)
  br label %continue_after_recursive_free706

continue_after_recursive_free706:                 ; preds = %recursive_free_struct705, %recursive_free_struct700
  %free_field_1711 = getelementptr inbounds %MLP, ptr %field_val_to_free698, i32 0, i32 1
  %field_val_to_free712 = load ptr, ptr %free_field_1711, align 8
  %is_not_null713 = icmp ne ptr %field_val_to_free712, null
  br i1 %is_not_null713, label %recursive_free_struct714, label %continue_after_recursive_free715

recursive_free_struct714:                         ; preds = %continue_after_recursive_free706
  %free_field_0716 = getelementptr inbounds %Linear, ptr %field_val_to_free712, i32 0, i32 0
  %field_val_to_free717 = load ptr, ptr %free_field_0716, align 8
  call void @tl_tensor_free(ptr %field_val_to_free717)
  %free_field_1718 = getelementptr inbounds %Linear, ptr %field_val_to_free712, i32 0, i32 1
  %field_val_to_free719 = load ptr, ptr %free_field_1718, align 8
  call void @tl_tensor_free(ptr %field_val_to_free719)
  call void @free(ptr %field_val_to_free712)
  br label %continue_after_recursive_free715

continue_after_recursive_free715:                 ; preds = %recursive_free_struct714, %continue_after_recursive_free706
  call void @free(ptr %field_val_to_free698)
  br label %continue_after_recursive_free701

recursive_free_struct722:                         ; preds = %continue_after_recursive_free637
  %free_field_0724 = getelementptr inbounds %Block, ptr %field_val_to_free720, i32 0, i32 0
  %field_val_to_free725 = load ptr, ptr %free_field_0724, align 8
  %is_not_null726 = icmp ne ptr %field_val_to_free725, null
  br i1 %is_not_null726, label %recursive_free_struct727, label %continue_after_recursive_free728

continue_after_recursive_free723:                 ; preds = %continue_after_recursive_free787, %continue_after_recursive_free637
  %free_field_5 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 5
  %field_val_to_free806 = load ptr, ptr %free_field_5, align 8
  %is_not_null807 = icmp ne ptr %field_val_to_free806, null
  br i1 %is_not_null807, label %recursive_free_struct808, label %continue_after_recursive_free809

recursive_free_struct727:                         ; preds = %recursive_free_struct722
  %free_field_0729 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free725, i32 0, i32 0
  %field_val_to_free730 = load ptr, ptr %free_field_0729, align 8
  call void @tl_tensor_free(ptr %field_val_to_free730)
  %free_field_1731 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free725, i32 0, i32 1
  %field_val_to_free732 = load ptr, ptr %free_field_1731, align 8
  call void @tl_tensor_free(ptr %field_val_to_free732)
  call void @free(ptr %field_val_to_free725)
  br label %continue_after_recursive_free728

continue_after_recursive_free728:                 ; preds = %recursive_free_struct727, %recursive_free_struct722
  %free_field_1733 = getelementptr inbounds %Block, ptr %field_val_to_free720, i32 0, i32 1
  %field_val_to_free734 = load ptr, ptr %free_field_1733, align 8
  %is_not_null735 = icmp ne ptr %field_val_to_free734, null
  br i1 %is_not_null735, label %recursive_free_struct736, label %continue_after_recursive_free737

recursive_free_struct736:                         ; preds = %continue_after_recursive_free728
  %free_field_0738 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free734, i32 0, i32 0
  %field_val_to_free739 = load ptr, ptr %free_field_0738, align 8
  %is_not_null740 = icmp ne ptr %field_val_to_free739, null
  br i1 %is_not_null740, label %recursive_free_struct741, label %continue_after_recursive_free742

continue_after_recursive_free737:                 ; preds = %continue_after_recursive_free769, %continue_after_recursive_free728
  %free_field_2774 = getelementptr inbounds %Block, ptr %field_val_to_free720, i32 0, i32 2
  %field_val_to_free775 = load ptr, ptr %free_field_2774, align 8
  %is_not_null776 = icmp ne ptr %field_val_to_free775, null
  br i1 %is_not_null776, label %recursive_free_struct777, label %continue_after_recursive_free778

recursive_free_struct741:                         ; preds = %recursive_free_struct736
  %free_field_0743 = getelementptr inbounds %Linear, ptr %field_val_to_free739, i32 0, i32 0
  %field_val_to_free744 = load ptr, ptr %free_field_0743, align 8
  call void @tl_tensor_free(ptr %field_val_to_free744)
  %free_field_1745 = getelementptr inbounds %Linear, ptr %field_val_to_free739, i32 0, i32 1
  %field_val_to_free746 = load ptr, ptr %free_field_1745, align 8
  call void @tl_tensor_free(ptr %field_val_to_free746)
  call void @free(ptr %field_val_to_free739)
  br label %continue_after_recursive_free742

continue_after_recursive_free742:                 ; preds = %recursive_free_struct741, %recursive_free_struct736
  %free_field_1747 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free734, i32 0, i32 1
  %field_val_to_free748 = load ptr, ptr %free_field_1747, align 8
  %is_not_null749 = icmp ne ptr %field_val_to_free748, null
  br i1 %is_not_null749, label %recursive_free_struct750, label %continue_after_recursive_free751

recursive_free_struct750:                         ; preds = %continue_after_recursive_free742
  %free_field_0752 = getelementptr inbounds %Linear, ptr %field_val_to_free748, i32 0, i32 0
  %field_val_to_free753 = load ptr, ptr %free_field_0752, align 8
  call void @tl_tensor_free(ptr %field_val_to_free753)
  %free_field_1754 = getelementptr inbounds %Linear, ptr %field_val_to_free748, i32 0, i32 1
  %field_val_to_free755 = load ptr, ptr %free_field_1754, align 8
  call void @tl_tensor_free(ptr %field_val_to_free755)
  call void @free(ptr %field_val_to_free748)
  br label %continue_after_recursive_free751

continue_after_recursive_free751:                 ; preds = %recursive_free_struct750, %continue_after_recursive_free742
  %free_field_2756 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free734, i32 0, i32 2
  %field_val_to_free757 = load ptr, ptr %free_field_2756, align 8
  %is_not_null758 = icmp ne ptr %field_val_to_free757, null
  br i1 %is_not_null758, label %recursive_free_struct759, label %continue_after_recursive_free760

recursive_free_struct759:                         ; preds = %continue_after_recursive_free751
  %free_field_0761 = getelementptr inbounds %Linear, ptr %field_val_to_free757, i32 0, i32 0
  %field_val_to_free762 = load ptr, ptr %free_field_0761, align 8
  call void @tl_tensor_free(ptr %field_val_to_free762)
  %free_field_1763 = getelementptr inbounds %Linear, ptr %field_val_to_free757, i32 0, i32 1
  %field_val_to_free764 = load ptr, ptr %free_field_1763, align 8
  call void @tl_tensor_free(ptr %field_val_to_free764)
  call void @free(ptr %field_val_to_free757)
  br label %continue_after_recursive_free760

continue_after_recursive_free760:                 ; preds = %recursive_free_struct759, %continue_after_recursive_free751
  %free_field_3765 = getelementptr inbounds %CausalSelfAttention, ptr %field_val_to_free734, i32 0, i32 3
  %field_val_to_free766 = load ptr, ptr %free_field_3765, align 8
  %is_not_null767 = icmp ne ptr %field_val_to_free766, null
  br i1 %is_not_null767, label %recursive_free_struct768, label %continue_after_recursive_free769

recursive_free_struct768:                         ; preds = %continue_after_recursive_free760
  %free_field_0770 = getelementptr inbounds %Linear, ptr %field_val_to_free766, i32 0, i32 0
  %field_val_to_free771 = load ptr, ptr %free_field_0770, align 8
  call void @tl_tensor_free(ptr %field_val_to_free771)
  %free_field_1772 = getelementptr inbounds %Linear, ptr %field_val_to_free766, i32 0, i32 1
  %field_val_to_free773 = load ptr, ptr %free_field_1772, align 8
  call void @tl_tensor_free(ptr %field_val_to_free773)
  call void @free(ptr %field_val_to_free766)
  br label %continue_after_recursive_free769

continue_after_recursive_free769:                 ; preds = %recursive_free_struct768, %continue_after_recursive_free760
  call void @free(ptr %field_val_to_free734)
  br label %continue_after_recursive_free737

recursive_free_struct777:                         ; preds = %continue_after_recursive_free737
  %free_field_0779 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free775, i32 0, i32 0
  %field_val_to_free780 = load ptr, ptr %free_field_0779, align 8
  call void @tl_tensor_free(ptr %field_val_to_free780)
  %free_field_1781 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free775, i32 0, i32 1
  %field_val_to_free782 = load ptr, ptr %free_field_1781, align 8
  call void @tl_tensor_free(ptr %field_val_to_free782)
  call void @free(ptr %field_val_to_free775)
  br label %continue_after_recursive_free778

continue_after_recursive_free778:                 ; preds = %recursive_free_struct777, %continue_after_recursive_free737
  %free_field_3783 = getelementptr inbounds %Block, ptr %field_val_to_free720, i32 0, i32 3
  %field_val_to_free784 = load ptr, ptr %free_field_3783, align 8
  %is_not_null785 = icmp ne ptr %field_val_to_free784, null
  br i1 %is_not_null785, label %recursive_free_struct786, label %continue_after_recursive_free787

recursive_free_struct786:                         ; preds = %continue_after_recursive_free778
  %free_field_0788 = getelementptr inbounds %MLP, ptr %field_val_to_free784, i32 0, i32 0
  %field_val_to_free789 = load ptr, ptr %free_field_0788, align 8
  %is_not_null790 = icmp ne ptr %field_val_to_free789, null
  br i1 %is_not_null790, label %recursive_free_struct791, label %continue_after_recursive_free792

continue_after_recursive_free787:                 ; preds = %continue_after_recursive_free801, %continue_after_recursive_free778
  call void @free(ptr %field_val_to_free720)
  br label %continue_after_recursive_free723

recursive_free_struct791:                         ; preds = %recursive_free_struct786
  %free_field_0793 = getelementptr inbounds %Linear, ptr %field_val_to_free789, i32 0, i32 0
  %field_val_to_free794 = load ptr, ptr %free_field_0793, align 8
  call void @tl_tensor_free(ptr %field_val_to_free794)
  %free_field_1795 = getelementptr inbounds %Linear, ptr %field_val_to_free789, i32 0, i32 1
  %field_val_to_free796 = load ptr, ptr %free_field_1795, align 8
  call void @tl_tensor_free(ptr %field_val_to_free796)
  call void @free(ptr %field_val_to_free789)
  br label %continue_after_recursive_free792

continue_after_recursive_free792:                 ; preds = %recursive_free_struct791, %recursive_free_struct786
  %free_field_1797 = getelementptr inbounds %MLP, ptr %field_val_to_free784, i32 0, i32 1
  %field_val_to_free798 = load ptr, ptr %free_field_1797, align 8
  %is_not_null799 = icmp ne ptr %field_val_to_free798, null
  br i1 %is_not_null799, label %recursive_free_struct800, label %continue_after_recursive_free801

recursive_free_struct800:                         ; preds = %continue_after_recursive_free792
  %free_field_0802 = getelementptr inbounds %Linear, ptr %field_val_to_free798, i32 0, i32 0
  %field_val_to_free803 = load ptr, ptr %free_field_0802, align 8
  call void @tl_tensor_free(ptr %field_val_to_free803)
  %free_field_1804 = getelementptr inbounds %Linear, ptr %field_val_to_free798, i32 0, i32 1
  %field_val_to_free805 = load ptr, ptr %free_field_1804, align 8
  call void @tl_tensor_free(ptr %field_val_to_free805)
  call void @free(ptr %field_val_to_free798)
  br label %continue_after_recursive_free801

continue_after_recursive_free801:                 ; preds = %recursive_free_struct800, %continue_after_recursive_free792
  call void @free(ptr %field_val_to_free784)
  br label %continue_after_recursive_free787

recursive_free_struct808:                         ; preds = %continue_after_recursive_free723
  %free_field_0810 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free806, i32 0, i32 0
  %field_val_to_free811 = load ptr, ptr %free_field_0810, align 8
  call void @tl_tensor_free(ptr %field_val_to_free811)
  %free_field_1812 = getelementptr inbounds %LayerNorm, ptr %field_val_to_free806, i32 0, i32 1
  %field_val_to_free813 = load ptr, ptr %free_field_1812, align 8
  call void @tl_tensor_free(ptr %field_val_to_free813)
  call void @free(ptr %field_val_to_free806)
  br label %continue_after_recursive_free809

continue_after_recursive_free809:                 ; preds = %recursive_free_struct808, %continue_after_recursive_free723
  %free_field_6 = getelementptr inbounds %GPT, ptr %struct_to_free, i32 0, i32 6
  %field_val_to_free814 = load ptr, ptr %free_field_6, align 8
  %is_not_null815 = icmp ne ptr %field_val_to_free814, null
  br i1 %is_not_null815, label %recursive_free_struct816, label %continue_after_recursive_free817

recursive_free_struct816:                         ; preds = %continue_after_recursive_free809
  %free_field_0818 = getelementptr inbounds %Linear, ptr %field_val_to_free814, i32 0, i32 0
  %field_val_to_free819 = load ptr, ptr %free_field_0818, align 8
  call void @tl_tensor_free(ptr %field_val_to_free819)
  %free_field_1820 = getelementptr inbounds %Linear, ptr %field_val_to_free814, i32 0, i32 1
  %field_val_to_free821 = load ptr, ptr %free_field_1820, align 8
  call void @tl_tensor_free(ptr %field_val_to_free821)
  call void @free(ptr %field_val_to_free814)
  br label %continue_after_recursive_free817

continue_after_recursive_free817:                 ; preds = %recursive_free_struct816, %continue_after_recursive_free809
  call void @free(ptr %struct_to_free)
  br label %continue_after_recursive_free
}
