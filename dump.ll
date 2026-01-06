; ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

%Linear = type { ptr, ptr }
%Embedding = type { ptr }
%LayerNorm = type { ptr, ptr }
%CausalSelfAttention = type { ptr, ptr, ptr, ptr }
%MLP = type { ptr, ptr }
%Block = type { ptr, ptr, ptr, ptr }
%GPTHeavy = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }

@str_literal = private unnamed_addr constant [7 x i8] c"Epoch:\00", align 1
@key_str = private unnamed_addr constant [4 x i8] c"w.w\00", align 1
@key_str.104 = private unnamed_addr constant [5 x i8] c"wp.w\00", align 1
@key_str.105 = private unnamed_addr constant [8 x i8] c"b1.l1.w\00", align 1
@key_str.106 = private unnamed_addr constant [8 x i8] c"b1.l1.b\00", align 1
@key_str.107 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.W\00", align 1
@key_str.108 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.b\00", align 1
@key_str.109 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.W\00", align 1
@key_str.110 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.b\00", align 1
@key_str.111 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.W\00", align 1
@key_str.112 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.b\00", align 1
@key_str.113 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.W\00", align 1
@key_str.114 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.b\00", align 1
@key_str.115 = private unnamed_addr constant [8 x i8] c"b1.l2.w\00", align 1
@key_str.116 = private unnamed_addr constant [8 x i8] c"b1.l2.b\00", align 1
@key_str.117 = private unnamed_addr constant [9 x i8] c"b1.m.f.W\00", align 1
@key_str.118 = private unnamed_addr constant [9 x i8] c"b1.m.f.b\00", align 1
@key_str.119 = private unnamed_addr constant [9 x i8] c"b1.m.p.W\00", align 1
@key_str.120 = private unnamed_addr constant [9 x i8] c"b1.m.p.b\00", align 1
@key_str.121 = private unnamed_addr constant [8 x i8] c"b2.l1.w\00", align 1
@key_str.122 = private unnamed_addr constant [8 x i8] c"b2.l1.b\00", align 1
@key_str.123 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.W\00", align 1
@key_str.124 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.b\00", align 1
@key_str.125 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.W\00", align 1
@key_str.126 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.b\00", align 1
@key_str.127 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.W\00", align 1
@key_str.128 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.b\00", align 1
@key_str.129 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.W\00", align 1
@key_str.130 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.b\00", align 1
@key_str.131 = private unnamed_addr constant [8 x i8] c"b2.l2.w\00", align 1
@key_str.132 = private unnamed_addr constant [8 x i8] c"b2.l2.b\00", align 1
@key_str.133 = private unnamed_addr constant [9 x i8] c"b2.m.f.W\00", align 1
@key_str.134 = private unnamed_addr constant [9 x i8] c"b2.m.f.b\00", align 1
@key_str.135 = private unnamed_addr constant [9 x i8] c"b2.m.p.W\00", align 1
@key_str.136 = private unnamed_addr constant [9 x i8] c"b2.m.p.b\00", align 1
@key_str.137 = private unnamed_addr constant [8 x i8] c"b3.l1.w\00", align 1
@key_str.138 = private unnamed_addr constant [8 x i8] c"b3.l1.b\00", align 1
@key_str.139 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.W\00", align 1
@key_str.140 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.b\00", align 1
@key_str.141 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.W\00", align 1
@key_str.142 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.b\00", align 1
@key_str.143 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.W\00", align 1
@key_str.144 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.b\00", align 1
@key_str.145 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.W\00", align 1
@key_str.146 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.b\00", align 1
@key_str.147 = private unnamed_addr constant [8 x i8] c"b3.l2.w\00", align 1
@key_str.148 = private unnamed_addr constant [8 x i8] c"b3.l2.b\00", align 1
@key_str.149 = private unnamed_addr constant [9 x i8] c"b3.m.f.W\00", align 1
@key_str.150 = private unnamed_addr constant [9 x i8] c"b3.m.f.b\00", align 1
@key_str.151 = private unnamed_addr constant [9 x i8] c"b3.m.p.W\00", align 1
@key_str.152 = private unnamed_addr constant [9 x i8] c"b3.m.p.b\00", align 1
@key_str.153 = private unnamed_addr constant [8 x i8] c"b4.l1.w\00", align 1
@key_str.154 = private unnamed_addr constant [8 x i8] c"b4.l1.b\00", align 1
@key_str.155 = private unnamed_addr constant [14 x i8] c"b4.a.q_proj.W\00", align 1
@key_str.156 = private unnamed_addr constant [14 x i8] c"b4.a.q_proj.b\00", align 1
@key_str.157 = private unnamed_addr constant [14 x i8] c"b4.a.k_proj.W\00", align 1
@key_str.158 = private unnamed_addr constant [14 x i8] c"b4.a.k_proj.b\00", align 1
@key_str.159 = private unnamed_addr constant [14 x i8] c"b4.a.v_proj.W\00", align 1
@key_str.160 = private unnamed_addr constant [14 x i8] c"b4.a.v_proj.b\00", align 1
@key_str.161 = private unnamed_addr constant [14 x i8] c"b4.a.p_proj.W\00", align 1
@key_str.162 = private unnamed_addr constant [14 x i8] c"b4.a.p_proj.b\00", align 1
@key_str.163 = private unnamed_addr constant [8 x i8] c"b4.l2.w\00", align 1
@key_str.164 = private unnamed_addr constant [8 x i8] c"b4.l2.b\00", align 1
@key_str.165 = private unnamed_addr constant [9 x i8] c"b4.m.f.W\00", align 1
@key_str.166 = private unnamed_addr constant [9 x i8] c"b4.m.f.b\00", align 1
@key_str.167 = private unnamed_addr constant [9 x i8] c"b4.m.p.W\00", align 1
@key_str.168 = private unnamed_addr constant [9 x i8] c"b4.m.p.b\00", align 1
@key_str.169 = private unnamed_addr constant [4 x i8] c"l.w\00", align 1
@key_str.170 = private unnamed_addr constant [4 x i8] c"l.b\00", align 1
@key_str.171 = private unnamed_addr constant [4 x i8] c"h.W\00", align 1
@key_str.172 = private unnamed_addr constant [4 x i8] c"h.b\00", align 1
@str_literal.173 = private unnamed_addr constant [24 x i8] c"model_heavy.safetensors\00", align 1
@str_literal.174 = private unnamed_addr constant [70 x i8] c"Training Heavy Model: 4-layer GPT, d_model=384, lr=0.0005, 500 epochs\00", align 1
@key_str.175 = private unnamed_addr constant [4 x i8] c"w.w\00", align 1
@key_str.176 = private unnamed_addr constant [5 x i8] c"wp.w\00", align 1
@key_str.177 = private unnamed_addr constant [8 x i8] c"b1.l1.w\00", align 1
@key_str.178 = private unnamed_addr constant [8 x i8] c"b1.l1.b\00", align 1
@key_str.179 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.W\00", align 1
@key_str.180 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.b\00", align 1
@key_str.181 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.W\00", align 1
@key_str.182 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.b\00", align 1
@key_str.183 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.W\00", align 1
@key_str.184 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.b\00", align 1
@key_str.185 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.W\00", align 1
@key_str.186 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.b\00", align 1
@key_str.187 = private unnamed_addr constant [8 x i8] c"b1.l2.w\00", align 1
@key_str.188 = private unnamed_addr constant [8 x i8] c"b1.l2.b\00", align 1
@key_str.189 = private unnamed_addr constant [9 x i8] c"b1.m.f.W\00", align 1
@key_str.190 = private unnamed_addr constant [9 x i8] c"b1.m.f.b\00", align 1
@key_str.191 = private unnamed_addr constant [9 x i8] c"b1.m.p.W\00", align 1
@key_str.192 = private unnamed_addr constant [9 x i8] c"b1.m.p.b\00", align 1
@key_str.193 = private unnamed_addr constant [8 x i8] c"b2.l1.w\00", align 1
@key_str.194 = private unnamed_addr constant [8 x i8] c"b2.l1.b\00", align 1
@key_str.195 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.W\00", align 1
@key_str.196 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.b\00", align 1
@key_str.197 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.W\00", align 1
@key_str.198 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.b\00", align 1
@key_str.199 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.W\00", align 1
@key_str.200 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.b\00", align 1
@key_str.201 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.W\00", align 1
@key_str.202 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.b\00", align 1
@key_str.203 = private unnamed_addr constant [8 x i8] c"b2.l2.w\00", align 1
@key_str.204 = private unnamed_addr constant [8 x i8] c"b2.l2.b\00", align 1
@key_str.205 = private unnamed_addr constant [9 x i8] c"b2.m.f.W\00", align 1
@key_str.206 = private unnamed_addr constant [9 x i8] c"b2.m.f.b\00", align 1
@key_str.207 = private unnamed_addr constant [9 x i8] c"b2.m.p.W\00", align 1
@key_str.208 = private unnamed_addr constant [9 x i8] c"b2.m.p.b\00", align 1
@key_str.209 = private unnamed_addr constant [8 x i8] c"b3.l1.w\00", align 1
@key_str.210 = private unnamed_addr constant [8 x i8] c"b3.l1.b\00", align 1
@key_str.211 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.W\00", align 1
@key_str.212 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.b\00", align 1
@key_str.213 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.W\00", align 1
@key_str.214 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.b\00", align 1
@key_str.215 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.W\00", align 1
@key_str.216 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.b\00", align 1
@key_str.217 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.W\00", align 1
@key_str.218 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.b\00", align 1
@key_str.219 = private unnamed_addr constant [8 x i8] c"b3.l2.w\00", align 1
@key_str.220 = private unnamed_addr constant [8 x i8] c"b3.l2.b\00", align 1
@key_str.221 = private unnamed_addr constant [9 x i8] c"b3.m.f.W\00", align 1
@key_str.222 = private unnamed_addr constant [9 x i8] c"b3.m.f.b\00", align 1
@key_str.223 = private unnamed_addr constant [9 x i8] c"b3.m.p.W\00", align 1
@key_str.224 = private unnamed_addr constant [9 x i8] c"b3.m.p.b\00", align 1
@key_str.225 = private unnamed_addr constant [8 x i8] c"b4.l1.w\00", align 1
@key_str.226 = private unnamed_addr constant [8 x i8] c"b4.l1.b\00", align 1
@key_str.227 = private unnamed_addr constant [14 x i8] c"b4.a.q_proj.W\00", align 1
@key_str.228 = private unnamed_addr constant [14 x i8] c"b4.a.q_proj.b\00", align 1
@key_str.229 = private unnamed_addr constant [14 x i8] c"b4.a.k_proj.W\00", align 1
@key_str.230 = private unnamed_addr constant [14 x i8] c"b4.a.k_proj.b\00", align 1
@key_str.231 = private unnamed_addr constant [14 x i8] c"b4.a.v_proj.W\00", align 1
@key_str.232 = private unnamed_addr constant [14 x i8] c"b4.a.v_proj.b\00", align 1
@key_str.233 = private unnamed_addr constant [14 x i8] c"b4.a.p_proj.W\00", align 1
@key_str.234 = private unnamed_addr constant [14 x i8] c"b4.a.p_proj.b\00", align 1
@key_str.235 = private unnamed_addr constant [8 x i8] c"b4.l2.w\00", align 1
@key_str.236 = private unnamed_addr constant [8 x i8] c"b4.l2.b\00", align 1
@key_str.237 = private unnamed_addr constant [9 x i8] c"b4.m.f.W\00", align 1
@key_str.238 = private unnamed_addr constant [9 x i8] c"b4.m.f.b\00", align 1
@key_str.239 = private unnamed_addr constant [9 x i8] c"b4.m.p.W\00", align 1
@key_str.240 = private unnamed_addr constant [9 x i8] c"b4.m.p.b\00", align 1
@key_str.241 = private unnamed_addr constant [4 x i8] c"l.w\00", align 1
@key_str.242 = private unnamed_addr constant [4 x i8] c"l.b\00", align 1
@key_str.243 = private unnamed_addr constant [4 x i8] c"h.W\00", align 1
@key_str.244 = private unnamed_addr constant [4 x i8] c"h.b\00", align 1
@str_literal.245 = private unnamed_addr constant [24 x i8] c"model_heavy.safetensors\00", align 1
@str_literal.246 = private unnamed_addr constant [18 x i8] c"Saved checkpoint.\00", align 1
@str_literal.247 = private unnamed_addr constant [19 x i8] c"Training Complete!\00", align 1
@key_str.248 = private unnamed_addr constant [4 x i8] c"w.w\00", align 1
@key_str.249 = private unnamed_addr constant [5 x i8] c"wp.w\00", align 1
@key_str.250 = private unnamed_addr constant [8 x i8] c"b1.l1.w\00", align 1
@key_str.251 = private unnamed_addr constant [8 x i8] c"b1.l1.b\00", align 1
@key_str.252 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.W\00", align 1
@key_str.253 = private unnamed_addr constant [14 x i8] c"b1.a.q_proj.b\00", align 1
@key_str.254 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.W\00", align 1
@key_str.255 = private unnamed_addr constant [14 x i8] c"b1.a.k_proj.b\00", align 1
@key_str.256 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.W\00", align 1
@key_str.257 = private unnamed_addr constant [14 x i8] c"b1.a.v_proj.b\00", align 1
@key_str.258 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.W\00", align 1
@key_str.259 = private unnamed_addr constant [14 x i8] c"b1.a.p_proj.b\00", align 1
@key_str.260 = private unnamed_addr constant [8 x i8] c"b1.l2.w\00", align 1
@key_str.261 = private unnamed_addr constant [8 x i8] c"b1.l2.b\00", align 1
@key_str.262 = private unnamed_addr constant [9 x i8] c"b1.m.f.W\00", align 1
@key_str.263 = private unnamed_addr constant [9 x i8] c"b1.m.f.b\00", align 1
@key_str.264 = private unnamed_addr constant [9 x i8] c"b1.m.p.W\00", align 1
@key_str.265 = private unnamed_addr constant [9 x i8] c"b1.m.p.b\00", align 1
@key_str.266 = private unnamed_addr constant [8 x i8] c"b2.l1.w\00", align 1
@key_str.267 = private unnamed_addr constant [8 x i8] c"b2.l1.b\00", align 1
@key_str.268 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.W\00", align 1
@key_str.269 = private unnamed_addr constant [14 x i8] c"b2.a.q_proj.b\00", align 1
@key_str.270 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.W\00", align 1
@key_str.271 = private unnamed_addr constant [14 x i8] c"b2.a.k_proj.b\00", align 1
@key_str.272 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.W\00", align 1
@key_str.273 = private unnamed_addr constant [14 x i8] c"b2.a.v_proj.b\00", align 1
@key_str.274 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.W\00", align 1
@key_str.275 = private unnamed_addr constant [14 x i8] c"b2.a.p_proj.b\00", align 1
@key_str.276 = private unnamed_addr constant [8 x i8] c"b2.l2.w\00", align 1
@key_str.277 = private unnamed_addr constant [8 x i8] c"b2.l2.b\00", align 1
@key_str.278 = private unnamed_addr constant [9 x i8] c"b2.m.f.W\00", align 1
@key_str.279 = private unnamed_addr constant [9 x i8] c"b2.m.f.b\00", align 1
@key_str.280 = private unnamed_addr constant [9 x i8] c"b2.m.p.W\00", align 1
@key_str.281 = private unnamed_addr constant [9 x i8] c"b2.m.p.b\00", align 1
@key_str.282 = private unnamed_addr constant [8 x i8] c"b3.l1.w\00", align 1
@key_str.283 = private unnamed_addr constant [8 x i8] c"b3.l1.b\00", align 1
@key_str.284 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.W\00", align 1
@key_str.285 = private unnamed_addr constant [14 x i8] c"b3.a.q_proj.b\00", align 1
@key_str.286 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.W\00", align 1
@key_str.287 = private unnamed_addr constant [14 x i8] c"b3.a.k_proj.b\00", align 1
@key_str.288 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.W\00", align 1
@key_str.289 = private unnamed_addr constant [14 x i8] c"b3.a.v_proj.b\00", align 1
@key_str.290 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.W\00", align 1
@key_str.291 = private unnamed_addr constant [14 x i8] c"b3.a.p_proj.b\00", align 1
@key_str.292 = private unnamed_addr constant [8 x i8] c"b3.l2.w\00", align 1
@key_str.293 = private unnamed_addr constant [8 x i8] c"b3.l2.b\00", align 1
@key_str.294 = private unnamed_addr constant [9 x i8] c"b3.m.f.W\00", align 1
@key_str.295 = private unnamed_addr constant [9 x i8] c"b3.m.f.b\00", align 1
@key_str.296 = private unnamed_addr constant [9 x i8] c"b3.m.p.W\00", align 1
@key_str.297 = private unnamed_addr constant [9 x i8] c"b3.m.p.b\00", align 1
@key_str.298 = private unnamed_addr constant [8 x i8] c"b4.l1.w\00", align 1
@key_str.299 = private unnamed_addr constant [8 x i8] c"b4.l1.b\00", align 1
@key_str.300 = private unnamed_addr constant [14 x i8] c"b4.a.q_proj.W\00", align 1
@key_str.301 = private unnamed_addr constant [14 x i8] c"b4.a.q_proj.b\00", align 1
@key_str.302 = private unnamed_addr constant [14 x i8] c"b4.a.k_proj.W\00", align 1
@key_str.303 = private unnamed_addr constant [14 x i8] c"b4.a.k_proj.b\00", align 1
@key_str.304 = private unnamed_addr constant [14 x i8] c"b4.a.v_proj.W\00", align 1
@key_str.305 = private unnamed_addr constant [14 x i8] c"b4.a.v_proj.b\00", align 1
@key_str.306 = private unnamed_addr constant [14 x i8] c"b4.a.p_proj.W\00", align 1
@key_str.307 = private unnamed_addr constant [14 x i8] c"b4.a.p_proj.b\00", align 1
@key_str.308 = private unnamed_addr constant [8 x i8] c"b4.l2.w\00", align 1
@key_str.309 = private unnamed_addr constant [8 x i8] c"b4.l2.b\00", align 1
@key_str.310 = private unnamed_addr constant [9 x i8] c"b4.m.f.W\00", align 1
@key_str.311 = private unnamed_addr constant [9 x i8] c"b4.m.f.b\00", align 1
@key_str.312 = private unnamed_addr constant [9 x i8] c"b4.m.p.W\00", align 1
@key_str.313 = private unnamed_addr constant [9 x i8] c"b4.m.p.b\00", align 1
@key_str.314 = private unnamed_addr constant [4 x i8] c"l.w\00", align 1
@key_str.315 = private unnamed_addr constant [4 x i8] c"l.b\00", align 1
@key_str.316 = private unnamed_addr constant [4 x i8] c"h.W\00", align 1
@key_str.317 = private unnamed_addr constant [4 x i8] c"h.b\00", align 1
@str_literal.318 = private unnamed_addr constant [24 x i8] c"model_heavy.safetensors\00", align 1

declare void @tl_print_i64(i64)

declare void @tl_print_f32(float)

declare void @tl_print_string(ptr)

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
  call void @tl_tensor_free(ptr %binop_res)
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
  call void @tl_tensor_free(ptr %binop_res18)
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
  call void @tl_tensor_free(ptr %binop_res14)
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
  call void @tl_tensor_free(ptr %binop_res26)
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
  %tensor_to_free = load ptr, ptr %gb, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free34 = load ptr, ptr %gW, align 8
  call void @tl_tensor_free(ptr %tensor_to_free34)
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
  call void @tl_tensor_free(ptr %binop_res)
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
  call void @tl_tensor_free(ptr %binop_res12)
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
  call void @tl_tensor_free(ptr %binop_res6)
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
  call void @tl_tensor_free(ptr %binop_res18)
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
  call void @tl_tensor_free(ptr %binop_res12)
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
  store float 0x3FAA209AA0000000, ptr %scalar_data_rhs, align 4
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
  %tensor_to_free = load ptr, ptr %y, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free18 = load ptr, ptr %k, align 8
  call void @tl_tensor_free(ptr %tensor_to_free18)
  %tensor_to_free19 = load ptr, ptr %q, align 8
  call void @tl_tensor_free(ptr %tensor_to_free19)
  %tensor_to_free20 = load ptr, ptr %v, align 8
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

define ptr @tl_GPTHeavy_new(i64 %v, i64 %d) {
entry:
  %d2 = alloca i64, align 16
  %v1 = alloca i64, align 16
  call void @tl_mem_enter_scope()
  store i64 %v, ptr %v1, align 8
  store i64 %d, ptr %d2, align 8
  %struct_malloc = call ptr @malloc(i64 ptrtoint (ptr getelementptr (%GPTHeavy, ptr null, i32 1) to i64))
  %v3 = load i64, ptr %v1, align 8
  %d4 = load i64, ptr %d2, align 8
  %static_call = call ptr @tl_Embedding_new(i64 %v3, i64 %d4)
  %init_field = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 0
  store ptr %static_call, ptr %init_field, align 8
  %d5 = load i64, ptr %d2, align 8
  %static_call6 = call ptr @tl_Embedding_new(i64 12, i64 %d5)
  %init_field7 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 1
  store ptr %static_call6, ptr %init_field7, align 8
  %d8 = load i64, ptr %d2, align 8
  %static_call9 = call ptr @tl_Block_new(i64 %d8)
  %init_field10 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 2
  store ptr %static_call9, ptr %init_field10, align 8
  %d11 = load i64, ptr %d2, align 8
  %static_call12 = call ptr @tl_Block_new(i64 %d11)
  %init_field13 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 3
  store ptr %static_call12, ptr %init_field13, align 8
  %d14 = load i64, ptr %d2, align 8
  %static_call15 = call ptr @tl_Block_new(i64 %d14)
  %init_field16 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 4
  store ptr %static_call15, ptr %init_field16, align 8
  %d17 = load i64, ptr %d2, align 8
  %static_call18 = call ptr @tl_Block_new(i64 %d17)
  %init_field19 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 5
  store ptr %static_call18, ptr %init_field19, align 8
  %d20 = load i64, ptr %d2, align 8
  %static_call21 = call ptr @tl_LayerNorm_new(i64 %d20)
  %init_field22 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 6
  store ptr %static_call21, ptr %init_field22, align 8
  %d23 = load i64, ptr %d2, align 8
  %v24 = load i64, ptr %v1, align 8
  %static_call25 = call ptr @tl_Linear_new(i64 %d23, i64 %v24)
  %init_field26 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 7
  store ptr %static_call25, ptr %init_field26, align 8
  call void @tl_mem_unregister(ptr %struct_malloc)
  %unreg_field_0 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_027 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val28 = load ptr, ptr %unreg_field_027, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_1 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 1
  %field_val29 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val29)
  %unreg_field_030 = getelementptr inbounds %Embedding, ptr %field_val29, i32 0, i32 0
  %field_val31 = load ptr, ptr %unreg_field_030, align 8
  call void @tl_mem_unregister(ptr %field_val31)
  %unreg_field_2 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 2
  %field_val32 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val32)
  %unreg_field_033 = getelementptr inbounds %Block, ptr %field_val32, i32 0, i32 0
  %field_val34 = load ptr, ptr %unreg_field_033, align 8
  call void @tl_mem_unregister(ptr %field_val34)
  %unreg_field_035 = getelementptr inbounds %LayerNorm, ptr %field_val34, i32 0, i32 0
  %field_val36 = load ptr, ptr %unreg_field_035, align 8
  call void @tl_mem_unregister(ptr %field_val36)
  %unreg_field_137 = getelementptr inbounds %LayerNorm, ptr %field_val34, i32 0, i32 1
  %field_val38 = load ptr, ptr %unreg_field_137, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_139 = getelementptr inbounds %Block, ptr %field_val32, i32 0, i32 1
  %field_val40 = load ptr, ptr %unreg_field_139, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_041 = getelementptr inbounds %CausalSelfAttention, ptr %field_val40, i32 0, i32 0
  %field_val42 = load ptr, ptr %unreg_field_041, align 8
  call void @tl_mem_unregister(ptr %field_val42)
  %unreg_field_043 = getelementptr inbounds %Linear, ptr %field_val42, i32 0, i32 0
  %field_val44 = load ptr, ptr %unreg_field_043, align 8
  call void @tl_mem_unregister(ptr %field_val44)
  %unreg_field_145 = getelementptr inbounds %Linear, ptr %field_val42, i32 0, i32 1
  %field_val46 = load ptr, ptr %unreg_field_145, align 8
  call void @tl_mem_unregister(ptr %field_val46)
  %unreg_field_147 = getelementptr inbounds %CausalSelfAttention, ptr %field_val40, i32 0, i32 1
  %field_val48 = load ptr, ptr %unreg_field_147, align 8
  call void @tl_mem_unregister(ptr %field_val48)
  %unreg_field_049 = getelementptr inbounds %Linear, ptr %field_val48, i32 0, i32 0
  %field_val50 = load ptr, ptr %unreg_field_049, align 8
  call void @tl_mem_unregister(ptr %field_val50)
  %unreg_field_151 = getelementptr inbounds %Linear, ptr %field_val48, i32 0, i32 1
  %field_val52 = load ptr, ptr %unreg_field_151, align 8
  call void @tl_mem_unregister(ptr %field_val52)
  %unreg_field_253 = getelementptr inbounds %CausalSelfAttention, ptr %field_val40, i32 0, i32 2
  %field_val54 = load ptr, ptr %unreg_field_253, align 8
  call void @tl_mem_unregister(ptr %field_val54)
  %unreg_field_055 = getelementptr inbounds %Linear, ptr %field_val54, i32 0, i32 0
  %field_val56 = load ptr, ptr %unreg_field_055, align 8
  call void @tl_mem_unregister(ptr %field_val56)
  %unreg_field_157 = getelementptr inbounds %Linear, ptr %field_val54, i32 0, i32 1
  %field_val58 = load ptr, ptr %unreg_field_157, align 8
  call void @tl_mem_unregister(ptr %field_val58)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val40, i32 0, i32 3
  %field_val59 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val59)
  %unreg_field_060 = getelementptr inbounds %Linear, ptr %field_val59, i32 0, i32 0
  %field_val61 = load ptr, ptr %unreg_field_060, align 8
  call void @tl_mem_unregister(ptr %field_val61)
  %unreg_field_162 = getelementptr inbounds %Linear, ptr %field_val59, i32 0, i32 1
  %field_val63 = load ptr, ptr %unreg_field_162, align 8
  call void @tl_mem_unregister(ptr %field_val63)
  %unreg_field_264 = getelementptr inbounds %Block, ptr %field_val32, i32 0, i32 2
  %field_val65 = load ptr, ptr %unreg_field_264, align 8
  call void @tl_mem_unregister(ptr %field_val65)
  %unreg_field_066 = getelementptr inbounds %LayerNorm, ptr %field_val65, i32 0, i32 0
  %field_val67 = load ptr, ptr %unreg_field_066, align 8
  call void @tl_mem_unregister(ptr %field_val67)
  %unreg_field_168 = getelementptr inbounds %LayerNorm, ptr %field_val65, i32 0, i32 1
  %field_val69 = load ptr, ptr %unreg_field_168, align 8
  call void @tl_mem_unregister(ptr %field_val69)
  %unreg_field_370 = getelementptr inbounds %Block, ptr %field_val32, i32 0, i32 3
  %field_val71 = load ptr, ptr %unreg_field_370, align 8
  call void @tl_mem_unregister(ptr %field_val71)
  %unreg_field_072 = getelementptr inbounds %MLP, ptr %field_val71, i32 0, i32 0
  %field_val73 = load ptr, ptr %unreg_field_072, align 8
  call void @tl_mem_unregister(ptr %field_val73)
  %unreg_field_074 = getelementptr inbounds %Linear, ptr %field_val73, i32 0, i32 0
  %field_val75 = load ptr, ptr %unreg_field_074, align 8
  call void @tl_mem_unregister(ptr %field_val75)
  %unreg_field_176 = getelementptr inbounds %Linear, ptr %field_val73, i32 0, i32 1
  %field_val77 = load ptr, ptr %unreg_field_176, align 8
  call void @tl_mem_unregister(ptr %field_val77)
  %unreg_field_178 = getelementptr inbounds %MLP, ptr %field_val71, i32 0, i32 1
  %field_val79 = load ptr, ptr %unreg_field_178, align 8
  call void @tl_mem_unregister(ptr %field_val79)
  %unreg_field_080 = getelementptr inbounds %Linear, ptr %field_val79, i32 0, i32 0
  %field_val81 = load ptr, ptr %unreg_field_080, align 8
  call void @tl_mem_unregister(ptr %field_val81)
  %unreg_field_182 = getelementptr inbounds %Linear, ptr %field_val79, i32 0, i32 1
  %field_val83 = load ptr, ptr %unreg_field_182, align 8
  call void @tl_mem_unregister(ptr %field_val83)
  %unreg_field_384 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 3
  %field_val85 = load ptr, ptr %unreg_field_384, align 8
  call void @tl_mem_unregister(ptr %field_val85)
  %unreg_field_086 = getelementptr inbounds %Block, ptr %field_val85, i32 0, i32 0
  %field_val87 = load ptr, ptr %unreg_field_086, align 8
  call void @tl_mem_unregister(ptr %field_val87)
  %unreg_field_088 = getelementptr inbounds %LayerNorm, ptr %field_val87, i32 0, i32 0
  %field_val89 = load ptr, ptr %unreg_field_088, align 8
  call void @tl_mem_unregister(ptr %field_val89)
  %unreg_field_190 = getelementptr inbounds %LayerNorm, ptr %field_val87, i32 0, i32 1
  %field_val91 = load ptr, ptr %unreg_field_190, align 8
  call void @tl_mem_unregister(ptr %field_val91)
  %unreg_field_192 = getelementptr inbounds %Block, ptr %field_val85, i32 0, i32 1
  %field_val93 = load ptr, ptr %unreg_field_192, align 8
  call void @tl_mem_unregister(ptr %field_val93)
  %unreg_field_094 = getelementptr inbounds %CausalSelfAttention, ptr %field_val93, i32 0, i32 0
  %field_val95 = load ptr, ptr %unreg_field_094, align 8
  call void @tl_mem_unregister(ptr %field_val95)
  %unreg_field_096 = getelementptr inbounds %Linear, ptr %field_val95, i32 0, i32 0
  %field_val97 = load ptr, ptr %unreg_field_096, align 8
  call void @tl_mem_unregister(ptr %field_val97)
  %unreg_field_198 = getelementptr inbounds %Linear, ptr %field_val95, i32 0, i32 1
  %field_val99 = load ptr, ptr %unreg_field_198, align 8
  call void @tl_mem_unregister(ptr %field_val99)
  %unreg_field_1100 = getelementptr inbounds %CausalSelfAttention, ptr %field_val93, i32 0, i32 1
  %field_val101 = load ptr, ptr %unreg_field_1100, align 8
  call void @tl_mem_unregister(ptr %field_val101)
  %unreg_field_0102 = getelementptr inbounds %Linear, ptr %field_val101, i32 0, i32 0
  %field_val103 = load ptr, ptr %unreg_field_0102, align 8
  call void @tl_mem_unregister(ptr %field_val103)
  %unreg_field_1104 = getelementptr inbounds %Linear, ptr %field_val101, i32 0, i32 1
  %field_val105 = load ptr, ptr %unreg_field_1104, align 8
  call void @tl_mem_unregister(ptr %field_val105)
  %unreg_field_2106 = getelementptr inbounds %CausalSelfAttention, ptr %field_val93, i32 0, i32 2
  %field_val107 = load ptr, ptr %unreg_field_2106, align 8
  call void @tl_mem_unregister(ptr %field_val107)
  %unreg_field_0108 = getelementptr inbounds %Linear, ptr %field_val107, i32 0, i32 0
  %field_val109 = load ptr, ptr %unreg_field_0108, align 8
  call void @tl_mem_unregister(ptr %field_val109)
  %unreg_field_1110 = getelementptr inbounds %Linear, ptr %field_val107, i32 0, i32 1
  %field_val111 = load ptr, ptr %unreg_field_1110, align 8
  call void @tl_mem_unregister(ptr %field_val111)
  %unreg_field_3112 = getelementptr inbounds %CausalSelfAttention, ptr %field_val93, i32 0, i32 3
  %field_val113 = load ptr, ptr %unreg_field_3112, align 8
  call void @tl_mem_unregister(ptr %field_val113)
  %unreg_field_0114 = getelementptr inbounds %Linear, ptr %field_val113, i32 0, i32 0
  %field_val115 = load ptr, ptr %unreg_field_0114, align 8
  call void @tl_mem_unregister(ptr %field_val115)
  %unreg_field_1116 = getelementptr inbounds %Linear, ptr %field_val113, i32 0, i32 1
  %field_val117 = load ptr, ptr %unreg_field_1116, align 8
  call void @tl_mem_unregister(ptr %field_val117)
  %unreg_field_2118 = getelementptr inbounds %Block, ptr %field_val85, i32 0, i32 2
  %field_val119 = load ptr, ptr %unreg_field_2118, align 8
  call void @tl_mem_unregister(ptr %field_val119)
  %unreg_field_0120 = getelementptr inbounds %LayerNorm, ptr %field_val119, i32 0, i32 0
  %field_val121 = load ptr, ptr %unreg_field_0120, align 8
  call void @tl_mem_unregister(ptr %field_val121)
  %unreg_field_1122 = getelementptr inbounds %LayerNorm, ptr %field_val119, i32 0, i32 1
  %field_val123 = load ptr, ptr %unreg_field_1122, align 8
  call void @tl_mem_unregister(ptr %field_val123)
  %unreg_field_3124 = getelementptr inbounds %Block, ptr %field_val85, i32 0, i32 3
  %field_val125 = load ptr, ptr %unreg_field_3124, align 8
  call void @tl_mem_unregister(ptr %field_val125)
  %unreg_field_0126 = getelementptr inbounds %MLP, ptr %field_val125, i32 0, i32 0
  %field_val127 = load ptr, ptr %unreg_field_0126, align 8
  call void @tl_mem_unregister(ptr %field_val127)
  %unreg_field_0128 = getelementptr inbounds %Linear, ptr %field_val127, i32 0, i32 0
  %field_val129 = load ptr, ptr %unreg_field_0128, align 8
  call void @tl_mem_unregister(ptr %field_val129)
  %unreg_field_1130 = getelementptr inbounds %Linear, ptr %field_val127, i32 0, i32 1
  %field_val131 = load ptr, ptr %unreg_field_1130, align 8
  call void @tl_mem_unregister(ptr %field_val131)
  %unreg_field_1132 = getelementptr inbounds %MLP, ptr %field_val125, i32 0, i32 1
  %field_val133 = load ptr, ptr %unreg_field_1132, align 8
  call void @tl_mem_unregister(ptr %field_val133)
  %unreg_field_0134 = getelementptr inbounds %Linear, ptr %field_val133, i32 0, i32 0
  %field_val135 = load ptr, ptr %unreg_field_0134, align 8
  call void @tl_mem_unregister(ptr %field_val135)
  %unreg_field_1136 = getelementptr inbounds %Linear, ptr %field_val133, i32 0, i32 1
  %field_val137 = load ptr, ptr %unreg_field_1136, align 8
  call void @tl_mem_unregister(ptr %field_val137)
  %unreg_field_4 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 4
  %field_val138 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val138)
  %unreg_field_0139 = getelementptr inbounds %Block, ptr %field_val138, i32 0, i32 0
  %field_val140 = load ptr, ptr %unreg_field_0139, align 8
  call void @tl_mem_unregister(ptr %field_val140)
  %unreg_field_0141 = getelementptr inbounds %LayerNorm, ptr %field_val140, i32 0, i32 0
  %field_val142 = load ptr, ptr %unreg_field_0141, align 8
  call void @tl_mem_unregister(ptr %field_val142)
  %unreg_field_1143 = getelementptr inbounds %LayerNorm, ptr %field_val140, i32 0, i32 1
  %field_val144 = load ptr, ptr %unreg_field_1143, align 8
  call void @tl_mem_unregister(ptr %field_val144)
  %unreg_field_1145 = getelementptr inbounds %Block, ptr %field_val138, i32 0, i32 1
  %field_val146 = load ptr, ptr %unreg_field_1145, align 8
  call void @tl_mem_unregister(ptr %field_val146)
  %unreg_field_0147 = getelementptr inbounds %CausalSelfAttention, ptr %field_val146, i32 0, i32 0
  %field_val148 = load ptr, ptr %unreg_field_0147, align 8
  call void @tl_mem_unregister(ptr %field_val148)
  %unreg_field_0149 = getelementptr inbounds %Linear, ptr %field_val148, i32 0, i32 0
  %field_val150 = load ptr, ptr %unreg_field_0149, align 8
  call void @tl_mem_unregister(ptr %field_val150)
  %unreg_field_1151 = getelementptr inbounds %Linear, ptr %field_val148, i32 0, i32 1
  %field_val152 = load ptr, ptr %unreg_field_1151, align 8
  call void @tl_mem_unregister(ptr %field_val152)
  %unreg_field_1153 = getelementptr inbounds %CausalSelfAttention, ptr %field_val146, i32 0, i32 1
  %field_val154 = load ptr, ptr %unreg_field_1153, align 8
  call void @tl_mem_unregister(ptr %field_val154)
  %unreg_field_0155 = getelementptr inbounds %Linear, ptr %field_val154, i32 0, i32 0
  %field_val156 = load ptr, ptr %unreg_field_0155, align 8
  call void @tl_mem_unregister(ptr %field_val156)
  %unreg_field_1157 = getelementptr inbounds %Linear, ptr %field_val154, i32 0, i32 1
  %field_val158 = load ptr, ptr %unreg_field_1157, align 8
  call void @tl_mem_unregister(ptr %field_val158)
  %unreg_field_2159 = getelementptr inbounds %CausalSelfAttention, ptr %field_val146, i32 0, i32 2
  %field_val160 = load ptr, ptr %unreg_field_2159, align 8
  call void @tl_mem_unregister(ptr %field_val160)
  %unreg_field_0161 = getelementptr inbounds %Linear, ptr %field_val160, i32 0, i32 0
  %field_val162 = load ptr, ptr %unreg_field_0161, align 8
  call void @tl_mem_unregister(ptr %field_val162)
  %unreg_field_1163 = getelementptr inbounds %Linear, ptr %field_val160, i32 0, i32 1
  %field_val164 = load ptr, ptr %unreg_field_1163, align 8
  call void @tl_mem_unregister(ptr %field_val164)
  %unreg_field_3165 = getelementptr inbounds %CausalSelfAttention, ptr %field_val146, i32 0, i32 3
  %field_val166 = load ptr, ptr %unreg_field_3165, align 8
  call void @tl_mem_unregister(ptr %field_val166)
  %unreg_field_0167 = getelementptr inbounds %Linear, ptr %field_val166, i32 0, i32 0
  %field_val168 = load ptr, ptr %unreg_field_0167, align 8
  call void @tl_mem_unregister(ptr %field_val168)
  %unreg_field_1169 = getelementptr inbounds %Linear, ptr %field_val166, i32 0, i32 1
  %field_val170 = load ptr, ptr %unreg_field_1169, align 8
  call void @tl_mem_unregister(ptr %field_val170)
  %unreg_field_2171 = getelementptr inbounds %Block, ptr %field_val138, i32 0, i32 2
  %field_val172 = load ptr, ptr %unreg_field_2171, align 8
  call void @tl_mem_unregister(ptr %field_val172)
  %unreg_field_0173 = getelementptr inbounds %LayerNorm, ptr %field_val172, i32 0, i32 0
  %field_val174 = load ptr, ptr %unreg_field_0173, align 8
  call void @tl_mem_unregister(ptr %field_val174)
  %unreg_field_1175 = getelementptr inbounds %LayerNorm, ptr %field_val172, i32 0, i32 1
  %field_val176 = load ptr, ptr %unreg_field_1175, align 8
  call void @tl_mem_unregister(ptr %field_val176)
  %unreg_field_3177 = getelementptr inbounds %Block, ptr %field_val138, i32 0, i32 3
  %field_val178 = load ptr, ptr %unreg_field_3177, align 8
  call void @tl_mem_unregister(ptr %field_val178)
  %unreg_field_0179 = getelementptr inbounds %MLP, ptr %field_val178, i32 0, i32 0
  %field_val180 = load ptr, ptr %unreg_field_0179, align 8
  call void @tl_mem_unregister(ptr %field_val180)
  %unreg_field_0181 = getelementptr inbounds %Linear, ptr %field_val180, i32 0, i32 0
  %field_val182 = load ptr, ptr %unreg_field_0181, align 8
  call void @tl_mem_unregister(ptr %field_val182)
  %unreg_field_1183 = getelementptr inbounds %Linear, ptr %field_val180, i32 0, i32 1
  %field_val184 = load ptr, ptr %unreg_field_1183, align 8
  call void @tl_mem_unregister(ptr %field_val184)
  %unreg_field_1185 = getelementptr inbounds %MLP, ptr %field_val178, i32 0, i32 1
  %field_val186 = load ptr, ptr %unreg_field_1185, align 8
  call void @tl_mem_unregister(ptr %field_val186)
  %unreg_field_0187 = getelementptr inbounds %Linear, ptr %field_val186, i32 0, i32 0
  %field_val188 = load ptr, ptr %unreg_field_0187, align 8
  call void @tl_mem_unregister(ptr %field_val188)
  %unreg_field_1189 = getelementptr inbounds %Linear, ptr %field_val186, i32 0, i32 1
  %field_val190 = load ptr, ptr %unreg_field_1189, align 8
  call void @tl_mem_unregister(ptr %field_val190)
  %unreg_field_5 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 5
  %field_val191 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val191)
  %unreg_field_0192 = getelementptr inbounds %Block, ptr %field_val191, i32 0, i32 0
  %field_val193 = load ptr, ptr %unreg_field_0192, align 8
  call void @tl_mem_unregister(ptr %field_val193)
  %unreg_field_0194 = getelementptr inbounds %LayerNorm, ptr %field_val193, i32 0, i32 0
  %field_val195 = load ptr, ptr %unreg_field_0194, align 8
  call void @tl_mem_unregister(ptr %field_val195)
  %unreg_field_1196 = getelementptr inbounds %LayerNorm, ptr %field_val193, i32 0, i32 1
  %field_val197 = load ptr, ptr %unreg_field_1196, align 8
  call void @tl_mem_unregister(ptr %field_val197)
  %unreg_field_1198 = getelementptr inbounds %Block, ptr %field_val191, i32 0, i32 1
  %field_val199 = load ptr, ptr %unreg_field_1198, align 8
  call void @tl_mem_unregister(ptr %field_val199)
  %unreg_field_0200 = getelementptr inbounds %CausalSelfAttention, ptr %field_val199, i32 0, i32 0
  %field_val201 = load ptr, ptr %unreg_field_0200, align 8
  call void @tl_mem_unregister(ptr %field_val201)
  %unreg_field_0202 = getelementptr inbounds %Linear, ptr %field_val201, i32 0, i32 0
  %field_val203 = load ptr, ptr %unreg_field_0202, align 8
  call void @tl_mem_unregister(ptr %field_val203)
  %unreg_field_1204 = getelementptr inbounds %Linear, ptr %field_val201, i32 0, i32 1
  %field_val205 = load ptr, ptr %unreg_field_1204, align 8
  call void @tl_mem_unregister(ptr %field_val205)
  %unreg_field_1206 = getelementptr inbounds %CausalSelfAttention, ptr %field_val199, i32 0, i32 1
  %field_val207 = load ptr, ptr %unreg_field_1206, align 8
  call void @tl_mem_unregister(ptr %field_val207)
  %unreg_field_0208 = getelementptr inbounds %Linear, ptr %field_val207, i32 0, i32 0
  %field_val209 = load ptr, ptr %unreg_field_0208, align 8
  call void @tl_mem_unregister(ptr %field_val209)
  %unreg_field_1210 = getelementptr inbounds %Linear, ptr %field_val207, i32 0, i32 1
  %field_val211 = load ptr, ptr %unreg_field_1210, align 8
  call void @tl_mem_unregister(ptr %field_val211)
  %unreg_field_2212 = getelementptr inbounds %CausalSelfAttention, ptr %field_val199, i32 0, i32 2
  %field_val213 = load ptr, ptr %unreg_field_2212, align 8
  call void @tl_mem_unregister(ptr %field_val213)
  %unreg_field_0214 = getelementptr inbounds %Linear, ptr %field_val213, i32 0, i32 0
  %field_val215 = load ptr, ptr %unreg_field_0214, align 8
  call void @tl_mem_unregister(ptr %field_val215)
  %unreg_field_1216 = getelementptr inbounds %Linear, ptr %field_val213, i32 0, i32 1
  %field_val217 = load ptr, ptr %unreg_field_1216, align 8
  call void @tl_mem_unregister(ptr %field_val217)
  %unreg_field_3218 = getelementptr inbounds %CausalSelfAttention, ptr %field_val199, i32 0, i32 3
  %field_val219 = load ptr, ptr %unreg_field_3218, align 8
  call void @tl_mem_unregister(ptr %field_val219)
  %unreg_field_0220 = getelementptr inbounds %Linear, ptr %field_val219, i32 0, i32 0
  %field_val221 = load ptr, ptr %unreg_field_0220, align 8
  call void @tl_mem_unregister(ptr %field_val221)
  %unreg_field_1222 = getelementptr inbounds %Linear, ptr %field_val219, i32 0, i32 1
  %field_val223 = load ptr, ptr %unreg_field_1222, align 8
  call void @tl_mem_unregister(ptr %field_val223)
  %unreg_field_2224 = getelementptr inbounds %Block, ptr %field_val191, i32 0, i32 2
  %field_val225 = load ptr, ptr %unreg_field_2224, align 8
  call void @tl_mem_unregister(ptr %field_val225)
  %unreg_field_0226 = getelementptr inbounds %LayerNorm, ptr %field_val225, i32 0, i32 0
  %field_val227 = load ptr, ptr %unreg_field_0226, align 8
  call void @tl_mem_unregister(ptr %field_val227)
  %unreg_field_1228 = getelementptr inbounds %LayerNorm, ptr %field_val225, i32 0, i32 1
  %field_val229 = load ptr, ptr %unreg_field_1228, align 8
  call void @tl_mem_unregister(ptr %field_val229)
  %unreg_field_3230 = getelementptr inbounds %Block, ptr %field_val191, i32 0, i32 3
  %field_val231 = load ptr, ptr %unreg_field_3230, align 8
  call void @tl_mem_unregister(ptr %field_val231)
  %unreg_field_0232 = getelementptr inbounds %MLP, ptr %field_val231, i32 0, i32 0
  %field_val233 = load ptr, ptr %unreg_field_0232, align 8
  call void @tl_mem_unregister(ptr %field_val233)
  %unreg_field_0234 = getelementptr inbounds %Linear, ptr %field_val233, i32 0, i32 0
  %field_val235 = load ptr, ptr %unreg_field_0234, align 8
  call void @tl_mem_unregister(ptr %field_val235)
  %unreg_field_1236 = getelementptr inbounds %Linear, ptr %field_val233, i32 0, i32 1
  %field_val237 = load ptr, ptr %unreg_field_1236, align 8
  call void @tl_mem_unregister(ptr %field_val237)
  %unreg_field_1238 = getelementptr inbounds %MLP, ptr %field_val231, i32 0, i32 1
  %field_val239 = load ptr, ptr %unreg_field_1238, align 8
  call void @tl_mem_unregister(ptr %field_val239)
  %unreg_field_0240 = getelementptr inbounds %Linear, ptr %field_val239, i32 0, i32 0
  %field_val241 = load ptr, ptr %unreg_field_0240, align 8
  call void @tl_mem_unregister(ptr %field_val241)
  %unreg_field_1242 = getelementptr inbounds %Linear, ptr %field_val239, i32 0, i32 1
  %field_val243 = load ptr, ptr %unreg_field_1242, align 8
  call void @tl_mem_unregister(ptr %field_val243)
  %unreg_field_6 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 6
  %field_val244 = load ptr, ptr %unreg_field_6, align 8
  call void @tl_mem_unregister(ptr %field_val244)
  %unreg_field_0245 = getelementptr inbounds %LayerNorm, ptr %field_val244, i32 0, i32 0
  %field_val246 = load ptr, ptr %unreg_field_0245, align 8
  call void @tl_mem_unregister(ptr %field_val246)
  %unreg_field_1247 = getelementptr inbounds %LayerNorm, ptr %field_val244, i32 0, i32 1
  %field_val248 = load ptr, ptr %unreg_field_1247, align 8
  call void @tl_mem_unregister(ptr %field_val248)
  %unreg_field_7 = getelementptr inbounds %GPTHeavy, ptr %struct_malloc, i32 0, i32 7
  %field_val249 = load ptr, ptr %unreg_field_7, align 8
  call void @tl_mem_unregister(ptr %field_val249)
  %unreg_field_0250 = getelementptr inbounds %Linear, ptr %field_val249, i32 0, i32 0
  %field_val251 = load ptr, ptr %unreg_field_0250, align 8
  call void @tl_mem_unregister(ptr %field_val251)
  %unreg_field_1252 = getelementptr inbounds %Linear, ptr %field_val249, i32 0, i32 1
  %field_val253 = load ptr, ptr %unreg_field_1252, align 8
  call void @tl_mem_unregister(ptr %field_val253)
  call void @tl_mem_exit_scope()
  ret ptr %struct_malloc
}

define ptr @tl_GPTHeavy_forward(ptr %self, ptr %i) {
entry:
  %x40 = alloca ptr, align 16
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
  %ptr_w = getelementptr inbounds %GPTHeavy, ptr %self15, i32 0, i32 0
  %w = load ptr, ptr %ptr_w, align 8
  %i16 = load ptr, ptr %i2, align 8
  %call_method = call ptr @tl_Embedding_forward(ptr %w, ptr %i16)
  call void @tl_mem_register_tensor(ptr %call_method)
  call void @tl_mem_unregister(ptr %call_method)
  store ptr %call_method, ptr %tok_emb, align 8
  %self17 = load ptr, ptr %self1, align 8
  %ptr_wp = getelementptr inbounds %GPTHeavy, ptr %self17, i32 0, i32 1
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
  %ptr_b1 = getelementptr inbounds %GPTHeavy, ptr %self22, i32 0, i32 2
  %b1 = load ptr, ptr %ptr_b1, align 8
  %x23 = load ptr, ptr %x, align 8
  %call_method24 = call ptr @tl_Block_forward(ptr %b1, ptr %x23)
  call void @tl_mem_register_tensor(ptr %call_method24)
  call void @tl_mem_unregister(ptr %call_method24)
  %old_shadowed = load ptr, ptr %x, align 8
  call void @tl_mem_unregister(ptr %old_shadowed)
  store ptr %call_method24, ptr %x25, align 8
  %self26 = load ptr, ptr %self1, align 8
  %ptr_b2 = getelementptr inbounds %GPTHeavy, ptr %self26, i32 0, i32 3
  %b2 = load ptr, ptr %ptr_b2, align 8
  %x27 = load ptr, ptr %x25, align 8
  %call_method28 = call ptr @tl_Block_forward(ptr %b2, ptr %x27)
  call void @tl_mem_register_tensor(ptr %call_method28)
  call void @tl_mem_unregister(ptr %call_method28)
  %old_shadowed29 = load ptr, ptr %x25, align 8
  call void @tl_mem_unregister(ptr %old_shadowed29)
  store ptr %call_method28, ptr %x30, align 8
  %self31 = load ptr, ptr %self1, align 8
  %ptr_b3 = getelementptr inbounds %GPTHeavy, ptr %self31, i32 0, i32 4
  %b3 = load ptr, ptr %ptr_b3, align 8
  %x32 = load ptr, ptr %x30, align 8
  %call_method33 = call ptr @tl_Block_forward(ptr %b3, ptr %x32)
  call void @tl_mem_register_tensor(ptr %call_method33)
  call void @tl_mem_unregister(ptr %call_method33)
  %old_shadowed34 = load ptr, ptr %x30, align 8
  call void @tl_mem_unregister(ptr %old_shadowed34)
  store ptr %call_method33, ptr %x35, align 8
  %self36 = load ptr, ptr %self1, align 8
  %ptr_b4 = getelementptr inbounds %GPTHeavy, ptr %self36, i32 0, i32 5
  %b4 = load ptr, ptr %ptr_b4, align 8
  %x37 = load ptr, ptr %x35, align 8
  %call_method38 = call ptr @tl_Block_forward(ptr %b4, ptr %x37)
  call void @tl_mem_register_tensor(ptr %call_method38)
  call void @tl_mem_unregister(ptr %call_method38)
  %old_shadowed39 = load ptr, ptr %x35, align 8
  call void @tl_mem_unregister(ptr %old_shadowed39)
  store ptr %call_method38, ptr %x40, align 8
  %self41 = load ptr, ptr %self1, align 8
  %ptr_h = getelementptr inbounds %GPTHeavy, ptr %self41, i32 0, i32 7
  %h = load ptr, ptr %ptr_h, align 8
  %self42 = load ptr, ptr %self1, align 8
  %ptr_l = getelementptr inbounds %GPTHeavy, ptr %self42, i32 0, i32 6
  %l = load ptr, ptr %ptr_l, align 8
  %x43 = load ptr, ptr %x40, align 8
  %call_method44 = call ptr @tl_LayerNorm_forward(ptr %l, ptr %x43)
  call void @tl_mem_register_tensor(ptr %call_method44)
  %call_method45 = call ptr @tl_Linear_forward(ptr %h, ptr %call_method44)
  call void @tl_tensor_free(ptr %call_method44)
  call void @tl_mem_register_tensor(ptr %call_method45)
  call void @tl_mem_unregister(ptr %call_method45)
  %tensor_to_free = load ptr, ptr %pos_data, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free46 = load ptr, ptr %x40, align 8
  call void @tl_tensor_free(ptr %tensor_to_free46)
  %tensor_to_free47 = load ptr, ptr %pos_emb, align 8
  call void @tl_tensor_free(ptr %tensor_to_free47)
  %tensor_to_free48 = load ptr, ptr %tok_emb, align 8
  call void @tl_tensor_free(ptr %tensor_to_free48)
  %tensor_to_free49 = load ptr, ptr %pos, align 8
  call void @tl_tensor_free(ptr %tensor_to_free49)
  call void @tl_mem_exit_scope()
  ret ptr %call_method45
}

define ptr @tl_GPTHeavy_step(ptr %self, float %lr) {
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
  %ptr_w = getelementptr inbounds %GPTHeavy, ptr %s4, i32 0, i32 0
  %s5 = load ptr, ptr %s, align 8
  %ptr_w6 = getelementptr inbounds %GPTHeavy, ptr %s5, i32 0, i32 0
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
  %ptr_wp = getelementptr inbounds %GPTHeavy, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_wp10 = getelementptr inbounds %GPTHeavy, ptr %s9, i32 0, i32 1
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
  %ptr_b1 = getelementptr inbounds %GPTHeavy, ptr %s19, i32 0, i32 2
  %s20 = load ptr, ptr %s, align 8
  %ptr_b121 = getelementptr inbounds %GPTHeavy, ptr %s20, i32 0, i32 2
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
  %ptr_b2 = getelementptr inbounds %GPTHeavy, ptr %s80, i32 0, i32 3
  %s81 = load ptr, ptr %s, align 8
  %ptr_b282 = getelementptr inbounds %GPTHeavy, ptr %s81, i32 0, i32 3
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
  %ptr_b3 = getelementptr inbounds %GPTHeavy, ptr %s141, i32 0, i32 4
  %s142 = load ptr, ptr %s, align 8
  %ptr_b3143 = getelementptr inbounds %GPTHeavy, ptr %s142, i32 0, i32 4
  %b3 = load ptr, ptr %ptr_b3143, align 8
  %lr144 = load float, ptr %lr2, align 4
  %call_method145 = call ptr @tl_Block_step(ptr %b3, float %lr144)
  call void @tl_mem_register_struct(ptr %call_method145)
  %old_field_val146 = load ptr, ptr %ptr_b3, align 8
  %cnt_free_diff147 = icmp ne ptr %old_field_val146, %call_method145
  br i1 %cnt_free_diff147, label %free_old_val148, label %skip_free149

free_old_val148:                                  ; preds = %skip_free88
  %field_gep150 = getelementptr inbounds %Block, ptr %old_field_val146, i32 0, i32 0
  %field_load151 = load ptr, ptr %field_gep150, align 8
  %field_gep152 = getelementptr inbounds %LayerNorm, ptr %field_load151, i32 0, i32 0
  %field_load153 = load ptr, ptr %field_gep152, align 8
  call void @tl_tensor_free(ptr %field_load153)
  %field_gep154 = getelementptr inbounds %LayerNorm, ptr %field_load151, i32 0, i32 1
  %field_load155 = load ptr, ptr %field_gep154, align 8
  call void @tl_tensor_free(ptr %field_load155)
  %field_gep156 = getelementptr inbounds %Block, ptr %old_field_val146, i32 0, i32 1
  %field_load157 = load ptr, ptr %field_gep156, align 8
  %field_gep158 = getelementptr inbounds %CausalSelfAttention, ptr %field_load157, i32 0, i32 0
  %field_load159 = load ptr, ptr %field_gep158, align 8
  %field_gep160 = getelementptr inbounds %Linear, ptr %field_load159, i32 0, i32 0
  %field_load161 = load ptr, ptr %field_gep160, align 8
  call void @tl_tensor_free(ptr %field_load161)
  %field_gep162 = getelementptr inbounds %Linear, ptr %field_load159, i32 0, i32 1
  %field_load163 = load ptr, ptr %field_gep162, align 8
  call void @tl_tensor_free(ptr %field_load163)
  %field_gep164 = getelementptr inbounds %CausalSelfAttention, ptr %field_load157, i32 0, i32 1
  %field_load165 = load ptr, ptr %field_gep164, align 8
  %field_gep166 = getelementptr inbounds %Linear, ptr %field_load165, i32 0, i32 0
  %field_load167 = load ptr, ptr %field_gep166, align 8
  call void @tl_tensor_free(ptr %field_load167)
  %field_gep168 = getelementptr inbounds %Linear, ptr %field_load165, i32 0, i32 1
  %field_load169 = load ptr, ptr %field_gep168, align 8
  call void @tl_tensor_free(ptr %field_load169)
  %field_gep170 = getelementptr inbounds %CausalSelfAttention, ptr %field_load157, i32 0, i32 2
  %field_load171 = load ptr, ptr %field_gep170, align 8
  %field_gep172 = getelementptr inbounds %Linear, ptr %field_load171, i32 0, i32 0
  %field_load173 = load ptr, ptr %field_gep172, align 8
  call void @tl_tensor_free(ptr %field_load173)
  %field_gep174 = getelementptr inbounds %Linear, ptr %field_load171, i32 0, i32 1
  %field_load175 = load ptr, ptr %field_gep174, align 8
  call void @tl_tensor_free(ptr %field_load175)
  %field_gep176 = getelementptr inbounds %CausalSelfAttention, ptr %field_load157, i32 0, i32 3
  %field_load177 = load ptr, ptr %field_gep176, align 8
  %field_gep178 = getelementptr inbounds %Linear, ptr %field_load177, i32 0, i32 0
  %field_load179 = load ptr, ptr %field_gep178, align 8
  call void @tl_tensor_free(ptr %field_load179)
  %field_gep180 = getelementptr inbounds %Linear, ptr %field_load177, i32 0, i32 1
  %field_load181 = load ptr, ptr %field_gep180, align 8
  call void @tl_tensor_free(ptr %field_load181)
  %field_gep182 = getelementptr inbounds %Block, ptr %old_field_val146, i32 0, i32 2
  %field_load183 = load ptr, ptr %field_gep182, align 8
  %field_gep184 = getelementptr inbounds %LayerNorm, ptr %field_load183, i32 0, i32 0
  %field_load185 = load ptr, ptr %field_gep184, align 8
  call void @tl_tensor_free(ptr %field_load185)
  %field_gep186 = getelementptr inbounds %LayerNorm, ptr %field_load183, i32 0, i32 1
  %field_load187 = load ptr, ptr %field_gep186, align 8
  call void @tl_tensor_free(ptr %field_load187)
  %field_gep188 = getelementptr inbounds %Block, ptr %old_field_val146, i32 0, i32 3
  %field_load189 = load ptr, ptr %field_gep188, align 8
  %field_gep190 = getelementptr inbounds %MLP, ptr %field_load189, i32 0, i32 0
  %field_load191 = load ptr, ptr %field_gep190, align 8
  %field_gep192 = getelementptr inbounds %Linear, ptr %field_load191, i32 0, i32 0
  %field_load193 = load ptr, ptr %field_gep192, align 8
  call void @tl_tensor_free(ptr %field_load193)
  %field_gep194 = getelementptr inbounds %Linear, ptr %field_load191, i32 0, i32 1
  %field_load195 = load ptr, ptr %field_gep194, align 8
  call void @tl_tensor_free(ptr %field_load195)
  %field_gep196 = getelementptr inbounds %MLP, ptr %field_load189, i32 0, i32 1
  %field_load197 = load ptr, ptr %field_gep196, align 8
  %field_gep198 = getelementptr inbounds %Linear, ptr %field_load197, i32 0, i32 0
  %field_load199 = load ptr, ptr %field_gep198, align 8
  call void @tl_tensor_free(ptr %field_load199)
  %field_gep200 = getelementptr inbounds %Linear, ptr %field_load197, i32 0, i32 1
  %field_load201 = load ptr, ptr %field_gep200, align 8
  call void @tl_tensor_free(ptr %field_load201)
  br label %skip_free149

skip_free149:                                     ; preds = %free_old_val148, %skip_free88
  store ptr %call_method145, ptr %ptr_b3, align 8
  call void @tl_mem_unregister(ptr %call_method145)
  %s202 = load ptr, ptr %s, align 8
  %ptr_b4 = getelementptr inbounds %GPTHeavy, ptr %s202, i32 0, i32 5
  %s203 = load ptr, ptr %s, align 8
  %ptr_b4204 = getelementptr inbounds %GPTHeavy, ptr %s203, i32 0, i32 5
  %b4 = load ptr, ptr %ptr_b4204, align 8
  %lr205 = load float, ptr %lr2, align 4
  %call_method206 = call ptr @tl_Block_step(ptr %b4, float %lr205)
  call void @tl_mem_register_struct(ptr %call_method206)
  %old_field_val207 = load ptr, ptr %ptr_b4, align 8
  %cnt_free_diff208 = icmp ne ptr %old_field_val207, %call_method206
  br i1 %cnt_free_diff208, label %free_old_val209, label %skip_free210

free_old_val209:                                  ; preds = %skip_free149
  %field_gep211 = getelementptr inbounds %Block, ptr %old_field_val207, i32 0, i32 0
  %field_load212 = load ptr, ptr %field_gep211, align 8
  %field_gep213 = getelementptr inbounds %LayerNorm, ptr %field_load212, i32 0, i32 0
  %field_load214 = load ptr, ptr %field_gep213, align 8
  call void @tl_tensor_free(ptr %field_load214)
  %field_gep215 = getelementptr inbounds %LayerNorm, ptr %field_load212, i32 0, i32 1
  %field_load216 = load ptr, ptr %field_gep215, align 8
  call void @tl_tensor_free(ptr %field_load216)
  %field_gep217 = getelementptr inbounds %Block, ptr %old_field_val207, i32 0, i32 1
  %field_load218 = load ptr, ptr %field_gep217, align 8
  %field_gep219 = getelementptr inbounds %CausalSelfAttention, ptr %field_load218, i32 0, i32 0
  %field_load220 = load ptr, ptr %field_gep219, align 8
  %field_gep221 = getelementptr inbounds %Linear, ptr %field_load220, i32 0, i32 0
  %field_load222 = load ptr, ptr %field_gep221, align 8
  call void @tl_tensor_free(ptr %field_load222)
  %field_gep223 = getelementptr inbounds %Linear, ptr %field_load220, i32 0, i32 1
  %field_load224 = load ptr, ptr %field_gep223, align 8
  call void @tl_tensor_free(ptr %field_load224)
  %field_gep225 = getelementptr inbounds %CausalSelfAttention, ptr %field_load218, i32 0, i32 1
  %field_load226 = load ptr, ptr %field_gep225, align 8
  %field_gep227 = getelementptr inbounds %Linear, ptr %field_load226, i32 0, i32 0
  %field_load228 = load ptr, ptr %field_gep227, align 8
  call void @tl_tensor_free(ptr %field_load228)
  %field_gep229 = getelementptr inbounds %Linear, ptr %field_load226, i32 0, i32 1
  %field_load230 = load ptr, ptr %field_gep229, align 8
  call void @tl_tensor_free(ptr %field_load230)
  %field_gep231 = getelementptr inbounds %CausalSelfAttention, ptr %field_load218, i32 0, i32 2
  %field_load232 = load ptr, ptr %field_gep231, align 8
  %field_gep233 = getelementptr inbounds %Linear, ptr %field_load232, i32 0, i32 0
  %field_load234 = load ptr, ptr %field_gep233, align 8
  call void @tl_tensor_free(ptr %field_load234)
  %field_gep235 = getelementptr inbounds %Linear, ptr %field_load232, i32 0, i32 1
  %field_load236 = load ptr, ptr %field_gep235, align 8
  call void @tl_tensor_free(ptr %field_load236)
  %field_gep237 = getelementptr inbounds %CausalSelfAttention, ptr %field_load218, i32 0, i32 3
  %field_load238 = load ptr, ptr %field_gep237, align 8
  %field_gep239 = getelementptr inbounds %Linear, ptr %field_load238, i32 0, i32 0
  %field_load240 = load ptr, ptr %field_gep239, align 8
  call void @tl_tensor_free(ptr %field_load240)
  %field_gep241 = getelementptr inbounds %Linear, ptr %field_load238, i32 0, i32 1
  %field_load242 = load ptr, ptr %field_gep241, align 8
  call void @tl_tensor_free(ptr %field_load242)
  %field_gep243 = getelementptr inbounds %Block, ptr %old_field_val207, i32 0, i32 2
  %field_load244 = load ptr, ptr %field_gep243, align 8
  %field_gep245 = getelementptr inbounds %LayerNorm, ptr %field_load244, i32 0, i32 0
  %field_load246 = load ptr, ptr %field_gep245, align 8
  call void @tl_tensor_free(ptr %field_load246)
  %field_gep247 = getelementptr inbounds %LayerNorm, ptr %field_load244, i32 0, i32 1
  %field_load248 = load ptr, ptr %field_gep247, align 8
  call void @tl_tensor_free(ptr %field_load248)
  %field_gep249 = getelementptr inbounds %Block, ptr %old_field_val207, i32 0, i32 3
  %field_load250 = load ptr, ptr %field_gep249, align 8
  %field_gep251 = getelementptr inbounds %MLP, ptr %field_load250, i32 0, i32 0
  %field_load252 = load ptr, ptr %field_gep251, align 8
  %field_gep253 = getelementptr inbounds %Linear, ptr %field_load252, i32 0, i32 0
  %field_load254 = load ptr, ptr %field_gep253, align 8
  call void @tl_tensor_free(ptr %field_load254)
  %field_gep255 = getelementptr inbounds %Linear, ptr %field_load252, i32 0, i32 1
  %field_load256 = load ptr, ptr %field_gep255, align 8
  call void @tl_tensor_free(ptr %field_load256)
  %field_gep257 = getelementptr inbounds %MLP, ptr %field_load250, i32 0, i32 1
  %field_load258 = load ptr, ptr %field_gep257, align 8
  %field_gep259 = getelementptr inbounds %Linear, ptr %field_load258, i32 0, i32 0
  %field_load260 = load ptr, ptr %field_gep259, align 8
  call void @tl_tensor_free(ptr %field_load260)
  %field_gep261 = getelementptr inbounds %Linear, ptr %field_load258, i32 0, i32 1
  %field_load262 = load ptr, ptr %field_gep261, align 8
  call void @tl_tensor_free(ptr %field_load262)
  br label %skip_free210

skip_free210:                                     ; preds = %free_old_val209, %skip_free149
  store ptr %call_method206, ptr %ptr_b4, align 8
  call void @tl_mem_unregister(ptr %call_method206)
  %s263 = load ptr, ptr %s, align 8
  %ptr_l = getelementptr inbounds %GPTHeavy, ptr %s263, i32 0, i32 6
  %s264 = load ptr, ptr %s, align 8
  %ptr_l265 = getelementptr inbounds %GPTHeavy, ptr %s264, i32 0, i32 6
  %l = load ptr, ptr %ptr_l265, align 8
  %lr266 = load float, ptr %lr2, align 4
  %call_method267 = call ptr @tl_LayerNorm_step(ptr %l, float %lr266)
  call void @tl_mem_register_struct(ptr %call_method267)
  %old_field_val268 = load ptr, ptr %ptr_l, align 8
  %cnt_free_diff269 = icmp ne ptr %old_field_val268, %call_method267
  br i1 %cnt_free_diff269, label %free_old_val270, label %skip_free271

free_old_val270:                                  ; preds = %skip_free210
  %field_gep272 = getelementptr inbounds %LayerNorm, ptr %old_field_val268, i32 0, i32 0
  %field_load273 = load ptr, ptr %field_gep272, align 8
  call void @tl_tensor_free(ptr %field_load273)
  %field_gep274 = getelementptr inbounds %LayerNorm, ptr %old_field_val268, i32 0, i32 1
  %field_load275 = load ptr, ptr %field_gep274, align 8
  call void @tl_tensor_free(ptr %field_load275)
  br label %skip_free271

skip_free271:                                     ; preds = %free_old_val270, %skip_free210
  store ptr %call_method267, ptr %ptr_l, align 8
  call void @tl_mem_unregister(ptr %call_method267)
  %s276 = load ptr, ptr %s, align 8
  %ptr_h = getelementptr inbounds %GPTHeavy, ptr %s276, i32 0, i32 7
  %s277 = load ptr, ptr %s, align 8
  %ptr_h278 = getelementptr inbounds %GPTHeavy, ptr %s277, i32 0, i32 7
  %h = load ptr, ptr %ptr_h278, align 8
  %lr279 = load float, ptr %lr2, align 4
  %call_method280 = call ptr @tl_Linear_step(ptr %h, float %lr279)
  call void @tl_mem_register_struct(ptr %call_method280)
  %old_field_val281 = load ptr, ptr %ptr_h, align 8
  %cnt_free_diff282 = icmp ne ptr %old_field_val281, %call_method280
  br i1 %cnt_free_diff282, label %free_old_val283, label %skip_free284

free_old_val283:                                  ; preds = %skip_free271
  %field_gep285 = getelementptr inbounds %Linear, ptr %old_field_val281, i32 0, i32 0
  %field_load286 = load ptr, ptr %field_gep285, align 8
  call void @tl_tensor_free(ptr %field_load286)
  %field_gep287 = getelementptr inbounds %Linear, ptr %old_field_val281, i32 0, i32 1
  %field_load288 = load ptr, ptr %field_gep287, align 8
  call void @tl_tensor_free(ptr %field_load288)
  br label %skip_free284

skip_free284:                                     ; preds = %free_old_val283, %skip_free271
  store ptr %call_method280, ptr %ptr_h, align 8
  call void @tl_mem_unregister(ptr %call_method280)
  %s289 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s289)
  %unreg_field_0 = getelementptr inbounds %GPTHeavy, ptr %s289, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_0290 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val291 = load ptr, ptr %unreg_field_0290, align 8
  call void @tl_mem_unregister(ptr %field_val291)
  %unreg_field_1 = getelementptr inbounds %GPTHeavy, ptr %s289, i32 0, i32 1
  %field_val292 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val292)
  %unreg_field_0293 = getelementptr inbounds %Embedding, ptr %field_val292, i32 0, i32 0
  %field_val294 = load ptr, ptr %unreg_field_0293, align 8
  call void @tl_mem_unregister(ptr %field_val294)
  %unreg_field_2 = getelementptr inbounds %GPTHeavy, ptr %s289, i32 0, i32 2
  %field_val295 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val295)
  %unreg_field_0296 = getelementptr inbounds %Block, ptr %field_val295, i32 0, i32 0
  %field_val297 = load ptr, ptr %unreg_field_0296, align 8
  call void @tl_mem_unregister(ptr %field_val297)
  %unreg_field_0298 = getelementptr inbounds %LayerNorm, ptr %field_val297, i32 0, i32 0
  %field_val299 = load ptr, ptr %unreg_field_0298, align 8
  call void @tl_mem_unregister(ptr %field_val299)
  %unreg_field_1300 = getelementptr inbounds %LayerNorm, ptr %field_val297, i32 0, i32 1
  %field_val301 = load ptr, ptr %unreg_field_1300, align 8
  call void @tl_mem_unregister(ptr %field_val301)
  %unreg_field_1302 = getelementptr inbounds %Block, ptr %field_val295, i32 0, i32 1
  %field_val303 = load ptr, ptr %unreg_field_1302, align 8
  call void @tl_mem_unregister(ptr %field_val303)
  %unreg_field_0304 = getelementptr inbounds %CausalSelfAttention, ptr %field_val303, i32 0, i32 0
  %field_val305 = load ptr, ptr %unreg_field_0304, align 8
  call void @tl_mem_unregister(ptr %field_val305)
  %unreg_field_0306 = getelementptr inbounds %Linear, ptr %field_val305, i32 0, i32 0
  %field_val307 = load ptr, ptr %unreg_field_0306, align 8
  call void @tl_mem_unregister(ptr %field_val307)
  %unreg_field_1308 = getelementptr inbounds %Linear, ptr %field_val305, i32 0, i32 1
  %field_val309 = load ptr, ptr %unreg_field_1308, align 8
  call void @tl_mem_unregister(ptr %field_val309)
  %unreg_field_1310 = getelementptr inbounds %CausalSelfAttention, ptr %field_val303, i32 0, i32 1
  %field_val311 = load ptr, ptr %unreg_field_1310, align 8
  call void @tl_mem_unregister(ptr %field_val311)
  %unreg_field_0312 = getelementptr inbounds %Linear, ptr %field_val311, i32 0, i32 0
  %field_val313 = load ptr, ptr %unreg_field_0312, align 8
  call void @tl_mem_unregister(ptr %field_val313)
  %unreg_field_1314 = getelementptr inbounds %Linear, ptr %field_val311, i32 0, i32 1
  %field_val315 = load ptr, ptr %unreg_field_1314, align 8
  call void @tl_mem_unregister(ptr %field_val315)
  %unreg_field_2316 = getelementptr inbounds %CausalSelfAttention, ptr %field_val303, i32 0, i32 2
  %field_val317 = load ptr, ptr %unreg_field_2316, align 8
  call void @tl_mem_unregister(ptr %field_val317)
  %unreg_field_0318 = getelementptr inbounds %Linear, ptr %field_val317, i32 0, i32 0
  %field_val319 = load ptr, ptr %unreg_field_0318, align 8
  call void @tl_mem_unregister(ptr %field_val319)
  %unreg_field_1320 = getelementptr inbounds %Linear, ptr %field_val317, i32 0, i32 1
  %field_val321 = load ptr, ptr %unreg_field_1320, align 8
  call void @tl_mem_unregister(ptr %field_val321)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val303, i32 0, i32 3
  %field_val322 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val322)
  %unreg_field_0323 = getelementptr inbounds %Linear, ptr %field_val322, i32 0, i32 0
  %field_val324 = load ptr, ptr %unreg_field_0323, align 8
  call void @tl_mem_unregister(ptr %field_val324)
  %unreg_field_1325 = getelementptr inbounds %Linear, ptr %field_val322, i32 0, i32 1
  %field_val326 = load ptr, ptr %unreg_field_1325, align 8
  call void @tl_mem_unregister(ptr %field_val326)
  %unreg_field_2327 = getelementptr inbounds %Block, ptr %field_val295, i32 0, i32 2
  %field_val328 = load ptr, ptr %unreg_field_2327, align 8
  call void @tl_mem_unregister(ptr %field_val328)
  %unreg_field_0329 = getelementptr inbounds %LayerNorm, ptr %field_val328, i32 0, i32 0
  %field_val330 = load ptr, ptr %unreg_field_0329, align 8
  call void @tl_mem_unregister(ptr %field_val330)
  %unreg_field_1331 = getelementptr inbounds %LayerNorm, ptr %field_val328, i32 0, i32 1
  %field_val332 = load ptr, ptr %unreg_field_1331, align 8
  call void @tl_mem_unregister(ptr %field_val332)
  %unreg_field_3333 = getelementptr inbounds %Block, ptr %field_val295, i32 0, i32 3
  %field_val334 = load ptr, ptr %unreg_field_3333, align 8
  call void @tl_mem_unregister(ptr %field_val334)
  %unreg_field_0335 = getelementptr inbounds %MLP, ptr %field_val334, i32 0, i32 0
  %field_val336 = load ptr, ptr %unreg_field_0335, align 8
  call void @tl_mem_unregister(ptr %field_val336)
  %unreg_field_0337 = getelementptr inbounds %Linear, ptr %field_val336, i32 0, i32 0
  %field_val338 = load ptr, ptr %unreg_field_0337, align 8
  call void @tl_mem_unregister(ptr %field_val338)
  %unreg_field_1339 = getelementptr inbounds %Linear, ptr %field_val336, i32 0, i32 1
  %field_val340 = load ptr, ptr %unreg_field_1339, align 8
  call void @tl_mem_unregister(ptr %field_val340)
  %unreg_field_1341 = getelementptr inbounds %MLP, ptr %field_val334, i32 0, i32 1
  %field_val342 = load ptr, ptr %unreg_field_1341, align 8
  call void @tl_mem_unregister(ptr %field_val342)
  %unreg_field_0343 = getelementptr inbounds %Linear, ptr %field_val342, i32 0, i32 0
  %field_val344 = load ptr, ptr %unreg_field_0343, align 8
  call void @tl_mem_unregister(ptr %field_val344)
  %unreg_field_1345 = getelementptr inbounds %Linear, ptr %field_val342, i32 0, i32 1
  %field_val346 = load ptr, ptr %unreg_field_1345, align 8
  call void @tl_mem_unregister(ptr %field_val346)
  %unreg_field_3347 = getelementptr inbounds %GPTHeavy, ptr %s289, i32 0, i32 3
  %field_val348 = load ptr, ptr %unreg_field_3347, align 8
  call void @tl_mem_unregister(ptr %field_val348)
  %unreg_field_0349 = getelementptr inbounds %Block, ptr %field_val348, i32 0, i32 0
  %field_val350 = load ptr, ptr %unreg_field_0349, align 8
  call void @tl_mem_unregister(ptr %field_val350)
  %unreg_field_0351 = getelementptr inbounds %LayerNorm, ptr %field_val350, i32 0, i32 0
  %field_val352 = load ptr, ptr %unreg_field_0351, align 8
  call void @tl_mem_unregister(ptr %field_val352)
  %unreg_field_1353 = getelementptr inbounds %LayerNorm, ptr %field_val350, i32 0, i32 1
  %field_val354 = load ptr, ptr %unreg_field_1353, align 8
  call void @tl_mem_unregister(ptr %field_val354)
  %unreg_field_1355 = getelementptr inbounds %Block, ptr %field_val348, i32 0, i32 1
  %field_val356 = load ptr, ptr %unreg_field_1355, align 8
  call void @tl_mem_unregister(ptr %field_val356)
  %unreg_field_0357 = getelementptr inbounds %CausalSelfAttention, ptr %field_val356, i32 0, i32 0
  %field_val358 = load ptr, ptr %unreg_field_0357, align 8
  call void @tl_mem_unregister(ptr %field_val358)
  %unreg_field_0359 = getelementptr inbounds %Linear, ptr %field_val358, i32 0, i32 0
  %field_val360 = load ptr, ptr %unreg_field_0359, align 8
  call void @tl_mem_unregister(ptr %field_val360)
  %unreg_field_1361 = getelementptr inbounds %Linear, ptr %field_val358, i32 0, i32 1
  %field_val362 = load ptr, ptr %unreg_field_1361, align 8
  call void @tl_mem_unregister(ptr %field_val362)
  %unreg_field_1363 = getelementptr inbounds %CausalSelfAttention, ptr %field_val356, i32 0, i32 1
  %field_val364 = load ptr, ptr %unreg_field_1363, align 8
  call void @tl_mem_unregister(ptr %field_val364)
  %unreg_field_0365 = getelementptr inbounds %Linear, ptr %field_val364, i32 0, i32 0
  %field_val366 = load ptr, ptr %unreg_field_0365, align 8
  call void @tl_mem_unregister(ptr %field_val366)
  %unreg_field_1367 = getelementptr inbounds %Linear, ptr %field_val364, i32 0, i32 1
  %field_val368 = load ptr, ptr %unreg_field_1367, align 8
  call void @tl_mem_unregister(ptr %field_val368)
  %unreg_field_2369 = getelementptr inbounds %CausalSelfAttention, ptr %field_val356, i32 0, i32 2
  %field_val370 = load ptr, ptr %unreg_field_2369, align 8
  call void @tl_mem_unregister(ptr %field_val370)
  %unreg_field_0371 = getelementptr inbounds %Linear, ptr %field_val370, i32 0, i32 0
  %field_val372 = load ptr, ptr %unreg_field_0371, align 8
  call void @tl_mem_unregister(ptr %field_val372)
  %unreg_field_1373 = getelementptr inbounds %Linear, ptr %field_val370, i32 0, i32 1
  %field_val374 = load ptr, ptr %unreg_field_1373, align 8
  call void @tl_mem_unregister(ptr %field_val374)
  %unreg_field_3375 = getelementptr inbounds %CausalSelfAttention, ptr %field_val356, i32 0, i32 3
  %field_val376 = load ptr, ptr %unreg_field_3375, align 8
  call void @tl_mem_unregister(ptr %field_val376)
  %unreg_field_0377 = getelementptr inbounds %Linear, ptr %field_val376, i32 0, i32 0
  %field_val378 = load ptr, ptr %unreg_field_0377, align 8
  call void @tl_mem_unregister(ptr %field_val378)
  %unreg_field_1379 = getelementptr inbounds %Linear, ptr %field_val376, i32 0, i32 1
  %field_val380 = load ptr, ptr %unreg_field_1379, align 8
  call void @tl_mem_unregister(ptr %field_val380)
  %unreg_field_2381 = getelementptr inbounds %Block, ptr %field_val348, i32 0, i32 2
  %field_val382 = load ptr, ptr %unreg_field_2381, align 8
  call void @tl_mem_unregister(ptr %field_val382)
  %unreg_field_0383 = getelementptr inbounds %LayerNorm, ptr %field_val382, i32 0, i32 0
  %field_val384 = load ptr, ptr %unreg_field_0383, align 8
  call void @tl_mem_unregister(ptr %field_val384)
  %unreg_field_1385 = getelementptr inbounds %LayerNorm, ptr %field_val382, i32 0, i32 1
  %field_val386 = load ptr, ptr %unreg_field_1385, align 8
  call void @tl_mem_unregister(ptr %field_val386)
  %unreg_field_3387 = getelementptr inbounds %Block, ptr %field_val348, i32 0, i32 3
  %field_val388 = load ptr, ptr %unreg_field_3387, align 8
  call void @tl_mem_unregister(ptr %field_val388)
  %unreg_field_0389 = getelementptr inbounds %MLP, ptr %field_val388, i32 0, i32 0
  %field_val390 = load ptr, ptr %unreg_field_0389, align 8
  call void @tl_mem_unregister(ptr %field_val390)
  %unreg_field_0391 = getelementptr inbounds %Linear, ptr %field_val390, i32 0, i32 0
  %field_val392 = load ptr, ptr %unreg_field_0391, align 8
  call void @tl_mem_unregister(ptr %field_val392)
  %unreg_field_1393 = getelementptr inbounds %Linear, ptr %field_val390, i32 0, i32 1
  %field_val394 = load ptr, ptr %unreg_field_1393, align 8
  call void @tl_mem_unregister(ptr %field_val394)
  %unreg_field_1395 = getelementptr inbounds %MLP, ptr %field_val388, i32 0, i32 1
  %field_val396 = load ptr, ptr %unreg_field_1395, align 8
  call void @tl_mem_unregister(ptr %field_val396)
  %unreg_field_0397 = getelementptr inbounds %Linear, ptr %field_val396, i32 0, i32 0
  %field_val398 = load ptr, ptr %unreg_field_0397, align 8
  call void @tl_mem_unregister(ptr %field_val398)
  %unreg_field_1399 = getelementptr inbounds %Linear, ptr %field_val396, i32 0, i32 1
  %field_val400 = load ptr, ptr %unreg_field_1399, align 8
  call void @tl_mem_unregister(ptr %field_val400)
  %unreg_field_4 = getelementptr inbounds %GPTHeavy, ptr %s289, i32 0, i32 4
  %field_val401 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val401)
  %unreg_field_0402 = getelementptr inbounds %Block, ptr %field_val401, i32 0, i32 0
  %field_val403 = load ptr, ptr %unreg_field_0402, align 8
  call void @tl_mem_unregister(ptr %field_val403)
  %unreg_field_0404 = getelementptr inbounds %LayerNorm, ptr %field_val403, i32 0, i32 0
  %field_val405 = load ptr, ptr %unreg_field_0404, align 8
  call void @tl_mem_unregister(ptr %field_val405)
  %unreg_field_1406 = getelementptr inbounds %LayerNorm, ptr %field_val403, i32 0, i32 1
  %field_val407 = load ptr, ptr %unreg_field_1406, align 8
  call void @tl_mem_unregister(ptr %field_val407)
  %unreg_field_1408 = getelementptr inbounds %Block, ptr %field_val401, i32 0, i32 1
  %field_val409 = load ptr, ptr %unreg_field_1408, align 8
  call void @tl_mem_unregister(ptr %field_val409)
  %unreg_field_0410 = getelementptr inbounds %CausalSelfAttention, ptr %field_val409, i32 0, i32 0
  %field_val411 = load ptr, ptr %unreg_field_0410, align 8
  call void @tl_mem_unregister(ptr %field_val411)
  %unreg_field_0412 = getelementptr inbounds %Linear, ptr %field_val411, i32 0, i32 0
  %field_val413 = load ptr, ptr %unreg_field_0412, align 8
  call void @tl_mem_unregister(ptr %field_val413)
  %unreg_field_1414 = getelementptr inbounds %Linear, ptr %field_val411, i32 0, i32 1
  %field_val415 = load ptr, ptr %unreg_field_1414, align 8
  call void @tl_mem_unregister(ptr %field_val415)
  %unreg_field_1416 = getelementptr inbounds %CausalSelfAttention, ptr %field_val409, i32 0, i32 1
  %field_val417 = load ptr, ptr %unreg_field_1416, align 8
  call void @tl_mem_unregister(ptr %field_val417)
  %unreg_field_0418 = getelementptr inbounds %Linear, ptr %field_val417, i32 0, i32 0
  %field_val419 = load ptr, ptr %unreg_field_0418, align 8
  call void @tl_mem_unregister(ptr %field_val419)
  %unreg_field_1420 = getelementptr inbounds %Linear, ptr %field_val417, i32 0, i32 1
  %field_val421 = load ptr, ptr %unreg_field_1420, align 8
  call void @tl_mem_unregister(ptr %field_val421)
  %unreg_field_2422 = getelementptr inbounds %CausalSelfAttention, ptr %field_val409, i32 0, i32 2
  %field_val423 = load ptr, ptr %unreg_field_2422, align 8
  call void @tl_mem_unregister(ptr %field_val423)
  %unreg_field_0424 = getelementptr inbounds %Linear, ptr %field_val423, i32 0, i32 0
  %field_val425 = load ptr, ptr %unreg_field_0424, align 8
  call void @tl_mem_unregister(ptr %field_val425)
  %unreg_field_1426 = getelementptr inbounds %Linear, ptr %field_val423, i32 0, i32 1
  %field_val427 = load ptr, ptr %unreg_field_1426, align 8
  call void @tl_mem_unregister(ptr %field_val427)
  %unreg_field_3428 = getelementptr inbounds %CausalSelfAttention, ptr %field_val409, i32 0, i32 3
  %field_val429 = load ptr, ptr %unreg_field_3428, align 8
  call void @tl_mem_unregister(ptr %field_val429)
  %unreg_field_0430 = getelementptr inbounds %Linear, ptr %field_val429, i32 0, i32 0
  %field_val431 = load ptr, ptr %unreg_field_0430, align 8
  call void @tl_mem_unregister(ptr %field_val431)
  %unreg_field_1432 = getelementptr inbounds %Linear, ptr %field_val429, i32 0, i32 1
  %field_val433 = load ptr, ptr %unreg_field_1432, align 8
  call void @tl_mem_unregister(ptr %field_val433)
  %unreg_field_2434 = getelementptr inbounds %Block, ptr %field_val401, i32 0, i32 2
  %field_val435 = load ptr, ptr %unreg_field_2434, align 8
  call void @tl_mem_unregister(ptr %field_val435)
  %unreg_field_0436 = getelementptr inbounds %LayerNorm, ptr %field_val435, i32 0, i32 0
  %field_val437 = load ptr, ptr %unreg_field_0436, align 8
  call void @tl_mem_unregister(ptr %field_val437)
  %unreg_field_1438 = getelementptr inbounds %LayerNorm, ptr %field_val435, i32 0, i32 1
  %field_val439 = load ptr, ptr %unreg_field_1438, align 8
  call void @tl_mem_unregister(ptr %field_val439)
  %unreg_field_3440 = getelementptr inbounds %Block, ptr %field_val401, i32 0, i32 3
  %field_val441 = load ptr, ptr %unreg_field_3440, align 8
  call void @tl_mem_unregister(ptr %field_val441)
  %unreg_field_0442 = getelementptr inbounds %MLP, ptr %field_val441, i32 0, i32 0
  %field_val443 = load ptr, ptr %unreg_field_0442, align 8
  call void @tl_mem_unregister(ptr %field_val443)
  %unreg_field_0444 = getelementptr inbounds %Linear, ptr %field_val443, i32 0, i32 0
  %field_val445 = load ptr, ptr %unreg_field_0444, align 8
  call void @tl_mem_unregister(ptr %field_val445)
  %unreg_field_1446 = getelementptr inbounds %Linear, ptr %field_val443, i32 0, i32 1
  %field_val447 = load ptr, ptr %unreg_field_1446, align 8
  call void @tl_mem_unregister(ptr %field_val447)
  %unreg_field_1448 = getelementptr inbounds %MLP, ptr %field_val441, i32 0, i32 1
  %field_val449 = load ptr, ptr %unreg_field_1448, align 8
  call void @tl_mem_unregister(ptr %field_val449)
  %unreg_field_0450 = getelementptr inbounds %Linear, ptr %field_val449, i32 0, i32 0
  %field_val451 = load ptr, ptr %unreg_field_0450, align 8
  call void @tl_mem_unregister(ptr %field_val451)
  %unreg_field_1452 = getelementptr inbounds %Linear, ptr %field_val449, i32 0, i32 1
  %field_val453 = load ptr, ptr %unreg_field_1452, align 8
  call void @tl_mem_unregister(ptr %field_val453)
  %unreg_field_5 = getelementptr inbounds %GPTHeavy, ptr %s289, i32 0, i32 5
  %field_val454 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val454)
  %unreg_field_0455 = getelementptr inbounds %Block, ptr %field_val454, i32 0, i32 0
  %field_val456 = load ptr, ptr %unreg_field_0455, align 8
  call void @tl_mem_unregister(ptr %field_val456)
  %unreg_field_0457 = getelementptr inbounds %LayerNorm, ptr %field_val456, i32 0, i32 0
  %field_val458 = load ptr, ptr %unreg_field_0457, align 8
  call void @tl_mem_unregister(ptr %field_val458)
  %unreg_field_1459 = getelementptr inbounds %LayerNorm, ptr %field_val456, i32 0, i32 1
  %field_val460 = load ptr, ptr %unreg_field_1459, align 8
  call void @tl_mem_unregister(ptr %field_val460)
  %unreg_field_1461 = getelementptr inbounds %Block, ptr %field_val454, i32 0, i32 1
  %field_val462 = load ptr, ptr %unreg_field_1461, align 8
  call void @tl_mem_unregister(ptr %field_val462)
  %unreg_field_0463 = getelementptr inbounds %CausalSelfAttention, ptr %field_val462, i32 0, i32 0
  %field_val464 = load ptr, ptr %unreg_field_0463, align 8
  call void @tl_mem_unregister(ptr %field_val464)
  %unreg_field_0465 = getelementptr inbounds %Linear, ptr %field_val464, i32 0, i32 0
  %field_val466 = load ptr, ptr %unreg_field_0465, align 8
  call void @tl_mem_unregister(ptr %field_val466)
  %unreg_field_1467 = getelementptr inbounds %Linear, ptr %field_val464, i32 0, i32 1
  %field_val468 = load ptr, ptr %unreg_field_1467, align 8
  call void @tl_mem_unregister(ptr %field_val468)
  %unreg_field_1469 = getelementptr inbounds %CausalSelfAttention, ptr %field_val462, i32 0, i32 1
  %field_val470 = load ptr, ptr %unreg_field_1469, align 8
  call void @tl_mem_unregister(ptr %field_val470)
  %unreg_field_0471 = getelementptr inbounds %Linear, ptr %field_val470, i32 0, i32 0
  %field_val472 = load ptr, ptr %unreg_field_0471, align 8
  call void @tl_mem_unregister(ptr %field_val472)
  %unreg_field_1473 = getelementptr inbounds %Linear, ptr %field_val470, i32 0, i32 1
  %field_val474 = load ptr, ptr %unreg_field_1473, align 8
  call void @tl_mem_unregister(ptr %field_val474)
  %unreg_field_2475 = getelementptr inbounds %CausalSelfAttention, ptr %field_val462, i32 0, i32 2
  %field_val476 = load ptr, ptr %unreg_field_2475, align 8
  call void @tl_mem_unregister(ptr %field_val476)
  %unreg_field_0477 = getelementptr inbounds %Linear, ptr %field_val476, i32 0, i32 0
  %field_val478 = load ptr, ptr %unreg_field_0477, align 8
  call void @tl_mem_unregister(ptr %field_val478)
  %unreg_field_1479 = getelementptr inbounds %Linear, ptr %field_val476, i32 0, i32 1
  %field_val480 = load ptr, ptr %unreg_field_1479, align 8
  call void @tl_mem_unregister(ptr %field_val480)
  %unreg_field_3481 = getelementptr inbounds %CausalSelfAttention, ptr %field_val462, i32 0, i32 3
  %field_val482 = load ptr, ptr %unreg_field_3481, align 8
  call void @tl_mem_unregister(ptr %field_val482)
  %unreg_field_0483 = getelementptr inbounds %Linear, ptr %field_val482, i32 0, i32 0
  %field_val484 = load ptr, ptr %unreg_field_0483, align 8
  call void @tl_mem_unregister(ptr %field_val484)
  %unreg_field_1485 = getelementptr inbounds %Linear, ptr %field_val482, i32 0, i32 1
  %field_val486 = load ptr, ptr %unreg_field_1485, align 8
  call void @tl_mem_unregister(ptr %field_val486)
  %unreg_field_2487 = getelementptr inbounds %Block, ptr %field_val454, i32 0, i32 2
  %field_val488 = load ptr, ptr %unreg_field_2487, align 8
  call void @tl_mem_unregister(ptr %field_val488)
  %unreg_field_0489 = getelementptr inbounds %LayerNorm, ptr %field_val488, i32 0, i32 0
  %field_val490 = load ptr, ptr %unreg_field_0489, align 8
  call void @tl_mem_unregister(ptr %field_val490)
  %unreg_field_1491 = getelementptr inbounds %LayerNorm, ptr %field_val488, i32 0, i32 1
  %field_val492 = load ptr, ptr %unreg_field_1491, align 8
  call void @tl_mem_unregister(ptr %field_val492)
  %unreg_field_3493 = getelementptr inbounds %Block, ptr %field_val454, i32 0, i32 3
  %field_val494 = load ptr, ptr %unreg_field_3493, align 8
  call void @tl_mem_unregister(ptr %field_val494)
  %unreg_field_0495 = getelementptr inbounds %MLP, ptr %field_val494, i32 0, i32 0
  %field_val496 = load ptr, ptr %unreg_field_0495, align 8
  call void @tl_mem_unregister(ptr %field_val496)
  %unreg_field_0497 = getelementptr inbounds %Linear, ptr %field_val496, i32 0, i32 0
  %field_val498 = load ptr, ptr %unreg_field_0497, align 8
  call void @tl_mem_unregister(ptr %field_val498)
  %unreg_field_1499 = getelementptr inbounds %Linear, ptr %field_val496, i32 0, i32 1
  %field_val500 = load ptr, ptr %unreg_field_1499, align 8
  call void @tl_mem_unregister(ptr %field_val500)
  %unreg_field_1501 = getelementptr inbounds %MLP, ptr %field_val494, i32 0, i32 1
  %field_val502 = load ptr, ptr %unreg_field_1501, align 8
  call void @tl_mem_unregister(ptr %field_val502)
  %unreg_field_0503 = getelementptr inbounds %Linear, ptr %field_val502, i32 0, i32 0
  %field_val504 = load ptr, ptr %unreg_field_0503, align 8
  call void @tl_mem_unregister(ptr %field_val504)
  %unreg_field_1505 = getelementptr inbounds %Linear, ptr %field_val502, i32 0, i32 1
  %field_val506 = load ptr, ptr %unreg_field_1505, align 8
  call void @tl_mem_unregister(ptr %field_val506)
  %unreg_field_6 = getelementptr inbounds %GPTHeavy, ptr %s289, i32 0, i32 6
  %field_val507 = load ptr, ptr %unreg_field_6, align 8
  call void @tl_mem_unregister(ptr %field_val507)
  %unreg_field_0508 = getelementptr inbounds %LayerNorm, ptr %field_val507, i32 0, i32 0
  %field_val509 = load ptr, ptr %unreg_field_0508, align 8
  call void @tl_mem_unregister(ptr %field_val509)
  %unreg_field_1510 = getelementptr inbounds %LayerNorm, ptr %field_val507, i32 0, i32 1
  %field_val511 = load ptr, ptr %unreg_field_1510, align 8
  call void @tl_mem_unregister(ptr %field_val511)
  %unreg_field_7 = getelementptr inbounds %GPTHeavy, ptr %s289, i32 0, i32 7
  %field_val512 = load ptr, ptr %unreg_field_7, align 8
  call void @tl_mem_unregister(ptr %field_val512)
  %unreg_field_0513 = getelementptr inbounds %Linear, ptr %field_val512, i32 0, i32 0
  %field_val514 = load ptr, ptr %unreg_field_0513, align 8
  call void @tl_mem_unregister(ptr %field_val514)
  %unreg_field_1515 = getelementptr inbounds %Linear, ptr %field_val512, i32 0, i32 1
  %field_val516 = load ptr, ptr %unreg_field_1515, align 8
  call void @tl_mem_unregister(ptr %field_val516)
  call void @tl_mem_exit_scope()
  ret ptr %s289
}

define ptr @train_step(ptr %model, float %lr, i64 %i, i64 %j) {
entry:
  %loss = alloca ptr, align 16
  %Y_flat = alloca ptr, align 16
  %dims_alloca148 = alloca [1 x i64], align 8
  %logits_flat = alloca ptr, align 16
  %dims_alloca142 = alloca [2 x i64], align 8
  %logits = alloca ptr, align 16
  %Y = alloca ptr, align 16
  %dims_alloca134 = alloca [2 x i64], align 8
  %X = alloca ptr, align 16
  %dims_alloca = alloca [2 x i64], align 8
  %target = alloca ptr, align 16
  %data = alloca ptr, align 16
  %v_s_d100 = alloca float, align 16
  %scalar_shape88 = alloca i64, align 16
  %scalar_data87 = alloca float, align 16
  %scalar_shape85 = alloca i64, align 16
  %scalar_data83 = alloca float, align 16
  %v_s_d10 = alloca float, align 16
  %scalar_shape78 = alloca i64, align 16
  %scalar_data77 = alloca float, align 16
  %scalar_shape75 = alloca i64, align 16
  %scalar_data73 = alloca float, align 16
  %v_s_d1 = alloca float, align 16
  %scalar_shape68 = alloca i64, align 16
  %scalar_data67 = alloca float, align 16
  %scalar_shape65 = alloca i64, align 16
  %scalar_data63 = alloca float, align 16
  %v_j_d10 = alloca float, align 16
  %scalar_shape58 = alloca i64, align 16
  %scalar_data57 = alloca float, align 16
  %scalar_shape55 = alloca i64, align 16
  %scalar_data53 = alloca float, align 16
  %v_j_d1 = alloca float, align 16
  %scalar_shape48 = alloca i64, align 16
  %scalar_data47 = alloca float, align 16
  %scalar_shape45 = alloca i64, align 16
  %scalar_data43 = alloca float, align 16
  %v_i_d10 = alloca float, align 16
  %scalar_shape38 = alloca i64, align 16
  %scalar_data37 = alloca float, align 16
  %scalar_shape35 = alloca i64, align 16
  %scalar_data33 = alloca float, align 16
  %v_i_d1 = alloca float, align 16
  %scalar_shape30 = alloca i64, align 16
  %scalar_data29 = alloca float, align 16
  %scalar_shape = alloca i64, align 16
  %scalar_data = alloca float, align 16
  %s_d1 = alloca i64, align 16
  %s_d10 = alloca i64, align 16
  %rem = alloca i64, align 16
  %s_d100 = alloca i64, align 16
  %j_d1 = alloca i64, align 16
  %j_d10 = alloca i64, align 16
  %i_d1 = alloca i64, align 16
  %i_d10 = alloca i64, align 16
  %sum = alloca i64, align 16
  %j4 = alloca i64, align 16
  %i3 = alloca i64, align 16
  %lr2 = alloca float, align 16
  %model1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %model, ptr %model1, align 8
  store float %lr, ptr %lr2, align 4
  store i64 %i, ptr %i3, align 8
  store i64 %j, ptr %j4, align 8
  %i5 = load i64, ptr %i3, align 8
  %j6 = load i64, ptr %j4, align 8
  %addtmp = add i64 %i5, %j6
  store i64 %addtmp, ptr %sum, align 8
  %i7 = load i64, ptr %i3, align 8
  %divtmp = sdiv i64 %i7, 10
  store i64 %divtmp, ptr %i_d10, align 8
  %i8 = load i64, ptr %i3, align 8
  %i_d109 = load i64, ptr %i_d10, align 8
  %multmp = mul i64 %i_d109, 10
  %subtmp = sub i64 %i8, %multmp
  store i64 %subtmp, ptr %i_d1, align 8
  %j10 = load i64, ptr %j4, align 8
  %divtmp11 = sdiv i64 %j10, 10
  store i64 %divtmp11, ptr %j_d10, align 8
  %j12 = load i64, ptr %j4, align 8
  %j_d1013 = load i64, ptr %j_d10, align 8
  %multmp14 = mul i64 %j_d1013, 10
  %subtmp15 = sub i64 %j12, %multmp14
  store i64 %subtmp15, ptr %j_d1, align 8
  %sum16 = load i64, ptr %sum, align 8
  %divtmp17 = sdiv i64 %sum16, 100
  store i64 %divtmp17, ptr %s_d100, align 8
  %sum18 = load i64, ptr %sum, align 8
  %s_d10019 = load i64, ptr %s_d100, align 8
  %multmp20 = mul i64 %s_d10019, 100
  %subtmp21 = sub i64 %sum18, %multmp20
  store i64 %subtmp21, ptr %rem, align 8
  %rem22 = load i64, ptr %rem, align 8
  %divtmp23 = sdiv i64 %rem22, 10
  store i64 %divtmp23, ptr %s_d10, align 8
  %rem24 = load i64, ptr %rem, align 8
  %s_d1025 = load i64, ptr %s_d10, align 8
  %multmp26 = mul i64 %s_d1025, 10
  %subtmp27 = sub i64 %rem24, %multmp26
  store i64 %subtmp27, ptr %s_d1, align 8
  %i_d128 = load i64, ptr %i_d1, align 8
  %cast_i64_f32 = sitofp i64 %i_d128 to float
  store float %cast_i64_f32, ptr %scalar_data, align 4
  %scalar_tensor = call ptr @tl_tensor_new(ptr %scalar_data, i64 0, ptr %scalar_shape)
  store float 1.000000e+00, ptr %scalar_data29, align 4
  %scalar_tensor31 = call ptr @tl_tensor_new(ptr %scalar_data29, i64 0, ptr %scalar_shape30)
  %pow_res = call ptr @tl_tensor_pow(ptr %scalar_tensor, ptr %scalar_tensor31)
  call void @tl_tensor_free(ptr %scalar_tensor31)
  %get_res = call float @tl_tensor_get(ptr %pow_res, i64 0)
  call void @tl_tensor_free(ptr %pow_res)
  store float %get_res, ptr %v_i_d1, align 4
  %i_d1032 = load i64, ptr %i_d10, align 8
  %cast_i64_f3234 = sitofp i64 %i_d1032 to float
  store float %cast_i64_f3234, ptr %scalar_data33, align 4
  %scalar_tensor36 = call ptr @tl_tensor_new(ptr %scalar_data33, i64 0, ptr %scalar_shape35)
  store float 1.000000e+00, ptr %scalar_data37, align 4
  %scalar_tensor39 = call ptr @tl_tensor_new(ptr %scalar_data37, i64 0, ptr %scalar_shape38)
  %pow_res40 = call ptr @tl_tensor_pow(ptr %scalar_tensor36, ptr %scalar_tensor39)
  call void @tl_tensor_free(ptr %scalar_tensor39)
  %get_res41 = call float @tl_tensor_get(ptr %pow_res40, i64 0)
  call void @tl_tensor_free(ptr %pow_res40)
  store float %get_res41, ptr %v_i_d10, align 4
  %j_d142 = load i64, ptr %j_d1, align 8
  %cast_i64_f3244 = sitofp i64 %j_d142 to float
  store float %cast_i64_f3244, ptr %scalar_data43, align 4
  %scalar_tensor46 = call ptr @tl_tensor_new(ptr %scalar_data43, i64 0, ptr %scalar_shape45)
  store float 1.000000e+00, ptr %scalar_data47, align 4
  %scalar_tensor49 = call ptr @tl_tensor_new(ptr %scalar_data47, i64 0, ptr %scalar_shape48)
  %pow_res50 = call ptr @tl_tensor_pow(ptr %scalar_tensor46, ptr %scalar_tensor49)
  call void @tl_tensor_free(ptr %scalar_tensor49)
  %get_res51 = call float @tl_tensor_get(ptr %pow_res50, i64 0)
  call void @tl_tensor_free(ptr %pow_res50)
  store float %get_res51, ptr %v_j_d1, align 4
  %j_d1052 = load i64, ptr %j_d10, align 8
  %cast_i64_f3254 = sitofp i64 %j_d1052 to float
  store float %cast_i64_f3254, ptr %scalar_data53, align 4
  %scalar_tensor56 = call ptr @tl_tensor_new(ptr %scalar_data53, i64 0, ptr %scalar_shape55)
  store float 1.000000e+00, ptr %scalar_data57, align 4
  %scalar_tensor59 = call ptr @tl_tensor_new(ptr %scalar_data57, i64 0, ptr %scalar_shape58)
  %pow_res60 = call ptr @tl_tensor_pow(ptr %scalar_tensor56, ptr %scalar_tensor59)
  call void @tl_tensor_free(ptr %scalar_tensor59)
  %get_res61 = call float @tl_tensor_get(ptr %pow_res60, i64 0)
  call void @tl_tensor_free(ptr %pow_res60)
  store float %get_res61, ptr %v_j_d10, align 4
  %s_d162 = load i64, ptr %s_d1, align 8
  %cast_i64_f3264 = sitofp i64 %s_d162 to float
  store float %cast_i64_f3264, ptr %scalar_data63, align 4
  %scalar_tensor66 = call ptr @tl_tensor_new(ptr %scalar_data63, i64 0, ptr %scalar_shape65)
  store float 1.000000e+00, ptr %scalar_data67, align 4
  %scalar_tensor69 = call ptr @tl_tensor_new(ptr %scalar_data67, i64 0, ptr %scalar_shape68)
  %pow_res70 = call ptr @tl_tensor_pow(ptr %scalar_tensor66, ptr %scalar_tensor69)
  call void @tl_tensor_free(ptr %scalar_tensor69)
  %get_res71 = call float @tl_tensor_get(ptr %pow_res70, i64 0)
  call void @tl_tensor_free(ptr %pow_res70)
  store float %get_res71, ptr %v_s_d1, align 4
  %s_d1072 = load i64, ptr %s_d10, align 8
  %cast_i64_f3274 = sitofp i64 %s_d1072 to float
  store float %cast_i64_f3274, ptr %scalar_data73, align 4
  %scalar_tensor76 = call ptr @tl_tensor_new(ptr %scalar_data73, i64 0, ptr %scalar_shape75)
  store float 1.000000e+00, ptr %scalar_data77, align 4
  %scalar_tensor79 = call ptr @tl_tensor_new(ptr %scalar_data77, i64 0, ptr %scalar_shape78)
  %pow_res80 = call ptr @tl_tensor_pow(ptr %scalar_tensor76, ptr %scalar_tensor79)
  call void @tl_tensor_free(ptr %scalar_tensor79)
  %get_res81 = call float @tl_tensor_get(ptr %pow_res80, i64 0)
  call void @tl_tensor_free(ptr %pow_res80)
  store float %get_res81, ptr %v_s_d10, align 4
  %s_d10082 = load i64, ptr %s_d100, align 8
  %cast_i64_f3284 = sitofp i64 %s_d10082 to float
  store float %cast_i64_f3284, ptr %scalar_data83, align 4
  %scalar_tensor86 = call ptr @tl_tensor_new(ptr %scalar_data83, i64 0, ptr %scalar_shape85)
  store float 1.000000e+00, ptr %scalar_data87, align 4
  %scalar_tensor89 = call ptr @tl_tensor_new(ptr %scalar_data87, i64 0, ptr %scalar_shape88)
  %pow_res90 = call ptr @tl_tensor_pow(ptr %scalar_tensor86, ptr %scalar_tensor89)
  call void @tl_tensor_free(ptr %scalar_tensor89)
  %get_res91 = call float @tl_tensor_get(ptr %pow_res90, i64 0)
  call void @tl_tensor_free(ptr %pow_res90)
  store float %get_res91, ptr %v_s_d100, align 4
  %buf_void = call ptr @tl_alloc_tmp(i64 48)
  %v_i_d192 = load float, ptr %v_i_d1, align 4
  %elem_ptr = getelementptr inbounds float, ptr %buf_void, i64 0
  store float %v_i_d192, ptr %elem_ptr, align 4
  %v_i_d1093 = load float, ptr %v_i_d10, align 4
  %elem_ptr94 = getelementptr inbounds float, ptr %buf_void, i64 1
  store float %v_i_d1093, ptr %elem_ptr94, align 4
  %elem_ptr95 = getelementptr inbounds float, ptr %buf_void, i64 2
  store float 1.000000e+01, ptr %elem_ptr95, align 4
  %v_j_d196 = load float, ptr %v_j_d1, align 4
  %elem_ptr97 = getelementptr inbounds float, ptr %buf_void, i64 3
  store float %v_j_d196, ptr %elem_ptr97, align 4
  %v_j_d1098 = load float, ptr %v_j_d10, align 4
  %elem_ptr99 = getelementptr inbounds float, ptr %buf_void, i64 4
  store float %v_j_d1098, ptr %elem_ptr99, align 4
  %elem_ptr100 = getelementptr inbounds float, ptr %buf_void, i64 5
  store float 1.100000e+01, ptr %elem_ptr100, align 4
  %v_s_d1101 = load float, ptr %v_s_d1, align 4
  %elem_ptr102 = getelementptr inbounds float, ptr %buf_void, i64 6
  store float %v_s_d1101, ptr %elem_ptr102, align 4
  %v_s_d10103 = load float, ptr %v_s_d10, align 4
  %elem_ptr104 = getelementptr inbounds float, ptr %buf_void, i64 7
  store float %v_s_d10103, ptr %elem_ptr104, align 4
  %v_s_d100105 = load float, ptr %v_s_d100, align 4
  %elem_ptr106 = getelementptr inbounds float, ptr %buf_void, i64 8
  store float %v_s_d100105, ptr %elem_ptr106, align 4
  %elem_ptr107 = getelementptr inbounds float, ptr %buf_void, i64 9
  store float 1.200000e+01, ptr %elem_ptr107, align 4
  %elem_ptr108 = getelementptr inbounds float, ptr %buf_void, i64 10
  store float 1.200000e+01, ptr %elem_ptr108, align 4
  %elem_ptr109 = getelementptr inbounds float, ptr %buf_void, i64 11
  store float 1.200000e+01, ptr %elem_ptr109, align 4
  %shape_alloc = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr = getelementptr inbounds i64, ptr %shape_alloc, i64 0
  store i64 12, ptr %shape_ptr, align 8
  %new_tensor = call ptr @tl_tensor_new(ptr %buf_void, i64 1, ptr %shape_alloc)
  call void @tl_free_tmp(ptr %buf_void)
  call void @tl_free_tmp(ptr %shape_alloc)
  call void @tl_mem_unregister(ptr %new_tensor)
  store ptr %new_tensor, ptr %data, align 8
  %buf_void110 = call ptr @tl_alloc_tmp(i64 48)
  %v_i_d10111 = load float, ptr %v_i_d10, align 4
  %elem_ptr112 = getelementptr inbounds float, ptr %buf_void110, i64 0
  store float %v_i_d10111, ptr %elem_ptr112, align 4
  %elem_ptr113 = getelementptr inbounds float, ptr %buf_void110, i64 1
  store float 1.000000e+01, ptr %elem_ptr113, align 4
  %v_j_d1114 = load float, ptr %v_j_d1, align 4
  %elem_ptr115 = getelementptr inbounds float, ptr %buf_void110, i64 2
  store float %v_j_d1114, ptr %elem_ptr115, align 4
  %v_j_d10116 = load float, ptr %v_j_d10, align 4
  %elem_ptr117 = getelementptr inbounds float, ptr %buf_void110, i64 3
  store float %v_j_d10116, ptr %elem_ptr117, align 4
  %elem_ptr118 = getelementptr inbounds float, ptr %buf_void110, i64 4
  store float 1.100000e+01, ptr %elem_ptr118, align 4
  %v_s_d1119 = load float, ptr %v_s_d1, align 4
  %elem_ptr120 = getelementptr inbounds float, ptr %buf_void110, i64 5
  store float %v_s_d1119, ptr %elem_ptr120, align 4
  %v_s_d10121 = load float, ptr %v_s_d10, align 4
  %elem_ptr122 = getelementptr inbounds float, ptr %buf_void110, i64 6
  store float %v_s_d10121, ptr %elem_ptr122, align 4
  %v_s_d100123 = load float, ptr %v_s_d100, align 4
  %elem_ptr124 = getelementptr inbounds float, ptr %buf_void110, i64 7
  store float %v_s_d100123, ptr %elem_ptr124, align 4
  %elem_ptr125 = getelementptr inbounds float, ptr %buf_void110, i64 8
  store float 1.200000e+01, ptr %elem_ptr125, align 4
  %elem_ptr126 = getelementptr inbounds float, ptr %buf_void110, i64 9
  store float 1.200000e+01, ptr %elem_ptr126, align 4
  %elem_ptr127 = getelementptr inbounds float, ptr %buf_void110, i64 10
  store float 1.200000e+01, ptr %elem_ptr127, align 4
  %elem_ptr128 = getelementptr inbounds float, ptr %buf_void110, i64 11
  store float 1.200000e+01, ptr %elem_ptr128, align 4
  %shape_alloc129 = call ptr @tl_alloc_tmp(i64 8)
  %shape_ptr130 = getelementptr inbounds i64, ptr %shape_alloc129, i64 0
  store i64 12, ptr %shape_ptr130, align 8
  %new_tensor131 = call ptr @tl_tensor_new(ptr %buf_void110, i64 1, ptr %shape_alloc129)
  call void @tl_free_tmp(ptr %buf_void110)
  call void @tl_free_tmp(ptr %shape_alloc129)
  call void @tl_mem_unregister(ptr %new_tensor131)
  store ptr %new_tensor131, ptr %target, align 8
  %data132 = load ptr, ptr %data, align 8
  %dim_ptr_0 = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0, align 8
  %dim_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 1
  store i64 12, ptr %dim_ptr, align 8
  %dims_ptr = getelementptr [2 x i64], ptr %dims_alloca, i64 0, i64 0
  %reshape_dims_res = call ptr @tl_tensor_reshape_dims(ptr %data132, ptr %dims_ptr, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res)
  store ptr %reshape_dims_res, ptr %X, align 8
  %target133 = load ptr, ptr %target, align 8
  %dim_ptr_0135 = getelementptr [2 x i64], ptr %dims_alloca134, i64 0, i64 0
  store i64 1, ptr %dim_ptr_0135, align 8
  %dim_ptr136 = getelementptr [2 x i64], ptr %dims_alloca134, i64 0, i64 1
  store i64 12, ptr %dim_ptr136, align 8
  %dims_ptr137 = getelementptr [2 x i64], ptr %dims_alloca134, i64 0, i64 0
  %reshape_dims_res138 = call ptr @tl_tensor_reshape_dims(ptr %target133, ptr %dims_ptr137, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res138)
  store ptr %reshape_dims_res138, ptr %Y, align 8
  %model139 = load ptr, ptr %model1, align 8
  %X140 = load ptr, ptr %X, align 8
  %call_method = call ptr @tl_GPTHeavy_forward(ptr %model139, ptr %X140)
  call void @tl_mem_register_tensor(ptr %call_method)
  call void @tl_mem_unregister(ptr %call_method)
  store ptr %call_method, ptr %logits, align 8
  %logits141 = load ptr, ptr %logits, align 8
  %dim_ptr_0143 = getelementptr [2 x i64], ptr %dims_alloca142, i64 0, i64 0
  store i64 12, ptr %dim_ptr_0143, align 8
  %dim_ptr144 = getelementptr [2 x i64], ptr %dims_alloca142, i64 0, i64 1
  store i64 13, ptr %dim_ptr144, align 8
  %dims_ptr145 = getelementptr [2 x i64], ptr %dims_alloca142, i64 0, i64 0
  %reshape_dims_res146 = call ptr @tl_tensor_reshape_dims(ptr %logits141, ptr %dims_ptr145, i64 2)
  call void @tl_mem_unregister(ptr %reshape_dims_res146)
  store ptr %reshape_dims_res146, ptr %logits_flat, align 8
  %Y147 = load ptr, ptr %Y, align 8
  %dim_ptr_0149 = getelementptr [1 x i64], ptr %dims_alloca148, i64 0, i64 0
  store i64 12, ptr %dim_ptr_0149, align 8
  %dims_ptr150 = getelementptr [1 x i64], ptr %dims_alloca148, i64 0, i64 0
  %reshape_dims_res151 = call ptr @tl_tensor_reshape_dims(ptr %Y147, ptr %dims_ptr150, i64 1)
  call void @tl_mem_unregister(ptr %reshape_dims_res151)
  store ptr %reshape_dims_res151, ptr %Y_flat, align 8
  %logits_flat152 = load ptr, ptr %logits_flat, align 8
  %Y_flat153 = load ptr, ptr %Y_flat, align 8
  %ce_res = call ptr @tl_tensor_cross_entropy(ptr %logits_flat152, ptr %Y_flat153)
  call void @tl_mem_unregister(ptr %ce_res)
  store ptr %ce_res, ptr %loss, align 8
  %loss154 = load ptr, ptr %loss, align 8
  call void @tl_tensor_backward(ptr %loss154)
  %model155 = load ptr, ptr %model1, align 8
  %lr156 = load float, ptr %lr2, align 4
  %call_method157 = call ptr @tl_GPTHeavy_step(ptr %model155, float %lr156)
  call void @tl_mem_register_struct(ptr %call_method157)
  call void @tl_mem_unregister(ptr %call_method157)
  %old_struct_to_free = load ptr, ptr %model1, align 8
  %is_not_null = icmp ne ptr %old_struct_to_free, null
  %are_diff = icmp ne ptr %old_struct_to_free, %call_method157
  %can_free_1 = and i1 %is_not_null, false
  %can_free = and i1 %can_free_1, %are_diff
  br i1 %can_free, label %free_struct, label %continue_after_free

free_struct:                                      ; preds = %entry
  %field_gep = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 0
  %field_load = load ptr, ptr %field_gep, align 8
  %field_gep158 = getelementptr inbounds %Embedding, ptr %field_load, i32 0, i32 0
  %field_load159 = load ptr, ptr %field_gep158, align 8
  call void @tl_tensor_free(ptr %field_load159)
  %field_gep160 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 1
  %field_load161 = load ptr, ptr %field_gep160, align 8
  %field_gep162 = getelementptr inbounds %Embedding, ptr %field_load161, i32 0, i32 0
  %field_load163 = load ptr, ptr %field_gep162, align 8
  call void @tl_tensor_free(ptr %field_load163)
  %field_gep164 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 2
  %field_load165 = load ptr, ptr %field_gep164, align 8
  %field_gep166 = getelementptr inbounds %Block, ptr %field_load165, i32 0, i32 0
  %field_load167 = load ptr, ptr %field_gep166, align 8
  %field_gep168 = getelementptr inbounds %LayerNorm, ptr %field_load167, i32 0, i32 0
  %field_load169 = load ptr, ptr %field_gep168, align 8
  call void @tl_tensor_free(ptr %field_load169)
  %field_gep170 = getelementptr inbounds %LayerNorm, ptr %field_load167, i32 0, i32 1
  %field_load171 = load ptr, ptr %field_gep170, align 8
  call void @tl_tensor_free(ptr %field_load171)
  %field_gep172 = getelementptr inbounds %Block, ptr %field_load165, i32 0, i32 1
  %field_load173 = load ptr, ptr %field_gep172, align 8
  %field_gep174 = getelementptr inbounds %CausalSelfAttention, ptr %field_load173, i32 0, i32 0
  %field_load175 = load ptr, ptr %field_gep174, align 8
  %field_gep176 = getelementptr inbounds %Linear, ptr %field_load175, i32 0, i32 0
  %field_load177 = load ptr, ptr %field_gep176, align 8
  call void @tl_tensor_free(ptr %field_load177)
  %field_gep178 = getelementptr inbounds %Linear, ptr %field_load175, i32 0, i32 1
  %field_load179 = load ptr, ptr %field_gep178, align 8
  call void @tl_tensor_free(ptr %field_load179)
  %field_gep180 = getelementptr inbounds %CausalSelfAttention, ptr %field_load173, i32 0, i32 1
  %field_load181 = load ptr, ptr %field_gep180, align 8
  %field_gep182 = getelementptr inbounds %Linear, ptr %field_load181, i32 0, i32 0
  %field_load183 = load ptr, ptr %field_gep182, align 8
  call void @tl_tensor_free(ptr %field_load183)
  %field_gep184 = getelementptr inbounds %Linear, ptr %field_load181, i32 0, i32 1
  %field_load185 = load ptr, ptr %field_gep184, align 8
  call void @tl_tensor_free(ptr %field_load185)
  %field_gep186 = getelementptr inbounds %CausalSelfAttention, ptr %field_load173, i32 0, i32 2
  %field_load187 = load ptr, ptr %field_gep186, align 8
  %field_gep188 = getelementptr inbounds %Linear, ptr %field_load187, i32 0, i32 0
  %field_load189 = load ptr, ptr %field_gep188, align 8
  call void @tl_tensor_free(ptr %field_load189)
  %field_gep190 = getelementptr inbounds %Linear, ptr %field_load187, i32 0, i32 1
  %field_load191 = load ptr, ptr %field_gep190, align 8
  call void @tl_tensor_free(ptr %field_load191)
  %field_gep192 = getelementptr inbounds %CausalSelfAttention, ptr %field_load173, i32 0, i32 3
  %field_load193 = load ptr, ptr %field_gep192, align 8
  %field_gep194 = getelementptr inbounds %Linear, ptr %field_load193, i32 0, i32 0
  %field_load195 = load ptr, ptr %field_gep194, align 8
  call void @tl_tensor_free(ptr %field_load195)
  %field_gep196 = getelementptr inbounds %Linear, ptr %field_load193, i32 0, i32 1
  %field_load197 = load ptr, ptr %field_gep196, align 8
  call void @tl_tensor_free(ptr %field_load197)
  %field_gep198 = getelementptr inbounds %Block, ptr %field_load165, i32 0, i32 2
  %field_load199 = load ptr, ptr %field_gep198, align 8
  %field_gep200 = getelementptr inbounds %LayerNorm, ptr %field_load199, i32 0, i32 0
  %field_load201 = load ptr, ptr %field_gep200, align 8
  call void @tl_tensor_free(ptr %field_load201)
  %field_gep202 = getelementptr inbounds %LayerNorm, ptr %field_load199, i32 0, i32 1
  %field_load203 = load ptr, ptr %field_gep202, align 8
  call void @tl_tensor_free(ptr %field_load203)
  %field_gep204 = getelementptr inbounds %Block, ptr %field_load165, i32 0, i32 3
  %field_load205 = load ptr, ptr %field_gep204, align 8
  %field_gep206 = getelementptr inbounds %MLP, ptr %field_load205, i32 0, i32 0
  %field_load207 = load ptr, ptr %field_gep206, align 8
  %field_gep208 = getelementptr inbounds %Linear, ptr %field_load207, i32 0, i32 0
  %field_load209 = load ptr, ptr %field_gep208, align 8
  call void @tl_tensor_free(ptr %field_load209)
  %field_gep210 = getelementptr inbounds %Linear, ptr %field_load207, i32 0, i32 1
  %field_load211 = load ptr, ptr %field_gep210, align 8
  call void @tl_tensor_free(ptr %field_load211)
  %field_gep212 = getelementptr inbounds %MLP, ptr %field_load205, i32 0, i32 1
  %field_load213 = load ptr, ptr %field_gep212, align 8
  %field_gep214 = getelementptr inbounds %Linear, ptr %field_load213, i32 0, i32 0
  %field_load215 = load ptr, ptr %field_gep214, align 8
  call void @tl_tensor_free(ptr %field_load215)
  %field_gep216 = getelementptr inbounds %Linear, ptr %field_load213, i32 0, i32 1
  %field_load217 = load ptr, ptr %field_gep216, align 8
  call void @tl_tensor_free(ptr %field_load217)
  %field_gep218 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 3
  %field_load219 = load ptr, ptr %field_gep218, align 8
  %field_gep220 = getelementptr inbounds %Block, ptr %field_load219, i32 0, i32 0
  %field_load221 = load ptr, ptr %field_gep220, align 8
  %field_gep222 = getelementptr inbounds %LayerNorm, ptr %field_load221, i32 0, i32 0
  %field_load223 = load ptr, ptr %field_gep222, align 8
  call void @tl_tensor_free(ptr %field_load223)
  %field_gep224 = getelementptr inbounds %LayerNorm, ptr %field_load221, i32 0, i32 1
  %field_load225 = load ptr, ptr %field_gep224, align 8
  call void @tl_tensor_free(ptr %field_load225)
  %field_gep226 = getelementptr inbounds %Block, ptr %field_load219, i32 0, i32 1
  %field_load227 = load ptr, ptr %field_gep226, align 8
  %field_gep228 = getelementptr inbounds %CausalSelfAttention, ptr %field_load227, i32 0, i32 0
  %field_load229 = load ptr, ptr %field_gep228, align 8
  %field_gep230 = getelementptr inbounds %Linear, ptr %field_load229, i32 0, i32 0
  %field_load231 = load ptr, ptr %field_gep230, align 8
  call void @tl_tensor_free(ptr %field_load231)
  %field_gep232 = getelementptr inbounds %Linear, ptr %field_load229, i32 0, i32 1
  %field_load233 = load ptr, ptr %field_gep232, align 8
  call void @tl_tensor_free(ptr %field_load233)
  %field_gep234 = getelementptr inbounds %CausalSelfAttention, ptr %field_load227, i32 0, i32 1
  %field_load235 = load ptr, ptr %field_gep234, align 8
  %field_gep236 = getelementptr inbounds %Linear, ptr %field_load235, i32 0, i32 0
  %field_load237 = load ptr, ptr %field_gep236, align 8
  call void @tl_tensor_free(ptr %field_load237)
  %field_gep238 = getelementptr inbounds %Linear, ptr %field_load235, i32 0, i32 1
  %field_load239 = load ptr, ptr %field_gep238, align 8
  call void @tl_tensor_free(ptr %field_load239)
  %field_gep240 = getelementptr inbounds %CausalSelfAttention, ptr %field_load227, i32 0, i32 2
  %field_load241 = load ptr, ptr %field_gep240, align 8
  %field_gep242 = getelementptr inbounds %Linear, ptr %field_load241, i32 0, i32 0
  %field_load243 = load ptr, ptr %field_gep242, align 8
  call void @tl_tensor_free(ptr %field_load243)
  %field_gep244 = getelementptr inbounds %Linear, ptr %field_load241, i32 0, i32 1
  %field_load245 = load ptr, ptr %field_gep244, align 8
  call void @tl_tensor_free(ptr %field_load245)
  %field_gep246 = getelementptr inbounds %CausalSelfAttention, ptr %field_load227, i32 0, i32 3
  %field_load247 = load ptr, ptr %field_gep246, align 8
  %field_gep248 = getelementptr inbounds %Linear, ptr %field_load247, i32 0, i32 0
  %field_load249 = load ptr, ptr %field_gep248, align 8
  call void @tl_tensor_free(ptr %field_load249)
  %field_gep250 = getelementptr inbounds %Linear, ptr %field_load247, i32 0, i32 1
  %field_load251 = load ptr, ptr %field_gep250, align 8
  call void @tl_tensor_free(ptr %field_load251)
  %field_gep252 = getelementptr inbounds %Block, ptr %field_load219, i32 0, i32 2
  %field_load253 = load ptr, ptr %field_gep252, align 8
  %field_gep254 = getelementptr inbounds %LayerNorm, ptr %field_load253, i32 0, i32 0
  %field_load255 = load ptr, ptr %field_gep254, align 8
  call void @tl_tensor_free(ptr %field_load255)
  %field_gep256 = getelementptr inbounds %LayerNorm, ptr %field_load253, i32 0, i32 1
  %field_load257 = load ptr, ptr %field_gep256, align 8
  call void @tl_tensor_free(ptr %field_load257)
  %field_gep258 = getelementptr inbounds %Block, ptr %field_load219, i32 0, i32 3
  %field_load259 = load ptr, ptr %field_gep258, align 8
  %field_gep260 = getelementptr inbounds %MLP, ptr %field_load259, i32 0, i32 0
  %field_load261 = load ptr, ptr %field_gep260, align 8
  %field_gep262 = getelementptr inbounds %Linear, ptr %field_load261, i32 0, i32 0
  %field_load263 = load ptr, ptr %field_gep262, align 8
  call void @tl_tensor_free(ptr %field_load263)
  %field_gep264 = getelementptr inbounds %Linear, ptr %field_load261, i32 0, i32 1
  %field_load265 = load ptr, ptr %field_gep264, align 8
  call void @tl_tensor_free(ptr %field_load265)
  %field_gep266 = getelementptr inbounds %MLP, ptr %field_load259, i32 0, i32 1
  %field_load267 = load ptr, ptr %field_gep266, align 8
  %field_gep268 = getelementptr inbounds %Linear, ptr %field_load267, i32 0, i32 0
  %field_load269 = load ptr, ptr %field_gep268, align 8
  call void @tl_tensor_free(ptr %field_load269)
  %field_gep270 = getelementptr inbounds %Linear, ptr %field_load267, i32 0, i32 1
  %field_load271 = load ptr, ptr %field_gep270, align 8
  call void @tl_tensor_free(ptr %field_load271)
  %field_gep272 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 4
  %field_load273 = load ptr, ptr %field_gep272, align 8
  %field_gep274 = getelementptr inbounds %Block, ptr %field_load273, i32 0, i32 0
  %field_load275 = load ptr, ptr %field_gep274, align 8
  %field_gep276 = getelementptr inbounds %LayerNorm, ptr %field_load275, i32 0, i32 0
  %field_load277 = load ptr, ptr %field_gep276, align 8
  call void @tl_tensor_free(ptr %field_load277)
  %field_gep278 = getelementptr inbounds %LayerNorm, ptr %field_load275, i32 0, i32 1
  %field_load279 = load ptr, ptr %field_gep278, align 8
  call void @tl_tensor_free(ptr %field_load279)
  %field_gep280 = getelementptr inbounds %Block, ptr %field_load273, i32 0, i32 1
  %field_load281 = load ptr, ptr %field_gep280, align 8
  %field_gep282 = getelementptr inbounds %CausalSelfAttention, ptr %field_load281, i32 0, i32 0
  %field_load283 = load ptr, ptr %field_gep282, align 8
  %field_gep284 = getelementptr inbounds %Linear, ptr %field_load283, i32 0, i32 0
  %field_load285 = load ptr, ptr %field_gep284, align 8
  call void @tl_tensor_free(ptr %field_load285)
  %field_gep286 = getelementptr inbounds %Linear, ptr %field_load283, i32 0, i32 1
  %field_load287 = load ptr, ptr %field_gep286, align 8
  call void @tl_tensor_free(ptr %field_load287)
  %field_gep288 = getelementptr inbounds %CausalSelfAttention, ptr %field_load281, i32 0, i32 1
  %field_load289 = load ptr, ptr %field_gep288, align 8
  %field_gep290 = getelementptr inbounds %Linear, ptr %field_load289, i32 0, i32 0
  %field_load291 = load ptr, ptr %field_gep290, align 8
  call void @tl_tensor_free(ptr %field_load291)
  %field_gep292 = getelementptr inbounds %Linear, ptr %field_load289, i32 0, i32 1
  %field_load293 = load ptr, ptr %field_gep292, align 8
  call void @tl_tensor_free(ptr %field_load293)
  %field_gep294 = getelementptr inbounds %CausalSelfAttention, ptr %field_load281, i32 0, i32 2
  %field_load295 = load ptr, ptr %field_gep294, align 8
  %field_gep296 = getelementptr inbounds %Linear, ptr %field_load295, i32 0, i32 0
  %field_load297 = load ptr, ptr %field_gep296, align 8
  call void @tl_tensor_free(ptr %field_load297)
  %field_gep298 = getelementptr inbounds %Linear, ptr %field_load295, i32 0, i32 1
  %field_load299 = load ptr, ptr %field_gep298, align 8
  call void @tl_tensor_free(ptr %field_load299)
  %field_gep300 = getelementptr inbounds %CausalSelfAttention, ptr %field_load281, i32 0, i32 3
  %field_load301 = load ptr, ptr %field_gep300, align 8
  %field_gep302 = getelementptr inbounds %Linear, ptr %field_load301, i32 0, i32 0
  %field_load303 = load ptr, ptr %field_gep302, align 8
  call void @tl_tensor_free(ptr %field_load303)
  %field_gep304 = getelementptr inbounds %Linear, ptr %field_load301, i32 0, i32 1
  %field_load305 = load ptr, ptr %field_gep304, align 8
  call void @tl_tensor_free(ptr %field_load305)
  %field_gep306 = getelementptr inbounds %Block, ptr %field_load273, i32 0, i32 2
  %field_load307 = load ptr, ptr %field_gep306, align 8
  %field_gep308 = getelementptr inbounds %LayerNorm, ptr %field_load307, i32 0, i32 0
  %field_load309 = load ptr, ptr %field_gep308, align 8
  call void @tl_tensor_free(ptr %field_load309)
  %field_gep310 = getelementptr inbounds %LayerNorm, ptr %field_load307, i32 0, i32 1
  %field_load311 = load ptr, ptr %field_gep310, align 8
  call void @tl_tensor_free(ptr %field_load311)
  %field_gep312 = getelementptr inbounds %Block, ptr %field_load273, i32 0, i32 3
  %field_load313 = load ptr, ptr %field_gep312, align 8
  %field_gep314 = getelementptr inbounds %MLP, ptr %field_load313, i32 0, i32 0
  %field_load315 = load ptr, ptr %field_gep314, align 8
  %field_gep316 = getelementptr inbounds %Linear, ptr %field_load315, i32 0, i32 0
  %field_load317 = load ptr, ptr %field_gep316, align 8
  call void @tl_tensor_free(ptr %field_load317)
  %field_gep318 = getelementptr inbounds %Linear, ptr %field_load315, i32 0, i32 1
  %field_load319 = load ptr, ptr %field_gep318, align 8
  call void @tl_tensor_free(ptr %field_load319)
  %field_gep320 = getelementptr inbounds %MLP, ptr %field_load313, i32 0, i32 1
  %field_load321 = load ptr, ptr %field_gep320, align 8
  %field_gep322 = getelementptr inbounds %Linear, ptr %field_load321, i32 0, i32 0
  %field_load323 = load ptr, ptr %field_gep322, align 8
  call void @tl_tensor_free(ptr %field_load323)
  %field_gep324 = getelementptr inbounds %Linear, ptr %field_load321, i32 0, i32 1
  %field_load325 = load ptr, ptr %field_gep324, align 8
  call void @tl_tensor_free(ptr %field_load325)
  %field_gep326 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 5
  %field_load327 = load ptr, ptr %field_gep326, align 8
  %field_gep328 = getelementptr inbounds %Block, ptr %field_load327, i32 0, i32 0
  %field_load329 = load ptr, ptr %field_gep328, align 8
  %field_gep330 = getelementptr inbounds %LayerNorm, ptr %field_load329, i32 0, i32 0
  %field_load331 = load ptr, ptr %field_gep330, align 8
  call void @tl_tensor_free(ptr %field_load331)
  %field_gep332 = getelementptr inbounds %LayerNorm, ptr %field_load329, i32 0, i32 1
  %field_load333 = load ptr, ptr %field_gep332, align 8
  call void @tl_tensor_free(ptr %field_load333)
  %field_gep334 = getelementptr inbounds %Block, ptr %field_load327, i32 0, i32 1
  %field_load335 = load ptr, ptr %field_gep334, align 8
  %field_gep336 = getelementptr inbounds %CausalSelfAttention, ptr %field_load335, i32 0, i32 0
  %field_load337 = load ptr, ptr %field_gep336, align 8
  %field_gep338 = getelementptr inbounds %Linear, ptr %field_load337, i32 0, i32 0
  %field_load339 = load ptr, ptr %field_gep338, align 8
  call void @tl_tensor_free(ptr %field_load339)
  %field_gep340 = getelementptr inbounds %Linear, ptr %field_load337, i32 0, i32 1
  %field_load341 = load ptr, ptr %field_gep340, align 8
  call void @tl_tensor_free(ptr %field_load341)
  %field_gep342 = getelementptr inbounds %CausalSelfAttention, ptr %field_load335, i32 0, i32 1
  %field_load343 = load ptr, ptr %field_gep342, align 8
  %field_gep344 = getelementptr inbounds %Linear, ptr %field_load343, i32 0, i32 0
  %field_load345 = load ptr, ptr %field_gep344, align 8
  call void @tl_tensor_free(ptr %field_load345)
  %field_gep346 = getelementptr inbounds %Linear, ptr %field_load343, i32 0, i32 1
  %field_load347 = load ptr, ptr %field_gep346, align 8
  call void @tl_tensor_free(ptr %field_load347)
  %field_gep348 = getelementptr inbounds %CausalSelfAttention, ptr %field_load335, i32 0, i32 2
  %field_load349 = load ptr, ptr %field_gep348, align 8
  %field_gep350 = getelementptr inbounds %Linear, ptr %field_load349, i32 0, i32 0
  %field_load351 = load ptr, ptr %field_gep350, align 8
  call void @tl_tensor_free(ptr %field_load351)
  %field_gep352 = getelementptr inbounds %Linear, ptr %field_load349, i32 0, i32 1
  %field_load353 = load ptr, ptr %field_gep352, align 8
  call void @tl_tensor_free(ptr %field_load353)
  %field_gep354 = getelementptr inbounds %CausalSelfAttention, ptr %field_load335, i32 0, i32 3
  %field_load355 = load ptr, ptr %field_gep354, align 8
  %field_gep356 = getelementptr inbounds %Linear, ptr %field_load355, i32 0, i32 0
  %field_load357 = load ptr, ptr %field_gep356, align 8
  call void @tl_tensor_free(ptr %field_load357)
  %field_gep358 = getelementptr inbounds %Linear, ptr %field_load355, i32 0, i32 1
  %field_load359 = load ptr, ptr %field_gep358, align 8
  call void @tl_tensor_free(ptr %field_load359)
  %field_gep360 = getelementptr inbounds %Block, ptr %field_load327, i32 0, i32 2
  %field_load361 = load ptr, ptr %field_gep360, align 8
  %field_gep362 = getelementptr inbounds %LayerNorm, ptr %field_load361, i32 0, i32 0
  %field_load363 = load ptr, ptr %field_gep362, align 8
  call void @tl_tensor_free(ptr %field_load363)
  %field_gep364 = getelementptr inbounds %LayerNorm, ptr %field_load361, i32 0, i32 1
  %field_load365 = load ptr, ptr %field_gep364, align 8
  call void @tl_tensor_free(ptr %field_load365)
  %field_gep366 = getelementptr inbounds %Block, ptr %field_load327, i32 0, i32 3
  %field_load367 = load ptr, ptr %field_gep366, align 8
  %field_gep368 = getelementptr inbounds %MLP, ptr %field_load367, i32 0, i32 0
  %field_load369 = load ptr, ptr %field_gep368, align 8
  %field_gep370 = getelementptr inbounds %Linear, ptr %field_load369, i32 0, i32 0
  %field_load371 = load ptr, ptr %field_gep370, align 8
  call void @tl_tensor_free(ptr %field_load371)
  %field_gep372 = getelementptr inbounds %Linear, ptr %field_load369, i32 0, i32 1
  %field_load373 = load ptr, ptr %field_gep372, align 8
  call void @tl_tensor_free(ptr %field_load373)
  %field_gep374 = getelementptr inbounds %MLP, ptr %field_load367, i32 0, i32 1
  %field_load375 = load ptr, ptr %field_gep374, align 8
  %field_gep376 = getelementptr inbounds %Linear, ptr %field_load375, i32 0, i32 0
  %field_load377 = load ptr, ptr %field_gep376, align 8
  call void @tl_tensor_free(ptr %field_load377)
  %field_gep378 = getelementptr inbounds %Linear, ptr %field_load375, i32 0, i32 1
  %field_load379 = load ptr, ptr %field_gep378, align 8
  call void @tl_tensor_free(ptr %field_load379)
  %field_gep380 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 6
  %field_load381 = load ptr, ptr %field_gep380, align 8
  %field_gep382 = getelementptr inbounds %LayerNorm, ptr %field_load381, i32 0, i32 0
  %field_load383 = load ptr, ptr %field_gep382, align 8
  call void @tl_tensor_free(ptr %field_load383)
  %field_gep384 = getelementptr inbounds %LayerNorm, ptr %field_load381, i32 0, i32 1
  %field_load385 = load ptr, ptr %field_gep384, align 8
  call void @tl_tensor_free(ptr %field_load385)
  %field_gep386 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 7
  %field_load387 = load ptr, ptr %field_gep386, align 8
  %field_gep388 = getelementptr inbounds %Linear, ptr %field_load387, i32 0, i32 0
  %field_load389 = load ptr, ptr %field_gep388, align 8
  call void @tl_tensor_free(ptr %field_load389)
  %field_gep390 = getelementptr inbounds %Linear, ptr %field_load387, i32 0, i32 1
  %field_load391 = load ptr, ptr %field_gep390, align 8
  call void @tl_tensor_free(ptr %field_load391)
  call void @tl_mem_unregister(ptr %old_struct_to_free)
  br label %continue_after_free

continue_after_free:                              ; preds = %free_struct, %entry
  store ptr %call_method157, ptr %model1, align 8
  %model392 = load ptr, ptr %model1, align 8
  call void @tl_mem_unregister(ptr %model392)
  %unreg_field_0 = getelementptr inbounds %GPTHeavy, ptr %model392, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_0393 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val394 = load ptr, ptr %unreg_field_0393, align 8
  call void @tl_mem_unregister(ptr %field_val394)
  %unreg_field_1 = getelementptr inbounds %GPTHeavy, ptr %model392, i32 0, i32 1
  %field_val395 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val395)
  %unreg_field_0396 = getelementptr inbounds %Embedding, ptr %field_val395, i32 0, i32 0
  %field_val397 = load ptr, ptr %unreg_field_0396, align 8
  call void @tl_mem_unregister(ptr %field_val397)
  %unreg_field_2 = getelementptr inbounds %GPTHeavy, ptr %model392, i32 0, i32 2
  %field_val398 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val398)
  %unreg_field_0399 = getelementptr inbounds %Block, ptr %field_val398, i32 0, i32 0
  %field_val400 = load ptr, ptr %unreg_field_0399, align 8
  call void @tl_mem_unregister(ptr %field_val400)
  %unreg_field_0401 = getelementptr inbounds %LayerNorm, ptr %field_val400, i32 0, i32 0
  %field_val402 = load ptr, ptr %unreg_field_0401, align 8
  call void @tl_mem_unregister(ptr %field_val402)
  %unreg_field_1403 = getelementptr inbounds %LayerNorm, ptr %field_val400, i32 0, i32 1
  %field_val404 = load ptr, ptr %unreg_field_1403, align 8
  call void @tl_mem_unregister(ptr %field_val404)
  %unreg_field_1405 = getelementptr inbounds %Block, ptr %field_val398, i32 0, i32 1
  %field_val406 = load ptr, ptr %unreg_field_1405, align 8
  call void @tl_mem_unregister(ptr %field_val406)
  %unreg_field_0407 = getelementptr inbounds %CausalSelfAttention, ptr %field_val406, i32 0, i32 0
  %field_val408 = load ptr, ptr %unreg_field_0407, align 8
  call void @tl_mem_unregister(ptr %field_val408)
  %unreg_field_0409 = getelementptr inbounds %Linear, ptr %field_val408, i32 0, i32 0
  %field_val410 = load ptr, ptr %unreg_field_0409, align 8
  call void @tl_mem_unregister(ptr %field_val410)
  %unreg_field_1411 = getelementptr inbounds %Linear, ptr %field_val408, i32 0, i32 1
  %field_val412 = load ptr, ptr %unreg_field_1411, align 8
  call void @tl_mem_unregister(ptr %field_val412)
  %unreg_field_1413 = getelementptr inbounds %CausalSelfAttention, ptr %field_val406, i32 0, i32 1
  %field_val414 = load ptr, ptr %unreg_field_1413, align 8
  call void @tl_mem_unregister(ptr %field_val414)
  %unreg_field_0415 = getelementptr inbounds %Linear, ptr %field_val414, i32 0, i32 0
  %field_val416 = load ptr, ptr %unreg_field_0415, align 8
  call void @tl_mem_unregister(ptr %field_val416)
  %unreg_field_1417 = getelementptr inbounds %Linear, ptr %field_val414, i32 0, i32 1
  %field_val418 = load ptr, ptr %unreg_field_1417, align 8
  call void @tl_mem_unregister(ptr %field_val418)
  %unreg_field_2419 = getelementptr inbounds %CausalSelfAttention, ptr %field_val406, i32 0, i32 2
  %field_val420 = load ptr, ptr %unreg_field_2419, align 8
  call void @tl_mem_unregister(ptr %field_val420)
  %unreg_field_0421 = getelementptr inbounds %Linear, ptr %field_val420, i32 0, i32 0
  %field_val422 = load ptr, ptr %unreg_field_0421, align 8
  call void @tl_mem_unregister(ptr %field_val422)
  %unreg_field_1423 = getelementptr inbounds %Linear, ptr %field_val420, i32 0, i32 1
  %field_val424 = load ptr, ptr %unreg_field_1423, align 8
  call void @tl_mem_unregister(ptr %field_val424)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val406, i32 0, i32 3
  %field_val425 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val425)
  %unreg_field_0426 = getelementptr inbounds %Linear, ptr %field_val425, i32 0, i32 0
  %field_val427 = load ptr, ptr %unreg_field_0426, align 8
  call void @tl_mem_unregister(ptr %field_val427)
  %unreg_field_1428 = getelementptr inbounds %Linear, ptr %field_val425, i32 0, i32 1
  %field_val429 = load ptr, ptr %unreg_field_1428, align 8
  call void @tl_mem_unregister(ptr %field_val429)
  %unreg_field_2430 = getelementptr inbounds %Block, ptr %field_val398, i32 0, i32 2
  %field_val431 = load ptr, ptr %unreg_field_2430, align 8
  call void @tl_mem_unregister(ptr %field_val431)
  %unreg_field_0432 = getelementptr inbounds %LayerNorm, ptr %field_val431, i32 0, i32 0
  %field_val433 = load ptr, ptr %unreg_field_0432, align 8
  call void @tl_mem_unregister(ptr %field_val433)
  %unreg_field_1434 = getelementptr inbounds %LayerNorm, ptr %field_val431, i32 0, i32 1
  %field_val435 = load ptr, ptr %unreg_field_1434, align 8
  call void @tl_mem_unregister(ptr %field_val435)
  %unreg_field_3436 = getelementptr inbounds %Block, ptr %field_val398, i32 0, i32 3
  %field_val437 = load ptr, ptr %unreg_field_3436, align 8
  call void @tl_mem_unregister(ptr %field_val437)
  %unreg_field_0438 = getelementptr inbounds %MLP, ptr %field_val437, i32 0, i32 0
  %field_val439 = load ptr, ptr %unreg_field_0438, align 8
  call void @tl_mem_unregister(ptr %field_val439)
  %unreg_field_0440 = getelementptr inbounds %Linear, ptr %field_val439, i32 0, i32 0
  %field_val441 = load ptr, ptr %unreg_field_0440, align 8
  call void @tl_mem_unregister(ptr %field_val441)
  %unreg_field_1442 = getelementptr inbounds %Linear, ptr %field_val439, i32 0, i32 1
  %field_val443 = load ptr, ptr %unreg_field_1442, align 8
  call void @tl_mem_unregister(ptr %field_val443)
  %unreg_field_1444 = getelementptr inbounds %MLP, ptr %field_val437, i32 0, i32 1
  %field_val445 = load ptr, ptr %unreg_field_1444, align 8
  call void @tl_mem_unregister(ptr %field_val445)
  %unreg_field_0446 = getelementptr inbounds %Linear, ptr %field_val445, i32 0, i32 0
  %field_val447 = load ptr, ptr %unreg_field_0446, align 8
  call void @tl_mem_unregister(ptr %field_val447)
  %unreg_field_1448 = getelementptr inbounds %Linear, ptr %field_val445, i32 0, i32 1
  %field_val449 = load ptr, ptr %unreg_field_1448, align 8
  call void @tl_mem_unregister(ptr %field_val449)
  %unreg_field_3450 = getelementptr inbounds %GPTHeavy, ptr %model392, i32 0, i32 3
  %field_val451 = load ptr, ptr %unreg_field_3450, align 8
  call void @tl_mem_unregister(ptr %field_val451)
  %unreg_field_0452 = getelementptr inbounds %Block, ptr %field_val451, i32 0, i32 0
  %field_val453 = load ptr, ptr %unreg_field_0452, align 8
  call void @tl_mem_unregister(ptr %field_val453)
  %unreg_field_0454 = getelementptr inbounds %LayerNorm, ptr %field_val453, i32 0, i32 0
  %field_val455 = load ptr, ptr %unreg_field_0454, align 8
  call void @tl_mem_unregister(ptr %field_val455)
  %unreg_field_1456 = getelementptr inbounds %LayerNorm, ptr %field_val453, i32 0, i32 1
  %field_val457 = load ptr, ptr %unreg_field_1456, align 8
  call void @tl_mem_unregister(ptr %field_val457)
  %unreg_field_1458 = getelementptr inbounds %Block, ptr %field_val451, i32 0, i32 1
  %field_val459 = load ptr, ptr %unreg_field_1458, align 8
  call void @tl_mem_unregister(ptr %field_val459)
  %unreg_field_0460 = getelementptr inbounds %CausalSelfAttention, ptr %field_val459, i32 0, i32 0
  %field_val461 = load ptr, ptr %unreg_field_0460, align 8
  call void @tl_mem_unregister(ptr %field_val461)
  %unreg_field_0462 = getelementptr inbounds %Linear, ptr %field_val461, i32 0, i32 0
  %field_val463 = load ptr, ptr %unreg_field_0462, align 8
  call void @tl_mem_unregister(ptr %field_val463)
  %unreg_field_1464 = getelementptr inbounds %Linear, ptr %field_val461, i32 0, i32 1
  %field_val465 = load ptr, ptr %unreg_field_1464, align 8
  call void @tl_mem_unregister(ptr %field_val465)
  %unreg_field_1466 = getelementptr inbounds %CausalSelfAttention, ptr %field_val459, i32 0, i32 1
  %field_val467 = load ptr, ptr %unreg_field_1466, align 8
  call void @tl_mem_unregister(ptr %field_val467)
  %unreg_field_0468 = getelementptr inbounds %Linear, ptr %field_val467, i32 0, i32 0
  %field_val469 = load ptr, ptr %unreg_field_0468, align 8
  call void @tl_mem_unregister(ptr %field_val469)
  %unreg_field_1470 = getelementptr inbounds %Linear, ptr %field_val467, i32 0, i32 1
  %field_val471 = load ptr, ptr %unreg_field_1470, align 8
  call void @tl_mem_unregister(ptr %field_val471)
  %unreg_field_2472 = getelementptr inbounds %CausalSelfAttention, ptr %field_val459, i32 0, i32 2
  %field_val473 = load ptr, ptr %unreg_field_2472, align 8
  call void @tl_mem_unregister(ptr %field_val473)
  %unreg_field_0474 = getelementptr inbounds %Linear, ptr %field_val473, i32 0, i32 0
  %field_val475 = load ptr, ptr %unreg_field_0474, align 8
  call void @tl_mem_unregister(ptr %field_val475)
  %unreg_field_1476 = getelementptr inbounds %Linear, ptr %field_val473, i32 0, i32 1
  %field_val477 = load ptr, ptr %unreg_field_1476, align 8
  call void @tl_mem_unregister(ptr %field_val477)
  %unreg_field_3478 = getelementptr inbounds %CausalSelfAttention, ptr %field_val459, i32 0, i32 3
  %field_val479 = load ptr, ptr %unreg_field_3478, align 8
  call void @tl_mem_unregister(ptr %field_val479)
  %unreg_field_0480 = getelementptr inbounds %Linear, ptr %field_val479, i32 0, i32 0
  %field_val481 = load ptr, ptr %unreg_field_0480, align 8
  call void @tl_mem_unregister(ptr %field_val481)
  %unreg_field_1482 = getelementptr inbounds %Linear, ptr %field_val479, i32 0, i32 1
  %field_val483 = load ptr, ptr %unreg_field_1482, align 8
  call void @tl_mem_unregister(ptr %field_val483)
  %unreg_field_2484 = getelementptr inbounds %Block, ptr %field_val451, i32 0, i32 2
  %field_val485 = load ptr, ptr %unreg_field_2484, align 8
  call void @tl_mem_unregister(ptr %field_val485)
  %unreg_field_0486 = getelementptr inbounds %LayerNorm, ptr %field_val485, i32 0, i32 0
  %field_val487 = load ptr, ptr %unreg_field_0486, align 8
  call void @tl_mem_unregister(ptr %field_val487)
  %unreg_field_1488 = getelementptr inbounds %LayerNorm, ptr %field_val485, i32 0, i32 1
  %field_val489 = load ptr, ptr %unreg_field_1488, align 8
  call void @tl_mem_unregister(ptr %field_val489)
  %unreg_field_3490 = getelementptr inbounds %Block, ptr %field_val451, i32 0, i32 3
  %field_val491 = load ptr, ptr %unreg_field_3490, align 8
  call void @tl_mem_unregister(ptr %field_val491)
  %unreg_field_0492 = getelementptr inbounds %MLP, ptr %field_val491, i32 0, i32 0
  %field_val493 = load ptr, ptr %unreg_field_0492, align 8
  call void @tl_mem_unregister(ptr %field_val493)
  %unreg_field_0494 = getelementptr inbounds %Linear, ptr %field_val493, i32 0, i32 0
  %field_val495 = load ptr, ptr %unreg_field_0494, align 8
  call void @tl_mem_unregister(ptr %field_val495)
  %unreg_field_1496 = getelementptr inbounds %Linear, ptr %field_val493, i32 0, i32 1
  %field_val497 = load ptr, ptr %unreg_field_1496, align 8
  call void @tl_mem_unregister(ptr %field_val497)
  %unreg_field_1498 = getelementptr inbounds %MLP, ptr %field_val491, i32 0, i32 1
  %field_val499 = load ptr, ptr %unreg_field_1498, align 8
  call void @tl_mem_unregister(ptr %field_val499)
  %unreg_field_0500 = getelementptr inbounds %Linear, ptr %field_val499, i32 0, i32 0
  %field_val501 = load ptr, ptr %unreg_field_0500, align 8
  call void @tl_mem_unregister(ptr %field_val501)
  %unreg_field_1502 = getelementptr inbounds %Linear, ptr %field_val499, i32 0, i32 1
  %field_val503 = load ptr, ptr %unreg_field_1502, align 8
  call void @tl_mem_unregister(ptr %field_val503)
  %unreg_field_4 = getelementptr inbounds %GPTHeavy, ptr %model392, i32 0, i32 4
  %field_val504 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val504)
  %unreg_field_0505 = getelementptr inbounds %Block, ptr %field_val504, i32 0, i32 0
  %field_val506 = load ptr, ptr %unreg_field_0505, align 8
  call void @tl_mem_unregister(ptr %field_val506)
  %unreg_field_0507 = getelementptr inbounds %LayerNorm, ptr %field_val506, i32 0, i32 0
  %field_val508 = load ptr, ptr %unreg_field_0507, align 8
  call void @tl_mem_unregister(ptr %field_val508)
  %unreg_field_1509 = getelementptr inbounds %LayerNorm, ptr %field_val506, i32 0, i32 1
  %field_val510 = load ptr, ptr %unreg_field_1509, align 8
  call void @tl_mem_unregister(ptr %field_val510)
  %unreg_field_1511 = getelementptr inbounds %Block, ptr %field_val504, i32 0, i32 1
  %field_val512 = load ptr, ptr %unreg_field_1511, align 8
  call void @tl_mem_unregister(ptr %field_val512)
  %unreg_field_0513 = getelementptr inbounds %CausalSelfAttention, ptr %field_val512, i32 0, i32 0
  %field_val514 = load ptr, ptr %unreg_field_0513, align 8
  call void @tl_mem_unregister(ptr %field_val514)
  %unreg_field_0515 = getelementptr inbounds %Linear, ptr %field_val514, i32 0, i32 0
  %field_val516 = load ptr, ptr %unreg_field_0515, align 8
  call void @tl_mem_unregister(ptr %field_val516)
  %unreg_field_1517 = getelementptr inbounds %Linear, ptr %field_val514, i32 0, i32 1
  %field_val518 = load ptr, ptr %unreg_field_1517, align 8
  call void @tl_mem_unregister(ptr %field_val518)
  %unreg_field_1519 = getelementptr inbounds %CausalSelfAttention, ptr %field_val512, i32 0, i32 1
  %field_val520 = load ptr, ptr %unreg_field_1519, align 8
  call void @tl_mem_unregister(ptr %field_val520)
  %unreg_field_0521 = getelementptr inbounds %Linear, ptr %field_val520, i32 0, i32 0
  %field_val522 = load ptr, ptr %unreg_field_0521, align 8
  call void @tl_mem_unregister(ptr %field_val522)
  %unreg_field_1523 = getelementptr inbounds %Linear, ptr %field_val520, i32 0, i32 1
  %field_val524 = load ptr, ptr %unreg_field_1523, align 8
  call void @tl_mem_unregister(ptr %field_val524)
  %unreg_field_2525 = getelementptr inbounds %CausalSelfAttention, ptr %field_val512, i32 0, i32 2
  %field_val526 = load ptr, ptr %unreg_field_2525, align 8
  call void @tl_mem_unregister(ptr %field_val526)
  %unreg_field_0527 = getelementptr inbounds %Linear, ptr %field_val526, i32 0, i32 0
  %field_val528 = load ptr, ptr %unreg_field_0527, align 8
  call void @tl_mem_unregister(ptr %field_val528)
  %unreg_field_1529 = getelementptr inbounds %Linear, ptr %field_val526, i32 0, i32 1
  %field_val530 = load ptr, ptr %unreg_field_1529, align 8
  call void @tl_mem_unregister(ptr %field_val530)
  %unreg_field_3531 = getelementptr inbounds %CausalSelfAttention, ptr %field_val512, i32 0, i32 3
  %field_val532 = load ptr, ptr %unreg_field_3531, align 8
  call void @tl_mem_unregister(ptr %field_val532)
  %unreg_field_0533 = getelementptr inbounds %Linear, ptr %field_val532, i32 0, i32 0
  %field_val534 = load ptr, ptr %unreg_field_0533, align 8
  call void @tl_mem_unregister(ptr %field_val534)
  %unreg_field_1535 = getelementptr inbounds %Linear, ptr %field_val532, i32 0, i32 1
  %field_val536 = load ptr, ptr %unreg_field_1535, align 8
  call void @tl_mem_unregister(ptr %field_val536)
  %unreg_field_2537 = getelementptr inbounds %Block, ptr %field_val504, i32 0, i32 2
  %field_val538 = load ptr, ptr %unreg_field_2537, align 8
  call void @tl_mem_unregister(ptr %field_val538)
  %unreg_field_0539 = getelementptr inbounds %LayerNorm, ptr %field_val538, i32 0, i32 0
  %field_val540 = load ptr, ptr %unreg_field_0539, align 8
  call void @tl_mem_unregister(ptr %field_val540)
  %unreg_field_1541 = getelementptr inbounds %LayerNorm, ptr %field_val538, i32 0, i32 1
  %field_val542 = load ptr, ptr %unreg_field_1541, align 8
  call void @tl_mem_unregister(ptr %field_val542)
  %unreg_field_3543 = getelementptr inbounds %Block, ptr %field_val504, i32 0, i32 3
  %field_val544 = load ptr, ptr %unreg_field_3543, align 8
  call void @tl_mem_unregister(ptr %field_val544)
  %unreg_field_0545 = getelementptr inbounds %MLP, ptr %field_val544, i32 0, i32 0
  %field_val546 = load ptr, ptr %unreg_field_0545, align 8
  call void @tl_mem_unregister(ptr %field_val546)
  %unreg_field_0547 = getelementptr inbounds %Linear, ptr %field_val546, i32 0, i32 0
  %field_val548 = load ptr, ptr %unreg_field_0547, align 8
  call void @tl_mem_unregister(ptr %field_val548)
  %unreg_field_1549 = getelementptr inbounds %Linear, ptr %field_val546, i32 0, i32 1
  %field_val550 = load ptr, ptr %unreg_field_1549, align 8
  call void @tl_mem_unregister(ptr %field_val550)
  %unreg_field_1551 = getelementptr inbounds %MLP, ptr %field_val544, i32 0, i32 1
  %field_val552 = load ptr, ptr %unreg_field_1551, align 8
  call void @tl_mem_unregister(ptr %field_val552)
  %unreg_field_0553 = getelementptr inbounds %Linear, ptr %field_val552, i32 0, i32 0
  %field_val554 = load ptr, ptr %unreg_field_0553, align 8
  call void @tl_mem_unregister(ptr %field_val554)
  %unreg_field_1555 = getelementptr inbounds %Linear, ptr %field_val552, i32 0, i32 1
  %field_val556 = load ptr, ptr %unreg_field_1555, align 8
  call void @tl_mem_unregister(ptr %field_val556)
  %unreg_field_5 = getelementptr inbounds %GPTHeavy, ptr %model392, i32 0, i32 5
  %field_val557 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val557)
  %unreg_field_0558 = getelementptr inbounds %Block, ptr %field_val557, i32 0, i32 0
  %field_val559 = load ptr, ptr %unreg_field_0558, align 8
  call void @tl_mem_unregister(ptr %field_val559)
  %unreg_field_0560 = getelementptr inbounds %LayerNorm, ptr %field_val559, i32 0, i32 0
  %field_val561 = load ptr, ptr %unreg_field_0560, align 8
  call void @tl_mem_unregister(ptr %field_val561)
  %unreg_field_1562 = getelementptr inbounds %LayerNorm, ptr %field_val559, i32 0, i32 1
  %field_val563 = load ptr, ptr %unreg_field_1562, align 8
  call void @tl_mem_unregister(ptr %field_val563)
  %unreg_field_1564 = getelementptr inbounds %Block, ptr %field_val557, i32 0, i32 1
  %field_val565 = load ptr, ptr %unreg_field_1564, align 8
  call void @tl_mem_unregister(ptr %field_val565)
  %unreg_field_0566 = getelementptr inbounds %CausalSelfAttention, ptr %field_val565, i32 0, i32 0
  %field_val567 = load ptr, ptr %unreg_field_0566, align 8
  call void @tl_mem_unregister(ptr %field_val567)
  %unreg_field_0568 = getelementptr inbounds %Linear, ptr %field_val567, i32 0, i32 0
  %field_val569 = load ptr, ptr %unreg_field_0568, align 8
  call void @tl_mem_unregister(ptr %field_val569)
  %unreg_field_1570 = getelementptr inbounds %Linear, ptr %field_val567, i32 0, i32 1
  %field_val571 = load ptr, ptr %unreg_field_1570, align 8
  call void @tl_mem_unregister(ptr %field_val571)
  %unreg_field_1572 = getelementptr inbounds %CausalSelfAttention, ptr %field_val565, i32 0, i32 1
  %field_val573 = load ptr, ptr %unreg_field_1572, align 8
  call void @tl_mem_unregister(ptr %field_val573)
  %unreg_field_0574 = getelementptr inbounds %Linear, ptr %field_val573, i32 0, i32 0
  %field_val575 = load ptr, ptr %unreg_field_0574, align 8
  call void @tl_mem_unregister(ptr %field_val575)
  %unreg_field_1576 = getelementptr inbounds %Linear, ptr %field_val573, i32 0, i32 1
  %field_val577 = load ptr, ptr %unreg_field_1576, align 8
  call void @tl_mem_unregister(ptr %field_val577)
  %unreg_field_2578 = getelementptr inbounds %CausalSelfAttention, ptr %field_val565, i32 0, i32 2
  %field_val579 = load ptr, ptr %unreg_field_2578, align 8
  call void @tl_mem_unregister(ptr %field_val579)
  %unreg_field_0580 = getelementptr inbounds %Linear, ptr %field_val579, i32 0, i32 0
  %field_val581 = load ptr, ptr %unreg_field_0580, align 8
  call void @tl_mem_unregister(ptr %field_val581)
  %unreg_field_1582 = getelementptr inbounds %Linear, ptr %field_val579, i32 0, i32 1
  %field_val583 = load ptr, ptr %unreg_field_1582, align 8
  call void @tl_mem_unregister(ptr %field_val583)
  %unreg_field_3584 = getelementptr inbounds %CausalSelfAttention, ptr %field_val565, i32 0, i32 3
  %field_val585 = load ptr, ptr %unreg_field_3584, align 8
  call void @tl_mem_unregister(ptr %field_val585)
  %unreg_field_0586 = getelementptr inbounds %Linear, ptr %field_val585, i32 0, i32 0
  %field_val587 = load ptr, ptr %unreg_field_0586, align 8
  call void @tl_mem_unregister(ptr %field_val587)
  %unreg_field_1588 = getelementptr inbounds %Linear, ptr %field_val585, i32 0, i32 1
  %field_val589 = load ptr, ptr %unreg_field_1588, align 8
  call void @tl_mem_unregister(ptr %field_val589)
  %unreg_field_2590 = getelementptr inbounds %Block, ptr %field_val557, i32 0, i32 2
  %field_val591 = load ptr, ptr %unreg_field_2590, align 8
  call void @tl_mem_unregister(ptr %field_val591)
  %unreg_field_0592 = getelementptr inbounds %LayerNorm, ptr %field_val591, i32 0, i32 0
  %field_val593 = load ptr, ptr %unreg_field_0592, align 8
  call void @tl_mem_unregister(ptr %field_val593)
  %unreg_field_1594 = getelementptr inbounds %LayerNorm, ptr %field_val591, i32 0, i32 1
  %field_val595 = load ptr, ptr %unreg_field_1594, align 8
  call void @tl_mem_unregister(ptr %field_val595)
  %unreg_field_3596 = getelementptr inbounds %Block, ptr %field_val557, i32 0, i32 3
  %field_val597 = load ptr, ptr %unreg_field_3596, align 8
  call void @tl_mem_unregister(ptr %field_val597)
  %unreg_field_0598 = getelementptr inbounds %MLP, ptr %field_val597, i32 0, i32 0
  %field_val599 = load ptr, ptr %unreg_field_0598, align 8
  call void @tl_mem_unregister(ptr %field_val599)
  %unreg_field_0600 = getelementptr inbounds %Linear, ptr %field_val599, i32 0, i32 0
  %field_val601 = load ptr, ptr %unreg_field_0600, align 8
  call void @tl_mem_unregister(ptr %field_val601)
  %unreg_field_1602 = getelementptr inbounds %Linear, ptr %field_val599, i32 0, i32 1
  %field_val603 = load ptr, ptr %unreg_field_1602, align 8
  call void @tl_mem_unregister(ptr %field_val603)
  %unreg_field_1604 = getelementptr inbounds %MLP, ptr %field_val597, i32 0, i32 1
  %field_val605 = load ptr, ptr %unreg_field_1604, align 8
  call void @tl_mem_unregister(ptr %field_val605)
  %unreg_field_0606 = getelementptr inbounds %Linear, ptr %field_val605, i32 0, i32 0
  %field_val607 = load ptr, ptr %unreg_field_0606, align 8
  call void @tl_mem_unregister(ptr %field_val607)
  %unreg_field_1608 = getelementptr inbounds %Linear, ptr %field_val605, i32 0, i32 1
  %field_val609 = load ptr, ptr %unreg_field_1608, align 8
  call void @tl_mem_unregister(ptr %field_val609)
  %unreg_field_6 = getelementptr inbounds %GPTHeavy, ptr %model392, i32 0, i32 6
  %field_val610 = load ptr, ptr %unreg_field_6, align 8
  call void @tl_mem_unregister(ptr %field_val610)
  %unreg_field_0611 = getelementptr inbounds %LayerNorm, ptr %field_val610, i32 0, i32 0
  %field_val612 = load ptr, ptr %unreg_field_0611, align 8
  call void @tl_mem_unregister(ptr %field_val612)
  %unreg_field_1613 = getelementptr inbounds %LayerNorm, ptr %field_val610, i32 0, i32 1
  %field_val614 = load ptr, ptr %unreg_field_1613, align 8
  call void @tl_mem_unregister(ptr %field_val614)
  %unreg_field_7 = getelementptr inbounds %GPTHeavy, ptr %model392, i32 0, i32 7
  %field_val615 = load ptr, ptr %unreg_field_7, align 8
  call void @tl_mem_unregister(ptr %field_val615)
  %unreg_field_0616 = getelementptr inbounds %Linear, ptr %field_val615, i32 0, i32 0
  %field_val617 = load ptr, ptr %unreg_field_0616, align 8
  call void @tl_mem_unregister(ptr %field_val617)
  %unreg_field_1618 = getelementptr inbounds %Linear, ptr %field_val615, i32 0, i32 1
  %field_val619 = load ptr, ptr %unreg_field_1618, align 8
  call void @tl_mem_unregister(ptr %field_val619)
  %tensor_to_free = load ptr, ptr %Y, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free620 = load ptr, ptr %target, align 8
  call void @tl_tensor_free(ptr %tensor_to_free620)
  %tensor_to_free621 = load ptr, ptr %logits, align 8
  call void @tl_tensor_free(ptr %tensor_to_free621)
  %tensor_to_free622 = load ptr, ptr %Y_flat, align 8
  call void @tl_tensor_free(ptr %tensor_to_free622)
  %tensor_to_free623 = load ptr, ptr %loss, align 8
  call void @tl_tensor_free(ptr %tensor_to_free623)
  %tensor_to_free624 = load ptr, ptr %data, align 8
  call void @tl_tensor_free(ptr %tensor_to_free624)
  %tensor_to_free625 = load ptr, ptr %logits_flat, align 8
  call void @tl_tensor_free(ptr %tensor_to_free625)
  %tensor_to_free626 = load ptr, ptr %X, align 8
  call void @tl_tensor_free(ptr %tensor_to_free626)
  call void @tl_mem_exit_scope()
  ret ptr %model392
}

define ptr @train_epoch(ptr %model, float %lr, i64 %epoch) {
entry:
  %j = alloca i64, align 16
  %i = alloca i64, align 16
  %idx = alloca i64, align 16
  %raw = alloca i64, align 16
  %s = alloca i64, align 16
  %offset = alloca i64, align 16
  %stride = alloca i64, align 16
  %total_steps = alloca i64, align 16
  %epoch3 = alloca i64, align 16
  %lr2 = alloca float, align 16
  %model1 = alloca ptr, align 16
  call void @tl_mem_enter_scope()
  store ptr %model, ptr %model1, align 8
  store float %lr, ptr %lr2, align 4
  store i64 %epoch, ptr %epoch3, align 8
  store i64 200, ptr %total_steps, align 8
  store i64 149, ptr %stride, align 8
  %epoch4 = load i64, ptr %epoch3, align 8
  %multmp = mul i64 %epoch4, 37
  store i64 %multmp, ptr %offset, align 8
  %total_steps5 = load i64, ptr %total_steps, align 8
  br label %for_header

for_header:                                       ; preds = %continue_after_free, %entry
  %for_idx = phi i64 [ %next_idx, %continue_after_free ], [ 0, %entry ]
  %for_cond = icmp slt i64 %for_idx, %total_steps5
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %s, align 8
  %s6 = load i64, ptr %s, align 8
  %stride7 = load i64, ptr %stride, align 8
  %multmp8 = mul i64 %s6, %stride7
  %offset9 = load i64, ptr %offset, align 8
  %addtmp = add i64 %multmp8, %offset9
  store i64 %addtmp, ptr %raw, align 8
  %raw10 = load i64, ptr %raw, align 8
  %raw11 = load i64, ptr %raw, align 8
  %divtmp = sdiv i64 %raw11, 10000
  %multmp12 = mul i64 %divtmp, 10000
  %subtmp = sub i64 %raw10, %multmp12
  store i64 %subtmp, ptr %idx, align 8
  %idx13 = load i64, ptr %idx, align 8
  %divtmp14 = sdiv i64 %idx13, 100
  store i64 %divtmp14, ptr %i, align 8
  %idx15 = load i64, ptr %idx, align 8
  %idx16 = load i64, ptr %idx, align 8
  %divtmp17 = sdiv i64 %idx16, 100
  %multmp18 = mul i64 %divtmp17, 100
  %subtmp19 = sub i64 %idx15, %multmp18
  store i64 %subtmp19, ptr %j, align 8
  %model20 = load ptr, ptr %model1, align 8
  %lr21 = load float, ptr %lr2, align 4
  %i22 = load i64, ptr %i, align 8
  %j23 = load i64, ptr %j, align 8
  %call_tmp = call ptr @train_step(ptr %model20, float %lr21, i64 %i22, i64 %j23)
  call void @tl_mem_unregister(ptr %call_tmp)
  %old_struct_to_free = load ptr, ptr %model1, align 8
  %is_not_null = icmp ne ptr %old_struct_to_free, null
  %are_diff = icmp ne ptr %old_struct_to_free, %call_tmp
  %can_free_1 = and i1 %is_not_null, false
  %can_free = and i1 %can_free_1, %are_diff
  br i1 %can_free, label %free_struct, label %continue_after_free

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal)
  %epoch485 = load i64, ptr %epoch3, align 8
  call void @tl_print_i64(i64 %epoch485)
  %model486 = load ptr, ptr %model1, align 8
  call void @tl_mem_unregister(ptr %model486)
  %unreg_field_0487 = getelementptr inbounds %GPTHeavy, ptr %model486, i32 0, i32 0
  %field_val488 = load ptr, ptr %unreg_field_0487, align 8
  call void @tl_mem_unregister(ptr %field_val488)
  %unreg_field_0489 = getelementptr inbounds %Embedding, ptr %field_val488, i32 0, i32 0
  %field_val490 = load ptr, ptr %unreg_field_0489, align 8
  call void @tl_mem_unregister(ptr %field_val490)
  %unreg_field_1491 = getelementptr inbounds %GPTHeavy, ptr %model486, i32 0, i32 1
  %field_val492 = load ptr, ptr %unreg_field_1491, align 8
  call void @tl_mem_unregister(ptr %field_val492)
  %unreg_field_0493 = getelementptr inbounds %Embedding, ptr %field_val492, i32 0, i32 0
  %field_val494 = load ptr, ptr %unreg_field_0493, align 8
  call void @tl_mem_unregister(ptr %field_val494)
  %unreg_field_2495 = getelementptr inbounds %GPTHeavy, ptr %model486, i32 0, i32 2
  %field_val496 = load ptr, ptr %unreg_field_2495, align 8
  call void @tl_mem_unregister(ptr %field_val496)
  %unreg_field_0497 = getelementptr inbounds %Block, ptr %field_val496, i32 0, i32 0
  %field_val498 = load ptr, ptr %unreg_field_0497, align 8
  call void @tl_mem_unregister(ptr %field_val498)
  %unreg_field_0499 = getelementptr inbounds %LayerNorm, ptr %field_val498, i32 0, i32 0
  %field_val500 = load ptr, ptr %unreg_field_0499, align 8
  call void @tl_mem_unregister(ptr %field_val500)
  %unreg_field_1501 = getelementptr inbounds %LayerNorm, ptr %field_val498, i32 0, i32 1
  %field_val502 = load ptr, ptr %unreg_field_1501, align 8
  call void @tl_mem_unregister(ptr %field_val502)
  %unreg_field_1503 = getelementptr inbounds %Block, ptr %field_val496, i32 0, i32 1
  %field_val504 = load ptr, ptr %unreg_field_1503, align 8
  call void @tl_mem_unregister(ptr %field_val504)
  %unreg_field_0505 = getelementptr inbounds %CausalSelfAttention, ptr %field_val504, i32 0, i32 0
  %field_val506 = load ptr, ptr %unreg_field_0505, align 8
  call void @tl_mem_unregister(ptr %field_val506)
  %unreg_field_0507 = getelementptr inbounds %Linear, ptr %field_val506, i32 0, i32 0
  %field_val508 = load ptr, ptr %unreg_field_0507, align 8
  call void @tl_mem_unregister(ptr %field_val508)
  %unreg_field_1509 = getelementptr inbounds %Linear, ptr %field_val506, i32 0, i32 1
  %field_val510 = load ptr, ptr %unreg_field_1509, align 8
  call void @tl_mem_unregister(ptr %field_val510)
  %unreg_field_1511 = getelementptr inbounds %CausalSelfAttention, ptr %field_val504, i32 0, i32 1
  %field_val512 = load ptr, ptr %unreg_field_1511, align 8
  call void @tl_mem_unregister(ptr %field_val512)
  %unreg_field_0513 = getelementptr inbounds %Linear, ptr %field_val512, i32 0, i32 0
  %field_val514 = load ptr, ptr %unreg_field_0513, align 8
  call void @tl_mem_unregister(ptr %field_val514)
  %unreg_field_1515 = getelementptr inbounds %Linear, ptr %field_val512, i32 0, i32 1
  %field_val516 = load ptr, ptr %unreg_field_1515, align 8
  call void @tl_mem_unregister(ptr %field_val516)
  %unreg_field_2517 = getelementptr inbounds %CausalSelfAttention, ptr %field_val504, i32 0, i32 2
  %field_val518 = load ptr, ptr %unreg_field_2517, align 8
  call void @tl_mem_unregister(ptr %field_val518)
  %unreg_field_0519 = getelementptr inbounds %Linear, ptr %field_val518, i32 0, i32 0
  %field_val520 = load ptr, ptr %unreg_field_0519, align 8
  call void @tl_mem_unregister(ptr %field_val520)
  %unreg_field_1521 = getelementptr inbounds %Linear, ptr %field_val518, i32 0, i32 1
  %field_val522 = load ptr, ptr %unreg_field_1521, align 8
  call void @tl_mem_unregister(ptr %field_val522)
  %unreg_field_3523 = getelementptr inbounds %CausalSelfAttention, ptr %field_val504, i32 0, i32 3
  %field_val524 = load ptr, ptr %unreg_field_3523, align 8
  call void @tl_mem_unregister(ptr %field_val524)
  %unreg_field_0525 = getelementptr inbounds %Linear, ptr %field_val524, i32 0, i32 0
  %field_val526 = load ptr, ptr %unreg_field_0525, align 8
  call void @tl_mem_unregister(ptr %field_val526)
  %unreg_field_1527 = getelementptr inbounds %Linear, ptr %field_val524, i32 0, i32 1
  %field_val528 = load ptr, ptr %unreg_field_1527, align 8
  call void @tl_mem_unregister(ptr %field_val528)
  %unreg_field_2529 = getelementptr inbounds %Block, ptr %field_val496, i32 0, i32 2
  %field_val530 = load ptr, ptr %unreg_field_2529, align 8
  call void @tl_mem_unregister(ptr %field_val530)
  %unreg_field_0531 = getelementptr inbounds %LayerNorm, ptr %field_val530, i32 0, i32 0
  %field_val532 = load ptr, ptr %unreg_field_0531, align 8
  call void @tl_mem_unregister(ptr %field_val532)
  %unreg_field_1533 = getelementptr inbounds %LayerNorm, ptr %field_val530, i32 0, i32 1
  %field_val534 = load ptr, ptr %unreg_field_1533, align 8
  call void @tl_mem_unregister(ptr %field_val534)
  %unreg_field_3535 = getelementptr inbounds %Block, ptr %field_val496, i32 0, i32 3
  %field_val536 = load ptr, ptr %unreg_field_3535, align 8
  call void @tl_mem_unregister(ptr %field_val536)
  %unreg_field_0537 = getelementptr inbounds %MLP, ptr %field_val536, i32 0, i32 0
  %field_val538 = load ptr, ptr %unreg_field_0537, align 8
  call void @tl_mem_unregister(ptr %field_val538)
  %unreg_field_0539 = getelementptr inbounds %Linear, ptr %field_val538, i32 0, i32 0
  %field_val540 = load ptr, ptr %unreg_field_0539, align 8
  call void @tl_mem_unregister(ptr %field_val540)
  %unreg_field_1541 = getelementptr inbounds %Linear, ptr %field_val538, i32 0, i32 1
  %field_val542 = load ptr, ptr %unreg_field_1541, align 8
  call void @tl_mem_unregister(ptr %field_val542)
  %unreg_field_1543 = getelementptr inbounds %MLP, ptr %field_val536, i32 0, i32 1
  %field_val544 = load ptr, ptr %unreg_field_1543, align 8
  call void @tl_mem_unregister(ptr %field_val544)
  %unreg_field_0545 = getelementptr inbounds %Linear, ptr %field_val544, i32 0, i32 0
  %field_val546 = load ptr, ptr %unreg_field_0545, align 8
  call void @tl_mem_unregister(ptr %field_val546)
  %unreg_field_1547 = getelementptr inbounds %Linear, ptr %field_val544, i32 0, i32 1
  %field_val548 = load ptr, ptr %unreg_field_1547, align 8
  call void @tl_mem_unregister(ptr %field_val548)
  %unreg_field_3549 = getelementptr inbounds %GPTHeavy, ptr %model486, i32 0, i32 3
  %field_val550 = load ptr, ptr %unreg_field_3549, align 8
  call void @tl_mem_unregister(ptr %field_val550)
  %unreg_field_0551 = getelementptr inbounds %Block, ptr %field_val550, i32 0, i32 0
  %field_val552 = load ptr, ptr %unreg_field_0551, align 8
  call void @tl_mem_unregister(ptr %field_val552)
  %unreg_field_0553 = getelementptr inbounds %LayerNorm, ptr %field_val552, i32 0, i32 0
  %field_val554 = load ptr, ptr %unreg_field_0553, align 8
  call void @tl_mem_unregister(ptr %field_val554)
  %unreg_field_1555 = getelementptr inbounds %LayerNorm, ptr %field_val552, i32 0, i32 1
  %field_val556 = load ptr, ptr %unreg_field_1555, align 8
  call void @tl_mem_unregister(ptr %field_val556)
  %unreg_field_1557 = getelementptr inbounds %Block, ptr %field_val550, i32 0, i32 1
  %field_val558 = load ptr, ptr %unreg_field_1557, align 8
  call void @tl_mem_unregister(ptr %field_val558)
  %unreg_field_0559 = getelementptr inbounds %CausalSelfAttention, ptr %field_val558, i32 0, i32 0
  %field_val560 = load ptr, ptr %unreg_field_0559, align 8
  call void @tl_mem_unregister(ptr %field_val560)
  %unreg_field_0561 = getelementptr inbounds %Linear, ptr %field_val560, i32 0, i32 0
  %field_val562 = load ptr, ptr %unreg_field_0561, align 8
  call void @tl_mem_unregister(ptr %field_val562)
  %unreg_field_1563 = getelementptr inbounds %Linear, ptr %field_val560, i32 0, i32 1
  %field_val564 = load ptr, ptr %unreg_field_1563, align 8
  call void @tl_mem_unregister(ptr %field_val564)
  %unreg_field_1565 = getelementptr inbounds %CausalSelfAttention, ptr %field_val558, i32 0, i32 1
  %field_val566 = load ptr, ptr %unreg_field_1565, align 8
  call void @tl_mem_unregister(ptr %field_val566)
  %unreg_field_0567 = getelementptr inbounds %Linear, ptr %field_val566, i32 0, i32 0
  %field_val568 = load ptr, ptr %unreg_field_0567, align 8
  call void @tl_mem_unregister(ptr %field_val568)
  %unreg_field_1569 = getelementptr inbounds %Linear, ptr %field_val566, i32 0, i32 1
  %field_val570 = load ptr, ptr %unreg_field_1569, align 8
  call void @tl_mem_unregister(ptr %field_val570)
  %unreg_field_2571 = getelementptr inbounds %CausalSelfAttention, ptr %field_val558, i32 0, i32 2
  %field_val572 = load ptr, ptr %unreg_field_2571, align 8
  call void @tl_mem_unregister(ptr %field_val572)
  %unreg_field_0573 = getelementptr inbounds %Linear, ptr %field_val572, i32 0, i32 0
  %field_val574 = load ptr, ptr %unreg_field_0573, align 8
  call void @tl_mem_unregister(ptr %field_val574)
  %unreg_field_1575 = getelementptr inbounds %Linear, ptr %field_val572, i32 0, i32 1
  %field_val576 = load ptr, ptr %unreg_field_1575, align 8
  call void @tl_mem_unregister(ptr %field_val576)
  %unreg_field_3577 = getelementptr inbounds %CausalSelfAttention, ptr %field_val558, i32 0, i32 3
  %field_val578 = load ptr, ptr %unreg_field_3577, align 8
  call void @tl_mem_unregister(ptr %field_val578)
  %unreg_field_0579 = getelementptr inbounds %Linear, ptr %field_val578, i32 0, i32 0
  %field_val580 = load ptr, ptr %unreg_field_0579, align 8
  call void @tl_mem_unregister(ptr %field_val580)
  %unreg_field_1581 = getelementptr inbounds %Linear, ptr %field_val578, i32 0, i32 1
  %field_val582 = load ptr, ptr %unreg_field_1581, align 8
  call void @tl_mem_unregister(ptr %field_val582)
  %unreg_field_2583 = getelementptr inbounds %Block, ptr %field_val550, i32 0, i32 2
  %field_val584 = load ptr, ptr %unreg_field_2583, align 8
  call void @tl_mem_unregister(ptr %field_val584)
  %unreg_field_0585 = getelementptr inbounds %LayerNorm, ptr %field_val584, i32 0, i32 0
  %field_val586 = load ptr, ptr %unreg_field_0585, align 8
  call void @tl_mem_unregister(ptr %field_val586)
  %unreg_field_1587 = getelementptr inbounds %LayerNorm, ptr %field_val584, i32 0, i32 1
  %field_val588 = load ptr, ptr %unreg_field_1587, align 8
  call void @tl_mem_unregister(ptr %field_val588)
  %unreg_field_3589 = getelementptr inbounds %Block, ptr %field_val550, i32 0, i32 3
  %field_val590 = load ptr, ptr %unreg_field_3589, align 8
  call void @tl_mem_unregister(ptr %field_val590)
  %unreg_field_0591 = getelementptr inbounds %MLP, ptr %field_val590, i32 0, i32 0
  %field_val592 = load ptr, ptr %unreg_field_0591, align 8
  call void @tl_mem_unregister(ptr %field_val592)
  %unreg_field_0593 = getelementptr inbounds %Linear, ptr %field_val592, i32 0, i32 0
  %field_val594 = load ptr, ptr %unreg_field_0593, align 8
  call void @tl_mem_unregister(ptr %field_val594)
  %unreg_field_1595 = getelementptr inbounds %Linear, ptr %field_val592, i32 0, i32 1
  %field_val596 = load ptr, ptr %unreg_field_1595, align 8
  call void @tl_mem_unregister(ptr %field_val596)
  %unreg_field_1597 = getelementptr inbounds %MLP, ptr %field_val590, i32 0, i32 1
  %field_val598 = load ptr, ptr %unreg_field_1597, align 8
  call void @tl_mem_unregister(ptr %field_val598)
  %unreg_field_0599 = getelementptr inbounds %Linear, ptr %field_val598, i32 0, i32 0
  %field_val600 = load ptr, ptr %unreg_field_0599, align 8
  call void @tl_mem_unregister(ptr %field_val600)
  %unreg_field_1601 = getelementptr inbounds %Linear, ptr %field_val598, i32 0, i32 1
  %field_val602 = load ptr, ptr %unreg_field_1601, align 8
  call void @tl_mem_unregister(ptr %field_val602)
  %unreg_field_4603 = getelementptr inbounds %GPTHeavy, ptr %model486, i32 0, i32 4
  %field_val604 = load ptr, ptr %unreg_field_4603, align 8
  call void @tl_mem_unregister(ptr %field_val604)
  %unreg_field_0605 = getelementptr inbounds %Block, ptr %field_val604, i32 0, i32 0
  %field_val606 = load ptr, ptr %unreg_field_0605, align 8
  call void @tl_mem_unregister(ptr %field_val606)
  %unreg_field_0607 = getelementptr inbounds %LayerNorm, ptr %field_val606, i32 0, i32 0
  %field_val608 = load ptr, ptr %unreg_field_0607, align 8
  call void @tl_mem_unregister(ptr %field_val608)
  %unreg_field_1609 = getelementptr inbounds %LayerNorm, ptr %field_val606, i32 0, i32 1
  %field_val610 = load ptr, ptr %unreg_field_1609, align 8
  call void @tl_mem_unregister(ptr %field_val610)
  %unreg_field_1611 = getelementptr inbounds %Block, ptr %field_val604, i32 0, i32 1
  %field_val612 = load ptr, ptr %unreg_field_1611, align 8
  call void @tl_mem_unregister(ptr %field_val612)
  %unreg_field_0613 = getelementptr inbounds %CausalSelfAttention, ptr %field_val612, i32 0, i32 0
  %field_val614 = load ptr, ptr %unreg_field_0613, align 8
  call void @tl_mem_unregister(ptr %field_val614)
  %unreg_field_0615 = getelementptr inbounds %Linear, ptr %field_val614, i32 0, i32 0
  %field_val616 = load ptr, ptr %unreg_field_0615, align 8
  call void @tl_mem_unregister(ptr %field_val616)
  %unreg_field_1617 = getelementptr inbounds %Linear, ptr %field_val614, i32 0, i32 1
  %field_val618 = load ptr, ptr %unreg_field_1617, align 8
  call void @tl_mem_unregister(ptr %field_val618)
  %unreg_field_1619 = getelementptr inbounds %CausalSelfAttention, ptr %field_val612, i32 0, i32 1
  %field_val620 = load ptr, ptr %unreg_field_1619, align 8
  call void @tl_mem_unregister(ptr %field_val620)
  %unreg_field_0621 = getelementptr inbounds %Linear, ptr %field_val620, i32 0, i32 0
  %field_val622 = load ptr, ptr %unreg_field_0621, align 8
  call void @tl_mem_unregister(ptr %field_val622)
  %unreg_field_1623 = getelementptr inbounds %Linear, ptr %field_val620, i32 0, i32 1
  %field_val624 = load ptr, ptr %unreg_field_1623, align 8
  call void @tl_mem_unregister(ptr %field_val624)
  %unreg_field_2625 = getelementptr inbounds %CausalSelfAttention, ptr %field_val612, i32 0, i32 2
  %field_val626 = load ptr, ptr %unreg_field_2625, align 8
  call void @tl_mem_unregister(ptr %field_val626)
  %unreg_field_0627 = getelementptr inbounds %Linear, ptr %field_val626, i32 0, i32 0
  %field_val628 = load ptr, ptr %unreg_field_0627, align 8
  call void @tl_mem_unregister(ptr %field_val628)
  %unreg_field_1629 = getelementptr inbounds %Linear, ptr %field_val626, i32 0, i32 1
  %field_val630 = load ptr, ptr %unreg_field_1629, align 8
  call void @tl_mem_unregister(ptr %field_val630)
  %unreg_field_3631 = getelementptr inbounds %CausalSelfAttention, ptr %field_val612, i32 0, i32 3
  %field_val632 = load ptr, ptr %unreg_field_3631, align 8
  call void @tl_mem_unregister(ptr %field_val632)
  %unreg_field_0633 = getelementptr inbounds %Linear, ptr %field_val632, i32 0, i32 0
  %field_val634 = load ptr, ptr %unreg_field_0633, align 8
  call void @tl_mem_unregister(ptr %field_val634)
  %unreg_field_1635 = getelementptr inbounds %Linear, ptr %field_val632, i32 0, i32 1
  %field_val636 = load ptr, ptr %unreg_field_1635, align 8
  call void @tl_mem_unregister(ptr %field_val636)
  %unreg_field_2637 = getelementptr inbounds %Block, ptr %field_val604, i32 0, i32 2
  %field_val638 = load ptr, ptr %unreg_field_2637, align 8
  call void @tl_mem_unregister(ptr %field_val638)
  %unreg_field_0639 = getelementptr inbounds %LayerNorm, ptr %field_val638, i32 0, i32 0
  %field_val640 = load ptr, ptr %unreg_field_0639, align 8
  call void @tl_mem_unregister(ptr %field_val640)
  %unreg_field_1641 = getelementptr inbounds %LayerNorm, ptr %field_val638, i32 0, i32 1
  %field_val642 = load ptr, ptr %unreg_field_1641, align 8
  call void @tl_mem_unregister(ptr %field_val642)
  %unreg_field_3643 = getelementptr inbounds %Block, ptr %field_val604, i32 0, i32 3
  %field_val644 = load ptr, ptr %unreg_field_3643, align 8
  call void @tl_mem_unregister(ptr %field_val644)
  %unreg_field_0645 = getelementptr inbounds %MLP, ptr %field_val644, i32 0, i32 0
  %field_val646 = load ptr, ptr %unreg_field_0645, align 8
  call void @tl_mem_unregister(ptr %field_val646)
  %unreg_field_0647 = getelementptr inbounds %Linear, ptr %field_val646, i32 0, i32 0
  %field_val648 = load ptr, ptr %unreg_field_0647, align 8
  call void @tl_mem_unregister(ptr %field_val648)
  %unreg_field_1649 = getelementptr inbounds %Linear, ptr %field_val646, i32 0, i32 1
  %field_val650 = load ptr, ptr %unreg_field_1649, align 8
  call void @tl_mem_unregister(ptr %field_val650)
  %unreg_field_1651 = getelementptr inbounds %MLP, ptr %field_val644, i32 0, i32 1
  %field_val652 = load ptr, ptr %unreg_field_1651, align 8
  call void @tl_mem_unregister(ptr %field_val652)
  %unreg_field_0653 = getelementptr inbounds %Linear, ptr %field_val652, i32 0, i32 0
  %field_val654 = load ptr, ptr %unreg_field_0653, align 8
  call void @tl_mem_unregister(ptr %field_val654)
  %unreg_field_1655 = getelementptr inbounds %Linear, ptr %field_val652, i32 0, i32 1
  %field_val656 = load ptr, ptr %unreg_field_1655, align 8
  call void @tl_mem_unregister(ptr %field_val656)
  %unreg_field_5657 = getelementptr inbounds %GPTHeavy, ptr %model486, i32 0, i32 5
  %field_val658 = load ptr, ptr %unreg_field_5657, align 8
  call void @tl_mem_unregister(ptr %field_val658)
  %unreg_field_0659 = getelementptr inbounds %Block, ptr %field_val658, i32 0, i32 0
  %field_val660 = load ptr, ptr %unreg_field_0659, align 8
  call void @tl_mem_unregister(ptr %field_val660)
  %unreg_field_0661 = getelementptr inbounds %LayerNorm, ptr %field_val660, i32 0, i32 0
  %field_val662 = load ptr, ptr %unreg_field_0661, align 8
  call void @tl_mem_unregister(ptr %field_val662)
  %unreg_field_1663 = getelementptr inbounds %LayerNorm, ptr %field_val660, i32 0, i32 1
  %field_val664 = load ptr, ptr %unreg_field_1663, align 8
  call void @tl_mem_unregister(ptr %field_val664)
  %unreg_field_1665 = getelementptr inbounds %Block, ptr %field_val658, i32 0, i32 1
  %field_val666 = load ptr, ptr %unreg_field_1665, align 8
  call void @tl_mem_unregister(ptr %field_val666)
  %unreg_field_0667 = getelementptr inbounds %CausalSelfAttention, ptr %field_val666, i32 0, i32 0
  %field_val668 = load ptr, ptr %unreg_field_0667, align 8
  call void @tl_mem_unregister(ptr %field_val668)
  %unreg_field_0669 = getelementptr inbounds %Linear, ptr %field_val668, i32 0, i32 0
  %field_val670 = load ptr, ptr %unreg_field_0669, align 8
  call void @tl_mem_unregister(ptr %field_val670)
  %unreg_field_1671 = getelementptr inbounds %Linear, ptr %field_val668, i32 0, i32 1
  %field_val672 = load ptr, ptr %unreg_field_1671, align 8
  call void @tl_mem_unregister(ptr %field_val672)
  %unreg_field_1673 = getelementptr inbounds %CausalSelfAttention, ptr %field_val666, i32 0, i32 1
  %field_val674 = load ptr, ptr %unreg_field_1673, align 8
  call void @tl_mem_unregister(ptr %field_val674)
  %unreg_field_0675 = getelementptr inbounds %Linear, ptr %field_val674, i32 0, i32 0
  %field_val676 = load ptr, ptr %unreg_field_0675, align 8
  call void @tl_mem_unregister(ptr %field_val676)
  %unreg_field_1677 = getelementptr inbounds %Linear, ptr %field_val674, i32 0, i32 1
  %field_val678 = load ptr, ptr %unreg_field_1677, align 8
  call void @tl_mem_unregister(ptr %field_val678)
  %unreg_field_2679 = getelementptr inbounds %CausalSelfAttention, ptr %field_val666, i32 0, i32 2
  %field_val680 = load ptr, ptr %unreg_field_2679, align 8
  call void @tl_mem_unregister(ptr %field_val680)
  %unreg_field_0681 = getelementptr inbounds %Linear, ptr %field_val680, i32 0, i32 0
  %field_val682 = load ptr, ptr %unreg_field_0681, align 8
  call void @tl_mem_unregister(ptr %field_val682)
  %unreg_field_1683 = getelementptr inbounds %Linear, ptr %field_val680, i32 0, i32 1
  %field_val684 = load ptr, ptr %unreg_field_1683, align 8
  call void @tl_mem_unregister(ptr %field_val684)
  %unreg_field_3685 = getelementptr inbounds %CausalSelfAttention, ptr %field_val666, i32 0, i32 3
  %field_val686 = load ptr, ptr %unreg_field_3685, align 8
  call void @tl_mem_unregister(ptr %field_val686)
  %unreg_field_0687 = getelementptr inbounds %Linear, ptr %field_val686, i32 0, i32 0
  %field_val688 = load ptr, ptr %unreg_field_0687, align 8
  call void @tl_mem_unregister(ptr %field_val688)
  %unreg_field_1689 = getelementptr inbounds %Linear, ptr %field_val686, i32 0, i32 1
  %field_val690 = load ptr, ptr %unreg_field_1689, align 8
  call void @tl_mem_unregister(ptr %field_val690)
  %unreg_field_2691 = getelementptr inbounds %Block, ptr %field_val658, i32 0, i32 2
  %field_val692 = load ptr, ptr %unreg_field_2691, align 8
  call void @tl_mem_unregister(ptr %field_val692)
  %unreg_field_0693 = getelementptr inbounds %LayerNorm, ptr %field_val692, i32 0, i32 0
  %field_val694 = load ptr, ptr %unreg_field_0693, align 8
  call void @tl_mem_unregister(ptr %field_val694)
  %unreg_field_1695 = getelementptr inbounds %LayerNorm, ptr %field_val692, i32 0, i32 1
  %field_val696 = load ptr, ptr %unreg_field_1695, align 8
  call void @tl_mem_unregister(ptr %field_val696)
  %unreg_field_3697 = getelementptr inbounds %Block, ptr %field_val658, i32 0, i32 3
  %field_val698 = load ptr, ptr %unreg_field_3697, align 8
  call void @tl_mem_unregister(ptr %field_val698)
  %unreg_field_0699 = getelementptr inbounds %MLP, ptr %field_val698, i32 0, i32 0
  %field_val700 = load ptr, ptr %unreg_field_0699, align 8
  call void @tl_mem_unregister(ptr %field_val700)
  %unreg_field_0701 = getelementptr inbounds %Linear, ptr %field_val700, i32 0, i32 0
  %field_val702 = load ptr, ptr %unreg_field_0701, align 8
  call void @tl_mem_unregister(ptr %field_val702)
  %unreg_field_1703 = getelementptr inbounds %Linear, ptr %field_val700, i32 0, i32 1
  %field_val704 = load ptr, ptr %unreg_field_1703, align 8
  call void @tl_mem_unregister(ptr %field_val704)
  %unreg_field_1705 = getelementptr inbounds %MLP, ptr %field_val698, i32 0, i32 1
  %field_val706 = load ptr, ptr %unreg_field_1705, align 8
  call void @tl_mem_unregister(ptr %field_val706)
  %unreg_field_0707 = getelementptr inbounds %Linear, ptr %field_val706, i32 0, i32 0
  %field_val708 = load ptr, ptr %unreg_field_0707, align 8
  call void @tl_mem_unregister(ptr %field_val708)
  %unreg_field_1709 = getelementptr inbounds %Linear, ptr %field_val706, i32 0, i32 1
  %field_val710 = load ptr, ptr %unreg_field_1709, align 8
  call void @tl_mem_unregister(ptr %field_val710)
  %unreg_field_6711 = getelementptr inbounds %GPTHeavy, ptr %model486, i32 0, i32 6
  %field_val712 = load ptr, ptr %unreg_field_6711, align 8
  call void @tl_mem_unregister(ptr %field_val712)
  %unreg_field_0713 = getelementptr inbounds %LayerNorm, ptr %field_val712, i32 0, i32 0
  %field_val714 = load ptr, ptr %unreg_field_0713, align 8
  call void @tl_mem_unregister(ptr %field_val714)
  %unreg_field_1715 = getelementptr inbounds %LayerNorm, ptr %field_val712, i32 0, i32 1
  %field_val716 = load ptr, ptr %unreg_field_1715, align 8
  call void @tl_mem_unregister(ptr %field_val716)
  %unreg_field_7717 = getelementptr inbounds %GPTHeavy, ptr %model486, i32 0, i32 7
  %field_val718 = load ptr, ptr %unreg_field_7717, align 8
  call void @tl_mem_unregister(ptr %field_val718)
  %unreg_field_0719 = getelementptr inbounds %Linear, ptr %field_val718, i32 0, i32 0
  %field_val720 = load ptr, ptr %unreg_field_0719, align 8
  call void @tl_mem_unregister(ptr %field_val720)
  %unreg_field_1721 = getelementptr inbounds %Linear, ptr %field_val718, i32 0, i32 1
  %field_val722 = load ptr, ptr %unreg_field_1721, align 8
  call void @tl_mem_unregister(ptr %field_val722)
  call void @tl_mem_exit_scope()
  ret ptr %model486

free_struct:                                      ; preds = %for_body
  %field_gep = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 0
  %field_load = load ptr, ptr %field_gep, align 8
  %field_gep24 = getelementptr inbounds %Embedding, ptr %field_load, i32 0, i32 0
  %field_load25 = load ptr, ptr %field_gep24, align 8
  call void @tl_tensor_free(ptr %field_load25)
  %field_gep26 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 1
  %field_load27 = load ptr, ptr %field_gep26, align 8
  %field_gep28 = getelementptr inbounds %Embedding, ptr %field_load27, i32 0, i32 0
  %field_load29 = load ptr, ptr %field_gep28, align 8
  call void @tl_tensor_free(ptr %field_load29)
  %field_gep30 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 2
  %field_load31 = load ptr, ptr %field_gep30, align 8
  %field_gep32 = getelementptr inbounds %Block, ptr %field_load31, i32 0, i32 0
  %field_load33 = load ptr, ptr %field_gep32, align 8
  %field_gep34 = getelementptr inbounds %LayerNorm, ptr %field_load33, i32 0, i32 0
  %field_load35 = load ptr, ptr %field_gep34, align 8
  call void @tl_tensor_free(ptr %field_load35)
  %field_gep36 = getelementptr inbounds %LayerNorm, ptr %field_load33, i32 0, i32 1
  %field_load37 = load ptr, ptr %field_gep36, align 8
  call void @tl_tensor_free(ptr %field_load37)
  %field_gep38 = getelementptr inbounds %Block, ptr %field_load31, i32 0, i32 1
  %field_load39 = load ptr, ptr %field_gep38, align 8
  %field_gep40 = getelementptr inbounds %CausalSelfAttention, ptr %field_load39, i32 0, i32 0
  %field_load41 = load ptr, ptr %field_gep40, align 8
  %field_gep42 = getelementptr inbounds %Linear, ptr %field_load41, i32 0, i32 0
  %field_load43 = load ptr, ptr %field_gep42, align 8
  call void @tl_tensor_free(ptr %field_load43)
  %field_gep44 = getelementptr inbounds %Linear, ptr %field_load41, i32 0, i32 1
  %field_load45 = load ptr, ptr %field_gep44, align 8
  call void @tl_tensor_free(ptr %field_load45)
  %field_gep46 = getelementptr inbounds %CausalSelfAttention, ptr %field_load39, i32 0, i32 1
  %field_load47 = load ptr, ptr %field_gep46, align 8
  %field_gep48 = getelementptr inbounds %Linear, ptr %field_load47, i32 0, i32 0
  %field_load49 = load ptr, ptr %field_gep48, align 8
  call void @tl_tensor_free(ptr %field_load49)
  %field_gep50 = getelementptr inbounds %Linear, ptr %field_load47, i32 0, i32 1
  %field_load51 = load ptr, ptr %field_gep50, align 8
  call void @tl_tensor_free(ptr %field_load51)
  %field_gep52 = getelementptr inbounds %CausalSelfAttention, ptr %field_load39, i32 0, i32 2
  %field_load53 = load ptr, ptr %field_gep52, align 8
  %field_gep54 = getelementptr inbounds %Linear, ptr %field_load53, i32 0, i32 0
  %field_load55 = load ptr, ptr %field_gep54, align 8
  call void @tl_tensor_free(ptr %field_load55)
  %field_gep56 = getelementptr inbounds %Linear, ptr %field_load53, i32 0, i32 1
  %field_load57 = load ptr, ptr %field_gep56, align 8
  call void @tl_tensor_free(ptr %field_load57)
  %field_gep58 = getelementptr inbounds %CausalSelfAttention, ptr %field_load39, i32 0, i32 3
  %field_load59 = load ptr, ptr %field_gep58, align 8
  %field_gep60 = getelementptr inbounds %Linear, ptr %field_load59, i32 0, i32 0
  %field_load61 = load ptr, ptr %field_gep60, align 8
  call void @tl_tensor_free(ptr %field_load61)
  %field_gep62 = getelementptr inbounds %Linear, ptr %field_load59, i32 0, i32 1
  %field_load63 = load ptr, ptr %field_gep62, align 8
  call void @tl_tensor_free(ptr %field_load63)
  %field_gep64 = getelementptr inbounds %Block, ptr %field_load31, i32 0, i32 2
  %field_load65 = load ptr, ptr %field_gep64, align 8
  %field_gep66 = getelementptr inbounds %LayerNorm, ptr %field_load65, i32 0, i32 0
  %field_load67 = load ptr, ptr %field_gep66, align 8
  call void @tl_tensor_free(ptr %field_load67)
  %field_gep68 = getelementptr inbounds %LayerNorm, ptr %field_load65, i32 0, i32 1
  %field_load69 = load ptr, ptr %field_gep68, align 8
  call void @tl_tensor_free(ptr %field_load69)
  %field_gep70 = getelementptr inbounds %Block, ptr %field_load31, i32 0, i32 3
  %field_load71 = load ptr, ptr %field_gep70, align 8
  %field_gep72 = getelementptr inbounds %MLP, ptr %field_load71, i32 0, i32 0
  %field_load73 = load ptr, ptr %field_gep72, align 8
  %field_gep74 = getelementptr inbounds %Linear, ptr %field_load73, i32 0, i32 0
  %field_load75 = load ptr, ptr %field_gep74, align 8
  call void @tl_tensor_free(ptr %field_load75)
  %field_gep76 = getelementptr inbounds %Linear, ptr %field_load73, i32 0, i32 1
  %field_load77 = load ptr, ptr %field_gep76, align 8
  call void @tl_tensor_free(ptr %field_load77)
  %field_gep78 = getelementptr inbounds %MLP, ptr %field_load71, i32 0, i32 1
  %field_load79 = load ptr, ptr %field_gep78, align 8
  %field_gep80 = getelementptr inbounds %Linear, ptr %field_load79, i32 0, i32 0
  %field_load81 = load ptr, ptr %field_gep80, align 8
  call void @tl_tensor_free(ptr %field_load81)
  %field_gep82 = getelementptr inbounds %Linear, ptr %field_load79, i32 0, i32 1
  %field_load83 = load ptr, ptr %field_gep82, align 8
  call void @tl_tensor_free(ptr %field_load83)
  %field_gep84 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 3
  %field_load85 = load ptr, ptr %field_gep84, align 8
  %field_gep86 = getelementptr inbounds %Block, ptr %field_load85, i32 0, i32 0
  %field_load87 = load ptr, ptr %field_gep86, align 8
  %field_gep88 = getelementptr inbounds %LayerNorm, ptr %field_load87, i32 0, i32 0
  %field_load89 = load ptr, ptr %field_gep88, align 8
  call void @tl_tensor_free(ptr %field_load89)
  %field_gep90 = getelementptr inbounds %LayerNorm, ptr %field_load87, i32 0, i32 1
  %field_load91 = load ptr, ptr %field_gep90, align 8
  call void @tl_tensor_free(ptr %field_load91)
  %field_gep92 = getelementptr inbounds %Block, ptr %field_load85, i32 0, i32 1
  %field_load93 = load ptr, ptr %field_gep92, align 8
  %field_gep94 = getelementptr inbounds %CausalSelfAttention, ptr %field_load93, i32 0, i32 0
  %field_load95 = load ptr, ptr %field_gep94, align 8
  %field_gep96 = getelementptr inbounds %Linear, ptr %field_load95, i32 0, i32 0
  %field_load97 = load ptr, ptr %field_gep96, align 8
  call void @tl_tensor_free(ptr %field_load97)
  %field_gep98 = getelementptr inbounds %Linear, ptr %field_load95, i32 0, i32 1
  %field_load99 = load ptr, ptr %field_gep98, align 8
  call void @tl_tensor_free(ptr %field_load99)
  %field_gep100 = getelementptr inbounds %CausalSelfAttention, ptr %field_load93, i32 0, i32 1
  %field_load101 = load ptr, ptr %field_gep100, align 8
  %field_gep102 = getelementptr inbounds %Linear, ptr %field_load101, i32 0, i32 0
  %field_load103 = load ptr, ptr %field_gep102, align 8
  call void @tl_tensor_free(ptr %field_load103)
  %field_gep104 = getelementptr inbounds %Linear, ptr %field_load101, i32 0, i32 1
  %field_load105 = load ptr, ptr %field_gep104, align 8
  call void @tl_tensor_free(ptr %field_load105)
  %field_gep106 = getelementptr inbounds %CausalSelfAttention, ptr %field_load93, i32 0, i32 2
  %field_load107 = load ptr, ptr %field_gep106, align 8
  %field_gep108 = getelementptr inbounds %Linear, ptr %field_load107, i32 0, i32 0
  %field_load109 = load ptr, ptr %field_gep108, align 8
  call void @tl_tensor_free(ptr %field_load109)
  %field_gep110 = getelementptr inbounds %Linear, ptr %field_load107, i32 0, i32 1
  %field_load111 = load ptr, ptr %field_gep110, align 8
  call void @tl_tensor_free(ptr %field_load111)
  %field_gep112 = getelementptr inbounds %CausalSelfAttention, ptr %field_load93, i32 0, i32 3
  %field_load113 = load ptr, ptr %field_gep112, align 8
  %field_gep114 = getelementptr inbounds %Linear, ptr %field_load113, i32 0, i32 0
  %field_load115 = load ptr, ptr %field_gep114, align 8
  call void @tl_tensor_free(ptr %field_load115)
  %field_gep116 = getelementptr inbounds %Linear, ptr %field_load113, i32 0, i32 1
  %field_load117 = load ptr, ptr %field_gep116, align 8
  call void @tl_tensor_free(ptr %field_load117)
  %field_gep118 = getelementptr inbounds %Block, ptr %field_load85, i32 0, i32 2
  %field_load119 = load ptr, ptr %field_gep118, align 8
  %field_gep120 = getelementptr inbounds %LayerNorm, ptr %field_load119, i32 0, i32 0
  %field_load121 = load ptr, ptr %field_gep120, align 8
  call void @tl_tensor_free(ptr %field_load121)
  %field_gep122 = getelementptr inbounds %LayerNorm, ptr %field_load119, i32 0, i32 1
  %field_load123 = load ptr, ptr %field_gep122, align 8
  call void @tl_tensor_free(ptr %field_load123)
  %field_gep124 = getelementptr inbounds %Block, ptr %field_load85, i32 0, i32 3
  %field_load125 = load ptr, ptr %field_gep124, align 8
  %field_gep126 = getelementptr inbounds %MLP, ptr %field_load125, i32 0, i32 0
  %field_load127 = load ptr, ptr %field_gep126, align 8
  %field_gep128 = getelementptr inbounds %Linear, ptr %field_load127, i32 0, i32 0
  %field_load129 = load ptr, ptr %field_gep128, align 8
  call void @tl_tensor_free(ptr %field_load129)
  %field_gep130 = getelementptr inbounds %Linear, ptr %field_load127, i32 0, i32 1
  %field_load131 = load ptr, ptr %field_gep130, align 8
  call void @tl_tensor_free(ptr %field_load131)
  %field_gep132 = getelementptr inbounds %MLP, ptr %field_load125, i32 0, i32 1
  %field_load133 = load ptr, ptr %field_gep132, align 8
  %field_gep134 = getelementptr inbounds %Linear, ptr %field_load133, i32 0, i32 0
  %field_load135 = load ptr, ptr %field_gep134, align 8
  call void @tl_tensor_free(ptr %field_load135)
  %field_gep136 = getelementptr inbounds %Linear, ptr %field_load133, i32 0, i32 1
  %field_load137 = load ptr, ptr %field_gep136, align 8
  call void @tl_tensor_free(ptr %field_load137)
  %field_gep138 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 4
  %field_load139 = load ptr, ptr %field_gep138, align 8
  %field_gep140 = getelementptr inbounds %Block, ptr %field_load139, i32 0, i32 0
  %field_load141 = load ptr, ptr %field_gep140, align 8
  %field_gep142 = getelementptr inbounds %LayerNorm, ptr %field_load141, i32 0, i32 0
  %field_load143 = load ptr, ptr %field_gep142, align 8
  call void @tl_tensor_free(ptr %field_load143)
  %field_gep144 = getelementptr inbounds %LayerNorm, ptr %field_load141, i32 0, i32 1
  %field_load145 = load ptr, ptr %field_gep144, align 8
  call void @tl_tensor_free(ptr %field_load145)
  %field_gep146 = getelementptr inbounds %Block, ptr %field_load139, i32 0, i32 1
  %field_load147 = load ptr, ptr %field_gep146, align 8
  %field_gep148 = getelementptr inbounds %CausalSelfAttention, ptr %field_load147, i32 0, i32 0
  %field_load149 = load ptr, ptr %field_gep148, align 8
  %field_gep150 = getelementptr inbounds %Linear, ptr %field_load149, i32 0, i32 0
  %field_load151 = load ptr, ptr %field_gep150, align 8
  call void @tl_tensor_free(ptr %field_load151)
  %field_gep152 = getelementptr inbounds %Linear, ptr %field_load149, i32 0, i32 1
  %field_load153 = load ptr, ptr %field_gep152, align 8
  call void @tl_tensor_free(ptr %field_load153)
  %field_gep154 = getelementptr inbounds %CausalSelfAttention, ptr %field_load147, i32 0, i32 1
  %field_load155 = load ptr, ptr %field_gep154, align 8
  %field_gep156 = getelementptr inbounds %Linear, ptr %field_load155, i32 0, i32 0
  %field_load157 = load ptr, ptr %field_gep156, align 8
  call void @tl_tensor_free(ptr %field_load157)
  %field_gep158 = getelementptr inbounds %Linear, ptr %field_load155, i32 0, i32 1
  %field_load159 = load ptr, ptr %field_gep158, align 8
  call void @tl_tensor_free(ptr %field_load159)
  %field_gep160 = getelementptr inbounds %CausalSelfAttention, ptr %field_load147, i32 0, i32 2
  %field_load161 = load ptr, ptr %field_gep160, align 8
  %field_gep162 = getelementptr inbounds %Linear, ptr %field_load161, i32 0, i32 0
  %field_load163 = load ptr, ptr %field_gep162, align 8
  call void @tl_tensor_free(ptr %field_load163)
  %field_gep164 = getelementptr inbounds %Linear, ptr %field_load161, i32 0, i32 1
  %field_load165 = load ptr, ptr %field_gep164, align 8
  call void @tl_tensor_free(ptr %field_load165)
  %field_gep166 = getelementptr inbounds %CausalSelfAttention, ptr %field_load147, i32 0, i32 3
  %field_load167 = load ptr, ptr %field_gep166, align 8
  %field_gep168 = getelementptr inbounds %Linear, ptr %field_load167, i32 0, i32 0
  %field_load169 = load ptr, ptr %field_gep168, align 8
  call void @tl_tensor_free(ptr %field_load169)
  %field_gep170 = getelementptr inbounds %Linear, ptr %field_load167, i32 0, i32 1
  %field_load171 = load ptr, ptr %field_gep170, align 8
  call void @tl_tensor_free(ptr %field_load171)
  %field_gep172 = getelementptr inbounds %Block, ptr %field_load139, i32 0, i32 2
  %field_load173 = load ptr, ptr %field_gep172, align 8
  %field_gep174 = getelementptr inbounds %LayerNorm, ptr %field_load173, i32 0, i32 0
  %field_load175 = load ptr, ptr %field_gep174, align 8
  call void @tl_tensor_free(ptr %field_load175)
  %field_gep176 = getelementptr inbounds %LayerNorm, ptr %field_load173, i32 0, i32 1
  %field_load177 = load ptr, ptr %field_gep176, align 8
  call void @tl_tensor_free(ptr %field_load177)
  %field_gep178 = getelementptr inbounds %Block, ptr %field_load139, i32 0, i32 3
  %field_load179 = load ptr, ptr %field_gep178, align 8
  %field_gep180 = getelementptr inbounds %MLP, ptr %field_load179, i32 0, i32 0
  %field_load181 = load ptr, ptr %field_gep180, align 8
  %field_gep182 = getelementptr inbounds %Linear, ptr %field_load181, i32 0, i32 0
  %field_load183 = load ptr, ptr %field_gep182, align 8
  call void @tl_tensor_free(ptr %field_load183)
  %field_gep184 = getelementptr inbounds %Linear, ptr %field_load181, i32 0, i32 1
  %field_load185 = load ptr, ptr %field_gep184, align 8
  call void @tl_tensor_free(ptr %field_load185)
  %field_gep186 = getelementptr inbounds %MLP, ptr %field_load179, i32 0, i32 1
  %field_load187 = load ptr, ptr %field_gep186, align 8
  %field_gep188 = getelementptr inbounds %Linear, ptr %field_load187, i32 0, i32 0
  %field_load189 = load ptr, ptr %field_gep188, align 8
  call void @tl_tensor_free(ptr %field_load189)
  %field_gep190 = getelementptr inbounds %Linear, ptr %field_load187, i32 0, i32 1
  %field_load191 = load ptr, ptr %field_gep190, align 8
  call void @tl_tensor_free(ptr %field_load191)
  %field_gep192 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 5
  %field_load193 = load ptr, ptr %field_gep192, align 8
  %field_gep194 = getelementptr inbounds %Block, ptr %field_load193, i32 0, i32 0
  %field_load195 = load ptr, ptr %field_gep194, align 8
  %field_gep196 = getelementptr inbounds %LayerNorm, ptr %field_load195, i32 0, i32 0
  %field_load197 = load ptr, ptr %field_gep196, align 8
  call void @tl_tensor_free(ptr %field_load197)
  %field_gep198 = getelementptr inbounds %LayerNorm, ptr %field_load195, i32 0, i32 1
  %field_load199 = load ptr, ptr %field_gep198, align 8
  call void @tl_tensor_free(ptr %field_load199)
  %field_gep200 = getelementptr inbounds %Block, ptr %field_load193, i32 0, i32 1
  %field_load201 = load ptr, ptr %field_gep200, align 8
  %field_gep202 = getelementptr inbounds %CausalSelfAttention, ptr %field_load201, i32 0, i32 0
  %field_load203 = load ptr, ptr %field_gep202, align 8
  %field_gep204 = getelementptr inbounds %Linear, ptr %field_load203, i32 0, i32 0
  %field_load205 = load ptr, ptr %field_gep204, align 8
  call void @tl_tensor_free(ptr %field_load205)
  %field_gep206 = getelementptr inbounds %Linear, ptr %field_load203, i32 0, i32 1
  %field_load207 = load ptr, ptr %field_gep206, align 8
  call void @tl_tensor_free(ptr %field_load207)
  %field_gep208 = getelementptr inbounds %CausalSelfAttention, ptr %field_load201, i32 0, i32 1
  %field_load209 = load ptr, ptr %field_gep208, align 8
  %field_gep210 = getelementptr inbounds %Linear, ptr %field_load209, i32 0, i32 0
  %field_load211 = load ptr, ptr %field_gep210, align 8
  call void @tl_tensor_free(ptr %field_load211)
  %field_gep212 = getelementptr inbounds %Linear, ptr %field_load209, i32 0, i32 1
  %field_load213 = load ptr, ptr %field_gep212, align 8
  call void @tl_tensor_free(ptr %field_load213)
  %field_gep214 = getelementptr inbounds %CausalSelfAttention, ptr %field_load201, i32 0, i32 2
  %field_load215 = load ptr, ptr %field_gep214, align 8
  %field_gep216 = getelementptr inbounds %Linear, ptr %field_load215, i32 0, i32 0
  %field_load217 = load ptr, ptr %field_gep216, align 8
  call void @tl_tensor_free(ptr %field_load217)
  %field_gep218 = getelementptr inbounds %Linear, ptr %field_load215, i32 0, i32 1
  %field_load219 = load ptr, ptr %field_gep218, align 8
  call void @tl_tensor_free(ptr %field_load219)
  %field_gep220 = getelementptr inbounds %CausalSelfAttention, ptr %field_load201, i32 0, i32 3
  %field_load221 = load ptr, ptr %field_gep220, align 8
  %field_gep222 = getelementptr inbounds %Linear, ptr %field_load221, i32 0, i32 0
  %field_load223 = load ptr, ptr %field_gep222, align 8
  call void @tl_tensor_free(ptr %field_load223)
  %field_gep224 = getelementptr inbounds %Linear, ptr %field_load221, i32 0, i32 1
  %field_load225 = load ptr, ptr %field_gep224, align 8
  call void @tl_tensor_free(ptr %field_load225)
  %field_gep226 = getelementptr inbounds %Block, ptr %field_load193, i32 0, i32 2
  %field_load227 = load ptr, ptr %field_gep226, align 8
  %field_gep228 = getelementptr inbounds %LayerNorm, ptr %field_load227, i32 0, i32 0
  %field_load229 = load ptr, ptr %field_gep228, align 8
  call void @tl_tensor_free(ptr %field_load229)
  %field_gep230 = getelementptr inbounds %LayerNorm, ptr %field_load227, i32 0, i32 1
  %field_load231 = load ptr, ptr %field_gep230, align 8
  call void @tl_tensor_free(ptr %field_load231)
  %field_gep232 = getelementptr inbounds %Block, ptr %field_load193, i32 0, i32 3
  %field_load233 = load ptr, ptr %field_gep232, align 8
  %field_gep234 = getelementptr inbounds %MLP, ptr %field_load233, i32 0, i32 0
  %field_load235 = load ptr, ptr %field_gep234, align 8
  %field_gep236 = getelementptr inbounds %Linear, ptr %field_load235, i32 0, i32 0
  %field_load237 = load ptr, ptr %field_gep236, align 8
  call void @tl_tensor_free(ptr %field_load237)
  %field_gep238 = getelementptr inbounds %Linear, ptr %field_load235, i32 0, i32 1
  %field_load239 = load ptr, ptr %field_gep238, align 8
  call void @tl_tensor_free(ptr %field_load239)
  %field_gep240 = getelementptr inbounds %MLP, ptr %field_load233, i32 0, i32 1
  %field_load241 = load ptr, ptr %field_gep240, align 8
  %field_gep242 = getelementptr inbounds %Linear, ptr %field_load241, i32 0, i32 0
  %field_load243 = load ptr, ptr %field_gep242, align 8
  call void @tl_tensor_free(ptr %field_load243)
  %field_gep244 = getelementptr inbounds %Linear, ptr %field_load241, i32 0, i32 1
  %field_load245 = load ptr, ptr %field_gep244, align 8
  call void @tl_tensor_free(ptr %field_load245)
  %field_gep246 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 6
  %field_load247 = load ptr, ptr %field_gep246, align 8
  %field_gep248 = getelementptr inbounds %LayerNorm, ptr %field_load247, i32 0, i32 0
  %field_load249 = load ptr, ptr %field_gep248, align 8
  call void @tl_tensor_free(ptr %field_load249)
  %field_gep250 = getelementptr inbounds %LayerNorm, ptr %field_load247, i32 0, i32 1
  %field_load251 = load ptr, ptr %field_gep250, align 8
  call void @tl_tensor_free(ptr %field_load251)
  %field_gep252 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 7
  %field_load253 = load ptr, ptr %field_gep252, align 8
  %field_gep254 = getelementptr inbounds %Linear, ptr %field_load253, i32 0, i32 0
  %field_load255 = load ptr, ptr %field_gep254, align 8
  call void @tl_tensor_free(ptr %field_load255)
  %field_gep256 = getelementptr inbounds %Linear, ptr %field_load253, i32 0, i32 1
  %field_load257 = load ptr, ptr %field_gep256, align 8
  call void @tl_tensor_free(ptr %field_load257)
  call void @tl_mem_unregister(ptr %old_struct_to_free)
  br label %continue_after_free

continue_after_free:                              ; preds = %free_struct, %for_body
  call void @tl_mem_unregister(ptr %call_tmp)
  %unreg_field_0 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_0258 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val259 = load ptr, ptr %unreg_field_0258, align 8
  call void @tl_mem_unregister(ptr %field_val259)
  %unreg_field_1 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 1
  %field_val260 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val260)
  %unreg_field_0261 = getelementptr inbounds %Embedding, ptr %field_val260, i32 0, i32 0
  %field_val262 = load ptr, ptr %unreg_field_0261, align 8
  call void @tl_mem_unregister(ptr %field_val262)
  %unreg_field_2 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 2
  %field_val263 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val263)
  %unreg_field_0264 = getelementptr inbounds %Block, ptr %field_val263, i32 0, i32 0
  %field_val265 = load ptr, ptr %unreg_field_0264, align 8
  call void @tl_mem_unregister(ptr %field_val265)
  %unreg_field_0266 = getelementptr inbounds %LayerNorm, ptr %field_val265, i32 0, i32 0
  %field_val267 = load ptr, ptr %unreg_field_0266, align 8
  call void @tl_mem_unregister(ptr %field_val267)
  %unreg_field_1268 = getelementptr inbounds %LayerNorm, ptr %field_val265, i32 0, i32 1
  %field_val269 = load ptr, ptr %unreg_field_1268, align 8
  call void @tl_mem_unregister(ptr %field_val269)
  %unreg_field_1270 = getelementptr inbounds %Block, ptr %field_val263, i32 0, i32 1
  %field_val271 = load ptr, ptr %unreg_field_1270, align 8
  call void @tl_mem_unregister(ptr %field_val271)
  %unreg_field_0272 = getelementptr inbounds %CausalSelfAttention, ptr %field_val271, i32 0, i32 0
  %field_val273 = load ptr, ptr %unreg_field_0272, align 8
  call void @tl_mem_unregister(ptr %field_val273)
  %unreg_field_0274 = getelementptr inbounds %Linear, ptr %field_val273, i32 0, i32 0
  %field_val275 = load ptr, ptr %unreg_field_0274, align 8
  call void @tl_mem_unregister(ptr %field_val275)
  %unreg_field_1276 = getelementptr inbounds %Linear, ptr %field_val273, i32 0, i32 1
  %field_val277 = load ptr, ptr %unreg_field_1276, align 8
  call void @tl_mem_unregister(ptr %field_val277)
  %unreg_field_1278 = getelementptr inbounds %CausalSelfAttention, ptr %field_val271, i32 0, i32 1
  %field_val279 = load ptr, ptr %unreg_field_1278, align 8
  call void @tl_mem_unregister(ptr %field_val279)
  %unreg_field_0280 = getelementptr inbounds %Linear, ptr %field_val279, i32 0, i32 0
  %field_val281 = load ptr, ptr %unreg_field_0280, align 8
  call void @tl_mem_unregister(ptr %field_val281)
  %unreg_field_1282 = getelementptr inbounds %Linear, ptr %field_val279, i32 0, i32 1
  %field_val283 = load ptr, ptr %unreg_field_1282, align 8
  call void @tl_mem_unregister(ptr %field_val283)
  %unreg_field_2284 = getelementptr inbounds %CausalSelfAttention, ptr %field_val271, i32 0, i32 2
  %field_val285 = load ptr, ptr %unreg_field_2284, align 8
  call void @tl_mem_unregister(ptr %field_val285)
  %unreg_field_0286 = getelementptr inbounds %Linear, ptr %field_val285, i32 0, i32 0
  %field_val287 = load ptr, ptr %unreg_field_0286, align 8
  call void @tl_mem_unregister(ptr %field_val287)
  %unreg_field_1288 = getelementptr inbounds %Linear, ptr %field_val285, i32 0, i32 1
  %field_val289 = load ptr, ptr %unreg_field_1288, align 8
  call void @tl_mem_unregister(ptr %field_val289)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val271, i32 0, i32 3
  %field_val290 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val290)
  %unreg_field_0291 = getelementptr inbounds %Linear, ptr %field_val290, i32 0, i32 0
  %field_val292 = load ptr, ptr %unreg_field_0291, align 8
  call void @tl_mem_unregister(ptr %field_val292)
  %unreg_field_1293 = getelementptr inbounds %Linear, ptr %field_val290, i32 0, i32 1
  %field_val294 = load ptr, ptr %unreg_field_1293, align 8
  call void @tl_mem_unregister(ptr %field_val294)
  %unreg_field_2295 = getelementptr inbounds %Block, ptr %field_val263, i32 0, i32 2
  %field_val296 = load ptr, ptr %unreg_field_2295, align 8
  call void @tl_mem_unregister(ptr %field_val296)
  %unreg_field_0297 = getelementptr inbounds %LayerNorm, ptr %field_val296, i32 0, i32 0
  %field_val298 = load ptr, ptr %unreg_field_0297, align 8
  call void @tl_mem_unregister(ptr %field_val298)
  %unreg_field_1299 = getelementptr inbounds %LayerNorm, ptr %field_val296, i32 0, i32 1
  %field_val300 = load ptr, ptr %unreg_field_1299, align 8
  call void @tl_mem_unregister(ptr %field_val300)
  %unreg_field_3301 = getelementptr inbounds %Block, ptr %field_val263, i32 0, i32 3
  %field_val302 = load ptr, ptr %unreg_field_3301, align 8
  call void @tl_mem_unregister(ptr %field_val302)
  %unreg_field_0303 = getelementptr inbounds %MLP, ptr %field_val302, i32 0, i32 0
  %field_val304 = load ptr, ptr %unreg_field_0303, align 8
  call void @tl_mem_unregister(ptr %field_val304)
  %unreg_field_0305 = getelementptr inbounds %Linear, ptr %field_val304, i32 0, i32 0
  %field_val306 = load ptr, ptr %unreg_field_0305, align 8
  call void @tl_mem_unregister(ptr %field_val306)
  %unreg_field_1307 = getelementptr inbounds %Linear, ptr %field_val304, i32 0, i32 1
  %field_val308 = load ptr, ptr %unreg_field_1307, align 8
  call void @tl_mem_unregister(ptr %field_val308)
  %unreg_field_1309 = getelementptr inbounds %MLP, ptr %field_val302, i32 0, i32 1
  %field_val310 = load ptr, ptr %unreg_field_1309, align 8
  call void @tl_mem_unregister(ptr %field_val310)
  %unreg_field_0311 = getelementptr inbounds %Linear, ptr %field_val310, i32 0, i32 0
  %field_val312 = load ptr, ptr %unreg_field_0311, align 8
  call void @tl_mem_unregister(ptr %field_val312)
  %unreg_field_1313 = getelementptr inbounds %Linear, ptr %field_val310, i32 0, i32 1
  %field_val314 = load ptr, ptr %unreg_field_1313, align 8
  call void @tl_mem_unregister(ptr %field_val314)
  %unreg_field_3315 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 3
  %field_val316 = load ptr, ptr %unreg_field_3315, align 8
  call void @tl_mem_unregister(ptr %field_val316)
  %unreg_field_0317 = getelementptr inbounds %Block, ptr %field_val316, i32 0, i32 0
  %field_val318 = load ptr, ptr %unreg_field_0317, align 8
  call void @tl_mem_unregister(ptr %field_val318)
  %unreg_field_0319 = getelementptr inbounds %LayerNorm, ptr %field_val318, i32 0, i32 0
  %field_val320 = load ptr, ptr %unreg_field_0319, align 8
  call void @tl_mem_unregister(ptr %field_val320)
  %unreg_field_1321 = getelementptr inbounds %LayerNorm, ptr %field_val318, i32 0, i32 1
  %field_val322 = load ptr, ptr %unreg_field_1321, align 8
  call void @tl_mem_unregister(ptr %field_val322)
  %unreg_field_1323 = getelementptr inbounds %Block, ptr %field_val316, i32 0, i32 1
  %field_val324 = load ptr, ptr %unreg_field_1323, align 8
  call void @tl_mem_unregister(ptr %field_val324)
  %unreg_field_0325 = getelementptr inbounds %CausalSelfAttention, ptr %field_val324, i32 0, i32 0
  %field_val326 = load ptr, ptr %unreg_field_0325, align 8
  call void @tl_mem_unregister(ptr %field_val326)
  %unreg_field_0327 = getelementptr inbounds %Linear, ptr %field_val326, i32 0, i32 0
  %field_val328 = load ptr, ptr %unreg_field_0327, align 8
  call void @tl_mem_unregister(ptr %field_val328)
  %unreg_field_1329 = getelementptr inbounds %Linear, ptr %field_val326, i32 0, i32 1
  %field_val330 = load ptr, ptr %unreg_field_1329, align 8
  call void @tl_mem_unregister(ptr %field_val330)
  %unreg_field_1331 = getelementptr inbounds %CausalSelfAttention, ptr %field_val324, i32 0, i32 1
  %field_val332 = load ptr, ptr %unreg_field_1331, align 8
  call void @tl_mem_unregister(ptr %field_val332)
  %unreg_field_0333 = getelementptr inbounds %Linear, ptr %field_val332, i32 0, i32 0
  %field_val334 = load ptr, ptr %unreg_field_0333, align 8
  call void @tl_mem_unregister(ptr %field_val334)
  %unreg_field_1335 = getelementptr inbounds %Linear, ptr %field_val332, i32 0, i32 1
  %field_val336 = load ptr, ptr %unreg_field_1335, align 8
  call void @tl_mem_unregister(ptr %field_val336)
  %unreg_field_2337 = getelementptr inbounds %CausalSelfAttention, ptr %field_val324, i32 0, i32 2
  %field_val338 = load ptr, ptr %unreg_field_2337, align 8
  call void @tl_mem_unregister(ptr %field_val338)
  %unreg_field_0339 = getelementptr inbounds %Linear, ptr %field_val338, i32 0, i32 0
  %field_val340 = load ptr, ptr %unreg_field_0339, align 8
  call void @tl_mem_unregister(ptr %field_val340)
  %unreg_field_1341 = getelementptr inbounds %Linear, ptr %field_val338, i32 0, i32 1
  %field_val342 = load ptr, ptr %unreg_field_1341, align 8
  call void @tl_mem_unregister(ptr %field_val342)
  %unreg_field_3343 = getelementptr inbounds %CausalSelfAttention, ptr %field_val324, i32 0, i32 3
  %field_val344 = load ptr, ptr %unreg_field_3343, align 8
  call void @tl_mem_unregister(ptr %field_val344)
  %unreg_field_0345 = getelementptr inbounds %Linear, ptr %field_val344, i32 0, i32 0
  %field_val346 = load ptr, ptr %unreg_field_0345, align 8
  call void @tl_mem_unregister(ptr %field_val346)
  %unreg_field_1347 = getelementptr inbounds %Linear, ptr %field_val344, i32 0, i32 1
  %field_val348 = load ptr, ptr %unreg_field_1347, align 8
  call void @tl_mem_unregister(ptr %field_val348)
  %unreg_field_2349 = getelementptr inbounds %Block, ptr %field_val316, i32 0, i32 2
  %field_val350 = load ptr, ptr %unreg_field_2349, align 8
  call void @tl_mem_unregister(ptr %field_val350)
  %unreg_field_0351 = getelementptr inbounds %LayerNorm, ptr %field_val350, i32 0, i32 0
  %field_val352 = load ptr, ptr %unreg_field_0351, align 8
  call void @tl_mem_unregister(ptr %field_val352)
  %unreg_field_1353 = getelementptr inbounds %LayerNorm, ptr %field_val350, i32 0, i32 1
  %field_val354 = load ptr, ptr %unreg_field_1353, align 8
  call void @tl_mem_unregister(ptr %field_val354)
  %unreg_field_3355 = getelementptr inbounds %Block, ptr %field_val316, i32 0, i32 3
  %field_val356 = load ptr, ptr %unreg_field_3355, align 8
  call void @tl_mem_unregister(ptr %field_val356)
  %unreg_field_0357 = getelementptr inbounds %MLP, ptr %field_val356, i32 0, i32 0
  %field_val358 = load ptr, ptr %unreg_field_0357, align 8
  call void @tl_mem_unregister(ptr %field_val358)
  %unreg_field_0359 = getelementptr inbounds %Linear, ptr %field_val358, i32 0, i32 0
  %field_val360 = load ptr, ptr %unreg_field_0359, align 8
  call void @tl_mem_unregister(ptr %field_val360)
  %unreg_field_1361 = getelementptr inbounds %Linear, ptr %field_val358, i32 0, i32 1
  %field_val362 = load ptr, ptr %unreg_field_1361, align 8
  call void @tl_mem_unregister(ptr %field_val362)
  %unreg_field_1363 = getelementptr inbounds %MLP, ptr %field_val356, i32 0, i32 1
  %field_val364 = load ptr, ptr %unreg_field_1363, align 8
  call void @tl_mem_unregister(ptr %field_val364)
  %unreg_field_0365 = getelementptr inbounds %Linear, ptr %field_val364, i32 0, i32 0
  %field_val366 = load ptr, ptr %unreg_field_0365, align 8
  call void @tl_mem_unregister(ptr %field_val366)
  %unreg_field_1367 = getelementptr inbounds %Linear, ptr %field_val364, i32 0, i32 1
  %field_val368 = load ptr, ptr %unreg_field_1367, align 8
  call void @tl_mem_unregister(ptr %field_val368)
  %unreg_field_4 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 4
  %field_val369 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val369)
  %unreg_field_0370 = getelementptr inbounds %Block, ptr %field_val369, i32 0, i32 0
  %field_val371 = load ptr, ptr %unreg_field_0370, align 8
  call void @tl_mem_unregister(ptr %field_val371)
  %unreg_field_0372 = getelementptr inbounds %LayerNorm, ptr %field_val371, i32 0, i32 0
  %field_val373 = load ptr, ptr %unreg_field_0372, align 8
  call void @tl_mem_unregister(ptr %field_val373)
  %unreg_field_1374 = getelementptr inbounds %LayerNorm, ptr %field_val371, i32 0, i32 1
  %field_val375 = load ptr, ptr %unreg_field_1374, align 8
  call void @tl_mem_unregister(ptr %field_val375)
  %unreg_field_1376 = getelementptr inbounds %Block, ptr %field_val369, i32 0, i32 1
  %field_val377 = load ptr, ptr %unreg_field_1376, align 8
  call void @tl_mem_unregister(ptr %field_val377)
  %unreg_field_0378 = getelementptr inbounds %CausalSelfAttention, ptr %field_val377, i32 0, i32 0
  %field_val379 = load ptr, ptr %unreg_field_0378, align 8
  call void @tl_mem_unregister(ptr %field_val379)
  %unreg_field_0380 = getelementptr inbounds %Linear, ptr %field_val379, i32 0, i32 0
  %field_val381 = load ptr, ptr %unreg_field_0380, align 8
  call void @tl_mem_unregister(ptr %field_val381)
  %unreg_field_1382 = getelementptr inbounds %Linear, ptr %field_val379, i32 0, i32 1
  %field_val383 = load ptr, ptr %unreg_field_1382, align 8
  call void @tl_mem_unregister(ptr %field_val383)
  %unreg_field_1384 = getelementptr inbounds %CausalSelfAttention, ptr %field_val377, i32 0, i32 1
  %field_val385 = load ptr, ptr %unreg_field_1384, align 8
  call void @tl_mem_unregister(ptr %field_val385)
  %unreg_field_0386 = getelementptr inbounds %Linear, ptr %field_val385, i32 0, i32 0
  %field_val387 = load ptr, ptr %unreg_field_0386, align 8
  call void @tl_mem_unregister(ptr %field_val387)
  %unreg_field_1388 = getelementptr inbounds %Linear, ptr %field_val385, i32 0, i32 1
  %field_val389 = load ptr, ptr %unreg_field_1388, align 8
  call void @tl_mem_unregister(ptr %field_val389)
  %unreg_field_2390 = getelementptr inbounds %CausalSelfAttention, ptr %field_val377, i32 0, i32 2
  %field_val391 = load ptr, ptr %unreg_field_2390, align 8
  call void @tl_mem_unregister(ptr %field_val391)
  %unreg_field_0392 = getelementptr inbounds %Linear, ptr %field_val391, i32 0, i32 0
  %field_val393 = load ptr, ptr %unreg_field_0392, align 8
  call void @tl_mem_unregister(ptr %field_val393)
  %unreg_field_1394 = getelementptr inbounds %Linear, ptr %field_val391, i32 0, i32 1
  %field_val395 = load ptr, ptr %unreg_field_1394, align 8
  call void @tl_mem_unregister(ptr %field_val395)
  %unreg_field_3396 = getelementptr inbounds %CausalSelfAttention, ptr %field_val377, i32 0, i32 3
  %field_val397 = load ptr, ptr %unreg_field_3396, align 8
  call void @tl_mem_unregister(ptr %field_val397)
  %unreg_field_0398 = getelementptr inbounds %Linear, ptr %field_val397, i32 0, i32 0
  %field_val399 = load ptr, ptr %unreg_field_0398, align 8
  call void @tl_mem_unregister(ptr %field_val399)
  %unreg_field_1400 = getelementptr inbounds %Linear, ptr %field_val397, i32 0, i32 1
  %field_val401 = load ptr, ptr %unreg_field_1400, align 8
  call void @tl_mem_unregister(ptr %field_val401)
  %unreg_field_2402 = getelementptr inbounds %Block, ptr %field_val369, i32 0, i32 2
  %field_val403 = load ptr, ptr %unreg_field_2402, align 8
  call void @tl_mem_unregister(ptr %field_val403)
  %unreg_field_0404 = getelementptr inbounds %LayerNorm, ptr %field_val403, i32 0, i32 0
  %field_val405 = load ptr, ptr %unreg_field_0404, align 8
  call void @tl_mem_unregister(ptr %field_val405)
  %unreg_field_1406 = getelementptr inbounds %LayerNorm, ptr %field_val403, i32 0, i32 1
  %field_val407 = load ptr, ptr %unreg_field_1406, align 8
  call void @tl_mem_unregister(ptr %field_val407)
  %unreg_field_3408 = getelementptr inbounds %Block, ptr %field_val369, i32 0, i32 3
  %field_val409 = load ptr, ptr %unreg_field_3408, align 8
  call void @tl_mem_unregister(ptr %field_val409)
  %unreg_field_0410 = getelementptr inbounds %MLP, ptr %field_val409, i32 0, i32 0
  %field_val411 = load ptr, ptr %unreg_field_0410, align 8
  call void @tl_mem_unregister(ptr %field_val411)
  %unreg_field_0412 = getelementptr inbounds %Linear, ptr %field_val411, i32 0, i32 0
  %field_val413 = load ptr, ptr %unreg_field_0412, align 8
  call void @tl_mem_unregister(ptr %field_val413)
  %unreg_field_1414 = getelementptr inbounds %Linear, ptr %field_val411, i32 0, i32 1
  %field_val415 = load ptr, ptr %unreg_field_1414, align 8
  call void @tl_mem_unregister(ptr %field_val415)
  %unreg_field_1416 = getelementptr inbounds %MLP, ptr %field_val409, i32 0, i32 1
  %field_val417 = load ptr, ptr %unreg_field_1416, align 8
  call void @tl_mem_unregister(ptr %field_val417)
  %unreg_field_0418 = getelementptr inbounds %Linear, ptr %field_val417, i32 0, i32 0
  %field_val419 = load ptr, ptr %unreg_field_0418, align 8
  call void @tl_mem_unregister(ptr %field_val419)
  %unreg_field_1420 = getelementptr inbounds %Linear, ptr %field_val417, i32 0, i32 1
  %field_val421 = load ptr, ptr %unreg_field_1420, align 8
  call void @tl_mem_unregister(ptr %field_val421)
  %unreg_field_5 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 5
  %field_val422 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val422)
  %unreg_field_0423 = getelementptr inbounds %Block, ptr %field_val422, i32 0, i32 0
  %field_val424 = load ptr, ptr %unreg_field_0423, align 8
  call void @tl_mem_unregister(ptr %field_val424)
  %unreg_field_0425 = getelementptr inbounds %LayerNorm, ptr %field_val424, i32 0, i32 0
  %field_val426 = load ptr, ptr %unreg_field_0425, align 8
  call void @tl_mem_unregister(ptr %field_val426)
  %unreg_field_1427 = getelementptr inbounds %LayerNorm, ptr %field_val424, i32 0, i32 1
  %field_val428 = load ptr, ptr %unreg_field_1427, align 8
  call void @tl_mem_unregister(ptr %field_val428)
  %unreg_field_1429 = getelementptr inbounds %Block, ptr %field_val422, i32 0, i32 1
  %field_val430 = load ptr, ptr %unreg_field_1429, align 8
  call void @tl_mem_unregister(ptr %field_val430)
  %unreg_field_0431 = getelementptr inbounds %CausalSelfAttention, ptr %field_val430, i32 0, i32 0
  %field_val432 = load ptr, ptr %unreg_field_0431, align 8
  call void @tl_mem_unregister(ptr %field_val432)
  %unreg_field_0433 = getelementptr inbounds %Linear, ptr %field_val432, i32 0, i32 0
  %field_val434 = load ptr, ptr %unreg_field_0433, align 8
  call void @tl_mem_unregister(ptr %field_val434)
  %unreg_field_1435 = getelementptr inbounds %Linear, ptr %field_val432, i32 0, i32 1
  %field_val436 = load ptr, ptr %unreg_field_1435, align 8
  call void @tl_mem_unregister(ptr %field_val436)
  %unreg_field_1437 = getelementptr inbounds %CausalSelfAttention, ptr %field_val430, i32 0, i32 1
  %field_val438 = load ptr, ptr %unreg_field_1437, align 8
  call void @tl_mem_unregister(ptr %field_val438)
  %unreg_field_0439 = getelementptr inbounds %Linear, ptr %field_val438, i32 0, i32 0
  %field_val440 = load ptr, ptr %unreg_field_0439, align 8
  call void @tl_mem_unregister(ptr %field_val440)
  %unreg_field_1441 = getelementptr inbounds %Linear, ptr %field_val438, i32 0, i32 1
  %field_val442 = load ptr, ptr %unreg_field_1441, align 8
  call void @tl_mem_unregister(ptr %field_val442)
  %unreg_field_2443 = getelementptr inbounds %CausalSelfAttention, ptr %field_val430, i32 0, i32 2
  %field_val444 = load ptr, ptr %unreg_field_2443, align 8
  call void @tl_mem_unregister(ptr %field_val444)
  %unreg_field_0445 = getelementptr inbounds %Linear, ptr %field_val444, i32 0, i32 0
  %field_val446 = load ptr, ptr %unreg_field_0445, align 8
  call void @tl_mem_unregister(ptr %field_val446)
  %unreg_field_1447 = getelementptr inbounds %Linear, ptr %field_val444, i32 0, i32 1
  %field_val448 = load ptr, ptr %unreg_field_1447, align 8
  call void @tl_mem_unregister(ptr %field_val448)
  %unreg_field_3449 = getelementptr inbounds %CausalSelfAttention, ptr %field_val430, i32 0, i32 3
  %field_val450 = load ptr, ptr %unreg_field_3449, align 8
  call void @tl_mem_unregister(ptr %field_val450)
  %unreg_field_0451 = getelementptr inbounds %Linear, ptr %field_val450, i32 0, i32 0
  %field_val452 = load ptr, ptr %unreg_field_0451, align 8
  call void @tl_mem_unregister(ptr %field_val452)
  %unreg_field_1453 = getelementptr inbounds %Linear, ptr %field_val450, i32 0, i32 1
  %field_val454 = load ptr, ptr %unreg_field_1453, align 8
  call void @tl_mem_unregister(ptr %field_val454)
  %unreg_field_2455 = getelementptr inbounds %Block, ptr %field_val422, i32 0, i32 2
  %field_val456 = load ptr, ptr %unreg_field_2455, align 8
  call void @tl_mem_unregister(ptr %field_val456)
  %unreg_field_0457 = getelementptr inbounds %LayerNorm, ptr %field_val456, i32 0, i32 0
  %field_val458 = load ptr, ptr %unreg_field_0457, align 8
  call void @tl_mem_unregister(ptr %field_val458)
  %unreg_field_1459 = getelementptr inbounds %LayerNorm, ptr %field_val456, i32 0, i32 1
  %field_val460 = load ptr, ptr %unreg_field_1459, align 8
  call void @tl_mem_unregister(ptr %field_val460)
  %unreg_field_3461 = getelementptr inbounds %Block, ptr %field_val422, i32 0, i32 3
  %field_val462 = load ptr, ptr %unreg_field_3461, align 8
  call void @tl_mem_unregister(ptr %field_val462)
  %unreg_field_0463 = getelementptr inbounds %MLP, ptr %field_val462, i32 0, i32 0
  %field_val464 = load ptr, ptr %unreg_field_0463, align 8
  call void @tl_mem_unregister(ptr %field_val464)
  %unreg_field_0465 = getelementptr inbounds %Linear, ptr %field_val464, i32 0, i32 0
  %field_val466 = load ptr, ptr %unreg_field_0465, align 8
  call void @tl_mem_unregister(ptr %field_val466)
  %unreg_field_1467 = getelementptr inbounds %Linear, ptr %field_val464, i32 0, i32 1
  %field_val468 = load ptr, ptr %unreg_field_1467, align 8
  call void @tl_mem_unregister(ptr %field_val468)
  %unreg_field_1469 = getelementptr inbounds %MLP, ptr %field_val462, i32 0, i32 1
  %field_val470 = load ptr, ptr %unreg_field_1469, align 8
  call void @tl_mem_unregister(ptr %field_val470)
  %unreg_field_0471 = getelementptr inbounds %Linear, ptr %field_val470, i32 0, i32 0
  %field_val472 = load ptr, ptr %unreg_field_0471, align 8
  call void @tl_mem_unregister(ptr %field_val472)
  %unreg_field_1473 = getelementptr inbounds %Linear, ptr %field_val470, i32 0, i32 1
  %field_val474 = load ptr, ptr %unreg_field_1473, align 8
  call void @tl_mem_unregister(ptr %field_val474)
  %unreg_field_6 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 6
  %field_val475 = load ptr, ptr %unreg_field_6, align 8
  call void @tl_mem_unregister(ptr %field_val475)
  %unreg_field_0476 = getelementptr inbounds %LayerNorm, ptr %field_val475, i32 0, i32 0
  %field_val477 = load ptr, ptr %unreg_field_0476, align 8
  call void @tl_mem_unregister(ptr %field_val477)
  %unreg_field_1478 = getelementptr inbounds %LayerNorm, ptr %field_val475, i32 0, i32 1
  %field_val479 = load ptr, ptr %unreg_field_1478, align 8
  call void @tl_mem_unregister(ptr %field_val479)
  %unreg_field_7 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 7
  %field_val480 = load ptr, ptr %unreg_field_7, align 8
  call void @tl_mem_unregister(ptr %field_val480)
  %unreg_field_0481 = getelementptr inbounds %Linear, ptr %field_val480, i32 0, i32 0
  %field_val482 = load ptr, ptr %unreg_field_0481, align 8
  call void @tl_mem_unregister(ptr %field_val482)
  %unreg_field_1483 = getelementptr inbounds %Linear, ptr %field_val480, i32 0, i32 1
  %field_val484 = load ptr, ptr %unreg_field_1483, align 8
  call void @tl_mem_unregister(ptr %field_val484)
  call void @tl_mem_unregister(ptr %call_tmp)
  store ptr %call_tmp, ptr %model1, align 8
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
  call void @tl_arena_init(i64 256000)
  store i64 13, ptr %vocab_size, align 8
  store i64 384, ptr %d_model, align 8
  %vocab_size1 = load i64, ptr %vocab_size, align 8
  %d_model2 = load i64, ptr %d_model, align 8
  %static_call = call ptr @tl_GPTHeavy_new(i64 %vocab_size1, i64 %d_model2)
  call void @tl_mem_unregister(ptr %static_call)
  store ptr %static_call, ptr %model, align 8
  %model3 = load ptr, ptr %model, align 8
  %w = getelementptr inbounds %GPTHeavy, ptr %model3, i32 0, i32 0
  %sub_ptr = load ptr, ptr %w, align 8
  %w4 = getelementptr inbounds %Embedding, ptr %sub_ptr, i32 0, i32 0
  %w5 = load ptr, ptr %w4, align 8
  call void @tl_add_parameter(ptr @key_str, ptr %w5)
  %wp = getelementptr inbounds %GPTHeavy, ptr %model3, i32 0, i32 1
  %sub_ptr6 = load ptr, ptr %wp, align 8
  %w7 = getelementptr inbounds %Embedding, ptr %sub_ptr6, i32 0, i32 0
  %w8 = load ptr, ptr %w7, align 8
  call void @tl_add_parameter(ptr @key_str.104, ptr %w8)
  %b1 = getelementptr inbounds %GPTHeavy, ptr %model3, i32 0, i32 2
  %sub_ptr9 = load ptr, ptr %b1, align 8
  %l1 = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 0
  %sub_ptr10 = load ptr, ptr %l1, align 8
  %w11 = getelementptr inbounds %LayerNorm, ptr %sub_ptr10, i32 0, i32 0
  %w12 = load ptr, ptr %w11, align 8
  call void @tl_add_parameter(ptr @key_str.105, ptr %w12)
  %b = getelementptr inbounds %LayerNorm, ptr %sub_ptr10, i32 0, i32 1
  %b13 = load ptr, ptr %b, align 8
  call void @tl_add_parameter(ptr @key_str.106, ptr %b13)
  %a = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 1
  %sub_ptr14 = load ptr, ptr %a, align 8
  %q_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 0
  %sub_ptr15 = load ptr, ptr %q_proj, align 8
  %W = getelementptr inbounds %Linear, ptr %sub_ptr15, i32 0, i32 0
  %W16 = load ptr, ptr %W, align 8
  call void @tl_add_parameter(ptr @key_str.107, ptr %W16)
  %b17 = getelementptr inbounds %Linear, ptr %sub_ptr15, i32 0, i32 1
  %b18 = load ptr, ptr %b17, align 8
  call void @tl_add_parameter(ptr @key_str.108, ptr %b18)
  %k_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 1
  %sub_ptr19 = load ptr, ptr %k_proj, align 8
  %W20 = getelementptr inbounds %Linear, ptr %sub_ptr19, i32 0, i32 0
  %W21 = load ptr, ptr %W20, align 8
  call void @tl_add_parameter(ptr @key_str.109, ptr %W21)
  %b22 = getelementptr inbounds %Linear, ptr %sub_ptr19, i32 0, i32 1
  %b23 = load ptr, ptr %b22, align 8
  call void @tl_add_parameter(ptr @key_str.110, ptr %b23)
  %v_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 2
  %sub_ptr24 = load ptr, ptr %v_proj, align 8
  %W25 = getelementptr inbounds %Linear, ptr %sub_ptr24, i32 0, i32 0
  %W26 = load ptr, ptr %W25, align 8
  call void @tl_add_parameter(ptr @key_str.111, ptr %W26)
  %b27 = getelementptr inbounds %Linear, ptr %sub_ptr24, i32 0, i32 1
  %b28 = load ptr, ptr %b27, align 8
  call void @tl_add_parameter(ptr @key_str.112, ptr %b28)
  %p_proj = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr14, i32 0, i32 3
  %sub_ptr29 = load ptr, ptr %p_proj, align 8
  %W30 = getelementptr inbounds %Linear, ptr %sub_ptr29, i32 0, i32 0
  %W31 = load ptr, ptr %W30, align 8
  call void @tl_add_parameter(ptr @key_str.113, ptr %W31)
  %b32 = getelementptr inbounds %Linear, ptr %sub_ptr29, i32 0, i32 1
  %b33 = load ptr, ptr %b32, align 8
  call void @tl_add_parameter(ptr @key_str.114, ptr %b33)
  %l2 = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 2
  %sub_ptr34 = load ptr, ptr %l2, align 8
  %w35 = getelementptr inbounds %LayerNorm, ptr %sub_ptr34, i32 0, i32 0
  %w36 = load ptr, ptr %w35, align 8
  call void @tl_add_parameter(ptr @key_str.115, ptr %w36)
  %b37 = getelementptr inbounds %LayerNorm, ptr %sub_ptr34, i32 0, i32 1
  %b38 = load ptr, ptr %b37, align 8
  call void @tl_add_parameter(ptr @key_str.116, ptr %b38)
  %m = getelementptr inbounds %Block, ptr %sub_ptr9, i32 0, i32 3
  %sub_ptr39 = load ptr, ptr %m, align 8
  %f = getelementptr inbounds %MLP, ptr %sub_ptr39, i32 0, i32 0
  %sub_ptr40 = load ptr, ptr %f, align 8
  %W41 = getelementptr inbounds %Linear, ptr %sub_ptr40, i32 0, i32 0
  %W42 = load ptr, ptr %W41, align 8
  call void @tl_add_parameter(ptr @key_str.117, ptr %W42)
  %b43 = getelementptr inbounds %Linear, ptr %sub_ptr40, i32 0, i32 1
  %b44 = load ptr, ptr %b43, align 8
  call void @tl_add_parameter(ptr @key_str.118, ptr %b44)
  %p = getelementptr inbounds %MLP, ptr %sub_ptr39, i32 0, i32 1
  %sub_ptr45 = load ptr, ptr %p, align 8
  %W46 = getelementptr inbounds %Linear, ptr %sub_ptr45, i32 0, i32 0
  %W47 = load ptr, ptr %W46, align 8
  call void @tl_add_parameter(ptr @key_str.119, ptr %W47)
  %b48 = getelementptr inbounds %Linear, ptr %sub_ptr45, i32 0, i32 1
  %b49 = load ptr, ptr %b48, align 8
  call void @tl_add_parameter(ptr @key_str.120, ptr %b49)
  %b2 = getelementptr inbounds %GPTHeavy, ptr %model3, i32 0, i32 3
  %sub_ptr50 = load ptr, ptr %b2, align 8
  %l151 = getelementptr inbounds %Block, ptr %sub_ptr50, i32 0, i32 0
  %sub_ptr52 = load ptr, ptr %l151, align 8
  %w53 = getelementptr inbounds %LayerNorm, ptr %sub_ptr52, i32 0, i32 0
  %w54 = load ptr, ptr %w53, align 8
  call void @tl_add_parameter(ptr @key_str.121, ptr %w54)
  %b55 = getelementptr inbounds %LayerNorm, ptr %sub_ptr52, i32 0, i32 1
  %b56 = load ptr, ptr %b55, align 8
  call void @tl_add_parameter(ptr @key_str.122, ptr %b56)
  %a57 = getelementptr inbounds %Block, ptr %sub_ptr50, i32 0, i32 1
  %sub_ptr58 = load ptr, ptr %a57, align 8
  %q_proj59 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr58, i32 0, i32 0
  %sub_ptr60 = load ptr, ptr %q_proj59, align 8
  %W61 = getelementptr inbounds %Linear, ptr %sub_ptr60, i32 0, i32 0
  %W62 = load ptr, ptr %W61, align 8
  call void @tl_add_parameter(ptr @key_str.123, ptr %W62)
  %b63 = getelementptr inbounds %Linear, ptr %sub_ptr60, i32 0, i32 1
  %b64 = load ptr, ptr %b63, align 8
  call void @tl_add_parameter(ptr @key_str.124, ptr %b64)
  %k_proj65 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr58, i32 0, i32 1
  %sub_ptr66 = load ptr, ptr %k_proj65, align 8
  %W67 = getelementptr inbounds %Linear, ptr %sub_ptr66, i32 0, i32 0
  %W68 = load ptr, ptr %W67, align 8
  call void @tl_add_parameter(ptr @key_str.125, ptr %W68)
  %b69 = getelementptr inbounds %Linear, ptr %sub_ptr66, i32 0, i32 1
  %b70 = load ptr, ptr %b69, align 8
  call void @tl_add_parameter(ptr @key_str.126, ptr %b70)
  %v_proj71 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr58, i32 0, i32 2
  %sub_ptr72 = load ptr, ptr %v_proj71, align 8
  %W73 = getelementptr inbounds %Linear, ptr %sub_ptr72, i32 0, i32 0
  %W74 = load ptr, ptr %W73, align 8
  call void @tl_add_parameter(ptr @key_str.127, ptr %W74)
  %b75 = getelementptr inbounds %Linear, ptr %sub_ptr72, i32 0, i32 1
  %b76 = load ptr, ptr %b75, align 8
  call void @tl_add_parameter(ptr @key_str.128, ptr %b76)
  %p_proj77 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr58, i32 0, i32 3
  %sub_ptr78 = load ptr, ptr %p_proj77, align 8
  %W79 = getelementptr inbounds %Linear, ptr %sub_ptr78, i32 0, i32 0
  %W80 = load ptr, ptr %W79, align 8
  call void @tl_add_parameter(ptr @key_str.129, ptr %W80)
  %b81 = getelementptr inbounds %Linear, ptr %sub_ptr78, i32 0, i32 1
  %b82 = load ptr, ptr %b81, align 8
  call void @tl_add_parameter(ptr @key_str.130, ptr %b82)
  %l283 = getelementptr inbounds %Block, ptr %sub_ptr50, i32 0, i32 2
  %sub_ptr84 = load ptr, ptr %l283, align 8
  %w85 = getelementptr inbounds %LayerNorm, ptr %sub_ptr84, i32 0, i32 0
  %w86 = load ptr, ptr %w85, align 8
  call void @tl_add_parameter(ptr @key_str.131, ptr %w86)
  %b87 = getelementptr inbounds %LayerNorm, ptr %sub_ptr84, i32 0, i32 1
  %b88 = load ptr, ptr %b87, align 8
  call void @tl_add_parameter(ptr @key_str.132, ptr %b88)
  %m89 = getelementptr inbounds %Block, ptr %sub_ptr50, i32 0, i32 3
  %sub_ptr90 = load ptr, ptr %m89, align 8
  %f91 = getelementptr inbounds %MLP, ptr %sub_ptr90, i32 0, i32 0
  %sub_ptr92 = load ptr, ptr %f91, align 8
  %W93 = getelementptr inbounds %Linear, ptr %sub_ptr92, i32 0, i32 0
  %W94 = load ptr, ptr %W93, align 8
  call void @tl_add_parameter(ptr @key_str.133, ptr %W94)
  %b95 = getelementptr inbounds %Linear, ptr %sub_ptr92, i32 0, i32 1
  %b96 = load ptr, ptr %b95, align 8
  call void @tl_add_parameter(ptr @key_str.134, ptr %b96)
  %p97 = getelementptr inbounds %MLP, ptr %sub_ptr90, i32 0, i32 1
  %sub_ptr98 = load ptr, ptr %p97, align 8
  %W99 = getelementptr inbounds %Linear, ptr %sub_ptr98, i32 0, i32 0
  %W100 = load ptr, ptr %W99, align 8
  call void @tl_add_parameter(ptr @key_str.135, ptr %W100)
  %b101 = getelementptr inbounds %Linear, ptr %sub_ptr98, i32 0, i32 1
  %b102 = load ptr, ptr %b101, align 8
  call void @tl_add_parameter(ptr @key_str.136, ptr %b102)
  %b3 = getelementptr inbounds %GPTHeavy, ptr %model3, i32 0, i32 4
  %sub_ptr103 = load ptr, ptr %b3, align 8
  %l1104 = getelementptr inbounds %Block, ptr %sub_ptr103, i32 0, i32 0
  %sub_ptr105 = load ptr, ptr %l1104, align 8
  %w106 = getelementptr inbounds %LayerNorm, ptr %sub_ptr105, i32 0, i32 0
  %w107 = load ptr, ptr %w106, align 8
  call void @tl_add_parameter(ptr @key_str.137, ptr %w107)
  %b108 = getelementptr inbounds %LayerNorm, ptr %sub_ptr105, i32 0, i32 1
  %b109 = load ptr, ptr %b108, align 8
  call void @tl_add_parameter(ptr @key_str.138, ptr %b109)
  %a110 = getelementptr inbounds %Block, ptr %sub_ptr103, i32 0, i32 1
  %sub_ptr111 = load ptr, ptr %a110, align 8
  %q_proj112 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr111, i32 0, i32 0
  %sub_ptr113 = load ptr, ptr %q_proj112, align 8
  %W114 = getelementptr inbounds %Linear, ptr %sub_ptr113, i32 0, i32 0
  %W115 = load ptr, ptr %W114, align 8
  call void @tl_add_parameter(ptr @key_str.139, ptr %W115)
  %b116 = getelementptr inbounds %Linear, ptr %sub_ptr113, i32 0, i32 1
  %b117 = load ptr, ptr %b116, align 8
  call void @tl_add_parameter(ptr @key_str.140, ptr %b117)
  %k_proj118 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr111, i32 0, i32 1
  %sub_ptr119 = load ptr, ptr %k_proj118, align 8
  %W120 = getelementptr inbounds %Linear, ptr %sub_ptr119, i32 0, i32 0
  %W121 = load ptr, ptr %W120, align 8
  call void @tl_add_parameter(ptr @key_str.141, ptr %W121)
  %b122 = getelementptr inbounds %Linear, ptr %sub_ptr119, i32 0, i32 1
  %b123 = load ptr, ptr %b122, align 8
  call void @tl_add_parameter(ptr @key_str.142, ptr %b123)
  %v_proj124 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr111, i32 0, i32 2
  %sub_ptr125 = load ptr, ptr %v_proj124, align 8
  %W126 = getelementptr inbounds %Linear, ptr %sub_ptr125, i32 0, i32 0
  %W127 = load ptr, ptr %W126, align 8
  call void @tl_add_parameter(ptr @key_str.143, ptr %W127)
  %b128 = getelementptr inbounds %Linear, ptr %sub_ptr125, i32 0, i32 1
  %b129 = load ptr, ptr %b128, align 8
  call void @tl_add_parameter(ptr @key_str.144, ptr %b129)
  %p_proj130 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr111, i32 0, i32 3
  %sub_ptr131 = load ptr, ptr %p_proj130, align 8
  %W132 = getelementptr inbounds %Linear, ptr %sub_ptr131, i32 0, i32 0
  %W133 = load ptr, ptr %W132, align 8
  call void @tl_add_parameter(ptr @key_str.145, ptr %W133)
  %b134 = getelementptr inbounds %Linear, ptr %sub_ptr131, i32 0, i32 1
  %b135 = load ptr, ptr %b134, align 8
  call void @tl_add_parameter(ptr @key_str.146, ptr %b135)
  %l2136 = getelementptr inbounds %Block, ptr %sub_ptr103, i32 0, i32 2
  %sub_ptr137 = load ptr, ptr %l2136, align 8
  %w138 = getelementptr inbounds %LayerNorm, ptr %sub_ptr137, i32 0, i32 0
  %w139 = load ptr, ptr %w138, align 8
  call void @tl_add_parameter(ptr @key_str.147, ptr %w139)
  %b140 = getelementptr inbounds %LayerNorm, ptr %sub_ptr137, i32 0, i32 1
  %b141 = load ptr, ptr %b140, align 8
  call void @tl_add_parameter(ptr @key_str.148, ptr %b141)
  %m142 = getelementptr inbounds %Block, ptr %sub_ptr103, i32 0, i32 3
  %sub_ptr143 = load ptr, ptr %m142, align 8
  %f144 = getelementptr inbounds %MLP, ptr %sub_ptr143, i32 0, i32 0
  %sub_ptr145 = load ptr, ptr %f144, align 8
  %W146 = getelementptr inbounds %Linear, ptr %sub_ptr145, i32 0, i32 0
  %W147 = load ptr, ptr %W146, align 8
  call void @tl_add_parameter(ptr @key_str.149, ptr %W147)
  %b148 = getelementptr inbounds %Linear, ptr %sub_ptr145, i32 0, i32 1
  %b149 = load ptr, ptr %b148, align 8
  call void @tl_add_parameter(ptr @key_str.150, ptr %b149)
  %p150 = getelementptr inbounds %MLP, ptr %sub_ptr143, i32 0, i32 1
  %sub_ptr151 = load ptr, ptr %p150, align 8
  %W152 = getelementptr inbounds %Linear, ptr %sub_ptr151, i32 0, i32 0
  %W153 = load ptr, ptr %W152, align 8
  call void @tl_add_parameter(ptr @key_str.151, ptr %W153)
  %b154 = getelementptr inbounds %Linear, ptr %sub_ptr151, i32 0, i32 1
  %b155 = load ptr, ptr %b154, align 8
  call void @tl_add_parameter(ptr @key_str.152, ptr %b155)
  %b4 = getelementptr inbounds %GPTHeavy, ptr %model3, i32 0, i32 5
  %sub_ptr156 = load ptr, ptr %b4, align 8
  %l1157 = getelementptr inbounds %Block, ptr %sub_ptr156, i32 0, i32 0
  %sub_ptr158 = load ptr, ptr %l1157, align 8
  %w159 = getelementptr inbounds %LayerNorm, ptr %sub_ptr158, i32 0, i32 0
  %w160 = load ptr, ptr %w159, align 8
  call void @tl_add_parameter(ptr @key_str.153, ptr %w160)
  %b161 = getelementptr inbounds %LayerNorm, ptr %sub_ptr158, i32 0, i32 1
  %b162 = load ptr, ptr %b161, align 8
  call void @tl_add_parameter(ptr @key_str.154, ptr %b162)
  %a163 = getelementptr inbounds %Block, ptr %sub_ptr156, i32 0, i32 1
  %sub_ptr164 = load ptr, ptr %a163, align 8
  %q_proj165 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr164, i32 0, i32 0
  %sub_ptr166 = load ptr, ptr %q_proj165, align 8
  %W167 = getelementptr inbounds %Linear, ptr %sub_ptr166, i32 0, i32 0
  %W168 = load ptr, ptr %W167, align 8
  call void @tl_add_parameter(ptr @key_str.155, ptr %W168)
  %b169 = getelementptr inbounds %Linear, ptr %sub_ptr166, i32 0, i32 1
  %b170 = load ptr, ptr %b169, align 8
  call void @tl_add_parameter(ptr @key_str.156, ptr %b170)
  %k_proj171 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr164, i32 0, i32 1
  %sub_ptr172 = load ptr, ptr %k_proj171, align 8
  %W173 = getelementptr inbounds %Linear, ptr %sub_ptr172, i32 0, i32 0
  %W174 = load ptr, ptr %W173, align 8
  call void @tl_add_parameter(ptr @key_str.157, ptr %W174)
  %b175 = getelementptr inbounds %Linear, ptr %sub_ptr172, i32 0, i32 1
  %b176 = load ptr, ptr %b175, align 8
  call void @tl_add_parameter(ptr @key_str.158, ptr %b176)
  %v_proj177 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr164, i32 0, i32 2
  %sub_ptr178 = load ptr, ptr %v_proj177, align 8
  %W179 = getelementptr inbounds %Linear, ptr %sub_ptr178, i32 0, i32 0
  %W180 = load ptr, ptr %W179, align 8
  call void @tl_add_parameter(ptr @key_str.159, ptr %W180)
  %b181 = getelementptr inbounds %Linear, ptr %sub_ptr178, i32 0, i32 1
  %b182 = load ptr, ptr %b181, align 8
  call void @tl_add_parameter(ptr @key_str.160, ptr %b182)
  %p_proj183 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr164, i32 0, i32 3
  %sub_ptr184 = load ptr, ptr %p_proj183, align 8
  %W185 = getelementptr inbounds %Linear, ptr %sub_ptr184, i32 0, i32 0
  %W186 = load ptr, ptr %W185, align 8
  call void @tl_add_parameter(ptr @key_str.161, ptr %W186)
  %b187 = getelementptr inbounds %Linear, ptr %sub_ptr184, i32 0, i32 1
  %b188 = load ptr, ptr %b187, align 8
  call void @tl_add_parameter(ptr @key_str.162, ptr %b188)
  %l2189 = getelementptr inbounds %Block, ptr %sub_ptr156, i32 0, i32 2
  %sub_ptr190 = load ptr, ptr %l2189, align 8
  %w191 = getelementptr inbounds %LayerNorm, ptr %sub_ptr190, i32 0, i32 0
  %w192 = load ptr, ptr %w191, align 8
  call void @tl_add_parameter(ptr @key_str.163, ptr %w192)
  %b193 = getelementptr inbounds %LayerNorm, ptr %sub_ptr190, i32 0, i32 1
  %b194 = load ptr, ptr %b193, align 8
  call void @tl_add_parameter(ptr @key_str.164, ptr %b194)
  %m195 = getelementptr inbounds %Block, ptr %sub_ptr156, i32 0, i32 3
  %sub_ptr196 = load ptr, ptr %m195, align 8
  %f197 = getelementptr inbounds %MLP, ptr %sub_ptr196, i32 0, i32 0
  %sub_ptr198 = load ptr, ptr %f197, align 8
  %W199 = getelementptr inbounds %Linear, ptr %sub_ptr198, i32 0, i32 0
  %W200 = load ptr, ptr %W199, align 8
  call void @tl_add_parameter(ptr @key_str.165, ptr %W200)
  %b201 = getelementptr inbounds %Linear, ptr %sub_ptr198, i32 0, i32 1
  %b202 = load ptr, ptr %b201, align 8
  call void @tl_add_parameter(ptr @key_str.166, ptr %b202)
  %p203 = getelementptr inbounds %MLP, ptr %sub_ptr196, i32 0, i32 1
  %sub_ptr204 = load ptr, ptr %p203, align 8
  %W205 = getelementptr inbounds %Linear, ptr %sub_ptr204, i32 0, i32 0
  %W206 = load ptr, ptr %W205, align 8
  call void @tl_add_parameter(ptr @key_str.167, ptr %W206)
  %b207 = getelementptr inbounds %Linear, ptr %sub_ptr204, i32 0, i32 1
  %b208 = load ptr, ptr %b207, align 8
  call void @tl_add_parameter(ptr @key_str.168, ptr %b208)
  %l = getelementptr inbounds %GPTHeavy, ptr %model3, i32 0, i32 6
  %sub_ptr209 = load ptr, ptr %l, align 8
  %w210 = getelementptr inbounds %LayerNorm, ptr %sub_ptr209, i32 0, i32 0
  %w211 = load ptr, ptr %w210, align 8
  call void @tl_add_parameter(ptr @key_str.169, ptr %w211)
  %b212 = getelementptr inbounds %LayerNorm, ptr %sub_ptr209, i32 0, i32 1
  %b213 = load ptr, ptr %b212, align 8
  call void @tl_add_parameter(ptr @key_str.170, ptr %b213)
  %h = getelementptr inbounds %GPTHeavy, ptr %model3, i32 0, i32 7
  %sub_ptr214 = load ptr, ptr %h, align 8
  %W215 = getelementptr inbounds %Linear, ptr %sub_ptr214, i32 0, i32 0
  %W216 = load ptr, ptr %W215, align 8
  call void @tl_add_parameter(ptr @key_str.171, ptr %W216)
  %b217 = getelementptr inbounds %Linear, ptr %sub_ptr214, i32 0, i32 1
  %b218 = load ptr, ptr %b217, align 8
  call void @tl_add_parameter(ptr @key_str.172, ptr %b218)
  call void @tl_load_all_params(ptr @str_literal.173)
  store float 0x3F40624DE0000000, ptr %lr, align 4
  store i64 500, ptr %epochs, align 8
  call void @tl_print_string(ptr @str_literal.174)
  %epochs219 = load i64, ptr %epochs, align 8
  br label %for_header

for_header:                                       ; preds = %merge, %entry
  %for_idx = phi i64 [ %next_idx, %merge ], [ 0, %entry ]
  %for_cond = icmp slt i64 %for_idx, %epochs219
  br i1 %for_cond, label %for_body, label %for_end

for_body:                                         ; preds = %for_header
  call void @tl_mem_enter_scope()
  store i64 %for_idx, ptr %epoch, align 8
  %model220 = load ptr, ptr %model, align 8
  %lr221 = load float, ptr %lr, align 4
  %epoch222 = load i64, ptr %epoch, align 8
  %call_tmp = call ptr @train_epoch(ptr %model220, float %lr221, i64 %epoch222)
  call void @tl_mem_unregister(ptr %call_tmp)
  %old_struct_to_free = load ptr, ptr %model, align 8
  %is_not_null = icmp ne ptr %old_struct_to_free, null
  %are_diff = icmp ne ptr %old_struct_to_free, %call_tmp
  %can_free_1 = and i1 %is_not_null, true
  %can_free = and i1 %can_free_1, %are_diff
  br i1 %can_free, label %free_struct, label %continue_after_free

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal.247)
  %model923 = load ptr, ptr %model, align 8
  %w924 = getelementptr inbounds %GPTHeavy, ptr %model923, i32 0, i32 0
  %sub_ptr925 = load ptr, ptr %w924, align 8
  %w926 = getelementptr inbounds %Embedding, ptr %sub_ptr925, i32 0, i32 0
  %w927 = load ptr, ptr %w926, align 8
  call void @tl_add_parameter(ptr @key_str.248, ptr %w927)
  %wp928 = getelementptr inbounds %GPTHeavy, ptr %model923, i32 0, i32 1
  %sub_ptr929 = load ptr, ptr %wp928, align 8
  %w930 = getelementptr inbounds %Embedding, ptr %sub_ptr929, i32 0, i32 0
  %w931 = load ptr, ptr %w930, align 8
  call void @tl_add_parameter(ptr @key_str.249, ptr %w931)
  %b1932 = getelementptr inbounds %GPTHeavy, ptr %model923, i32 0, i32 2
  %sub_ptr933 = load ptr, ptr %b1932, align 8
  %l1934 = getelementptr inbounds %Block, ptr %sub_ptr933, i32 0, i32 0
  %sub_ptr935 = load ptr, ptr %l1934, align 8
  %w936 = getelementptr inbounds %LayerNorm, ptr %sub_ptr935, i32 0, i32 0
  %w937 = load ptr, ptr %w936, align 8
  call void @tl_add_parameter(ptr @key_str.250, ptr %w937)
  %b938 = getelementptr inbounds %LayerNorm, ptr %sub_ptr935, i32 0, i32 1
  %b939 = load ptr, ptr %b938, align 8
  call void @tl_add_parameter(ptr @key_str.251, ptr %b939)
  %a940 = getelementptr inbounds %Block, ptr %sub_ptr933, i32 0, i32 1
  %sub_ptr941 = load ptr, ptr %a940, align 8
  %q_proj942 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr941, i32 0, i32 0
  %sub_ptr943 = load ptr, ptr %q_proj942, align 8
  %W944 = getelementptr inbounds %Linear, ptr %sub_ptr943, i32 0, i32 0
  %W945 = load ptr, ptr %W944, align 8
  call void @tl_add_parameter(ptr @key_str.252, ptr %W945)
  %b946 = getelementptr inbounds %Linear, ptr %sub_ptr943, i32 0, i32 1
  %b947 = load ptr, ptr %b946, align 8
  call void @tl_add_parameter(ptr @key_str.253, ptr %b947)
  %k_proj948 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr941, i32 0, i32 1
  %sub_ptr949 = load ptr, ptr %k_proj948, align 8
  %W950 = getelementptr inbounds %Linear, ptr %sub_ptr949, i32 0, i32 0
  %W951 = load ptr, ptr %W950, align 8
  call void @tl_add_parameter(ptr @key_str.254, ptr %W951)
  %b952 = getelementptr inbounds %Linear, ptr %sub_ptr949, i32 0, i32 1
  %b953 = load ptr, ptr %b952, align 8
  call void @tl_add_parameter(ptr @key_str.255, ptr %b953)
  %v_proj954 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr941, i32 0, i32 2
  %sub_ptr955 = load ptr, ptr %v_proj954, align 8
  %W956 = getelementptr inbounds %Linear, ptr %sub_ptr955, i32 0, i32 0
  %W957 = load ptr, ptr %W956, align 8
  call void @tl_add_parameter(ptr @key_str.256, ptr %W957)
  %b958 = getelementptr inbounds %Linear, ptr %sub_ptr955, i32 0, i32 1
  %b959 = load ptr, ptr %b958, align 8
  call void @tl_add_parameter(ptr @key_str.257, ptr %b959)
  %p_proj960 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr941, i32 0, i32 3
  %sub_ptr961 = load ptr, ptr %p_proj960, align 8
  %W962 = getelementptr inbounds %Linear, ptr %sub_ptr961, i32 0, i32 0
  %W963 = load ptr, ptr %W962, align 8
  call void @tl_add_parameter(ptr @key_str.258, ptr %W963)
  %b964 = getelementptr inbounds %Linear, ptr %sub_ptr961, i32 0, i32 1
  %b965 = load ptr, ptr %b964, align 8
  call void @tl_add_parameter(ptr @key_str.259, ptr %b965)
  %l2966 = getelementptr inbounds %Block, ptr %sub_ptr933, i32 0, i32 2
  %sub_ptr967 = load ptr, ptr %l2966, align 8
  %w968 = getelementptr inbounds %LayerNorm, ptr %sub_ptr967, i32 0, i32 0
  %w969 = load ptr, ptr %w968, align 8
  call void @tl_add_parameter(ptr @key_str.260, ptr %w969)
  %b970 = getelementptr inbounds %LayerNorm, ptr %sub_ptr967, i32 0, i32 1
  %b971 = load ptr, ptr %b970, align 8
  call void @tl_add_parameter(ptr @key_str.261, ptr %b971)
  %m972 = getelementptr inbounds %Block, ptr %sub_ptr933, i32 0, i32 3
  %sub_ptr973 = load ptr, ptr %m972, align 8
  %f974 = getelementptr inbounds %MLP, ptr %sub_ptr973, i32 0, i32 0
  %sub_ptr975 = load ptr, ptr %f974, align 8
  %W976 = getelementptr inbounds %Linear, ptr %sub_ptr975, i32 0, i32 0
  %W977 = load ptr, ptr %W976, align 8
  call void @tl_add_parameter(ptr @key_str.262, ptr %W977)
  %b978 = getelementptr inbounds %Linear, ptr %sub_ptr975, i32 0, i32 1
  %b979 = load ptr, ptr %b978, align 8
  call void @tl_add_parameter(ptr @key_str.263, ptr %b979)
  %p980 = getelementptr inbounds %MLP, ptr %sub_ptr973, i32 0, i32 1
  %sub_ptr981 = load ptr, ptr %p980, align 8
  %W982 = getelementptr inbounds %Linear, ptr %sub_ptr981, i32 0, i32 0
  %W983 = load ptr, ptr %W982, align 8
  call void @tl_add_parameter(ptr @key_str.264, ptr %W983)
  %b984 = getelementptr inbounds %Linear, ptr %sub_ptr981, i32 0, i32 1
  %b985 = load ptr, ptr %b984, align 8
  call void @tl_add_parameter(ptr @key_str.265, ptr %b985)
  %b2986 = getelementptr inbounds %GPTHeavy, ptr %model923, i32 0, i32 3
  %sub_ptr987 = load ptr, ptr %b2986, align 8
  %l1988 = getelementptr inbounds %Block, ptr %sub_ptr987, i32 0, i32 0
  %sub_ptr989 = load ptr, ptr %l1988, align 8
  %w990 = getelementptr inbounds %LayerNorm, ptr %sub_ptr989, i32 0, i32 0
  %w991 = load ptr, ptr %w990, align 8
  call void @tl_add_parameter(ptr @key_str.266, ptr %w991)
  %b992 = getelementptr inbounds %LayerNorm, ptr %sub_ptr989, i32 0, i32 1
  %b993 = load ptr, ptr %b992, align 8
  call void @tl_add_parameter(ptr @key_str.267, ptr %b993)
  %a994 = getelementptr inbounds %Block, ptr %sub_ptr987, i32 0, i32 1
  %sub_ptr995 = load ptr, ptr %a994, align 8
  %q_proj996 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr995, i32 0, i32 0
  %sub_ptr997 = load ptr, ptr %q_proj996, align 8
  %W998 = getelementptr inbounds %Linear, ptr %sub_ptr997, i32 0, i32 0
  %W999 = load ptr, ptr %W998, align 8
  call void @tl_add_parameter(ptr @key_str.268, ptr %W999)
  %b1000 = getelementptr inbounds %Linear, ptr %sub_ptr997, i32 0, i32 1
  %b1001 = load ptr, ptr %b1000, align 8
  call void @tl_add_parameter(ptr @key_str.269, ptr %b1001)
  %k_proj1002 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr995, i32 0, i32 1
  %sub_ptr1003 = load ptr, ptr %k_proj1002, align 8
  %W1004 = getelementptr inbounds %Linear, ptr %sub_ptr1003, i32 0, i32 0
  %W1005 = load ptr, ptr %W1004, align 8
  call void @tl_add_parameter(ptr @key_str.270, ptr %W1005)
  %b1006 = getelementptr inbounds %Linear, ptr %sub_ptr1003, i32 0, i32 1
  %b1007 = load ptr, ptr %b1006, align 8
  call void @tl_add_parameter(ptr @key_str.271, ptr %b1007)
  %v_proj1008 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr995, i32 0, i32 2
  %sub_ptr1009 = load ptr, ptr %v_proj1008, align 8
  %W1010 = getelementptr inbounds %Linear, ptr %sub_ptr1009, i32 0, i32 0
  %W1011 = load ptr, ptr %W1010, align 8
  call void @tl_add_parameter(ptr @key_str.272, ptr %W1011)
  %b1012 = getelementptr inbounds %Linear, ptr %sub_ptr1009, i32 0, i32 1
  %b1013 = load ptr, ptr %b1012, align 8
  call void @tl_add_parameter(ptr @key_str.273, ptr %b1013)
  %p_proj1014 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr995, i32 0, i32 3
  %sub_ptr1015 = load ptr, ptr %p_proj1014, align 8
  %W1016 = getelementptr inbounds %Linear, ptr %sub_ptr1015, i32 0, i32 0
  %W1017 = load ptr, ptr %W1016, align 8
  call void @tl_add_parameter(ptr @key_str.274, ptr %W1017)
  %b1018 = getelementptr inbounds %Linear, ptr %sub_ptr1015, i32 0, i32 1
  %b1019 = load ptr, ptr %b1018, align 8
  call void @tl_add_parameter(ptr @key_str.275, ptr %b1019)
  %l21020 = getelementptr inbounds %Block, ptr %sub_ptr987, i32 0, i32 2
  %sub_ptr1021 = load ptr, ptr %l21020, align 8
  %w1022 = getelementptr inbounds %LayerNorm, ptr %sub_ptr1021, i32 0, i32 0
  %w1023 = load ptr, ptr %w1022, align 8
  call void @tl_add_parameter(ptr @key_str.276, ptr %w1023)
  %b1024 = getelementptr inbounds %LayerNorm, ptr %sub_ptr1021, i32 0, i32 1
  %b1025 = load ptr, ptr %b1024, align 8
  call void @tl_add_parameter(ptr @key_str.277, ptr %b1025)
  %m1026 = getelementptr inbounds %Block, ptr %sub_ptr987, i32 0, i32 3
  %sub_ptr1027 = load ptr, ptr %m1026, align 8
  %f1028 = getelementptr inbounds %MLP, ptr %sub_ptr1027, i32 0, i32 0
  %sub_ptr1029 = load ptr, ptr %f1028, align 8
  %W1030 = getelementptr inbounds %Linear, ptr %sub_ptr1029, i32 0, i32 0
  %W1031 = load ptr, ptr %W1030, align 8
  call void @tl_add_parameter(ptr @key_str.278, ptr %W1031)
  %b1032 = getelementptr inbounds %Linear, ptr %sub_ptr1029, i32 0, i32 1
  %b1033 = load ptr, ptr %b1032, align 8
  call void @tl_add_parameter(ptr @key_str.279, ptr %b1033)
  %p1034 = getelementptr inbounds %MLP, ptr %sub_ptr1027, i32 0, i32 1
  %sub_ptr1035 = load ptr, ptr %p1034, align 8
  %W1036 = getelementptr inbounds %Linear, ptr %sub_ptr1035, i32 0, i32 0
  %W1037 = load ptr, ptr %W1036, align 8
  call void @tl_add_parameter(ptr @key_str.280, ptr %W1037)
  %b1038 = getelementptr inbounds %Linear, ptr %sub_ptr1035, i32 0, i32 1
  %b1039 = load ptr, ptr %b1038, align 8
  call void @tl_add_parameter(ptr @key_str.281, ptr %b1039)
  %b31040 = getelementptr inbounds %GPTHeavy, ptr %model923, i32 0, i32 4
  %sub_ptr1041 = load ptr, ptr %b31040, align 8
  %l11042 = getelementptr inbounds %Block, ptr %sub_ptr1041, i32 0, i32 0
  %sub_ptr1043 = load ptr, ptr %l11042, align 8
  %w1044 = getelementptr inbounds %LayerNorm, ptr %sub_ptr1043, i32 0, i32 0
  %w1045 = load ptr, ptr %w1044, align 8
  call void @tl_add_parameter(ptr @key_str.282, ptr %w1045)
  %b1046 = getelementptr inbounds %LayerNorm, ptr %sub_ptr1043, i32 0, i32 1
  %b1047 = load ptr, ptr %b1046, align 8
  call void @tl_add_parameter(ptr @key_str.283, ptr %b1047)
  %a1048 = getelementptr inbounds %Block, ptr %sub_ptr1041, i32 0, i32 1
  %sub_ptr1049 = load ptr, ptr %a1048, align 8
  %q_proj1050 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr1049, i32 0, i32 0
  %sub_ptr1051 = load ptr, ptr %q_proj1050, align 8
  %W1052 = getelementptr inbounds %Linear, ptr %sub_ptr1051, i32 0, i32 0
  %W1053 = load ptr, ptr %W1052, align 8
  call void @tl_add_parameter(ptr @key_str.284, ptr %W1053)
  %b1054 = getelementptr inbounds %Linear, ptr %sub_ptr1051, i32 0, i32 1
  %b1055 = load ptr, ptr %b1054, align 8
  call void @tl_add_parameter(ptr @key_str.285, ptr %b1055)
  %k_proj1056 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr1049, i32 0, i32 1
  %sub_ptr1057 = load ptr, ptr %k_proj1056, align 8
  %W1058 = getelementptr inbounds %Linear, ptr %sub_ptr1057, i32 0, i32 0
  %W1059 = load ptr, ptr %W1058, align 8
  call void @tl_add_parameter(ptr @key_str.286, ptr %W1059)
  %b1060 = getelementptr inbounds %Linear, ptr %sub_ptr1057, i32 0, i32 1
  %b1061 = load ptr, ptr %b1060, align 8
  call void @tl_add_parameter(ptr @key_str.287, ptr %b1061)
  %v_proj1062 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr1049, i32 0, i32 2
  %sub_ptr1063 = load ptr, ptr %v_proj1062, align 8
  %W1064 = getelementptr inbounds %Linear, ptr %sub_ptr1063, i32 0, i32 0
  %W1065 = load ptr, ptr %W1064, align 8
  call void @tl_add_parameter(ptr @key_str.288, ptr %W1065)
  %b1066 = getelementptr inbounds %Linear, ptr %sub_ptr1063, i32 0, i32 1
  %b1067 = load ptr, ptr %b1066, align 8
  call void @tl_add_parameter(ptr @key_str.289, ptr %b1067)
  %p_proj1068 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr1049, i32 0, i32 3
  %sub_ptr1069 = load ptr, ptr %p_proj1068, align 8
  %W1070 = getelementptr inbounds %Linear, ptr %sub_ptr1069, i32 0, i32 0
  %W1071 = load ptr, ptr %W1070, align 8
  call void @tl_add_parameter(ptr @key_str.290, ptr %W1071)
  %b1072 = getelementptr inbounds %Linear, ptr %sub_ptr1069, i32 0, i32 1
  %b1073 = load ptr, ptr %b1072, align 8
  call void @tl_add_parameter(ptr @key_str.291, ptr %b1073)
  %l21074 = getelementptr inbounds %Block, ptr %sub_ptr1041, i32 0, i32 2
  %sub_ptr1075 = load ptr, ptr %l21074, align 8
  %w1076 = getelementptr inbounds %LayerNorm, ptr %sub_ptr1075, i32 0, i32 0
  %w1077 = load ptr, ptr %w1076, align 8
  call void @tl_add_parameter(ptr @key_str.292, ptr %w1077)
  %b1078 = getelementptr inbounds %LayerNorm, ptr %sub_ptr1075, i32 0, i32 1
  %b1079 = load ptr, ptr %b1078, align 8
  call void @tl_add_parameter(ptr @key_str.293, ptr %b1079)
  %m1080 = getelementptr inbounds %Block, ptr %sub_ptr1041, i32 0, i32 3
  %sub_ptr1081 = load ptr, ptr %m1080, align 8
  %f1082 = getelementptr inbounds %MLP, ptr %sub_ptr1081, i32 0, i32 0
  %sub_ptr1083 = load ptr, ptr %f1082, align 8
  %W1084 = getelementptr inbounds %Linear, ptr %sub_ptr1083, i32 0, i32 0
  %W1085 = load ptr, ptr %W1084, align 8
  call void @tl_add_parameter(ptr @key_str.294, ptr %W1085)
  %b1086 = getelementptr inbounds %Linear, ptr %sub_ptr1083, i32 0, i32 1
  %b1087 = load ptr, ptr %b1086, align 8
  call void @tl_add_parameter(ptr @key_str.295, ptr %b1087)
  %p1088 = getelementptr inbounds %MLP, ptr %sub_ptr1081, i32 0, i32 1
  %sub_ptr1089 = load ptr, ptr %p1088, align 8
  %W1090 = getelementptr inbounds %Linear, ptr %sub_ptr1089, i32 0, i32 0
  %W1091 = load ptr, ptr %W1090, align 8
  call void @tl_add_parameter(ptr @key_str.296, ptr %W1091)
  %b1092 = getelementptr inbounds %Linear, ptr %sub_ptr1089, i32 0, i32 1
  %b1093 = load ptr, ptr %b1092, align 8
  call void @tl_add_parameter(ptr @key_str.297, ptr %b1093)
  %b41094 = getelementptr inbounds %GPTHeavy, ptr %model923, i32 0, i32 5
  %sub_ptr1095 = load ptr, ptr %b41094, align 8
  %l11096 = getelementptr inbounds %Block, ptr %sub_ptr1095, i32 0, i32 0
  %sub_ptr1097 = load ptr, ptr %l11096, align 8
  %w1098 = getelementptr inbounds %LayerNorm, ptr %sub_ptr1097, i32 0, i32 0
  %w1099 = load ptr, ptr %w1098, align 8
  call void @tl_add_parameter(ptr @key_str.298, ptr %w1099)
  %b1100 = getelementptr inbounds %LayerNorm, ptr %sub_ptr1097, i32 0, i32 1
  %b1101 = load ptr, ptr %b1100, align 8
  call void @tl_add_parameter(ptr @key_str.299, ptr %b1101)
  %a1102 = getelementptr inbounds %Block, ptr %sub_ptr1095, i32 0, i32 1
  %sub_ptr1103 = load ptr, ptr %a1102, align 8
  %q_proj1104 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr1103, i32 0, i32 0
  %sub_ptr1105 = load ptr, ptr %q_proj1104, align 8
  %W1106 = getelementptr inbounds %Linear, ptr %sub_ptr1105, i32 0, i32 0
  %W1107 = load ptr, ptr %W1106, align 8
  call void @tl_add_parameter(ptr @key_str.300, ptr %W1107)
  %b1108 = getelementptr inbounds %Linear, ptr %sub_ptr1105, i32 0, i32 1
  %b1109 = load ptr, ptr %b1108, align 8
  call void @tl_add_parameter(ptr @key_str.301, ptr %b1109)
  %k_proj1110 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr1103, i32 0, i32 1
  %sub_ptr1111 = load ptr, ptr %k_proj1110, align 8
  %W1112 = getelementptr inbounds %Linear, ptr %sub_ptr1111, i32 0, i32 0
  %W1113 = load ptr, ptr %W1112, align 8
  call void @tl_add_parameter(ptr @key_str.302, ptr %W1113)
  %b1114 = getelementptr inbounds %Linear, ptr %sub_ptr1111, i32 0, i32 1
  %b1115 = load ptr, ptr %b1114, align 8
  call void @tl_add_parameter(ptr @key_str.303, ptr %b1115)
  %v_proj1116 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr1103, i32 0, i32 2
  %sub_ptr1117 = load ptr, ptr %v_proj1116, align 8
  %W1118 = getelementptr inbounds %Linear, ptr %sub_ptr1117, i32 0, i32 0
  %W1119 = load ptr, ptr %W1118, align 8
  call void @tl_add_parameter(ptr @key_str.304, ptr %W1119)
  %b1120 = getelementptr inbounds %Linear, ptr %sub_ptr1117, i32 0, i32 1
  %b1121 = load ptr, ptr %b1120, align 8
  call void @tl_add_parameter(ptr @key_str.305, ptr %b1121)
  %p_proj1122 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr1103, i32 0, i32 3
  %sub_ptr1123 = load ptr, ptr %p_proj1122, align 8
  %W1124 = getelementptr inbounds %Linear, ptr %sub_ptr1123, i32 0, i32 0
  %W1125 = load ptr, ptr %W1124, align 8
  call void @tl_add_parameter(ptr @key_str.306, ptr %W1125)
  %b1126 = getelementptr inbounds %Linear, ptr %sub_ptr1123, i32 0, i32 1
  %b1127 = load ptr, ptr %b1126, align 8
  call void @tl_add_parameter(ptr @key_str.307, ptr %b1127)
  %l21128 = getelementptr inbounds %Block, ptr %sub_ptr1095, i32 0, i32 2
  %sub_ptr1129 = load ptr, ptr %l21128, align 8
  %w1130 = getelementptr inbounds %LayerNorm, ptr %sub_ptr1129, i32 0, i32 0
  %w1131 = load ptr, ptr %w1130, align 8
  call void @tl_add_parameter(ptr @key_str.308, ptr %w1131)
  %b1132 = getelementptr inbounds %LayerNorm, ptr %sub_ptr1129, i32 0, i32 1
  %b1133 = load ptr, ptr %b1132, align 8
  call void @tl_add_parameter(ptr @key_str.309, ptr %b1133)
  %m1134 = getelementptr inbounds %Block, ptr %sub_ptr1095, i32 0, i32 3
  %sub_ptr1135 = load ptr, ptr %m1134, align 8
  %f1136 = getelementptr inbounds %MLP, ptr %sub_ptr1135, i32 0, i32 0
  %sub_ptr1137 = load ptr, ptr %f1136, align 8
  %W1138 = getelementptr inbounds %Linear, ptr %sub_ptr1137, i32 0, i32 0
  %W1139 = load ptr, ptr %W1138, align 8
  call void @tl_add_parameter(ptr @key_str.310, ptr %W1139)
  %b1140 = getelementptr inbounds %Linear, ptr %sub_ptr1137, i32 0, i32 1
  %b1141 = load ptr, ptr %b1140, align 8
  call void @tl_add_parameter(ptr @key_str.311, ptr %b1141)
  %p1142 = getelementptr inbounds %MLP, ptr %sub_ptr1135, i32 0, i32 1
  %sub_ptr1143 = load ptr, ptr %p1142, align 8
  %W1144 = getelementptr inbounds %Linear, ptr %sub_ptr1143, i32 0, i32 0
  %W1145 = load ptr, ptr %W1144, align 8
  call void @tl_add_parameter(ptr @key_str.312, ptr %W1145)
  %b1146 = getelementptr inbounds %Linear, ptr %sub_ptr1143, i32 0, i32 1
  %b1147 = load ptr, ptr %b1146, align 8
  call void @tl_add_parameter(ptr @key_str.313, ptr %b1147)
  %l1148 = getelementptr inbounds %GPTHeavy, ptr %model923, i32 0, i32 6
  %sub_ptr1149 = load ptr, ptr %l1148, align 8
  %w1150 = getelementptr inbounds %LayerNorm, ptr %sub_ptr1149, i32 0, i32 0
  %w1151 = load ptr, ptr %w1150, align 8
  call void @tl_add_parameter(ptr @key_str.314, ptr %w1151)
  %b1152 = getelementptr inbounds %LayerNorm, ptr %sub_ptr1149, i32 0, i32 1
  %b1153 = load ptr, ptr %b1152, align 8
  call void @tl_add_parameter(ptr @key_str.315, ptr %b1153)
  %h1154 = getelementptr inbounds %GPTHeavy, ptr %model923, i32 0, i32 7
  %sub_ptr1155 = load ptr, ptr %h1154, align 8
  %W1156 = getelementptr inbounds %Linear, ptr %sub_ptr1155, i32 0, i32 0
  %W1157 = load ptr, ptr %W1156, align 8
  call void @tl_add_parameter(ptr @key_str.316, ptr %W1157)
  %b1158 = getelementptr inbounds %Linear, ptr %sub_ptr1155, i32 0, i32 1
  %b1159 = load ptr, ptr %b1158, align 8
  call void @tl_add_parameter(ptr @key_str.317, ptr %b1159)
  call void @tl_save_all_params(ptr @str_literal.318)
  %struct_to_free = load ptr, ptr %model, align 8
  %field_gep1160 = getelementptr inbounds %GPTHeavy, ptr %struct_to_free, i32 0, i32 0
  %field_load1161 = load ptr, ptr %field_gep1160, align 8
  %field_gep1162 = getelementptr inbounds %Embedding, ptr %field_load1161, i32 0, i32 0
  %field_load1163 = load ptr, ptr %field_gep1162, align 8
  call void @tl_tensor_free(ptr %field_load1163)
  %field_gep1164 = getelementptr inbounds %GPTHeavy, ptr %struct_to_free, i32 0, i32 1
  %field_load1165 = load ptr, ptr %field_gep1164, align 8
  %field_gep1166 = getelementptr inbounds %Embedding, ptr %field_load1165, i32 0, i32 0
  %field_load1167 = load ptr, ptr %field_gep1166, align 8
  call void @tl_tensor_free(ptr %field_load1167)
  %field_gep1168 = getelementptr inbounds %GPTHeavy, ptr %struct_to_free, i32 0, i32 2
  %field_load1169 = load ptr, ptr %field_gep1168, align 8
  %field_gep1170 = getelementptr inbounds %Block, ptr %field_load1169, i32 0, i32 0
  %field_load1171 = load ptr, ptr %field_gep1170, align 8
  %field_gep1172 = getelementptr inbounds %LayerNorm, ptr %field_load1171, i32 0, i32 0
  %field_load1173 = load ptr, ptr %field_gep1172, align 8
  call void @tl_tensor_free(ptr %field_load1173)
  %field_gep1174 = getelementptr inbounds %LayerNorm, ptr %field_load1171, i32 0, i32 1
  %field_load1175 = load ptr, ptr %field_gep1174, align 8
  call void @tl_tensor_free(ptr %field_load1175)
  %field_gep1176 = getelementptr inbounds %Block, ptr %field_load1169, i32 0, i32 1
  %field_load1177 = load ptr, ptr %field_gep1176, align 8
  %field_gep1178 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1177, i32 0, i32 0
  %field_load1179 = load ptr, ptr %field_gep1178, align 8
  %field_gep1180 = getelementptr inbounds %Linear, ptr %field_load1179, i32 0, i32 0
  %field_load1181 = load ptr, ptr %field_gep1180, align 8
  call void @tl_tensor_free(ptr %field_load1181)
  %field_gep1182 = getelementptr inbounds %Linear, ptr %field_load1179, i32 0, i32 1
  %field_load1183 = load ptr, ptr %field_gep1182, align 8
  call void @tl_tensor_free(ptr %field_load1183)
  %field_gep1184 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1177, i32 0, i32 1
  %field_load1185 = load ptr, ptr %field_gep1184, align 8
  %field_gep1186 = getelementptr inbounds %Linear, ptr %field_load1185, i32 0, i32 0
  %field_load1187 = load ptr, ptr %field_gep1186, align 8
  call void @tl_tensor_free(ptr %field_load1187)
  %field_gep1188 = getelementptr inbounds %Linear, ptr %field_load1185, i32 0, i32 1
  %field_load1189 = load ptr, ptr %field_gep1188, align 8
  call void @tl_tensor_free(ptr %field_load1189)
  %field_gep1190 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1177, i32 0, i32 2
  %field_load1191 = load ptr, ptr %field_gep1190, align 8
  %field_gep1192 = getelementptr inbounds %Linear, ptr %field_load1191, i32 0, i32 0
  %field_load1193 = load ptr, ptr %field_gep1192, align 8
  call void @tl_tensor_free(ptr %field_load1193)
  %field_gep1194 = getelementptr inbounds %Linear, ptr %field_load1191, i32 0, i32 1
  %field_load1195 = load ptr, ptr %field_gep1194, align 8
  call void @tl_tensor_free(ptr %field_load1195)
  %field_gep1196 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1177, i32 0, i32 3
  %field_load1197 = load ptr, ptr %field_gep1196, align 8
  %field_gep1198 = getelementptr inbounds %Linear, ptr %field_load1197, i32 0, i32 0
  %field_load1199 = load ptr, ptr %field_gep1198, align 8
  call void @tl_tensor_free(ptr %field_load1199)
  %field_gep1200 = getelementptr inbounds %Linear, ptr %field_load1197, i32 0, i32 1
  %field_load1201 = load ptr, ptr %field_gep1200, align 8
  call void @tl_tensor_free(ptr %field_load1201)
  %field_gep1202 = getelementptr inbounds %Block, ptr %field_load1169, i32 0, i32 2
  %field_load1203 = load ptr, ptr %field_gep1202, align 8
  %field_gep1204 = getelementptr inbounds %LayerNorm, ptr %field_load1203, i32 0, i32 0
  %field_load1205 = load ptr, ptr %field_gep1204, align 8
  call void @tl_tensor_free(ptr %field_load1205)
  %field_gep1206 = getelementptr inbounds %LayerNorm, ptr %field_load1203, i32 0, i32 1
  %field_load1207 = load ptr, ptr %field_gep1206, align 8
  call void @tl_tensor_free(ptr %field_load1207)
  %field_gep1208 = getelementptr inbounds %Block, ptr %field_load1169, i32 0, i32 3
  %field_load1209 = load ptr, ptr %field_gep1208, align 8
  %field_gep1210 = getelementptr inbounds %MLP, ptr %field_load1209, i32 0, i32 0
  %field_load1211 = load ptr, ptr %field_gep1210, align 8
  %field_gep1212 = getelementptr inbounds %Linear, ptr %field_load1211, i32 0, i32 0
  %field_load1213 = load ptr, ptr %field_gep1212, align 8
  call void @tl_tensor_free(ptr %field_load1213)
  %field_gep1214 = getelementptr inbounds %Linear, ptr %field_load1211, i32 0, i32 1
  %field_load1215 = load ptr, ptr %field_gep1214, align 8
  call void @tl_tensor_free(ptr %field_load1215)
  %field_gep1216 = getelementptr inbounds %MLP, ptr %field_load1209, i32 0, i32 1
  %field_load1217 = load ptr, ptr %field_gep1216, align 8
  %field_gep1218 = getelementptr inbounds %Linear, ptr %field_load1217, i32 0, i32 0
  %field_load1219 = load ptr, ptr %field_gep1218, align 8
  call void @tl_tensor_free(ptr %field_load1219)
  %field_gep1220 = getelementptr inbounds %Linear, ptr %field_load1217, i32 0, i32 1
  %field_load1221 = load ptr, ptr %field_gep1220, align 8
  call void @tl_tensor_free(ptr %field_load1221)
  %field_gep1222 = getelementptr inbounds %GPTHeavy, ptr %struct_to_free, i32 0, i32 3
  %field_load1223 = load ptr, ptr %field_gep1222, align 8
  %field_gep1224 = getelementptr inbounds %Block, ptr %field_load1223, i32 0, i32 0
  %field_load1225 = load ptr, ptr %field_gep1224, align 8
  %field_gep1226 = getelementptr inbounds %LayerNorm, ptr %field_load1225, i32 0, i32 0
  %field_load1227 = load ptr, ptr %field_gep1226, align 8
  call void @tl_tensor_free(ptr %field_load1227)
  %field_gep1228 = getelementptr inbounds %LayerNorm, ptr %field_load1225, i32 0, i32 1
  %field_load1229 = load ptr, ptr %field_gep1228, align 8
  call void @tl_tensor_free(ptr %field_load1229)
  %field_gep1230 = getelementptr inbounds %Block, ptr %field_load1223, i32 0, i32 1
  %field_load1231 = load ptr, ptr %field_gep1230, align 8
  %field_gep1232 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1231, i32 0, i32 0
  %field_load1233 = load ptr, ptr %field_gep1232, align 8
  %field_gep1234 = getelementptr inbounds %Linear, ptr %field_load1233, i32 0, i32 0
  %field_load1235 = load ptr, ptr %field_gep1234, align 8
  call void @tl_tensor_free(ptr %field_load1235)
  %field_gep1236 = getelementptr inbounds %Linear, ptr %field_load1233, i32 0, i32 1
  %field_load1237 = load ptr, ptr %field_gep1236, align 8
  call void @tl_tensor_free(ptr %field_load1237)
  %field_gep1238 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1231, i32 0, i32 1
  %field_load1239 = load ptr, ptr %field_gep1238, align 8
  %field_gep1240 = getelementptr inbounds %Linear, ptr %field_load1239, i32 0, i32 0
  %field_load1241 = load ptr, ptr %field_gep1240, align 8
  call void @tl_tensor_free(ptr %field_load1241)
  %field_gep1242 = getelementptr inbounds %Linear, ptr %field_load1239, i32 0, i32 1
  %field_load1243 = load ptr, ptr %field_gep1242, align 8
  call void @tl_tensor_free(ptr %field_load1243)
  %field_gep1244 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1231, i32 0, i32 2
  %field_load1245 = load ptr, ptr %field_gep1244, align 8
  %field_gep1246 = getelementptr inbounds %Linear, ptr %field_load1245, i32 0, i32 0
  %field_load1247 = load ptr, ptr %field_gep1246, align 8
  call void @tl_tensor_free(ptr %field_load1247)
  %field_gep1248 = getelementptr inbounds %Linear, ptr %field_load1245, i32 0, i32 1
  %field_load1249 = load ptr, ptr %field_gep1248, align 8
  call void @tl_tensor_free(ptr %field_load1249)
  %field_gep1250 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1231, i32 0, i32 3
  %field_load1251 = load ptr, ptr %field_gep1250, align 8
  %field_gep1252 = getelementptr inbounds %Linear, ptr %field_load1251, i32 0, i32 0
  %field_load1253 = load ptr, ptr %field_gep1252, align 8
  call void @tl_tensor_free(ptr %field_load1253)
  %field_gep1254 = getelementptr inbounds %Linear, ptr %field_load1251, i32 0, i32 1
  %field_load1255 = load ptr, ptr %field_gep1254, align 8
  call void @tl_tensor_free(ptr %field_load1255)
  %field_gep1256 = getelementptr inbounds %Block, ptr %field_load1223, i32 0, i32 2
  %field_load1257 = load ptr, ptr %field_gep1256, align 8
  %field_gep1258 = getelementptr inbounds %LayerNorm, ptr %field_load1257, i32 0, i32 0
  %field_load1259 = load ptr, ptr %field_gep1258, align 8
  call void @tl_tensor_free(ptr %field_load1259)
  %field_gep1260 = getelementptr inbounds %LayerNorm, ptr %field_load1257, i32 0, i32 1
  %field_load1261 = load ptr, ptr %field_gep1260, align 8
  call void @tl_tensor_free(ptr %field_load1261)
  %field_gep1262 = getelementptr inbounds %Block, ptr %field_load1223, i32 0, i32 3
  %field_load1263 = load ptr, ptr %field_gep1262, align 8
  %field_gep1264 = getelementptr inbounds %MLP, ptr %field_load1263, i32 0, i32 0
  %field_load1265 = load ptr, ptr %field_gep1264, align 8
  %field_gep1266 = getelementptr inbounds %Linear, ptr %field_load1265, i32 0, i32 0
  %field_load1267 = load ptr, ptr %field_gep1266, align 8
  call void @tl_tensor_free(ptr %field_load1267)
  %field_gep1268 = getelementptr inbounds %Linear, ptr %field_load1265, i32 0, i32 1
  %field_load1269 = load ptr, ptr %field_gep1268, align 8
  call void @tl_tensor_free(ptr %field_load1269)
  %field_gep1270 = getelementptr inbounds %MLP, ptr %field_load1263, i32 0, i32 1
  %field_load1271 = load ptr, ptr %field_gep1270, align 8
  %field_gep1272 = getelementptr inbounds %Linear, ptr %field_load1271, i32 0, i32 0
  %field_load1273 = load ptr, ptr %field_gep1272, align 8
  call void @tl_tensor_free(ptr %field_load1273)
  %field_gep1274 = getelementptr inbounds %Linear, ptr %field_load1271, i32 0, i32 1
  %field_load1275 = load ptr, ptr %field_gep1274, align 8
  call void @tl_tensor_free(ptr %field_load1275)
  %field_gep1276 = getelementptr inbounds %GPTHeavy, ptr %struct_to_free, i32 0, i32 4
  %field_load1277 = load ptr, ptr %field_gep1276, align 8
  %field_gep1278 = getelementptr inbounds %Block, ptr %field_load1277, i32 0, i32 0
  %field_load1279 = load ptr, ptr %field_gep1278, align 8
  %field_gep1280 = getelementptr inbounds %LayerNorm, ptr %field_load1279, i32 0, i32 0
  %field_load1281 = load ptr, ptr %field_gep1280, align 8
  call void @tl_tensor_free(ptr %field_load1281)
  %field_gep1282 = getelementptr inbounds %LayerNorm, ptr %field_load1279, i32 0, i32 1
  %field_load1283 = load ptr, ptr %field_gep1282, align 8
  call void @tl_tensor_free(ptr %field_load1283)
  %field_gep1284 = getelementptr inbounds %Block, ptr %field_load1277, i32 0, i32 1
  %field_load1285 = load ptr, ptr %field_gep1284, align 8
  %field_gep1286 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1285, i32 0, i32 0
  %field_load1287 = load ptr, ptr %field_gep1286, align 8
  %field_gep1288 = getelementptr inbounds %Linear, ptr %field_load1287, i32 0, i32 0
  %field_load1289 = load ptr, ptr %field_gep1288, align 8
  call void @tl_tensor_free(ptr %field_load1289)
  %field_gep1290 = getelementptr inbounds %Linear, ptr %field_load1287, i32 0, i32 1
  %field_load1291 = load ptr, ptr %field_gep1290, align 8
  call void @tl_tensor_free(ptr %field_load1291)
  %field_gep1292 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1285, i32 0, i32 1
  %field_load1293 = load ptr, ptr %field_gep1292, align 8
  %field_gep1294 = getelementptr inbounds %Linear, ptr %field_load1293, i32 0, i32 0
  %field_load1295 = load ptr, ptr %field_gep1294, align 8
  call void @tl_tensor_free(ptr %field_load1295)
  %field_gep1296 = getelementptr inbounds %Linear, ptr %field_load1293, i32 0, i32 1
  %field_load1297 = load ptr, ptr %field_gep1296, align 8
  call void @tl_tensor_free(ptr %field_load1297)
  %field_gep1298 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1285, i32 0, i32 2
  %field_load1299 = load ptr, ptr %field_gep1298, align 8
  %field_gep1300 = getelementptr inbounds %Linear, ptr %field_load1299, i32 0, i32 0
  %field_load1301 = load ptr, ptr %field_gep1300, align 8
  call void @tl_tensor_free(ptr %field_load1301)
  %field_gep1302 = getelementptr inbounds %Linear, ptr %field_load1299, i32 0, i32 1
  %field_load1303 = load ptr, ptr %field_gep1302, align 8
  call void @tl_tensor_free(ptr %field_load1303)
  %field_gep1304 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1285, i32 0, i32 3
  %field_load1305 = load ptr, ptr %field_gep1304, align 8
  %field_gep1306 = getelementptr inbounds %Linear, ptr %field_load1305, i32 0, i32 0
  %field_load1307 = load ptr, ptr %field_gep1306, align 8
  call void @tl_tensor_free(ptr %field_load1307)
  %field_gep1308 = getelementptr inbounds %Linear, ptr %field_load1305, i32 0, i32 1
  %field_load1309 = load ptr, ptr %field_gep1308, align 8
  call void @tl_tensor_free(ptr %field_load1309)
  %field_gep1310 = getelementptr inbounds %Block, ptr %field_load1277, i32 0, i32 2
  %field_load1311 = load ptr, ptr %field_gep1310, align 8
  %field_gep1312 = getelementptr inbounds %LayerNorm, ptr %field_load1311, i32 0, i32 0
  %field_load1313 = load ptr, ptr %field_gep1312, align 8
  call void @tl_tensor_free(ptr %field_load1313)
  %field_gep1314 = getelementptr inbounds %LayerNorm, ptr %field_load1311, i32 0, i32 1
  %field_load1315 = load ptr, ptr %field_gep1314, align 8
  call void @tl_tensor_free(ptr %field_load1315)
  %field_gep1316 = getelementptr inbounds %Block, ptr %field_load1277, i32 0, i32 3
  %field_load1317 = load ptr, ptr %field_gep1316, align 8
  %field_gep1318 = getelementptr inbounds %MLP, ptr %field_load1317, i32 0, i32 0
  %field_load1319 = load ptr, ptr %field_gep1318, align 8
  %field_gep1320 = getelementptr inbounds %Linear, ptr %field_load1319, i32 0, i32 0
  %field_load1321 = load ptr, ptr %field_gep1320, align 8
  call void @tl_tensor_free(ptr %field_load1321)
  %field_gep1322 = getelementptr inbounds %Linear, ptr %field_load1319, i32 0, i32 1
  %field_load1323 = load ptr, ptr %field_gep1322, align 8
  call void @tl_tensor_free(ptr %field_load1323)
  %field_gep1324 = getelementptr inbounds %MLP, ptr %field_load1317, i32 0, i32 1
  %field_load1325 = load ptr, ptr %field_gep1324, align 8
  %field_gep1326 = getelementptr inbounds %Linear, ptr %field_load1325, i32 0, i32 0
  %field_load1327 = load ptr, ptr %field_gep1326, align 8
  call void @tl_tensor_free(ptr %field_load1327)
  %field_gep1328 = getelementptr inbounds %Linear, ptr %field_load1325, i32 0, i32 1
  %field_load1329 = load ptr, ptr %field_gep1328, align 8
  call void @tl_tensor_free(ptr %field_load1329)
  %field_gep1330 = getelementptr inbounds %GPTHeavy, ptr %struct_to_free, i32 0, i32 5
  %field_load1331 = load ptr, ptr %field_gep1330, align 8
  %field_gep1332 = getelementptr inbounds %Block, ptr %field_load1331, i32 0, i32 0
  %field_load1333 = load ptr, ptr %field_gep1332, align 8
  %field_gep1334 = getelementptr inbounds %LayerNorm, ptr %field_load1333, i32 0, i32 0
  %field_load1335 = load ptr, ptr %field_gep1334, align 8
  call void @tl_tensor_free(ptr %field_load1335)
  %field_gep1336 = getelementptr inbounds %LayerNorm, ptr %field_load1333, i32 0, i32 1
  %field_load1337 = load ptr, ptr %field_gep1336, align 8
  call void @tl_tensor_free(ptr %field_load1337)
  %field_gep1338 = getelementptr inbounds %Block, ptr %field_load1331, i32 0, i32 1
  %field_load1339 = load ptr, ptr %field_gep1338, align 8
  %field_gep1340 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1339, i32 0, i32 0
  %field_load1341 = load ptr, ptr %field_gep1340, align 8
  %field_gep1342 = getelementptr inbounds %Linear, ptr %field_load1341, i32 0, i32 0
  %field_load1343 = load ptr, ptr %field_gep1342, align 8
  call void @tl_tensor_free(ptr %field_load1343)
  %field_gep1344 = getelementptr inbounds %Linear, ptr %field_load1341, i32 0, i32 1
  %field_load1345 = load ptr, ptr %field_gep1344, align 8
  call void @tl_tensor_free(ptr %field_load1345)
  %field_gep1346 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1339, i32 0, i32 1
  %field_load1347 = load ptr, ptr %field_gep1346, align 8
  %field_gep1348 = getelementptr inbounds %Linear, ptr %field_load1347, i32 0, i32 0
  %field_load1349 = load ptr, ptr %field_gep1348, align 8
  call void @tl_tensor_free(ptr %field_load1349)
  %field_gep1350 = getelementptr inbounds %Linear, ptr %field_load1347, i32 0, i32 1
  %field_load1351 = load ptr, ptr %field_gep1350, align 8
  call void @tl_tensor_free(ptr %field_load1351)
  %field_gep1352 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1339, i32 0, i32 2
  %field_load1353 = load ptr, ptr %field_gep1352, align 8
  %field_gep1354 = getelementptr inbounds %Linear, ptr %field_load1353, i32 0, i32 0
  %field_load1355 = load ptr, ptr %field_gep1354, align 8
  call void @tl_tensor_free(ptr %field_load1355)
  %field_gep1356 = getelementptr inbounds %Linear, ptr %field_load1353, i32 0, i32 1
  %field_load1357 = load ptr, ptr %field_gep1356, align 8
  call void @tl_tensor_free(ptr %field_load1357)
  %field_gep1358 = getelementptr inbounds %CausalSelfAttention, ptr %field_load1339, i32 0, i32 3
  %field_load1359 = load ptr, ptr %field_gep1358, align 8
  %field_gep1360 = getelementptr inbounds %Linear, ptr %field_load1359, i32 0, i32 0
  %field_load1361 = load ptr, ptr %field_gep1360, align 8
  call void @tl_tensor_free(ptr %field_load1361)
  %field_gep1362 = getelementptr inbounds %Linear, ptr %field_load1359, i32 0, i32 1
  %field_load1363 = load ptr, ptr %field_gep1362, align 8
  call void @tl_tensor_free(ptr %field_load1363)
  %field_gep1364 = getelementptr inbounds %Block, ptr %field_load1331, i32 0, i32 2
  %field_load1365 = load ptr, ptr %field_gep1364, align 8
  %field_gep1366 = getelementptr inbounds %LayerNorm, ptr %field_load1365, i32 0, i32 0
  %field_load1367 = load ptr, ptr %field_gep1366, align 8
  call void @tl_tensor_free(ptr %field_load1367)
  %field_gep1368 = getelementptr inbounds %LayerNorm, ptr %field_load1365, i32 0, i32 1
  %field_load1369 = load ptr, ptr %field_gep1368, align 8
  call void @tl_tensor_free(ptr %field_load1369)
  %field_gep1370 = getelementptr inbounds %Block, ptr %field_load1331, i32 0, i32 3
  %field_load1371 = load ptr, ptr %field_gep1370, align 8
  %field_gep1372 = getelementptr inbounds %MLP, ptr %field_load1371, i32 0, i32 0
  %field_load1373 = load ptr, ptr %field_gep1372, align 8
  %field_gep1374 = getelementptr inbounds %Linear, ptr %field_load1373, i32 0, i32 0
  %field_load1375 = load ptr, ptr %field_gep1374, align 8
  call void @tl_tensor_free(ptr %field_load1375)
  %field_gep1376 = getelementptr inbounds %Linear, ptr %field_load1373, i32 0, i32 1
  %field_load1377 = load ptr, ptr %field_gep1376, align 8
  call void @tl_tensor_free(ptr %field_load1377)
  %field_gep1378 = getelementptr inbounds %MLP, ptr %field_load1371, i32 0, i32 1
  %field_load1379 = load ptr, ptr %field_gep1378, align 8
  %field_gep1380 = getelementptr inbounds %Linear, ptr %field_load1379, i32 0, i32 0
  %field_load1381 = load ptr, ptr %field_gep1380, align 8
  call void @tl_tensor_free(ptr %field_load1381)
  %field_gep1382 = getelementptr inbounds %Linear, ptr %field_load1379, i32 0, i32 1
  %field_load1383 = load ptr, ptr %field_gep1382, align 8
  call void @tl_tensor_free(ptr %field_load1383)
  %field_gep1384 = getelementptr inbounds %GPTHeavy, ptr %struct_to_free, i32 0, i32 6
  %field_load1385 = load ptr, ptr %field_gep1384, align 8
  %field_gep1386 = getelementptr inbounds %LayerNorm, ptr %field_load1385, i32 0, i32 0
  %field_load1387 = load ptr, ptr %field_gep1386, align 8
  call void @tl_tensor_free(ptr %field_load1387)
  %field_gep1388 = getelementptr inbounds %LayerNorm, ptr %field_load1385, i32 0, i32 1
  %field_load1389 = load ptr, ptr %field_gep1388, align 8
  call void @tl_tensor_free(ptr %field_load1389)
  %field_gep1390 = getelementptr inbounds %GPTHeavy, ptr %struct_to_free, i32 0, i32 7
  %field_load1391 = load ptr, ptr %field_gep1390, align 8
  %field_gep1392 = getelementptr inbounds %Linear, ptr %field_load1391, i32 0, i32 0
  %field_load1393 = load ptr, ptr %field_gep1392, align 8
  call void @tl_tensor_free(ptr %field_load1393)
  %field_gep1394 = getelementptr inbounds %Linear, ptr %field_load1391, i32 0, i32 1
  %field_load1395 = load ptr, ptr %field_gep1394, align 8
  call void @tl_tensor_free(ptr %field_load1395)
  call void @tl_mem_unregister(ptr %struct_to_free)
  call void @tl_mem_exit_scope()
  ret void

free_struct:                                      ; preds = %for_body
  %field_gep = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 0
  %field_load = load ptr, ptr %field_gep, align 8
  %field_gep223 = getelementptr inbounds %Embedding, ptr %field_load, i32 0, i32 0
  %field_load224 = load ptr, ptr %field_gep223, align 8
  call void @tl_tensor_free(ptr %field_load224)
  %field_gep225 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 1
  %field_load226 = load ptr, ptr %field_gep225, align 8
  %field_gep227 = getelementptr inbounds %Embedding, ptr %field_load226, i32 0, i32 0
  %field_load228 = load ptr, ptr %field_gep227, align 8
  call void @tl_tensor_free(ptr %field_load228)
  %field_gep229 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 2
  %field_load230 = load ptr, ptr %field_gep229, align 8
  %field_gep231 = getelementptr inbounds %Block, ptr %field_load230, i32 0, i32 0
  %field_load232 = load ptr, ptr %field_gep231, align 8
  %field_gep233 = getelementptr inbounds %LayerNorm, ptr %field_load232, i32 0, i32 0
  %field_load234 = load ptr, ptr %field_gep233, align 8
  call void @tl_tensor_free(ptr %field_load234)
  %field_gep235 = getelementptr inbounds %LayerNorm, ptr %field_load232, i32 0, i32 1
  %field_load236 = load ptr, ptr %field_gep235, align 8
  call void @tl_tensor_free(ptr %field_load236)
  %field_gep237 = getelementptr inbounds %Block, ptr %field_load230, i32 0, i32 1
  %field_load238 = load ptr, ptr %field_gep237, align 8
  %field_gep239 = getelementptr inbounds %CausalSelfAttention, ptr %field_load238, i32 0, i32 0
  %field_load240 = load ptr, ptr %field_gep239, align 8
  %field_gep241 = getelementptr inbounds %Linear, ptr %field_load240, i32 0, i32 0
  %field_load242 = load ptr, ptr %field_gep241, align 8
  call void @tl_tensor_free(ptr %field_load242)
  %field_gep243 = getelementptr inbounds %Linear, ptr %field_load240, i32 0, i32 1
  %field_load244 = load ptr, ptr %field_gep243, align 8
  call void @tl_tensor_free(ptr %field_load244)
  %field_gep245 = getelementptr inbounds %CausalSelfAttention, ptr %field_load238, i32 0, i32 1
  %field_load246 = load ptr, ptr %field_gep245, align 8
  %field_gep247 = getelementptr inbounds %Linear, ptr %field_load246, i32 0, i32 0
  %field_load248 = load ptr, ptr %field_gep247, align 8
  call void @tl_tensor_free(ptr %field_load248)
  %field_gep249 = getelementptr inbounds %Linear, ptr %field_load246, i32 0, i32 1
  %field_load250 = load ptr, ptr %field_gep249, align 8
  call void @tl_tensor_free(ptr %field_load250)
  %field_gep251 = getelementptr inbounds %CausalSelfAttention, ptr %field_load238, i32 0, i32 2
  %field_load252 = load ptr, ptr %field_gep251, align 8
  %field_gep253 = getelementptr inbounds %Linear, ptr %field_load252, i32 0, i32 0
  %field_load254 = load ptr, ptr %field_gep253, align 8
  call void @tl_tensor_free(ptr %field_load254)
  %field_gep255 = getelementptr inbounds %Linear, ptr %field_load252, i32 0, i32 1
  %field_load256 = load ptr, ptr %field_gep255, align 8
  call void @tl_tensor_free(ptr %field_load256)
  %field_gep257 = getelementptr inbounds %CausalSelfAttention, ptr %field_load238, i32 0, i32 3
  %field_load258 = load ptr, ptr %field_gep257, align 8
  %field_gep259 = getelementptr inbounds %Linear, ptr %field_load258, i32 0, i32 0
  %field_load260 = load ptr, ptr %field_gep259, align 8
  call void @tl_tensor_free(ptr %field_load260)
  %field_gep261 = getelementptr inbounds %Linear, ptr %field_load258, i32 0, i32 1
  %field_load262 = load ptr, ptr %field_gep261, align 8
  call void @tl_tensor_free(ptr %field_load262)
  %field_gep263 = getelementptr inbounds %Block, ptr %field_load230, i32 0, i32 2
  %field_load264 = load ptr, ptr %field_gep263, align 8
  %field_gep265 = getelementptr inbounds %LayerNorm, ptr %field_load264, i32 0, i32 0
  %field_load266 = load ptr, ptr %field_gep265, align 8
  call void @tl_tensor_free(ptr %field_load266)
  %field_gep267 = getelementptr inbounds %LayerNorm, ptr %field_load264, i32 0, i32 1
  %field_load268 = load ptr, ptr %field_gep267, align 8
  call void @tl_tensor_free(ptr %field_load268)
  %field_gep269 = getelementptr inbounds %Block, ptr %field_load230, i32 0, i32 3
  %field_load270 = load ptr, ptr %field_gep269, align 8
  %field_gep271 = getelementptr inbounds %MLP, ptr %field_load270, i32 0, i32 0
  %field_load272 = load ptr, ptr %field_gep271, align 8
  %field_gep273 = getelementptr inbounds %Linear, ptr %field_load272, i32 0, i32 0
  %field_load274 = load ptr, ptr %field_gep273, align 8
  call void @tl_tensor_free(ptr %field_load274)
  %field_gep275 = getelementptr inbounds %Linear, ptr %field_load272, i32 0, i32 1
  %field_load276 = load ptr, ptr %field_gep275, align 8
  call void @tl_tensor_free(ptr %field_load276)
  %field_gep277 = getelementptr inbounds %MLP, ptr %field_load270, i32 0, i32 1
  %field_load278 = load ptr, ptr %field_gep277, align 8
  %field_gep279 = getelementptr inbounds %Linear, ptr %field_load278, i32 0, i32 0
  %field_load280 = load ptr, ptr %field_gep279, align 8
  call void @tl_tensor_free(ptr %field_load280)
  %field_gep281 = getelementptr inbounds %Linear, ptr %field_load278, i32 0, i32 1
  %field_load282 = load ptr, ptr %field_gep281, align 8
  call void @tl_tensor_free(ptr %field_load282)
  %field_gep283 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 3
  %field_load284 = load ptr, ptr %field_gep283, align 8
  %field_gep285 = getelementptr inbounds %Block, ptr %field_load284, i32 0, i32 0
  %field_load286 = load ptr, ptr %field_gep285, align 8
  %field_gep287 = getelementptr inbounds %LayerNorm, ptr %field_load286, i32 0, i32 0
  %field_load288 = load ptr, ptr %field_gep287, align 8
  call void @tl_tensor_free(ptr %field_load288)
  %field_gep289 = getelementptr inbounds %LayerNorm, ptr %field_load286, i32 0, i32 1
  %field_load290 = load ptr, ptr %field_gep289, align 8
  call void @tl_tensor_free(ptr %field_load290)
  %field_gep291 = getelementptr inbounds %Block, ptr %field_load284, i32 0, i32 1
  %field_load292 = load ptr, ptr %field_gep291, align 8
  %field_gep293 = getelementptr inbounds %CausalSelfAttention, ptr %field_load292, i32 0, i32 0
  %field_load294 = load ptr, ptr %field_gep293, align 8
  %field_gep295 = getelementptr inbounds %Linear, ptr %field_load294, i32 0, i32 0
  %field_load296 = load ptr, ptr %field_gep295, align 8
  call void @tl_tensor_free(ptr %field_load296)
  %field_gep297 = getelementptr inbounds %Linear, ptr %field_load294, i32 0, i32 1
  %field_load298 = load ptr, ptr %field_gep297, align 8
  call void @tl_tensor_free(ptr %field_load298)
  %field_gep299 = getelementptr inbounds %CausalSelfAttention, ptr %field_load292, i32 0, i32 1
  %field_load300 = load ptr, ptr %field_gep299, align 8
  %field_gep301 = getelementptr inbounds %Linear, ptr %field_load300, i32 0, i32 0
  %field_load302 = load ptr, ptr %field_gep301, align 8
  call void @tl_tensor_free(ptr %field_load302)
  %field_gep303 = getelementptr inbounds %Linear, ptr %field_load300, i32 0, i32 1
  %field_load304 = load ptr, ptr %field_gep303, align 8
  call void @tl_tensor_free(ptr %field_load304)
  %field_gep305 = getelementptr inbounds %CausalSelfAttention, ptr %field_load292, i32 0, i32 2
  %field_load306 = load ptr, ptr %field_gep305, align 8
  %field_gep307 = getelementptr inbounds %Linear, ptr %field_load306, i32 0, i32 0
  %field_load308 = load ptr, ptr %field_gep307, align 8
  call void @tl_tensor_free(ptr %field_load308)
  %field_gep309 = getelementptr inbounds %Linear, ptr %field_load306, i32 0, i32 1
  %field_load310 = load ptr, ptr %field_gep309, align 8
  call void @tl_tensor_free(ptr %field_load310)
  %field_gep311 = getelementptr inbounds %CausalSelfAttention, ptr %field_load292, i32 0, i32 3
  %field_load312 = load ptr, ptr %field_gep311, align 8
  %field_gep313 = getelementptr inbounds %Linear, ptr %field_load312, i32 0, i32 0
  %field_load314 = load ptr, ptr %field_gep313, align 8
  call void @tl_tensor_free(ptr %field_load314)
  %field_gep315 = getelementptr inbounds %Linear, ptr %field_load312, i32 0, i32 1
  %field_load316 = load ptr, ptr %field_gep315, align 8
  call void @tl_tensor_free(ptr %field_load316)
  %field_gep317 = getelementptr inbounds %Block, ptr %field_load284, i32 0, i32 2
  %field_load318 = load ptr, ptr %field_gep317, align 8
  %field_gep319 = getelementptr inbounds %LayerNorm, ptr %field_load318, i32 0, i32 0
  %field_load320 = load ptr, ptr %field_gep319, align 8
  call void @tl_tensor_free(ptr %field_load320)
  %field_gep321 = getelementptr inbounds %LayerNorm, ptr %field_load318, i32 0, i32 1
  %field_load322 = load ptr, ptr %field_gep321, align 8
  call void @tl_tensor_free(ptr %field_load322)
  %field_gep323 = getelementptr inbounds %Block, ptr %field_load284, i32 0, i32 3
  %field_load324 = load ptr, ptr %field_gep323, align 8
  %field_gep325 = getelementptr inbounds %MLP, ptr %field_load324, i32 0, i32 0
  %field_load326 = load ptr, ptr %field_gep325, align 8
  %field_gep327 = getelementptr inbounds %Linear, ptr %field_load326, i32 0, i32 0
  %field_load328 = load ptr, ptr %field_gep327, align 8
  call void @tl_tensor_free(ptr %field_load328)
  %field_gep329 = getelementptr inbounds %Linear, ptr %field_load326, i32 0, i32 1
  %field_load330 = load ptr, ptr %field_gep329, align 8
  call void @tl_tensor_free(ptr %field_load330)
  %field_gep331 = getelementptr inbounds %MLP, ptr %field_load324, i32 0, i32 1
  %field_load332 = load ptr, ptr %field_gep331, align 8
  %field_gep333 = getelementptr inbounds %Linear, ptr %field_load332, i32 0, i32 0
  %field_load334 = load ptr, ptr %field_gep333, align 8
  call void @tl_tensor_free(ptr %field_load334)
  %field_gep335 = getelementptr inbounds %Linear, ptr %field_load332, i32 0, i32 1
  %field_load336 = load ptr, ptr %field_gep335, align 8
  call void @tl_tensor_free(ptr %field_load336)
  %field_gep337 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 4
  %field_load338 = load ptr, ptr %field_gep337, align 8
  %field_gep339 = getelementptr inbounds %Block, ptr %field_load338, i32 0, i32 0
  %field_load340 = load ptr, ptr %field_gep339, align 8
  %field_gep341 = getelementptr inbounds %LayerNorm, ptr %field_load340, i32 0, i32 0
  %field_load342 = load ptr, ptr %field_gep341, align 8
  call void @tl_tensor_free(ptr %field_load342)
  %field_gep343 = getelementptr inbounds %LayerNorm, ptr %field_load340, i32 0, i32 1
  %field_load344 = load ptr, ptr %field_gep343, align 8
  call void @tl_tensor_free(ptr %field_load344)
  %field_gep345 = getelementptr inbounds %Block, ptr %field_load338, i32 0, i32 1
  %field_load346 = load ptr, ptr %field_gep345, align 8
  %field_gep347 = getelementptr inbounds %CausalSelfAttention, ptr %field_load346, i32 0, i32 0
  %field_load348 = load ptr, ptr %field_gep347, align 8
  %field_gep349 = getelementptr inbounds %Linear, ptr %field_load348, i32 0, i32 0
  %field_load350 = load ptr, ptr %field_gep349, align 8
  call void @tl_tensor_free(ptr %field_load350)
  %field_gep351 = getelementptr inbounds %Linear, ptr %field_load348, i32 0, i32 1
  %field_load352 = load ptr, ptr %field_gep351, align 8
  call void @tl_tensor_free(ptr %field_load352)
  %field_gep353 = getelementptr inbounds %CausalSelfAttention, ptr %field_load346, i32 0, i32 1
  %field_load354 = load ptr, ptr %field_gep353, align 8
  %field_gep355 = getelementptr inbounds %Linear, ptr %field_load354, i32 0, i32 0
  %field_load356 = load ptr, ptr %field_gep355, align 8
  call void @tl_tensor_free(ptr %field_load356)
  %field_gep357 = getelementptr inbounds %Linear, ptr %field_load354, i32 0, i32 1
  %field_load358 = load ptr, ptr %field_gep357, align 8
  call void @tl_tensor_free(ptr %field_load358)
  %field_gep359 = getelementptr inbounds %CausalSelfAttention, ptr %field_load346, i32 0, i32 2
  %field_load360 = load ptr, ptr %field_gep359, align 8
  %field_gep361 = getelementptr inbounds %Linear, ptr %field_load360, i32 0, i32 0
  %field_load362 = load ptr, ptr %field_gep361, align 8
  call void @tl_tensor_free(ptr %field_load362)
  %field_gep363 = getelementptr inbounds %Linear, ptr %field_load360, i32 0, i32 1
  %field_load364 = load ptr, ptr %field_gep363, align 8
  call void @tl_tensor_free(ptr %field_load364)
  %field_gep365 = getelementptr inbounds %CausalSelfAttention, ptr %field_load346, i32 0, i32 3
  %field_load366 = load ptr, ptr %field_gep365, align 8
  %field_gep367 = getelementptr inbounds %Linear, ptr %field_load366, i32 0, i32 0
  %field_load368 = load ptr, ptr %field_gep367, align 8
  call void @tl_tensor_free(ptr %field_load368)
  %field_gep369 = getelementptr inbounds %Linear, ptr %field_load366, i32 0, i32 1
  %field_load370 = load ptr, ptr %field_gep369, align 8
  call void @tl_tensor_free(ptr %field_load370)
  %field_gep371 = getelementptr inbounds %Block, ptr %field_load338, i32 0, i32 2
  %field_load372 = load ptr, ptr %field_gep371, align 8
  %field_gep373 = getelementptr inbounds %LayerNorm, ptr %field_load372, i32 0, i32 0
  %field_load374 = load ptr, ptr %field_gep373, align 8
  call void @tl_tensor_free(ptr %field_load374)
  %field_gep375 = getelementptr inbounds %LayerNorm, ptr %field_load372, i32 0, i32 1
  %field_load376 = load ptr, ptr %field_gep375, align 8
  call void @tl_tensor_free(ptr %field_load376)
  %field_gep377 = getelementptr inbounds %Block, ptr %field_load338, i32 0, i32 3
  %field_load378 = load ptr, ptr %field_gep377, align 8
  %field_gep379 = getelementptr inbounds %MLP, ptr %field_load378, i32 0, i32 0
  %field_load380 = load ptr, ptr %field_gep379, align 8
  %field_gep381 = getelementptr inbounds %Linear, ptr %field_load380, i32 0, i32 0
  %field_load382 = load ptr, ptr %field_gep381, align 8
  call void @tl_tensor_free(ptr %field_load382)
  %field_gep383 = getelementptr inbounds %Linear, ptr %field_load380, i32 0, i32 1
  %field_load384 = load ptr, ptr %field_gep383, align 8
  call void @tl_tensor_free(ptr %field_load384)
  %field_gep385 = getelementptr inbounds %MLP, ptr %field_load378, i32 0, i32 1
  %field_load386 = load ptr, ptr %field_gep385, align 8
  %field_gep387 = getelementptr inbounds %Linear, ptr %field_load386, i32 0, i32 0
  %field_load388 = load ptr, ptr %field_gep387, align 8
  call void @tl_tensor_free(ptr %field_load388)
  %field_gep389 = getelementptr inbounds %Linear, ptr %field_load386, i32 0, i32 1
  %field_load390 = load ptr, ptr %field_gep389, align 8
  call void @tl_tensor_free(ptr %field_load390)
  %field_gep391 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 5
  %field_load392 = load ptr, ptr %field_gep391, align 8
  %field_gep393 = getelementptr inbounds %Block, ptr %field_load392, i32 0, i32 0
  %field_load394 = load ptr, ptr %field_gep393, align 8
  %field_gep395 = getelementptr inbounds %LayerNorm, ptr %field_load394, i32 0, i32 0
  %field_load396 = load ptr, ptr %field_gep395, align 8
  call void @tl_tensor_free(ptr %field_load396)
  %field_gep397 = getelementptr inbounds %LayerNorm, ptr %field_load394, i32 0, i32 1
  %field_load398 = load ptr, ptr %field_gep397, align 8
  call void @tl_tensor_free(ptr %field_load398)
  %field_gep399 = getelementptr inbounds %Block, ptr %field_load392, i32 0, i32 1
  %field_load400 = load ptr, ptr %field_gep399, align 8
  %field_gep401 = getelementptr inbounds %CausalSelfAttention, ptr %field_load400, i32 0, i32 0
  %field_load402 = load ptr, ptr %field_gep401, align 8
  %field_gep403 = getelementptr inbounds %Linear, ptr %field_load402, i32 0, i32 0
  %field_load404 = load ptr, ptr %field_gep403, align 8
  call void @tl_tensor_free(ptr %field_load404)
  %field_gep405 = getelementptr inbounds %Linear, ptr %field_load402, i32 0, i32 1
  %field_load406 = load ptr, ptr %field_gep405, align 8
  call void @tl_tensor_free(ptr %field_load406)
  %field_gep407 = getelementptr inbounds %CausalSelfAttention, ptr %field_load400, i32 0, i32 1
  %field_load408 = load ptr, ptr %field_gep407, align 8
  %field_gep409 = getelementptr inbounds %Linear, ptr %field_load408, i32 0, i32 0
  %field_load410 = load ptr, ptr %field_gep409, align 8
  call void @tl_tensor_free(ptr %field_load410)
  %field_gep411 = getelementptr inbounds %Linear, ptr %field_load408, i32 0, i32 1
  %field_load412 = load ptr, ptr %field_gep411, align 8
  call void @tl_tensor_free(ptr %field_load412)
  %field_gep413 = getelementptr inbounds %CausalSelfAttention, ptr %field_load400, i32 0, i32 2
  %field_load414 = load ptr, ptr %field_gep413, align 8
  %field_gep415 = getelementptr inbounds %Linear, ptr %field_load414, i32 0, i32 0
  %field_load416 = load ptr, ptr %field_gep415, align 8
  call void @tl_tensor_free(ptr %field_load416)
  %field_gep417 = getelementptr inbounds %Linear, ptr %field_load414, i32 0, i32 1
  %field_load418 = load ptr, ptr %field_gep417, align 8
  call void @tl_tensor_free(ptr %field_load418)
  %field_gep419 = getelementptr inbounds %CausalSelfAttention, ptr %field_load400, i32 0, i32 3
  %field_load420 = load ptr, ptr %field_gep419, align 8
  %field_gep421 = getelementptr inbounds %Linear, ptr %field_load420, i32 0, i32 0
  %field_load422 = load ptr, ptr %field_gep421, align 8
  call void @tl_tensor_free(ptr %field_load422)
  %field_gep423 = getelementptr inbounds %Linear, ptr %field_load420, i32 0, i32 1
  %field_load424 = load ptr, ptr %field_gep423, align 8
  call void @tl_tensor_free(ptr %field_load424)
  %field_gep425 = getelementptr inbounds %Block, ptr %field_load392, i32 0, i32 2
  %field_load426 = load ptr, ptr %field_gep425, align 8
  %field_gep427 = getelementptr inbounds %LayerNorm, ptr %field_load426, i32 0, i32 0
  %field_load428 = load ptr, ptr %field_gep427, align 8
  call void @tl_tensor_free(ptr %field_load428)
  %field_gep429 = getelementptr inbounds %LayerNorm, ptr %field_load426, i32 0, i32 1
  %field_load430 = load ptr, ptr %field_gep429, align 8
  call void @tl_tensor_free(ptr %field_load430)
  %field_gep431 = getelementptr inbounds %Block, ptr %field_load392, i32 0, i32 3
  %field_load432 = load ptr, ptr %field_gep431, align 8
  %field_gep433 = getelementptr inbounds %MLP, ptr %field_load432, i32 0, i32 0
  %field_load434 = load ptr, ptr %field_gep433, align 8
  %field_gep435 = getelementptr inbounds %Linear, ptr %field_load434, i32 0, i32 0
  %field_load436 = load ptr, ptr %field_gep435, align 8
  call void @tl_tensor_free(ptr %field_load436)
  %field_gep437 = getelementptr inbounds %Linear, ptr %field_load434, i32 0, i32 1
  %field_load438 = load ptr, ptr %field_gep437, align 8
  call void @tl_tensor_free(ptr %field_load438)
  %field_gep439 = getelementptr inbounds %MLP, ptr %field_load432, i32 0, i32 1
  %field_load440 = load ptr, ptr %field_gep439, align 8
  %field_gep441 = getelementptr inbounds %Linear, ptr %field_load440, i32 0, i32 0
  %field_load442 = load ptr, ptr %field_gep441, align 8
  call void @tl_tensor_free(ptr %field_load442)
  %field_gep443 = getelementptr inbounds %Linear, ptr %field_load440, i32 0, i32 1
  %field_load444 = load ptr, ptr %field_gep443, align 8
  call void @tl_tensor_free(ptr %field_load444)
  %field_gep445 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 6
  %field_load446 = load ptr, ptr %field_gep445, align 8
  %field_gep447 = getelementptr inbounds %LayerNorm, ptr %field_load446, i32 0, i32 0
  %field_load448 = load ptr, ptr %field_gep447, align 8
  call void @tl_tensor_free(ptr %field_load448)
  %field_gep449 = getelementptr inbounds %LayerNorm, ptr %field_load446, i32 0, i32 1
  %field_load450 = load ptr, ptr %field_gep449, align 8
  call void @tl_tensor_free(ptr %field_load450)
  %field_gep451 = getelementptr inbounds %GPTHeavy, ptr %old_struct_to_free, i32 0, i32 7
  %field_load452 = load ptr, ptr %field_gep451, align 8
  %field_gep453 = getelementptr inbounds %Linear, ptr %field_load452, i32 0, i32 0
  %field_load454 = load ptr, ptr %field_gep453, align 8
  call void @tl_tensor_free(ptr %field_load454)
  %field_gep455 = getelementptr inbounds %Linear, ptr %field_load452, i32 0, i32 1
  %field_load456 = load ptr, ptr %field_gep455, align 8
  call void @tl_tensor_free(ptr %field_load456)
  call void @tl_mem_unregister(ptr %old_struct_to_free)
  br label %continue_after_free

continue_after_free:                              ; preds = %free_struct, %for_body
  call void @tl_mem_unregister(ptr %call_tmp)
  %unreg_field_0 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_0457 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val458 = load ptr, ptr %unreg_field_0457, align 8
  call void @tl_mem_unregister(ptr %field_val458)
  %unreg_field_1 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 1
  %field_val459 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val459)
  %unreg_field_0460 = getelementptr inbounds %Embedding, ptr %field_val459, i32 0, i32 0
  %field_val461 = load ptr, ptr %unreg_field_0460, align 8
  call void @tl_mem_unregister(ptr %field_val461)
  %unreg_field_2 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 2
  %field_val462 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val462)
  %unreg_field_0463 = getelementptr inbounds %Block, ptr %field_val462, i32 0, i32 0
  %field_val464 = load ptr, ptr %unreg_field_0463, align 8
  call void @tl_mem_unregister(ptr %field_val464)
  %unreg_field_0465 = getelementptr inbounds %LayerNorm, ptr %field_val464, i32 0, i32 0
  %field_val466 = load ptr, ptr %unreg_field_0465, align 8
  call void @tl_mem_unregister(ptr %field_val466)
  %unreg_field_1467 = getelementptr inbounds %LayerNorm, ptr %field_val464, i32 0, i32 1
  %field_val468 = load ptr, ptr %unreg_field_1467, align 8
  call void @tl_mem_unregister(ptr %field_val468)
  %unreg_field_1469 = getelementptr inbounds %Block, ptr %field_val462, i32 0, i32 1
  %field_val470 = load ptr, ptr %unreg_field_1469, align 8
  call void @tl_mem_unregister(ptr %field_val470)
  %unreg_field_0471 = getelementptr inbounds %CausalSelfAttention, ptr %field_val470, i32 0, i32 0
  %field_val472 = load ptr, ptr %unreg_field_0471, align 8
  call void @tl_mem_unregister(ptr %field_val472)
  %unreg_field_0473 = getelementptr inbounds %Linear, ptr %field_val472, i32 0, i32 0
  %field_val474 = load ptr, ptr %unreg_field_0473, align 8
  call void @tl_mem_unregister(ptr %field_val474)
  %unreg_field_1475 = getelementptr inbounds %Linear, ptr %field_val472, i32 0, i32 1
  %field_val476 = load ptr, ptr %unreg_field_1475, align 8
  call void @tl_mem_unregister(ptr %field_val476)
  %unreg_field_1477 = getelementptr inbounds %CausalSelfAttention, ptr %field_val470, i32 0, i32 1
  %field_val478 = load ptr, ptr %unreg_field_1477, align 8
  call void @tl_mem_unregister(ptr %field_val478)
  %unreg_field_0479 = getelementptr inbounds %Linear, ptr %field_val478, i32 0, i32 0
  %field_val480 = load ptr, ptr %unreg_field_0479, align 8
  call void @tl_mem_unregister(ptr %field_val480)
  %unreg_field_1481 = getelementptr inbounds %Linear, ptr %field_val478, i32 0, i32 1
  %field_val482 = load ptr, ptr %unreg_field_1481, align 8
  call void @tl_mem_unregister(ptr %field_val482)
  %unreg_field_2483 = getelementptr inbounds %CausalSelfAttention, ptr %field_val470, i32 0, i32 2
  %field_val484 = load ptr, ptr %unreg_field_2483, align 8
  call void @tl_mem_unregister(ptr %field_val484)
  %unreg_field_0485 = getelementptr inbounds %Linear, ptr %field_val484, i32 0, i32 0
  %field_val486 = load ptr, ptr %unreg_field_0485, align 8
  call void @tl_mem_unregister(ptr %field_val486)
  %unreg_field_1487 = getelementptr inbounds %Linear, ptr %field_val484, i32 0, i32 1
  %field_val488 = load ptr, ptr %unreg_field_1487, align 8
  call void @tl_mem_unregister(ptr %field_val488)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val470, i32 0, i32 3
  %field_val489 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val489)
  %unreg_field_0490 = getelementptr inbounds %Linear, ptr %field_val489, i32 0, i32 0
  %field_val491 = load ptr, ptr %unreg_field_0490, align 8
  call void @tl_mem_unregister(ptr %field_val491)
  %unreg_field_1492 = getelementptr inbounds %Linear, ptr %field_val489, i32 0, i32 1
  %field_val493 = load ptr, ptr %unreg_field_1492, align 8
  call void @tl_mem_unregister(ptr %field_val493)
  %unreg_field_2494 = getelementptr inbounds %Block, ptr %field_val462, i32 0, i32 2
  %field_val495 = load ptr, ptr %unreg_field_2494, align 8
  call void @tl_mem_unregister(ptr %field_val495)
  %unreg_field_0496 = getelementptr inbounds %LayerNorm, ptr %field_val495, i32 0, i32 0
  %field_val497 = load ptr, ptr %unreg_field_0496, align 8
  call void @tl_mem_unregister(ptr %field_val497)
  %unreg_field_1498 = getelementptr inbounds %LayerNorm, ptr %field_val495, i32 0, i32 1
  %field_val499 = load ptr, ptr %unreg_field_1498, align 8
  call void @tl_mem_unregister(ptr %field_val499)
  %unreg_field_3500 = getelementptr inbounds %Block, ptr %field_val462, i32 0, i32 3
  %field_val501 = load ptr, ptr %unreg_field_3500, align 8
  call void @tl_mem_unregister(ptr %field_val501)
  %unreg_field_0502 = getelementptr inbounds %MLP, ptr %field_val501, i32 0, i32 0
  %field_val503 = load ptr, ptr %unreg_field_0502, align 8
  call void @tl_mem_unregister(ptr %field_val503)
  %unreg_field_0504 = getelementptr inbounds %Linear, ptr %field_val503, i32 0, i32 0
  %field_val505 = load ptr, ptr %unreg_field_0504, align 8
  call void @tl_mem_unregister(ptr %field_val505)
  %unreg_field_1506 = getelementptr inbounds %Linear, ptr %field_val503, i32 0, i32 1
  %field_val507 = load ptr, ptr %unreg_field_1506, align 8
  call void @tl_mem_unregister(ptr %field_val507)
  %unreg_field_1508 = getelementptr inbounds %MLP, ptr %field_val501, i32 0, i32 1
  %field_val509 = load ptr, ptr %unreg_field_1508, align 8
  call void @tl_mem_unregister(ptr %field_val509)
  %unreg_field_0510 = getelementptr inbounds %Linear, ptr %field_val509, i32 0, i32 0
  %field_val511 = load ptr, ptr %unreg_field_0510, align 8
  call void @tl_mem_unregister(ptr %field_val511)
  %unreg_field_1512 = getelementptr inbounds %Linear, ptr %field_val509, i32 0, i32 1
  %field_val513 = load ptr, ptr %unreg_field_1512, align 8
  call void @tl_mem_unregister(ptr %field_val513)
  %unreg_field_3514 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 3
  %field_val515 = load ptr, ptr %unreg_field_3514, align 8
  call void @tl_mem_unregister(ptr %field_val515)
  %unreg_field_0516 = getelementptr inbounds %Block, ptr %field_val515, i32 0, i32 0
  %field_val517 = load ptr, ptr %unreg_field_0516, align 8
  call void @tl_mem_unregister(ptr %field_val517)
  %unreg_field_0518 = getelementptr inbounds %LayerNorm, ptr %field_val517, i32 0, i32 0
  %field_val519 = load ptr, ptr %unreg_field_0518, align 8
  call void @tl_mem_unregister(ptr %field_val519)
  %unreg_field_1520 = getelementptr inbounds %LayerNorm, ptr %field_val517, i32 0, i32 1
  %field_val521 = load ptr, ptr %unreg_field_1520, align 8
  call void @tl_mem_unregister(ptr %field_val521)
  %unreg_field_1522 = getelementptr inbounds %Block, ptr %field_val515, i32 0, i32 1
  %field_val523 = load ptr, ptr %unreg_field_1522, align 8
  call void @tl_mem_unregister(ptr %field_val523)
  %unreg_field_0524 = getelementptr inbounds %CausalSelfAttention, ptr %field_val523, i32 0, i32 0
  %field_val525 = load ptr, ptr %unreg_field_0524, align 8
  call void @tl_mem_unregister(ptr %field_val525)
  %unreg_field_0526 = getelementptr inbounds %Linear, ptr %field_val525, i32 0, i32 0
  %field_val527 = load ptr, ptr %unreg_field_0526, align 8
  call void @tl_mem_unregister(ptr %field_val527)
  %unreg_field_1528 = getelementptr inbounds %Linear, ptr %field_val525, i32 0, i32 1
  %field_val529 = load ptr, ptr %unreg_field_1528, align 8
  call void @tl_mem_unregister(ptr %field_val529)
  %unreg_field_1530 = getelementptr inbounds %CausalSelfAttention, ptr %field_val523, i32 0, i32 1
  %field_val531 = load ptr, ptr %unreg_field_1530, align 8
  call void @tl_mem_unregister(ptr %field_val531)
  %unreg_field_0532 = getelementptr inbounds %Linear, ptr %field_val531, i32 0, i32 0
  %field_val533 = load ptr, ptr %unreg_field_0532, align 8
  call void @tl_mem_unregister(ptr %field_val533)
  %unreg_field_1534 = getelementptr inbounds %Linear, ptr %field_val531, i32 0, i32 1
  %field_val535 = load ptr, ptr %unreg_field_1534, align 8
  call void @tl_mem_unregister(ptr %field_val535)
  %unreg_field_2536 = getelementptr inbounds %CausalSelfAttention, ptr %field_val523, i32 0, i32 2
  %field_val537 = load ptr, ptr %unreg_field_2536, align 8
  call void @tl_mem_unregister(ptr %field_val537)
  %unreg_field_0538 = getelementptr inbounds %Linear, ptr %field_val537, i32 0, i32 0
  %field_val539 = load ptr, ptr %unreg_field_0538, align 8
  call void @tl_mem_unregister(ptr %field_val539)
  %unreg_field_1540 = getelementptr inbounds %Linear, ptr %field_val537, i32 0, i32 1
  %field_val541 = load ptr, ptr %unreg_field_1540, align 8
  call void @tl_mem_unregister(ptr %field_val541)
  %unreg_field_3542 = getelementptr inbounds %CausalSelfAttention, ptr %field_val523, i32 0, i32 3
  %field_val543 = load ptr, ptr %unreg_field_3542, align 8
  call void @tl_mem_unregister(ptr %field_val543)
  %unreg_field_0544 = getelementptr inbounds %Linear, ptr %field_val543, i32 0, i32 0
  %field_val545 = load ptr, ptr %unreg_field_0544, align 8
  call void @tl_mem_unregister(ptr %field_val545)
  %unreg_field_1546 = getelementptr inbounds %Linear, ptr %field_val543, i32 0, i32 1
  %field_val547 = load ptr, ptr %unreg_field_1546, align 8
  call void @tl_mem_unregister(ptr %field_val547)
  %unreg_field_2548 = getelementptr inbounds %Block, ptr %field_val515, i32 0, i32 2
  %field_val549 = load ptr, ptr %unreg_field_2548, align 8
  call void @tl_mem_unregister(ptr %field_val549)
  %unreg_field_0550 = getelementptr inbounds %LayerNorm, ptr %field_val549, i32 0, i32 0
  %field_val551 = load ptr, ptr %unreg_field_0550, align 8
  call void @tl_mem_unregister(ptr %field_val551)
  %unreg_field_1552 = getelementptr inbounds %LayerNorm, ptr %field_val549, i32 0, i32 1
  %field_val553 = load ptr, ptr %unreg_field_1552, align 8
  call void @tl_mem_unregister(ptr %field_val553)
  %unreg_field_3554 = getelementptr inbounds %Block, ptr %field_val515, i32 0, i32 3
  %field_val555 = load ptr, ptr %unreg_field_3554, align 8
  call void @tl_mem_unregister(ptr %field_val555)
  %unreg_field_0556 = getelementptr inbounds %MLP, ptr %field_val555, i32 0, i32 0
  %field_val557 = load ptr, ptr %unreg_field_0556, align 8
  call void @tl_mem_unregister(ptr %field_val557)
  %unreg_field_0558 = getelementptr inbounds %Linear, ptr %field_val557, i32 0, i32 0
  %field_val559 = load ptr, ptr %unreg_field_0558, align 8
  call void @tl_mem_unregister(ptr %field_val559)
  %unreg_field_1560 = getelementptr inbounds %Linear, ptr %field_val557, i32 0, i32 1
  %field_val561 = load ptr, ptr %unreg_field_1560, align 8
  call void @tl_mem_unregister(ptr %field_val561)
  %unreg_field_1562 = getelementptr inbounds %MLP, ptr %field_val555, i32 0, i32 1
  %field_val563 = load ptr, ptr %unreg_field_1562, align 8
  call void @tl_mem_unregister(ptr %field_val563)
  %unreg_field_0564 = getelementptr inbounds %Linear, ptr %field_val563, i32 0, i32 0
  %field_val565 = load ptr, ptr %unreg_field_0564, align 8
  call void @tl_mem_unregister(ptr %field_val565)
  %unreg_field_1566 = getelementptr inbounds %Linear, ptr %field_val563, i32 0, i32 1
  %field_val567 = load ptr, ptr %unreg_field_1566, align 8
  call void @tl_mem_unregister(ptr %field_val567)
  %unreg_field_4 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 4
  %field_val568 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val568)
  %unreg_field_0569 = getelementptr inbounds %Block, ptr %field_val568, i32 0, i32 0
  %field_val570 = load ptr, ptr %unreg_field_0569, align 8
  call void @tl_mem_unregister(ptr %field_val570)
  %unreg_field_0571 = getelementptr inbounds %LayerNorm, ptr %field_val570, i32 0, i32 0
  %field_val572 = load ptr, ptr %unreg_field_0571, align 8
  call void @tl_mem_unregister(ptr %field_val572)
  %unreg_field_1573 = getelementptr inbounds %LayerNorm, ptr %field_val570, i32 0, i32 1
  %field_val574 = load ptr, ptr %unreg_field_1573, align 8
  call void @tl_mem_unregister(ptr %field_val574)
  %unreg_field_1575 = getelementptr inbounds %Block, ptr %field_val568, i32 0, i32 1
  %field_val576 = load ptr, ptr %unreg_field_1575, align 8
  call void @tl_mem_unregister(ptr %field_val576)
  %unreg_field_0577 = getelementptr inbounds %CausalSelfAttention, ptr %field_val576, i32 0, i32 0
  %field_val578 = load ptr, ptr %unreg_field_0577, align 8
  call void @tl_mem_unregister(ptr %field_val578)
  %unreg_field_0579 = getelementptr inbounds %Linear, ptr %field_val578, i32 0, i32 0
  %field_val580 = load ptr, ptr %unreg_field_0579, align 8
  call void @tl_mem_unregister(ptr %field_val580)
  %unreg_field_1581 = getelementptr inbounds %Linear, ptr %field_val578, i32 0, i32 1
  %field_val582 = load ptr, ptr %unreg_field_1581, align 8
  call void @tl_mem_unregister(ptr %field_val582)
  %unreg_field_1583 = getelementptr inbounds %CausalSelfAttention, ptr %field_val576, i32 0, i32 1
  %field_val584 = load ptr, ptr %unreg_field_1583, align 8
  call void @tl_mem_unregister(ptr %field_val584)
  %unreg_field_0585 = getelementptr inbounds %Linear, ptr %field_val584, i32 0, i32 0
  %field_val586 = load ptr, ptr %unreg_field_0585, align 8
  call void @tl_mem_unregister(ptr %field_val586)
  %unreg_field_1587 = getelementptr inbounds %Linear, ptr %field_val584, i32 0, i32 1
  %field_val588 = load ptr, ptr %unreg_field_1587, align 8
  call void @tl_mem_unregister(ptr %field_val588)
  %unreg_field_2589 = getelementptr inbounds %CausalSelfAttention, ptr %field_val576, i32 0, i32 2
  %field_val590 = load ptr, ptr %unreg_field_2589, align 8
  call void @tl_mem_unregister(ptr %field_val590)
  %unreg_field_0591 = getelementptr inbounds %Linear, ptr %field_val590, i32 0, i32 0
  %field_val592 = load ptr, ptr %unreg_field_0591, align 8
  call void @tl_mem_unregister(ptr %field_val592)
  %unreg_field_1593 = getelementptr inbounds %Linear, ptr %field_val590, i32 0, i32 1
  %field_val594 = load ptr, ptr %unreg_field_1593, align 8
  call void @tl_mem_unregister(ptr %field_val594)
  %unreg_field_3595 = getelementptr inbounds %CausalSelfAttention, ptr %field_val576, i32 0, i32 3
  %field_val596 = load ptr, ptr %unreg_field_3595, align 8
  call void @tl_mem_unregister(ptr %field_val596)
  %unreg_field_0597 = getelementptr inbounds %Linear, ptr %field_val596, i32 0, i32 0
  %field_val598 = load ptr, ptr %unreg_field_0597, align 8
  call void @tl_mem_unregister(ptr %field_val598)
  %unreg_field_1599 = getelementptr inbounds %Linear, ptr %field_val596, i32 0, i32 1
  %field_val600 = load ptr, ptr %unreg_field_1599, align 8
  call void @tl_mem_unregister(ptr %field_val600)
  %unreg_field_2601 = getelementptr inbounds %Block, ptr %field_val568, i32 0, i32 2
  %field_val602 = load ptr, ptr %unreg_field_2601, align 8
  call void @tl_mem_unregister(ptr %field_val602)
  %unreg_field_0603 = getelementptr inbounds %LayerNorm, ptr %field_val602, i32 0, i32 0
  %field_val604 = load ptr, ptr %unreg_field_0603, align 8
  call void @tl_mem_unregister(ptr %field_val604)
  %unreg_field_1605 = getelementptr inbounds %LayerNorm, ptr %field_val602, i32 0, i32 1
  %field_val606 = load ptr, ptr %unreg_field_1605, align 8
  call void @tl_mem_unregister(ptr %field_val606)
  %unreg_field_3607 = getelementptr inbounds %Block, ptr %field_val568, i32 0, i32 3
  %field_val608 = load ptr, ptr %unreg_field_3607, align 8
  call void @tl_mem_unregister(ptr %field_val608)
  %unreg_field_0609 = getelementptr inbounds %MLP, ptr %field_val608, i32 0, i32 0
  %field_val610 = load ptr, ptr %unreg_field_0609, align 8
  call void @tl_mem_unregister(ptr %field_val610)
  %unreg_field_0611 = getelementptr inbounds %Linear, ptr %field_val610, i32 0, i32 0
  %field_val612 = load ptr, ptr %unreg_field_0611, align 8
  call void @tl_mem_unregister(ptr %field_val612)
  %unreg_field_1613 = getelementptr inbounds %Linear, ptr %field_val610, i32 0, i32 1
  %field_val614 = load ptr, ptr %unreg_field_1613, align 8
  call void @tl_mem_unregister(ptr %field_val614)
  %unreg_field_1615 = getelementptr inbounds %MLP, ptr %field_val608, i32 0, i32 1
  %field_val616 = load ptr, ptr %unreg_field_1615, align 8
  call void @tl_mem_unregister(ptr %field_val616)
  %unreg_field_0617 = getelementptr inbounds %Linear, ptr %field_val616, i32 0, i32 0
  %field_val618 = load ptr, ptr %unreg_field_0617, align 8
  call void @tl_mem_unregister(ptr %field_val618)
  %unreg_field_1619 = getelementptr inbounds %Linear, ptr %field_val616, i32 0, i32 1
  %field_val620 = load ptr, ptr %unreg_field_1619, align 8
  call void @tl_mem_unregister(ptr %field_val620)
  %unreg_field_5 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 5
  %field_val621 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val621)
  %unreg_field_0622 = getelementptr inbounds %Block, ptr %field_val621, i32 0, i32 0
  %field_val623 = load ptr, ptr %unreg_field_0622, align 8
  call void @tl_mem_unregister(ptr %field_val623)
  %unreg_field_0624 = getelementptr inbounds %LayerNorm, ptr %field_val623, i32 0, i32 0
  %field_val625 = load ptr, ptr %unreg_field_0624, align 8
  call void @tl_mem_unregister(ptr %field_val625)
  %unreg_field_1626 = getelementptr inbounds %LayerNorm, ptr %field_val623, i32 0, i32 1
  %field_val627 = load ptr, ptr %unreg_field_1626, align 8
  call void @tl_mem_unregister(ptr %field_val627)
  %unreg_field_1628 = getelementptr inbounds %Block, ptr %field_val621, i32 0, i32 1
  %field_val629 = load ptr, ptr %unreg_field_1628, align 8
  call void @tl_mem_unregister(ptr %field_val629)
  %unreg_field_0630 = getelementptr inbounds %CausalSelfAttention, ptr %field_val629, i32 0, i32 0
  %field_val631 = load ptr, ptr %unreg_field_0630, align 8
  call void @tl_mem_unregister(ptr %field_val631)
  %unreg_field_0632 = getelementptr inbounds %Linear, ptr %field_val631, i32 0, i32 0
  %field_val633 = load ptr, ptr %unreg_field_0632, align 8
  call void @tl_mem_unregister(ptr %field_val633)
  %unreg_field_1634 = getelementptr inbounds %Linear, ptr %field_val631, i32 0, i32 1
  %field_val635 = load ptr, ptr %unreg_field_1634, align 8
  call void @tl_mem_unregister(ptr %field_val635)
  %unreg_field_1636 = getelementptr inbounds %CausalSelfAttention, ptr %field_val629, i32 0, i32 1
  %field_val637 = load ptr, ptr %unreg_field_1636, align 8
  call void @tl_mem_unregister(ptr %field_val637)
  %unreg_field_0638 = getelementptr inbounds %Linear, ptr %field_val637, i32 0, i32 0
  %field_val639 = load ptr, ptr %unreg_field_0638, align 8
  call void @tl_mem_unregister(ptr %field_val639)
  %unreg_field_1640 = getelementptr inbounds %Linear, ptr %field_val637, i32 0, i32 1
  %field_val641 = load ptr, ptr %unreg_field_1640, align 8
  call void @tl_mem_unregister(ptr %field_val641)
  %unreg_field_2642 = getelementptr inbounds %CausalSelfAttention, ptr %field_val629, i32 0, i32 2
  %field_val643 = load ptr, ptr %unreg_field_2642, align 8
  call void @tl_mem_unregister(ptr %field_val643)
  %unreg_field_0644 = getelementptr inbounds %Linear, ptr %field_val643, i32 0, i32 0
  %field_val645 = load ptr, ptr %unreg_field_0644, align 8
  call void @tl_mem_unregister(ptr %field_val645)
  %unreg_field_1646 = getelementptr inbounds %Linear, ptr %field_val643, i32 0, i32 1
  %field_val647 = load ptr, ptr %unreg_field_1646, align 8
  call void @tl_mem_unregister(ptr %field_val647)
  %unreg_field_3648 = getelementptr inbounds %CausalSelfAttention, ptr %field_val629, i32 0, i32 3
  %field_val649 = load ptr, ptr %unreg_field_3648, align 8
  call void @tl_mem_unregister(ptr %field_val649)
  %unreg_field_0650 = getelementptr inbounds %Linear, ptr %field_val649, i32 0, i32 0
  %field_val651 = load ptr, ptr %unreg_field_0650, align 8
  call void @tl_mem_unregister(ptr %field_val651)
  %unreg_field_1652 = getelementptr inbounds %Linear, ptr %field_val649, i32 0, i32 1
  %field_val653 = load ptr, ptr %unreg_field_1652, align 8
  call void @tl_mem_unregister(ptr %field_val653)
  %unreg_field_2654 = getelementptr inbounds %Block, ptr %field_val621, i32 0, i32 2
  %field_val655 = load ptr, ptr %unreg_field_2654, align 8
  call void @tl_mem_unregister(ptr %field_val655)
  %unreg_field_0656 = getelementptr inbounds %LayerNorm, ptr %field_val655, i32 0, i32 0
  %field_val657 = load ptr, ptr %unreg_field_0656, align 8
  call void @tl_mem_unregister(ptr %field_val657)
  %unreg_field_1658 = getelementptr inbounds %LayerNorm, ptr %field_val655, i32 0, i32 1
  %field_val659 = load ptr, ptr %unreg_field_1658, align 8
  call void @tl_mem_unregister(ptr %field_val659)
  %unreg_field_3660 = getelementptr inbounds %Block, ptr %field_val621, i32 0, i32 3
  %field_val661 = load ptr, ptr %unreg_field_3660, align 8
  call void @tl_mem_unregister(ptr %field_val661)
  %unreg_field_0662 = getelementptr inbounds %MLP, ptr %field_val661, i32 0, i32 0
  %field_val663 = load ptr, ptr %unreg_field_0662, align 8
  call void @tl_mem_unregister(ptr %field_val663)
  %unreg_field_0664 = getelementptr inbounds %Linear, ptr %field_val663, i32 0, i32 0
  %field_val665 = load ptr, ptr %unreg_field_0664, align 8
  call void @tl_mem_unregister(ptr %field_val665)
  %unreg_field_1666 = getelementptr inbounds %Linear, ptr %field_val663, i32 0, i32 1
  %field_val667 = load ptr, ptr %unreg_field_1666, align 8
  call void @tl_mem_unregister(ptr %field_val667)
  %unreg_field_1668 = getelementptr inbounds %MLP, ptr %field_val661, i32 0, i32 1
  %field_val669 = load ptr, ptr %unreg_field_1668, align 8
  call void @tl_mem_unregister(ptr %field_val669)
  %unreg_field_0670 = getelementptr inbounds %Linear, ptr %field_val669, i32 0, i32 0
  %field_val671 = load ptr, ptr %unreg_field_0670, align 8
  call void @tl_mem_unregister(ptr %field_val671)
  %unreg_field_1672 = getelementptr inbounds %Linear, ptr %field_val669, i32 0, i32 1
  %field_val673 = load ptr, ptr %unreg_field_1672, align 8
  call void @tl_mem_unregister(ptr %field_val673)
  %unreg_field_6 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 6
  %field_val674 = load ptr, ptr %unreg_field_6, align 8
  call void @tl_mem_unregister(ptr %field_val674)
  %unreg_field_0675 = getelementptr inbounds %LayerNorm, ptr %field_val674, i32 0, i32 0
  %field_val676 = load ptr, ptr %unreg_field_0675, align 8
  call void @tl_mem_unregister(ptr %field_val676)
  %unreg_field_1677 = getelementptr inbounds %LayerNorm, ptr %field_val674, i32 0, i32 1
  %field_val678 = load ptr, ptr %unreg_field_1677, align 8
  call void @tl_mem_unregister(ptr %field_val678)
  %unreg_field_7 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 7
  %field_val679 = load ptr, ptr %unreg_field_7, align 8
  call void @tl_mem_unregister(ptr %field_val679)
  %unreg_field_0680 = getelementptr inbounds %Linear, ptr %field_val679, i32 0, i32 0
  %field_val681 = load ptr, ptr %unreg_field_0680, align 8
  call void @tl_mem_unregister(ptr %field_val681)
  %unreg_field_1682 = getelementptr inbounds %Linear, ptr %field_val679, i32 0, i32 1
  %field_val683 = load ptr, ptr %unreg_field_1682, align 8
  call void @tl_mem_unregister(ptr %field_val683)
  call void @tl_mem_unregister(ptr %call_tmp)
  store ptr %call_tmp, ptr %model, align 8
  %epoch684 = load i64, ptr %epoch, align 8
  %epoch685 = load i64, ptr %epoch, align 8
  %divtmp = sdiv i64 %epoch685, 5
  %multmp = mul i64 %divtmp, 5
  %subtmp = sub i64 %epoch684, %multmp
  %eqtmp = icmp eq i64 %subtmp, 4
  br i1 %eqtmp, label %then, label %else

then:                                             ; preds = %continue_after_free
  call void @tl_mem_enter_scope()
  %model686 = load ptr, ptr %model, align 8
  %w687 = getelementptr inbounds %GPTHeavy, ptr %model686, i32 0, i32 0
  %sub_ptr688 = load ptr, ptr %w687, align 8
  %w689 = getelementptr inbounds %Embedding, ptr %sub_ptr688, i32 0, i32 0
  %w690 = load ptr, ptr %w689, align 8
  call void @tl_add_parameter(ptr @key_str.175, ptr %w690)
  %wp691 = getelementptr inbounds %GPTHeavy, ptr %model686, i32 0, i32 1
  %sub_ptr692 = load ptr, ptr %wp691, align 8
  %w693 = getelementptr inbounds %Embedding, ptr %sub_ptr692, i32 0, i32 0
  %w694 = load ptr, ptr %w693, align 8
  call void @tl_add_parameter(ptr @key_str.176, ptr %w694)
  %b1695 = getelementptr inbounds %GPTHeavy, ptr %model686, i32 0, i32 2
  %sub_ptr696 = load ptr, ptr %b1695, align 8
  %l1697 = getelementptr inbounds %Block, ptr %sub_ptr696, i32 0, i32 0
  %sub_ptr698 = load ptr, ptr %l1697, align 8
  %w699 = getelementptr inbounds %LayerNorm, ptr %sub_ptr698, i32 0, i32 0
  %w700 = load ptr, ptr %w699, align 8
  call void @tl_add_parameter(ptr @key_str.177, ptr %w700)
  %b701 = getelementptr inbounds %LayerNorm, ptr %sub_ptr698, i32 0, i32 1
  %b702 = load ptr, ptr %b701, align 8
  call void @tl_add_parameter(ptr @key_str.178, ptr %b702)
  %a703 = getelementptr inbounds %Block, ptr %sub_ptr696, i32 0, i32 1
  %sub_ptr704 = load ptr, ptr %a703, align 8
  %q_proj705 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr704, i32 0, i32 0
  %sub_ptr706 = load ptr, ptr %q_proj705, align 8
  %W707 = getelementptr inbounds %Linear, ptr %sub_ptr706, i32 0, i32 0
  %W708 = load ptr, ptr %W707, align 8
  call void @tl_add_parameter(ptr @key_str.179, ptr %W708)
  %b709 = getelementptr inbounds %Linear, ptr %sub_ptr706, i32 0, i32 1
  %b710 = load ptr, ptr %b709, align 8
  call void @tl_add_parameter(ptr @key_str.180, ptr %b710)
  %k_proj711 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr704, i32 0, i32 1
  %sub_ptr712 = load ptr, ptr %k_proj711, align 8
  %W713 = getelementptr inbounds %Linear, ptr %sub_ptr712, i32 0, i32 0
  %W714 = load ptr, ptr %W713, align 8
  call void @tl_add_parameter(ptr @key_str.181, ptr %W714)
  %b715 = getelementptr inbounds %Linear, ptr %sub_ptr712, i32 0, i32 1
  %b716 = load ptr, ptr %b715, align 8
  call void @tl_add_parameter(ptr @key_str.182, ptr %b716)
  %v_proj717 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr704, i32 0, i32 2
  %sub_ptr718 = load ptr, ptr %v_proj717, align 8
  %W719 = getelementptr inbounds %Linear, ptr %sub_ptr718, i32 0, i32 0
  %W720 = load ptr, ptr %W719, align 8
  call void @tl_add_parameter(ptr @key_str.183, ptr %W720)
  %b721 = getelementptr inbounds %Linear, ptr %sub_ptr718, i32 0, i32 1
  %b722 = load ptr, ptr %b721, align 8
  call void @tl_add_parameter(ptr @key_str.184, ptr %b722)
  %p_proj723 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr704, i32 0, i32 3
  %sub_ptr724 = load ptr, ptr %p_proj723, align 8
  %W725 = getelementptr inbounds %Linear, ptr %sub_ptr724, i32 0, i32 0
  %W726 = load ptr, ptr %W725, align 8
  call void @tl_add_parameter(ptr @key_str.185, ptr %W726)
  %b727 = getelementptr inbounds %Linear, ptr %sub_ptr724, i32 0, i32 1
  %b728 = load ptr, ptr %b727, align 8
  call void @tl_add_parameter(ptr @key_str.186, ptr %b728)
  %l2729 = getelementptr inbounds %Block, ptr %sub_ptr696, i32 0, i32 2
  %sub_ptr730 = load ptr, ptr %l2729, align 8
  %w731 = getelementptr inbounds %LayerNorm, ptr %sub_ptr730, i32 0, i32 0
  %w732 = load ptr, ptr %w731, align 8
  call void @tl_add_parameter(ptr @key_str.187, ptr %w732)
  %b733 = getelementptr inbounds %LayerNorm, ptr %sub_ptr730, i32 0, i32 1
  %b734 = load ptr, ptr %b733, align 8
  call void @tl_add_parameter(ptr @key_str.188, ptr %b734)
  %m735 = getelementptr inbounds %Block, ptr %sub_ptr696, i32 0, i32 3
  %sub_ptr736 = load ptr, ptr %m735, align 8
  %f737 = getelementptr inbounds %MLP, ptr %sub_ptr736, i32 0, i32 0
  %sub_ptr738 = load ptr, ptr %f737, align 8
  %W739 = getelementptr inbounds %Linear, ptr %sub_ptr738, i32 0, i32 0
  %W740 = load ptr, ptr %W739, align 8
  call void @tl_add_parameter(ptr @key_str.189, ptr %W740)
  %b741 = getelementptr inbounds %Linear, ptr %sub_ptr738, i32 0, i32 1
  %b742 = load ptr, ptr %b741, align 8
  call void @tl_add_parameter(ptr @key_str.190, ptr %b742)
  %p743 = getelementptr inbounds %MLP, ptr %sub_ptr736, i32 0, i32 1
  %sub_ptr744 = load ptr, ptr %p743, align 8
  %W745 = getelementptr inbounds %Linear, ptr %sub_ptr744, i32 0, i32 0
  %W746 = load ptr, ptr %W745, align 8
  call void @tl_add_parameter(ptr @key_str.191, ptr %W746)
  %b747 = getelementptr inbounds %Linear, ptr %sub_ptr744, i32 0, i32 1
  %b748 = load ptr, ptr %b747, align 8
  call void @tl_add_parameter(ptr @key_str.192, ptr %b748)
  %b2749 = getelementptr inbounds %GPTHeavy, ptr %model686, i32 0, i32 3
  %sub_ptr750 = load ptr, ptr %b2749, align 8
  %l1751 = getelementptr inbounds %Block, ptr %sub_ptr750, i32 0, i32 0
  %sub_ptr752 = load ptr, ptr %l1751, align 8
  %w753 = getelementptr inbounds %LayerNorm, ptr %sub_ptr752, i32 0, i32 0
  %w754 = load ptr, ptr %w753, align 8
  call void @tl_add_parameter(ptr @key_str.193, ptr %w754)
  %b755 = getelementptr inbounds %LayerNorm, ptr %sub_ptr752, i32 0, i32 1
  %b756 = load ptr, ptr %b755, align 8
  call void @tl_add_parameter(ptr @key_str.194, ptr %b756)
  %a757 = getelementptr inbounds %Block, ptr %sub_ptr750, i32 0, i32 1
  %sub_ptr758 = load ptr, ptr %a757, align 8
  %q_proj759 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr758, i32 0, i32 0
  %sub_ptr760 = load ptr, ptr %q_proj759, align 8
  %W761 = getelementptr inbounds %Linear, ptr %sub_ptr760, i32 0, i32 0
  %W762 = load ptr, ptr %W761, align 8
  call void @tl_add_parameter(ptr @key_str.195, ptr %W762)
  %b763 = getelementptr inbounds %Linear, ptr %sub_ptr760, i32 0, i32 1
  %b764 = load ptr, ptr %b763, align 8
  call void @tl_add_parameter(ptr @key_str.196, ptr %b764)
  %k_proj765 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr758, i32 0, i32 1
  %sub_ptr766 = load ptr, ptr %k_proj765, align 8
  %W767 = getelementptr inbounds %Linear, ptr %sub_ptr766, i32 0, i32 0
  %W768 = load ptr, ptr %W767, align 8
  call void @tl_add_parameter(ptr @key_str.197, ptr %W768)
  %b769 = getelementptr inbounds %Linear, ptr %sub_ptr766, i32 0, i32 1
  %b770 = load ptr, ptr %b769, align 8
  call void @tl_add_parameter(ptr @key_str.198, ptr %b770)
  %v_proj771 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr758, i32 0, i32 2
  %sub_ptr772 = load ptr, ptr %v_proj771, align 8
  %W773 = getelementptr inbounds %Linear, ptr %sub_ptr772, i32 0, i32 0
  %W774 = load ptr, ptr %W773, align 8
  call void @tl_add_parameter(ptr @key_str.199, ptr %W774)
  %b775 = getelementptr inbounds %Linear, ptr %sub_ptr772, i32 0, i32 1
  %b776 = load ptr, ptr %b775, align 8
  call void @tl_add_parameter(ptr @key_str.200, ptr %b776)
  %p_proj777 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr758, i32 0, i32 3
  %sub_ptr778 = load ptr, ptr %p_proj777, align 8
  %W779 = getelementptr inbounds %Linear, ptr %sub_ptr778, i32 0, i32 0
  %W780 = load ptr, ptr %W779, align 8
  call void @tl_add_parameter(ptr @key_str.201, ptr %W780)
  %b781 = getelementptr inbounds %Linear, ptr %sub_ptr778, i32 0, i32 1
  %b782 = load ptr, ptr %b781, align 8
  call void @tl_add_parameter(ptr @key_str.202, ptr %b782)
  %l2783 = getelementptr inbounds %Block, ptr %sub_ptr750, i32 0, i32 2
  %sub_ptr784 = load ptr, ptr %l2783, align 8
  %w785 = getelementptr inbounds %LayerNorm, ptr %sub_ptr784, i32 0, i32 0
  %w786 = load ptr, ptr %w785, align 8
  call void @tl_add_parameter(ptr @key_str.203, ptr %w786)
  %b787 = getelementptr inbounds %LayerNorm, ptr %sub_ptr784, i32 0, i32 1
  %b788 = load ptr, ptr %b787, align 8
  call void @tl_add_parameter(ptr @key_str.204, ptr %b788)
  %m789 = getelementptr inbounds %Block, ptr %sub_ptr750, i32 0, i32 3
  %sub_ptr790 = load ptr, ptr %m789, align 8
  %f791 = getelementptr inbounds %MLP, ptr %sub_ptr790, i32 0, i32 0
  %sub_ptr792 = load ptr, ptr %f791, align 8
  %W793 = getelementptr inbounds %Linear, ptr %sub_ptr792, i32 0, i32 0
  %W794 = load ptr, ptr %W793, align 8
  call void @tl_add_parameter(ptr @key_str.205, ptr %W794)
  %b795 = getelementptr inbounds %Linear, ptr %sub_ptr792, i32 0, i32 1
  %b796 = load ptr, ptr %b795, align 8
  call void @tl_add_parameter(ptr @key_str.206, ptr %b796)
  %p797 = getelementptr inbounds %MLP, ptr %sub_ptr790, i32 0, i32 1
  %sub_ptr798 = load ptr, ptr %p797, align 8
  %W799 = getelementptr inbounds %Linear, ptr %sub_ptr798, i32 0, i32 0
  %W800 = load ptr, ptr %W799, align 8
  call void @tl_add_parameter(ptr @key_str.207, ptr %W800)
  %b801 = getelementptr inbounds %Linear, ptr %sub_ptr798, i32 0, i32 1
  %b802 = load ptr, ptr %b801, align 8
  call void @tl_add_parameter(ptr @key_str.208, ptr %b802)
  %b3803 = getelementptr inbounds %GPTHeavy, ptr %model686, i32 0, i32 4
  %sub_ptr804 = load ptr, ptr %b3803, align 8
  %l1805 = getelementptr inbounds %Block, ptr %sub_ptr804, i32 0, i32 0
  %sub_ptr806 = load ptr, ptr %l1805, align 8
  %w807 = getelementptr inbounds %LayerNorm, ptr %sub_ptr806, i32 0, i32 0
  %w808 = load ptr, ptr %w807, align 8
  call void @tl_add_parameter(ptr @key_str.209, ptr %w808)
  %b809 = getelementptr inbounds %LayerNorm, ptr %sub_ptr806, i32 0, i32 1
  %b810 = load ptr, ptr %b809, align 8
  call void @tl_add_parameter(ptr @key_str.210, ptr %b810)
  %a811 = getelementptr inbounds %Block, ptr %sub_ptr804, i32 0, i32 1
  %sub_ptr812 = load ptr, ptr %a811, align 8
  %q_proj813 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr812, i32 0, i32 0
  %sub_ptr814 = load ptr, ptr %q_proj813, align 8
  %W815 = getelementptr inbounds %Linear, ptr %sub_ptr814, i32 0, i32 0
  %W816 = load ptr, ptr %W815, align 8
  call void @tl_add_parameter(ptr @key_str.211, ptr %W816)
  %b817 = getelementptr inbounds %Linear, ptr %sub_ptr814, i32 0, i32 1
  %b818 = load ptr, ptr %b817, align 8
  call void @tl_add_parameter(ptr @key_str.212, ptr %b818)
  %k_proj819 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr812, i32 0, i32 1
  %sub_ptr820 = load ptr, ptr %k_proj819, align 8
  %W821 = getelementptr inbounds %Linear, ptr %sub_ptr820, i32 0, i32 0
  %W822 = load ptr, ptr %W821, align 8
  call void @tl_add_parameter(ptr @key_str.213, ptr %W822)
  %b823 = getelementptr inbounds %Linear, ptr %sub_ptr820, i32 0, i32 1
  %b824 = load ptr, ptr %b823, align 8
  call void @tl_add_parameter(ptr @key_str.214, ptr %b824)
  %v_proj825 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr812, i32 0, i32 2
  %sub_ptr826 = load ptr, ptr %v_proj825, align 8
  %W827 = getelementptr inbounds %Linear, ptr %sub_ptr826, i32 0, i32 0
  %W828 = load ptr, ptr %W827, align 8
  call void @tl_add_parameter(ptr @key_str.215, ptr %W828)
  %b829 = getelementptr inbounds %Linear, ptr %sub_ptr826, i32 0, i32 1
  %b830 = load ptr, ptr %b829, align 8
  call void @tl_add_parameter(ptr @key_str.216, ptr %b830)
  %p_proj831 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr812, i32 0, i32 3
  %sub_ptr832 = load ptr, ptr %p_proj831, align 8
  %W833 = getelementptr inbounds %Linear, ptr %sub_ptr832, i32 0, i32 0
  %W834 = load ptr, ptr %W833, align 8
  call void @tl_add_parameter(ptr @key_str.217, ptr %W834)
  %b835 = getelementptr inbounds %Linear, ptr %sub_ptr832, i32 0, i32 1
  %b836 = load ptr, ptr %b835, align 8
  call void @tl_add_parameter(ptr @key_str.218, ptr %b836)
  %l2837 = getelementptr inbounds %Block, ptr %sub_ptr804, i32 0, i32 2
  %sub_ptr838 = load ptr, ptr %l2837, align 8
  %w839 = getelementptr inbounds %LayerNorm, ptr %sub_ptr838, i32 0, i32 0
  %w840 = load ptr, ptr %w839, align 8
  call void @tl_add_parameter(ptr @key_str.219, ptr %w840)
  %b841 = getelementptr inbounds %LayerNorm, ptr %sub_ptr838, i32 0, i32 1
  %b842 = load ptr, ptr %b841, align 8
  call void @tl_add_parameter(ptr @key_str.220, ptr %b842)
  %m843 = getelementptr inbounds %Block, ptr %sub_ptr804, i32 0, i32 3
  %sub_ptr844 = load ptr, ptr %m843, align 8
  %f845 = getelementptr inbounds %MLP, ptr %sub_ptr844, i32 0, i32 0
  %sub_ptr846 = load ptr, ptr %f845, align 8
  %W847 = getelementptr inbounds %Linear, ptr %sub_ptr846, i32 0, i32 0
  %W848 = load ptr, ptr %W847, align 8
  call void @tl_add_parameter(ptr @key_str.221, ptr %W848)
  %b849 = getelementptr inbounds %Linear, ptr %sub_ptr846, i32 0, i32 1
  %b850 = load ptr, ptr %b849, align 8
  call void @tl_add_parameter(ptr @key_str.222, ptr %b850)
  %p851 = getelementptr inbounds %MLP, ptr %sub_ptr844, i32 0, i32 1
  %sub_ptr852 = load ptr, ptr %p851, align 8
  %W853 = getelementptr inbounds %Linear, ptr %sub_ptr852, i32 0, i32 0
  %W854 = load ptr, ptr %W853, align 8
  call void @tl_add_parameter(ptr @key_str.223, ptr %W854)
  %b855 = getelementptr inbounds %Linear, ptr %sub_ptr852, i32 0, i32 1
  %b856 = load ptr, ptr %b855, align 8
  call void @tl_add_parameter(ptr @key_str.224, ptr %b856)
  %b4857 = getelementptr inbounds %GPTHeavy, ptr %model686, i32 0, i32 5
  %sub_ptr858 = load ptr, ptr %b4857, align 8
  %l1859 = getelementptr inbounds %Block, ptr %sub_ptr858, i32 0, i32 0
  %sub_ptr860 = load ptr, ptr %l1859, align 8
  %w861 = getelementptr inbounds %LayerNorm, ptr %sub_ptr860, i32 0, i32 0
  %w862 = load ptr, ptr %w861, align 8
  call void @tl_add_parameter(ptr @key_str.225, ptr %w862)
  %b863 = getelementptr inbounds %LayerNorm, ptr %sub_ptr860, i32 0, i32 1
  %b864 = load ptr, ptr %b863, align 8
  call void @tl_add_parameter(ptr @key_str.226, ptr %b864)
  %a865 = getelementptr inbounds %Block, ptr %sub_ptr858, i32 0, i32 1
  %sub_ptr866 = load ptr, ptr %a865, align 8
  %q_proj867 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr866, i32 0, i32 0
  %sub_ptr868 = load ptr, ptr %q_proj867, align 8
  %W869 = getelementptr inbounds %Linear, ptr %sub_ptr868, i32 0, i32 0
  %W870 = load ptr, ptr %W869, align 8
  call void @tl_add_parameter(ptr @key_str.227, ptr %W870)
  %b871 = getelementptr inbounds %Linear, ptr %sub_ptr868, i32 0, i32 1
  %b872 = load ptr, ptr %b871, align 8
  call void @tl_add_parameter(ptr @key_str.228, ptr %b872)
  %k_proj873 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr866, i32 0, i32 1
  %sub_ptr874 = load ptr, ptr %k_proj873, align 8
  %W875 = getelementptr inbounds %Linear, ptr %sub_ptr874, i32 0, i32 0
  %W876 = load ptr, ptr %W875, align 8
  call void @tl_add_parameter(ptr @key_str.229, ptr %W876)
  %b877 = getelementptr inbounds %Linear, ptr %sub_ptr874, i32 0, i32 1
  %b878 = load ptr, ptr %b877, align 8
  call void @tl_add_parameter(ptr @key_str.230, ptr %b878)
  %v_proj879 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr866, i32 0, i32 2
  %sub_ptr880 = load ptr, ptr %v_proj879, align 8
  %W881 = getelementptr inbounds %Linear, ptr %sub_ptr880, i32 0, i32 0
  %W882 = load ptr, ptr %W881, align 8
  call void @tl_add_parameter(ptr @key_str.231, ptr %W882)
  %b883 = getelementptr inbounds %Linear, ptr %sub_ptr880, i32 0, i32 1
  %b884 = load ptr, ptr %b883, align 8
  call void @tl_add_parameter(ptr @key_str.232, ptr %b884)
  %p_proj885 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr866, i32 0, i32 3
  %sub_ptr886 = load ptr, ptr %p_proj885, align 8
  %W887 = getelementptr inbounds %Linear, ptr %sub_ptr886, i32 0, i32 0
  %W888 = load ptr, ptr %W887, align 8
  call void @tl_add_parameter(ptr @key_str.233, ptr %W888)
  %b889 = getelementptr inbounds %Linear, ptr %sub_ptr886, i32 0, i32 1
  %b890 = load ptr, ptr %b889, align 8
  call void @tl_add_parameter(ptr @key_str.234, ptr %b890)
  %l2891 = getelementptr inbounds %Block, ptr %sub_ptr858, i32 0, i32 2
  %sub_ptr892 = load ptr, ptr %l2891, align 8
  %w893 = getelementptr inbounds %LayerNorm, ptr %sub_ptr892, i32 0, i32 0
  %w894 = load ptr, ptr %w893, align 8
  call void @tl_add_parameter(ptr @key_str.235, ptr %w894)
  %b895 = getelementptr inbounds %LayerNorm, ptr %sub_ptr892, i32 0, i32 1
  %b896 = load ptr, ptr %b895, align 8
  call void @tl_add_parameter(ptr @key_str.236, ptr %b896)
  %m897 = getelementptr inbounds %Block, ptr %sub_ptr858, i32 0, i32 3
  %sub_ptr898 = load ptr, ptr %m897, align 8
  %f899 = getelementptr inbounds %MLP, ptr %sub_ptr898, i32 0, i32 0
  %sub_ptr900 = load ptr, ptr %f899, align 8
  %W901 = getelementptr inbounds %Linear, ptr %sub_ptr900, i32 0, i32 0
  %W902 = load ptr, ptr %W901, align 8
  call void @tl_add_parameter(ptr @key_str.237, ptr %W902)
  %b903 = getelementptr inbounds %Linear, ptr %sub_ptr900, i32 0, i32 1
  %b904 = load ptr, ptr %b903, align 8
  call void @tl_add_parameter(ptr @key_str.238, ptr %b904)
  %p905 = getelementptr inbounds %MLP, ptr %sub_ptr898, i32 0, i32 1
  %sub_ptr906 = load ptr, ptr %p905, align 8
  %W907 = getelementptr inbounds %Linear, ptr %sub_ptr906, i32 0, i32 0
  %W908 = load ptr, ptr %W907, align 8
  call void @tl_add_parameter(ptr @key_str.239, ptr %W908)
  %b909 = getelementptr inbounds %Linear, ptr %sub_ptr906, i32 0, i32 1
  %b910 = load ptr, ptr %b909, align 8
  call void @tl_add_parameter(ptr @key_str.240, ptr %b910)
  %l911 = getelementptr inbounds %GPTHeavy, ptr %model686, i32 0, i32 6
  %sub_ptr912 = load ptr, ptr %l911, align 8
  %w913 = getelementptr inbounds %LayerNorm, ptr %sub_ptr912, i32 0, i32 0
  %w914 = load ptr, ptr %w913, align 8
  call void @tl_add_parameter(ptr @key_str.241, ptr %w914)
  %b915 = getelementptr inbounds %LayerNorm, ptr %sub_ptr912, i32 0, i32 1
  %b916 = load ptr, ptr %b915, align 8
  call void @tl_add_parameter(ptr @key_str.242, ptr %b916)
  %h917 = getelementptr inbounds %GPTHeavy, ptr %model686, i32 0, i32 7
  %sub_ptr918 = load ptr, ptr %h917, align 8
  %W919 = getelementptr inbounds %Linear, ptr %sub_ptr918, i32 0, i32 0
  %W920 = load ptr, ptr %W919, align 8
  call void @tl_add_parameter(ptr @key_str.243, ptr %W920)
  %b921 = getelementptr inbounds %Linear, ptr %sub_ptr918, i32 0, i32 1
  %b922 = load ptr, ptr %b921, align 8
  call void @tl_add_parameter(ptr @key_str.244, ptr %b922)
  call void @tl_save_all_params(ptr @str_literal.245)
  call void @tl_print_string(ptr @str_literal.246)
  call void @tl_mem_exit_scope()
  br label %merge

else:                                             ; preds = %continue_after_free
  call void @tl_mem_enter_scope()
  call void @tl_mem_exit_scope()
  br label %merge

merge:                                            ; preds = %else, %then
  call void @tl_mem_exit_scope()
  %next_idx = add i64 %for_idx, 1
  br label %for_header
}
