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
  %tensor_to_free = load ptr, ptr %gW, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free34 = load ptr, ptr %gb, align 8
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
  %tensor_to_free19 = load ptr, ptr %v, align 8
  call void @tl_tensor_free(ptr %tensor_to_free19)
  %tensor_to_free20 = load ptr, ptr %q, align 8
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
  br label %skip_free

skip_free:                                        ; preds = %free_old_val, %entry
  store ptr %call_method, ptr %ptr_q_proj, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s8 = load ptr, ptr %s, align 8
  %ptr_k_proj = getelementptr inbounds %CausalSelfAttention, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_k_proj10 = getelementptr inbounds %CausalSelfAttention, ptr %s9, i32 0, i32 1
  %k_proj = load ptr, ptr %ptr_k_proj10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_Linear_step(ptr %k_proj, float %lr11)
  call void @tl_mem_register_struct(ptr %call_method12)
  %old_field_val13 = load ptr, ptr %ptr_k_proj, align 8
  %cnt_free_diff14 = icmp ne ptr %old_field_val13, %call_method12
  br i1 %cnt_free_diff14, label %free_old_val15, label %skip_free16

free_old_val15:                                   ; preds = %skip_free
  br label %skip_free16

skip_free16:                                      ; preds = %free_old_val15, %skip_free
  store ptr %call_method12, ptr %ptr_k_proj, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s17 = load ptr, ptr %s, align 8
  %ptr_v_proj = getelementptr inbounds %CausalSelfAttention, ptr %s17, i32 0, i32 2
  %s18 = load ptr, ptr %s, align 8
  %ptr_v_proj19 = getelementptr inbounds %CausalSelfAttention, ptr %s18, i32 0, i32 2
  %v_proj = load ptr, ptr %ptr_v_proj19, align 8
  %lr20 = load float, ptr %lr2, align 4
  %call_method21 = call ptr @tl_Linear_step(ptr %v_proj, float %lr20)
  call void @tl_mem_register_struct(ptr %call_method21)
  %old_field_val22 = load ptr, ptr %ptr_v_proj, align 8
  %cnt_free_diff23 = icmp ne ptr %old_field_val22, %call_method21
  br i1 %cnt_free_diff23, label %free_old_val24, label %skip_free25

free_old_val24:                                   ; preds = %skip_free16
  br label %skip_free25

skip_free25:                                      ; preds = %free_old_val24, %skip_free16
  store ptr %call_method21, ptr %ptr_v_proj, align 8
  call void @tl_mem_unregister(ptr %call_method21)
  %s26 = load ptr, ptr %s, align 8
  %ptr_p_proj = getelementptr inbounds %CausalSelfAttention, ptr %s26, i32 0, i32 3
  %s27 = load ptr, ptr %s, align 8
  %ptr_p_proj28 = getelementptr inbounds %CausalSelfAttention, ptr %s27, i32 0, i32 3
  %p_proj = load ptr, ptr %ptr_p_proj28, align 8
  %lr29 = load float, ptr %lr2, align 4
  %call_method30 = call ptr @tl_Linear_step(ptr %p_proj, float %lr29)
  call void @tl_mem_register_struct(ptr %call_method30)
  %old_field_val31 = load ptr, ptr %ptr_p_proj, align 8
  %cnt_free_diff32 = icmp ne ptr %old_field_val31, %call_method30
  br i1 %cnt_free_diff32, label %free_old_val33, label %skip_free34

free_old_val33:                                   ; preds = %skip_free25
  br label %skip_free34

skip_free34:                                      ; preds = %free_old_val33, %skip_free25
  store ptr %call_method30, ptr %ptr_p_proj, align 8
  call void @tl_mem_unregister(ptr %call_method30)
  %s35 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s35)
  %unreg_field_0 = getelementptr inbounds %CausalSelfAttention, ptr %s35, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_036 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 0
  %field_val37 = load ptr, ptr %unreg_field_036, align 8
  call void @tl_mem_unregister(ptr %field_val37)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 1
  %field_val38 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_139 = getelementptr inbounds %CausalSelfAttention, ptr %s35, i32 0, i32 1
  %field_val40 = load ptr, ptr %unreg_field_139, align 8
  call void @tl_mem_unregister(ptr %field_val40)
  %unreg_field_041 = getelementptr inbounds %Linear, ptr %field_val40, i32 0, i32 0
  %field_val42 = load ptr, ptr %unreg_field_041, align 8
  call void @tl_mem_unregister(ptr %field_val42)
  %unreg_field_143 = getelementptr inbounds %Linear, ptr %field_val40, i32 0, i32 1
  %field_val44 = load ptr, ptr %unreg_field_143, align 8
  call void @tl_mem_unregister(ptr %field_val44)
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %s35, i32 0, i32 2
  %field_val45 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val45)
  %unreg_field_046 = getelementptr inbounds %Linear, ptr %field_val45, i32 0, i32 0
  %field_val47 = load ptr, ptr %unreg_field_046, align 8
  call void @tl_mem_unregister(ptr %field_val47)
  %unreg_field_148 = getelementptr inbounds %Linear, ptr %field_val45, i32 0, i32 1
  %field_val49 = load ptr, ptr %unreg_field_148, align 8
  call void @tl_mem_unregister(ptr %field_val49)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %s35, i32 0, i32 3
  %field_val50 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val50)
  %unreg_field_051 = getelementptr inbounds %Linear, ptr %field_val50, i32 0, i32 0
  %field_val52 = load ptr, ptr %unreg_field_051, align 8
  call void @tl_mem_unregister(ptr %field_val52)
  %unreg_field_153 = getelementptr inbounds %Linear, ptr %field_val50, i32 0, i32 1
  %field_val54 = load ptr, ptr %unreg_field_153, align 8
  call void @tl_mem_unregister(ptr %field_val54)
  call void @tl_mem_exit_scope()
  ret ptr %s35
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
  br label %skip_free

skip_free:                                        ; preds = %free_old_val, %entry
  store ptr %call_method, ptr %ptr_f, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s8 = load ptr, ptr %s, align 8
  %ptr_p = getelementptr inbounds %MLP, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_p10 = getelementptr inbounds %MLP, ptr %s9, i32 0, i32 1
  %p = load ptr, ptr %ptr_p10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_Linear_step(ptr %p, float %lr11)
  call void @tl_mem_register_struct(ptr %call_method12)
  %old_field_val13 = load ptr, ptr %ptr_p, align 8
  %cnt_free_diff14 = icmp ne ptr %old_field_val13, %call_method12
  br i1 %cnt_free_diff14, label %free_old_val15, label %skip_free16

free_old_val15:                                   ; preds = %skip_free
  br label %skip_free16

skip_free16:                                      ; preds = %free_old_val15, %skip_free
  store ptr %call_method12, ptr %ptr_p, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s17 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s17)
  %unreg_field_0 = getelementptr inbounds %MLP, ptr %s17, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_018 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 0
  %field_val19 = load ptr, ptr %unreg_field_018, align 8
  call void @tl_mem_unregister(ptr %field_val19)
  %unreg_field_1 = getelementptr inbounds %Linear, ptr %field_val, i32 0, i32 1
  %field_val20 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val20)
  %unreg_field_121 = getelementptr inbounds %MLP, ptr %s17, i32 0, i32 1
  %field_val22 = load ptr, ptr %unreg_field_121, align 8
  call void @tl_mem_unregister(ptr %field_val22)
  %unreg_field_023 = getelementptr inbounds %Linear, ptr %field_val22, i32 0, i32 0
  %field_val24 = load ptr, ptr %unreg_field_023, align 8
  call void @tl_mem_unregister(ptr %field_val24)
  %unreg_field_125 = getelementptr inbounds %Linear, ptr %field_val22, i32 0, i32 1
  %field_val26 = load ptr, ptr %unreg_field_125, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  call void @tl_mem_exit_scope()
  ret ptr %s17
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
  br label %skip_free

skip_free:                                        ; preds = %free_old_val, %entry
  store ptr %call_method, ptr %ptr_l1, align 8
  call void @tl_mem_unregister(ptr %call_method)
  %s8 = load ptr, ptr %s, align 8
  %ptr_a = getelementptr inbounds %Block, ptr %s8, i32 0, i32 1
  %s9 = load ptr, ptr %s, align 8
  %ptr_a10 = getelementptr inbounds %Block, ptr %s9, i32 0, i32 1
  %a = load ptr, ptr %ptr_a10, align 8
  %lr11 = load float, ptr %lr2, align 4
  %call_method12 = call ptr @tl_CausalSelfAttention_step(ptr %a, float %lr11)
  call void @tl_mem_register_struct(ptr %call_method12)
  %old_field_val13 = load ptr, ptr %ptr_a, align 8
  %cnt_free_diff14 = icmp ne ptr %old_field_val13, %call_method12
  br i1 %cnt_free_diff14, label %free_old_val15, label %skip_free16

free_old_val15:                                   ; preds = %skip_free
  br label %skip_free16

skip_free16:                                      ; preds = %free_old_val15, %skip_free
  store ptr %call_method12, ptr %ptr_a, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s17 = load ptr, ptr %s, align 8
  %ptr_l2 = getelementptr inbounds %Block, ptr %s17, i32 0, i32 2
  %s18 = load ptr, ptr %s, align 8
  %ptr_l219 = getelementptr inbounds %Block, ptr %s18, i32 0, i32 2
  %l2 = load ptr, ptr %ptr_l219, align 8
  %lr20 = load float, ptr %lr2, align 4
  %call_method21 = call ptr @tl_LayerNorm_step(ptr %l2, float %lr20)
  call void @tl_mem_register_struct(ptr %call_method21)
  %old_field_val22 = load ptr, ptr %ptr_l2, align 8
  %cnt_free_diff23 = icmp ne ptr %old_field_val22, %call_method21
  br i1 %cnt_free_diff23, label %free_old_val24, label %skip_free25

free_old_val24:                                   ; preds = %skip_free16
  br label %skip_free25

skip_free25:                                      ; preds = %free_old_val24, %skip_free16
  store ptr %call_method21, ptr %ptr_l2, align 8
  call void @tl_mem_unregister(ptr %call_method21)
  %s26 = load ptr, ptr %s, align 8
  %ptr_m = getelementptr inbounds %Block, ptr %s26, i32 0, i32 3
  %s27 = load ptr, ptr %s, align 8
  %ptr_m28 = getelementptr inbounds %Block, ptr %s27, i32 0, i32 3
  %m = load ptr, ptr %ptr_m28, align 8
  %lr29 = load float, ptr %lr2, align 4
  %call_method30 = call ptr @tl_MLP_step(ptr %m, float %lr29)
  call void @tl_mem_register_struct(ptr %call_method30)
  %old_field_val31 = load ptr, ptr %ptr_m, align 8
  %cnt_free_diff32 = icmp ne ptr %old_field_val31, %call_method30
  br i1 %cnt_free_diff32, label %free_old_val33, label %skip_free34

free_old_val33:                                   ; preds = %skip_free25
  br label %skip_free34

skip_free34:                                      ; preds = %free_old_val33, %skip_free25
  store ptr %call_method30, ptr %ptr_m, align 8
  call void @tl_mem_unregister(ptr %call_method30)
  %s35 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s35)
  %unreg_field_0 = getelementptr inbounds %Block, ptr %s35, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_036 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 0
  %field_val37 = load ptr, ptr %unreg_field_036, align 8
  call void @tl_mem_unregister(ptr %field_val37)
  %unreg_field_1 = getelementptr inbounds %LayerNorm, ptr %field_val, i32 0, i32 1
  %field_val38 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val38)
  %unreg_field_139 = getelementptr inbounds %Block, ptr %s35, i32 0, i32 1
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
  %unreg_field_2 = getelementptr inbounds %CausalSelfAttention, ptr %field_val40, i32 0, i32 2
  %field_val53 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val53)
  %unreg_field_054 = getelementptr inbounds %Linear, ptr %field_val53, i32 0, i32 0
  %field_val55 = load ptr, ptr %unreg_field_054, align 8
  call void @tl_mem_unregister(ptr %field_val55)
  %unreg_field_156 = getelementptr inbounds %Linear, ptr %field_val53, i32 0, i32 1
  %field_val57 = load ptr, ptr %unreg_field_156, align 8
  call void @tl_mem_unregister(ptr %field_val57)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val40, i32 0, i32 3
  %field_val58 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val58)
  %unreg_field_059 = getelementptr inbounds %Linear, ptr %field_val58, i32 0, i32 0
  %field_val60 = load ptr, ptr %unreg_field_059, align 8
  call void @tl_mem_unregister(ptr %field_val60)
  %unreg_field_161 = getelementptr inbounds %Linear, ptr %field_val58, i32 0, i32 1
  %field_val62 = load ptr, ptr %unreg_field_161, align 8
  call void @tl_mem_unregister(ptr %field_val62)
  %unreg_field_263 = getelementptr inbounds %Block, ptr %s35, i32 0, i32 2
  %field_val64 = load ptr, ptr %unreg_field_263, align 8
  call void @tl_mem_unregister(ptr %field_val64)
  %unreg_field_065 = getelementptr inbounds %LayerNorm, ptr %field_val64, i32 0, i32 0
  %field_val66 = load ptr, ptr %unreg_field_065, align 8
  call void @tl_mem_unregister(ptr %field_val66)
  %unreg_field_167 = getelementptr inbounds %LayerNorm, ptr %field_val64, i32 0, i32 1
  %field_val68 = load ptr, ptr %unreg_field_167, align 8
  call void @tl_mem_unregister(ptr %field_val68)
  %unreg_field_369 = getelementptr inbounds %Block, ptr %s35, i32 0, i32 3
  %field_val70 = load ptr, ptr %unreg_field_369, align 8
  call void @tl_mem_unregister(ptr %field_val70)
  %unreg_field_071 = getelementptr inbounds %MLP, ptr %field_val70, i32 0, i32 0
  %field_val72 = load ptr, ptr %unreg_field_071, align 8
  call void @tl_mem_unregister(ptr %field_val72)
  %unreg_field_073 = getelementptr inbounds %Linear, ptr %field_val72, i32 0, i32 0
  %field_val74 = load ptr, ptr %unreg_field_073, align 8
  call void @tl_mem_unregister(ptr %field_val74)
  %unreg_field_175 = getelementptr inbounds %Linear, ptr %field_val72, i32 0, i32 1
  %field_val76 = load ptr, ptr %unreg_field_175, align 8
  call void @tl_mem_unregister(ptr %field_val76)
  %unreg_field_177 = getelementptr inbounds %MLP, ptr %field_val70, i32 0, i32 1
  %field_val78 = load ptr, ptr %unreg_field_177, align 8
  call void @tl_mem_unregister(ptr %field_val78)
  %unreg_field_079 = getelementptr inbounds %Linear, ptr %field_val78, i32 0, i32 0
  %field_val80 = load ptr, ptr %unreg_field_079, align 8
  call void @tl_mem_unregister(ptr %field_val80)
  %unreg_field_181 = getelementptr inbounds %Linear, ptr %field_val78, i32 0, i32 1
  %field_val82 = load ptr, ptr %unreg_field_181, align 8
  call void @tl_mem_unregister(ptr %field_val82)
  call void @tl_mem_exit_scope()
  ret ptr %s35
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
  br label %skip_free16

skip_free16:                                      ; preds = %free_old_val15, %skip_free
  store ptr %call_method12, ptr %ptr_wp, align 8
  call void @tl_mem_unregister(ptr %call_method12)
  %s17 = load ptr, ptr %s, align 8
  %ptr_b1 = getelementptr inbounds %GPTHeavy, ptr %s17, i32 0, i32 2
  %s18 = load ptr, ptr %s, align 8
  %ptr_b119 = getelementptr inbounds %GPTHeavy, ptr %s18, i32 0, i32 2
  %b1 = load ptr, ptr %ptr_b119, align 8
  %lr20 = load float, ptr %lr2, align 4
  %call_method21 = call ptr @tl_Block_step(ptr %b1, float %lr20)
  call void @tl_mem_register_struct(ptr %call_method21)
  %old_field_val22 = load ptr, ptr %ptr_b1, align 8
  %cnt_free_diff23 = icmp ne ptr %old_field_val22, %call_method21
  br i1 %cnt_free_diff23, label %free_old_val24, label %skip_free25

free_old_val24:                                   ; preds = %skip_free16
  br label %skip_free25

skip_free25:                                      ; preds = %free_old_val24, %skip_free16
  store ptr %call_method21, ptr %ptr_b1, align 8
  call void @tl_mem_unregister(ptr %call_method21)
  %s26 = load ptr, ptr %s, align 8
  %ptr_b2 = getelementptr inbounds %GPTHeavy, ptr %s26, i32 0, i32 3
  %s27 = load ptr, ptr %s, align 8
  %ptr_b228 = getelementptr inbounds %GPTHeavy, ptr %s27, i32 0, i32 3
  %b2 = load ptr, ptr %ptr_b228, align 8
  %lr29 = load float, ptr %lr2, align 4
  %call_method30 = call ptr @tl_Block_step(ptr %b2, float %lr29)
  call void @tl_mem_register_struct(ptr %call_method30)
  %old_field_val31 = load ptr, ptr %ptr_b2, align 8
  %cnt_free_diff32 = icmp ne ptr %old_field_val31, %call_method30
  br i1 %cnt_free_diff32, label %free_old_val33, label %skip_free34

free_old_val33:                                   ; preds = %skip_free25
  br label %skip_free34

skip_free34:                                      ; preds = %free_old_val33, %skip_free25
  store ptr %call_method30, ptr %ptr_b2, align 8
  call void @tl_mem_unregister(ptr %call_method30)
  %s35 = load ptr, ptr %s, align 8
  %ptr_b3 = getelementptr inbounds %GPTHeavy, ptr %s35, i32 0, i32 4
  %s36 = load ptr, ptr %s, align 8
  %ptr_b337 = getelementptr inbounds %GPTHeavy, ptr %s36, i32 0, i32 4
  %b3 = load ptr, ptr %ptr_b337, align 8
  %lr38 = load float, ptr %lr2, align 4
  %call_method39 = call ptr @tl_Block_step(ptr %b3, float %lr38)
  call void @tl_mem_register_struct(ptr %call_method39)
  %old_field_val40 = load ptr, ptr %ptr_b3, align 8
  %cnt_free_diff41 = icmp ne ptr %old_field_val40, %call_method39
  br i1 %cnt_free_diff41, label %free_old_val42, label %skip_free43

free_old_val42:                                   ; preds = %skip_free34
  br label %skip_free43

skip_free43:                                      ; preds = %free_old_val42, %skip_free34
  store ptr %call_method39, ptr %ptr_b3, align 8
  call void @tl_mem_unregister(ptr %call_method39)
  %s44 = load ptr, ptr %s, align 8
  %ptr_b4 = getelementptr inbounds %GPTHeavy, ptr %s44, i32 0, i32 5
  %s45 = load ptr, ptr %s, align 8
  %ptr_b446 = getelementptr inbounds %GPTHeavy, ptr %s45, i32 0, i32 5
  %b4 = load ptr, ptr %ptr_b446, align 8
  %lr47 = load float, ptr %lr2, align 4
  %call_method48 = call ptr @tl_Block_step(ptr %b4, float %lr47)
  call void @tl_mem_register_struct(ptr %call_method48)
  %old_field_val49 = load ptr, ptr %ptr_b4, align 8
  %cnt_free_diff50 = icmp ne ptr %old_field_val49, %call_method48
  br i1 %cnt_free_diff50, label %free_old_val51, label %skip_free52

free_old_val51:                                   ; preds = %skip_free43
  br label %skip_free52

skip_free52:                                      ; preds = %free_old_val51, %skip_free43
  store ptr %call_method48, ptr %ptr_b4, align 8
  call void @tl_mem_unregister(ptr %call_method48)
  %s53 = load ptr, ptr %s, align 8
  %ptr_l = getelementptr inbounds %GPTHeavy, ptr %s53, i32 0, i32 6
  %s54 = load ptr, ptr %s, align 8
  %ptr_l55 = getelementptr inbounds %GPTHeavy, ptr %s54, i32 0, i32 6
  %l = load ptr, ptr %ptr_l55, align 8
  %lr56 = load float, ptr %lr2, align 4
  %call_method57 = call ptr @tl_LayerNorm_step(ptr %l, float %lr56)
  call void @tl_mem_register_struct(ptr %call_method57)
  %old_field_val58 = load ptr, ptr %ptr_l, align 8
  %cnt_free_diff59 = icmp ne ptr %old_field_val58, %call_method57
  br i1 %cnt_free_diff59, label %free_old_val60, label %skip_free61

free_old_val60:                                   ; preds = %skip_free52
  br label %skip_free61

skip_free61:                                      ; preds = %free_old_val60, %skip_free52
  store ptr %call_method57, ptr %ptr_l, align 8
  call void @tl_mem_unregister(ptr %call_method57)
  %s62 = load ptr, ptr %s, align 8
  %ptr_h = getelementptr inbounds %GPTHeavy, ptr %s62, i32 0, i32 7
  %s63 = load ptr, ptr %s, align 8
  %ptr_h64 = getelementptr inbounds %GPTHeavy, ptr %s63, i32 0, i32 7
  %h = load ptr, ptr %ptr_h64, align 8
  %lr65 = load float, ptr %lr2, align 4
  %call_method66 = call ptr @tl_Linear_step(ptr %h, float %lr65)
  call void @tl_mem_register_struct(ptr %call_method66)
  %old_field_val67 = load ptr, ptr %ptr_h, align 8
  %cnt_free_diff68 = icmp ne ptr %old_field_val67, %call_method66
  br i1 %cnt_free_diff68, label %free_old_val69, label %skip_free70

free_old_val69:                                   ; preds = %skip_free61
  br label %skip_free70

skip_free70:                                      ; preds = %free_old_val69, %skip_free61
  store ptr %call_method66, ptr %ptr_h, align 8
  call void @tl_mem_unregister(ptr %call_method66)
  %s71 = load ptr, ptr %s, align 8
  call void @tl_mem_unregister(ptr %s71)
  %unreg_field_0 = getelementptr inbounds %GPTHeavy, ptr %s71, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_072 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val73 = load ptr, ptr %unreg_field_072, align 8
  call void @tl_mem_unregister(ptr %field_val73)
  %unreg_field_1 = getelementptr inbounds %GPTHeavy, ptr %s71, i32 0, i32 1
  %field_val74 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val74)
  %unreg_field_075 = getelementptr inbounds %Embedding, ptr %field_val74, i32 0, i32 0
  %field_val76 = load ptr, ptr %unreg_field_075, align 8
  call void @tl_mem_unregister(ptr %field_val76)
  %unreg_field_2 = getelementptr inbounds %GPTHeavy, ptr %s71, i32 0, i32 2
  %field_val77 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val77)
  %unreg_field_078 = getelementptr inbounds %Block, ptr %field_val77, i32 0, i32 0
  %field_val79 = load ptr, ptr %unreg_field_078, align 8
  call void @tl_mem_unregister(ptr %field_val79)
  %unreg_field_080 = getelementptr inbounds %LayerNorm, ptr %field_val79, i32 0, i32 0
  %field_val81 = load ptr, ptr %unreg_field_080, align 8
  call void @tl_mem_unregister(ptr %field_val81)
  %unreg_field_182 = getelementptr inbounds %LayerNorm, ptr %field_val79, i32 0, i32 1
  %field_val83 = load ptr, ptr %unreg_field_182, align 8
  call void @tl_mem_unregister(ptr %field_val83)
  %unreg_field_184 = getelementptr inbounds %Block, ptr %field_val77, i32 0, i32 1
  %field_val85 = load ptr, ptr %unreg_field_184, align 8
  call void @tl_mem_unregister(ptr %field_val85)
  %unreg_field_086 = getelementptr inbounds %CausalSelfAttention, ptr %field_val85, i32 0, i32 0
  %field_val87 = load ptr, ptr %unreg_field_086, align 8
  call void @tl_mem_unregister(ptr %field_val87)
  %unreg_field_088 = getelementptr inbounds %Linear, ptr %field_val87, i32 0, i32 0
  %field_val89 = load ptr, ptr %unreg_field_088, align 8
  call void @tl_mem_unregister(ptr %field_val89)
  %unreg_field_190 = getelementptr inbounds %Linear, ptr %field_val87, i32 0, i32 1
  %field_val91 = load ptr, ptr %unreg_field_190, align 8
  call void @tl_mem_unregister(ptr %field_val91)
  %unreg_field_192 = getelementptr inbounds %CausalSelfAttention, ptr %field_val85, i32 0, i32 1
  %field_val93 = load ptr, ptr %unreg_field_192, align 8
  call void @tl_mem_unregister(ptr %field_val93)
  %unreg_field_094 = getelementptr inbounds %Linear, ptr %field_val93, i32 0, i32 0
  %field_val95 = load ptr, ptr %unreg_field_094, align 8
  call void @tl_mem_unregister(ptr %field_val95)
  %unreg_field_196 = getelementptr inbounds %Linear, ptr %field_val93, i32 0, i32 1
  %field_val97 = load ptr, ptr %unreg_field_196, align 8
  call void @tl_mem_unregister(ptr %field_val97)
  %unreg_field_298 = getelementptr inbounds %CausalSelfAttention, ptr %field_val85, i32 0, i32 2
  %field_val99 = load ptr, ptr %unreg_field_298, align 8
  call void @tl_mem_unregister(ptr %field_val99)
  %unreg_field_0100 = getelementptr inbounds %Linear, ptr %field_val99, i32 0, i32 0
  %field_val101 = load ptr, ptr %unreg_field_0100, align 8
  call void @tl_mem_unregister(ptr %field_val101)
  %unreg_field_1102 = getelementptr inbounds %Linear, ptr %field_val99, i32 0, i32 1
  %field_val103 = load ptr, ptr %unreg_field_1102, align 8
  call void @tl_mem_unregister(ptr %field_val103)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val85, i32 0, i32 3
  %field_val104 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val104)
  %unreg_field_0105 = getelementptr inbounds %Linear, ptr %field_val104, i32 0, i32 0
  %field_val106 = load ptr, ptr %unreg_field_0105, align 8
  call void @tl_mem_unregister(ptr %field_val106)
  %unreg_field_1107 = getelementptr inbounds %Linear, ptr %field_val104, i32 0, i32 1
  %field_val108 = load ptr, ptr %unreg_field_1107, align 8
  call void @tl_mem_unregister(ptr %field_val108)
  %unreg_field_2109 = getelementptr inbounds %Block, ptr %field_val77, i32 0, i32 2
  %field_val110 = load ptr, ptr %unreg_field_2109, align 8
  call void @tl_mem_unregister(ptr %field_val110)
  %unreg_field_0111 = getelementptr inbounds %LayerNorm, ptr %field_val110, i32 0, i32 0
  %field_val112 = load ptr, ptr %unreg_field_0111, align 8
  call void @tl_mem_unregister(ptr %field_val112)
  %unreg_field_1113 = getelementptr inbounds %LayerNorm, ptr %field_val110, i32 0, i32 1
  %field_val114 = load ptr, ptr %unreg_field_1113, align 8
  call void @tl_mem_unregister(ptr %field_val114)
  %unreg_field_3115 = getelementptr inbounds %Block, ptr %field_val77, i32 0, i32 3
  %field_val116 = load ptr, ptr %unreg_field_3115, align 8
  call void @tl_mem_unregister(ptr %field_val116)
  %unreg_field_0117 = getelementptr inbounds %MLP, ptr %field_val116, i32 0, i32 0
  %field_val118 = load ptr, ptr %unreg_field_0117, align 8
  call void @tl_mem_unregister(ptr %field_val118)
  %unreg_field_0119 = getelementptr inbounds %Linear, ptr %field_val118, i32 0, i32 0
  %field_val120 = load ptr, ptr %unreg_field_0119, align 8
  call void @tl_mem_unregister(ptr %field_val120)
  %unreg_field_1121 = getelementptr inbounds %Linear, ptr %field_val118, i32 0, i32 1
  %field_val122 = load ptr, ptr %unreg_field_1121, align 8
  call void @tl_mem_unregister(ptr %field_val122)
  %unreg_field_1123 = getelementptr inbounds %MLP, ptr %field_val116, i32 0, i32 1
  %field_val124 = load ptr, ptr %unreg_field_1123, align 8
  call void @tl_mem_unregister(ptr %field_val124)
  %unreg_field_0125 = getelementptr inbounds %Linear, ptr %field_val124, i32 0, i32 0
  %field_val126 = load ptr, ptr %unreg_field_0125, align 8
  call void @tl_mem_unregister(ptr %field_val126)
  %unreg_field_1127 = getelementptr inbounds %Linear, ptr %field_val124, i32 0, i32 1
  %field_val128 = load ptr, ptr %unreg_field_1127, align 8
  call void @tl_mem_unregister(ptr %field_val128)
  %unreg_field_3129 = getelementptr inbounds %GPTHeavy, ptr %s71, i32 0, i32 3
  %field_val130 = load ptr, ptr %unreg_field_3129, align 8
  call void @tl_mem_unregister(ptr %field_val130)
  %unreg_field_0131 = getelementptr inbounds %Block, ptr %field_val130, i32 0, i32 0
  %field_val132 = load ptr, ptr %unreg_field_0131, align 8
  call void @tl_mem_unregister(ptr %field_val132)
  %unreg_field_0133 = getelementptr inbounds %LayerNorm, ptr %field_val132, i32 0, i32 0
  %field_val134 = load ptr, ptr %unreg_field_0133, align 8
  call void @tl_mem_unregister(ptr %field_val134)
  %unreg_field_1135 = getelementptr inbounds %LayerNorm, ptr %field_val132, i32 0, i32 1
  %field_val136 = load ptr, ptr %unreg_field_1135, align 8
  call void @tl_mem_unregister(ptr %field_val136)
  %unreg_field_1137 = getelementptr inbounds %Block, ptr %field_val130, i32 0, i32 1
  %field_val138 = load ptr, ptr %unreg_field_1137, align 8
  call void @tl_mem_unregister(ptr %field_val138)
  %unreg_field_0139 = getelementptr inbounds %CausalSelfAttention, ptr %field_val138, i32 0, i32 0
  %field_val140 = load ptr, ptr %unreg_field_0139, align 8
  call void @tl_mem_unregister(ptr %field_val140)
  %unreg_field_0141 = getelementptr inbounds %Linear, ptr %field_val140, i32 0, i32 0
  %field_val142 = load ptr, ptr %unreg_field_0141, align 8
  call void @tl_mem_unregister(ptr %field_val142)
  %unreg_field_1143 = getelementptr inbounds %Linear, ptr %field_val140, i32 0, i32 1
  %field_val144 = load ptr, ptr %unreg_field_1143, align 8
  call void @tl_mem_unregister(ptr %field_val144)
  %unreg_field_1145 = getelementptr inbounds %CausalSelfAttention, ptr %field_val138, i32 0, i32 1
  %field_val146 = load ptr, ptr %unreg_field_1145, align 8
  call void @tl_mem_unregister(ptr %field_val146)
  %unreg_field_0147 = getelementptr inbounds %Linear, ptr %field_val146, i32 0, i32 0
  %field_val148 = load ptr, ptr %unreg_field_0147, align 8
  call void @tl_mem_unregister(ptr %field_val148)
  %unreg_field_1149 = getelementptr inbounds %Linear, ptr %field_val146, i32 0, i32 1
  %field_val150 = load ptr, ptr %unreg_field_1149, align 8
  call void @tl_mem_unregister(ptr %field_val150)
  %unreg_field_2151 = getelementptr inbounds %CausalSelfAttention, ptr %field_val138, i32 0, i32 2
  %field_val152 = load ptr, ptr %unreg_field_2151, align 8
  call void @tl_mem_unregister(ptr %field_val152)
  %unreg_field_0153 = getelementptr inbounds %Linear, ptr %field_val152, i32 0, i32 0
  %field_val154 = load ptr, ptr %unreg_field_0153, align 8
  call void @tl_mem_unregister(ptr %field_val154)
  %unreg_field_1155 = getelementptr inbounds %Linear, ptr %field_val152, i32 0, i32 1
  %field_val156 = load ptr, ptr %unreg_field_1155, align 8
  call void @tl_mem_unregister(ptr %field_val156)
  %unreg_field_3157 = getelementptr inbounds %CausalSelfAttention, ptr %field_val138, i32 0, i32 3
  %field_val158 = load ptr, ptr %unreg_field_3157, align 8
  call void @tl_mem_unregister(ptr %field_val158)
  %unreg_field_0159 = getelementptr inbounds %Linear, ptr %field_val158, i32 0, i32 0
  %field_val160 = load ptr, ptr %unreg_field_0159, align 8
  call void @tl_mem_unregister(ptr %field_val160)
  %unreg_field_1161 = getelementptr inbounds %Linear, ptr %field_val158, i32 0, i32 1
  %field_val162 = load ptr, ptr %unreg_field_1161, align 8
  call void @tl_mem_unregister(ptr %field_val162)
  %unreg_field_2163 = getelementptr inbounds %Block, ptr %field_val130, i32 0, i32 2
  %field_val164 = load ptr, ptr %unreg_field_2163, align 8
  call void @tl_mem_unregister(ptr %field_val164)
  %unreg_field_0165 = getelementptr inbounds %LayerNorm, ptr %field_val164, i32 0, i32 0
  %field_val166 = load ptr, ptr %unreg_field_0165, align 8
  call void @tl_mem_unregister(ptr %field_val166)
  %unreg_field_1167 = getelementptr inbounds %LayerNorm, ptr %field_val164, i32 0, i32 1
  %field_val168 = load ptr, ptr %unreg_field_1167, align 8
  call void @tl_mem_unregister(ptr %field_val168)
  %unreg_field_3169 = getelementptr inbounds %Block, ptr %field_val130, i32 0, i32 3
  %field_val170 = load ptr, ptr %unreg_field_3169, align 8
  call void @tl_mem_unregister(ptr %field_val170)
  %unreg_field_0171 = getelementptr inbounds %MLP, ptr %field_val170, i32 0, i32 0
  %field_val172 = load ptr, ptr %unreg_field_0171, align 8
  call void @tl_mem_unregister(ptr %field_val172)
  %unreg_field_0173 = getelementptr inbounds %Linear, ptr %field_val172, i32 0, i32 0
  %field_val174 = load ptr, ptr %unreg_field_0173, align 8
  call void @tl_mem_unregister(ptr %field_val174)
  %unreg_field_1175 = getelementptr inbounds %Linear, ptr %field_val172, i32 0, i32 1
  %field_val176 = load ptr, ptr %unreg_field_1175, align 8
  call void @tl_mem_unregister(ptr %field_val176)
  %unreg_field_1177 = getelementptr inbounds %MLP, ptr %field_val170, i32 0, i32 1
  %field_val178 = load ptr, ptr %unreg_field_1177, align 8
  call void @tl_mem_unregister(ptr %field_val178)
  %unreg_field_0179 = getelementptr inbounds %Linear, ptr %field_val178, i32 0, i32 0
  %field_val180 = load ptr, ptr %unreg_field_0179, align 8
  call void @tl_mem_unregister(ptr %field_val180)
  %unreg_field_1181 = getelementptr inbounds %Linear, ptr %field_val178, i32 0, i32 1
  %field_val182 = load ptr, ptr %unreg_field_1181, align 8
  call void @tl_mem_unregister(ptr %field_val182)
  %unreg_field_4 = getelementptr inbounds %GPTHeavy, ptr %s71, i32 0, i32 4
  %field_val183 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val183)
  %unreg_field_0184 = getelementptr inbounds %Block, ptr %field_val183, i32 0, i32 0
  %field_val185 = load ptr, ptr %unreg_field_0184, align 8
  call void @tl_mem_unregister(ptr %field_val185)
  %unreg_field_0186 = getelementptr inbounds %LayerNorm, ptr %field_val185, i32 0, i32 0
  %field_val187 = load ptr, ptr %unreg_field_0186, align 8
  call void @tl_mem_unregister(ptr %field_val187)
  %unreg_field_1188 = getelementptr inbounds %LayerNorm, ptr %field_val185, i32 0, i32 1
  %field_val189 = load ptr, ptr %unreg_field_1188, align 8
  call void @tl_mem_unregister(ptr %field_val189)
  %unreg_field_1190 = getelementptr inbounds %Block, ptr %field_val183, i32 0, i32 1
  %field_val191 = load ptr, ptr %unreg_field_1190, align 8
  call void @tl_mem_unregister(ptr %field_val191)
  %unreg_field_0192 = getelementptr inbounds %CausalSelfAttention, ptr %field_val191, i32 0, i32 0
  %field_val193 = load ptr, ptr %unreg_field_0192, align 8
  call void @tl_mem_unregister(ptr %field_val193)
  %unreg_field_0194 = getelementptr inbounds %Linear, ptr %field_val193, i32 0, i32 0
  %field_val195 = load ptr, ptr %unreg_field_0194, align 8
  call void @tl_mem_unregister(ptr %field_val195)
  %unreg_field_1196 = getelementptr inbounds %Linear, ptr %field_val193, i32 0, i32 1
  %field_val197 = load ptr, ptr %unreg_field_1196, align 8
  call void @tl_mem_unregister(ptr %field_val197)
  %unreg_field_1198 = getelementptr inbounds %CausalSelfAttention, ptr %field_val191, i32 0, i32 1
  %field_val199 = load ptr, ptr %unreg_field_1198, align 8
  call void @tl_mem_unregister(ptr %field_val199)
  %unreg_field_0200 = getelementptr inbounds %Linear, ptr %field_val199, i32 0, i32 0
  %field_val201 = load ptr, ptr %unreg_field_0200, align 8
  call void @tl_mem_unregister(ptr %field_val201)
  %unreg_field_1202 = getelementptr inbounds %Linear, ptr %field_val199, i32 0, i32 1
  %field_val203 = load ptr, ptr %unreg_field_1202, align 8
  call void @tl_mem_unregister(ptr %field_val203)
  %unreg_field_2204 = getelementptr inbounds %CausalSelfAttention, ptr %field_val191, i32 0, i32 2
  %field_val205 = load ptr, ptr %unreg_field_2204, align 8
  call void @tl_mem_unregister(ptr %field_val205)
  %unreg_field_0206 = getelementptr inbounds %Linear, ptr %field_val205, i32 0, i32 0
  %field_val207 = load ptr, ptr %unreg_field_0206, align 8
  call void @tl_mem_unregister(ptr %field_val207)
  %unreg_field_1208 = getelementptr inbounds %Linear, ptr %field_val205, i32 0, i32 1
  %field_val209 = load ptr, ptr %unreg_field_1208, align 8
  call void @tl_mem_unregister(ptr %field_val209)
  %unreg_field_3210 = getelementptr inbounds %CausalSelfAttention, ptr %field_val191, i32 0, i32 3
  %field_val211 = load ptr, ptr %unreg_field_3210, align 8
  call void @tl_mem_unregister(ptr %field_val211)
  %unreg_field_0212 = getelementptr inbounds %Linear, ptr %field_val211, i32 0, i32 0
  %field_val213 = load ptr, ptr %unreg_field_0212, align 8
  call void @tl_mem_unregister(ptr %field_val213)
  %unreg_field_1214 = getelementptr inbounds %Linear, ptr %field_val211, i32 0, i32 1
  %field_val215 = load ptr, ptr %unreg_field_1214, align 8
  call void @tl_mem_unregister(ptr %field_val215)
  %unreg_field_2216 = getelementptr inbounds %Block, ptr %field_val183, i32 0, i32 2
  %field_val217 = load ptr, ptr %unreg_field_2216, align 8
  call void @tl_mem_unregister(ptr %field_val217)
  %unreg_field_0218 = getelementptr inbounds %LayerNorm, ptr %field_val217, i32 0, i32 0
  %field_val219 = load ptr, ptr %unreg_field_0218, align 8
  call void @tl_mem_unregister(ptr %field_val219)
  %unreg_field_1220 = getelementptr inbounds %LayerNorm, ptr %field_val217, i32 0, i32 1
  %field_val221 = load ptr, ptr %unreg_field_1220, align 8
  call void @tl_mem_unregister(ptr %field_val221)
  %unreg_field_3222 = getelementptr inbounds %Block, ptr %field_val183, i32 0, i32 3
  %field_val223 = load ptr, ptr %unreg_field_3222, align 8
  call void @tl_mem_unregister(ptr %field_val223)
  %unreg_field_0224 = getelementptr inbounds %MLP, ptr %field_val223, i32 0, i32 0
  %field_val225 = load ptr, ptr %unreg_field_0224, align 8
  call void @tl_mem_unregister(ptr %field_val225)
  %unreg_field_0226 = getelementptr inbounds %Linear, ptr %field_val225, i32 0, i32 0
  %field_val227 = load ptr, ptr %unreg_field_0226, align 8
  call void @tl_mem_unregister(ptr %field_val227)
  %unreg_field_1228 = getelementptr inbounds %Linear, ptr %field_val225, i32 0, i32 1
  %field_val229 = load ptr, ptr %unreg_field_1228, align 8
  call void @tl_mem_unregister(ptr %field_val229)
  %unreg_field_1230 = getelementptr inbounds %MLP, ptr %field_val223, i32 0, i32 1
  %field_val231 = load ptr, ptr %unreg_field_1230, align 8
  call void @tl_mem_unregister(ptr %field_val231)
  %unreg_field_0232 = getelementptr inbounds %Linear, ptr %field_val231, i32 0, i32 0
  %field_val233 = load ptr, ptr %unreg_field_0232, align 8
  call void @tl_mem_unregister(ptr %field_val233)
  %unreg_field_1234 = getelementptr inbounds %Linear, ptr %field_val231, i32 0, i32 1
  %field_val235 = load ptr, ptr %unreg_field_1234, align 8
  call void @tl_mem_unregister(ptr %field_val235)
  %unreg_field_5 = getelementptr inbounds %GPTHeavy, ptr %s71, i32 0, i32 5
  %field_val236 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val236)
  %unreg_field_0237 = getelementptr inbounds %Block, ptr %field_val236, i32 0, i32 0
  %field_val238 = load ptr, ptr %unreg_field_0237, align 8
  call void @tl_mem_unregister(ptr %field_val238)
  %unreg_field_0239 = getelementptr inbounds %LayerNorm, ptr %field_val238, i32 0, i32 0
  %field_val240 = load ptr, ptr %unreg_field_0239, align 8
  call void @tl_mem_unregister(ptr %field_val240)
  %unreg_field_1241 = getelementptr inbounds %LayerNorm, ptr %field_val238, i32 0, i32 1
  %field_val242 = load ptr, ptr %unreg_field_1241, align 8
  call void @tl_mem_unregister(ptr %field_val242)
  %unreg_field_1243 = getelementptr inbounds %Block, ptr %field_val236, i32 0, i32 1
  %field_val244 = load ptr, ptr %unreg_field_1243, align 8
  call void @tl_mem_unregister(ptr %field_val244)
  %unreg_field_0245 = getelementptr inbounds %CausalSelfAttention, ptr %field_val244, i32 0, i32 0
  %field_val246 = load ptr, ptr %unreg_field_0245, align 8
  call void @tl_mem_unregister(ptr %field_val246)
  %unreg_field_0247 = getelementptr inbounds %Linear, ptr %field_val246, i32 0, i32 0
  %field_val248 = load ptr, ptr %unreg_field_0247, align 8
  call void @tl_mem_unregister(ptr %field_val248)
  %unreg_field_1249 = getelementptr inbounds %Linear, ptr %field_val246, i32 0, i32 1
  %field_val250 = load ptr, ptr %unreg_field_1249, align 8
  call void @tl_mem_unregister(ptr %field_val250)
  %unreg_field_1251 = getelementptr inbounds %CausalSelfAttention, ptr %field_val244, i32 0, i32 1
  %field_val252 = load ptr, ptr %unreg_field_1251, align 8
  call void @tl_mem_unregister(ptr %field_val252)
  %unreg_field_0253 = getelementptr inbounds %Linear, ptr %field_val252, i32 0, i32 0
  %field_val254 = load ptr, ptr %unreg_field_0253, align 8
  call void @tl_mem_unregister(ptr %field_val254)
  %unreg_field_1255 = getelementptr inbounds %Linear, ptr %field_val252, i32 0, i32 1
  %field_val256 = load ptr, ptr %unreg_field_1255, align 8
  call void @tl_mem_unregister(ptr %field_val256)
  %unreg_field_2257 = getelementptr inbounds %CausalSelfAttention, ptr %field_val244, i32 0, i32 2
  %field_val258 = load ptr, ptr %unreg_field_2257, align 8
  call void @tl_mem_unregister(ptr %field_val258)
  %unreg_field_0259 = getelementptr inbounds %Linear, ptr %field_val258, i32 0, i32 0
  %field_val260 = load ptr, ptr %unreg_field_0259, align 8
  call void @tl_mem_unregister(ptr %field_val260)
  %unreg_field_1261 = getelementptr inbounds %Linear, ptr %field_val258, i32 0, i32 1
  %field_val262 = load ptr, ptr %unreg_field_1261, align 8
  call void @tl_mem_unregister(ptr %field_val262)
  %unreg_field_3263 = getelementptr inbounds %CausalSelfAttention, ptr %field_val244, i32 0, i32 3
  %field_val264 = load ptr, ptr %unreg_field_3263, align 8
  call void @tl_mem_unregister(ptr %field_val264)
  %unreg_field_0265 = getelementptr inbounds %Linear, ptr %field_val264, i32 0, i32 0
  %field_val266 = load ptr, ptr %unreg_field_0265, align 8
  call void @tl_mem_unregister(ptr %field_val266)
  %unreg_field_1267 = getelementptr inbounds %Linear, ptr %field_val264, i32 0, i32 1
  %field_val268 = load ptr, ptr %unreg_field_1267, align 8
  call void @tl_mem_unregister(ptr %field_val268)
  %unreg_field_2269 = getelementptr inbounds %Block, ptr %field_val236, i32 0, i32 2
  %field_val270 = load ptr, ptr %unreg_field_2269, align 8
  call void @tl_mem_unregister(ptr %field_val270)
  %unreg_field_0271 = getelementptr inbounds %LayerNorm, ptr %field_val270, i32 0, i32 0
  %field_val272 = load ptr, ptr %unreg_field_0271, align 8
  call void @tl_mem_unregister(ptr %field_val272)
  %unreg_field_1273 = getelementptr inbounds %LayerNorm, ptr %field_val270, i32 0, i32 1
  %field_val274 = load ptr, ptr %unreg_field_1273, align 8
  call void @tl_mem_unregister(ptr %field_val274)
  %unreg_field_3275 = getelementptr inbounds %Block, ptr %field_val236, i32 0, i32 3
  %field_val276 = load ptr, ptr %unreg_field_3275, align 8
  call void @tl_mem_unregister(ptr %field_val276)
  %unreg_field_0277 = getelementptr inbounds %MLP, ptr %field_val276, i32 0, i32 0
  %field_val278 = load ptr, ptr %unreg_field_0277, align 8
  call void @tl_mem_unregister(ptr %field_val278)
  %unreg_field_0279 = getelementptr inbounds %Linear, ptr %field_val278, i32 0, i32 0
  %field_val280 = load ptr, ptr %unreg_field_0279, align 8
  call void @tl_mem_unregister(ptr %field_val280)
  %unreg_field_1281 = getelementptr inbounds %Linear, ptr %field_val278, i32 0, i32 1
  %field_val282 = load ptr, ptr %unreg_field_1281, align 8
  call void @tl_mem_unregister(ptr %field_val282)
  %unreg_field_1283 = getelementptr inbounds %MLP, ptr %field_val276, i32 0, i32 1
  %field_val284 = load ptr, ptr %unreg_field_1283, align 8
  call void @tl_mem_unregister(ptr %field_val284)
  %unreg_field_0285 = getelementptr inbounds %Linear, ptr %field_val284, i32 0, i32 0
  %field_val286 = load ptr, ptr %unreg_field_0285, align 8
  call void @tl_mem_unregister(ptr %field_val286)
  %unreg_field_1287 = getelementptr inbounds %Linear, ptr %field_val284, i32 0, i32 1
  %field_val288 = load ptr, ptr %unreg_field_1287, align 8
  call void @tl_mem_unregister(ptr %field_val288)
  %unreg_field_6 = getelementptr inbounds %GPTHeavy, ptr %s71, i32 0, i32 6
  %field_val289 = load ptr, ptr %unreg_field_6, align 8
  call void @tl_mem_unregister(ptr %field_val289)
  %unreg_field_0290 = getelementptr inbounds %LayerNorm, ptr %field_val289, i32 0, i32 0
  %field_val291 = load ptr, ptr %unreg_field_0290, align 8
  call void @tl_mem_unregister(ptr %field_val291)
  %unreg_field_1292 = getelementptr inbounds %LayerNorm, ptr %field_val289, i32 0, i32 1
  %field_val293 = load ptr, ptr %unreg_field_1292, align 8
  call void @tl_mem_unregister(ptr %field_val293)
  %unreg_field_7 = getelementptr inbounds %GPTHeavy, ptr %s71, i32 0, i32 7
  %field_val294 = load ptr, ptr %unreg_field_7, align 8
  call void @tl_mem_unregister(ptr %field_val294)
  %unreg_field_0295 = getelementptr inbounds %Linear, ptr %field_val294, i32 0, i32 0
  %field_val296 = load ptr, ptr %unreg_field_0295, align 8
  call void @tl_mem_unregister(ptr %field_val296)
  %unreg_field_1297 = getelementptr inbounds %Linear, ptr %field_val294, i32 0, i32 1
  %field_val298 = load ptr, ptr %unreg_field_1297, align 8
  call void @tl_mem_unregister(ptr %field_val298)
  call void @tl_mem_exit_scope()
  ret ptr %s71
}

define ptr @train_step(ptr %model, float %lr, i64 %i, i64 %j) {
entry:
  %_discard_0 = alloca ptr, align 16
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
  store ptr %call_method157, ptr %_discard_0, align 8
  %model158 = load ptr, ptr %model1, align 8
  call void @tl_mem_unregister(ptr %model158)
  %unreg_field_0 = getelementptr inbounds %GPTHeavy, ptr %model158, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_0159 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val160 = load ptr, ptr %unreg_field_0159, align 8
  call void @tl_mem_unregister(ptr %field_val160)
  %unreg_field_1 = getelementptr inbounds %GPTHeavy, ptr %model158, i32 0, i32 1
  %field_val161 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val161)
  %unreg_field_0162 = getelementptr inbounds %Embedding, ptr %field_val161, i32 0, i32 0
  %field_val163 = load ptr, ptr %unreg_field_0162, align 8
  call void @tl_mem_unregister(ptr %field_val163)
  %unreg_field_2 = getelementptr inbounds %GPTHeavy, ptr %model158, i32 0, i32 2
  %field_val164 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val164)
  %unreg_field_0165 = getelementptr inbounds %Block, ptr %field_val164, i32 0, i32 0
  %field_val166 = load ptr, ptr %unreg_field_0165, align 8
  call void @tl_mem_unregister(ptr %field_val166)
  %unreg_field_0167 = getelementptr inbounds %LayerNorm, ptr %field_val166, i32 0, i32 0
  %field_val168 = load ptr, ptr %unreg_field_0167, align 8
  call void @tl_mem_unregister(ptr %field_val168)
  %unreg_field_1169 = getelementptr inbounds %LayerNorm, ptr %field_val166, i32 0, i32 1
  %field_val170 = load ptr, ptr %unreg_field_1169, align 8
  call void @tl_mem_unregister(ptr %field_val170)
  %unreg_field_1171 = getelementptr inbounds %Block, ptr %field_val164, i32 0, i32 1
  %field_val172 = load ptr, ptr %unreg_field_1171, align 8
  call void @tl_mem_unregister(ptr %field_val172)
  %unreg_field_0173 = getelementptr inbounds %CausalSelfAttention, ptr %field_val172, i32 0, i32 0
  %field_val174 = load ptr, ptr %unreg_field_0173, align 8
  call void @tl_mem_unregister(ptr %field_val174)
  %unreg_field_0175 = getelementptr inbounds %Linear, ptr %field_val174, i32 0, i32 0
  %field_val176 = load ptr, ptr %unreg_field_0175, align 8
  call void @tl_mem_unregister(ptr %field_val176)
  %unreg_field_1177 = getelementptr inbounds %Linear, ptr %field_val174, i32 0, i32 1
  %field_val178 = load ptr, ptr %unreg_field_1177, align 8
  call void @tl_mem_unregister(ptr %field_val178)
  %unreg_field_1179 = getelementptr inbounds %CausalSelfAttention, ptr %field_val172, i32 0, i32 1
  %field_val180 = load ptr, ptr %unreg_field_1179, align 8
  call void @tl_mem_unregister(ptr %field_val180)
  %unreg_field_0181 = getelementptr inbounds %Linear, ptr %field_val180, i32 0, i32 0
  %field_val182 = load ptr, ptr %unreg_field_0181, align 8
  call void @tl_mem_unregister(ptr %field_val182)
  %unreg_field_1183 = getelementptr inbounds %Linear, ptr %field_val180, i32 0, i32 1
  %field_val184 = load ptr, ptr %unreg_field_1183, align 8
  call void @tl_mem_unregister(ptr %field_val184)
  %unreg_field_2185 = getelementptr inbounds %CausalSelfAttention, ptr %field_val172, i32 0, i32 2
  %field_val186 = load ptr, ptr %unreg_field_2185, align 8
  call void @tl_mem_unregister(ptr %field_val186)
  %unreg_field_0187 = getelementptr inbounds %Linear, ptr %field_val186, i32 0, i32 0
  %field_val188 = load ptr, ptr %unreg_field_0187, align 8
  call void @tl_mem_unregister(ptr %field_val188)
  %unreg_field_1189 = getelementptr inbounds %Linear, ptr %field_val186, i32 0, i32 1
  %field_val190 = load ptr, ptr %unreg_field_1189, align 8
  call void @tl_mem_unregister(ptr %field_val190)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val172, i32 0, i32 3
  %field_val191 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val191)
  %unreg_field_0192 = getelementptr inbounds %Linear, ptr %field_val191, i32 0, i32 0
  %field_val193 = load ptr, ptr %unreg_field_0192, align 8
  call void @tl_mem_unregister(ptr %field_val193)
  %unreg_field_1194 = getelementptr inbounds %Linear, ptr %field_val191, i32 0, i32 1
  %field_val195 = load ptr, ptr %unreg_field_1194, align 8
  call void @tl_mem_unregister(ptr %field_val195)
  %unreg_field_2196 = getelementptr inbounds %Block, ptr %field_val164, i32 0, i32 2
  %field_val197 = load ptr, ptr %unreg_field_2196, align 8
  call void @tl_mem_unregister(ptr %field_val197)
  %unreg_field_0198 = getelementptr inbounds %LayerNorm, ptr %field_val197, i32 0, i32 0
  %field_val199 = load ptr, ptr %unreg_field_0198, align 8
  call void @tl_mem_unregister(ptr %field_val199)
  %unreg_field_1200 = getelementptr inbounds %LayerNorm, ptr %field_val197, i32 0, i32 1
  %field_val201 = load ptr, ptr %unreg_field_1200, align 8
  call void @tl_mem_unregister(ptr %field_val201)
  %unreg_field_3202 = getelementptr inbounds %Block, ptr %field_val164, i32 0, i32 3
  %field_val203 = load ptr, ptr %unreg_field_3202, align 8
  call void @tl_mem_unregister(ptr %field_val203)
  %unreg_field_0204 = getelementptr inbounds %MLP, ptr %field_val203, i32 0, i32 0
  %field_val205 = load ptr, ptr %unreg_field_0204, align 8
  call void @tl_mem_unregister(ptr %field_val205)
  %unreg_field_0206 = getelementptr inbounds %Linear, ptr %field_val205, i32 0, i32 0
  %field_val207 = load ptr, ptr %unreg_field_0206, align 8
  call void @tl_mem_unregister(ptr %field_val207)
  %unreg_field_1208 = getelementptr inbounds %Linear, ptr %field_val205, i32 0, i32 1
  %field_val209 = load ptr, ptr %unreg_field_1208, align 8
  call void @tl_mem_unregister(ptr %field_val209)
  %unreg_field_1210 = getelementptr inbounds %MLP, ptr %field_val203, i32 0, i32 1
  %field_val211 = load ptr, ptr %unreg_field_1210, align 8
  call void @tl_mem_unregister(ptr %field_val211)
  %unreg_field_0212 = getelementptr inbounds %Linear, ptr %field_val211, i32 0, i32 0
  %field_val213 = load ptr, ptr %unreg_field_0212, align 8
  call void @tl_mem_unregister(ptr %field_val213)
  %unreg_field_1214 = getelementptr inbounds %Linear, ptr %field_val211, i32 0, i32 1
  %field_val215 = load ptr, ptr %unreg_field_1214, align 8
  call void @tl_mem_unregister(ptr %field_val215)
  %unreg_field_3216 = getelementptr inbounds %GPTHeavy, ptr %model158, i32 0, i32 3
  %field_val217 = load ptr, ptr %unreg_field_3216, align 8
  call void @tl_mem_unregister(ptr %field_val217)
  %unreg_field_0218 = getelementptr inbounds %Block, ptr %field_val217, i32 0, i32 0
  %field_val219 = load ptr, ptr %unreg_field_0218, align 8
  call void @tl_mem_unregister(ptr %field_val219)
  %unreg_field_0220 = getelementptr inbounds %LayerNorm, ptr %field_val219, i32 0, i32 0
  %field_val221 = load ptr, ptr %unreg_field_0220, align 8
  call void @tl_mem_unregister(ptr %field_val221)
  %unreg_field_1222 = getelementptr inbounds %LayerNorm, ptr %field_val219, i32 0, i32 1
  %field_val223 = load ptr, ptr %unreg_field_1222, align 8
  call void @tl_mem_unregister(ptr %field_val223)
  %unreg_field_1224 = getelementptr inbounds %Block, ptr %field_val217, i32 0, i32 1
  %field_val225 = load ptr, ptr %unreg_field_1224, align 8
  call void @tl_mem_unregister(ptr %field_val225)
  %unreg_field_0226 = getelementptr inbounds %CausalSelfAttention, ptr %field_val225, i32 0, i32 0
  %field_val227 = load ptr, ptr %unreg_field_0226, align 8
  call void @tl_mem_unregister(ptr %field_val227)
  %unreg_field_0228 = getelementptr inbounds %Linear, ptr %field_val227, i32 0, i32 0
  %field_val229 = load ptr, ptr %unreg_field_0228, align 8
  call void @tl_mem_unregister(ptr %field_val229)
  %unreg_field_1230 = getelementptr inbounds %Linear, ptr %field_val227, i32 0, i32 1
  %field_val231 = load ptr, ptr %unreg_field_1230, align 8
  call void @tl_mem_unregister(ptr %field_val231)
  %unreg_field_1232 = getelementptr inbounds %CausalSelfAttention, ptr %field_val225, i32 0, i32 1
  %field_val233 = load ptr, ptr %unreg_field_1232, align 8
  call void @tl_mem_unregister(ptr %field_val233)
  %unreg_field_0234 = getelementptr inbounds %Linear, ptr %field_val233, i32 0, i32 0
  %field_val235 = load ptr, ptr %unreg_field_0234, align 8
  call void @tl_mem_unregister(ptr %field_val235)
  %unreg_field_1236 = getelementptr inbounds %Linear, ptr %field_val233, i32 0, i32 1
  %field_val237 = load ptr, ptr %unreg_field_1236, align 8
  call void @tl_mem_unregister(ptr %field_val237)
  %unreg_field_2238 = getelementptr inbounds %CausalSelfAttention, ptr %field_val225, i32 0, i32 2
  %field_val239 = load ptr, ptr %unreg_field_2238, align 8
  call void @tl_mem_unregister(ptr %field_val239)
  %unreg_field_0240 = getelementptr inbounds %Linear, ptr %field_val239, i32 0, i32 0
  %field_val241 = load ptr, ptr %unreg_field_0240, align 8
  call void @tl_mem_unregister(ptr %field_val241)
  %unreg_field_1242 = getelementptr inbounds %Linear, ptr %field_val239, i32 0, i32 1
  %field_val243 = load ptr, ptr %unreg_field_1242, align 8
  call void @tl_mem_unregister(ptr %field_val243)
  %unreg_field_3244 = getelementptr inbounds %CausalSelfAttention, ptr %field_val225, i32 0, i32 3
  %field_val245 = load ptr, ptr %unreg_field_3244, align 8
  call void @tl_mem_unregister(ptr %field_val245)
  %unreg_field_0246 = getelementptr inbounds %Linear, ptr %field_val245, i32 0, i32 0
  %field_val247 = load ptr, ptr %unreg_field_0246, align 8
  call void @tl_mem_unregister(ptr %field_val247)
  %unreg_field_1248 = getelementptr inbounds %Linear, ptr %field_val245, i32 0, i32 1
  %field_val249 = load ptr, ptr %unreg_field_1248, align 8
  call void @tl_mem_unregister(ptr %field_val249)
  %unreg_field_2250 = getelementptr inbounds %Block, ptr %field_val217, i32 0, i32 2
  %field_val251 = load ptr, ptr %unreg_field_2250, align 8
  call void @tl_mem_unregister(ptr %field_val251)
  %unreg_field_0252 = getelementptr inbounds %LayerNorm, ptr %field_val251, i32 0, i32 0
  %field_val253 = load ptr, ptr %unreg_field_0252, align 8
  call void @tl_mem_unregister(ptr %field_val253)
  %unreg_field_1254 = getelementptr inbounds %LayerNorm, ptr %field_val251, i32 0, i32 1
  %field_val255 = load ptr, ptr %unreg_field_1254, align 8
  call void @tl_mem_unregister(ptr %field_val255)
  %unreg_field_3256 = getelementptr inbounds %Block, ptr %field_val217, i32 0, i32 3
  %field_val257 = load ptr, ptr %unreg_field_3256, align 8
  call void @tl_mem_unregister(ptr %field_val257)
  %unreg_field_0258 = getelementptr inbounds %MLP, ptr %field_val257, i32 0, i32 0
  %field_val259 = load ptr, ptr %unreg_field_0258, align 8
  call void @tl_mem_unregister(ptr %field_val259)
  %unreg_field_0260 = getelementptr inbounds %Linear, ptr %field_val259, i32 0, i32 0
  %field_val261 = load ptr, ptr %unreg_field_0260, align 8
  call void @tl_mem_unregister(ptr %field_val261)
  %unreg_field_1262 = getelementptr inbounds %Linear, ptr %field_val259, i32 0, i32 1
  %field_val263 = load ptr, ptr %unreg_field_1262, align 8
  call void @tl_mem_unregister(ptr %field_val263)
  %unreg_field_1264 = getelementptr inbounds %MLP, ptr %field_val257, i32 0, i32 1
  %field_val265 = load ptr, ptr %unreg_field_1264, align 8
  call void @tl_mem_unregister(ptr %field_val265)
  %unreg_field_0266 = getelementptr inbounds %Linear, ptr %field_val265, i32 0, i32 0
  %field_val267 = load ptr, ptr %unreg_field_0266, align 8
  call void @tl_mem_unregister(ptr %field_val267)
  %unreg_field_1268 = getelementptr inbounds %Linear, ptr %field_val265, i32 0, i32 1
  %field_val269 = load ptr, ptr %unreg_field_1268, align 8
  call void @tl_mem_unregister(ptr %field_val269)
  %unreg_field_4 = getelementptr inbounds %GPTHeavy, ptr %model158, i32 0, i32 4
  %field_val270 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val270)
  %unreg_field_0271 = getelementptr inbounds %Block, ptr %field_val270, i32 0, i32 0
  %field_val272 = load ptr, ptr %unreg_field_0271, align 8
  call void @tl_mem_unregister(ptr %field_val272)
  %unreg_field_0273 = getelementptr inbounds %LayerNorm, ptr %field_val272, i32 0, i32 0
  %field_val274 = load ptr, ptr %unreg_field_0273, align 8
  call void @tl_mem_unregister(ptr %field_val274)
  %unreg_field_1275 = getelementptr inbounds %LayerNorm, ptr %field_val272, i32 0, i32 1
  %field_val276 = load ptr, ptr %unreg_field_1275, align 8
  call void @tl_mem_unregister(ptr %field_val276)
  %unreg_field_1277 = getelementptr inbounds %Block, ptr %field_val270, i32 0, i32 1
  %field_val278 = load ptr, ptr %unreg_field_1277, align 8
  call void @tl_mem_unregister(ptr %field_val278)
  %unreg_field_0279 = getelementptr inbounds %CausalSelfAttention, ptr %field_val278, i32 0, i32 0
  %field_val280 = load ptr, ptr %unreg_field_0279, align 8
  call void @tl_mem_unregister(ptr %field_val280)
  %unreg_field_0281 = getelementptr inbounds %Linear, ptr %field_val280, i32 0, i32 0
  %field_val282 = load ptr, ptr %unreg_field_0281, align 8
  call void @tl_mem_unregister(ptr %field_val282)
  %unreg_field_1283 = getelementptr inbounds %Linear, ptr %field_val280, i32 0, i32 1
  %field_val284 = load ptr, ptr %unreg_field_1283, align 8
  call void @tl_mem_unregister(ptr %field_val284)
  %unreg_field_1285 = getelementptr inbounds %CausalSelfAttention, ptr %field_val278, i32 0, i32 1
  %field_val286 = load ptr, ptr %unreg_field_1285, align 8
  call void @tl_mem_unregister(ptr %field_val286)
  %unreg_field_0287 = getelementptr inbounds %Linear, ptr %field_val286, i32 0, i32 0
  %field_val288 = load ptr, ptr %unreg_field_0287, align 8
  call void @tl_mem_unregister(ptr %field_val288)
  %unreg_field_1289 = getelementptr inbounds %Linear, ptr %field_val286, i32 0, i32 1
  %field_val290 = load ptr, ptr %unreg_field_1289, align 8
  call void @tl_mem_unregister(ptr %field_val290)
  %unreg_field_2291 = getelementptr inbounds %CausalSelfAttention, ptr %field_val278, i32 0, i32 2
  %field_val292 = load ptr, ptr %unreg_field_2291, align 8
  call void @tl_mem_unregister(ptr %field_val292)
  %unreg_field_0293 = getelementptr inbounds %Linear, ptr %field_val292, i32 0, i32 0
  %field_val294 = load ptr, ptr %unreg_field_0293, align 8
  call void @tl_mem_unregister(ptr %field_val294)
  %unreg_field_1295 = getelementptr inbounds %Linear, ptr %field_val292, i32 0, i32 1
  %field_val296 = load ptr, ptr %unreg_field_1295, align 8
  call void @tl_mem_unregister(ptr %field_val296)
  %unreg_field_3297 = getelementptr inbounds %CausalSelfAttention, ptr %field_val278, i32 0, i32 3
  %field_val298 = load ptr, ptr %unreg_field_3297, align 8
  call void @tl_mem_unregister(ptr %field_val298)
  %unreg_field_0299 = getelementptr inbounds %Linear, ptr %field_val298, i32 0, i32 0
  %field_val300 = load ptr, ptr %unreg_field_0299, align 8
  call void @tl_mem_unregister(ptr %field_val300)
  %unreg_field_1301 = getelementptr inbounds %Linear, ptr %field_val298, i32 0, i32 1
  %field_val302 = load ptr, ptr %unreg_field_1301, align 8
  call void @tl_mem_unregister(ptr %field_val302)
  %unreg_field_2303 = getelementptr inbounds %Block, ptr %field_val270, i32 0, i32 2
  %field_val304 = load ptr, ptr %unreg_field_2303, align 8
  call void @tl_mem_unregister(ptr %field_val304)
  %unreg_field_0305 = getelementptr inbounds %LayerNorm, ptr %field_val304, i32 0, i32 0
  %field_val306 = load ptr, ptr %unreg_field_0305, align 8
  call void @tl_mem_unregister(ptr %field_val306)
  %unreg_field_1307 = getelementptr inbounds %LayerNorm, ptr %field_val304, i32 0, i32 1
  %field_val308 = load ptr, ptr %unreg_field_1307, align 8
  call void @tl_mem_unregister(ptr %field_val308)
  %unreg_field_3309 = getelementptr inbounds %Block, ptr %field_val270, i32 0, i32 3
  %field_val310 = load ptr, ptr %unreg_field_3309, align 8
  call void @tl_mem_unregister(ptr %field_val310)
  %unreg_field_0311 = getelementptr inbounds %MLP, ptr %field_val310, i32 0, i32 0
  %field_val312 = load ptr, ptr %unreg_field_0311, align 8
  call void @tl_mem_unregister(ptr %field_val312)
  %unreg_field_0313 = getelementptr inbounds %Linear, ptr %field_val312, i32 0, i32 0
  %field_val314 = load ptr, ptr %unreg_field_0313, align 8
  call void @tl_mem_unregister(ptr %field_val314)
  %unreg_field_1315 = getelementptr inbounds %Linear, ptr %field_val312, i32 0, i32 1
  %field_val316 = load ptr, ptr %unreg_field_1315, align 8
  call void @tl_mem_unregister(ptr %field_val316)
  %unreg_field_1317 = getelementptr inbounds %MLP, ptr %field_val310, i32 0, i32 1
  %field_val318 = load ptr, ptr %unreg_field_1317, align 8
  call void @tl_mem_unregister(ptr %field_val318)
  %unreg_field_0319 = getelementptr inbounds %Linear, ptr %field_val318, i32 0, i32 0
  %field_val320 = load ptr, ptr %unreg_field_0319, align 8
  call void @tl_mem_unregister(ptr %field_val320)
  %unreg_field_1321 = getelementptr inbounds %Linear, ptr %field_val318, i32 0, i32 1
  %field_val322 = load ptr, ptr %unreg_field_1321, align 8
  call void @tl_mem_unregister(ptr %field_val322)
  %unreg_field_5 = getelementptr inbounds %GPTHeavy, ptr %model158, i32 0, i32 5
  %field_val323 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val323)
  %unreg_field_0324 = getelementptr inbounds %Block, ptr %field_val323, i32 0, i32 0
  %field_val325 = load ptr, ptr %unreg_field_0324, align 8
  call void @tl_mem_unregister(ptr %field_val325)
  %unreg_field_0326 = getelementptr inbounds %LayerNorm, ptr %field_val325, i32 0, i32 0
  %field_val327 = load ptr, ptr %unreg_field_0326, align 8
  call void @tl_mem_unregister(ptr %field_val327)
  %unreg_field_1328 = getelementptr inbounds %LayerNorm, ptr %field_val325, i32 0, i32 1
  %field_val329 = load ptr, ptr %unreg_field_1328, align 8
  call void @tl_mem_unregister(ptr %field_val329)
  %unreg_field_1330 = getelementptr inbounds %Block, ptr %field_val323, i32 0, i32 1
  %field_val331 = load ptr, ptr %unreg_field_1330, align 8
  call void @tl_mem_unregister(ptr %field_val331)
  %unreg_field_0332 = getelementptr inbounds %CausalSelfAttention, ptr %field_val331, i32 0, i32 0
  %field_val333 = load ptr, ptr %unreg_field_0332, align 8
  call void @tl_mem_unregister(ptr %field_val333)
  %unreg_field_0334 = getelementptr inbounds %Linear, ptr %field_val333, i32 0, i32 0
  %field_val335 = load ptr, ptr %unreg_field_0334, align 8
  call void @tl_mem_unregister(ptr %field_val335)
  %unreg_field_1336 = getelementptr inbounds %Linear, ptr %field_val333, i32 0, i32 1
  %field_val337 = load ptr, ptr %unreg_field_1336, align 8
  call void @tl_mem_unregister(ptr %field_val337)
  %unreg_field_1338 = getelementptr inbounds %CausalSelfAttention, ptr %field_val331, i32 0, i32 1
  %field_val339 = load ptr, ptr %unreg_field_1338, align 8
  call void @tl_mem_unregister(ptr %field_val339)
  %unreg_field_0340 = getelementptr inbounds %Linear, ptr %field_val339, i32 0, i32 0
  %field_val341 = load ptr, ptr %unreg_field_0340, align 8
  call void @tl_mem_unregister(ptr %field_val341)
  %unreg_field_1342 = getelementptr inbounds %Linear, ptr %field_val339, i32 0, i32 1
  %field_val343 = load ptr, ptr %unreg_field_1342, align 8
  call void @tl_mem_unregister(ptr %field_val343)
  %unreg_field_2344 = getelementptr inbounds %CausalSelfAttention, ptr %field_val331, i32 0, i32 2
  %field_val345 = load ptr, ptr %unreg_field_2344, align 8
  call void @tl_mem_unregister(ptr %field_val345)
  %unreg_field_0346 = getelementptr inbounds %Linear, ptr %field_val345, i32 0, i32 0
  %field_val347 = load ptr, ptr %unreg_field_0346, align 8
  call void @tl_mem_unregister(ptr %field_val347)
  %unreg_field_1348 = getelementptr inbounds %Linear, ptr %field_val345, i32 0, i32 1
  %field_val349 = load ptr, ptr %unreg_field_1348, align 8
  call void @tl_mem_unregister(ptr %field_val349)
  %unreg_field_3350 = getelementptr inbounds %CausalSelfAttention, ptr %field_val331, i32 0, i32 3
  %field_val351 = load ptr, ptr %unreg_field_3350, align 8
  call void @tl_mem_unregister(ptr %field_val351)
  %unreg_field_0352 = getelementptr inbounds %Linear, ptr %field_val351, i32 0, i32 0
  %field_val353 = load ptr, ptr %unreg_field_0352, align 8
  call void @tl_mem_unregister(ptr %field_val353)
  %unreg_field_1354 = getelementptr inbounds %Linear, ptr %field_val351, i32 0, i32 1
  %field_val355 = load ptr, ptr %unreg_field_1354, align 8
  call void @tl_mem_unregister(ptr %field_val355)
  %unreg_field_2356 = getelementptr inbounds %Block, ptr %field_val323, i32 0, i32 2
  %field_val357 = load ptr, ptr %unreg_field_2356, align 8
  call void @tl_mem_unregister(ptr %field_val357)
  %unreg_field_0358 = getelementptr inbounds %LayerNorm, ptr %field_val357, i32 0, i32 0
  %field_val359 = load ptr, ptr %unreg_field_0358, align 8
  call void @tl_mem_unregister(ptr %field_val359)
  %unreg_field_1360 = getelementptr inbounds %LayerNorm, ptr %field_val357, i32 0, i32 1
  %field_val361 = load ptr, ptr %unreg_field_1360, align 8
  call void @tl_mem_unregister(ptr %field_val361)
  %unreg_field_3362 = getelementptr inbounds %Block, ptr %field_val323, i32 0, i32 3
  %field_val363 = load ptr, ptr %unreg_field_3362, align 8
  call void @tl_mem_unregister(ptr %field_val363)
  %unreg_field_0364 = getelementptr inbounds %MLP, ptr %field_val363, i32 0, i32 0
  %field_val365 = load ptr, ptr %unreg_field_0364, align 8
  call void @tl_mem_unregister(ptr %field_val365)
  %unreg_field_0366 = getelementptr inbounds %Linear, ptr %field_val365, i32 0, i32 0
  %field_val367 = load ptr, ptr %unreg_field_0366, align 8
  call void @tl_mem_unregister(ptr %field_val367)
  %unreg_field_1368 = getelementptr inbounds %Linear, ptr %field_val365, i32 0, i32 1
  %field_val369 = load ptr, ptr %unreg_field_1368, align 8
  call void @tl_mem_unregister(ptr %field_val369)
  %unreg_field_1370 = getelementptr inbounds %MLP, ptr %field_val363, i32 0, i32 1
  %field_val371 = load ptr, ptr %unreg_field_1370, align 8
  call void @tl_mem_unregister(ptr %field_val371)
  %unreg_field_0372 = getelementptr inbounds %Linear, ptr %field_val371, i32 0, i32 0
  %field_val373 = load ptr, ptr %unreg_field_0372, align 8
  call void @tl_mem_unregister(ptr %field_val373)
  %unreg_field_1374 = getelementptr inbounds %Linear, ptr %field_val371, i32 0, i32 1
  %field_val375 = load ptr, ptr %unreg_field_1374, align 8
  call void @tl_mem_unregister(ptr %field_val375)
  %unreg_field_6 = getelementptr inbounds %GPTHeavy, ptr %model158, i32 0, i32 6
  %field_val376 = load ptr, ptr %unreg_field_6, align 8
  call void @tl_mem_unregister(ptr %field_val376)
  %unreg_field_0377 = getelementptr inbounds %LayerNorm, ptr %field_val376, i32 0, i32 0
  %field_val378 = load ptr, ptr %unreg_field_0377, align 8
  call void @tl_mem_unregister(ptr %field_val378)
  %unreg_field_1379 = getelementptr inbounds %LayerNorm, ptr %field_val376, i32 0, i32 1
  %field_val380 = load ptr, ptr %unreg_field_1379, align 8
  call void @tl_mem_unregister(ptr %field_val380)
  %unreg_field_7 = getelementptr inbounds %GPTHeavy, ptr %model158, i32 0, i32 7
  %field_val381 = load ptr, ptr %unreg_field_7, align 8
  call void @tl_mem_unregister(ptr %field_val381)
  %unreg_field_0382 = getelementptr inbounds %Linear, ptr %field_val381, i32 0, i32 0
  %field_val383 = load ptr, ptr %unreg_field_0382, align 8
  call void @tl_mem_unregister(ptr %field_val383)
  %unreg_field_1384 = getelementptr inbounds %Linear, ptr %field_val381, i32 0, i32 1
  %field_val385 = load ptr, ptr %unreg_field_1384, align 8
  call void @tl_mem_unregister(ptr %field_val385)
  %tensor_to_free = load ptr, ptr %X, align 8
  call void @tl_tensor_free(ptr %tensor_to_free)
  %tensor_to_free386 = load ptr, ptr %Y_flat, align 8
  call void @tl_tensor_free(ptr %tensor_to_free386)
  %tensor_to_free387 = load ptr, ptr %data, align 8
  call void @tl_tensor_free(ptr %tensor_to_free387)
  %struct_to_free = load ptr, ptr %_discard_0, align 8
  call void @tl_mem_unregister(ptr %struct_to_free)
  %tensor_to_free388 = load ptr, ptr %logits, align 8
  call void @tl_tensor_free(ptr %tensor_to_free388)
  %tensor_to_free389 = load ptr, ptr %Y, align 8
  call void @tl_tensor_free(ptr %tensor_to_free389)
  %tensor_to_free390 = load ptr, ptr %target, align 8
  call void @tl_tensor_free(ptr %tensor_to_free390)
  %tensor_to_free391 = load ptr, ptr %logits_flat, align 8
  call void @tl_tensor_free(ptr %tensor_to_free391)
  %tensor_to_free392 = load ptr, ptr %loss, align 8
  call void @tl_tensor_free(ptr %tensor_to_free392)
  call void @tl_mem_exit_scope()
  ret ptr %model158
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
  %can_free = and i1 %is_not_null, false
  br i1 %can_free, label %free_struct, label %continue_after_free

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal)
  %epoch251 = load i64, ptr %epoch3, align 8
  call void @tl_print_i64(i64 %epoch251)
  %model252 = load ptr, ptr %model1, align 8
  call void @tl_mem_unregister(ptr %model252)
  %unreg_field_0253 = getelementptr inbounds %GPTHeavy, ptr %model252, i32 0, i32 0
  %field_val254 = load ptr, ptr %unreg_field_0253, align 8
  call void @tl_mem_unregister(ptr %field_val254)
  %unreg_field_0255 = getelementptr inbounds %Embedding, ptr %field_val254, i32 0, i32 0
  %field_val256 = load ptr, ptr %unreg_field_0255, align 8
  call void @tl_mem_unregister(ptr %field_val256)
  %unreg_field_1257 = getelementptr inbounds %GPTHeavy, ptr %model252, i32 0, i32 1
  %field_val258 = load ptr, ptr %unreg_field_1257, align 8
  call void @tl_mem_unregister(ptr %field_val258)
  %unreg_field_0259 = getelementptr inbounds %Embedding, ptr %field_val258, i32 0, i32 0
  %field_val260 = load ptr, ptr %unreg_field_0259, align 8
  call void @tl_mem_unregister(ptr %field_val260)
  %unreg_field_2261 = getelementptr inbounds %GPTHeavy, ptr %model252, i32 0, i32 2
  %field_val262 = load ptr, ptr %unreg_field_2261, align 8
  call void @tl_mem_unregister(ptr %field_val262)
  %unreg_field_0263 = getelementptr inbounds %Block, ptr %field_val262, i32 0, i32 0
  %field_val264 = load ptr, ptr %unreg_field_0263, align 8
  call void @tl_mem_unregister(ptr %field_val264)
  %unreg_field_0265 = getelementptr inbounds %LayerNorm, ptr %field_val264, i32 0, i32 0
  %field_val266 = load ptr, ptr %unreg_field_0265, align 8
  call void @tl_mem_unregister(ptr %field_val266)
  %unreg_field_1267 = getelementptr inbounds %LayerNorm, ptr %field_val264, i32 0, i32 1
  %field_val268 = load ptr, ptr %unreg_field_1267, align 8
  call void @tl_mem_unregister(ptr %field_val268)
  %unreg_field_1269 = getelementptr inbounds %Block, ptr %field_val262, i32 0, i32 1
  %field_val270 = load ptr, ptr %unreg_field_1269, align 8
  call void @tl_mem_unregister(ptr %field_val270)
  %unreg_field_0271 = getelementptr inbounds %CausalSelfAttention, ptr %field_val270, i32 0, i32 0
  %field_val272 = load ptr, ptr %unreg_field_0271, align 8
  call void @tl_mem_unregister(ptr %field_val272)
  %unreg_field_0273 = getelementptr inbounds %Linear, ptr %field_val272, i32 0, i32 0
  %field_val274 = load ptr, ptr %unreg_field_0273, align 8
  call void @tl_mem_unregister(ptr %field_val274)
  %unreg_field_1275 = getelementptr inbounds %Linear, ptr %field_val272, i32 0, i32 1
  %field_val276 = load ptr, ptr %unreg_field_1275, align 8
  call void @tl_mem_unregister(ptr %field_val276)
  %unreg_field_1277 = getelementptr inbounds %CausalSelfAttention, ptr %field_val270, i32 0, i32 1
  %field_val278 = load ptr, ptr %unreg_field_1277, align 8
  call void @tl_mem_unregister(ptr %field_val278)
  %unreg_field_0279 = getelementptr inbounds %Linear, ptr %field_val278, i32 0, i32 0
  %field_val280 = load ptr, ptr %unreg_field_0279, align 8
  call void @tl_mem_unregister(ptr %field_val280)
  %unreg_field_1281 = getelementptr inbounds %Linear, ptr %field_val278, i32 0, i32 1
  %field_val282 = load ptr, ptr %unreg_field_1281, align 8
  call void @tl_mem_unregister(ptr %field_val282)
  %unreg_field_2283 = getelementptr inbounds %CausalSelfAttention, ptr %field_val270, i32 0, i32 2
  %field_val284 = load ptr, ptr %unreg_field_2283, align 8
  call void @tl_mem_unregister(ptr %field_val284)
  %unreg_field_0285 = getelementptr inbounds %Linear, ptr %field_val284, i32 0, i32 0
  %field_val286 = load ptr, ptr %unreg_field_0285, align 8
  call void @tl_mem_unregister(ptr %field_val286)
  %unreg_field_1287 = getelementptr inbounds %Linear, ptr %field_val284, i32 0, i32 1
  %field_val288 = load ptr, ptr %unreg_field_1287, align 8
  call void @tl_mem_unregister(ptr %field_val288)
  %unreg_field_3289 = getelementptr inbounds %CausalSelfAttention, ptr %field_val270, i32 0, i32 3
  %field_val290 = load ptr, ptr %unreg_field_3289, align 8
  call void @tl_mem_unregister(ptr %field_val290)
  %unreg_field_0291 = getelementptr inbounds %Linear, ptr %field_val290, i32 0, i32 0
  %field_val292 = load ptr, ptr %unreg_field_0291, align 8
  call void @tl_mem_unregister(ptr %field_val292)
  %unreg_field_1293 = getelementptr inbounds %Linear, ptr %field_val290, i32 0, i32 1
  %field_val294 = load ptr, ptr %unreg_field_1293, align 8
  call void @tl_mem_unregister(ptr %field_val294)
  %unreg_field_2295 = getelementptr inbounds %Block, ptr %field_val262, i32 0, i32 2
  %field_val296 = load ptr, ptr %unreg_field_2295, align 8
  call void @tl_mem_unregister(ptr %field_val296)
  %unreg_field_0297 = getelementptr inbounds %LayerNorm, ptr %field_val296, i32 0, i32 0
  %field_val298 = load ptr, ptr %unreg_field_0297, align 8
  call void @tl_mem_unregister(ptr %field_val298)
  %unreg_field_1299 = getelementptr inbounds %LayerNorm, ptr %field_val296, i32 0, i32 1
  %field_val300 = load ptr, ptr %unreg_field_1299, align 8
  call void @tl_mem_unregister(ptr %field_val300)
  %unreg_field_3301 = getelementptr inbounds %Block, ptr %field_val262, i32 0, i32 3
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
  %unreg_field_3315 = getelementptr inbounds %GPTHeavy, ptr %model252, i32 0, i32 3
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
  %unreg_field_4369 = getelementptr inbounds %GPTHeavy, ptr %model252, i32 0, i32 4
  %field_val370 = load ptr, ptr %unreg_field_4369, align 8
  call void @tl_mem_unregister(ptr %field_val370)
  %unreg_field_0371 = getelementptr inbounds %Block, ptr %field_val370, i32 0, i32 0
  %field_val372 = load ptr, ptr %unreg_field_0371, align 8
  call void @tl_mem_unregister(ptr %field_val372)
  %unreg_field_0373 = getelementptr inbounds %LayerNorm, ptr %field_val372, i32 0, i32 0
  %field_val374 = load ptr, ptr %unreg_field_0373, align 8
  call void @tl_mem_unregister(ptr %field_val374)
  %unreg_field_1375 = getelementptr inbounds %LayerNorm, ptr %field_val372, i32 0, i32 1
  %field_val376 = load ptr, ptr %unreg_field_1375, align 8
  call void @tl_mem_unregister(ptr %field_val376)
  %unreg_field_1377 = getelementptr inbounds %Block, ptr %field_val370, i32 0, i32 1
  %field_val378 = load ptr, ptr %unreg_field_1377, align 8
  call void @tl_mem_unregister(ptr %field_val378)
  %unreg_field_0379 = getelementptr inbounds %CausalSelfAttention, ptr %field_val378, i32 0, i32 0
  %field_val380 = load ptr, ptr %unreg_field_0379, align 8
  call void @tl_mem_unregister(ptr %field_val380)
  %unreg_field_0381 = getelementptr inbounds %Linear, ptr %field_val380, i32 0, i32 0
  %field_val382 = load ptr, ptr %unreg_field_0381, align 8
  call void @tl_mem_unregister(ptr %field_val382)
  %unreg_field_1383 = getelementptr inbounds %Linear, ptr %field_val380, i32 0, i32 1
  %field_val384 = load ptr, ptr %unreg_field_1383, align 8
  call void @tl_mem_unregister(ptr %field_val384)
  %unreg_field_1385 = getelementptr inbounds %CausalSelfAttention, ptr %field_val378, i32 0, i32 1
  %field_val386 = load ptr, ptr %unreg_field_1385, align 8
  call void @tl_mem_unregister(ptr %field_val386)
  %unreg_field_0387 = getelementptr inbounds %Linear, ptr %field_val386, i32 0, i32 0
  %field_val388 = load ptr, ptr %unreg_field_0387, align 8
  call void @tl_mem_unregister(ptr %field_val388)
  %unreg_field_1389 = getelementptr inbounds %Linear, ptr %field_val386, i32 0, i32 1
  %field_val390 = load ptr, ptr %unreg_field_1389, align 8
  call void @tl_mem_unregister(ptr %field_val390)
  %unreg_field_2391 = getelementptr inbounds %CausalSelfAttention, ptr %field_val378, i32 0, i32 2
  %field_val392 = load ptr, ptr %unreg_field_2391, align 8
  call void @tl_mem_unregister(ptr %field_val392)
  %unreg_field_0393 = getelementptr inbounds %Linear, ptr %field_val392, i32 0, i32 0
  %field_val394 = load ptr, ptr %unreg_field_0393, align 8
  call void @tl_mem_unregister(ptr %field_val394)
  %unreg_field_1395 = getelementptr inbounds %Linear, ptr %field_val392, i32 0, i32 1
  %field_val396 = load ptr, ptr %unreg_field_1395, align 8
  call void @tl_mem_unregister(ptr %field_val396)
  %unreg_field_3397 = getelementptr inbounds %CausalSelfAttention, ptr %field_val378, i32 0, i32 3
  %field_val398 = load ptr, ptr %unreg_field_3397, align 8
  call void @tl_mem_unregister(ptr %field_val398)
  %unreg_field_0399 = getelementptr inbounds %Linear, ptr %field_val398, i32 0, i32 0
  %field_val400 = load ptr, ptr %unreg_field_0399, align 8
  call void @tl_mem_unregister(ptr %field_val400)
  %unreg_field_1401 = getelementptr inbounds %Linear, ptr %field_val398, i32 0, i32 1
  %field_val402 = load ptr, ptr %unreg_field_1401, align 8
  call void @tl_mem_unregister(ptr %field_val402)
  %unreg_field_2403 = getelementptr inbounds %Block, ptr %field_val370, i32 0, i32 2
  %field_val404 = load ptr, ptr %unreg_field_2403, align 8
  call void @tl_mem_unregister(ptr %field_val404)
  %unreg_field_0405 = getelementptr inbounds %LayerNorm, ptr %field_val404, i32 0, i32 0
  %field_val406 = load ptr, ptr %unreg_field_0405, align 8
  call void @tl_mem_unregister(ptr %field_val406)
  %unreg_field_1407 = getelementptr inbounds %LayerNorm, ptr %field_val404, i32 0, i32 1
  %field_val408 = load ptr, ptr %unreg_field_1407, align 8
  call void @tl_mem_unregister(ptr %field_val408)
  %unreg_field_3409 = getelementptr inbounds %Block, ptr %field_val370, i32 0, i32 3
  %field_val410 = load ptr, ptr %unreg_field_3409, align 8
  call void @tl_mem_unregister(ptr %field_val410)
  %unreg_field_0411 = getelementptr inbounds %MLP, ptr %field_val410, i32 0, i32 0
  %field_val412 = load ptr, ptr %unreg_field_0411, align 8
  call void @tl_mem_unregister(ptr %field_val412)
  %unreg_field_0413 = getelementptr inbounds %Linear, ptr %field_val412, i32 0, i32 0
  %field_val414 = load ptr, ptr %unreg_field_0413, align 8
  call void @tl_mem_unregister(ptr %field_val414)
  %unreg_field_1415 = getelementptr inbounds %Linear, ptr %field_val412, i32 0, i32 1
  %field_val416 = load ptr, ptr %unreg_field_1415, align 8
  call void @tl_mem_unregister(ptr %field_val416)
  %unreg_field_1417 = getelementptr inbounds %MLP, ptr %field_val410, i32 0, i32 1
  %field_val418 = load ptr, ptr %unreg_field_1417, align 8
  call void @tl_mem_unregister(ptr %field_val418)
  %unreg_field_0419 = getelementptr inbounds %Linear, ptr %field_val418, i32 0, i32 0
  %field_val420 = load ptr, ptr %unreg_field_0419, align 8
  call void @tl_mem_unregister(ptr %field_val420)
  %unreg_field_1421 = getelementptr inbounds %Linear, ptr %field_val418, i32 0, i32 1
  %field_val422 = load ptr, ptr %unreg_field_1421, align 8
  call void @tl_mem_unregister(ptr %field_val422)
  %unreg_field_5423 = getelementptr inbounds %GPTHeavy, ptr %model252, i32 0, i32 5
  %field_val424 = load ptr, ptr %unreg_field_5423, align 8
  call void @tl_mem_unregister(ptr %field_val424)
  %unreg_field_0425 = getelementptr inbounds %Block, ptr %field_val424, i32 0, i32 0
  %field_val426 = load ptr, ptr %unreg_field_0425, align 8
  call void @tl_mem_unregister(ptr %field_val426)
  %unreg_field_0427 = getelementptr inbounds %LayerNorm, ptr %field_val426, i32 0, i32 0
  %field_val428 = load ptr, ptr %unreg_field_0427, align 8
  call void @tl_mem_unregister(ptr %field_val428)
  %unreg_field_1429 = getelementptr inbounds %LayerNorm, ptr %field_val426, i32 0, i32 1
  %field_val430 = load ptr, ptr %unreg_field_1429, align 8
  call void @tl_mem_unregister(ptr %field_val430)
  %unreg_field_1431 = getelementptr inbounds %Block, ptr %field_val424, i32 0, i32 1
  %field_val432 = load ptr, ptr %unreg_field_1431, align 8
  call void @tl_mem_unregister(ptr %field_val432)
  %unreg_field_0433 = getelementptr inbounds %CausalSelfAttention, ptr %field_val432, i32 0, i32 0
  %field_val434 = load ptr, ptr %unreg_field_0433, align 8
  call void @tl_mem_unregister(ptr %field_val434)
  %unreg_field_0435 = getelementptr inbounds %Linear, ptr %field_val434, i32 0, i32 0
  %field_val436 = load ptr, ptr %unreg_field_0435, align 8
  call void @tl_mem_unregister(ptr %field_val436)
  %unreg_field_1437 = getelementptr inbounds %Linear, ptr %field_val434, i32 0, i32 1
  %field_val438 = load ptr, ptr %unreg_field_1437, align 8
  call void @tl_mem_unregister(ptr %field_val438)
  %unreg_field_1439 = getelementptr inbounds %CausalSelfAttention, ptr %field_val432, i32 0, i32 1
  %field_val440 = load ptr, ptr %unreg_field_1439, align 8
  call void @tl_mem_unregister(ptr %field_val440)
  %unreg_field_0441 = getelementptr inbounds %Linear, ptr %field_val440, i32 0, i32 0
  %field_val442 = load ptr, ptr %unreg_field_0441, align 8
  call void @tl_mem_unregister(ptr %field_val442)
  %unreg_field_1443 = getelementptr inbounds %Linear, ptr %field_val440, i32 0, i32 1
  %field_val444 = load ptr, ptr %unreg_field_1443, align 8
  call void @tl_mem_unregister(ptr %field_val444)
  %unreg_field_2445 = getelementptr inbounds %CausalSelfAttention, ptr %field_val432, i32 0, i32 2
  %field_val446 = load ptr, ptr %unreg_field_2445, align 8
  call void @tl_mem_unregister(ptr %field_val446)
  %unreg_field_0447 = getelementptr inbounds %Linear, ptr %field_val446, i32 0, i32 0
  %field_val448 = load ptr, ptr %unreg_field_0447, align 8
  call void @tl_mem_unregister(ptr %field_val448)
  %unreg_field_1449 = getelementptr inbounds %Linear, ptr %field_val446, i32 0, i32 1
  %field_val450 = load ptr, ptr %unreg_field_1449, align 8
  call void @tl_mem_unregister(ptr %field_val450)
  %unreg_field_3451 = getelementptr inbounds %CausalSelfAttention, ptr %field_val432, i32 0, i32 3
  %field_val452 = load ptr, ptr %unreg_field_3451, align 8
  call void @tl_mem_unregister(ptr %field_val452)
  %unreg_field_0453 = getelementptr inbounds %Linear, ptr %field_val452, i32 0, i32 0
  %field_val454 = load ptr, ptr %unreg_field_0453, align 8
  call void @tl_mem_unregister(ptr %field_val454)
  %unreg_field_1455 = getelementptr inbounds %Linear, ptr %field_val452, i32 0, i32 1
  %field_val456 = load ptr, ptr %unreg_field_1455, align 8
  call void @tl_mem_unregister(ptr %field_val456)
  %unreg_field_2457 = getelementptr inbounds %Block, ptr %field_val424, i32 0, i32 2
  %field_val458 = load ptr, ptr %unreg_field_2457, align 8
  call void @tl_mem_unregister(ptr %field_val458)
  %unreg_field_0459 = getelementptr inbounds %LayerNorm, ptr %field_val458, i32 0, i32 0
  %field_val460 = load ptr, ptr %unreg_field_0459, align 8
  call void @tl_mem_unregister(ptr %field_val460)
  %unreg_field_1461 = getelementptr inbounds %LayerNorm, ptr %field_val458, i32 0, i32 1
  %field_val462 = load ptr, ptr %unreg_field_1461, align 8
  call void @tl_mem_unregister(ptr %field_val462)
  %unreg_field_3463 = getelementptr inbounds %Block, ptr %field_val424, i32 0, i32 3
  %field_val464 = load ptr, ptr %unreg_field_3463, align 8
  call void @tl_mem_unregister(ptr %field_val464)
  %unreg_field_0465 = getelementptr inbounds %MLP, ptr %field_val464, i32 0, i32 0
  %field_val466 = load ptr, ptr %unreg_field_0465, align 8
  call void @tl_mem_unregister(ptr %field_val466)
  %unreg_field_0467 = getelementptr inbounds %Linear, ptr %field_val466, i32 0, i32 0
  %field_val468 = load ptr, ptr %unreg_field_0467, align 8
  call void @tl_mem_unregister(ptr %field_val468)
  %unreg_field_1469 = getelementptr inbounds %Linear, ptr %field_val466, i32 0, i32 1
  %field_val470 = load ptr, ptr %unreg_field_1469, align 8
  call void @tl_mem_unregister(ptr %field_val470)
  %unreg_field_1471 = getelementptr inbounds %MLP, ptr %field_val464, i32 0, i32 1
  %field_val472 = load ptr, ptr %unreg_field_1471, align 8
  call void @tl_mem_unregister(ptr %field_val472)
  %unreg_field_0473 = getelementptr inbounds %Linear, ptr %field_val472, i32 0, i32 0
  %field_val474 = load ptr, ptr %unreg_field_0473, align 8
  call void @tl_mem_unregister(ptr %field_val474)
  %unreg_field_1475 = getelementptr inbounds %Linear, ptr %field_val472, i32 0, i32 1
  %field_val476 = load ptr, ptr %unreg_field_1475, align 8
  call void @tl_mem_unregister(ptr %field_val476)
  %unreg_field_6477 = getelementptr inbounds %GPTHeavy, ptr %model252, i32 0, i32 6
  %field_val478 = load ptr, ptr %unreg_field_6477, align 8
  call void @tl_mem_unregister(ptr %field_val478)
  %unreg_field_0479 = getelementptr inbounds %LayerNorm, ptr %field_val478, i32 0, i32 0
  %field_val480 = load ptr, ptr %unreg_field_0479, align 8
  call void @tl_mem_unregister(ptr %field_val480)
  %unreg_field_1481 = getelementptr inbounds %LayerNorm, ptr %field_val478, i32 0, i32 1
  %field_val482 = load ptr, ptr %unreg_field_1481, align 8
  call void @tl_mem_unregister(ptr %field_val482)
  %unreg_field_7483 = getelementptr inbounds %GPTHeavy, ptr %model252, i32 0, i32 7
  %field_val484 = load ptr, ptr %unreg_field_7483, align 8
  call void @tl_mem_unregister(ptr %field_val484)
  %unreg_field_0485 = getelementptr inbounds %Linear, ptr %field_val484, i32 0, i32 0
  %field_val486 = load ptr, ptr %unreg_field_0485, align 8
  call void @tl_mem_unregister(ptr %field_val486)
  %unreg_field_1487 = getelementptr inbounds %Linear, ptr %field_val484, i32 0, i32 1
  %field_val488 = load ptr, ptr %unreg_field_1487, align 8
  call void @tl_mem_unregister(ptr %field_val488)
  call void @tl_mem_exit_scope()
  ret ptr %model252

free_struct:                                      ; preds = %for_body
  br label %continue_after_free

continue_after_free:                              ; preds = %free_struct, %for_body
  call void @tl_mem_unregister(ptr %call_tmp)
  %unreg_field_0 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_024 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val25 = load ptr, ptr %unreg_field_024, align 8
  call void @tl_mem_unregister(ptr %field_val25)
  %unreg_field_1 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 1
  %field_val26 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val26)
  %unreg_field_027 = getelementptr inbounds %Embedding, ptr %field_val26, i32 0, i32 0
  %field_val28 = load ptr, ptr %unreg_field_027, align 8
  call void @tl_mem_unregister(ptr %field_val28)
  %unreg_field_2 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 2
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
  %unreg_field_381 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 3
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
  %unreg_field_4 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 4
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
  %unreg_field_5 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 5
  %field_val188 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val188)
  %unreg_field_0189 = getelementptr inbounds %Block, ptr %field_val188, i32 0, i32 0
  %field_val190 = load ptr, ptr %unreg_field_0189, align 8
  call void @tl_mem_unregister(ptr %field_val190)
  %unreg_field_0191 = getelementptr inbounds %LayerNorm, ptr %field_val190, i32 0, i32 0
  %field_val192 = load ptr, ptr %unreg_field_0191, align 8
  call void @tl_mem_unregister(ptr %field_val192)
  %unreg_field_1193 = getelementptr inbounds %LayerNorm, ptr %field_val190, i32 0, i32 1
  %field_val194 = load ptr, ptr %unreg_field_1193, align 8
  call void @tl_mem_unregister(ptr %field_val194)
  %unreg_field_1195 = getelementptr inbounds %Block, ptr %field_val188, i32 0, i32 1
  %field_val196 = load ptr, ptr %unreg_field_1195, align 8
  call void @tl_mem_unregister(ptr %field_val196)
  %unreg_field_0197 = getelementptr inbounds %CausalSelfAttention, ptr %field_val196, i32 0, i32 0
  %field_val198 = load ptr, ptr %unreg_field_0197, align 8
  call void @tl_mem_unregister(ptr %field_val198)
  %unreg_field_0199 = getelementptr inbounds %Linear, ptr %field_val198, i32 0, i32 0
  %field_val200 = load ptr, ptr %unreg_field_0199, align 8
  call void @tl_mem_unregister(ptr %field_val200)
  %unreg_field_1201 = getelementptr inbounds %Linear, ptr %field_val198, i32 0, i32 1
  %field_val202 = load ptr, ptr %unreg_field_1201, align 8
  call void @tl_mem_unregister(ptr %field_val202)
  %unreg_field_1203 = getelementptr inbounds %CausalSelfAttention, ptr %field_val196, i32 0, i32 1
  %field_val204 = load ptr, ptr %unreg_field_1203, align 8
  call void @tl_mem_unregister(ptr %field_val204)
  %unreg_field_0205 = getelementptr inbounds %Linear, ptr %field_val204, i32 0, i32 0
  %field_val206 = load ptr, ptr %unreg_field_0205, align 8
  call void @tl_mem_unregister(ptr %field_val206)
  %unreg_field_1207 = getelementptr inbounds %Linear, ptr %field_val204, i32 0, i32 1
  %field_val208 = load ptr, ptr %unreg_field_1207, align 8
  call void @tl_mem_unregister(ptr %field_val208)
  %unreg_field_2209 = getelementptr inbounds %CausalSelfAttention, ptr %field_val196, i32 0, i32 2
  %field_val210 = load ptr, ptr %unreg_field_2209, align 8
  call void @tl_mem_unregister(ptr %field_val210)
  %unreg_field_0211 = getelementptr inbounds %Linear, ptr %field_val210, i32 0, i32 0
  %field_val212 = load ptr, ptr %unreg_field_0211, align 8
  call void @tl_mem_unregister(ptr %field_val212)
  %unreg_field_1213 = getelementptr inbounds %Linear, ptr %field_val210, i32 0, i32 1
  %field_val214 = load ptr, ptr %unreg_field_1213, align 8
  call void @tl_mem_unregister(ptr %field_val214)
  %unreg_field_3215 = getelementptr inbounds %CausalSelfAttention, ptr %field_val196, i32 0, i32 3
  %field_val216 = load ptr, ptr %unreg_field_3215, align 8
  call void @tl_mem_unregister(ptr %field_val216)
  %unreg_field_0217 = getelementptr inbounds %Linear, ptr %field_val216, i32 0, i32 0
  %field_val218 = load ptr, ptr %unreg_field_0217, align 8
  call void @tl_mem_unregister(ptr %field_val218)
  %unreg_field_1219 = getelementptr inbounds %Linear, ptr %field_val216, i32 0, i32 1
  %field_val220 = load ptr, ptr %unreg_field_1219, align 8
  call void @tl_mem_unregister(ptr %field_val220)
  %unreg_field_2221 = getelementptr inbounds %Block, ptr %field_val188, i32 0, i32 2
  %field_val222 = load ptr, ptr %unreg_field_2221, align 8
  call void @tl_mem_unregister(ptr %field_val222)
  %unreg_field_0223 = getelementptr inbounds %LayerNorm, ptr %field_val222, i32 0, i32 0
  %field_val224 = load ptr, ptr %unreg_field_0223, align 8
  call void @tl_mem_unregister(ptr %field_val224)
  %unreg_field_1225 = getelementptr inbounds %LayerNorm, ptr %field_val222, i32 0, i32 1
  %field_val226 = load ptr, ptr %unreg_field_1225, align 8
  call void @tl_mem_unregister(ptr %field_val226)
  %unreg_field_3227 = getelementptr inbounds %Block, ptr %field_val188, i32 0, i32 3
  %field_val228 = load ptr, ptr %unreg_field_3227, align 8
  call void @tl_mem_unregister(ptr %field_val228)
  %unreg_field_0229 = getelementptr inbounds %MLP, ptr %field_val228, i32 0, i32 0
  %field_val230 = load ptr, ptr %unreg_field_0229, align 8
  call void @tl_mem_unregister(ptr %field_val230)
  %unreg_field_0231 = getelementptr inbounds %Linear, ptr %field_val230, i32 0, i32 0
  %field_val232 = load ptr, ptr %unreg_field_0231, align 8
  call void @tl_mem_unregister(ptr %field_val232)
  %unreg_field_1233 = getelementptr inbounds %Linear, ptr %field_val230, i32 0, i32 1
  %field_val234 = load ptr, ptr %unreg_field_1233, align 8
  call void @tl_mem_unregister(ptr %field_val234)
  %unreg_field_1235 = getelementptr inbounds %MLP, ptr %field_val228, i32 0, i32 1
  %field_val236 = load ptr, ptr %unreg_field_1235, align 8
  call void @tl_mem_unregister(ptr %field_val236)
  %unreg_field_0237 = getelementptr inbounds %Linear, ptr %field_val236, i32 0, i32 0
  %field_val238 = load ptr, ptr %unreg_field_0237, align 8
  call void @tl_mem_unregister(ptr %field_val238)
  %unreg_field_1239 = getelementptr inbounds %Linear, ptr %field_val236, i32 0, i32 1
  %field_val240 = load ptr, ptr %unreg_field_1239, align 8
  call void @tl_mem_unregister(ptr %field_val240)
  %unreg_field_6 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 6
  %field_val241 = load ptr, ptr %unreg_field_6, align 8
  call void @tl_mem_unregister(ptr %field_val241)
  %unreg_field_0242 = getelementptr inbounds %LayerNorm, ptr %field_val241, i32 0, i32 0
  %field_val243 = load ptr, ptr %unreg_field_0242, align 8
  call void @tl_mem_unregister(ptr %field_val243)
  %unreg_field_1244 = getelementptr inbounds %LayerNorm, ptr %field_val241, i32 0, i32 1
  %field_val245 = load ptr, ptr %unreg_field_1244, align 8
  call void @tl_mem_unregister(ptr %field_val245)
  %unreg_field_7 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 7
  %field_val246 = load ptr, ptr %unreg_field_7, align 8
  call void @tl_mem_unregister(ptr %field_val246)
  %unreg_field_0247 = getelementptr inbounds %Linear, ptr %field_val246, i32 0, i32 0
  %field_val248 = load ptr, ptr %unreg_field_0247, align 8
  call void @tl_mem_unregister(ptr %field_val248)
  %unreg_field_1249 = getelementptr inbounds %Linear, ptr %field_val246, i32 0, i32 1
  %field_val250 = load ptr, ptr %unreg_field_1249, align 8
  call void @tl_mem_unregister(ptr %field_val250)
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
  %can_free = and i1 %is_not_null, true
  br i1 %can_free, label %free_struct, label %continue_after_free

for_end:                                          ; preds = %for_header
  call void @tl_print_string(ptr @str_literal.247)
  %model689 = load ptr, ptr %model, align 8
  %w690 = getelementptr inbounds %GPTHeavy, ptr %model689, i32 0, i32 0
  %sub_ptr691 = load ptr, ptr %w690, align 8
  %w692 = getelementptr inbounds %Embedding, ptr %sub_ptr691, i32 0, i32 0
  %w693 = load ptr, ptr %w692, align 8
  call void @tl_add_parameter(ptr @key_str.248, ptr %w693)
  %wp694 = getelementptr inbounds %GPTHeavy, ptr %model689, i32 0, i32 1
  %sub_ptr695 = load ptr, ptr %wp694, align 8
  %w696 = getelementptr inbounds %Embedding, ptr %sub_ptr695, i32 0, i32 0
  %w697 = load ptr, ptr %w696, align 8
  call void @tl_add_parameter(ptr @key_str.249, ptr %w697)
  %b1698 = getelementptr inbounds %GPTHeavy, ptr %model689, i32 0, i32 2
  %sub_ptr699 = load ptr, ptr %b1698, align 8
  %l1700 = getelementptr inbounds %Block, ptr %sub_ptr699, i32 0, i32 0
  %sub_ptr701 = load ptr, ptr %l1700, align 8
  %w702 = getelementptr inbounds %LayerNorm, ptr %sub_ptr701, i32 0, i32 0
  %w703 = load ptr, ptr %w702, align 8
  call void @tl_add_parameter(ptr @key_str.250, ptr %w703)
  %b704 = getelementptr inbounds %LayerNorm, ptr %sub_ptr701, i32 0, i32 1
  %b705 = load ptr, ptr %b704, align 8
  call void @tl_add_parameter(ptr @key_str.251, ptr %b705)
  %a706 = getelementptr inbounds %Block, ptr %sub_ptr699, i32 0, i32 1
  %sub_ptr707 = load ptr, ptr %a706, align 8
  %q_proj708 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr707, i32 0, i32 0
  %sub_ptr709 = load ptr, ptr %q_proj708, align 8
  %W710 = getelementptr inbounds %Linear, ptr %sub_ptr709, i32 0, i32 0
  %W711 = load ptr, ptr %W710, align 8
  call void @tl_add_parameter(ptr @key_str.252, ptr %W711)
  %b712 = getelementptr inbounds %Linear, ptr %sub_ptr709, i32 0, i32 1
  %b713 = load ptr, ptr %b712, align 8
  call void @tl_add_parameter(ptr @key_str.253, ptr %b713)
  %k_proj714 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr707, i32 0, i32 1
  %sub_ptr715 = load ptr, ptr %k_proj714, align 8
  %W716 = getelementptr inbounds %Linear, ptr %sub_ptr715, i32 0, i32 0
  %W717 = load ptr, ptr %W716, align 8
  call void @tl_add_parameter(ptr @key_str.254, ptr %W717)
  %b718 = getelementptr inbounds %Linear, ptr %sub_ptr715, i32 0, i32 1
  %b719 = load ptr, ptr %b718, align 8
  call void @tl_add_parameter(ptr @key_str.255, ptr %b719)
  %v_proj720 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr707, i32 0, i32 2
  %sub_ptr721 = load ptr, ptr %v_proj720, align 8
  %W722 = getelementptr inbounds %Linear, ptr %sub_ptr721, i32 0, i32 0
  %W723 = load ptr, ptr %W722, align 8
  call void @tl_add_parameter(ptr @key_str.256, ptr %W723)
  %b724 = getelementptr inbounds %Linear, ptr %sub_ptr721, i32 0, i32 1
  %b725 = load ptr, ptr %b724, align 8
  call void @tl_add_parameter(ptr @key_str.257, ptr %b725)
  %p_proj726 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr707, i32 0, i32 3
  %sub_ptr727 = load ptr, ptr %p_proj726, align 8
  %W728 = getelementptr inbounds %Linear, ptr %sub_ptr727, i32 0, i32 0
  %W729 = load ptr, ptr %W728, align 8
  call void @tl_add_parameter(ptr @key_str.258, ptr %W729)
  %b730 = getelementptr inbounds %Linear, ptr %sub_ptr727, i32 0, i32 1
  %b731 = load ptr, ptr %b730, align 8
  call void @tl_add_parameter(ptr @key_str.259, ptr %b731)
  %l2732 = getelementptr inbounds %Block, ptr %sub_ptr699, i32 0, i32 2
  %sub_ptr733 = load ptr, ptr %l2732, align 8
  %w734 = getelementptr inbounds %LayerNorm, ptr %sub_ptr733, i32 0, i32 0
  %w735 = load ptr, ptr %w734, align 8
  call void @tl_add_parameter(ptr @key_str.260, ptr %w735)
  %b736 = getelementptr inbounds %LayerNorm, ptr %sub_ptr733, i32 0, i32 1
  %b737 = load ptr, ptr %b736, align 8
  call void @tl_add_parameter(ptr @key_str.261, ptr %b737)
  %m738 = getelementptr inbounds %Block, ptr %sub_ptr699, i32 0, i32 3
  %sub_ptr739 = load ptr, ptr %m738, align 8
  %f740 = getelementptr inbounds %MLP, ptr %sub_ptr739, i32 0, i32 0
  %sub_ptr741 = load ptr, ptr %f740, align 8
  %W742 = getelementptr inbounds %Linear, ptr %sub_ptr741, i32 0, i32 0
  %W743 = load ptr, ptr %W742, align 8
  call void @tl_add_parameter(ptr @key_str.262, ptr %W743)
  %b744 = getelementptr inbounds %Linear, ptr %sub_ptr741, i32 0, i32 1
  %b745 = load ptr, ptr %b744, align 8
  call void @tl_add_parameter(ptr @key_str.263, ptr %b745)
  %p746 = getelementptr inbounds %MLP, ptr %sub_ptr739, i32 0, i32 1
  %sub_ptr747 = load ptr, ptr %p746, align 8
  %W748 = getelementptr inbounds %Linear, ptr %sub_ptr747, i32 0, i32 0
  %W749 = load ptr, ptr %W748, align 8
  call void @tl_add_parameter(ptr @key_str.264, ptr %W749)
  %b750 = getelementptr inbounds %Linear, ptr %sub_ptr747, i32 0, i32 1
  %b751 = load ptr, ptr %b750, align 8
  call void @tl_add_parameter(ptr @key_str.265, ptr %b751)
  %b2752 = getelementptr inbounds %GPTHeavy, ptr %model689, i32 0, i32 3
  %sub_ptr753 = load ptr, ptr %b2752, align 8
  %l1754 = getelementptr inbounds %Block, ptr %sub_ptr753, i32 0, i32 0
  %sub_ptr755 = load ptr, ptr %l1754, align 8
  %w756 = getelementptr inbounds %LayerNorm, ptr %sub_ptr755, i32 0, i32 0
  %w757 = load ptr, ptr %w756, align 8
  call void @tl_add_parameter(ptr @key_str.266, ptr %w757)
  %b758 = getelementptr inbounds %LayerNorm, ptr %sub_ptr755, i32 0, i32 1
  %b759 = load ptr, ptr %b758, align 8
  call void @tl_add_parameter(ptr @key_str.267, ptr %b759)
  %a760 = getelementptr inbounds %Block, ptr %sub_ptr753, i32 0, i32 1
  %sub_ptr761 = load ptr, ptr %a760, align 8
  %q_proj762 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr761, i32 0, i32 0
  %sub_ptr763 = load ptr, ptr %q_proj762, align 8
  %W764 = getelementptr inbounds %Linear, ptr %sub_ptr763, i32 0, i32 0
  %W765 = load ptr, ptr %W764, align 8
  call void @tl_add_parameter(ptr @key_str.268, ptr %W765)
  %b766 = getelementptr inbounds %Linear, ptr %sub_ptr763, i32 0, i32 1
  %b767 = load ptr, ptr %b766, align 8
  call void @tl_add_parameter(ptr @key_str.269, ptr %b767)
  %k_proj768 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr761, i32 0, i32 1
  %sub_ptr769 = load ptr, ptr %k_proj768, align 8
  %W770 = getelementptr inbounds %Linear, ptr %sub_ptr769, i32 0, i32 0
  %W771 = load ptr, ptr %W770, align 8
  call void @tl_add_parameter(ptr @key_str.270, ptr %W771)
  %b772 = getelementptr inbounds %Linear, ptr %sub_ptr769, i32 0, i32 1
  %b773 = load ptr, ptr %b772, align 8
  call void @tl_add_parameter(ptr @key_str.271, ptr %b773)
  %v_proj774 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr761, i32 0, i32 2
  %sub_ptr775 = load ptr, ptr %v_proj774, align 8
  %W776 = getelementptr inbounds %Linear, ptr %sub_ptr775, i32 0, i32 0
  %W777 = load ptr, ptr %W776, align 8
  call void @tl_add_parameter(ptr @key_str.272, ptr %W777)
  %b778 = getelementptr inbounds %Linear, ptr %sub_ptr775, i32 0, i32 1
  %b779 = load ptr, ptr %b778, align 8
  call void @tl_add_parameter(ptr @key_str.273, ptr %b779)
  %p_proj780 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr761, i32 0, i32 3
  %sub_ptr781 = load ptr, ptr %p_proj780, align 8
  %W782 = getelementptr inbounds %Linear, ptr %sub_ptr781, i32 0, i32 0
  %W783 = load ptr, ptr %W782, align 8
  call void @tl_add_parameter(ptr @key_str.274, ptr %W783)
  %b784 = getelementptr inbounds %Linear, ptr %sub_ptr781, i32 0, i32 1
  %b785 = load ptr, ptr %b784, align 8
  call void @tl_add_parameter(ptr @key_str.275, ptr %b785)
  %l2786 = getelementptr inbounds %Block, ptr %sub_ptr753, i32 0, i32 2
  %sub_ptr787 = load ptr, ptr %l2786, align 8
  %w788 = getelementptr inbounds %LayerNorm, ptr %sub_ptr787, i32 0, i32 0
  %w789 = load ptr, ptr %w788, align 8
  call void @tl_add_parameter(ptr @key_str.276, ptr %w789)
  %b790 = getelementptr inbounds %LayerNorm, ptr %sub_ptr787, i32 0, i32 1
  %b791 = load ptr, ptr %b790, align 8
  call void @tl_add_parameter(ptr @key_str.277, ptr %b791)
  %m792 = getelementptr inbounds %Block, ptr %sub_ptr753, i32 0, i32 3
  %sub_ptr793 = load ptr, ptr %m792, align 8
  %f794 = getelementptr inbounds %MLP, ptr %sub_ptr793, i32 0, i32 0
  %sub_ptr795 = load ptr, ptr %f794, align 8
  %W796 = getelementptr inbounds %Linear, ptr %sub_ptr795, i32 0, i32 0
  %W797 = load ptr, ptr %W796, align 8
  call void @tl_add_parameter(ptr @key_str.278, ptr %W797)
  %b798 = getelementptr inbounds %Linear, ptr %sub_ptr795, i32 0, i32 1
  %b799 = load ptr, ptr %b798, align 8
  call void @tl_add_parameter(ptr @key_str.279, ptr %b799)
  %p800 = getelementptr inbounds %MLP, ptr %sub_ptr793, i32 0, i32 1
  %sub_ptr801 = load ptr, ptr %p800, align 8
  %W802 = getelementptr inbounds %Linear, ptr %sub_ptr801, i32 0, i32 0
  %W803 = load ptr, ptr %W802, align 8
  call void @tl_add_parameter(ptr @key_str.280, ptr %W803)
  %b804 = getelementptr inbounds %Linear, ptr %sub_ptr801, i32 0, i32 1
  %b805 = load ptr, ptr %b804, align 8
  call void @tl_add_parameter(ptr @key_str.281, ptr %b805)
  %b3806 = getelementptr inbounds %GPTHeavy, ptr %model689, i32 0, i32 4
  %sub_ptr807 = load ptr, ptr %b3806, align 8
  %l1808 = getelementptr inbounds %Block, ptr %sub_ptr807, i32 0, i32 0
  %sub_ptr809 = load ptr, ptr %l1808, align 8
  %w810 = getelementptr inbounds %LayerNorm, ptr %sub_ptr809, i32 0, i32 0
  %w811 = load ptr, ptr %w810, align 8
  call void @tl_add_parameter(ptr @key_str.282, ptr %w811)
  %b812 = getelementptr inbounds %LayerNorm, ptr %sub_ptr809, i32 0, i32 1
  %b813 = load ptr, ptr %b812, align 8
  call void @tl_add_parameter(ptr @key_str.283, ptr %b813)
  %a814 = getelementptr inbounds %Block, ptr %sub_ptr807, i32 0, i32 1
  %sub_ptr815 = load ptr, ptr %a814, align 8
  %q_proj816 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr815, i32 0, i32 0
  %sub_ptr817 = load ptr, ptr %q_proj816, align 8
  %W818 = getelementptr inbounds %Linear, ptr %sub_ptr817, i32 0, i32 0
  %W819 = load ptr, ptr %W818, align 8
  call void @tl_add_parameter(ptr @key_str.284, ptr %W819)
  %b820 = getelementptr inbounds %Linear, ptr %sub_ptr817, i32 0, i32 1
  %b821 = load ptr, ptr %b820, align 8
  call void @tl_add_parameter(ptr @key_str.285, ptr %b821)
  %k_proj822 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr815, i32 0, i32 1
  %sub_ptr823 = load ptr, ptr %k_proj822, align 8
  %W824 = getelementptr inbounds %Linear, ptr %sub_ptr823, i32 0, i32 0
  %W825 = load ptr, ptr %W824, align 8
  call void @tl_add_parameter(ptr @key_str.286, ptr %W825)
  %b826 = getelementptr inbounds %Linear, ptr %sub_ptr823, i32 0, i32 1
  %b827 = load ptr, ptr %b826, align 8
  call void @tl_add_parameter(ptr @key_str.287, ptr %b827)
  %v_proj828 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr815, i32 0, i32 2
  %sub_ptr829 = load ptr, ptr %v_proj828, align 8
  %W830 = getelementptr inbounds %Linear, ptr %sub_ptr829, i32 0, i32 0
  %W831 = load ptr, ptr %W830, align 8
  call void @tl_add_parameter(ptr @key_str.288, ptr %W831)
  %b832 = getelementptr inbounds %Linear, ptr %sub_ptr829, i32 0, i32 1
  %b833 = load ptr, ptr %b832, align 8
  call void @tl_add_parameter(ptr @key_str.289, ptr %b833)
  %p_proj834 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr815, i32 0, i32 3
  %sub_ptr835 = load ptr, ptr %p_proj834, align 8
  %W836 = getelementptr inbounds %Linear, ptr %sub_ptr835, i32 0, i32 0
  %W837 = load ptr, ptr %W836, align 8
  call void @tl_add_parameter(ptr @key_str.290, ptr %W837)
  %b838 = getelementptr inbounds %Linear, ptr %sub_ptr835, i32 0, i32 1
  %b839 = load ptr, ptr %b838, align 8
  call void @tl_add_parameter(ptr @key_str.291, ptr %b839)
  %l2840 = getelementptr inbounds %Block, ptr %sub_ptr807, i32 0, i32 2
  %sub_ptr841 = load ptr, ptr %l2840, align 8
  %w842 = getelementptr inbounds %LayerNorm, ptr %sub_ptr841, i32 0, i32 0
  %w843 = load ptr, ptr %w842, align 8
  call void @tl_add_parameter(ptr @key_str.292, ptr %w843)
  %b844 = getelementptr inbounds %LayerNorm, ptr %sub_ptr841, i32 0, i32 1
  %b845 = load ptr, ptr %b844, align 8
  call void @tl_add_parameter(ptr @key_str.293, ptr %b845)
  %m846 = getelementptr inbounds %Block, ptr %sub_ptr807, i32 0, i32 3
  %sub_ptr847 = load ptr, ptr %m846, align 8
  %f848 = getelementptr inbounds %MLP, ptr %sub_ptr847, i32 0, i32 0
  %sub_ptr849 = load ptr, ptr %f848, align 8
  %W850 = getelementptr inbounds %Linear, ptr %sub_ptr849, i32 0, i32 0
  %W851 = load ptr, ptr %W850, align 8
  call void @tl_add_parameter(ptr @key_str.294, ptr %W851)
  %b852 = getelementptr inbounds %Linear, ptr %sub_ptr849, i32 0, i32 1
  %b853 = load ptr, ptr %b852, align 8
  call void @tl_add_parameter(ptr @key_str.295, ptr %b853)
  %p854 = getelementptr inbounds %MLP, ptr %sub_ptr847, i32 0, i32 1
  %sub_ptr855 = load ptr, ptr %p854, align 8
  %W856 = getelementptr inbounds %Linear, ptr %sub_ptr855, i32 0, i32 0
  %W857 = load ptr, ptr %W856, align 8
  call void @tl_add_parameter(ptr @key_str.296, ptr %W857)
  %b858 = getelementptr inbounds %Linear, ptr %sub_ptr855, i32 0, i32 1
  %b859 = load ptr, ptr %b858, align 8
  call void @tl_add_parameter(ptr @key_str.297, ptr %b859)
  %b4860 = getelementptr inbounds %GPTHeavy, ptr %model689, i32 0, i32 5
  %sub_ptr861 = load ptr, ptr %b4860, align 8
  %l1862 = getelementptr inbounds %Block, ptr %sub_ptr861, i32 0, i32 0
  %sub_ptr863 = load ptr, ptr %l1862, align 8
  %w864 = getelementptr inbounds %LayerNorm, ptr %sub_ptr863, i32 0, i32 0
  %w865 = load ptr, ptr %w864, align 8
  call void @tl_add_parameter(ptr @key_str.298, ptr %w865)
  %b866 = getelementptr inbounds %LayerNorm, ptr %sub_ptr863, i32 0, i32 1
  %b867 = load ptr, ptr %b866, align 8
  call void @tl_add_parameter(ptr @key_str.299, ptr %b867)
  %a868 = getelementptr inbounds %Block, ptr %sub_ptr861, i32 0, i32 1
  %sub_ptr869 = load ptr, ptr %a868, align 8
  %q_proj870 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr869, i32 0, i32 0
  %sub_ptr871 = load ptr, ptr %q_proj870, align 8
  %W872 = getelementptr inbounds %Linear, ptr %sub_ptr871, i32 0, i32 0
  %W873 = load ptr, ptr %W872, align 8
  call void @tl_add_parameter(ptr @key_str.300, ptr %W873)
  %b874 = getelementptr inbounds %Linear, ptr %sub_ptr871, i32 0, i32 1
  %b875 = load ptr, ptr %b874, align 8
  call void @tl_add_parameter(ptr @key_str.301, ptr %b875)
  %k_proj876 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr869, i32 0, i32 1
  %sub_ptr877 = load ptr, ptr %k_proj876, align 8
  %W878 = getelementptr inbounds %Linear, ptr %sub_ptr877, i32 0, i32 0
  %W879 = load ptr, ptr %W878, align 8
  call void @tl_add_parameter(ptr @key_str.302, ptr %W879)
  %b880 = getelementptr inbounds %Linear, ptr %sub_ptr877, i32 0, i32 1
  %b881 = load ptr, ptr %b880, align 8
  call void @tl_add_parameter(ptr @key_str.303, ptr %b881)
  %v_proj882 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr869, i32 0, i32 2
  %sub_ptr883 = load ptr, ptr %v_proj882, align 8
  %W884 = getelementptr inbounds %Linear, ptr %sub_ptr883, i32 0, i32 0
  %W885 = load ptr, ptr %W884, align 8
  call void @tl_add_parameter(ptr @key_str.304, ptr %W885)
  %b886 = getelementptr inbounds %Linear, ptr %sub_ptr883, i32 0, i32 1
  %b887 = load ptr, ptr %b886, align 8
  call void @tl_add_parameter(ptr @key_str.305, ptr %b887)
  %p_proj888 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr869, i32 0, i32 3
  %sub_ptr889 = load ptr, ptr %p_proj888, align 8
  %W890 = getelementptr inbounds %Linear, ptr %sub_ptr889, i32 0, i32 0
  %W891 = load ptr, ptr %W890, align 8
  call void @tl_add_parameter(ptr @key_str.306, ptr %W891)
  %b892 = getelementptr inbounds %Linear, ptr %sub_ptr889, i32 0, i32 1
  %b893 = load ptr, ptr %b892, align 8
  call void @tl_add_parameter(ptr @key_str.307, ptr %b893)
  %l2894 = getelementptr inbounds %Block, ptr %sub_ptr861, i32 0, i32 2
  %sub_ptr895 = load ptr, ptr %l2894, align 8
  %w896 = getelementptr inbounds %LayerNorm, ptr %sub_ptr895, i32 0, i32 0
  %w897 = load ptr, ptr %w896, align 8
  call void @tl_add_parameter(ptr @key_str.308, ptr %w897)
  %b898 = getelementptr inbounds %LayerNorm, ptr %sub_ptr895, i32 0, i32 1
  %b899 = load ptr, ptr %b898, align 8
  call void @tl_add_parameter(ptr @key_str.309, ptr %b899)
  %m900 = getelementptr inbounds %Block, ptr %sub_ptr861, i32 0, i32 3
  %sub_ptr901 = load ptr, ptr %m900, align 8
  %f902 = getelementptr inbounds %MLP, ptr %sub_ptr901, i32 0, i32 0
  %sub_ptr903 = load ptr, ptr %f902, align 8
  %W904 = getelementptr inbounds %Linear, ptr %sub_ptr903, i32 0, i32 0
  %W905 = load ptr, ptr %W904, align 8
  call void @tl_add_parameter(ptr @key_str.310, ptr %W905)
  %b906 = getelementptr inbounds %Linear, ptr %sub_ptr903, i32 0, i32 1
  %b907 = load ptr, ptr %b906, align 8
  call void @tl_add_parameter(ptr @key_str.311, ptr %b907)
  %p908 = getelementptr inbounds %MLP, ptr %sub_ptr901, i32 0, i32 1
  %sub_ptr909 = load ptr, ptr %p908, align 8
  %W910 = getelementptr inbounds %Linear, ptr %sub_ptr909, i32 0, i32 0
  %W911 = load ptr, ptr %W910, align 8
  call void @tl_add_parameter(ptr @key_str.312, ptr %W911)
  %b912 = getelementptr inbounds %Linear, ptr %sub_ptr909, i32 0, i32 1
  %b913 = load ptr, ptr %b912, align 8
  call void @tl_add_parameter(ptr @key_str.313, ptr %b913)
  %l914 = getelementptr inbounds %GPTHeavy, ptr %model689, i32 0, i32 6
  %sub_ptr915 = load ptr, ptr %l914, align 8
  %w916 = getelementptr inbounds %LayerNorm, ptr %sub_ptr915, i32 0, i32 0
  %w917 = load ptr, ptr %w916, align 8
  call void @tl_add_parameter(ptr @key_str.314, ptr %w917)
  %b918 = getelementptr inbounds %LayerNorm, ptr %sub_ptr915, i32 0, i32 1
  %b919 = load ptr, ptr %b918, align 8
  call void @tl_add_parameter(ptr @key_str.315, ptr %b919)
  %h920 = getelementptr inbounds %GPTHeavy, ptr %model689, i32 0, i32 7
  %sub_ptr921 = load ptr, ptr %h920, align 8
  %W922 = getelementptr inbounds %Linear, ptr %sub_ptr921, i32 0, i32 0
  %W923 = load ptr, ptr %W922, align 8
  call void @tl_add_parameter(ptr @key_str.316, ptr %W923)
  %b924 = getelementptr inbounds %Linear, ptr %sub_ptr921, i32 0, i32 1
  %b925 = load ptr, ptr %b924, align 8
  call void @tl_add_parameter(ptr @key_str.317, ptr %b925)
  call void @tl_save_all_params(ptr @str_literal.318)
  %struct_to_free = load ptr, ptr %model, align 8
  call void @tl_mem_unregister(ptr %struct_to_free)
  call void @tl_mem_exit_scope()
  ret void

free_struct:                                      ; preds = %for_body
  br label %continue_after_free

continue_after_free:                              ; preds = %free_struct, %for_body
  call void @tl_mem_unregister(ptr %call_tmp)
  %unreg_field_0 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 0
  %field_val = load ptr, ptr %unreg_field_0, align 8
  call void @tl_mem_unregister(ptr %field_val)
  %unreg_field_0223 = getelementptr inbounds %Embedding, ptr %field_val, i32 0, i32 0
  %field_val224 = load ptr, ptr %unreg_field_0223, align 8
  call void @tl_mem_unregister(ptr %field_val224)
  %unreg_field_1 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 1
  %field_val225 = load ptr, ptr %unreg_field_1, align 8
  call void @tl_mem_unregister(ptr %field_val225)
  %unreg_field_0226 = getelementptr inbounds %Embedding, ptr %field_val225, i32 0, i32 0
  %field_val227 = load ptr, ptr %unreg_field_0226, align 8
  call void @tl_mem_unregister(ptr %field_val227)
  %unreg_field_2 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 2
  %field_val228 = load ptr, ptr %unreg_field_2, align 8
  call void @tl_mem_unregister(ptr %field_val228)
  %unreg_field_0229 = getelementptr inbounds %Block, ptr %field_val228, i32 0, i32 0
  %field_val230 = load ptr, ptr %unreg_field_0229, align 8
  call void @tl_mem_unregister(ptr %field_val230)
  %unreg_field_0231 = getelementptr inbounds %LayerNorm, ptr %field_val230, i32 0, i32 0
  %field_val232 = load ptr, ptr %unreg_field_0231, align 8
  call void @tl_mem_unregister(ptr %field_val232)
  %unreg_field_1233 = getelementptr inbounds %LayerNorm, ptr %field_val230, i32 0, i32 1
  %field_val234 = load ptr, ptr %unreg_field_1233, align 8
  call void @tl_mem_unregister(ptr %field_val234)
  %unreg_field_1235 = getelementptr inbounds %Block, ptr %field_val228, i32 0, i32 1
  %field_val236 = load ptr, ptr %unreg_field_1235, align 8
  call void @tl_mem_unregister(ptr %field_val236)
  %unreg_field_0237 = getelementptr inbounds %CausalSelfAttention, ptr %field_val236, i32 0, i32 0
  %field_val238 = load ptr, ptr %unreg_field_0237, align 8
  call void @tl_mem_unregister(ptr %field_val238)
  %unreg_field_0239 = getelementptr inbounds %Linear, ptr %field_val238, i32 0, i32 0
  %field_val240 = load ptr, ptr %unreg_field_0239, align 8
  call void @tl_mem_unregister(ptr %field_val240)
  %unreg_field_1241 = getelementptr inbounds %Linear, ptr %field_val238, i32 0, i32 1
  %field_val242 = load ptr, ptr %unreg_field_1241, align 8
  call void @tl_mem_unregister(ptr %field_val242)
  %unreg_field_1243 = getelementptr inbounds %CausalSelfAttention, ptr %field_val236, i32 0, i32 1
  %field_val244 = load ptr, ptr %unreg_field_1243, align 8
  call void @tl_mem_unregister(ptr %field_val244)
  %unreg_field_0245 = getelementptr inbounds %Linear, ptr %field_val244, i32 0, i32 0
  %field_val246 = load ptr, ptr %unreg_field_0245, align 8
  call void @tl_mem_unregister(ptr %field_val246)
  %unreg_field_1247 = getelementptr inbounds %Linear, ptr %field_val244, i32 0, i32 1
  %field_val248 = load ptr, ptr %unreg_field_1247, align 8
  call void @tl_mem_unregister(ptr %field_val248)
  %unreg_field_2249 = getelementptr inbounds %CausalSelfAttention, ptr %field_val236, i32 0, i32 2
  %field_val250 = load ptr, ptr %unreg_field_2249, align 8
  call void @tl_mem_unregister(ptr %field_val250)
  %unreg_field_0251 = getelementptr inbounds %Linear, ptr %field_val250, i32 0, i32 0
  %field_val252 = load ptr, ptr %unreg_field_0251, align 8
  call void @tl_mem_unregister(ptr %field_val252)
  %unreg_field_1253 = getelementptr inbounds %Linear, ptr %field_val250, i32 0, i32 1
  %field_val254 = load ptr, ptr %unreg_field_1253, align 8
  call void @tl_mem_unregister(ptr %field_val254)
  %unreg_field_3 = getelementptr inbounds %CausalSelfAttention, ptr %field_val236, i32 0, i32 3
  %field_val255 = load ptr, ptr %unreg_field_3, align 8
  call void @tl_mem_unregister(ptr %field_val255)
  %unreg_field_0256 = getelementptr inbounds %Linear, ptr %field_val255, i32 0, i32 0
  %field_val257 = load ptr, ptr %unreg_field_0256, align 8
  call void @tl_mem_unregister(ptr %field_val257)
  %unreg_field_1258 = getelementptr inbounds %Linear, ptr %field_val255, i32 0, i32 1
  %field_val259 = load ptr, ptr %unreg_field_1258, align 8
  call void @tl_mem_unregister(ptr %field_val259)
  %unreg_field_2260 = getelementptr inbounds %Block, ptr %field_val228, i32 0, i32 2
  %field_val261 = load ptr, ptr %unreg_field_2260, align 8
  call void @tl_mem_unregister(ptr %field_val261)
  %unreg_field_0262 = getelementptr inbounds %LayerNorm, ptr %field_val261, i32 0, i32 0
  %field_val263 = load ptr, ptr %unreg_field_0262, align 8
  call void @tl_mem_unregister(ptr %field_val263)
  %unreg_field_1264 = getelementptr inbounds %LayerNorm, ptr %field_val261, i32 0, i32 1
  %field_val265 = load ptr, ptr %unreg_field_1264, align 8
  call void @tl_mem_unregister(ptr %field_val265)
  %unreg_field_3266 = getelementptr inbounds %Block, ptr %field_val228, i32 0, i32 3
  %field_val267 = load ptr, ptr %unreg_field_3266, align 8
  call void @tl_mem_unregister(ptr %field_val267)
  %unreg_field_0268 = getelementptr inbounds %MLP, ptr %field_val267, i32 0, i32 0
  %field_val269 = load ptr, ptr %unreg_field_0268, align 8
  call void @tl_mem_unregister(ptr %field_val269)
  %unreg_field_0270 = getelementptr inbounds %Linear, ptr %field_val269, i32 0, i32 0
  %field_val271 = load ptr, ptr %unreg_field_0270, align 8
  call void @tl_mem_unregister(ptr %field_val271)
  %unreg_field_1272 = getelementptr inbounds %Linear, ptr %field_val269, i32 0, i32 1
  %field_val273 = load ptr, ptr %unreg_field_1272, align 8
  call void @tl_mem_unregister(ptr %field_val273)
  %unreg_field_1274 = getelementptr inbounds %MLP, ptr %field_val267, i32 0, i32 1
  %field_val275 = load ptr, ptr %unreg_field_1274, align 8
  call void @tl_mem_unregister(ptr %field_val275)
  %unreg_field_0276 = getelementptr inbounds %Linear, ptr %field_val275, i32 0, i32 0
  %field_val277 = load ptr, ptr %unreg_field_0276, align 8
  call void @tl_mem_unregister(ptr %field_val277)
  %unreg_field_1278 = getelementptr inbounds %Linear, ptr %field_val275, i32 0, i32 1
  %field_val279 = load ptr, ptr %unreg_field_1278, align 8
  call void @tl_mem_unregister(ptr %field_val279)
  %unreg_field_3280 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 3
  %field_val281 = load ptr, ptr %unreg_field_3280, align 8
  call void @tl_mem_unregister(ptr %field_val281)
  %unreg_field_0282 = getelementptr inbounds %Block, ptr %field_val281, i32 0, i32 0
  %field_val283 = load ptr, ptr %unreg_field_0282, align 8
  call void @tl_mem_unregister(ptr %field_val283)
  %unreg_field_0284 = getelementptr inbounds %LayerNorm, ptr %field_val283, i32 0, i32 0
  %field_val285 = load ptr, ptr %unreg_field_0284, align 8
  call void @tl_mem_unregister(ptr %field_val285)
  %unreg_field_1286 = getelementptr inbounds %LayerNorm, ptr %field_val283, i32 0, i32 1
  %field_val287 = load ptr, ptr %unreg_field_1286, align 8
  call void @tl_mem_unregister(ptr %field_val287)
  %unreg_field_1288 = getelementptr inbounds %Block, ptr %field_val281, i32 0, i32 1
  %field_val289 = load ptr, ptr %unreg_field_1288, align 8
  call void @tl_mem_unregister(ptr %field_val289)
  %unreg_field_0290 = getelementptr inbounds %CausalSelfAttention, ptr %field_val289, i32 0, i32 0
  %field_val291 = load ptr, ptr %unreg_field_0290, align 8
  call void @tl_mem_unregister(ptr %field_val291)
  %unreg_field_0292 = getelementptr inbounds %Linear, ptr %field_val291, i32 0, i32 0
  %field_val293 = load ptr, ptr %unreg_field_0292, align 8
  call void @tl_mem_unregister(ptr %field_val293)
  %unreg_field_1294 = getelementptr inbounds %Linear, ptr %field_val291, i32 0, i32 1
  %field_val295 = load ptr, ptr %unreg_field_1294, align 8
  call void @tl_mem_unregister(ptr %field_val295)
  %unreg_field_1296 = getelementptr inbounds %CausalSelfAttention, ptr %field_val289, i32 0, i32 1
  %field_val297 = load ptr, ptr %unreg_field_1296, align 8
  call void @tl_mem_unregister(ptr %field_val297)
  %unreg_field_0298 = getelementptr inbounds %Linear, ptr %field_val297, i32 0, i32 0
  %field_val299 = load ptr, ptr %unreg_field_0298, align 8
  call void @tl_mem_unregister(ptr %field_val299)
  %unreg_field_1300 = getelementptr inbounds %Linear, ptr %field_val297, i32 0, i32 1
  %field_val301 = load ptr, ptr %unreg_field_1300, align 8
  call void @tl_mem_unregister(ptr %field_val301)
  %unreg_field_2302 = getelementptr inbounds %CausalSelfAttention, ptr %field_val289, i32 0, i32 2
  %field_val303 = load ptr, ptr %unreg_field_2302, align 8
  call void @tl_mem_unregister(ptr %field_val303)
  %unreg_field_0304 = getelementptr inbounds %Linear, ptr %field_val303, i32 0, i32 0
  %field_val305 = load ptr, ptr %unreg_field_0304, align 8
  call void @tl_mem_unregister(ptr %field_val305)
  %unreg_field_1306 = getelementptr inbounds %Linear, ptr %field_val303, i32 0, i32 1
  %field_val307 = load ptr, ptr %unreg_field_1306, align 8
  call void @tl_mem_unregister(ptr %field_val307)
  %unreg_field_3308 = getelementptr inbounds %CausalSelfAttention, ptr %field_val289, i32 0, i32 3
  %field_val309 = load ptr, ptr %unreg_field_3308, align 8
  call void @tl_mem_unregister(ptr %field_val309)
  %unreg_field_0310 = getelementptr inbounds %Linear, ptr %field_val309, i32 0, i32 0
  %field_val311 = load ptr, ptr %unreg_field_0310, align 8
  call void @tl_mem_unregister(ptr %field_val311)
  %unreg_field_1312 = getelementptr inbounds %Linear, ptr %field_val309, i32 0, i32 1
  %field_val313 = load ptr, ptr %unreg_field_1312, align 8
  call void @tl_mem_unregister(ptr %field_val313)
  %unreg_field_2314 = getelementptr inbounds %Block, ptr %field_val281, i32 0, i32 2
  %field_val315 = load ptr, ptr %unreg_field_2314, align 8
  call void @tl_mem_unregister(ptr %field_val315)
  %unreg_field_0316 = getelementptr inbounds %LayerNorm, ptr %field_val315, i32 0, i32 0
  %field_val317 = load ptr, ptr %unreg_field_0316, align 8
  call void @tl_mem_unregister(ptr %field_val317)
  %unreg_field_1318 = getelementptr inbounds %LayerNorm, ptr %field_val315, i32 0, i32 1
  %field_val319 = load ptr, ptr %unreg_field_1318, align 8
  call void @tl_mem_unregister(ptr %field_val319)
  %unreg_field_3320 = getelementptr inbounds %Block, ptr %field_val281, i32 0, i32 3
  %field_val321 = load ptr, ptr %unreg_field_3320, align 8
  call void @tl_mem_unregister(ptr %field_val321)
  %unreg_field_0322 = getelementptr inbounds %MLP, ptr %field_val321, i32 0, i32 0
  %field_val323 = load ptr, ptr %unreg_field_0322, align 8
  call void @tl_mem_unregister(ptr %field_val323)
  %unreg_field_0324 = getelementptr inbounds %Linear, ptr %field_val323, i32 0, i32 0
  %field_val325 = load ptr, ptr %unreg_field_0324, align 8
  call void @tl_mem_unregister(ptr %field_val325)
  %unreg_field_1326 = getelementptr inbounds %Linear, ptr %field_val323, i32 0, i32 1
  %field_val327 = load ptr, ptr %unreg_field_1326, align 8
  call void @tl_mem_unregister(ptr %field_val327)
  %unreg_field_1328 = getelementptr inbounds %MLP, ptr %field_val321, i32 0, i32 1
  %field_val329 = load ptr, ptr %unreg_field_1328, align 8
  call void @tl_mem_unregister(ptr %field_val329)
  %unreg_field_0330 = getelementptr inbounds %Linear, ptr %field_val329, i32 0, i32 0
  %field_val331 = load ptr, ptr %unreg_field_0330, align 8
  call void @tl_mem_unregister(ptr %field_val331)
  %unreg_field_1332 = getelementptr inbounds %Linear, ptr %field_val329, i32 0, i32 1
  %field_val333 = load ptr, ptr %unreg_field_1332, align 8
  call void @tl_mem_unregister(ptr %field_val333)
  %unreg_field_4 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 4
  %field_val334 = load ptr, ptr %unreg_field_4, align 8
  call void @tl_mem_unregister(ptr %field_val334)
  %unreg_field_0335 = getelementptr inbounds %Block, ptr %field_val334, i32 0, i32 0
  %field_val336 = load ptr, ptr %unreg_field_0335, align 8
  call void @tl_mem_unregister(ptr %field_val336)
  %unreg_field_0337 = getelementptr inbounds %LayerNorm, ptr %field_val336, i32 0, i32 0
  %field_val338 = load ptr, ptr %unreg_field_0337, align 8
  call void @tl_mem_unregister(ptr %field_val338)
  %unreg_field_1339 = getelementptr inbounds %LayerNorm, ptr %field_val336, i32 0, i32 1
  %field_val340 = load ptr, ptr %unreg_field_1339, align 8
  call void @tl_mem_unregister(ptr %field_val340)
  %unreg_field_1341 = getelementptr inbounds %Block, ptr %field_val334, i32 0, i32 1
  %field_val342 = load ptr, ptr %unreg_field_1341, align 8
  call void @tl_mem_unregister(ptr %field_val342)
  %unreg_field_0343 = getelementptr inbounds %CausalSelfAttention, ptr %field_val342, i32 0, i32 0
  %field_val344 = load ptr, ptr %unreg_field_0343, align 8
  call void @tl_mem_unregister(ptr %field_val344)
  %unreg_field_0345 = getelementptr inbounds %Linear, ptr %field_val344, i32 0, i32 0
  %field_val346 = load ptr, ptr %unreg_field_0345, align 8
  call void @tl_mem_unregister(ptr %field_val346)
  %unreg_field_1347 = getelementptr inbounds %Linear, ptr %field_val344, i32 0, i32 1
  %field_val348 = load ptr, ptr %unreg_field_1347, align 8
  call void @tl_mem_unregister(ptr %field_val348)
  %unreg_field_1349 = getelementptr inbounds %CausalSelfAttention, ptr %field_val342, i32 0, i32 1
  %field_val350 = load ptr, ptr %unreg_field_1349, align 8
  call void @tl_mem_unregister(ptr %field_val350)
  %unreg_field_0351 = getelementptr inbounds %Linear, ptr %field_val350, i32 0, i32 0
  %field_val352 = load ptr, ptr %unreg_field_0351, align 8
  call void @tl_mem_unregister(ptr %field_val352)
  %unreg_field_1353 = getelementptr inbounds %Linear, ptr %field_val350, i32 0, i32 1
  %field_val354 = load ptr, ptr %unreg_field_1353, align 8
  call void @tl_mem_unregister(ptr %field_val354)
  %unreg_field_2355 = getelementptr inbounds %CausalSelfAttention, ptr %field_val342, i32 0, i32 2
  %field_val356 = load ptr, ptr %unreg_field_2355, align 8
  call void @tl_mem_unregister(ptr %field_val356)
  %unreg_field_0357 = getelementptr inbounds %Linear, ptr %field_val356, i32 0, i32 0
  %field_val358 = load ptr, ptr %unreg_field_0357, align 8
  call void @tl_mem_unregister(ptr %field_val358)
  %unreg_field_1359 = getelementptr inbounds %Linear, ptr %field_val356, i32 0, i32 1
  %field_val360 = load ptr, ptr %unreg_field_1359, align 8
  call void @tl_mem_unregister(ptr %field_val360)
  %unreg_field_3361 = getelementptr inbounds %CausalSelfAttention, ptr %field_val342, i32 0, i32 3
  %field_val362 = load ptr, ptr %unreg_field_3361, align 8
  call void @tl_mem_unregister(ptr %field_val362)
  %unreg_field_0363 = getelementptr inbounds %Linear, ptr %field_val362, i32 0, i32 0
  %field_val364 = load ptr, ptr %unreg_field_0363, align 8
  call void @tl_mem_unregister(ptr %field_val364)
  %unreg_field_1365 = getelementptr inbounds %Linear, ptr %field_val362, i32 0, i32 1
  %field_val366 = load ptr, ptr %unreg_field_1365, align 8
  call void @tl_mem_unregister(ptr %field_val366)
  %unreg_field_2367 = getelementptr inbounds %Block, ptr %field_val334, i32 0, i32 2
  %field_val368 = load ptr, ptr %unreg_field_2367, align 8
  call void @tl_mem_unregister(ptr %field_val368)
  %unreg_field_0369 = getelementptr inbounds %LayerNorm, ptr %field_val368, i32 0, i32 0
  %field_val370 = load ptr, ptr %unreg_field_0369, align 8
  call void @tl_mem_unregister(ptr %field_val370)
  %unreg_field_1371 = getelementptr inbounds %LayerNorm, ptr %field_val368, i32 0, i32 1
  %field_val372 = load ptr, ptr %unreg_field_1371, align 8
  call void @tl_mem_unregister(ptr %field_val372)
  %unreg_field_3373 = getelementptr inbounds %Block, ptr %field_val334, i32 0, i32 3
  %field_val374 = load ptr, ptr %unreg_field_3373, align 8
  call void @tl_mem_unregister(ptr %field_val374)
  %unreg_field_0375 = getelementptr inbounds %MLP, ptr %field_val374, i32 0, i32 0
  %field_val376 = load ptr, ptr %unreg_field_0375, align 8
  call void @tl_mem_unregister(ptr %field_val376)
  %unreg_field_0377 = getelementptr inbounds %Linear, ptr %field_val376, i32 0, i32 0
  %field_val378 = load ptr, ptr %unreg_field_0377, align 8
  call void @tl_mem_unregister(ptr %field_val378)
  %unreg_field_1379 = getelementptr inbounds %Linear, ptr %field_val376, i32 0, i32 1
  %field_val380 = load ptr, ptr %unreg_field_1379, align 8
  call void @tl_mem_unregister(ptr %field_val380)
  %unreg_field_1381 = getelementptr inbounds %MLP, ptr %field_val374, i32 0, i32 1
  %field_val382 = load ptr, ptr %unreg_field_1381, align 8
  call void @tl_mem_unregister(ptr %field_val382)
  %unreg_field_0383 = getelementptr inbounds %Linear, ptr %field_val382, i32 0, i32 0
  %field_val384 = load ptr, ptr %unreg_field_0383, align 8
  call void @tl_mem_unregister(ptr %field_val384)
  %unreg_field_1385 = getelementptr inbounds %Linear, ptr %field_val382, i32 0, i32 1
  %field_val386 = load ptr, ptr %unreg_field_1385, align 8
  call void @tl_mem_unregister(ptr %field_val386)
  %unreg_field_5 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 5
  %field_val387 = load ptr, ptr %unreg_field_5, align 8
  call void @tl_mem_unregister(ptr %field_val387)
  %unreg_field_0388 = getelementptr inbounds %Block, ptr %field_val387, i32 0, i32 0
  %field_val389 = load ptr, ptr %unreg_field_0388, align 8
  call void @tl_mem_unregister(ptr %field_val389)
  %unreg_field_0390 = getelementptr inbounds %LayerNorm, ptr %field_val389, i32 0, i32 0
  %field_val391 = load ptr, ptr %unreg_field_0390, align 8
  call void @tl_mem_unregister(ptr %field_val391)
  %unreg_field_1392 = getelementptr inbounds %LayerNorm, ptr %field_val389, i32 0, i32 1
  %field_val393 = load ptr, ptr %unreg_field_1392, align 8
  call void @tl_mem_unregister(ptr %field_val393)
  %unreg_field_1394 = getelementptr inbounds %Block, ptr %field_val387, i32 0, i32 1
  %field_val395 = load ptr, ptr %unreg_field_1394, align 8
  call void @tl_mem_unregister(ptr %field_val395)
  %unreg_field_0396 = getelementptr inbounds %CausalSelfAttention, ptr %field_val395, i32 0, i32 0
  %field_val397 = load ptr, ptr %unreg_field_0396, align 8
  call void @tl_mem_unregister(ptr %field_val397)
  %unreg_field_0398 = getelementptr inbounds %Linear, ptr %field_val397, i32 0, i32 0
  %field_val399 = load ptr, ptr %unreg_field_0398, align 8
  call void @tl_mem_unregister(ptr %field_val399)
  %unreg_field_1400 = getelementptr inbounds %Linear, ptr %field_val397, i32 0, i32 1
  %field_val401 = load ptr, ptr %unreg_field_1400, align 8
  call void @tl_mem_unregister(ptr %field_val401)
  %unreg_field_1402 = getelementptr inbounds %CausalSelfAttention, ptr %field_val395, i32 0, i32 1
  %field_val403 = load ptr, ptr %unreg_field_1402, align 8
  call void @tl_mem_unregister(ptr %field_val403)
  %unreg_field_0404 = getelementptr inbounds %Linear, ptr %field_val403, i32 0, i32 0
  %field_val405 = load ptr, ptr %unreg_field_0404, align 8
  call void @tl_mem_unregister(ptr %field_val405)
  %unreg_field_1406 = getelementptr inbounds %Linear, ptr %field_val403, i32 0, i32 1
  %field_val407 = load ptr, ptr %unreg_field_1406, align 8
  call void @tl_mem_unregister(ptr %field_val407)
  %unreg_field_2408 = getelementptr inbounds %CausalSelfAttention, ptr %field_val395, i32 0, i32 2
  %field_val409 = load ptr, ptr %unreg_field_2408, align 8
  call void @tl_mem_unregister(ptr %field_val409)
  %unreg_field_0410 = getelementptr inbounds %Linear, ptr %field_val409, i32 0, i32 0
  %field_val411 = load ptr, ptr %unreg_field_0410, align 8
  call void @tl_mem_unregister(ptr %field_val411)
  %unreg_field_1412 = getelementptr inbounds %Linear, ptr %field_val409, i32 0, i32 1
  %field_val413 = load ptr, ptr %unreg_field_1412, align 8
  call void @tl_mem_unregister(ptr %field_val413)
  %unreg_field_3414 = getelementptr inbounds %CausalSelfAttention, ptr %field_val395, i32 0, i32 3
  %field_val415 = load ptr, ptr %unreg_field_3414, align 8
  call void @tl_mem_unregister(ptr %field_val415)
  %unreg_field_0416 = getelementptr inbounds %Linear, ptr %field_val415, i32 0, i32 0
  %field_val417 = load ptr, ptr %unreg_field_0416, align 8
  call void @tl_mem_unregister(ptr %field_val417)
  %unreg_field_1418 = getelementptr inbounds %Linear, ptr %field_val415, i32 0, i32 1
  %field_val419 = load ptr, ptr %unreg_field_1418, align 8
  call void @tl_mem_unregister(ptr %field_val419)
  %unreg_field_2420 = getelementptr inbounds %Block, ptr %field_val387, i32 0, i32 2
  %field_val421 = load ptr, ptr %unreg_field_2420, align 8
  call void @tl_mem_unregister(ptr %field_val421)
  %unreg_field_0422 = getelementptr inbounds %LayerNorm, ptr %field_val421, i32 0, i32 0
  %field_val423 = load ptr, ptr %unreg_field_0422, align 8
  call void @tl_mem_unregister(ptr %field_val423)
  %unreg_field_1424 = getelementptr inbounds %LayerNorm, ptr %field_val421, i32 0, i32 1
  %field_val425 = load ptr, ptr %unreg_field_1424, align 8
  call void @tl_mem_unregister(ptr %field_val425)
  %unreg_field_3426 = getelementptr inbounds %Block, ptr %field_val387, i32 0, i32 3
  %field_val427 = load ptr, ptr %unreg_field_3426, align 8
  call void @tl_mem_unregister(ptr %field_val427)
  %unreg_field_0428 = getelementptr inbounds %MLP, ptr %field_val427, i32 0, i32 0
  %field_val429 = load ptr, ptr %unreg_field_0428, align 8
  call void @tl_mem_unregister(ptr %field_val429)
  %unreg_field_0430 = getelementptr inbounds %Linear, ptr %field_val429, i32 0, i32 0
  %field_val431 = load ptr, ptr %unreg_field_0430, align 8
  call void @tl_mem_unregister(ptr %field_val431)
  %unreg_field_1432 = getelementptr inbounds %Linear, ptr %field_val429, i32 0, i32 1
  %field_val433 = load ptr, ptr %unreg_field_1432, align 8
  call void @tl_mem_unregister(ptr %field_val433)
  %unreg_field_1434 = getelementptr inbounds %MLP, ptr %field_val427, i32 0, i32 1
  %field_val435 = load ptr, ptr %unreg_field_1434, align 8
  call void @tl_mem_unregister(ptr %field_val435)
  %unreg_field_0436 = getelementptr inbounds %Linear, ptr %field_val435, i32 0, i32 0
  %field_val437 = load ptr, ptr %unreg_field_0436, align 8
  call void @tl_mem_unregister(ptr %field_val437)
  %unreg_field_1438 = getelementptr inbounds %Linear, ptr %field_val435, i32 0, i32 1
  %field_val439 = load ptr, ptr %unreg_field_1438, align 8
  call void @tl_mem_unregister(ptr %field_val439)
  %unreg_field_6 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 6
  %field_val440 = load ptr, ptr %unreg_field_6, align 8
  call void @tl_mem_unregister(ptr %field_val440)
  %unreg_field_0441 = getelementptr inbounds %LayerNorm, ptr %field_val440, i32 0, i32 0
  %field_val442 = load ptr, ptr %unreg_field_0441, align 8
  call void @tl_mem_unregister(ptr %field_val442)
  %unreg_field_1443 = getelementptr inbounds %LayerNorm, ptr %field_val440, i32 0, i32 1
  %field_val444 = load ptr, ptr %unreg_field_1443, align 8
  call void @tl_mem_unregister(ptr %field_val444)
  %unreg_field_7 = getelementptr inbounds %GPTHeavy, ptr %call_tmp, i32 0, i32 7
  %field_val445 = load ptr, ptr %unreg_field_7, align 8
  call void @tl_mem_unregister(ptr %field_val445)
  %unreg_field_0446 = getelementptr inbounds %Linear, ptr %field_val445, i32 0, i32 0
  %field_val447 = load ptr, ptr %unreg_field_0446, align 8
  call void @tl_mem_unregister(ptr %field_val447)
  %unreg_field_1448 = getelementptr inbounds %Linear, ptr %field_val445, i32 0, i32 1
  %field_val449 = load ptr, ptr %unreg_field_1448, align 8
  call void @tl_mem_unregister(ptr %field_val449)
  call void @tl_mem_unregister(ptr %call_tmp)
  store ptr %call_tmp, ptr %model, align 8
  %epoch450 = load i64, ptr %epoch, align 8
  %epoch451 = load i64, ptr %epoch, align 8
  %divtmp = sdiv i64 %epoch451, 5
  %multmp = mul i64 %divtmp, 5
  %subtmp = sub i64 %epoch450, %multmp
  %eqtmp = icmp eq i64 %subtmp, 4
  br i1 %eqtmp, label %then, label %else

then:                                             ; preds = %continue_after_free
  call void @tl_mem_enter_scope()
  %model452 = load ptr, ptr %model, align 8
  %w453 = getelementptr inbounds %GPTHeavy, ptr %model452, i32 0, i32 0
  %sub_ptr454 = load ptr, ptr %w453, align 8
  %w455 = getelementptr inbounds %Embedding, ptr %sub_ptr454, i32 0, i32 0
  %w456 = load ptr, ptr %w455, align 8
  call void @tl_add_parameter(ptr @key_str.175, ptr %w456)
  %wp457 = getelementptr inbounds %GPTHeavy, ptr %model452, i32 0, i32 1
  %sub_ptr458 = load ptr, ptr %wp457, align 8
  %w459 = getelementptr inbounds %Embedding, ptr %sub_ptr458, i32 0, i32 0
  %w460 = load ptr, ptr %w459, align 8
  call void @tl_add_parameter(ptr @key_str.176, ptr %w460)
  %b1461 = getelementptr inbounds %GPTHeavy, ptr %model452, i32 0, i32 2
  %sub_ptr462 = load ptr, ptr %b1461, align 8
  %l1463 = getelementptr inbounds %Block, ptr %sub_ptr462, i32 0, i32 0
  %sub_ptr464 = load ptr, ptr %l1463, align 8
  %w465 = getelementptr inbounds %LayerNorm, ptr %sub_ptr464, i32 0, i32 0
  %w466 = load ptr, ptr %w465, align 8
  call void @tl_add_parameter(ptr @key_str.177, ptr %w466)
  %b467 = getelementptr inbounds %LayerNorm, ptr %sub_ptr464, i32 0, i32 1
  %b468 = load ptr, ptr %b467, align 8
  call void @tl_add_parameter(ptr @key_str.178, ptr %b468)
  %a469 = getelementptr inbounds %Block, ptr %sub_ptr462, i32 0, i32 1
  %sub_ptr470 = load ptr, ptr %a469, align 8
  %q_proj471 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr470, i32 0, i32 0
  %sub_ptr472 = load ptr, ptr %q_proj471, align 8
  %W473 = getelementptr inbounds %Linear, ptr %sub_ptr472, i32 0, i32 0
  %W474 = load ptr, ptr %W473, align 8
  call void @tl_add_parameter(ptr @key_str.179, ptr %W474)
  %b475 = getelementptr inbounds %Linear, ptr %sub_ptr472, i32 0, i32 1
  %b476 = load ptr, ptr %b475, align 8
  call void @tl_add_parameter(ptr @key_str.180, ptr %b476)
  %k_proj477 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr470, i32 0, i32 1
  %sub_ptr478 = load ptr, ptr %k_proj477, align 8
  %W479 = getelementptr inbounds %Linear, ptr %sub_ptr478, i32 0, i32 0
  %W480 = load ptr, ptr %W479, align 8
  call void @tl_add_parameter(ptr @key_str.181, ptr %W480)
  %b481 = getelementptr inbounds %Linear, ptr %sub_ptr478, i32 0, i32 1
  %b482 = load ptr, ptr %b481, align 8
  call void @tl_add_parameter(ptr @key_str.182, ptr %b482)
  %v_proj483 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr470, i32 0, i32 2
  %sub_ptr484 = load ptr, ptr %v_proj483, align 8
  %W485 = getelementptr inbounds %Linear, ptr %sub_ptr484, i32 0, i32 0
  %W486 = load ptr, ptr %W485, align 8
  call void @tl_add_parameter(ptr @key_str.183, ptr %W486)
  %b487 = getelementptr inbounds %Linear, ptr %sub_ptr484, i32 0, i32 1
  %b488 = load ptr, ptr %b487, align 8
  call void @tl_add_parameter(ptr @key_str.184, ptr %b488)
  %p_proj489 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr470, i32 0, i32 3
  %sub_ptr490 = load ptr, ptr %p_proj489, align 8
  %W491 = getelementptr inbounds %Linear, ptr %sub_ptr490, i32 0, i32 0
  %W492 = load ptr, ptr %W491, align 8
  call void @tl_add_parameter(ptr @key_str.185, ptr %W492)
  %b493 = getelementptr inbounds %Linear, ptr %sub_ptr490, i32 0, i32 1
  %b494 = load ptr, ptr %b493, align 8
  call void @tl_add_parameter(ptr @key_str.186, ptr %b494)
  %l2495 = getelementptr inbounds %Block, ptr %sub_ptr462, i32 0, i32 2
  %sub_ptr496 = load ptr, ptr %l2495, align 8
  %w497 = getelementptr inbounds %LayerNorm, ptr %sub_ptr496, i32 0, i32 0
  %w498 = load ptr, ptr %w497, align 8
  call void @tl_add_parameter(ptr @key_str.187, ptr %w498)
  %b499 = getelementptr inbounds %LayerNorm, ptr %sub_ptr496, i32 0, i32 1
  %b500 = load ptr, ptr %b499, align 8
  call void @tl_add_parameter(ptr @key_str.188, ptr %b500)
  %m501 = getelementptr inbounds %Block, ptr %sub_ptr462, i32 0, i32 3
  %sub_ptr502 = load ptr, ptr %m501, align 8
  %f503 = getelementptr inbounds %MLP, ptr %sub_ptr502, i32 0, i32 0
  %sub_ptr504 = load ptr, ptr %f503, align 8
  %W505 = getelementptr inbounds %Linear, ptr %sub_ptr504, i32 0, i32 0
  %W506 = load ptr, ptr %W505, align 8
  call void @tl_add_parameter(ptr @key_str.189, ptr %W506)
  %b507 = getelementptr inbounds %Linear, ptr %sub_ptr504, i32 0, i32 1
  %b508 = load ptr, ptr %b507, align 8
  call void @tl_add_parameter(ptr @key_str.190, ptr %b508)
  %p509 = getelementptr inbounds %MLP, ptr %sub_ptr502, i32 0, i32 1
  %sub_ptr510 = load ptr, ptr %p509, align 8
  %W511 = getelementptr inbounds %Linear, ptr %sub_ptr510, i32 0, i32 0
  %W512 = load ptr, ptr %W511, align 8
  call void @tl_add_parameter(ptr @key_str.191, ptr %W512)
  %b513 = getelementptr inbounds %Linear, ptr %sub_ptr510, i32 0, i32 1
  %b514 = load ptr, ptr %b513, align 8
  call void @tl_add_parameter(ptr @key_str.192, ptr %b514)
  %b2515 = getelementptr inbounds %GPTHeavy, ptr %model452, i32 0, i32 3
  %sub_ptr516 = load ptr, ptr %b2515, align 8
  %l1517 = getelementptr inbounds %Block, ptr %sub_ptr516, i32 0, i32 0
  %sub_ptr518 = load ptr, ptr %l1517, align 8
  %w519 = getelementptr inbounds %LayerNorm, ptr %sub_ptr518, i32 0, i32 0
  %w520 = load ptr, ptr %w519, align 8
  call void @tl_add_parameter(ptr @key_str.193, ptr %w520)
  %b521 = getelementptr inbounds %LayerNorm, ptr %sub_ptr518, i32 0, i32 1
  %b522 = load ptr, ptr %b521, align 8
  call void @tl_add_parameter(ptr @key_str.194, ptr %b522)
  %a523 = getelementptr inbounds %Block, ptr %sub_ptr516, i32 0, i32 1
  %sub_ptr524 = load ptr, ptr %a523, align 8
  %q_proj525 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr524, i32 0, i32 0
  %sub_ptr526 = load ptr, ptr %q_proj525, align 8
  %W527 = getelementptr inbounds %Linear, ptr %sub_ptr526, i32 0, i32 0
  %W528 = load ptr, ptr %W527, align 8
  call void @tl_add_parameter(ptr @key_str.195, ptr %W528)
  %b529 = getelementptr inbounds %Linear, ptr %sub_ptr526, i32 0, i32 1
  %b530 = load ptr, ptr %b529, align 8
  call void @tl_add_parameter(ptr @key_str.196, ptr %b530)
  %k_proj531 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr524, i32 0, i32 1
  %sub_ptr532 = load ptr, ptr %k_proj531, align 8
  %W533 = getelementptr inbounds %Linear, ptr %sub_ptr532, i32 0, i32 0
  %W534 = load ptr, ptr %W533, align 8
  call void @tl_add_parameter(ptr @key_str.197, ptr %W534)
  %b535 = getelementptr inbounds %Linear, ptr %sub_ptr532, i32 0, i32 1
  %b536 = load ptr, ptr %b535, align 8
  call void @tl_add_parameter(ptr @key_str.198, ptr %b536)
  %v_proj537 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr524, i32 0, i32 2
  %sub_ptr538 = load ptr, ptr %v_proj537, align 8
  %W539 = getelementptr inbounds %Linear, ptr %sub_ptr538, i32 0, i32 0
  %W540 = load ptr, ptr %W539, align 8
  call void @tl_add_parameter(ptr @key_str.199, ptr %W540)
  %b541 = getelementptr inbounds %Linear, ptr %sub_ptr538, i32 0, i32 1
  %b542 = load ptr, ptr %b541, align 8
  call void @tl_add_parameter(ptr @key_str.200, ptr %b542)
  %p_proj543 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr524, i32 0, i32 3
  %sub_ptr544 = load ptr, ptr %p_proj543, align 8
  %W545 = getelementptr inbounds %Linear, ptr %sub_ptr544, i32 0, i32 0
  %W546 = load ptr, ptr %W545, align 8
  call void @tl_add_parameter(ptr @key_str.201, ptr %W546)
  %b547 = getelementptr inbounds %Linear, ptr %sub_ptr544, i32 0, i32 1
  %b548 = load ptr, ptr %b547, align 8
  call void @tl_add_parameter(ptr @key_str.202, ptr %b548)
  %l2549 = getelementptr inbounds %Block, ptr %sub_ptr516, i32 0, i32 2
  %sub_ptr550 = load ptr, ptr %l2549, align 8
  %w551 = getelementptr inbounds %LayerNorm, ptr %sub_ptr550, i32 0, i32 0
  %w552 = load ptr, ptr %w551, align 8
  call void @tl_add_parameter(ptr @key_str.203, ptr %w552)
  %b553 = getelementptr inbounds %LayerNorm, ptr %sub_ptr550, i32 0, i32 1
  %b554 = load ptr, ptr %b553, align 8
  call void @tl_add_parameter(ptr @key_str.204, ptr %b554)
  %m555 = getelementptr inbounds %Block, ptr %sub_ptr516, i32 0, i32 3
  %sub_ptr556 = load ptr, ptr %m555, align 8
  %f557 = getelementptr inbounds %MLP, ptr %sub_ptr556, i32 0, i32 0
  %sub_ptr558 = load ptr, ptr %f557, align 8
  %W559 = getelementptr inbounds %Linear, ptr %sub_ptr558, i32 0, i32 0
  %W560 = load ptr, ptr %W559, align 8
  call void @tl_add_parameter(ptr @key_str.205, ptr %W560)
  %b561 = getelementptr inbounds %Linear, ptr %sub_ptr558, i32 0, i32 1
  %b562 = load ptr, ptr %b561, align 8
  call void @tl_add_parameter(ptr @key_str.206, ptr %b562)
  %p563 = getelementptr inbounds %MLP, ptr %sub_ptr556, i32 0, i32 1
  %sub_ptr564 = load ptr, ptr %p563, align 8
  %W565 = getelementptr inbounds %Linear, ptr %sub_ptr564, i32 0, i32 0
  %W566 = load ptr, ptr %W565, align 8
  call void @tl_add_parameter(ptr @key_str.207, ptr %W566)
  %b567 = getelementptr inbounds %Linear, ptr %sub_ptr564, i32 0, i32 1
  %b568 = load ptr, ptr %b567, align 8
  call void @tl_add_parameter(ptr @key_str.208, ptr %b568)
  %b3569 = getelementptr inbounds %GPTHeavy, ptr %model452, i32 0, i32 4
  %sub_ptr570 = load ptr, ptr %b3569, align 8
  %l1571 = getelementptr inbounds %Block, ptr %sub_ptr570, i32 0, i32 0
  %sub_ptr572 = load ptr, ptr %l1571, align 8
  %w573 = getelementptr inbounds %LayerNorm, ptr %sub_ptr572, i32 0, i32 0
  %w574 = load ptr, ptr %w573, align 8
  call void @tl_add_parameter(ptr @key_str.209, ptr %w574)
  %b575 = getelementptr inbounds %LayerNorm, ptr %sub_ptr572, i32 0, i32 1
  %b576 = load ptr, ptr %b575, align 8
  call void @tl_add_parameter(ptr @key_str.210, ptr %b576)
  %a577 = getelementptr inbounds %Block, ptr %sub_ptr570, i32 0, i32 1
  %sub_ptr578 = load ptr, ptr %a577, align 8
  %q_proj579 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr578, i32 0, i32 0
  %sub_ptr580 = load ptr, ptr %q_proj579, align 8
  %W581 = getelementptr inbounds %Linear, ptr %sub_ptr580, i32 0, i32 0
  %W582 = load ptr, ptr %W581, align 8
  call void @tl_add_parameter(ptr @key_str.211, ptr %W582)
  %b583 = getelementptr inbounds %Linear, ptr %sub_ptr580, i32 0, i32 1
  %b584 = load ptr, ptr %b583, align 8
  call void @tl_add_parameter(ptr @key_str.212, ptr %b584)
  %k_proj585 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr578, i32 0, i32 1
  %sub_ptr586 = load ptr, ptr %k_proj585, align 8
  %W587 = getelementptr inbounds %Linear, ptr %sub_ptr586, i32 0, i32 0
  %W588 = load ptr, ptr %W587, align 8
  call void @tl_add_parameter(ptr @key_str.213, ptr %W588)
  %b589 = getelementptr inbounds %Linear, ptr %sub_ptr586, i32 0, i32 1
  %b590 = load ptr, ptr %b589, align 8
  call void @tl_add_parameter(ptr @key_str.214, ptr %b590)
  %v_proj591 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr578, i32 0, i32 2
  %sub_ptr592 = load ptr, ptr %v_proj591, align 8
  %W593 = getelementptr inbounds %Linear, ptr %sub_ptr592, i32 0, i32 0
  %W594 = load ptr, ptr %W593, align 8
  call void @tl_add_parameter(ptr @key_str.215, ptr %W594)
  %b595 = getelementptr inbounds %Linear, ptr %sub_ptr592, i32 0, i32 1
  %b596 = load ptr, ptr %b595, align 8
  call void @tl_add_parameter(ptr @key_str.216, ptr %b596)
  %p_proj597 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr578, i32 0, i32 3
  %sub_ptr598 = load ptr, ptr %p_proj597, align 8
  %W599 = getelementptr inbounds %Linear, ptr %sub_ptr598, i32 0, i32 0
  %W600 = load ptr, ptr %W599, align 8
  call void @tl_add_parameter(ptr @key_str.217, ptr %W600)
  %b601 = getelementptr inbounds %Linear, ptr %sub_ptr598, i32 0, i32 1
  %b602 = load ptr, ptr %b601, align 8
  call void @tl_add_parameter(ptr @key_str.218, ptr %b602)
  %l2603 = getelementptr inbounds %Block, ptr %sub_ptr570, i32 0, i32 2
  %sub_ptr604 = load ptr, ptr %l2603, align 8
  %w605 = getelementptr inbounds %LayerNorm, ptr %sub_ptr604, i32 0, i32 0
  %w606 = load ptr, ptr %w605, align 8
  call void @tl_add_parameter(ptr @key_str.219, ptr %w606)
  %b607 = getelementptr inbounds %LayerNorm, ptr %sub_ptr604, i32 0, i32 1
  %b608 = load ptr, ptr %b607, align 8
  call void @tl_add_parameter(ptr @key_str.220, ptr %b608)
  %m609 = getelementptr inbounds %Block, ptr %sub_ptr570, i32 0, i32 3
  %sub_ptr610 = load ptr, ptr %m609, align 8
  %f611 = getelementptr inbounds %MLP, ptr %sub_ptr610, i32 0, i32 0
  %sub_ptr612 = load ptr, ptr %f611, align 8
  %W613 = getelementptr inbounds %Linear, ptr %sub_ptr612, i32 0, i32 0
  %W614 = load ptr, ptr %W613, align 8
  call void @tl_add_parameter(ptr @key_str.221, ptr %W614)
  %b615 = getelementptr inbounds %Linear, ptr %sub_ptr612, i32 0, i32 1
  %b616 = load ptr, ptr %b615, align 8
  call void @tl_add_parameter(ptr @key_str.222, ptr %b616)
  %p617 = getelementptr inbounds %MLP, ptr %sub_ptr610, i32 0, i32 1
  %sub_ptr618 = load ptr, ptr %p617, align 8
  %W619 = getelementptr inbounds %Linear, ptr %sub_ptr618, i32 0, i32 0
  %W620 = load ptr, ptr %W619, align 8
  call void @tl_add_parameter(ptr @key_str.223, ptr %W620)
  %b621 = getelementptr inbounds %Linear, ptr %sub_ptr618, i32 0, i32 1
  %b622 = load ptr, ptr %b621, align 8
  call void @tl_add_parameter(ptr @key_str.224, ptr %b622)
  %b4623 = getelementptr inbounds %GPTHeavy, ptr %model452, i32 0, i32 5
  %sub_ptr624 = load ptr, ptr %b4623, align 8
  %l1625 = getelementptr inbounds %Block, ptr %sub_ptr624, i32 0, i32 0
  %sub_ptr626 = load ptr, ptr %l1625, align 8
  %w627 = getelementptr inbounds %LayerNorm, ptr %sub_ptr626, i32 0, i32 0
  %w628 = load ptr, ptr %w627, align 8
  call void @tl_add_parameter(ptr @key_str.225, ptr %w628)
  %b629 = getelementptr inbounds %LayerNorm, ptr %sub_ptr626, i32 0, i32 1
  %b630 = load ptr, ptr %b629, align 8
  call void @tl_add_parameter(ptr @key_str.226, ptr %b630)
  %a631 = getelementptr inbounds %Block, ptr %sub_ptr624, i32 0, i32 1
  %sub_ptr632 = load ptr, ptr %a631, align 8
  %q_proj633 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr632, i32 0, i32 0
  %sub_ptr634 = load ptr, ptr %q_proj633, align 8
  %W635 = getelementptr inbounds %Linear, ptr %sub_ptr634, i32 0, i32 0
  %W636 = load ptr, ptr %W635, align 8
  call void @tl_add_parameter(ptr @key_str.227, ptr %W636)
  %b637 = getelementptr inbounds %Linear, ptr %sub_ptr634, i32 0, i32 1
  %b638 = load ptr, ptr %b637, align 8
  call void @tl_add_parameter(ptr @key_str.228, ptr %b638)
  %k_proj639 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr632, i32 0, i32 1
  %sub_ptr640 = load ptr, ptr %k_proj639, align 8
  %W641 = getelementptr inbounds %Linear, ptr %sub_ptr640, i32 0, i32 0
  %W642 = load ptr, ptr %W641, align 8
  call void @tl_add_parameter(ptr @key_str.229, ptr %W642)
  %b643 = getelementptr inbounds %Linear, ptr %sub_ptr640, i32 0, i32 1
  %b644 = load ptr, ptr %b643, align 8
  call void @tl_add_parameter(ptr @key_str.230, ptr %b644)
  %v_proj645 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr632, i32 0, i32 2
  %sub_ptr646 = load ptr, ptr %v_proj645, align 8
  %W647 = getelementptr inbounds %Linear, ptr %sub_ptr646, i32 0, i32 0
  %W648 = load ptr, ptr %W647, align 8
  call void @tl_add_parameter(ptr @key_str.231, ptr %W648)
  %b649 = getelementptr inbounds %Linear, ptr %sub_ptr646, i32 0, i32 1
  %b650 = load ptr, ptr %b649, align 8
  call void @tl_add_parameter(ptr @key_str.232, ptr %b650)
  %p_proj651 = getelementptr inbounds %CausalSelfAttention, ptr %sub_ptr632, i32 0, i32 3
  %sub_ptr652 = load ptr, ptr %p_proj651, align 8
  %W653 = getelementptr inbounds %Linear, ptr %sub_ptr652, i32 0, i32 0
  %W654 = load ptr, ptr %W653, align 8
  call void @tl_add_parameter(ptr @key_str.233, ptr %W654)
  %b655 = getelementptr inbounds %Linear, ptr %sub_ptr652, i32 0, i32 1
  %b656 = load ptr, ptr %b655, align 8
  call void @tl_add_parameter(ptr @key_str.234, ptr %b656)
  %l2657 = getelementptr inbounds %Block, ptr %sub_ptr624, i32 0, i32 2
  %sub_ptr658 = load ptr, ptr %l2657, align 8
  %w659 = getelementptr inbounds %LayerNorm, ptr %sub_ptr658, i32 0, i32 0
  %w660 = load ptr, ptr %w659, align 8
  call void @tl_add_parameter(ptr @key_str.235, ptr %w660)
  %b661 = getelementptr inbounds %LayerNorm, ptr %sub_ptr658, i32 0, i32 1
  %b662 = load ptr, ptr %b661, align 8
  call void @tl_add_parameter(ptr @key_str.236, ptr %b662)
  %m663 = getelementptr inbounds %Block, ptr %sub_ptr624, i32 0, i32 3
  %sub_ptr664 = load ptr, ptr %m663, align 8
  %f665 = getelementptr inbounds %MLP, ptr %sub_ptr664, i32 0, i32 0
  %sub_ptr666 = load ptr, ptr %f665, align 8
  %W667 = getelementptr inbounds %Linear, ptr %sub_ptr666, i32 0, i32 0
  %W668 = load ptr, ptr %W667, align 8
  call void @tl_add_parameter(ptr @key_str.237, ptr %W668)
  %b669 = getelementptr inbounds %Linear, ptr %sub_ptr666, i32 0, i32 1
  %b670 = load ptr, ptr %b669, align 8
  call void @tl_add_parameter(ptr @key_str.238, ptr %b670)
  %p671 = getelementptr inbounds %MLP, ptr %sub_ptr664, i32 0, i32 1
  %sub_ptr672 = load ptr, ptr %p671, align 8
  %W673 = getelementptr inbounds %Linear, ptr %sub_ptr672, i32 0, i32 0
  %W674 = load ptr, ptr %W673, align 8
  call void @tl_add_parameter(ptr @key_str.239, ptr %W674)
  %b675 = getelementptr inbounds %Linear, ptr %sub_ptr672, i32 0, i32 1
  %b676 = load ptr, ptr %b675, align 8
  call void @tl_add_parameter(ptr @key_str.240, ptr %b676)
  %l677 = getelementptr inbounds %GPTHeavy, ptr %model452, i32 0, i32 6
  %sub_ptr678 = load ptr, ptr %l677, align 8
  %w679 = getelementptr inbounds %LayerNorm, ptr %sub_ptr678, i32 0, i32 0
  %w680 = load ptr, ptr %w679, align 8
  call void @tl_add_parameter(ptr @key_str.241, ptr %w680)
  %b681 = getelementptr inbounds %LayerNorm, ptr %sub_ptr678, i32 0, i32 1
  %b682 = load ptr, ptr %b681, align 8
  call void @tl_add_parameter(ptr @key_str.242, ptr %b682)
  %h683 = getelementptr inbounds %GPTHeavy, ptr %model452, i32 0, i32 7
  %sub_ptr684 = load ptr, ptr %h683, align 8
  %W685 = getelementptr inbounds %Linear, ptr %sub_ptr684, i32 0, i32 0
  %W686 = load ptr, ptr %W685, align 8
  call void @tl_add_parameter(ptr @key_str.243, ptr %W686)
  %b687 = getelementptr inbounds %Linear, ptr %sub_ptr684, i32 0, i32 1
  %b688 = load ptr, ptr %b687, align 8
  call void @tl_add_parameter(ptr @key_str.244, ptr %b688)
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
