#include <cuda_runtime.h>
#include <math.h>

// =====================================================================
// one_hot カーネル
// =====================================================================
__global__ void one_hot_kernel(
    const long long* indices,
    float* output,
    int batch,
    int classes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch) {
        int idx = (int)indices[i];
        if (idx >= 0 && idx < classes) {
            output[i * classes + idx] = 1.0f;
        }
    }
}

// =====================================================================
// scatter_add カーネル
// =====================================================================
__global__ void scatter_add_kernel(
    const float* grad,
    const long long* indices,
    float* output,
    int seq_len,
    int dim,
    int vocab
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * dim;
    if (tid < total) {
        int i = tid / dim;
        int j = tid % dim;
        int target_row = (int)indices[i];
        if (target_row >= 0 && target_row < vocab) {
            atomicAdd(&output[target_row * dim + j], grad[i * dim + j]);
        }
    }
}

// =====================================================================
// scalar ops カーネル (element-wise: y[i] = op(x[i], scalar))
// =====================================================================
__global__ void add_scalar_kernel(const float* x, float* y, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + s;
}

__global__ void mul_scalar_kernel(const float* x, float* y, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * s;
}

__global__ void div_scalar_kernel(const float* x, float* y, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] / s;
}

__global__ void pow_scalar_kernel(const float* x, float* y, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = powf(x[i], s);
}

__global__ void clamp_kernel(const float* x, float* y, int n, float lo, float hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fminf(fmaxf(x[i], lo), hi);
}

// =====================================================================
// binary ops カーネル (element-wise same-shape: y[i] = op(a[i], b[i]))
// =====================================================================
__global__ void add_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

__global__ void sub_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] - b[i];
}

__global__ void mul_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}

__global__ void div_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] / b[i];
}

__global__ void pow_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = powf(a[i], b[i]);
}

__global__ void rem_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmodf(a[i], b[i]);
}

// comparison ops
__global__ void eq_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (fabsf(a[i] - b[i]) < 1e-6f) ? 1.0f : 0.0f;
}
__global__ void ne_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (fabsf(a[i] - b[i]) >= 1e-6f) ? 1.0f : 0.0f;
}
__global__ void lt_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
}
__global__ void le_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (a[i] <= b[i]) ? 1.0f : 0.0f;
}
__global__ void gt_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
}
__global__ void ge_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (a[i] >= b[i]) ? 1.0f : 0.0f;
}

// =====================================================================
// unary ops カーネル
// =====================================================================
__global__ void neg_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = -x[i];
}

__global__ void abs_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fabsf(x[i]);
}

__global__ void exp_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = expf(x[i]);
}

__global__ void log_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = logf(x[i]);
}

__global__ void sqrt_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sqrtf(x[i]);
}

// =====================================================================
// activation カーネル
// =====================================================================
__global__ void relu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(x[i], 0.0f);
}

__global__ void sigmoid_kernel2(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ void tanh_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i]);
}

__global__ void gelu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float c = 0.7978845608f; // sqrt(2/pi)
        float inner = c * (v + 0.044715f * v * v * v);
        y[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

__global__ void silu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] / (1.0f + expf(-x[i]));
}

__global__ void sin_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sinf(x[i]);
}

__global__ void cos_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = cosf(x[i]);
}

__global__ void tan_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanf(x[i]);
}

__global__ void floor_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = floorf(x[i]);
}

__global__ void ceil_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = ceilf(x[i]);
}

__global__ void round_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = roundf(x[i]);
}

// =====================================================================
// max/min reduction カーネル
// =====================================================================
__global__ void max_all_kernel(const float* x, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? x[i] : -3.402823466e+38f; // -FLT_MAX
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        // atomic max for float via atomicCAS
        float val = sdata[0];
        unsigned int* addr = (unsigned int*)out;
        unsigned int old = *addr, assumed;
        do {
            assumed = old;
            old = atomicCAS(addr, assumed,
                __float_as_uint(fmaxf(__uint_as_float(assumed), val)));
        } while (assumed != old);
    }
}

__global__ void min_all_kernel(const float* x, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? x[i] : 3.402823466e+38f; // FLT_MAX
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        float val = sdata[0];
        unsigned int* addr = (unsigned int*)out;
        unsigned int old = *addr, assumed;
        do {
            assumed = old;
            old = atomicCAS(addr, assumed,
                __float_as_uint(fminf(__uint_as_float(assumed), val)));
        } while (assumed != old);
    }
}

// =====================================================================
// reduce カーネル (single block for simplicity; works for small tensors)
// =====================================================================
__global__ void sum_all_kernel(const float* x, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// =====================================================================
// matmul カーネル (naive: 各スレッドが C[batch,i,j] の 1 要素を計算)
// =====================================================================
__global__ void matmul_naive_kernel(
    const float* A, const float* B, float* C,
    int m, int k, int n, int batch, int b_batched
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * m * n;
    if (tid < total) {
        int b_idx = tid / (m * n);
        int remainder = tid % (m * n);
        int i = remainder / n;
        int j = remainder % n;

        int a_off = b_idx * m * k;
        int b_off = b_batched ? (b_idx * k * n) : 0;

        float sum = 0.0f;
        for (int p = 0; p < k; p++) {
            sum += A[a_off + i * k + p] * B[b_off + p * n + j];
        }
        C[b_idx * m * n + i * n + j] = sum;
    }
}

// =====================================================================
// softmax カーネル (行ごと: 各ブロックが1行を処理)
// input[outer * axis_size * inner], output 同 shape
// 簡易版: 各スレッドが1つの (outer, inner) ペアを処理
// =====================================================================
__global__ void softmax_kernel(
    const float* input, float* output,
    int outer, int axis_size, int inner
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (tid < total) {
        int o = tid / inner;
        int i = tid % inner;
        int base = o * axis_size * inner + i;

        // max
        float max_val = -3.402823466e+38f;
        for (int k = 0; k < axis_size; k++) {
            float v = input[base + k * inner];
            if (v > max_val) max_val = v;
        }
        // exp + sum
        float sum_exp = 0.0f;
        for (int k = 0; k < axis_size; k++) {
            float e = expf(input[base + k * inner] - max_val);
            output[base + k * inner] = e;
            sum_exp += e;
        }
        // normalize
        float inv_sum = 1.0f / sum_exp;
        for (int k = 0; k < axis_size; k++) {
            output[base + k * inner] *= inv_sum;
        }
    }
}

// =====================================================================
// embedding (gather) カーネル
// weight[vocab, dim], indices[seq] → output[seq, dim]
// =====================================================================
__global__ void embedding_kernel(
    const float* weight, const long long* indices,
    float* output, int seq_len, int embed_dim, int vocab_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * embed_dim;
    if (tid < total) {
        int i = tid / embed_dim;
        int j = tid % embed_dim;
        int idx = (int)indices[i];
        if (idx >= 0 && idx < vocab_size) {
            output[i * embed_dim + j] = weight[idx * embed_dim + j];
        }
    }
}

// =====================================================================
// cross_entropy カーネル
// logits[N, C], targets[N] → loss[N] (per-sample loss)
// =====================================================================
__global__ void cross_entropy_kernel(
    const float* logits, const long long* targets,
    float* losses, int n, int c
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float* row = logits + i * c;
        // max for stability
        float max_val = -3.402823466e+38f;
        for (int j = 0; j < c; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < c; j++) {
            sum_exp += expf(row[j] - max_val);
        }
        int target_idx = (int)targets[i];
        losses[i] = (max_val + logf(sum_exp)) - row[target_idx];
    }
}

// =====================================================================
// tril カーネル (下三角以外を 0 にする)
// =====================================================================
__global__ void tril_kernel(
    const float* input, float* output,
    int rows, int cols, int batch, int diagonal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * rows * cols;
    if (tid < total) {
        int b = tid / (rows * cols);
        int remainder = tid % (rows * cols);
        int r = remainder / cols;
        int c_idx = remainder % cols;
        (void)b;
        output[tid] = (c_idx <= r + diagonal) ? input[tid] : 0.0f;
    }
}

// =====================================================================
// where_cond カーネル (ternary: cond > 0 ? x : y)
// =====================================================================
__global__ void where_cond_kernel(
    const float* cond, const float* x, const float* y,
    float* output, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = (cond[i] > 0.0f) ? x[i] : y[i];
    }
}

// =====================================================================
// rms_norm カーネル (行ごと: x * rsqrt(mean(x²) + eps))
// =====================================================================
__global__ void rms_norm_kernel(
    const float* input, float* output,
    int outer, int norm_size, float eps
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < outer) {
        int offset = o * norm_size;
        float sum_sq = 0.0f;
        for (int i = 0; i < norm_size; i++) {
            float v = input[offset + i];
            sum_sq += v * v;
        }
        float inv_rms = rsqrtf(sum_sq / (float)norm_size + eps);
        for (int i = 0; i < norm_size; i++) {
            output[offset + i] = input[offset + i] * inv_rms;
        }
    }
}

// =====================================================================
// causal_mask カーネル (上三角を -inf にする)
// =====================================================================
__global__ void causal_mask_kernel(float* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = size * size;
    if (tid < total) {
        int r = tid / size;
        int c = tid % size;
        output[tid] = (c > r) ? -3.402823466e+38f : 0.0f;
    }
}

// =====================================================================
// transpose カーネル (任意の2次元スワップ)
// shape/strides 情報は launch 関数側で計算して渡す
// =====================================================================
__global__ void transpose_2d_kernel(
    const float* input, float* output,
    int rows, int cols
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (tid < total) {
        int r = tid / cols;
        int c = tid % cols;
        output[c * rows + r] = input[r * cols + c];
    }
}

// =====================================================================
// broadcast binary カーネル
// out[i] = op(a[broadcast_a(i)], b[broadcast_b(i)])
// shape 情報は GPU constant memory に渡す
// 簡易版: 最大8次元対応
// =====================================================================
__global__ void broadcast_add_kernel(
    const float* a, const float* b, float* out,
    const int* out_shape, const int* a_shape, const int* b_shape,
    int ndim, int total
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int idx = tid;
        int a_idx = 0, b_idx = 0;
        int a_stride = 1, b_stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            int coord = idx % out_shape[d];
            idx /= out_shape[d];
            int a_dim = a_shape[d];
            int b_dim = b_shape[d];
            if (a_dim > 1) a_idx += coord * a_stride;
            if (b_dim > 1) b_idx += coord * b_stride;
            a_stride *= a_dim;
            b_stride *= b_dim;
        }
        out[tid] = a[a_idx] + b[b_idx];
    }
}

// =====================================================================
// reduce_axis カーネル (sum)
// 各スレッドが1つの出力要素を計算: 軸方向に sum
// =====================================================================
__global__ void reduce_axis_sum_kernel(
    const float* input, float* output,
    int outer, int axis_size, int inner
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (tid < total) {
        int o = tid / inner;
        int i = tid % inner;
        float sum = 0.0f;
        for (int k = 0; k < axis_size; k++) {
            sum += input[o * axis_size * inner + k * inner + i];
        }
        output[tid] = sum;
    }
}

__global__ void reduce_axis_max_kernel(
    const float* input, float* output,
    int outer, int axis_size, int inner
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (tid < total) {
        int o = tid / inner;
        int i = tid % inner;
        float mx = -3.402823466e+38f;
        for (int k = 0; k < axis_size; k++) {
            float v = input[o * axis_size * inner + k * inner + i];
            if (v > mx) mx = v;
        }
        output[tid] = mx;
    }
}

__global__ void reduce_axis_min_kernel(
    const float* input, float* output,
    int outer, int axis_size, int inner
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (tid < total) {
        int o = tid / inner;
        int i = tid % inner;
        float mn = 3.402823466e+38f;
        for (int k = 0; k < axis_size; k++) {
            float v = input[o * axis_size * inner + k * inner + i];
            if (v < mn) mn = v;
        }
        output[tid] = mn;
    }
}

// =====================================================================
// narrow (slice) カーネル
// =====================================================================
__global__ void narrow_kernel(
    const float* input, float* output,
    int outer, int inner, int old_dim, int new_dim, int start
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * new_dim * inner;
    if (tid < total) {
        int o = tid / (new_dim * inner);
        int rem = tid % (new_dim * inner);
        int d = rem / inner;
        int i = rem % inner;
        int src = o * old_dim * inner + (d + start) * inner + i;
        output[tid] = input[src];
    }
}

// =====================================================================
// cat カーネル（2つのテンソルを指定次元で結合）
// =====================================================================
__global__ void cat_kernel(
    const float* a, const float* b, float* output,
    int outer, int inner, int a_dim, int b_dim
) {
    int total_dim = a_dim + b_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * total_dim * inner;
    if (tid < total) {
        int o = tid / (total_dim * inner);
        int rem = tid % (total_dim * inner);
        int d = rem / inner;
        int i = rem % inner;
        if (d < a_dim) {
            output[tid] = a[o * a_dim * inner + d * inner + i];
        } else {
            output[tid] = b[o * b_dim * inner + (d - a_dim) * inner + i];
        }
    }
}

// =====================================================================
// broadcast_to カーネル
// =====================================================================
__global__ void broadcast_to_kernel(
    const float* input, float* output,
    const int* target_shape, const int* src_shape,
    int ndim, int total
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int idx = tid;
        int src_idx = 0;
        int src_stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            int coord = idx % target_shape[d];
            idx /= target_shape[d];
            int sd = src_shape[d];
            if (sd > 1) src_idx += coord * src_stride;
            src_stride *= sd;
        }
        output[tid] = input[src_idx];
    }
}

// =====================================================================
// conv2d カーネル
// input[N,C_in,H,W], weight[C_out,C_in,kH,kW] → output[N,C_out,H_out,W_out]
// =====================================================================
__global__ void conv2d_kernel(
    const float* input, const float* weight, float* output,
    int n, int c_in, int h_in, int w_in,
    int c_out, int kh, int kw,
    int h_out, int w_out,
    int stride_h, int stride_w, int pad_h, int pad_w
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c_out * h_out * w_out;
    if (tid < total) {
        int batch = tid / (c_out * h_out * w_out);
        int rem = tid % (c_out * h_out * w_out);
        int co = rem / (h_out * w_out);
        rem = rem % (h_out * w_out);
        int oh = rem / w_out;
        int ow = rem % w_out;

        float sum = 0.0f;
        for (int ci = 0; ci < c_in; ci++) {
            for (int khi = 0; khi < kh; khi++) {
                for (int kwi = 0; kwi < kw; kwi++) {
                    int ih = oh * stride_h + khi - pad_h;
                    int iw = ow * stride_w + kwi - pad_w;
                    if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                        int in_idx = batch * c_in * h_in * w_in + ci * h_in * w_in + ih * w_in + iw;
                        int w_idx = co * c_in * kh * kw + ci * kh * kw + khi * kw + kwi;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        output[tid] = sum;
    }
}

// =====================================================================
// batch_norm カーネル
// input[N,C,...], gamma[C], beta[C], mean[C], var[C]
// =====================================================================
__global__ void batch_norm_kernel(
    const float* input, const float* gamma, const float* beta,
    const float* mean, const float* var, float* output,
    int n, int c, int spatial, float eps
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * spatial;
    if (tid < total) {
        int ch = (tid / spatial) % c;
        float inv_std = rsqrtf(var[ch] + eps);
        output[tid] = gamma[ch] * (input[tid] - mean[ch]) * inv_std + beta[ch];
    }
}

// =====================================================================
// layer_norm カーネル (行ごと: 各スレッドが1行を処理)
// =====================================================================
__global__ void layer_norm_kernel(
    const float* input, const float* gamma, const float* beta,
    float* output, int outer, int norm_size, float eps
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < outer) {
        int offset = o * norm_size;
        // mean
        float sum = 0.0f;
        for (int i = 0; i < norm_size; i++) sum += input[offset + i];
        float mean = sum / (float)norm_size;
        // var
        float var_sum = 0.0f;
        for (int i = 0; i < norm_size; i++) {
            float d = input[offset + i] - mean;
            var_sum += d * d;
        }
        float inv_std = rsqrtf(var_sum / (float)norm_size + eps);
        // normalize
        for (int i = 0; i < norm_size; i++) {
            output[offset + i] = gamma[i] * (input[offset + i] - mean) * inv_std + beta[i];
        }
    }
}

// =====================================================================
// max_pool2d カーネル
// =====================================================================
__global__ void max_pool2d_kernel(
    const float* input, float* output,
    int n, int c, int h, int w,
    int h_out, int w_out,
    int kh, int kw, int stride_h, int stride_w
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * h_out * w_out;
    if (tid < total) {
        int batch = tid / (c * h_out * w_out);
        int rem = tid % (c * h_out * w_out);
        int ch = rem / (h_out * w_out);
        rem = rem % (h_out * w_out);
        int oh = rem / w_out;
        int ow = rem % w_out;

        float mx = -3.402823466e+38f;
        for (int ki = 0; ki < kh; ki++) {
            for (int kj = 0; kj < kw; kj++) {
                int ih = oh * stride_h + ki;
                int iw = ow * stride_w + kj;
                int idx = batch * c * h * w + ch * h * w + ih * w + iw;
                if (input[idx] > mx) mx = input[idx];
            }
        }
        output[tid] = mx;
    }
}

// =====================================================================
// avg_pool2d カーネル
// =====================================================================
__global__ void avg_pool2d_kernel(
    const float* input, float* output,
    int n, int c, int h, int w,
    int h_out, int w_out,
    int kh, int kw, int stride_h, int stride_w
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * h_out * w_out;
    if (tid < total) {
        int batch = tid / (c * h_out * w_out);
        int rem = tid % (c * h_out * w_out);
        int ch = rem / (h_out * w_out);
        rem = rem % (h_out * w_out);
        int oh = rem / w_out;
        int ow = rem % w_out;

        float sum = 0.0f;
        for (int ki = 0; ki < kh; ki++) {
            for (int kj = 0; kj < kw; kj++) {
                int ih = oh * stride_h + ki;
                int iw = ow * stride_w + kj;
                int idx = batch * c * h * w + ch * h * w + ih * w + iw;
                sum += input[idx];
            }
        }
        output[tid] = sum / (float)(kh * kw);
    }
}

// =====================================================================
// dropout カーネル (hash-based deterministic mask)
// =====================================================================
__global__ void dropout_kernel(
    const float* input, float* output,
    int n, float p, float scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // simple hash-based pseudo-random
        unsigned int hash = (unsigned int)i * 2654435761u;
        float r = (float)(hash >> 16) / 65536.0f;
        output[i] = (r < p) ? 0.0f : input[i] * scale;
    }
}

// =====================================================================
// C wrappers
// =====================================================================
extern "C" {

// --- scatter/one_hot ---
void launch_one_hot_kernel(const long long* indices, float* output,
    int batch, int classes, cudaStream_t stream) {
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    one_hot_kernel<<<blocks, threads, 0, stream>>>(indices, output, batch, classes);
}

void launch_scatter_add_kernel(const float* grad, const long long* indices,
    float* output, int seq_len, int dim, int vocab, cudaStream_t stream) {
    int total = seq_len * dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    scatter_add_kernel<<<blocks, threads, 0, stream>>>(grad, indices, output, seq_len, dim, vocab);
}

// --- scalar ops ---
#define LAUNCH_SCALAR(name) \
void launch_##name##_kernel(const float* x, float* y, int n, float s, cudaStream_t stream) { \
    int threads = 256; int blocks = (n + threads - 1) / threads; \
    name##_kernel<<<blocks, threads, 0, stream>>>(x, y, n, s); \
}

LAUNCH_SCALAR(add_scalar)
LAUNCH_SCALAR(mul_scalar)
LAUNCH_SCALAR(div_scalar)
LAUNCH_SCALAR(pow_scalar)

void launch_clamp_kernel(const float* x, float* y, int n, float lo, float hi, cudaStream_t stream) {
    int threads = 256; int blocks = (n + threads - 1) / threads;
    clamp_kernel<<<blocks, threads, 0, stream>>>(x, y, n, lo, hi);
}

// --- binary ops ---
#define LAUNCH_BINARY(name) \
void launch_##name##_kernel(const float* a, const float* b, float* y, int n, cudaStream_t stream) { \
    int threads = 256; int blocks = (n + threads - 1) / threads; \
    name##_kernel<<<blocks, threads, 0, stream>>>(a, b, y, n); \
}

LAUNCH_BINARY(add)
LAUNCH_BINARY(sub)
LAUNCH_BINARY(mul)
LAUNCH_BINARY(div)
LAUNCH_BINARY(pow)
LAUNCH_BINARY(rem)
LAUNCH_BINARY(eq)
LAUNCH_BINARY(ne)
LAUNCH_BINARY(lt)
LAUNCH_BINARY(le)
LAUNCH_BINARY(gt)
LAUNCH_BINARY(ge)

// --- unary ops ---
#define LAUNCH_UNARY(name) \
void launch_##name##_kernel(const float* x, float* y, int n, cudaStream_t stream) { \
    int threads = 256; int blocks = (n + threads - 1) / threads; \
    name##_kernel<<<blocks, threads, 0, stream>>>(x, y, n); \
}

LAUNCH_UNARY(neg)
LAUNCH_UNARY(abs)
LAUNCH_UNARY(exp)
LAUNCH_UNARY(log)
LAUNCH_UNARY(sqrt)
LAUNCH_UNARY(relu)
LAUNCH_UNARY(tanh)
LAUNCH_UNARY(gelu)
LAUNCH_UNARY(silu)
LAUNCH_UNARY(sin)
LAUNCH_UNARY(cos)
LAUNCH_UNARY(tan)
LAUNCH_UNARY(floor)
LAUNCH_UNARY(ceil)
LAUNCH_UNARY(round)

void launch_sigmoid_kernel2(const float* x, float* y, int n, cudaStream_t stream) {
    int threads = 256; int blocks = (n + threads - 1) / threads;
    sigmoid_kernel2<<<blocks, threads, 0, stream>>>(x, y, n);
}

// --- max/min reduce ---
void launch_max_all_kernel(const float* x, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    max_all_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(x, out, n);
}

void launch_min_all_kernel(const float* x, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    min_all_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(x, out, n);
}

// --- reduce ---
void launch_sum_all_kernel(const float* x, float* out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sum_all_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(x, out, n);
}

// --- matmul ---
void launch_matmul_kernel(const float* a, const float* b, float* c,
    int m, int k, int n, int batch, int b_batched, cudaStream_t stream) {
    int total = batch * m * n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    matmul_naive_kernel<<<blocks, threads, 0, stream>>>(a, b, c, m, k, n, batch, b_batched);
}

// --- special ops ---
void launch_softmax_kernel(const float* input, float* output,
    int outer, int axis_size, int inner, cudaStream_t stream) {
    int total = outer * inner;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    softmax_kernel<<<blocks, threads, 0, stream>>>(input, output, outer, axis_size, inner);
}

void launch_embedding_kernel(const float* weight, const long long* indices,
    float* output, int seq_len, int embed_dim, int vocab_size, cudaStream_t stream) {
    int total = seq_len * embed_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    embedding_kernel<<<blocks, threads, 0, stream>>>(weight, indices, output, seq_len, embed_dim, vocab_size);
}

void launch_cross_entropy_kernel(const float* logits, const long long* targets,
    float* losses, int n, int c, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cross_entropy_kernel<<<blocks, threads, 0, stream>>>(logits, targets, losses, n, c);
}

void launch_tril_kernel(const float* input, float* output,
    int rows, int cols, int batch, int diagonal, cudaStream_t stream) {
    int total = batch * rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    tril_kernel<<<blocks, threads, 0, stream>>>(input, output, rows, cols, batch, diagonal);
}

void launch_where_cond_kernel(const float* cond, const float* x, const float* y,
    float* output, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    where_cond_kernel<<<blocks, threads, 0, stream>>>(cond, x, y, output, n);
}

void launch_rms_norm_kernel(const float* input, float* output,
    int outer, int norm_size, float eps, cudaStream_t stream) {
    int threads = 256;
    int blocks = (outer + threads - 1) / threads;
    rms_norm_kernel<<<blocks, threads, 0, stream>>>(input, output, outer, norm_size, eps);
}

void launch_causal_mask_kernel(float* output, int size, cudaStream_t stream) {
    int total = size * size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    causal_mask_kernel<<<blocks, threads, 0, stream>>>(output, size);
}

// --- transpose ---
void launch_transpose_2d_kernel(const float* input, float* output,
    int rows, int cols, cudaStream_t stream) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    transpose_2d_kernel<<<blocks, threads, 0, stream>>>(input, output, rows, cols);
}

// --- reduce_axis ---
void launch_reduce_axis_sum_kernel(const float* input, float* output,
    int outer, int axis_size, int inner, cudaStream_t stream) {
    int total = outer * inner;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reduce_axis_sum_kernel<<<blocks, threads, 0, stream>>>(input, output, outer, axis_size, inner);
}

void launch_reduce_axis_max_kernel(const float* input, float* output,
    int outer, int axis_size, int inner, cudaStream_t stream) {
    int total = outer * inner;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reduce_axis_max_kernel<<<blocks, threads, 0, stream>>>(input, output, outer, axis_size, inner);
}

void launch_reduce_axis_min_kernel(const float* input, float* output,
    int outer, int axis_size, int inner, cudaStream_t stream) {
    int total = outer * inner;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reduce_axis_min_kernel<<<blocks, threads, 0, stream>>>(input, output, outer, axis_size, inner);
}

// --- narrow ---
void launch_narrow_kernel(const float* input, float* output,
    int outer, int inner, int old_dim, int new_dim, int start, cudaStream_t stream) {
    int total = outer * new_dim * inner;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    narrow_kernel<<<blocks, threads, 0, stream>>>(input, output, outer, inner, old_dim, new_dim, start);
}

// --- cat ---
void launch_cat_kernel(const float* a, const float* b, float* output,
    int outer, int inner, int a_dim, int b_dim, cudaStream_t stream) {
    int total = outer * (a_dim + b_dim) * inner;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cat_kernel<<<blocks, threads, 0, stream>>>(a, b, output, outer, inner, a_dim, b_dim);
}

// --- broadcast_to ---
void launch_broadcast_to_kernel(const float* input, float* output,
    const int* target_shape, const int* src_shape,
    int ndim, int total, cudaStream_t stream) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    broadcast_to_kernel<<<blocks, threads, 0, stream>>>(input, output, target_shape, src_shape, ndim, total);
}

// --- nn ops ---
void launch_conv2d_kernel(const float* input, const float* weight, float* output,
    int n, int c_in, int h_in, int w_in,
    int c_out, int kh, int kw,
    int h_out, int w_out,
    int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream) {
    int total = n * c_out * h_out * w_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_kernel<<<blocks, threads, 0, stream>>>(input, weight, output,
        n, c_in, h_in, w_in, c_out, kh, kw, h_out, w_out,
        stride_h, stride_w, pad_h, pad_w);
}

void launch_batch_norm_kernel(const float* input, const float* gamma, const float* beta,
    const float* mean, const float* var, float* output,
    int n, int c, int spatial, float eps, cudaStream_t stream) {
    int total = n * c * spatial;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    batch_norm_kernel<<<blocks, threads, 0, stream>>>(input, gamma, beta, mean, var, output, n, c, spatial, eps);
}

void launch_layer_norm_kernel(const float* input, const float* gamma, const float* beta,
    float* output, int outer, int norm_size, float eps, cudaStream_t stream) {
    int threads = 256;
    int blocks = (outer + threads - 1) / threads;
    layer_norm_kernel<<<blocks, threads, 0, stream>>>(input, gamma, beta, output, outer, norm_size, eps);
}

void launch_max_pool2d_kernel(const float* input, float* output,
    int n, int c, int h, int w,
    int h_out, int w_out,
    int kh, int kw, int stride_h, int stride_w,
    cudaStream_t stream) {
    int total = n * c * h_out * w_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    max_pool2d_kernel<<<blocks, threads, 0, stream>>>(input, output, n, c, h, w, h_out, w_out, kh, kw, stride_h, stride_w);
}

void launch_avg_pool2d_kernel(const float* input, float* output,
    int n, int c, int h, int w,
    int h_out, int w_out,
    int kh, int kw, int stride_h, int stride_w,
    cudaStream_t stream) {
    int total = n * c * h_out * w_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(input, output, n, c, h, w, h_out, w_out, kh, kw, stride_h, stride_w);
}

void launch_dropout_kernel(const float* input, float* output,
    int n, float p, float scale, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dropout_kernel<<<blocks, threads, 0, stream>>>(input, output, n, p, scale);
}

}
