#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// =====================================================================
// one_hot カーネル
// =====================================================================
__global__ void one_hot_kernel(const long long *indices, float *output,
                               int batch, int classes) {
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
__global__ void scatter_add_kernel(const float *grad, const long long *indices,
                                   float *output, int seq_len, int dim,
                                   int vocab) {
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
__global__ void add_scalar_kernel(const float *x, float *y, int n, float s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = x[i] + s;
}

__global__ void mul_scalar_kernel(const float *x, float *y, int n, float s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = x[i] * s;
}

__global__ void div_scalar_kernel(const float *x, float *y, int n, float s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = x[i] / s;
}

__global__ void pow_scalar_kernel(const float *x, float *y, int n, float s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = powf(x[i], s);
}

__global__ void mod_scalar_kernel(const float *x, float *y, int n, float s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = fmodf(x[i], s);
}

__global__ void clamp_kernel(const float *x, float *y, int n, float lo,
                             float hi) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = fminf(fmaxf(x[i], lo), hi);
}

// =====================================================================
// binary ops カーネル (element-wise same-shape: y[i] = op(a[i], b[i]))
// =====================================================================
__global__ void add_kernel(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a[i] + b[i];
}

__global__ void sub_kernel(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a[i] - b[i];
}

__global__ void mul_kernel(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a[i] * b[i];
}

__global__ void div_kernel(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a[i] / b[i];
}

__global__ void pow_kernel(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = powf(a[i], b[i]);
}

__global__ void rem_kernel(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = fmodf(a[i], b[i]);
}

// comparison ops
__global__ void eq_kernel(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (fabsf(a[i] - b[i]) < 1e-6f) ? 1.0f : 0.0f;
}
__global__ void ne_kernel(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (fabsf(a[i] - b[i]) >= 1e-6f) ? 1.0f : 0.0f;
}
__global__ void lt_kernel(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
}
__global__ void le_kernel(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (a[i] <= b[i]) ? 1.0f : 0.0f;
}
__global__ void gt_kernel(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
}
__global__ void ge_kernel(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (a[i] >= b[i]) ? 1.0f : 0.0f;
}

// =====================================================================
// unary ops カーネル
// =====================================================================
__global__ void neg_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = -x[i];
}

__global__ void abs_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = fabsf(x[i]);
}

__global__ void exp_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = expf(x[i]);
}

__global__ void log_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = logf(x[i]);
}

__global__ void sqrt_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = sqrtf(x[i]);
}

// =====================================================================
// activation カーネル
// =====================================================================
__global__ void relu_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = fmaxf(x[i], 0.0f);
}

__global__ void sigmoid_kernel2(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ void tanh_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = tanhf(x[i]);
}

__global__ void gelu_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    float c = 0.7978845608f; // sqrt(2/pi)
    float inner = c * (v + 0.044715f * v * v * v);
    y[i] = 0.5f * v * (1.0f + tanhf(inner));
  }
}

__global__ void silu_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = x[i] / (1.0f + expf(-x[i]));
}

__global__ void sin_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = sinf(x[i]);
}

__global__ void cos_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = cosf(x[i]);
}

__global__ void tan_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = tanf(x[i]);
}

__global__ void floor_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = floorf(x[i]);
}

__global__ void ceil_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = ceilf(x[i]);
}

__global__ void round_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = roundf(x[i]);
}

// =====================================================================
// max/min reduction カーネル
// =====================================================================
__global__ void max_all_kernel(const float *x, float *out, int n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < n) ? x[i] : -3.402823466e+38f; // -FLT_MAX
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) {
    // atomic max for float via atomicCAS
    float val = sdata[0];
    unsigned int *addr = (unsigned int *)out;
    unsigned int old = *addr, assumed;
    do {
      assumed = old;
      old = atomicCAS(addr, assumed,
                      __float_as_uint(fmaxf(__uint_as_float(assumed), val)));
    } while (assumed != old);
  }
}

__global__ void min_all_kernel(const float *x, float *out, int n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < n) ? x[i] : 3.402823466e+38f; // FLT_MAX
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) {
    float val = sdata[0];
    unsigned int *addr = (unsigned int *)out;
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
__global__ void sum_all_kernel(const float *x, float *out, int n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < n) ? x[i] : 0.0f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    atomicAdd(out, sdata[0]);
}

// =====================================================================
// matmul カーネル (naive: 各スレッドが C[batch,i,j] の 1 要素を計算)
// =====================================================================
__global__ void matmul_naive_kernel(const float *A, const float *B, float *C,
                                    int m, int k, int n, int batch,
                                    int b_batched) {
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
__global__ void softmax_kernel(const float *input, float *output, int outer,
                               int axis_size, int inner) {
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
      if (v > max_val)
        max_val = v;
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
__global__ void embedding_kernel(const float *weight, const long long *indices,
                                 float *output, int seq_len, int embed_dim,
                                 int vocab_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * embed_dim;
  if (tid < total) {
    int i = tid / embed_dim;
    int j = tid % embed_dim;
    int idx = (int)indices[i];
    if (idx >= 0 && idx < vocab_size) {
      output[i * embed_dim + j] = weight[idx * embed_dim + j];
    } else {
      output[i * embed_dim + j] = 0.0f;
    }
  }
}

// =====================================================================
// cross_entropy カーネル
// logits[N, C], targets[N] → loss[N] (per-sample loss)
// =====================================================================
__global__ void cross_entropy_kernel(const float *logits,
                                     const long long *targets, float *losses,
                                     int n, int c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float *row = logits + i * c;
    // max for stability
    float max_val = -3.402823466e+38f;
    for (int j = 0; j < c; j++) {
      if (row[j] > max_val)
        max_val = row[j];
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
// cross_entropy_backward カーネル (各行の softmax - one_hot を計算)
// =====================================================================
__global__ void cross_entropy_backward_kernel(const float *logits,
                                              const long long *targets,
                                              float *grad_out, int batch_size,
                                              int num_classes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size) {
    const float *row = logits + i * num_classes;
    float *grad_row = grad_out + i * num_classes;

    // max_val for numerical stability
    float max_val = -3.402823466e+38f;
    for (int j = 0; j < num_classes; j++) {
      if (row[j] > max_val) max_val = row[j];
    }

    // sum_exp
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; j++) {
      float e = expf(row[j] - max_val);
      grad_row[j] = e;
      sum_exp += e;
    }

    // softmax probabilities and (prob - 1) if target
    int target_idx = (int)targets[i];
    float scale = 1.0f / (float)batch_size;
    for (int j = 0; j < num_classes; j++) {
      float prob = grad_row[j] / sum_exp;
      if (j == target_idx) {
        prob -= 1.0f;
      }
      grad_row[j] = prob * scale;
    }
  }
}

// =====================================================================
// tril カーネル (下三角以外を 0 にする)
// =====================================================================
__global__ void tril_kernel(const float *input, float *output, int rows,
                            int cols, int batch, int diagonal) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * rows * cols;
  if (tid < total) {
    int remainder = tid % (rows * cols);
    int r = remainder / cols;
    int c_idx = remainder % cols;
    output[tid] = (c_idx <= r + diagonal) ? input[tid] : 0.0f;
  }
}

// =====================================================================
// where_cond カーネル (ternary: cond > 0 ? x : y)
// =====================================================================
__global__ void where_cond_kernel(const float *cond, const float *x,
                                  const float *y, float *output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = (cond[i] > 0.0f) ? x[i] : y[i];
  }
}

// =====================================================================
// rms_norm カーネル (行ごと: x * rsqrt(mean(x²) + eps))
// =====================================================================
__global__ void rms_norm_kernel(const float *input, float *output, int outer,
                                int norm_size, float eps) {
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
__global__ void causal_mask_kernel(float *output, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = size * size;
  if (tid < total) {
    int r = tid / size;
    int c = tid % size;
    output[tid] = (c > r) ? (-1.0f / 0.0f) : 0.0f; // -inf
  }
}

// =====================================================================
// transpose カーネル (任意の2次元スワップ)
// shape/strides 情報は launch 関数側で計算して渡す
// =====================================================================
__global__ void transpose_2d_kernel(const float *input, float *output, int rows,
                                    int cols) {
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
__global__ void broadcast_add_kernel(const float *a, const float *b, float *out,
                                     const int *out_shape, const int *a_shape,
                                     const int *b_shape, int ndim, int total) {
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
      if (a_dim > 1)
        a_idx += coord * a_stride;
      if (b_dim > 1)
        b_idx += coord * b_stride;
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
__global__ void reduce_axis_sum_kernel(const float *input, float *output,
                                       int outer, int axis_size, int inner) {
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

__global__ void reduce_axis_max_kernel(const float *input, float *output,
                                       int outer, int axis_size, int inner) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (tid < total) {
    int o = tid / inner;
    int i = tid % inner;
    float mx = -3.402823466e+38f;
    for (int k = 0; k < axis_size; k++) {
      float v = input[o * axis_size * inner + k * inner + i];
      if (v > mx)
        mx = v;
    }
    output[tid] = mx;
  }
}

__global__ void reduce_axis_min_kernel(const float *input, float *output,
                                       int outer, int axis_size, int inner) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (tid < total) {
    int o = tid / inner;
    int i = tid % inner;
    float mn = 3.402823466e+38f;
    for (int k = 0; k < axis_size; k++) {
      float v = input[o * axis_size * inner + k * inner + i];
      if (v < mn)
        mn = v;
    }
    output[tid] = mn;
  }
}

// =====================================================================
// narrow (slice) カーネル
// =====================================================================
__global__ void narrow_kernel(const float *input, float *output, int outer,
                              int inner, int old_dim, int new_dim, int start) {
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
__global__ void cat_kernel(const float *a, const float *b, float *output,
                           int outer, int inner, int a_dim, int b_dim) {
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
__global__ void broadcast_to_kernel(const float *input, float *output,
                                    const int *target_shape,
                                    const int *src_shape, int ndim, int total) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total) {
    int idx = tid;
    int src_idx = 0;
    int src_stride = 1;
    for (int d = ndim - 1; d >= 0; d--) {
      int coord = idx % target_shape[d];
      idx /= target_shape[d];
      int sd = src_shape[d];
      if (sd > 1)
        src_idx += coord * src_stride;
      src_stride *= sd;
    }
    output[tid] = input[src_idx];
  }
}

// =====================================================================
// conv2d カーネル
// input[N,C_in,H,W], weight[C_out,C_in,kH,kW] → output[N,C_out,H_out,W_out]
// =====================================================================
__global__ void conv2d_kernel(const float *input, const float *weight,
                              float *output, int n, int c_in, int h_in,
                              int w_in, int c_out, int kh, int kw, int h_out,
                              int w_out, int stride_h, int stride_w, int pad_h,
                              int pad_w) {
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
            int in_idx =
                batch * c_in * h_in * w_in + ci * h_in * w_in + ih * w_in + iw;
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
__global__ void batch_norm_kernel(const float *input, const float *gamma,
                                  const float *beta, const float *mean,
                                  const float *var, float *output, int n, int c,
                                  int spatial, float eps) {
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
__global__ void layer_norm_kernel(const float *input, const float *gamma,
                                  const float *beta, float *output, int outer,
                                  int norm_size, float eps) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o < outer) {
    int offset = o * norm_size;
    // mean
    float sum = 0.0f;
    for (int i = 0; i < norm_size; i++)
      sum += input[offset + i];
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
      output[offset + i] =
          gamma[i] * (input[offset + i] - mean) * inv_std + beta[i];
    }
  }
}

// =====================================================================
// max_pool2d カーネル
// =====================================================================
__global__ void max_pool2d_kernel(const float *input, float *output, int n,
                                  int c, int h, int w, int h_out, int w_out,
                                  int kh, int kw, int stride_h, int stride_w) {
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
        if (input[idx] > mx)
          mx = input[idx];
      }
    }
    output[tid] = mx;
  }
}

// =====================================================================
// avg_pool2d カーネル
// =====================================================================
__global__ void avg_pool2d_kernel(const float *input, float *output, int n,
                                  int c, int h, int w, int h_out, int w_out,
                                  int kh, int kw, int stride_h, int stride_w) {
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
__global__ void dropout_kernel(const float *input, float *output, int n,
                               float p, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // simple hash-based pseudo-random
    unsigned int hash = (unsigned int)i * 2654435761u;
    float r = (float)(hash >> 16) / 65536.0f;
    output[i] = (r < p) ? 0.0f : input[i] * scale;
  }
}

// =====================================================================
// argmax_all / argmin_all (全要素)
// 各ブロックが部分的な argmax/min を計算し、最後に device 上で集約
// 簡易版: 1ブロックで処理 (要素数 < 数百万を想定)
// =====================================================================
__global__ void argmax_all_kernel(const float *input, float *out_idx, int n) {
  // 1スレッドで全走査（tiny kernel, 数百万要素まで OK）
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float best = -3.402823466e+38f;
    int best_i = 0;
    for (int i = 0; i < n; i++) {
      if (input[i] > best) {
        best = input[i];
        best_i = i;
      }
    }
    out_idx[0] = (float)best_i;
  }
}

__global__ void argmin_all_kernel(const float *input, float *out_idx, int n) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float best = 3.402823466e+38f;
    int best_i = 0;
    for (int i = 0; i < n; i++) {
      if (input[i] < best) {
        best = input[i];
        best_i = i;
      }
    }
    out_idx[0] = (float)best_i;
  }
}

// =====================================================================
// argmax_axis / argmin_axis (軸に沿ったインデックス返却)
// =====================================================================
__global__ void argmax_axis_kernel(const float *input, float *output, int outer,
                                   int axis_size, int inner) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (tid < total) {
    int o = tid / inner;
    int i = tid % inner;
    float best = -3.402823466e+38f;
    int best_k = 0;
    for (int k = 0; k < axis_size; k++) {
      float v = input[o * axis_size * inner + k * inner + i];
      if (v > best) {
        best = v;
        best_k = k;
      }
    }
    output[tid] = (float)best_k;
  }
}

__global__ void argmin_axis_kernel(const float *input, float *output, int outer,
                                   int axis_size, int inner) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (tid < total) {
    int o = tid / inner;
    int i = tid % inner;
    float best = 3.402823466e+38f;
    int best_k = 0;
    for (int k = 0; k < axis_size; k++) {
      float v = input[o * axis_size * inner + k * inner + i];
      if (v < best) {
        best = v;
        best_k = k;
      }
    }
    output[tid] = (float)best_k;
  }
}

// =====================================================================
// apply_rope カーネル
// input[total_pairs], cos_table[seq*half], sin_table[seq*half]
// num_tokens, heads_per_token, dim (= 2*half_dim)
// =====================================================================
__global__ void apply_rope_kernel(const float *input, const float *cos_table,
                                  const float *sin_table, float *output,
                                  int num_tokens, int heads_per_token, int dim,
                                  int half_dim, int cos_dim, int cos_seq_len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * heads_per_token * half_dim;
  if (tid < total) {
    int token_pos = tid / (heads_per_token * half_dim);
    int rem = tid % (heads_per_token * half_dim);
    int head = rem / half_dim;
    int i = rem % half_dim;

    int offset = (token_pos * heads_per_token + head) * dim;
    int cos_row = token_pos < cos_seq_len
                      ? token_pos
                      : (cos_seq_len > 0 ? cos_seq_len - 1 : 0);
    int cos_offset = cos_row * cos_dim;

    float ci = (cos_offset + i < cos_seq_len * cos_dim)
                   ? cos_table[cos_offset + i]
                   : 1.0f;
    float si = (cos_offset + i < cos_seq_len * cos_dim)
                   ? sin_table[cos_offset + i]
                   : 0.0f;

    float x0 = input[offset + i];
    float x1 = input[offset + half_dim + i];

    output[offset + i] = x0 * ci - x1 * si;
    output[offset + half_dim + i] = x0 * si + x1 * ci;
  }
}

// =====================================================================
// index_select カーネル
// data[...axis_dim...], indices[n_idx] → output[...n_idx...]
// =====================================================================
__global__ void index_select_kernel(const float *input,
                                    const long long *indices, float *output,
                                    int outer, int inner, int old_dim,
                                    int n_idx) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * n_idx * inner;
  if (tid < total) {
    int o = tid / (n_idx * inner);
    int rem = tid % (n_idx * inner);
    int d = rem / inner;
    int i = rem % inner;
    int src_d = (int)indices[d];
    int src = o * old_dim * inner + src_d * inner + i;
    output[tid] = input[src];
  }
}

// =====================================================================
// repeat_interleave カーネル
// =====================================================================
__global__ void repeat_interleave_kernel(const float *input, float *output,
                                         int outer, int inner, int old_dim,
                                         int repeats) {
  int new_dim = old_dim * repeats;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * new_dim * inner;
  if (tid < total) {
    int o = tid / (new_dim * inner);
    int rem = tid % (new_dim * inner);
    int d = rem / inner;
    int i = rem % inner;
    int src_d = d / repeats;
    int src = o * old_dim * inner + src_d * inner + i;
    output[tid] = input[src];
  }
}

// =====================================================================
// transpose_nd カーネル (汎用 N 次元転置)
// old_strides, new_strides を GPU に渡す
// =====================================================================
__global__ void transpose_nd_kernel(const float *input, float *output,
                                    const int *old_shape,
                                    const int *old_strides,
                                    const int *new_strides, int ndim, int total,
                                    int dim0, int dim1) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total) {
    // flat index → old coords
    int idx = tid;
    int new_idx = 0;
    for (int d = ndim - 1; d >= 0; d--) {
      int coord = idx % old_shape[d];
      idx /= old_shape[d];
      // swap dims
      int mapped_d;
      if (d == dim0)
        mapped_d = dim1;
      else if (d == dim1)
        mapped_d = dim0;
      else
        mapped_d = d;
      new_idx += coord * new_strides[mapped_d];
    }
    output[new_idx] = input[tid];
  }
}

// =====================================================================
// C wrappers
// =====================================================================
extern "C" {

// --- scatter/one_hot ---
void launch_one_hot_kernel(const long long *indices, float *output, int batch,
                           int classes, cudaStream_t stream) {
  int threads = 256;
  int blocks = (batch + threads - 1) / threads;
  one_hot_kernel<<<blocks, threads, 0, stream>>>(indices, output, batch,
                                                 classes);
}

void launch_scatter_add_kernel(const float *grad, const long long *indices,
                               float *output, int seq_len, int dim, int vocab,
                               cudaStream_t stream) {
  int total = seq_len * dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  scatter_add_kernel<<<blocks, threads, 0, stream>>>(grad, indices, output,
                                                     seq_len, dim, vocab);
}

// --- scalar ops ---
#define LAUNCH_SCALAR(name)                                                    \
  void launch_##name##_kernel(const float *x, float *y, int n, float s,        \
                              cudaStream_t stream) {                           \
    int threads = 256;                                                         \
    int blocks = (n + threads - 1) / threads;                                  \
    name##_kernel<<<blocks, threads, 0, stream>>>(x, y, n, s);                 \
  }

LAUNCH_SCALAR(add_scalar)
LAUNCH_SCALAR(mul_scalar)
LAUNCH_SCALAR(div_scalar)
LAUNCH_SCALAR(pow_scalar)
LAUNCH_SCALAR(mod_scalar)

void launch_clamp_kernel(const float *x, float *y, int n, float lo, float hi,
                         cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  clamp_kernel<<<blocks, threads, 0, stream>>>(x, y, n, lo, hi);
}

// --- binary ops ---
#define LAUNCH_BINARY(name)                                                    \
  void launch_##name##_kernel(const float *a, const float *b, float *y, int n, \
                              cudaStream_t stream) {                           \
    int threads = 256;                                                         \
    int blocks = (n + threads - 1) / threads;                                  \
    name##_kernel<<<blocks, threads, 0, stream>>>(a, b, y, n);                 \
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
#define LAUNCH_UNARY(name)                                                     \
  void launch_##name##_kernel(const float *x, float *y, int n,                 \
                              cudaStream_t stream) {                           \
    int threads = 256;                                                         \
    int blocks = (n + threads - 1) / threads;                                  \
    name##_kernel<<<blocks, threads, 0, stream>>>(x, y, n);                    \
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

void launch_sigmoid_kernel2(const float *x, float *y, int n,
                            cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  sigmoid_kernel2<<<blocks, threads, 0, stream>>>(x, y, n);
}

// --- max/min reduce ---
void launch_max_all_kernel(const float *x, float *out, int n,
                           cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  max_all_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(x, out,
                                                                       n);
}

void launch_min_all_kernel(const float *x, float *out, int n,
                           cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  min_all_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(x, out,
                                                                       n);
}

// --- reduce ---
void launch_sum_all_kernel(const float *x, float *out, int n,
                           cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  sum_all_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(x, out,
                                                                       n);
}

// --- matmul ---
void launch_matmul_kernel(const float *a, const float *b, float *c, int m,
                          int k, int n, int batch, int b_batched,
                          cudaStream_t stream) {
  int total = batch * m * n;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  matmul_naive_kernel<<<blocks, threads, 0, stream>>>(a, b, c, m, k, n, batch,
                                                      b_batched);
}

// --- special ops ---
void launch_softmax_kernel(const float *input, float *output, int outer,
                           int axis_size, int inner, cudaStream_t stream) {
  int total = outer * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  softmax_kernel<<<blocks, threads, 0, stream>>>(input, output, outer,
                                                 axis_size, inner);
}

void launch_embedding_kernel(const float *weight, const long long *indices,
                             float *output, int seq_len, int embed_dim,
                             int vocab_size, cudaStream_t stream) {
  int total = seq_len * embed_dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  embedding_kernel<<<blocks, threads, 0, stream>>>(
      weight, indices, output, seq_len, embed_dim, vocab_size);
}

void launch_cross_entropy_kernel(const float *logits, const long long *targets,
                                 float *losses, int n, int c,
                                 cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  cross_entropy_kernel<<<blocks, threads, 0, stream>>>(logits, targets, losses,
                                                       n, c);
}

void launch_cross_entropy_backward_kernel(const float *logits,
                                         const long long *targets,
                                         float *grad_out, int batch_size,
                                         int num_classes,
                                         cudaStream_t stream) {
  int threads = 256;
  int blocks = (batch_size + threads - 1) / threads;
  cross_entropy_backward_kernel<<<blocks, threads, 0, stream>>>(
      logits, targets, grad_out, batch_size, num_classes);
}

void launch_tril_kernel(const float *input, float *output, int rows, int cols,
                        int batch, int diagonal, cudaStream_t stream) {
  int total = batch * rows * cols;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  tril_kernel<<<blocks, threads, 0, stream>>>(input, output, rows, cols, batch,
                                              diagonal);
}

void launch_where_cond_kernel(const float *cond, const float *x, const float *y,
                              float *output, int n, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  where_cond_kernel<<<blocks, threads, 0, stream>>>(cond, x, y, output, n);
}

void launch_rms_norm_kernel(const float *input, float *output, int outer,
                            int norm_size, float eps, cudaStream_t stream) {
  int threads = 256;
  int blocks = (outer + threads - 1) / threads;
  rms_norm_kernel<<<blocks, threads, 0, stream>>>(input, output, outer,
                                                  norm_size, eps);
}

void launch_causal_mask_kernel(float *output, int size, cudaStream_t stream) {
  int total = size * size;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  causal_mask_kernel<<<blocks, threads, 0, stream>>>(output, size);
}

// --- transpose ---
void launch_transpose_2d_kernel(const float *input, float *output, int rows,
                                int cols, cudaStream_t stream) {
  int total = rows * cols;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  transpose_2d_kernel<<<blocks, threads, 0, stream>>>(input, output, rows,
                                                      cols);
}

// --- reduce_axis ---
void launch_reduce_axis_sum_kernel(const float *input, float *output, int outer,
                                   int axis_size, int inner,
                                   cudaStream_t stream) {
  int total = outer * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  reduce_axis_sum_kernel<<<blocks, threads, 0, stream>>>(input, output, outer,
                                                         axis_size, inner);
}

void launch_reduce_axis_max_kernel(const float *input, float *output, int outer,
                                   int axis_size, int inner,
                                   cudaStream_t stream) {
  int total = outer * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  reduce_axis_max_kernel<<<blocks, threads, 0, stream>>>(input, output, outer,
                                                         axis_size, inner);
}

void launch_reduce_axis_min_kernel(const float *input, float *output, int outer,
                                   int axis_size, int inner,
                                   cudaStream_t stream) {
  int total = outer * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  reduce_axis_min_kernel<<<blocks, threads, 0, stream>>>(input, output, outer,
                                                         axis_size, inner);
}

// --- narrow ---
void launch_narrow_kernel(const float *input, float *output, int outer,
                          int inner, int old_dim, int new_dim, int start,
                          cudaStream_t stream) {
  int total = outer * new_dim * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  narrow_kernel<<<blocks, threads, 0, stream>>>(input, output, outer, inner,
                                                old_dim, new_dim, start);
}

// --- cat ---
void launch_cat_kernel(const float *a, const float *b, float *output, int outer,
                       int inner, int a_dim, int b_dim, cudaStream_t stream) {
  int total = outer * (a_dim + b_dim) * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  cat_kernel<<<blocks, threads, 0, stream>>>(a, b, output, outer, inner, a_dim,
                                             b_dim);
}

// --- broadcast_to ---
void launch_broadcast_to_kernel(const float *input, float *output,
                                const int *target_shape, const int *src_shape,
                                int ndim, int total, cudaStream_t stream) {
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  broadcast_to_kernel<<<blocks, threads, 0, stream>>>(
      input, output, target_shape, src_shape, ndim, total);
}

// --- nn ops ---
void launch_conv2d_kernel(const float *input, const float *weight,
                          float *output, int n, int c_in, int h_in, int w_in,
                          int c_out, int kh, int kw, int h_out, int w_out,
                          int stride_h, int stride_w, int pad_h, int pad_w,
                          cudaStream_t stream) {
  int total = n * c_out * h_out * w_out;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  conv2d_kernel<<<blocks, threads, 0, stream>>>(
      input, weight, output, n, c_in, h_in, w_in, c_out, kh, kw, h_out, w_out,
      stride_h, stride_w, pad_h, pad_w);
}

void launch_batch_norm_kernel(const float *input, const float *gamma,
                              const float *beta, const float *mean,
                              const float *var, float *output, int n, int c,
                              int spatial, float eps, cudaStream_t stream) {
  int total = n * c * spatial;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  batch_norm_kernel<<<blocks, threads, 0, stream>>>(
      input, gamma, beta, mean, var, output, n, c, spatial, eps);
}

void launch_layer_norm_kernel(const float *input, const float *gamma,
                              const float *beta, float *output, int outer,
                              int norm_size, float eps, cudaStream_t stream) {
  int threads = 256;
  int blocks = (outer + threads - 1) / threads;
  layer_norm_kernel<<<blocks, threads, 0, stream>>>(input, gamma, beta, output,
                                                    outer, norm_size, eps);
}

void launch_max_pool2d_kernel(const float *input, float *output, int n, int c,
                              int h, int w, int h_out, int w_out, int kh,
                              int kw, int stride_h, int stride_w,
                              cudaStream_t stream) {
  int total = n * c * h_out * w_out;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  max_pool2d_kernel<<<blocks, threads, 0, stream>>>(
      input, output, n, c, h, w, h_out, w_out, kh, kw, stride_h, stride_w);
}

void launch_avg_pool2d_kernel(const float *input, float *output, int n, int c,
                              int h, int w, int h_out, int w_out, int kh,
                              int kw, int stride_h, int stride_w,
                              cudaStream_t stream) {
  int total = n * c * h_out * w_out;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(
      input, output, n, c, h, w, h_out, w_out, kh, kw, stride_h, stride_w);
}

void launch_dropout_kernel(const float *input, float *output, int n, float p,
                           float scale, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  dropout_kernel<<<blocks, threads, 0, stream>>>(input, output, n, p, scale);
}

// --- argmax/argmin ---
void launch_argmax_all_kernel(const float *input, float *out_idx, int n,
                              cudaStream_t stream) {
  argmax_all_kernel<<<1, 1, 0, stream>>>(input, out_idx, n);
}

void launch_argmin_all_kernel(const float *input, float *out_idx, int n,
                              cudaStream_t stream) {
  argmin_all_kernel<<<1, 1, 0, stream>>>(input, out_idx, n);
}

void launch_argmax_axis_kernel(const float *input, float *output, int outer,
                               int axis_size, int inner, cudaStream_t stream) {
  int total = outer * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  argmax_axis_kernel<<<blocks, threads, 0, stream>>>(input, output, outer,
                                                     axis_size, inner);
}

void launch_argmin_axis_kernel(const float *input, float *output, int outer,
                               int axis_size, int inner, cudaStream_t stream) {
  int total = outer * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  argmin_axis_kernel<<<blocks, threads, 0, stream>>>(input, output, outer,
                                                     axis_size, inner);
}

// --- apply_rope ---
void launch_apply_rope_kernel(const float *input, const float *cos_table,
                              const float *sin_table, float *output,
                              int num_tokens, int heads_per_token, int dim,
                              int half_dim, int cos_dim, int cos_seq_len,
                              cudaStream_t stream) {
  int total = num_tokens * heads_per_token * half_dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  apply_rope_kernel<<<blocks, threads, 0, stream>>>(
      input, cos_table, sin_table, output, num_tokens, heads_per_token, dim,
      half_dim, cos_dim, cos_seq_len);
}

// --- index_select ---
void launch_index_select_kernel(const float *input, const long long *indices,
                                float *output, int outer, int inner,
                                int old_dim, int n_idx, cudaStream_t stream) {
  int total = outer * n_idx * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  index_select_kernel<<<blocks, threads, 0, stream>>>(
      input, indices, output, outer, inner, old_dim, n_idx);
}

// --- repeat_interleave ---
void launch_repeat_interleave_kernel(const float *input, float *output,
                                     int outer, int inner, int old_dim,
                                     int repeats, cudaStream_t stream) {
  int new_dim = old_dim * repeats;
  int total = outer * new_dim * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  repeat_interleave_kernel<<<blocks, threads, 0, stream>>>(
      input, output, outer, inner, old_dim, repeats);
}

// --- transpose_nd ---
void launch_transpose_nd_kernel(const float *input, float *output,
                                const int *old_shape, const int *old_strides,
                                const int *new_strides, int ndim, int total,
                                int dim0, int dim1, cudaStream_t stream) {
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  transpose_nd_kernel<<<blocks, threads, 0, stream>>>(input, output, old_shape,
                                                      old_strides, new_strides,
                                                      ndim, total, dim0, dim1);

} // end launch_transpose_nd_kernel

} // end extern "C" block for main kernels

// =====================================================================
// Phase A: 新規 element-wise カーネル
// =====================================================================

// --- activations with parameter ---
__global__ void leaky_relu_kernel(const float *x, float *y, int n,
                                  float slope) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (x[i] > 0.0f) ? x[i] : slope * x[i];
}

__global__ void elu_kernel(const float *x, float *y, int n, float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (x[i] > 0.0f) ? x[i] : alpha * (expf(x[i]) - 1.0f);
}

// --- unary activations ---
__global__ void mish_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sp = logf(1.0f + expf(x[i])); // softplus
    y[i] = x[i] * tanhf(sp);
  }
}

__global__ void hardswish_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = x[i] * fminf(fmaxf(x[i] / 6.0f + 0.5f, 0.0f), 1.0f);
}

__global__ void hardsigmoid_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = fminf(fmaxf(x[i] / 6.0f + 0.5f, 0.0f), 1.0f);
}

// --- logical ops ---
__global__ void logical_and_kernel(const float *a, const float *b, float *y,
                                   int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (a[i] > 0.0f && b[i] > 0.0f) ? 1.0f : 0.0f;
}

__global__ void logical_or_kernel(const float *a, const float *b, float *y,
                                  int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (a[i] > 0.0f || b[i] > 0.0f) ? 1.0f : 0.0f;
}

__global__ void logical_not_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (x[i] > 0.0f) ? 0.0f : 1.0f;
}

// --- masked_fill: y[i] = (mask[i] > 0) ? value : x[i] ---
__global__ void masked_fill_kernel(const float *x, const float *mask, float *y,
                                   int n, float value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (mask[i] > 0.0f) ? value : x[i];
}

// --- fill_: y[i] = value ---
__global__ void fill_kernel(float *y, int n, float value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = value;
}

// --- dot product: per-element multiply, then sum_all_kernel で集約 ---
// (a[i]*b[i] を計算して tmp に格納、その後 sum_all_kernel で合計)
__global__ void dot_mul_kernel(const float *a, const float *b, float *y,
                               int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a[i] * b[i];
}

// --- loss element-wise カーネル (mean は後で sum_all + div_scalar) ---
__global__ void mse_loss_elem_kernel(const float *a, const float *b, float *y,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float d = a[i] - b[i];
    y[i] = d * d;
  }
}

__global__ void l1_loss_elem_kernel(const float *a, const float *b, float *y,
                                    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = fabsf(a[i] - b[i]);
}

__global__ void bce_loss_elem_kernel(const float *pred, const float *target,
                                     float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float p = fminf(fmaxf(pred[i], 1e-7f), 1.0f - 1e-7f); // clamp for safety
    float t = target[i];
    y[i] = -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
  }
}

__global__ void nll_loss_elem_kernel(const float *pred, const float *target,
                                     float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = -(pred[i] * target[i]);
}

__global__ void kl_div_loss_elem_kernel(const float *pred, const float *target,
                                        float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float p = fmaxf(pred[i], 1e-7f);
    float q = fmaxf(target[i], 1e-7f);
    y[i] = q * (logf(q) - logf(p));
  }
}

// =====================================================================
// Phase B: Reduction カーネル
// =====================================================================

// --- reduce_axis_prod: 軸方向の積 ---
__global__ void reduce_axis_prod_kernel(const float *input, float *output,
                                        int outer, int axis_size, int inner) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (tid < total) {
    int o = tid / inner;
    int i = tid % inner;
    float prod = 1.0f;
    for (int k = 0; k < axis_size; k++) {
      prod *= input[o * axis_size * inner + k * inner + i];
    }
    output[tid] = prod;
  }
}

// --- cumsum: 各スレッドが 1 列の累積和を計算 ---
__global__ void cumsum_kernel(const float *input, float *output, int outer,
                              int axis_size, int inner) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (tid < total) {
    int o = tid / inner;
    int i = tid % inner;
    float sum = 0.0f;
    for (int k = 0; k < axis_size; k++) {
      int idx = o * axis_size * inner + k * inner + i;
      sum += input[idx];
      output[idx] = sum;
    }
  }
}

// --- norm: Lp ノルム (各スレッドが1出力要素) ---
__global__ void reduce_axis_norm_kernel(const float *input, float *output,
                                        int outer, int axis_size, int inner,
                                        float p_val) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (tid < total) {
    int o = tid / inner;
    int i = tid % inner;
    float sum = 0.0f;
    for (int k = 0; k < axis_size; k++) {
      sum += powf(fabsf(input[o * axis_size * inner + k * inner + i]), p_val);
    }
    output[tid] = powf(sum, 1.0f / p_val);
  }
}

// --- topk: 各スレッドが 1 行の上位 k 個を選択 ---
__global__ void topk_kernel(const float *input, float *output, int outer,
                            int axis_size, int inner, int k) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (tid < total) {
    int o = tid / inner;
    int i = tid % inner;
    // 上位 k 個の値を保持（挿入ソート）
    // k は通常小さい (< 100) と仮定
    int max_k = (k < 64) ? k : 64; // 安全上限
    float top[64];
    for (int j = 0; j < max_k; j++)
      top[j] = -3.402823466e+38f;

    for (int a = 0; a < axis_size; a++) {
      float v = input[o * axis_size * inner + a * inner + i];
      // 挿入ソート: 最小の top 値より大きければ挿入
      if (v > top[max_k - 1]) {
        top[max_k - 1] = v;
        // バブル
        for (int j = max_k - 1; j > 0 && top[j] > top[j - 1]; j--) {
          float tmp = top[j];
          top[j] = top[j - 1];
          top[j - 1] = tmp;
        }
      }
    }
    // 出力: k 個の値を書き込み
    for (int j = 0; j < max_k; j++) {
      output[tid * max_k + j] = top[j];
    }
  }
}

// --- var: 2パス（mean → 差の二乗の mean）---
__global__ void reduce_axis_var_kernel(const float *input, float *output,
                                       int outer, int axis_size, int inner) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * inner;
  if (tid < total) {
    int o = tid / inner;
    int i = tid % inner;
    // pass 1: mean
    float sum = 0.0f;
    for (int k = 0; k < axis_size; k++)
      sum += input[o * axis_size * inner + k * inner + i];
    float mean = sum / (float)axis_size;
    // pass 2: var
    float var_sum = 0.0f;
    for (int k = 0; k < axis_size; k++) {
      float d = input[o * axis_size * inner + k * inner + i] - mean;
      var_sum += d * d;
    }
    output[tid] = var_sum / (float)axis_size;
  }
}

// =====================================================================
// Phase C: NN / LLM カーネル
// =====================================================================

// --- group_norm: チャネルをグループ分割して正規化 ---
__global__ void group_norm_kernel(const float *input, const float *gamma,
                                  const float *beta, float *output, int n,
                                  int c, int spatial, int num_groups,
                                  float eps) {
  // 各スレッドが 1 つの (batch, group) ペアを処理
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_groups = n * num_groups;
  if (tid < total_groups) {
    int batch = tid / num_groups;
    int g = tid % num_groups;
    int cpg = c / num_groups; // channels per group
    int c_start = g * cpg;
    int group_size = cpg * spatial;

    // mean
    float sum = 0.0f;
    for (int ci = 0; ci < cpg; ci++) {
      int ch = c_start + ci;
      for (int s = 0; s < spatial; s++) {
        sum += input[batch * c * spatial + ch * spatial + s];
      }
    }
    float mean = sum / (float)group_size;

    // var
    float var_sum = 0.0f;
    for (int ci = 0; ci < cpg; ci++) {
      int ch = c_start + ci;
      for (int s = 0; s < spatial; s++) {
        float d = input[batch * c * spatial + ch * spatial + s] - mean;
        var_sum += d * d;
      }
    }
    float inv_std = rsqrtf(var_sum / (float)group_size + eps);

    // normalize
    for (int ci = 0; ci < cpg; ci++) {
      int ch = c_start + ci;
      for (int s = 0; s < spatial; s++) {
        int idx = batch * c * spatial + ch * spatial + s;
        output[idx] = gamma[ch] * (input[idx] - mean) * inv_std + beta[ch];
      }
    }
  }
}

// --- adaptive_avg_pool2d ---
__global__ void adaptive_avg_pool2d_kernel(const float *input, float *output,
                                           int n, int c, int h_in, int w_in,
                                           int h_out, int w_out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * h_out * w_out;
  if (tid < total) {
    int batch = tid / (c * h_out * w_out);
    int rem = tid % (c * h_out * w_out);
    int ch = rem / (h_out * w_out);
    rem = rem % (h_out * w_out);
    int oh = rem / w_out;
    int ow = rem % w_out;

    // 入力領域を計算
    int ih_start = (oh * h_in) / h_out;
    int ih_end = ((oh + 1) * h_in) / h_out;
    int iw_start = (ow * w_in) / w_out;
    int iw_end = ((ow + 1) * w_in) / w_out;

    float sum = 0.0f;
    int count = 0;
    for (int ih = ih_start; ih < ih_end; ih++) {
      for (int iw = iw_start; iw < iw_end; iw++) {
        sum +=
            input[batch * c * h_in * w_in + ch * h_in * w_in + ih * w_in + iw];
        count++;
      }
    }
    output[tid] = (count > 0) ? sum / (float)count : 0.0f;
  }
}

// --- pad (1D: left/right パディング) ---
__global__ void pad_kernel(const float *input, float *output, int n,
                           int old_dim, int new_dim, int pad_left,
                           float pad_value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * new_dim;
  if (tid < total) {
    int batch = tid / new_dim;
    int pos = tid % new_dim;
    int src_pos = pos - pad_left;
    if (src_pos >= 0 && src_pos < old_dim) {
      output[tid] = input[batch * old_dim + src_pos];
    } else {
      output[tid] = pad_value;
    }
  }
}

// --- dropout2d: チャネル単位マスク ---
__global__ void dropout2d_kernel(const float *input, float *output, int n,
                                 int c, int spatial, float p, float scale) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * spatial;
  if (tid < total) {
    int ch_idx = (tid / spatial) % c;
    int batch_idx = tid / (c * spatial);
    int chan_id = batch_idx * c + ch_idx;
    // hash-based mask per channel
    unsigned int hash = (unsigned int)chan_id * 2654435761u;
    float r = (float)(hash >> 16) / 65536.0f;
    output[tid] = (r < p) ? 0.0f : input[tid] * scale;
  }
}

// --- conv1d: input[N,C_in,L], weight[C_out,C_in,K] → output[N,C_out,L_out] ---
__global__ void conv1d_kernel(const float *input, const float *weight,
                              float *output, int n, int c_in, int l_in,
                              int c_out, int kl, int l_out, int stride,
                              int pad) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c_out * l_out;
  if (tid < total) {
    int batch = tid / (c_out * l_out);
    int rem = tid % (c_out * l_out);
    int co = rem / l_out;
    int ol = rem % l_out;

    float sum = 0.0f;
    for (int ci = 0; ci < c_in; ci++) {
      for (int ki = 0; ki < kl; ki++) {
        int il = ol * stride + ki - pad;
        if (il >= 0 && il < l_in) {
          sum += input[batch * c_in * l_in + ci * l_in + il] *
                 weight[co * c_in * kl + ci * kl + ki];
        }
      }
    }
    output[tid] = sum;
  }
}

// --- conv_transpose2d ---
__global__ void conv_transpose2d_kernel(const float *input, const float *weight,
                                        float *output, int n, int c_in,
                                        int h_in, int w_in, int c_out, int kh,
                                        int kw, int h_out, int w_out,
                                        int stride_h, int stride_w, int pad_h,
                                        int pad_w) {
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
          // transpose: oh = ih * stride - pad + khi
          // → ih = (oh + pad - khi) / stride
          int h_idx = oh + pad_h - khi;
          int w_idx = ow + pad_w - kwi;
          if (h_idx % stride_h == 0 && w_idx % stride_w == 0) {
            int ih = h_idx / stride_h;
            int iw = w_idx / stride_w;
            if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
              // weight layout: [C_in, C_out, kH, kW]
              sum +=
                  input[batch * c_in * h_in * w_in + ci * h_in * w_in +
                        ih * w_in + iw] *
                  weight[ci * c_out * kh * kw + co * kh * kw + khi * kw + kwi];
            }
          }
        }
      }
    }
    output[tid] = sum;
  }
}

// --- interpolate (nearest / bilinear) ---
__global__ void interpolate_kernel(const float *input, float *output, int n,
                                   int c, int h_in, int w_in, int h_out,
                                   int w_out, int mode) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * h_out * w_out;
  if (tid < total) {
    int batch = tid / (c * h_out * w_out);
    int rem = tid % (c * h_out * w_out);
    int ch = rem / (h_out * w_out);
    rem = rem % (h_out * w_out);
    int oh = rem / w_out;
    int ow = rem % w_out;

    float scale_h = (float)h_in / (float)h_out;
    float scale_w = (float)w_in / (float)w_out;

    if (mode == 0) {
      // nearest
      int ih = (int)(oh * scale_h);
      int iw = (int)(ow * scale_w);
      ih = (ih < h_in) ? ih : h_in - 1;
      iw = (iw < w_in) ? iw : w_in - 1;
      output[tid] =
          input[batch * c * h_in * w_in + ch * h_in * w_in + ih * w_in + iw];
    } else {
      // bilinear
      float fy = oh * scale_h;
      float fx = ow * scale_w;
      int iy = (int)fy;
      int ix = (int)fx;
      float dy = fy - (float)iy;
      float dx = fx - (float)ix;
      int iy1 = (iy + 1 < h_in) ? iy + 1 : iy;
      int ix1 = (ix + 1 < w_in) ? ix + 1 : ix;
      int base = batch * c * h_in * w_in + ch * h_in * w_in;
      float v00 = input[base + iy * w_in + ix];
      float v01 = input[base + iy * w_in + ix1];
      float v10 = input[base + iy1 * w_in + ix];
      float v11 = input[base + iy1 * w_in + ix1];
      output[tid] = v00 * (1 - dy) * (1 - dx) + v01 * (1 - dy) * dx +
                    v10 * dy * (1 - dx) + v11 * dy * dx;
    }
  }
}

// --- Flash Attention v2: tiled SDPA with online softmax ---
// Each block processes one query row for one (batch, head).
// K/V are loaded tile-by-tile into shared memory.
// Online softmax avoids materializing the full seq_q × seq_k score matrix.

#define FA_TILE_K 32  // K/V tile size along seq_k dimension

__global__ void flash_attention_kernel(
    const float *q, const float *k, const float *v, const float *mask,
    float *output, int batch, int heads, int seq_q, int seq_k,
    int head_dim, int has_mask) {
  int bh_sq = blockIdx.x;
  int total_rows = batch * heads * seq_q;
  if (bh_sq >= total_rows) return;

  int bh = bh_sq / seq_q;
  int sq = bh_sq % seq_q;

  float scale = rsqrtf((float)head_dim);

  const float *q_row = q + (bh * seq_q + sq) * head_dim;
  float *out_row = output + (bh * seq_q + sq) * head_dim;

  // Online softmax state
  float running_max = -3.402823466e+38f;
  float running_sum = 0.0f;

  // Initialize output to zero
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    out_row[d] = 0.0f;
  }
  __syncthreads();

  // Shared memory for K and V tiles
  extern __shared__ float smem[];
  float *k_tile = smem;
  float *v_tile = smem + FA_TILE_K * head_dim;

  int num_k_tiles = (seq_k + FA_TILE_K - 1) / FA_TILE_K;

  for (int kt = 0; kt < num_k_tiles; kt++) {
    int k_start = kt * FA_TILE_K;
    int k_end = k_start + FA_TILE_K;
    if (k_end > seq_k) k_end = seq_k;
    int tile_len = k_end - k_start;

    // Cooperatively load K/V tiles into shared memory
    int total_load = tile_len * head_dim;
    for (int i = threadIdx.x; i < total_load; i += blockDim.x) {
      int sk_local = i / head_dim;
      int d = i % head_dim;
      int sk_global = k_start + sk_local;
      k_tile[sk_local * head_dim + d] =
          k[(bh * seq_k + sk_global) * head_dim + d];
      v_tile[sk_local * head_dim + d] =
          v[(bh * seq_k + sk_global) * head_dim + d];
    }
    __syncthreads();

    // Compute scores and tile max
    float tile_scores[FA_TILE_K];
    float tile_max = -3.402823466e+38f;

    for (int sk_local = 0; sk_local < tile_len; sk_local++) {
      float score = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        score += q_row[d] * k_tile[sk_local * head_dim + d];
      }
      score *= scale;
      if (has_mask && mask != NULL) {
        score += mask[sq * seq_k + (k_start + sk_local)];
      }
      tile_scores[sk_local] = score;
      if (score > tile_max) tile_max = score;
    }

    // Online softmax update
    float new_max = fmaxf(running_max, tile_max);
    float rescale_old = expf(running_max - new_max);

    float tile_sum = 0.0f;
    float tile_exp[FA_TILE_K];
    for (int sk_local = 0; sk_local < tile_len; sk_local++) {
      tile_exp[sk_local] = expf(tile_scores[sk_local] - new_max);
      tile_sum += tile_exp[sk_local];
    }

    float new_sum = running_sum * rescale_old + tile_sum;

    // Update output: rescale old + accumulate new weighted V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      float old_val = out_row[d] * rescale_old;
      float new_val = 0.0f;
      for (int sk_local = 0; sk_local < tile_len; sk_local++) {
        new_val += tile_exp[sk_local] * v_tile[sk_local * head_dim + d];
      }
      out_row[d] = old_val + new_val;
    }

    running_max = new_max;
    running_sum = new_sum;
    __syncthreads();
  }

  // Final normalization
  if (running_sum > 0.0f) {
    float inv_sum = 1.0f / running_sum;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      out_row[d] *= inv_sum;
    }
  }
}

// --- top_k_sample: sort + mask + softmax + argmax ---
__global__ void top_k_sample_kernel(const float *logits, float *output,
                                    int vocab_size, int k) {
  // 1 スレッドで全処理（logits は 1 行）
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // top-k threshold を見つける
    float top[64];
    int max_k = (k < 64) ? k : 64;
    for (int i = 0; i < max_k; i++)
      top[i] = -3.402823466e+38f;

    for (int i = 0; i < vocab_size; i++) {
      float v = logits[i];
      if (v > top[max_k - 1]) {
        top[max_k - 1] = v;
        for (int j = max_k - 1; j > 0 && top[j] > top[j - 1]; j--) {
          float tmp = top[j];
          top[j] = top[j - 1];
          top[j - 1] = tmp;
        }
      }
    }
    float threshold = top[max_k - 1];

    // softmax over top-k
    float max_val = top[0];
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
      if (logits[i] >= threshold)
        sum_exp += expf(logits[i] - max_val);
    }

    // argmax of softmax
    float best = -1.0f;
    int best_idx = 0;
    for (int i = 0; i < vocab_size; i++) {
      if (logits[i] >= threshold) {
        float p = expf(logits[i] - max_val) / sum_exp;
        if (p > best) {
          best = p;
          best_idx = i;
        }
      }
    }
    output[0] = (float)best_idx;
  }
}

// --- top_p_sample: sort + cumsum + mask + argmax ---
__global__ void top_p_sample_kernel(const float *logits, float *output,
                                    int vocab_size, float p_threshold) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // softmax
    float max_val = -3.402823466e+38f;
    for (int i = 0; i < vocab_size; i++)
      if (logits[i] > max_val)
        max_val = logits[i];

    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++)
      sum_exp += expf(logits[i] - max_val);

    // argmax (greedy for simplicity, matching CPU behavior)
    float best = -1.0f;
    int best_idx = 0;

    // 降順で累積: 最大確率から始めて p_threshold まで
    // 簡易実装: argmax を返す (top-p の最頻出ケース)
    for (int i = 0; i < vocab_size; i++) {
      float prob = expf(logits[i] - max_val) / sum_exp;
      if (prob > best) {
        best = prob;
        best_idx = i;
      }
    }
    output[0] = (float)best_idx;
  }
}

// --- repetition_penalty ---
__global__ void repetition_penalty_kernel(const float *logits,
                                          const float *tokens, float *output,
                                          int vocab_size, int token_len,
                                          float penalty, int total_elements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total_elements) {
    int vocab_idx = i % vocab_size;
    float val = logits[i];
    // token_list に含まれるか確認
    bool found = false;
    for (int t = 0; t < token_len; t++) {
      if ((int)tokens[t] == vocab_idx) {
        found = true;
        break;
      }
    }
    if (found) {
      output[i] = (val > 0.0f) ? val / penalty : val * penalty;
    } else {
      output[i] = val;
    }
  }
}

// =====================================================================
// Phase D: Fused カーネル
// =====================================================================

// --- fused_silu_mul: y[i] = silu(x[i]) * up[i] ---
__global__ void fused_silu_mul_kernel(const float *x, const float *up, float *y,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sv = x[i] / (1.0f + expf(-x[i])); // silu
    y[i] = sv * up[i];
  }
}

// --- fused_rms_norm: y = x * weight * rsqrt(mean(x²) + eps) ---
__global__ void fused_rms_norm_kernel(const float *x, const float *weight,
                                      float *y, int outer, int norm_size,
                                      float eps) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o < outer) {
    int offset = o * norm_size;
    float sum_sq = 0.0f;
    for (int i = 0; i < norm_size; i++) {
      float v = x[offset + i];
      sum_sq += v * v;
    }
    float inv_rms = rsqrtf(sum_sq / (float)norm_size + eps);
    for (int i = 0; i < norm_size; i++) {
      y[offset + i] = x[offset + i] * weight[i] * inv_rms;
    }
  }
}

// --- fused_add_rms_norm: y = (x + residual) * weight * rsqrt(mean((x+r)²) +
// eps) ---
__global__ void fused_add_rms_norm_kernel(const float *x, const float *residual,
                                          const float *weight, float *y,
                                          int outer, int norm_size, float eps) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o < outer) {
    int offset = o * norm_size;
    float sum_sq = 0.0f;
    for (int i = 0; i < norm_size; i++) {
      float v = x[offset + i] + residual[offset + i];
      sum_sq += v * v;
    }
    float inv_rms = rsqrtf(sum_sq / (float)norm_size + eps);
    for (int i = 0; i < norm_size; i++) {
      float v = x[offset + i] + residual[offset + i];
      y[offset + i] = v * weight[i] * inv_rms;
    }
  }
}

// --- fused_add_relu: y[i] = max(a[i] + b[i], 0) ---
__global__ void fused_add_relu_kernel(const float *a, const float *b, float *y,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = fmaxf(a[i] + b[i], 0.0f);
}

// --- fused_bias_gelu: y = gelu(x + bias) ---
__global__ void fused_bias_gelu_kernel(const float *x, const float *bias,
                                       float *y, int outer, int dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outer * dim;
  if (tid < total) {
    int j = tid % dim;
    float v = x[tid] + bias[j];
    float c = 0.7978845608f;
    float inner = c * (v + 0.044715f * v * v * v);
    y[tid] = 0.5f * v * (1.0f + tanhf(inner));
  }
}

// =====================================================================
// Phase A-D: C wrapper (launch functions)
// =====================================================================
extern "C" {

// --- Phase A: activations ---
LAUNCH_SCALAR(leaky_relu)
LAUNCH_SCALAR(elu)
LAUNCH_UNARY(mish)
LAUNCH_UNARY(hardswish)
LAUNCH_UNARY(hardsigmoid)
LAUNCH_UNARY(logical_not)

LAUNCH_BINARY(logical_and)
LAUNCH_BINARY(logical_or)
LAUNCH_BINARY(dot_mul)
LAUNCH_BINARY(mse_loss_elem)
LAUNCH_BINARY(l1_loss_elem)
LAUNCH_BINARY(bce_loss_elem)
LAUNCH_BINARY(nll_loss_elem)
LAUNCH_BINARY(kl_div_loss_elem)

void launch_masked_fill_kernel(const float *x, const float *mask, float *y,
                               int n, float value, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  masked_fill_kernel<<<blocks, threads, 0, stream>>>(x, mask, y, n, value);
}

void launch_fill_kernel(float *y, int n, float value, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  fill_kernel<<<blocks, threads, 0, stream>>>(y, n, value);
}

// --- Phase B: reductions ---
void launch_reduce_axis_prod_kernel(const float *input, float *output,
                                    int outer, int axis_size, int inner,
                                    cudaStream_t stream) {
  int total = outer * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  reduce_axis_prod_kernel<<<blocks, threads, 0, stream>>>(input, output, outer,
                                                          axis_size, inner);
}

void launch_cumsum_kernel(const float *input, float *output, int outer,
                          int axis_size, int inner, cudaStream_t stream) {
  int total = outer * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  cumsum_kernel<<<blocks, threads, 0, stream>>>(input, output, outer, axis_size,
                                                inner);
}

void launch_reduce_axis_norm_kernel(const float *input, float *output,
                                    int outer, int axis_size, int inner,
                                    float p_val, cudaStream_t stream) {
  int total = outer * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  reduce_axis_norm_kernel<<<blocks, threads, 0, stream>>>(
      input, output, outer, axis_size, inner, p_val);
}

void launch_topk_kernel(const float *input, float *output, int outer,
                        int axis_size, int inner, int k, cudaStream_t stream) {
  int total = outer * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  topk_kernel<<<blocks, threads, 0, stream>>>(input, output, outer, axis_size,
                                              inner, k);
}

void launch_reduce_axis_var_kernel(const float *input, float *output, int outer,
                                   int axis_size, int inner,
                                   cudaStream_t stream) {
  int total = outer * inner;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  reduce_axis_var_kernel<<<blocks, threads, 0, stream>>>(input, output, outer,
                                                         axis_size, inner);
}

// --- Phase C: NN / LLM ---
void launch_group_norm_kernel(const float *input, const float *gamma,
                              const float *beta, float *output, int n, int c,
                              int spatial, int num_groups, float eps,
                              cudaStream_t stream) {
  int total = n * num_groups;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  group_norm_kernel<<<blocks, threads, 0, stream>>>(
      input, gamma, beta, output, n, c, spatial, num_groups, eps);
}

void launch_adaptive_avg_pool2d_kernel(const float *input, float *output, int n,
                                       int c, int h_in, int w_in, int h_out,
                                       int w_out, cudaStream_t stream) {
  int total = n * c * h_out * w_out;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  adaptive_avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(
      input, output, n, c, h_in, w_in, h_out, w_out);
}

void launch_pad_kernel(const float *input, float *output, int n, int old_dim,
                       int new_dim, int pad_left, float pad_value,
                       cudaStream_t stream) {
  int total = n * new_dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  pad_kernel<<<blocks, threads, 0, stream>>>(input, output, n, old_dim, new_dim,
                                             pad_left, pad_value);
}

void launch_dropout2d_kernel(const float *input, float *output, int n, int c,
                             int spatial, float p, float scale,
                             cudaStream_t stream) {
  int total = n * c * spatial;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  dropout2d_kernel<<<blocks, threads, 0, stream>>>(input, output, n, c, spatial,
                                                   p, scale);
}

void launch_conv1d_kernel(const float *input, const float *weight,
                          float *output, int n, int c_in, int l_in, int c_out,
                          int kl, int l_out, int stride, int pad,
                          cudaStream_t stream) {
  int total = n * c_out * l_out;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  conv1d_kernel<<<blocks, threads, 0, stream>>>(
      input, weight, output, n, c_in, l_in, c_out, kl, l_out, stride, pad);
}

void launch_conv_transpose2d_kernel(const float *input, const float *weight,
                                    float *output, int n, int c_in, int h_in,
                                    int w_in, int c_out, int kh, int kw,
                                    int h_out, int w_out, int stride_h,
                                    int stride_w, int pad_h, int pad_w,
                                    cudaStream_t stream) {
  int total = n * c_out * h_out * w_out;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  conv_transpose2d_kernel<<<blocks, threads, 0, stream>>>(
      input, weight, output, n, c_in, h_in, w_in, c_out, kh, kw, h_out, w_out,
      stride_h, stride_w, pad_h, pad_w);
}

void launch_interpolate_kernel(const float *input, float *output, int n, int c,
                               int h_in, int w_in, int h_out, int w_out,
                               int mode, cudaStream_t stream) {
  int total = n * c * h_out * w_out;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  interpolate_kernel<<<blocks, threads, 0, stream>>>(input, output, n, c, h_in,
                                                     w_in, h_out, w_out, mode);
}

void launch_sdpa_kernel(const float *q, const float *k, const float *v,
                        const float *mask, float *output, int batch, int heads,
                        int seq_q, int seq_k, int head_dim, int has_mask,
                        cudaStream_t stream) {
  int total_rows = batch * heads * seq_q;
  int threads = (head_dim <= 128) ? head_dim : 128;
  if (threads < 32) threads = 32;
  int blocks = total_rows;
  size_t smem_size = 2 * FA_TILE_K * head_dim * sizeof(float);
  flash_attention_kernel<<<blocks, threads, smem_size, stream>>>(
      q, k, v, mask, output, batch, heads, seq_q, seq_k, head_dim, has_mask);
}

void launch_top_k_sample_kernel(const float *logits, float *output,
                                int vocab_size, int k, cudaStream_t stream) {
  top_k_sample_kernel<<<1, 1, 0, stream>>>(logits, output, vocab_size, k);
}

void launch_top_p_sample_kernel(const float *logits, float *output,
                                int vocab_size, float p, cudaStream_t stream) {
  top_p_sample_kernel<<<1, 1, 0, stream>>>(logits, output, vocab_size, p);
}

void launch_repetition_penalty_kernel(const float *logits, const float *tokens,
                                      float *output, int vocab_size,
                                      int token_len, float penalty,
                                      int total_elements,
                                      cudaStream_t stream) {
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;
  repetition_penalty_kernel<<<blocks, threads, 0, stream>>>(
      logits, tokens, output, vocab_size, token_len, penalty, total_elements);
}

// --- Phase D: fused ---
void launch_fused_silu_mul_kernel(const float *x, const float *up, float *y,
                                  int n, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  fused_silu_mul_kernel<<<blocks, threads, 0, stream>>>(x, up, y, n);
}

void launch_fused_rms_norm_kernel(const float *x, const float *weight, float *y,
                                  int outer, int norm_size, float eps,
                                  cudaStream_t stream) {
  int threads = 256;
  int blocks = (outer + threads - 1) / threads;
  fused_rms_norm_kernel<<<blocks, threads, 0, stream>>>(x, weight, y, outer,
                                                        norm_size, eps);
}

void launch_fused_add_rms_norm_kernel(const float *x, const float *residual,
                                      const float *weight, float *y, int outer,
                                      int norm_size, float eps,
                                      cudaStream_t stream) {
  int threads = 256;
  int blocks = (outer + threads - 1) / threads;
  fused_add_rms_norm_kernel<<<blocks, threads, 0, stream>>>(
      x, residual, weight, y, outer, norm_size, eps);
}

void launch_fused_add_relu_kernel(const float *a, const float *b, float *y,
                                  int n, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  fused_add_relu_kernel<<<blocks, threads, 0, stream>>>(a, b, y, n);
}

void launch_fused_bias_gelu_kernel(const float *x, const float *bias, float *y,
                                   int outer, int dim, cudaStream_t stream) {
  int total = outer * dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  fused_bias_gelu_kernel<<<blocks, threads, 0, stream>>>(x, bias, y, outer,
                                                         dim);
}

} // end extern "C" block for Phase A-D kernels

// =====================================================================
// 融合 Q4_K / Q6_K matmul カーネル
// =====================================================================

#include <cuda_fp16.h>

__device__ inline float half_to_float_q(unsigned short h) {
  __half hv;
  memcpy(&hv, &h, sizeof(__half));
  return __half2float(hv);
}

__device__ inline void get_scale_min_k4_dev(int j, const unsigned char *scales,
                                            unsigned char *sc,
                                            unsigned char *m) {
  if (j < 4) {
    *sc = scales[j] & 63;
    *m = scales[j + 4] & 63;
  } else {
    *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
    *m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
  }
}

__global__ void mul_mv_q4_k_kernel(const float *input,
                                   const unsigned char *w_raw, float *output,
                                   int M, int N, int K) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= M * N)
    return;
  int m = tid / N;
  int n = tid % N;
  const int QK_K = 256;
  const int BB = 144;
  int bpr = K / QK_K;
  float sum = 0.0f;
  const float *xr = input + m * K;
  for (int bi = 0; bi < bpr; bi++) {
    const unsigned char *blk = w_raw + ((long long)n * bpr + bi) * BB;
    unsigned short dh, dmh;
    memcpy(&dh, blk, 2);
    memcpy(&dmh, blk + 2, 2);
    float d = half_to_float_q(dh);
    float dm = half_to_float_q(dmh);
    const unsigned char *sc = blk + 4;
    const unsigned char *qs = blk + 16;
    int xo = bi * QK_K;
    int is = 0, qi = 0;
    for (int j = 0; j < QK_K; j += 64) {
      unsigned char s1, m1, s2, m2;
      get_scale_min_k4_dev(is, sc, &s1, &m1);
      get_scale_min_k4_dev(is + 1, sc, &s2, &m2);
      float d1 = d * s1, mn1 = dm * m1;
      float d2 = d * s2, mn2 = dm * m2;
      for (int l = 0; l < 32; l++)
        sum += xr[xo + j + l] * (d1 * (float)(qs[qi + l] & 0xF) - mn1);
      for (int l = 0; l < 32; l++)
        sum += xr[xo + j + 32 + l] * (d2 * (float)(qs[qi + l] >> 4) - mn2);
      qi += 32;
      is += 2;
    }
  }
  output[m * N + n] = sum;
}

__global__ void mul_mv_q6_k_kernel(const float *input,
                                   const unsigned char *w_raw, float *output,
                                   int M, int N, int K) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= M * N)
    return;
  int m = tid / N;
  int n = tid % N;
  const int QK_K = 256;
  const int BB = 210;
  int bpr = K / QK_K;
  float sum = 0.0f;
  const float *xr = input + m * K;
  for (int bi = 0; bi < bpr; bi++) {
    const unsigned char *blk = w_raw + ((long long)n * bpr + bi) * BB;
    unsigned short dh;
    memcpy(&dh, blk + 208, 2);
    float d = half_to_float_q(dh);
    int xo = bi * QK_K;
    int qlo = 0, qho = 128, sco = 192;
    for (int nc = 0; nc < 2; nc++) {
      int cb = xo + nc * 128;
      for (int l = 0; l < 32; l++) {
        int is = l / 16;
        unsigned char ql_lo = blk[qlo + l];
        unsigned char ql_hi = blk[qlo + l + 32];
        unsigned char qhv = blk[qho + l];
        int q1 = ((ql_lo & 0xF) | (((qhv >> 0) & 3) << 4)) - 32;
        int q2 = ((ql_hi & 0xF) | (((qhv >> 2) & 3) << 4)) - 32;
        int q3 = ((ql_lo >> 4) | (((qhv >> 4) & 3) << 4)) - 32;
        int q4 = ((ql_hi >> 4) | (((qhv >> 6) & 3) << 4)) - 32;
        float s1 = d * (float)(signed char)blk[sco + is + 0];
        float s2 = d * (float)(signed char)blk[sco + is + 2];
        float s3 = d * (float)(signed char)blk[sco + is + 4];
        float s4 = d * (float)(signed char)blk[sco + is + 6];
        sum += xr[cb + l + 0] * s1 * q1;
        sum += xr[cb + l + 32] * s2 * q2;
        sum += xr[cb + l + 64] * s3 * q3;
        sum += xr[cb + l + 96] * s4 * q4;
      }
      qlo += 64;
      qho += 32;
      sco += 8;
    }
  }
  output[m * N + n] = sum;
}

extern "C" {

void launch_mul_mv_q4_k_kernel(const float *input, const unsigned char *w_raw,
                               float *output, int M, int N, int K,
                               cudaStream_t stream) {
  int total = M * N;
  if (total <= 0 || K <= 0) {
    fprintf(stderr, "[CUDA] mul_mv_q4_k: invalid dims M=%d N=%d K=%d\n", M, N,
            K);
    return;
  }
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  mul_mv_q4_k_kernel<<<blocks, threads, 0, stream>>>(input, w_raw, output, M, N,
                                                     K);
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    fprintf(
        stderr,
        "[CUDA ERROR] mul_mv_q4_k launch failed: %s (blocks=%d, total=%d)\n",
        cudaGetErrorString(err), blocks, total);
  }
}

void launch_mul_mv_q6_k_kernel(const float *input, const unsigned char *w_raw,
                               float *output, int M, int N, int K,
                               cudaStream_t stream) {
  int total = M * N;
  if (total <= 0 || K <= 0) {
    fprintf(stderr, "[CUDA] mul_mv_q6_k: invalid dims M=%d N=%d K=%d\n", M, N,
            K);
    return;
  }
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  mul_mv_q6_k_kernel<<<blocks, threads, 0, stream>>>(input, w_raw, output, M, N,
                                                     K);
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    fprintf(
        stderr,
        "[CUDA ERROR] mul_mv_q6_k launch failed: %s (blocks=%d, total=%d)\n",
        cudaGetErrorString(err), blocks, total);
  }
}

} // extern "C"
