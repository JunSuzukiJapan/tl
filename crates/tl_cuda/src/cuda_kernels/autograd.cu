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

}
