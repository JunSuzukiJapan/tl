#include <cuda_runtime.h>

// =====================================================================
// one_hot カーネル
// indices[batch] (i64) → output[batch * classes] (f32)
// output は事前に 0 初期化されている前提
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
// grad[seq_len * dim] (f32), indices[seq_len] (i64)
//   → output[vocab * dim] (f32)
// output は事前に 0 初期化されている前提
// atomicAdd で並行書き込みの競合を解決
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
        int i = tid / dim;  // seq index
        int j = tid % dim;  // dim index
        int target_row = (int)indices[i];
        if (target_row >= 0 && target_row < vocab) {
            atomicAdd(&output[target_row * dim + j], grad[i * dim + j]);
        }
    }
}

extern "C" {

void launch_one_hot_kernel(
    const long long* indices,
    float* output,
    int batch,
    int classes,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    one_hot_kernel<<<blocks, threads, 0, stream>>>(indices, output, batch, classes);
}

void launch_scatter_add_kernel(
    const float* grad,
    const long long* indices,
    float* output,
    int seq_len,
    int dim,
    int vocab,
    cudaStream_t stream
) {
    int total = seq_len * dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    scatter_add_kernel<<<blocks, threads, 0, stream>>>(grad, indices, output, seq_len, dim, vocab);
}

}
