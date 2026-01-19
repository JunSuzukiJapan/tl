#include <cuda_runtime.h>
#include <math.h>

// Simple Sigmoid Kernel: y = 1 / (1 + exp(-x))
__global__ void sigmoid_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

extern "C" {

// C-wrapper to call the kernel
void launch_sigmoid_kernel(const float* x, float* y, int n, cudaStream_t stream) {
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    sigmoid_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(x, y, n);
}

}
