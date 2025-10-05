#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

// Kernel function declarations
__global__ void normalize_kernel(float* input, float* output, size_t n, float min_val, float max_val);
__global__ void convolve_5tap_kernel(float* input, float* output, float* weights, size_t n);
__global__ void add_kernel(float* a, float* b, float* result, size_t n);

// Host wrapper functions for kernel launches
void launch_normalize_kernel(float* d_input, float* d_output, 
                           float min_val, float max_val, 
                           size_t base, int n_tile, int N_total,
                           int block_size, cudaStream_t stream);

void launch_convolve_kernel(float* d_input, float* d_output, float* d_weights, 
                          size_t base, int n_tile, int N_total,
                          int block_size, cudaStream_t stream);

// CPU reference implementations for verification
void normalize_cpu_reference(float* input, float* output, size_t n, float min_val, float max_val);
void convolve_cpu_reference(float* input, float* output, float* weights, size_t n);

// Kernel configuration helpers
dim3 calculate_grid_size(size_t n, int block_size);
int calculate_optimal_block_size(size_t n, int max_block_size);

#endif // KERNELS_H
