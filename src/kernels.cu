#include "kernels.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <algorithm>

/**
 * Normalization kernel - Min-max scaling
 * Maps input values to range [0, 1] using: (x - min) / (max - min)
 */
__global__ void normalize_kernel(float* input, float* output, size_t n, float min_val, float max_val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float range = max_val - min_val;
        if (range > 0.0f) {
            output[idx] = (input[idx] - min_val) / range;
        } else {
            output[idx] = 0.0f;  // Handle edge case where min == max
        }
    }
}

/**
 * 1D Convolution kernel with 5-tap filter
 * Implements edge detection or smoothing filter
 */
__global__ void convolve_5tap_kernel(float* input, float* output, float* weights, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float result = 0.0f;
        
        // Apply 5-tap convolution
        for (int i = 0; i < 5; i++) {
            int input_idx = (int)idx - 2 + i;  // Center the filter
            
            // Handle boundary conditions with clamping
            if (input_idx < 0) {
                input_idx = 0;
            } else if (input_idx >= (int)n) {
                input_idx = n - 1;
            }
            
            result += input[input_idx] * weights[i];
        }
        
        output[idx] = result;
    }
}

/**
 * Simple addition kernel for testing
 */
__global__ void add_kernel(float* a, float* b, float* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

// Host wrapper functions for kernel launches

void launch_normalize_kernel(float* d_input, float* d_output, size_t n, 
                           float min_val, float max_val, 
                           int block_size, cudaStream_t stream) {
    dim3 grid = calculate_grid_size(n, block_size);
    dim3 block(block_size);
    
    normalize_kernel<<<grid, block, 0, stream>>>(d_input, d_output, n, min_val, max_val);
    CHECK(cudaGetLastError());
}

void launch_convolve_kernel(float* d_input, float* d_output, float* d_weights, 
                          size_t n, int block_size, cudaStream_t stream) {
    dim3 grid = calculate_grid_size(n, block_size);
    dim3 block(block_size);
    
    convolve_5tap_kernel<<<grid, block, 0, stream>>>(d_input, d_output, d_weights, n);
    CHECK(cudaGetLastError());
}

// CPU reference implementations for verification

void normalize_cpu_reference(float* input, float* output, size_t n, float min_val, float max_val) {
    float range = max_val - min_val;
    
    for (size_t i = 0; i < n; i++) {
        if (range > 0.0f) {
            output[i] = (input[i] - min_val) / range;
        } else {
            output[i] = 0.0f;
        }
    }
}

void convolve_cpu_reference(float* input, float* output, float* weights, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float result = 0.0f;
        
        for (int j = 0; j < 5; j++) {
            int input_idx = (int)i - 2 + j;  // Center the filter
            
            // Handle boundary conditions with clamping
            if (input_idx < 0) {
                input_idx = 0;
            } else if (input_idx >= (int)n) {
                input_idx = n - 1;
            }
            
            result += input[input_idx] * weights[j];
        }
        
        output[i] = result;
    }
}

// Kernel configuration helpers

dim3 calculate_grid_size(size_t n, int block_size) {
    int grid_size = (n + block_size - 1) / block_size;
    return dim3(grid_size);
}

int calculate_optimal_block_size(size_t n, int max_block_size) {
    // Simple heuristic: use the largest block size that fits the data
    // In practice, you might want to benchmark different sizes
    return std::min((int)n, max_block_size);
}
