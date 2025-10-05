#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA error checking macro
#define CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Constants for the pipeline
#define MAX_STREAMS 4
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 64
#define DEFAULT_BLOCK_SIZE 256
#define DEFAULT_ARRAY_SIZE 1000000
#define DEFAULT_ITERATIONS 10
#define TILES_PER_STREAM 2

// Device capability structure
typedef struct {
    int device_id;
    char device_name[256];
    int major;
    int minor;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor;
    int deviceOverlap;
    int asyncEngineCount;
    int concurrentKernels;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
} DeviceInfo;

// Pipeline configuration structure
typedef struct {
    size_t array_size;
    int block_size;
    int num_streams;
    int iterations;
    int verify;
    char csv_output[256];
    char operation[16];  // "normalize", "conv", or "both"
} PipelineConfig;

// Performance timing structure
typedef struct {
    float total_time;
    float h2d_time;
    float normalize_time;
    float convolve_time;
    float d2h_time;
    float throughput_mel_s;  // millions of elements per second
    float bandwidth_gbps;    // bandwidth in GB/s
    float speedup;
} PerformanceMetrics;

// Function declarations
void print_device_info(DeviceInfo* info);
void print_usage(const char* program_name);
int parse_arguments(int argc, char* argv[], PipelineConfig* config);
void validate_config(PipelineConfig* config, DeviceInfo* device_info);
void print_config(PipelineConfig* config);
void print_performance_table(PerformanceMetrics* metrics, int num_configs);
void write_csv_header(FILE* file);
void write_csv_row(FILE* file, PipelineConfig* config, PerformanceMetrics* metrics);

// Utility functions
float median(float* values, int count);
void generate_test_data(float* data, size_t size);
int compare_arrays(const float* a, const float* b, size_t size, float tolerance);

#endif // CUDA_UTILS_H
