#ifndef PIPELINE_H
#define PIPELINE_H

#include <cuda_runtime.h>
#include "cuda_utils.h"

// Pipeline class structure
typedef struct {
    // Configuration
    PipelineConfig config;
    DeviceInfo device_info;
    
    // Memory management
    float* h_input;           // Pinned host memory
    float* h_output;          // Pinned host memory
    float* d_input;           // Device memory
    float* d_output;          // Device memory
    float* d_temp;            // Device temporary memory
    float* d_weights;         // Device convolution weights
    
    // Streams and events
    cudaStream_t streams[MAX_STREAMS];
    cudaEvent_t h2d_start[MAX_STREAMS];
    cudaEvent_t h2d_end[MAX_STREAMS];
    cudaEvent_t normalize_end[MAX_STREAMS];
    cudaEvent_t convolve_end[MAX_STREAMS];
    cudaEvent_t d2h_end[MAX_STREAMS];
    
    // Performance metrics
    PerformanceMetrics metrics;
    
    // Internal state
    int initialized;
} Pipeline;

// Pipeline management functions
int pipeline_init(Pipeline* pipeline, PipelineConfig* config);
void pipeline_cleanup(Pipeline* pipeline);
void pipeline_print_info(Pipeline* pipeline);

// Main pipeline execution
int pipeline_run(Pipeline* pipeline);
int pipeline_run_single_stream(Pipeline* pipeline);
int pipeline_run_multi_stream(Pipeline* pipeline);

// Memory management
int pipeline_allocate_memory(Pipeline* pipeline);
void pipeline_free_memory(Pipeline* pipeline);
int pipeline_setup_data(Pipeline* pipeline);

// Stream and event management
int pipeline_create_streams(Pipeline* pipeline);
int pipeline_create_events(Pipeline* pipeline);
void pipeline_destroy_streams(Pipeline* pipeline);
void pipeline_destroy_events(Pipeline* pipeline);

// Performance measurement
int pipeline_measure_performance(Pipeline* pipeline, float* total_time, 
                                float* h2d_time, float* norm_time, 
                                float* conv_time, float* d2h_time);
float pipeline_get_event_time(cudaEvent_t start, cudaEvent_t end);
void pipeline_calculate_metrics(Pipeline* pipeline, float total_time, 
                               float h2d_time, float norm_time, 
                               float conv_time, float d2h_time, float baseline_time);

// Verification
int pipeline_verify_correctness(Pipeline* pipeline);

// Utility functions
size_t pipeline_calculate_tile_size(Pipeline* pipeline);
void pipeline_print_timing_results(Pipeline* pipeline);

#endif // PIPELINE_H
