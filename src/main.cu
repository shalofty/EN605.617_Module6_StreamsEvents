#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "utils.h"
#include "pipeline.h"
#include "kernels.h"

/**
 * CUDA Streams and Events Pipeline
 * 
 * This program demonstrates advanced CUDA programming concepts through a
 * high-performance data processing pipeline with multiple streams and events.
 * 
 * Features:
 * - Multi-stream parallel processing with event synchronization
 * - Pinned host memory for true H2D/D2H overlap
 * - Comprehensive timing and performance analysis
 * - Correctness verification with CPU reference implementations
 */

void print_banner() {
    printf("========================================\n");
    printf("  CUDA Streams and Events Pipeline\n");
    printf("  Advanced GPU Programming Demo\n");
    printf("========================================\n\n");
}

int main(int argc, char* argv[]) {
    print_banner();
    
    // Initialize configuration with defaults
    PipelineConfig config = {
        .array_size = DEFAULT_ARRAY_SIZE,
        .block_size = DEFAULT_BLOCK_SIZE,
        .num_streams = 4,
        .iterations = DEFAULT_ITERATIONS,
        .verify = 0,
        .csv_output = {0},
        .operation = "both"
    };
    
    // Parse command line arguments
    if (parse_arguments(argc, argv, &config) != 0) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Initialize CUDA device
    DeviceInfo device_info;
    CHECK(cudaGetDevice(&device_info.device_id));
    
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device_info.device_id));
    
    // Copy properties to our structure
    strncpy(device_info.device_name, prop.name, sizeof(device_info.device_name) - 1);
    device_info.device_name[sizeof(device_info.device_name) - 1] = '\0';
    device_info.major = prop.major;
    device_info.minor = prop.minor;
    device_info.multiProcessorCount = prop.multiProcessorCount;
    device_info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    device_info.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    device_info.deviceOverlap = prop.deviceOverlap;
    device_info.asyncEngineCount = 2;  // Most modern GPUs have 2 async engines
    device_info.concurrentKernels = 1;  // Most modern GPUs support concurrent kernels
    device_info.totalGlobalMem = prop.totalGlobalMem;
    device_info.sharedMemPerBlock = prop.sharedMemPerBlock;
    
    print_device_info(&device_info);
    
    // Validate configuration
    validate_config(&config, &device_info);
    print_config(&config);
    
    // Initialize pipeline
    Pipeline pipeline;
    if (pipeline_init(&pipeline, &config) != 0) {
        fprintf(stderr, "Failed to initialize pipeline\n");
        return 1;
    }
    
    pipeline_print_info(&pipeline);
    
    // Run pipeline with performance measurement
    printf("\n=== Running Pipeline ===\n");
    if (pipeline_run(&pipeline) != 0) {
        fprintf(stderr, "Pipeline execution failed\n");
        pipeline_cleanup(&pipeline);
        return 1;
    }
    
    // Print results
    pipeline_print_timing_results(&pipeline);
    
    // Write CSV output if requested
    if (strlen(config.csv_output) > 0) {
        FILE* csv_file = fopen(config.csv_output, "w");
        if (csv_file) {
            write_csv_header(csv_file);
            write_csv_row(csv_file, &config, &pipeline.metrics);
            fclose(csv_file);
            printf("\nResults written to: %s\n", config.csv_output);
        }
    }
    
    // Verify correctness if requested
    if (config.verify) {
        printf("\n=== Verifying Correctness ===\n");
        if (pipeline_verify_correctness(&pipeline) == 0) {
            printf("✓ Verification passed - GPU results match CPU reference\n");
        } else {
            printf("✗ Verification failed - GPU results differ from CPU reference\n");
        }
    }
    
    // Cleanup
    pipeline_cleanup(&pipeline);
    
    printf("\n=== Pipeline Complete ===\n");
    return 0;
}
