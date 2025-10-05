#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuda_utils.h"
#include "utils.h"

void print_device_info(DeviceInfo* info) {
    printf("=== Device Information ===\n");
    printf("Device ID: %d\n", info->device_id);
    printf("Device Name: %s\n", info->device_name);
    printf("Compute Capability: %d.%d\n", info->major, info->minor);
    printf("Multiprocessors: %d\n", info->multiProcessorCount);
    printf("Max Threads per Block: %d\n", info->maxThreadsPerBlock);
    printf("Max Threads per Multiprocessor: %d\n", info->maxThreadsPerMultiProcessor);
    printf("Device Overlap: %s\n", info->deviceOverlap ? "Yes" : "No");
    printf("Async Engine Count: %d\n", info->asyncEngineCount);
    printf("Concurrent Kernels: %s\n", info->concurrentKernels ? "Yes" : "No");
    printf("Total Global Memory: %.2f GB\n", info->totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory per Block: %zu KB\n", info->sharedMemPerBlock / 1024);
    printf("\n");
}

void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n\n", program_name);
    printf("Options:\n");
    printf("  --n SIZE        Array size (default: %d)\n", DEFAULT_ARRAY_SIZE);
    printf("  --block SIZE    Thread block size (%d-%d, default: %d)\n", 
           MIN_BLOCK_SIZE, MAX_BLOCK_SIZE, DEFAULT_BLOCK_SIZE);
    printf("  --streams NUM   Number of streams (1|2|4, default: 4)\n");
    printf("  --iters NUM     Number of iterations (≥2, default: %d)\n", DEFAULT_ITERATIONS);
    printf("  --op OP         Operation: normalize|conv|both (default: both)\n");
    printf("  --verify        Enable correctness verification\n");
    printf("  --csv FILE      Output results to CSV file\n");
    printf("  --help          Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s --n 1000000 --block 256 --streams 4 --iters 10\n", program_name);
    printf("  %s --n 10000000 --block 512 --streams 2 --iters 5 --verify\n", program_name);
    printf("  %s --n 500000 --block 128 --streams 1 --iters 20 --csv results.csv\n", program_name);
}

int parse_arguments(int argc, char* argv[], PipelineConfig* config) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            return -1;  // Signal to show help
        }
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            config->array_size = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            config->block_size = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--streams") == 0 && i + 1 < argc) {
            config->num_streams = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            config->iterations = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--op") == 0 && i + 1 < argc) {
            strncpy(config->operation, argv[++i], sizeof(config->operation) - 1);
        }
        else if (strcmp(argv[i], "--verify") == 0) {
            config->verify = 1;
        }
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            strncpy(config->csv_output, argv[++i], sizeof(config->csv_output) - 1);
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return -1;
        }
    }
    return 0;
}

void validate_config(PipelineConfig* config, DeviceInfo* device_info) {
    // Validate block size
    if (config->block_size < MIN_BLOCK_SIZE || config->block_size > MAX_BLOCK_SIZE) {
        fprintf(stderr, "Error: Block size must be between %d and %d\n", 
                MIN_BLOCK_SIZE, MAX_BLOCK_SIZE);
        exit(1);
    }
    
    if (config->block_size > device_info->maxThreadsPerBlock) {
        fprintf(stderr, "Error: Block size (%d) exceeds device maximum (%d)\n",
                config->block_size, device_info->maxThreadsPerBlock);
        exit(1);
    }
    
    // Validate number of streams
    if (config->num_streams != 1 && config->num_streams != 2 && config->num_streams != 4) {
        fprintf(stderr, "Error: Number of streams must be 1, 2, or 4\n");
        exit(1);
    }
    
    // Validate iterations
    if (config->iterations < 2) {
        fprintf(stderr, "Error: Number of iterations must be at least 2 (rubric requirement)\n");
        exit(1);
    }
    
    // Validate array size
    if (config->array_size <= 0) {
        fprintf(stderr, "Error: Array size must be positive\n");
        exit(1);
    }
    
    // Validate operation
    if (strcmp(config->operation, "normalize") != 0 && 
        strcmp(config->operation, "conv") != 0 && 
        strcmp(config->operation, "both") != 0) {
        fprintf(stderr, "Error: Operation must be 'normalize', 'conv', or 'both'\n");
        exit(1);
    }
}

void print_config(PipelineConfig* config) {
    printf("=== Pipeline Configuration ===\n");
    printf("Array Size: %zu elements\n", config->array_size);
    printf("Block Size: %d threads\n", config->block_size);
    printf("Streams: %d\n", config->num_streams);
    printf("Iterations: %d\n", config->iterations);
    printf("Operation: %s\n", config->operation);
    printf("Verification: %s\n", config->verify ? "Enabled" : "Disabled");
    if (strlen(config->csv_output) > 0) {
        printf("CSV Output: %s\n", config->csv_output);
    }
    printf("\n");
}

void print_performance_table(PerformanceMetrics* metrics, int num_configs) {
    printf("=== Performance Results ===\n");
    printf("%-8s %-12s %-12s %-12s %-10s\n", 
           "Streams", "Total (ms)", "Throughput", "Bandwidth", "Speedup");
    printf("%-8s %-12s %-12s %-12s %-10s\n", 
           "", "", "(MEl/s)", "(GB/s)", "");
    printf("-----------------------------------------------------------\n");
    
    for (int i = 0; i < num_configs; i++) {
        printf("%-8d %-12.2f %-12.2f %-12.2f %-10.2fx\n",
               i + 1,
               metrics[i].total_time,
               metrics[i].throughput_mel_s,
               metrics[i].bandwidth_gbps,
               metrics[i].speedup);
    }
    printf("\n");
}

void write_csv_header(FILE* file) {
    fprintf(file, "streams,block_size,array_size,iterations,total_ms,h2d_ms,norm_ms,conv_ms,d2h_ms,throughput_mel_s,bandwidth_gbps,speedup\n");
}

void write_csv_row(FILE* file, PipelineConfig* config, PerformanceMetrics* metrics) {
    fprintf(file, "%d,%d,%zu,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
            config->num_streams,
            config->block_size,
            config->array_size,
            config->iterations,
            metrics->total_time,
            metrics->h2d_time,
            metrics->normalize_time,
            metrics->convolve_time,
            metrics->d2h_time,
            metrics->throughput_mel_s,
            metrics->bandwidth_gbps,
            metrics->speedup);
}

// Simple bubble sort for median calculation
static void bubble_sort(float* arr, int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                float temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

float median(float* values, int count) {
    // Create a copy for sorting
    float* sorted = (float*)malloc(count * sizeof(float));
    memcpy(sorted, values, count * sizeof(float));
    
    // Sort the array
    bubble_sort(sorted, count);
    
    float result;
    if (count % 2 == 0) {
        result = (sorted[count/2 - 1] + sorted[count/2]) / 2.0f;
    } else {
        result = sorted[count/2];
    }
    
    free(sorted);
    return result;
}

void generate_test_data(float* data, size_t size) {
    // Generate test data with known patterns for verification
    for (size_t i = 0; i < size; i++) {
        data[i] = (float)(i % 1000) / 1000.0f;  // Values between 0 and 1
    }
}

int compare_arrays(const float* a, const float* b, size_t size, float tolerance) {
    for (size_t i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > tolerance) {
            printf("Mismatch at index %zu: %f vs %f (diff: %f)\n", 
                   i, a[i], b[i], fabs(a[i] - b[i]));
            return 0;  // Arrays differ
        }
    }
    return 1;  // Arrays match
}
