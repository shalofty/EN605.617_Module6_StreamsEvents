# CUDA Streams and Events Pipeline Program

## 📋 Program Overview

**Project Name**: Double/Triple-Buffered Processing Pipeline with CUDA Streams and Events  
**Objective**: Demonstrate advanced CUDA programming concepts through a high-performance data processing pipeline  
**Target Grade**: 100/100 + 5% bonus for final project integration

## 🎯 Program Architecture

### Core Concept
A multi-stream pipeline that processes large datasets through a series of mathematical operations, demonstrating:
- **Stream-based parallelization** for overlapping computation and memory transfers
- **Event-driven synchronization** for precise timing and dependency management
- **Performance scaling** across different thread and stream configurations

### Pipeline Operations
```
Input Data → Normalization → 1D Convolution → Output Data
     ↓             ↓              ↓
   [H2D]        [Kernel]       [D2H]
```

**Stage 1: Normalization**
- Min-max scaling or z-score normalization
- Prepares data for convolution processing

**Stage 2: 1D Convolution**
- 5-tap filter kernel (edge detection, smoothing, or custom)
- Non-trivial mathematical operation with memory access patterns

## 🏗️ Technical Implementation

### Stream Architecture with Event Dependencies
```cpp
// Multi-stream pipeline (configurable 1-4 streams)
Stream 0: [H2D Copy] → [Normalize Kernel] → [Convolve Kernel] → [D2H Copy]
Stream 1: [H2D Copy] → [Normalize Kernel] → [Convolve Kernel] → [D2H Copy]
Stream 2: [H2D Copy] → [Normalize Kernel] → [Convolve Kernel] → [D2H Copy]
Stream 3: [H2D Copy] → [Normalize Kernel] → [Convolve Kernel] → [D2H Copy]

// Event synchronization with explicit dependencies:
cudaEvent_t e_h2d_start[S], e_h2d_end[S], e_norm_end[S], e_conv_end[S], e_d2h_end[S];

// Intra-stream dependency demonstration:
cudaEventRecord(e_norm_end[s], stream[s]);
cudaStreamWaitEvent(stream[s], e_norm_end[s], 0);  // Explicit normalize→conv dependency
```

### Memory Management (Pinned Host Memory)
```cpp
// Pinned host memory for true overlap
float *hx, *hy;
CHECK(cudaHostAlloc(&hx, N*sizeof(float), cudaHostAllocDefault));
CHECK(cudaHostAlloc(&hy, N*sizeof(float), cudaHostAllocDefault));

// Device memory
float *dx, *dy, *dtmp;
CHECK(cudaMalloc(&dx, N*sizeof(float)));
CHECK(cudaMalloc(&dy, N*sizeof(float)));
CHECK(cudaMalloc(&dtmp, N*sizeof(float)));
```

### Chunked Processing with Multiple Tiles
```cpp
// Multiple tiles per stream for visible overlap
const int tiles_per_stream = 2;     // Configurable
const size_t tile_size = (N + (streams*tiles_per_stream) - 1) / (streams*tiles_per_stream);

// Enqueue two tiles per stream for better overlap visibility
for (int tile = 0; tile < tiles_per_stream; tile++) {
    size_t offset = stream_id * tile_size + tile * tile_size;
    // Process tile with full pipeline
}
```

### Event-Driven Pipeline Implementation
```cpp
// Event layout per stream
cudaEvent_t e_h2d_start[S], e_h2d_end[S], e_norm_end[S], e_conv_end[S], e_d2h_end[S];
for (int s = 0; s < S; ++s) {
    CHECK(cudaEventCreate(&e_h2d_start[s]));
    CHECK(cudaEventCreate(&e_h2d_end[s]));
    CHECK(cudaEventCreate(&e_norm_end[s]));
    CHECK(cudaEventCreate(&e_conv_end[s]));
    CHECK(cudaEventCreate(&e_d2h_end[s]));
}

// Pipeline execution with event timing
// H2D
CHECK(cudaEventRecord(e_h2d_start[s], stream[s]));
CHECK(cudaMemcpyAsync(dx+offset, hx+offset, bytes, cudaMemcpyHostToDevice, stream[s]));
CHECK(cudaEventRecord(e_h2d_end[s], stream[s]));

// Normalize -> mark end
normalize<<<grid, block, 0, stream[s]>>>(dx+offset, dtmp+offset, n, params...);
CHECK(cudaEventRecord(e_norm_end[s], stream[s]));

// Explicit intra-stream dependency
CHECK(cudaStreamWaitEvent(stream[s], e_norm_end[s], 0));

// Convolution -> mark end
convolve5<<<grid, block, 0, stream[s]>>>(dtmp+offset, dy+offset, w, n);
CHECK(cudaEventRecord(e_conv_end[s], stream[s]));

// D2H
CHECK(cudaMemcpyAsync(hy+offset, dy+offset, bytes, cudaMemcpyDeviceToHost, stream[s]));
CHECK(cudaEventRecord(e_d2h_end[s], stream[s]));
```

## 📊 Performance Metrics & Analysis

### Measured Metrics
1. **Per-Stage Timing**: H2D, Normalize, Convolve, D2H (using events)
2. **Total Pipeline Time**: End-to-end processing time per stream configuration
3. **Throughput Metrics**: Elements/second and GB/s bandwidth
4. **Memory Bandwidth**: `gbps = ((H2D_bytes + D2H_bytes) / 1e9) / (total_ms/1000.0)`
5. **Speedup**: Performance improvement vs single-stream baseline
6. **Device Capabilities**: Print `deviceOverlap`, `asyncEngineCount`, `concurrentKernels`

### Robust Timing Methodology
```cpp
// Warm-up passes (not recorded)
for (int warmup = 0; warmup < 2; warmup++) {
    // Run full pipeline without timing
}

// Main timing runs with median calculation
std::vector<float> times;
for (int iter = 0; iter < iterations; iter++) {
    float total_time = run_pipeline();
    times.push_back(total_time);
}

// Report median (less noisy than average)
auto median = [](std::vector<float> v) {
    std::nth_element(v.begin(), v.begin()+v.size()/2, v.end());
    return v[v.size()/2];
};
```

### Comparison Studies
- **Stream Scaling**: 1 vs 2 vs 4 streams performance comparison
- **Block Size Impact**: 128, 256, 512, 1024 thread block analysis
- **Array Size Scaling**: 100K, 1M, 10M element processing
- **Iteration Averaging**: Multiple runs for statistical significance

## 🖥️ Command Line Interface

```bash
./pipeline [OPTIONS]

Options:
  --n SIZE        Array size (default: 1000000)
  --block SIZE    Thread block size (64-1024, default: 256)
  --streams NUM   Number of streams (1|2|4, default: 4)
  --iters NUM     Number of iterations (≥2, default: 10)
  --op OP         Operation: normalize|conv|both (default: both)
  --verify        Enable correctness verification
  --csv FILE      Output results to CSV file
  --help          Show help message

Examples:
  ./pipeline --n 1000000 --block 256 --streams 4 --iters 10
  ./pipeline --n 10000000 --block 512 --streams 2 --iters 5 --verify
  ./pipeline --n 500000 --block 128 --streams 1 --iters 20 --csv results.csv
```

### Input Validation
- **Block size**: Validated against device maximum threads per block
- **Streams**: Limited to 1, 2, or 4 for meaningful comparisons
- **Iterations**: Minimum 2 to meet rubric requirements
- **Array size**: Must be positive and reasonable for available memory

## 📁 Project Structure

```
CUDA_Stream_Events/
├── docs/
│   ├── assignment.md
│   ├── program.md
│   └── CUDA_Streams_and_Events_assignment.pdf
├── src/
│   ├── main.cpp              # Main program entry point
│   ├── pipeline.h            # Pipeline class declaration
│   ├── pipeline.cu           # Pipeline implementation
│   ├── kernels.cu            # CUDA kernel implementations
│   ├── kernels.h             # Kernel declarations
│   └── utils.h               # Utility functions
├── include/
│   └── cuda_utils.h          # CUDA helper functions
├── Makefile                  # Build configuration
├── README.md                 # Project documentation
└── results/                  # Performance results and screenshots
    ├── screenshots/
    └── performance_data/
```

## 🎯 Implementation Milestones

### Phase 1: Core Infrastructure (Week 1)
- [ ] **Milestone 1.1**: Project setup and build system
  - [ ] Create project directory structure
  - [ ] Set up Makefile with CUDA compilation
  - [ ] Implement command line argument parsing with validation
  - [ ] Create CHECK macro for CUDA error handling
  - [ ] Add device capability detection and reporting

- [ ] **Milestone 1.2**: Pinned memory management
  - [ ] Implement pinned host memory allocation (`cudaHostAlloc`)
  - [ ] Add device memory allocation with proper cleanup
  - [ ] Create chunked processing with multiple tiles per stream
  - [ ] Implement memory bandwidth calculation utilities

### Phase 2: Kernels + Streams/Events (Week 1-2)
- [ ] **Milestone 2.1**: Kernel development with immediate stream integration
  - [ ] Implement min-max normalization kernel (simpler than z-score)
  - [ ] Implement 1D convolution with 5-tap filter
  - [ ] Create CPU reference implementations for verification
  - [ ] Add kernel correctness validation with `--verify` flag

- [ ] **Milestone 2.2**: Event-driven pipeline
  - [ ] Create event arrays for per-stage timing
  - [ ] Implement explicit intra-stream dependencies
  - [ ] Add warm-up passes and median timing calculation
  - [ ] Wire streams/events immediately after kernel development

### Phase 4: Pipeline Integration (Week 2-3)
- [ ] **Milestone 4.1**: Complete pipeline
  - [ ] Integrate all components into working pipeline
  - [ ] Implement overlapped memory transfers
  - [ ] Add pipeline dependency management
  - [ ] Test end-to-end functionality

- [ ] **Milestone 4.2**: Performance optimization
  - [ ] Optimize memory access patterns
  - [ ] Tune thread block sizes for different operations
  - [ ] Implement dynamic stream allocation
  - [ ] Add performance profiling capabilities

### Phase 5: Testing and Validation (Week 3)
- [ ] **Milestone 5.1**: Comprehensive testing
  - [ ] Create test harness with multiple runs (≥2 iterations)
  - [ ] Implement correctness validation with CPU references
  - [ ] Add performance regression testing
  - [ ] Test edge cases and error conditions
  - [ ] Generate run.sh script for automated configuration matrix

- [ ] **Milestone 5.2**: Performance analysis and reporting
  - [ ] Generate CSV output with all metrics
  - [ ] Create performance comparison tables
  - [ ] Report median + IQR (p25-p75) for statistical robustness
  - [ ] Optional: Capture Nsight Systems timeline for visual proof of overlap

### Phase 6: Documentation and Submission (Week 3-4)
- [ ] **Milestone 6.1**: Code quality
  - [ ] Add comprehensive code comments
  - [ ] Implement consistent naming conventions
  - [ ] Add input validation and error handling
  - [ ] Create code documentation

- [ ] **Milestone 6.2**: Results and demonstration
  - [ ] Generate performance screenshots
  - [ ] Create demonstration videos
  - [ ] Document performance findings
  - [ ] Prepare submission materials

## 📈 Expected Performance Results

### Stream Scaling Performance
```
Array Size: 1,000,000 elements
Block Size: 256 threads
Iterations: 10

Streams | Total Time (ms) | Throughput (MEl/s) | Bandwidth (GB/s) | Speedup
--------|----------------|-------------------|------------------|---------
   1    |     45.2      |       22.1        |      185         |  1.00x
   2    |     28.7      |       34.8        |      292         |  1.57x
   4    |     18.3      |       54.6        |      458         |  2.47x
```

### Per-Stage Timing Breakdown (4 streams)
```
Stage     | Time (ms) | % of Total | Bandwidth (GB/s)
----------|-----------|------------|-----------------
H2D Copy  |    4.2    |    23%     |      445
Normalize |    6.8    |    37%     |      275
Convolve  |    5.1    |    28%     |      367
D2H Copy  |    2.2    |    12%     |      455
Total     |   18.3    |   100%     |      458
```

### Device Capabilities Report
```
Device: NVIDIA GeForce RTX 3080
Concurrent Kernels: Yes (8)
Async Engine Count: 2
Device Overlap: Yes
Max Threads per Block: 1024
```

### CSV Output Format
```csv
streams,block_size,array_size,iterations,total_ms,h2d_ms,norm_ms,conv_ms,d2h_ms,throughput_mel_s,bandwidth_gbps,speedup
1,256,1000000,10,45.2,8.1,18.4,14.2,4.5,22100,185.0,1.00
2,256,1000000,10,28.7,4.2,12.1,9.8,2.6,34800,292.0,1.57
4,256,1000000,10,18.3,4.2,6.8,5.1,2.2,54600,458.0,2.47
```

## 🎯 Rubric Alignment

| Requirement | Implementation | Points |
|-------------|---------------|--------|
| **CUDA Streams and Events** | Multi-stream pipeline with event synchronization | 50/50 |
| **Two separate kernels** | Normalization + Convolution kernels | 10/10 |
| **Timing metrics** | Comprehensive performance analysis | 10/10 |
| **Code quality** | Constants, naming, comments, validation | 20/20 |
| **Command line args** | --n, --block, --streams, --iters | 10/10 |
| **Final project bonus** | Pipeline framework for ML preprocessing | 5/5 |
| **Total** | | **105/100** |

## 🚀 Future Project Integration

### Final Project Integration (+5% Bonus)
This pipeline (multi-stream H2D → normalize → convolve → D2H with event timing) will be reused in my final project as the data pre-processing stage that feeds batched inference. The same double/triple-buffered structure lets me overlap transfers with per-batch transforms, maintaining a steady GPU input queue and improving end-to-end throughput. I will swap the 1D conv with my project's feature transforms, but the streams/events backbone remains identical.

### Extension Opportunities
- **Machine Learning**: Data preprocessing pipeline for neural networks
- **Image Processing**: Real-time image filtering and enhancement  
- **Signal Processing**: Audio/video processing applications
- **Scientific Computing**: Numerical analysis and simulation pipelines

The modular design allows easy integration of additional processing stages and optimization for specific application domains.

## 📝 Success Criteria

- [ ] Program compiles and runs without errors
- [ ] Demonstrates clear performance improvement with multiple streams
- [ ] Shows measurable timing differences across configurations
- [ ] Generates compelling performance visualization
- [ ] Includes comprehensive documentation and comments
- [ ] Provides clear command-line interface
- [ ] Achieves target performance benchmarks
- [ ] Ready for final project integration
