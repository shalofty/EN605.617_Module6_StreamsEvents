# CUDA Streams and Events Pipeline - Makefile
# Compiler and flags
NVCC = nvcc
CC = gcc

# CUDA compute capability (adjust based on your GPU)
CUDA_ARCH = -arch=sm_75

# Compiler flags
NVCC_FLAGS = $(CUDA_ARCH) -O3 -std=c++14 -lineinfo
CC_FLAGS = -O3 -std=c99 -Wall -Wextra

# Include directories
INCLUDES = -I./include

# Source files
CUDA_SOURCES = src/main.cu src/kernels.cu src/pipeline.cu src/utils.cu

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
OBJECTS = $(CUDA_OBJECTS)

# Executable name
TARGET = pipeline

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $(TARGET)

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Clean everything including results
distclean: clean
	rm -rf results/performance_data/*.csv results/screenshots/*.png

# Run basic test
test: $(TARGET)
	./$(TARGET) --n 1000000 --block 256 --streams 4 --iters 5

# Run verification test
verify: $(TARGET)
	./$(TARGET) --n 100000 --block 256 --streams 2 --iters 3 --verify

# Run performance benchmark
benchmark: $(TARGET)
	./$(TARGET) --n 10000000 --block 512 --streams 4 --iters 10 --csv results/performance_data/benchmark.csv

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Build the pipeline executable"
	@echo "  clean     - Remove build artifacts"
	@echo "  distclean - Remove build artifacts and results"
	@echo "  test      - Run basic functionality test"
	@echo "  verify    - Run correctness verification test"
	@echo "  benchmark - Run performance benchmark"
	@echo "  help      - Show this help message"

# Declare phony targets
.PHONY: all clean distclean test verify benchmark help
