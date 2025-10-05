#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declarations for utility functions
float median(float* values, int count);
void generate_test_data(float* data, size_t size);
int compare_arrays(const float* a, const float* b, size_t size, float tolerance);

#endif // UTILS_H
