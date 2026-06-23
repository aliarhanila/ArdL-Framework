#ifndef ARDL_MATH_H
#define ARDL_MATH_H

#include "ardl_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

// --- Data Structures ---
typedef struct {
    int rows;
    int cols;
    float *data;
} Matrix;

//--- Matrix Creation---
Matrix* create_matrix_area(MemoryArena *arena, int rows, int cols);

// --- Matrix Operations ---
void transpose_matrix(const Matrix *src, Matrix *dst);
void matrix_dot(const Matrix *A, const Matrix *B, Matrix *C);
void matrix_dot_optimized(const Matrix *A, const Matrix *B_T, Matrix *C);

//--- Random Numbers Generators---
float random_randn();
void randomize_matrix_he(Matrix *mat, int fan_in);
void randomize_matrix_xavier(Matrix *mat, int fan_in);


void im2col(const float* data_im, int channels, int height, int width, int kernel_size, int stride, int padding, float* data_col);
void col2im(const float* data_col, int channels, int height, int width, int kernel_size, int stride, int padding, float* data_im);



#endif
