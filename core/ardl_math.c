#include "ardl_math.h"


#ifndef __cplusplus
#define nullptr ((void*)0)
#endif

/*
#---------------------------------
# Matrix Creations and release
#----------------------------------
*/
Matrix* create_matrix_area(MemoryArena *arena, int rows, int cols){
    Matrix *mat = (Matrix*)arena_alloc(arena, sizeof(Matrix));
    if(!mat) return nullptr;

    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float*)arena_alloc(arena, rows * cols * sizeof(float));
    
    if(!mat->data) return nullptr;
    
    return mat;
}

/*
#----------------------------
# Random Number Generator
#---------------------------
*/
float random_randn() {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    
    // Prevent log(0) calculation
    if (u1 <= 1e-7f) u1 = 1e-7f; 
    
    // Box-Muller Transformation
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.1415926535f * u2);
}

/*
#-----------------------------
# Random Number Matrix
#-----------------------------
*/
void randomize_matrix_xavier(Matrix *mat, int fan_in){
    float scale = sqrtf(1.0f / fan_in);
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = random_randn() * scale;
    }
}

void randomize_matrix_he(Matrix *mat, int fan_in){
    float scale = sqrtf(2.0f / fan_in);
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = random_randn() * scale;
    }
}

/*
#-------------------------
# Matrix Operations 
#-------------------------
*/
void transpose_matrix(const Matrix *src, Matrix *dst) {
    if (src->rows != dst->cols || src->cols != dst->rows) {
        printf("Error (transpose_matrix): Dimension mismatch!\n");
        return;
    }
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            float val = src->data[i * src->cols + j];
            dst->data[j * dst->cols + i] = val;
        }
    }
}

void matrix_dot(const Matrix *A, const Matrix *B, Matrix *C) {
    if(A->cols != B->rows) return;
    
    #pragma omp parallel for
    for(int i = 0; i < C->rows * C->cols; i++) C->data[i] = 0.0f;
    
    #pragma omp parallel for
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < B->cols; j++) {
            float sum = 0.0f;
            for(int k = 0; k < A->cols; k++) {
                sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
            C->data[i * C->cols + j] = sum;
        }
    }
}

void matrix_dot_optimized(const Matrix *A, const Matrix *B_T, Matrix *C) {
    if(A->cols != B_T->cols) return;
    
    #pragma omp parallel for
    for(int i = 0; i < C->rows * C->cols; i++) C->data[i] = 0.0f;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < B_T->rows; j++) {
            float sum = 0.0f;
            for(int k = 0; k < A->cols; k++) {
                sum += A->data[i * A->cols + k] * B_T->data[j * B_T->cols + k];
            }
            C->data[i * C->cols + j] = sum;
        }
    }
}



/*
#---------------------
# im2col -forward
#---------------------
*/
void im2col(const float* data_im, int channels, int height, int width, int kernel_size, int stride, int padding, float* data_col){
    // Feature map dimensions
    int out_h = (height + 2 * padding - kernel_size) / stride + 1; 
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    
    // data_col matrix total number of rows (1 filter window flatten dimension)
    int channel_col = channels * kernel_size * kernel_size;

    #pragma omp parallel for 
    for(int c = 0; c < channel_col; ++c){
        int w_offset = c % kernel_size;
        int h_offset = (c / kernel_size) % kernel_size;
        int c_im = c / (kernel_size * kernel_size);

        for(int h = 0; h < out_h; ++h){
            for(int w = 0; w < out_w; ++w){
                int im_row = h_offset + h * stride - padding;
                int im_col = w_offset + w * stride - padding;

                int col_index = (c * out_h + h) * out_w + w;

                if(im_row >= 0 && im_row < height && im_col >= 0 && im_col < width){
                    int im_index = (c_im * height + im_row) * width + im_col;
                    data_col[col_index] = data_im[im_index];
                }
                else{
                    data_col[col_index] = 0.0f;
                }
            }
        }
    }
}
/*
#-----------------------------------
# col2im (Gradients Accumulation)
#-----------------------------------
*/
void col2im(const float* data_col, int channels, int height, int width, int kernel_size, int stride, int padding, float* data_im) {
    int out_h = (height + 2 * padding - kernel_size) / stride + 1; 
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    int channel_col = channels * kernel_size * kernel_size;
    int spatial_dim = channels * height * width;

    // target matris is reset
    #pragma omp parallel for
    for(int i = 0; i < spatial_dim; i++){
        data_im[i] = 0.0f;
    }

    #pragma omp parallel for 
    for(int c = 0; c < channel_col; ++c){
        int w_offset = c % kernel_size;
        int h_offset = (c / kernel_size) % kernel_size;
        int c_im = c / (kernel_size * kernel_size);

        for(int h = 0; h < out_h; ++h){
            for(int w = 0; w < out_w; ++w){
                int im_row = h_offset + h * stride - padding;
                int im_col = w_offset + w * stride - padding;
                int col_index = (c * out_h + h) * out_w + w;

                if(im_row >= 0 && im_row < height && im_col >= 0 && im_col < width){
                    int im_index = (c_im * height + im_row) * width + im_col;
                    
                    
                    #pragma omp atomic //race condition obstacles
                    data_im[im_index] += data_col[col_index];
                }
            }
        }
    }
}

