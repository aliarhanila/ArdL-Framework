#ifndef ARDL_CORE_THREAD_H
#define ARDL_CORE_THREAD_H

#include <stddef.h> // size_t için
#include <stdio.h>  // FILE için

// --- Memory Structures ---
typedef struct {
    unsigned char *buffer;
    size_t size;
    size_t offset;
} MemoryArena;

// --- Data Structures ---
typedef struct {
    int rows;
    int cols;
    float *data;
} Matrix;

//---Conv2d struct ---
typedef struct{
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
 
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    int batch_size;

    Matrix *kernels;
    Matrix *biases;
    Matrix *im2col_buffer;
    
    Matrix *dx_col_buffer;

    Matrix *z;
    Matrix *a;
    Matrix *delta;
    Matrix *dK;
    Matrix *db;

    int activation_type;

} Conv2DLayer; 

//---MaxPool2D Layer Structure---
typedef struct {
    int channels;       
    int pool_size;      
    int stride;         
    int padding;        
    
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    int batch_size;

    Matrix *a;          // Activation output
    Matrix *delta;      // Gradient
    
    // BACKPROP ROUTING MEMORY: Keeps the 1D index of the winning pixel
    int *max_indices;   
} MaxPool2DLayer;

//--- Dense Layer Structes ---
typedef struct {
    Matrix *weights;
    Matrix *weights_T;
    Matrix *biases;
    Matrix *z;
    Matrix *a;
    Matrix *delta;
    Matrix *dW;
    Matrix *db;

    int activation_type;
} DenseLayer;

// --- 1. Memory Management ---
MemoryArena* arena_create(size_t size);
void* arena_alloc(MemoryArena *arena, size_t size);
void arena_reset(MemoryArena *arena);
void arena_free(MemoryArena *arena);

// --- 2. Matrix & Layers ---
Matrix* create_matrix_area(MemoryArena *arena, int rows, int cols);
DenseLayer* init_dense_layer(MemoryArena *arena, int input_dim , int output_dim, int batch_size, int activation_type);
Conv2DLayer* init_conv2d_layer(MemoryArena *arena, int in_channels, int out_channels, int kernel_size, int stride, int padding,int input_height, int input_width, int batch_size, int activation_type);
MaxPool2DLayer* init_maxpool2d_layer(MemoryArena *arena, int channels, int pool_size, int stride, int padding, int input_height, int input_width, int batch_size);

// --- Forward and Backward direction geometry engines ---  
void im2col(const float* data_im, int channels, int height, int width, int kernel_size, int stride, int padding, float* data_col);
void col2im(const float* data_col, int channels, int height, int width, int kernel_size, int stride, int padding, float* data_im);

// --- 3. Activations & Utils ---
float ReLu(float x);
float ReLu_grad(float x);
float LeakyReLu(float x);
float Tanh(float x);
float Tanh_grad(float x);
float LeakyReLu_grad(float x);
float sigmoid(float x);
float sigmoid_grad(float x);
void softmax(Matrix *z, Matrix *a);
void apply_activations(Matrix *z, Matrix *a, int activation_type);

float random_randn();
void randomize_matrix_he(Matrix *mat, int fan_in);
void randomize_matrix_xavier(Matrix *mat, int fan_in);

// --- 4. Matrix Operations ---
void transpose_matrix(const Matrix *src, Matrix *dst);
void matrix_dot(const Matrix *A, const Matrix *B, Matrix *C);
void matrix_dot_optimized(const Matrix *A, const Matrix *B_T, Matrix *C);

// --- 5. Forward / Backward ---
// Dense
void forward_pass(DenseLayer *layer, Matrix *input_a);
void compute_gradients(DenseLayer *layer , Matrix *input_a);
void update_weights(DenseLayer *layer, float lr);

// Conv2D
void forward_conv2d(Conv2DLayer *layer, Matrix *input_a);
void compute_gradients_conv2d(Conv2DLayer *layer, Matrix *input_a, Matrix *prev_layer_delta);

// MaxPool2D
void forward_maxpool2d(MaxPool2DLayer *layer, Matrix *input_a);
void backward_maxpool2d(MaxPool2DLayer *layer, Matrix *prev_layer_delta);

// --- 6. Loss Functions ---
float binary_crossentropy(Matrix *y_true, Matrix *y_pred);
float categorical_crossentropy(Matrix *y_true, Matrix *y_pred);
float mse_loss(Matrix *y_true, Matrix *y_pred);
float msle_loss(Matrix *y_true, Matrix *y_pred);

// --- Core Serialization ---
void core_serialize_weights(DenseLayer *layer, FILE *file);
void core_deserialize_weights(DenseLayer *layer, FILE *file);

#endif