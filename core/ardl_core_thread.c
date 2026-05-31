// ardl_core_thread.c
// Final, Cache-Optimized & Multithreaded Forward Pass Engine
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <omp.h>
#include "ardl_core_thread.h"

/*
#----------------------------
# Memory Area Functions
#---------------------------
*/
MemoryArena* arena_create(size_t size){
    MemoryArena *arena = malloc(sizeof(MemoryArena));
    arena->buffer = malloc(size);
    arena->size = size;
    arena->offset = 0;
    return arena;
}

void* arena_alloc(MemoryArena *arena, size_t size){
    size = (size + 7) & ~7; // 8-byte alignment for stable memory mapping

    if(arena->offset + size > arena->size){
        printf("Error (Memory): Arena out of memory!\n");
        return NULL;
    }

    void *ptr = (void*)(arena->buffer + arena->offset);
    arena->offset += size;
    return ptr;
}

void arena_reset(MemoryArena *arena){
    arena->offset = 0;
}

void arena_free(MemoryArena *arena){
    free(arena->buffer);
    free(arena);
}

/*
# ---------------------------
#  Activations
# ---------------------------
*/
float ReLu(float x){
    return fmaxf(0.0f, x);
}

float ReLu_grad(float x){
    return (x > 0.0f) ? 1.0f : 0.0f;
}

float Tanh(float x) {
    return tanhf(x); 
}

float Tanh_grad(float x) {
    float t = tanhf(x);
    return 1.0f - (t * t);
}

float LeakyReLu(float x){
    return (x > 0.0f) ? x : 0.01f * x;
}

float LeakyReLu_grad(float x){
    return (x > 0.0f) ? 1.0f : 0.01f;
}

float sigmoid(float x){
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_grad(float x){
    float s = sigmoid(x);
    return s * (1.0f - s);
}

void softmax(Matrix *z, Matrix *a) {
    int batch_size = z->rows;
    int out_dim = z->cols;
    
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        // Step 1: Find max value for numerical stability (prevent overflow)
        float max_val = z->data[i * out_dim];
        for (int j = 1; j < out_dim; j++) {
            if (z->data[i * out_dim + j] > max_val) {
                max_val = z->data[i * out_dim + j];
            }
        }

        // Step 2: Subtract max_val, apply exp(), and calculate sum
        float sum_exp = 0.0f;
        for (int j = 0; j < out_dim; j++) {
            float e = expf(z->data[i * out_dim + j] - max_val);
            a->data[i * out_dim + j] = e;
            sum_exp += e;
        }

        // Step 3: Divide by sum to get probability distribution
        for (int j = 0; j < out_dim; j++) {
            a->data[i * out_dim + j] /= sum_exp;
        }
    }
}

/*
#---------------------------------
# Matrix Creations and release
#----------------------------------
*/
Matrix* create_matrix_area(MemoryArena *arena, int rows, int cols){
    Matrix *mat = (Matrix*)arena_alloc(arena, sizeof(Matrix));
    if(!mat) return NULL;

    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float*)arena_alloc(arena, rows * cols * sizeof(float));
    
    if(!mat->data) return NULL;
    
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
# ----------------------
# Layer Initialization
# ----------------------
*/
DenseLayer* init_dense_layer(MemoryArena *arena, int input_dim , int output_dim, int batch_size, int activation_type){
    DenseLayer *layer = (DenseLayer*)arena_alloc(arena, sizeof(DenseLayer));
    if(!layer) return NULL;
    
    // Assign activation type to the layer
    layer->activation_type = activation_type; 

    layer->weights = create_matrix_area(arena, input_dim, output_dim);

    if (activation_type == 0) {
        randomize_matrix_he(layer->weights, input_dim);
    } else {
        randomize_matrix_xavier(layer->weights, input_dim);
    }

    layer->weights_T = create_matrix_area(arena, output_dim, input_dim);
    transpose_matrix(layer->weights, layer->weights_T);

    layer->biases = create_matrix_area(arena, 1, output_dim);
    for (int i = 0; i < output_dim; i++){
        layer->biases->data[i] = 0.0f;
    }

    layer->z = create_matrix_area(arena, batch_size, output_dim);
    layer->a = create_matrix_area(arena, batch_size, output_dim);

    layer->delta = create_matrix_area(arena, batch_size, output_dim);
    layer->dW    = create_matrix_area(arena, input_dim, output_dim);
    layer->db    = create_matrix_area(arena, 1, output_dim);
    
    return layer; 
}

/*
#--------------------------
# Loss functions
#------------------------
*/
float mse_loss(Matrix *y_true, Matrix *y_pred){
    float total_loss = 0.0f;
    int total_elements = y_true->rows * y_true->cols;
    
    #pragma omp parallel for reduction(+:total_loss)
    for(int i = 0 ; i < total_elements ; i++){
        float err = y_pred->data[i] - y_true->data[i];
        total_loss += err * err;
    }
    
    return total_loss / (float)y_true->rows;
}

float binary_crossentropy(Matrix *y_true, Matrix *y_pred){
    float total_loss = 0.0f;
    int batch_size = y_true->rows;
    int total_elements = y_true->rows * y_true->cols;
    
    #pragma omp parallel for reduction(+:total_loss)
    for(int i = 0 ; i < total_elements ; i++){
        float y_t = y_true->data[i];
        float y_p = y_pred->data[i];
        
        // Add epsilon (1e-9) to prevent log(0) calculation
        total_loss += -(y_t * logf(y_p + 1e-9f) + (1.0f - y_t) * logf(1.0f - y_p + 1e-9f));
    }
    
    return (total_loss / (float)batch_size);
}

float categorical_crossentropy(Matrix *y_true, Matrix *y_pred){
    float total_loss = 0.0f;
    int batch_size = y_true->rows;
    int total_elements = y_true->rows * y_true->cols;
    
    #pragma omp parallel for reduction(+:total_loss)
    for(int i = 0 ; i < total_elements ; i++){
        float y_t = y_true->data[i];
        float y_p = y_pred->data[i];
        total_loss += y_t * logf(y_p + 1e-9f);
    }
    
    return -(total_loss / (float)batch_size);
}

float msle_loss(Matrix *y_true, Matrix *y_pred){
    float total_loss = 0.0f;
    int total_elements = y_true->rows * y_true->cols;
    
    #pragma omp parallel for reduction(+:total_loss)
    for(int i = 0 ; i < total_elements ; i++){
        // Add fmaxf and +1.0f to avoid negative values (log(0) is undefined)
        float log_true = logf(fmaxf(y_true->data[i], 0.0f) + 1.0f);
        float log_pred = logf(fmaxf(y_pred->data[i], 0.0f) + 1.0f);
        float err = log_pred - log_true;
        total_loss += err * err;
    }
    
    return total_loss / (float)y_true->rows;
}

/*
#-----------------------------
# Forward Propagation
#-----------------------------
*/
void forward_pass(DenseLayer *layer, Matrix *input_a)
{
    // 1. Z = W^T * X
    matrix_dot_optimized(input_a, layer->weights_T, layer->z);

    // 2. Z = Z + bias
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < layer->z->rows; i++){
        for(int j = 0; j < layer->z->cols; j++){
            layer->z->data[i * layer->z->cols + j] += layer->biases->data[j];
        }
    }

    // 3. Activation based on new API indices
    if (layer->activation_type == 5) { // 5 = Softmax (Vectorial)
        softmax(layer->z, layer->a);
    } 
    else {
        // Scalar (element-wise) activations
        int total = layer->z->rows * layer->z->cols;
        
        #pragma omp parallel for
        for(int i = 0; i < total; i++){
            float val = layer->z->data[i];
            
            if (layer->activation_type == 0)      // 0 = Linear
                layer->a->data[i] = val; 
            else if (layer->activation_type == 1) // 1 = ReLU
                layer->a->data[i] = ReLu(val);
            else if (layer->activation_type == 2) // 2 = LeakyReLU
                layer->a->data[i] = LeakyReLu(val);
            else if (layer->activation_type == 3) // 3 = Tanh
                layer->a->data[i] = Tanh(val);
            else if (layer->activation_type == 4) // 4 = Sigmoid
                layer->a->data[i] = sigmoid(val);
        }
    }
}

/*
#-----------------------
# Backpropagation
#-----------------------
*/
void compute_gradients(DenseLayer *layer, Matrix *input_a){
    float batch_size = (float)layer->delta->rows; 

    // Biases Gradient
    #pragma omp parallel for schedule(static)
    for(int j = 0; j < layer->db->cols; j++){
        float sum = 0.0f;
        for(int i = 0; i < layer->delta->rows; i++){
            sum += layer->delta->data[i * layer->delta->cols + j];
        }
        // Gradient Normalization
        layer->db->data[j] = sum / batch_size;
    }

    // Weights Gradient (BOTTLENECK RESOLVED: Full power with collapse(2))
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i = 0; i < input_a->cols; i++){
        for(int j = 0; j < layer->delta->cols; j++){
            float sum = 0.0f;
            for(int k = 0; k < input_a->rows; k++){
                float a_val = input_a->data[k * input_a->cols + i];
                float d_val = layer->delta->data[k * layer->delta->cols + j];
                sum += a_val * d_val;
            }
            // Gradient Normalization
            layer->dW->data[i * layer->dW->cols + j] = sum / batch_size;
        }
    }
}

void update_weights(DenseLayer *layer, float lr){
    int w_elements = layer->weights->rows * layer->weights->cols;
    
    #pragma omp parallel for
    for(int i = 0; i < w_elements; i++){
        layer->weights->data[i] -= lr * layer->dW->data[i]; 
    }

    int b_elements = layer->biases->rows * layer->biases->cols;
    
    #pragma omp parallel for
    for(int i = 0; i < b_elements; i++){
        layer->biases->data[i] -= lr * layer->db->data[i]; 
    }

    transpose_matrix(layer->weights, layer->weights_T);
}

/*
#-----------------------------
# Core serialize funcs
#-----------------------------
*/
void core_serialize_weights(DenseLayer *layer, FILE *file) {
    int w_size = layer->weights->rows * layer->weights->cols;
    fwrite(layer->weights->data, sizeof(float), w_size, file);

    int b_size = layer->biases->rows * layer->biases->cols;
    fwrite(layer->biases->data, sizeof(float), b_size, file);
}

void core_deserialize_weights(DenseLayer *layer, FILE *file) {
    int w_size = layer->weights->rows * layer->weights->cols;
    fread(layer->weights->data, sizeof(float), w_size, file);

    int b_size = layer->biases->rows * layer->biases->cols;
    fread(layer->biases->data, sizeof(float), b_size, file);

    transpose_matrix(layer->weights, layer->weights_T);
}