//ardl_core.c
//Final , Cache-Optimized Forward Pass Engine
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include "ardl_core.h"

/*
#----------------------------
# Memory Area Function
#---------------------------
*/

//Arena İnit
MemoryArena* arena_create(size_t size){
    MemoryArena *arena = malloc(sizeof(MemoryArena));
    arena->buffer = malloc(size);
    arena->size = size;
    arena->offset = 0;
    return arena;
}

//Arena Alloc Function
void* arena_alloc(MemoryArena *arena,size_t size){
    //alignment 
    size = (size + 7) & ~7; // 8 byte alignment(for embedded systems)

    if(arena->offset + size > arena->size){
        printf("Error (Memory): Arena out of memory!\n");
        return NULL;
    }

    void *ptr = (void*)(arena->buffer + arena->offset);

    arena->offset +=size;

    return ptr;
}

//arena offset reset function
void arena_reset(MemoryArena *arena){
    arena->offset = 0;
}

//Erase Arena function
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
    return fmaxf(0.0f,x);
}
float ReLu_grad(float x){
    return (x > 0.0f) ? 1.0f : 0.0f;
}

float sigmoid(float x){
    return 1.0f/(1.0f+expf(-x));
}

float sigmoid_grad(float x){
    float s = sigmoid(x);
    return s * (1.0f - s);
}
/*
#---------------------------------
# Matrix Creations and release
#----------------------------------
*/
Matrix* create_matrix_area(MemoryArena *arena,int rows, int cols){
    Matrix *mat = (Matrix*)arena_alloc(arena ,sizeof(Matrix));
    if(!mat){
        return NULL;
    } 

    mat->rows = rows;
    mat->cols = cols;
    
    mat->data = (float*)arena_alloc(arena , rows * cols * sizeof(float));
    
    // if memory data is not allocated
    if(!mat->data){
        return NULL;
    }
    

    return mat;
}


/*
#----------------------------
#Random Number Generator Function
#---------------------------
*/
float random_randn() {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    
    //(epsilon)
    if (u1 <= 1e-7f) u1 = 1e-7f; 
    
    // Box-Muller Transformation
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.1415926535f * u2);
}

/*
#-----------------------------
# Random Number Matrix
#-----------------------------
*/
void randomize_matrix(Matrix *mat) {
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = random_randn() * 0.1f; 
    }
}


/*
#-------------------------
# Matrix Operations 
#-------------------------
*/

// Transpose matrix for cache optimization (Row-Major to Column-Major mapping)
void transpose_matrix(const Matrix *src, Matrix *dst) {
    
    // 1. Safety Check: Destination matrix dimensions must be the inverse of the source.
    if (src->rows != dst->cols || src->cols != dst->rows) {
        printf("Error (transpose_matrix): Dimension mismatch! SRC(%d,%d) -> DST(%d,%d)\n", 
               src->rows, src->cols, dst->rows, dst->cols);
        return;
    }

    // 2. Transpose Loop
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            
            // Read formula from source matrix: (row * total_columns + col)
            float val = src->data[i * src->cols + j];
            
            // Write formula to destination matrix (Critical Point):
            // i and j are swapped! dst->cols is actually equal to src->rows.
            dst->data[j * dst->cols + i] = val;
        }
    }
}

// General Matrix Multiplication (Dot Product)
void matrix_dot(const Matrix *A, const Matrix *B, Matrix *C) {
    
    // Matrix dimensions check
    if(A->cols != B->rows) {
        printf("Error (matrix_dot): Matrix dimensions are incompatible A(%d,%d) x B(%d,%d)\n", 
               A->rows, A->cols, B->rows, B->cols);
        return;
    }
    
    // Clear destination matrix C
    for(int i = 0; i < C->rows * C->cols; i++) {
        C->data[i] = 0.0f;
    }
    
    // GEMM Loops
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < B->cols; j++) {
            float sum = 0.0f;

            for(int k = 0; k < A->cols; k++) {
                
                float a_val = A->data[i * A->cols + k];
                float b_val = B->data[k * B->cols + j]; 

                sum += a_val * b_val;
            }

            C->data[i * C->cols + j] = sum;
        }
    }
}



// Optimized GEMM (Uses Transposed B to prevent Cache Misses)
void matrix_dot_optimized(const Matrix *A, const Matrix *B_T, Matrix *C) {
    
    // Check (B_T->cols is actually original B's rows)
    if(A->cols != B_T->cols) {
        printf("Error (matrix_dot_opt): Dimensions incompatible!\n");
        return;
    }
    
    for(int i = 0; i < C->rows * C->cols; i++) {
        C->data[i] = 0.0f;
    }
    
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < B_T->rows; j++) { // Iterating over B_T's rows
            float sum = 0.0f;

            for(int k = 0; k < A->cols; k++) {
                
                float a_val = A->data[i * A->cols + k];
                float b_val = B_T->data[j * B_T->cols + k]; 

                sum += a_val * b_val;
            }
            C->data[i * C->cols + j] = sum;
        }
    }
}


/*
# ----------------------
# Layer İnitilazation
# ----------------------
*/

//Allocates memory for a complate dense layer and initializes weights/biases

DenseLayer* init_dense_layer(MemoryArena *arena, int input_dim , int output_dim ,int batch_size){
    DenseLayer *layer = (DenseLayer*)arena_alloc(arena,sizeof(DenseLayer));
    //if layer is null
    if(!layer){
        return NULL;
    }

    //İnitilaze Weights
    layer->weights = create_matrix_area(arena,input_dim,output_dim);
    randomize_matrix(layer->weights);
    //optimizing cache
    layer->weights_T = create_matrix_area(arena,output_dim, input_dim);
    transpose_matrix(layer->weights, layer->weights_T);

    //İnitilaze Biases
    layer->biases = create_matrix_area(arena,1,output_dim);

    //Allocate Memory for Forward Pass States
    layer->z = create_matrix_area(arena,batch_size,output_dim);
    layer->a = create_matrix_area(arena,batch_size,output_dim);

    //Allocate Memory for BackPropagation Gradients
    layer->delta = create_matrix_area(arena,batch_size,output_dim);
    layer->dW    = create_matrix_area(arena,input_dim,output_dim);
    layer->db    = create_matrix_area(arena,1,output_dim);
    
    return layer ; 
}

/*
#-----------------------------
# Layer Destruction
#-----------------------------
*/
// Safely frees all allocated matrix memory within the layer to prevent memory leaks

/*
#--------------------------
# Loss function
#------------------------
*/
float categorical_crossentropy(Matrix *y_true,Matrix *y_pred){
    if(y_true->rows != y_pred->rows || y_true->cols != y_pred->cols){
        printf("Error(crossentropy): Dimensions incompatible! Y_true(%d,%d) vs Y_pred(%d,%d)\n", 
               y_true->rows, y_true->cols, y_pred->rows, y_pred->cols);
        return -1.0f;
    }
    
    float total_loss = 0.0f;
    int batch_size = y_true->rows;
    int total_elements = y_true->rows *y_true->cols;
    
    for(int i = 0 ; i < total_elements ; i++){
        float y_t = y_true->data[i];
        float y_p = y_pred->data[i];


        total_loss += y_t * logf(y_p + 1e-9f);
    }
    
    return -(total_loss/ (float)batch_size);
}

/*
#-----------------------------
# Forward Propagation
#-----------------------------
*/
// Executes the forward pass for a single layer: Z = Input * W + b, then A = Activation(Z)
// activation_type: 0 for ReLu, 1 for Sigmoid
void forward_pass(DenseLayer *layer,Matrix *input_a,int activation_type){
    //Compute matrix multiplication (Z = input_a *W)
    matrix_dot_optimized(input_a ,layer->weights_T , layer->z);

    //Add Bias vector with broadcasting over the batch rows (Z = Z + b)
    for(int i = 0 ; i < layer->z->rows ; i++){
        for(int j = 0 ; j<layer->z->cols ; j++){
            // Index mapping for Row Major 1D array execution
            layer->z->data[i * layer->z->cols +j] += layer->biases->data[j];
        }
    }

    int total_elements = layer->z->rows *layer->z->cols;
    for(int i = 0 ; i < total_elements ; i++){
        if (activation_type == 0){
            layer->a->data[i] = ReLu(layer->z->data[i]);
        }
        else if (activation_type == 1){
            layer->a->data[i] = sigmoid(layer->z->data[i]);
        }
    }

}

/*
#-----------------------
# Backpropagation
#-----------------------
*/

// Calculates Gradients(dW,db) based on the layer's current delta
void compute_gradients(DenseLayer *layer , Matrix *input_a){

    //Biases Gradient
    for(int j = 0 ; j < layer->db->cols ; j++){
        float sum = 0.0f;
        for(int i = 0 ; i < layer->delta->rows ; i++){//batch size
            sum += layer->delta->data[i * layer->delta->cols +j];
        }
        layer->db->data[j] = sum;
    }

    //Weights Gradient
    // We do a live transpose read to avoid allocating memory for input_a^T
    for(int i = 0 ; i < input_a->cols ; i++){
        for(int j = 0 ; j < layer->delta->cols; j++){
            float sum = 0.0f;
            for(int k = 0 ; k < input_a->rows; k++){
                // input_a^T[i, k] is read as input_a[k, i] from memory
                float a_val = input_a->data[k * input_a->cols + i];
                float d_val = layer->delta->data[k * layer->delta->cols + j];
                
                sum += a_val * d_val;
            }
            // Write the computed gradient to dW
            layer->dW->data[i * layer->dW->cols + j] = sum;
        }
    }

}


// Applies Gradient Descent to Update Weights And Biases
void update_weights(DenseLayer *layer,float lr){

    //Update Weights
    int w_elements = layer->weights->rows*layer->weights->cols;
    for(int i = 0 ; i < w_elements ; i++){
        layer->weights->data[i] -= lr * layer->dW->data[i]; 
    }

    //Update Biases
    int b_elements = layer->biases->rows*layer->biases->cols;
    for(int i = 0 ; i < b_elements ; i++){
        layer->biases->data[i] -= lr * layer->db->data[i]; 
    }

    transpose_matrix(layer->weights ,layer->weights_T);
}
