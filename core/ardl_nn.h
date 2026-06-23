#ifndef ARDL_NN_H
#define ARDL_NN_H

#include "ardl_memory.h"
#include "ardl_math.h"
#include "ardl_loss.h"
#include "ardl_activations.h"
#include <stdio.h> // FILE 



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

// --- Initilazitions ---
DenseLayer* init_dense_layer(MemoryArena *arena, int input_dim , int output_dim, int batch_size, int activation_type);
Conv2DLayer* init_conv2d_layer(MemoryArena *arena, int in_channels, int out_channels, int kernel_size, int stride, int padding,int input_height, int input_width, int batch_size, int activation_type);
MaxPool2DLayer* init_maxpool2d_layer(MemoryArena *arena, int channels, int pool_size, int stride, int padding, int input_height, int input_width, int batch_size);


// --- Forward / Backward ---
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



// --- Core Serialization ---
void core_serialize_weights(DenseLayer *layer, FILE *file);
void core_deserialize_weights(DenseLayer *layer, FILE *file);



#endif
