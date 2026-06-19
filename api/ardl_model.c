// ardl_model.c
// High-Level API Implementation and Training Loop
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h> 
#include "ardl_model.h"

/*
#-----------------------------------
# Model Initialization & Layers
#-----------------------------------
*/
SequentialModel* ardl_create_sequential(size_t arena_mb, int batch_size) {
    SequentialModel *model = malloc(sizeof(SequentialModel));
    model->arena = arena_create(1024 * 1024 * arena_mb);
    model->layer_count = 0;
    model->batch_size = batch_size;
    return model;
}

void ardl_add_dense(SequentialModel *model, int input_dim, int output_dim, int activation_type) {
    if (model->layer_count >= MAX_LAYERS) return;
    DenseLayer *layer = init_dense_layer(model->arena, input_dim, output_dim, model->batch_size, activation_type);
    
    model->layers[model->layer_count].type = L_DENSE;
    model->layers[model->layer_count].instance = layer;
    model->layers[model->layer_count].a = layer->a;
    model->layers[model->layer_count].delta = layer->delta;
    model->layer_count++;
}

void ardl_add_conv2d(SequentialModel *model, int in_channels, int out_channels, int kernel_size, int stride, int padding, int input_height, int input_width, int activation_type) {
    if (model->layer_count >= MAX_LAYERS) return;
    Conv2DLayer *layer = init_conv2d_layer(model->arena, in_channels, out_channels, kernel_size, stride, padding, input_height, input_width, model->batch_size, activation_type);
    
    model->layers[model->layer_count].type = L_CONV2D;
    model->layers[model->layer_count].instance = layer;
    model->layers[model->layer_count].a = layer->a;
    model->layers[model->layer_count].delta = layer->delta;
    model->layer_count++;
}

void ardl_add_maxpool2d(SequentialModel *model, int channels, int pool_size, int stride, int padding, int input_height, int input_width) {
    if (model->layer_count >= MAX_LAYERS) return;
    MaxPool2DLayer *layer = init_maxpool2d_layer(model->arena, channels, pool_size, stride, padding, input_height, input_width, model->batch_size);
    
    model->layers[model->layer_count].type = L_MAXPOOL;
    model->layers[model->layer_count].instance = layer;
    model->layers[model->layer_count].a = layer->a;
    model->layers[model->layer_count].delta = layer->delta;
    model->layer_count++;
}

/*
#-----------------------------------
# Conv2D Weight Update Helper
#-----------------------------------
*/
static void update_weights_conv2d(Conv2DLayer *layer, float lr) {
    int k_elements = layer->out_channels * (layer->in_channels * layer->kernel_size * layer->kernel_size);
    #pragma omp parallel for
    for(int i = 0; i < k_elements; i++){
        layer->kernels->data[i] -= lr * layer->dK->data[i]; 
    }
    #pragma omp parallel for
    for(int i = 0; i < layer->out_channels; i++){
        layer->biases->data[i] -= lr * layer->db->data[i]; 
    }
}

/*
#-----------------------------------
# Universal Fit Engine (Training Loop)
#-----------------------------------
*/
void ardl_compile_and_fit(SequentialModel *model, Matrix *X, Matrix *Y, int epochs, float initial_lr, int decay_steps, float decay_rate) {
    float lr = initial_lr;
    int lc = model->layer_count;
    
    for (int e = 0; e <= epochs; e++) {
        if (e > 0 && e % decay_steps == 0) lr *= decay_rate;

        // 1. FORWARD PASS
        for (int i = 0; i < lc; i++) {
            Matrix *input = (i == 0) ? X : model->layers[i-1].a;
            int type = model->layers[i].type;
            
            if (type == L_DENSE) forward_pass((DenseLayer*)model->layers[i].instance, input);
            else if (type == L_CONV2D) forward_conv2d((Conv2DLayer*)model->layers[i].instance, input);
            else if (type == L_MAXPOOL) forward_maxpool2d((MaxPool2DLayer*)model->layers[i].instance, input);
        }

        // 2. CALCULATE OUTPUT ERROR
        NetLayer *last = &model->layers[lc - 1];
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < model->batch_size * last->delta->cols; i++) {
            last->delta->data[i] = last->a->data[i] - Y->data[i];
        }

        // 3. BACKPROPAGATION CHAIN & WEIGHT UPDATE
        for (int i = lc - 1; i >= 0; i--) {
            NetLayer *curr = &model->layers[i];
            NetLayer *prev = (i > 0) ? &model->layers[i-1] : NULL;
            Matrix *input = (i == 0) ? X : prev->a;

            // A. Calculate Gradients and Pass Error to Previous Layer (prev->delta)
            if (curr->type == L_DENSE) {
                compute_gradients((DenseLayer*)curr->instance, input);
                update_weights((DenseLayer*)curr->instance, lr);
                
                // Pass error to previous layer (Dense Logic)
                if (prev) {
                    DenseLayer *dl = (DenseLayer*)curr->instance;
                    #pragma omp parallel for collapse(2)
                    for (int r = 0; r < prev->delta->rows; r++) {
                        for (int c = 0; c < prev->delta->cols; c++) {
                            float sum = 0.0f;
                            for (int k = 0; k < dl->delta->cols; k++) {
                                sum += dl->delta->data[r * dl->delta->cols + k] * dl->weights->data[c * dl->weights->cols + k];
                            }
                            prev->delta->data[r * prev->delta->cols + c] = sum;
                        }
                    }
                }
            } 
            else if (curr->type == L_CONV2D) {
                compute_gradients_conv2d((Conv2DLayer*)curr->instance, input, (prev) ? prev->delta : NULL);
                update_weights_conv2d((Conv2DLayer*)curr->instance, lr);
            } 
            else if (curr->type == L_MAXPOOL) {
                backward_maxpool2d((MaxPool2DLayer*)curr->instance, (prev) ? prev->delta : NULL);
            }

            // B. Chain Rule: Apply Previous Layer's Activation Derivative (f'(z))
            if (prev && (prev->type == L_DENSE || prev->type == L_CONV2D)) {
                int act_type = (prev->type == L_DENSE) ? ((DenseLayer*)prev->instance)->activation_type : ((Conv2DLayer*)prev->instance)->activation_type;
                Matrix *prev_z = (prev->type == L_DENSE) ? ((DenseLayer*)prev->instance)->z : ((Conv2DLayer*)prev->instance)->z;
                
                if (act_type != LINEAR && act_type != SOFTMAX) {
                    #pragma omp parallel for
                    for (int idx = 0; idx < prev->delta->rows * prev->delta->cols; idx++) {
                        float z_val = prev_z->data[idx];
                        float grad = 1.0f;
                        if (act_type == RELU)            grad = ReLu_grad(z_val);
                        else if (act_type == LEAKY_RELU) grad = LeakyReLu_grad(z_val);
                        else if (act_type == TANH)       grad = Tanh_grad(z_val);
                        else if (act_type == SIGMOID)    grad = sigmoid_grad(z_val);
                        
                        prev->delta->data[idx] *= grad; // Multiply by derivative to bound the error
                    }
                }
            }
        }

        // Metrics Reporting (e.g., MSLE/MSE)
        if (e % (epochs / 10) == 0) {
            float loss = mse_loss(Y, last->a);
            printf("Epoch: %6d | LR: %.5f | MSE Loss: %.6f\n", e, lr, loss);
        }
    }
}

/*
#-----------------------------------
# Model Summary & Profiling
#-----------------------------------
*/
void ardl_model_summary(SequentialModel *model) {
    if (!model) return;

    printf("\n=================================================================\n");
    printf(" %-15s %-15s %-15s\n", "Layer (Type)", "Output Shape", "Activation");
    printf("=================================================================\n");

    for (int i = 0; i < model->layer_count; i++) {
        NetLayer *layer = &model->layers[i];
        
        if (layer->type == L_DENSE) {
            DenseLayer *dl = (DenseLayer*)layer->instance;
            char act_name[15];
            if(dl->activation_type == RELU) strcpy(act_name, "ReLU");
            else if(dl->activation_type == SIGMOID) strcpy(act_name, "Sigmoid");
            else if(dl->activation_type == TANH) strcpy(act_name, "Tanh");
            else if(dl->activation_type == LINEAR) strcpy(act_name, "Linear");
            else strcpy(act_name, "Other");

            printf(" %-15s (None, %-7d) %-15s\n", "Dense", dl->weights->cols, act_name);
        } 
        else if (layer->type == L_CONV2D) {
            Conv2DLayer *cl = (Conv2DLayer*)layer->instance;
            char act_name[15];
            if(cl->activation_type == RELU) strcpy(act_name, "ReLU");
            else if(cl->activation_type == TANH) strcpy(act_name, "Tanh");
            else strcpy(act_name, "Linear");

            // Output format: (Channels, Height, Width)
            char shape_str[30];
            sprintf(shape_str, "(%d, %d, %d)", cl->out_channels, cl->output_height, cl->output_width);
            printf(" %-15s %-15s %-15s\n", "Conv2D", shape_str, act_name);
        }
        else if (layer->type == L_MAXPOOL) {
            MaxPool2DLayer *mpl = (MaxPool2DLayer*)layer->instance;
            char shape_str[30];
            sprintf(shape_str, "(%d, %d, %d)", mpl->channels, mpl->output_height, mpl->output_width);
            printf(" %-15s %-15s %-15s\n", "MaxPool2D", shape_str, "-");
        }
    }
    
    printf("=================================================================\n");
    
    // Memory Calculations (O(1) complexity thanks to Arena!)
    size_t used_bytes = model->arena->offset;
    float used_kb = (float)used_bytes / 1024.0f;
    float used_mb = used_kb / 1024.0f;
    float total_mb = (float)model->arena->size / (1024.0f * 1024.0f);

    printf(" Total Reserved Arena      : %.2f MB\n", total_mb);
    printf(" Actual Memory Used        : %zu Bytes (%.2f KB / %.4f MB)\n", used_bytes, used_kb, used_mb);
    printf("=================================================================\n\n");
}

void ardl_free_model(SequentialModel *model) {
    if (model) { arena_free(model->arena); free(model); }
}
