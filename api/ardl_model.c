// ardl_model.c
// High-Level API Implementation and Training Loop
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h> 
#include "ardl_model.h"

/*
#----------------------------
# Model Initialization
#---------------------------
*/
SequentialModel* ardl_create_sequential(size_t arena_mb, int batch_size) {
    SequentialModel *model = malloc(sizeof(SequentialModel));
    model->arena = arena_create(1024 * 1024 * arena_mb);
    model->layer_count = 0;
    model->batch_size = batch_size;
    return model;
}

void ardl_add_dense(SequentialModel *model, int input_dim, int output_dim, int activation_type) {
    if (model->layer_count >= MAX_LAYERS) {
        printf("Error (Model): Maximum layer limit reached!\n");
        return;
    }
    model->layers[model->layer_count] = init_dense_layer(model->arena, input_dim, output_dim, model->batch_size, activation_type);
    model->layer_count++;
}

/*
#----------------------------
# Dynamic Backpropagation
#---------------------------
*/
static void backprop_hidden_dynamic(DenseLayer *cur, DenseLayer *next) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cur->delta->rows; i++) {
        for (int j = 0; j < cur->delta->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < next->delta->cols; k++) {
                sum += next->delta->data[i * next->delta->cols + k] * next->weights->data[j * next->weights->cols + k];
            }
            float z = cur->z->data[i * cur->z->cols + j];
            float grad = 1.0f; 
            
            if (cur->activation_type == RELU)            grad = ReLu_grad(z);
            else if (cur->activation_type == LEAKY_RELU) grad = LeakyReLu_grad(z);
            else if (cur->activation_type == TANH)       grad = Tanh_grad(z);
            else if (cur->activation_type == SIGMOID)    grad = sigmoid_grad(z);

            cur->delta->data[i * cur->delta->cols + j] = sum * grad;
        }
    }
}

/*
#----------------------------
# Fit Engine (Training Loop)
#---------------------------
*/
void ardl_compile_and_fit(SequentialModel *model, Matrix *X, Matrix *Y, int epochs, float initial_lr, int decay_steps, float decay_rate) {
    float lr = initial_lr;
    int lc = model->layer_count;
    
    for (int e = 0; e <= epochs; e++) {
        // Learning Rate Decay
        if (e > 0 && e % decay_steps == 0) lr *= decay_rate;

        // Forward Pass
        for (int i = 0; i < lc; i++) {
            Matrix *input = (i == 0) ? X : model->layers[i-1]->a;
            forward_pass(model->layers[i], input);
        }

        // Output Layer Error Calculation
        DenseLayer *last_layer = model->layers[lc - 1];
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < model->batch_size; i++) {
            last_layer->delta->data[i] = last_layer->a->data[i] - Y->data[i];
        }

        // Backpropagation Chain
        for (int i = lc - 2; i >= 0; i--) {
            backprop_hidden_dynamic(model->layers[i], model->layers[i+1]);
        }

        // Gradient Computation & Weight Update
        for (int i = lc - 1; i >= 0; i--) {
            Matrix *input = (i == 0) ? X : model->layers[i-1]->a;
            compute_gradients(model->layers[i], input);
            update_weights(model->layers[i], lr);
        }

        // Metrics Logging (Using MSLE for scale-invariant accuracy tracking)
        if (e % (epochs / 20) == 0) {
            float loss = msle_loss(Y, last_layer->a);
            float r2 = ardl_calculate_r2(Y, last_layer->a);
            printf("Epoch: %6d | LR: %.5f | MSLE Loss: %.6f | R2: %%%05.2f\n", e, lr, loss, r2 * 100.0f);
        }
    }
}

/*
#----------------------------
# Data Processing Utilities
#---------------------------
*/
void ardl_load_csv(const char *filename, Matrix *X, Matrix *Y) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error (I/O): File %s not found!\n", filename);
        exit(1);
    }
    
    char line[256]; 
    int row = 0;
    
    if (fgets(line, sizeof(line), file)) { /* skip header */ }
    
    while (fgets(line, sizeof(line), file) && row < X->rows) {
        float m, v, e;
        if (sscanf(line, "%f,%f,%f", &m, &v, &e) == 3) {
            X->data[row * 2 + 0] = m; 
            X->data[row * 2 + 1] = v; 
            Y->data[row] = e;
            row++;
        }
    }
    fclose(file);
}

float ardl_calculate_r2(Matrix *y_true, Matrix *y_pred) {
    float ss_res = 0.0f, ss_tot = 0.0f, y_mean = 0.0f;
    for (int i = 0; i < y_true->rows; i++) y_mean += y_true->data[i];
    y_mean /= y_true->rows;
    
    for (int i = 0; i < y_true->rows; i++) {
        ss_res += powf(y_true->data[i] - y_pred->data[i], 2);
        ss_tot += powf(y_true->data[i] - y_mean, 2);
    }
    return (ss_tot == 0.0f) ? 1.0f : 1.0f - (ss_res / ss_tot);
}

void ardl_normalize_zscore(Matrix *X, float *out_mean, float *out_std) {
    int batch = X->rows;
    for (int i = 0; i < batch; i++) { 
        out_mean[0] += X->data[i*2+0]; 
        out_mean[1] += X->data[i*2+1]; 
    }
    out_mean[0] /= batch; 
    out_mean[1] /= batch;
    
    for (int i = 0; i < batch; i++) { 
        out_std[0] += powf(X->data[i*2+0] - out_mean[0], 2); 
        out_std[1] += powf(X->data[i*2+1] - out_mean[1], 2); 
    }
    out_std[0] = sqrtf(out_std[0]/batch); 
    out_std[1] = sqrtf(out_std[1]/batch);
    
    for (int i = 0; i < batch; i++) {
        X->data[i*2+0] = (X->data[i*2+0] - out_mean[0]) / (out_std[0] + 1e-8f);
        X->data[i*2+1] = (X->data[i*2+1] - out_mean[1]) / (out_std[1] + 1e-8f);
    }
}

void ardl_normalize_minmax(Matrix *Y, float *out_min, float *out_max) {
    int batch = Y->rows;
    *out_min = Y->data[0]; 
    *out_max = Y->data[0];
    
    for (int i = 1; i < batch; i++) {
        if (Y->data[i] < *out_min) *out_min = Y->data[i];
        if (Y->data[i] > *out_max) *out_max = Y->data[i];
    }
    
    for (int i = 0; i < batch; i++) {
        Y->data[i] = (Y->data[i] - *out_min) / (*out_max - *out_min + 1e-8f);
    }
}

/*
#----------------------------
# Memory Management
#---------------------------
*/
void ardl_free_model(SequentialModel *model) {
    if (model) { 
        arena_free(model->arena); 
        free(model); 
    }
}

/*
#----------------------------
# Serialization & Export
#---------------------------
*/
void ardl_model_save(SequentialModel *model, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) return;
    
    char magic[4] = "ARDL";
    fwrite(magic, sizeof(char), 4, file);
    fwrite(&model->layer_count, sizeof(int), 1, file);
    
    for (int i = 0; i < model->layer_count; i++) {
        DenseLayer *l = model->layers[i];
        int in_dim = l->weights->rows; 
        int out_dim = l->weights->cols; 
        int act = l->activation_type;
        
        fwrite(&in_dim, sizeof(int), 1, file); 
        fwrite(&out_dim, sizeof(int), 1, file); 
        fwrite(&act, sizeof(int), 1, file);
        
        core_serialize_weights(l, file);
    }
    fclose(file);
    printf("[INFO] Model architecture and weights successfully saved: %s\n", filename);
}

void ardl_model_load(SequentialModel *model, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error (I/O): Model file %s not found!\n", filename);
        exit(1);
    }
    
    char magic[4]; 
    fread(magic, sizeof(char), 4, file);
    if (strncmp(magic, "ARDL", 4) != 0) { 
        printf("Error (I/O): Invalid model format!\n");
        fclose(file); 
        exit(1); 
    }
    
    int file_layer_count; 
    fread(&file_layer_count, sizeof(int), 1, file);
    model->layer_count = 0;
    
    for (int i = 0; i < file_layer_count; i++) {
        int in_dim, out_dim, act;
        fread(&in_dim, sizeof(int), 1, file); 
        fread(&out_dim, sizeof(int), 1, file); 
        fread(&act, sizeof(int), 1, file);
        
        ardl_add_dense(model, in_dim, out_dim, act);
        core_deserialize_weights(model->layers[i], file);
    }
    fclose(file);
    printf("[INFO] %d-layer model successfully built and loaded from: %s\n", file_layer_count, filename);
}

void ardl_model_export_header(SequentialModel *model, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) return;
    
    fprintf(file, "/* ARDL STATIC MODEL CONFIGURATION */\n");
    fprintf(file, "#ifndef ARDL_EMBEDDED_MODEL_H\n#define ARDL_EMBEDDED_MODEL_H\n\n");
    fprintf(file, "#define ARDL_MODEL_LAYER_COUNT %d\n\n", model->layer_count);
    
    for (int l = 0; l < model->layer_count; l++) {
        DenseLayer *layer = model->layers[l];
        int in_dim = layer->weights->rows; 
        int out_dim = layer->weights->cols; 
        int act = layer->activation_type;
        
        fprintf(file, "/* LAYER %d */\n", l+1);
        fprintf(file, "#define ARDL_L%d_IN_DIM %d\n", l+1, in_dim);
        fprintf(file, "#define ARDL_L%d_OUT_DIM %d\n", l+1, out_dim);
        fprintf(file, "#define ARDL_L%d_ACTIVATION %d\n\n", l+1, act);
        
        fprintf(file, "const float ardl_l%d_weights[%d] = {\n    ", l+1, in_dim * out_dim);
        for (int i = 0; i < in_dim * out_dim; i++) {
            fprintf(file, "%.8ff%s", layer->weights->data[i], (i < (in_dim * out_dim) - 1) ? ", " : "");
            if ((i + 1) % 8 == 0 && i < (in_dim * out_dim) - 1) fprintf(file, "\n    ");
        }
        
        fprintf(file, "\n};\n\nconst float ardl_l%d_biases[%d] = {\n    ", l+1, out_dim);
        for (int i = 0; i < out_dim; i++) {
            fprintf(file, "%.8ff%s", layer->biases->data[i], (i < out_dim - 1) ? ", " : "");
            if ((i + 1) % 8 == 0 && i < out_dim - 1) fprintf(file, "\n    ");
        }
        fprintf(file, "\n};\n\n");
    }
    
    fprintf(file, "#endif\n"); 
    fclose(file);
    printf("[INFO] Model exported as static C Header for Embedded Deployment: %s\n", filename);
}

/*
#----------------------------
# Real-Time Inference
#---------------------------
*/
float ardl_predict_single(SequentialModel *model, float input_m, float input_v, float mean_m, float std_m, float mean_v, float std_v, float min_Y, float max_Y) {
    if (!model || model->layer_count == 0) return 0.0f;
    
    // Z-Score Scaling
    float norm_m = (input_m - mean_m) / (std_m + 1e-8f);
    float norm_v = (input_v - mean_v) / (std_v + 1e-8f);
    
    // Stack-allocated matrix for memory safety
    Matrix X_test_struct; 
    float x_data[2] = {norm_m, norm_v};
    X_test_struct.rows = 1; 
    X_test_struct.cols = 2; 
    X_test_struct.data = x_data;
    Matrix *X_test = &X_test_struct;
    
    int original_batch = model->batch_size;
    
    for (int i = 0; i < model->layer_count; i++) {
        model->layers[i]->z->rows = 1; 
        model->layers[i]->a->rows = 1;
        Matrix *in = (i == 0) ? X_test : model->layers[i-1]->a;
        forward_pass(model->layers[i], in);
    }
    
    float norm_out = model->layers[model->layer_count - 1]->a->data[0];
    float final_joule = (norm_out * (max_Y - min_Y)) + min_Y;
    
    // Restore architecture properties
    for (int i = 0; i < model->layer_count; i++) {
        model->layers[i]->z->rows = original_batch; 
        model->layers[i]->a->rows = original_batch;
    }
    
    return final_joule;
}