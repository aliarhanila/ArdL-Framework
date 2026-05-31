// ardl_model.h
// Keras-like High-Level API and Architecture Manager
#ifndef ARDL_MODEL_H
#define ARDL_MODEL_H

#include "ardl_core_thread.h"

// --- User-Friendly Activation Macros ---
#define LINEAR 0
#define RELU 1
#define LEAKY_RELU 2
#define TANH 3
#define SIGMOID 4
#define SOFTMAX 5

#define MAX_LAYERS 20

/*
#----------------------------
# Core Model Structure
#---------------------------
*/
typedef struct {
    MemoryArena *arena;
    DenseLayer *layers[MAX_LAYERS];
    int layer_count;
    int batch_size;
} SequentialModel;

/*
#----------------------------
# API Functions
#---------------------------
*/
SequentialModel* ardl_create_sequential(size_t arena_mb, int batch_size);
void ardl_add_dense(SequentialModel *model, int input_dim, int output_dim, int activation_type);
void ardl_compile_and_fit(SequentialModel *model, Matrix *X, Matrix *Y, int epochs, float initial_lr, int decay_steps, float decay_rate);

/*
#----------------------------
# Data & Metric Helpers
#---------------------------
*/
void ardl_load_csv(const char *filename, Matrix *X, Matrix *Y);
float ardl_calculate_r2(Matrix *y_true, Matrix *y_pred);
void ardl_normalize_zscore(Matrix *X, float *out_mean, float *out_std);
void ardl_normalize_minmax(Matrix *Y, float *out_min, float *out_max);

/*
#----------------------------
# Memory Management
#---------------------------
*/
void ardl_free_model(SequentialModel *model);

/*
#----------------------------
# Serialization & Export
#---------------------------
*/
void ardl_model_save(SequentialModel *model, const char *filename);
void ardl_model_load(SequentialModel *model, const char *filename);
void ardl_model_export_header(SequentialModel *model, const char *filename);

/*
#----------------------------
# Real-Time Inference
#---------------------------
*/
float ardl_predict_single(SequentialModel *model, float input_m, float input_v, float mean_m, float std_m, float mean_v, float std_v, float min_Y, float max_Y);

#endif