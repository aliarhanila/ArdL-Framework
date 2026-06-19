// ardl_model.h
// Keras-like High-Level API and Architecture Manager
#ifndef ARDL_MODEL_H
#define ARDL_MODEL_H

#include "../core/ardl_core_thread.h"

// --- User-Friendly Macros ---
#define L_DENSE 0
#define L_CONV2D 1
#define L_MAXPOOL 2

#define LINEAR 0
#define RELU 1
#define LEAKY_RELU 2
#define TANH 3
#define SIGMOID 4
#define SOFTMAX 5

#define MAX_LAYERS 20

/*
#-----------------------------------
# Core Model Structure (Polymorphic)
#-----------------------------------
*/
typedef struct {
    int type;           // L_DENSE, L_CONV2D, or L_MAXPOOL
    void *instance;     // Pointer to the actual layer struct
    Matrix *a;          // Pointer to output activations (for easy chaining)
    Matrix *delta;      // Pointer to gradients (for easy backprop)
} NetLayer;

typedef struct {
    MemoryArena *arena;
    NetLayer layers[MAX_LAYERS];
    int layer_count;
    int batch_size;
} SequentialModel;

/*
#-----------------------------------
# API Functions
#-----------------------------------
*/
SequentialModel* ardl_create_sequential(size_t arena_mb, int batch_size);

// Katman Ekleme Fonksiyonları
void ardl_add_dense(SequentialModel *model, int input_dim, int output_dim, int activation_type);
void ardl_add_conv2d(SequentialModel *model, int in_channels, int out_channels, int kernel_size, int stride, int padding, int input_height, int input_width, int activation_type);
void ardl_add_maxpool2d(SequentialModel *model, int channels, int pool_size, int stride, int padding, int input_height, int input_width);

// Eğitim Motoru
void ardl_compile_and_fit(SequentialModel *model, Matrix *X, Matrix *Y, int epochs, float initial_lr, int decay_steps, float decay_rate);

// Veri & Metrik Yardımcıları
void ardl_load_csv(const char *filename, Matrix *X, Matrix *Y);
float ardl_calculate_r2(Matrix *y_true, Matrix *y_pred);
void ardl_normalize_zscore(Matrix *X, float *out_mean, float *out_std);
void ardl_normalize_minmax(Matrix *Y, float *out_min, float *out_max);
void ardl_model_summary(SequentialModel *model);
// Bellek Temizliği
void ardl_free_model(SequentialModel *model);

#endif