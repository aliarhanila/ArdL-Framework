#include <stdio.h>
#include "ardl_model.h"

int main() {
    printf("\n=== ArdL Framework v1.0 (High-Level API) ===\n\n");

    int batch_size = 5000;
    
    // 1. Initialize Sequential Model and Arena Memory (20 MB allocation)
    SequentialModel *model = ardl_create_sequential(20, batch_size);

    // 2. Memory Space and Data Processing Pipeline
    Matrix *X = create_matrix_area(model->arena, batch_size, 2);
    Matrix *Y = create_matrix_area(model->arena, batch_size, 1);
    ardl_load_csv("kinetik_enerji_veriseti.csv", X, Y);

    float mean_X[2] = {0}, std_X[2] = {0}, min_Y = 0, max_Y = 0;
    ardl_normalize_zscore(X, mean_X, std_X);
    ardl_normalize_minmax(Y, &min_Y, &max_Y);

    // 3. Define Neural Network Architecture (Keras-like Layout)
    ardl_add_dense(model, 2, 64, TANH);
    ardl_add_dense(model, 64, 32, TANH);
    ardl_add_dense(model, 32, 16, TANH);
    ardl_add_dense(model, 16, 1, LINEAR);

    // 4. Model Training Routine (Epochs, Init_LR, Decay_Steps, Decay_Rate)
    printf("[INFO] Training Pipeline Triggered...\n");
    ardl_compile_and_fit(model, X, Y, 120000, 0.1f, 15000, 0.5f);

    printf("\n[INFO] Pipeline Completed Successfully!\n");
    
    // Save binary model structure and architecture metrics
    ardl_model_save(model, "kinetik_motor.ardl"); 

    // Export static C Header configuration for edge computing / MCU environments
    ardl_model_export_header(model, "ardl_embedded_model.h");

    // Flush allocation records and free memory blocks
    ardl_free_model(model);
    return 0;
}