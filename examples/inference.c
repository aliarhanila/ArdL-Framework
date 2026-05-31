// inference.c
// Real-Time Inference and Testing Engine
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ardl_model.h"

int main() {
    printf("\n=== ArdL Inference (Real-Time Kinetic Engine) ===\n\n");

    int batch_size = 5000; 
    SequentialModel *model = ardl_create_sequential(20, batch_size);

    /*
    #----------------------------
    # 1. Scaler Setup
    #---------------------------
    */
    Matrix *X = create_matrix_area(model->arena, batch_size, 2);
    Matrix *Y = create_matrix_area(model->arena, batch_size, 1);
    
    FILE *f_check = fopen("kinetik_enerji_veriseti.csv", "r");
    if(!f_check) {
        printf("Error: 'kinetik_enerji_veriseti.csv' not found. Normalization coefficients cannot be read.\n");
        return 1;
    }
    fclose(f_check);

    ardl_load_csv("kinetik_enerji_veriseti.csv", X, Y);

    float mean_X[2] = {0}, std_X[2] = {0}, min_Y = 0, max_Y = 0;
    ardl_normalize_zscore(X, mean_X, std_X);
    ardl_normalize_minmax(Y, &min_Y, &max_Y);

    /*
    #----------------------------
    # 2. Load Unified Weights
    #---------------------------
    */
    printf("[INFO] Fetching trained parameters from disk...\n");
    ardl_model_load(model, "kinetik_motor.ardl");

    printf("\n[INFO] Engine running. Type '0' in Mass selection to shutdown the system.\n");
    printf("----------------------------------------------------------\n");

    /*
    #----------------------------
    # 3. Interactive Terminal Loop
    #---------------------------
    */
    float input_m = 0.0f, input_v = 0.0f;
    while (1) {
        printf("\n> Enter Mass (kg) : ");
        if (scanf("%f", &input_m) != 1) break; 
        
        if (input_m <= 0.0f) {
            printf("\nShutting down ArdL Inference Engine...\n");
            break;
        }

        printf("> Enter Velocity (m/s) : ");
        if (scanf("%f", &input_v) != 1) break;

        // Execute Real-Time Prediction Pass
        float predicted_E = ardl_predict_single(model, input_m, input_v, 
                                                mean_X[0], std_X[0], 
                                                mean_X[1], std_X[1], 
                                                min_Y, max_Y);
        
        if (predicted_E < 0) predicted_E = 0.0f;

        float actual_E = 0.5f * input_m * powf(input_v, 2.0f);
        float error_margin = fabsf(actual_E - predicted_E) / (actual_E + 1e-8f) * 100.0f;

        printf("\n=== COLLISION EVALUATION ===\n");
        printf("  Theoretical Physics (1/2mv^2)  : %.0f Joule\n", actual_E);
        printf("  ArdL Network Prediction        : %.0f Joule\n", predicted_E);
        printf("  Error Margin                   : %%%.2f\n", error_margin);
        printf("==============================\n");
    }

    ardl_free_model(model);
    return 0;
}