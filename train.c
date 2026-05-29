#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ardl_core.h"

// =========================
// CONFIGURATION
// =========================
// Set to 1: Train the model and save the weights to disk.
// Set to 0: Skip training, load weights from disk, and run inference.
#define TRAIN_MODE 1 

// =========================
// DATASET PARSER
// =========================
void load_iris_dataset(const char *filename, Matrix *X, Matrix *Y) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("ERROR: Dataset file '%s' not found!\n", filename);
        exit(1);
    }

    char line[256];
    int row = 0;
    
    // Read line by line until X capacity is reached
    while (fgets(line, sizeof(line), file) && row < X->rows) {
        float f1, f2, f3, f4;
        char label[64];
        
        // Expected CSV Format: 5.1,3.5,1.4,0.2,Iris-setosa
        if (sscanf(line, "%f,%f,%f,%f,%s", &f1, &f2, &f3, &f4, label) == 5) {
            
            // Input Features (4 dimensions)
            X->data[row * 4 + 0] = f1;
            X->data[row * 4 + 1] = f2;
            X->data[row * 4 + 2] = f3;
            X->data[row * 4 + 3] = f4;

            // One-Hot Encoding Initialization
            Y->data[row * 3 + 0] = 0.0f;
            Y->data[row * 3 + 1] = 0.0f;
            Y->data[row * 3 + 2] = 0.0f;

            // Set the corresponding class to 1.0
            if (strstr(label, "setosa")) {
                Y->data[row * 3 + 0] = 1.0f;
            } else if (strstr(label, "versicolor")) {
                Y->data[row * 3 + 1] = 1.0f;
            } else if (strstr(label, "virginica")) {
                Y->data[row * 3 + 2] = 1.0f;
            }

            row++;
        }
    }
    fclose(file);
    printf(">> Dataset loaded successfully: %d samples read.\n\n", row);
}

// =========================
// BACKPROPAGATION HELPER
// =========================
void backprop_hidden(DenseLayer *cur, DenseLayer *next) {
    for (int i = 0; i < cur->delta->rows; i++) {
        for (int j = 0; j < cur->delta->cols; j++) {
            float sum = 0;
            for (int k = 0; k < next->delta->cols; k++) {
                float d = next->delta->data[i * next->delta->cols + k];
                float w = next->weights->data[j * next->weights->cols + k];
                sum += d * w;
            }
            float z = cur->z->data[i * cur->z->cols + j];
            
            // Using LeakyReLU gradient for hidden layers
            cur->delta->data[i * cur->delta->cols + j] = sum * LeakyReLu_grad(z);
        }
    }
}

// =========================
// MAIN ROUTINE
// =========================
int main() {
    printf("=== IRIS DATASET MULTI-CLASS CLASSIFICATION ===\n\n");

    MemoryArena *arena = arena_create(1024 * 1024); // Allocate 1 MB memory

    // Full-batch training (Iris dataset has 150 rows)
    int batch = 150; 
    
    // Memory allocation for Inputs (4) and Targets (3)
    Matrix *X = create_matrix_area(arena, batch, 4);
    Matrix *Y = create_matrix_area(arena, batch, 3);

    // Load data from CSV
    load_iris_dataset("data.csv", X, Y);

    // MODEL ARCHITECTURE (4 Input -> 16 -> 8 -> 3 Output)
    DenseLayer *l1 = init_dense_layer(arena, 4, 16, batch, 0); // 0: LeakyReLU
    DenseLayer *l2 = init_dense_layer(arena, 16, 8, batch, 0); // 0: LeakyReLU
    DenseLayer *l3 = init_dense_layer(arena, 8, 3, batch, 3);  // 3: Softmax

    if (TRAIN_MODE == 1) {
        printf("=== MODE 1: TRAINING AND EXPORT ===\n\n");
        
        int epochs = 5000;
        float lr = 0.01f;

        for (int e = 0; e <= epochs; e++) {
            // 1. Forward Pass
            forward_pass(l1, X);
            forward_pass(l2, l1->a);
            forward_pass(l3, l2->a);

            // 2. Output Error Calculation (Softmax + Categorical Cross-Entropy shortcut: p - y)
            int total_elements = batch * 3;
            for (int i = 0; i < total_elements; i++) {
                l3->delta->data[i] = (l3->a->data[i] - Y->data[i]);
            }

            float loss = categorical_crossentropy(Y, l3->a);

            // 3. Backward Pass (Backpropagation)
            backprop_hidden(l2, l3);
            backprop_hidden(l1, l2);

            compute_gradients(l3, l2->a);
            compute_gradients(l2, l1->a);
            compute_gradients(l1, X);

            // 4. Update Weights
            update_weights(l3, lr);
            update_weights(l2, lr);
            update_weights(l1, lr);

            // Display metrics every 500 epochs
            if (e % 500 == 0) {
                int correct = 0;
                for (int i = 0; i < batch; i++) {
                    int true_class = -1, pred_class = -1;
                    float max_p = -1.0f;

                    for (int j = 0; j < 3; j++) {
                        if (Y->data[i * 3 + j] == 1.0f) true_class = j;
                        if (l3->a->data[i * 3 + j] > max_p) {
                            max_p = l3->a->data[i * 3 + j];
                            pred_class = j;
                        }
                    }
                    if (true_class == pred_class) correct++;
                }
                float accuracy = (float)correct / batch * 100.0f;
                printf("Epoch %4d | Loss: %.4f | Accuracy: %%%.1f\n", e, loss, accuracy);
            }
        }

        // Save trained weights to binary files
        printf("\n=== EXPORTING TRAINED MODEL ===\n");
        save_layer(l1, "model_l1.bin");
        save_layer(l2, "model_l2.bin");
        save_layer(l3, "model_l3.bin");

    } else {
        // INFERENCE MODE (TRAIN_MODE == 0)
        printf("=== MODE 0: IMPORT AND INFERENCE ===\n\n");
        
        // Skip training and overwrite random initialization with saved weights
        load_layer(l1, "model_l1.bin");
        load_layer(l2, "model_l2.bin");
        load_layer(l3, "model_l3.bin");
    }

    // ==========================================
    // MODEL TESTING (INFERENCE)
    // Run forward pass once to test specific samples
    // ==========================================
    printf("\n=== RUNNING INFERENCE TEST ===\n");
    
    forward_pass(l1, X);
    forward_pass(l2, l1->a);
    forward_pass(l3, l2->a);

    // Pick 3 specific rows to test (one from each class)
    int test_indices[] = {0, 60, 120}; 
    const char *classes[] = {"Setosa", "Versicolor", "Virginica"};

    for(int k = 0; k < 3; k++) {
        int i = test_indices[k];
        printf("\nSample [%d] Features (cm): %.1f, %.1f, %.1f, %.1f\n", i, 
               X->data[i*4+0], X->data[i*4+1], X->data[i*4+2], X->data[i*4+3]);
        
        printf("Model Predictions:\n");
        printf("  - %-12s: %5.2f%%\n", classes[0], l3->a->data[i*3+0] * 100.0f);
        printf("  - %-12s: %5.2f%%\n", classes[1], l3->a->data[i*3+1] * 100.0f);
        printf("  - %-12s: %5.2f%%\n", classes[2], l3->a->data[i*3+2] * 100.0f);
    }

    // Clean up memory
    arena_free(arena);
    printf("\n>> Process completed.\n");
    return 0;
}