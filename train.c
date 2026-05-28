// train.c
// ARdL Engine: XOR Training (Arena Allocator Version)

#include <stdio.h>
#include <stdlib.h>
#include "nn_layers.h"

int main() {

    printf("=== ARdL C-Engine: XOR Training ===\n\n");

    // ==========================================
    // 1. ARENA INIT
    // ==========================================
    size_t arena_size = 1024 * 1024; // 1 MB (for test)
    MemoryArena *arena = arena_create(arena_size);

    printf("[INFO] Arena created: %zu bytes\n", arena->size);

    // ==========================================
    // 2. PARAMETERS
    // ==========================================
    int epochs = 2000;
    float lr = 0.1f;
    int batch_size = 4;

    // ==========================================
    // 3. DATASET (ARENA)
    // ==========================================
    Matrix *X = create_matrix_area(arena, batch_size, 2);
    Matrix *Y = create_matrix_area(arena, batch_size, 1);

    // XOR input
    X->data[0] = 0.0f; X->data[1] = 0.0f;
    X->data[2] = 0.0f; X->data[3] = 1.0f;
    X->data[4] = 1.0f; X->data[5] = 0.0f;
    X->data[6] = 1.0f; X->data[7] = 1.0f;

    // XOR output
    Y->data[0] = 0.0f;
    Y->data[1] = 1.0f;
    Y->data[2] = 1.0f;
    Y->data[3] = 0.0f;

    printf("[INFO] Dataset allocated\n");

    // ==========================================
    // 4. MODEL INIT
    // ==========================================
    DenseLayer *layer1 = init_dense_layer(arena, 2, 4, batch_size);
    DenseLayer *layer2 = init_dense_layer(arena, 4, 1, batch_size);

    printf("[INFO] Model initialized\n");

    // Memory check before training
    size_t mem_before = arena->offset;

    printf("[INFO] Memory before training: %zu bytes\n\n", mem_before);

    // ==========================================
    // 5. TRAINING LOOP
    // ==========================================
    printf("[INFO] Training started (%d epochs)\n\n", epochs);

    for (int epoch = 0; epoch <= epochs; epoch++) {

        // --- FORWARD ---
        forward_pass(layer1, X, 0);
        forward_pass(layer2, layer1->a, 1);

        float total_loss = 0.0f;

        // --- OUTPUT DELTA ---
        for (int i = 0; i < batch_size; i++) {
            float y_pred = layer2->a->data[i];
            float y_true = Y->data[i];

            float error = y_pred - y_true;

            layer2->delta->data[i] = error;
            total_loss += error * error;
        }

        total_loss /= batch_size;

        // --- HIDDEN DELTA ---
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < layer1->a->cols; j++) {

                float back_error = layer2->delta->data[i] *
                                   layer2->weights->data[j];

                float z_val = layer1->z->data[i * layer1->z->cols + j];

                layer1->delta->data[i * layer1->delta->cols + j] =
                    back_error * ReLu_grad(z_val);
            }
        }

        // --- BACKWARD ---
        compute_gradients(layer2, layer1->a);
        compute_gradients(layer1, X);

        // --- UPDATE ---
        update_weights(layer2, lr);
        update_weights(layer1, lr);

        // --- LOG ---
        if (epoch % 200 == 0) {
            printf("Epoch %4d | Loss: %.6f | Arena: %zu bytes\n",
                   epoch, total_loss, arena->offset);
        }
    }

    // ==========================================
    // 6. MEMORY CHECK
    // ==========================================
    size_t mem_after = arena->offset;

    printf("\n[INFO] Memory after training: %zu bytes\n", mem_after);

    if (mem_before == mem_after) {
        printf("[OK] No memory leak detected ✅\n");
    } else {
        printf("[WARNING] Memory changed! Possible leak ❌\n");
    }

    // ==========================================
    // 7. RESULTS
    // ==========================================
    printf("\n=== XOR Predictions ===\n");

    for (int i = 0; i < batch_size; i++) {
        int x1 = (int)X->data[i * 2 + 0];
        int x2 = (int)X->data[i * 2 + 1];

        float pred = layer2->a->data[i];

        printf("[%d, %d] -> %.4f\n", x1, x2, pred);
    }

    // ==========================================
    // 8. CLEANUP
    // ==========================================
    arena_free(arena);

    printf("\n[INFO] Arena freed. Shutdown complete.\n");

    return 0;
}
