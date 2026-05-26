// train.c
// ARdL Engine: XOR Gate Training 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn_layers.h"

int main() {
    printf("--- ARdL C-Engine: XOR Training Starting ---\n\n");

    // 1. Network and Training Parameters
    int epochs = 2000;
    float lr = 0.1f; // Learning Rate
    int batch_size = 4; // 4 different states for XOR

    // 2. Create XOR Dataset
    // Inputs (X): [0,0], [0,1], [1,0], [1,1]
    Matrix *X = create_matrix(batch_size, 2);
    X->data[0] = 0.0f; X->data[1] = 0.0f;
    X->data[2] = 0.0f; X->data[3] = 1.0f;
    X->data[4] = 1.0f; X->data[5] = 0.0f;
    X->data[6] = 1.0f; X->data[7] = 1.0f;

    // Ground Truth / Targets (Y): [0], [1], [1], [0]
    Matrix *Y = create_matrix(batch_size, 1);
    Y->data[0] = 0.0f; 
    Y->data[1] = 1.0f; 
    Y->data[2] = 1.0f; 
    Y->data[3] = 0.0f;

    // 3. Initialize Layers (Architecture: 2 Inputs -> 4 Hidden Neurons -> 1 Output)
    printf("[INFO] Allocating memory...\n");
    DenseLayer *layer1 = init_dense_layer(2, 4, batch_size); // Hidden Layer
    DenseLayer *layer2 = init_dense_layer(4, 1, batch_size); // Output Layer

    printf("[INFO] Training Loop Starting (%d Epochs)...\n\n", epochs);

    // ==========================================
    // 4. THE TRAINING LOOP
    // ==========================================
    for (int epoch = 0; epoch <= epochs; epoch++) {
        
        // --- A. FORWARD PASS ---
        forward_pass(layer1, X, 0);        // Layer 1 (Input: X, Activation: 0 = ReLU)
        forward_pass(layer2, layer1->a, 1); // Layer 2 (Input: layer1's output, Activation: 1 = Sigmoid)

        // Loss Calculation (MSE - Mean Squared Error, just for logging)
        float total_loss = 0.0f;
        
        // --- B. ERROR (DELTA) CALCULATION ---
        
        // 1. Output Layer Delta (layer2): Delta = (Y_pred - Y_true)
        for (int i = 0; i < batch_size; i++) {
            float y_pred = layer2->a->data[i];
            float y_true = Y->data[i];
            
            float error = y_pred - y_true;
            layer2->delta->data[i] = error; 
            
            total_loss += (error * error); // Squaring and summing for MSE
        }
        total_loss /= batch_size;

        // 2. Hidden Layer Delta (layer1): Delta = (layer2_Delta * W2^T) * ReLU_grad(Z1)
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < layer1->a->cols; j++) { // cols = 4 neurons
                
                // Multiply the error from the upper layer (layer2) by the weights (W)
                // Since layer2->weights is (4x1), it is read as [j * 1 + 0]
                float back_error = layer2->delta->data[i] * layer2->weights->data[j];
                
                // Multiply by the derivative of its own Z value (Chain Rule)
                float z_val = layer1->z->data[i * layer1->z->cols + j];
                layer1->delta->data[i * layer1->delta->cols + j] = back_error * ReLu_grad(z_val);
            }
        }

        // --- C. BACKWARD PASS ---
        // Now that deltas are ready, calculate gradients (dW, db)
        compute_gradients(layer2, layer1->a); // Layer 2's input is Layer 1's A (activation)
        compute_gradients(layer1, X);         // Layer 1's input is X

        // --- D. WEIGHT UPDATE (OPTIMIZER) ---
        update_weights(layer2, lr);
        update_weights(layer1, lr);

        // Log status every 200 epochs
        if (epoch % 200 == 0) {
            printf("Epoch %4d | Loss: %.6f\n", epoch, total_loss);
        }
    }

    // ==========================================
    // 5. TEST AND RESULTS
    // ==========================================
    printf("\n--- TRAINING RESULTS / PREDICTIONS (XOR) ---\n");
    for (int i = 0; i < batch_size; i++) {
        int x1 = (int)X->data[i * 2 + 0];
        int x2 = (int)X->data[i * 2 + 1];
        float target = Y->data[i];
        float prediction = layer2->a->data[i];
        
        printf("Input: [%d, %d] | Target: %.0f | Network Prediction: %.4f\n", x1, x2, target, prediction);
    }

    // 6. Memory Cleanup (Preventing Memory Leaks)
    free_matrix(X);
    free_matrix(Y);
    free_dense_layer(layer1);
    free_dense_layer(layer2);

    printf("\nEngine successfully shut down.\n");
    return 0;
}