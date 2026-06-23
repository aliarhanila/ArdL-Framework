#ifndef ARDL_ACTIVATIONS_H
#define ARDL_ACTIVATIONS_H

#include "ardl_math.h"

//---Activations Macros ----
#define LINEAR 0
#define RELU 1
#define LEAKY_RELU 2
#define TANH 3
#define SIGMOID 4
#define SOFTMAX 5

// --- Activations & Utils ---
float ReLu(float x);
float ReLu_grad(float x);
float LeakyReLu(float x);
float Tanh(float x);
float Tanh_grad(float x);
float LeakyReLu_grad(float x);
float sigmoid(float x);
float sigmoid_grad(float x);


void softmax(Matrix *z, Matrix *a);
void apply_activations(Matrix *z, Matrix *a, int activation_type);

#endif
