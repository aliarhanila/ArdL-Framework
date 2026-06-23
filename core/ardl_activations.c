#include "ardl_activations.h"





/*
# ---------------------------
#  Activations
# ---------------------------
*/
float ReLu(float x){
    return fmaxf(0.0f, x);
}

float ReLu_grad(float x){
    return (x > 0.0f) ? 1.0f : 0.0f;
}

float Tanh(float x) {
    return tanhf(x); 
}

float Tanh_grad(float x) {
    float t = tanhf(x);
    return 1.0f - (t * t);
}

float LeakyReLu(float x){
    return (x > 0.0f) ? x : 0.01f * x;
}

float LeakyReLu_grad(float x){
    return (x > 0.0f) ? 1.0f : 0.01f;
}

float sigmoid(float x){
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_grad(float x){
    float s = sigmoid(x);
    return s * (1.0f - s);
}

void softmax(Matrix *z, Matrix *a) {
    int batch_size = z->rows;
    int out_dim = z->cols;
    
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        // Step 1: Find max value for numerical stability (prevent overflow)
        float max_val = z->data[i * out_dim];
        for (int j = 1; j < out_dim; j++) {
            if (z->data[i * out_dim + j] > max_val) {
                max_val = z->data[i * out_dim + j];
            }
        }

        // Step 2: Subtract max_val, apply exp(), and calculate sum
        float sum_exp = 0.0f;
        for (int j = 0; j < out_dim; j++) {
            float e = expf(z->data[i * out_dim + j] - max_val);
            a->data[i * out_dim + j] = e;
            sum_exp += e;
        }

        // Step 3: Divide by sum to get probability distribution
        for (int j = 0; j < out_dim; j++) {
            a->data[i * out_dim + j] /= sum_exp;
        }
    }
}




/*
#---------------------------
# Activation Applicator
#---------------------------
*/
// Common activation applicator for any matrix (Dense or Conv2D)
void apply_activations(Matrix *z, Matrix *a, int activation_type) 
{
    if (activation_type == 5) { // SOFTMAX
        softmax(z, a);
    } 
    else {
        int total = z->rows * z->cols;
        
        #pragma omp parallel for
        for(int i = 0; i < total; i++){
            float val = z->data[i];
            
            if (activation_type == 0)      a->data[i] = val;         // LINEAR
            else if (activation_type == 1) a->data[i] = ReLu(val);   // RELU
            else if (activation_type == 2) a->data[i] = LeakyReLu(val); // LEAKY_RELU
            else if (activation_type == 3) a->data[i] = Tanh(val);   // TANH
            else if (activation_type == 4) a->data[i] = sigmoid(val); // SIGMOID
        }
    }
}
