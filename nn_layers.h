// nn_layers.h
#ifndef NN_LAYERS_H
#define NN_LAYERS_H

// --- Data Structures ---
typedef struct {
    int rows;
    int cols;
    float *data;
} Matrix;

typedef struct {
    Matrix *weights;
    Matrix *weights_T;
    Matrix *biases;
    Matrix *z;
    Matrix *a;
    Matrix *delta;
    Matrix *dW;
    Matrix *db;
} DenseLayer;

// --- Function Prototypes ---

// 1. Memory Management & Layer Initialization
Matrix* create_matrix(int rows, int cols);
void free_matrix(Matrix *mat);
DenseLayer* init_dense_layer(int input_dim, int output_dim, int batch_size);
void free_dense_layer(DenseLayer *layer);

// 2. Mathematics & Activations
float ReLu(float x);
float sigmoid(float x);
float ReLu_grad(float x);
float sigmoid_grad(float x);
float random_randn();
void randomize_matrix(Matrix *mat);

// 3. Matrix Operations & Forward Propagation
void transpose_matrix(Matrix *src, Matrix *dst);
void matrix_dot(Matrix *A, Matrix *B, Matrix *C);
void forward_pass(DenseLayer *layer, Matrix *input_a, int activation_type);
void matrix_dot_optimized(Matrix *A, Matrix *B_T, Matrix *C);
void compute_gradients(DenseLayer *layer , Matrix *input_a);
void update_weights(DenseLayer *layer,float lr);



#endif // NN_LAYERS_H