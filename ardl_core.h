#ifndef NN_LAYERS_H
#define NN_LAYERS_H


// --- Memory Structures ---
typedef struct {
    unsigned char *buffer;
    size_t size;
    size_t offset;
} MemoryArena;

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

// --- 1. Memory Management ---
MemoryArena* arena_create(size_t size);
void* arena_alloc(MemoryArena *arena,size_t size);
void arena_reset(MemoryArena *arena);
void arena_free(MemoryArena *arena);

// --- 2. Matrix & Layer ---
Matrix* create_matrix_area(MemoryArena *arena,int rows, int cols);
DenseLayer* init_dense_layer(MemoryArena *arena,int input_dim, int output_dim, int batch_size);

// --- 3. Activations & Utils ---
float ReLu(float x);
float sigmoid(float x);
float ReLu_grad(float x);
float sigmoid_grad(float x);
float random_randn();
void randomize_matrix(Matrix *mat);

// --- 4. Matrix Operations ---
void transpose_matrix(const Matrix *src, Matrix *dst);
void matrix_dot(const Matrix *A, const Matrix *B, Matrix *C);
void matrix_dot_optimized(const Matrix *A, const Matrix *B_T, Matrix *C);

// --- 5. Forward / Backward ---
void forward_pass(DenseLayer *layer, Matrix *input_a, int activation_type);
void compute_gradients(DenseLayer *layer , Matrix *input_a);
void update_weights(DenseLayer *layer,float lr);

#endif
