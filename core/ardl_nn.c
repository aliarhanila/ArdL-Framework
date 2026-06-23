#include "ardl_nn.h"



// Pointer safety macro for standard C compatibility
#ifndef __cplusplus
#define nullptr ((void*)0)
#endif



/*
# ----------------------
# İnitilazitions
# ----------------------
*/
Conv2DLayer* init_conv2d_layer(MemoryArena *arena, int in_channels, int out_channels, 
                               int kernel_size, int stride, int padding,
                               int input_height, int input_width, int batch_size, int activation_type) 
{
    Conv2DLayer *layer = (Conv2DLayer*)arena_alloc(arena, sizeof(Conv2DLayer));
    if(layer == nullptr) return nullptr;

    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->input_height = input_height;
    layer->input_width = input_width;
    layer->batch_size = batch_size;
    layer->activation_type = activation_type;

    // 1. Calculate Output Dimensions: ((Input - Filter + 2*Padding) / Stride) + 1
    layer->output_height = ((input_height - kernel_size + 2 * padding) / stride) + 1;
    layer->output_width = ((input_width - kernel_size + 2 * padding) / stride) + 1;

    // 2. Filter (Kernel) Memory Allocation
    // Each filter has in_channels * k_size * k_size parameters.
    int kernel_flat_dim = in_channels * kernel_size * kernel_size;
    layer->kernels = create_matrix_area(arena, out_channels, kernel_flat_dim);
    
    // He or Xavier initialization based on activation type
    if (activation_type == 0) { 
        randomize_matrix_he(layer->kernels, kernel_flat_dim);
    } else {
        randomize_matrix_xavier(layer->kernels, kernel_flat_dim);
    }

    layer->biases = create_matrix_area(arena, 1, out_channels);
    for(int i = 0; i < out_channels; i++) {
        layer->biases->data[i] = 0.0f;
    }

    // 3. Output Matrices Abstracting Tensors
    // Total number of features produced for each batch element
    int output_flat_dim = out_channels * layer->output_height * layer->output_width;
    
    layer->z = create_matrix_area(arena, batch_size, output_flat_dim);
    layer->a = create_matrix_area(arena, batch_size, output_flat_dim);
    
    // 4. Gradient Memory Allocation
    layer->delta = create_matrix_area(arena, batch_size, output_flat_dim);
    layer->dK = create_matrix_area(arena, out_channels, kernel_flat_dim);
    layer->db = create_matrix_area(arena, 1, out_channels);
    
    // im2col matrix dimensions: 
    // Rows: in_channels * kernel_size * kernel_size
    // Cols: output_height * output_width
    int im2col_rows = in_channels * kernel_size * kernel_size;
    int im2col_cols = layer->output_height * layer->output_width;
    
    layer->im2col_buffer = create_matrix_area(arena, im2col_rows, im2col_cols);

    return layer;
}

DenseLayer* init_dense_layer(MemoryArena *arena, int input_dim , int output_dim, int batch_size, int activation_type){
    DenseLayer *layer = (DenseLayer*)arena_alloc(arena, sizeof(DenseLayer));
    if(!layer) return nullptr;
    
    // Assign activation type to the layer
    layer->activation_type = activation_type; 

    layer->weights = create_matrix_area(arena, input_dim, output_dim);

    if (activation_type == 0) {
        randomize_matrix_he(layer->weights, input_dim);
    } else {
        randomize_matrix_xavier(layer->weights, input_dim);
    }

    layer->weights_T = create_matrix_area(arena, output_dim, input_dim);
    transpose_matrix(layer->weights, layer->weights_T);

    layer->biases = create_matrix_area(arena, 1, output_dim);
    for (int i = 0; i < output_dim; i++){
        layer->biases->data[i] = 0.0f;
    }

    layer->z = create_matrix_area(arena, batch_size, output_dim);
    layer->a = create_matrix_area(arena, batch_size, output_dim);

    layer->delta = create_matrix_area(arena, batch_size, output_dim);
    layer->dW    = create_matrix_area(arena, input_dim, output_dim);
    layer->db    = create_matrix_area(arena, 1, output_dim);
    
    return layer; 
}

/*
#-----------------------------------
# MaxPool2D Initialization
#-----------------------------------
*/
MaxPool2DLayer* init_maxpool2d_layer(MemoryArena *arena, int channels, 
                                     int pool_size, int stride, int padding,
                                     int input_height, int input_width, int batch_size) 
{
    MaxPool2DLayer *layer = (MaxPool2DLayer*)arena_alloc(arena, sizeof(MaxPool2DLayer));
    if(layer == nullptr) return nullptr;

    layer->channels = channels;
    layer->pool_size = pool_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->input_height = input_height;
    layer->input_width = input_width;
    layer->batch_size = batch_size;

    layer->output_height = ((input_height - pool_size + 2 * padding) / stride) + 1;
    layer->output_width = ((input_width - pool_size + 2 * padding) / stride) + 1;

    int output_flat_dim = channels * layer->output_height * layer->output_width;

    layer->a = create_matrix_area(arena, batch_size, output_flat_dim);
    layer->delta = create_matrix_area(arena, batch_size, output_flat_dim);

    // Allocate integer array for index tracking (Backprop memory)
    layer->max_indices = (int*)arena_alloc(arena, batch_size * output_flat_dim * sizeof(int));

    return layer;
}



/*
#-----------------------------
# Forward Propagation Funcs
#-----------------------------
*/
//---forward pass---
void forward_pass(DenseLayer *layer, Matrix *input_a)
{
    // 1. Z = W^T * X
    matrix_dot_optimized(input_a, layer->weights_T, layer->z);

    // 2. Z = Z + bias
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < layer->z->rows; i++){
        for(int j = 0; j < layer->z->cols; j++){
            layer->z->data[i * layer->z->cols + j] += layer->biases->data[j];
        }
    }

    // 3. Activation based on new DRY helper
    apply_activations(layer->z, layer->a, layer->activation_type);
}

//--- conv2d forward pass---
void forward_conv2d(Conv2DLayer *layer, Matrix *input_a) 
{
    int batch_size = layer->batch_size;
    int in_features_per_img = layer->in_channels * layer->input_height * layer->input_width;
    int out_features_per_img = layer->out_channels * layer->output_height * layer->output_width;

    // 1. PROCESS IMAGES (im2col + GEMM)
    for (int b = 0; b < batch_size; b++) {
        float *img_ptr = input_a->data + (b * in_features_per_img);

        im2col(img_ptr, 
               layer->in_channels, layer->input_height, layer->input_width, 
               layer->kernel_size, layer->stride, layer->padding, 
               layer->im2col_buffer->data);

        float *z_ptr = layer->z->data + (b * out_features_per_img);
        Matrix z_out = {layer->out_channels, layer->output_height * layer->output_width, z_ptr};

        matrix_dot(layer->kernels, layer->im2col_buffer, &z_out);
    }

    // 2. CONV-SPECIFIC BIAS ADDITION (Broadcasting)
    int total_elements = batch_size * out_features_per_img;
    int spatial_size = layer->output_height * layer->output_width;

    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        int channel_idx = (i / spatial_size) % layer->out_channels;
        layer->z->data[i] += layer->biases->data[channel_idx];
    }

    // 3. APPLY COMMON ACTIVATION
    apply_activations(layer->z, layer->a, layer->activation_type);
}

//---Maxpool forward pass ---
void forward_maxpool2d(MaxPool2DLayer *layer, Matrix *input_a) 
{
    int batch = layer->batch_size;
    int channels = layer->channels;
    int in_h = layer->input_height;
    int in_w = layer->input_width;
    int out_h = layer->output_height;
    int out_w = layer->output_width;
    int pool_size = layer->pool_size;
    int stride = layer->stride;
    int padding = layer->padding;

    int in_spatial = in_h * in_w;
    int out_spatial = out_h * out_w;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                
                for (int ow = 0; ow < out_w; ow++) {
                    
                    int h_start = oh * stride - padding;
                    int w_start = ow * stride - padding;

                    float max_val = -INFINITY;
                    int max_idx = -1;

                    // Scan the pool window
                    for (int kh = 0; kh < pool_size; kh++) {
                        for (int kw = 0; kw < pool_size; kw++) {
                            int ih = h_start + kh;
                            int iw = w_start + kw;

                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int in_index = (b * channels + c) * in_spatial + (ih * in_w + iw);
                                float val = input_a->data[in_index];
                                
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = in_index; // Track the winner!
                                }
                            }
                        }
                    }

                    int out_index = (b * channels + c) * out_spatial + (oh * out_w + ow);
                    
                    layer->a->data[out_index] = max_val;
                    layer->max_indices[out_index] = max_idx; // Save for backprop
                }
            }
        }
    }
}

/*
#-----------------------
# Backpropagation
#-----------------------
*/

//Conv2D Backpropagation

void compute_gradients_conv2d(Conv2DLayer *layer, Matrix *input_a, Matrix *prev_layer_delta) 
{
    int batch_size = layer->batch_size;
    int in_features_per_img = layer->in_channels * layer->input_height * layer->input_width;
    int out_features_per_img = layer->out_channels * layer->output_height * layer->output_width;
    
    int kernel_flat_dim = layer->in_channels * layer->kernel_size * layer->kernel_size;
    int out_spatial = layer->output_height * layer->output_width;

    // 1. BIAS GRADYANLARI (db)
    #pragma omp parallel for schedule(static)
    for(int oc = 0; oc < layer->out_channels; oc++) {
        float sum = 0.0f;
        for(int b = 0; b < batch_size; b++) {
            for(int spatial = 0; spatial < out_spatial; spatial++) {
                sum += layer->delta->data[(b * out_features_per_img) + (oc * out_spatial) + spatial];
            }
        }
        layer->db->data[oc] = sum / batch_size;
    }

    // dK Matrisini sıfırla (Üzerine ekleme yapacağız)
    int dk_elements = layer->out_channels * kernel_flat_dim;
    #pragma omp parallel for
    for(int i = 0; i < dk_elements; i++) layer->dK->data[i] = 0.0f;

    // 2. KERNEL (dK) VE GİRDİ (dX) GRADYANLARI
    for(int b = 0; b < batch_size; b++) {
        
        float *img_ptr = input_a->data + (b * in_features_per_img);
        float *delta_ptr = layer->delta->data + (b * out_features_per_img);
        
        // Orijinal input'u tekrar im2col formatına getir (dK hesabı için)
        im2col(img_ptr, layer->in_channels, layer->input_height, layer->input_width, 
               layer->kernel_size, layer->stride, layer->padding, layer->im2col_buffer->data);

        // A. dK HESABI (dK += delta * im2col^T) -> Sanal Transpoze GEMM
        #pragma omp parallel for collapse(2) schedule(static)
        for(int oc = 0; oc < layer->out_channels; oc++) {
            for(int ic = 0; ic < kernel_flat_dim; ic++) {
                float sum = 0.0f;
                for(int spatial = 0; spatial < out_spatial; spatial++) {
                    sum += delta_ptr[oc * out_spatial + spatial] * layer->im2col_buffer->data[ic * out_spatial + spatial];
                }
                #pragma omp atomic
                layer->dK->data[oc * kernel_flat_dim + ic] += sum / batch_size;
            }
        }

        // Eğer önceki bir katman varsa (Girdi katmanı değilse), hatayı geriye ilet
        if (prev_layer_delta != nullptr) {
            
            float *dx_col_ptr = layer->dx_col_buffer->data;
            float *prev_delta_ptr = prev_layer_delta->data + (b * in_features_per_img);

            // B. dX_col HESABI (dX_col = K^T * delta) -> Sanal Transpoze GEMM
            #pragma omp parallel for collapse(2) schedule(static)
            for(int ic = 0; ic < kernel_flat_dim; ic++) {
                for(int spatial = 0; spatial < out_spatial; spatial++) {
                    float sum = 0.0f;
                    for(int oc = 0; oc < layer->out_channels; oc++) {
                        sum += layer->kernels->data[oc * kernel_flat_dim + ic] * delta_ptr[oc * out_spatial + spatial];
                    }
                    dx_col_ptr[ic * out_spatial + spatial] = sum;
                }
            }

            // C. col2im İLE GÖRSEL FORMATINA DÖNÜŞ (dX_col -> prev_layer_delta)
            col2im(dx_col_ptr, layer->in_channels, layer->input_height, layer->input_width,
                   layer->kernel_size, layer->stride, layer->padding, prev_delta_ptr);
        }
    }
}

// MaxPool2D Backpropagation
void backward_maxpool2d(MaxPool2DLayer *layer, Matrix *prev_layer_delta) 
{
    int batch = layer->batch_size;
    int channels = layer->channels;
    int in_h = layer->input_height;
    int in_w = layer->input_width;
    int out_h = layer->output_height;
    int out_w = layer->output_width;

    int in_spatial = in_h * in_w;
    int out_spatial = out_h * out_w;

    int total_prev_elements = batch * channels * in_spatial;
    int total_out_elements = batch * channels * out_spatial;

    // 1. Reset the previous layer's delta tensor to 0
    #pragma omp parallel for
    for (int i = 0; i < total_prev_elements; i++) {
        prev_layer_delta->data[i] = 0.0f;
    }

    // 2. Route gradients only to the winning pixels from forward pass
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total_out_elements; i++) {
        int max_idx = layer->max_indices[i];
        
        // Defensive boundary check to ensure stable memory mapping
        if (max_idx >= 0 && max_idx < total_prev_elements) {
            #pragma omp atomic
            prev_layer_delta->data[max_idx] += layer->delta->data[i];
        }
    }
}
// MLP BACKPROP FUNC
void compute_gradients(DenseLayer *layer, Matrix *input_a){
    float batch_size = (float)layer->delta->rows; 

    // Biases Gradient
    #pragma omp parallel for schedule(static)
    for(int j = 0; j < layer->db->cols; j++){
        float sum = 0.0f;
        for(int i = 0; i < layer->delta->rows; i++){
            sum += layer->delta->data[i * layer->delta->cols + j];
        }
        // Gradient Normalization
        layer->db->data[j] = sum / batch_size;
    }

    // Weights Gradient (BOTTLENECK RESOLVED: Full power with collapse(2))
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i = 0; i < input_a->cols; i++){
        for(int j = 0; j < layer->delta->cols; j++){
            float sum = 0.0f;
            for(int k = 0; k < input_a->rows; k++){
                float a_val = input_a->data[k * input_a->cols + i];
                float d_val = layer->delta->data[k * layer->delta->cols + j];
                sum += a_val * d_val;
            }
            // Gradient Normalization
            layer->dW->data[i * layer->dW->cols + j] = sum / batch_size;
        }
    }
}

void update_weights(DenseLayer *layer, float lr){
    int w_elements = layer->weights->rows * layer->weights->cols;
    
    #pragma omp parallel for
    for(int i = 0; i < w_elements; i++){
        layer->weights->data[i] -= lr * layer->dW->data[i]; 
    }

    int b_elements = layer->biases->rows * layer->biases->cols;
    
    #pragma omp parallel for
    for(int i = 0; i < b_elements; i++){
        layer->biases->data[i] -= lr * layer->db->data[i]; 
    }

    transpose_matrix(layer->weights, layer->weights_T);
}

/*
#-----------------------------
# Core serialize funcs
#-----------------------------
*/
void core_serialize_weights(DenseLayer *layer, FILE *file) {
    int w_size = layer->weights->rows * layer->weights->cols;
    fwrite(layer->weights->data, sizeof(float), w_size, file);

    int b_size = layer->biases->rows * layer->biases->cols;
    fwrite(layer->biases->data, sizeof(float), b_size, file);
}

void core_deserialize_weights(DenseLayer *layer, FILE *file) {
    int w_size = layer->weights->rows * layer->weights->cols;
    fread(layer->weights->data, sizeof(float), w_size, file);

    int b_size = layer->biases->rows * layer->biases->cols;
    fread(layer->biases->data, sizeof(float), b_size, file);

    transpose_matrix(layer->weights, layer->weights_T);
}