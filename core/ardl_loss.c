
#include "ardl_loss.h"


/*
#------------------------
# Loss functions
#------------------------
*/
float mse_loss(Matrix *y_true, Matrix *y_pred){
    float total_loss = 0.0f;
    int total_elements = y_true->rows * y_true->cols;
    
    #pragma omp parallel for reduction(+:total_loss)
    for(int i = 0 ; i < total_elements ; i++){
        float err = y_pred->data[i] - y_true->data[i];
        total_loss += err * err;
    }
    
    return total_loss / (float)y_true->rows;
}

float binary_crossentropy(Matrix *y_true, Matrix *y_pred){
    float total_loss = 0.0f;
    int batch_size = y_true->rows;
    int total_elements = y_true->rows * y_true->cols;
    
    #pragma omp parallel for reduction(+:total_loss)
    for(int i = 0 ; i < total_elements ; i++){
        float y_t = y_true->data[i];
        float y_p = y_pred->data[i];
        
        // Add epsilon (1e-9) to prevent log(0) calculation
        total_loss += -(y_t * logf(y_p + 1e-9f) + (1.0f - y_t) * logf(1.0f - y_p + 1e-9f));
    }
    
    return (total_loss / (float)batch_size);
}

float categorical_crossentropy(Matrix *y_true, Matrix *y_pred){
    float total_loss = 0.0f;
    int batch_size = y_true->rows;
    int total_elements = y_true->rows * y_true->cols;
    
    #pragma omp parallel for reduction(+:total_loss)
    for(int i = 0 ; i < total_elements ; i++){
        float y_t = y_true->data[i];
        float y_p = y_pred->data[i];
        total_loss += y_t * logf(y_p + 1e-9f);
    }
    
    return -(total_loss / (float)batch_size);
}

float msle_loss(Matrix *y_true, Matrix *y_pred){
    float total_loss = 0.0f;
    int total_elements = y_true->rows * y_true->cols;
    
    #pragma omp parallel for reduction(+:total_loss)
    for(int i = 0 ; i < total_elements ; i++){
        // Add fmaxf and +1.0f to avoid negative values (log(0) is undefined)
        float log_true = logf(fmaxf(y_true->data[i], 0.0f) + 1.0f);
        float log_pred = logf(fmaxf(y_pred->data[i], 0.0f) + 1.0f);
        float err = log_pred - log_true;
        total_loss += err * err;
    }
    
    return total_loss / (float)y_true->rows;
}
