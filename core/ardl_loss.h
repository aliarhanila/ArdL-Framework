#ifndef ARDL_LOSS_H
#define ARDL_LOSS_H

#include "ardl_math.h"


// --- 6. Loss Functions ---
float binary_crossentropy(Matrix *y_true, Matrix *y_pred);
float categorical_crossentropy(Matrix *y_true, Matrix *y_pred);
float mse_loss(Matrix *y_true, Matrix *y_pred);
float msle_loss(Matrix *y_true, Matrix *y_pred);

#endif