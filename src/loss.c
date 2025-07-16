#include <stdlib.h>
#include "../include/loss.h"
#include <math.h>

float cross_entropy_loss(float *predictions, float *targets, int size)
{
    float loss = 0.0f;
    for(int i = 0; i < size; i++)
        if(targets[i] > 0.0f)
            loss -= targets[i] * logf(predictions[i] + 1e-9f);
    
    return loss / size;
}

void softmax(float *input, float *output, int size)
{
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val); // numerically stable
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

float binary_cross_entropy_loss(float prediction, float target) {
    return -(target * logf(prediction + 1e-9f) + (1.0f - target) * logf(1.0f - prediction + 1e-9f));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}
