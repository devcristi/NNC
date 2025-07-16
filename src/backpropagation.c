#include "../include/backpropagation.h"
#include "../include/loss.h"
#include "../include/relu.h"
#include <stdlib.h>

// Derivata funcției ReLU
static float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

void backpropagation(NeuralNetwork *nn, float *input_data, float *targets, float learning_rate)
{
    int last_index = nn->num_layers - 1;
    Layer *output_layer = nn->layers[last_index];

    // 1. Calculăm eroarea pentru output layer (softmax + cross-entropy derivative)
    for (int i = 0; i < output_layer->output_size; i++) {
        //* float a = output_layer->output[i];
        //! output_layer->delta[i] = (a - targets[i]) * sigmoid_derivative(output_layer->z[i]);  
        float a = output_layer->output[i];              // activarea (sigmoid(sum))
        float t = targets[i];                           // target (0 sau 1)
        // derivata BCE w.r.t. z: (a - t) * sigmoid'(z)
        // iar sigmoid'(z) = a * (1 - a)
        output_layer->delta[i] = (a - t) * (a * (1.0f - a));  
    }

    // 2. Backpropagation pentru layerele ascunse
    for (int l = last_index - 1; l >= 0; l--) {
        Layer *layer = nn->layers[l];
        Layer *next = nn->layers[l + 1];

        for (int i = 0; i < layer->output_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < next->output_size; j++) {
                sum += next->weights[j][i] * next->delta[j];
            }
            layer->delta[i] = relu_derivative(layer->output[i]) * sum;
        }
    }

    // 3. Update pentru weights și biases
    for (int l = 0; l < nn->num_layers; l++) {
        Layer *layer = nn->layers[l];
        float *input = (l == 0) ? input_data : nn->layers[l - 1]->output;

        for (int i = 0; i < layer->output_size; i++) {
            for (int j = 0; j < layer->input_size; j++) {
                layer->weights[i][j] -= learning_rate * layer->delta[i] * input[j];
            }
            layer->biases[i] -= learning_rate * layer->delta[i];
        }
    }
}
