#include <string.h>
#include "../include/neural_network.h"
#include "../include/relu.h"
#include "../include/loss.h"
#include "../include/forward.h"

void forward_neural_network(NeuralNetwork *nn, float *input) {
    float *current_input = input;

    for (int i = 0; i < nn->num_layers; i++) {
        Layer *layer = nn->layers[i];
        forward_layer(layer, current_input);
        current_input = layer->output;  // Output-ul devine input pentru next layer
    }

    // Aplicăm softmax pe ultimul layer
    Layer *last = nn->layers[nn->num_layers - 1];
    softmax(last->output, last->output, last->output_size);
}

void forward_layer(Layer *layer, float *input) {
    for (int i = 0; i < layer->output_size; i++) {
        float sum = layer->biases[i];
        for (int j = 0; j < layer->input_size; j++) {
            sum += layer->weights[i][j] * input[j];
        }
        layer->output[i] = relu(sum);
    }
}