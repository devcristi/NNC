#include <stdio.h>
#include <math.h>
#include "./include/neural_network.h"
#include "./include/relu.h"
#include "./include/loss.h"
#include "./include/forward.h"
#include "./include/backpropagation.h"

int predict_label(float *probs, int size) {
    int max_index = 0;
    float max_value = probs[0];
    for (int i = 1; i < size; i++) {
        if (probs[i] > max_value) {
            max_value = probs[i];
            max_index = i;
        }
    }
    return max_index;
}

int main() {
    NeuralNetwork *nn = create_neural_network(3, (int[]){2, 4, 1});

    if (!nn) {
        fprintf(stderr, "Failed to create neural network\n");
        return 1;
    }

    float inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    float targets[4][1] = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    int epochs = 50000;
    float learning_rate = 0.1f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        for (int i = 0; i < 4; i++) {
            forward_neural_network(nn, inputs[i]);
            total_loss += cross_entropy_loss(nn->layers[nn->num_layers - 1]->output, targets[i], 1);
            backpropagation(nn, inputs[i], targets[i], learning_rate);
        }
        if (epoch % 1000 == 0) {
            printf("Epoch %d, Loss: %.6f\n", epoch, total_loss / 4);
        }
    }

    printf("\nTesting network on XOR inputs:\n");
    for (int i = 0; i < 4; i++) {
        forward_neural_network(nn, inputs[i]);
        float *output = nn->layers[nn->num_layers - 1]->output;
        printf("Input: %.1f %.1f -> Output: %.4f\n", inputs[i][0], inputs[i][1], output[0]);
    }

    free_neural_network(nn);
    return 0;
}
