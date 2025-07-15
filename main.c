#include <stdio.h>
#include <math.h>
#include "./include/neural_network.h"
#include "./include/relu.h"
#include "./include/loss.h"
#include "./include/forward.h"

// Funcție pentru determinarea indexului clasei cu probabilitate maximă
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
    NeuralNetwork *nn = create_neural_network(2, (int[]){4, 3});
    if (!nn) {
        fprintf(stderr, "Failed to create neural network\n");
        return 1;
    }

    printf("Neural network created with %d layer(s)\n", nn->num_layers);
    for (int i = 0; i < nn->num_layers; i++) {
        Layer *l = nn->layers[i];
        printf("Layer %d: input = %d, output = %d\n", i, l->input_size, l->output_size);

        for (int n = 0; n < l->output_size; n++) {
            printf("  Neuron %d weights: ", n);
            for (int w = 0; w < l->input_size; w++) {
                printf("%.2f ", l->weights[n][w]);
            }
            printf("\n");
        }
    }

    // Forward pass cu input arbitrar
    float input[] = {1.0f, 0.5f, -0.5f, 0.0f};
    forward_neural_network(nn, input);

    printf("\nOutput of the last layer:\n");
    Layer *last_layer = nn->layers[nn->num_layers - 1];
    for (int i = 0; i < last_layer->output_size; i++) {
        printf("%.4f ", last_layer->output[i]);
    }
    printf("\n");

    // Aplicăm ReLU (doar ca test, inutil pe output softmax)
    printf("\nApplying ReLU activation (for demo only):\n");
    for (int i = 0; i < last_layer->output_size; i++) {
        last_layer->output[i] = relu(last_layer->output[i]);
        printf("%.4f ", last_layer->output[i]);
    }
    printf("\n");

    // Test cu softmax și cross-entropy loss
    float logits[] = {2.5f, 0.3f, -1.2f};
    float probs[3];
    float target[] = {0.0f, 1.0f, 0.0f};  // Clasa corectă: 1

    softmax(logits, probs, 3);

    printf("\nSoftmax output for logits {2.5, 0.3, -1.2}:\n");
    for (int i = 0; i < 3; i++) {
        printf("%.4f ", probs[i]);
    }
    printf("\n");

    float loss = cross_entropy_loss(probs, target, 3);
    printf("Cross-entropy loss: %.6f\n", loss);

    int predicted = predict_label(probs, 3);
    printf("Predicted class: %d\n", predicted);

    free_neural_network(nn);
    return 0;
}
