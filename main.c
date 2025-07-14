#include <stdio.h>
#include "./include/neural_network.h"

int main()
{
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

    free_neural_network(nn);
    return 0;
}