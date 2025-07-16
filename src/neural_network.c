#include <stdlib.h>
#include "../include/neural_network.h"

Layer *create_layer(int input_size, int output_size)
{
    Layer *layer = malloc(sizeof(*layer)); //**alocarea memoriei pt layere
    if (!layer) {
        return NULL; // Handle memory allocation failure
    }

    // intializam propietatile layerului
    layer->input_size = input_size; 
    layer->output_size = output_size;
    layer->weights = malloc(output_size * sizeof(float *));

    if(!layer->weights)
    {
        free(layer);
        return NULL;
    }

    for(int i = 0; i < output_size; i++)
    {
        layer->weights[i] = malloc(input_size *sizeof(float));
        if(!layer->weights[i])
        {
            for(int j = 0; j < i; j++)
            
                free(layer->weights[j]);
            free(layer->weights);
            free(layer);

            return NULL;
        }

        // random init weights între -1 și 1
        for (int j = 0; j < input_size; j++) {
            layer->weights[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }

    layer->biases = malloc(output_size * sizeof(float));
    if (!layer->biases) {
        for (int i = 0; i < output_size; i++) {
            free(layer->weights[i]);
        }
        free(layer->weights);
        free(layer);
        return NULL;
    }

    for(int i = 0; i < output_size; i++) {
        layer->biases[i] = 0.0f; // Initializare biasuri la 0
    }

    // output
    layer->output = calloc(output_size, sizeof(float));
    layer->delta  = calloc(output_size, sizeof(float));
    if (!layer->output || !layer->delta) {
        free(layer->biases);
        for (int i = 0; i < output_size; i++) free(layer->weights[i]);
        free(layer->weights);
        free(layer->output); // chiar dacă NULL e safe
        free(layer->delta);
        // !free(layer->z);
        free(layer);
        return NULL;
    }
    return layer;
}

void free_layer(Layer *layer)
{
    if (!layer) return;

    // eliberezi fiecare rând din matricea weights
    for (int i = 0; i < layer->output_size; i++) {
        free(layer->weights[i]);
    }

    // eliberezi vectorul de pointeri
    free(layer->weights);

    // eliberezi restul vectorilor
    free(layer->biases);
    free(layer->output);
    free(layer->delta);
    //! free(layer->z);

    // eliberezi structura în sine
    free(layer);
}


NeuralNetwork *create_neural_network(int num_layers, int *layer_sizes)
{
    NeuralNetwork *nn = malloc(sizeof(*nn));
    if (!nn) {
        return NULL; // Handle memory allocation failure
    }

    nn->num_layers = num_layers;
    nn->layers = malloc(num_layers * sizeof(Layer *));
    if (!nn->layers) {
        free(nn);
        return NULL; // Handle memory allocation failure
    }

    for (int i = 0; i < num_layers; i++) {
        int input_size = (i == 0) ? layer_sizes[i] : layer_sizes[i - 1];
        nn->layers[i] = create_layer(input_size, layer_sizes[i]);
        if (!nn->layers[i]) {
            for (int j = 0; j < i; j++) {
                free_layer(nn->layers[j]);
            }
            free(nn->layers);
            free(nn);
            return NULL; // Handle memory allocation failure
        }
    }

    return nn;
}

void free_neural_network(NeuralNetwork *nn)
{
    if (!nn) return;

    for (int i = 0; i < nn->num_layers; i++) {
        free_layer(nn->layers[i]);
    }

    free(nn->layers);
    free(nn);
}