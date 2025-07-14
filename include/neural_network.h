#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

// ** structura generala a retelei neurale

typedef struct
{
    int input_size;
    int output_size;

    float **weights; //** matricea de greutati weights[output_size][input_size]
    float *biases;   //** vectorul de biasuri biases[output_size]

    //? pt backpropagation
    float *output; //** vectorul de iesire output[output_size]
    float *delta;  //** eroarea locală a layer-ului [output_size]
} Layer;

typedef struct
{
    int num_layers;
    Layer **layers;
} NeuralNetwork;

// ** functii pentru initializarea retelei neurale
Layer *create_layer(int input_size, int output_size);
void free_layer(Layer *layer);
NeuralNetwork *create_neural_network(int num_layers, int *layer_sizes);
void free_neural_network(NeuralNetwork *nn);

#endif