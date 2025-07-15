#ifndef FORWARD_H
#define FORWARD_H
#include "neural_network.h"

void forward_layer(Layer *layer, float *input);
void forward_neural_network(NeuralNetwork *nn, float *input);

#endif