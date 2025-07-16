#ifndef LOSS_H
#define LOSS_H

float cross_entropy_loss(float *predictions, float *targets, int size);
void softmax(float *input, float *output, int size);
float binary_cross_entropy_loss(float prediction, float target);
float sigmoid(float x);
float sigmoid_derivative(float x);

#endif