#ifndef LOSS_H
#define LOSS_H

float cross_entropy_loss(float *predictions, float *targets, int size);
void softmax(float *input, float *output, int size);

#endif