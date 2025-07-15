#include <stdlib.h>
#include "../include/relu.h"

float relu(float x)
{
    return x>0 ? x:0;
}

void relu_array(float *array, int size)
{
    for(int i = 0; i < size; i++)
        array[i] = relu(array[i]);
}