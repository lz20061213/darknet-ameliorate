#ifndef REWEIGHT_LAYER_H
#define REWEIGHT_LAYER_H
#include "network.h"
#include "layer.h"
#include "activations.h"

typedef layer reweight_layer;

reweight_layer make_reweight_layer(int batch, int n, int *input_layers, int *input_size, ACTIVATION activation);
void forward_reweight_layer(const reweight_layer l, network net);
void backward_reweight_layer(const reweight_layer l, network net);
void resize_reweight_layer(reweight_layer *l, network *net);

#ifdef GPU
void forward_reweight_layer_gpu(const reweight_layer l, network net);
void backward_reweight_layer_gpu(const reweight_layer l, network net);
#endif

#endif