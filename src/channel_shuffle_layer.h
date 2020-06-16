#ifndef CHANNEL_SHUFFLE_LAYER_H
#define CHANNEL_SHUFFLE_LAYER_H

#include "layer.h"
#include "network.h"

layer make_channel_shuffle_layer(int batch, int w, int h, int c, int groups);

void resize_channel_shuffle_layer(layer *l, int h, int w);

void forward_channel_shuffle_layer(const layer l, network net);
void backward_channel_shuffle_layer(const layer l, network net);

#ifdef GPU
void forward_channel_shuffle_layer_gpu(const layer l, network net);
void backward_channel_shuffle_layer_gpu(const layer l, network net);
#endif

#endif // CHANNEL_SHUFFLE_LAYER_H