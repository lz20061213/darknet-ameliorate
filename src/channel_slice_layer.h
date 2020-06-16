#ifndef CHANNEL_SLICE_LAYER_H
#define CHANNEL_SLICE_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer channel_slice_layer;

channel_slice_layer make_channel_slice_layer(int batch, int w, int h, int c, \
                                             int begin_slice_point, int end_slice_point, int axis,\
                                             int n, int *input_layers, int *input_sizes);

void resize_channel_slice_layer(channel_slice_layer *l, int h, int w);

void forward_channel_slice_layer(const channel_slice_layer l, network net);
void backward_channel_slice_layer(const channel_slice_layer l, network net);

#ifdef GPU
void forward_channel_slice_layer_gpu(const channel_slice_layer l, network net);
void backward_channel_slice_layer_gpu(const channel_slice_layer l, network net);
#endif

#endif //CHANNEL_SLICE_LAYER_H