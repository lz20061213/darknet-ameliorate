#ifndef HEATMAP_LAYER_H
#define HEATMAP_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_heatmap_layer(int batch, int w, int h, int keypoints_num);
void forward_heatmap_layer(const layer l, network net);
void backward_heatmap_layer(const layer l, network net);
void resize_heatmap_layer(layer *l, int w, int h);

#ifdef GPU
void forward_heatmap_layer_gpu(const layer l, network net);
void backward_heatmap_layer_gpu(layer l, network net);
#endif

#endif
