#ifndef DISTILL_YOLO_LAYER_H
#define DISTILL_YOLO_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_distill_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int distill_index);

void forward_distill_yolo_layer(const layer l, network snet, network tnet);

void backward_distill_yolo_layer(const layer l, network net);

void resize_distill_yolo_layer(layer *l, int w, int h);

#ifdef GPU
void forward_distill_yolo_layer_gpu(const layer l, network snet, network tnet);
void backward_distill_yolo_layer_gpu(layer l, network net);
#endif

#endif
