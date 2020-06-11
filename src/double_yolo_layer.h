#ifndef DOUBLE_YOLO_LAYER_H
#define DOUBLE_YOLO_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_double_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int *input_layers);
void forward_double_yolo_layer(const layer l, network net);
void backward_double_yolo_layer(const layer l, network net);
void resize_double_yolo_layer(layer *l, int w, int h);
int double_yolo_num_detections(layer l, network net, float thresh);

#ifdef GPU
void forward_double_yolo_layer_gpu(const layer l, network net);
void backward_double_yolo_layer_gpu(layer l, network net);
#endif

#endif
