#ifndef KEYPOINT_YOLO_LAYER_H
#define KEYPOINT_YOLO_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_keypoint_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int keypoints_num);
void forward_keypoint_yolo_layer(const layer l, network net);
void backward_keypoint_yolo_layer(const layer l, network net);
void resize_keypoint_yolo_layer(layer *l, int w, int h);
int keypoint_yolo_num_detection_with_keypoints(layer l, float thresh);

#ifdef GPU
void forward_keypoint_yolo_layer_gpu(const layer l, network net);
void backward_keypoint_yolo_layer_gpu(layer l, network net);
#endif

#endif
