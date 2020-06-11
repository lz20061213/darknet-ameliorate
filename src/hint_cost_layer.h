#ifndef HINT_COST_LAYER_H
#define HINT_COST_LAYER_H

#include "layer.h"
#include "network.h"
#include "cost_layer.h"

typedef layer hint_cost_layer;

//COST_TYPE get_cost_type(char *s);

//char *get_cost_string(COST_TYPE a);

hint_cost_layer make_hint_cost_layer(int batch, int inputs, COST_TYPE type, float scale, int hint_index, float margin);

void forward_hint_cost_layer(const hint_cost_layer l, network snet, network tnet);

void backward_hint_cost_layer(const hint_cost_layer l, network snet);

void resize_hint_cost_layer(hint_cost_layer *l, int inputs);

#ifdef GPU
void forward_mimic_hint_cost_layer_gpu(hint_cost_layer l, network snet, network tnet);
void forward_mutual_hint_cost_layer_gpu(hint_cost_layer ls, network snet, network pnet);
void backward_hint_cost_layer_gpu(const hint_cost_layer l, network net);
#endif

#endif
