#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "activation_layer.h"
#include "logistic_layer.h"
#include "l2norm_layer.h"
#include "activations.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "hint_cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "keypoint_yolo_layer.h"
#include "heatmap_layer.h"
#include "distill_yolo_layer.h"
#include "mutual_yolo_layer.h"
#include "mimicutual_yolo_layer.h"
#include "double_yolo_layer.h"
#include "iseg_layer.h"
#include "reorg_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "reweight_layer.h"
#include "channel_slice_layer.h"
#include "channel_shuffle_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "lstm_layer.h"
#include "utils.h"

typedef struct{
    char *type;
    list *options;
}section;

list *read_cfg(char *filename);

LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, "[shortcut]")==0) return SHORTCUT;
    if (strcmp(type, "[crop]")==0) return CROP;
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[hint_cost]")==0) return HINT_COST;
    if (strcmp(type, "[detection]")==0) return DETECTION;
    if (strcmp(type, "[region]")==0) return REGION;
    if (strcmp(type, "[yolo]")==0) return YOLO;
    if (strcmp(type, "[keypoint_yolo]")==0) return KEYPOINT_YOLO;
    if (strcmp(type, "[heatmap]")==0) return HEATMAP;
    if (strcmp(type, "[distill_yolo]")==0) return DISTILL_YOLO;
    if (strcmp(type, "[mutual_yolo]")==0) return MUTUAL_YOLO;
    if (strcmp(type, "[mimicutual_yolo]")==0) return MIMICUTUAL_YOLO;
    if (strcmp(type, "[double_yolo]")==0) return DOUBLE_YOLO;
    if (strcmp(type, "[iseg]")==0) return ISEG;
    if (strcmp(type, "[local]")==0) return LOCAL;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[deconv]")==0
            || strcmp(type, "[deconvolutional]")==0) return DECONVOLUTIONAL;
    if (strcmp(type, "[activation]")==0) return ACTIVE;
    if (strcmp(type, "[logistic]")==0) return LOGXENT;
    if (strcmp(type, "[l2norm]")==0) return L2NORM;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[crnn]")==0) return CRNN;
    if (strcmp(type, "[gru]")==0) return GRU;
    if (strcmp(type, "[lstm]") == 0) return LSTM;
    if (strcmp(type, "[rnn]")==0) return RNN;
    if (strcmp(type, "[conn]")==0
            || strcmp(type, "[connected]")==0) return CONNECTED;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[reorg]")==0) return REORG;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[dropout]")==0) return DROPOUT;
    if (strcmp(type, "[lrn]")==0
            || strcmp(type, "[normalization]")==0) return NORMALIZATION;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    if (strcmp(type, "[route]")==0) return ROUTE;
    if (strcmp(type, "[reweight]")==0) return REWEIGHT;
    if (strcmp(type, "[upsample]")==0) return UPSAMPLE;
    if (strcmp(type, "[channel_slice]")==0) return CHANNEL_SLICE;
    if (strcmp(type, "[channel_shuffle]")==0) return CHANNEL_SHUFFLE;
    return BLANK;
}

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

void parse_data(char *data, float *a, int n)
{
    int i;
    if(!data) return;
    char *curr = data;
    char *next = data;
    int done = 0;
    for(i = 0; i < n && !done; ++i){
        while(*++next !='\0' && *next != ',');
        if(*next == '\0') done = 1;
        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network *net;
} size_params;

local_layer parse_local(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int(options, "pad",0);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before local layer must output image.");

    local_layer layer = make_local_layer(batch,h,w,c,n,size,stride,pad,activation);

    return layer;
}

layer parse_deconvolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before deconvolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    if(pad) padding = size/2;

    layer l = make_deconvolutional_layer(batch,h,w,c,n,size,stride,padding, activation, batch_normalize, params.net->adam);

    return l;
}


convolutional_layer parse_convolutional(list *options, size_params params, network *net)
{
    int n = option_find_int(options, "filters", 1);
    int size = option_find_int(options, "size", 1);
    int stride = option_find_int(options, "stride", 1);
    int dilation = option_find_int_quiet(options, "dilation", 1);
    if (size == 1) dilation = 1;
    int pad = option_find_int_quiet(options, "pad", 0);
    int padding = option_find_int_quiet(options, "padding", 0);
    int groups = option_find_int_quiet(options, "groups", 1);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);
    int quantize = option_find_int_quiet(options, "quantize", net->quantize);

    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,dilation,padding,activation, batch_normalize, binary, xnor, params.net->adam, quantize);

    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);
    layer.quantize_feature = option_find_int_quiet(options, "quantize_feature", 1);
    layer.scale_weight = option_find_int_quiet(options, "scale_weight", 0);
    layer.isprune = option_find_int_quiet(options, "isprune", 0);
    layer.quantize_weight_bitwidth = option_find_int_quiet(options, "weight_bitwidth", net->quantize_weight_bitwidth);
    layer.quantize_weight_fraction_bitwidth = option_find_int_quiet(options, "weight_fraction_bitwidth", net->quantize_weight_fraction_bitwidth);
    layer.quantize_bias_bitwidth = option_find_int_quiet(options, "bias_bitwidth", net->quantize_bias_bitwidth);
    layer.quantize_bias_fraction_bitwidth = option_find_int_quiet(options, "bias_fraction_bitwidth", net->quantize_bias_fraction_bitwidth);
    layer.quantize_feature_bitwidth = option_find_int_quiet(options, "feature_bitwidth", net->quantize_feature_bitwidth);
    layer.quantize_feature_fraction_bitwidth = option_find_int_quiet(options, "feature_fraction_bitwidth", net->quantize_feature_fraction_bitwidth);
    layer.quantize_per_channel = option_find_int_quiet(options, "per_channel", net->quantize_per_channel);
    layer.quantize_per_channel_keep_rate = option_find_float_quiet(options, "per_channel_keep_rate", net->quantize_per_channel_keep_rate);

    layer.post_training_quantization = option_find_int_quiet(options, "post_training_quantization", net->post_training_quantization);
    layer.quantization_aware_training = option_find_int_quiet(options, "quantization_aware_training", net->quantization_aware_training);

    assert(layer.quantize == layer.post_training_quantization + layer.quantization_aware_training);

    if (layer.post_training_quantization) {
        if (layer.quantize_per_channel) {
            layer.conv_fls = calloc(layer.n, sizeof(int));
            layer.bias_fls = calloc(layer.n, sizeof(int));
            // read from .weights
            int i;
            for(i=0; i<layer.n; ++i) {
                layer.conv_fls[i] = 0;
                layer.bias_fls[i] = 0;
            }
        } else {
            layer.conv_fl = calloc(1, sizeof(int));
            layer.bias_fl = calloc(1, sizeof(int));
            // read from .weights
            *(layer.conv_fl) = option_find_int_quiet(options, "conv_fl", 0);
            *(layer.bias_fl) = option_find_int_quiet(options, "bias_fl", 0);
        }
        layer.x_fl = calloc(1, sizeof(int));
        // read from .cfg
        *(layer.x_fl) = option_find_int_quiet(options, "x_fl", 0);
    }

    if (layer.quantization_aware_training) {
        if (layer.quantize_per_channel) {
            //printf("allow memory for bitwidths\n");
            layer.quantize_weight_fraction_bitwidths = calloc(layer.n, sizeof(int));
            layer.quantize_bias_fraction_bitwidths = calloc(layer.n, sizeof(int));
        }
    }

    layer.leaky_rate = option_find_float_quiet(options, "leaky_rate", .1);

    return layer;
}

layer parse_crnn(list *options, size_params params)
{
    int output_filters = option_find_int(options, "output_filters",1);
    int hidden_filters = option_find_int(options, "hidden_filters",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_crnn_layer(params.batch, params.w, params.h, params.c, hidden_filters, output_filters, params.time_steps, activation, batch_normalize);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

layer parse_rnn(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_rnn_layer(params.batch, params.inputs, output, params.time_steps, activation, batch_normalize, params.net->adam);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

layer parse_gru(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_gru_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);
    l.tanh = option_find_int_quiet(options, "tanh", 0);

    return l;
}

layer parse_lstm(list *options, size_params params)
{
    int output = option_find_int(options, "output", 1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_lstm_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);

    return l;
}

layer parse_connected(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam);
    return l;
}

layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups",1);
    layer l = make_softmax_layer(params.batch, params.inputs, groups);
    l.temperature = option_find_float_quiet(options, "temperature", 1);
    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    l.w = params.w;
    l.h = params.h;
    l.c = params.c;
    l.spatial = option_find_float_quiet(options, "spatial", 0);
    l.noloss =  option_find_int_quiet(options, "noloss", 0);
    return l;
}

int *parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',')+1;
        }
        *num = n;
    }
    return mask;
}

layer parse_yolo(list *options, size_params params, network *net)
{
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes);
    assert(l.outputs == params.inputs);

    l.max_boxes = option_find_int_quiet(options, "max",90);
    l.jitter = option_find_float(options, "jitter", .2);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.random = option_find_int_quiet(options, "random", 0);
    l.rescore = option_find_int_quiet(options, "rescore", 0);
    l.lb_dis_max_thresh = option_find_float_quiet(options, "lb_dis_max_thresh", 10);
    l.lb_dis_ignore_thresh = option_find_float_quiet(options, "lb_dis_ignore_thresh", 5);
    l.lb_dis_truth_thresh = option_find_float_quiet(options, "lb_dis_truth_thresh", 0);
    l.scale_xy = option_find_float_quiet(options, "scale_xy", 1);
    l.use_center_regression = option_find_int(options, "use_center_regression", 0);
    l.object_focal_loss = option_find_int(options, "object_focal_loss", 0);

    l.iou_normalizer = option_find_float_quiet(options, "iou_normalizer", 1);
    l.cls_normalizer = option_find_float_quiet(options, "cls_normalizer", 1);

    l.label_smooth_rate = option_find_float_quiet(options, "label_smooth_rate", 0);

    l.quantize =  option_find_int_quiet(options, "quantize", net->quantize);
    l.post_training_quantization = option_find_int_quiet(options, "post_training_quantization", net->post_training_quantization);
    l.quantization_aware_training = option_find_int_quiet(options, "quantization_aware_training", net->quantization_aware_training);

    char *iou_loss = option_find_str_quiet(options, "iou_loss", "mse");   //
    if (strcmp(iou_loss, "mse") == 0) l.iou_loss = MSE;
    else if (strcmp(iou_loss, "giou") == 0) l.iou_loss = GIOU;
    else if (strcmp(iou_loss, "diou") == 0) l.iou_loss = DIOU;
    else if (strcmp(iou_loss, "ciou") == 0) l.iou_loss = CIOU;
    else l.iou_loss = IOU;

    char *iou_thresh_kind_str = option_find_str_quiet(options, "iou_thresh_kind", "iou");
    if (strcmp(iou_thresh_kind_str, "iou") == 0) l.iou_thresh_kind = IOU;
    else if (strcmp(iou_thresh_kind_str, "giou") == 0) l.iou_thresh_kind = GIOU;
    else if (strcmp(iou_thresh_kind_str, "diou") == 0) l.iou_thresh_kind = DIOU;
    else if (strcmp(iou_thresh_kind_str, "ciou") == 0) l.iou_thresh_kind = CIOU;
    else {
        fprintf(stderr, " Wrong iou_thresh_kind = %s \n", iou_thresh_kind_str);
        l.iou_thresh_kind = IOU;
    }

    l.beta_nms = option_find_float_quiet(options, "beta_nms", 0.6);
    char *nms_kind = option_find_str_quiet(options, "nms_kind", "default");
    if (strcmp(nms_kind, "default") == 0) l.nms_kind = DEFAULT_NMS;
    else {
        if (strcmp(nms_kind, "greedynms") == 0) l.nms_kind = GREEDY_NMS;
        else if (strcmp(nms_kind, "diounms") == 0) l.nms_kind = DIOU_NMS;
        else l.nms_kind = DEFAULT_NMS;
        printf("nms_kind: %s (%d), beta = %f \n", nms_kind, l.nms_kind, l.beta_nms);
    }

    l.atss =  option_find_int_quiet(options, "atss", 0);

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

layer parse_keypoint_yolo(list *options, size_params params, network *net)
{
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);

    int keypoints_num = option_find_int(options, "keypoints_num", 5);
    layer l = make_keypoint_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, keypoints_num);
    assert(l.outputs == params.inputs);

    l.max_boxes = option_find_int_quiet(options, "max",90);
    l.jitter = option_find_float(options, "jitter", .2);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.random = option_find_int_quiet(options, "random", 0);
    l.rescore = option_find_int_quiet(options, "rescore", 0);
    l.lb_dis_max_thresh = option_find_float_quiet(options, "lb_dis_max_thresh", 10);
    l.lb_dis_ignore_thresh = option_find_float_quiet(options, "lb_dis_ignore_thresh", 5);
    l.lb_dis_truth_thresh = option_find_float_quiet(options, "lb_dis_truth_thresh", 0);
    l.scale_xy = option_find_float_quiet(options, "scale_xy", 1);
    l.use_center_regression = option_find_int(options, "use_center_regression", 0);
    l.object_focal_loss = option_find_int(options, "object_focal_loss", 0);

    l.iou_normalizer = option_find_float_quiet(options, "iou_normalizer", 1);
    l.cls_normalizer = option_find_float_quiet(options, "cls_normalizer", 1);

    l.label_smooth_rate = option_find_float_quiet(options, "label_smooth_rate", 0);

    l.quantize =  option_find_int_quiet(options, "quantize", net->quantize);
    l.post_training_quantization = option_find_int_quiet(options, "post_training_quantization", net->post_training_quantization);
    l.quantization_aware_training = option_find_int_quiet(options, "quantization_aware_training", net->quantization_aware_training);

    char *iou_loss = option_find_str_quiet(options, "iou_loss", "mse");   //
    if (strcmp(iou_loss, "mse") == 0) l.iou_loss = MSE;
    else if (strcmp(iou_loss, "giou") == 0) l.iou_loss = GIOU;
    else if (strcmp(iou_loss, "diou") == 0) l.iou_loss = DIOU;
    else if (strcmp(iou_loss, "ciou") == 0) l.iou_loss = CIOU;
    else l.iou_loss = IOU;

    char *iou_thresh_kind_str = option_find_str_quiet(options, "iou_thresh_kind", "iou");
    if (strcmp(iou_thresh_kind_str, "iou") == 0) l.iou_thresh_kind = IOU;
    else if (strcmp(iou_thresh_kind_str, "giou") == 0) l.iou_thresh_kind = GIOU;
    else if (strcmp(iou_thresh_kind_str, "diou") == 0) l.iou_thresh_kind = DIOU;
    else if (strcmp(iou_thresh_kind_str, "ciou") == 0) l.iou_thresh_kind = CIOU;
    else {
        fprintf(stderr, " Wrong iou_thresh_kind = %s \n", iou_thresh_kind_str);
        l.iou_thresh_kind = IOU;
    }

    l.beta_nms = option_find_float_quiet(options, "beta_nms", 0.6);
    char *nms_kind = option_find_str_quiet(options, "nms_kind", "default");
    if (strcmp(nms_kind, "default") == 0) l.nms_kind = DEFAULT_NMS;
    else {
        if (strcmp(nms_kind, "greedynms") == 0) l.nms_kind = GREEDY_NMS;
        else if (strcmp(nms_kind, "diounms") == 0) l.nms_kind = DIOU_NMS;
        else l.nms_kind = DEFAULT_NMS;
        printf("nms_kind: %s (%d), beta = %f \n", nms_kind, l.nms_kind, l.beta_nms);
    }

    l.atss =  option_find_int_quiet(options, "atss", 0);

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

layer parse_distill_yolo(list *options, size_params params) {
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);

    int distill_index = option_find_int(options, "distill_index", 0);
    layer l = make_distill_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, distill_index);
    assert(l.outputs == params.inputs);

    l.max_boxes = option_find_int_quiet(options, "max", 90);
    l.jitter = option_find_float(options, "jitter", .2);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.random = option_find_int_quiet(options, "random", 0);
    l.margin = option_find_float_quiet(options, "margin", 0.1);
    l.alpha = option_find_float_quiet(options, "alpha", 0.7);

    l.scale_xy = option_find_float_quiet(options, "scale_xy", 1);
    l.use_center_regression = option_find_int(options, "use_center_regression", 0);
    l.object_focal_loss = option_find_int(options, "object_focal_loss", 0);

    l.iou_normalizer = option_find_float_quiet(options, "iou_normalizer", 1);
    l.cls_normalizer = option_find_float_quiet(options, "cls_normalizer", 1);

    l.label_smooth_rate = option_find_float_quiet(options, "label_smooth_rate", 0);

    char *iou_loss = option_find_str_quiet(options, "iou_loss", "mse");   //
    if (strcmp(iou_loss, "mse") == 0) l.iou_loss = MSE;
    else if (strcmp(iou_loss, "giou") == 0) l.iou_loss = GIOU;
    else if (strcmp(iou_loss, "diou") == 0) l.iou_loss = DIOU;
    else if (strcmp(iou_loss, "ciou") == 0) l.iou_loss = CIOU;
    else l.iou_loss = IOU;

    char *iou_thresh_kind_str = option_find_str_quiet(options, "iou_thresh_kind", "iou");
    if (strcmp(iou_thresh_kind_str, "iou") == 0) l.iou_thresh_kind = IOU;
    else if (strcmp(iou_thresh_kind_str, "giou") == 0) l.iou_thresh_kind = GIOU;
    else if (strcmp(iou_thresh_kind_str, "diou") == 0) l.iou_thresh_kind = DIOU;
    else if (strcmp(iou_thresh_kind_str, "ciou") == 0) l.iou_thresh_kind = CIOU;
    else {
        fprintf(stderr, " Wrong iou_thresh_kind = %s \n", iou_thresh_kind_str);
        l.iou_thresh_kind = IOU;
    }

    l.beta_nms = option_find_float_quiet(options, "beta_nms", 0.6);
    char *nms_kind = option_find_str_quiet(options, "nms_kind", "default");
    if (strcmp(nms_kind, "default") == 0) l.nms_kind = DEFAULT_NMS;
    else {
        if (strcmp(nms_kind, "greedynms") == 0) l.nms_kind = GREEDY_NMS;
        else if (strcmp(nms_kind, "diounms") == 0) l.nms_kind = DIOU_NMS;
        else l.nms_kind = DEFAULT_NMS;
        printf("nms_kind: %s (%d), beta = %f \n", nms_kind, l.nms_kind, l.beta_nms);
    }

    l.atss =  option_find_int_quiet(options, "atss", 0);

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        //printf("anchor: ");
        for (i = 0; i < n; ++i) {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',') + 1;
            //printf("%f ", bias);
        }
        //printf("\n");
    }
    return l;
}

layer parse_mutual_yolo(list *options, size_params params) {
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);

    int mutual_index = option_find_int(options, "mutual_index", 0);
    layer l = make_mutual_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, mutual_index);
    assert(l.outputs == params.inputs);

    l.max_boxes = option_find_int_quiet(options, "max", 90);
    l.jitter = option_find_float(options, "jitter", .2);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.random = option_find_int_quiet(options, "random", 0);
    l.margin = option_find_float_quiet(options, "margin", 0.1);
    l.alpha = option_find_float_quiet(options, "alpha", 0.7);

    l.scale_xy = option_find_float_quiet(options, "scale_xy", 1);
    l.use_center_regression = option_find_int(options, "use_center_regression", 0);
    l.object_focal_loss = option_find_int(options, "object_focal_loss", 0);

    l.iou_normalizer = option_find_float_quiet(options, "iou_normalizer", 1);
    l.cls_normalizer = option_find_float_quiet(options, "cls_normalizer", 1);

    l.label_smooth_rate = option_find_float_quiet(options, "label_smooth_rate", 0);

    char *iou_loss = option_find_str_quiet(options, "iou_loss", "mse");   //
    if (strcmp(iou_loss, "mse") == 0) l.iou_loss = MSE;
    else if (strcmp(iou_loss, "giou") == 0) l.iou_loss = GIOU;
    else if (strcmp(iou_loss, "diou") == 0) l.iou_loss = DIOU;
    else if (strcmp(iou_loss, "ciou") == 0) l.iou_loss = CIOU;
    else l.iou_loss = IOU;

    char *iou_thresh_kind_str = option_find_str_quiet(options, "iou_thresh_kind", "iou");
    if (strcmp(iou_thresh_kind_str, "iou") == 0) l.iou_thresh_kind = IOU;
    else if (strcmp(iou_thresh_kind_str, "giou") == 0) l.iou_thresh_kind = GIOU;
    else if (strcmp(iou_thresh_kind_str, "diou") == 0) l.iou_thresh_kind = DIOU;
    else if (strcmp(iou_thresh_kind_str, "ciou") == 0) l.iou_thresh_kind = CIOU;
    else {
        fprintf(stderr, " Wrong iou_thresh_kind = %s \n", iou_thresh_kind_str);
        l.iou_thresh_kind = IOU;
    }

    l.beta_nms = option_find_float_quiet(options, "beta_nms", 0.6);
    char *nms_kind = option_find_str_quiet(options, "nms_kind", "default");
    if (strcmp(nms_kind, "default") == 0) l.nms_kind = DEFAULT_NMS;
    else {
        if (strcmp(nms_kind, "greedynms") == 0) l.nms_kind = GREEDY_NMS;
        else if (strcmp(nms_kind, "diounms") == 0) l.nms_kind = DIOU_NMS;
        else l.nms_kind = DEFAULT_NMS;
        printf("nms_kind: %s (%d), beta = %f \n", nms_kind, l.nms_kind, l.beta_nms);
    }

    l.atss =  option_find_int_quiet(options, "atss", 0);

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        //printf("anchor: ");
        for (i = 0; i < n; ++i) {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',') + 1;
            //printf("%f ", bias);
        }
        //printf("\n");
    }
    return l;
}

layer parse_mimicutual_yolo(list *options, size_params params) {
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);

    int distill_index = option_find_int(options, "distill_index", 0);
    int mutual_index = option_find_int(options, "mutual_index", 0);
    layer l = make_mimicutual_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, distill_index, mutual_index);
    assert(l.outputs == params.inputs);

    l.max_boxes = option_find_int_quiet(options, "max", 90);
    l.jitter = option_find_float(options, "jitter", .2);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.random = option_find_int_quiet(options, "random", 0);
    l.margin = option_find_float_quiet(options, "margin", 0.1);
    l.alpha = option_find_float_quiet(options, "alpha", 0.7);

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        //printf("anchor: ");
        for (i = 0; i < n; ++i) {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',') + 1;
            //printf("%f ", bias);
        }
        //printf("\n");
    }
    return l;
}

layer parse_double_yolo(list *options, size_params params)
{

    char *layerstr = option_find(options, "layers");
    int len = strlen(layerstr);
    if(!layerstr) error("Double_yolo Layer must specify input layers");
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (layerstr[i] == ',') ++n;
    }

    int *layers = calloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index = atoi(layerstr);
        layerstr = strchr(layerstr, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
    }

    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);
    layer l = make_double_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, layers);

    //printf("%d %d\n", l.outputs, params.inputs);
    //assert(l.outputs == params.inputs);

    l.max_boxes = option_find_int_quiet(options, "max",90);
    l.jitter = option_find_float(options, "jitter", .2);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.random = option_find_int_quiet(options, "random", 0);
    l.rescore = option_find_int_quiet(options, "rescore", 0);
    l.lb_dis_max_thresh = option_find_float_quiet(options, "lb_dis_max_thresh", 10);
    l.lb_dis_ignore_thresh = option_find_float_quiet(options, "lb_dis_ignore_thresh", 5);
    l.lb_dis_truth_thresh = option_find_float_quiet(options, "lb_dis_truth_thresh", 0);
    l.scale_xy = option_find_float_quiet(options, "scale_xy", 1);

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }

    return l;
}

layer parse_heatmap(list *options, size_params params)
{
    int keypoints_num = option_find_int(options, "keypoints_num", 5);
    layer l = make_heatmap_layer(params.batch, params.w, params.h, keypoints_num);
    assert(l.outputs == params.inputs);
    l.max_boxes = option_find_int_quiet(options, "max", 90);
    l.truths = l.max_boxes * (4 + keypoints_num * 3 + 1);
    l.alpha = option_find_float_quiet(options, "alpha", 2);
    l.beta = option_find_float_quiet(options, "beta", 4);
    l.scale = option_find_float_quiet(options, "scale", 1.);
    return l;
}

layer parse_iseg(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);
    int ids = option_find_int(options, "ids", 32);
    layer l = make_iseg_layer(params.batch, params.w, params.h, classes, ids);
    assert(l.outputs == params.inputs);
    return l;
}

layer parse_region(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 4);
    int classes = option_find_int(options, "classes", 20);
    int num = option_find_int(options, "num", 1);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
    assert(l.outputs == params.inputs);

    l.log = option_find_int_quiet(options, "log", 0);
    l.sqrt = option_find_int_quiet(options, "sqrt", 0);

    l.softmax = option_find_int(options, "softmax", 0);
    l.background = option_find_int_quiet(options, "background", 0);
    l.max_boxes = option_find_int_quiet(options, "max",30);
    l.jitter = option_find_float(options, "jitter", .2);
    l.rescore = option_find_int_quiet(options, "rescore",0);

    l.thresh = option_find_float(options, "thresh", .5);
    l.classfix = option_find_int_quiet(options, "classfix", 0);
    l.absolute = option_find_int_quiet(options, "absolute", 0);
    l.random = option_find_int_quiet(options, "random", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.mask_scale = option_find_float(options, "mask_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.bias_match = option_find_int_quiet(options, "bias_match",0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    char *a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

detection_layer parse_detection(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 1);
    int classes = option_find_int(options, "classes", 1);
    int rescore = option_find_int(options, "rescore", 0);
    int num = option_find_int(options, "num", 1);
    int side = option_find_int(options, "side", 7);
    detection_layer layer = make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

    layer.softmax = option_find_int(options, "softmax", 0);
    layer.sqrt = option_find_int(options, "sqrt", 0);

    layer.max_boxes = option_find_int_quiet(options, "max",90);
    layer.coord_scale = option_find_float(options, "coord_scale", 1);
    layer.forced = option_find_int(options, "forced", 0);
    layer.object_scale = option_find_float(options, "object_scale", 1);
    layer.noobject_scale = option_find_float(options, "noobject_scale", 1);
    layer.class_scale = option_find_float(options, "class_scale", 1);
    layer.jitter = option_find_float(options, "jitter", .2);
    layer.random = option_find_int_quiet(options, "random", 0);
    layer.reorg = option_find_int_quiet(options, "reorg", 0);
    return layer;
}

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale",1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
    layer.ratio =  option_find_float_quiet(options, "ratio",0);
    layer.noobject_scale =  option_find_float_quiet(options, "noobj", 1);
    layer.thresh =  option_find_float_quiet(options, "thresh",0);
    return layer;
}

hint_cost_layer parse_hint_cost(list *options, size_params params) {
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    int hint_index = option_find_int(options, "hint_index", 0);
    float scale = option_find_float_quiet(options, "scale", 1);
    float margin = option_find_float_quiet(options, "mutual_margin", 0);
    hint_cost_layer layer = make_hint_cost_layer(params.batch, params.inputs, type, scale, hint_index, margin);
    layer.ratio = option_find_float_quiet(options, "ratio", 0);
    layer.noobject_scale = option_find_float_quiet(options, "noobj", 1);
    layer.thresh = option_find_float_quiet(options, "thresh", 0);
    return layer;
}

crop_layer parse_crop(list *options, size_params params)
{
    int crop_height = option_find_int(options, "crop_height",1);
    int crop_width = option_find_int(options, "crop_width",1);
    int flip = option_find_int(options, "flip",0);
    float angle = option_find_float(options, "angle",0);
    float saturation = option_find_float(options, "saturation",1);
    float exposure = option_find_float(options, "exposure",1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before crop layer must output image.");

    int noadjust = option_find_int_quiet(options, "noadjust",0);

    crop_layer l = make_crop_layer(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);
    l.shift = option_find_float(options, "shift", 0);
    l.noadjust = noadjust;
    return l;
}

layer parse_reorg(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int reverse = option_find_int_quiet(options, "reverse",0);
    int flatten = option_find_int_quiet(options, "flatten",0);
    int extra = option_find_int_quiet(options, "extra",0);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before reorg layer must output image.");

    layer layer = make_reorg_layer(batch,w,h,c,stride,reverse, flatten, extra);
    return layer;
}

maxpool_layer parse_maxpool(list *options, size_params params, network *net)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch,h,w,c,size,stride,padding);
    layer.post_training_quantization = option_find_int_quiet(options, "post_training_quantization", net->post_training_quantization);
    return layer;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before avgpool layer must output image.");

    avgpool_layer layer = make_avgpool_layer(batch,w,h,c);
    return layer;
}

dropout_layer parse_dropout(list *options, size_params params)
{
    float probability = option_find_float(options, "probability", .5);
    dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}

layer parse_normalization(list *options, size_params params)
{
    float alpha = option_find_float(options, "alpha", .0001);
    float beta =  option_find_float(options, "beta" , .75);
    float kappa = option_find_float(options, "kappa", 1);
    int size = option_find_int(options, "size", 5);
    layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    return l;
}

layer parse_batchnorm(list *options, size_params params)
{
    layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.c);
    return l;
}

layer parse_shortcut(list *options, size_params params, network *net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if(index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net->layers[index];

    layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    s.alpha = option_find_float_quiet(options, "alpha", 1);
    s.beta = option_find_float_quiet(options, "beta", 1);
    s.leaky_rate = option_find_float_quiet(options, "leaky_rate", .1);
    return s;
}


layer parse_l2norm(list *options, size_params params)
{
    layer l = make_l2norm_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}


layer parse_logistic(list *options, size_params params)
{
    layer l = make_logistic_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}

layer parse_activation(list *options, size_params params)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    layer l = make_activation_layer(params.batch, params.inputs, activation);

    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;

    return l;
}

layer parse_upsample(list *options, size_params params, network *net)
{

    int stride = option_find_int(options, "stride",2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, "scale", 1);

    l.post_training_quantization = option_find_int_quiet(options, "post_training_quantization", net->post_training_quantization);
    l.quantize_per_channel = option_find_int_quiet(options, "quantize_per_channel", net->quantize_per_channel);
    if (l.post_training_quantization) {
        l.x_fl = calloc(1, sizeof(int));
        *(l.x_fl) = 0;
    }

    return l;
}

route_layer parse_route(list *options, size_params params, network *net)
{
    char *l = option_find(options, "layers");
    int len = strlen(l);
    if(!l) error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }

    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
    int batch = params.batch;

    route_layer layer = make_route_layer(batch, n, layers, sizes);

    convolutional_layer first = net->layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for(i = 1; i < n; ++i){
        int index = layers[i];
        convolutional_layer next = net->layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            layer.out_c += next.out_c;
        }else{
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }

    layer.post_training_quantization = option_find_int_quiet(options, "post_training_quantization", net->post_training_quantization);
    layer.quantize_per_channel = option_find_int_quiet(options, "quantize_per_channel", net->quantize_per_channel);
    if (layer.post_training_quantization) {
        layer.x_fl = calloc(1, sizeof(int));
        *(layer.x_fl) = 0;
    }

    return layer;
}

reweight_layer parse_reweight(list *options, size_params params, network *net)
{
    char *l = option_find(options, "layers");
    int len = strlen(l);
    if(!l) error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    // here we just ensure that y = a * x, so the input layer will only be scale layer and data layer
    assert(n == 2);

    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
    int batch = params.batch;

    reweight_layer layer = make_reweight_layer(batch, n, layers, sizes, activation);

    convolutional_layer first = net->layers[layers[0]];
    convolutional_layer second = net->layers[layers[1]];
    if (first.out_c == second.out_c) {
        layer.out_w = second.out_w;
        layer.out_h = second.out_h;
        layer.out_c = second.out_c;
    } else {
        layer.out_w = layer.out_h = layer.out_c = 0;
    }

    return layer;
}

channel_slice_layer parse_channel_slice(list *options, size_params params, network *net){

    int batch, h, w;
    h = params.h;
    w = params.w;
    //c = params.c;
    batch = params.batch;
    char *l = option_find(options, "from");
    int len = strlen(l);
    if(!l) error("Channel Slice Layer must specify input layers");
    int begin_slice_point = option_find_int(options, "start", 2);
    int end_slice_point = option_find_int(options, "end", 2);
    int axis = option_find_int(options, "axis", 1);
    int n = 1;
    int *layers = (int*)calloc(n,sizeof(int));
    int* sizes = (int*)calloc(n, sizeof(int));
    int *c = (int*)calloc(n, sizeof(int));
    for(int i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
        c[i] = net->layers[index].out_c;
    }
    channel_slice_layer layer = make_channel_slice_layer(batch, w, h, c[0], begin_slice_point, end_slice_point, axis, n, layers, sizes);
    return layer;
}

layer parse_channel_shuffle(list *options, size_params params){

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    int groups = option_find_int(options, "groups", 2);
    layer l = make_channel_shuffle_layer(batch, w, h, c, groups);
    return l;
}

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random")==0) return RANDOM;
    if (strcmp(s, "poly")==0) return POLY;
    if (strcmp(s, "constant")==0) return CONSTANT;
    if (strcmp(s, "step")==0) return STEP;
    if (strcmp(s, "exp")==0) return EXP;
    if (strcmp(s, "sigmoid")==0) return SIG;
    if (strcmp(s, "steps")==0) return STEPS;
    if (strcmp(s, "cosine") == 0) return COSINE;
    if (strcmp(s, "sgdr") == 0) return SGDR;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);

    net->learning_rate_min = option_find_float_quiet(options, "learning_rate_min", .00001);
    net->batches_per_cycle = option_find_int_quiet(options, "sgdr_cycle", 1000);
    net->batches_cycle_mult = option_find_int_quiet(options, "sgdr_mult", 2);

    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->notruth = option_find_int_quiet(options, "notruth",0);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->random = option_find_int_quiet(options, "random", 0);

    net->adam = option_find_int_quiet(options, "adam", 0);
    if(net->adam){
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .0000001);
    }

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop", net->w*2);
    net->min_crop = option_find_int_quiet(options, "min_crop", net->w);
    net->max_ratio = option_find_float_quiet(options, "max_ratio", (float) net->max_crop / net->w);
    net->min_ratio = option_find_float_quiet(options, "min_ratio", (float) net->min_crop / net->w);
    net->center = option_find_int_quiet(options, "center",0);
    net->clip = option_find_float_quiet(options, "clip", 0);

    // augment args
    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);
    net->openscale = option_find_int_quiet(options, "openscale", 0);

    net->train_slimming = option_find_int_quiet(options, "train_slimming", 0);
    net->consistent_slimming = option_find_int_quiet(options, "consistent_slimming", 0);
    net->slimming_scale = option_find_float_quiet(options, "slimming_scale", 0.0001);
    net->slimming_alpha = option_find_float_quiet(options, "slimming_alpha", 0.1);

    net->poly_slimming = option_find_int_quiet(options, "poly_slimming", 1);
    net->slimming_min_scale = option_find_float_quiet(options, "slimming_min_scale", 0.1);
    net->slimming_max_scale = option_find_float_quiet(options, "slimming_max_scale", 1.0);;
    net->slimming_start_batch = option_find_int_quiet(options, "slimming_start_batch", 0);

    net->filter_thresh = option_find_float_quiet(options, "filter_thresh", 0);

     // 0: original, 1: mixup, 2: cutmix, 3: mosaic, 4: mosaic + cutmix, ...
    net->data_fusion_type = option_find_int_quiet(options, "data_fusion_type", 0);
    // fusion_prob for each image
    net->data_fusion_prob = option_find_float_quiet(options, "data_fusion_prob", 0);
    net->mosaic_min_offset = option_find_float_quiet(options, "mosaic_min_offset", 0.2);

    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    net->power = option_find_float_quiet(options, "power", 4);
    if(net->policy == STEP){
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    } else if (net->policy == STEPS){
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        if(!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (l[i] == ',') ++n;
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
    } else if (net->policy == COSINE) {
        net->alpha = option_find_float(options, "alpha", 0);
    }
    net->max_batches = option_find_int(options, "max_batches", 0);

    char *l = option_find_str(options, "hint_layers", "-1,-1,-1");
    char *p = option_find_str(options, "distill_layers", "-1,-1,-1");
    char *q = option_find_str(options, "mutual_layers", "-1,-1,-1");
    //if (!l || !p) error("mimic_train must have hint_layers and distill_layers in cfg file.");

    net->log_step = option_find_int(options, "log_step", 1);

    int len = strlen(l);
    int n = 1;
    int i;
    for (i = 0; i < len; ++i) {
        if (l[i] == ',') ++n;
    }
    int *hint_layers = calloc(n, sizeof(int));
    int *distill_layers = calloc(n, sizeof(int));
    int *mutual_layers = calloc(n, sizeof(int));
    for (i = 0; i < n; ++i) {
        int hint_layer = atoi(l);
        int distill_layer = atoi(p);
        int mutual_layer = atoi(q);
        l = strchr(l, ',') + 1;
        p = strchr(p, ',') + 1;
        q = strchr(q, ',') + 1;
        hint_layers[i] = hint_layer;
        distill_layers[i] = distill_layer;
        mutual_layers[i] = mutual_layer;
    }
    net->hint_layers = hint_layers;
    net->distill_layers = distill_layers;
    net->mutual_layers = mutual_layers;
    net->num_mimic_layer = n;

    net->quantize = option_find_int_quiet(options, "quantize", 0);
    net->post_training_quantization = option_find_int_quiet(options, "post_training_quantization", 0);
    net->quantization_aware_training = option_find_int_quiet(options, "quantization_aware_training", 0);
    assert(net->quantize == net->post_training_quantization + net->quantization_aware_training);
    net->transfer_input = option_find_int_quiet(options, "transfer_input", 0);
    net->transfer_todct = option_find_int_quiet(options, "transfer_todct", 0);
    net->dct_onlyY = option_find_int_quiet(options, "dct_onlyY", 1);

    net->convx_bias_align = option_find_int_quiet(options, "convx_bias_align", 0);
    net->write_statistic_fl = option_find_int_quiet(options, "write_statistic_fl", 0);
    net->write_input = option_find_int_quiet(options, "write_input", 0);
    net->write_results = option_find_int_quiet(options, "write_results", 0);
    net->write_yolo_output = option_find_int_quiet(options, "write_yolo_output", 0);
    net->write_statistic_features = option_find_int_quiet(options, "write_statistic_features", 0);

    net->quantize_weight_bitwidth = option_find_int_quiet(options, "weight_bitwidth", 8);
    net->quantize_weight_fraction_bitwidth = option_find_int_quiet(options, "weight_fraction_bitwidth", 6);
    net->quantize_feature_bitwidth = option_find_int_quiet(options, "feature_bitwidth", 8);
    net->quantize_feature_fraction_bitwidth = option_find_int_quiet(options, "feature_fraction_bitwidth", 3);
    net->quantize_bias_bitwidth = option_find_int_quiet(options, "bias_bitwidth", 16);
    net->quantize_bias_fraction_bitwidth = option_find_int_quiet(options, "bias_fraction_bitwidth", 10);
    net->quantize_freezeBN_iterpoint = option_find_int_quiet(options, "quantize_freezeBN_iterpoint", 30000);

    net->quantize_per_channel = option_find_int_quiet(options, "per_channel", 0);
    net->quantize_per_channel_keep_rate = option_find_float_quiet(options, "per_channel_keep_rate", 0.8);


    net->downsample_scale = option_find_int_quiet(options, "downsample_scale", 32);

}

int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}

network *parse_network_cfg(char *filename)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections");
    network *net = make_network(sections->size - 1);
    net->gpu_index = gpu_index;
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, net);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;
    if (net->transfer_todct) {
        params.h = net->h / 8;
        params.w = net->w / 8;
        params.c = 32;
    }

    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");
    while(n){
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = {0};
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(options, params, net);
        }else if(lt == DECONVOLUTIONAL){
            l = parse_deconvolutional(options, params);
        }else if(lt == LOCAL){
            l = parse_local(options, params);
        }else if(lt == ACTIVE){
            l = parse_activation(options, params);
        }else if(lt == LOGXENT){
            l = parse_logistic(options, params);
        }else if(lt == L2NORM){
            l = parse_l2norm(options, params);
        }else if(lt == RNN){
            l = parse_rnn(options, params);
        }else if(lt == GRU){
            l = parse_gru(options, params);
        }else if (lt == LSTM) {
            l = parse_lstm(options, params);
        }else if(lt == CRNN){
            l = parse_crnn(options, params);
        }else if(lt == CONNECTED){
            l = parse_connected(options, params);
        }else if(lt == CROP){
            l = parse_crop(options, params);
        }else if(lt == COST){
            l = parse_cost(options, params);
        }else if (lt == HINT_COST) {
            l = parse_hint_cost(options, params);
        }else if(lt == REGION){
            l = parse_region(options, params);
        }else if(lt == YOLO){
            l = parse_yolo(options, params, net);
        }else if(lt == KEYPOINT_YOLO){
            l = parse_keypoint_yolo(options, params, net);
        }else if(lt == DISTILL_YOLO) {
            l = parse_distill_yolo(options, params);
        }else if(lt == MUTUAL_YOLO) {
            l = parse_mutual_yolo(options, params);
        }else if(lt == MIMICUTUAL_YOLO) {
            l = parse_mimicutual_yolo(options, params);
        }else if(lt == DOUBLE_YOLO) {
            l = parse_double_yolo(options, params);
        }else if(lt == HEATMAP) {
            l = parse_heatmap(options, params);
        }else if(lt == ISEG){
            l = parse_iseg(options, params);
        }else if(lt == DETECTION){
            l = parse_detection(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
            net->hierarchy = l.softmax_tree;
        }else if(lt == NORMALIZATION){
            l = parse_normalization(options, params);
        }else if(lt == BATCHNORM){
            l = parse_batchnorm(options, params);
        }else if(lt == MAXPOOL){
            l = parse_maxpool(options, params, net);
        }else if(lt == REORG){
            l = parse_reorg(options, params);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else if(lt == ROUTE){
            l = parse_route(options, params, net);
        }else if(lt == REWEIGHT){
            l = parse_reweight(options, params, net);
        }else if(lt == CHANNEL_SLICE) {
            l = parse_channel_slice(options, params, net);
        }else if(lt == CHANNEL_SHUFFLE) {
            l = parse_channel_shuffle(options, params);
        }else if(lt == UPSAMPLE){
            l = parse_upsample(options, params, net);
        }else if(lt == SHORTCUT){
            l = parse_shortcut(options, params, net);
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params);
            l.output = net->layers[count-1].output;
            l.delta = net->layers[count-1].delta;
#ifdef GPU
            l.output_gpu = net->layers[count-1].output_gpu;
            l.delta_gpu = net->layers[count-1].delta_gpu;
#endif
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        l.clip = net->clip;
        l.truth = option_find_int_quiet(options, "truth", 0);
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.dontsave = option_find_int_quiet(options, "dontsave", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.skipload = option_find_int_quiet(options, "skipload", 0);
        l.skipfilters = option_find_int_quiet(options, "skipfilters", 0);
        l.numload = option_find_int_quiet(options, "numload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);
        option_unused(options);
        net->layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    layer out = get_network_output_layer(net);
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    int size = net->inputs;
    if (net->transfer_todct) {
        size = net->layers[0].inputs;
    }
    net->input = calloc(size*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
    net->output_gpu = out.output_gpu;
    net->input_gpu = cuda_make_array(net->input, size*net->batch);
    net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
#endif
    if(workspace_size){
        //printf("%ld\n", workspace_size);
#ifdef GPU
        if(gpu_index >= 0){
            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }else {
            net->workspace = calloc(1, workspace_size);
        }
#else
        net->workspace = calloc(1, workspace_size);
#endif
    }
    return net;
}

list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = malloc(sizeof(section));
                list_insert(options, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

void save_convolutional_weights_binary(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
    int size = l.c*l.size*l.size;
    int i, j, k;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    for(i = 0; i < l.n; ++i){
        float mean = l.binary_weights[i*size];
        if(mean < 0) mean = -mean;
        fwrite(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                if (l.binary_weights[index + k] > 0) c = (c | 1<<k);
            }
            fwrite(&c, sizeof(char), 1, fp);
        }
    }
}

void save_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    int num = l.nweights;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_batchnorm_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_batchnorm_layer(l);
    }
#endif
    fwrite(l.scales, sizeof(float), l.c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_connected_layer(l);
    }
#endif
    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}

void save_weights_upto(network *net, char *filename, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    int major = 0;
    int minor = 2;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(size_t), 1, fp);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            save_convolutional_weights(l, fp);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
        } if(l.type == BATCHNORM){
            save_batchnorm_weights(l, fp);
        } if(l.type == RNN){
            save_connected_weights(*(l.input_layer), fp);
            save_connected_weights(*(l.self_layer), fp);
            save_connected_weights(*(l.output_layer), fp);
        } if (l.type == LSTM) {
            save_connected_weights(*(l.wi), fp);
            save_connected_weights(*(l.wf), fp);
            save_connected_weights(*(l.wo), fp);
            save_connected_weights(*(l.wg), fp);
            save_connected_weights(*(l.ui), fp);
            save_connected_weights(*(l.uf), fp);
            save_connected_weights(*(l.uo), fp);
            save_connected_weights(*(l.ug), fp);
        } if (l.type == GRU) {
            if(1){
                save_connected_weights(*(l.wz), fp);
                save_connected_weights(*(l.wr), fp);
                save_connected_weights(*(l.wh), fp);
                save_connected_weights(*(l.uz), fp);
                save_connected_weights(*(l.ur), fp);
                save_connected_weights(*(l.uh), fp);
            }else{
                save_connected_weights(*(l.reset_layer), fp);
                save_connected_weights(*(l.update_layer), fp);
                save_connected_weights(*(l.state_layer), fp);
            }
        }  if(l.type == CRNN){
            save_convolutional_weights(*(l.input_layer), fp);
            save_convolutional_weights(*(l.self_layer), fp);
            save_convolutional_weights(*(l.output_layer), fp);
        } if(l.type == LOCAL){
#ifdef GPU
            if(gpu_index >= 0){
                pull_local_layer(l);
            }
#endif
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fwrite(l.biases, sizeof(float), l.outputs, fp);
            fwrite(l.weights, sizeof(float), size, fp);
        }
    }
    fclose(fp);
}

void save_weights(network *net, char *filename)
{
    save_weights_upto(net, filename, net->n);
}

void transpose_matrix(float *a, int rows, int cols)
{
    float *transpose = calloc(rows*cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if(transpose){
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    //printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
    //printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_connected_layer(l);
    }
#endif
}

void load_batchnorm_weights(layer l, FILE *fp)
{
    fread(l.scales, sizeof(float), l.c, fp);
    fread(l.rolling_mean, sizeof(float), l.c, fp);
    fread(l.rolling_variance, sizeof(float), l.c, fp);
#ifdef GPU
    if(gpu_index >= 0){
        push_batchnorm_layer(l);
    }
#endif
}

void load_convolutional_weights_binary(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    int size = l.c*l.size*l.size;
    int i, j, k;
    for(i = 0; i < l.n; ++i){
        float mean = 0;
        fread(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            fread(&c, sizeof(char), 1, fp);
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                l.weights[index + k] = (c & 1<<k) ? mean : -mean;
            }
        }
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void load_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    if(l.numload) l.n = l.numload;
    int num = l.c/l.groups*l.n*l.size*l.size;
    if (l.skipload) {
        fseek(fp, sizeof(float)*l.skipfilters, 1);
    } else {
        fread(l.biases, sizeof(float), l.n, fp);
    }

    if(l.post_training_quantization) {
        if (l.quantize_per_channel) {
            fread(l.bias_fls, sizeof(int), l.n, fp);
        } else {
            fread(l.bias_fl, sizeof(int), 1, fp);
            //printf("l.bias_fl: %d ", *(l.bias_fl));
        }
    }

    if (l.batch_normalize && (!l.dontloadscales)){
        //printf("load batchnorm\n");
        if (l.skipload) {
            fseek(fp, sizeof(float)*l.skipfilters*3, 1);
        }
        else {
            fread(l.scales, sizeof(float), l.n, fp);
            fread(l.rolling_mean, sizeof(float), l.n, fp);
            fread(l.rolling_variance, sizeof(float), l.n, fp);
        }
        //printf("l.scales: %f %f %f\n", l.scales[0], l.scales[1], l.scales[2]);
        //printf("l.rolling_mean: %f %f %f\n", l.rolling_mean[0], l.rolling_mean[1], l.rolling_mean[2]);
        //printf("l.rolling_variance: %f %f %f\n", l.rolling_variance[0], l.rolling_variance[1], l.rolling_variance[2]);
        if(0){
            int i;
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
        if(0){
            fill_cpu(l.n, 0, l.rolling_mean, 1);
            fill_cpu(l.n, 0, l.rolling_variance, 1);
        }
        if(0){
            int i;
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
    }

    if (l.skipload) {
        int skipnum = l.c/l.groups*l.skipfilters*l.size*l.size;
        fseek(fp, sizeof(float)*skipnum, 1);
    }
    else {
        fread(l.weights, sizeof(float), num, fp);
    }

    if(l.post_training_quantization) {
        if (l.quantize_per_channel) {
            fread(l.conv_fls, sizeof(int), l.n, fp);
        } else {
            fread(l.conv_fl, sizeof(int), 1, fp);
        }
        //printf("l.conv_fl: %d\n", *(l.conv_fl));
    }

    // CHECK /255 for the first convolution in quantization_aware_training
    if(l.quantization_aware_training && l.scale_weight) {
        scal_cpu(l.nweights, 1 / 255., l.weights, 1);
    }

    //printf("l.weights: %f %f %f\n", l.weights[0], l.weights[1], l.weights[2]);

    //if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    if (l.flipped) {
        transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
    }
    //if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}


void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(net->seen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
        }
        if(l.type == BATCHNORM){
            load_batchnorm_weights(l, fp);
        }
        if(l.type == CRNN){
            load_convolutional_weights(*(l.input_layer), fp);
            load_convolutional_weights(*(l.self_layer), fp);
            load_convolutional_weights(*(l.output_layer), fp);
        }
        if(l.type == RNN){
            load_connected_weights(*(l.input_layer), fp, transpose);
            load_connected_weights(*(l.self_layer), fp, transpose);
            load_connected_weights(*(l.output_layer), fp, transpose);
        }
        if (l.type == LSTM) {
            load_connected_weights(*(l.wi), fp, transpose);
            load_connected_weights(*(l.wf), fp, transpose);
            load_connected_weights(*(l.wo), fp, transpose);
            load_connected_weights(*(l.wg), fp, transpose);
            load_connected_weights(*(l.ui), fp, transpose);
            load_connected_weights(*(l.uf), fp, transpose);
            load_connected_weights(*(l.uo), fp, transpose);
            load_connected_weights(*(l.ug), fp, transpose);
        }
        if (l.type == GRU) {
            if(1){
                load_connected_weights(*(l.wz), fp, transpose);
                load_connected_weights(*(l.wr), fp, transpose);
                load_connected_weights(*(l.wh), fp, transpose);
                load_connected_weights(*(l.uz), fp, transpose);
                load_connected_weights(*(l.ur), fp, transpose);
                load_connected_weights(*(l.uh), fp, transpose);
            }else{
                load_connected_weights(*(l.reset_layer), fp, transpose);
                load_connected_weights(*(l.update_layer), fp, transpose);
                load_connected_weights(*(l.state_layer), fp, transpose);
            }
        }
        if(l.type == LOCAL){
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.weights, sizeof(float), size, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_local_layer(l);
            }
#endif
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, 0, net->n);
}

void fuse_conv_batchnorm(network *net)
{
    int j;
    for (j = 0; j < net->n; ++j) {

        layer *l = &net->layers[j];
        if (l->type == CONVOLUTIONAL) {

            if (l->batch_normalize) {
                int f;
                for (f = 0; f < l->n; ++f)
                {
                    l->biases[f] = l->biases[f] - (double)l->scales[f] * l->rolling_mean[f] / (sqrt((double)l->rolling_variance[f] + .00001));

                    const size_t filter_size = l->size*l->size*l->c / l->groups;
                    int i;
                    for (i = 0; i < filter_size; ++i) {
                        int w_index = f*filter_size + i;
                        l->weights[w_index] = (double)l->weights[w_index] * l->scales[f] / (sqrt((double)l->rolling_variance[f] + .00001));
                    }
                }

                l->batch_normalize = 0;
#ifdef GPU
                if (gpu_index >= 0) {
                    push_convolutional_layer(*l);
                }
#endif
            }
        }
        else {
            //printf(" Fusion skip layer type: %d \n", l->type);
        }
    }
}

int get_appr_range(float *X, int n, int bitwidth, float keep_rate) {
    //printf("enter get_appr_range\n");
    float Xmax, Xmin;
    int i, j, total_shift, appr_shift;

    get_max_min(X, n, &Xmax, &Xmin);
    //printf("printf max %f, min %f\n", Xmax, Xmin);

    total_shift = ceil(log(fabs(Xmax) > fabs(Xmin) ? fabs(Xmax):fabs(Xmin)) / log(2));
    //total_bound = pow(2, total_shift);
    //printf("printf %d %f\n", total_shift, total_bound);

    appr_shift = total_shift;
    for (i = 1; i <= 5; ++i) {
        float bound = pow(2, total_shift - i);
        float precsion = pow(2, -(bitwidth - 1 - (total_shift - i)));
        //printf("printf bound %f, precsion %f\n", bound, precsion);
        // sum the num of X range in [-bound, bound - precsion]
        int count = 0;
        for (j = 0; j < n; ++j) {
            if (X[j] >= -bound && X[j] <= bound - precsion)
                count++;
        }
        //printf("count %d, keep: %f\n", count, n * keep_rate);
        if (count >= n * keep_rate) {
            appr_shift = total_shift - i;
        }
    }
    //printf("appr_shift: %d\n", bitwidth - 1 - appr_shift);
    return (bitwidth - 1 - appr_shift);
}

void calculate_appr_fracs_network(network *net) {
    int i, j, k;
    int net_fl = 0;
    for(i = 0; i < net->n; ++i) {
        layer *l = &net->layers[i];
        if (l->type == CONVOLUTIONAL) {
            //printf("convolutional %d\n", i);
            const size_t filter_size = l->size*l->size*l->c / l->groups;
            float *merge_a_per_channel = calloc(filter_size, sizeof(float));
            //float merge_b_per_channel;
            if (l->batch_normalize) {
                for (j = 0; j < l->n; ++j) {
                    for (k = 0; k < filter_size; ++k) {
                        int w_index = j*filter_size + k;
                        merge_a_per_channel[k] = (double)l->weights[w_index] * l->scales[j] / (sqrt((double)l->rolling_variance[j] + .00001));
                    }

                    //merge_b_per_channel = l->biases[j] - (double)l->scales[j] * l->rolling_mean[j] / (sqrt((double)l->rolling_variance[j] + .00001));

                    // get_appr_range of weights and bias
                    l->quantize_weight_fraction_bitwidths[j] = get_appr_range(merge_a_per_channel, filter_size, l->quantize_weight_bitwidth, l->quantize_per_channel_keep_rate);
                    // align bias to conv(x)
                    l->quantize_bias_fraction_bitwidths[j] = l->quantize_weight_fraction_bitwidths[j] + net_fl;
                    //printf("channel %d: %d %d\n", j, l->quantize_weight_fraction_bitwidths[j], l->quantize_bias_fraction_bitwidths[j]);
                }
            } else {
                for (j = 0; j < l->n; ++j) {
                    for (k = 0; k < filter_size; ++k) {
                        int w_index = j*filter_size + k;
                        merge_a_per_channel[k] = (double)l->weights[w_index];
                    }
                    //merge_b_per_channel = l->biases[j];

                    // get_appr_range of weights and bias
                    l->quantize_weight_fraction_bitwidths[j] = get_appr_range(merge_a_per_channel, filter_size, l->quantize_weight_bitwidth, l->quantize_per_channel_keep_rate);

                    // align bias to conv(x)
                    l->quantize_bias_fraction_bitwidths[j] = l->quantize_weight_fraction_bitwidths[j] + net_fl;
                    //printf("channel %d: %d %d\n", j, l->quantize_weight_fraction_bitwidths[j], l->quantize_bias_fraction_bitwidths[j]);
                }
            }
            free(merge_a_per_channel);
            net_fl = l->quantize_feature_fraction_bitwidth;
            //printf("net_fl: %d\n", net_fl);
        } else if (l->type == ROUTE) {
            int index = l->input_layers[0];
            if (net->layers[index].type == UPSAMPLE)
                index -= 1;
            net_fl = net->layers[index].quantize_feature_fraction_bitwidth;
        }
    }
}

void copy_appr_fracs_network(network *src, network *dst) {
    int j, k;
    // copy the quantize_weight_fraction_bitwidths/quantize_bias_fraction_bitwidths from src_net to dst_net
    for (j = 0; j < src->n; ++j) {
        if (src->layers[j].type == CONVOLUTIONAL) {
            for (k = 0; k < src->layers[j].n; ++k) {
                dst->layers[j].quantize_weight_fraction_bitwidths[k] = src->layers[j].quantize_weight_fraction_bitwidths[k];
                dst->layers[j].quantize_bias_fraction_bitwidths[k] = src->layers[j].quantize_bias_fraction_bitwidths[k];
            }
        }
    }
}

void calculate_appr_fracs_networks(network **nets, int n)
{
    int i;

    // we use network 0 for calculation
    network *rep = nets[0];
    calculate_appr_fracs_network(rep);

    // copy the quantize_weight_fraction_bitwidths/quantize_bias_fraction_bitwidths to all networks
    for (i = 1; i < n; ++i) {
        copy_appr_fracs_network(rep, nets[i]);
    }
}
