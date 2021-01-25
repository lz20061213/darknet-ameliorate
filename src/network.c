#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "keypoint_yolo_layer.h"
#include "distill_yolo_layer.h"
#include "mutual_yolo_layer.h"
#include "mimicutual_yolo_layer.h"
#include "double_yolo_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "hint_cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "parser.h"
#include "data.h"
#include "bgr2dct.h"

load_args get_base_args(network *net)
{
    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.size = net->w;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.openscale = net->openscale;
    args.filter_thresh = net->filter_thresh;
    args.data_fusion_type = net->data_fusion_type;
    args.data_fusion_prob = net->data_fusion_prob;
    args.mosaic_min_offset = net->mosaic_min_offset;
    args.keypoints_num = net->keypoints_num;
    return args;
}

network *load_network(char *cfg, char *weights, int clear)
{
    network *net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    if(clear) (*net->seen) = 0;
    return net;
}

size_t get_current_batch(network *net)
{
    size_t batch_num = (*net->seen)/(net->batch*net->subdivisions);
    return batch_num;
}

void reset_network_state(network *net, int b)
{
    int i;
    for (i = 0; i < net->n; ++i) {
        #ifdef GPU
        layer l = net->layers[i];
        if(l.state_gpu){
            fill_gpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
        }
        if(l.h_gpu){
            fill_gpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
        }
        #endif
    }
}

void reset_rnn(network *net)
{
    reset_network_state(net, 0);
}

float get_current_rate(network *net)
{
    size_t batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
    switch (net->policy) {
        case CONSTANT:
            return net->learning_rate;
        case STEP:
            return net->learning_rate * pow(net->scale, batch_num/net->step);
        case STEPS:
            rate = net->learning_rate;
            for(i = 0; i < net->num_steps; ++i){
                if(net->steps[i] > batch_num) return rate;
                rate *= net->scales[i];
            }
            return rate;
        case EXP:
            return net->learning_rate * pow(net->gamma, batch_num);
        case POLY:
            return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
        case RANDOM:
            return net->learning_rate * pow(rand_uniform(0,1), net->power);
        case SIG:
            return net->learning_rate * (1./(1.+exp(net->gamma*(batch_num - net->step))));
        case COSINE:
            rate = net->alpha;
            return net->learning_rate * ((1 - rate) * 0.5 * (1 + cos(TWO_PI / 2 * (float) batch_num / net->max_batches)) + rate);
        case SGDR:
        {
            int last_iter_start = 0;
            int cycle_size = net->batches_per_cycle;
            while ((last_iter_start + cycle_size) < batch_num)
            {
                last_iter_start += cycle_size;
                cycle_size *= net->batches_cycle_mult;
            }
            rate = net->learning_rate_min + 0.5*(net->learning_rate - net->learning_rate_min)*(1. + cos((float)(batch_num - last_iter_start)*TWO_PI/cycle_size));
            return rate;
        }
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net->learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case LSTM:
	    return "lstm";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case YOLO:
            return "yolo";
        case DISTILL_YOLO:
            return "distill_yolo";
        case MUTUAL_YOLO:
            return "mutual_yolo";
        case MIMICUTUAL_YOLO:
            return "mimicutual_yolo";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case HINT_COST:
            return "hint_cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case UPSAMPLE:
            return "upsample";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}

network *make_network(int n)
{
    network *net = calloc(1, sizeof(network));
    net->n = n;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    return net;
}

void forward_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        forward_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta){
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, net);
        net.input = l.output;
        if(l.truth) {
            net.truth = l.output;
        }
    }
    calc_network_cost(netp);
}

void mimic_forward_network(network *netp, network *netq) {
#ifdef GPU
    if (netp->gpu_index >= 0) {
        mimic_forward_network_gpu(netp, netq);
        return;
    }
#endif
    network snet = *netp;
    network tnet = *netq;
    // unimplemented
}

void mutual_forward_network(network *netp, network *netq) {
#ifdef GPU
    if (netp->gpu_index >= 0) {
        mutual_forward_network_gpu(netp, netq);
        return;
    }
#endif
    network snet = *netp;
    network pnet = *netq;
    // unimplemented
}

void mimicutual_forward_network(network *netp, network *netq, network *netr) {
#ifdef GPU
    if (netp->gpu_index >= 0) {
        mimicutual_forward_network_gpu(netp, netq, netr);
        return;
    }
#endif
    network snet = *netp;
    network pnet = *netq;
    network tnet = *netr;
    // unimplemented
}

void update_network(network *netp) {
#ifdef GPU
    if(netp->gpu_index >= 0){
        update_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = *net.t;

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){
            l.update(l, a);
        }
    }
}

void calc_network_cost(network *netp)
{
    network net = *netp;
    if (!net.train) return;

    int i, index;

    // COST, HINT_COST, YOLO, DISTILL_YOLO, KEYPOINT_YOLO, MUTUAL_YOLO, MIMICUTUAL_YOLO, DOUBLE_YOLO, HEATMAP
    int cost_type_count = 9;
    float *sums = calloc(cost_type_count, sizeof(float));
    int *counts = calloc(cost_type_count, sizeof(int));
    memset(counts, 0, cost_type_count*sizeof(float));
    memset(sums, 0, cost_type_count*sizeof(int));

    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            switch (net.layers[i].type) {
                case COST:
                    index = 0;
                    break;
                case HINT_COST:
                    index = 1;
                    break;
                case YOLO:
                    index = 2;
                    break;
                case DISTILL_YOLO:
                    index = 3;
                    break;
                case KEYPOINT_YOLO:
                    index = 4;
                    break;
                case MUTUAL_YOLO:
                    index = 5;
                    break;
                case MIMICUTUAL_YOLO:
                    index = 6;
                    break;
                case DOUBLE_YOLO:
                    index = 7;
                    break;
                case HEATMAP:
                    index = 8;
                    break;
                default:
                    index = -1;
                    break;
            }
            if (index >= 0) {
                counts[index]++;
                sums[index] += net.layers[i].cost[0];
            }
        }
    }

    *net.cost = 0;
    printf("loss for ");
    for (i = 0; i < cost_type_count; ++i) {
        if (counts[i] > 0) {
            float cost_i = sums[i] / counts[i];
            *net.cost += cost_i;
            switch (i) {
                case 0:
                    printf("classification: %.2f ", cost_i);
                    break;
                case 1:
                    printf("hint: %.2f ", cost_i);
                    break;
                case 2:
                    printf("yolo: %.2f ", cost_i);
                    break;
                case 3:
                    printf("distill_yolo: %.2f ", cost_i);
                    break;
                case 4:
                    printf("keypoint_yolo: %.2f ", cost_i);
                    break;
                case 5:
                    printf("mutual_yolo: %.2f ", cost_i);
                    break;
                case 6:
                    printf("mimicutual_yolo: %.2f ", cost_i);
                    break;
                case 7:
                    printf("double_yolo: %.2f ", cost_i);
                    break;
                case 8:
                    printf("heatmap: %.2f ", cost_i);
                    break;
            }
        }
    }
    printf("\n");
    free(counts);
    free(sums);
}

int get_predicted_class_network(network *net)
{
    return max_index(net->output, net->outputs);
}

void backward_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        backward_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    network orig = net;
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
        }
        net.index = i;
        l.backward(l, net);
    }
}

// penalty for slimming sparse scale
// there maybe some question
void updateBN(network *net) {

    size_t batch_num = get_current_batch(net);
    if(batch_num < net->slimming_start_batch) return;
    int i, j;
    int *penalty = calloc(net->n, sizeof(int));
    memset(penalty, 0, net->n * sizeof(int));

    if(net->poly_slimming) {
        net->slimming_scale = net->slimming_min_scale + (float)(batch_num - net->slimming_start_batch) / (net->max_batches - net->slimming_start_batch) * (net->slimming_max_scale - net->slimming_min_scale);
    }

    if (net->consistent_slimming) {
        i = net->n;
        while (i>=0) {
            layer l = net->layers[i];
            if (l.type != SHORTCUT) {
                i--;
                continue;
            }
            int* indexes = calloc(10, sizeof(int));
            int count = 0;
            int next_index = -1;
            while (l.type == SHORTCUT) {
                indexes[count] = l.current_layer_index -1;
                count++;
                next_index = l.index;
                l = net->layers[next_index];
            }
            indexes[count] = next_index;
            count++;
            // do consistent l1-loss, which means l1,2 loss for a matrix,  | sqrt(||X(i, :)||2) |
            int out_c = net->layers[indexes[0]].out_c;
            float* sqrt_row = calloc(out_c, sizeof(float));
            memset(sqrt_row, 0, out_c * sizeof(float));
            float* sqrt_row_gpu = cuda_make_array(sqrt_row, out_c);
            layer corrl;
            for(j=0; j<count; ++j) {
                corrl = net->layers[indexes[j]];
                if (corrl.isprune)
                    add_pow_gpu(out_c, 2, corrl.scales_gpu, 1, sqrt_row_gpu, 1);
            }
            pow_gpu(out_c, 0.5, sqrt_row_gpu, 1, sqrt_row_gpu, 1);

            for(j=0; j<count; ++j) {
                corrl = net->layers[indexes[j]];
                // add corr-l1 loss
                if (corrl.isprune) {
                    add_consistent_l1_delta_gpu(out_c, -net->slimming_scale, corrl.scales_gpu, sqrt_row_gpu,
                                corrl.scale_updates_gpu);
                    penalty[indexes[j]] = 1;
                }
            }

            // release the memory
            cuda_free(sqrt_row_gpu);
            free(sqrt_row);
            free(indexes);

            i = next_index - 1;
        }
    }

    // add l1 loss for corresponding layers (eltwise),  |sum(|gamma|)|
    for (i=0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.isprune && penalty[i] == 0) {
            add_l1_delta_gpu(l.out_c, -net->slimming_scale, l.scales_gpu, l.scale_updates_gpu);
        }
    }

    free(penalty);
}

float train_network_datum(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    forward_network(net);
    backward_network(net);
    if (net->train_slimming) updateBN(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float mimic_train_network_datum(network *netp, network *netq) {
    *netp->seen += netp->batch;
    netp->train = 1;
    netq->train = 0;
    mimic_forward_network(netp, netq);
    backward_network(netp);
    float error = *netp->cost;
    if (((*netp->seen) / netp->batch) % netp->subdivisions == 0) update_network(netp);
    return error;
}

mutual_error mutual_train_network_datum(network *netp, network *netq) {
    *netp->seen += netp->batch;
    *netq->seen += netq->batch;
    netp->train = 1;
    netq->train = 1;
    mutual_forward_network(netp, netq);
    mutual_error errs;
    backward_network(netp);
    backward_network(netq);
    errs.errorp = *netp -> cost;
    errs.errorq = *netq -> cost;
    if (((*netp->seen) / netp->batch) % netp->subdivisions == 0) {
        update_network(netp);
        update_network(netq);
    }
    return errs;
}

mutual_error mimicutual_train_network_datum(network *netp, network *netq, network *netr) {
    *netp->seen += netp->batch;
    *netq->seen += netq->batch;
    *netr->seen += netr->batch;
    netp->train = 1;
    netq->train = 1;
    netr->train = 0;
    //printf("enter mimicutual_forward_network\n");
    mimicutual_forward_network(netp, netq, netr);
    //printf("finish mimicutual_forward_network\n");
    mutual_error errs;
    backward_network(netp);
    backward_network(netq);
    errs.errorp = *netp -> cost;
    errs.errorq = *netq -> cost;
    if (((*netp->seen) / netp->batch) % netp->subdivisions == 0) {
        update_network(netp);
        update_network(netq);
    }
    return errs;
}

float train_network_sgd(network *net, data d, int n) {
    int batch = net->batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float train_network(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;

    if (net->transfer_input) {
        for (i = 0; i < d.X.rows; ++i) {
            scal_cpu_int(d.X.cols, 255.0, d.X.vals[i], 1);
            add_cpu(d.X.cols, -128, d.X.vals[i], 1);
        }
    }

    if (net->transfer_todct) {
        data dct = {0};
        d.X.rows = d.X.rows;
        d.X.vals = calloc(d.X.rows, sizeof(float*));
        d.X.cols = net->h * net->w / 2;  // /8 * /8 * 32
        for(i = 0; i < d.X.rows; ++i) {
            DCT dct_outputs;
            // 1. convert 0~1 to 0~255
            scal_cpu_int(d.X.cols, 255.0, d.X.vals[i], 1);
            // 2. rgb to bgr, and chw to hwc
            uint8_t *bgr_image = calloc(d.X.cols, sizeof(uint8_t));
            // 3. bgr to dct
            rgbgr_chwhc(d.X.vals[i], bgr_image, net->h, net->w, net->c);
            free(bgr_image);
            // 4. scale dct / 8
            float *dct_data = calloc(dct.X.cols, sizeof(float));
            short2float(dct_outputs.data, dct_data, dct.X.cols);
            free(dct_outputs.data);
            dct.X.vals[i] = dct_data;
            scal_cpu_int(dct.X.cols, 1.0/8, dct.X.vals[i], 1);
        }
        d.X.rows = dct.X.rows;
        d.X.cols = dct.X.cols;
        for (i = 0; i < d.X.rows; ++i) {
            d.X.vals[i] = realloc(d.X.vals[i], d.X.cols*sizeof(float));
            whchw(dct.X.vals[i], d.X.vals[i], net->h / 8, net->w / 8, 32);
        }
        free_data(dct);
    }

    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float mimic_train_network(network *netp, network *netq, data d) {
    assert(d.X.rows % netp->batch == 0);
    int batch = netp->batch;
    int n = d.X.rows / batch;

    int i;
    if (netp->transfer_input) {
        for (i = 0; i < d.X.rows; ++i) {
            scal_cpu_int(d.X.cols, 255.0, d.X.vals[i], 1);
            add_cpu(d.X.cols, -128, d.X.vals[i], 1);
        }
    }

    float sum = 0;
    for (i = 0; i < n; ++i) {
        get_next_batch(d, batch, i * batch, netp->input, netp->truth);
        get_next_batch(d, batch, i * batch, netq->input, netq->truth);
        float err = mimic_train_network_datum(netp, netq);
        sum += err;
    }
    return (float) sum / (n * batch);
 }

mutual_error mutual_train_network(network *netp, network *netq, data d) {
    assert(d.X.rows % netp->batch == 0);
    int batch = netp->batch;
    int n = d.X.rows / batch;

    int i;
    if (netp->transfer_input || netq->transfer_input) {
        for (i = 0; i < d.X.rows; ++i) {
            scal_cpu_int(d.X.cols, 255.0, d.X.vals[i], 1);
            add_cpu(d.X.cols, -128, d.X.vals[i], 1);
        }
    }
    float sump = 0, sumq = 0;
    mutual_error errs;
    errs.errorp = 0;
    errs.errorq = 0;
    for (i = 0; i < n; ++i) {
        get_next_batch(d, batch, i * batch, netp->input, netp->truth);
        get_next_batch(d, batch, i * batch, netq->input, netq->truth);
        mutual_error errs = mutual_train_network_datum(netp, netq);
        sump += errs.errorp;
        sumq += errs.errorq;
    }
    errs.errorp = (float) sump / (n * batch);
    errs.errorq = (float) sumq / (n * batch);
    return errs;
}

mutual_error mimicutual_train_network(network *netp, network *netq, network *netr, data d) {
    assert(d.X.rows % netp->batch == 0);
    int batch = netp->batch;
    int n = d.X.rows / batch;

    int i;
    float sump = 0, sumq = 0;
    mutual_error errs;
    errs.errorp = 0;
    errs.errorq = 0;
    for (i = 0; i < n; ++i) {
        get_next_batch(d, batch, i * batch, netp->input, netp->truth);
        get_next_batch(d, batch, i * batch, netq->input, netq->truth);
        get_next_batch(d, batch, i * batch, netr->input, netr->truth);
        mutual_error errs = mimicutual_train_network_datum(netp, netq, netr);
        sump += errs.errorp;
        sumq += errs.errorq;
    }
    errs.errorp = (float) sump / (n * batch);
    errs.errorq = (float) sumq / (n * batch);
    return errs;
}


void set_temp_network(network *net, float t) {
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].temperature = t;
    }
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
            cudnn_convolutional_setup(net->layers + i);
        }
        if(net->layers[i].type == DECONVOLUTIONAL){
            layer *l = net->layers + i;
            cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, l->out_h, l->out_w);
            cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
        }
#endif
    }
}

int resize_network(network *net, int w, int h)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
    cuda_free(net->workspace);
#endif
    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    //fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }else if(l.type == CROP){
            resize_crop_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if(l.type == REGION){
            resize_region_layer(&l, w, h);
        }else if(l.type == YOLO){
            resize_yolo_layer(&l, w, h);
        }else if(l.type == DISTILL_YOLO){
            resize_distill_yolo_layer(&l, w, h);
        }else if(l.type == MUTUAL_YOLO){
            resize_mutual_yolo_layer(&l, w, h);
        }else if(l.type == MIMICUTUAL_YOLO){
            resize_mimicutual_yolo_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
        }else if(l.type == SHORTCUT){
            resize_shortcut_layer(&l, net, w, h);
        }else if(l.type == UPSAMPLE){
            resize_upsample_layer(&l, w, h);
        }else if(l.type == REORG){
            resize_reorg_layer(&l, w, h);
        }else if(l.type == AVGPOOL){
            resize_avgpool_layer(&l, w, h);
        }else if(l.type == NORMALIZATION){
            resize_normalization_layer(&l, w, h);
        }else if(l.type == COST){
            resize_cost_layer(&l, inputs);
        }else if(l.type == HINT_COST){
            resize_hint_cost_layer(&l, inputs);
        }else{
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if(l.workspace_size > 2000000000) assert(0);
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        //printf("%d: out w, h: %d %d\n", i, w, h);
        if(l.type == AVGPOOL) break;
    }
    layer out = get_network_output_layer(net);
    net->inputs = net->layers[0].inputs;
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    free(net->input);
    free(net->truth);
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
    if(gpu_index >= 0){
        cuda_free(net->input_gpu);
        cuda_free(net->truth_gpu);
        net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
        net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
        if(workspace_size){
            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }
    }else {
        free(net->workspace);
        net->workspace = calloc(1, workspace_size);
    }
#else
    free(net->workspace);
    net->workspace = calloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}

layer get_network_detection_layer(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        if(net->layers[i].type == DETECTION){
            return net->layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    layer l = {0};
    return l;
}

image get_network_image_layer(network *net, int i)
{
    layer l = net->layers[i];
#ifdef GPU
    //cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network *net)
{
    int i;
    for(i = net->n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network *net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net->n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    } 
}

void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}


float *network_predict(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;

    int size = net->h*net->w*net->c;
    int dct_size = net->h*net->w/2;

    if (net->transfer_input) {
        scal_cpu_int(size, 255.0, net->input, 1);
        add_cpu(size, -128, net->input, 1);
    }

    if (net->transfer_todct) {
        DCT dct_outputs;
        // 1. convert 0~1 to 0~255
        scal_cpu_int(size, 255.0, input, 1);
        // 2. rgb to bgr, and chw to hwc
        uint8_t *bgr_image = calloc(size, sizeof(uint8_t));
        rgbgr_chwhc(input, bgr_image, net->h, net->w, net->c);
        // 3. bgr to dct
        dct_outputs = bgr2dct(bgr_image, net->w, net->h, net->dct_onlyY);
        free(bgr_image);
        // 4. scale dct result
        float *dct_data = calloc(dct_size, sizeof(float));
        short2float(dct_outputs.data, dct_data, dct_size);
        free(dct_outputs.data);
        scal_cpu_int(dct_size, 1.0/8, dct_data, 1);
        //free(input);
        input = realloc(input, dct_size*sizeof(float));
        whchw(dct_data, input, net->h / 8, net->w / 8, 32);
        free(dct_data);
        size = dct_size;
        net->input = input;
    }

    // write input to input.dat
    if (net->write_input) {
        printf("input size: %d\n", size);
        FILE *fp;
        fp = fopen("ship/input.dat", "wb");
        fwrite(net->input, sizeof(float), size, fp);
        fclose(fp);
    }

    fuse_conv_batchnorm(net);

    if (net->post_training_quantization) {
        if (!net->quantize_per_channel) {
            net->fl = calloc(1, sizeof(int));
            *(net->fl) = 0;
        }
    }

    forward_network(net);
    float *out = net->output;
    *net = orig;
    return out;
}

int num_detections(network *net, float thresh)
{
    //printf("thresh in num_detections: %f\n", thresh);
    int i;
    int s = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO){
            s += yolo_num_detections(l, thresh);
        }
        if(l.type == DOUBLE_YOLO){
            s += double_yolo_num_detections(l, *net, thresh);
        }
        if(l.type == DETECTION || l.type == REGION){
            s += l.w*l.h*l.n;
        }
        if(l.type == KEYPOINT_YOLO){
            s += keypoint_yolo_num_detection_with_keypoints(l, thresh);
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    //printf("nboxes: %d\n", nboxes);
    if(num) *num = nboxes;
    detection *dets = calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = calloc(l.classes, sizeof(float));
        if(l.coords > 4){
            dets[i].mask = calloc(l.coords-4, sizeof(float));
        }
    }
    return dets;
}

detection_with_keypoints *make_network_boxes_with_keypoints(network *net, float thresh, int *num)
{
    int i, j;
    layer l = {0};
    for (i=net->n-1; i>=0; --i) {
        l = net->layers[i];
        if (l.type == KEYPOINT_YOLO)
            break;
    }
    int nboxes_with_keypoints = num_detections(net, thresh);
    if(num) *num = nboxes_with_keypoints;
    detection_with_keypoints *dets = calloc(nboxes_with_keypoints, sizeof(detection_with_keypoints));
    for(i = 0; i < nboxes_with_keypoints; ++i) {
        dets[i].prob = calloc(l.classes, sizeof(float));
        if(l.coords > 4) {
            dets[i].mask = calloc(l.coords-4, sizeof(float));
        }
        // alloc in keypoint_yolo_layer get_keypoint_yolo_detections_with_keypoints
        //for(j = 0; j < l.keypoints_num; ++j) {
            //dets[i].bkps.kps = calloc(l.keypoints_num, sizeof(keypoint));
        //}
    }
    return dets;
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    for(j = 0; j < net->n; ++j){
        layer l = net->layers[j];
        if(l.type == YOLO){
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
            dets += count;
        }
        if(l.type == DOUBLE_YOLO) {
            //printf("before get_double_yolo_detections\n");
            int count = get_double_yolo_detections(l, *net, w, h, net->w, net->h, thresh, map, relative, dets);
            //printf("after get_double_yolo_detections\n");
            dets += count;
        }
        if(l.type == REGION){
            get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if(l.type == DETECTION){
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

void fill_network_boxes_with_keypoints(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection_with_keypoints *dets)
{
    int j;
    for(j = 0; j < net->n; ++j){
        layer l = net->layers[j];
        if(l.type == KEYPOINT_YOLO){
            int count = get_keypoint_yolo_detections_with_keypoints(l, w, h, net->w, net->h, thresh, map, relative, dets);
            dets += count;
        }
    }
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes(net, thresh, num);
    //printf("after make_network_boxes\n");
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
    //printf("after fill_network_boxes\n");
    return dets;
}

detection_with_keypoints *get_network_boxes_with_keypoints(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection_with_keypoints *dets = make_network_boxes_with_keypoints(net, thresh, num);
    fill_network_boxes_with_keypoints(net, w, h, thresh, hier, map, relative, dets);
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}


void free_detections_with_keypoints(detection_with_keypoints *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i) {
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
        free(dets[i].bkps.kps);
    }
    free(dets);
}

float *network_predict_image(network *net, image im)
{
    image imr = letterbox_image(im, net->w, net->h);
    set_batch_network(net, 1);
    float *p = network_predict(net, imr.data);
    free_image(imr);
    return p;
}

int network_width(network *net){return net->w;}
int network_height(network *net){return net->h;}

matrix network_predict_data_multi(network *net, data test, int n)
{
    int i,j,b,m;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net->batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;   
}

matrix network_predict_data(network *net, data test)
{
    int i,j,b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}

void print_network(network *net)
{
    int i,j;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network *n1, network *n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den); 
}

float network_accuracy(network *net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network *net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

layer get_network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

float network_accuracy_multi(network *net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if(net->input) free(net->input);
    if(net->truth) free(net->truth);
#ifdef GPU
    if(net->input_gpu) cuda_free(net->input_gpu);
    if(net->truth_gpu) cuda_free(net->truth_gpu);
#endif
    free(net);
}

// Some day...
// ^ What the hell is this comment for?


layer network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

int network_inputs(network *net)
{
    return net->layers[0].inputs;
}

int network_outputs(network *net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
    return network_output_layer(net).output;
}

#ifdef GPU

void forward_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int size = net.inputs;
    if (net.transfer_todct)
        size = net.layers[0].inputs;
    cuda_push_array(net.input_gpu, net.input, size*net.batch);
    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }

    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        net.layers[i].current_layer_index = i;
        if(l.delta_gpu){
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        //printf("index: %d\n", i);
        l.forward_gpu(l, net);
        //printf("finish\n");
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }

    pull_network_output(netp);
    calc_network_cost(netp);
}

void backward_network_gpu(network *netp)
{
    int i;
    network net = *netp;
    network orig = net;
    cuda_set_device(net.gpu_index);
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
            net.input_gpu = prev.output_gpu;
            net.delta_gpu = prev.delta_gpu;
        }
        net.index = i;
        l.backward_gpu(l, net);
    }
}

void update_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = (*net.t);

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update_gpu){
            l.update_gpu(l, a);
        }
    }
}

typedef struct {
    network *net;
    data d;
    float *err;
} train_args;

typedef struct {
    network *netp;
    network *netq;
    data d;
    float *err;
} mimic_train_args;

typedef struct {
    network *netp;
    network *netq;
    data d;
    mutual_error *err;
} mutual_train_args;

typedef struct {
    network *netp;
    network *netq;
    network *netr;
    data d;
    mutual_error *err;
} mimicutual_train_args;

void *forward_thread(void *ptr)
{
    //printf("forwarding\n");
    train_args args = *(train_args*)ptr;
    free(ptr);
    //printf("id: %d\n", args.net->gpu_index);
    cuda_set_device(args.net->gpu_index);
    int i;
    for (i = 0; i < args.net->n; ++i) {
        args.net->index = i;
        layer l = args.net->layers[i];
        args.net->layers[i].current_layer_index = i;
        if (l.type == DISTILL_YOLO || l.type == MUTUAL_YOLO || l.type == MIMICUTUAL_YOLO || l.type == HINT_COST) continue;
        if (l.delta_gpu) {
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, *args.net);
        args.net->input_gpu = l.output_gpu;
        args.net->input = l.output;
        if(l.truth) {
            args.net->truth_gpu = l.output_gpu;
            args.net->truth = l.output;
        }
    }
    return 0;
}

pthread_t forward_network_in_thread(network *net) {
    //printf("creating forward thread\n");
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    if(pthread_create(&thread, 0, forward_thread, ptr)) error("Thread creation failed");
    return thread;
}

void mimic_forward_network_gpu(network *netp, network *netq) {
    network snet = *netp;
    network tnet = *netq;
    // push data to gpu memory for student net
    cuda_set_device(snet.gpu_index);
    cuda_push_array(snet.input_gpu, snet.input, snet.inputs*snet.batch);
    if (snet.truth) {
        cuda_push_array(snet.truth_gpu, snet.truth, snet.truths*snet.batch);
    }
    // push data to gpu memory for teacher net
    cuda_set_device(tnet.gpu_index);
    cuda_push_array(tnet.input_gpu, tnet.input, tnet.inputs*tnet.batch);
    if (tnet.truth) {
        cuda_push_array(tnet.truth_gpu, tnet.truth, tnet.truths*tnet.batch);
    }

    // first get the output of the teacher network
    pthread_t sforward_thread = forward_network_in_thread(&snet);
    pthread_t tforward_thread = forward_network_in_thread(&tnet);
    pthread_join(sforward_thread, 0);
    pthread_join(tforward_thread, 0);

    pull_network_output(netq);

    // get the hint features form the teacher network
    // this path move to the forward function

    cuda_set_device(snet.gpu_index);
    // forward distill_yolo layer
    int i;
    for(i = 0; i < snet.n; ++i) {
        snet.index = i;
        layer l = snet.layers[i];
        if (l.type == DISTILL_YOLO) {
            if (l.delta_gpu) {
                fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
            }
            l.mimic_forward_gpu(l, snet, tnet);
        }
        snet.input_gpu = l.output_gpu;
        snet.input = l.output;
        //printf("forward %d\n", i);
    }
    // forward hint_cost layer
    for(i = 0; i < snet.n; ++i) {
        layer l = snet.layers[i];
        if (l.type == HINT_COST) {
            if (l.delta_gpu) {
                fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
            }
            l.mimic_forward_gpu(l, snet, tnet);
        }
        snet.input_gpu = l.output_gpu;
        snet.input = l.output;
        //printf("forward %d\n", i);
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void mutual_forward_network_gpu(network *netp, network *netq) {
    network snet = *netp;
    network pnet = *netq;

    // push data to gpu memory for student net
    cuda_set_device(snet.gpu_index);
    cuda_push_array(snet.input_gpu, snet.input, snet.inputs*snet.batch);
    if (snet.truth) {
        cuda_push_array(snet.truth_gpu, snet.truth, snet.truths*snet.batch);
    }
    // push data to gpu memory for peer net
    cuda_set_device(pnet.gpu_index);
    cuda_push_array(pnet.input_gpu, pnet.input, pnet.inputs*pnet.batch);
    if (pnet.truth) {
        cuda_push_array(pnet.truth_gpu, pnet.truth, pnet.truths*pnet.batch);
    }
//    printf("sinput: %f %f %f %f %f\n",
//            snet.input[259582],
//            snet.input[259583],
//            snet.input[259584],
//            snet.input[259585],
//            snet.input[259586]
//            );
//    printf("struth: %f %f %f %f %f\n",
//            snet.truth[0],
//            snet.truth[1],
//            snet.truth[2],
//            snet.truth[3],
//            snet.truth[4]
//            );
//    printf("pinput: %f %f %f %f %f\n",
//            pnet.input[259582],
//            pnet.input[259583],
//            pnet.input[259584],
//            pnet.input[259585],
//            pnet.input[259586]
//            );
//    printf("ptruth: %f %f %f %f %f\n",
//            pnet.truth[0],
//            pnet.truth[1],
//            pnet.truth[2],
//            pnet.truth[3],
//            pnet.truth[4]
//            );

    // can be parallel, here serial
    // forward student and peer net, skip yolo layer (yolo1, yolo2, yolo3)
    cuda_set_device(snet.gpu_index);
    int i;
    for (i = 0; i < snet.n; ++i) {
        snet.index = i;
        layer l = snet.layers[i];
//        if (l.type == CONVOLUTIONAL) {
//            cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
//            printf("%d params: %f %f %f %f %f\n", i, l.weights[0], l.weights[1], l.weights[2], l.weights[3], l.weights[4]);
//        } else
//        {
//            printf("%d params:\n", i);
//        }
        snet.layers[i].current_layer_index = i;
        //printf("%d truth: %d\n", i, l.truth);
        //printf("%d %s\n", i, get_layer_string(l.type));
        if (l.type == MUTUAL_YOLO || l.type == HINT_COST) continue;
        if (l.delta_gpu) {
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, snet);
        snet.input_gpu = l.output_gpu;
        snet.input = l.output;
        if(l.truth) {
            snet.truth_gpu = l.output_gpu;
            snet.truth = l.output;
        }
    }

    cuda_set_device(pnet.gpu_index);
    for (i = 0; i < pnet.n; ++i) {
        pnet.index = i;
        layer l = pnet.layers[i];
        pnet.layers[i].current_layer_index = i;
        //printf("%d truth: %d\n", i, l.truth);
        if (l.type == MUTUAL_YOLO || l.type == HINT_COST) continue;
        if (l.delta_gpu) {
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, pnet);
        pnet.input_gpu = l.output_gpu;
        pnet.input = l.output;
        if(l.truth) {
            pnet.truth_gpu = l.output_gpu;
            pnet.truth = l.output;
        }
    }

    // forward mutual_yolo layer and hint_layer
    for (i = 0; i < snet.n; ++i) {
        snet.index = i;
        layer ls = snet.layers[i];
        if (ls.type == MUTUAL_YOLO) {
            int p_index = snet.mutual_layers[ls.mutual_index];
            layer lp = pnet.layers[p_index];
            if (ls.delta_gpu && lp.delta_gpu) {
                cuda_set_device(snet.gpu_index);
                fill_gpu(ls.outputs * ls.batch, 0, ls.delta_gpu, 1);
                cuda_set_device(pnet.gpu_index);
                fill_gpu(lp.outputs * lp.batch, 0, lp.delta_gpu, 1);
            }
            ls.mutual_forward_gpu(ls, snet, pnet);
            snet.input_gpu = ls.output_gpu;
            snet.input = ls.output;
            pnet.input_gpu = lp.output_gpu;
            pnet.input = lp.output;
        }
        else if (ls.type == HINT_COST) {
            if (ls.delta_gpu) {
                cuda_set_device(snet.gpu_index);
                fill_gpu(ls.outputs * ls.batch, 0, ls.delta_gpu, 1);
            }
            ls.mutual_forward_gpu(ls, snet, pnet);
            snet.input_gpu = ls.output_gpu;
            snet.input = ls.output;
        }
    }

    for (i = 0; i < pnet.n; ++i) {
        pnet.index = i;
        layer lp = pnet.layers[i];
        if (lp.type == HINT_COST) {
            if (lp.delta_gpu) {
                cuda_set_device(pnet.gpu_index);
                fill_gpu(lp.outputs * lp.batch, 0, lp.delta_gpu, 1);
            }
            lp.mutual_forward_gpu(lp, pnet, snet);
            pnet.input_gpu = lp.output_gpu;
            pnet.input = lp.output;
        }
    }
    //assert ( 1== 2);

    pull_network_output(netp);
    pull_network_output(netq);
    calc_network_cost(netp);
    calc_network_cost(netq);
}

void mimicutual_forward_network_gpu(network *netp, network *netq, network *netr) {
    network snet = *netp;
    network pnet = *netq;
    network tnet = *netr;

    // push data to gpu memory for student net
    cuda_set_device(snet.gpu_index);
    cuda_push_array(snet.input_gpu, snet.input, snet.inputs*snet.batch);
    if (snet.truth) {
        cuda_push_array(snet.truth_gpu, snet.truth, snet.truths*snet.batch);
    }
    // push data to gpu memory for peer net
    cuda_set_device(pnet.gpu_index);
    cuda_push_array(pnet.input_gpu, pnet.input, pnet.inputs*pnet.batch);
    if (pnet.truth) {
        cuda_push_array(pnet.truth_gpu, pnet.truth, pnet.truths*pnet.batch);
    }
    // push data to gpu memory for teacher net
    cuda_set_device(tnet.gpu_index);
    cuda_push_array(tnet.input_gpu, tnet.input, tnet.inputs*tnet.batch);
    if (tnet.truth) {
        cuda_push_array(tnet.truth_gpu, tnet.truth, tnet.truths*tnet.batch);
    }

//    printf("sinput: %f %f %f %f %f\n",
//            snet.input[259582],
//            snet.input[259583],
//            snet.input[259584],
//            snet.input[259585],
//            snet.input[259586]
//            );
//    printf("struth: %f %f %f %f %f\n",
//            snet.truth[0],
//            snet.truth[1],
//            snet.truth[2],
//            snet.truth[3],
//            snet.truth[4]
//            );
//    printf("pinput: %f %f %f %f %f\n",
//            pnet.input[259582],
//            pnet.input[259583],
//            pnet.input[259584],
//            pnet.input[259585],
//            pnet.input[259586]
//            );
//    printf("ptruth: %f %f %f %f %f\n",
//            pnet.truth[0],
//            pnet.truth[1],
//            pnet.truth[2],
//            pnet.truth[3],
//            pnet.truth[4]
//            );

    //printf("before thread\n");
    pthread_t sforward_thread = forward_network_in_thread(&snet);
    pthread_t pforward_thread = forward_network_in_thread(&pnet);
    pthread_t tforward_thread = forward_network_in_thread(&tnet);
    pthread_join(sforward_thread, 0);
    pthread_join(pforward_thread, 0);
    pthread_join(tforward_thread, 0);
    //printf("after thread\n");

    pull_network_output(netr);

    /*
    // first get the output of the teacher network
    int i;
    for(i = 0; i < tnet.n; ++i) {
        tnet.index = i;
        layer l = tnet.layers[i];
        if (l.delta_gpu) {
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, tnet);
        tnet.input_gpu = l.output_gpu;
        tnet.input = l.output;
        if(l.truth) {
            tnet.truth_gpu = l.output_gpu;
            tnet.truth = l.output;
        }
    }
    pull_network_output(netr);

    // can be parallel, here serial
    // forward student and peer net, skip yolo layer (yolo1, yolo2, yolo3)
    cuda_set_device(snet.gpu_index);
    int i;
    for (i = 0; i < snet.n; ++i) {
        snet.index = i;
        layer l = snet.layers[i];
//        if (l.type == CONVOLUTIONAL) {
//            cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
//            printf("%d params: %f %f %f %f %f\n", i, l.weights[0], l.weights[1], l.weights[2], l.weights[3], l.weights[4]);
//        } else
//        {
//            printf("%d params:\n", i);
//        }
        snet.layers[i].current_layer_index = i;
        //printf("%d truth: %d\n", i, l.truth);
        if (l.type == MUTUAL_YOLO) continue;
        if (l.delta_gpu) {
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, snet);
        snet.input_gpu = l.output_gpu;
        snet.input = l.output;
        if(l.truth) {
            snet.truth_gpu = l.output_gpu;
            snet.truth = l.output;
        }
    }

    cuda_set_device(pnet.gpu_index);
    for (i = 0; i < pnet.n; ++i) {
        pnet.index = i;
        layer l = pnet.layers[i];
        pnet.layers[i].current_layer_index = i;
        //printf("%d truth: %d\n", i, l.truth);
        if (l.type == MUTUAL_YOLO) continue;
        if (l.delta_gpu) {
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, pnet);
        pnet.input_gpu = l.output_gpu;
        pnet.input = l.output;
        if(l.truth) {
            pnet.truth_gpu = l.output_gpu;
            pnet.truth = l.output;
        }
    }
    */
    int i;
    // forward mutual_yolo layer
    for (i = 0; i < snet.n; ++i) {
        snet.index = i;
        layer ls = snet.layers[i];
        if (ls.type == MIMICUTUAL_YOLO) {
            int p_index = snet.mutual_layers[ls.mutual_index];
            layer lp = pnet.layers[p_index];
            if (ls.delta_gpu && lp.delta_gpu) {
                cuda_set_device(snet.gpu_index);
                fill_gpu(ls.outputs * ls.batch, 0, ls.delta_gpu, 1);
                cuda_set_device(pnet.gpu_index);
                fill_gpu(lp.outputs * lp.batch, 0, lp.delta_gpu, 1);
            }
            ls.mimicutual_forward_gpu(ls, snet, pnet, tnet);
            snet.input_gpu = ls.output_gpu;
            snet.input = ls.output;
            pnet.input_gpu = lp.output_gpu;
            pnet.input = lp.output;
        }
    }
    // forward hint_cost layer
    for (i = 0; i < snet.n; ++i) {
        layer ls = snet.layers[i];
        if (ls.type == HINT_COST) {
            if (ls.delta_gpu) {
                cuda_set_device(snet.gpu_index);
                fill_gpu(ls.outputs * ls.batch, 0, ls.delta_gpu, 1);
            }
            ls.mimic_forward_gpu(ls, snet, tnet);
        }
        snet.input_gpu = ls.output_gpu;
        snet.input = ls.output;
    }
    for (i = 0; i < pnet.n; ++i) {
        layer lp = pnet.layers[i];
        if (lp.type == HINT_COST) {
            if (lp.delta_gpu) {
                cuda_set_device(pnet.gpu_index);
                fill_gpu(lp.outputs * lp.batch, 0, lp.delta_gpu, 1);
            }
            lp.mimic_forward_gpu(lp, pnet, tnet);
        }
        pnet.input_gpu = lp.output_gpu;
        pnet.input = lp.output;
    }
    //assert ( 1 == 2);

    pull_network_output(netp);
    pull_network_output(netq);
    calc_network_cost(netp);
    calc_network_cost(netq);
}

void harmless_update_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.weight_updates_gpu) fill_gpu(l.nweights, 0, l.weight_updates_gpu, 1);
        if(l.bias_updates_gpu) fill_gpu(l.nbiases, 0, l.bias_updates_gpu, 1);
        if(l.scale_updates_gpu) fill_gpu(l.nbiases, 0, l.scale_updates_gpu, 1);
    }
}

void *train_thread(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    cuda_set_device(args.net->gpu_index);
    *args.err = train_network(args.net, args.d);
    return 0;
}

void *mimic_train_thread(void *ptr)
{
    mimic_train_args args = *(mimic_train_args*)ptr;
    free(ptr);
    cuda_set_device(args.netp->gpu_index);
    *args.err = mimic_train_network(args.netp, args.netq, args.d);
    return 0;
}

void *mutual_train_thread(void *ptr)
{
    mutual_train_args args = *(mutual_train_args*)ptr;
    free(ptr);
    cuda_set_device(args.netp->gpu_index);
    *args.err = mutual_train_network(args.netp, args.netq, args.d);
    return 0;
}

void *mimicutual_train_thread(void *ptr)
{
    mimicutual_train_args args = *(mimicutual_train_args*)ptr;
    free(ptr);
    cuda_set_device(args.netp->gpu_index);
    *args.err = mimicutual_train_network(args.netp, args.netq, args.netr, args.d);
    return 0;
}


pthread_t train_network_in_thread(network *net, data d, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
    return thread;
}

pthread_t mimic_train_network_in_thread(network *netp, network *netq, data d, float *err)
{
    pthread_t thread;
    mimic_train_args *ptr = (mimic_train_args *)calloc(1, sizeof(mimic_train_args));
    ptr->netp = netp;
    ptr->netq = netq;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, mimic_train_thread, ptr)) error("Thread creation failed");
    return thread;
}

pthread_t mutual_train_network_in_thread(network *netp, network *netq, data d, mutual_error *err)
{
    pthread_t thread;
    mutual_train_args *ptr = (mutual_train_args *)calloc(1, sizeof(mutual_train_args));
    ptr->netp = netp;
    ptr->netq = netq;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, mutual_train_thread, ptr)) error("Thread creation failed");
    return thread;
}

pthread_t mimicutual_train_network_in_thread(network *netp, network *netq, network *netr, data d, mutual_error *err)
{
    pthread_t thread;
    mimicutual_train_args *ptr = (mimicutual_train_args *)calloc(1, sizeof(mimicutual_train_args));
    ptr->netp = netp;
    ptr->netq = netq;
    ptr->netr = netr;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, mimicutual_train_thread, ptr)) error("Thread creation failed");
    return thread;

}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.nweights, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}


void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.nweights);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.nweights);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.nweights);
        if (base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if (l.type == CONNECTED) {
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}


/*

   void pull_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }

   void push_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }

   void update_layer(layer l, network net)
   {
   int update_batch = net.batch*net.subdivisions;
   float rate = get_current_rate(net);
   l.t = get_current_batch(net);
   if(l.update_gpu){
   l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
   }
   }
   void merge_updates(layer l, layer base)
   {
   if (l.type == CONVOLUTIONAL) {
   axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weight_updates, 1);
   if (l.scale_updates) {
   axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
   }
   } else if(l.type == CONNECTED) {
   axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
   }
   }

   void distribute_updates(layer l, layer base)
   {
   if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
   if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
   }
   }
 */

/*
   void sync_layer(network *nets, int n, int j)
   {
   int i;
   network net = nets[0];
   layer base = net.layers[j];
   scale_weights(base, 0);
   for (i = 0; i < n; ++i) {
   cuda_set_device(nets[i].gpu_index);
   layer l = nets[i].layers[j];
   pull_weights(l);
   merge_weights(l, base);
   }
   scale_weights(base, 1./n);
   for (i = 0; i < n; ++i) {
   cuda_set_device(nets[i].gpu_index);
   layer l = nets[i].layers[j];
   distribute_weights(l, base);
   }
   }
 */

void sync_layer(network **nets, int n, int j)
{
    int i;
    network *net = nets[0];
    layer base = net->layers[j];
    scale_weights(base, 0);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        distribute_weights(l, base);
    }
}

typedef struct{
    network **nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network **nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
    return thread;
}

void sync_nets(network **nets, int n, int interval)
{
    int j;
    int layers = nets[0]->n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *(nets[0]->seen) += interval * (n-1) * nets[0]->batch * nets[0]->subdivisions;
    for (j = 0; j < n; ++j){
        *(nets[j]->seen) = *(nets[0]->seen);
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network **nets, int n, data d, int interval)
{
    int i;
    int batch = nets[0]->batch;
    int subdivisions = nets[0]->subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    //cudaDeviceSynchronize();
    if (get_current_batch(nets[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}

float mimic_train_networks(network **netps, network **netqs, int n, data d, int interval)
{
    int i;
    int batch = netps[0]->batch;
    int subdivisions = netps[0]->subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = mimic_train_network_in_thread(netps[i], netqs[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        sum += errors[i];
    }
    printf("loss: ");
    for(i = 0; i < n; ++i){
        printf("%.2f ", errors[i]);
    }
    printf("\n");
    //cudaDeviceSynchronize();
    if (get_current_batch(netps[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(netps, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}

mutual_error mutual_train_networks(network **netps, network **netqs, int n, data d, int interval)
{
    int i;
    int batch = netps[0]->batch;
    int subdivisions = netps[0]->subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    mutual_error *errors = (mutual_error *) calloc(n, sizeof(mutual_error));

    float sump = 0, sumq = 0;
    mutual_error avg_error;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = mutual_train_network_in_thread(netps[i], netqs[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        sump += errors[i].errorp;
        sumq += errors[i].errorq;
    }
    printf("netp->loss: ");
    for(i = 0; i < n; ++i){
        printf("%.2f ", errors[i].errorp);
    }
    printf("\n");
    printf("netq->loss: ");
    for(i = 0; i < n; ++i){
        printf("%.2f ", errors[i].errorq);
    }
    printf("\n");
    avg_error.errorp = (float)sump/(n);
    avg_error.errorq = (float)sumq/(n);
    //cudaDeviceSynchronize();
    //cudaDeviceSynchronize();
    if (get_current_batch(netps[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(netps, n, interval);
        sync_nets(netqs, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);

    return avg_error;
}

mutual_error mimicutual_train_networks(network **netps, network **netqs, network **netrs, int n, data d, int interval)
{
    int i;
    int batch = netps[0]->batch;
    int subdivisions = netps[0]->subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t*) calloc(n, sizeof(pthread_t));
    mutual_error *errors = (mutual_error*) calloc(n, sizeof(mutual_error));

    float sump = 0, sumq = 0;
    mutual_error avg_error;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = mimicutual_train_network_in_thread(netps[i], netqs[i], netrs[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        sump += errors[i].errorp;
        sumq += errors[i].errorq;
    }
    printf("netp->loss: ");
    for(i = 0; i < n; ++i){
        printf("%.2f ", errors[i].errorp);
    }
    printf("\n");
    printf("netq->loss: ");
    for(i = 0; i < n; ++i){
        printf("%.2f ", errors[i].errorq);
    }
    printf("\n");
    avg_error.errorp = (float)sump/(n);
    avg_error.errorq = (float)sumq/(n);
    //
    if (get_current_batch(netps[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(netps, n, interval);
        sync_nets(netqs, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);

    return avg_error;
}

void pull_network_output(network *net)
{
    layer l = get_network_output_layer(net);
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
}

#endif
