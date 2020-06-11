#include "hint_cost_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

/*
COST_TYPE get_cost_type(char *s) {
    if (strcmp(s, "seg") == 0) return SEG;
    if (strcmp(s, "sse") == 0) return SSE;
    if (strcmp(s, "masked") == 0) return MASKED;
    if (strcmp(s, "smooth") == 0) return SMOOTH;
    if (strcmp(s, "L1") == 0) return L1;
    if (strcmp(s, "wgan") == 0) return WGAN;
    fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
    return SSE;
}

char *get_cost_string(COST_TYPE a) {
    switch (a) {
        case SEG:
            return "seg";
        case SSE:
            return "sse";
        case MASKED:
            return "masked";
        case SMOOTH:
            return "smooth";
        case L1:
            return "L1";
        case WGAN:
            return "wgan";
    }
    return "sse";
}
*/

hint_cost_layer make_hint_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale, int hint_index, float margin) {
    fprintf(stderr, "hint_cost                                           %4d\n", inputs);
    hint_cost_layer l = {0};
    l.type = HINT_COST;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    l.delta = calloc(inputs * batch, sizeof(float));
    l.output = calloc(inputs * batch, sizeof(float));
    l.mimic_truth = calloc(inputs * batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.hint_index = hint_index;

    l.margin = margin;

    l.mimic_forward = forward_hint_cost_layer;
    l.backward = backward_hint_cost_layer;
#ifdef GPU
    l.mimic_forward_gpu = forward_mimic_hint_cost_layer_gpu;
    l.mutual_forward_gpu = forward_mutual_hint_cost_layer_gpu;
    l.backward_gpu = backward_hint_cost_layer_gpu;

    l.delta_gpu = cuda_make_array(l.output, inputs*batch);
    l.mimic_truth_gpu = cuda_make_array(l.mimic_truth, inputs*batch);
    l.output_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    return l;
}

void resize_hint_cost_layer(hint_cost_layer *l, int inputs) {
    l->inputs = inputs;
    l->outputs = inputs;
    l->delta = realloc(l->delta, inputs * l->batch * sizeof(float));
    l->output = realloc(l->output, inputs * l->batch * sizeof(float));
    l->mimic_truth = realloc(l->mimic_truth, inputs * l->batch * sizeof(float));
#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->mimic_truth_gpu);
    l->delta_gpu = cuda_make_array(l->delta, inputs*l->batch);
    l->output_gpu = cuda_make_array(l->output, inputs*l->batch);
    l->mimic_truth_gpu = cuda_make_array(l->mimic_truth, inputs*l->batch);
#endif
}

void forward_hint_cost_layer(hint_cost_layer l, network snet, network tnet) {
    int t_index = snet.hint_layers[l.hint_index];
    if (!tnet.layers[t_index].output) return;
    if (l.cost_type == MASKED) {
        int i;
        for (i = 0; i < l.batch * l.inputs; ++i) {
            if (tnet.layers[t_index].output[i] == SECRET_NUM) snet.input[i] = SECRET_NUM;
        }
    }
    if (l.cost_type == SMOOTH) {
        smooth_l1_cpu(l.batch * l.inputs, snet.input, tnet.layers[t_index].output, l.delta, l.output);
    } else if (l.cost_type == L1) {
        l1_cpu(l.batch * l.inputs, snet.input, tnet.layers[t_index].output, l.delta, l.output);
    } else {
        l2_cpu(l.batch * l.inputs, snet.input, tnet.layers[t_index].output, l.delta, l.output);
    }
    l.cost[0] = sum_array(l.output, l.batch * l.inputs);
}

void backward_hint_cost_layer(const hint_cost_layer l, network net) {
    axpy_cpu(l.batch * l.inputs, l.scale, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_hint_cost_layer(hint_cost_layer l)
{
    cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

void push_hint_cost_layer(hint_cost_layer l)
{
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

/*
int float_abs_compare (const void * a, const void * b)
{
    float fa = *(const float*) a;
    if(fa < 0) fa = -fa;
    float fb = *(const float*) b;
    if(fb < 0) fb = -fb;
    return (fa > fb) - (fa < fb);
}*/

void forward_mimic_hint_cost_layer_gpu(hint_cost_layer l, network snet, network tnet)
{
    int t_index = snet.hint_layers[l.hint_index];
    //printf("hint_layer: %d\n", t_index);
    layer lt = tnet.layers[t_index];
    if (!lt.output_gpu) return;

    // pull the data form tnet gpu to cpu
    cuda_set_device(tnet.gpu_index);
    //printf("assert: %d %d %d\n", t_index, l.outputs, tnet.layers[t_index].outputs);
    assert(l.outputs == lt.outputs);
    cuda_pull_array(lt.output_gpu, l.mimic_truth, l.batch*l.inputs);

    // push the data from cpu to snet gpu
    cuda_set_device(snet.gpu_index);
    cuda_push_array(l.mimic_truth_gpu, l.mimic_truth, l.batch*l.inputs);

    // for test
    /*
    printf("in forward hint cost\n");
    cuda_pull_array(snet.input_gpu, snet.input, l.batch*l.inputs);
    printf("input(from network) of hint_cost_layer: %f %f %f %f %f\n",
            snet.input[0], snet.input[1], snet.input[2], snet.input[3], snet.input[4]);
    //printf("layer_index: %d\n", l.current_layer_index);
    cuda_pull_array(snet.layers[l.current_layer_index-1].output_gpu, snet.layers[l.current_layer_index-1].output, l.batch*l.inputs);
    printf("input(from prev layer) of hint_cost_layer: %f %f %f %f %f\n",
            snet.layers[l.current_layer_index-1].output[0],
            snet.layers[l.current_layer_index-1].output[1],
            snet.layers[l.current_layer_index-1].output[2],
            snet.layers[l.current_layer_index-1].output[3],
            snet.layers[l.current_layer_index-1].output[4]);
    */

    if(l.smooth){
        scal_gpu(l.batch*l.inputs, (1-l.smooth), l.mimic_truth_gpu, 1);
        add_gpu(l.batch*l.inputs, l.smooth * 1./l.inputs, l.mimic_truth_gpu, 1);
    }

    if(l.cost_type == SMOOTH){
        smooth_l1_gpu(l.batch*l.inputs, snet.input_gpu, l.mimic_truth_gpu, l.delta_gpu, l.output_gpu);
    } else if (l.cost_type == L1){
        l1_gpu(l.batch*l.inputs, snet.input_gpu, l.mimic_truth_gpu, l.delta_gpu, l.output_gpu);
    } else if (l.cost_type == WGAN){
        wgan_gpu(l.batch*l.inputs, snet.input_gpu, l.mimic_truth_gpu, l.delta_gpu, l.output_gpu);
    } else {
        l2_gpu(l.batch*l.inputs, snet.input_gpu, l.mimic_truth_gpu, l.delta_gpu, l.output_gpu);
    }

    if (l.cost_type == SEG && l.noobject_scale != 1) {
        scale_mask_gpu(l.batch*l.inputs, l.delta_gpu, 0, l.mimic_truth_gpu, l.noobject_scale);
        scale_mask_gpu(l.batch*l.inputs, l.output_gpu, 0, l.mimic_truth_gpu, l.noobject_scale);
    }
    if (l.cost_type == MASKED) {
        mask_gpu(l.batch*l.inputs, snet.delta_gpu, SECRET_NUM, l.mimic_truth_gpu, 0);
    }

    if(l.ratio){
        cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
        qsort(l.delta, l.batch*l.inputs, sizeof(float), float_abs_compare);
        int n = (1-l.ratio) * l.batch*l.inputs;
        float thresh = l.delta[n];
        thresh = 0;
        printf("%f\n", thresh);
        supp_gpu(l.batch*l.inputs, thresh, l.delta_gpu, 1);
    }

    if(l.thresh){
        supp_gpu(l.batch*l.inputs, l.thresh*1./l.inputs, l.delta_gpu, 1);
    }

    cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
    l.cost[0] = sum_array(l.output, l.batch*l.inputs) / (l.batch * l.inputs);
    //printf("Layer  %d loss: %f\n", snet.index, l.cost[0]);

    //printf("finish hint_cout\n");
}

void forward_mutual_hint_cost_layer_gpu(hint_cost_layer ls, network snet, network pnet)
{
    // change scale
    // snet.layers[ls.current_layer_index].scale = 1. / ls.inputs;

    // get output of snet and pnet
    cuda_set_device(snet.gpu_index);
    copy_gpu(ls.batch*ls.inputs, snet.layers[ls.current_layer_index-1].output_gpu, 1, ls.output_gpu, 1);

    cuda_set_device(pnet.gpu_index);
    int p_index = snet.hint_layers[ls.hint_index];
    //printf("hint_layer: %d\n", p_index);
    layer lp = pnet.layers[p_index];
    assert(ls.outputs == lp.outputs);

    cuda_pull_array(pnet.layers[p_index].output_gpu, ls.mimic_truth, lp.batch*lp.inputs);

    // push the data from cpu to snet gpu
    cuda_set_device(snet.gpu_index);
    cuda_push_array(ls.mimic_truth_gpu, ls.mimic_truth, ls.batch*ls.inputs);

    // smooth output (unused)
    if(ls.smooth){
        scal_gpu(ls.batch*ls.inputs, (1-ls.smooth), ls.output_gpu, 1);
        add_gpu(ls.batch*ls.inputs, ls.smooth * 1./ls.inputs, ls.output_gpu, 1);
    }

    if(lp.smooth){
        scal_gpu(ls.batch*ls.inputs, (1-ls.smooth), ls.output_gpu, 1);
        add_gpu(ls.batch*ls.inputs, ls.smooth * 1./ls.inputs, ls.output_gpu, 1);
    }

    // cost_type is sse
    if(ls.cost_type == SMOOTH){
        // TODO
    } else if (ls.cost_type == L1){
        // TODO
    } else if (ls.cost_type == WGAN){
        // TODO
    } else if (ls.cost_type == L1_MARGIN) {
        // modify here, we use margin in tripletloss
        // loss = max(0, min(ls, lp)-max(ls, lp) + margin) = max(0, -|ls - lp| + margin)
        // error(ls) = error(lp)
        l1_margin_gpu(ls.batch*ls.inputs, ls.output_gpu, ls.mimic_truth_gpu, ls.margin, ls.delta_gpu, ls.output_gpu);
    } else if (ls.cost_type == SSE) {
        //printf("sse\n");
        l2_gpu(ls.batch*ls.inputs, ls.output_gpu, ls.mimic_truth_gpu, ls.delta_gpu, ls.output_gpu);
    }

    // unused
    if (ls.cost_type == SEG && ls.noobject_scale != 1) {
        // TODO
    }
    if (ls.cost_type == MASKED) {
        // TODO
    }

    if(ls.ratio){
        // TODO
    }

    if(ls.thresh){
        // TODO
    }


    cuda_pull_array(ls.output_gpu, ls.output, ls.batch*ls.inputs);
    //printf("scale: %f\n", ls.scale);
    //float cost = sum_array(ls.output, ls.batch*ls.inputs);
    //printf("%d: all diff: %f, average diff: %f\n", ls.current_layer_index, cost, cost / (ls.batch*ls.inputs));
    //printf("w: %d, h: %d\n", snet.layers[ls.current_layer_index-1].out_w, snet.layers[ls.current_layer_index-1].out_h);
    //ls.cost[0] = sum_array(ls.output, ls.batch*ls.inputs) / (ls.batch * ls.inputs);
    ls.cost[0] = sum_array(ls.output, ls.batch*ls.inputs) / ls.inputs;
}

void backward_hint_cost_layer_gpu(const hint_cost_layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

