#include "reweight_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>

reweight_layer make_reweight_layer(int batch, int n, int *input_layers, int *input_sizes, ACTIVATION activation)
{
    fprintf(stderr,"reweight ");
    reweight_layer l = {0};
    l.type = REWEIGHT;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    int i;
    int outputs = 0;
    for(i = 0; i < n; ++i){
        fprintf(stderr," %d", input_layers[i]);
    }
    outputs += input_sizes[1];
    fprintf(stderr, "\n");
    l.outputs = outputs;
    l.inputs = outputs;
    l.delta =  calloc(outputs*batch, sizeof(float));
    l.output = calloc(outputs*batch, sizeof(float));;

    l.forward = forward_reweight_layer;
    l.backward = backward_reweight_layer;
    #ifdef GPU
    l.forward_gpu = forward_reweight_layer_gpu;
    l.backward_gpu = backward_reweight_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, outputs*batch);
    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    #endif
    l.activation = activation;
    return l;
}

void resize_reweight_layer(reweight_layer *l, network *net)
{
    layer second = net->layers[l->input_layers[1]];
    l->out_w = second.out_w;
    l->out_h = second.out_h;
    l->out_c = second.out_c;
    l->outputs = second.outputs;
    l->input_sizes[1] = second.outputs;

    layer first = net->layers[l->input_layers[0]];
    if(first.out_c == second.out_c){
        l->out_c = second.out_c;
    }else{
        printf("%d %d\n", first.out_c, second.out_c);
        l->out_h = l->out_w = l->out_c = 0;
    }
    l->inputs = l->outputs;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif

}

void forward_reweight_layer(const reweight_layer l, network net)
{
    int i, j;
    int scale_offset = 0;
    int x_offset = 0;

    float *scale = net.layers[l.input_layers[0]].output;
    float *x = net.layers[l.input_layers[1]].output;

    for(i = 0; i < l.batch; ++i){
        for (j = 0; j < l.out_c; ++j) {
            scale_offset = i * l.out_c + j;
            x_offset = i * l.outputs + j * l.out_h * l.out_w;
            copy_cpu(l.out_h*l.out_w, x + x_offset, 1, l.output + x_offset, 1);
            scal_cpu(l.out_h*l.out_w, scale[scale_offset], l.output + x_offset, 1);
        }
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);

}

void backward_reweight_layer(const reweight_layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    int i, j, k;
    int scale_offset = 0;
    int x_offset = 0;

    float *scale = net.layers[l.input_layers[0]].output;
    float *scale_delta = net.layers[l.input_layers[0]].delta;
    float *x = net.layers[l.input_layers[1]].output;
    float *x_delta = net.layers[l.input_layers[1]].delta;

    for(i = 0; i < l.batch; ++i){
        for (j = 0; j < l.out_c; ++j) {
            scale_offset = i * l.out_c + j;
            x_offset = i * l.outputs + j * l.out_h * l.out_w;
            axpy_cpu(l.h*l.w, scale[scale_offset], l.delta + x_offset, 1, x_delta+x_offset, 1);
            scale_delta[scale_offset] = 0;
            for (k = 0; k < l.out_h * l.out_w; ++k) {
                scale_delta[scale_offset] += l.delta[x_offset+k] * x[x_offset+k];
            }
        }
    }
}

#ifdef GPU
void forward_reweight_layer_gpu(const reweight_layer l, network net)
{
    int i, j;
    int scale_offset = 0;
    int x_offset = 0;

    float *scale = net.layers[l.input_layers[0]].output_gpu;
    float *x = net.layers[l.input_layers[1]].output_gpu;

    for(i = 0; i < l.batch; ++i){
        for (j = 0; j < l.out_c; ++j) {
            scale_offset = i * l.out_c + j;
            x_offset = i * l.outputs + j * l.out_h * l.out_w;
            reweight_gpu(l.out_h*l.out_w, scale, scale_offset, x + x_offset, 1, l.output_gpu + x_offset, 1);
        }
    }

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

}

void backward_reweight_layer_gpu(const reweight_layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    int i, j, k;
    int scale_offset = 0;
    int x_offset = 0;

    int scale_num = net.layers[l.input_layers[0]].batch * net.layers[l.input_layers[0]].outputs;
    int x_num = net.layers[l.input_layers[1]].batch * net.layers[l.input_layers[1]].outputs;


    cuda_pull_array(net.layers[l.input_layers[0]].output_gpu, net.layers[l.input_layers[0]].output, scale_num);
    cuda_pull_array(net.layers[l.input_layers[1]].output_gpu, net.layers[l.input_layers[1]].output, x_num);
    cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.outputs);

    float *scale = net.layers[l.input_layers[0]].output;
    float *scale_delta = net.layers[l.input_layers[0]].delta;
    float *x = net.layers[l.input_layers[1]].output;
    float *x_delta = net.layers[l.input_layers[1]].delta_gpu;

    for(i = 0; i < l.batch; ++i){
        for (j = 0; j < l.out_c; ++j) {
            scale_offset = i * l.out_c + j;
            x_offset = i * l.outputs + j * l.out_h * l.out_w;
            // x_delta
            axpy_gpu(l.out_h*l.out_w, scale[scale_offset], l.delta_gpu + x_offset, 1, x_delta+x_offset, 1);
            // scale_delta
            scale_delta[scale_offset] = 0;
            for (k = 0; k < l.out_h*l.out_w; ++k) {
                scale_delta[scale_offset] += l.delta[x_offset+k] * x[x_offset+k];
            }
        }
    }

    cuda_push_array(net.layers[l.input_layers[0]].delta_gpu, scale_delta, scale_num);
}
#endif