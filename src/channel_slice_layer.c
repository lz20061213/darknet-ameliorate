#include "channel_slice_layer.h"
#include "darknet.h"
#include "blas.h"

channel_slice_layer make_channel_slice_layer(int batch, int w, int h, int c,
    int begin_slice_point, int end_slice_point, int axis, int n, int *input_layers, int *input_sizes)
{
    channel_slice_layer l = {0};
    l.type = CHANNEL_SLICE;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w;
    l.out_h = h;
    l.n = n;
    l.out_c = end_slice_point - begin_slice_point;
    l.axis = axis;
    l.begin_slice_point = begin_slice_point;
    l.end_slice_point = end_slice_point;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    int output_size = l.outputs * batch;
    l.delta = (float*)calloc(output_size, sizeof(float));
    l.output = (float*)calloc(output_size,sizeof(float));
    l.forward = forward_channel_slice_layer;
    l.backward = backward_channel_slice_layer;
    fprintf(stderr, "channel_slice              %4d x%4d x%4d   ->  %4d x%4d x%4d \n", w, h, c, l.out_w, l.out_h, l.out_c);
#ifdef GPU
    l.forward_gpu = forward_channel_slice_layer_gpu;
    l.backward_gpu = backward_channel_slice_layer_gpu;

    l.output_gpu = cuda_make_array(l.output,output_size);
    l.delta_gpu = cuda_make_array(l.delta,output_size);
#endif
    return l;
}

void resize_channel_slice_layer(channel_slice_layer *l, int h, int w)
{
    l->h = h;
    l->w = w;
    l->out_h = h;
    l->out_w = w;
    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->h * l->w * l->c;
    l->delta = (float*)realloc(l->delta,l->outputs * l->batch * sizeof(float));
    l->output = (float*)realloc(l->output,l->outputs * l->batch * sizeof(float));
    int output_size = l->outputs * l->batch;
#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
#endif
}

void forward_channel_slice_layer(const channel_slice_layer l, network net)
{
    int input_channels = l.c;
    int output_channels = l.out_c;
    int begin_slice_pt = l.begin_slice_point;
    int batch_size = l.batch;
    int spatial_size = l.h * l.w;
    // here l.n=1, just one layer input
    int i, n;
    for(i=0; i<l.n; ++i) {
        int index = l.input_layers[i];
        float *input = net.layers[index].output;
        for(n=0; n<batch_size; ++n)
        {
            const int input_offset = (n * input_channels + begin_slice_pt) * spatial_size;
            const int output_offset = n * output_channels * spatial_size;
            float *p_i = input + input_offset;
            float *p_o = l.output + output_offset;
            copy_cpu(output_channels * spatial_size, p_i, 1, p_o, 1);
        }
    }
}

void backward_channel_slice_layer(const channel_slice_layer l, network net)
{
    int input_channels = l.c;
    int output_channels = l.out_c;
    int begin_slice_pt = l.begin_slice_point;
    int batch_size = l.batch;
    int spatial_size = l.h * l.w;
    int i, n;
    for(i=0; i<l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta;
        for(n=0; n < batch_size; ++n)
        {
            const int input_offset = n * output_channels * spatial_size;
            const int output_offset = (n * input_channels + begin_slice_pt) * spatial_size;
            float *delta_i = l.delta + input_offset;
            float *delta_o = delta + output_offset;
            axpy_cpu(output_channels * spatial_size, 1, delta_i, 1, delta_o, 1);
        }
    }
}

#ifdef GPU
void forward_channel_slice_layer_gpu(const channel_slice_layer l, network net)
{
    int input_channels = l.c;
    int output_channels = l.out_c;
    int begin_slice_pt = l.begin_slice_point;
    int batch_size = l.batch;
    int spatial_size = l.h * l.w;
    // here l.n=1, just one layer input
    int i, n;
    for(i=0; i<l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output_gpu;
        for(n=0; n<batch_size; ++n)
        {
            const int input_offset = (n * input_channels + begin_slice_pt) * spatial_size;
            const int output_offset = n * output_channels * spatial_size;
            float *p_i = input + input_offset;
            float *p_o = l.output_gpu + output_offset;
            copy_gpu(output_channels * spatial_size, p_i, 1, p_o, 1);
        }
    }
}

void backward_channel_slice_layer_gpu(const channel_slice_layer l, network net)
{
    int input_channels = l.c;
    int output_channels = l.out_c;
    int begin_slice_pt = l.begin_slice_point;
    int batch_size = l.batch;
    int spatial_size = l.h * l.w;
    int i, n;
    for(i=0; i<l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta_gpu;
        for(n=0; n < batch_size; ++n)
        {
            const int input_offset = n * output_channels * spatial_size;
            const int output_offset = (n * input_channels + begin_slice_pt) * spatial_size;
            float *delta_i = l.delta_gpu + input_offset;
            float *delta_o = delta + output_offset;
            axpy_gpu(output_channels * spatial_size, 1, delta_i, 1, delta_o, 1);
        }
    }
}

#endif