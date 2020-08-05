#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += fabsf(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += fabsf(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}

void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        net.input_gpu = l.binary_input_gpu;
    }

    if(l.quantize){
        // quantize the input feature maps
        copy_gpu(l.inputs*l.batch, net.input_gpu, 1, l.quantize_input_gpu, 1);
        if (l.quantize_feature) {
            quantize_gpu(l.quantize_input_gpu, l.inputs*l.batch, net.quantize_feature_bitwidth, net.quantize_feature_fraction_bitwidth);
        }
        if(l.batch_normalize) {
#ifdef CUDNN
            float one = 1;
            cudnnConvolutionForward(cudnn_handle(),
                        &one,
                        l.srcTensorDesc,
                        l.quantize_input_gpu,
                        l.weightDesc,
                        l.weights_gpu,
                        l.convDesc,
                        l.fw_algo,
                        net.workspace,
                        l.workspace_size,
                        &one,
                        l.dstTensorDesc,
                        l.output_gpu);
#else
            int i, j;
            int m = l.n/l.groups;
            int k = l.size*l.size*l.c/l.groups;
            int n = l.out_w*l.out_h;
            for(i = 0; i < l.batch; ++i){
                for(j = 0; j < l.groups; ++j){
                    float *a = l.weights_gpu + j*l.nweights/l.groups;
                    float *b = net.workspace;
                    float *c = l.output_gpu + (i*l.groups + j)*n*m;
                    float *im = l.quantize_input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

                    if (l.size == 1){
                        b = im;
                    } else {
                        im2col_gpu_ext(im,          // input
                            l.c / l.groups,         // input channels
                            l.h, l.w,               // input size (h, w)
                            l.size, l.size,         // kernel size (h, w)
                            l.pad, l.pad,           // padding (h, w)
                            l.stride, l.stride,     // stride (h, w)
                            l.dilation, l.dilation, // dilation (h, w)
                            b);       // output
                    }
                    gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
                }
            }
#endif
            // copy x_stat, used for backward
            copy_gpu(l.batch*l.outputs, l.output_gpu, 1, l.x_stat_gpu, 1);

            // calculate the mean and variance of l.output_gpu
            fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
            fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);

            copy_gpu(l.nweights, l.weights_gpu, 1, l.merge_weights_gpu, 1);
            scale_weights_gpu(l.merge_weights_gpu, l.nweights, l.n, l.scales_gpu, l.rolling_variance_gpu);

            copy_gpu(l.nweights, l.merge_weights_gpu, 1, l.scale_weights_gpu, 1);
            quantize_gpu(l.merge_weights_gpu, l.nweights, net.quantize_weight_bitwidth, net.quantize_weight_fraction_bitwidth);

#ifdef CUDNN
            // free the results of output_gpu in first conv, or use zero in cudnnConvolutionForward
            fill_gpu(l.batch*l.outputs, 0, l.output_gpu, 1);
            one = 1;
            cudnnConvolutionForward(cudnn_handle(),
                        &one,
                        l.srcTensorDesc,
                        l.quantize_input_gpu,
                        l.weightDesc,
                        l.merge_weights_gpu,
                        l.convDesc,
                        l.fw_algo,
                        net.workspace,
                        l.workspace_size,
                        &one,
                        l.dstTensorDesc,
                        l.output_gpu);
#else
            int i, j;
            int m = l.n/l.groups;
            int k = l.size*l.size*l.c/l.groups;
            int n = l.out_w*l.out_h;
            for(i = 0; i < l.batch; ++i){
                for(j = 0; j < l.groups; ++j){
                    float *a = l.merge_weights_gpu + j*l.nweights/l.groups;
                    float *b = net.workspace;
                    float *c = l.output_gpu + (i*l.groups + j)*n*m;
                    float *im = l.quantize_input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

                    if (l.size == 1){
                        b = im;
                    } else {
                        im2col_gpu_ext(im,          // input
                            l.c / l.groups,         // input channels
                            l.h, l.w,               // input size (h, w)
                            l.size, l.size,         // kernel size (h, w)
                            l.pad, l.pad,           // padding (h, w)
                            l.stride, l.stride,     // stride (h, w)
                            l.dilation, l.dilation, // dilation (h, w)
                            b);       // output
                    }
                    gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
                }
            }
#endif

            copy_gpu(l.n, l.biases_gpu, 1, l.merge_biases_gpu, 1);
            // update the rolling_mean_gpu and rolling_variance_gpu
            if (!net.quantize_freezeBN) {
                // todo: keep outputs before scal, used in backward
                copy_gpu(l.batch*l.outputs, l.output_gpu, 1, l.x_norm_gpu, 1);

                scale_outputs_gpu(l.output_gpu, l.variance_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_w*l.out_h);

                shift_bias_gpu(l.merge_biases_gpu, l.n, l.scales_gpu, l.biases_gpu, l.mean_gpu, l.variance_gpu);

                // todo: keep rolling_variance before scal, used in backward
                //scal_gpu(l.out_c, .99, l.rolling_mean_gpu, 1);
                //axpy_gpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
                //scal_gpu(l.out_c, .99, l.rolling_variance_gpu, 1);
                //axpy_gpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);
            } else {
                shift_bias_gpu(l.merge_biases_gpu, l.n, l.scales_gpu, l.biases_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu);
            }

            copy_gpu(l.n, l.merge_biases_gpu, 1, l.shift_biases_gpu, 1);
            quantize_gpu(l.merge_biases_gpu, l.n, net.quantize_bias_bitwidth, net.quantize_bias_fraction_bitwidth); // todo
            add_bias_gpu(l.output_gpu, l.merge_biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
        } else {
            copy_gpu(l.nweights, l.weights_gpu, 1, l.merge_weights_gpu, 1);
            quantize_gpu(l.merge_weights_gpu, l.nweights, net.quantize_weight_bitwidth, net.quantize_weight_fraction_bitwidth);
#ifdef CUDNN
            float one = 1;
            cudnnConvolutionForward(cudnn_handle(),
                        &one,
                        l.srcTensorDesc,
                        l.quantize_input_gpu,
                        l.weightDesc,
                        l.merge_weights_gpu,
                        l.convDesc,
                        l.fw_algo,
                        net.workspace,
                        l.workspace_size,
                        &one,
                        l.dstTensorDesc,
                        l.output_gpu);
#else
            int i, j;
            int m = l.n/l.groups;
            int k = l.size*l.size*l.c/l.groups;
            int n = l.out_w*l.out_h;
            for(i = 0; i < l.batch; ++i){
                for(j = 0; j < l.groups; ++j){
                    float *a = l.merge_weights_gpu + j*l.nweights/l.groups;
                    float *b = net.workspace;
                    float *c = l.output_gpu + (i*l.groups + j)*n*m;
                    float *im = l.quantize_input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

                    if (l.size == 1){
                        b = im;
                    } else {
                        im2col_gpu_ext(im,          // input
                            l.c / l.groups,         // input channels
                            l.h, l.w,               // input size (h, w)
                            l.size, l.size,         // kernel size (h, w)
                            l.pad, l.pad,           // padding (h, w)
                            l.stride, l.stride,     // stride (h, w)
                            l.dilation, l.dilation, // dilation (h, w)
                            b);       // output
                    }
                    gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
                }
            }
#endif
            copy_gpu(l.n, l.biases_gpu, 1, l.merge_biases_gpu, 1);
            quantize_gpu(l.merge_biases_gpu, l.n, net.quantize_bias_bitwidth, net.quantize_bias_fraction_bitwidth);
            add_bias_gpu(l.output_gpu, l.merge_biases_gpu, l.batch, l.n, l.out_w*l.out_h);
        }
    } else {
#ifdef CUDNN
        float one = 1;
        cudnnConvolutionForward(cudnn_handle(),
                    &one,
                    l.srcTensorDesc,
                    net.input_gpu,
                    l.weightDesc,
                    l.weights_gpu,
                    l.convDesc,
                    l.fw_algo,
                    net.workspace,
                    l.workspace_size,
                    &one,
                    l.dstTensorDesc,
                    l.output_gpu);
#else
        int i, j;
        int m = l.n/l.groups;
        int k = l.size*l.size*l.c/l.groups;
        int n = l.out_w*l.out_h;
        for(i = 0; i < l.batch; ++i){
            for(j = 0; j < l.groups; ++j){
                float *a = l.weights_gpu + j*l.nweights/l.groups;
                float *b = net.workspace;
                float *c = l.output_gpu + (i*l.groups + j)*n*m;
                float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

                if (l.size == 1){
                    b = im;
                } else {
                    im2col_gpu_ext(im,          // input
                        l.c / l.groups,         // input channels
                        l.h, l.w,               // input size (h, w)
                        l.size, l.size,         // kernel size (h, w)
                        l.pad, l.pad,           // padding (h, w)
                        l.stride, l.stride,     // stride (h, w)
                        l.dilation, l.dilation, // dilation (h, w)
                        b);       // output
                }
                gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
            }
        }
#endif
        if (l.batch_normalize) {
            forward_batchnorm_layer_gpu(l, net);
        } else {
            add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
        }
    }

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
}

__global__ void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.f);
    int h_offset = -(size/2.f);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }
    //constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    // CHECK since the gradient accumulate each backward, so there need to use temp updates
    if(l.quantize) {
        float one = 1, zero = 0;
        if(l.batch_normalize) {
            // 1. for bias_updates_gpu
            fill_gpu(l.n, 0, l.bias_updates_gpu_part, 1);
            // CHECK backward y = conv(x) + merge_bias, for merge_bias
            backward_bias_gpu(l.bias_updates_gpu_part, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
            // CHECK backward merge_bias = quantize_gpu(shift_bias)
            backward_quantize_gpu(l.bias_updates_gpu_part, l.shift_biases_gpu, l.n, net.quantize_bias_bitwidth, net.quantize_bias_fraction_bitwidth);
            // CHECK backward shift_bias = bias - gamma * mean / sqrt(var+.00001f), bias_updates_gpu unchanged
            axpy_gpu(l.n, 1, l.bias_updates_gpu_part, 1, l.bias_updates_gpu, 1);

            // 2. for weight_updates_gpu, delta (mean, var, rolling_mean, rolling_var) = 0
            // CHECK backward y = scale * conv(x) + merge_bias, for conv (merge_weights)
            if (!net.quantize_freezeBN) {
                // CHECK backward y = scale * conv(x)
                fill_gpu(l.n, 0, l.variance_delta_gpu, 1);
                fill_gpu(l.n, 0, l.variance_delta_part_gpu, 1);
                // CHECK backward var: 1, y = sqrt(moving_var+.00001f) / sqrt(var+.00001f) * conv(x)
                backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_delta_part_gpu);
                backward_variance_outputs_gpu(l.variance_delta_gpu, l.variance_delta_part_gpu, l.n, l.rolling_variance_gpu, l.variance_gpu);

                scale_outputs_gpu(l.delta_gpu, l.variance_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
            }
#ifdef CUDNN
            // CHECK backward conv(x) = merge_weights * quantize_input
            cudnnConvolutionBackwardFilter(cudnn_handle(),
                    &one,
                    l.srcTensorDesc,
                    l.quantize_input_gpu,
                    l.ddstTensorDesc,
                    l.delta_gpu,
                    l.convDesc,
                    l.bf_algo,
                    net.workspace,
                    l.workspace_size,
                    &zero,
                    l.dweightDesc,
                    l.weight_updates_gpu_part);
#endif

            // CHECK backward merge_weights = quantize_gpu(scale_weights)
            backward_quantize_gpu(l.weight_updates_gpu_part, l.scale_weights_gpu, l.nweights, net.quantize_weight_bitwidth, net.quantize_weight_fraction_bitwidth);

            // CHECK backward scale_weights = gamma / sqrt(var+.00001f) * weights, gamma first, then update weights
            fill_gpu(l.n, 0, l.scale_updates_gpu_part, 1);
            // CHECK move 1 / sqrt(var+.00001f) to backward_shift_gamma_gpu
            backward_gamma_gpu(l.weights_gpu, l.weight_updates_gpu_part, l.c, l.n, l.size*l.size, l.rolling_variance_gpu, l.scale_updates_gpu_part);
            // CHECK backward shift_bias = bias - gamma * mean / sqrt(var+.00001f)
            if (!net.quantize_freezeBN) {
                backward_shift_gamma_gpu(l.scale_updates_gpu_part, l.n, l.bias_updates_gpu_part, l.rolling_variance_gpu, l.mean_gpu, l.variance_gpu);
            } else {
                backward_shift_gamma_gpu(l.scale_updates_gpu_part, l.n, l.bias_updates_gpu_part, l.rolling_variance_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu);
            }
            axpy_gpu(l.n, 1, l.scale_updates_gpu_part, 1, l.scale_updates_gpu, 1);

            scale_weights_gpu(l.weight_updates_gpu_part, l.nweights, l.n, l.scales_gpu, l.rolling_variance_gpu);

            if (!net.quantize_freezeBN) {
                // CHECK backward mean, shift_bias = bias - gamma * mean / sqrt(var+.00001f)
                backward_mean_gpu(l.mean_delta_gpu, l.bias_updates_gpu_part, l.n, l.scales_gpu, l.variance_gpu);

                // CHECK backward var: 2, shift_bias = bias - gamma * mean / sqrt(var+.00001f)
                backward_variance_bias_gpu(l.variance_delta_gpu, l.bias_updates_gpu_part, l.n, l.scales_gpu, l.mean_gpu, l.variance_gpu);

                // CHECK backward mean, var = moments(outputs) -> x_stat_gpu
                fill_gpu(l.n, 0, l.x_stat_sum_gpu, 1);
                sum_outputs_gpu(l.x_stat_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.x_stat_sum_gpu);
                backward_stat_mean_var_gpu(l.x_stat_delta_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.x_stat_gpu, l.mean_gpu, l.x_stat_sum_gpu);

#ifdef CUDNN
                // CHECK backward, x_stat = weights * quantize_input
                cudnnConvolutionBackwardFilter(cudnn_handle(),
                        &one,
                        l.srcTensorDesc,
                        l.quantize_input_gpu,
                        l.ddstTensorDesc,
                        l.x_stat_delta_gpu,
                        l.convDesc,
                        l.bf_algo,
                        net.workspace,
                        l.workspace_size,
                        &one,
                        l.dweightDesc,
                        l.weight_updates_gpu_part);

                axpy_gpu(l.nweights, 1, l.weight_updates_gpu_part, 1, l.weight_updates_gpu, 1);

                if (!net.quantize_freezeBN) {
                     // CHECK backward x_stat = conv(x), for x
                    cudnnConvolutionBackwardData(cudnn_handle(),
                        &one,
                        l.weightDesc,
                        l.weights_gpu,
                        l.ddstTensorDesc,
                        l.x_stat_delta_gpu,
                        l.convDesc,
                        l.bd_algo,
                        net.workspace,
                        l.workspace_size,
                        &one,
                        l.dsrcTensorDesc,
                        net.delta_gpu);
                }
#endif
            }
        }
        else {
            fill_gpu(l.n, 0, l.bias_updates_gpu_part, 1);
            // CHECK backward y = conv(x) + merge_bias, for merge_bias
            backward_bias_gpu(l.bias_updates_gpu_part, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
            // CHECK backward merge_bias = quantize_gpu(bias)
            backward_quantize_gpu(l.bias_updates_gpu_part, l.biases_gpu, l.n, net.quantize_bias_bitwidth, net.quantize_bias_fraction_bitwidth);
            axpy_gpu(l.n, 1, l.bias_updates_gpu_part, 1, l.bias_updates_gpu, 1);

            // CHECK backward y = conv(x) + merge_bias, for conv (w)
            cudnnConvolutionBackwardFilter(cudnn_handle(),
                    &one,
                    l.srcTensorDesc,
                    l.quantize_input_gpu,
                    l.ddstTensorDesc,
                    l.delta_gpu,
                    l.convDesc,
                    l.bf_algo,
                    net.workspace,
                    l.workspace_size,
                    &zero,
                    l.dweightDesc,
                    l.weight_updates_gpu_part);
            // CHECK bacward merge_weights = quantize_gpu(weights)
            backward_quantize_gpu(l.weight_updates_gpu_part, l.weights_gpu, l.nweights, net.quantize_weight_bitwidth, net.quantize_weight_fraction_bitwidth);
            axpy_gpu(l.nweights, 1, l.weight_updates_gpu_part, 1, l.weight_updates_gpu, 1);
        }

        if(net.delta_gpu){
            // CHECK backward y = conv(x) + merge_bias, for x
#ifdef CUDNN
            cudnnConvolutionBackwardData(cudnn_handle(),
                    &one,
                    l.weightDesc,
                    l.merge_weights_gpu,
                    l.ddstTensorDesc,
                    l.delta_gpu,
                    l.convDesc,
                    l.bd_algo,
                    net.workspace,
                    l.workspace_size,
                    &one,
                    l.dsrcTensorDesc,
                    net.delta_gpu);
#endif
            if (l.quantize_feature) {
                // CHECK backward net.quantize_input_gpu = quantize(net.input_gpu)
                backward_quantize_gpu(net.delta_gpu, net.input_gpu, l.batch * l.inputs, net.quantize_feature_bitwidth, net.quantize_feature_fraction_bitwidth);
            }
        }

        if (l.batch_normalize) {
            // update rolling_mean_gpu and rolling_variance_gpu after backward
            scal_gpu(l.out_c, .99, l.rolling_mean_gpu, 1);
            axpy_gpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
            scal_gpu(l.out_c, .99, l.rolling_variance_gpu, 1);
            axpy_gpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);
        }
    }
    else {
        if(l.batch_normalize){
            backward_batchnorm_layer_gpu(l, net);
        } else {
            backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
        }
        float *original_input = net.input_gpu;

        if(l.xnor) net.input_gpu = l.binary_input_gpu;
#ifdef CUDNN
        float one = 1;
        cudnnConvolutionBackwardFilter(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                net.input_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bf_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dweightDesc,
                l.weight_updates_gpu);
        if(net.delta_gpu){
            if(l.binary || l.xnor) swap_binary(&l);
            cudnnConvolutionBackwardData(cudnn_handle(),
                    &one,
                    l.weightDesc,
                    l.weights_gpu,
                    l.ddstTensorDesc,
                    l.delta_gpu,
                    l.convDesc,
                    l.bd_algo,
                    net.workspace,
                    l.workspace_size,
                    &one,
                    l.dsrcTensorDesc,
                    net.delta_gpu);
            if(l.binary || l.xnor) swap_binary(&l);
            if(l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
        }
#else
        int m = l.n/l.groups;
        int n = l.size*l.size*l.c/l.groups;
        int k = l.out_w*l.out_h;

        int i, j;
        for(i = 0; i < l.batch; ++i){
            for(j = 0; j < l.groups; ++j){
                float *a = l.delta_gpu + (i*l.groups + j)*m*k;
                float *b = net.workspace;
                float *c = l.weight_updates_gpu + j*l.nweights/l.groups;

                float *im  = net.input_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;
                float *imd = net.delta_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;

                im2col_gpu_ext(im,          // input
                    l.c / l.groups,         // input channels
                    l.h, l.w,               // input size (h, w)
                    l.size, l.size,         // kernel size (h, w)
                    l.pad, l.pad,           // padding (h, w)
                    l.stride, l.stride,     // stride (h, w)
                    l.dilation, l.dilation, // dilation (h, w)
                    b);       // output

                gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

                if (net.delta_gpu) {
                    if (l.binary || l.xnor) swap_binary(&l);
                    a = l.weights_gpu + j*l.nweights/l.groups;
                    b = l.delta_gpu + (i*l.groups + j)*m*k;
                    c = net.workspace;
                    if (l.size == 1) {
                        c = imd;
                    }

                    gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

                    if (l.size != 1) {
                        col2im_gpu_ext(net.workspace,   // input
                            l.c/l.groups,   // input channels
                            l.h, l.w,   // input size (h, w)
                            l.size, l.size, // kernel size (h, w)
                            l.pad, l.pad,   // padding size (h, w)
                            l.stride, l.stride, // stride size (h, w)
                            l.dilation, l.dilation, // dilation size (h, w)
                            imd);   // output (delta)
                    }
                    if(l.binary || l.xnor) {
                        swap_binary(&l);
                    }
                }
                if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
            }
        }
#endif
    }
}

void pull_convolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_convolutional_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_convolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
    if(l.clip){
        constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
    }
}


