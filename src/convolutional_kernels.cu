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
#include "cost_layer.h"
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
        // CHECK: l.quantization_aware_training
        if (l.quantization_aware_training) {
            /*
            //printf("feature quantize: %d %d\n", l.quantize_feature_bitwidth, l.quantize_feature_fraction_bitwidth);
            cuda_pull_array(net.input_gpu, net.input, l.batch*l.inputs);
            int o, p, q;
            printf("layer %d, input:\n", l.current_layer_index);
            for (o=0; o<2; ++o) {
                for(p=5; p<10; ++p) {
                    for(q=5; q<10; ++q) {
                        printf("%f ", net.input[o*l.h*l.w+p*l.w+q]);
                    }
                    printf("\n");
                }
            }
            printf("\n");
            */

            if(l.batch_normalize) {
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
                if (l.quantize_per_channel) {
                    int k, filter_size;
                    filter_size = l.nweights / l.n;
                    for(k = 0; k < l.n; ++k) {
                        quantize_gpu(l.merge_weights_gpu+k*filter_size, filter_size, l.quantize_weight_bitwidth, l.quantize_weight_fraction_bitwidths[k], 0, 1);
                    }
                } else {
                    quantize_gpu(l.merge_weights_gpu, l.nweights, l.quantize_weight_bitwidth, l.quantize_weight_fraction_bitwidth, 0, 1);
                }

    #ifdef CUDNN
                // free the results of output_gpu in first conv, or use zero in cudnnConvolutionForward
                fill_gpu(l.batch*l.outputs, 0, l.output_gpu, 1);
                one = 1;
                cudnnConvolutionForward(cudnn_handle(),
                            &one,
                            l.srcTensorDesc,
                            net.input_gpu,
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
                if (l.quantize_per_channel) {
                    int k;
                    for (k = 0; k < l.n; ++k) {
                        quantize_gpu(l.merge_biases_gpu+k, 1, l.quantize_bias_bitwidth, l.quantize_bias_fraction_bitwidths[k], 0, 1);
                    }
                } else {
                    quantize_gpu(l.merge_biases_gpu, l.n, l.quantize_bias_bitwidth, l.quantize_bias_fraction_bitwidth, 0, 1);
                }
                add_bias_gpu(l.output_gpu, l.merge_biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
            } else {

                copy_gpu(l.nweights, l.weights_gpu, 1, l.merge_weights_gpu, 1);
                if (l.quantize_per_channel) {
                    int k, filter_size;
                    filter_size = l.nweights / l.n;
                    for(k = 0; k < l.n; ++k) {
                        quantize_gpu(l.merge_weights_gpu+k*filter_size, filter_size, l.quantize_weight_bitwidth, l.quantize_weight_fraction_bitwidths[k], 0, 1);
                    }
                } else {
                    quantize_gpu(l.merge_weights_gpu, l.nweights, l.quantize_weight_bitwidth, l.quantize_weight_fraction_bitwidth, 0, 1);
                }

                /*
                cuda_pull_array(l.merge_weights_gpu, l.weights, l.nweights);
                printf("layer %d quantization weights: %.10f %.10f %.10f %.10f %.10f\n", l.current_layer_index, l.weights[5], l.weights[6], l.weights[7], l.weights[8], l.weights[9]);
                */

    #ifdef CUDNN
                float one = 1;
                cudnnConvolutionForward(cudnn_handle(),
                            &one,
                            l.srcTensorDesc,
                            net.input_gpu,
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
                /*
                cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
                printf("layer %d, conv output:\n", l.current_layer_index);
                for (o=0; o<2; ++o) {
                    for(p=5; p<10; ++p) {
                        for(q=5; q<10; ++q) {
                            printf("%f ", l.output[o*l.h*l.w+p*l.w+q]);
                        }
                        printf("\n");
                    }
                }
                printf("\n");
                */

                if (net.write_results) {
                    char buff[100];
                    sprintf(buff, "ship/statistics/outputs/convolution_conv_%02d.dat", l.current_layer_index);
                    FILE *fp;
                    fp = fopen(buff, "wb");
                    fwrite(l.output, sizeof(float), l.batch*l.outputs, fp);
                    fclose(fp);
                }


                copy_gpu(l.n, l.biases_gpu, 1, l.merge_biases_gpu, 1);
                if (l.quantize_per_channel) {
                    int k;
                    for (k = 0; k < l.n; ++k) {
                        quantize_gpu(l.merge_biases_gpu+k, 1, l.quantize_bias_bitwidth, l.quantize_bias_fraction_bitwidths[k], 0, 1);
                    }
                } else {
                    quantize_gpu(l.merge_biases_gpu, l.n, l.quantize_bias_bitwidth, l.quantize_bias_fraction_bitwidth, 0, 1);
                }

                /*
                cuda_pull_array(l.merge_biases_gpu, l.biases, l.n);
                printf("layer %d: quantization biases: %f %f %f %f %f\n", l.current_layer_index, l.biases[0], l.biases[1], l.biases[2], l.biases[3], l.biases[4]);
                */

                add_bias_gpu(l.output_gpu, l.merge_biases_gpu, l.batch, l.n, l.out_w*l.out_h);

                /*
                cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
                printf("layer %d, add bias output:\n", l.current_layer_index);
                for (o=0; o<2; ++o) {
                    for(p=5; p<10; ++p) {
                        for(q=5; q<10; ++q) {
                            printf("%f ", l.output[o*l.h*l.w+p*l.w+q]);
                        }
                        printf("\n");
                    }
                }
                printf("\n");
                */

                if (net.write_results) {
                    char buff[100];
                    sprintf(buff, "ship/statistics/outputs/convolution_bias_%02d.dat", l.current_layer_index);
                    FILE *fp;
                    fp = fopen(buff, "wb");
                    fwrite(l.output, sizeof(float), l.batch*l.outputs, fp);
                    fclose(fp);
                }
            }
        }
        // CHECK: l.post_training_quantization
        else
        {
            /*
            cuda_pull_array(net.input_gpu, net.input, l.batch*l.inputs);
            if (l.quantize_per_channel) {
                int j, pre_index = -1;
                for (j = l.current_layer_index - 1; j >= 0; j--) {
                    if (net.layers[j].type == CONVOLUTIONAL) {
                        pre_index = j;
                        break;
                    }
                    if (net.layers[j].type == ROUTE) {
                        int index = net.layers[j].input_layers[0];
                        j = index;
                    }
                }
                //printf("pre_index: %d\n", pre_index);
                if (pre_index >= 0)
                    restore(net.input, l.batch*l.inputs, *(net.layers[pre_index].x_fl));
            } else {
                restore(net.input, l.batch*l.inputs, *(net.fl));
            }
            int o, p, q;
            printf("layer %d, input:\n", l.current_layer_index);
            for (o=0; o<2; ++o) {
                for(p=5; p<10; ++p) {
                    for(q=5; q<10; ++q) {
                        printf("%f ", net.input[o*l.h*l.w+p*l.w+q]);
                    }
                    printf("\n");
                }
            }
            printf("\n");
            */

            /*
            cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
            if (l.quantize_per_channel) {
                int c, spatial_size;
                spatial_size = l.nweights / l.n;
                printf("conv_fls: ");
                for(c = 0; c < l.n; ++c) {
                    printf("%d ", l.conv_fls[c]);
                    restore(l.weights + spatial_size * c, spatial_size, l.conv_fls[c]);
                }
                printf("\n");
            } else {
                restore(l.weights, l.nweights, *(l.conv_fl));
            }
            printf("layer %d quantization weights: %.10f %.10f %.10f %.10f %.10f\n", l.current_layer_index, l.weights[5], l.weights[6], l.weights[7], l.weights[8], l.weights[9]);
            */

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

            if (!l.quantize_per_channel)
                *(net.fl) += *(l.conv_fl);

            /*
            cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
            if (l.quantize_per_channel) {
                int c, spatial_size;
                spatial_size = l.outputs / l.n;
                for(c = 0; c < l.n; ++c) {
                    restore(l.output + spatial_size * c, spatial_size, l.bias_fls[c]);
                }
            } else {
                restore(l.output, l.batch*l.outputs, *(net.fl));
            }
            printf("layer %d, conv output:\n", l.current_layer_index);
            for (o=0; o<2; ++o) {
                for(p=5; p<10; ++p) {
                    for(q=5; q<10; ++q) {
                        printf("%f ", l.output[o*l.h*l.w+p*l.w+q]);
                    }
                    printf("\n");
                }
            }
            printf("\n");
            */

            if (net.write_results) {
                char buff[100];
                sprintf(buff, "ship/statistics/outputs/convolution_conv_%02d.dat", l.current_layer_index);
                FILE *fp;
                fp = fopen(buff, "wb");
                fwrite(l.output, sizeof(float), l.batch*l.outputs, fp);
                fclose(fp);
            }


            if (!net.convx_bias_align) {
                // quantize: shift output to bias, and add bias, update net.fl
                int diff = *(l.bias_fl) - *(net.fl);
                //printf("3. diff %d\n", diff);
                cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
                //printf("3.5\n");
                logicShift(l.output, l.batch*l.outputs, diff);
                cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
                *(net.fl) = *(l.bias_fl);
            }

            /*
            cuda_pull_array(l.biases_gpu, l.biases, l.n);
            if (l.quantize_per_channel) {
                int c;
                printf("bias_fl: ");
                for(c = 0; c < l.n; ++c) {
                    printf("%d ", l.bias_fls[c]);
                    restore(l.biases + c, 1, l.bias_fls[c]);
                }
                printf("\n");
            } else {
                restore(l.biases, l.n, *(l.bias_fl));
            }
            printf("layer %d: quantization biases: %f %f %f %f %f\n", l.current_layer_index, l.biases[0], l.biases[1], l.biases[2], l.biases[3], l.biases[4]);
            */

            add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);

            /*
            cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
            if (l.quantize_per_channel) {
                int c, spatial_size;
                spatial_size = l.outputs / l.n;
                for(c = 0; c < l.n; ++c) {
                    restore(l.output + spatial_size * c, spatial_size, l.bias_fls[c]);
                }
            } else {
                restore(l.output, l.batch*l.outputs, *(net.fl));
            }
            printf("layer %d, add bias output:\n", l.current_layer_index);
            for (o=0; o<2; ++o) {
                for(p=5; p<10; ++p) {
                    for(q=5; q<10; ++q) {
                        printf("%f ", l.output[o*l.h*l.w+p*l.w+q]);
                    }
                    printf("\n");
                }
            }
            printf("\n");
            */

            if (net.write_results) {
                char buff[100];
                sprintf(buff, "ship/statistics/outputs/convolution_bias_%02d.dat", l.current_layer_index);
                FILE *fp;
                fp = fopen(buff, "wb");
                fwrite(l.output, sizeof(float), l.batch*l.outputs, fp);
                fclose(fp);
            }

        }
    }
    else
    {
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

    if (l.activation == SWISH)
        copy_gpu(l.batch*l.outputs, l.output_gpu, 1, l.output_afterbn_gpu, 1);

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.leaky_rate);

    /*
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    if (l.post_training_quantization) {
        if (l.quantize_per_channel) {
            int c, spatial_size;
            spatial_size = l.outputs / l.n;
            for(c = 0; c < l.n; ++c) {
                restore(l.output + spatial_size * c, spatial_size, l.bias_fls[c]);
            }
        } else {
            restore(l.output, l.batch*l.outputs, *(net.fl));
        }
    }
    printf("layer %d, leaky output:\n", l.current_layer_index);
    int o,p,q;
    for (o=0; o<2; ++o) {
        for(p=5; p<10; ++p) {
            for(q=5; q<10; ++q) {
                printf("%f ", l.output[o*l.h*l.w+p*l.w+q]);
            }
            printf("\n");
        }
    }
    printf("\n");
    */

    if (net.write_results) {
        char buff[100];
        sprintf(buff, "ship/statistics/outputs/convolution_leaky_%02d.dat", l.current_layer_index);
        FILE *fp;
        fp = fopen(buff, "wb");
        fwrite(l.output, sizeof(float), l.batch*l.outputs, fp);
        fclose(fp);
    }

    if (l.quantize) {
        if(l.post_training_quantization) {
            if (l.activation != LINEAR) {
                cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
                if (!net.convx_bias_align) {
                    int x_fl = quantizeOutputs(l.output, l.batch*l.outputs);
                    *(net.fl) += x_fl;
                }
                else {
                    if (!l.quantize_per_channel) {
                        int diff = *(net.fl) - *(l.x_fl);
                        //printf("convolution: %d, diff %d, fearue_bitwidth: %d\n", l.current_layer_index, diff, l.quantize_feature_bitwidth);
                        logicShiftAlign(l.output, l.batch*l.outputs, l.quantize_feature_bitwidth, -diff);
                        *(net.fl) = *(l.x_fl);
                    } else {
                        int c, diff, spatial_size;
                        spatial_size = l.outputs / l.n;
                        // l.batch = 1
                        for (c = 0; c < l.n; ++c) {
                            diff = l.bias_fls[c] - *(l.x_fl);
                            logicShiftAlign(l.output+spatial_size * c, spatial_size, l.quantize_feature_bitwidth, -diff);
                        }
                    }
                }
                cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);

                /*
                if (l.quantize_per_channel) {
                    restore(l.output, l.batch*l.outputs, *(l.x_fl));
                } else {
                    restore(l.output, l.batch*l.outputs, *(net.fl));
                }
                printf("layer %d, quantize output:\n", l.current_layer_index);
                for (o=0; o<2; ++o) {
                    for(p=5; p<10; ++p) {
                        for(q=5; q<10; ++q) {
                            printf("%f ", l.output[o*l.h*l.w+p*l.w+q]);
                        }
                        printf("\n");
                    }
                }
                printf("\n");
                */
            }

            if (!l.quantize_per_channel)
                *(l.x_fl) = *(net.fl);

            if (net.write_statistic_fl) {
                char buff[20];
                sprintf(buff, "convolution %d: %d\n", l.current_layer_index, *(l.x_fl));
                fwrite(buff, sizeof(char), strlen(buff), net.filewriter_fl);
            }

        } else {
            // quantize the features
            if (l.quantize_feature) {
                copy_gpu(l.batch*l.outputs, l.output_gpu, 1, l.quantize_output_gpu, 1);

                if (l.quantize_per_channel) {
                    int b, c, spatial_size;
                    spatial_size = l.outputs / l.n;
                    for (b = 0; b < l.batch; ++b) {
                        for (c = 0; c < l.n; ++c) {
                            quantize_gpu(l.output_gpu + l.outputs * b + spatial_size * c, spatial_size,
                                l.quantize_feature_bitwidth, l.quantize_feature_fraction_bitwidth, l.quantize_bias_fraction_bitwidths[c], 0);
                        }
                    }
                } else {
                    quantize_gpu(l.output_gpu, l.batch*l.outputs, l.quantize_feature_bitwidth,
                            l.quantize_feature_fraction_bitwidth, l.quantize_bias_fraction_bitwidth, 0);
                }

                /*
                cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
                printf("layer %d, quantize output:\n", l.current_layer_index);
                for (o=0; o<2; ++o) {
                    for(p=5; p<10; ++p) {
                        for(q=5; q<10; ++q) {
                            printf("%f ", l.output[o*l.h*l.w+p*l.w+q]);
                        }
                        printf("\n");
                    }
                }
                printf("\n");
                */
            }
        }
    }

    if (net.write_results) {
        if(l.activation == LINEAR) cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        char buff[100];
        sprintf(buff, "ship/statistics/outputs/convolution_%02d.dat", l.current_layer_index);
        FILE *fp;
        fp = fopen(buff, "wb");
        fwrite(l.output, sizeof(float), l.batch*l.outputs, fp);
        fclose(fp);
    }

    if (net.write_statistic_features) {
        // quantize: statistic the feature maps, for statistic lower feature_fraction_bitwidth
        cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
        qsort(l.output, l.batch*l.outputs, sizeof(float), float_compare);
        float mean;
        float variance;
        //printf("before get_mean_variance\n");
        get_mean_variance(l.output, l.outputs*l.batch, &mean, &variance);
        //printf("after get_mean_variance\n");
        //printf("convolution %d output, min: %.4f, max: %.4f\n", l.current_layer_index, l.output[0], l.output[l.outputs*l.batch-1]);
        fprintf(net.filewriter_features, "convolution %d output, min: %.4f, max: %.4f, mean: %.4f, var: %.4f\n", l.current_layer_index, l.output[0], l.output[l.outputs*l.batch-1], mean, variance);
    }

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

    if(l.quantization_aware_training) {
        if (l.quantize_feature) {
            // CHECK backward l.output_gpu = quantize(l.output_gpu);
            backward_quantize_gpu(l.delta_gpu, l.quantize_output_gpu, l.batch*l.outputs, l.quantize_feature_bitwidth, l.quantize_feature_fraction_bitwidth);
        }
    }

    if (l.activation == SWISH) {
        gradient_array_gpu(l.output_afterbn_gpu, l.outputs*l.batch, l.activation, l.delta_gpu, 0);
    } else {
        gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu, l.leaky_rate);
    }

    // CHECK since the gradient accumulate each backward, so there need to use temp updates
    if(l.quantize) {
        float one = 1, zero = 0;
        if(l.batch_normalize) {
            // 1. for bias_updates_gpu
            fill_gpu(l.n, 0, l.bias_updates_gpu_part, 1);
            // CHECK backward y = conv(x) + merge_bias, for merge_bias
            backward_bias_gpu(l.bias_updates_gpu_part, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
            // CHECK backward merge_bias = quantize_gpu(shift_bias)
            if (l.quantize_per_channel) {
                int k;
                for (k = 0; k < l.n; ++k) {
                    backward_quantize_gpu(l.bias_updates_gpu_part+k, l.shift_biases_gpu+k, 1, l.quantize_bias_bitwidth, l.quantize_bias_fraction_bitwidths[k]);
                }
            } else {
                backward_quantize_gpu(l.bias_updates_gpu_part, l.shift_biases_gpu, l.n, l.quantize_bias_bitwidth, l.quantize_bias_fraction_bitwidth);
            }
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
            // CHECK backward conv(x) = merge_weights * input
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
                    &zero,
                    l.dweightDesc,
                    l.weight_updates_gpu_part);
#endif

            // CHECK backward merge_weights = quantize_gpu(scale_weights)
            if (l.quantize_per_channel) {
                int k, filter_size;
                filter_size = l.nweights / l.n;
                for (k = 0; k < l.n; ++k) {
                    backward_quantize_gpu(l.weight_updates_gpu_part+filter_size*k, l.scale_weights_gpu+filter_size*k, filter_size, l.quantize_weight_bitwidth, l.quantize_weight_fraction_bitwidths[k]);
                }
            } else {
                backward_quantize_gpu(l.weight_updates_gpu_part, l.scale_weights_gpu, l.nweights, l.quantize_weight_bitwidth, l.quantize_weight_fraction_bitwidth);
            }

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
                // CHECK backward, x_stat = weights * input
                cudnnConvolutionBackwardFilter(cudnn_handle(),
                        &one,
                        l.srcTensorDesc,
                        net.input_gpu,
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
            if (l.quantize_per_channel) {
                int k;
                for (k = 0; k < l.n; ++k) {
                    backward_quantize_gpu(l.bias_updates_gpu_part+k, l.shift_biases_gpu+k, 1, l.quantize_bias_bitwidth, l.quantize_bias_fraction_bitwidths[k]);
                }
            } else {
                backward_quantize_gpu(l.bias_updates_gpu_part, l.shift_biases_gpu, l.n, l.quantize_bias_bitwidth, l.quantize_bias_fraction_bitwidth);
            }
            axpy_gpu(l.n, 1, l.bias_updates_gpu_part, 1, l.bias_updates_gpu, 1);

            // CHECK backward y = conv(x) + merge_bias, for conv (w)
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
                    &zero,
                    l.dweightDesc,
                    l.weight_updates_gpu_part);
            // CHECK bacward merge_weights = quantize_gpu(weights)
            if (l.quantize_per_channel) {
                int k, filter_size;
                filter_size = l.nweights / l.n;
                for (k = 0; k < l.n; ++k) {
                    backward_quantize_gpu(l.weight_updates_gpu_part+filter_size*k, l.weights_gpu+filter_size*k, filter_size, l.quantize_weight_bitwidth, l.quantize_weight_fraction_bitwidths[k]);
                }
            } else {
                backward_quantize_gpu(l.weight_updates_gpu_part, l.weights_gpu, l.nweights, l.quantize_weight_bitwidth, l.quantize_weight_fraction_bitwidth);
            }

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
        }

        if (l.batch_normalize) {
            if (!net.quantize_freezeBN) {
                // update rolling_mean_gpu and rolling_variance_gpu after backward
                scal_gpu(l.out_c, .99, l.rolling_mean_gpu, 1);
                axpy_gpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
                scal_gpu(l.out_c, .99, l.rolling_variance_gpu, 1);
                axpy_gpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);
            }
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
            if(l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu, 0);
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
                if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w, 0);
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


