#include "heatmap_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_heatmap_layer(int batch, int w, int h, int keypoints_num)
{
    layer l = {0};
    l.type = HEATMAP;
    l.keypoints_num = keypoints_num;

    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = keypoints_num + 2;
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.cost = calloc(1, sizeof(float));
    l.outputs = h*w*l.out_c;
    l.inputs = l.outputs;
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));

    l.forward = forward_heatmap_layer;
    l.backward = backward_heatmap_layer;
#ifdef GPU
    l.forward_gpu = forward_heatmap_layer_gpu;
    l.backward_gpu = backward_heatmap_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "heatmap\n");
    srand(0);

    return l;
}

void resize_heatmap_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->out_w = w;
    l->out_h = h;

    l->outputs = h*w*(l->keypoints_num+2);
    l->inputs = l->outputs;
    //printf("outputs: %d\n", l->outputs);
    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

float gaussian_radius(float h, float w, float min_overlap)
{
    float a, b, c, sq, r1, r2, r3;

    // solve quadric function, x = (-b +/- sqrt(b^2 - 4ac)) / 2a
    // case 1: inner cut and outer cut
    // overlap = ((h - r) * (w - r)) / (2 * h * w - (h - r) * (w - r))
    a = 1;
    b = h + w;
    c = w * h * (1 - min_overlap) / (1 + min_overlap);
    sq = sqrtf(b * b - 4 * a * c);
    r1 = (b - sq) / (2*a);

    // case 2: inner cut
    // overlap = ((h - 2r) * (w - 2r)) / (h * w)
    a = 4;
    b = 2 * (h + w);
    c = (1 - min_overlap) * w * h;
    sq = sqrtf(b * b - 4 * a * c);
    r2 = (b - sq) / (2*a);

    // case 3: outer cut
    // overlap = (h * w)/((h + 2r) * (w + 2r))
    a  = 4 * min_overlap;
    b  = -2 * min_overlap * (h + w);
    c  = (min_overlap - 1) * w * h;
    sq = sqrtf(b * b - 4 * a * c);
    r3  = (b + sq) / (2*a);

    //printf("r1, r2, r3: %.4f, %.4f, %.4f\n", r1, r2, r3);

    return fminf(fminf(r1, r2), r3);
}

void draw_umich_gaussian(float *heatmaps, const layer l, int b, int k, int x, int y, int radius, int s, int v)
{
    //printf("%d %d %d %d %d %d\n", b, k, x, y, radius, s);
    float *heatmap = heatmaps + b * l.outputs + k * l.out_h * l.out_w;
    int diameter = 2 * radius + 1;
    int i, j, mapi, mapj;
    int left, right, top, bottom;

    // generate gaussian map, no need
    // (-h:h; -w:w), here h = w = radius
    // gaussian[j, i] = exp(-(vi * vi + vj * vj)) / (2 * sigma * sigma), here sigma = diameter
    // vj = -h + j, vi = -w + i;
    float gvalue, gx, gy;

    left = fminf(x, radius);
    right = fminf(l.out_w - x, radius + 1);
    top = fminf(y, radius);
    bottom = fminf(l.out_h - y, radius + 1);

    //printf("%d %d %d %d\n", left, right, top, bottom);

    for (j = y - top; j < y + bottom; ++j) {
        for (i = x - left; i < x + right; ++i) {
            // corresponding index
            mapj = radius - top + j - (y - top);
            mapi = radius - left + i - (x - left);
            gx = -radius + mapi;
            gy = -radius + mapj;
            gvalue = expf(-(gx * gx + gy * gy) / (2 * diameter / 6. * diameter / 6.));
            gvalue *= s;
            if (v == -1) gvalue *= -1;
            if (fabsf(heatmap[j * l.out_w + i]) < fabsf(gvalue))
                heatmap[j * l.out_w + i] = gvalue;
        }
    }
}

void get_heatmap_truths(float *heatmaps, int *heatmap_mask, const layer l, network net)
{
    // generate the heatmap_truths, including the heatmap, heatmap_offset, heatmap_mask
    // refer to datasets.py/__getitem__(self, index)
    int b, i, j, c, t, k;
    float w, h;
    float *keypoints, *keypoint;
    int radius, int_kpx, int_kpy;
    float vis, kpx, kpy;
    //int count = 0;
    for (b = 0; b < l.batch; ++b) {
        keypoints = net.truth + b * l.truths; // for batch
        for(t = 0; t < l.max_boxes; ++t) {
            keypoint = keypoints + t * (4 + l.keypoints_num * 3 + 1); // for each keypoint(x, y, w, h, keypoints+ .., id)
            //printf("x, y, w, h: %.2f, %.2f, %.2f, %.2f\n", keypoint[0], keypoint[1], keypoint[2], keypoint[3]);
            if(!keypoint[0]) break;
            w = keypoint[2] * l.out_w;
            h = keypoint[3] * l.out_h;
            //printf("h, w: %f, %f\n", h, w);
            radius = fmaxf(0, (int)(gaussian_radius(h, w, 0.8)));
            //printf("radius: %d\n", radius);
            //printf("keypoints_num: %d\n", l.keypoints_num);
            for (k = 0; k < l.keypoints_num; ++k) {
                vis = keypoint[4 + k * 3 + 0];
                kpx = keypoint[4 + k * 3 + 1];
                kpy = keypoint[4 + k * 3 + 2];
                if ( fabsf(vis) > 0 && kpx > 0 && kpx < 1 && kpy > 0 && kpy < 1) {
                    int_kpx = (int)(kpx * l.out_w);
                    int_kpy = (int)(kpy * l.out_h);
                    //printf("int_kpx, int_kpy: %d %d\n", int_kpx, int_kpy);
                    // set heatmap_mask
                    heatmap_mask[b * l.out_h * l.out_w + int_kpy * l.out_w + int_kpx] = 1;
                    // set heatmap_offset
                    heatmaps[b * l.outputs + l.keypoints_num * l.out_h * l.out_w + int_kpy * l.out_w + int_kpx] = kpx * l.out_w - int_kpx;
                    heatmaps[b * l.outputs + (l.keypoints_num + 1) * l.out_h * l.out_w + int_kpy * l.out_w + int_kpx] = kpy * l.out_h - int_kpy;
                    // set heatmap
                    draw_umich_gaussian(heatmaps, l, b, k, int_kpx, int_kpy, radius, 1, vis);
                    //count++;
                }
            }
        }
    }
    //printf("count: %d\n", count);
}

void forward_heatmap_layer(const layer l, network net)
{
    int i, j, b, c, index;
    int heatmap_visible_count = 0, heatmap_occlusion_count = 0, heatmap_count = 0;
    int heatmap_offset_count = 0;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    activate_array(l.output, l.batch*l.outputs, LOGISTIC);
    //activate_array(l.output, l.batch*l.keypoitns_num*l.out_h*l.out_w, TANH);
    //activate_array(l.output+l.batch*l.keypoitns_num*l.out_h*l.out_w, l.batch*2*l.out_h*l.out_w, LOGISTIC);
    clamp_cpu(l.batch*l.outputs, l.output, 1, 0.0001, 0.9999);
    //fabsf_clamp_cpu(l.batch*l.outputs, l.output, 1, 0.0001, 0.9999);
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    *(l.cost) = 0;

    // generate heatmap by groundtruth
    float *heatmap_truth = calloc(l.batch * l.outputs, sizeof(float));
    int *heatmap_mask = calloc(l.batch * 1 * l.out_h * l.out_w, sizeof(int));   // keep ground-truth index
    memset(heatmap_truth, 0, l.batch * l.outputs * sizeof(float));
    memset(heatmap_mask, 0, l.batch * 1 * l.out_h * l.out_w * sizeof(int));
    get_heatmap_truths(heatmap_truth, heatmap_mask, l, net);

    //printf("after get_heatmap_truths\n");
    //printf("alpha, beta: %f, %f\n", l.alpha, l.beta);

    float heatmap_loss = 0;
    float heatmap_offset_loss = 0;

    //printf("batch, %d\n", l.batch);

    for (b = 0; b < l.batch; ++b) {
        for (c = 0; c < l.out_c; ++c) {
            for (j = 0; j < l.out_h; ++j) {
                for (i = 0; i < l.out_w; ++i) {
                    // first keypoints_num for heatmap, then for heatmap_offset
                    // heatmap: focal_loss in centernet, refer https://github.com/xingyizhou/CenterNet
                    index = b * l.outputs + c * l.out_h * l.out_w + j * l.out_w + i;
                    if (c < l.keypoints_num) {
                        if (fabsf(heatmap_truth[index]) == 1) {
                            if (heatmap_truth[index] == 1) {
                                heatmap_visible_count += 1;
                            }
                            else {
                                heatmap_occlusion_count += 1;
                            }
                            // CHECK: l.output -> 1
                            float floss_part1, floss_part2;
                            floss_part1 = powf(1-l.output[index], l.alpha);
                            floss_part2 = logf(l.output[index]);
                            //printf("keypoints loss, %.4f %.4f\n", floss_part1, floss_part2);
                            heatmap_loss += (floss_part1 * floss_part2);  // normalization -1/N in last
                            l.delta[index] = floss_part1 / l.output[index] - l.alpha * floss_part1 * floss_part2 / (1 - l.output[index]);
                        }
                        else
                        {
                            float floss_part1, floss_part2, floss_part3;
                            // CHECK: l.output -> 0
                            floss_part1 = powf((1-fabsf(heatmap_truth[index])), l.beta);
                            floss_part2 = powf(l.output[index], l.alpha);
                            floss_part3 = logf(1-l.output[index]);
                            heatmap_loss += (floss_part1 * floss_part2 * floss_part3);
                            //printf("after heatmap_loss: %f\n", heatmap_loss);
                            l.delta[index] = floss_part1 * (l.alpha * floss_part2 / l.output[index] * floss_part3 - floss_part2 / (1 - l.output[index]));
                        }
                    }
                    else {
                        int mask_index = b * l.out_h * l.out_w + j * l.out_w + i;
                        if (heatmap_mask[mask_index] == 1) {
                            heatmap_offset_count += 1;
                            heatmap_offset_loss += fabsf(l.output[index] - heatmap_truth[index]);
                            if (l.output[index] > heatmap_truth[index]) {
                                l.delta[index] = -1;
                            }
                            else if (l.output[index] < heatmap_truth[index]) {
                                l.delta[index] = 1;
                            }
                        }
                    }
                }
            }
        }
    }

    free(heatmap_mask);
    free(heatmap_truth);

    heatmap_count = heatmap_visible_count + heatmap_occlusion_count;
    if (heatmap_count > 0) heatmap_loss /= (-heatmap_count);
    if (heatmap_offset_count > 0) heatmap_offset_loss = heatmap_offset_loss * 2 / (heatmap_offset_count + 0.0001);

    *(l.cost) = heatmap_loss + heatmap_offset_loss;

    // normalize delta
    for (b = 0; b < l.batch; ++b) {
        for (c = 0; c < l.out_c; ++c) {
            for (j = 0; j < l.out_h; ++j) {
                for (i = 0; i < l.out_w; ++i) {
                    index = b * l.outputs + c * l.out_h * l.out_w + j * l.out_w + i;
                    if (c < l.keypoints_num) {
                        l.delta[index] *= (1. / heatmap_count);
                    }
                    else {
                        int mask_index = b * l.out_h * l.out_w + j * l.out_w + i;
                        if (heatmap_mask[mask_index] == 1) {
                            l.delta[index] *= (2. / (heatmap_offset_count + 0.0001));
                        }
                    }
                }
            }
        }
    }

    if (get_current_batch(&net) % net.log_step == 0)
        fprintf(stderr, "heatmap %d, heatmap_visible_count: %d,  heatmap_occlusion_count: %d, heatmap_offset_count: %d, heatmap_loss = %f, heatmap_offset_loss = %f, total_loss = %f \n",
            net.index,
            heatmap_visible_count,
            heatmap_occlusion_count,
            heatmap_offset_count,
            heatmap_loss,
            heatmap_offset_loss,
            *(l.cost));
}

void backward_heatmap_layer(const layer l, network net)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU
void forward_heatmap_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);

    if (l.post_training_quantization) {
        //printf("yolo %d, restore: %d\n", l.current_layer_index, *(net.fl));
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        layer pre_l = net.layers[l.current_layer_index - 1];
        if (pre_l.quantize_per_channel) {
            int c, spatial_size;
            spatial_size = l.out_w * l.out_h;
            for (c = 0; c < l.c; ++c) {
                restore(l.output + spatial_size * c, spatial_size, pre_l.bias_fls[c]);
            }
        } else {
            restore(l.output, l.batch*l.outputs, *(net.fl));
        }
        cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    }

    // sigmoid activation
    activate_array_gpu(l.output_gpu, l.batch * l.outputs, LOGISTIC, 0);
    // TANH for first keypoints_nums, then LOGISTIC
    //activate_array_gpu(l.output_gpu, l.batch * l.keypoints_num * l.out_h * l.out_w, TANH, 0);
    //activate_array_gpu(l.output_gpu + l.batch * l.keypoints_num * l.out_h * l.out_w, l.batch * 2 * l.out_h * l.out_w, LOGISTIC, 0);
    // clamp
    clamp_gpu(l.batch * l.outputs, l.output_gpu, 1, 0.0001, 0.9999);
    //fabsf_clamp_gpu(l.batch * l.outputs, l.output_gpu, 1, 0.0001, 0.9999);  // CHECK: fabsf(x) ~ (0.0001, 0.9999)

    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_heatmap_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_heatmap_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif