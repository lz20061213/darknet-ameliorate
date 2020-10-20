#include "distill_yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_distill_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int distill_index) {
    int i;
    layer l = {0};
    l.type = DISTILL_YOLO;

    l.n = n;  // anchor nums in this layer
    l.total = total;  // total anchor nums in this network
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    //l.c = n * (classes + 12 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total * 2, sizeof(float));
    if (mask) l.mask = mask;
    else {
        l.mask = calloc(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            l.mask[i] = i;
        }
    }
    l.distill_index = distill_index;

    l.bias_updates = calloc(n * 2, sizeof(float));
    //l.outputs = h*w*n*(classes + 4 + 1);
    l.outputs = h * w * l.c;
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    //l.truths = 90 * (12 + 1);
    l.delta = calloc(batch * l.outputs, sizeof(float));
    l.output = calloc(batch * l.outputs, sizeof(float));
    //l.mimic_truth = calloc(batch * l.outputs, sizeof(float));
    for (i = 0; i < total * 2; ++i) {
        l.biases[i] = .5;
    }

    l.mimic_forward = forward_distill_yolo_layer;
    l.backward = backward_distill_yolo_layer;
#ifdef GPU
    l.mimic_forward_gpu = forward_distill_yolo_layer_gpu;
    l.backward_gpu = backward_distill_yolo_layer_gpu;
    //l.mimic_truth_gpu = cuda_make_array(l.mimic_truth, batch*l.outputs);
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "distill yolo\n");
    srand(0);

    return l;
}

void resize_distill_yolo_layer(layer *l, int w, int h) {
    l->w = w;
    l->h = h;
    l->out_w = w;
    l->out_h = h;
    //printf("resize_distill_yolo: %d %d\n", w, h);

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    //printf("outputs,: %d\n", l->outputs);
    //l->outputs = h * w * l->n * (l->classes + 12 + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch * l->outputs * sizeof(float));
    l->delta = realloc(l->delta, l->batch * l->outputs * sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_distill_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
    box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

float delta_distill_yolo_box(box truth, float *x, float *mimic_truth, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h,
                     float *delta, float scale, int stride, float margin) {
    box spred = get_distill_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    box tpred = get_distill_yolo_box(mimic_truth, biases, n, index, i, j, lw, lh, w, h, stride);
    float siou = box_iou(spred, truth);
    float tiou = box_iou(tpred, truth);

    //printf("105 in delta_distill_yolo_box, siou: %f, tiou: %f, scale: %f\n", siou, tiou, scale);

    float tx = (truth.x * lw - i);
    float ty = (truth.y * lh - j);
    float tw = log(truth.w * w / biases[2 * n]);
    float th = log(truth.h * h / biases[2 * n + 1]);

    if (siou + margin < tiou)  scale *= 1.5;

    delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
    delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
    delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);

    return siou;
}

void delta_distill_yolo_class(float *output, float *mimic_truth, float *delta, int index, int class, int classes,
                              int stride, float alpha, float *avg_cat) {
    int n;
    float hard, soft;
    //printf("distill_yolo_layer in 125, output: %f, mimic_truth: %f\n", output[index + stride * class], mimic_truth[index + stride * class]);
    //the return will be fused in mimic_train format
    if (delta[index]) {
        hard = 1 - output[index + stride * class];
        soft = mimic_truth[index + stride * class] - output[index + stride * class];  // corss-entropy
        delta[index + stride * class] = alpha * hard + (1-alpha) * soft;
        if (avg_cat) *avg_cat += output[index + stride * class];
        return;
    }
    for (n = 0; n < classes; ++n) {
        hard = ((n == class)?1 : 0) - output[index + stride * n];
        soft = mimic_truth[index + stride * n] - output[index + stride * n];
        delta[index + stride * n] = alpha * hard + (1-alpha) * soft;
        if (n == class && avg_cat) *avg_cat += output[index + stride * n];
    }
}

static int entry_index(layer l, int batch, int location, int entry) {
    int n = location / (l.w * l.h);
    int loc = location % (l.w * l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
    //return batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1) + entry * l.w * l.h + loc;
}

void forward_distill_yolo_layer(const layer l, network snet, network tnet) {
    int i, j, b, t, n;
    memcpy(l.output, snet.input, l.outputs * l.batch * sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n * l.w * l.h, 0);   // bbox/poly active, linear
            activate_array(l.output + index, 2 * l.w * l.h, LOGISTIC);  // x, y activate
            index = entry_index(l, b, n*l.w*l.h, 4);  // class active
            activate_array(l.output + index, (1 + l.classes) * l.w * l.h, LOGISTIC);
        }
    }
#endif

    //cuda_set_device(tnet.gpu_index);
    int t_index = snet.distill_layers[l.distill_index];
    layer lt = tnet.layers[t_index];
    //printf("assert: %d %d %d\n", t_index, l.outputs, tnet.layers[t_index].outputs);
    assert(l.outputs == lt.outputs);
    //cuda_pull_array(tnet.layers[t_index].output_gpu, lt.output, l.batch*l.outputs);
    //memcpy(lt.output, tnet.layers[t_index].output, l.batch*l.outputs*sizeof(float));

    // no need push the data from cpu to snet gpu

    // printf("yolo 106: %d, %d, %d, %f\n", l.w, l.h, l.n, l.output[0]);
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if (!snet.train) return;
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 0);
                    box pred = get_distill_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, snet.w, snet.h,
                                              l.w * l.h);
                    float best_iou = 0;
                    int best_t = 0;
                    for (t = 0; t < l.max_boxes; ++t) {
                        box truth = float_to_box(snet.truth + t * (4 + 1) + b * l.truths, 1);
                        if (!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4);
                    avg_anyobj += l.output[obj_index];
                    // for objectness
                    float hard, soft;
                    hard = 0 - l.output[obj_index];
                    soft = lt.output[obj_index] - l.output[obj_index];
                    l.delta[obj_index] = l.alpha * hard + (1-l.alpha) * soft;
                    if (best_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;
                    }
                    if (best_iou > l.truth_thresh) {
                        hard = 1 - l.output[obj_index];
                        l.delta[obj_index] = l.alpha * hard + (1-l.alpha) * soft;

                        int class = snet.truth[best_t * (4 + 1) + b * l.truths + 4];
                        if (l.map) class = l.map[class];
                        int class_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4 + 1);
                        delta_distill_yolo_class(l.output, lt.output, l.delta, class_index, class, l.classes, l.w * l.h, l.alpha, 0);
                        box truth = float_to_box(snet.truth + best_t * (4 + 1) + b * l.truths, 1);
                        delta_distill_yolo_box(truth, l.output, lt.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, snet.w, snet.h,
                                        l.delta, (2 - truth.w * truth.h), l.w * l.h, l.margin);
                    }
                }
            }
        }
        //printf("distill_yolo_layer in 227\n");
        for (t = 0; t < l.max_boxes; ++t) {
            box truth = float_to_box(snet.truth + t * (4 + 1) + b * l.truths, 1);

//            printf("distill_yolo_layer in 230: %f %f %f %f\n", truth.x, truth.y, truth.w, truth.h);

            if (!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for (n = 0; n < l.total; ++n) {
                box pred = {0};
                pred.w = l.biases[2 * n] / snet.w;
                pred.h = l.biases[2 * n + 1] / snet.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            if (mask_n >= 0) {
                int box_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 0);
                float iou = delta_distill_yolo_box(truth, l.output, lt.output, l.biases, best_n, box_index, i, j, l.w, l.h, snet.w, snet.h,
                                            l.delta, (2 - truth.w * truth.h), l.w * l.h, l.margin);
                int obj_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4);
                avg_obj += l.output[obj_index];

                float hard, soft;
                hard = 1 - l.output[obj_index];
                soft = lt.output[obj_index] - l.output[obj_index];
                //printf("distill_yolo_layer.c in 260: mimic_truth: %f, output: %f\n", lt.output[obj_index], l.output[obj_index]);
                l.delta[obj_index] = l.alpha * hard + (1-l.alpha) * soft;

                int class = snet.truth[t * (4 + 1) + b * l.truths + 4];
                if (l.map) class = l.map[class];
                int class_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4 + 1);
                delta_distill_yolo_class(l.output, lt.output, l.delta, class_index, class, l.classes, l.w * l.h, l.alpha, &avg_cat);

                ++count;
                ++class_count;
                if (iou > .5) recall += 1;
                if (iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);

    // for check
    /*
    int k;
    if (*(l.cost) > 100 || (count>=1 && *(l.cost) < 0.5)) {
        for (b = 0; b < l.batch; ++b) {
            printf("batch %d:\n", b);
            for (j = 0; j < l.h; ++j) {
                for (i = 0; i < l.w; ++i) {
                    for (n = 0; n < l.n; ++n) {
                        int box_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 0);
                        int obj_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4);
                        int class_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4 + 1);
                        int stride = l.w*l.h;

                        box spred = get_distill_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, snet.w, snet.h,
                                                  l.w * l.h);
                        printf("spred: %.4f %.4f %.4f %.4f ", spred.x, spred.y, spred.w, spred.h);
                        printf("%.4f ", l.output[obj_index]);
                        for(k=0; k < l.classes; ++k) {
                            printf("%.4f ", l.output[class_index+k*stride]);
                        }
                        printf("\n");

                        box tpred = get_distill_yolo_box(lt.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, snet.w, snet.h,
                                                  l.w * l.h);
                        printf("tpred: %.4f %.4f %.4f %.4f ", tpred.x, tpred.y, tpred.w, tpred.h);
                        printf("%.4f ", lt.output[obj_index]);
                        for(k=0; k < l.classes; ++k) {
                            printf("%.4f ", lt.output[class_index+k*stride]);
                        }
                        printf("\n");

                        printf("delta: %.4f %.4f %.4f %.4f ", l.delta[box_index+0*stride], l.delta[box_index+1*stride], l.delta[box_index+2*stride], l.delta[box_index+3*stride]);
                        printf("%.4f ", l.delta[obj_index]);
                        for(k=0; k < l.classes; ++k) {
                            printf("%.4f ", l.delta[class_index+k*stride]);
                        }
                        printf("\n");
                    }
                }
            }
            printf("begin to check gt boxes\n");
            for (t = 0; t < l.max_boxes; ++t) {
                box truth = float_to_box(snet.truth + t * (4 + 1) + b * l.truths, 1);
                box t_truth = float_to_box(tnet.truth + t * (4 + 1) + b * l.truths, 1);
                printf("enter\n");
                printf("struth: %.4f %.4f %.4f %.4f\n", truth.x, truth.y, truth.w, truth.h);
                printf("ttruth: %.4f %.4f %.4f %.4f\n", t_truth.x, t_truth.y, t_truth.w, t_truth.h);
                if (!truth.x) break;
                float best_iou = 0;
                int best_n = 0;
                i = (truth.x * l.w);
                j = (truth.y * l.h);
                box truth_shift = truth;
                truth_shift.x = truth_shift.y = 0;
                for (n = 0; n < l.total; ++n) {
                    box pred = {0};
                    pred.w = l.biases[2 * n] / snet.w;
                    pred.h = l.biases[2 * n + 1] / snet.h;
                    float iou = box_iou(pred, truth_shift);
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_n = n;
                    }
                }

                int mask_n = int_index(l.mask, best_n, l.n);
                if (mask_n >= 0) {
                    int class = snet.truth[t * (4 + 1) + b * l.truths + 4];
                    printf("gt: %.4f %.4f %.4f %.4f %d\n", truth.x, truth.y, truth.w, truth.h, class);
                    int box_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 0);
                    int obj_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4);
                    int class_index = entry_index(l, b, mask_n * l.w * l.h + j * l.w + i, 4 + 1);
                    int stride = l.w*l.h;

                    box spred = get_distill_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, snet.w, snet.h,
                                              l.w * l.h);

                    printf("spred: %.4f %.4f %.4f %.4f ", spred.x, spred.y, spred.w, spred.h);
                    printf("%.4f ", l.output[obj_index]);
                    for(k=0; k < l.classes; ++k) {
                        printf("%.4f ", l.output[class_index+k*stride]);
                    }
                    printf("\n");

                    box tpred = get_distill_yolo_box(lt.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, snet.w, snet.h,
                                              l.w * l.h);
                    printf("tpred: %.4f %.4f %.4f %.4f ", tpred.x, tpred.y, tpred.w, tpred.h);
                    printf("%.4f ", lt.output[obj_index]);
                    for(k=0; k < l.classes; ++k) {
                        printf("%.4f ", lt.output[class_index+k*stride]);
                    }
                    printf("\n");

                    printf("delta: %.4f %.4f %.4f %.4f ", l.delta[box_index+0*stride], l.delta[box_index+1*stride], l.delta[box_index+2*stride], l.delta[box_index+3*stride]);
                    printf("%.4f ", l.delta[obj_index]);
                    for(k=0; k < l.classes; ++k) {
                        printf("%.4f ", l.delta[class_index+k*stride]);
                    }
                    printf("\n");

                }
            }
        }
    }
    */

    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d,  loss: %f\n", snet.index,
           avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w * l.h * l.n * l.batch),
           recall / count, recall75 / count, count, *(l.cost));
}

void backward_distill_yolo_layer(const layer l, network net) {
    axpy_cpu(l.batch * l.inputs, 1, l.delta, 1, net.delta, 1);
}

void avg_flipped_distill_yolo(layer l) {
    int i, j, n, z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w / 2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for (z = 0; z < l.classes + 4 + 1; ++z) {
                    int i1 = z * l.w * l.h * l.n + n * l.w * l.h + j * l.w + i;
                    int i2 = z * l.w * l.h * l.n + n * l.w * l.h + j * l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if (z == 0) {
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for (i = 0; i < l.outputs; ++i) {
        l.output[i] = (l.output[i] + flip[i]) / 2.;
    }
}

#ifdef GPU

void forward_distill_yolo_layer_gpu(const layer l, network snet, network tnet)
{
    copy_gpu(l.batch*l.inputs, snet.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC, 0);  // x, y activate
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC, 0);
        }
    }
    if(!snet.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, snet.input, l.batch*l.inputs);
    forward_distill_yolo_layer(l, snet, tnet);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_distill_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

