#include "mimicutual_yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_mimicutual_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int distill_index, int mutual_index) {
    int i;
    layer l = {0};
    l.type = MIMICUTUAL_YOLO;

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
    l.mutual_index = mutual_index;

    l.bias_updates = calloc(n * 2, sizeof(float));
    //l.outputs = h*w*n*(classes + 4 + 1);
    l.outputs = h * w * l.c;
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    //l.truths = 90 * (12 + 1);
    l.delta = calloc(batch * l.outputs, sizeof(float));
    l.output = calloc(batch * l.outputs, sizeof(float));
    //printf("l.output: %d\n", batch * l.outputs);
    for (i = 0; i < total * 2; ++i) {
        l.biases[i] = .5;
    }

    l.mimicutual_forward = forward_mimicutual_yolo_layer;
    l.backward = backward_mimicutual_yolo_layer;
#ifdef GPU
    l.mimicutual_forward_gpu = forward_mimicutual_yolo_layer_gpu;
    l.backward_gpu = backward_mimicutual_yolo_layer_gpu;
    //l.mutual_output_gpu = cuda_make_array(l.mutual_output, batch*l.outputs);
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "mimicutual yolo\n");
    srand(0);

    return l;
}

void resize_mimicutual_yolo_layer(layer *l, int w, int h) {
    l->w = w;
    l->h = h;
    l->out_w = w;
    l->out_h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
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

box get_mimicutual_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
    box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

float delta_mimicutual_yolo_box(box truth, float *x, float *mutual_x, float *mimic_x, float *biases, int n, int index, int i, int j, int lw,
                     int lh, int w, int h, float *delta, float scale, int stride, float margin) {
    box spred = get_mimicutual_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    box tpred = get_mimicutual_yolo_box(mimic_x, biases, n, index, i, j, lw, lh, w, h, stride);

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

void delta_mimicutual_yolo_class(float *output, float *mutual_output, float *mimic_output, float *delta, int index, int class, int classes,
                              int stride, float alpha, float *avg_cat) {
    int n;
    float hard, mutu, soft;
    //printf("distill_yolo_layer in 125, output: %f, mimic_truth: %f\n", output[index + stride * class], mimic_truth[index + stride * class]);
    // loss netp Lp + KL(q||p)
    if (delta[index]) {
        hard = 1 - output[index + stride * class];
        mutu = mutual_output[index + stride * class] - output[index + stride * class];  // KL divergence
        soft = mimic_output[index + stride * class] - output[index + stride * class];  // Cross entropy
        delta[index + stride * class] = alpha * (hard + mutu) + (1 - alpha) * soft;
        //delta[index + stride * class] = hard;
        if (avg_cat) *avg_cat += output[index + stride * class];
        return;
    }
    for (n = 0; n < classes; ++n) {
        hard = ((n == class)?1 : 0) - output[index + stride * n];
        mutu = mutual_output[index + stride * n] - output[index + stride * n];
        soft = mimic_output[index + stride * n] - output[index + stride * n];
        delta[index + stride * n] = alpha * (hard + mutu) + (1 - alpha) * soft;
        //delta[index + stride * n] = hard;
        if (n == class && avg_cat) *avg_cat += output[index + stride * n];
    }
}

static int entry_index(layer l, int batch, int location, int entry) {
    int n = location / (l.w * l.h);
    int loc = location % (l.w * l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
    //return batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1) + entry * l.w * l.h + loc;
}

void forward_mimicutual_yolo_layer(const layer ls, network snet, network pnet, network tnet) {
    int i, j, b, t, n;

     // get layer lp and copy input
    int p_index = snet.mutual_layers[ls.mutual_index];
    layer lp = pnet.layers[p_index];

#ifdef GPU
    cuda_set_device(snet.gpu_index);
    cuda_pull_array(ls.output_gpu, ls.output, ls.batch*ls.outputs);
    cuda_set_device(pnet.gpu_index);
    cuda_pull_array(lp.output_gpu, lp.output, lp.batch*lp.outputs);
    assert(ls.outputs == lp.outputs);
#endif

#ifndef GPU
    // get input for yolo layer from previous layer output
    memcpy(ls.output, snet.layers[ls.current_layer_index-1].output, ls.outputs * ls.batch * sizeof(float));
    memcpy(lp.output, pnet.layers[p_index-1].output, lp.outputs * lp.batch * sizeof(float));
    assert(ls.outputs == lp.outputs);

    // activate student network
    for (b = 0; b < ls.batch; ++b) {
        for (n = 0; n < ls.n; ++n) {
            int index = entry_index(ls, b, n * ls.w * ls.h, 0);   // bbox/poly active, linear
            activate_array(ls.output + index, 2 * ls.w * ls.h, LOGISTIC);  // x, y activate
            index = entry_index(ls, b, n*ls.w*ls.h, 4);  // class active
            activate_array(ls.output + index, (1 + ls.classes) * ls.w * ls.h, LOGISTIC);
        }
    }
    // activate peer network
    for (b = 0; b < lp.batch; ++b) {
        for (n = 0; n < lp.n; ++n) {
            int index = entry_index(lp, b, n * lp.w * lp.h, 0);   // bbox/poly active, linear
            activate_array(lp.output + index, 2 * lp.w * lp.h, LOGISTIC);  // x, y activate
            index = entry_index(lp, b, n*lp.w*lp.h, 4);  // class active
            activate_array(lp.output + index, (1 + lp.classes) * lp.w * lp.h, LOGISTIC);
        }
    }

#endif

    // get mimic_truth
    int t_index = snet.distill_layers[ls.distill_index];
    layer lt = tnet.layers[t_index];
    assert(ls.outputs == lt.outputs);

    // printf("yolo 106: %d, %d, %d, %f\n", l.w, l.h, l.n, l.output[0]);
    memset(ls.delta, 0, ls.outputs * ls.batch * sizeof(float));
    memset(lp.delta, 0, lp.outputs * lp.batch * sizeof(float));
    if (!snet.train || !pnet.train) return;
    float savg_iou = 0, pavg_iou = 0;
    float srecall = 0, precall = 0;
    float srecall75 = 0, precall75 = 0;
    float savg_cat = 0, pavg_cat = 0;
    float savg_obj = 0, pavg_obj = 0;
    float savg_anyobj = 0, pavg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(ls.cost) = 0; *(lp.cost) = 0;
    // ls dimensions is same as lp
    for (b = 0; b < ls.batch; ++b) {
        for (j = 0; j < ls.h; ++j) {
            for (i = 0; i < ls.w; ++i) {
                for (n = 0; n < ls.n; ++n) {
                    int box_index = entry_index(ls, b, n * ls.w * ls.h + j * ls.w + i, 0);
                    box spred = get_mimicutual_yolo_box(ls.output, ls.biases, ls.mask[n], box_index, i, j, ls.w, ls.h, snet.w, snet.h,
                                              ls.w * ls.h);
                    //printf("biases in 214: %f %f\n", ls.biases[0], lp.biases[0]);
                    //printf("mask in 215: %d %d\n", ls.mask[n], lp.mask[n]);
                    //printf("h, w in 216: %d %d %d %d\n", ls.w, ls.h, lp.w, lp.h);
                    //printf("net h, w in 217: %d %d %d %d\n", snet.h, snet.w, pnet.h, pnet.w);
                    box ppred = get_mimicutual_yolo_box(lp.output, lp.biases, lp.mask[n], box_index, i, j, lp.w, lp.h, pnet.w, pnet.h,
                                              lp.w * lp.h);
                    float sbest_iou = 0, pbest_iou = 0;
                    int sbest_t = 0, pbest_t = 0;
                    for (t = 0; t < ls.max_boxes; ++t) {
                        box truth = float_to_box(snet.truth + t * (4 + 1) + b * ls.truths, 1);
                        if (!truth.x) break;
                        float siou = box_iou(spred, truth);
                        float piou = box_iou(ppred, truth);
                        if (siou > sbest_iou) {
                            sbest_iou = siou;
                            sbest_t = t;
                        }
                        if (piou > pbest_iou) {
                            pbest_iou = piou;
                            pbest_t = t;
                        }
                    }
                    //printf("sbset_t: %d, pbset_t: %d\n", sbest_t, nbest_t);
                    int obj_index = entry_index(ls, b, n * ls.w * ls.h + j * ls.w + i, 4);
                    savg_anyobj += ls.output[obj_index];
                    pavg_anyobj += lp.output[obj_index];
                    // for objectness
                    float hard, mutu, soft;
                    // student network
                    hard = 0 - ls.output[obj_index];
                    mutu = lp.output[obj_index] - ls.output[obj_index];
                    soft = lt.output[obj_index] - ls.output[obj_index];
                    ls.delta[obj_index] = ls.alpha * (hard + mutu) + (1 - ls.alpha) * soft;
                    //ls.delta[obj_index] = hard;
                    // peer network
                    hard = 0 - lp.output[obj_index];
                    mutu = ls.output[obj_index] - lp.output[obj_index];
                    soft = lt.output[obj_index] - lp.output[obj_index];
                    lp.delta[obj_index] = lp.alpha * (hard + mutu) + (1 - lp.alpha) * soft;
                    //lp.delta[obj_index] = hard;
                    if (sbest_iou > ls.ignore_thresh) {
                        ls.delta[obj_index] = 0;
                    }
                    if (pbest_iou > lp.ignore_thresh) {
                        lp.delta[obj_index] = 0;
                    }
                    if (sbest_iou > ls.truth_thresh) {
                        hard = 1 - ls.output[obj_index];
                        mutu = lp.output[obj_index] - ls.output[obj_index];
                        soft = lt.output[obj_index] - ls.output[obj_index];
                        ls.delta[obj_index] = ls.alpha * (hard + mutu) + (1 - ls.alpha) * soft;
                        //ls.delta[obj_index] = hard;

                        int sclass = snet.truth[sbest_t * (4 + 1) + b * ls.truths + 4];
                        if (ls.map) sclass = ls.map[sclass];
                        int class_index = entry_index(ls, b, n * ls.w * ls.h + j * ls.w + i, 4 + 1);
                        delta_mimicutual_yolo_class(ls.output, lp.output, lt.output, ls.delta, class_index, sclass, ls.classes,
                                                 ls.w * ls.h, ls.alpha, 0);

                        box struth = float_to_box(snet.truth + sbest_t * (4 + 1) + b * ls.truths, 1);
                        delta_mimicutual_yolo_box(struth, ls.output, lp.output, lt.output, ls.biases, ls.mask[n], box_index, i, j,
                                               ls.w, ls.h, snet.w, snet.h, ls.delta, (2 - struth.w * struth.h),
                                               ls.w * ls.h, ls.margin);
                    }
                    if (pbest_iou > lp.truth_thresh) {
                        hard = 1 - lp.output[obj_index];
                        mutu = ls.output[obj_index] - lp.output[obj_index];
                        soft = lt.output[obj_index] - lp.output[obj_index];
                        lp.delta[obj_index] = lp.alpha * (hard + mutu) + (1 - lp.alpha) * soft;
                        //lp.delta[obj_index] = hard;

                        int pclass = pnet.truth[pbest_t * (4 + 1) + b * lp.truths + 4];
                        if (lp.map) pclass = lp.map[pclass];
                        int class_index = entry_index(lp, b, n * lp.w * lp.h + j * lp.w + i, 4 + 1);
                        delta_mimicutual_yolo_class(lp.output, ls.output, lt.output, lp.delta, class_index, pclass, lp.classes,
                                                 lp.w * lp.h, lp.alpha, 0);

                        box ptruth = float_to_box(pnet.truth + pbest_t * (4 + 1) + b * lp.truths, 1);
                        delta_mimicutual_yolo_box(ptruth, lp.output, ls.output, lt.output, lp.biases, lp.mask[n], box_index, i, j,
                                               lp.w, lp.h, pnet.w, pnet.h, lp.delta, (2 - ptruth.w * ptruth.h),
                                               lp.w * lp.h, lp.margin);
                    }
                }
            }
        }
        //printf("distill_yolo_layer in 227\n");
        for (t = 0; t < ls.max_boxes; ++t) {
            box truth = float_to_box(snet.truth + t * (4 + 1) + b * ls.truths, 1);
            //box truthp = float_to_box(pnet.truth + t * (4 + 1) + b * lp.truths, 1);
            //printf("mutual_yolo_layer in 230: %f %f %f %f\n", truth.x, truth.y, truth.w, truth.h);
            //printf("mutual_yolo_layer in 231: %f %f %f %f\n", truthp.x, truthp.y, truthp.w, truthp.h);

            if (!truth.x) break;

            float best_iou = 0;
            int best_n = 0;

            i = (truth.x * ls.w);
            j = (truth.y * ls.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for (n = 0; n < ls.total; ++n) {
                box pred = {0};
                pred.w = ls.biases[2 * n] / snet.w;
                pred.h = ls.biases[2 * n + 1] / snet.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(ls.mask, best_n, ls.n);
            if (mask_n >= 0) {
                int box_index = entry_index(ls, b, mask_n * ls.w * ls.h + j * ls.w + i, 0);
                float siou = delta_mimicutual_yolo_box(truth, ls.output, lp.output, lt.output, ls.biases, best_n, box_index, i, j,
                                        ls.w, ls.h, snet.w, snet.h, ls.delta, (2 - truth.w * truth.h), ls.w * ls.h, ls.margin);
                float piou = delta_mimicutual_yolo_box(truth, lp.output, ls.output, lt.output, lp.biases, best_n, box_index, i, j,
                                        lp.w, lp.h, pnet.w, pnet.h, lp.delta, (2 - truth.w * truth.h), lp.w * lp.h, lp.margin);

                int obj_index = entry_index(ls, b, mask_n * ls.w * ls.h + j * ls.w + i, 4);
                savg_obj += ls.output[obj_index];
                pavg_obj += lp.output[obj_index];

                // for objectness
                float hard, mutu, soft;
                // student network
                hard = 1 - ls.output[obj_index];
                mutu = lp.output[obj_index] - ls.output[obj_index];
                soft = lt.output[obj_index] - ls.output[obj_index];
                ls.delta[obj_index] = ls.alpha * (hard + mutu) + (1 - ls.alpha) * soft;
                //ls.delta[obj_index] = hard;

                // peer network
                hard = 1 - lp.output[obj_index];
                mutu = ls.output[obj_index] - lp.output[obj_index];
                soft = lt.output[obj_index] - lp.output[obj_index];
                lp.delta[obj_index] = lp.alpha * (hard + mutu) + (1 - lp.alpha) * soft;
                //lp.delta[obj_index] = hard;

                int class = snet.truth[t * (4 + 1) + b * ls.truths + 4];
                //printf("truth: %d %f %f %f %f\n", class, truth.x, truth.y, truth.w, truth.h);
                if (ls.map) class = ls.map[class];
                int class_index = entry_index(ls, b, mask_n * ls.w * ls.h + j * ls.w + i, 4 + 1);
                delta_mimicutual_yolo_class(ls.output, lp.output, lt.output, ls.delta, class_index, class, ls.classes, ls.w * ls.h,
                                         ls.alpha, &savg_cat);
                delta_mimicutual_yolo_class(lp.output, ls.output, lt.output, lp.delta, class_index, class, lp.classes, lp.w * lp.h,
                                         lp.alpha, &pavg_cat);

                ++count;
                ++class_count;
                if (siou > .5) srecall += 1;
                if (piou > .5) precall += 1;
                if (siou > .75) srecall75 += 1;
                if (piou > .75) precall75 += 1;
                savg_iou += siou;
                pavg_iou += piou;
            }
        }
    }
    *(ls.cost) = pow(mag_array(ls.delta, ls.outputs * ls.batch), 2);
    *(lp.cost) = pow(mag_array(lp.delta, lp.outputs * lp.batch), 2);

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

                        box tpred = get_distill_yolo_box(l.mimic_truth, l.biases, l.mask[n], box_index, i, j, l.w, l.h, snet.w, snet.h,
                                                  l.w * l.h);
                        printf("tpred: %.4f %.4f %.4f %.4f ", tpred.x, tpred.y, tpred.w, tpred.h);
                        printf("%.4f ", l.mimic_truth[obj_index]);
                        for(k=0; k < l.classes; ++k) {
                            printf("%.4f ", l.mimic_truth[class_index+k*stride]);
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

                    box tpred = get_distill_yolo_box(l.mimic_truth, l.biases, l.mask[n], box_index, i, j, l.w, l.h, snet.w, snet.h,
                                              l.w * l.h);
                    printf("tpred: %.4f %.4f %.4f %.4f ", tpred.x, tpred.y, tpred.w, tpred.h);
                    printf("%.4f ", l.mimic_truth[obj_index]);
                    for(k=0; k < l.classes; ++k) {
                        printf("%.4f ", l.mimic_truth[class_index+k*stride]);
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

    printf("snet->Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d,  loss: %f\n",
           ls.current_layer_index, savg_iou / count, savg_cat / class_count, savg_obj / count, savg_anyobj / (ls.w * ls.h * ls.n * ls.batch),
           srecall / count, srecall75 / count, count, *(ls.cost));
    printf("pnet->Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d,  loss: %f\n",
           lp.current_layer_index, pavg_iou / count, pavg_cat / class_count, pavg_obj / count, pavg_anyobj / (lp.w * lp.h * lp.n * lp.batch),
           precall / count, precall75 / count, count, *(lp.cost));
}

void backward_mimicutual_yolo_layer(const layer l, network net) {
    axpy_cpu(l.batch * l.inputs, 1, l.delta, 1, net.delta, 1);
}

void avg_flipped_mimicutual_yolo(layer l) {
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

void forward_mimicutual_yolo_layer_gpu(const layer ls, network snet, network pnet, network tnet)
{
    // activate student network
    cuda_set_device(snet.gpu_index);
    copy_gpu(ls.batch*ls.inputs, snet.layers[ls.current_layer_index-1].output_gpu, 1, ls.output_gpu, 1);
    int b, n;
    for (b = 0; b < ls.batch; ++b){
        for(n = 0; n < ls.n; ++n){
            int index = entry_index(ls, b, n*ls.w*ls.h, 0);
            activate_array_gpu(ls.output_gpu + index, 2*ls.w*ls.h, LOGISTIC, 0);  // x, y activate
            index = entry_index(ls, b, n*ls.w*ls.h, 4);
            activate_array_gpu(ls.output_gpu + index, (1+ls.classes)*ls.w*ls.h, LOGISTIC, 0);
        }
    }
    // activate peer network
    cuda_set_device(pnet.gpu_index);
    int p_index = snet.mutual_layers[ls.mutual_index];
    layer lp = pnet.layers[p_index];
    assert(ls.outputs == lp.outputs);
    copy_gpu(lp.batch*lp.inputs, pnet.layers[p_index-1].output_gpu, 1, lp.output_gpu, 1);
    for (b = 0; b < lp.batch; ++b){
        for(n = 0; n < lp.n; ++n){
            int index = entry_index(lp, b, n*lp.w*lp.h, 0);
            activate_array_gpu(lp.output_gpu + index, 2*lp.w*lp.h, LOGISTIC, 0);  // x, y activate
            index = entry_index(lp, b, n*lp.w*lp.h, 4);
            activate_array_gpu(lp.output_gpu + index, (1+lp.classes)*lp.w*lp.h, LOGISTIC, 0);
        }
    }

    // for test
//    int t;
//    cuda_pull_array(ls.output_gpu, ls.output, ls.batch*ls.outputs);
//    printf("ls->yolo: %d output: ", ls.current_layer_index);
//    for (t = 0; t < 10; ++t) {
//        printf("%f ", ls.output[t]);
//    }
//    printf("\n");
//    cuda_pull_array(lp.output_gpu, lp.output, lp.batch*lp.outputs);
//    printf("lp->yolo: %d output: ", lp.current_layer_index);
//    for (t = 0; t < 10; ++t) {
//        printf("%f ", lp.output[t]);
//    }
//    printf("\n");


    if(!snet.train || ls.onlyforward || !pnet.train || lp.onlyforward){
        cuda_set_device(snet.gpu_index);
        cuda_pull_array(ls.output_gpu, ls.output, ls.batch*ls.outputs);
        cuda_set_device(pnet.gpu_index);
        cuda_pull_array(lp.output_gpu, lp.output, lp.batch*lp.outputs);
        return;
    }

    forward_mimicutual_yolo_layer(ls, snet, pnet, tnet);

    cuda_set_device(snet.gpu_index);
    cuda_push_array(ls.delta_gpu, ls.delta, ls.batch*ls.outputs);
    cuda_set_device(pnet.gpu_index);
    cuda_push_array(lp.delta_gpu, lp.delta, lp.batch*lp.outputs);
}

void backward_mimicutual_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

