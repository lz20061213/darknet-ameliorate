#include "mutual_yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_mutual_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int mutual_index) {
    int i;
    layer l = {0};
    l.type = MUTUAL_YOLO;

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

    l.mutual_forward = forward_mutual_yolo_layer;
    l.backward = backward_mutual_yolo_layer;
#ifdef GPU
    l.mutual_forward_gpu = forward_mutual_yolo_layer_gpu;
    l.backward_gpu = backward_mutual_yolo_layer_gpu;
    //l.mutual_output_gpu = cuda_make_array(l.mutual_output, batch*l.outputs);
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "mutual yolo\n");
    srand(0);

    return l;
}

void resize_mutual_yolo_layer(layer *l, int w, int h) {
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

box get_mutual_yolo_box(float *x, float *biases, int n, int index, int use_center_regression, int i, int j, int lw, int lh, int w, int h, int stride) {
    box b;
    if (use_center_regression) {
        b.x = (i + 0.5 + x[index + 0*stride]) / lw;
        b.y = (j + 0.5 + x[index + 1*stride]) / lh;
    } else {
        b.x = (i + x[index + 0*stride]) / lw;
        b.y = (j + x[index + 1*stride]) / lh;
    }
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

ious delta_mutual_yolo_box(box truth, float *x, float *mutual_x, float *biases, int n, int index, int use_center_regression, int i, int j, int lw,
                     int lh, int w, int h, float *delta, float scale, int stride, float margin, float iou_normalizer, IOU_LOSS iou_loss) {
    ious all_sious = {0};
    ious all_pious = {0};

    box spred = get_mutual_yolo_box(x, biases, n, index, use_center_regression, i, j, lw, lh, w, h, stride);
    box ppred = get_mutual_yolo_box(mutual_x, biases, n, index, use_center_regression, i, j, lw, lh, w, h, stride);

    all_sious.iou = box_iou(spred, truth);
    all_sious.giou = box_giou(spred, truth);
    all_sious.diou = box_diou(spred, truth);
    all_sious.ciou = box_ciou(spred, truth);

    all_pious.iou = box_iou(ppred, truth);
    all_pious.giou = box_giou(ppred, truth);
    all_pious.diou = box_diou(ppred, truth);
    all_pious.ciou = box_ciou(ppred, truth);
    // avoid nan in dx_box_iou
    if (spred.w == 0) { spred.w = 1.0; }
    if (spred.h == 0) { spred.h = 1.0; }

    if (iou_loss == MSE) {
        float tx = 0;
        float ty = 0;
        if (use_center_regression) {
            tx = (truth.x * lw - i - 0.5);
            ty = (truth.y * lh - j - 0.5);
        } else {
            tx = (truth.x * lw - i);
            ty = (truth.h * lh - j);
        }
        float tw = log(truth.w * w / biases[2 * n]);
        float th = log(truth.h * h / biases[2 * n + 1]);

        if (all_sious.iou + 2 * margin < all_pious.iou)  scale *= 1.5;

        delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]) * iou_normalizer;
        delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]) * iou_normalizer;
        delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]) * iou_normalizer;
        delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]) * iou_normalizer;
    } else {
        if (iou_loss == GIOU) {
            if (all_sious.giou + 2 * margin < all_pious.giou) scale *= 1.5;
        } else if (iou_loss == DIOU) {
            if (all_sious.diou + 2 * margin < all_pious.diou) scale *= 1.5;
        } else if (iou_loss == CIOU) {
            if (all_sious.ciou + 2 * margin < all_pious.ciou) scale *= 1.5;
        }
        // https://github.com/generalized-iou/g-darknet
        // https://arxiv.org/abs/1902.09630v2
        // https://giou.stanford.edu/
        all_sious.dx_iou = dx_box_iou(spred, truth, iou_loss);
        // jacobian^t (transpose)
        float dx = all_sious.dx_iou.dt;
        float dy = all_sious.dx_iou.db;
        float dw = all_sious.dx_iou.dl;
        float dh = all_sious.dx_iou.dr;
         // predict exponential, apply gradient of e^delta_t ONLY for w,h
        dw *= exp(x[index + 2 * stride]);
        dh *= exp(x[index + 3 * stride]);

        // normalize iou weight
        dx *= iou_normalizer;
        dy *= iou_normalizer;
        dw *= iou_normalizer;
        dh *= iou_normalizer;

        // delta
        delta[index + 0 * stride] = scale * dx;
        delta[index + 1 * stride] = scale * dy;
        delta[index + 2 * stride] = scale * dw;
        delta[index + 3 * stride] = scale * dh;
    }
    return all_sious;
}

void delta_mutual_yolo_class(float *output, float *mutual_output, float *delta, int index, int class, int classes,
                              int stride, float alpha, float *avg_cat, float label_smooth_rate) {
    int n;
    float hard, soft;
    //printf("distill_yolo_layer in 125, output: %f, mimic_truth: %f\n", output[index + stride * class], mimic_truth[index + stride * class]);
    // loss netp Lp + KL(q||p)
    if (delta[index]) {
        float label = 1;
        if (label_smooth_rate) label = label * (1 - label_smooth_rate) + 0.5 * label_smooth_rate;
        hard = label - output[index + stride * class];
        soft = mutual_output[index + stride * class] - output[index + stride * class];  // KL divergence
        delta[index + stride * class] = hard + soft;
        //delta[index + stride * class] = hard;
        if (avg_cat) *avg_cat += output[index + stride * class];
        return;
    }
    for (n = 0; n < classes; ++n) {
        float label = ((n == class)?1 : 0);
        if (label_smooth_rate) label = label * (1 - label_smooth_rate) + 0.5 * label_smooth_rate;
        hard = label - output[index + stride * n];
        soft = mutual_output[index + stride * n] - output[index + stride * n];
        delta[index + stride * n] = hard + soft;
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

void forward_mutual_yolo_layer(const layer ls, network snet, network pnet) {
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
            if (ls.use_center_regression) {
                activate_array(ls.output + index, 2 * ls.w * ls.h, HALFTANH);
            } else {
                activate_array(ls.output + index, 2 * ls.w * ls.h, LOGISTIC);  // x, y activate
            }
            if (ls.scale_xy != 1.0) {
                scal_add_cpu(2*ls.w*ls.h, ls.scale_xy, -0.5*(ls.scale_xy - 1), ls.output + index, 1); // scale x,y
            }
            index = entry_index(ls, b, n*ls.w*ls.h, 4);  // class active
            activate_array(ls.output + index, (1 + ls.classes) * ls.w * ls.h, LOGISTIC);
        }
    }
    // activate peer network
    for (b = 0; b < lp.batch; ++b) {
        for (n = 0; n < lp.n; ++n) {
            int index = entry_index(lp, b, n * lp.w * lp.h, 0);   // bbox/poly active, linear
            if (lp.use_center_regression) {
                activate_array(lp.output + index, 2 * lp.w * lp.h, HALFTANH);
            } else {
                activate_array(lp.output + index, 2 * lp.w * lp.h, LOGISTIC);  // x, y activate
            }
            if (lp.scale_xy != 1.0) {
                scal_add_cpu(2*lp.w*lp.h, lp.scale_xy, -0.5*(lp.scale_xy - 1), lp.output + index, 1); // scale x,y
            }
            index = entry_index(lp, b, n*lp.w*lp.h, 4);  // class active
            activate_array(lp.output + index, (1 + lp.classes) * lp.w * lp.h, LOGISTIC);
        }
    }

#endif

    // printf("yolo 106: %d, %d, %d, %f\n", l.w, l.h, l.n, l.output[0]);
    memset(ls.delta, 0, ls.outputs * ls.batch * sizeof(float));
    memset(lp.delta, 0, lp.outputs * lp.batch * sizeof(float));
    if (!snet.train || !pnet.train) return;
    float stot_iou = 0, ptot_iou = 0;
    float stot_giou = 0, ptot_giou = 0;
    float stot_diou = 0, ptot_diou = 0;
    float stot_ciou = 0, ptot_ciou = 0;
    float stot_iou_loss = 0, ptot_iou_loss = 0;
    float stot_giou_loss = 0, ptot_giou_loss = 0;
    float stot_diou_loss = 0, ptot_diou_loss = 0;
    float stot_ciou_loss = 0, ptot_ciou_loss = 0;
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
                    box spred = get_mutual_yolo_box(ls.output, ls.biases, ls.mask[n], box_index, ls.use_center_regression, i, j, ls.w, ls.h, snet.w, snet.h,
                                              ls.w * ls.h);
                    //printf("biases in 214: %f %f\n", ls.biases[0], lp.biases[0]);
                    //printf("mask in 215: %d %d\n", ls.mask[n], lp.mask[n]);
                    //printf("h, w in 216: %d %d %d %d\n", ls.w, ls.h, lp.w, lp.h);
                    //printf("net h, w in 217: %d %d %d %d\n", snet.h, snet.w, pnet.h, pnet.w);
                    box ppred = get_mutual_yolo_box(lp.output, lp.biases, lp.mask[n], box_index, lp.use_center_regression, i, j, lp.w, lp.h, pnet.w, pnet.h,
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
                    float hard, soft;
                    // student network
                    hard = 0 - ls.output[obj_index];
                    soft = lp.output[obj_index] - ls.output[obj_index];
                    //ls.delta[obj_index] = ls.alpha * hard + (1 - ls.alpha) * soft;
                    ls.delta[obj_index] = hard + soft;
                    //ls.delta[obj_index] = hard;
                    // peer network
                    hard = 0 - lp.output[obj_index];
                    soft = ls.output[obj_index] - lp.output[obj_index];
                    //lp.delta[obj_index] = lp.alpha * hard + (1 - lp.alpha) * soft;
                    lp.delta[obj_index] = hard + soft;
                    //lp.delta[obj_index] = hard;
                    if (sbest_iou > ls.ignore_thresh) {
                        ls.delta[obj_index] = 0;
                    }
                    if (pbest_iou > lp.ignore_thresh) {
                        lp.delta[obj_index] = 0;
                    }
                    if (sbest_iou > ls.truth_thresh) {
                        hard = 1 - ls.output[obj_index];
                        soft = lp.output[obj_index] - ls.output[obj_index];
                        //ls.delta[obj_index] = ls.alpha * hard + (1 - ls.alpha) * soft;
                        ls.delta[obj_index] = hard + soft;
                        //ls.delta[obj_index] = hard;

                        int sclass = snet.truth[sbest_t * (4 + 1) + b * ls.truths + 4];
                        if (ls.map) sclass = ls.map[sclass];
                        int class_index = entry_index(ls, b, n * ls.w * ls.h + j * ls.w + i, 4 + 1);
                        delta_mutual_yolo_class(ls.output, lp.output, ls.delta, class_index, sclass, ls.classes,
                                                 ls.w * ls.h, ls.alpha, 0, ls.label_smooth_rate);

                        box struth = float_to_box(snet.truth + sbest_t * (4 + 1) + b * ls.truths, 1);
                        delta_mutual_yolo_box(struth, ls.output, lp.output, ls.biases, ls.mask[n], box_index, ls.use_center_regression, i, j,
                                               ls.w, ls.h, snet.w, snet.h, ls.delta, (2 - struth.w * struth.h),
                                               ls.w * ls.h, ls.margin, ls.iou_normalizer, ls.iou_loss);
                    }
                    if (pbest_iou > lp.truth_thresh) {
                        hard = 1 - lp.output[obj_index];
                        soft = ls.output[obj_index] - lp.output[obj_index];
                        //lp.delta[obj_index] = lp.alpha * hard + (1 - lp.alpha) * soft;
                        lp.delta[obj_index] = hard + soft;
                        //lp.delta[obj_index] = hard;

                        int pclass = pnet.truth[pbest_t * (4 + 1) + b * lp.truths + 4];
                        if (lp.map) pclass = lp.map[pclass];
                        int class_index = entry_index(lp, b, n * lp.w * lp.h + j * lp.w + i, 4 + 1);
                        delta_mutual_yolo_class(lp.output, ls.output, lp.delta, class_index, pclass, lp.classes,
                                                 lp.w * lp.h, lp.alpha, 0, lp.label_smooth_rate);

                        box ptruth = float_to_box(pnet.truth + pbest_t * (4 + 1) + b * lp.truths, 1);
                        delta_mutual_yolo_box(ptruth, lp.output, ls.output, lp.biases, lp.mask[n], box_index, lp.use_center_regression, i, j,
                                               lp.w, lp.h, pnet.w, pnet.h, lp.delta, (2 - ptruth.w * ptruth.h),
                                               lp.w * lp.h, lp.margin, lp.iou_normalizer, lp.iou_loss);
                    }
                }
            }
        }
        //printf("distill_yolo_layer in 227\n");
        for (t = 0; t < ls.max_boxes; ++t) {
            box truth = float_to_box(snet.truth + t * (4 + 1) + b * ls.truths, 1);
            box truthp = float_to_box(pnet.truth + t * (4 + 1) + b * lp.truths, 1);
            //printf("mutual_yolo_layer in 230: %f %f %f %f\n", truth.x, truth.y, truth.w, truth.h);
            //printf("mutual_yolo_layer in 231: %f %f %f %f\n", truthp.x, truthp.y, truthp.w, truthp.h);

            if (!truth.x) break;

            float best_iou = 0;
            int best_n = 0;

            i = (truth.x * ls.w);
            j = (truth.y * ls.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;

            // for atss, get mean and std of all iou(anchor, truth)
            float *per_ious = calloc(ls.total, sizeof(float));
            float mean_iou = 0;
            float std_iou = 0;
            float target_iou = 0;
            for(n = 0; n < ls.total; ++n){
                box pred = {0};
                pred.w = ls.biases[2*n] / snet.w;
                pred.h = ls.biases[2*n+1] / snet.h;
                float iou = box_iou(pred, truth_shift);
                mean_iou += iou;
                per_ious[n] = iou;
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
                if (n==ls.total - 1) mean_iou /= ls.total;
            }

            for(n = 0; n < ls.total; ++n) {
                std_iou += pow(per_ious[n] - mean_iou, 2);
            }
            std_iou = sqrt(std_iou/ls.total);

            target_iou = mean_iou + std_iou;

            for (n = 0; n < ls.total; ++n) {
                int is_positive = 0;
                if (ls.atss && per_ious[n] >= target_iou) is_positive = 1;
                if (n == best_n) is_positive = 1;

                int mask_n = int_index(ls.mask, n, ls.n);
                if (mask_n >= 0 && is_positive) {
                    int box_index = entry_index(ls, b, mask_n * ls.w * ls.h + j * ls.w + i, 0);
                    ious all_sious = delta_mutual_yolo_box(truth, ls.output, lp.output, ls.biases, best_n, box_index, ls.use_center_regression, i, j,
                                            ls.w, ls.h, snet.w, snet.h, ls.delta, (2 - truth.w * truth.h), ls.w * ls.h, ls.margin, ls.iou_normalizer, ls.iou_loss);
                    ious all_pious = delta_mutual_yolo_box(truth, lp.output, ls.output, lp.biases, best_n, box_index, lp.use_center_regression, i, j,
                                            lp.w, lp.h, pnet.w, pnet.h, lp.delta, (2 - truth.w * truth.h), lp.w * lp.h, lp.margin, lp.iou_normalizer, lp.iou_loss);

                    // range is 0 <= 1
                    stot_iou += all_sious.iou;
                    stot_iou_loss += 1 - all_sious.iou;
                    ptot_iou += all_pious.iou;
                    ptot_iou_loss += 1 - all_pious.iou;
                    // range is -1 <= giou <= 1
                    stot_giou += all_sious.giou;
                    stot_giou_loss += 1 - all_sious.giou;
                    ptot_giou += all_pious.giou;
                    ptot_giou_loss += 1 - all_pious.giou;

                    stot_diou += all_sious.diou;
                    stot_diou_loss += 1 - all_sious.diou;
                    ptot_diou += all_pious.diou;
                    ptot_diou_loss += 1 - all_pious.diou;

                    stot_ciou += all_sious.ciou;
                    stot_ciou_loss += 1 - all_sious.ciou;
                    ptot_ciou += all_pious.ciou;
                    ptot_ciou_loss += 1 - all_pious.ciou;

                    int obj_index = entry_index(ls, b, mask_n * ls.w * ls.h + j * ls.w + i, 4);
                    savg_obj += ls.output[obj_index];
                    pavg_obj += lp.output[obj_index];

                    // for objectness
                    float hard, soft;
                    // student network
                    hard = 1 - ls.output[obj_index];
                    soft = lp.output[obj_index] - ls.output[obj_index];
                    //ls.delta[obj_index] = ls.alpha * hard + (1 - ls.alpha) * soft;
                    ls.delta[obj_index] = hard + soft;
                    //ls.delta[obj_index] = hard;

                    // peer network
                    hard = 1 - lp.output[obj_index];
                    soft = ls.output[obj_index] - lp.output[obj_index];
                    //lp.delta[obj_index] = lp.alpha * hard + (1 - lp.alpha) * soft;
                    lp.delta[obj_index] = hard + soft;
                    //lp.delta[obj_index] = hard;

                    int class = snet.truth[t * (4 + 1) + b * ls.truths + 4];
                    //printf("truth: %d %f %f %f %f\n", class, truth.x, truth.y, truth.w, truth.h);
                    if (ls.map) class = ls.map[class];
                    int class_index = entry_index(ls, b, mask_n * ls.w * ls.h + j * ls.w + i, 4 + 1);
                    delta_mutual_yolo_class(ls.output, lp.output, ls.delta, class_index, class, ls.classes, ls.w * ls.h,
                                             ls.alpha, &savg_cat, ls.label_smooth_rate);
                    delta_mutual_yolo_class(lp.output, ls.output, lp.delta, class_index, class, lp.classes, lp.w * lp.h,
                                             lp.alpha, &pavg_cat, lp.label_smooth_rate);

                    ++count;
                    ++class_count;
                    if (all_sious.iou > .5) srecall += 1;
                    if (all_pious.iou > .5) precall += 1;
                    if (all_sious.iou > .75) srecall75 += 1;
                    if (all_pious.iou > .75) precall75 += 1;
                }
            }
        }
    }

    if (count == 0) count = 1;
    if (class_count == 0) class_count = 1;

    // Always compute classification loss both for iou + cls loss and for logging with mse loss
    // TODO: remove IOU loss fields before computing MSE on class
    //   probably split into two arrays
    int stride = ls.w*ls.h;
    float* no_iou_loss_delta = (float *)calloc(ls.batch * ls.outputs, sizeof(float));
    memcpy(no_iou_loss_delta, ls.delta, ls.batch * ls.outputs * sizeof(float));
    for (b = 0; b < ls.batch; ++b) {
        for (j = 0; j < ls.h; ++j) {
            for (i = 0; i < ls.w; ++i) {
                for (n = 0; n < ls.n; ++n) {
                    int index = entry_index(ls, b, n*ls.w*ls.h + j*ls.w + i, 0);
                    no_iou_loss_delta[index + 0 * stride] = 0;
                    no_iou_loss_delta[index + 1 * stride] = 0;
                    no_iou_loss_delta[index + 2 * stride] = 0;
                    no_iou_loss_delta[index + 3 * stride] = 0;
                }
            }
        }
    }
    float sclassification_loss = ls.cls_normalizer * pow(mag_array(no_iou_loss_delta, ls.outputs * ls.batch), 2);
    memcpy(no_iou_loss_delta, lp.delta, lp.batch * lp.outputs * sizeof(float));
    for (b = 0; b < lp.batch; ++b) {
        for (j = 0; j < lp.h; ++j) {
            for (i = 0; i < lp.w; ++i) {
                for (n = 0; n < lp.n; ++n) {
                    int index = entry_index(lp, b, n*lp.w*lp.h + j*lp.w + i, 0);
                    no_iou_loss_delta[index + 0 * stride] = 0;
                    no_iou_loss_delta[index + 1 * stride] = 0;
                    no_iou_loss_delta[index + 2 * stride] = 0;
                    no_iou_loss_delta[index + 3 * stride] = 0;
                }
            }
        }
    }
    float pclassification_loss = lp.cls_normalizer * pow(mag_array(no_iou_loss_delta, lp.outputs * lp.batch), 2);
    free(no_iou_loss_delta);

    float savg_iou_loss = 0, pavg_iou_loss = 0;
    // gIOU loss + MSE (objectness) loss
    if (ls.iou_loss == MSE) {
        *(ls.cost) = pow(mag_array(ls.delta, ls.outputs * ls.batch), 2);
        savg_iou_loss = count > 0 ? (*(ls.cost) - sclassification_loss) / count : 0;
    } else {
        if (ls.iou_loss == IOU) {
            savg_iou_loss = count > 0 ? ls.iou_normalizer * (stot_iou_loss / count) : 0;
        }
        else if (ls.iou_loss == GIOU) {
            savg_iou_loss = count > 0 ? ls.iou_normalizer * (stot_giou_loss / count) : 0;
        }
        else if (ls.iou_loss == DIOU) {
            savg_iou_loss = count > 0 ? ls.iou_normalizer * (stot_diou_loss / count) : 0;
        }
        else if (ls.iou_loss == CIOU) {
            savg_iou_loss = count > 0 ? ls.iou_normalizer * (stot_ciou_loss / count) : 0;
        }
        *(ls.cost) = savg_iou_loss + sclassification_loss;
    }

    if (lp.iou_loss == MSE) {
        *(lp.cost) = pow(mag_array(lp.delta, lp.outputs * lp.batch), 2);
        pavg_iou_loss = count > 0 ? (*(lp.cost) - pclassification_loss) / count : 0;
    } else {
        if (lp.iou_loss == IOU) {
            pavg_iou_loss = count > 0 ? lp.iou_normalizer * (ptot_iou_loss / count) : 0;
        }
        else if (lp.iou_loss == GIOU) {
            pavg_iou_loss = count > 0 ? lp.iou_normalizer * (ptot_giou_loss / count) : 0;
        }
        else if (lp.iou_loss == DIOU) {
            pavg_iou_loss = count > 0 ? lp.iou_normalizer * (ptot_diou_loss / count) : 0;
        }
        else if (lp.iou_loss == CIOU) {
            pavg_iou_loss = count > 0 ? lp.iou_normalizer * (ptot_ciou_loss / count) : 0;
        }
        *(lp.cost) = pavg_iou_loss + pclassification_loss;
    }

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
    fprintf(stderr, "(snet->%s loss, normalizer: (iou: %.2f, cls: %.2f) Region %d Avg (iou: %f, %s: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d, class_loss = %f, avg_iou_loss = %f, total_loss = %f \n",
        ((ls.iou_loss == MSE) ? "mse" : (ls.iou_loss == IOU ? "iou" : (ls.iou_loss == GIOU ? "giou" : (ls.iou_loss == DIOU ? "diou" : "ciou")))),
        ls.iou_normalizer, ls.cls_normalizer, snet.index,
        stot_iou / count,
        ((ls.iou_loss == MSE) ? "mse" : (ls.iou_loss == IOU ? "iou" : (ls.iou_loss == GIOU ? "giou" : (ls.iou_loss == DIOU ? "diou" : "ciou")))),
        (ls.iou_loss == MSE ?  stot_iou / count : (ls.iou_loss == IOU ? stot_iou / count : (ls.iou_loss == GIOU ? stot_giou / count : (ls.iou_loss == DIOU ? stot_diou / count : stot_ciou / count)))),
        savg_cat / class_count, savg_obj / count, savg_anyobj / (ls.w*ls.h*ls.n*ls.batch), srecall / count, srecall75 / count, count,
        sclassification_loss, savg_iou_loss, *(ls.cost));
    fprintf(stderr, "(pnet->%s loss, normalizer: (iou: %.2f, cls: %.2f) Region %d Avg (iou: %f, %s: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d, class_loss = %f, avg_iou_loss = %f, total_loss = %f \n",
        ((lp.iou_loss == MSE) ? "mse" : (lp.iou_loss == IOU ? "iou" : (lp.iou_loss == GIOU ? "giou" : (lp.iou_loss == DIOU ? "diou" : "ciou")))),
        lp.iou_normalizer, lp.cls_normalizer, pnet.index,
        ptot_iou / count,
        ((lp.iou_loss == MSE) ? "mse" : (lp.iou_loss == IOU ? "iou" : (lp.iou_loss == GIOU ? "giou" : (lp.iou_loss == DIOU ? "diou" : "ciou")))),
        (lp.iou_loss == MSE ?  ptot_iou / count : (lp.iou_loss == IOU ? ptot_iou / count : (lp.iou_loss == GIOU ? ptot_giou / count : (lp.iou_loss == DIOU ? ptot_diou / count : ptot_ciou / count)))),
        pavg_cat / class_count, pavg_obj / count, pavg_anyobj / (lp.w*lp.h*lp.n*lp.batch), precall / count, precall75 / count, count,
        pclassification_loss, pavg_iou_loss, *(lp.cost));
}

void backward_mutual_yolo_layer(const layer l, network net) {
    axpy_cpu(l.batch * l.inputs, 1, l.delta, 1, net.delta, 1);
}

void avg_flipped_mutual_yolo(layer l) {
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

void forward_mutual_yolo_layer_gpu(const layer ls, network snet, network pnet)
{
    // activate student network
    cuda_set_device(snet.gpu_index);
    copy_gpu(ls.batch*ls.inputs, snet.layers[ls.current_layer_index-1].output_gpu, 1, ls.output_gpu, 1);
    int b, n;
    for (b = 0; b < ls.batch; ++b){
        for(n = 0; n < ls.n; ++n){
            int index = entry_index(ls, b, n*ls.w*ls.h, 0);
            if (ls.use_center_regression) {
                activate_array_gpu(ls.output_gpu + index, 2*ls.w*ls.h, HALFTANH, 0);
            } else {
                activate_array_gpu(ls.output_gpu + index, 2*ls.w*ls.h, LOGISTIC, 0);
            }
            if (ls.scale_xy != 1.0) {
                scal_add_gpu(2*ls.w*ls.h, ls.scale_xy, -0.5*(ls.scale_xy - 1), ls.output_gpu + index, 1); // scale x,y
            }
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
            if (lp.use_center_regression) {
                activate_array_gpu(lp.output_gpu + index, 2*lp.w*lp.h, HALFTANH, 0);
            } else {
                activate_array_gpu(lp.output_gpu + index, 2*lp.w*lp.h, LOGISTIC, 0);
            }
            if (lp.scale_xy != 1.0) {
                scal_add_gpu(2*lp.w*lp.h, lp.scale_xy, -0.5*(lp.scale_xy - 1), lp.output_gpu + index, 1); // scale x,y
            }
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

    forward_mutual_yolo_layer(ls, snet, pnet);

    cuda_set_device(snet.gpu_index);
    cuda_push_array(ls.delta_gpu, ls.delta, ls.batch*ls.outputs);
    cuda_set_device(pnet.gpu_index);
    cuda_push_array(lp.delta_gpu, lp.delta, lp.batch*lp.outputs);
}

void backward_mutual_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

