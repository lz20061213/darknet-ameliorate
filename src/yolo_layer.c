#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->out_w = w;
    l->out_h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
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

box get_yolo_box(float *x, float *biases, int n, int index, int use_center_regression, int i, int j, int lw, int lh, int w, int h, int stride)
{
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

ious delta_yolo_box(box truth, float *x, float *biases, int n, int index, int use_center_regression, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride,
    float iou_normalizer, IOU_LOSS iou_loss)
{
    ious all_ious = {0};

    box pred = get_yolo_box(x, biases, n, index, use_center_regression, i, j, lw, lh, w, h, stride);

    all_ious.iou = box_iou(pred, truth);
    all_ious.giou = box_giou(pred, truth);
    all_ious.diou = box_diou(pred, truth);
    all_ious.ciou = box_ciou(pred, truth);

    // avoid nan in dx_box_iou
    if (pred.w == 0) { pred.w = 1.0; }
    if (pred.h == 0) { pred.h = 1.0; }

    if (iou_loss == MSE) {
        float tx = 0;
        float ty = 0;
        if (use_center_regression) {
            tx = (truth.x*lw - i - 0.5);
            ty = (truth.y*lh - j - 0.5);
            //printf("tx ty: %f %f\n", tx, ty);
        } else {
            tx = (truth.x*lw - i);
            ty = (truth.y*lh - j);
        }
        float tw = log(truth.w*w / biases[2*n]);
        float th = log(truth.h*h / biases[2*n + 1]);

        delta[index + 0*stride] = scale * (tx - x[index + 0*stride]) * iou_normalizer;
        delta[index + 1*stride] = scale * (ty - x[index + 1*stride]) * iou_normalizer;
        delta[index + 2*stride] = scale * (tw - x[index + 2*stride]) * iou_normalizer;
        delta[index + 3*stride] = scale * (th - x[index + 3*stride]) * iou_normalizer;
    } else {
        // https://github.com/generalized-iou/g-darknet
        // https://arxiv.org/abs/1902.09630v2
        // https://giou.stanford.edu/
        all_ious.dx_iou = dx_box_iou(pred, truth, iou_loss);
        // jacobian^t (transpose)
        float dx = all_ious.dx_iou.dt;
        float dy = all_ious.dx_iou.db;
        float dw = all_ious.dx_iou.dl;
        float dh = all_ious.dx_iou.dr;
         // predict exponential, apply gradient of e^delta_t ONLY for w,h
        dw *= exp(x[index + 2 * stride]);
        dh *= exp(x[index + 3 * stride]);

        // normalize iou weight
        dx *= iou_normalizer;
        dy *= iou_normalizer;
        dw *= iou_normalizer;
        dh *= iou_normalizer;

        // delta
        delta[index + 0 * stride] = dx;
        delta[index + 1 * stride] = dy;
        delta[index + 2 * stride] = dw;
        delta[index + 3 * stride] = dh;
    }

    return all_ious;
}


void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat, float label_smooth_rate)
{
    int n;
    if (delta[index]){
        float label = 1;
        if (label_smooth_rate) label = label * (1 - label_smooth_rate) + 0.5 * label_smooth_rate;
        delta[index + stride*class] = label - output[index + stride*class];
        if(avg_cat) *avg_cat += output[index + stride*class];
        return;
    }
    for(n = 0; n < classes; ++n){
        float label = ((n == class)?1 : 0);
        if (label_smooth_rate) label = label * (1 - label_smooth_rate) + 0.5 * label_smooth_rate;
        delta[index + stride*n] = label - output[index + stride*n];
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            if (l.use_center_regression) {
                activate_array(l.output + index, 2*l.w*l.h, HALFTANH);
            } else {
                activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            }
            if (l.scale_xy != 1.0) {
                scal_add_cpu(2*l.w*l.h, l.scale_xy, -0.5*(l.scale_xy - 1), l.output + index, 1); // scale x,y
            }
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float tot_iou = 0;
    float tot_giou = 0;
    float tot_diou = 0;
    float tot_ciou = 0;
    float tot_iou_loss = 0;
    float tot_giou_loss = 0;
    float tot_diou_loss = 0;
    float tot_ciou_loss = 0;
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
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, l.use_center_regression, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < l.max_boxes; ++t){
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];

                    box truth = float_to_box(net.truth + best_t*(4 + 1)+ b*l.truths, 1);
                    float lb_dis = box_lb_dis(pred, truth);
                    // here we modify objectness to loc confidence
                    l.delta[obj_index] = 0 - l.output[obj_index];

                    if (l.rescore) {
                        l.delta[obj_index] = two_way_max(0, 1 - lb_dis / l.lb_dis_max_thresh) - l.output[obj_index];
                    }

                    // https://blog.csdn.net/linmingan/article/details/77885832
                    // noobj should be different from obj
                    if (l.object_focal_loss) {
                        float alpha = 0.5;
                        // gamma = 2;
                        float pt = l.output[obj_index];
                        float grad =  - 2 * pt * (1- pt) * logf(1 - pt) + pt * pt;
                        l.delta[obj_index] *= alpha * grad;
                    }

                    if (best_iou > l.ignore_thresh) {
                        if (l.rescore) {
                            if (lb_dis < l.lb_dis_ignore_thresh) {
                                l.delta[obj_index] = 0;
                            }
                        } else {
                            l.delta[obj_index] = 0;
                        }
                    }
                    if (best_iou > l.truth_thresh) {
                        if (l.rescore) {
                            if (lb_dis < l.lb_dis_truth_thresh) {
                                l.delta[obj_index] = 1 - l.output[obj_index];
                            }
                        } else {
                            l.delta[obj_index] = 1 - l.output[obj_index];
                        }

                        if (l.object_focal_loss) {
                            float alpha = 0.5;
                            // gamma = 2;
                            float pt = l.output[obj_index] + 0.000000000000001F;
                            float grad = -(1 - pt) * (2 * pt * logf(pt) + pt - 1);
                            l.delta[obj_index] *= alpha * grad;

                        }

                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0, l.label_smooth_rate);
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, l.use_center_regression, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h, l.iou_normalizer, l.iou_loss);
                    }
                }
            }
        }
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;

            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;

            // get mean and std of all iou(anchor, truth)
            float *per_ious = calloc(l.total, sizeof(float));
            float mean_iou = 0;
            float std_iou = 0;
            float target_iou = 0;
            for(n = 0; n < l.total; ++n){
                box pred = {0};
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift);
                mean_iou += iou;
                per_ious[n] = iou;
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
                if (n==l.total - 1) mean_iou /= l.total;
            }

            for(n = 0; n < l.total; ++n) {
                std_iou += pow(per_ious[n] - mean_iou, 2);
            }
            std_iou = sqrt(std_iou/l.total);

            target_iou = mean_iou + std_iou;

            //printf("mean: %.4f, std: %.4f, tagert: %.4f\n", mean_iou, std_iou, target_iou);

            for(n = 0; n < l.total; ++n) {
                int is_positive = 0;
                if(l.atss && per_ious[n] >= target_iou) is_positive = 1;
                if (n == best_n) is_positive = 1;

                int mask_n = int_index(l.mask, n, l.n);

                if (mask_n >= 0 && is_positive) {

                    int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                    ious all_ious = delta_yolo_box(truth, l.output, l.biases, n, box_index, l.use_center_regression, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h, l.iou_normalizer, l.iou_loss);

                    // range is 0 <= 1
                    tot_iou += all_ious.iou;
                    tot_iou_loss += 1 - all_ious.iou;
                    // range is -1 <= giou <= 1
                    tot_giou += all_ious.giou;
                    tot_giou_loss += 1 - all_ious.giou;

                    tot_diou += all_ious.diou;
                    tot_diou_loss += 1 - all_ious.diou;

                    tot_ciou += all_ious.ciou;
                    tot_ciou_loss += 1 - all_ious.ciou;

                    int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                    avg_obj += l.output[obj_index];

                    box pred = get_yolo_box(l.output, l.biases, n, box_index, l.use_center_regression, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
                    float lb_dis = box_lb_dis(pred, truth);

                    l.delta[obj_index] = 1 - l.output[obj_index];

                    if (l.rescore) {
                        l.delta[obj_index] = two_way_max(0, 1 - lb_dis / l.lb_dis_max_thresh) - l.output[obj_index];
                    }

                    if (l.object_focal_loss) {
                        float alpha = 0.5;
                        // gamma = 2;
                        float pt = l.output[obj_index] + 0.000000000000001F;
                        float grad = -(1 - pt) * (2 * pt * logf(pt) + pt - 1);
                        l.delta[obj_index] *= alpha * grad;
                    }

                    int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                    if (l.map) class = l.map[class];
                    int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                    delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat, l.label_smooth_rate);

                    ++count;
                    ++class_count;
                    if(all_ious.iou > .5) recall += 1;
                    if(all_ious.iou > .75) recall75 += 1;
                    //avg_iou += all_ious.iou;
                }
            }
        }
    }

    if (count == 0) count = 1;
    if (class_count == 0) class_count = 1;

    // original mse loss (not true, just for show)
    //*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    //printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);

    // Always compute classification loss both for iou + cls loss and for logging with mse loss
    // TODO: remove IOU loss fields before computing MSE on class
    //   probably split into two arrays
    int stride = l.w*l.h;
    float* no_iou_loss_delta = (float *)calloc(l.batch * l.outputs, sizeof(float));
    memcpy(no_iou_loss_delta, l.delta, l.batch * l.outputs * sizeof(float));
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    no_iou_loss_delta[index + 0 * stride] = 0;
                    no_iou_loss_delta[index + 1 * stride] = 0;
                    no_iou_loss_delta[index + 2 * stride] = 0;
                    no_iou_loss_delta[index + 3 * stride] = 0;
                }
            }
        }
    }
    float classification_loss = l.cls_normalizer * pow(mag_array(no_iou_loss_delta, l.outputs * l.batch), 2);
    free(no_iou_loss_delta);

    float avg_iou_loss = 0;
    // gIOU loss + MSE (objectness) loss
    if (l.iou_loss == MSE) {
        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
        avg_iou_loss = count > 0 ? (*(l.cost) - classification_loss) / count : 0;
    } else {
        if (l.iou_loss == IOU) {
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_iou_loss / count) : 0;
        }
        else if (l.iou_loss == GIOU) {
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_giou_loss / count) : 0;
        }
        else if (l.iou_loss == DIOU) {
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_diou_loss / count) : 0;
        }
        else if (l.iou_loss == CIOU) {
            avg_iou_loss = count > 0 ? l.iou_normalizer * (tot_ciou_loss / count) : 0;
        }
        *(l.cost) = avg_iou_loss + classification_loss;
    }


    fprintf(stderr, "(%s loss, normalizer: (iou: %.2f, cls: %.2f) Region %d Avg (iou: %f, %s: %f), Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f, count: %d, class_loss = %f, avg_iou_loss = %f, total_loss = %f \n",
        ((l.iou_loss == MSE) ? "mse" : (l.iou_loss == IOU ? "iou" : (l.iou_loss == GIOU ? "giou" : (l.iou_loss == DIOU ? "diou" : "ciou")))),
        l.iou_normalizer, l.cls_normalizer, net.index,
        tot_iou / count,
        ((l.iou_loss == MSE) ? "mse" : (l.iou_loss == IOU ? "iou" : (l.iou_loss == GIOU ? "giou" : (l.iou_loss == DIOU ? "diou" : "ciou")))),
        (l.iou_loss == MSE ?  tot_iou / count : (l.iou_loss == IOU ? tot_iou / count : (l.iou_loss == GIOU ? tot_giou / count : (l.iou_loss == DIOU ? tot_diou / count : tot_ciou / count)))),
        avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count,
        classification_loss, avg_iou_loss, *(l.cost));

}

void backward_yolo_layer(const layer l, network net)
{
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, l.use_center_regression, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    cuda_pull_array(net.input_gpu, net.input, l.batch*l.inputs);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            if (l.use_center_regression) {
                activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, HALFTANH);
            } else {
                activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            }
            if (l.scale_xy != 1.0) {
                scal_add_gpu(2*l.w*l.h, l.scale_xy, -0.5*(l.scale_xy - 1), l.output_gpu + index, 1); // scale x,y
            }
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }

    /*
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    int t;
    printf("yolo: %d output: ", l.index);
    for (t = 0; t < 10; ++t) {
        printf("%f ", l.output[t]);
    }
    printf("\n");
    */

    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

