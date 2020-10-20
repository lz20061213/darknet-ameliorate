#include "double_yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_double_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int *input_layers)
{
    int i;
    layer l = {0};
    l.type = DOUBLE_YOLO;

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
    //printf("double_yolo: %d %d %d %d %d\n", l.batch, l.out_w, l.out_h, l.out_c, l.outputs);
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.input_layers = input_layers;


    l.forward = forward_double_yolo_layer;
    l.backward = backward_double_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_double_yolo_layer_gpu;
    l.backward_gpu = backward_double_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "double_yolo\n");
    srand(0);

    return l;
}

void resize_double_yolo_layer(layer *l, int w, int h)
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

box get_double_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

float delta_double_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_double_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}


void delta_double_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)
{
    int n;
    if (delta[index]){
        delta[index + stride*class] = 1 - output[index + stride*class];
        if(avg_cat) *avg_cat += output[index + stride*class];
        return;
    }
    for(n = 0; n < classes; ++n){
        delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

static int entry_cls_index(layer l, network net, int batch, int location, int entry)
{
    layer cls = net.layers[l.input_layers[0]];
    int n =   location / (l.w*l.h);
    int coor = location % (l.w*l.h);
    return batch * cls.outputs + n * l.w * l.h * (1 + l.classes) + entry * l.w * l.h + coor;
}

static int entry_loc_index(layer l, network net, int batch, int location, int entry)
{
    layer cls = net.layers[l.input_layers[0]];
    layer loc = net.layers[l.input_layers[1]];
    int offset = cls.batch * cls.outputs;
    int n =   location / (l.w*l.h);
    int coor = location % (l.w*l.h);
    return offset + batch*loc.outputs + n * l.w * l.h * 4 + entry * l.w * l.h + coor;
}

void forward_double_yolo_layer(const layer l, network net)
{
#ifndef GPU
    int cls_layer_index, loc_layer_index;
    cls_layer_index = l.input_layers[0];
    loc_layer_index = l.input_layers[1];

    layer cls = net.layers[cls_layer_index];
    layer loc = net.layers[loc_layer_index];

    float *cls_input = cls.output; // (1+classes) * l.w * l.h
    float *loc_input = loc.output; // (dx, dy, dw, dh)

    // copy cls_input and loc_input to output_gpu
    copy_cpu(cls.batch * cls.outputs, cls_input, 1, l.output, 1);
    copy_cpu(loc.batch * loc.outputs, loc_input, 1, l.output + cls.batch * cls.outputs, 1);
#else
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
#endif

    int i,j,b,t,n;

#ifndef GPU
    // activate cls
    activate_array(l.output, cls.batch * cls.outputs, LOGISTIC);
    // activate loc
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_loc_index(l, net, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            if (l.scale_xy != 1.0) {
                scal_add_cpu(2*l.w*l.h, l.scale_xy, -0.5*(l.scale_xy - 1), l.output + index, 1); // scale x,y
            }
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
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
                    int box_index = entry_loc_index(l, net, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_double_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
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
                    int obj_index = entry_cls_index(l, net, b, n*l.w*l.h + j*l.w + i, 0);
                    avg_anyobj += l.output[obj_index];

                    box truth = float_to_box(net.truth + best_t*(4 + 1)+ b*l.truths, 1);
                    float lb_dis = box_lb_dis(pred, truth);
                    // here we modify objectness to loc confidence
                    l.delta[obj_index] = 0 - l.output[obj_index];
                    if (l.rescore) {
                        l.delta[obj_index] = two_way_max(0, 1 - lb_dis / l.lb_dis_max_thresh) - l.output[obj_index];
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
                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];
                        int class_index = entry_cls_index(l, net, b, n*l.w*l.h + j*l.w + i, 1);
                        delta_double_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_double_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
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
            for(n = 0; n < l.total; ++n){
                box pred = {0};
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(l.mask, best_n, l.n);
            if(mask_n >= 0){
                int box_index = entry_loc_index(l, net, b, mask_n*l.w*l.h + j*l.w + i, 0);
                float iou = delta_double_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);

                int obj_index = entry_cls_index(l, net, b, mask_n*l.w*l.h + j*l.w + i, 0);
                avg_obj += l.output[obj_index];

                box pred = get_double_yolo_box(l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
                float lb_dis = box_lb_dis(pred, truth);

                l.delta[obj_index] = 1 - l.output[obj_index];

                if (l.rescore) {
                    l.delta[obj_index] = two_way_max(0, 1 - lb_dis / l.lb_dis_max_thresh) - l.output[obj_index];
                }

                int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class = l.map[class];
                int class_index = entry_cls_index(l, net, b, mask_n*l.w*l.h + j*l.w + i, 1);
                delta_double_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);

    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}

void backward_double_yolo_layer(const layer l, network net)
{
    int cls_layer_index, loc_layer_index;
    cls_layer_index = l.input_layers[0];
    loc_layer_index = l.input_layers[1];

    layer cls = net.layers[cls_layer_index];
    layer loc = net.layers[loc_layer_index];

    axpy_gpu(cls.batch * cls.outputs, 1, l.delta, 1, cls.delta, 1);
    axpy_gpu(loc.batch * loc.outputs, 1, l.delta+cls.batch * cls.outputs, 1, loc.delta, 1);
}

void correct_double_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
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

int double_yolo_num_detections(layer l, network net, float thresh)
{
    //printf("thresh in double_yolo_num_detections: %f\n", thresh);
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_cls_index(l, net, 0, n*l.w*l.h + i, 0);
            //printf("objectness: %f\n", l.output[obj_index]);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_double_yolo(layer l)
{
   // no implementation
}

int get_double_yolo_detections(layer l, network net, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    //printf("batch %d\n", l.batch);
    if (l.batch == 2) avg_flipped_double_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_cls_index(l, net, 0, n*l.w*l.h + i, 0);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            //printf("objectness: %f\n", objectness);
            int box_index  = entry_loc_index(l, net, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_double_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            //printf("bbox: %f %f %f %f\n", dets[count].bbox.x, dets[count].bbox.y, dets[count].bbox.w, dets[count].bbox.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            //printf("classes: %d\n", l.classes);
            for(j = 0; j < l.classes; ++j){
                //printf("j: %d\n", j);
                int class_index = entry_cls_index(l, net, 0, n*l.w*l.h + i, 1 + j);
                //printf("class_index: %d\n", class_index);
                float prob = objectness*predictions[class_index];
                //printf("after get prob: %f, thresh: %f\n", prob, thresh);
                //float test = (prob > thresh) ? prob : 0.0;
                //printf("test: %f\n", test);
                dets[count].prob[j] = (prob > thresh) ? prob : 0.0;
                //printf("after set prob\n");
            }
            ++count;
        }
    }
    correct_double_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU

void forward_double_yolo_layer_gpu(const layer l, network net)
{
    // get exactly cls and loc outputs
    int cls_layer_index, loc_layer_index;
    cls_layer_index = l.input_layers[0];
    loc_layer_index = l.input_layers[1];

    layer cls = net.layers[cls_layer_index];
    layer loc = net.layers[loc_layer_index];

    float *cls_input = cls.output_gpu; // (1+classes) * l.w * l.h
    float *loc_input = loc.output_gpu; // (dx, dy, dw, dh)

    //printf("double yolo forward ...\n");
    //printf("index: %d %d\n", cls_layer_index, loc_layer_index);
    //printf("channels: %d %d %d\n", l.out_c, cls.out_c, loc.out_c);
    assert(l.out_c == cls.out_c + loc.out_c);

    //printf("cls_outputs: %d\n", cls.batch * cls.outputs);
    //printf("loc_outputs: %d\n", loc.batch * loc.outputs);

    // copy cls_input and loc_input to output_gpu
    copy_gpu(cls.batch * cls.outputs, cls_input, 1, l.output_gpu, 1);
    copy_gpu(loc.batch * loc.outputs, loc_input, 1, l.output_gpu + cls.batch * cls.outputs, 1);

    //printf("after copy_gpu\n");
    // activate
    // activate all classes
    activate_array_gpu(l.output_gpu, cls.batch * cls.outputs, LOGISTIC, 0);
    // activate x, y
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_loc_index(l, net, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC, 0);
            if (l.scale_xy != 1.0) {
                scal_add_gpu(2*l.w*l.h, l.scale_xy, -0.5*(l.scale_xy - 1), l.output_gpu + index, 1); // scale x,y
            }
        }
    }

    //printf("after activate\n");

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

    //cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);

    //printf("begin forward_double_yolo1\n");
    forward_double_yolo_layer(l, net);
    //printf("after forward_double_yolo1\n");
    //printf("outputs size: %d %d\n", l.batch, l.outputs);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    //printf("after forward_double_yolo2\n");
}

void backward_double_yolo_layer_gpu(const layer l, network net)
{
    int cls_layer_index, loc_layer_index;
    cls_layer_index = l.input_layers[0];
    loc_layer_index = l.input_layers[1];

    layer cls = net.layers[cls_layer_index];
    layer loc = net.layers[loc_layer_index];

    axpy_gpu(cls.batch * cls.outputs, 1, l.delta_gpu, 1, cls.delta_gpu, 1);
    axpy_gpu(loc.batch * loc.outputs, 1, l.delta_gpu+cls.batch * cls.outputs, 1, loc.delta_gpu, 1);
}
#endif

