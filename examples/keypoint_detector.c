#include "darknet.h"
#include "image.h"
#include "utils.h"
#include "parser.h"
#include <assert.h>
#include <libgen.h>

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

#define SAVE_EVERY 1000

int index_of_array(char **names, char *name, int num) {
    int i;
    for (i = 0; i < num; ++i) {
        if (strcmp(names[i], name) == 0)
            return i;
    }
    return -1;
}

void get_keypoints_flip_map(load_args *args, char** keypoint_names, char **keypoint_maps)
{
    int i, j;
    args->keypoints_flip_map = calloc(args->keypoint_id_map_count, sizeof(keypoint_id_map));
    printf("flip mapping:\n");
    for (i = 0; i < args->keypoint_id_map_count; ++i) {
        char *map_string = keypoint_maps[i];
        list *s = split_str(map_string, ':');
        assert(s->size == 2);
        char **lrs = (char **)list_to_array(s);
        int l = index_of_array(keypoint_names, lrs[0], args->keypoints_num);
        int r = index_of_array(keypoint_names, lrs[1], args->keypoints_num);
        args->keypoints_flip_map[i].lid = l;
        args->keypoints_flip_map[i].rid = r;
        printf("%s(%d) -> %s(%d)\n", lrs[0], l, lrs[1], r);
        // CHECK: for free
    }
}

void train_keypoint_detector(char *datacfg, char *cfgfile, char *weightfile, char *backup_directory, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    if (!backup_directory) {
        backup_directory = option_find_str(options, "backup", "/backup/");
    }

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        //printf("before load_network\n");
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
        //printf("after load_network\n");
    }

    // calculate the weights_frac_bitwiths and biases_frac_bitwidths when use per_channel
    //printf("before calculate_appr_fracs\n");
    if (nets[0]->quantize) {
        if (nets[0]->quantize_per_channel)
            calculate_appr_fracs_networks(nets, ngpus);
    }
    //printf("after calculate_appr_fracs\n");

    //assert(1==2);

    srand(time(0));
    network *net = nets[0];
    int last_save = 0;

#ifdef GPU
    // scale the conv weights and its bias
    for (i = 0; i < ngpus; ++i) {
        if (nets[i]->train_slimming) scale_gamma(nets[i], nets[i]->slimming_alpha);
    }
#endif

    int imgs = net->batch * net->subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = {0};
    for (i=net->n-1; i>=0; --i) {
        l = net->layers[i];
        if (l.type == KEYPOINT_YOLO)
            break;
    }

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.keypoints_num = l.keypoints_num;
    args.d = &buffer;
    args.type = KEYPOINT_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;

    // get keypoints_flip_map
    char *name_path = option_find_str(options, "names", "data/names.list");
    list *name_list = get_paths(name_path);
    char **names = (char **)list_to_array(name_list);

    char *keypoint_map_path = option_find_str(options, "keypoint_flip_maps", "data/flip.list");
    list *keypoint_map_list = get_paths(keypoint_map_path);
    int map_count = keypoint_map_list -> size;
    char **keypoint_maps = (char **)list_to_array(keypoint_map_list);

    args.keypoint_id_map_count = map_count;
    if (map_count > 0)
        get_keypoints_flip_map(&args, names+1, keypoint_maps);

    int init_w = net->w;
    int init_h = net->h;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        #pragma omp parallel for
        for(i = 0; i < ngpus; ++i){
            if(get_current_batch(net) > net->quantize_freezeBN_iterpoint) {
                nets[i]->quantize_freezeBN = 1;
            }
            else {
                nets[i]->quantize_freezeBN = 0;
            }
        }
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int randi = rand_int(-5, 10);
            int dim_w = randi * net->downsample_scale + init_w;
            int dim_h = randi * net->downsample_scale + init_h;
            if (get_current_batch(net)+200 > net->max_batches) {
                dim_w = 10 * net->downsample_scale + init_w;
                dim_h = 10 * net->downsample_scale + init_h;
            }
            //int dim = (rand() % 4 + 16) * 32;
            printf("input: %d %d\n", dim_w, dim_h);
            args.w = dim_w;
            args.h = dim_h;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim_w, dim_h);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
         */
        /*
           int zz;
           for(zz = 0; zz < train.X.cols; ++zz){
           image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[zz] + k*5, 1);
           printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
           draw_bbox(im, b, 1, 1,0,0);
           }
           show_image(im, "truth11");
           cvWaitKey(0);
           save_image(im, "truth11");
           }
         */
        float load_time = what_time_is_it_now()-time;
        if (load_time > 0.001)
            printf("Loaded: %lf seconds\n", load_time);

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);

#ifdef GPU
            // scale the conv weights and its bias back
            if (net->train_slimming) scale_gamma(net, 1.0 / net->slimming_alpha);
#endif
            save_weights(net, buff);

#ifdef GPU
            // scale the conv weights and its bias
            if (net->train_slimming) scale_gamma(net, net->slimming_alpha);
#endif
        }
        if((i-last_save) > SAVE_EVERY || i % SAVE_EVERY == 0 || (i < SAVE_EVERY && i % 100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            last_save = i;
#ifdef GPU
            // scale the conv weights and its bias back
            if (net->train_slimming) scale_gamma(net, 1.0 / net->slimming_alpha);
#endif
            save_weights(net, buff);

#ifdef GPU
            // scale the conv weights and its bias
            if (net->train_slimming) scale_gamma(net, net->slimming_alpha);
#endif
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);

#ifdef GPU
    // scale the conv weights and its bias back
    if (net->train_slimming) scale_gamma(net, 1.0 / net->slimming_alpha);
#endif

    save_weights(net, buff);
}

void heatmap_nms(const layer l)
{
    // only one batch
    int size = 7;
    int stride = 1;

    int i,j,k,m,n;
    int w_offset = -size/2;
    int h_offset = -size/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.keypoints_num;

    float *nms_heatmap = calloc(c * h * w, sizeof(float));
    memset(nms_heatmap, 0, c * h * w * sizeof(float));

    for(k = 0; k < c; ++k){
        for(i = 0; i < h; ++i){
            for(j = 0; j < w; ++j){
                int out_index = j + w*(i + h*(k + c*0));
                float max = -1;
                int max_i = -1;
                for(n = 0; n < size; ++n){
                    for(m = 0; m < size; ++m){
                        int cur_h = h_offset + i*stride + n;
                        int cur_w = w_offset + j*stride + m;
                        int index = cur_w + l.w*(cur_h + l.h*(k + 0*l.c));
                        int valid = (cur_h >= 0 && cur_h < l.h &&
                                     cur_w >= 0 && cur_w < l.w);
                        float val = (valid != 0) ? l.output[index] : -FLT_MAX;
                        max_i = (val > max) ? index : max_i;
                        max   = (val > max) ? val   : max;
                    }
                }
                if (out_index == max_i)
                    nms_heatmap[out_index] = max;
            }
        }
    }

    memcpy(l.output, nms_heatmap, c * h * w * sizeof(float));

    free(nms_heatmap);
}

/*  use for keypoint instead
typedef struct heatinfo {
    float value;    // for value
    float x;  // for coord
    float y;   // for coord
}
*/

int value_comparator(const void *kpa, const void *kpb)
{
    keypoint a = *(keypoint *)kpa;
    keypoint b = *(keypoint *)kpb;
    // CHECK: sigmoid (-1, 0, 1 will be update for classification)
    float diff = abs(a.v) - abs(b.v);
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

keypoint **heatmap_topk(const layer l, int topk)
{
    int i, k;
    keypoint **hm_kps = calloc(l.keypoints_num, sizeof(keypoint*));
    for (i=0; i < l.keypoints_num; ++i)
        hm_kps[i] = calloc(topk, sizeof(keypoint));

    for (i=0; i < l.keypoints_num; ++i) {
        float *heatmap = l.output + i * l.out_h * l.out_w;
        keypoint *hm_kp = calloc(l.out_h*l.out_w, sizeof(keypoint));
        for (k = 0; k < l.out_h * l.out_w; ++k) {
            hm_kp[k].v = heatmap[k];
            hm_kp[k].x = k % l.out_w;
            hm_kp[k].y = k / l.out_w;
        }
        // sort heatmap by value
        qsort(hm_kp, l.out_h*l.out_w, sizeof(keypoint), value_comparator);
        // get topk hm_kps for each keypoint
        for (k = 0; k < topk; ++k) {
            hm_kps[i][k].v =  hm_kp[k].v;
            hm_kps[i][k].x =  hm_kp[k].x;
            hm_kps[i][k].y =  hm_kp[k].y;
        }
        free(hm_kp);
    }

    return hm_kps;
}


void keypoints_decode(detection_with_keypoints *dets, int num, network *net, float thresh)
{
    // get the output of heatmaps and heatmaps_offset
    int i, j, k, topk = 100;
    int flag = 0;
    layer l = {0};
    for(i=net->n-1; i>=0; --i) {
        l = net->layers[i];
        if (l.type == HEATMAP) {
            flag = 1;
            break;
        }
    }
    if (!flag) return;

    // nms for heatmap
    heatmap_nms(l);
    // get the top k of heatmap
    keypoint **hm_kps = heatmap_topk(l, topk);
    // resume to origin
    int xindex, yindex;
    int down_ratio = net->h / l.out_h;
    assert(down_ratio == net->w / l.out_w);
    for (i=0; i<l.keypoints_num; ++i) {
        for (j=0; j<topk; ++j) {
            xindex = l.keypoints_num * l.out_h * l.out_w + (int)hm_kps[i][j].y * l.out_w + (int)hm_kps[i][j].x;
            yindex = (l.keypoints_num + 1) * l.out_h * l.out_w + (int)hm_kps[i][j].y * l.out_w + (int)hm_kps[i][j].x;
            hm_kps[i][j].x += l.output[xindex];
            hm_kps[i][j].x /= l.out_w;
            hm_kps[i][j].y += l.output[yindex];
            hm_kps[i][j].y /= l.out_h;
        }
    }
    // update dets by hm_kps with some conditions
    for (i=0; i<num; ++i) {
        detection_with_keypoints det = dets[i];
        for (j=0; j<l.keypoints_num; ++j) {
            keypoint kp = det.bkps.kps[j];
            float min_dist = 10000.0;
            int min_index = -1;
            for (k=0; k<topk; ++k) {
                keypoint hi = hm_kps[j][k];
                float dist = sqrtf((kp.x-hi.x)*(kp.x-hi.x) + (kp.y-hi.y)*(kp.y-hi.y));
                if (dist < min_dist) {
                    min_dist = dist;
                    min_index = k;
                }
            }
            if (hm_kps[j][min_index].v >= thresh && min_dist < fmaxf(det.bkps.w, det.bkps.h) * 0.3) {
                det.bkps.kps[j].v = hm_kps[j][min_index].v;
                det.bkps.kps[j].x = hm_kps[j][min_index].x;
                det.bkps.kps[j].y = hm_kps[j][min_index].y;
            }
        }
    }
}

void test_keypoint_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();

    network *net = load_network(cfgfile, weightfile, 0);

    if (net->quantize) {
        if (net->quantization_aware_training) {
            if (net->quantize_per_channel) {
                //int i, j;
                char *float_weightfile = option_find_str(options, "float_weightfile", "backup/final.weights");
                network *float_net = load_network(cfgfile, float_weightfile, 0);
                calculate_appr_fracs_network(float_net);
                copy_appr_fracs_network(float_net, net);
                free_network(float_net);
                printf("finish weight and bias fraction bitwidths calculation and copying\n");
                /*
                for(i = 0; i < net->n; ++i) {
                    layer *l = &net->layers[i];
                    if (l->type == CONVOLUTIONAL) {
                        printf("convolutional %d\n", i);
                        for (j = 0; j < l->n; ++j) {
                            printf("channel %d: %d %d\n", j, l->quantize_weight_fraction_bitwidths[j], l->quantize_bias_fraction_bitwidths[j]);
                        }
                    }
                }
                */
            }
        }
    }

    int i, j;
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = {0};
        for(i=net->n-1; i>=0; --i) {
            l = net->layers[i];
            if (l.type == KEYPOINT_YOLO) break;
        }
        printf("classes: %d\n", l.classes);

        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes_with_keypoints = 0;
        // 1. get the results of top-down branch
        detection_with_keypoints *dets = get_network_boxes_with_keypoints(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes_with_keypoints);
        /*
        for (i = 0; i < nboxes_with_keypoints; ++i) {
            detection_with_keypoints det = dets[i];
            for(j = 0; j < l.classes; ++j){
                printf("prob: %f\n", det.prob[j]);
            }
            printf("x, y, w, h: %f %f %f %f\n", det.bkps.x, det.bkps.y, det.bkps.w, det.bkps.h);
            for (j = 0; j < det.bkps.keypoints_num; ++j) {
                printf("vis, x, y: %f %f %f\n", det.bkps.kps[j].v, det.bkps.kps[j].x, det.bkps.kps[j].y);
            }
        }
        */
        // do nms by box
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) {
                do_nms_sort_with_keypoints(dets, nboxes_with_keypoints, l.classes, nms);
            } else {
                diounms_sort_with_keypoints(dets, nboxes_with_keypoints, l.classes, nms, l.nms_kind, l.beta_nms);
            }
        }
        // 2. decode bottom-up keypoints and merge
        keypoints_decode(dets, nboxes_with_keypoints, net, 0.25);  // todo: hp-thres
        // 3. correct net to im coord
        correct_keypoint_yolo_boxes_with_keypoints(dets, nboxes_with_keypoints, im.w, im.h, net->w, net->h, 1);
        draw_detections_with_keypoints(im, dets, nboxes_with_keypoints, thresh, names, alphabet, l.classes);
        free_detections_with_keypoints(dets, nboxes_with_keypoints);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}

void test_list_keypoint_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    char *test_list = option_find_str(options, "test_list", "data/test.list");
    char *out_folder = option_find_str(options, "out_folder", "predicts");

    int show_result = option_find_int(options, "show_result", 0);

    image **alphabet = load_alphabet();

    network *net = load_network(cfgfile, weightfile, 0);

    if (net->quantize) {
        if (net->quantization_aware_training) {
            if (net->quantize_per_channel) {
                int i, j;
                char *float_weightfile = option_find_str(options, "float_weightfile", "backup/final.weights");
                network *float_net = load_network(cfgfile, float_weightfile, 0);
                calculate_appr_fracs_network(float_net);
                copy_appr_fracs_network(float_net, net);
                free_network(float_net);
                printf("finish weight and bias fraction bitwidths calculation and copying\n");
                /*
                for(i = 0; i < net->n; ++i) {
                    layer *l = &net->layers[i];
                    if (l->type == CONVOLUTIONAL) {
                        printf("convolutional %d\n", i);
                        for (j = 0; j < l->n; ++j) {
                            printf("channel %d: %d %d\n", j, l->quantize_weight_fraction_bitwidths[j], l->quantize_bias_fraction_bitwidths[j]);
                        }
                    }
                }
                */
            }
        }
    }

    int i, j, m;
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=0.45;

    layer l = {0};
    for(i=net->n-1; i>=0; --i) {
        l = net->layers[i];
        if (l.type == KEYPOINT_YOLO) break;
    }

    list *plist = get_paths(test_list);
    char **paths = (char **) list_to_array(plist);
    int num = plist->size;

    for (m = 0; m < num; ++m) {
        char *path = paths[m];
        //printf("test image path: %s\n", path);
        image im = load_image_color(path, 0, 0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        printf("classes: %d\n", l.classes);

        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", path, what_time_is_it_now()-time);
        int nboxes_with_keypoints = 0;
        // 1. get the results of top-down branch
        detection_with_keypoints *dets = get_network_boxes_with_keypoints(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes_with_keypoints);
        /*
        for (i = 0; i < nboxes_with_keypoints; ++i) {
            detection_with_keypoints det = dets[i];
            for(j = 0; j < l.classes; ++j){
                printf("prob: %f\n", det.prob[j]);
            }
            printf("x, y, w, h: %f %f %f %f\n", det.bkps.x, det.bkps.y, det.bkps.w, det.bkps.h);
            for (j = 0; j < det.bkps.keypoints_num; ++j) {
                printf("vis, x, y: %f %f %f\n", det.bkps.kps[j].v, det.bkps.kps[j].x, det.bkps.kps[j].y);
            }
        }
        */
        // do nms by box
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) {
                do_nms_sort_with_keypoints(dets, nboxes_with_keypoints, l.classes, nms);
            } else {
                diounms_sort_with_keypoints(dets, nboxes_with_keypoints, l.classes, nms, l.nms_kind, l.beta_nms);
            }
        }
        // 2. decode bottom-up keypoints and merge
        keypoints_decode(dets, nboxes_with_keypoints, net, 0.25);  // todo: hp-thres
        // 3. correct net to im coord
        correct_keypoint_yolo_boxes_with_keypoints(dets, nboxes_with_keypoints, im.w, im.h, net->w, net->h, 1);
        draw_detections_with_keypoints(im, dets, nboxes_with_keypoints, thresh, names, alphabet, l.classes);
        free_detections_with_keypoints(dets, nboxes_with_keypoints);

        if(out_folder){
            char outpath[256];
            find_replace(path, "images", out_folder, outpath);
            // create folder
            char *outfolder = dirname(outpath);
            if (access(outfolder, 0)==-1) {
                // mk folder revision
                int stat = folder_mkdirs(outfolder);
            }
            //find_replace(outpath, ".jpg/../", "", outpath);
            //save_image(im, outfile);
        }

        if (show_result) {
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
    }
}

void run_keypoint_detector(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    //int class = find_int_arg(argc, argv, "-class", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    char *backup_directory = (argc > 6) ? argv[6] : 0;

    float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
    int map_points = find_int_arg(argc, argv, "-points", 0);
    int letter_box = find_int_arg(argc, argv, "-letter_box", 1);

    if(0==strcmp(argv[2], "test")) test_keypoint_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "test_list")) test_list_keypoint_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train")) train_keypoint_detector(datacfg, cfg, weights, backup_directory, gpus, ngpus, clear);

}