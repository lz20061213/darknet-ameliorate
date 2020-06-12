#ifndef BOX_H
#define BOX_H
#include "darknet.h"

typedef struct{
    float dx, dy, dw, dh;
} dbox;

float box_rmse(box a, box b);
float box_iou(box a, box b);
float box_giou(box a, box b);
float box_diou(box a, box b);
float box_ciou(box a, box b);
float box_lb_dis(box a, box b);

boxabs to_tblr(box a);

dbox diou(box a, box b);
dxrep dx_box_iou(box a, box b, IOU_LOSS iou_loss);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
