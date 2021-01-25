#ifndef BGR2DCT_H
#define BGR2DCT_H

#include <stdio.h>

typedef struct
{
    short *data;
    int w, h, c;
} DCT;

typedef unsigned char uint8_t;

#ifdef __cplusplus
extern "C" {
#endif

DCT bgr2dct(uint8_t *bgr_image, int w, int h, int only_y);

#ifdef __cplusplus
}
#endif

#endif  // BGR2DCT_H