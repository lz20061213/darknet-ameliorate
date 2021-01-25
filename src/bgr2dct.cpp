
#include "bgr2dct.h"

#include <malloc.h>
#include <jpeglib.h>
#include <turbojpeg.h>

#include "dctfromjpg.h"

extern "C" {

DCT bgr2dct(uint8_t *bgr_image, int w, int h, int only_y)
{
    //printf("enter bgr2dct\n");
    // bgr_image: WHC BGR
    tjhandle handle = tjInitCompress();
    // YCbCr选取的通道 Y:0..22, Cb: 22..27, Cr: 27..32
    const int subset[32] = {0, 1, 2, 3, 4, 8, 9, 10, 11,
                            12, 16, 17, 18, 19, 20, 24, 25, 26,
                            27, 32, 33, 34, 0, 1, 2, 8, 9, 0, 1, 2, 8, 9};
    // BGR顺序
    int pixelfmt = TJPF_BGR;
    unsigned char *outjpg_buf = NULL;
    unsigned long outjpg_size;
    int subsamp = 2;
    int quality = 100;
    int flags = 0;
    // img2buffer
    //printf("begin tjCompress2\n");
    tjCompress2(handle, bgr_image, h, h * 3, w, pixelfmt, &outjpg_buf, &outjpg_size, subsamp, quality, flags);
    tjDestroy(handle);
    bool normalized = true;
    int channels = 3;
    band_info Y, Cb, Cr;
    //buffer2dct
    //printf("begin dct\n");
    read_dct_coefficients_from_buffer_((char *)outjpg_buf, outjpg_size, normalized, channels, &Y, &Cb, &Cr);
    free(outjpg_buf);
    short *dct_outputs = (short *)malloc(sizeof(short) * (Y.dct_w) * (Y.dct_h) * 32);
    if (only_y)
    {
        for (int w = 0; w < Y.dct_w; w++)
            for (int h = 0; h < Y.dct_h; h++)
                for (int c = 0; c < 32; c++)
                    dct_outputs[h * Y.dct_w * 32 + w * 32 + c] = Y.dct[h * Y.dct_w * Y.dct_b + w * Y.dct_b + c];
    }
    else
    {
        for (int w = 0; w < Y.dct_w; w++)
            for (int h = 0; h < Y.dct_h; h++)
                for (int c = 0; c < 32; c++)
                {
                    if (c < 22)
                        dct_outputs[h * Y.dct_w * 32 + w * 32 + c] = Y.dct[h * Y.dct_w * Y.dct_b + w * Y.dct_b + subset[c]];
                    else if (c < 27)
                        dct_outputs[h * Y.dct_w * 32 + w * 32 + c] = Cb.dct[h / 2 * Cb.dct_w * Cb.dct_b + w / 2 * Cb.dct_b + subset[c]];
                    else
                        dct_outputs[h * Y.dct_w * 32 + w * 32 + c] = Cr.dct[h / 2 * Cr.dct_w * Cr.dct_b + w / 2 * Cr.dct_b + subset[c]];
                }
    }
    DCT out;
    out.data = dct_outputs;
    // img.w = dct.h, img.h = dct.w
    out.w = Y.dct_h;
    out.h = Y.dct_w;
    out.c = 32;
    free(Y.dct);
    free(Cb.dct);
    free(Cr.dct);
    return out;
}

}