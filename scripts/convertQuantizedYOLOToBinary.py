#!/usr/bin/env python
# encoding: utf-8

# --------------------------------------------------------
# file: convertQuantizedYOLOToBinary.py
# Copyright(c) 2017-2020 SeetaTech
# Written by Zhuang Liu
# 2020/07/29 10:50
# --------------------------------------------------------

import os
import numpy as np
import argparse
import io
# import ConfigParser
import configparser   # need change to python3.5
from collections import defaultdict
import struct
import re

# load quantized weigths from numpy format
def loadNpz(npz_path):
    r = np.load(npz_path, allow_pickle=True)
    header = r['header'].tolist()
    infos = r['infos'].tolist()
    weights = r['weights'].tolist()

    return weights, infos

def writeBinary(path, weights, infos):
    with open(path, 'wb') as fd:
        for i in range(len(infos)):

            info = infos[i]
            weight = weights[i]
            #print(info)

            if info['type'] == 'convolution':

                conv = weight['conv'][0]
                conv_fl = weight['conv'][1]
                bias = weight['bias'][0]
                bias_fl = weight['bias'][1]

                # write layerparam, no need
                krow, kcol, in_c, out_c = conv.shape
                biasFlag = 1
                bnFlag = 0
                reluFlag = 2
                poolFlag = 0
                cnnStride = 1

                # write weightparam
                # 'b' for int8, 'i' for int32
                # first k_w, then k_h, then k_out, final k_in
                if conv is not None:
                    for c in range(conv.shape[2]):
                        for n in range(conv.shape[3]):
                            for h in range(conv.shape[0]):
                                for w in range(conv.shape[1]):
                                    print(conv[h, w, c, n])
                                    fd.write(struct.pack('b', conv[h, w, c, n]))

                if conv_fl is not None:
                    fd.write(struct.pack('b', conv_fl))

                if bias is not None:
                    for i in range(len(bias)):
                        fd.write(struct.pack('i', bias[i]))

                if conv_fl is not None:
                    fd.write(struct.pack('b', np.int8(bias_fl)))

if __name__ == '__main__':

    # static
    qnpz_path = '../backup/ship_tiny_yolov3_prune/float32_quantized_int8_static/yolov3-tiny-prune_quantized.npz'
    layer_weights, layer_infos = loadNpz(qnpz_path)

    # write binary_proto
    date = 2020082102  # change
    dst_model_path = 'binary/weights_{}.dat'.format(date)  # change
    writeBinary(dst_model_path, layer_weights, layer_infos)