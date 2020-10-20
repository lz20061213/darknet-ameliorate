#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);
void fuse_conv_batchnorm(network *net);
int get_appr_range(float *X, int n, int bitwidth, float keep_rate);
void calculate_appr_fracs(network **nets, int ngpus);
#endif
