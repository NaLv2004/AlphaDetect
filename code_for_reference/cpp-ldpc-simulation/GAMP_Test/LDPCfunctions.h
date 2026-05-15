#pragma once
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;
int8_t *DesegmentLDPC(int8_t **input, int cwlen, int F, int blklen, int C);
void SegmentLDPC(int C, int K, int K1, int L, unsigned char* a_data_crc, unsigned char** a_data_block, int8_t** a_data_block1);
void modified_DesegmentLDPC(int8_t **input, int **uhat1, int8_t *output, int L, int K1, int blklen, int C);
int** buildSubBlock(int Z, int shift);
void parityCheckMatrix(int **basegraph, int **H, int bgn, int Z);
void getH(int** V, int** H, int bgn, int Z, int setIdx);