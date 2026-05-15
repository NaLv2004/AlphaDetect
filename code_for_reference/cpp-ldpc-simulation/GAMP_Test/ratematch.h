#pragma once
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include"decode.h"
#define encode_outlen 3600///////50*Z
void RateMatchLDPC_pre(int *E_array, int &k0, int N, int C, int Zc, int bgn, int outlen, int rv, int modulation, int nlayers);
void modified_RateMatchLDPC(int8_t **input, int8_t *output, int8_t **temp, int *E_array, int k0, int N, int C, int Zc, int bgn, int outlen, int rv, int modulation);
void modified_cbsRateMatch(int8_t*d, int8_t*e1, int E, int k0, int Ncb, int Qm);
int8_t *RateMatchLDPC(int8_t **input, int N, int C, int Zc, int bgn, int outlen, int rv, int modulation, int nlayers);
int8_t * cbsRateMatch(int8_t*d, int E, int k0, int Ncb, int Qm);
void RateRecoverLDPC_pre(int G, int N, int C, int Zc, int bgn, int outlen, int rv, int modulation, int nlayers);
float **RateRecoverLDPC(float*input, int G, int N, int C, int Zc, int K, int K1, int bgn, int outlen, int rv, int modulation, int nlayers);
void modified_RateRecoverLDPC(float*input, float **temp, float **deconcatenated, float **LLR_in_C, int *E_array, int k0, int G, int N, int C, int Zc, int K, int K1, int bgn, int outlen, int rv, int modulation, int nlayers);
float * cbsRateRecover(float*d, int N, int E, int K, int Kd, int k0, int Ncb, int Qm);
void modified_cbsRateRecover(float*d, float*temp, int N, int E, int K, int Kd, int k0, int Ncb, int Qm);
int PDSCH_Transport_Block_Size_computation(float R, int modulation, int nlayers);