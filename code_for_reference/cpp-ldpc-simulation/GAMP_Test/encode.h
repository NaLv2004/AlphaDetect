#pragma once
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <stdint.h>
using namespace std;
const int batchsize = 1;
void encode_pre(int K, int Zc, int bgn);
void encode(int** infobits, int K, int** out, int Zc, int bgn, int setIdx);
void nrLdpcEncode(int** in, int K, int bgn, int** out);