#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cmath>

#include <bitset>
#include <algorithm>
#include <vector>
using namespace std;


void polar_codeconstruction(int CodeLength, float sigma, vector<int>& best_channel, vector<double>& P_ui);
//int* bitreorder(int* min, int len);
void polar_codeconstruction_punc(int RealLength, int CodeLength, float sigma, vector<int>& best_channel, vector<double>& mean);
void PolarEncode_xor_int8(int8_t* uout, int8_t* uin, int len);
void PolarEncode_xor(int* uout, int* uin, int len);
