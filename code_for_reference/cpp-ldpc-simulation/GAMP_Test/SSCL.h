#include <vector>
#include <bitset>

void R0_list(float** LLR, int** sum, int node, int& l, int last_start_sum, int last_start, float* PM, int** p, int& stage, int& count_info);
void R1_list(float** LLR, int** sum, int node, int& l, int last_start_sum, int last_start, float* PM, int n, int L, int** p, float* W, int* index, int* better, int* worse, int stage);
void REP_list(float** LLR, int** sum, int node, int& l, int last_start_sum, int last_start, float* PM, int n, int L, int** p, float* W, int* index, int* better, int* worse, int stage);

void sort_list(int* p, int* q, float* W, int L);
