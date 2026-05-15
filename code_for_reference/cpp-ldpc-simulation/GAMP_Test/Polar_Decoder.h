#include "Polar_Encoder.h"
void decode(float* LLR, int* sum, int j, int* Ac, int a, int N, int K, float* Ma_SC);
void decode_BP(float* LLR_y, float** LLR_L, float** LLR_R, int iter, int stage, int* A_Ac, int N, int* u, int* x, int* u_xor, int crc_length, int& iteration);
void decode_BP_oneiter(float* LLR_y, float** LLR_L, float** LLR_R, int iter, int stage, int* A_Ac, int N, int* u, int* x, int* u_xor, int crc_length, int& iteration);

void decode_list_with_no_copy(float** LLR, int**, int**, int j, int* A_Ac, int a, int N, int K, float* PM, int L, int m, int** p, int* fg,
	float* LLR_in, float* W, int* index, int* better, int* worse, int* path_number, int* st);
int cal_stage(int i, int n);
int* PM_sort(float* FilterArray1, int len);
void r1_tree(int j, int* A_Ac, int N, int K, vector<int>& r1_1);
void SCAN_decode(float* LLR, float* beta_L, float* beta_R, int* A_Ac, int N, int* st, int* fg);
void SCAN_decode_hardware(float* llr_y, float** LLR_L, float** LLR_R, int* A_Ac, int N, float* Ma_SC, float& theta, int& Flip_bit, int* uhat, int* st, int* fg);
void SCAN_decode_hardware_list(float** LLR, float** beta_L, float** beta_R, int* A_Ac, int N,
	int* st, int* fg, float* PM, int L, int m, int** p,
	float* LLR_in, float* W, int* index, int* better, int* worse, int* path_number);
void LDPC_BP_Decoder(float* llr, int** Nv, int* dc, int** Nc, int* dv, int* uhat, int N, int N_Info, int iterations);
void LDPC_BP_Decoder_SP(float* llr, int** Nv, int* dc, int** Nc, int* dv, int* uhat, int N, int N_Info, int iterations);