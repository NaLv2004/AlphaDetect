#pragma once
#include <vector>
#include <fstream>
#include <ctime>
#include <functional>
#include <algorithm>
#include <queue>
using namespace std;
void Gaussian_ele(int CodeLength, int N_info, int** G_aug_swapped);
void Gaussian_ele1(int CodeLength, int N_info, int** G_aug_swapped, vector<int>& index_indep, vector<int>& index_parity);
void GRAND_set_build(vector<vector<int>>& GRAND_flip_set, vector<int>& current_set, int CodeLength, int AB_LW, bool GRAND_type);
void crc_generator_matrix2(int A, int L, int* polynomial, int** G_crc);
void decodeOrdered_OSD(int CodeLength, int DataLength, int N_info, int CRC_Length, int AB_LW, float* LLR_in, float* LLR_in_abs, int** G_new, int** G_swapped, int** G_swapped2, int* uhat, int* uhat_swapped1, int* uhat_swapped2, unsigned char* uhat_crc_check, int* uhat_best, vector<int>& index_LLR, vector<int>& index_indep, vector<int>& index_parity, vector<vector<int>>& GRAND_flip_set, vector<int>& current_set);
void decodeOrdered_OSD_CRCaided(int CodeLength, int DataLength, int N_info, int CRC_Length, int AB_LW, float* LLR_in, float* LLR_in_abs, int** G_new, int** G_swapped, int** G_swapped2, int** H, int* uhat, int* uhat_swapped1, int* uhat_swapped2, unsigned char* uhat_crc_check, int* uhat_best, vector<int>& index_LLR, vector<int>& index_indep, vector<int>& index_parity, vector<vector<int>>& GRAND_flip_set, vector<int>& current_set);
void decodeOrdered_OSD_CRCaided_Hardware(int CodeLength, int DataLength, int N_info, int CRC_Length, int AB_LW, float* LLR_in, float* LLR_in_abs, int* pivot_flag, int** G_new, int** G_swapped, int** G_swapped2, int** H, int* uhat, int* uhat_encode, int* uhat_swapped2, unsigned char* uhat_crc_check, int* uhat_best, vector<int>& index_LLR, vector<int>& index_indep, vector<int>& index_parity, vector<vector<int>>& GRAND_flip_set, vector<int>& current_set);
void swap_row(int* a, int* b, int N);
void decode_OSD_modified_for_hardware(float* LLR_in, int* Bit_out, int M, int N, int** H, vector<int>& index_indep, vector<int>& index_parity, int* pivot_flag, float* LLR_in_abs, vector<int>& index_LLR, int** Htemp, int** H_swapped2, int* LRIP, int* LRIPtemp, int* syndrome, int* syndrometemp, int* Bit_temp);