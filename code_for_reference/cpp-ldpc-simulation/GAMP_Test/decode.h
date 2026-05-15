#pragma once
#include <vector>
#include <fstream>
#include <ctime>
#include <functional>
#include <algorithm>
#include <queue>
using namespace std;
void decodeLogDomainMinSum_converge(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, int** H);
void decodeLogDomainMinSum_rowlayered(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, vector<std::vector<float>>& Lqij_temp, vector<std::vector<float>>& Lqij, int** H, int quantization_flag);
void decodeLogDomain_adjMinSum_converge(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<std::vector<float>>& Lqij, vector<std::vector<float>>& Lqij_temp, vector<float>& LLR_VN, int** H, int quantization_flag);
void decodeLogDomain_adjMinSum_rowlayered(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, vector<std::vector<float>>& Lqij, int** H, int quantization_flag);
void decodeLogDomainSumProduct(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, int** H);
void decodeLogDomainSumProduct_rowlayered(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, vector<std::vector<float>>& Lqij, int** H);
void decodeLogDomainMinSum_adaptive_rowlayered(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, int Mb, int Zc, float N0, int adjusted_flag, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, vector<std::vector<float>>& Lqij, vector<std::vector<float>>& Lqij_temp, int** H, int quantization_flag);
void decodeLogDomain_ga_MinSum_rowlayered(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, int Zc, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, vector<std::vector<float>>& Lqij, vector<float>& Lqij_sort, vector<float>& Lqij_index, int** H, int quantization_flag);
void decodeLogDomain_RC_modified_MinSum_rowlayered(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, float N0, int adjusted_flag, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, vector<std::vector<float>>& Lqij, vector<float>& Lqij_sort, vector<float>& Lqij_index, int** H);
double sampleNormal();
float abs_f(float a);
double mysign(double sig);
float max_f(float a, float b);
float min_f(float a, float b);
float lambda_adjusted(float a);
void quantization_llr(float* LLR, int N, bool first_quan);
void quantization_v2c(vector<float>& LLR, int N, bool first_quan);
void quantization_c2v(float LLR);