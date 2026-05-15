#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <bitset>
#include <vector>
#include <functional>
#include <queue>
#include <string.h>
#include <fstream>
#include "Polar_Encoder.h"
#include "Polar_Function.h"
#include <xmmintrin.h>
#include <immintrin.h>
using namespace std;
#define hard(n) (n>=0?0:1)
//std::vector<std::vector<int>> path_min(32, std::vector<int>(4096));
int path_min[32][32768];
std::vector<int> path_mintemp(32768);
int path_mintemptemp[32768];
vector<float> NM_R1(32768);
vector<float> NM_R1_2(32768);

bool* temp_01 = new bool[32768];
ofstream cc("cc.txt");


struct CompSmall {
	CompSmall(const vector<float>& v) : _v(v) {}
	bool operator ()(int a, int b) { return _v[a] < _v[b]; }
	const vector<float>& _v;
};

void sort_list(int* p, int* q, float* W, int L)
{
	int p_index = 0; int q_index = 0;
	int cnt = 0;
	for (int i = 0; i < L; ++i) {
		for (int k = 1; k < L; ++k)
		{
			if (W[p[p_index]] > W[p[k]]) {
				p_index = k;
			}
			if (W[q[q_index]] < W[q[k]]) {
				q_index = k;
			}
		}
		if (W[p[p_index]] < W[q[q_index]])
		{
			swap(p[p_index], q[q_index]);
			p_index = 0; q_index = 0;
			cnt++;
		}
		else { //fout << "times = "<<cnt << endl;
			break;
		}
	}
}

struct Comp {
	Comp(const vector<float>& v) : _v(v) {}
	bool operator ()(int a, int b) { return _v[a] > _v[b]; }
	const vector<float>& _v;
};

float kth_elem_small(vector<float>& a, int* p, int low, int high, int k)
{
	int pivot = p[low];
	int low_temp = low;
	int high_temp = high;
	while (low < high)
	{
		while (low < high && a[p[high]] >= a[pivot])
			--high;
		p[low] = p[high];
		while (low < high && a[p[low]] < a[pivot])
			++low;
		p[high] = p[low];
	}
	p[low] = pivot;

	if (low == k - 1)
		return p[low];
	else if (low > k - 1)
		return kth_elem_small(a, p, low_temp, low - 1, k);
	else
		return kth_elem_small(a, p, low + 1, high_temp, k);
}

void R0_list(float** LLR, int** sum, int node, int& l, int last_start_sum, int last_start, float* PM, int** p, int& stage, int& count_info)
{
	for (int k = 0; k < l; k++) {
		memset(*(sum + k) + last_start_sum, 0, sizeof(int) * node);
		//p[k][stage] = k;
	}
	if (count_info > 0) {
		if (node >= 8)
		{
			__m256 zero = _mm256_setzero_ps();
			for (int k = 0; k < l; ++k) {
				__m256 PM_256 = _mm256_setzero_ps();
				for (int i = 0; i < node; i += 8) {
					__m256 a = _mm256_loadu_ps(*(LLR + k) + last_start + i);
					PM_256 = _mm256_add_ps(PM_256, _mm256_min_ps(a, zero));
				}
				float* ptr_PM = (float*)&PM_256;
				for (int i = 0; i < 8; ++i)
				{
					PM[k] += ptr_PM[i];
				}
			}
		}
		else
		{
			for (int k = 0; k < l; k++) {
				for (int i = 0; i < node; i++) {
					PM[k] += min_f(*(*(LLR + k) + last_start + i), 0.0);
				}
			}
		}
	}
}

void R1_list(float** LLR, int** sum, int node, int& l, int last_start_sum, int last_start, float* PM, int n, int L, int** p, float* W, int* index, int* better, int* worse, int stage)
{
	int sort_size = min(L - 1, node);
	//sort_size = 2;

	for (int k = 0; k < L; ++k) {  //Path expansion and new path metric calculation
		hard_SIMD(*(LLR + k) + last_start, *(sum + k) + last_start_sum, node);
		for (int i = 0; i < node; i++)
		{
			NM_R1[i] = abs_f(*(*(LLR + k) + last_start + i));
			path_min[k][i] = i;
		}

		kth_elem_small(NM_R1, path_min[k], 0, node - 1, sort_size);
		for (int i = 0; i < sort_size; i++)
		{
			path_mintemp[i] = i;
			NM_R1_2[i] = NM_R1[path_min[k][i]];
		}
		sort(path_mintemp.begin(), path_mintemp.begin() + sort_size, CompSmall(NM_R1_2));
		for (int i = 0; i < sort_size; i++)
		{
			path_mintemptemp[i] = path_min[k][i];
		}
		for (int i = 0; i < sort_size; i++)
		{
			path_min[k][i] = path_mintemptemp[path_mintemp[i]];
		}
		//sort(path_min[k].begin(), path_min[k].begin() + node, CompSmall(NM_R1));
	}

	for (int seq = 0; seq < sort_size; seq++) {
		if (seq == 0)
		{
			for (int k = 0; k < L; ++k) {
				W[(k << 1)] = PM[k];
				W[(k << 1) + 1] = PM[k] - abs_f(*(*(LLR + p[k][stage]) + last_start + path_min[p[k][stage]][0]));
				better[k] = (k << 1);
				worse[k] = (k << 1) + 1;
			}
		}
		else {
			for (int k = 0; k < L; ++k) {
				//path_min[k][0] = path_min[p[k][stage]][seq];
				W[(k << 1)] = PM[k];
				W[(k << 1) + 1] = PM[k] - abs_f(*(*(LLR + p[k][stage]) + last_start + path_min[p[k][stage]][seq]));;
				better[k] = (k << 1);
				worse[k] = (k << 1) + 1;
			}
		}
		sort_list(better, worse, W, L);  // Path metric sorting
										 //fout << endl;
		for (int k = 0; k < L; ++k) {
			index[k] = (better[k] >> 1);
			if (index[k] != k) {
				for (int s = stage; s <= n; s++) {
					p[k][s] = p[index[k]][s];
				}
				replace_sum(*(sum + index[k]) + last_start_sum, *(sum + k) + last_start_sum, node);
			}
		}

		for (int k = 0; k < L; ++k) {  //path updating
			sum[k][last_start_sum + path_min[p[k][stage]][seq]] ^= (better[k] % 2);
			PM[k] = W[better[k]];
		}
	}
}

void REP_list(float** LLR, int** sum, int node, int& l, int last_start_sum, int last_start, float* PM, int n, int L, int** p, float* W, int* index, int* better, int* worse, int stage)
{
	if (l < L) {
		for (int k = l - 1; k >= 0; k--) {
			float NM_0 = 0;
			float NM_1 = 0;
			if (node > 4) {
				__m128 NM_0_temp = _mm_castsi128_ps(_mm_set1_epi32(0x00000000));
				__m128 NM_1_temp = NM_0_temp;
				for (int i = 0; i < node; i += 4)
				{
					__m128 llr_128 = _mm_loadu_ps(*(LLR + k) + last_start + i);
					__m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x00000000));
					__m128 min_0 = _mm_min_ps(llr_128, SIGN_MASK);
					__m128 max_1 = _mm_max_ps(llr_128, SIGN_MASK);
					NM_0_temp = _mm_add_ps(NM_0_temp, min_0);
					NM_1_temp = _mm_add_ps(NM_1_temp, max_1);
				}
				float* ptr_0 = (float*)&NM_0_temp;
				NM_0 = 0 - ptr_0[0] - ptr_0[1] - ptr_0[2] - ptr_0[3];
				float* ptr_1 = (float*)&NM_1_temp;
				NM_1 = ptr_1[0] + ptr_1[1] + ptr_1[2] + ptr_1[3];
			}
			else {
				for (int i = 0; i < node; ++i) {
					NM_0 -= min_f(*(*(LLR + k) + last_start + i), 0);
					NM_1 += max_f(*(*(LLR + k) + last_start + i), 0);
				}
			}
			PM[(k << 1) + 1] = PM[k] - NM_1;
			PM[(k << 1)] = PM[k] - NM_0;

			for (int i = last_start_sum; i < last_start_sum + node; ++i) {
				sum[(k << 1)][i] = 0;
				sum[(k << 1) + 1][i] = 1;
			}
			for (int s = stage; s <= n; s++) {
				p[(k << 1) + 1][s] = p[k][s];
				p[(k << 1)][s] = p[k][s];
			}
		}
		l <<= 1;
	}
	else {

		for (int k = 0; k < l; ++k) {
			float NM_0 = 0;
			float NM_1 = 0;
			__m128 NM_0_temp = _mm_castsi128_ps(_mm_set1_epi32(0x00000000));
			__m128 NM_1_temp = NM_0_temp;
			for (int i = 0; i < node; i += 4)
			{
				__m128 llr_128 = _mm_loadu_ps(*(LLR + k) + last_start + i);
				__m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x00000000));
				__m128 min_0 = _mm_min_ps(llr_128, SIGN_MASK);
				__m128 max_1 = _mm_max_ps(llr_128, SIGN_MASK);
				NM_0_temp = _mm_add_ps(NM_0_temp, min_0);
				NM_1_temp = _mm_add_ps(NM_1_temp, max_1);
			}
			float* ptr_0 = (float*)&NM_0_temp;
			float* ptr_1 = (float*)&NM_1_temp;
			if (node == 1) {
				NM_0 = ptr_0[0]; NM_1 = ptr_1[0];
			}
			else if (node == 2) {
				NM_0 = ptr_0[0] + ptr_0[1];
				NM_1 = ptr_1[0] + ptr_1[1];
			}
			else {
				NM_0 = ptr_0[0] + ptr_0[1] + ptr_0[2] + ptr_0[3];
				NM_1 = ptr_1[0] + ptr_1[1] + ptr_1[2] + ptr_1[3];
			}
			W[(k << 1)] = PM[k] + NM_0;
			W[(k << 1) + 1] = PM[k] - NM_1;
			temp_01[(k << 1)] = 0;
			temp_01[(k << 1) + 1] = 1;
			if ((-NM_0) > NM_1) {
				better[k] = (k << 1) + 1;
				worse[k] = (k << 1);
			}
			else {
				better[k] = (k << 1);
				worse[k] = (k << 1) + 1;
			}
		}

		sort_list(better, worse, W, L);  // Path metric sorting

		for (int k = 0; k < L; ++k) {
			index[k] = better[k] >> 1;
			if (index[k] != k) {
				for (int s = stage; s <= n; s++) {
					p[k][s] = p[index[k]][s];
				}
			}
		}

		for (int k = 0; k < l; ++k) {  //path updating
			int cntBlock = (node + 3) >> 2;
			__m128i* p_new = (__m128i*)(sum[k] + last_start_sum);
			__m128i xidLoad = temp_01[better[k]] == 0 ? _mm_set1_epi32(0x00000000) : _mm_set1_epi32(0x00000001);
			for (int i = 0; i < cntBlock; ++i)
			{
				_mm_storeu_si128(p_new + i, xidLoad);
			}
			PM[k] = W[better[k]];
		}
	}
}