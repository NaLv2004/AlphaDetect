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
#include "SSCL.h"
#include <xmmintrin.h>
#include <immintrin.h>

int sumtemp[1024];
vector<float> NLLR_R1(256);
float W_R1[256];
int ppp[256];
int min1_index[256];
int min0_index[256];
//bool * temp_01 = new bool[2048];
int fpath_0[256];
int fpath_2[256];
float fWpath_0[256];
float fWpath_2[256];

#define qhard(n) (n>=0?0:1)
#define qalph 0.5
#define qmthres(n) (abs(n)+log(1 + exp(-qalph*abs(n))) / qalph)
#define qhthres(n) (log(1 + exp(-qalph*abs(n))) / qalph)

int min_index_sse(float* array, int size) {
	int minindex;
	if (size == 2)
		minindex = array[0] < array[1] ? 0 : 1;
	else
	{
		const __m128i increment = _mm_set1_epi32(4);
		__m128i indices = _mm_setr_epi32(0, 1, 2, 3);
		__m128i minindices = indices;
		__m128 minvalues_ps = _mm_loadu_ps(array);
		__m128i minvalues = _mm_cvtps_epi32(minvalues_ps);

		for (size_t i = 4; i < size; i += 4) {

			indices = _mm_add_epi32(indices, increment);

			const __m128i values = _mm_cvtps_epi32(_mm_loadu_ps(array + i));
			const __m128i lt = _mm_cmplt_epi32(values, minvalues);
			minindices = _mm_blendv_epi8(minindices, indices, lt);
			minvalues = _mm_min_epi32(values, minvalues);
		}

		// find min index in vector result (in an extremely naive way)
		int32_t values_array[4];
		uint32_t indices_array[4];

		_mm_storeu_si128((__m128i*)values_array, minvalues);
		_mm_storeu_si128((__m128i*)indices_array, minindices);

		minindex = indices_array[0];
		int32_t minvalue = values_array[0];

		int block_size = size < 4 ? size : 4;
		for (int i = 1; i < block_size; i++) {
			if (values_array[i] < minvalue) {
				minvalue = values_array[i];
				minindex = indices_array[i];
			}
			else if (values_array[i] == minvalue) {
				minindex = std::min(minindex, int(indices_array[i]));
			}
		}
	}
	return minindex;
}

int max_index_sse(float* array, int size) {
	int maxindex;
	if (size == 2)
		maxindex = array[0] > array[1] ? 0 : 1;
	else
	{
		const __m128i increment = _mm_set1_epi32(4);
		__m128i indices = _mm_setr_epi32(0, 1, 2, 3);
		__m128i maxindices = indices;
		__m128 maxvalues_ps = _mm_loadu_ps(array);
		__m128i maxvalues = _mm_cvtps_epi32(maxvalues_ps);

		for (size_t i = 4; i < size; i += 4) {

			indices = _mm_add_epi32(indices, increment);

			const __m128i values = _mm_cvtps_epi32(_mm_loadu_ps(array + i));
			const __m128i lt = _mm_cmpgt_epi32(values, maxvalues);
			maxindices = _mm_blendv_epi8(maxindices, indices, lt);
			maxvalues = _mm_max_epi32(values, maxvalues);
		}

		// find min index in vector result (in an extremely naive way)
		int32_t values_array[4];
		uint32_t indices_array[4];

		_mm_storeu_si128((__m128i*)values_array, maxvalues);
		_mm_storeu_si128((__m128i*)indices_array, maxindices);

		maxindex = indices_array[0];
		int32_t maxvalue = values_array[0];

		int block_size = size < 4 ? size : 4;
		for (int i = 1; i < block_size; i++) {
			if (values_array[i] > maxvalue) {
				maxvalue = values_array[i];
				maxindex = indices_array[i];
			}
			else if (values_array[i] == maxvalue) {
				maxindex = std::max(maxindex, int(indices_array[i]));
			}
		}
	}
	return maxindex;
}

void SCAN_function_noadd(float* r, float* a, float* b1, int size)
{
	if (size < 4)
	{
		for (int i = 0; i < size; i++)
		{
			r[i] = 1 * sgn(a[i])*sgn(b1[i])*min_f(abs_f(a[i]), abs_f(b1[i]));
		}
	}
	else if (size == 4)
	{
		__m128 a_sse = _mm_loadu_ps(a);//
		__m128 b_sse = _mm_loadu_ps(b1);
		__m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
		__m128 sign = _mm_and_ps(_mm_xor_ps(a_sse, b_sse), SIGN_MASK);
		__m128 abs_a_sse = _mm_andnot_ps(SIGN_MASK, a_sse);
		__m128 abs_b_sse = _mm_andnot_ps(SIGN_MASK, b_sse);
		__m128 s = _mm_or_ps(sign, _mm_min_ps(abs_a_sse, abs_b_sse));
		__m128 scale_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));
		__m128 scale_s = _mm_mul_ps(s, scale_MASK);
		_mm_storeu_ps(r, scale_s);
	}
	else
	{
		for (int i = 0; i < size; i += 8)
		{
			__m256 a_avx = _mm256_loadu_ps(a + i);
			__m256 b_avx = _mm256_loadu_ps(b1 + i);

			__m256 SIGN_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
			__m256 sign = _mm256_and_ps(_mm256_xor_ps(a_avx, b_avx), SIGN_MASK);
			__m256 abs_a_avx = _mm256_andnot_ps(SIGN_MASK, a_avx);
			__m256 abs_b_avx = _mm256_andnot_ps(SIGN_MASK, b_avx);

			__m256 s = _mm256_or_ps(sign, _mm256_min_ps(abs_a_avx, abs_b_avx));

			__m256 scale_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x3F800000));
			__m256 scale_s = _mm256_mul_ps(s, scale_MASK);
			_mm256_storeu_ps(r + i, scale_s);
		}
	}
	return;
}

void SCAN_function(float* r, float* a, float* b1, float* b2, int size)
{
	if (size < 4)
	{
		for (int i = 0; i < size; i++)
		{
			r[i] = 1 * sgn(a[i])*sgn(b1[i] + b2[i])*min_f(abs_f(a[i]), abs_f(b1[i] + b2[i]));
		}
	}
	else if (size == 4)
	{
		__m128 a_sse = _mm_loadu_ps(a);//
		__m128 b1_sse = _mm_loadu_ps(b1);
		__m128 b2_sse = _mm_loadu_ps(b2);
		__m128 b_sse = _mm_add_ps(b1_sse, b2_sse);
		__m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
		__m128 sign = _mm_and_ps(_mm_xor_ps(a_sse, b_sse), SIGN_MASK);
		__m128 abs_a_sse = _mm_andnot_ps(SIGN_MASK, a_sse);
		__m128 abs_b_sse = _mm_andnot_ps(SIGN_MASK, b_sse);
		__m128 s = _mm_or_ps(sign, _mm_min_ps(abs_a_sse, abs_b_sse));
		__m128 scale_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));
		__m128 scale_s = _mm_mul_ps(s, scale_MASK);
		_mm_storeu_ps(r, scale_s);
	}
	else
	{
		for (int i = 0; i < size; i += 8)
		{
			__m256 a_avx = _mm256_loadu_ps(a + i);
			__m256 b1_avx = _mm256_loadu_ps(b1 + i);
			__m256 b2_avx = _mm256_loadu_ps(b2 + i);
			__m256 b_avx = _mm256_add_ps(b1_avx, b2_avx);

			__m256 SIGN_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
			__m256 sign = _mm256_and_ps(_mm256_xor_ps(a_avx, b_avx), SIGN_MASK);
			__m256 abs_a_avx = _mm256_andnot_ps(SIGN_MASK, a_avx);
			__m256 abs_b_avx = _mm256_andnot_ps(SIGN_MASK, b_avx);

			__m256 s = _mm256_or_ps(sign, _mm256_min_ps(abs_a_avx, abs_b_avx));

			__m256 scale_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x3F800000));
			__m256 scale_s = _mm256_mul_ps(s, scale_MASK);
			_mm256_storeu_ps(r + i, scale_s);
		}
	}
	return;
}
void SCAN_function2(float* r, float* a, float* b1, float* b2, int size)
{
	if (size < 4)
	{
		for (int i = 0; i < size; i++)
		{
			r[i] = b2[i] + 1 * sgn(a[i])*sgn(b1[i])*min_f(abs_f(a[i]), abs_f(b1[i]));
		}
	}
	else if (size == 4)
	{
		__m128 a_sse = _mm_loadu_ps(a);//
		__m128 b1_sse = _mm_loadu_ps(b1);
		__m128 b2_sse = _mm_loadu_ps(b2);

		__m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
		__m128 sign = _mm_and_ps(_mm_xor_ps(a_sse, b1_sse), SIGN_MASK);
		__m128 abs_a_sse = _mm_andnot_ps(SIGN_MASK, a_sse);
		__m128 abs_b1_sse = _mm_andnot_ps(SIGN_MASK, b1_sse);
		__m128 s = _mm_or_ps(sign, _mm_min_ps(abs_a_sse, abs_b1_sse));
		__m128 scale_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));
		__m128 scale_s = _mm_mul_ps(s, scale_MASK);

		__m128 r_sse = _mm_add_ps(scale_s, b2_sse);
		_mm_storeu_ps(r, r_sse);
	}
	else
	{
		for (int i = 0; i < size; i += 8)
		{
			__m256 a_avx = _mm256_loadu_ps(a + i);
			__m256 b1_avx = _mm256_loadu_ps(b1 + i);
			__m256 b2_avx = _mm256_loadu_ps(b2 + i);

			__m256 SIGN_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
			__m256 sign = _mm256_and_ps(_mm256_xor_ps(a_avx, b1_avx), SIGN_MASK);
			__m256 abs_a_avx = _mm256_andnot_ps(SIGN_MASK, a_avx);
			__m256 abs_b1_avx = _mm256_andnot_ps(SIGN_MASK, b1_avx);

			__m256 s = _mm256_or_ps(sign, _mm256_min_ps(abs_a_avx, abs_b1_avx));

			__m256 scale_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x3F800000));
			__m256 scale_s = _mm256_mul_ps(s, scale_MASK);

			__m256 r_avx = _mm256_add_ps(scale_s, b2_avx);
			_mm256_storeu_ps(r + i, r_avx);
		}
	}
	return;
}


void SCAN_R0_FLIP(float** LLR_L, float** LLR_R, int stage, int Count, int node, float* Ma_SC, int* uhat)
{
	memset(uhat + Count, 0, sizeof(int) * node);

	for (int i = 0; i < node; i++)
	{
		LLR_R[stage][Count + i] = HUGE_VALF;
	}
	for (int cc = 0; cc < node; cc++)
	{
		Ma_SC[Count + cc] = HUGE_VALF;
	}
}

void SCAN_R1_FLIP(float** LLR_L, float** LLR_R, int stage, int Count, int node, float* Ma_SC, float& theta, int Flip_bit, int* uhat)
{
	memset(LLR_R[stage] + Count, 0, sizeof(float) * node);
	for (int cc = 0; cc < node; cc++)
	{
		Ma_SC[Count + cc] = theta + qmthres(LLR_L[stage][Count + cc]);
	}
	for (int cc = 0; cc < node; cc++)
	{
		theta += qhthres(LLR_L[stage][Count + cc]);
	}
	if ((Flip_bit >= Count) && (Flip_bit < Count + node))
		LLR_R[stage][Flip_bit] = HUGE_VALF * (2 * qhard(LLR_L[stage][Flip_bit]) - 1);

	for (int i = 0; i < node; i++)
	{
		sumtemp[i] = qhard(LLR_L[stage][Count + i] + LLR_R[stage][Count + i]);
	}
	PolarEncode_xor(uhat + Count, sumtemp, node);
}

void SCAN_REP_FLIP(float** LLR_L, float** LLR_R, int stage, int Count, int node, float* Ma_SC, float& theta, int Flip_bit, int* uhat)
{
	float temp = 0;
	for (int i = 0; i < node; i++)
	{
		temp += LLR_L[stage][Count + i];
	}

	memset(uhat + Count, 0, sizeof(int) * node);
	if (Flip_bit == Count + node - 1)
	{
		if (temp < 0)
		{
			for (int i = 0; i < node; i++)
			{
				LLR_R[stage][Count + i] = HUGE_VALF;
			}
		}
		else
		{
			uhat[Count + node - 1] ^= 1;
			for (int i = 0; i < node; i++)
			{
				LLR_R[stage][Count + i] = -HUGE_VALF;
			}
		}
	}
	else
	{
		if (temp < 0) uhat[Count + node - 1] ^= 1;

		for (int i = 0; i < node; i++)
		{
			LLR_R[stage][Count + i] = temp - LLR_L[stage][Count + i];
		}
	}
	for (int cc = 0; cc < node - 1; cc++)
	{
		Ma_SC[Count + cc] = HUGE_VALF;
	}
	Ma_SC[Count + node - 1] = theta + qmthres(temp);
	theta += qhthres(temp);
}
void SCAN_SPC_FLIP(float** LLR_L, float** LLR_R, int stage, int Count, int node, float* Ma_SC, float& theta, int Flip_bit, int* uhat)
{
	int FS, SS; // FS: First_Smallest, SS: Second_Smallest
	if (abs(LLR_L[stage][Count]) > abs(LLR_L[stage][Count + 1]))
	{
		FS = 1; SS = 0;
	}
	else
	{
		FS = 0; SS = 1;
	}
	int parity = qhard(LLR_L[stage][Count]) ^ qhard(LLR_L[stage][Count + 1]);

	if (node > 2)
	{
		for (int i = 2; i < node; i++)
		{
			parity ^= qhard(LLR_L[stage][Count + i]);
			if (abs(LLR_L[stage][Count + i]) < abs(LLR_L[stage][Count + FS]))
			{
				SS = FS; FS = i;
			}
			else
			{
				if (abs(LLR_L[stage][Count + i]) < abs(LLR_L[stage][Count + SS]))
				{
					SS = i;
				}
			}
		}
	}
	for (int i = 0; i < node; i++)
	{
		if (i == FS)
		{
			LLR_R[stage][Count + i] = (1 - 2 * (parity ^ qhard(LLR_L[stage][Count + FS]))) * abs(LLR_L[stage][Count + SS]);
		}
		else
		{
			LLR_R[stage][Count + i] = (1 - 2 * (parity ^ qhard(LLR_L[stage][Count + i]))) * abs(LLR_L[stage][Count + FS]);
		}
	}

	// 以下 Flip 待优化
	parity = 0;
	int index = 0;
	float temp = abs(LLR_L[stage][Count] + LLR_R[stage][Count]);
	for (int i = 0; i < node; i++)
	{
		parity ^= qhard(LLR_L[stage][Count + i] + LLR_R[stage][Count + i]);
		if (temp > abs(LLR_L[stage][Count + i] + LLR_R[stage][Count + i]))
		{
			temp = abs(LLR_L[stage][Count + i] + LLR_R[stage][Count + i]);
			index = i;
		}
	}

	index = FS;
	for (int cc = 0; cc < node; cc++)
	{
		Ma_SC[Count + cc] = theta + qmthres(LLR_L[stage][Count + cc] + LLR_R[stage][Count + cc]) + (1 - parity) * qmthres(LLR_L[stage][Count + index] + LLR_R[stage][Count + index]);
	}
	Ma_SC[Count + index] = HUGE_VALF;
	for (int cc = 0; cc < node; cc++)
	{
		theta += qhthres(LLR_L[stage][Count + cc] + LLR_R[stage][Count + cc]);
	}
	theta += parity * abs(LLR_L[stage][Count + index] + LLR_R[stage][Count + index]);

	//if (parity == 1) LLR_R[stage][index] = HUGE_VALF * (2 * hard(LLR_L[stage][index] + LLR_R[stage][index]) - 1);

	if ((Flip_bit >= Count) && (Flip_bit < Count + node))
	{
		LLR_R[stage][Flip_bit] = HUGE_VALF * (2 * qhard(LLR_L[stage][Flip_bit] + LLR_R[stage][Flip_bit]) - 1);
		LLR_R[stage][Count + index] = HUGE_VALF * (2 * qhard(LLR_L[stage][Count + index] + LLR_R[stage][Count + index]) - 1);
	}
	for (int i = 0; i < node; i++)
	{
		sumtemp[i] = qhard(LLR_L[stage][Count + i] + LLR_R[stage][Count + i]);
	}
	PolarEncode_xor(uhat + Count, sumtemp, node);
}

void SCAN_R0_LIST(float** LLR, float** beta_R, int beta_start, int Count, int node, int l, int last_start, float* PM, int count_info)
{
	for (int k = 0; k < l; k++)
	{
		for (int i = 0; i < node; i += 8)
		{
			_mm256_storeu_ps(beta_R[k] + beta_start + i, _mm256_set1_ps(HUGE_VALF));
		}
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

void SCAN_REP_LIST(float** LLR, float** beta, int node, int& l, int beta_start, int last_start, float* PM, int n, int L, int** p, float* W, int* index, int* better, int* worse, int stage)
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

			for (int i = beta_start; i < beta_start + node; ++i) {
				beta[(k << 1)][i] = HUGE_VALF;
				beta[(k << 1) + 1][i] = -HUGE_VALF;
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
			//temp_01[(k << 1)] = 0;
			//temp_01[(k << 1) + 1] = 1;
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
			if (better[k] % 2)
			{
				for (int i = beta_start; i < beta_start + node; ++i)
					beta[k][i] = -HUGE_VALF;
			}
			else
			{
				for (int i = beta_start; i < beta_start + node; ++i)
					beta[k][i] = HUGE_VALF;
			}
			PM[k] = W[better[k]];
		}
	}
}

void SCAN_R1_LIST(float** LLR, float** beta, int node, int& l, int beta_start, int last_start, float* PM, int n, int L, int** p, float* W, int* index, int* better, int* worse, int* min_index, int stage)
{
	int sort_size = min(L - 1, node);
	sort_size = 1;

	for (int k = 0; k < L; ++k) {  //Path expansion and new path metric calculation
		memset(beta[k] + beta_start, 0, sizeof(float)*node);
		min_index[k] = 0;
		for (int i = 0; i < node; i++)
		{
			if (abs_f(*(*(LLR + k) + last_start + i)) < abs_f(*(*(LLR + k) + last_start + min_index[k])))
				min_index[k] = i;
		}
	}

	for (int k = 0; k < L; ++k) {
		W[(k << 1)] = PM[k];
		W[(k << 1) + 1] = PM[k] - abs_f(*(*(LLR + p[k][stage]) + last_start + min_index[p[k][stage]]));
		better[k] = (k << 1);
		worse[k] = (k << 1) + 1;
	}

	sort_list(better, worse, W, L);  // Path metric sorting
									 //fout << endl;
	for (int k = 0; k < L; ++k) {
		index[k] = (better[k] >> 1);
		if (index[k] != k) {
			for (int s = stage; s <= n; s++) {
				p[k][s] = p[index[k]][s];
			}
			//replace_sum(*(sum + index[k]) + last_start_sum, *(sum + k) + last_start_sum, node);
		}
	}

	for (int k = 0; k < L; ++k) {  //path updating
								   //beta[k][beta_start + min_index[p[k][stage]]] = (better[k] % 2) == 1 ? -HUGE_VALF : HUGE_VALF;
		int FS, SS; // FS: First_Smallest, SS: Second_Smallest
		if (abs(LLR[p[k][stage]][last_start]) > abs(LLR[p[k][stage]][last_start + 1]))
		{
			FS = 1; SS = 0;
		}
		else
		{
			FS = 0; SS = 1;
		}
		int parity = qhard(LLR[p[k][stage]][last_start]) ^ qhard(LLR[p[k][stage]][last_start + 1]);

		if (node > 2)
		{
			for (int i = 2; i < node; i++)
			{
				parity ^= qhard(LLR[p[k][stage]][last_start + i]);
				if (abs(LLR[p[k][stage]][last_start + i]) < abs(LLR[p[k][stage]][last_start + FS]))
				{
					SS = FS; FS = i;
				}
				else
				{
					if (abs(LLR[p[k][stage]][last_start + i]) < abs(LLR[p[k][stage]][last_start + SS]))
					{
						SS = i;
					}
				}
			}
		}
		for (int i = 0; i < node; i++)
		{
			if (i == FS)
			{
				beta[k][beta_start + i] = (1 - 2 * (better[k] % 2))*(1 - 2 * parity) * (1 - 2 * (parity^qhard(LLR[p[k][stage]][last_start + FS])))*abs(LLR[p[k][stage]][last_start + SS]);
			}
			else
			{
				beta[k][beta_start + i] = (1 - 2 * (better[k] % 2))*(1 - 2 * parity) * (1 - 2 * (parity^qhard(LLR[p[k][stage]][last_start + i])))*abs(LLR[p[k][stage]][last_start + FS]);
			}
		}

		PM[k] = W[better[k]];
	}
}

void SCAN_R1_LIST_2(float** LLR, float** beta, int node, int& l, int beta_start, int last_start, float* PM, int n, int L, int** p, float* W, int* index, int* better, int* worse, int* path_number, int stage)
{
	int sort_size = min(L - 1, node);
	sort_size = 2;

	for (int k = 0; k < L; ++k) {  //Path expansion and new path metric calculation
		for (int i = 0; i < node; i++)
		{
			NLLR_R1[i] = abs_f(*(*(LLR + k) + last_start + i));
		}
		memset(beta[k] + beta_start, 0, sizeof(float)*node);
		min0_index[k] = 0;
		min1_index[k] = 1;
		for (int i = 0; i < node / 2; i++)
		{
			if (NLLR_R1[i * 2] < NLLR_R1[min0_index[k]])
			{
				min0_index[k] = i * 2;
			}
			if (NLLR_R1[i * 2 + 1] < NLLR_R1[min1_index[k]])
			{
				min1_index[k] = i * 2 + 1;
			}
		}
	}

	for (int k = 0; k < L; ++k) {
		W_R1[k] = PM[k];
		W_R1[L + k] = PM[k] - min(abs(LLR[p[k][stage]][last_start + min1_index[p[k][stage]]]), abs(LLR[p[k][stage]][last_start + min0_index[p[k][stage]]]));
		W_R1[L + L + k] = PM[k] - abs(LLR[p[k][stage]][last_start + min0_index[p[k][stage]]]) - abs(LLR[p[k][stage]][last_start + min1_index[p[k][stage]]]);
		W_R1[L + L + L + k] = PM[k] - max(abs(LLR[p[k][stage]][last_start + min1_index[p[k][stage]]]), abs(LLR[p[k][stage]][last_start + min0_index[p[k][stage]]]));
	}

	int n_can = (L << 2);

	for (int k = 0; k < n_can; k++)
	{
		ppp[k] = k;
	}
	int cnt = 0;
	while (cnt < L)
	{
		int p_index = min_index_sse(W_R1, L);
		int q_index = max_index_sse(W_R1 + L, 3 * L);
		//cout << " cnt = " << cnt << " p_index = " << p_index << " q_index = " << q_index + L << endl;
		if (W_R1[p_index] < W_R1[q_index + L])
		{
			fpath_0[cnt] = p_index;
			fpath_2[cnt] = q_index + L;
			fWpath_0[cnt] = W_R1[p_index];
			fWpath_2[cnt] = W_R1[q_index + L];
			W_R1[p_index] = 0;
			W_R1[q_index + L] = -HUGE_VALF;
			cnt++;
		}
		else
			break;
	}
	for (int k = 0; k < cnt; k++)
	{
		//cout << " cnt = " << k << " " << path_0[k] << " " << path_2[k] << endl;
		ppp[fpath_0[k]] = fpath_2[k];
		ppp[fpath_2[k]] = fpath_0[k];
		W_R1[fpath_0[k]] = fWpath_0[k];
		W_R1[fpath_2[k]] = fWpath_2[k];
	}
	for (int i = 0; i < L; i++)
	{
		index[i] = ppp[i] % L;
	}

	for (int k = 0; k < L; ++k) {
		if (index[k] != k) {
			for (int s = stage; s <= n; s++) {
				p[k][s] = p[index[k]][s];
			}
		}
	}

	for (int k = 0; k < L; ++k) {  //path updating
		int pc0, pc1;
		int pco, pce;
		if (node == 2)
		{
			pc0 = qhard(LLR[p[k][stage]][last_start]) ^ qhard(LLR[p[k][stage]][last_start + 1]);
			pc0 = (ppp[k] / L) % 2 == 0 ? pc0 : 1 - pc0;
			pc1 = pc0 == 0 ? qhard(LLR[p[k][stage]][last_start] + LLR[p[k][stage]][last_start + 1]) : qhard(LLR[p[k][stage]][last_start + 1] - LLR[p[k][stage]][last_start]);
			pc1 = (ppp[k] / L / 2) == 0 ? pc1 : 1 - pc1;

			beta[k][beta_start] = (1 - 2 * (pc0 ^ pc1)) * HUGE_VALF;
			beta[k][beta_start + 1] = (1 - 2 * pc1) * HUGE_VALF;
		}

		else
		{
			int FS_0, SS_0, FS_1, SS_1; // FS: First_Smallest, SS: Second_Smallest
			if (abs(LLR[p[k][stage]][last_start]) > abs(LLR[p[k][stage]][last_start + 2]))
			{
				FS_0 = 2; SS_0 = 0;
			}
			else
			{
				FS_0 = 0; SS_0 = 2;
			}
			if (abs(LLR[p[k][stage]][last_start + 1]) > abs(LLR[p[k][stage]][last_start + 3]))
			{
				FS_1 = 3; SS_1 = 1;
			}
			else
			{
				FS_1 = 1; SS_1 = 3;
			}
			pce = qhard(LLR[p[k][stage]][last_start]) ^ qhard(LLR[p[k][stage]][last_start + 2]);
			pco = qhard(LLR[p[k][stage]][last_start + 1]) ^ qhard(LLR[p[k][stage]][last_start + 3]);

			if (node > 4)
			{
				for (int i = 2; i < node / 2; i++)
				{
					pce ^= qhard(LLR[p[k][stage]][last_start + 2 * i]);
					pco ^= qhard(LLR[p[k][stage]][last_start + 2 * i + 1]);

					if (abs(LLR[p[k][stage]][last_start + 2 * i]) < abs(LLR[p[k][stage]][last_start + FS_0]))
					{
						SS_0 = FS_0; FS_0 = 2 * i;
					}
					else
					{
						if (abs(LLR[p[k][stage]][last_start + 2 * i]) < abs(LLR[p[k][stage]][last_start + SS_0]))
						{
							SS_0 = 2 * i;
						}
					}
					if (abs(LLR[p[k][stage]][last_start + 2 * i + 1]) < abs(LLR[p[k][stage]][last_start + FS_1]))
					{
						SS_1 = FS_1; FS_1 = 2 * i + 1;
					}
					else
					{
						if (abs(LLR[p[k][stage]][last_start + 2 * i + 1]) < abs(LLR[p[k][stage]][last_start + SS_1]))
						{
							SS_1 = 2 * i + 1;
						}
					}
				}
			}
			pc0 = pco ^ pce;
			pc0 = (ppp[k] / L) % 2 == 0 ? pc0 : 1 - pc0;
			pc1 = pc0 == 0 ? qhard((1 - 2 * pce)*abs(LLR[p[k][stage]][last_start + FS_0]) + (1 - 2 * pco)*abs(LLR[p[k][stage]][last_start + FS_1])) : qhard(-1 * (1 - 2 * pce)*abs(LLR[p[k][stage]][last_start + FS_0]) + (1 - 2 * pco)*abs(LLR[p[k][stage]][last_start + FS_1]));
			pc1 = (ppp[k] / L / 2) == 0 ? pc1 : 1 - pc1;

			for (int i = 0; i < node / 2; i++)
			{
				if (2 * i == FS_0)
				{
					beta[k][beta_start + 2 * i] = (1 - 2 * (pc0 ^ pc1))*(1 - 2 * (pce^qhard(LLR[p[k][stage]][last_start + FS_0])))*abs(LLR[p[k][stage]][last_start + SS_0]);
				}
				else
				{
					beta[k][beta_start + 2 * i] = (1 - 2 * (pc0 ^ pc1))*(1 - 2 * (pce^qhard(LLR[p[k][stage]][last_start + 2 * i])))*abs(LLR[p[k][stage]][last_start + FS_0]);
				}
				if (2 * i + 1 == FS_1)
				{
					beta[k][beta_start + 2 * i + 1] = (1 - 2 * pc1)*(1 - 2 * (pco^qhard(LLR[p[k][stage]][last_start + FS_1])))*abs(LLR[p[k][stage]][last_start + SS_1]);
				}
				else
				{
					beta[k][beta_start + 2 * i + 1] = (1 - 2 * pc1)*(1 - 2 * (pco^qhard(LLR[p[k][stage]][last_start + 2 * i + 1])))*abs(LLR[p[k][stage]][last_start + FS_1]);
				}
			}

		}
		PM[k] = W_R1[ppp[k]];
	}
}