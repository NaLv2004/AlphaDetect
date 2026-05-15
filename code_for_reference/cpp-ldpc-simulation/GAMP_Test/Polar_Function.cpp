#include <xmmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
using namespace std;
float abs_f(float a)
{
	return a < 0 ? -a : a;
}

float max_f(float a, float b)
{
	return a > b ? a : b;
}
float min_f(float a, float b)
{
	return a < b ? a : b;
}
int sgn(float a)
{
	if (a > 0)
		return 1;
	else if (a < 0)
		return -1;
	else return 0;
}
void BP_function_OL1(float* r, float* a, float* b1, float* b2, int size)
{
	if (size <= 4)
	{
		__m128 a_sse = _mm_loadu_ps(a);//
		__m128 b1_sse = _mm_loadu_ps(b1);
		__m128 b2_sse = _mm_loadu_ps(b2);
		__m128 b_sse = _mm_add_ps(b1_sse, b2_sse);
		__m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
		__m128 sign = _mm_and_ps(_mm_xor_ps(a_sse, b_sse), SIGN_MASK);
		__m128 abs_a_sse = _mm_andnot_ps(SIGN_MASK, a_sse);
		__m128 abs_b_sse = _mm_andnot_ps(SIGN_MASK, b_sse);
		__m128 max_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x00000000)); // W_20
		__m128 max_s = _mm_max_ps(_mm_sub_ps(_mm_min_ps(abs_a_sse, abs_b_sse), max_MASK), _mm_castsi128_ps(_mm_set1_epi32(0x00000000)));
		__m128 s = _mm_or_ps(sign, max_s);
		__m128 scale_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x3f800000)); // W_10
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
			__m256 max_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x00000000)); // W_20
			__m256 max_s = _mm256_max_ps(_mm256_sub_ps(_mm256_min_ps(abs_a_avx, abs_b_avx), max_MASK), _mm256_castsi256_ps(_mm256_set1_epi32(0x00000000)));
			__m256 s = _mm256_or_ps(sign, max_s);

			__m256 scale_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000)); // W_10
			__m256 scale_s = _mm256_mul_ps(s, scale_MASK);
			_mm256_storeu_ps(r + i, scale_s);
		}
	}
	return;
}
void BP_function_OR1(float* r, float* a, float* b1, float* b2, int size)
{
	if (size <= 4)
	{
		__m128 a_sse = _mm_loadu_ps(a);//
		__m128 b1_sse = _mm_loadu_ps(b1);
		__m128 b2_sse = _mm_loadu_ps(b2);
		__m128 b_sse = _mm_add_ps(b1_sse, b2_sse);
		__m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
		__m128 sign = _mm_and_ps(_mm_xor_ps(a_sse, b_sse), SIGN_MASK);
		__m128 abs_a_sse = _mm_andnot_ps(SIGN_MASK, a_sse);
		__m128 abs_b_sse = _mm_andnot_ps(SIGN_MASK, b_sse);
		__m128 max_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x3e800000)); // W_40
		__m128 max_s = _mm_max_ps(_mm_sub_ps(_mm_min_ps(abs_a_sse, abs_b_sse), max_MASK), _mm_castsi128_ps(_mm_set1_epi32(0x00000000)));
		__m128 s = _mm_or_ps(sign, max_s);
		__m128 scale_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x3f800000)); // W_30
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
			__m256 max_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x3e800000)); // W_40
			__m256 max_s = _mm256_max_ps(_mm256_sub_ps(_mm256_min_ps(abs_a_avx, abs_b_avx), max_MASK), _mm256_castsi256_ps(_mm256_set1_epi32(0x00000000)));
			__m256 s = _mm256_or_ps(sign, max_s);

			__m256 scale_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000)); // W_30
			__m256 scale_s = _mm256_mul_ps(s, scale_MASK);
			_mm256_storeu_ps(r + i, scale_s);
		}
	}
	return;
}
void BP_function_OL2(float* r, float* a, float* b1, float* b2, int size)
{
	if (size <= 4)
	{
		__m128 a_sse = _mm_loadu_ps(a);//
		__m128 b1_sse = _mm_loadu_ps(b1);
		__m128 b2_sse = _mm_loadu_ps(b2);

		__m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
		__m128 sign = _mm_and_ps(_mm_xor_ps(a_sse, b1_sse), SIGN_MASK);
		__m128 abs_a_sse = _mm_andnot_ps(SIGN_MASK, a_sse);
		__m128 abs_b1_sse = _mm_andnot_ps(SIGN_MASK, b1_sse);
		__m128 max_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x00000000)); // W_21
		__m128 max_s = _mm_max_ps(_mm_sub_ps(_mm_min_ps(abs_a_sse, abs_b1_sse), max_MASK), _mm_castsi128_ps(_mm_set1_epi32(0x00000000)));

		__m128 s = _mm_or_ps(sign, max_s);
		__m128 scale_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x3f800000)); // W_11
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
			__m256 max_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x00000000)); // W_21
			__m256 max_s = _mm256_max_ps(_mm256_sub_ps(_mm256_min_ps(abs_a_avx, abs_b1_avx), max_MASK), _mm256_castsi256_ps(_mm256_set1_epi32(0x00000000)));

			__m256 s = _mm256_or_ps(sign, max_s);

			__m256 scale_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000)); // W_11
			__m256 scale_s = _mm256_mul_ps(s, scale_MASK);

			__m256 r_avx = _mm256_add_ps(scale_s, b2_avx);
			_mm256_storeu_ps(r + i, r_avx);
		}
	}
	return;
}
void BP_function_OR2(float* r, float* a, float* b1, float* b2, int size)
{
	if (size <= 4)
	{
		__m128 a_sse = _mm_loadu_ps(a);//
		__m128 b1_sse = _mm_loadu_ps(b1);
		__m128 b2_sse = _mm_loadu_ps(b2);

		__m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
		__m128 sign = _mm_and_ps(_mm_xor_ps(a_sse, b1_sse), SIGN_MASK);
		__m128 abs_a_sse = _mm_andnot_ps(SIGN_MASK, a_sse);
		__m128 abs_b1_sse = _mm_andnot_ps(SIGN_MASK, b1_sse);
		__m128 max_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x3e800000)); // W_41
		__m128 max_s = _mm_max_ps(_mm_sub_ps(_mm_min_ps(abs_a_sse, abs_b1_sse), max_MASK), _mm_castsi128_ps(_mm_set1_epi32(0x00000000)));

		__m128 s = _mm_or_ps(sign, max_s);
		__m128 scale_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x3f800000)); // W_31
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
			__m256 max_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x3e800000)); // W_41
			__m256 max_s = _mm256_max_ps(_mm256_sub_ps(_mm256_min_ps(abs_a_avx, abs_b1_avx), max_MASK), _mm256_castsi256_ps(_mm256_set1_epi32(0x00000000)));

			__m256 s = _mm256_or_ps(sign, max_s);

			__m256 scale_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000)); // W_31
			__m256 scale_s = _mm256_mul_ps(s, scale_MASK);

			__m256 r_avx = _mm256_add_ps(scale_s, b2_avx);
			_mm256_storeu_ps(r + i, r_avx);
		}
	}
	return;
}

void BP_function(float* r, float* a, float* b1, float* b2, int size)
{
	if (size < 4)
	{
		for (int i = 0; i < size; i++)
		{
			r[i] = 1 * sgn(a[i]) * sgn(b1[i] + b2[i]) * min_f(abs_f(a[i]), abs_f(b1[i] + b2[i]));
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
void BP_function2(float* r, float* a, float* b1, float* b2, int size)
{
	if (size < 4)
	{
		for (int i = 0; i < size; i++)
		{
			r[i] = b2[i] + 1 * sgn(a[i]) * sgn(b1[i]) * min_f(abs_f(a[i]), abs_f(b1[i]));
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

void add_hard_SIMD(float* a1, float* a2, int* b, int size)
{
	switch (size) {
	case 2:
	{
		*b = (*a1 + *a2) > 0 ? 0 : 1;
		*(b + 1) = (*(a1 + 1) + *(a2 + 1)) > 0 ? 0 : 1;
	}
	case 4:
	{
		__m128i* ptr_b_128 = (__m128i*) b;
		for (int i = 0; i < size / 4; ++i)
		{
			__m128 a1_128 = _mm_loadu_ps(a1 + (i << 2));
			__m128 a2_128 = _mm_loadu_ps(a2 + (i << 2));

			__m128 llr_128 = _mm_add_ps(a1_128, a2_128);
			__m128 s = _mm_cmpge_ps(_mm_setzero_ps(), llr_128);
			__m128i mask = _mm_set1_epi32(0x00000001);
			__m128i sum_128 = _mm_and_si128(mask, _mm_castps_si128(s));
			_mm_storeu_si128(ptr_b_128 + i, sum_128);
		}

	}
	default:
	{
		__m256i* ptr_b_256 = (__m256i*) b;
		for (int i = 0; i < size / 8; ++i)
		{
			__m256 a1_256 = _mm256_loadu_ps(a1 + (i << 3));
			__m256 a2_256 = _mm256_loadu_ps(a2 + (i << 3));

			__m256 llr_256 = _mm256_add_ps(a1_256, a2_256);
			__m256 s = _mm256_cmp_ps(_mm256_setzero_ps(), llr_256, 14);
			__m256i mask = _mm256_set1_epi32(0x00000001);
			__m256i sum_256 = _mm256_and_si256(mask, _mm256_castps_si256(s));
			_mm256_storeu_si256(ptr_b_256 + i, sum_256);
		}
	}
	}
}

void f_function(float* a, int size)
{
	if (size <= 4) {
		for (int i = 0; i < size; i += 4)
		{
			__m128 a_left = _mm_loadu_ps(a + i);//
			__m128 a_right = _mm_loadu_ps(a + i + size);
			__m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
			__m128 sign = _mm_and_ps(_mm_xor_ps(a_left, a_right), SIGN_MASK);
			__m128 abs_a_left = _mm_andnot_ps(SIGN_MASK, a_left);
			__m128 abs_a_right = _mm_andnot_ps(SIGN_MASK, a_right);
			__m128 s = _mm_or_ps(sign, _mm_min_ps(abs_a_left, abs_a_right));
			_mm_storeu_ps(a + i + (size << 1), s);
		}
	}
	else {
		for (int i = 0; i < size; i += 8)
		{
			__m256 a_left = _mm256_loadu_ps(a + i);
			__m256 a_right = _mm256_loadu_ps(a + i + size);
			__m256 SIGN_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
			__m256 sign = _mm256_and_ps(_mm256_xor_ps(a_left, a_right), SIGN_MASK);
			__m256 abs_a_left = _mm256_andnot_ps(SIGN_MASK, a_left);
			__m256 abs_a_right = _mm256_andnot_ps(SIGN_MASK, a_right);
			__m256 s = _mm256_or_ps(sign, _mm256_min_ps(abs_a_left, abs_a_right));
			_mm256_storeu_ps(a + i + (size << 1), s);
		}
	}
}

void g_function(float* a, int* b, int size)
{
	if (size <= 4) {
		for (int i = 0; i < size; i += 4)
		{
			__m128 a_left = _mm_loadu_ps(a + i);
			__m128 a_right = _mm_loadu_ps(a + i + size);
			__m128 b_in = _mm_loadu_ps((float*)(b + i));
			__m128 MASK_0 = _mm_castsi128_ps(_mm_set1_epi32(0x00000000));
			__m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
			__m128 sign = _mm_and_ps(_mm_cmpneq_ps(b_in, MASK_0), SIGN_MASK);
			__m128 s = _mm_add_ps(a_right, _mm_xor_ps(sign, a_left));
			_mm_storeu_ps(a + i + (size << 1), s);
		}
	}
	else {
		for (int i = 0; i < size; i += 8)
		{
			__m256 a_left = _mm256_loadu_ps(a + i);
			__m256 a_right = _mm256_loadu_ps(a + i + size);
			__m256 b_in = _mm256_loadu_ps((float*)(b + i));
			__m256 MASK_0 = _mm256_castsi256_ps(_mm256_set1_epi32(0x00000000));
			__m256 SIGN_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
			__m256 sign = _mm256_and_ps(_mm256_cmp_ps(b_in, MASK_0, 0), SIGN_MASK);
			__m256 s = _mm256_sub_ps(a_right, _mm256_xor_ps(sign, a_left));
			_mm256_storeu_ps(a + i + (size << 1), s);
		}
	}
}
void f_function_index(float* a, float* a_new, int size)
{
	if (size <= 4) {
		for (int i = 0; i < size; i += 4)
		{
			__m128 a_left = _mm_loadu_ps(a + i);//
			__m128 a_right = _mm_loadu_ps(a + i + size);
			__m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
			__m128 sign = _mm_and_ps(_mm_xor_ps(a_left, a_right), SIGN_MASK);
			__m128 abs_a_left = _mm_andnot_ps(SIGN_MASK, a_left);
			__m128 abs_a_right = _mm_andnot_ps(SIGN_MASK, a_right);
			__m128 s = _mm_or_ps(sign, _mm_min_ps(abs_a_left, abs_a_right));
			_mm_storeu_ps(a_new + i + (size << 1), s);
		}
	}
	else {
		for (int i = 0; i < size; i += 8)
		{
			__m256 a_left = _mm256_loadu_ps(a + i);
			__m256 a_right = _mm256_loadu_ps(a + i + size);
			__m256 SIGN_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
			__m256 sign = _mm256_and_ps(_mm256_xor_ps(a_left, a_right), SIGN_MASK);
			__m256 abs_a_left = _mm256_andnot_ps(SIGN_MASK, a_left);
			__m256 abs_a_right = _mm256_andnot_ps(SIGN_MASK, a_right);
			__m256 s = _mm256_or_ps(sign, _mm256_min_ps(abs_a_left, abs_a_right));
			_mm256_storeu_ps(a_new + i + (size << 1), s);
		}
	}
}
void g_function_index(float* a, int* b, float* a_new, int size)
{
	if (size <= 4) {
		for (int i = 0; i < size; i += 4)
		{
			__m128 a_left = _mm_loadu_ps(a + i);
			__m128 a_right = _mm_loadu_ps(a + i + size);
			__m128 b_in = _mm_loadu_ps((float*)(b + i));
			__m128 MASK_0 = _mm_castsi128_ps(_mm_set1_epi32(0x00000000));
			__m128 SIGN_MASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
			__m128 sign = _mm_and_ps(_mm_cmpneq_ps(b_in, MASK_0), SIGN_MASK);
			__m128 s = _mm_add_ps(a_right, _mm_xor_ps(sign, a_left));
			_mm_storeu_ps(a_new + i + (size << 1), s);
		}
	}
	else {
		for (int i = 0; i < size; i += 8)
		{
			__m256 a_left = _mm256_loadu_ps(a + i);
			__m256 a_right = _mm256_loadu_ps(a + i + size);
			__m256 b_in = _mm256_loadu_ps((float*)(b + i));
			__m256 MASK_0 = _mm256_castsi256_ps(_mm256_set1_epi32(0x00000000));
			__m256 SIGN_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
			__m256 sign = _mm256_and_ps(_mm256_cmp_ps(b_in, MASK_0, 0), SIGN_MASK);
			__m256 s = _mm256_sub_ps(a_right, _mm256_xor_ps(sign, a_left));
			_mm256_storeu_ps(a_new + i + (size << 1), s);
		}
	}
}

void combine(int* a, int* b, int size)
{
	if (size <= 4) {
		__m128i* ptr_a_128 = (__m128i*) a;
		__m128i* ptr_b_128 = (__m128i*) b;
		for (int i = 0; i < size / 4; ++i)
		{
			__m128i a_128 = _mm_loadu_si128(ptr_a_128 + i);
			__m128i b_128 = _mm_loadu_si128(ptr_b_128 + i);
			__m128i s_128 = _mm_xor_si128(a_128, b_128);
			_mm_storeu_si128(ptr_a_128 + i, s_128);
		}
	}
	else {
		__m256i* ptr_a_256 = (__m256i*) a;
		__m256i* ptr_b_256 = (__m256i*) b;
		for (int i = 0; i < size / 8; ++i)
		{
			__m256i a_256 = _mm256_loadu_si256(ptr_a_256 + i);
			__m256i b_256 = _mm256_loadu_si256(ptr_b_256 + i);
			__m256i s_256 = _mm256_xor_si256(a_256, b_256);
			_mm256_storeu_si256(ptr_a_256 + i, s_256);
		}
	}
}
void combine_index(int* a, int* b, int* a_new, int size)
{
	if (size <= 4) {
		__m128i* ptr_a_128 = (__m128i*) a;
		__m128i* ptr_b_128 = (__m128i*) b;
		__m128i* ptr_a_new_128 = (__m128i*) a_new;
		for (int i = 0; i < size / 4; i++)
		{
			__m128i a_128 = _mm_loadu_si128(ptr_a_128 + i);
			__m128i b_128 = _mm_loadu_si128(ptr_b_128 + i);
			_mm_storeu_si128(ptr_a_new_128 + i + (size / 4), b_128);
			__m128i s_128 = _mm_xor_si128(a_128, b_128);
			_mm_storeu_si128(ptr_a_new_128 + i, s_128);
		}
	}
	else {
		__m256i* ptr_a_256 = (__m256i*) a;
		__m256i* ptr_b_256 = (__m256i*) b;
		__m256i* ptr_a_new_256 = (__m256i*) a_new;
		for (int i = 0; i < size / 8; i++)
		{
			__m256i a_256 = _mm256_loadu_si256(ptr_a_256 + i);
			__m256i b_256 = _mm256_loadu_si256(ptr_b_256 + i);
			_mm256_storeu_si256(ptr_a_new_256 + i + (size / 8), b_256);
			__m256i s_256 = _mm256_xor_si256(a_256, b_256);
			_mm256_storeu_si256(ptr_a_new_256 + i, s_256);
		}
	}
}
void hard_SIMD(float* a, int* b, int size)
{
	switch (size) {
	case 2:
	{
		*b = *a > 0 ? 0 : 1;
		*(b + 1) = *(a + 1) > 0 ? 0 : 1;
	}
	case 4:
	{
		__m128i* ptr_b_128 = (__m128i*) b;
		for (int i = 0; i < size / 4; ++i)
		{
			__m128 llr_128 = _mm_loadu_ps(a + (i << 2));
			__m128 s = _mm_cmpge_ps(_mm_setzero_ps(), llr_128);
			__m128i mask = _mm_set1_epi32(0x00000001);
			__m128i sum_128 = _mm_and_si128(mask, _mm_castps_si128(s));
			_mm_storeu_si128(ptr_b_128 + i, sum_128);
		}

	}
	default:
	{
		__m256i* ptr_b_256 = (__m256i*) b;
		for (int i = 0; i < size / 8; ++i)
		{
			__m256 llr_256 = _mm256_loadu_ps(a + (i << 3));
			__m256 s = _mm256_cmp_ps(_mm256_setzero_ps(), llr_256, 14);
			__m256i mask = _mm256_set1_epi32(0x00000001);
			__m256i sum_256 = _mm256_and_si256(mask, _mm256_castps_si256(s));
			_mm256_storeu_si256(ptr_b_256 + i, sum_256);
		}
	}
	}
}

float cal_sum(float* a, int len)
{
	float temp = 0;
	if (len <= 4) {
		int cnt = (len >> 2) - 1;
		__m128 s = _mm_loadu_ps(a);
		__m128 llr_128 = _mm_loadu_ps(a);
		s = _mm_add_ps(s, llr_128);
		s = _mm_hadd_ps(s, s);
		s = _mm_hadd_ps(s, s);
		temp += s.m128_f32[0];
	}
	else
	{
		int cnt = len >> 3;
		__m256 s = _mm256_loadu_ps(a);
		for (int i = 1; i < cnt; i++)
		{
			a += 8;
			__m256 llr_256 = _mm256_loadu_ps(a);
			s = _mm256_add_ps(s, llr_256);
		}
		s = _mm256_hadd_ps(s, s);
		s = _mm256_hadd_ps(s, s);
		temp += s.m256_f32[0];
		temp += s.m256_f32[5];
	}
	return temp;
}

void replace_sum(int* a, int* b, int len) {
	if (len < 8) {
		int cntBlock = len >> 2;
		int cntRem = len & 3;
		__m128i* p_new = (__m128i*)a;
		__m128i* p_old = (__m128i*)b;
		__m128i xidLoad;
		for (int i = 0; i < cntRem; ++i)
			*(b + len - i - 1) = *(a + len - i - 1);
		for (int i = 0; i < cntBlock; ++i)
		{
			xidLoad = _mm_loadu_si128(p_new);
			_mm_storeu_si128(p_old, xidLoad);
			p_old++; p_new++;
		}

	}
	else {
		int cntBlock = len >> 3;
		int cntRem = len & 7;
		__m256i* p_new = (__m256i*)a;
		__m256i* p_old = (__m256i*)b;
		__m256i xidLoad;
		for (int i = 0; i < cntRem; ++i)
			*(b + len - i - 1) = *(a + len - i - 1);
		for (int i = 0; i < cntBlock; ++i)
		{
			xidLoad = _mm256_loadu_si256(p_new);
			_mm256_storeu_si256(p_old, xidLoad);
			p_old++; p_new++;
		}

	}
}
void replace_LLR(float* a, float* b, int len) {
	if (len < 8) {
		int cntBlock = len >> 2;
		int cntRem = len & 3;
		__m128 xidLoad;
		for (int i = 0; i < cntBlock; ++i)
		{
			xidLoad = _mm_loadu_ps(a);
			_mm_storeu_ps(b, xidLoad);
			a += 4; b += 4;
		}
		for (int i = 0; i < cntRem; ++i)
			*(b - i - 1) = *(a - i - 1);
	}
	else {
		int cntBlock = len >> 3;
		int cntRem = len & 7;
		__m256 xidLoad;
		for (int i = 0; i < cntBlock; ++i)
		{
			xidLoad = _mm256_loadu_ps(a);
			_mm256_storeu_ps(b, xidLoad);
			a += 8; b += 8;
		}
		for (int i = 0; i < cntRem; ++i) {
			*(b - i - 1) = *(a - i - 1);
		}

	}
}
void set_PM(float** a, float* b, int L, int len)
{
	if (len == 2) {
		b[0] = *(*(a)+len);
		b[1] = *(*(a + 1) + len);
	}
	else if (len == 4) {
		__m128 xidLoad;
		__m128 ZERO = _mm_setzero_ps();
		for (int i = 0; i < L; i += 4)
		{
			xidLoad = _mm_setr_ps(*(*(a + i) + len), *(*(a + i + 1) + len), *(*(a + i + 2) + len), *(*(a + i + 3) + len));
			_mm_storeu_ps(b + i, _mm_add_ps(_mm_loadu_ps(b + i), _mm_min_ps(xidLoad, ZERO)));
		}
	}
	else
	{
		__m256 xidLoad;
		__m256 ZERO = _mm256_setzero_ps();
		for (int i = 0; i < L; i += 8)
		{
			xidLoad = _mm256_setr_ps(*(*(a + i) + len), *(*(a + i + 1) + len), *(*(a + i + 2) + len), *(*(a + i + 3) + len), *(*(a + i + 4) + len), *(*(a + i + 5) + len), *(*(a + i + 6) + len), *(*(a + i + 7) + len));
			_mm256_storeu_ps(b + i, _mm256_add_ps(_mm256_loadu_ps(b + i), _mm256_min_ps(xidLoad, ZERO)));
		}
	}
}
int index_posit(int a, int b)
{
	int c;
	int temp = 0;
	c = a ^ b;
	if (c == 0) return 0;
	else {
		while (c != 1) {
			c >>= 1;
			temp++;
		}
		return temp;
	}
}
int lower_posit(int a)
{
	int c = a + 1;
	int temp = 0;
	while ((c & 1) == 0) {
		c >>= 1;
		temp++;
	}
	return temp;
}

