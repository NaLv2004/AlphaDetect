#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cmath>
#include <bitset>
#include <algorithm>
#include <vector>
#include <limits>
#include <fstream>
#include <xmmintrin.h>
#include <immintrin.h>

using namespace std;

#define SIGN(n) (n==0?0:(n/abs(n)))

ofstream aout("POLAR_CONS.txt");
double qfunc(double x) {
	double y = (double)erfc((long double)x / sqrt(2)) / 2;
	return y;
}
double P_less(double x, double u, double s)
{
	double P = 1 - qfunc((x - u) / s);
	return P;
}
double P_large(double x, double u, double s)
{
	double P = qfunc((x - u) / s);
	return P;
}
double normal_pdf(double x, double u, double s)
{
	double P = (1 / s / sqrt(2 * 3.14159265358979)) * exp(-(x - u) * (x - u) / (2 * s * s));
	return P;
}
double min_GA(double x, double u1, double s1, double u2, double s2)
{
	double P;
	if (x >= 0) {
		P = P_less(0, u1, s1) * P_large(0, u2, s2) + P_large(0, u1, s1) * P_less(0, u2, s2) + P_large(0, u1, s1) * P_large(0, u2, s2) - P_large(x, u1, s1) * P_large(x, u2, s2) + P_less(0, u1, s1) * P_less(0, u2, s2) - P_less(-x, u1, s1) * P_less(-x, u2, s2);
	}
	else
	{
		P = P_less(x, u1, s1) * P_large(-x, u2, s2) + P_large(-x, u1, s1) * P_less(x, u2, s2);
	}
	return P;
}
double min_pdf(double x, double u1, double s1, double u2, double s2)
{
	double P;
	if (x >= 0) {
		P = normal_pdf(x, u1, s1) * P_large(x, u2, s2) + normal_pdf(-x, u2, s2) * P_less(-x, u1, s1) + normal_pdf(-x, u1, s1) * P_less(-x, u2, s2) + normal_pdf(x, u2, s2) * P_large(x, u1, s1);
	}
	else
	{
		P = normal_pdf(x, u1, s1) * P_large(-x, u2, s2) + normal_pdf(-x, u2, s2) * P_less(x, u1, s1) + normal_pdf(-x, u1, s1) * P_less(x, u2, s2) + normal_pdf(x, u2, s2) * P_large(-x, u1, s1);
	}
	return P;
}
double solve_a(double t1, double t2, double e, double u1, double s1, double u2, double s2)
{
	double tave = (t1 + t2) / 2.0;
	double y;
	double f3, f1;
	f3 = min_GA(tave, u1, s1, u2, s2) - 0.5;
	f1 = min_GA(t1, u1, s1, u2, s2) - 0.5;
	//printf("%s %2.20f %s %2.20f %s %2.20f ", " t1 = ", t1, " t2 = ", t2, " tave = ", tave);
	//cout << " f1 = " << f1 << " f3 = " << f3 << endl;
	if ((tave == t2) || (tave == t1)) return tave;
	if ((f1 >= 0) && (f3 < 0) || (f1 < 0) && (f3 >= 0))
	{
		double m = tave - t1;
		//cout << " delta = " << m << endl;
		//printf("%s %2.20f ", "delta = ", m);
		if (m > e)
		{
			t2 = tave;
			//printf("%s %2.20f \n\n", "tave = ", tave);
			return solve_a(t1, t2, e, u1, s1, u2, s2);
		}
		else
		{
			return tave;
		}
	}
	else
	{
		double m = t2 - tave;
		if (m > e)
		{
			t1 = tave;
			return solve_a(t1, t2, e, u1, s1, u2, s2);
		}
		else
		{
			return tave;
		}
	}
}
void cal_E(double u1, double s1, double u2, double s2, double& u3, double& s3)
{
	double e = 1e-200;//DBL_MIN;
	double t1 = -1000000;
	double t2 = 1000000;
	if ((u1 == HUGE_VAL) && (s1 == HUGE_VAL))
	{
		u3 = u2;
		s3 = s2;
	}
	else if ((u2 == HUGE_VAL) && (s2 == HUGE_VAL))
	{
		u3 = u1;
		s3 = s1;
	}
	else if ((u1 == 0) && (s1 == 0))
	{
		u3 = 0;
		s3 = 0;
	}
	else if ((u2 == 0) && (s2 == 0))
	{
		u3 = 0;
		s3 = 0;
	}
	else {
		if (min_GA(0, u1, s1, u2, s2) == 0.5)
		{
			u3 = 0;
		}
		else
		{
			u3 = solve_a(t1, t2, e, u1, s1, u2, s2);
		}
		e = 0.00000001;
		double b = (min_GA(u3 + e, u1, s1, u2, s2) - min_GA(u3, u1, s1, u2, s2)) / e;
		//double b = min_pdf(u3, u1, s1, u2, s2);
		s3 = 1 / b / sqrt(2 * 3.14159265358979);
	}
	return;
}
double stdv(double s1, double s2)
{
	double s;
	if ((s1 == HUGE_VAL) || (s2 == HUGE_VAL)) s = HUGE_VAL;
	else s = sqrt(s1 * s1 + s2 * s2);
	return s;
}
/*
double phi(double t) // AGA_4
{
	if (t <= 0.1910)
		return exp(0.1047*t*t - 0.4992*t);
	else if ((t > 0.1910) && (t <= 0.7420))//(0.1910 < t <= 0.7420)
		return 0.9981*exp(0.05315*t*t - 0.4795*t);
	else if ((t > 0.7420) && (t <= 9.2254))//(0.7420 < t <= 9.2254)
		return exp(-0.4527*pow(t, 0.86) + 0.0218);
	else
		return exp(-0.2832*t - 0.4254);
}
double phi_inv(double t)
{
	if (t == 0)
		return HUGE_VAL;
	else if ((t > 0) && (t < 0.0479))//(0 <t < 0.0479)
		return (-log(t) - 0.4254) / 0.2832;
	else if ((t >= 0.0479) && (t < 0.7201)) {//(0.0479 <= t < 0.7201) {
		return pow((log(t) - 0.0218) / -0.4527, 1/0.86);
	}
	else if ((t >= 0.7201) && (t < 0.9125)) //(0.7201 <= t < 0.9125)
		return (0.4795 - sqrt(0.4795*0.4795 + 4 * 0.05315*log(t / 0.9981))) / 2 / 0.05315;
	else
		return (0.4992 - sqrt(0.4992*0.4992 + 4 * 0.1047*log(t))) / 2 / 0.1047;
}*/

double phi(double t)
{
	if (t < 7.0633)
		return exp(0.0116 * t * t - 0.4212 * t);
	else // if(t >= phi_pivot)
		return exp(-0.2944 * t - 0.3169);
}

double phi_inv(double t)
{
	if (t < 0.0911)
		return (-log(t) - 0.3169) / 0.2944;
	else
		return (0.4212 - sqrt(0.4212 * 0.4212 + 4 * 0.0116 * log(t))) / 2 / 0.0116;
}
void polar_codeconstruction(int CodeLength, float sigma, vector<int>& best_channel, vector<double>& mean)
{
	int m = (int)log2(CodeLength);
	std::vector<double> z(CodeLength, 0);
	const double alpha = -0.4527;
	const double beta = 0.0218;
	const double gamma = 0.8600;
	const double bisection_max = std::numeric_limits<double>::max();
	const double epsilon = 0.00000000001;

	for (auto i = 0; i < CodeLength; i++)
		best_channel[i] = i;

	for (auto i = 0; i < std::exp2(m); i++)
		z[i] = 2.0 / std::pow((double)sigma, 2.0);
	for (auto l = 1; l <= m; l++)
	{
		auto o1 = (int)std::exp2(m - l + 1);
		auto o2 = (int)std::exp2(m - l);
		for (auto t = 0; t < (int)std::exp2(l - 1); t++)
		{
			double T = z[t * o1];

			z[t * o1] = phi_inv(1.0 - std::pow(1.0 - phi(T), 2.0));
			if (z[t * o1] == HUGE_VAL) {
				z[t * o1] = T + M_LN2 / (alpha * gamma);
			}

			z[t * o1 + o2] = 2.0 * T;
		}
	}
	std::sort(best_channel.begin(), best_channel.end(), [&](int i1, int i2) { return z[i1] > z[i2]; });

	for (int i = 0; i < CodeLength; i++)
	{
		//P_ui[i] = (erfc(sqrt(z[i])/2) / 2);
		mean[i] = z[i];
	}
}
void polar_codeconstruction_punc(int RealLength, int CodeLength, float sigma, vector<int>& best_channel, vector<double>& mean)
{
	int m = (int)log2(CodeLength);
	//std::vector<double> z(CodeLength, 0);
	vector<std::vector<double>> z(m + 1, std::vector<double>(CodeLength, 0));
	const double alpha = -0.4527;
	const double beta = 0.0218;
	const double gamma = 0.8600;
	const double bisection_max = std::numeric_limits<double>::max();
	const double epsilon = 0.00000000001;

	for (auto i = 0; i < CodeLength; i++)
		best_channel[i] = i;

	int PuncLength = CodeLength - RealLength;
	for (auto i = 0; i < PuncLength; i++)
		z[m][i] = 0;

	for (auto i = PuncLength; i < std::exp2(m); i++)
		z[m][i] = 2.0 / std::pow((double)sigma, 2.0);

	for (int k = m - 1; k >= 0; k--)
	{
		int block = pow(2, k + 1);
		int step = pow(2, k);
		for (int cnt = 0; cnt < CodeLength; cnt += block)
		{
			for (int i = 0; i < step; i++)
			{
				z[k][i + cnt] = phi_inv(1.0 - (1.0 - phi(z[k + 1][i + cnt])) * (1.0 - phi(z[k + 1][i + cnt + step])));
				if (z[k][i + cnt] == HUGE_VAL) {
					z[k][i + cnt] = z[k + 1][i + cnt + step] + M_LN2 / (alpha * gamma);
				}
				z[k][i + cnt + step] = z[k + 1][i + cnt] + z[k + 1][i + cnt + step];
			}
		}
		/*cout << " stage = " << k << endl;
		for (int i = 0; i < CodeLength; i++)
		{
			cout << z[k][i] << " ";
			if (i % 16 == 15)cout << endl;
		}*/
	}
	std::sort(best_channel.begin(), best_channel.end(), [&](int i1, int i2) { return z[0][i1] > z[0][i2]; });

	for (int i = 0; i < CodeLength; i++)
	{
		mean[i] = z[0][i];
	}
}


void PolarEncode_xor_int8(int8_t* uout, int8_t* uin, int len)
{
	int i, j, k;
	if (len == 1) return;
	else if (len == 2)
	{
		uout[0] = uin[0] + uin[1];
		uout[1] = uin[1];
	}
	else if (len == 4)
	{
		*(uout + 1) = *(uin + 1) ^ *(uin + 3);
		*(uout + 2) = *(uin + 2) ^ *(uin + 3);
		*(uout + 3) = *(uin + 3);
		*(uout) = *(uin) ^ *(uin + 1) ^ *(uout + 2);
	}
	else if (len == 8)
	{
		*(uout + 3) = *(uin + 3) ^ *(uin + 7);
		*(uout + 5) = *(uin + 5) ^ *(uin + 7);
		*(uout + 6) = *(uin + 6) ^ *(uin + 7);
		*(uout + 7) = *(uin + 7);

		*(uout + 4) = *(uin + 4) ^ *(uin + 5) ^ *(uout + 6);
		*(uout + 2) = *(uin + 2) ^ *(uin + 3) ^ *(uout + 6);
		*(uout + 1) = *(uin + 1) ^ *(uin + 3) ^ *(uout + 5);
		*(uout) = *(uin) ^ *(uin + 1) ^ *(uin + 2) ^ *(uin + 3) ^ *(uout + 4);
	}
	else if (len == 16)
	{
		*(uout + 3) = *(uin + 3) ^ *(uin + 7);
		*(uout + 5) = *(uin + 5) ^ *(uin + 7);
		*(uout + 6) = *(uin + 6) ^ *(uin + 7);
		*(uout + 7) = *(uin + 7);
		*(uout + 4) = *(uin + 4) ^ *(uin + 5) ^ *(uout + 6);
		*(uout + 2) = *(uin + 2) ^ *(uin + 3) ^ *(uout + 6);
		*(uout + 1) = *(uin + 1) ^ *(uin + 3) ^ *(uout + 5);
		*(uout) = *(uin) ^ *(uin + 1) ^ *(uin + 2) ^ *(uin + 3) ^ *(uout + 4);

		*(uout + 11) = *(uin + 11) ^ *(uin + 15);
		*(uout + 13) = *(uin + 13) ^ *(uin + 15);
		*(uout + 14) = *(uin + 14) ^ *(uin + 15);
		*(uout + 15) = *(uin + 15);

		*(uout + 12) = *(uin + 12) ^ *(uin + 13) ^ *(uout + 14);
		*(uout + 10) = *(uin + 10) ^ *(uin + 11) ^ *(uout + 14);
		*(uout + 9) = *(uin + 9) ^ *(uin + 11) ^ *(uout + 13);
		*(uout + 8) = *(uin + 8) ^ *(uin + 9) ^ *(uin + 10) ^ *(uin + 11) ^ *(uout + 12);

		for (i = 0; i < len; i += 16)
		{
			for (k = 0; k < 8; k++)
			{
				uout[i + k] ^= uout[i + k + 8];
			}
		}
	}
	else
	{
		for (i = 0; i < len; i += 2)
		{
			uout[i] = uin[i] ^ uin[i + 1];
			uout[i + 1] = uin[i + 1];

		}
		for (i = 0; i < len; i += 4)
		{
			for (k = 0; k < 2; k++)
			{
				uout[i + k] ^= uout[i + k + 2];
			}
		}
		for (i = 0; i < len; i += 8)
		{
			for (k = 0; k < 4; k++)
			{
				uout[i + k] ^= uout[i + k + 4];
			}
		}
		for (i = 0; i < len; i += 16)
		{
			for (k = 0; k < 8; k++)
			{
				uout[i + k] ^= uout[i + k + 8];
			}
		}
		int temp = 32;

		/*__m128i* p = (__m128i*)uout;
		for (i = 0; i < len; i += temp) {
			__m128i a_128i = _mm_loadu_si128(p);
			__m128i b_128i = _mm_loadu_si128(p + 1);
			_mm_storeu_si128(p, _mm_xor_si128(a_128i, b_128i));
			p += 2;
		}
		temp <<= 1;*/
		int cnt = 1;
		__m128i* q = (__m128i*)uout;
		while (len >= temp) {
			for (i = 0; i < len; i += temp) {
				for (j = 0; j < (temp >> 5); j++) { // temp/64
					__m128i a_128i = _mm_loadu_si128(q + j);
					__m128i b_128i = _mm_loadu_si128(q + j + cnt);
					_mm_storeu_si128(q + j, _mm_xor_si128(a_128i, b_128i));
				}
				q += (cnt << 1);
			}
			temp <<= 1;
			cnt <<= 1;
			q = (__m128i*)uout;
		}
	}
}

void PolarEncode_xor(int* uout, int* uin, int len)
{
	if (len == 1) return;
	else if (len == 2)
	{
		uout[0] = uin[0] ^ uin[1];
		uout[1] = uin[1];
	}
	else if (len == 4)
	{
		*(uout + 1) = *(uin + 1) ^ *(uin + 3);
		*(uout + 2) = *(uin + 2) ^ *(uin + 3);
		*(uout + 3) = *(uin + 3);
		*(uout) = *(uin) ^ *(uin + 1) ^ *(uout + 2);
	}
	else
	{
		for (int i = 0; i < len; i += 2)
		{
			uout[i] = uin[i] ^ uin[i + 1];
			uout[i + 1] = uin[i + 1];

		}
		for (int i = 0; i < len; i += 4)
		{
			for (int k = 0; k < 2; k++)
			{
				uout[i + k] ^= uout[i + k + 2];
			}
		}
		int temp = 8;
		__m128i* p = (__m128i*)uout;
		for (int i = 0; i < len; i += temp) {
			__m128i a_128i = _mm_loadu_si128(p);
			__m128i b_128i = _mm_loadu_si128(p + 1);
			_mm_storeu_si128(p, _mm_xor_si128(a_128i, b_128i));
			p += 2;
		}
		temp <<= 1;
		int cnt = 1;
		__m256i* q = (__m256i*)uout;
		while (len >= temp) {
			for (int i = 0; i < len; i += temp) {
				for (int j = 0; j < temp / 16; j++) {
					__m256i a_256i = _mm256_loadu_si256(q + j);
					__m256i b_256i = _mm256_loadu_si256(q + j + cnt);
					_mm256_storeu_si256(q + j, _mm256_xor_si256(a_256i, b_256i));
				}
				q += (cnt << 1);
			}
			temp <<= 1;
			cnt <<= 1;
			q = (__m256i*)uout;
		}
	}
}


