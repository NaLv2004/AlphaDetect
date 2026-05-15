#pragma once
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "ratematch.h"
#include"decode.h"
//#define encode_outlen 3600///////50*Z/////
//int Ncb = 12672;

// the function calculates starting position k0 and buffer length of each CodeBlock
void RateMatchLDPC_pre(int *E_array, int &k0, int N, int C, int Zc, int bgn, int outlen, int rv, int modulation, int nlayers)
{
	int Ncb = N;
	int E;
	int cnt = 0;
	//select the starting position k0
	if (bgn == 1)
	{
		if (rv == 0)
		{
			k0 = 0;
		}
		else if (rv == 1)
		{
			k0 = floor(float(17 * Ncb) / (66 * Zc))*Zc;
		}
		else if (rv == 2)
		{
			k0 = floor(float(33 * Ncb) / (66 * Zc))*Zc;
		}
		else if (rv == 3)
		{
			k0 = floor(float(56 * Ncb) / (66 * Zc))*Zc;
		}
	}
	else
	{
		if (rv == 0)
		{
			k0 = 0;
		}
		else if (rv == 1)
		{
			k0 = floor(float(13 * Ncb) / (50 * Zc))*Zc;
		}
		else if (rv == 2)
		{
			k0 = floor(float(25 * Ncb) / (50 * Zc))*Zc;
		}
		else if (rv == 3)
		{
			k0 = floor(float(43 * Ncb) / (50 * Zc))*Zc;
		}
	}
	//int8_t *output = new int8_t[outlen];
	for (int r = 0; r < C; r++)
	{
		if (r <= C - ( outlen / (nlayers * modulation)) % C - 1)
		{
			E = nlayers*modulation*floor(float(outlen) / float(nlayers*modulation*C));
			// lengthE less than N1
		}
		else
		{
			E = nlayers*modulation*ceil(float(outlen) / (nlayers*modulation*C));
		}
		E_array[r] = E;
		//temp[r] = new int8_t[E];
	}
	//return temp;
}
void modified_RateMatchLDPC(int8_t **input, int8_t *output, int8_t **temp, int *E_array, int k0, int N1, int C, int Zc, int bgn, int outlen, int rv, int modulation)
{   // input -- C x N1
	int Ncb = N1;
	int E;
	int cnt = 0;
	//int8_t **temp = new int8_t *[C];
	//int8_t *output = new int8_t[outlen];
	for (int r = 0; r < C; r++)
	{
		modified_cbsRateMatch(input[r], temp[r], E_array[r], k0, Ncb, modulation);
		// temp is matched bit
		// E_array is matched length of CodeBlock
		for (int i = 0; i < E_array[r]; i++)
		{
			*(output + cnt + i) = temp[r][i]; // concatenation
		}
		cnt = cnt + E_array[r];
	}

}
void modified_cbsRateMatch(int8_t*d, int8_t*e1, int E, int k0, int Ncb, int Qm)
{
	int k = 0;
	int j = 0;
	int8_t *e = new int8_t[E];
	// Ncb = N1  the encoded length of each CodeBlock
	// Qm sets to 1;
	// fill the RateMatching buffer with circulation of length Ncb
	while (k < E)
	{	// starting from k0 and total length is Ncb
		// length of E is less than Ncb
		// 会不会有些校验位无法传送？不会，校验矩阵限制了最低码率，有效校验位的计算包含在outlen里
		if (d[(k0 + j) % Ncb] != -1) // excluding filler bits
		{
			e[k] = d[(k0 + j) % Ncb];
			k++;
		}
		j++;
	}
	for (int i = 0; i < Qm; i++) // Bit interleaving actually it didn't
	{
		for (int j = 0; j < E / Qm; j++)
		{
			e1[j*Qm + i] = e[i*E / Qm + j];
		}
	}
	delete[]e;
}

int8_t *RateMatchLDPC(int8_t **input, int N, int C, int Zc, int bgn, int outlen, int rv, int modulation, int nlayers)
{
	int k0;
	int Ncb = N;
	int E;
	int cnt=0;
	int8_t **temp = new int8_t *[C];
	int8_t *output=new int8_t[outlen];
	if (bgn == 1)
	{
		if (rv == 0)
		{
			k0 = 0;
		}
		else if (rv == 1)
		{
			k0 = floor(float(17 * Ncb) / (66 * Zc))*Zc;
		}
		else if (rv == 2)
		{
			k0 = floor(float(33 * Ncb) / (66 * Zc))*Zc;
		}
		else if (rv == 3)
		{
			k0 = floor(float(56 * Ncb) / (66 * Zc))*Zc;
		}
	}
	else
	{
		if (rv == 0)
		{
			k0 = 0;
		}
		else if (rv == 1)
		{
			k0 = floor(float(13 * Ncb) / (50 * Zc))*Zc;
		}
		else if (rv == 2)
		{
			k0 = floor(float(25 * Ncb) / (50 * Zc))*Zc;
		}
		else if (rv == 3)
		{
			k0 = floor(float(43 * Ncb) / (50 * Zc))*Zc;
		}
	}
	for (int r = 0; r < C; r++)
	{
		if (r <= C - (outlen / (nlayers * modulation)) % C - 1)
		{
			E = nlayers*modulation*floor(float(outlen) / (nlayers*modulation*C));
		}
		else
		{
			E = nlayers*modulation*ceil(float(outlen) / (nlayers*modulation*C));
		}
		temp[r] = new int8_t[E];
		temp[r] = cbsRateMatch(input[r], E, k0, Ncb, modulation);
		for (int i = 0; i < E; i++)
		{
			*(output+cnt+i) = temp[r][i];
		}
		cnt = cnt + E;
	}
	for (int i = 0; i < C; i++)
	{
		delete[]temp[i];
	}
	delete[] temp;
	return output;

}
int8_t * cbsRateMatch(int8_t*d, int E, int k0, int Ncb, int Qm)
{
	int k = 0;
	int j = 0;
	int8_t *e = new int8_t[E];
	int8_t *e1 = new int8_t[E];


	// fill the buffer for ratematching 
	while (k < E)
	{
		if (d[(k0 + j) % Ncb] != -1)
		{
			e[k] = d[(k0 + j) % Ncb];
			k++;
		}
		j++;
	} 
	for (int i = 0; i < Qm; i++)
	{
		for (int j = 0; j < E / Qm; j++)
		{
			e1[j*Qm + i] = e[i*E / Qm + j];
		}
	}
	delete[]e;
	return e1;
}
void RateRecoverLDPC_pre(int G, int N, int C, int Zc, int bgn, int outlen, int rv, int modulation, int nlayers)
{
	int k0;
	//int Ncb = N;
	int E;
	int cnt = 0;
	int gIdx = 0;
	//float **deconcatenated = new float *[C];
	for (int r = 0; r < C; r++)
	{
		if (r <= C - ((G / (nlayers * modulation))) % C - 1)
		{
			E = nlayers*modulation*floor(float(outlen) / (nlayers*modulation*C));
		}
		else
		{
			E = nlayers*modulation*ceil(float(outlen) / (nlayers*modulation*C));
		}
		//float *deconcatenated = new float[E];
		//deconcatenated[r] = new float[E];
	}
	//return deconcatenated;

}
void modified_RateRecoverLDPC(float*input, float **temp, float **deconcatenated, float **LLR_in_C, int *E_array, int k0, int G, int N, int C, int Zc, int K, int K1, int bgn, int outlen, int rv, int modulation, int nlayers)
{   // temp is the result of raterecover without filling puncture C x N1
	// LLR_IN_C the temp after filling puncture
	// K1 --K_CB_Bit -- excluding filler bits
	// K -- the total bits 
	//outlen=G
	int Ncb = N;
	int E;
	int cnt = 0;
	int gIdx = 0; // used for deconcatenating
	K1 = K1 - 2 * Zc;
	K = K - 2 * Zc;  // message length of encoded codeword
	for (int r = 0; r < C; r++)
	{
		if (G < E_array[r])
		{
			for (int j = 0; j < G; j++)
			{
				deconcatenated[r][j] = input[j];
			}
			for (int j = 0; j < E_array[r] - G; j++)
			{
				deconcatenated[r][G + j] = 1; //如果速率匹配后bits length小于一个E_array的长度，则将后面填1
			}
		}
		else
		{
			for (int j = 0; j < E_array[r]; j++)
			{
				deconcatenated[r][j] = input[gIdx + j]; // deconcatenation 取出outlen中一个E_array的长度
			}
		}
		gIdx = gIdx + E_array[r];
		modified_cbsRateRecover(deconcatenated[r], temp[r], N, E_array[r], K, K1, k0, Ncb, modulation);
		// from the deconcatenation to the temp part
		// K and K1 are uesd to make sure the filler bits position

	}

	// fill the puncture part message with zeros
	for (int i = 0; i < C; i++) {
		for (int j = 0; j < 2 * Zc; j++) {
			LLR_in_C[i][j] = 0;
		}
	}
	// fill the RateRecovered bits to the output LLR_in_C
	for (int i = 0; i < C; i++) {
		for (int j = 2 * Zc; j <N; j++) {
			LLR_in_C[i][j] = temp[i][j - 2 * Zc];
		}
	}

}

void modified_cbsRateRecover(float*d, float*temp, int N, int E, int K, int Kd, int k0, int Ncb, int Qm)
{
	// Kd is the starting position of filler bits
	// K is the total length of message bits
	// d -input; temp -output
	int k = 0;
	int j = 0;
	int idx;
	float *e = new float[E];
	int *indices = new int[E];
	
	// de-interleaving from input -d to -e
	for (int i = 0; i < Qm; i++)
	{
		for (int j = 0; j < E / Qm; j++)
		{
			e[i*E / Qm + j] = d[j*Qm + i];
		}
	}
	// j=0?
	while (k < E)
	{
		idx = (k0 + j) % Ncb;
		if (!(idx >= Kd&&idx < (K)))
		{
			indices[k] = idx;  // take down the non filler position
			k++;
		}
		j++;
	}
	for (int j = 0; j < N; j++)
	{
		temp[j] = 0.0;  // initial the output with zero
	}
	for (int j = Kd; j < K; j++)
	{
		temp[j] = 999999999; // fill the filler bits position with inf number
		//temp[i] = 0.0;
	}
	// ratematching recover 首先是信息位的填充，然后是fillerbits 的填充；超过Earray的部分，填充为0；
	for (int n = 0; n < E; n++)
	{
		temp[indices[n]] = temp[indices[n]] + e[n]; //fill the message bits 
	}
	delete[]e;
	delete[]indices;
}
float * cbsRateRecover(float*d,int N, int E, int K, int Kd, int k0, int Ncb, int Qm)
{

	int k = 0;
	int j = 0;
	int idx;
	float *e = new float[E];
	float *e1 = new float[N];
	int *indices = new int[E];
	//int8_t **e1 = new int8_t*[E / Qm];
	//for (int i = 0; i < E / Qm; i++)
	//{
	//e1[i] = new int8_t[Qm];
	//}
	for (int i = 0; i < Qm; i++)
	{
		for (int j = 0; j < E / Qm; j++)
		{
			e[i*E / Qm + j] = d[j*Qm + i];
		}
	}
	while (k < E)
	{
		idx = (k0 + j) % Ncb;
		if (!(idx >= Kd&&idx < (K)))
		{
			indices[k] = idx;
			k++;
		}
		j++;
	}
	for (int j = 0; j < N; j++)
	{
		e1[j] = 0.0;
	}
	for (int i = Kd; i < K; i++)
	{
		e1[i] = INFINITY;
	}
	for (int n = 0; n < E; n++)
	{
		e1[indices[n]] = e[n];
	}
	delete[]e;
	delete[]indices;
	return e1;
}
float **RateRecoverLDPC(float*input, int G, int N, int C, int Zc, int K, int K1, int bgn, int outlen, int rv, int modulation, int nlayers)
{
	int k0;
	int Ncb = N;
	int E;
	int cnt = 0;
	int gIdx = 0;
	float **temp = new float *[C];
	float *output;
	K1 = K1 - 2 * Zc;
	K = K - 2 * Zc;
	if (bgn == 1)
	{
		if (rv == 0)
		{
			k0 = 0;
		}
		else if (rv == 1)
		{
			k0 = floor(float(17 * Ncb) / (66 * Zc)) * Zc;
		}
		else if (rv == 2)
		{
			k0 = floor(float(33 * Ncb) / (66 * Zc)) * Zc;
		}
		else if (rv == 3)
		{
			k0 = floor(float(56 * Ncb) / (66 * Zc)) * Zc;
		}
	}
	else
	{
		if (rv == 0)
		{
			k0 = 0;
		}
		else if (rv == 1)
		{
			k0 = floor(float(13 * Ncb) / (66 * Zc)) * Zc;//+float
		}
		else if (rv == 2)
		{
			k0 = floor(float(25 * Ncb) / (66 * Zc)) * Zc;
		}
		else if (rv == 3)
		{
			k0 = floor(float(43 * Ncb) / (66 * Zc)) * Zc;
		}
	}
	for (int r = 0; r < C; r++)
	{
		if (r <= C - ((G / (nlayers * modulation))) % C - 1)
		{
			E = nlayers*modulation*floor(float(outlen) / (nlayers*modulation*C));
		}
		else
		{
			E = nlayers*modulation*ceil(float(outlen) / (nlayers*modulation*C));
		}
		float *deconcatenated = new float[E];
		if (G < E)
		{
			for (int j = 0; j < G; j++)
			{
				deconcatenated[j] = input[j];
			}
			for (int j = 0; j < E - G; j++)
			{
				deconcatenated[G + j] = 1;
			}
		}
		else
		{
			for (int j = 0; j < E; j++)
			{
				deconcatenated[j] = input[gIdx + j];
			}
		}
		gIdx = gIdx + E;
		temp[r] = new float[N];
		temp[r] = cbsRateRecover(deconcatenated, N, E, K, K1, k0, Ncb, modulation);

	}

	return temp;

}
int PDSCH_Transport_Block_Size_computation(float R, int modulation, int nlayers)
{
	int N_sc_RB = 12; // Number of subcarriers in a physical resource block
	int N_symb_sh = 13;//Number of symbols of the PDSCH allocation within the slot
	int N_DMRS_PRB = 12; // Number of REs for DM - RS per PRB in the scheduled duration including the overhead of the DM - RS CDM groups
	int N_oh_PRB = 0;   // Overhead configured by higher layer parameter xOverhead in PDSCH - ServingCellConfig
	int theta_PRB = 4; // total number of allocated PRBs for the User Equipment(UE)
	int scale = 1;
	int TBS, C, n, power_2_n, NN_info;
	int LUT[93] = { 24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,208,224,240,256,272,288,304,320,336,352,368,384,408,432,456,480,504,528,552,576,608,640,672,704,736,768,808,848,888,928,984,1032,1064,1128,1160,1192,1224,1256,1288,1320,1352,1416,1480,1544,1608,1672,1736,1800,1864,1928,2024,2088,2152,2216,2280,2408,2472,2536,2600,2664,2728,2792,2856,2976,3104,3240,3368,3496,3624,3752,3824 };
	int N_RE = 156 * theta_PRB;
	int N_info = scale*N_RE*R*modulation*nlayers;
	if (N_info > 3824)
	{
		n = floor(float(log2(N_info - 24))) - 5;
		power_2_n = pow(2,n);
		NN_info = max_f(3840, power_2_n*round(float((N_info - 24)) / power_2_n));
		if (R > 0.25)
		{
			if (NN_info > 8424)
			{
				C = ceil(float((NN_info + 24)) / 8424);
				TBS = 8 * C * ceil(float((NN_info + 24)) / (8 * C)) - 24;
			}
			else
			{
				TBS = 8 * ceil(float((NN_info + 24)) / 8) - 24;
			}
		}
		else
		{
			C = ceil(float((NN_info + 24)) / 3816);
			TBS = 8 * C*ceil(float((NN_info + 24)) / (8 * C)) - 24;
		}
		return TBS;
	}
	else
	{
		n = max_f(3, floor(float(log2(N_info)) - 6));
		power_2_n = pow(2, n);
		NN_info = max_f(3840, power_2_n*round(float((N_info - 24)) / power_2_n));
		for (int i = 0; i < 93; i++)
		{
			if (LUT[i] >= NN_info)
			{
				TBS = LUT[i];
				return TBS;
			}
		}
	}
}
