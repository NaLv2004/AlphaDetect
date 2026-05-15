#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "LDPCfunctions.h"
#include "crc.h"

using namespace std;
//codewordLength = cbsInfo.N + 2*cbsInfo.Zc
//blk=(cwlen - F - 24)*C
#define bg_col 52

int8_t *DesegmentLDPC(int8_t **input, int cwlen, int F, int blklen, int C)
{//cwlen=codeword length=N ;input[C][cwlen]
//blklen=A+L=B
	int8_t **cbi = new int8_t*[C];
	for (int i = 0; i < C; i++)
	{
		cbi[i] = new int8_t[cwlen - F];
	}
	int8_t *blk = new int8_t[blklen];

	for (int i = 0; i < C; i++)
	{
		for (int j = 0; j < cwlen - F; j++)
		{
			cbi[i][j] = input[i][j];
		}
	}
	if (C == 1)
	{
		for (int i = 0; i < blklen; i++)
		{
			blk[i] = input[0][i];
		}
	}
	else
	{
		for (int i = 0; i < C; i++)
		{
			for (int j = 0; j < cwlen - F - 24; j++)
			{
				blk[i*(cwlen - F - 24) + j] = cbi[i][j];
			}
		}
	}
	for (int i = 0; i < C; i++)
	{
		delete[]cbi[i];
	}
	return blk;
}
void modified_DesegmentLDPC(int8_t **input, int **uhat1, int8_t *output, int L, int K1, int blklen, int C)
{//cwlen=K1, input[C][cwlen]
 //blklen=A+L=B
 // the size of uhat1 is C x N

 // get the bit excluding filler bits
	for (int i = 0; i < C; i++) {
		for (int j = 0; j < K1; j++) {
			input[i][j] = int8_t(uhat1[i][j]);
		}
	}

	if (C == 1) // 只有一个code block
	{
		for (int i = 0; i < blklen; i++)
		{
			output[i] = input[0][i];
		}
	}
	else
	{
		//excluding the CB-CRC
		for (int i = 0; i < C; i++)
		{
			for (int j = 0; j < K1 - L; j++)
			{
				output[i*(K1 - L) + j] = input[i][j];
			}
		}
	}
}

void SegmentLDPC(int C, int K, int K1, int L, unsigned char* a_data_crc, unsigned char** a_data_block, int8_t** a_data_block1)
{ // K1 -- all meaningful bitlength exclude filler bits
	int s = 0;
	if (C == 1) // only one CB no need for CB-CRC
	{
		for (int k = 0; k < K1; k++)
		{
			a_data_block[0][k] = a_data_crc[k];
			a_data_block1[0][k] = (int8_t)a_data_crc[k];
		}
		for (int k = K1; k < K; k++)
		{
			a_data_block1[0][k] = -1;
		}
	}
	else
	{
		for (int r = 0; r < C; r++)
		{
			for (int k = 0; k < K1 - L; k++)
			{
				a_data_block[r][k] = a_data_crc[s];
				a_data_block1[r][k] = (int8_t)a_data_crc[s];
				s++;
			}
			tx_append_crc(a_data_block[r], (K1 - L), L, 2);
			for (int k = K1 - L; k < K1; k++)
			{
				a_data_block1[r][k] = (int8_t)a_data_block[r][k];
			}
			for (int k = K1; k < K; k++)
			{
				a_data_block1[r][k] = -1;
			}
		}
	}
}

int** buildSubBlock(int Z, int shift)
{
	int** M = new int*[Z];
	// creat full zero Z x Z matrix M
	for (int i = 0; i < Z; i++)
	{
		M[i] = new int[Z];
		memset(M[i], 0, sizeof(int) * Z);
	} 

	if (shift == -1)
	{
		return M;  
	}
	else  //right shift of every element of basegraph
	{
		for (int i = 0; i < Z; i++)
		{
			M[i][(i + shift) % Z] = 1;  // 不用%Z 也行?
		}
		return M;
	}
}
void parityCheckMatrix(int **basegraph, int **H, int bgn, int Z)
{
	int Nb, Mb;
	int** M;
	if (bgn == 2)
	{
		Mb = 42;
		Nb = 52;
	}
	else
	{
		Mb = 46;
		Nb = 68;
	}
	for (int i = 0; i < Mb; i++)
	{
		for (int j = 0; j < Nb; j++)
		{
			basegraph[i][j] = basegraph[i][j] % Z;
		}
	}
	for (int i1 = 0; i1 < Mb; i1++)
	{
		for (int i2 = 0; i2 < Nb; i2++)
		{
			M = buildSubBlock(Z, basegraph[i1][i2]); //element of basegraph
			for (int j = 0; j < Z; j++)
			{
				for (int k = 0; k < Z; k++)
				{
					H[i1 * Z + j][i2 * Z + k] = M[j][k];  //extension of element in basegraph
				}
			}
		}
	}
}


void getH(int **V, int **H, int bgn, int Z, int setIdx) {
	// V is original base graph parity matrix
	switch (bgn) {
	case 1: {  // BG#1 V[46][68];
		switch (setIdx) {
		case 1: {
			fstream fileV;
			fileV.open("BG1S1.txt", ios::in);
			for (int i = 0; i < 46; i++)
				for (int j = 0; j < 68; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		case 2: {
			fstream fileV;
			fileV.open("BG1S2.txt", ios::in);
			for (int i = 0; i < 46; i++)
				for (int j = 0; j < 68; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		case 3: {
			fstream fileV;
			fileV.open("BG1S3.txt", ios::in);
			for (int i = 0; i < 46; i++)
				for (int j = 0; j < 68; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		case 4: {
			fstream fileV;
			fileV.open("BG1S4.txt", ios::in);
			for (int i = 0; i < 46; i++)
				for (int j = 0; j < 68; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		case 5: {
			fstream fileV;
			fileV.open("BG1S5.txt", ios::in);
			for (int i = 0; i < 46; i++)
				for (int j = 0; j < 68; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		case 6: {
			fstream fileV;
			fileV.open("BG1S6.txt", ios::in);
			for (int i = 0; i < 46; i++)
				for (int j = 0; j < 68; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		case 7: {
			fstream fileV;
			fileV.open("BG1S7.txt", ios::in);
			for (int i = 0; i < 46; i++)
				for (int j = 0; j < 68; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		default: {
			fstream fileV;
			fileV.open("BG1S8.txt", ios::in);
			for (int i = 0; i < 46; i++)
				for (int j = 0; j < 68; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		}
		break;
	}
	default: {  //BG#2 int V[42][52];
		switch (setIdx) {
		case 1: {
			fstream fileV;
			fileV.open("BG2S1.txt", ios::in);
			for (int i = 0; i < 42; i++)
				for (int j = 0; j < 52; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		case 2: {
			fstream fileV;
			fileV.open("BG2S2.txt", ios::in);
			for (int i = 0; i < 42; i++)
				for (int j = 0; j < 52; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		case 3: {
			fstream fileV;
			fileV.open("BG2S3.txt", ios::in);
			for (int i = 0; i < 42; i++)
				for (int j = 0; j < 52; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		case 4: {
			fstream fileV;
			fileV.open("BG2S4.txt", ios::in);
			for (int i = 0; i < 42; i++)
				for (int j = 0; j < 52; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		case 5: {
			fstream fileV;
			fileV.open("BG2S5.txt", ios::in);
			for (int i = 0; i < 42; i++)
				for (int j = 0; j < 52; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		case 6: {
			fstream fileV;
			fileV.open("BG2S6.txt", ios::in);
			for (int i = 0; i < 42; i++)
				for (int j = 0; j < 52; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		case 7: {
			fstream fileV;
			fileV.open("BG2S7.txt", ios::in);
			for (int i = 0; i < 42; i++)
				for (int j = 0; j < 52; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		default: {
			fstream fileV;
			fileV.open("BG2S8.txt", ios::in);
			for (int i = 0; i < 42; i++)
				for (int j = 0; j < 52; j++)
					fileV >> V[i][j];
			fileV.close();
			break;
		}
		}
	}
	}
	//ModifiedBaseGraph(V, bgn, Z);
	parityCheckMatrix(V, H, bgn, Z);  // V and Z-> Basegraph  V(Mb x Nb)-> real parity check matrix H(M X N) 
}//子CPP
