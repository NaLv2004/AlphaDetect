/*
* file name: modulation.c
*
* The following modulation types are adopted, QPSK, 16-QAM, 64-QAM.
*
* Created by Dongming Wang, Oct. 8, 2016
*
*/

#include "modem.h"

float table_qpsk[2] = { -0.707107f, 0.707107f };
float table_16qam[4] = { -0.316228f, -0.948683f,  0.316228f,  0.948683f };
float table_64qam[8] = { -0.462910f, -0.154303f, -0.771517f, -1.08012f,	0.462910f,  0.154303f,  0.771517f,  1.08012f };
float table_256qam[16] = { -0.383482f, -0.536875f, -0.230089f, -0.07669f,  -0.843661f, -0.690268f, -0.997054f, -1.15044f,
		0.383482f, 0.536875f, 0.230089f, 0.07669f,  0.843661f, 0.690268f, 0.997054f, 1.15044f };

//float table_qpsk[2] = { -1.0f, 1.0f };
//float table_16qam[4] = { -1.0f, -3.0f, 1.0f, 3.0f };
//float table_64qam[8] = { -3.0f, -1.0f, -5.0f, -7.0f,    3.0f,    1.0f,    5.0f,    7.0f };
//float table_256qam[16] = { -5.0f, -7.0f, -3.0f, -1.0f, -11.0f, -9.0f, -13.0f, -15.0f,    5.0f,    7.0f,    3.0f,    1.0f,
//							11.0f,    9.0f,   13.0f,   15.0f };


void modulation(int* input_bits, cf_t* symbols_out, int mod_type,
	int nof_symbols)
{
	int i, j;
	int tmp_i, tmp_q;
	int half_sym;

	half_sym = mod_type / 2;

	switch (mod_type)
	{
	case 2:
		for (i = 0; i < nof_symbols; i++)
		{
			tmp_i = 0;
			tmp_q = 0;

			for (j = 0; j < half_sym; j++)
			{
				tmp_i = tmp_i + (input_bits[i * mod_type + j] << (half_sym - j - 1));
				tmp_q = tmp_q + (input_bits[i * mod_type + half_sym + j] << (half_sym - j - 1));
			}

			symbols_out[i].real = table_qpsk[tmp_i];
			symbols_out[i].imag = table_qpsk[tmp_q];

		}
		break;
	case 4:
		for (i = 0; i < nof_symbols; i++)
		{
			tmp_i = 0;
			tmp_q = 0;

			for (j = 0; j < half_sym; j++)
			{
				tmp_i = tmp_i + (input_bits[i * mod_type + j] << (half_sym - j - 1));
				tmp_q = tmp_q + (input_bits[i * mod_type + half_sym + j] << (half_sym - j - 1));
			}

			symbols_out[i].real = table_16qam[tmp_i];
			symbols_out[i].imag = table_16qam[tmp_q];

		}
		break;
	case 6:
		for (i = 0; i < nof_symbols; i++)
		{
			tmp_i = 0;
			tmp_q = 0;

			for (j = 0; j < half_sym; j++)
			{
				tmp_i = tmp_i + (input_bits[i * mod_type + j] << (half_sym - j - 1));
				tmp_q = tmp_q + (input_bits[i * mod_type + half_sym + j] << (half_sym - j - 1));
			}

			symbols_out[i].real = table_64qam[tmp_i];
			symbols_out[i].imag = table_64qam[tmp_q];

		}
		break;

	case 8:
		for (i = 0; i < nof_symbols; i++)
		{
			tmp_i = 0;
			tmp_q = 0;

			for (j = 0; j < half_sym; j++)
			{
				tmp_i = tmp_i + (input_bits[i * mod_type + j] << (half_sym - j - 1));  //左移1位相当于乘2
				tmp_q = tmp_q + (input_bits[i * mod_type + half_sym + j] << (half_sym - j - 1));
			}

			symbols_out[i].real = table_256qam[tmp_i];
			symbols_out[i].imag = table_256qam[tmp_q];

		}
		break;
	default:
		printf("Invalid modulation type: 2:QPSK, 4:16-QAM \n");
	}
}

void soft_demodulation(float* llr, cf_t* received_symbols, int nof_symbols,
	int	mod_type)
{
	int	i;

	switch (mod_type)
	{
	case 2:
		for (i = 0; i < nof_symbols; i++) {

			llr[i * mod_type] = received_symbols[i].real;
			llr[i * mod_type + 1] = received_symbols[i].imag;

		}
		break;
	case 4:
		for (i = 0; i < nof_symbols; i++) {

			llr[i * mod_type] = received_symbols[i].real;
			llr[i * mod_type + 1] = fabs(received_symbols[i].real) - 0.6324555;
			llr[i * mod_type + 2] = received_symbols[i].imag;
			llr[i * mod_type + 3] = fabs(received_symbols[i].imag) - 0.6324555;

		}
		break;
	case 6:
		for (i = 0; i < nof_symbols; i++) {

			llr[i * mod_type] = received_symbols[i].real;
			llr[i * mod_type + 1] = fabs(llr[i * mod_type]) - 0.6172134;
			llr[i * mod_type + 2] = fabs(llr[i * mod_type + 1]) - 0.3086067;

			llr[i * mod_type + 3] = received_symbols[i].imag;
			llr[i * mod_type + 4] = fabs(llr[i * mod_type + 3]) - 0.6172134;
			llr[i * mod_type + 5] = fabs(llr[i * mod_type + 4]) - 0.3086067;

		}
		break;

	case 8:
		for (i = 0; i < nof_symbols; i++) {

			llr[i * mod_type] = received_symbols[i].real;
			llr[i * mod_type + 1] = fabs(llr[i * mod_type]) - 0.6135715;
			llr[i * mod_type + 2] = fabs(llr[i * mod_type + 1]) - 0.306786;
			llr[i * mod_type + 3] = fabs(llr[i * mod_type + 2]) - 0.1533895;

			llr[i * mod_type + 4] = received_symbols[i].imag;
			llr[i * mod_type + 5] = fabs(llr[i * mod_type + 4]) - 0.6135715;
			llr[i * mod_type + 6] = fabs(llr[i * mod_type + 5]) - 0.306786;
			llr[i * mod_type + 7] = fabs(llr[i * mod_type + 6]) - 0.1533895;

		}
		break;

	default:
		printf("Invalid modulation type: 2:QPSK, 4:16-QAM, 6:64-QAM! \n");
	}
}

void soft_demodulation(float* llr, float* received_symbols, int nof_symbols,
	int	mod_type)
{
	int	i;

	switch (mod_type)
	{

	case 2:
		for (i = 0; i < nof_symbols; i++) {

			llr[i * mod_type] = received_symbols[i];
			llr[i * mod_type + 1] = received_symbols[i + NOF_ANT_BS];
		}
		break;
	case 4:
		for (i = 0; i < nof_symbols; i++) {

			llr[i * mod_type] = received_symbols[i];
			llr[i * mod_type + 1] = fabs(received_symbols[i]) - 0.6324555;
			llr[i * mod_type + 2] = received_symbols[i + NOF_ANT_BS];
			llr[i * mod_type + 3] = fabs(received_symbols[i + NOF_ANT_BS]) - 0.6324555;
		}
		break;
	case 6:
		for (i = 0; i < nof_symbols; i++) {

			llr[i * mod_type] = received_symbols[i];
			llr[i * mod_type + 1] = fabs(llr[i * mod_type]) - 0.6172134;
			llr[i * mod_type + 2] = fabs(llr[i * mod_type + 1]) - 0.3086067;

			llr[i * mod_type + 3] = received_symbols[i + NOF_ANT_BS];
			llr[i * mod_type + 4] = fabs(llr[i * mod_type + 3]) - 0.6172134;
			llr[i * mod_type + 5] = fabs(llr[i * mod_type + 4]) - 0.3086067;
		}
		break;

	case 8:
		for (i = 0; i < nof_symbols; i++) {

			llr[i * mod_type] = received_symbols[i];
			llr[i * mod_type + 1] = fabs(llr[i * mod_type]) - 0.6135715;
			llr[i * mod_type + 2] = fabs(llr[i * mod_type + 1]) - 0.306786;
			llr[i * mod_type + 3] = fabs(llr[i * mod_type + 2]) - 0.1533895;

			llr[i * mod_type + 4] = received_symbols[i + NOF_ANT_BS];
			llr[i * mod_type + 5] = fabs(llr[i * mod_type + 4]) - 0.6135715;
			llr[i * mod_type + 6] = fabs(llr[i * mod_type + 5]) - 0.306786;
			llr[i * mod_type + 7] = fabs(llr[i * mod_type + 6]) - 0.1533895;

		}
		break;
	default:
		printf("Invalid modulation type: 2:QPSK, 4:16-QAM, 6:64-QAM! \n");
	}
}

