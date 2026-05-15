/*
 * file name: modem.h
 * header of  modem.c
 * modulation and demodulation for QPSK, 16-QAM, 64-QAM
 *
 * Created by Dongming Wang, Oct. 8, 2016
 *
 */


#ifndef _MODEM_H_
#define _MODEM_H_

#include "phy_config.h"

extern float table_qpsk[2];
extern float	table_16qam[4];
extern float	table_64qam[8];


/*
float table_qpsk[2]	= { -0.707107f, 0.707107f };
float	table_16qam[4]= {-0.316228f, -0.948683f,  0.316228f,  0.948683f};
float	table_64qam[8]= {-0.462910f, -0.154303f, -0.771517f, -1.08012f,
												0.462910f,  0.154303f,  0.771517f,  1.08012f};
*/
void modulation(int* input_bits, cf_t* symbols_out, int mod_type, int nof_symbols);
void soft_demodulation(float* llr, cf_t* received_symbols, int nof_symbols,
	int	mod_type);

void soft_demodulation(float* llr, float* received_symbols, int nof_symbols, int mod_type);

#endif
