#ifndef _PHY_CONFIG_H_
#define _PHY_CONFIG_H_

#define ARMA_DONT_USE_FORTRAN_HIDDEN_ARGS
#define ARMA_USE_MKL_TYPES

#include "mkl.h"
#include <Eigen/Dense>
#include <stdint.h>
#include <vector>
#include <bitset>

#include <armadillo>

using namespace Eigen;
using namespace arma;

/* fixed parameters of the system  */
#define NOF_ANT_UE 32 // The Rx ANT NUM
#define NOF_ANT_BS 12 // The Tx ANT NUM
#define MODE_TYPE 6 // THE MODULATION TYPE

#define NOF_CODE_LEN 1440 // The code length
#define CODE_RATE    0.8
#define NOF_CRC_LEN 16 // The CRC length
#define NOF_INFO_LEN NOF_CODE_LEN*CODE_RATE // The info length 


#define STAGE (size_t)(log(NOF_CODE_LEN)/log(2))

#define NOF_MIMO_BLCOK NOF_CODE_LEN/NOF_ANT_BS/MODE_TYPE


#define cf_t	MKL_Complex8

 

#if MODE_TYPE == 8 // 256QAM

static float cons[16] = { -0.383482f, -0.536875f, -0.230089f, -0.07669f,  -0.843661f, -0.690268f, -0.997054f, -1.15044f,
		0.383482f, 0.536875f, 0.230089f, 0.07669f,  0.843661f, 0.690268f, 0.997054f, 1.15044f };

static Row<float> cons_row(&cons[0], 16, true);
static Col<float> cons_col(&cons[0], 16, true);

#elif MODE_TYPE == 6  // 64QAM
static float cons[8] = { -0.462910f, -0.154303f, -0.771517f, -1.08012f,	0.462910f,  0.154303f,  0.771517f,  1.08012f };
static Row<float> cons_row(&cons[0], 8, true);
static Col<float> cons_col(&cons[0], 8, true);
#else
static float cons[4] = { -0.316228f, -0.948683f,  0.316228f,  0.948683f };
static Row<float> cons_row(&cons[0], 4, true);
static Col<float> cons_col(&cons[0], 4, true);

#endif

#endif
