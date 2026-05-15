#ifndef _MEM_INIT_H_
#define _MEM_INIT_H_

#include "phy_config.h"
#include "ch_mem.h"
#include "coding_mem.h"
#include "data_mem.h"
#include "det_mem.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>
#include <vector>
#include <string.h>
#include <chrono> 
#include "LDPCfunctions.h"

extern mem_ch_mtx_t ch_mtx_struct;
extern mem_coding_t coding_struct;
extern mem_data_t data_struct;
extern mem_det_t det_struct;

void thread_struct_init();
void thread_struct_free();
void FillValue(float* array, size_t n, float value);
void add_scalar(float* array, size_t length, size_t incx, float scalar);
void add_axpbyz(float* x, float* y, float* z, float alpha, float beta, size_t inc, size_t length);
void vec_copy(int length, float* x, int incx, float* y, int incy);
void vec_copy(int length, int* x, int incx, int* y, int incy);
void vec_mul_element(int length, float* x, float* y, float* z, float alpha);
void vec_div_element(int length, float* x, float* y, float* z, float alpha);
void vec_logic_element(int length, float* x, float* y, float value);
void vec_scalar_element(int length, float* x, float* y, float scalar);


using namespace std;
typedef struct {

	int** H;	//check matrix M x N
	int** V;  //base graph   Mb x Nb
	int* E_array; // length is C x 1 reserve the lengthBit of every CB in ratematching 
	int8_t* a_data_raw; //raw data buffer A x 1
	unsigned char* a_data_crc; // raw data+TB-CRC buffer B x 1
	unsigned char** a_data_block; // matrix for message data --size is C x K
	int8_t** a_data_block1;
	int8_t** a_data_ldpc;   // matrix for (message+check) data -- size is C x N1
	int8_t* a_data_match;   // outlen x 1
	int** a_data_col;      // col-format of message data --1 x K
	int** a_data_ldpc1;   // col-format of (message+check) data --1 x N1
	int8_t** rate_match_temp; // C x outlen/C
} mem_ldpc_transmitter_t;

typedef struct {

	float* a_data_y;  // size is outlen; message from detection part
	float* LLR_in;    // size is N
	float* y_array;   // size is N   temp message (extrinsic message) in decoding 
	float* LLR_out;   // size is  N  message result from decoding part
	float* LLR_out_tmp;// size is N
	int8_t* u_out_tmp; // size is N
	int8_t* uhat;		// size is B  transmit block after desegment
	int** uhat1;       // size is C x N  decode-decision result bits 
	int8_t** uhat2;    // size is C x K_CB_Bit

	// used in RateRecover
	float** LLR_in_C;   // size is C x N  RateRecovered results with filling two puncture columns
	float** LLR_in_tmp;    // size is C x N1 RateRecovered results without filling puncture columns 
	float** deconcatenated;  // size is C x (outlen / C) reserve deconcatenated streams of RateRecovered sequence
	
} mem_ldpc_receiver_t;


extern  mem_ldpc_transmitter_t ldpc_t_struct;
extern  mem_ldpc_receiver_t ldpc_r_struct;


void mem_ldpc_transmitter_init(mem_ldpc_transmitter_t* q, int A, int B, int L, int C, int K, int K1, int N, int N1, int Nb, int Mb, int Zc, int outlen);
void mem_ldpc_receiver_init(mem_ldpc_receiver_t* q, int A, int B, int L, int C, int K, int K1, int N, int N1, int Nb, int Mb, int Zc, int outlen);
void thread_struct_init_ldpc(int A, int B, int L, int C, int K, int K1, int N, int N1, int Nb, int Mb, int Zc, int outlen);


void mem_ldpc_transmitter_free(mem_ldpc_transmitter_t* q, int C, int K, int K1, int N, int N1, int Mb, int Zc);
void mem_ldpc_receiver_free(mem_ldpc_receiver_t* q, int C);
void thread_struct_free_ldpc(int C, int K, int K1, int N, int N1, int Mb, int Zc);


#endif // !_MEM_INIT_H_





