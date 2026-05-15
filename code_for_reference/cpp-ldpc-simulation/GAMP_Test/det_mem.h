#ifndef _DET_MEM_H_
#define _DET_MEM_H_

#include "phy_config.h"


typedef struct
{

	//Pointer for BsP Detector

	size_t**** sIndex;
	float*** alpha;
	float*** beta;
	float*** gamma;   // MIMO_block x nof_ant_bs*2 x slen (using size
	float**** Px; 
	float* sMean;
	float* sVar;

	Cube<float> alpha_;
	Cube<float> beta_;
	Cube<float> gamma_;
	Cube<uword> det_res_idx;

	field<Cube<float>> beta_field;

	/* pointer for MMSE float*/
	float* tmp_conv_intf_mtx_real;
	float* tmp_inv_mtx_real;
	float* eq_channel_mtx_col_real;
	float* mmse_filter_mtx_real;
	float** det_results_real;  // MIMOBlock x Nt2

	Mat<float> det_results_real_;  // Nt2 x MIMOBlock

} mem_det_t;

void mem_det_init(mem_det_t* q);
void mem_det_free(mem_det_t* q);


#endif // !_DET_MEM_H_

