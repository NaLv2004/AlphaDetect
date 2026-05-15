#include "coding_mem.h"

void mem_coding_init(mem_coding_t* q)
{
	size_t Csym = pow(2, MODE_TYPE / 2);

	q->llr = new float[NOF_CODE_LEN];
	q->uhat = new int[NOF_CODE_LEN];

	q->llr_cube = zeros<Cube<float>>(MODE_TYPE, NOF_ANT_BS, NOF_MIMO_BLCOK);
	q->llr_mmse = zeros<Cube<float>>(2 * NOF_ANT_BS, Csym, NOF_MIMO_BLCOK);
	q->llr_ = zeros<fvec>(NOF_CODE_LEN);	//¼ì²âbitµÄllr   1024
	q->uhat_ = zeros<Col<float>>(NOF_CODE_LEN);
}

void mem_coding_free(mem_coding_t* q)
{
	delete[] q->llr;
	delete[] q->uhat;
}
