#include "ch_mem.h"

void mem_ch_mtx_init(mem_ch_mtx_t* q)
{
	size_t i;
	q->H_mtx = new cf_t * [NOF_MIMO_BLCOK];
	q->H_real = new float * [NOF_MIMO_BLCOK];

	for (i = 0; i < NOF_MIMO_BLCOK; i++)
	{
		q->H_mtx[i] = new cf_t[NOF_ANT_BS * NOF_ANT_UE]();
		q->H_real[i] = new float[2 * NOF_ANT_UE * 2 * NOF_ANT_BS]();
	}
	q->H_real_ = zeros<Cube<float>>(2 * NOF_ANT_UE, 2 * NOF_ANT_BS, NOF_MIMO_BLCOK);
	q->H_flat_real = zeros<Row<float>>(2 * NOF_ANT_UE * 2 * NOF_ANT_BS);
	q->H_mtx_ = zeros<Cube<cx_float>>(NOF_ANT_UE, NOF_ANT_BS, NOF_MIMO_BLCOK);

}

void mem_ch_mtx_free(mem_ch_mtx_t* q)
{
	size_t i;

	for (i = 0; i < NOF_MIMO_BLCOK; i++)
	{
		delete[] q->H_real[i];
		delete[] q->H_mtx[i];
	}

	delete[] q->H_mtx;
	delete[] q->H_real;
}
