#include "det_mem.h"

void mem_det_init(mem_det_t* q)
{
	size_t i, j, s;


	size_t Csym = pow(2, MODE_TYPE / 2);

	q->sMean = new float[2 * NOF_ANT_BS];
	q->sVar = new float[2 * NOF_ANT_BS];

	q->alpha_ = zeros<Cube<float>>(2 * NOF_ANT_BS, Csym, 2 * NOF_ANT_UE);
	q->beta_ = zeros<Cube<float>>(2 * NOF_ANT_BS, Csym, 2 * NOF_ANT_UE);
	q->gamma_ = zeros<Cube<float>>(2 * NOF_ANT_BS, Csym, NOF_MIMO_BLCOK);
	q->det_res_idx = zeros<Cube<uword>>(2 * NOF_ANT_BS, Csym, NOF_MIMO_BLCOK);


	q->beta_field.set_size(NOF_MIMO_BLCOK);
	for (i = 0; i < NOF_MIMO_BLCOK; i++)
		q->beta_field(i) = zeros<Cube<float>>(2 * NOF_ANT_BS, Csym, 2 * NOF_ANT_UE);

	
	q->sIndex = new size_t*** [NOF_MIMO_BLCOK];
	q->Px = new float*** [NOF_MIMO_BLCOK];

	q->alpha = new float** [2 * NOF_ANT_UE];
	q->beta = new float** [2 * NOF_ANT_UE];
	
	q->gamma = new float** [NOF_MIMO_BLCOK];

	for (i = 0; i < NOF_MIMO_BLCOK; i++)
	{
		q->gamma[i] = new float* [2 * NOF_ANT_UE];
		q->Px[i] = new float** [2 * NOF_ANT_UE];
		q->sIndex[i] = new size_t** [2 * NOF_ANT_UE];

		for (j = 0; j < 2 * NOF_ANT_UE; j++)
		{
			q->gamma[i][j] = new float[Csym]();
			q->Px[i][j] = new float* [2 * NOF_ANT_UE];
			q->sIndex[i][j] = new size_t* [2 * NOF_ANT_UE];

			for (s = 0; s < 2 * NOF_ANT_UE; s++)
			{
				q->Px[i][j][s] = new float[Csym]();
				q->sIndex[i][j][s] = new size_t[Csym]();
			}

		}
	}

	for (i = 0; i < 2 * NOF_ANT_UE; i++)
	{
		q->alpha[i] = new float* [2 * NOF_ANT_UE];

		for (j = 0; j < 2 * NOF_ANT_UE; j++)
		{
			q->alpha[i][j] = new float[Csym]();
		}
	}

	for (i = 0; i < 2 * NOF_ANT_UE; i++)
	{
		q->beta[i] = new float* [2 * NOF_ANT_UE];
		for (j = 0; j < 2 * NOF_ANT_UE; j++)
		{
			q->beta[i][j] = new float[Csym]();
		}
	}


	q->tmp_conv_intf_mtx_real = new float[4 * NOF_ANT_UE * NOF_ANT_UE];
	q->tmp_inv_mtx_real = new float[4 * NOF_ANT_UE * NOF_ANT_UE];
	q->eq_channel_mtx_col_real = new float[4 * NOF_ANT_UE];
	q->mmse_filter_mtx_real = new float[4 * NOF_ANT_BS * NOF_ANT_UE];
	q->det_results_real = new float* [NOF_MIMO_BLCOK];

	for (i = 0; i < NOF_MIMO_BLCOK; i++)
	{
		q->det_results_real[i] = new float[2 * NOF_ANT_BS];
	}

	q->det_results_real_ = zeros<Mat<float>>(2 * NOF_ANT_BS, NOF_MIMO_BLCOK);

}

void mem_det_free(mem_det_t* q)
{
	size_t i, j, s;

	for (i = 0; i < NOF_MIMO_BLCOK; i++)
	{
		
		for (j = 0; j < 2 * NOF_ANT_UE; j++)
		{
			delete[] q->gamma[i][j];

			for (s = 0; s < 2 * NOF_ANT_UE; s++)
			{
				delete[] q->Px[i][j][s];
				delete[] q->sIndex[i][j][s];
			}

			delete[] q->Px[i][j];
			delete[] q->sIndex[i][j];
		}
		delete[] q->gamma[i];
		delete[] q->Px[i];
		delete[] q->sIndex[i];
	}

	delete[] q->Px;
	delete[] q->sIndex;
	delete[] q->gamma;


	for (i = 0; i < 2 * NOF_ANT_UE; i++)
	{
		for (j = 0; j < 2 * NOF_ANT_UE; j++)
		{
			delete[] q->alpha[i][j];
			
		}

		delete[] q->alpha[i];
		
	}

	delete[] q->alpha;
	

	for (i = 0; i < 2 * NOF_ANT_UE; i++)
	{
		for (j = 0; j < 2 * NOF_ANT_UE; j++)
		{
			delete[] q->beta[i][j];
		}
		delete[] q->beta[i];
	}

	delete[] q->beta;


	for (i = 0; i < NOF_MIMO_BLCOK; i++)
	{
		delete[] q->det_results_real[i];
	}

	delete[] q->tmp_conv_intf_mtx_real;
	delete[] q->tmp_inv_mtx_real;
	delete[] q->eq_channel_mtx_col_real;
	delete[] q->mmse_filter_mtx_real;
	delete[] q->det_results_real;
	delete[] q->sMean;
	delete[] q->sVar;

}
