#include "scan_mem.h"

void mem_scan_init(mem_scan_t* q)
{
	size_t i;
	q->LLR_R = new float* [STAGE + 1];
	q->LLR_L = new float* [STAGE + 1];
	q->Ma_SC = new float[NOF_CODE_LEN]();
	q->bit_stage = new int[NOF_CODE_LEN + 1]();
	q->fg = new int[STAGE + 1]();

	for (i = 0; i < STAGE + 1; i++)
	{
		q->LLR_R[i] = new float __declspec(align(64))[NOF_CODE_LEN]();
		q->LLR_L[i] = new float __declspec(align(64))[NOF_CODE_LEN]();
	}

}


void mem_scan_free(mem_scan_t* q)
{
	size_t i;
	for (i = 0; i < STAGE + 1; i++)
	{
		delete[] q->LLR_R[i];
		delete[] q->LLR_L[i];
	}

	delete[] q->LLR_L;
	delete[] q->LLR_R;
	delete[] q->Ma_SC;
	delete[] q->bit_stage;
	delete[] q->fg;

}