#ifndef _CODING_MEM_H_
#define _CODING_MEM_H_

#include "phy_config.h"


typedef struct {

	float* llr; // size is NOF_CODE_LEN
	int* uhat;
	Cube<float> llr_cube;
	Cube<float>  llr_mmse;
	Col<float> llr_;
	Col<float> uhat_;

} mem_coding_t;

void mem_coding_init(mem_coding_t* q);
void mem_coding_free(mem_coding_t* q);

#endif // !_CODING_MEM_H_






