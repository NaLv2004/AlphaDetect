#ifndef _CH_MEM_H_
#define _CH_MEM_H_

#include "phy_config.h"

typedef struct {
	cf_t** H_mtx;
	float** H_real;
	Cube<float> H_real_;
	Cube<cx_float>H_mtx_;
	Row<float> H_flat_real;
} mem_ch_mtx_t;

void mem_ch_mtx_init(mem_ch_mtx_t* q);
void mem_ch_mtx_free(mem_ch_mtx_t* q);




#endif // !_CH_MEM_H_



