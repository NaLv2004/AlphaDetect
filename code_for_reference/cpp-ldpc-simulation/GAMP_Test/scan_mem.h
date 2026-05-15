#ifndef _CODING_SCAN_H_
#define _CODING_SCAN_H_

#include "phy_config.h"

typedef struct {

	float** LLR_L; // 
	float** LLR_R; // 
	float* Ma_SC; // 
	int* bit_stage; // 
	int* fg;

} mem_scan_t;

void mem_scan_init(mem_scan_t* q);
void mem_scan_free(mem_scan_t* q);




#endif
