#ifndef _CODING_POLAR_H_
#define _CODING_POLAR_H_

#include "phy_config.h"

typedef struct {

	uint32_t* channel_Pe; // temp; channel quality
	uint32_t* Info_bit; // A; infomation bits index;
	uint32_t* Info_CRC; // A_CRC; CRC bit index
	uint32_t* frozen_bit; // Ac; frozen bit index
	int* frozen_index; // A_Ac; frozen bit index in source bit vector u;

	unsigned char* a_data; // // Information CodeWord
	unsigned char* u_crc;
	int8_t* u; // source bit u;
	int8_t* x; // codeword x;

	

} mem_polarcoding_t;

void mem_polarcoding_init(mem_polarcoding_t* q);
void mem_polarcoding_free(mem_polarcoding_t* q);

#endif
