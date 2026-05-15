#ifndef _DATA_MEM_H_
#define _DATA_MEM_H_

#include "phy_config.h"

typedef struct {

	int* tx_data;  // size is NOF_CODE_LEN
	cf_t* sym_buffer;
	cf_t** rx_buffer;
	float** rx_buffer_real;

	Col<int> tx_data_;
	Mat<cx_float> sym_buffer_;
	Mat<cx_float> rx_buffer_;
	Mat<float> sym_buffer_real_;
	Mat<float> rx_buffer_real_;
	Row<float> rx_buffer_real_row_;

} mem_data_t;

void mem_data_init(mem_data_t* q);
void mem_data_free(mem_data_t* q);


#endif // !_DATA_MEM_H_






