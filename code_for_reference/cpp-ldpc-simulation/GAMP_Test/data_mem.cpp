#include "data_mem.h"

void mem_data_init(mem_data_t* q)
{
	size_t i;

	q->tx_data = new int [NOF_CODE_LEN];
	q->rx_buffer = new cf_t* [NOF_MIMO_BLCOK];
	q->rx_buffer_real = new float* [NOF_MIMO_BLCOK];
	q->sym_buffer = new cf_t [NOF_CODE_LEN/MODE_TYPE];

	q->tx_data_ = zeros<Col<int>>(NOF_CODE_LEN);
	q->sym_buffer_ = zeros<Mat<cx_float>>(NOF_ANT_BS, NOF_MIMO_BLCOK);
	q->sym_buffer_real_ = zeros<Mat<float>>(2 * NOF_ANT_BS, NOF_MIMO_BLCOK);
	q->rx_buffer_ = zeros<Mat<cx_float>>(NOF_ANT_UE, NOF_MIMO_BLCOK);
	q->rx_buffer_real_ = zeros<Mat<float>>(2*NOF_ANT_UE,NOF_MIMO_BLCOK);
	q->rx_buffer_real_row_ = zeros<Row<float>>(2 * NOF_ANT_UE);


	for (i = 0; i < NOF_MIMO_BLCOK; i++) {

		/*allocate the buffer for the transmitted symbol*/
		q->rx_buffer[i] = new cf_t[NOF_ANT_UE];

		q->rx_buffer_real[i] = new float[2 * NOF_ANT_UE];
	}
}

void mem_data_free(mem_data_t* q)
{
	size_t i;
	for (i = 0; i < NOF_MIMO_BLCOK; i++)
	{
		delete[] q->rx_buffer[i];
		delete[] q->rx_buffer_real[i];
	}

	delete[] q->rx_buffer;
	delete[] q->rx_buffer_real;
	delete[] q->sym_buffer;
	delete[] q->tx_data;

}
