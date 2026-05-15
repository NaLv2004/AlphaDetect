#include "polar_coding.h"

void mem_polarcoding_init(mem_polarcoding_t* q)
{
	q->channel_Pe = new uint32_t[NOF_CODE_LEN]();
	q->Info_bit = new uint32_t[NOF_INFO_LEN]();
	q->Info_CRC = new uint32_t[NOF_CRC_LEN]();
	q->frozen_bit = new uint32_t[NOF_CODE_LEN-NOF_DATA_LEN]();
	q->frozen_index = new int[NOF_CODE_LEN]();

	q->a_data = new unsigned char[NOF_DATA_LEN]();
	q->u_crc = new unsigned char[NOF_DATA_LEN]();
	q->u = new int8_t[NOF_CODE_LEN]();
	q->x = new int8_t[NOF_CODE_LEN]();

}


void mem_polarcoding_free(mem_polarcoding_t* q)
{
	delete[] q->channel_Pe;
	delete[] q->Info_bit;
	delete[] q->Info_CRC;
	delete[] q->frozen_bit;
	delete[] q->frozen_index;
	delete[] q->a_data;
	delete[] q->u_crc;
	delete[] q->u;
	delete[] q->x;
}