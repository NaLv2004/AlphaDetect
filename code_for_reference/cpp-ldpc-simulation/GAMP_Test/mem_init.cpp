#include "mem_init.h"

void thread_struct_init()
{
	size_t code_length = NOF_CODE_LEN;
	size_t cons_size = pow(2, MODE_TYPE);

	mem_ch_mtx_init(&ch_mtx_struct);
	mem_coding_init(&coding_struct);
	mem_data_init(&data_struct);
	mem_det_init(&det_struct);
}


void thread_struct_free()
{
	mem_ch_mtx_free(&ch_mtx_struct);
	mem_det_free(&det_struct);
	mem_coding_free(&coding_struct);
	mem_data_free(&data_struct);

}


void FillValue(float* array, size_t n, float value)
{
	for (float* ptrstart = array, *ptrend = &array[n]; ptrstart < ptrend; ptrstart++)
	{
		*ptrstart = value;
	}
}

void add_scalar(float* array, size_t length, size_t incx, float scalar)
{
	for (float* ptr = array, *ptrend = &array[length]; ptr < ptrend; ptr = ptr + incx)
	{
		*ptr = *ptr + scalar;
	}
}

// z=alpha*x+beta*y 
// make sure the length of x, y, z vector is same;
void add_axpbyz(float* x, float* y, float* z, float alpha, float beta, size_t inc, size_t length)
{
	for (float* xptr = x, *yptr = y, *zptr = z, *xend = &x[length * inc]; xptr < xend; xptr = xptr + inc, yptr = yptr + inc, zptr = zptr + inc)
	{
		*zptr = alpha * (*xptr) + beta * (*yptr);
	}
}

void vec_copy(int length, float* x, int incx, float* y, int incy)
{
	for (float* xptr = x, *yptr = y, *xend = &x[length * incx]; xptr < xend; xptr = xptr + incx, yptr = yptr + incy)
	{
		*yptr = *xptr;
	}
}

void vec_copy(int length, int* x, int incx, int* y, int incy)
{
	for (int* xptr = x, *yptr = y, *xend = &x[length * incx]; xptr < xend; xptr = xptr + incx, yptr = yptr + incy)
	{
		*yptr = *xptr;
	}
}

//zi=alpha*xi.*yi
void vec_mul_element(int length, float* x, float* y, float* z, float alpha)
{
	for (float* xptr = x, *yptr = y, *zptr = z, *xend = &x[length]; xptr < xend; xptr++, yptr++, zptr++)
	{
		*zptr = alpha * (*xptr) * (*yptr);
	}
}

//zi=alpha*xi./yi
void vec_div_element(int length, float* x, float* y, float* z, float alpha)
{
	for (float* xptr = x, *yptr = y, *zptr = z, *xend = &x[length]; xptr < xend; xptr++, yptr++, zptr++)
	{
		*zptr = alpha * (*xptr) / (*yptr);
	}
}

void vec_logic_element(int length, float* x, float* y, float value)
{
	for (float* xptr = x, *yptr = y, *xend = &x[length]; xptr < xend; xptr++, yptr++)
	{
		*yptr = (*xptr > value) ? 1 : 0;
	}
}

void vec_scalar_element(int length, float* x, float* y, float scalar)
{
	for (float* xptr = x, *yptr = y, *xend = &x[length]; xptr < xend; xptr++, yptr++)
	{
		*yptr = scalar * (*xptr);
	}
}




void mem_ldpc_transmitter_init(mem_ldpc_transmitter_t* q, int A, int B, int L, int C, int K, int K1, int N, int N1, int Nb, int Mb, int Zc, int outlen)
{
	size_t i;

	// Initial check matrix --H
	q->H = new int* [Mb * Zc];
	for (i = 0; i < (Mb * Zc); i++) {

		/*allocate the buffer for the transmitted data*/
		q->H[i] = new int[Nb * Zc];
		if (!(q->H[i])) {
			perror("malloc error for H");
			exit(-1);
		}
	}

	// Initial base graph --BG#
	q->V = new int* [Mb];
	for (i = 0; i < Mb; i++) {

		/*allocate the buffer for the transmitted data*/
		q->V[i] = new int[Nb];
		if (!(q->V[i])) {
			perror("malloc error for each base graph");
			exit(-1);
		}
	}

	// size is cx1
	q->E_array = new int[C];
	if (!(q->E_array))
	{
		perror("malloc error for E_array");
		exit(-1);
	}

	//Initial buffer for transmit block size
	q->a_data_raw = new int8_t[A];
	if (!(q->a_data_raw))
	{
		perror("malloc error for a_data_raw");
		exit(-1);
	}

	//Initial buffer for transmit info + TB-CRC
	q->a_data_crc = new unsigned char[B];

	// size is C x K (including all CB-CRC filler number)
	q->a_data_block = new unsigned char* [C];

	// size is C x K (including all CB-CRC filler number) ?
	q->a_data_block1 = new int8_t * [C];

	// size is C x N1 (everyrow without puncture column) including check data
	q->a_data_ldpc = new int8_t * [C];

	q->a_data_match = new int8_t[outlen]; // transmitable data number

	// size is K x 1  将校验矩阵中每行的信息位转换成行的形式
	q->a_data_col = new int* [K];
	for (i = 0; i < K; i++) {
		q->a_data_col[i] = new int[1];
		if (!(q->a_data_col[i])) {
			perror("malloc error for a_data_col");
			exit(-1);
		}
	}

	// size is  1 x N1 
	q->a_data_ldpc1 = new int* [N1];
	for (i = 0; i < N1; i++) {
		q->a_data_ldpc1[i] = new int[1];
		if (!(q->a_data_ldpc1[i])) {
			perror("malloc error for a_data_ldpc1");
			exit(-1);
		}
	}

	// ≈ C x ((outlen=B/R) /C) 将outlen 分到每个CB中去实现
	q->rate_match_temp = new int8_t * [C];

	for (i = 0; i < C; i++) {

		/*allocate the buffer for the transmitted data*/
		q->a_data_block[i] = new unsigned char[K];
		if (!(q->a_data_block[i])) {
			perror("malloc error for each a_data_block");
			exit(-1);
		}
		q->a_data_block1[i] = new int8_t[K];
		if (!(q->a_data_block1[i])) {
			perror("malloc error for each a_data_block1");
			exit(-1);
		}
		q->a_data_ldpc[i] = new int8_t[N1];
		if (!(q->a_data_ldpc[i])) {
			perror("malloc error for each a_data_ldpc");
			exit(-1);
		}
		q->rate_match_temp[i] = new int8_t[ceil(float(outlen) / C)];//多开了！
		if (!(q->rate_match_temp[i])) {
			perror("malloc error for each rate_match_temp");
			exit(-1);
		}
	}

}

void mem_ldpc_transmitter_free(mem_ldpc_transmitter_t* q, int C, int K, int K1, int N, int N1, int Mb, int Zc)
{
	size_t i;
	for (i = 0; i < C; i++)
	{
		delete[] q->a_data_block[i];
		delete[] q->a_data_block1[i];
		delete[] q->a_data_ldpc[i];
		delete[] q->rate_match_temp[i];
	}
	for (i = 0; i < N1; i++)
	{
		delete[] q->a_data_ldpc1[i];
	}
	for (i = 0; i < K; i++)
	{
		delete[] q->a_data_col[i];
	}
	for (i = 0; i < Mb; i++)
	{
		delete[] q->V[i];
	}
	for (i = 0; i < Mb * Zc; i++)
	{
		delete[] q->H[i];
	}
	delete[] q->a_data_block;
	delete[] q->a_data_block1;
	delete[] q->a_data_ldpc;
	delete[] q->rate_match_temp;
	delete[] q->a_data_ldpc1;
	delete[] q->a_data_col;
	delete[] q->V;
	delete[] q->a_data_match;
	delete[] q->a_data_raw;
	delete[] q->a_data_crc;
	delete[] q->E_array;

}

void mem_ldpc_receiver_init(mem_ldpc_receiver_t* q, int A, int B, int L, int C, int K, int K1, int N, int N1, int Nb, int Mb, int Zc, int outlen)
{
	size_t i;

	q->a_data_y = new float[outlen];

	q->LLR_in = new float[N];

	q->y_array = new float[N];

	q->LLR_out = new float[N];
	q->LLR_out_tmp = new float[N];//
	q->u_out_tmp = new int8_t[N];//
	q->uhat = new int8_t[B];

	q->uhat1 = new int* [C];

	q->uhat2 = new int8_t * [C];

	q->LLR_in_C = new float* [C];

	q->LLR_in_tmp = new float* [C];

	q->deconcatenated = new float* [C];

	for (i = 0; i < C; i++) {

		/*allocate the buffer for the transmitted data*/
		q->uhat1[i] = new int[N];
		if (!(q->uhat1[i])) {
			perror("malloc error for each uhat1");
			exit(-1);
		}
		q->uhat2[i] = new int8_t[K1];
		if (!(q->uhat2[i])) {
			perror("malloc error for each uhat2");
			exit(-1);
		}
		q->LLR_in_C[i] = new float[N];
		if (!(q->LLR_in_C[i])) {
			perror("malloc error for each LLR_in_C");
			exit(-1);
		}
		q->LLR_in_tmp[i] = new float[N1];
		if (!(q->LLR_in_tmp[i])) {
			perror("malloc error for each LLR_in_tmp");
			exit(-1);
		}
		q->deconcatenated[i] = new float[ceil(float(outlen) / C)];
		if (!(q->deconcatenated[i])) {
			perror("malloc error for each deconcatenated");
			exit(-1);
		}
	}

}

void mem_ldpc_receiver_free(mem_ldpc_receiver_t* q, int C)
{
	size_t i;
	for (i = 0; i < C; i++)
	{
		delete[] q->uhat1[i];
		delete[] q->uhat2[i];
		delete[] q->LLR_in_C[i];
		delete[] q->LLR_in_tmp[i];
		delete[] q->deconcatenated[i];
	}
	delete[] q->a_data_y;
	delete[] q->LLR_in;
	delete[] q->y_array;
	delete[] q->LLR_out;
	delete[] q->LLR_out_tmp;
	delete[] q->u_out_tmp;
	delete[] q->uhat;
	delete[] q->uhat1;
	delete[] q->uhat2;
	delete[] q->LLR_in_C;
	delete[] q->LLR_in_tmp;
	delete[] q->deconcatenated;

}

void thread_struct_init_ldpc(int A, int B, int L, int C, int K, int K1, int N, int N1, int Nb, int Mb, int Zc, int outlen)
{

	mem_ldpc_transmitter_init(&ldpc_t_struct, A, B, L, C, K, K1, N, N1, Nb, Mb, Zc, outlen);
	mem_ldpc_receiver_init(&ldpc_r_struct, A, B, L, C, K, K1, N, N1, Nb, Mb, Zc, outlen);
}

void thread_struct_free_ldpc(int C, int K, int K1, int N, int N1, int Mb, int Zc)
{
	mem_ldpc_transmitter_free(&ldpc_t_struct, C, K, K1, N, N1, Mb, Zc);
	mem_ldpc_receiver_free(&ldpc_r_struct, C);

}