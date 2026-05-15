#include "utils.h"
#include "phy_config.h"

double sampleNormal() {
	double p = ((double)rand() / (RAND_MAX)) * 2 - 1;
	double v = ((double)rand() / (RAND_MAX)) * 2 - 1;
	double r = p * p + v * v;
	if (r == 0 || r > 1) return sampleNormal();
	double c = sqrt(-2 * log(r) / r);
	return p * c;
}

double rand_gen() {
	// return a uniformly distributed random value
	return ((double)(rand()) + 1.) / ((double)(RAND_MAX)+1.);
}
double normalRandom() {
	// return a normally distributed random value
	double v1 = rand_gen();
	double v2 = rand_gen();
	return cos(2 * 3.14 * v2) * sqrt(-2. * log(v1));
}


cf_t fmul(float a, cf_t b)
{
	cf_t c;
	c.real = a * b.real;
	c.imag = a * b.imag;
	return c;
}


cf_t cadd(cf_t a, cf_t b)
{
	cf_t c;
	c.real = a.real + b.real;
	c.imag = a.imag + b.imag;
	return c;
}

cf_t cmul(cf_t a, cf_t b)
{
	cf_t c;
	c.real = a.real * b.real - a.imag * b.imag;
	c.imag = a.real * b.imag + a.imag * b.real;
	return c;
}

cf_t cmul(cf_t* a, cf_t* b, size_t num)
{
	cf_t c = { 0,0 };

	for (size_t i = 0; i < num; i++)
	{
		c = cadd(c, cmul(a[i], b[i]));
	}

	return c;
}

cf_t* cmul(cf_t* a, cf_t* b, size_t row, size_t col) // 如何做安全性检测，其实是个问题；
{
	cf_t* c = new cf_t[row]();

	for (size_t i = 0; i < row; i++)
	{
		for (size_t j = 0; j < col; j++)
		{
			c[i] = cadd(c[i], cmul(a[i * col + j], b[j]));
		}

	}
	return c;
}


cf_t csub(cf_t a, cf_t b)
{
	cf_t c;
	c.real = a.real - b.real;
	c.imag = a.imag - b.imag;
	return c;
}

cf_t* csub(cf_t* a, cf_t* b, size_t num)
{
	cf_t* c = new cf_t[num];
	for (size_t i = 0; i < num; i++)
	{
		c[i].real = a[i].real - b[i].real;
		c[i].imag = a[i].imag - b[i].imag;
	}

	return c;
}


float cnorm2(cf_t a)
{
	float c = a.real * a.real + a.imag * a.imag;
	return c;
}

float cnorm2(cf_t* a, size_t b)
{
	float c = 0;
	for (size_t i = 0; i < b; i++)
	{
		c += (a[i].real * a[i].real + a[i].imag * a[i].imag);
	}
	return c;
}


size_t* sortArray(const float* PM, size_t num)
{
	int i, j;

	float* PM_tmp = new float[num];
	memcpy(PM_tmp, PM, sizeof(float) * num);

	size_t* index = new size_t[num];

	for (i = 0; i < num; i++)
	{
		index[i] = i;
	}

	float tmp = 0;
	size_t idx = 0;
	for (i = 1; i < num; i++)
	{
		tmp = PM_tmp[i];
		idx = index[i];
		for (j = i - 1; j >= 0; j--)
		{
			if (PM_tmp[j] < tmp) // < for descend sort; > for a ascend sort; 
			{
				PM_tmp[j + 1] = PM_tmp[j];
				index[j + 1] = index[j];
			}
			else
			{
				break;
			}
		}
		PM_tmp[j + 1] = tmp;
		index[j + 1] = idx;
	}

	delete[] PM_tmp;

	return index;
}




void inverseMatrix(cf_t* matrix, int dim)
{
	int* ivpv = (int*)malloc(sizeof(int) * dim);

	if (!LAPACKE_cgetrf(LAPACK_ROW_MAJOR, dim, dim, matrix, dim, ivpv))
		if (!LAPACKE_cgetri(LAPACK_ROW_MAJOR, dim, matrix, dim, ivpv));
		else
			printf("Inverse is failed!\n");
			/*debug_info.utl_inv_fail++;*/
	else
		printf("LU decompose is failed!\n");
		//debug_info.utl_LU_decomp_fail++;
	if (ivpv != NULL) free(ivpv);
}


void inverseMatrix(float* pSrc, int dim)
{
	int* ivpv = new int [sizeof(int) * dim];
	float* pSrcBak = new float[dim * dim];  // LAPACKE_sgesv会覆盖A矩阵，因而将pSrc备份
	memset(pSrcBak, 0, sizeof(float) * dim * dim);
	for (int i = 0; i < dim; ++i)
	{
		// LAPACKE_sgesv函数计算AX=B，当B为单位矩阵时，X为inv(A)
		pSrcBak[i * (dim + 1)] = 1.0f;
	}

	size_t flag = LAPACKE_sgesv(LAPACK_ROW_MAJOR, dim, dim, pSrc, dim, ivpv, pSrcBak, dim);
	// 调用LAPACKE_sgesv后，会将inv(A)覆盖到X（即pDst）
	if (flag)
	{
		printf("Inverse is failed!\n");
	}

	memcpy(pSrc, pSrcBak, sizeof(float) * dim * dim);


	delete[] ivpv;
	ivpv = nullptr;

	delete[] pSrcBak;
	pSrcBak = nullptr;
}
