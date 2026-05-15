#ifndef _UTILS_H_
#define _UTILS_H_

#include "phy_config.h"

double sampleNormal();

double normalRandom();

cf_t fmul(float a, cf_t b);

cf_t cadd(cf_t a, cf_t b);

cf_t cmul(cf_t a, cf_t b);

cf_t cmul(cf_t* a, cf_t* b, size_t num);

cf_t* cmul(cf_t* a, cf_t* b, size_t row, size_t col);

cf_t csub(cf_t a, cf_t b);

cf_t* csub(cf_t* a, cf_t* b, size_t num);

float cnorm2(cf_t a);

float cnorm2(cf_t* a, size_t b);


size_t* sortArray(const float* PM, size_t num);
void inverseMatrix(cf_t* matrix, int dim);
void inverseMatrix(float* pSrc, int dim);
#endif // !_UTILS_H_






