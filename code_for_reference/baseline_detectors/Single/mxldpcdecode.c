/*==========================================================
 * mxldpcdecode.c - example in MATLAB External Interfaces
 *
 *
 * The calling syntax is:
 *
 *		[bm_raw, Lms_out] = mxldpcdecode(bit_llr, Lms, scaling, offset, iter, index)
 *
 * This is a MEX-file for MATLAB.
 *
 * zxc 出品
 *========================================================*/

#include "mex.h"

/* The computational routine */
void arrayProduct(double x, double *y, double *z, mwSize n)
{
    mwSize i;
    /* multiply each element y by x */
    for (i=0; i<n; i++) {
        z[i] = x * y[i];
    }
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double multiplier;              /* input scalar */
    double *inMatrix;               /* 1xN input matrix */
    size_t ncols;                   /* size of matrix */
    double *outMatrix;              /* output matrix */

    double *bit_llr;
    double *Lms;
    double scaling;
    double offset;
    int iter;
    int* index

    double *bm_raw;
    double *Lms_out

    size_t check_num;
    size_t bit_num;
    size_t degree;


    // 输入
    bit_llr = mxGetDoubles(prhs[0]);
    Lms = mxGetDoubles(prhs[1]);
    scaling = mxGetScalar(prhs[2]);
    offset = mxGetScalar(prhs[3]);
    iter = mxGetScalar(prhs[4]);
    index = mxGetInt8s(prhs[5]);
    
    // 尺寸
    check_num=mxGetM(prhs[1]);
    degree=mxGetN(prhs[1]);
    bit_num= mxGetM(prhs[0]);

    // 输出
    plhs[0] = mxCreateDoubleMatrix(bit_num,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(check_num,degree,mxREAL);

    bm_raw = mxGetDoubles(plhs[0]);
    Lms_out = mxGetDoubles(plhs[1]);


}
