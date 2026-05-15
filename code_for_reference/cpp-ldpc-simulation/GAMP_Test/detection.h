#ifndef _DETECTION_H_
#define _DETECTION_H_

#include "phy_config.h"
#include "mem_init.h"
#include "utils.h"

//void EP_symtobit_llr(float* extr_mean, float* extr_var, float* llr_EP, size_t nt, size_t slen_real, size_t cnt);
void CFbsp_nm_mean_rd_idd_arma_llrpro(Mat<float>& H_mtx, Col<float> Rx, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma, Mat<uword>det_res_idx, Mat<float>& mmse_llr, Mat<float>& llr_mtx, size_t Nr, size_t Nt, size_t iterNum, size_t nm, size_t Csym_length, float damp_factor, float sigma2, float clipping_neg_value);

void gai_bp_arma(Mat<float>& H_mtx, Col<float> Rx, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma, Mat<float>& llr_mtx, float* det_results, size_t Nr, size_t Nt, size_t iterNum, size_t Csym_length, float damp_factor, float sigma2);

void bsp_dm1df1_rd_arma(Mat<float>& H_mtx, Col<float> Rx, float* det_results, size_t symbol_length, size_t Nr, size_t Nt, size_t iterNum, float sigma2,
	float delta, Cube<float>& beta, Mat<float>& gamma, Cube<float>& alpha, Mat<float>& llr_mtx);
void bsp_nm_mean_rd_idd_arma_llr(Mat<float>& H_mtx, Col<float> Rx, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma, Mat<uword>det_res_idx, Mat<float>& mmse_llr, Mat<float>& llr_mtx, size_t Nr, size_t Nt, size_t iterNum, size_t nm, size_t Csym_length, float damp_factor, float sigma2, float clipping_neg_value);

void bsp_nm_mean_rd_idd_arma_llrpro(Mat<float>& H_mtx, Col<float> Rx, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma, Mat<uword>det_res_idx, Mat<float>& mmse_llr, Mat<float>& llr_mtx, size_t Nr, size_t Nt, size_t iterNum, size_t nm, size_t Csym_length, float damp_factor, float sigma2, float clipping_neg_value);

void bsp_nm_mean_rd_idd_arma(Mat<float>& H_mtx, Col<float> Rx, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma, Mat<uword>det_res_idx, Mat<float>& llr_mtx, size_t Nr, size_t Nt, size_t iterNum, size_t nm, size_t Csym_length, float damp_factor, float sigma2);

void exbsp_nm_dm1df1_rd_idd_arma(Mat<float>& H_mtx, Col<float> Rx, float* det_results, size_t symbol_length, size_t Nr, size_t Nt, size_t iterNum, size_t nm, float sigma2, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma_arma, Mat<float>& llr_mtx, float damp_factor);

void bsp_mean_dm1df1_rd_idd_arma(Mat<float>& H_mtx, Col<float> Rx, float* det_results, size_t symbol_length, size_t Nr, size_t Nt, size_t iterNum, size_t nm, float sigma2, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma_arma, Mat<float>& llr_mtx, float damp_factor);
//void bsp_mean_dm1df1_rd_idd_arma(Mat<float>& H_mtx, Col<float> Rx, float* det_results, size_t symbol_length, size_t Nr, size_t Nt, size_t iterNum, float sigma2, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma, Mat<float>& llr_mtx, float*** Px, float* sMean, float* sVar, float*** alpha1, size_t*** sIndex);

void bsp_mean_dm1df1_rd_idd(float* H_mtx, float* Rx, float* det_results, float* tmp_inv_mtx, float* llr, size_t symbol_length, size_t Nr, size_t Nt, size_t iterNum, float sigma2, float delta, float*** Px, float* sMean, float* sVar, float*** beta, float** gamma, float*** alpha, size_t*** sIndex);

void bsp_dm1df1_rd(float* H_mtx, float* Rx, float* det_results, float* tmp_inv_mtx, float* llr, size_t symbol_length, size_t Nr, size_t Nt, size_t iterNum, float sigma2, float delta, float*** beta, float** gamma, float*** alpha, size_t* ems_sym);

void mmse_symtobit_llr(Col<float>& mmse_res, Mat<float>& llr_mtx, int MODETYPE, int Nt);

void mmse_detection_float(uint32_t Nt, uint32_t Nr, cf_t* H_mtx, cf_t* Rx, cf_t* tmp_inv_mtx, cf_t* tmp_conv_intf_mtx, cf_t* eq_channel_mtx_col, cf_t* mmse_filter_mtx, cf_t* det_results, float sigma2);

void mmse_detection_float(size_t Nt, size_t Nr, float* H_mtx, float* Rx, float* tmp_inv_mtx, float* tmp_conv_intf_mtx, float* eq_channel_mtx_col, float* mmse_filter_mtx, float* det_results, float sigma2);

#endif // !_DETECTION_H_

