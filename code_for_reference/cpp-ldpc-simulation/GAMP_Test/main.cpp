#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <random>
#include <cstdlib>


#include "mkl.h"
#include "modem.h"
#include "main.h"
#include <math.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include "detection.h"
#include "utils.h"
#include "phy_config.h"
#include "mem_init.h"
#include "mem_struct.h"
#include "crc.h"

#include "LDPCfunctions.h"
#include "encode.h"
#include "ratematch.h"
#include "decode.h"
using namespace std;
//#define cf_t			MKL_Complex8	/* complex float, 8 Bytes */

ofstream y_out("result.txt");

int main()
{
	cout << "5G LDPC Coded MIMO platform copyright(c) Yifei Shen & Wenyue Zhou & Zeqiong Tan - LEADS - SEU " << endl;
	cout << "BsP + 5G LDPC Based Armadillo Library " << endl;  // initial all 

	float minSNR, maxSNR, stepSNR;
	cout << "Enter min_SNR: ";
	cin >> minSNR;
	cout << "Enter max_SNR: ";
	cin >> maxSNR;
	cout << "Enter step_SNR: ";
	cin >> stepSNR;

	// Parameters for MIMO 
	size_t symbol_length = MODE_TYPE;
	size_t symbol_length_real = MODE_TYPE / 2;
	size_t Nr = NOF_ANT_UE;
	size_t Nt = NOF_ANT_BS;
	size_t Nr2 = 2*Nr;   
	size_t Nt2 = 2*Nt;  
	size_t nof_BitPerBlock = Nt * symbol_length;
	size_t Csym_len = pow(2, symbol_length);
	size_t Csym_len_real = pow(2, symbol_length_real); // 16
	size_t iterMIMO = 3;
	size_t nm = 4;
	size_t ex_nm = 4;
	size_t random_num = 624;
	size_t isCorr = 0;  // 0 -- Rayleigh Channel  1 -- WINNERII Channel
	float damp_factor = 0;
	//float clipping_neg_value = -100;

	// Other parameters
	size_t CodeLength = NOF_CODE_LEN;  // 1024
	size_t cntBlk = NOF_MIMO_BLCOK;
	size_t iterALL = 1;
	size_t i, j, cnt, s, iter, kk;
	size_t itweave_row = 90;
	
	
	srand((unsigned int)time(0));
	//arma_rng::set_seed_random();
	thread_struct_init();

	/********************generate WINNERII channel*************************/
#if 1
	Cube<cx_float> H_winner2(NOF_ANT_UE, NOF_ANT_BS, random_num, fill::zeros);
	if (isCorr == 1)
	{
		Mat<float> Hrealtmp, Himagtmp;
		//Hrealtmp.load("3GPPChannel_real.txt"); // 128 x 57
		//Himagtmp.load("3GPPChannel_imag.txt"); //
		//Hrealtmp.load("H_WINNERII_real.txt");  // 128 x 64
		//Himagtmp.load("H_WINNERII_imag.txt");
		//Hrealtmp.load("H_WINNERII_real_32_12.txt");  // 32 x 12
		//Himagtmp.load("H_WINNERII_imag_32_12.txt");
		Hrealtmp.load("H_platform_real.txt");  // 32 x 12
		Himagtmp.load("H_platform_imag.txt");
		for (i = 0; i < random_num; i++)
		{
			H_winner2.slice(i).set_real(Hrealtmp.rows(i * Nr, (i + 1) * Nr - 1));
			H_winner2.slice(i).set_imag(Himagtmp.rows(i * Nr, (i + 1) * Nr - 1));
		}
	}

#endif // 0


#if 1
	int A;		// A--transmit block size
	float codeRate;	
	int L;		// L--length of TB-CRC
	int B;      // bitNum within TB-CRC
	int bgn;    // type of LDPC Base Graph
	int iter_decoding = 25; // iteration number of LDPC Decoding


	A = NOF_INFO_LEN;
	codeRate = CODE_RATE;
	cout << "A: " << A << "\t" << "codeRate: " << codeRate << endl;

	// select TB-CRC BitLength
	if (A > 3824) L = 24;  //CRC Length
	else L = 16;
	B = A + L;

	// select LDPC base graph 
	if (A < 293 || (A < 3825 && codeRate <= 0.67) || codeRate <= 0.25) bgn = 2;
	else bgn = 1;



	/**********************************************************/
	int Kcb;			//the maximum code block size  
	int Kb;				//number of used message block size in BaseGraph
	int K;				//size of each code block (including CB-CRC bits and filler bits)
	int K_CB_Bit;	    //number of bits in each code block (including CB-CRC excluding filler bits)
	int C;				//Number of code block segments
	int B_CB_Bit;				// B1=B + C * L
	//Total number of bits in each code block (including CRC, CB-CRC and filler bits 
	int Zc = 0;			//Selected lifting size
	int NumFiller_cb;				//Number of filler bits in each code block
	int outlen = (int)ceil(float(A) / float(codeRate));  //used for Rating Matching   768+16)/0.766=1024

	//int cbz;			//Number of info-bits in each code block (excluding CB-CRC bits and filler bits)

	int Zlist[51] = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384 };

	//select the maximum code block size
	if (bgn == 1) Kcb = 8448;
	else Kcb = 3840;

	//calculate segment number C
	if (B < Kcb)
	{
		C = 1;
		B_CB_Bit = B;
	}
	else
	{
		C = (int)ceil(float(B) / float(Kcb - L));  // 最少需要的CBNum
		B_CB_Bit = B + C * L;
	}

	//Get number of bits in each code block(including CB-CRC but excluding filler bits)
	K_CB_Bit = ceil(float(B_CB_Bit) / float(C));

	//select the number of used block size
	if (bgn == 1)
	{
		Kb = 22;
	}
	else
	{
		if (B > 640)
		{
			Kb = 10;
		}
		else if (B > 560)
		{
			Kb = 9;
		}
		else if (B > 192)
		{
			Kb = 8;
		}
		else Kb = 6;
	}
 
	// select the lifting size
	int ii = 0;
	while (Kb * Zc < K_CB_Bit)
	{
		Zc = Zlist[ii];
		ii++;
	}
	if (bgn == 1) K = 22 * Zc;
	else K = 10 * Zc;			 // 信息位的总长度
	NumFiller_cb = K - K_CB_Bit; // fillerbits number


	/**********************set parameters about check matrix***************************/
	int Qm = 1;//modulation
	int N;     //Column number of check matrix
	int Nb;    //Column number of base graph
	int M;	   //Row number of check matrix
	int Mb;    //Row number of base graph
	if (bgn == 2)
	{
		Nb = 52;
		Mb = 42;
		N = 52 * Zc;
		M = 42 * Zc;
	}
	else
	{
		Nb = 68;
		Mb = 46;
		N = 68 * Zc;
		M = 46 * Zc;
	}

	cout <<"lifting size is: "<< Zc << endl;
	int N_Punctured = N - 2 * Zc;   //excluding puncture  coding bitlength
	int Num_Error = 0;
	int Num_Frame_Error = 0;
	int Zsets[8][8] = {    // used for building check matrix
	{2,4,8,16,32,64,128,256},
	{3,6,12,24,48,96,192,384},
	{5,10,20,40,80,160,320,0},
	{7,14,28,56,112,224,0,0},
	{9,18,36,72,144,288,0,0},
	{11,22,44,88,176,352,0,0},
	{13,26,52,104,208,0,0,0},
	{15,30,60,120,240,0,0,0}
	};

	int setIdx;
	// find Vij
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			if (Zc == Zsets[i][j]) {
				setIdx = i + 1;
			}
		}
	}
#endif
	thread_struct_init_ldpc(A, B, L, C, K, K_CB_Bit, N, N_Punctured, Nb, Mb, Zc, outlen);

	//build check matrix H(M x N) and base graph V(Mb x Nb)
	getH(ldpc_t_struct.V, ldpc_t_struct.H, bgn, Zc, setIdx);  

#if 0
	fstream file;
	file.open("5G_1024_0667.txt", ios::out);
	for (int i = 0; i < M; i++)//check
	{
		for (int j = 0; j < N; j++)
			file << ldpc_t_struct.H[i][j] << " ";
		file << endl;
	}
#endif
	
	/*********parameters used in LDPC Decodingand function preparation************/ 
	
	// CNtoVN:	M x Mi - none zero index in H
	// LLR_CNtoVN:	M x Mi  - message from CN to VN
	// LLR_VN : M x 1 - message of every VN, used to iterative temp and output 
	vector<std::vector<int>> CNtoVN(M, std::vector<int>(0));		 //size is M x dimension(dc) keep none-zero elements' index
	vector<std::vector<float>> LLR_CNtoVN(M, std::vector<float>(0)); // -- reserve edges' llr size is M x Mi
	vector<float> LLR_VN(N, 0);										 // reserve llr of variable node ;initial length is N ; value is zero
	for (int k = 0; k < M; k++)
	{
		for (int i = 0; i < N; i++)
		{
			if (ldpc_t_struct.H[k][i] == 1)
				CNtoVN[k].push_back(i);
		}
		LLR_CNtoVN[k].resize(CNtoVN[k].size());
	} 

 
	int k0 = 0; // rate matching starting position
	// preparation:
	// 1. calculate buffer of rate matching in every CB an
	// 2. calculate rate matching starting position
	RateMatchLDPC_pre(ldpc_t_struct.E_array, k0, N, C, Zc, bgn, outlen, 0, Qm, 1);

	// set buffer for every part
	encode_pre(K, Zc, bgn);
	cout << "outlen is "<<outlen << endl;

	/**********************************************************/
	size_t start, finish;
	double duration = 0;

	for (float SNR = minSNR; SNR <= maxSNR; SNR += stepSNR)
	{
		float stmp = pow(10, SNR / 10) * symbol_length * codeRate * Nt;
		float sigma2 = Nt * Nr / stmp;
		//float SNR_indB = pow(10, -SNR / 10) / NOF_ANT_UE / codeRate;
		size_t Num_Error, Num_Frame_Error, Max_frame, Num_Error1, Num_Frame_Error1;
		double Total_BER, Total_BER1;
		Num_Error = 0; Num_Error1 = 0;
		Num_Frame_Error = 0; Num_Frame_Error1 = 0;
		Max_frame = 0;
		Total_BER = 0; Total_BER1 = 0;
		duration = 0;
		Col<uword> flagblk(cntBlk, fill::zeros);
		int flag = 1;

		int Sample_Num = 1000;
		int thres_frame = 100;
		
		while (Num_Frame_Error < thres_frame)
		{
			Max_frame++;

			// generate original bits
			for (int i = 0; i < A; i++)
			{
				ldpc_t_struct.a_data_raw[i] = (int8_t)(rand() % 2); //generate raw information
				ldpc_t_struct.a_data_crc[i] = (unsigned char)ldpc_t_struct.a_data_raw[i];
			}

			//add crc to transmit bit
			tx_append_crc(ldpc_t_struct.a_data_crc, A, L, 1);			

			//LDPC_segment bit operation
			// a_data_block <unsigned char> used to add crc
			// a_data_block1 <int8_t> used to fill the filler bits -1
			// if C==1 block struct -> tb-data + tb-crc + fillerbit
			// if c!=1 block struct -> cb-data(consists of tb-data + tb-crc) + cb-crc + fillerbit
			SegmentLDPC(C, K, K_CB_Bit, L, ldpc_t_struct.a_data_crc, ldpc_t_struct.a_data_block, ldpc_t_struct.a_data_block1);

			//LDPC_encoding
			// the encoding length is N1 without puncture column;
			// input ->  a_data_block1 CB(C x K) 
			//output ->  a_data_ldpc   coded-CB (C x N1)
			for (int i = 0; i < C; i++) {
				for (int j = 0; j < K; j++) {
					ldpc_t_struct.a_data_col[j][0] = int(ldpc_t_struct.a_data_block1[i][j]);
				}
				nrLdpcEncode(ldpc_t_struct.a_data_col, K, bgn, ldpc_t_struct.a_data_ldpc1);
				for (int j = 0; j < N_Punctured; j++) {
					ldpc_t_struct.a_data_ldpc[i][j] = ldpc_t_struct.a_data_ldpc1[j][0];
				}
			}

			//LDPC_ratematching
			//input ->  a_data_ldpc   coded-CB (C x N1)
			//output -> a_data_match  outlen x 1        
			modified_RateMatchLDPC(ldpc_t_struct.a_data_ldpc, ldpc_t_struct.a_data_match, ldpc_t_struct.rate_match_temp, ldpc_t_struct.E_array, k0, N_Punctured, C, Zc, bgn, outlen, 0, Qm);





			// type conversion
			for (int i = 0; i < outlen; i++)
			{
				data_struct.tx_data[i] = ldpc_t_struct.a_data_match[i]; // outlen x 1 
			}

			/*---------test for interweave--------*/
			//cout << "before interweave" << endl;
			//for (int txi = 0; txi < outlen;  txi++)
			//{
			//	cout << data_struct.tx_data[txi] << " ";
			//	if (txi != 0 && txi % 32 == 0)
			//		cout << endl;
			//}

			// interweave
			// input: a_data_match
			// output: tx_data
			
			Mat<int> weaveMat(&data_struct.tx_data[0], itweave_row, outlen / itweave_row);  // save in column-way
			//weaveMat.print("weaveMat is");
			//cout << endl << "after interweave" << endl;
			Col<int> weaveCol = vectorise(trans(weaveMat));
			//data_struct.tx_data = weaveCol.memptr();
			for (int i = 0; i < outlen; i++)
			{
				data_struct.tx_data[i] = weaveCol(i); // outlen x 1 
			}

			//for (int txi = 0; txi < outlen; txi++)
			//{
			//	cout << data_struct.tx_data[txi] << " ";
			//	if (txi!=0 && txi % 32 == 0)
			//		cout << endl;
			//}
			
			// after zero-padding, puncture both filler bits and useless parity bits
			modulation(data_struct.tx_data, data_struct.sym_buffer, MODE_TYPE, CodeLength / MODE_TYPE);

			//reserve MKLComplex_8 to cx_float
			Col<float> sym_buffer_temp(&data_struct.sym_buffer[0].real, 2 * NOF_CODE_LEN / MODE_TYPE);
			Mat<float> sym_buffer_temp_mtx = reshape(sym_buffer_temp, 2, NOF_CODE_LEN / MODE_TYPE);
			Col<cx_float>sym_buffer_temp_cx(NOF_CODE_LEN / MODE_TYPE);
			sym_buffer_temp_cx.set_real(sym_buffer_temp_mtx.row(0).t());
			sym_buffer_temp_cx.set_imag(sym_buffer_temp_mtx.row(1).t());
			data_struct.sym_buffer_ = reshape(sym_buffer_temp_cx, NOF_ANT_BS, NOF_MIMO_BLCOK);

			data_struct.rx_buffer_real_.zeros();

			//complex to real
			data_struct.sym_buffer_real_.rows(0, Nt - 1) = real(data_struct.sym_buffer_); // 
			data_struct.sym_buffer_real_.rows(Nt, Nt2 - 1) = imag(data_struct.sym_buffer_);


			
#if 1		

			if (isCorr == 0)
			{
				Cube<float> H_real(NOF_ANT_UE, NOF_ANT_BS, NOF_MIMO_BLCOK);
				Cube<float> H_imag(NOF_ANT_UE, NOF_ANT_BS, NOF_MIMO_BLCOK);
				H_real = sqrt(0.5) * randn<Cube<float>>(NOF_ANT_UE, NOF_ANT_BS, NOF_MIMO_BLCOK);
				H_imag = sqrt(0.5) * randn<Cube<float>>(NOF_ANT_UE, NOF_ANT_BS, NOF_MIMO_BLCOK);
				ch_mtx_struct.H_mtx_.set_real(H_real);
				ch_mtx_struct.H_mtx_.set_imag(H_imag);
			}
			else
			{
				for (cnt = 0; cnt < cntBlk; cnt++)
				{
					int random_slice = randi(distr_param(0, random_num-1));
					ch_mtx_struct.H_mtx_.slice(cnt) = H_winner2.slice(random_slice);

				}
			}


			for (cnt = 0; cnt < cntBlk; cnt++)
			{
				data_struct.rx_buffer_.col(cnt) = ch_mtx_struct.H_mtx_.slice(cnt) * data_struct.sym_buffer_.col(cnt) + sqrt(0.5) * (randn<Col<cx_float>>(NOF_ANT_UE, distr_param(0.0, sqrt(sigma2))));

				data_struct.rx_buffer_real_(span(0, Nr - 1), cnt) = real(data_struct.rx_buffer_.col(cnt));
				data_struct.rx_buffer_real_(span(Nr, Nr2 - 1), cnt) = imag(data_struct.rx_buffer_.col(cnt));

				ch_mtx_struct.H_real_.slice(cnt)(span(0, Nr - 1), span(0, Nt - 1)) = real(ch_mtx_struct.H_mtx_.slice(cnt));
				ch_mtx_struct.H_real_.slice(cnt)(span(0, Nr - 1), span(Nt, Nt2 - 1)) = -imag(ch_mtx_struct.H_mtx_.slice(cnt));
				ch_mtx_struct.H_real_.slice(cnt)(span(Nr, Nr2 - 1), span(0, Nt - 1)) = imag(ch_mtx_struct.H_mtx_.slice(cnt));
				ch_mtx_struct.H_real_.slice(cnt)(span(Nr, Nr2 - 1), span(Nt, Nt2 - 1)) = real(ch_mtx_struct.H_mtx_.slice(cnt));
			}
#endif

			start = clock();
			
			//initialize the parameters every SNR iteration
			Row<float> cons_row_temp = cons_row;
			det_struct.gamma_.zeros();
			det_struct.alpha_.zeros();
			for (i = 0; i < cntBlk; i++)
				det_struct.beta_field(i).zeros();

 
 			// MMSE detector to active the BsP parameters in the first round of IDD
			for (cnt = 0; cnt < cntBlk; cnt++)
			{
				// input is armadillo data type
				//H_real matrix and Rx from col to Row  (RowMajor

				ch_mtx_struct.H_flat_real = ch_mtx_struct.H_real_.slice(cnt).as_row();
				data_struct.rx_buffer_real_row_ = data_struct.rx_buffer_real_.col(cnt).t();
				mmse_detection_float(Nt2, Nr2, &ch_mtx_struct.H_flat_real(0), &data_struct.rx_buffer_real_row_(0), det_struct.tmp_inv_mtx_real, det_struct.tmp_conv_intf_mtx_real, det_struct.eq_channel_mtx_col_real,
					det_struct.mmse_filter_mtx_real, det_struct.det_results_real[cnt], sigma2);				

				// Initialize the Px begin the first iteration. Init the gamma messages.
				Col<float> det_res_tmp(det_struct.det_results_real[cnt], Nt2);

				Mat<float> det_res_dis(Nt2, Csym_len_real);	   // tensor the initial result to all constellation
				det_res_dis.each_col() = det_res_tmp;   //reserve the det_results

				det_res_dis.each_row([&cons_row_temp](Row<float>& a) { a = 1 * abs(a - cons_row_temp); }); // calculate the distance between initial symbol and constellation


				for (i = 0; i < Nt2; i++)
				{
					det_struct.det_res_idx.slice(cnt).row(i) = trans(sort_index(det_res_dis.row(i), "acsend"));
				}


				mmse_symtobit_llr(det_res_tmp, coding_struct.llr_cube.slice(cnt), MODE_TYPE, Nt);	

#if 0				
				
				//协方差 covariance
				Mat<float> K_mmse = inv(ch_mtx_struct.H_real_.slice(cnt).t() * ch_mtx_struct.H_real_.slice(cnt) + sigma2 * eye<Mat<float>>(Nt2, Nt2));
				//tensor
				Mat<float> det_res_mat = repmat(det_res_tmp, 1, Csym_len_real);
				Mat<float> K_mmse_mat = repmat(K_mmse.diag(), 1, Csym_len_real);
				Mat<float> cons_row_mat = repmat(cons_row, Nt2, 1);

				Mat<float> symProb = -1 * square(abs(cons_row_mat - det_res_mat)) / (2 * abs(K_mmse_mat));
				Mat<float> symProb_mu0 = repmat(symProb.col(0), 1, Csym_len_real);
				coding_struct.llr_mmse.slice(cnt) = symProb - symProb_mu0;



					
				Col<uword> s_index(Nt2);							   // reserve the most likely symbol index
				s_index = det_struct.det_res_idx.slice(cnt).col(0);
				Col<uword> gamma_row_index = linspace<Col<uword>>(0, Nt2 - 1, Nt2); // turn the min_row_index into min_ele_index
				s_index = s_index * Nt2 + gamma_row_index;
				det_struct.gamma_.slice(cnt).elem(s_index).fill(7);
						
#endif
			}
			//cout << coding_struct.llr_mmse.has_nan() << endl;
			flag = 1;

			for (iter = 0; iter < iterALL; iter++)
			{

 				
				coding_struct.llr_ = vectorise(coding_struct.llr_cube); //cube to vector

				if (flag == 1)
				{
					coding_struct.llr_ = 32 * coding_struct.llr_;
				}


				// deinterweave
				// input: coding_struct.llr_
				// output: coding_struct.llr_

				Mat<float> deweaveMat = reshape(coding_struct.llr_, outlen / itweave_row, itweave_row); // save in column-way
				coding_struct.llr_ = vectorise(trans(deweaveMat));



				for (i = 0; i < outlen; i++)
				{
					if (iter > 0) {
						coding_struct.llr_(i) += 0.00 * ldpc_r_struct.LLR_out_tmp[i];  // adding the last LDPC results
					}

					coding_struct.llr_(i) = 1 * coding_struct.llr_(i);

					coding_struct.llr[i] = -coding_struct.llr_(i);
					ldpc_r_struct.a_data_y[i] = coding_struct.llr[i];
				}

				for (int i = 0; i < N; i++)
				{
					ldpc_r_struct.LLR_in_C[0][i] = 0;
				}

				//LDPC_RateRecover
				// input -> a_data_y	  outlen x 1
				// output -> LLR_in_tmp	  (C x N1)
				// output -> LLR_in_C	  (C x N) fill puncture parts
				modified_RateRecoverLDPC(ldpc_r_struct.a_data_y, ldpc_r_struct.LLR_in_tmp, ldpc_r_struct.deconcatenated, ldpc_r_struct.LLR_in_C, ldpc_t_struct.E_array, k0, outlen, N_Punctured, C, Zc, K, K_CB_Bit, bgn, outlen, 0, Qm, 1);

				// LDPC decoding
				//input LLR_in_C  (C x N)
				//output LLR _out (1 x N) llr resuls
				//output uhat1    (C x N) decoing results
				for (i = 0; i < C; i++)
				{
					decodeLogDomainMinSum_converge(ldpc_r_struct.LLR_in_C[i], ldpc_r_struct.LLR_out, ldpc_r_struct.y_array, iter_decoding, M, N, sigma2, CNtoVN, LLR_CNtoVN, LLR_VN, ldpc_t_struct.H);

					for (int j = 0; j < N; j++)
					{
						ldpc_r_struct.uhat1[i][j] = (ldpc_r_struct.LLR_out[j] >= 0 )? 0 : 1;
					}

				}
				
				//LDPC_desegment get the transmit block from code block
				// input -> uhat1 (result of decoding)
				// medium-bits uhat2 -> messagebits in uhat1
				//output -> uhat(transmit block) excluding CB-CRC and TB-CRC, means only baseband data
				modified_DesegmentLDPC(ldpc_r_struct.uhat2, ldpc_r_struct.uhat1, ldpc_r_struct.uhat, L, K_CB_Bit, B, C);
				
			 
				// length of u_out_tmp is N ; 
				// get the first outlen message bits
				for (i = 0; i < outlen; i++)
				{
					ldpc_r_struct.u_out_tmp[i] = ldpc_r_struct.LLR_out[i] < 0;
				}

				// if the crc check is satisfied, break the IDD
				// including zero padding bits (2*Zc)
				// take the first B bits of u_out_tmp for crc check
				if (rx_check_crc((unsigned char*)ldpc_r_struct.u_out_tmp, B, L) == true) {
					break;
				}
 
				

				// output LLR_out_tmp gets outlen's bits without no-transmition bits, including zero padding, filler bits and unused parity bits
				for (i = 2 * Zc; i < K_CB_Bit; i++) //no puncture part of message bits 
				{
					ldpc_r_struct.LLR_out_tmp[i - 2 * Zc] = -ldpc_r_struct.LLR_out[i];
				}
				for (i = K; i < outlen + K - K_CB_Bit + 2 * Zc; i++) // finding effective bits in outlen
				{
					ldpc_r_struct.LLR_out_tmp[i - K + (K_CB_Bit - 2 * Zc)] = -ldpc_r_struct.LLR_out[i];
				}


				// Interweave for MIMO detection
				Mat<float> weaveMat2(&ldpc_r_struct.LLR_out_tmp[0], itweave_row, outlen / itweave_row);  // save in column-way
				Col<float> weaveCol2 = vectorise(trans(weaveMat2));
				//weaveCol2.print("weaveCol is");
				for (i = 0; i < outlen; i++)
				{
					ldpc_r_struct.LLR_out_tmp[i] = weaveCol2(i); // outlen x 1 
				}

				//if (flag == 0)
				//{
				for (cnt = 0; cnt < cntBlk; cnt++) //gamma iteration
				{
					for (s = 0; s < Nt; s++)
					{
						size_t LLR_start1 = cnt * nof_BitPerBlock + s * symbol_length; // the pointer for bit LLR of each symbol, real;
						size_t LLR_start2 = cnt * nof_BitPerBlock + s * symbol_length + symbol_length_real; // the pointer for bit LLR of each symbol, imag;

						// bit llr -> sym llr
						// using LDPC bit-llr results to update gamma  
						for (kk = 0; kk < Csym_len_real; kk++)
						{
							if (MODE_TYPE == 8)
							{
								det_struct.gamma[cnt][s][kk] = ldpc_r_struct.LLR_out_tmp[LLR_start1] * ((uint8_t)kk >> (symbol_length_real - 1) & 0x01)
									+ ldpc_r_struct.LLR_out_tmp[LLR_start1 + 1] * ((uint8_t)kk >> (symbol_length_real - 2) & 0x01)
									+ ldpc_r_struct.LLR_out_tmp[LLR_start1 + 2] * ((uint8_t)kk >> (symbol_length_real - 3) & 0x01)
									+ ldpc_r_struct.LLR_out_tmp[LLR_start1 + 3] * ((uint8_t)kk >> (symbol_length_real - 4) & 0x01);

								det_struct.gamma[cnt][s + Nt][kk] = ldpc_r_struct.LLR_out_tmp[LLR_start2] * ((uint8_t)kk >> (symbol_length_real - 1) & 0x01)
									+ ldpc_r_struct.LLR_out_tmp[LLR_start2 + 1] * ((uint8_t)kk >> (symbol_length_real - 2) & 0x01)
									+ ldpc_r_struct.LLR_out_tmp[LLR_start2 + 2] * ((uint8_t)kk >> (symbol_length_real - 3) & 0x01)
									+ ldpc_r_struct.LLR_out_tmp[LLR_start2 + 3] * ((uint8_t)kk >> (symbol_length_real - 4) & 0x01);
							}
							else if (MODE_TYPE == 6)
							{
								det_struct.gamma[cnt][s][kk] = ldpc_r_struct.LLR_out_tmp[LLR_start1] * ((uint8_t)kk >> (symbol_length_real - 1) & 0x01)
									+ ldpc_r_struct.LLR_out_tmp[LLR_start1 + 1] * ((uint8_t)kk >> (symbol_length_real - 2) & 0x01)
									+ ldpc_r_struct.LLR_out_tmp[LLR_start1 + 2] * ((uint8_t)kk >> (symbol_length_real - 3) & 0x01);

								det_struct.gamma[cnt][s + Nt][kk] = ldpc_r_struct.LLR_out_tmp[LLR_start2] * ((uint8_t)kk >> (symbol_length_real - 1) & 0x01)
									+ ldpc_r_struct.LLR_out_tmp[LLR_start2 + 1] * ((uint8_t)kk >> (symbol_length_real - 2) & 0x01)
									+ ldpc_r_struct.LLR_out_tmp[LLR_start2 + 2] * ((uint8_t)kk >> (symbol_length_real - 3) & 0x01);
							}
							else
							{
								det_struct.gamma[cnt][s][kk] = ldpc_r_struct.LLR_out_tmp[LLR_start1] * ((uint8_t)kk >> (symbol_length_real - 1) & 0x01)
									+ ldpc_r_struct.LLR_out_tmp[LLR_start1 + 1] * ((uint8_t)kk >> (symbol_length_real - 2) & 0x01);

								det_struct.gamma[cnt][s + Nt][kk] = ldpc_r_struct.LLR_out_tmp[LLR_start2] * ((uint8_t)kk >> (symbol_length_real - 1) & 0x01)
									+ ldpc_r_struct.LLR_out_tmp[LLR_start2 + 1] * ((uint8_t)kk >> (symbol_length_real - 2) & 0x01);
							}


							det_struct.gamma_.slice(cnt)(s, kk) = 0.00 * det_struct.gamma_.slice(cnt)(s, kk) + 1 * det_struct.gamma[cnt][s][kk];
							det_struct.gamma_.slice(cnt)(s + Nt, kk) = 0.00 * det_struct.gamma_.slice(cnt)(s + Nt, kk) + 1 * det_struct.gamma[cnt][s + Nt][kk];
						}

							
					}

					

				//}
				}


				flag = 0;

				
			}

			finish = clock();
			duration += (double)(finish - start) / CLOCKS_PER_SEC * 1000000;	

			int Temp_Error = Num_Error;
			for (int i = 0; i < A; i++)
			{
				Num_Error += abs(int(ldpc_r_struct.uhat[i]) - int(ldpc_t_struct.a_data_crc[i]));
			}
			
			
			if (Temp_Error < Num_Error) {
				Num_Frame_Error++;
				Total_BER += (double)(Num_Error - Temp_Error) / (double)A;
			}


			if (Max_frame % 100 == 0)
				cout << sigma2 << " " << "RD-BsP IDD\tNow SNR: " << SNR << "\tnm: " << nm << "\tNow Code Frame: " << Max_frame << "\tNow Error Frame : " << Num_Frame_Error << "\tNow Error Bits: " << Num_Error << endl << " BER = " << Total_BER / Max_frame << " FER = " << (double)Num_Frame_Error / Max_frame << endl << endl;
		 
		}
		y_out << "SNR = " << SNR << "Latency = " << duration / Max_frame << " BER = " << Total_BER / Max_frame << " FER = " << (double)Num_Frame_Error / Max_frame << " MIMO BER = " << Total_BER1 / Max_frame << endl;

	}

	thread_struct_free();
	std::cout << "Hello World!\n";
	thread_struct_free_ldpc(C, K, K_CB_Bit, N, N_Punctured, Mb, Zc);
	system("pause");
	return 0;
}
