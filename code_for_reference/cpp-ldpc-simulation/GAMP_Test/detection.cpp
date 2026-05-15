
#include "detection.h"
using namespace std;

struct CompLarge {
	CompLarge(const vector<float>& v) : _v(v) {}
	bool operator ()(int a, int b) { return _v[a] > _v[b]; }
	const vector<float>& _v;
};



void gai_bp_arma(Mat<float>& H_mtx, Col<float> Rx, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma, Mat<float>& llr_mtx, float* det_results, size_t Nr, size_t Nt, size_t iterNum, size_t Csym_length, float damp_factor, float sigma2)
{
	size_t i, j, cnt;
	size_t Nt2 = 2 * Nt;
	size_t Nr2 = 2 * Nr;

	//Mat<float> Means(Nr2, Nt2, fill::zeros);
	//Mat<float> Vars(Nr2, Nt2, fill::zeros);

	//Col<float> Mu_all(Nr2);
	//Mat<float> Sigma_forsum(Nr2, Nt2);
	//Col<float> Sigma_all(Nr2);

	Col<float> sGAI(Nt2);          // mean of the two most likely symbol's constellation
	Col<float> sGAI_Var(Nt2);      // variance of the two most likely symbol's constellation
	Col<float> mean_col(Nt2);      // mean of each symbol node
	Col<float> var_col(Nt2);       // varaince of each symbol node
	Col<float> H_mtx_row(Nt2);     //convert the H_mtx_row(j) to Colvec

	Mat<float> mean_incoming(Nt2, Csym_length);  // incoming message's mean of every symbol nodes and their constellations
	Mat<float> var_incoming(Nt2, Csym_length);   // incoming message's variance
	Mat<float> HS_0(Nt2, Csym_length);            // used for the first constellation
	Mat<float> HS(Nt2, Csym_length, fill::zeros);// all constellation

	Mat<float> conss_mat(Nt2, Csym_length);      // reduce time consumption
	conss_mat.each_row() = cons_row;


	Cube<float> Px(Nt2, Csym_length, Nr2, fill::value((float)1 / Csym_length));
	//Cube<float> Px(Nt2, Nr2, Csym_length, fill::value(1 / Csym_length));

	Mat<float> K_mmse = inv(H_mtx.t() * H_mtx + sigma2 * eye<Mat<float>>(Nt2, Nt2));
	Col<float> S_mmse(det_results, Nt2);

	for (i = 0; i < Nt2; i++)
	{
		Col<float> symProb = exp(-1 * square(abs(cons_col - S_mmse(i))) / (2 * abs(K_mmse(i, i))));
		symProb = symProb / as_scalar(sum(symProb));

		//uword symidx = index_max(symProb);
		//symProb.zeros();
		//symProb(symidx) = 1.0;

		Px.row(i) = (repmat(symProb, 1, Nr2));
	}


#if 0
	// step2: update the beta messages
	conss = cons_col.elem(idx_max_.slice(j));  // 最大symbol对应的星座点	
	R_incoming_all = as_scalar(H_mtx.row(j) * conss); // 第j根天线接收的所有最大星座点
	H_mtx_row = H_mtx.row(j).t();

	// Compute each beta message;
	R_incoming.each_col() = R_incoming_all - H_mtx_row % conss; //除自己以外的点	

	HS_0.each_col() = H_mtx_row * cons[0];
	//Nt2*Csym_length each row is symbol nodes' all constellation probability
	HS.each_col() = H_mtx_row;
	HS.each_row([Csym_length](frowvec& a) { a = a % cons_row; });

	fmat llrk = Rx(j) - R_incoming - HS; // No.k constellation
	//llrk = pow(llrk, 2);
	fmat llr0 = Rx(j) - R_incoming - HS_0; // No.0 constellation
	//llr0 = pow(llr0, 2);

	beta.slice(j) = (-llrk % llrk + llr0 % llr0) / 2 / sigma2;
	gamma += beta.slice(j);     //add all slice to a matrix

#endif // 0


	for (cnt = 0; cnt < iterNum; cnt++)
	{
		//Cube<float> Px_mtx = reshape(Px, Px.n_rows * Px.n_cols, Px.n_slices, 1);
		////Mat<float> squeez_symProb = reshape(Mat<float>(symProb.))
		//Means = reshape(Px_mtx.slice(0) * cons_col, Px.n_rows, Px.n_cols);
		//Vars = reshape(Px_mtx.slice(0) * square(cons_col), Px.n_rows, Px.n_cols) - square(abs(Means));

		//Mu_all = sum(H_mtx % Means.t(), 1);
		//Sigma_forsum = square(abs(H_mtx)) % Vars.t();
		//Sigma_all = sum(Sigma_forsum, 1);

		gamma.zeros();

		for (j = 0; j < Nr2; j++)
		{
			H_mtx_row = H_mtx.row(j).t();
			sGAI = Px.slice(j) * cons_col; // nt2 * 1
			sGAI_Var = Px.slice(j) * ((abs(cons_col % cons_col)));
			//sGAI.print("sGAI is");
			//sGAI_Var.print("sGAI_Var is");
			mean_col = sGAI % H_mtx_row;
			var_col = (abs((H_mtx_row) % (H_mtx_row))) % (Px.slice(j) * ((abs(cons_col % cons_col))) - sGAI % sGAI);
			//var_col.print("var_col is");
			float mean_incoming_all = accu(mean_col);
			float var_incoming_all = accu(var_col);

			//cout << "mean_incoming_all is" << mean_incoming_all << endl;
			//cout << "var_incoming_all is" << var_incoming_all << endl;

			// Compute each beta message;
			mean_incoming.each_col() = mean_incoming_all - mean_col; //extrinsic message sum
			var_incoming.each_col() = var_incoming_all - var_col + sigma2;

			HS_0.each_col() = H_mtx_row * cons[0];
			HS.each_col() = H_mtx_row;
			HS = HS % conss_mat;

			fmat llrk = Rx(j) - mean_incoming - HS; // No.k constellation
			llrk = llrk % llrk;
			fmat llr0 = Rx(j) - mean_incoming - HS_0; // No.0 constellation
			llr0 = llr0 % llr0;

			beta.slice(j) = (-llrk + llr0) / 2 / var_incoming;

			gamma += beta.slice(j);     //update gamma

		}

		alpha = gamma - beta.each_slice();

		for (j = 0; j < Nr2; j++)
		{
			Mat<float> alpha_max_mat = repmat(max(alpha.slice(j), 1), 1, Csym_length);
			Mat<float> a_temp = exp(alpha.slice(j) - alpha_max_mat);
			Mat<float> alpha_sum_max = repmat(sum(a_temp, 1), 1, Csym_length);
			Px.slice(j) = (1 - damp_factor) * (a_temp / alpha_sum_max) + damp_factor * Px.slice(j);
		}

	}

	//gamma.print("gamma is");
	if (MODE_TYPE == 8)
	{
#if 1
		//256QAM in armadillo
		// calculate bit-llr using symbol llr
		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(8, 15)) - max(gamma_real.rows(0, 7));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_real.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_real.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(3) = max(gamma_real.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_real.rows(uvec{ 0,2,4,6,8,10,12,14 }));
		// imag part
		llr_mtx.row(4) = max(gamma_imag.rows(8, 15)) - max(gamma_imag.rows(0, 7));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(6) = max(gamma_imag.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(7) = max(gamma_imag.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_imag.rows(uvec{ 0,2,4,6,8,10,12,14 }));

#endif
	}
	else if (MODE_TYPE == 6)
	{
#if 1 // Soft output for 64-QAM modulation using armadillo
		// 64QAM 
		// compute the Bit LLR

		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(4, 7)) - max(gamma_real.rows(0, 3));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 2,3,6,7 })) - max(gamma_real.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 1,3,5,7 })) - max(gamma_real.rows(uvec{ 0,2,4,6 }));
		// imag part
		llr_mtx.row(3) = max(gamma_imag.rows(4, 7)) - max(gamma_imag.rows(0, 3));
		llr_mtx.row(4) = max(gamma_imag.rows(uvec{ 2,3,6,7 })) - max(gamma_imag.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 1,3,5,7 })) - max(gamma_imag.rows(uvec{ 0,2,4,6 }));


#endif
	}
	else if (MODE_TYPE == 4)
	{
#if 1 // Soft output for 16-QAM modulation using armadillo
		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(2, 3)) - max(gamma_real.rows(0, 1));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 1,3 })) - max(gamma_real.rows(uvec{ 0,2 }));
		// imag part
		llr_mtx.row(2) = max(gamma_imag.rows(2, 3)) - max(gamma_imag.rows(0, 1));
		llr_mtx.row(3) = max(gamma_imag.rows(uvec{ 1,3 })) - max(gamma_imag.rows(uvec{ 0,2 }));

#endif
	}
	else
	{
		printf("Invalid modulation type. \n");;
	}


}


 


void bsp_dm1df1_rd_arma(Mat<float>& H_mtx, Col<float> Rx, float* det_results, size_t symbol_length, size_t Nr, size_t Nt, size_t iterNum, float sigma2,
	float delta, Cube<float>& beta, Mat<float>& gamma, Cube<float>& alpha, Mat<float>& llr_mtx)
{
	size_t i, j, s, iter, k;
	size_t slen = symbol_length / 2;

	size_t Csym_length = pow(2, slen);

	size_t Nt2 = 2 * Nt;
	size_t Nr2 = 2 * Nr;

	float R_incoming_all = 0;
	Col<float> conss(Nt2);  // the position of biggest constellation 
	Col<float> H_mtx_row(Nt2);  //convert the H_mtx_row(j) to Colvec

	fmat R_incoming(Nt2, Csym_length);
	fmat HS_0(Nt2, Csym_length);
	fmat HS(Nt2, Csym_length, fill::zeros);

	//  initialize with MMSE; 简化版本
	for (i = 0; i < Nt2; i++)
	{
		gamma.row(i) = -abs(cons_row - det_results[i]);
	}
	alpha.each_slice() = gamma;


	for (iter = 0; iter < iterNum; iter++)
	{

		Cube<uword> idx_max_ = index_max(alpha, 1);
		gamma.zeros();
		for (j = 0; j < Nr2; j++)
		{
			// step2: update the beta messages
			conss = cons_col.elem(idx_max_.slice(j));  // 最大symbol对应的星座点	
			R_incoming_all = as_scalar(H_mtx.row(j) * conss); // 第j根天线接收的所有最大星座点
			H_mtx_row = H_mtx.row(j).t();

			// Compute each beta message;
			R_incoming.each_col() = R_incoming_all - H_mtx_row % conss; //除自己以外的点	

			HS_0.each_col() = H_mtx_row * cons[0];
			//Nt2*Csym_length each row is symbol nodes' all constellation probability
			HS.each_col() = H_mtx_row;
			HS.each_row([Csym_length](frowvec& a) { a = a % cons_row; });

			fmat llrk = Rx(j) - R_incoming - HS; // No.k constellation
			//llrk = pow(llrk, 2);
			fmat llr0 = Rx(j) - R_incoming - HS_0; // No.0 constellation
			//llr0 = pow(llr0, 2);

			beta.slice(j) = (-llrk % llrk + llr0 % llr0) / 2 / sigma2;
			gamma += beta.slice(j);     //add all slice to a matrix
		}

		alpha = gamma - beta.each_slice();

	}
	if (MODE_TYPE == 8)
	{
#if 1
		//256QAM in armadillo
		// calculate bit-llr using symbol llr
		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(8, 15)) - max(gamma_real.rows(0, 7));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_real.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_real.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(3) = max(gamma_real.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_real.rows(uvec{ 0,2,4,6,8,10,12,14 }));
		// imag part
		llr_mtx.row(4) = max(gamma_imag.rows(8, 15)) - max(gamma_imag.rows(0, 7));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(6) = max(gamma_imag.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(7) = max(gamma_imag.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_imag.rows(uvec{ 0,2,4,6,8,10,12,14 }));

#endif
	}
	else if (MODE_TYPE == 6)
	{
#if 1 // Soft output for 64-QAM modulation using armadillo
		// 64QAM 
		// compute the Bit LLR

		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(4, 7)) - max(gamma_real.rows(0, 3));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 2,3,6,7 })) - max(gamma_real.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 1,3,5,7 })) - max(gamma_real.rows(uvec{ 0,2,4,6 }));
		// imag part
		llr_mtx.row(3) = max(gamma_imag.rows(4, 7)) - max(gamma_imag.rows(0, 3));
		llr_mtx.row(4) = max(gamma_imag.rows(uvec{ 2,3,6,7 })) - max(gamma_imag.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 1,3,5,7 })) - max(gamma_imag.rows(uvec{ 0,2,4,6 }));


#endif
	}
	else if (MODE_TYPE == 4)
	{
#if 1 // Soft output for 16-QAM modulation using armadillo
		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(2, 3)) - max(gamma_real.rows(0, 1));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 1,3 })) - max(gamma_real.rows(uvec{ 0,2 }));
		// imag part
		llr_mtx.row(2) = max(gamma_imag.rows(2, 3)) - max(gamma_imag.rows(0, 1));
		llr_mtx.row(3) = max(gamma_imag.rows(uvec{ 1,3 })) - max(gamma_imag.rows(uvec{ 0,2 }));

#endif
	}
	else
	{
		printf("Invalid modulation type. \n");;
	}
# if 0
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;

		llr[i * symbol_length] = max({ gamma(i,8), gamma(i,9), gamma(i,10), gamma(i,11), gamma(i,12), gamma(i,13), gamma(i,14), gamma(i,15) }) -
			max({ gamma(i,0), gamma(i,1), gamma(i,2), gamma(i,3), gamma(i,4), gamma(i,5), gamma(i,6), gamma(i,7) });

		llr[i * symbol_length + 1] = max({ gamma(i,4), gamma(i,5), gamma(i,6), gamma(i,7), gamma(i,12), gamma(i,13), gamma(i,14), gamma(i,15) }) -
			max({ gamma(i,0), gamma(i,1), gamma(i,2), gamma(i,3), gamma(i,8), gamma(i,9), gamma(i,10), gamma(i,11) });

		llr[i * symbol_length + 2] = max({ gamma(i,2), gamma(i,3), gamma(i,6), gamma(i,7),gamma(i,10), gamma(i,11),gamma(i,14), gamma(i,15) }) -
			max({ gamma(i,0), gamma(i,1), gamma(i,4), gamma(i,5),gamma(i,8), gamma(i,9), gamma(i,12), gamma(i,13) });

		llr[i * symbol_length + 3] = max({ gamma(i,1), gamma(i,3), gamma(i,5), gamma(i,7),gamma(i,9), gamma(i,11),gamma(i,13), gamma(i,15) }) -
			max({ gamma(i,0), gamma(i,2), gamma(i,4), gamma(i,6),gamma(i,8), gamma(i,10), gamma(i,12), gamma(i,14) });


		// bits with respect to the imag part;
		llr[i * symbol_length + 4] = max({ gamma(i + Nt,8), gamma(i + Nt,9), gamma(i + Nt,10), gamma(i + Nt,11), gamma(i + Nt,12), gamma(i + Nt,13), gamma(i + Nt,14), gamma(i + Nt,15) }) -
			max({ gamma(i + Nt,0), gamma(i + Nt,1), gamma(i + Nt,2), gamma(i + Nt,3), gamma(i + Nt,4), gamma(i + Nt,5), gamma(i + Nt,6), gamma(i + Nt,7) });

		llr[i * symbol_length + 5] = max({ gamma(i + Nt,4), gamma(i + Nt,5), gamma(i + Nt,6), gamma(i + Nt,7), gamma(i + Nt,12), gamma(i + Nt,13), gamma(i + Nt,14), gamma(i + Nt,15) }) -
			max({ gamma(i + Nt,0), gamma(i + Nt,1), gamma(i + Nt,2), gamma(i + Nt,3), gamma(i + Nt,8), gamma(i + Nt,9), gamma(i + Nt,10), gamma(i + Nt,11) });

		llr[i * symbol_length + 6] = max({ gamma(i + Nt,2), gamma(i + Nt,3), gamma(i + Nt,6), gamma(i + Nt,7),gamma(i + Nt,10), gamma(i + Nt,11),gamma(i + Nt,14), gamma(i + Nt,15) }) -
			max({ gamma(i + Nt,0), gamma(i + Nt,1), gamma(i + Nt,4), gamma(i + Nt,5),gamma(i + Nt,8), gamma(i + Nt,9), gamma(i + Nt,12), gamma(i + Nt,13) });

		llr[i * symbol_length + 7] = max({ gamma(i + Nt,1), gamma(i + Nt,3), gamma(i + Nt,5), gamma(i + Nt,7),gamma(i + Nt,9), gamma(i + Nt,11),gamma(i + Nt,13), gamma(i + Nt,15) }) -
			max({ gamma(i + Nt,0), gamma(i + Nt,2), gamma(i + Nt,4), gamma(i + Nt,6),gamma(i + Nt,8), gamma(i + Nt,10), gamma(i + Nt,12), gamma(i + Nt,14) });

	}
#endif
}



void CFbsp_nm_mean_rd_idd_arma_llrpro(Mat<float>& H_mtx, Col<float> Rx, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma, Mat<uword>det_res_idx, Mat<float>& mmse_llr, Mat<float>& llr_mtx, size_t Nr, size_t Nt, size_t iterNum, size_t nm, size_t Csym_length, float damp_factor, float sigma2, float clipping_neg_value)
{
	size_t i, j, s, iter, k;
	//size_t slen = symbol_length / 2;
	//size_t Csym_length = pow(2, slen);

	//size_t dm = 2; // truncate to 2 symbol in every symbol_factor buffer

	size_t Nt2 = 2 * Nt;
	size_t Nr2 = 2 * Nr;

	float factor = 0.8;
	float llr_clipping_neg = -50;
	float geo_coeff = 0.5;   // geometric sequence
	float arith_coeff = 0.5; // 等差
	float scalar = 3;

	float mean_incoming_all = 0;
	float var_incoming_all = 0;

	Col<float> mean_col(Nt2);      // mean of each symbol node
	Col<float> var_col(Nt2);       // varaince of each symbol node
	Col<float> H_mtx_row(Nt2);     //convert the H_mtx_row(j) to Colvec

	Mat<float> mean_incoming(Nt2, Csym_length);  // incoming message's mean of every symbol nodes and their constellations
	Mat<float> var_incoming(Nt2, Csym_length);   // incoming message's variance
	Mat<float> HS_0(Nt2, Csym_length);            // used for the first constellation
	Mat<float> HS(Nt2, Csym_length, fill::zeros);// all constellation

	Mat<float> conss_mat(Nt2, Csym_length);      // reduce time consumption
	conss_mat.each_row() = cons_row;


	/*----------------- new parameters ---------------*/
	Cube<float> pro_miu0(Nt2, nm, Nr2);
	Cube<float> pro_cons(Nt2, nm, Nr2);
	Cube<float> sym_llr(Nt2, nm, Nr2);
	Cube<float> pro_sum(Nt2, 1, Nr2);
	Cube<float> alpha_nm(Nt2, nm, Nr2);
	Cube<float> symllr_max(Nt2, nm, Nr2);
	Mat<float> pro_cons_t(nm, Nt2);
	Mat<uword> pro_idx_mat(Nt2, nm);
	Mat<uword> pro_idx_nmtail(Nt2, Csym_length - nm);
	Row<uword> pro_idx_vec(Nt2 * nm);
	Cube<float> pro_temp(size(pro_cons));
	//Cube<float> pro_temp_1(size(pro_cons));

	Cube<float> sGAI(Nt2, 1, Nr2);
	Cube<float> sGAI_Var(Nt2, 1, Nr2);

	//no matter the iteration
	Mat<uword> det_idx_nm = det_res_idx.cols(0, nm - 1);   // select the constellation used in BsP
	Mat<uword> det_idx_nmtail = det_res_idx.cols(nm, Csym_length - 1); // these lattice need to fill LLR value 

	//det_idx_nm.print("pick nm index : ");


	pro_idx_vec = det_idx_nm.as_row();
	Mat<float> cons_nm = reshape(cons_row(pro_idx_vec), nm, Nt2);
	cons_nm = trans(cons_nm);

	Mat<uword> tensor1 = repmat(linspace<Col<uword>>(0, Nt2 - 1, Nt2), 1, nm);
	pro_idx_mat = det_idx_nm + Csym_length * tensor1;

	Mat<uword> tensor2 = repmat(linspace<Col<uword>>(0, Nt2 - 1, Nt2), 1, Csym_length - nm);
	pro_idx_nmtail = det_idx_nmtail + Csym_length * tensor2;

	//pro_idx_mat.print("pro idx mat : ");

	pro_idx_mat = pro_idx_mat.as_row();
	pro_idx_nmtail = pro_idx_nmtail.as_row();


	// the main iteration
	for (iter = 0; iter < iterNum; iter++)
	{

		if (iter == 0)
			alpha.each_slice() = gamma;
		else
			alpha = gamma - beta.each_slice();

		//alpha.each_slice() = gamma;
		gamma.zeros();

		/*---------- select the nm lattice to calculate probability -----------*/
		for (j = 0; j < Nr2; j++)
		{
			Mat<float> alpha_vec = alpha.slice(j).as_row();
			alpha_vec = alpha_vec(pro_idx_mat);
			sym_llr.slice(j) = trans(reshape(alpha_vec, nm, Nt2));
			Col<float> maxllr = max(sym_llr.slice(j), 1);
			symllr_max.slice(j).each_col() = maxllr;
		}

		/*------------------------------------*/

		pro_cons = exp(sym_llr - symllr_max);
		pro_sum = sum(pro_cons, 1);
		int islice = 0;
		pro_sum.each_slice([&pro_miu0, &islice, &nm](Mat<float>& X) {pro_miu0.slice(islice) = repmat(X, 1, nm); islice++; });
		pro_cons = pro_cons / pro_miu0;

		/*------------- add damping factor ------------------*/
		if (iter > 0)
		{
			pro_cons = (1 - damp_factor) * pro_cons + damp_factor * pro_temp;
		}
		pro_temp = pro_cons;

		/*---------------------------------*/

		pro_cons.each_slice([&cons_nm](Mat<float>& X) {X %= cons_nm; }); //calculate symbol means
		sGAI = sum(pro_cons, 1);
		pro_cons.each_slice([&cons_nm](Mat<float>& X) {X %= cons_nm; });  // calculate symbol variant
		sGAI_Var = sum(pro_cons, 1);


#if 1
		// 1 // Update beta

		for (j = 0; j < Nr2; j++)
		{
			H_mtx_row = H_mtx.row(j).t();
			mean_col = H_mtx_row % sGAI.slice(j);
			mean_incoming_all = accu(mean_col);
			//square(abs(H_mtx_row))
			var_col = (square(abs(H_mtx_row))) % (sGAI_Var.slice(j) - sGAI.slice(j) % sGAI.slice(j));
			var_incoming_all = accu(var_col);

			// Compute each beta message;
			mean_incoming.each_col() = mean_incoming_all - mean_col; //extrinsic message sum
			var_incoming.each_col() = var_incoming_all - var_col + sigma2;
			HS_0.each_col() = H_mtx_row * cons[0];
			HS.each_col() = H_mtx_row;
			HS = HS % conss_mat;

			fmat llrk = Rx(j) - mean_incoming - HS; // No.k constellation
			llrk = llrk % llrk;
			fmat llr0 = Rx(j) - mean_incoming - HS_0; // No.0 constellation
			llr0 = llr0 % llr0;

			beta.slice(j) = (-llrk + llr0) / 2 / var_incoming;
			//beta.slice(j) = (-llrk + llr0) / 2 / sigma2;
			/*---------------------------------------*/

			// only keep nm beta value, other beta keeps the minimum value.
			Mat<float> beta_vec = beta.slice(j).as_row();
			beta_vec = beta_vec(pro_idx_mat);

			//Mat<float>  beta_set(size(beta.slice(j)),fill::zeros);
			//Mat<float> bet_set(beta.cols)
			Row<float> beta_nm(Nt2 * Csym_length, fill::zeros);
			beta_nm(pro_idx_mat) = beta_vec;
			Mat<float> beta_set(size(beta.slice(j)));
			beta_set = trans(reshape(beta_nm, Csym_length, Nt2));
			gamma += beta_set;
			/*---------------------------------------*/

			//Mat<float> beta_vec = beta.slice(j).as_row();
			//beta_vec = beta_vec(pro_idx_mat);
			//Col<float> beta_min = min(beta.slice(j), 1);
			//Mat<float> beta_set(size(beta.slice(j)));
			//beta_set.each_col() = beta_min;
			////beta_set.zeros();
			//Mat<float> beta_set_vec = beta_set.as_row();
			//beta_set_vec(pro_idx_mat) = beta_vec;
			//beta_set = trans(reshape(beta_set_vec, Csym_length, Nt2));
			//gamma += beta_set;

			//Col<float> beta_min = min(beta.slice(j), 1);
			//Mat<float> beta_set(size(beta.slice(j)));
			//beta_set.each_col() = beta_min;
			////beta_set.zeros();
			//Mat<float> beta_set_vec = beta_set.as_row();
			//beta_set_vec(pro_idx_mat) = beta_vec;
			//beta_set = trans(reshape(beta_set_vec, Csym_length, Nt2));
			//gamma += beta_set;
			/*---------------------------------------*/

			//beta.slice(j).print("beta is");
			//gamma += beta.slice(j);     //update gamma

		}
#endif

	}



	// pick up the nm+1-th symbol llr
	Mat<float> gamma_vec = gamma.as_row();
	int nmtail = Csym_length - nm;
	//gamma.print("gamma is");

	Col<uword> nmllr_idx = linspace<Col<uword>>(nm - 1, Nt2 * nm - 1, Nt2);
	//nmllr_idx.print("nmllr_idx : ");
	nmllr_idx = pro_idx_mat(nmllr_idx); // get the min llr value from each row
	//nmllr_idx.print("nmllr index is");
	Col<float> nmllr = gamma_vec(nmllr_idx);
	Col<float> nmllr_tmp = nmllr - log((1 - pow (factor, (Csym_length - nm))) / (1 - factor));
	//Col<float> nmllr_tmp = nmllr - log(Csym_length - nm);


	Mat<float> nmtail_llr(Nt2, Csym_length - nm, fill::zeros); // the truncation part need to fill
	//float log_coeff = log(geo_coeff);
	//Row<float> tmp_coeff = linspace<Row<float>>(1, nmtail, nmtail) * log_coeff * scalar;
	////tmp_coeff.print("tmp coeff :");

	//Row<float> tmp_coeff = linspace<Row<float>>(0, Csym_length - nm - 1, Csym_length - nm) * log(factor);
	//Mat<float> coeffmat = repmat(tmp_coeff, Nt2, 1);
	//coeffmat.each_col() += nmllr_tmp;

	nmtail_llr.each_row() += linspace<Row<float>>(0, Csym_length - nm - 1, Csym_length - nm) * log(factor);
	nmtail_llr.each_col() += nmllr_tmp;
	//nmtail_llr.print("nmtail llr : ");


	/*----------- fill llr compensation back into gamm ------------*/
	Mat<float> gamma_vec1 = gamma.as_row();
	gamma_vec1(pro_idx_nmtail) = nmtail_llr.as_row();
	gamma = trans(reshape(gamma_vec1, Csym_length, Nt2));
	//gamma.print("gamma is");

#if 0

	//gamma.print("gamma is");
	Col<float> gamma_min = min(gamma, 1);
	/*------------ fill llr -------------*/
	Mat<float> gamma_vec = gamma.as_row();
	gamma_vec = gamma_vec(pro_idx_mat);   // select the nm lattice values
	gamma_vec = trans(reshape(gamma_vec, nm, Nt2));
	Col<float> gamma_nm_min = min(gamma_vec, 1);  // select the minimum value within the nm lattice values

	//gamma_min.print("gamma min is");

	/*------------- select the fill lattice -------------*/
	Mat<float> mmse_llr_nmtail = mmse_llr.as_row();
	mmse_llr_nmtail = mmse_llr_nmtail(pro_idx_nmtail); // sorted
	mmse_llr_nmtail = trans(reshape(mmse_llr_nmtail, Csym_length - nm, Nt2));

	Mat<float> mmse_llr_sign = sign(mmse_llr_nmtail);

	for (i = 0; i < Nt2; i++)
	{
		if (gamma_nm_min(i) > 0)
		{
			/*------------calculate the positive LLR---------------*/
			float posNum = sum(mmse_llr_nmtail.row(i) > 0); // calculate the positive number
			Row<float> mmse_llr_nmtail_row = mmse_llr_nmtail.row(i);  // 
			if (posNum > 0)
			{
				/********* llr compensation(arithmetic sequence) ********/

				//Row<float> posllr = linspace<Row<float>>(gamma_nm_min(i), 0, posNum + 2); 
				//mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row > 0)) = posllr.subvec(1, posNum);

				/*----------------------------------------------*/


				/********* llr compensation (geometric sequence) ********/
				Row<float> basevec_pos(posNum, fill::value(factor));
				Row<float> geo_seq_pos = linspace<Row<float>>(-1, -1 * posNum, posNum);
				Row<float> posllr = gamma_nm_min(i) * pow(basevec_pos, geo_seq_pos);
				mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row > 0)) = posllr;
				/*----------------------------------------------*/

			}


			/*-------------calculate the negative LLR-------------------*/
			float negNum = sum(mmse_llr_nmtail.row(i) < 0);
			if (negNum > 0)
			{
				/********* llr compensation(arithmetic sequence) ********/
				//Row<float> negllr = linspace<Row<float>>(0, gamma_min(i), negNum + 1);
				//mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row < 0)) = negllr.subvec(1, negNum);

				/********* llr compensation (geometric sequence) ********/
				Row<float> basevec_neg(negNum, fill::value(factor));
				Row<float> geo_seq_neg = linspace<Row<float>>(-1 * negNum, -1, negNum);
				Row<float> negllr = llr_clipping_neg * pow(basevec_neg, geo_seq_neg);
				//Row<float> negllr = gamma_min(i) * pow(basevec_neg, geo_seq_neg);
				mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row < 0)) = negllr;

			}



			mmse_llr_nmtail.row(i) = mmse_llr_nmtail_row;

		}
		else   // gamma_nm_min(i) <= 0
		{
			/*------------- only need to calculate the negative LLR --------------*/
			Row<float> mmse_llr_nmtail_row = mmse_llr_nmtail.row(i);  // 
			float negNum = sum(mmse_llr_nmtail.row(i) < 0);

			if (negNum > 0)
			{
				/********* llr compensation(arithmetic sequence) ********/
				//Row<float> negllr = linspace<Row<float>>(0, gamma_min(i), negNum + 1);
				//mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row < 0)) = negllr.subvec(1, negNum);

				/********* llr compensation (geometric sequence) ********/
				Row<float> basevec_neg(negNum, fill::value(factor));
				Row<float> geo_seq_neg = linspace<Row<float>>(-1 * negNum, -1, negNum);
				Row<float> negllr = llr_clipping_neg * pow(basevec_neg, geo_seq_neg);
				//Row<float> negllr = gamma_min(i) * pow(basevec_neg, geo_seq_neg);
				mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row < 0)) = negllr;
			}
			mmse_llr_nmtail.row(i) = mmse_llr_nmtail_row;


		}
	}

	/*----------- fill llr compensation back into gamm ------------*/
	Mat<float> gamma_vec1 = gamma.as_row();
	gamma_vec1(pro_idx_nmtail) = mmse_llr_nmtail.as_row();
	gamma = trans(reshape(gamma_vec1, Csym_length, Nt2));
#endif // 0


	//gamma.print("llr inserted gamma is");

#if 0 // Soft output for 16-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		llr[i * symbol_length] = max(gamma[i][2], gamma[i][3]) - max(gamma[i][0], gamma[i][1]);
		llr[i * symbol_length + 1] = max(gamma[i][1], gamma[i][3]) - max(gamma[i][0], gamma[i][2]);
		llr[i * symbol_length + 2] = max(gamma[i + Nt][2], gamma[i + Nt][3]) - max(gamma[i + Nt][0], gamma[i + Nt][1]);
		llr[i * symbol_length + 3] = max(gamma[i + Nt][1], gamma[i + Nt][3]) - max(gamma[i + Nt][0], gamma[i + Nt][2]);
	}
#endif

#if 0 // Soft output for 64-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3] });
		llr[i * symbol_length + 1] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5] });
		llr[i * symbol_length + 2] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 3] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3] });
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6] });
	}
#endif

	if (MODE_TYPE == 8)
	{
#if 1
		//256QAM in armadillo
		// calculate bit-llr using symbol llr
		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(8, 15)) - max(gamma_real.rows(0, 7));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_real.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_real.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(3) = max(gamma_real.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_real.rows(uvec{ 0,2,4,6,8,10,12,14 }));
		// imag part
		llr_mtx.row(4) = max(gamma_imag.rows(8, 15)) - max(gamma_imag.rows(0, 7));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(6) = max(gamma_imag.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(7) = max(gamma_imag.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_imag.rows(uvec{ 0,2,4,6,8,10,12,14 }));

#endif
	}
	else if (MODE_TYPE == 6)
	{
#if 1 // Soft output for 64-QAM modulation using armadillo
		// 64QAM 
		// compute the Bit LLR

		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(4, 7)) - max(gamma_real.rows(0, 3));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 2,3,6,7 })) - max(gamma_real.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 1,3,5,7 })) - max(gamma_real.rows(uvec{ 0,2,4,6 }));
		// imag part
		llr_mtx.row(3) = max(gamma_imag.rows(4, 7)) - max(gamma_imag.rows(0, 3));
		llr_mtx.row(4) = max(gamma_imag.rows(uvec{ 2,3,6,7 })) - max(gamma_imag.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 1,3,5,7 })) - max(gamma_imag.rows(uvec{ 0,2,4,6 }));


#endif
	}
	else if (MODE_TYPE == 4)
	{
#if 1 // Soft output for 16-QAM modulation using armadillo
		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(2, 3)) - max(gamma_real.rows(0, 1));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 1,3 })) - max(gamma_real.rows(uvec{ 0,2 }));
		// imag part
		llr_mtx.row(2) = max(gamma_imag.rows(2, 3)) - max(gamma_imag.rows(0, 1));
		llr_mtx.row(3) = max(gamma_imag.rows(uvec{ 1,3 })) - max(gamma_imag.rows(uvec{ 0,2 }));

#endif
	}
	else
	{
		printf("Invalid modulation type. \n");;
	}


#if 0 // Soft output for 256-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] });
		llr[i * symbol_length + 1] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11] });
		llr[i * symbol_length + 2] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7],gamma[i][10], gamma[i][11],gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5],gamma[i][8], gamma[i][9], gamma[i][12], gamma[i][13] });
		llr[i * symbol_length + 3] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7],gamma[i][9], gamma[i][11],gamma[i][13], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6],gamma[i][8], gamma[i][10], gamma[i][12], gamma[i][14] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11] });
		llr[i * symbol_length + 6] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7],gamma[i + Nt][10], gamma[i + Nt][11],gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5],gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][12], gamma[i + Nt][13] });
		llr[i * symbol_length + 7] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7],gamma[i + Nt][9], gamma[i + Nt][11],gamma[i + Nt][13], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6],gamma[i + Nt][8], gamma[i + Nt][10], gamma[i + Nt][12], gamma[i + Nt][14] });
	}
#endif


}





void bsp_nm_mean_rd_idd_arma_llrpro(Mat<float>& H_mtx, Col<float> Rx, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma, Mat<uword>det_res_idx, Mat<float>& mmse_llr, Mat<float>& llr_mtx, size_t Nr, size_t Nt, size_t iterNum, size_t nm, size_t Csym_length, float damp_factor, float sigma2, float clipping_neg_value)
{
	size_t i, j, s, iter, k;
	//size_t slen = symbol_length / 2;
	//size_t Csym_length = pow(2, slen);

	//size_t dm = 2; // truncate to 2 symbol in every symbol_factor buffer

	size_t Nt2 = 2 * Nt;
	size_t Nr2 = 2 * Nr;

	float factor = 1.25;
	float llr_clipping_neg = -50;
	float geo_coeff = 0.5;   // geometric sequence
	float arith_coeff = 0.5; // 等差
	float scalar = 3;

	float mean_incoming_all = 0;
	float var_incoming_all = 0;

	Col<float> mean_col(Nt2);      // mean of each symbol node
	Col<float> var_col(Nt2);       // varaince of each symbol node
	Col<float> H_mtx_row(Nt2);     //convert the H_mtx_row(j) to Colvec

	Mat<float> mean_incoming(Nt2, Csym_length);  // incoming message's mean of every symbol nodes and their constellations
	Mat<float> var_incoming(Nt2, Csym_length);   // incoming message's variance
	Mat<float> HS_0(Nt2, Csym_length);            // used for the first constellation
	Mat<float> HS(Nt2, Csym_length, fill::zeros);// all constellation

	Mat<float> conss_mat(Nt2, Csym_length);      // reduce time consumption
	conss_mat.each_row() = cons_row;


	/*----------------- new parameters ---------------*/
	Cube<float> pro_miu0(Nt2, nm, Nr2);
	Cube<float> pro_cons(Nt2, nm, Nr2);
	Cube<float> sym_llr(Nt2, nm, Nr2);
	Cube<float> pro_sum(Nt2, 1, Nr2);
	Cube<float> alpha_nm(Nt2, nm, Nr2);
	Cube<float> symllr_max(Nt2, nm, Nr2);
	Mat<float> pro_cons_t(nm, Nt2);
	Mat<uword> pro_idx_mat(Nt2, nm);
	Mat<uword> pro_idx_nmtail(Nt2, Csym_length - nm);
	Row<uword> pro_idx_vec(Nt2 * nm);
	Cube<float> pro_temp(size(pro_cons));
	//Cube<float> pro_temp_1(size(pro_cons));

	Cube<float> sGAI(Nt2, 1, Nr2);
	Cube<float> sGAI_Var(Nt2, 1, Nr2);

	//no matter the iteration
	Mat<uword> det_idx_nm = det_res_idx.cols(0, nm - 1);   // select the constellation used in BsP
	Mat<uword> det_idx_nmtail = det_res_idx.cols(nm, Csym_length - 1); // these lattice need to fill LLR value 

	//det_idx_nm.print("pick nm index : ");


	pro_idx_vec = det_idx_nm.as_row();
	Mat<float> cons_nm = reshape(cons_row(pro_idx_vec), nm, Nt2);
	cons_nm = trans(cons_nm);

	Mat<uword> tensor1 = repmat(linspace<Col<uword>>(0, Nt2 - 1, Nt2), 1, nm);
	pro_idx_mat = det_idx_nm + Csym_length * tensor1;

	Mat<uword> tensor2 = repmat(linspace<Col<uword>>(0, Nt2 - 1, Nt2), 1, Csym_length - nm);
	pro_idx_nmtail = det_idx_nmtail + Csym_length * tensor2;

	//pro_idx_mat.print("pro idx mat : ");

	pro_idx_mat = pro_idx_mat.as_row();
	pro_idx_nmtail = pro_idx_nmtail.as_row();


	// the main iteration
	for (iter = 0; iter < iterNum; iter++)
	{

		if (iter == 0)
			alpha.each_slice() = gamma;
		else
			alpha = gamma - beta.each_slice();

		//alpha.each_slice() = gamma;
		gamma.zeros();

		/*---------- select the nm lattice to calculate probability -----------*/
		for (j = 0; j < Nr2; j++)
		{
			Mat<float> alpha_vec = alpha.slice(j).as_row();
			alpha_vec = alpha_vec(pro_idx_mat);
			sym_llr.slice(j) = trans(reshape(alpha_vec, nm, Nt2));
			Col<float> maxllr = max(sym_llr.slice(j), 1);
			symllr_max.slice(j).each_col() = maxllr;
		}

		/*------------------------------------*/

		pro_cons = exp(sym_llr - symllr_max);
		pro_sum = sum(pro_cons, 1);
		int islice = 0;
		pro_sum.each_slice([&pro_miu0, &islice, &nm](Mat<float>& X) {pro_miu0.slice(islice) = repmat(X, 1, nm); islice++; });
		pro_cons = pro_cons / pro_miu0;

		/*------------- add damping factor ------------------*/
		if (iter > 0)
		{
			pro_cons = (1 - damp_factor) * pro_cons + damp_factor * pro_temp;
		}
		pro_temp = pro_cons;

		/*---------------------------------*/

		pro_cons.each_slice([&cons_nm](Mat<float>& X) {X %= cons_nm; }); //calculate symbol means
		sGAI = sum(pro_cons, 1);
		pro_cons.each_slice([&cons_nm](Mat<float>& X) {X %= cons_nm; });  // calculate symbol variant
		sGAI_Var = sum(pro_cons, 1);


#if 1
		// 1 // Update beta

		for (j = 0; j < Nr2; j++)
		{
			H_mtx_row = H_mtx.row(j).t();
			mean_col = H_mtx_row % sGAI.slice(j);
			mean_incoming_all = accu(mean_col);
			//square(abs(H_mtx_row))
			var_col = (square(abs(H_mtx_row))) % (sGAI_Var.slice(j) - sGAI.slice(j) % sGAI.slice(j));
			var_incoming_all = accu(var_col);

			// Compute each beta message;
			mean_incoming.each_col() = mean_incoming_all - mean_col; //extrinsic message sum
			var_incoming.each_col() = var_incoming_all - var_col + sigma2;
			HS_0.each_col() = H_mtx_row * cons[0];
			HS.each_col() = H_mtx_row;
			HS = HS % conss_mat;

			fmat llrk = Rx(j) - mean_incoming - HS; // No.k constellation
			llrk = llrk % llrk;
			fmat llr0 = Rx(j) - mean_incoming - HS_0; // No.0 constellation
			llr0 = llr0 % llr0;

			beta.slice(j) = (-llrk + llr0) / 2 / var_incoming;

			/*---------------------------------------*/

			// only keep nm beta value, other beta keeps the minimum value.
			Mat<float> beta_vec = beta.slice(j).as_row();
			beta_vec = beta_vec(pro_idx_mat);
			Col<float> beta_min = min(beta.slice(j), 1);
			Mat<float> beta_set(size(beta.slice(j)));
			beta_set.each_col() = beta_min;
			//beta_set.zeros();
			Mat<float> beta_set_vec = beta_set.as_row();
			beta_set_vec(pro_idx_mat) = beta_vec;
			beta_set = trans(reshape(beta_set_vec, Csym_length, Nt2));
			gamma += beta_set;
			/*---------------------------------------*/

			//beta.slice(j).print("beta is");
			//gamma += beta.slice(j);     //update gamma

		}
#endif

	}



	// pick up the nm+1-th symbol llr
	Mat<float> gamma_vec = gamma.as_row();
	int nmtail = Csym_length - nm;
	//gamma.print("gamma is");

	Col<uword> nmllr_idx = linspace<Col<uword>>(nm - 1, Nt2 * nm - 1, Nt2);
	//nmllr_idx.print("nmllr_idx : ");
	nmllr_idx = pro_idx_mat(nmllr_idx);
	//nmllr_idx.print("nmllr index is");
	Col<float> nmllr = gamma_vec(nmllr_idx);
	Col<float> nmllr_tmp = nmllr - log(Csym_length - nm);


	Mat<float> nmtail_llr(Nt2, Csym_length - nm); // the truncation part need to fill
	float log_coeff = log(geo_coeff);
	Row<float> tmp_coeff = linspace<Row<float>>(1, nmtail, nmtail) * log_coeff * scalar;
	//tmp_coeff.print("tmp coeff :");

	//nmtail_llr.each_row() = tmp_coeff;

	//nmtail_llr.print("nmtail llr : ");

	nmtail_llr.each_col() += nmllr_tmp;
	//nmtail_llr.print("nmtail llr : ");


	/*----------- fill llr compensation back into gamm ------------*/
	Mat<float> gamma_vec1 = gamma.as_row();
	gamma_vec1(pro_idx_nmtail) = nmtail_llr.as_row();
	gamma = trans(reshape(gamma_vec1, Csym_length, Nt2));
	//gamma.print("gamma is");

#if 0

	//gamma.print("gamma is");
	Col<float> gamma_min = min(gamma, 1);
	/*------------ fill llr -------------*/
	Mat<float> gamma_vec = gamma.as_row();
	gamma_vec = gamma_vec(pro_idx_mat);   // select the nm lattice values
	gamma_vec = trans(reshape(gamma_vec, nm, Nt2));
	Col<float> gamma_nm_min = min(gamma_vec, 1);  // select the minimum value within the nm lattice values

	//gamma_min.print("gamma min is");

	/*------------- select the fill lattice -------------*/
	Mat<float> mmse_llr_nmtail = mmse_llr.as_row();
	mmse_llr_nmtail = mmse_llr_nmtail(pro_idx_nmtail); // sorted
	mmse_llr_nmtail = trans(reshape(mmse_llr_nmtail, Csym_length - nm, Nt2));

	Mat<float> mmse_llr_sign = sign(mmse_llr_nmtail);

	for (i = 0; i < Nt2; i++)
	{
		if (gamma_nm_min(i) > 0)
		{
			/*------------calculate the positive LLR---------------*/
			float posNum = sum(mmse_llr_nmtail.row(i) > 0); // calculate the positive number
			Row<float> mmse_llr_nmtail_row = mmse_llr_nmtail.row(i);  // 
			if (posNum > 0)
			{
				/********* llr compensation(arithmetic sequence) ********/

				//Row<float> posllr = linspace<Row<float>>(gamma_nm_min(i), 0, posNum + 2); 
				//mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row > 0)) = posllr.subvec(1, posNum);

				/*----------------------------------------------*/


				/********* llr compensation (geometric sequence) ********/
				Row<float> basevec_pos(posNum, fill::value(factor));
				Row<float> geo_seq_pos = linspace<Row<float>>(-1, -1 * posNum, posNum);
				Row<float> posllr = gamma_nm_min(i) * pow(basevec_pos, geo_seq_pos);
				mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row > 0)) = posllr;
				/*----------------------------------------------*/

			}


			/*-------------calculate the negative LLR-------------------*/
			float negNum = sum(mmse_llr_nmtail.row(i) < 0);
			if (negNum > 0)
			{
				/********* llr compensation(arithmetic sequence) ********/
				//Row<float> negllr = linspace<Row<float>>(0, gamma_min(i), negNum + 1);
				//mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row < 0)) = negllr.subvec(1, negNum);

				/********* llr compensation (geometric sequence) ********/
				Row<float> basevec_neg(negNum, fill::value(factor));
				Row<float> geo_seq_neg = linspace<Row<float>>(-1 * negNum, -1, negNum);
				Row<float> negllr = llr_clipping_neg * pow(basevec_neg, geo_seq_neg);
				//Row<float> negllr = gamma_min(i) * pow(basevec_neg, geo_seq_neg);
				mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row < 0)) = negllr;

			}



			mmse_llr_nmtail.row(i) = mmse_llr_nmtail_row;

		}
		else   // gamma_nm_min(i) <= 0
		{
			/*------------- only need to calculate the negative LLR --------------*/
			Row<float> mmse_llr_nmtail_row = mmse_llr_nmtail.row(i);  // 
			float negNum = sum(mmse_llr_nmtail.row(i) < 0);

			if (negNum > 0)
			{
				/********* llr compensation(arithmetic sequence) ********/
				//Row<float> negllr = linspace<Row<float>>(0, gamma_min(i), negNum + 1);
				//mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row < 0)) = negllr.subvec(1, negNum);

				/********* llr compensation (geometric sequence) ********/
				Row<float> basevec_neg(negNum, fill::value(factor));
				Row<float> geo_seq_neg = linspace<Row<float>>(-1 * negNum, -1, negNum);
				Row<float> negllr = llr_clipping_neg * pow(basevec_neg, geo_seq_neg);
				//Row<float> negllr = gamma_min(i) * pow(basevec_neg, geo_seq_neg);
				mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row < 0)) = negllr;
			}
			mmse_llr_nmtail.row(i) = mmse_llr_nmtail_row;


		}
	}

	/*----------- fill llr compensation back into gamm ------------*/
	Mat<float> gamma_vec1 = gamma.as_row();
	gamma_vec1(pro_idx_nmtail) = mmse_llr_nmtail.as_row();
	gamma = trans(reshape(gamma_vec1, Csym_length, Nt2));
#endif // 0


	//gamma.print("llr inserted gamma is");

#if 0 // Soft output for 16-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		llr[i * symbol_length] = max(gamma[i][2], gamma[i][3]) - max(gamma[i][0], gamma[i][1]);
		llr[i * symbol_length + 1] = max(gamma[i][1], gamma[i][3]) - max(gamma[i][0], gamma[i][2]);
		llr[i * symbol_length + 2] = max(gamma[i + Nt][2], gamma[i + Nt][3]) - max(gamma[i + Nt][0], gamma[i + Nt][1]);
		llr[i * symbol_length + 3] = max(gamma[i + Nt][1], gamma[i + Nt][3]) - max(gamma[i + Nt][0], gamma[i + Nt][2]);
	}
#endif

#if 0 // Soft output for 64-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3] });
		llr[i * symbol_length + 1] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5] });
		llr[i * symbol_length + 2] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 3] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3] });
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6] });
	}
#endif

	if (MODE_TYPE == 8)
	{
#if 1
		//256QAM in armadillo
		// calculate bit-llr using symbol llr
		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(8, 15)) - max(gamma_real.rows(0, 7));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_real.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_real.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(3) = max(gamma_real.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_real.rows(uvec{ 0,2,4,6,8,10,12,14 }));
		// imag part
		llr_mtx.row(4) = max(gamma_imag.rows(8, 15)) - max(gamma_imag.rows(0, 7));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(6) = max(gamma_imag.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(7) = max(gamma_imag.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_imag.rows(uvec{ 0,2,4,6,8,10,12,14 }));

#endif
	}
	else if (MODE_TYPE == 6)
	{
#if 1 // Soft output for 64-QAM modulation using armadillo
		// 64QAM 
		// compute the Bit LLR

		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(4, 7)) - max(gamma_real.rows(0, 3));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 2,3,6,7 })) - max(gamma_real.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 1,3,5,7 })) - max(gamma_real.rows(uvec{ 0,2,4,6 }));
		// imag part
		llr_mtx.row(3) = max(gamma_imag.rows(4, 7)) - max(gamma_imag.rows(0, 3));
		llr_mtx.row(4) = max(gamma_imag.rows(uvec{ 2,3,6,7 })) - max(gamma_imag.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 1,3,5,7 })) - max(gamma_imag.rows(uvec{ 0,2,4,6 }));


#endif
	}
	else if (MODE_TYPE == 4)
	{
#if 1 // Soft output for 16-QAM modulation using armadillo
		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(2, 3)) - max(gamma_real.rows(0, 1));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 1,3 })) - max(gamma_real.rows(uvec{ 0,2 }));
		// imag part
		llr_mtx.row(2) = max(gamma_imag.rows(2, 3)) - max(gamma_imag.rows(0, 1));
		llr_mtx.row(3) = max(gamma_imag.rows(uvec{ 1,3 })) - max(gamma_imag.rows(uvec{ 0,2 }));

#endif
	}
	else
	{
		printf("Invalid modulation type. \n");;
	}


#if 0 // Soft output for 256-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] });
		llr[i * symbol_length + 1] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11] });
		llr[i * symbol_length + 2] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7],gamma[i][10], gamma[i][11],gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5],gamma[i][8], gamma[i][9], gamma[i][12], gamma[i][13] });
		llr[i * symbol_length + 3] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7],gamma[i][9], gamma[i][11],gamma[i][13], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6],gamma[i][8], gamma[i][10], gamma[i][12], gamma[i][14] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11] });
		llr[i * symbol_length + 6] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7],gamma[i + Nt][10], gamma[i + Nt][11],gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5],gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][12], gamma[i + Nt][13] });
		llr[i * symbol_length + 7] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7],gamma[i + Nt][9], gamma[i + Nt][11],gamma[i + Nt][13], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6],gamma[i + Nt][8], gamma[i + Nt][10], gamma[i + Nt][12], gamma[i + Nt][14] });
	}
#endif


}




void bsp_nm_mean_rd_idd_arma_llr(Mat<float>& H_mtx, Col<float> Rx, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma, Mat<uword>det_res_idx, Mat<float>& mmse_llr, Mat<float>& llr_mtx, size_t Nr, size_t Nt, size_t iterNum, size_t nm, size_t Csym_length, float damp_factor, float sigma2,float clipping_neg_value)
{
	size_t i, j, s, iter, k;
	//size_t slen = symbol_length / 2;
	//size_t Csym_length = pow(2, slen);

	//size_t dm = 2; // truncate to 2 symbol in every symbol_factor buffer

	size_t Nt2 = 2 * Nt;
	size_t Nr2 = 2 * Nr;

	float factor = 1.25;
	float llr_clipping_neg = -50;

	float mean_incoming_all = 0;
	float var_incoming_all = 0;

	Col<float> mean_col(Nt2);      // mean of each symbol node
	Col<float> var_col(Nt2);       // varaince of each symbol node
	Col<float> H_mtx_row(Nt2);     //convert the H_mtx_row(j) to Colvec

	Mat<float> mean_incoming(Nt2, Csym_length);  // incoming message's mean of every symbol nodes and their constellations
	Mat<float> var_incoming(Nt2, Csym_length);   // incoming message's variance
	Mat<float> HS_0(Nt2, Csym_length);            // used for the first constellation
	Mat<float> HS(Nt2, Csym_length, fill::zeros);// all constellation

	Mat<float> conss_mat(Nt2, Csym_length);      // reduce time consumption
	conss_mat.each_row() = cons_row;


	/*----------------- new parameters ---------------*/
	Cube<float> pro_miu0(Nt2, nm, Nr2);
	Cube<float> pro_cons(Nt2, nm, Nr2);
	Cube<float> sym_llr(Nt2, nm, Nr2);
	Cube<float> pro_sum(Nt2, 1, Nr2);
	Cube<float> alpha_nm(Nt2, nm, Nr2);
	Cube<float> symllr_max(Nt2, nm, Nr2);
	Mat<float> pro_cons_t(nm, Nt2);
	Mat<uword> pro_idx_mat(Nt2, nm);
	Mat<uword> pro_idx_nmtail(Nt2, Csym_length - nm);
	Row<uword> pro_idx_vec(Nt2 * nm);
	Cube<float> pro_temp(size(pro_cons));
	//Cube<float> pro_temp_1(size(pro_cons));

	Cube<float> sGAI(Nt2, 1, Nr2);
	Cube<float> sGAI_Var(Nt2, 1, Nr2);

	//no matter the iteration
	Mat<uword> det_idx_nm = det_res_idx.cols(0, nm - 1);   // select the constellation used in BsP
	Mat<uword> det_idx_nmtail = det_res_idx.cols(nm, Csym_length - 1); // these lattice need to fill LLR value 


	pro_idx_vec = det_idx_nm.as_row();
	Mat<float> cons_nm = reshape(cons_row(pro_idx_vec), nm, Nt2);
	cons_nm = trans(cons_nm);

	Mat<uword> tensor1 = repmat(linspace<Col<uword>>(0, Nt2 - 1, Nt2), 1, nm);
	pro_idx_mat = det_idx_nm + Csym_length * tensor1;

	Mat<uword> tensor2 = repmat(linspace<Col<uword>>(0, Nt2 - 1, Nt2), 1, Csym_length - nm);
	pro_idx_nmtail = det_idx_nmtail + Csym_length * tensor2;

	pro_idx_mat = pro_idx_mat.as_row();
	pro_idx_nmtail = pro_idx_nmtail.as_row();


	// the main iteration
	for (iter = 0; iter < iterNum; iter++)
	{

		if (iter == 0)
			alpha.each_slice() = gamma;
		else
			alpha = gamma - beta.each_slice();

		//alpha.each_slice() = gamma;
		gamma.zeros();

		/*---------- select the nm lattice to calculate probability -----------*/
		for (j = 0; j < Nr2; j++)
		{
			Mat<float> alpha_vec = alpha.slice(j).as_row();
			alpha_vec = alpha_vec(pro_idx_mat);
			sym_llr.slice(j) = trans(reshape(alpha_vec, nm, Nt2));
			Col<float> maxllr = max(sym_llr.slice(j), 1);
			symllr_max.slice(j).each_col() = maxllr;
		}

		/*------------------------------------*/

		pro_cons = exp(sym_llr - symllr_max);
		pro_sum = sum(pro_cons, 1);
		int islice = 0;
		pro_sum.each_slice([&pro_miu0, &islice, &nm](Mat<float>& X) {pro_miu0.slice(islice) = repmat(X, 1, nm); islice++; });
		pro_cons = pro_cons / pro_miu0;

		/*------------- add damping factor ------------------*/
		if (iter > 0)
		{
			pro_cons = (1 - damp_factor) * pro_cons + damp_factor * pro_temp;
		}
		pro_temp = pro_cons;

		/*---------------------------------*/

		pro_cons.each_slice([&cons_nm](Mat<float>& X) {X %= cons_nm; }); //calculate symbol means
		sGAI = sum(pro_cons, 1);
		pro_cons.each_slice([&cons_nm](Mat<float>& X) {X %= cons_nm; });  // calculate symbol variant
		sGAI_Var = sum(pro_cons, 1);


#if 1
		// 1 // Update beta

		for (j = 0; j < Nr2; j++)
		{
			H_mtx_row = H_mtx.row(j).t();
			mean_col = H_mtx_row % sGAI.slice(j);
			mean_incoming_all = accu(mean_col);
			//square(abs(H_mtx_row))
			var_col = (square(abs(H_mtx_row))) % (sGAI_Var.slice(j) - sGAI.slice(j) % sGAI.slice(j));
			var_incoming_all = accu(var_col);

			// Compute each beta message;
			mean_incoming.each_col() = mean_incoming_all - mean_col; //extrinsic message sum
			var_incoming.each_col() = var_incoming_all - var_col + sigma2;
			HS_0.each_col() = H_mtx_row * cons[0];
			HS.each_col() = H_mtx_row;
			HS = HS % conss_mat;

			fmat llrk = Rx(j) - mean_incoming - HS; // No.k constellation
			llrk = llrk % llrk;
			fmat llr0 = Rx(j) - mean_incoming - HS_0; // No.0 constellation
			llr0 = llr0 % llr0;

			beta.slice(j) = (-llrk + llr0) / 2 / var_incoming;

			/*---------------------------------------*/

			// only keep nm beta value, other beta keeps the minimum value.
			Mat<float> beta_vec = beta.slice(j).as_row();
			beta_vec = beta_vec(pro_idx_mat);
			Col<float> beta_min = min(beta.slice(j), 1);
			Mat<float> beta_set(size(beta.slice(j)));
			beta_set.each_col() = beta_min;
			//beta_set.zeros();
			Mat<float> beta_set_vec = beta_set.as_row();
			beta_set_vec(pro_idx_mat) = beta_vec;
			beta_set = trans(reshape(beta_set_vec, Csym_length, Nt2));
			gamma += beta_set;
			/*---------------------------------------*/

			//beta.slice(j).print("beta is");
			//gamma += beta.slice(j);     //update gamma

		}
#endif

	}

	//gamma.print("gamma is");
	Col<float> gamma_min = min(gamma, 1);
	/*------------ fill llr -------------*/
	Mat<float> gamma_vec = gamma.as_row();
	gamma_vec = gamma_vec(pro_idx_mat);   // select the nm lattice values
	gamma_vec = trans(reshape(gamma_vec, nm, Nt2));
	Col<float> gamma_nm_min = min(gamma_vec, 1);  // select the minimum value within the nm lattice values

	//gamma_min.print("gamma min is");

	/*------------- select the fill lattice -------------*/
	Mat<float> mmse_llr_nmtail = mmse_llr.as_row();
	mmse_llr_nmtail = mmse_llr_nmtail(pro_idx_nmtail); // sorted
	mmse_llr_nmtail = trans(reshape(mmse_llr_nmtail, Csym_length - nm, Nt2));

	Mat<float> mmse_llr_sign = sign(mmse_llr_nmtail);

	for (i = 0; i < Nt2; i++)
	{
		if (gamma_nm_min(i) > 0)
		{
			/*------------calculate the positive LLR---------------*/
			float posNum = sum(mmse_llr_nmtail.row(i) > 0); // calculate the positive number
			Row<float> mmse_llr_nmtail_row = mmse_llr_nmtail.row(i);  // 
			if (posNum > 0)
			{			
				/********* llr compensation(arithmetic sequence) ********/ 

				//Row<float> posllr = linspace<Row<float>>(gamma_nm_min(i), 0, posNum + 2); 
				//mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row > 0)) = posllr.subvec(1, posNum);

				/*----------------------------------------------*/


				/********* llr compensation (geometric sequence) ********/
				Row<float> basevec_pos(posNum, fill::value(factor));
				Row<float> geo_seq_pos = linspace<Row<float>>(-1, -1 * posNum, posNum);
				Row<float> posllr = gamma_nm_min(i) * pow(basevec_pos, geo_seq_pos);
				mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row > 0)) = posllr;
				/*----------------------------------------------*/
					
			}


			/*-------------calculate the negative LLR-------------------*/
			float negNum = sum(mmse_llr_nmtail.row(i) < 0);
			if (negNum > 0)
			{
				/********* llr compensation(arithmetic sequence) ********/
				//Row<float> negllr = linspace<Row<float>>(0, gamma_min(i), negNum + 1);
				//mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row < 0)) = negllr.subvec(1, negNum);

				/********* llr compensation (geometric sequence) ********/
				Row<float> basevec_neg(negNum, fill::value(factor));
				Row<float> geo_seq_neg = linspace<Row<float>>(-1 * negNum, -1, negNum);
				Row<float> negllr = llr_clipping_neg * pow(basevec_neg, geo_seq_neg);
				//Row<float> negllr = gamma_min(i) * pow(basevec_neg, geo_seq_neg);
				mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row < 0)) = negllr;

			}

				
			
			mmse_llr_nmtail.row(i) = mmse_llr_nmtail_row;

		}
		else   // gamma_nm_min(i) <= 0
		{
			/*------------- only need to calculate the negative LLR --------------*/
			Row<float> mmse_llr_nmtail_row = mmse_llr_nmtail.row(i);  // 
			float negNum = sum(mmse_llr_nmtail.row(i) < 0);
				
			if (negNum > 0)
			{
				/********* llr compensation(arithmetic sequence) ********/
				//Row<float> negllr = linspace<Row<float>>(0, gamma_min(i), negNum + 1);
				//mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row < 0)) = negllr.subvec(1, negNum);

				/********* llr compensation (geometric sequence) ********/
				Row<float> basevec_neg(negNum, fill::value(factor));
				Row<float> geo_seq_neg = linspace<Row<float>>(-1 * negNum, -1, negNum);
				Row<float> negllr = llr_clipping_neg * pow(basevec_neg, geo_seq_neg);
				//Row<float> negllr = gamma_min(i) * pow(basevec_neg, geo_seq_neg);
				mmse_llr_nmtail_row.elem(find(mmse_llr_nmtail_row < 0)) = negllr;				
			}
			mmse_llr_nmtail.row(i) = mmse_llr_nmtail_row;


		}
	}
	
	/*----------- fill llr compensation back into gamm ------------*/
	Mat<float> gamma_vec1 = gamma.as_row();
	gamma_vec1(pro_idx_nmtail) = mmse_llr_nmtail.as_row();
	gamma = trans(reshape(gamma_vec1, Csym_length, Nt2));

	//gamma.print("llr inserted gamma is");

#if 0 // Soft output for 16-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		llr[i * symbol_length] = max(gamma[i][2], gamma[i][3]) - max(gamma[i][0], gamma[i][1]);
		llr[i * symbol_length + 1] = max(gamma[i][1], gamma[i][3]) - max(gamma[i][0], gamma[i][2]);
		llr[i * symbol_length + 2] = max(gamma[i + Nt][2], gamma[i + Nt][3]) - max(gamma[i + Nt][0], gamma[i + Nt][1]);
		llr[i * symbol_length + 3] = max(gamma[i + Nt][1], gamma[i + Nt][3]) - max(gamma[i + Nt][0], gamma[i + Nt][2]);
	}
#endif

#if 0 // Soft output for 64-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3] });
		llr[i * symbol_length + 1] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5] });
		llr[i * symbol_length + 2] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 3] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3] });
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6] });
	}
#endif

	if (MODE_TYPE == 8)
	{
#if 1
		//256QAM in armadillo
		// calculate bit-llr using symbol llr
		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(8, 15)) - max(gamma_real.rows(0, 7));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_real.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_real.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(3) = max(gamma_real.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_real.rows(uvec{ 0,2,4,6,8,10,12,14 }));
		// imag part
		llr_mtx.row(4) = max(gamma_imag.rows(8, 15)) - max(gamma_imag.rows(0, 7));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(6) = max(gamma_imag.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(7) = max(gamma_imag.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_imag.rows(uvec{ 0,2,4,6,8,10,12,14 }));

#endif
	}
	else if (MODE_TYPE == 6)
	{
#if 1 // Soft output for 64-QAM modulation using armadillo
		// 64QAM 
		// compute the Bit LLR

		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(4, 7)) - max(gamma_real.rows(0, 3));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 2,3,6,7 })) - max(gamma_real.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 1,3,5,7 })) - max(gamma_real.rows(uvec{ 0,2,4,6 }));
		// imag part
		llr_mtx.row(3) = max(gamma_imag.rows(4, 7)) - max(gamma_imag.rows(0, 3));
		llr_mtx.row(4) = max(gamma_imag.rows(uvec{ 2,3,6,7 })) - max(gamma_imag.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 1,3,5,7 })) - max(gamma_imag.rows(uvec{ 0,2,4,6 }));


#endif
	}
	else if (MODE_TYPE == 4)
	{
#if 1 // Soft output for 16-QAM modulation using armadillo
		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(2, 3)) - max(gamma_real.rows(0, 1));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 1,3 })) - max(gamma_real.rows(uvec{ 0,2 }));
		// imag part
		llr_mtx.row(2) = max(gamma_imag.rows(2, 3)) - max(gamma_imag.rows(0, 1));
		llr_mtx.row(3) = max(gamma_imag.rows(uvec{ 1,3 })) - max(gamma_imag.rows(uvec{ 0,2 }));

#endif
	}
	else
	{
		printf("Invalid modulation type. \n");;
	}


#if 0 // Soft output for 256-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] });
		llr[i * symbol_length + 1] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11] });
		llr[i * symbol_length + 2] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7],gamma[i][10], gamma[i][11],gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5],gamma[i][8], gamma[i][9], gamma[i][12], gamma[i][13] });
		llr[i * symbol_length + 3] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7],gamma[i][9], gamma[i][11],gamma[i][13], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6],gamma[i][8], gamma[i][10], gamma[i][12], gamma[i][14] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11] });
		llr[i * symbol_length + 6] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7],gamma[i + Nt][10], gamma[i + Nt][11],gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5],gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][12], gamma[i + Nt][13] });
		llr[i * symbol_length + 7] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7],gamma[i + Nt][9], gamma[i + Nt][11],gamma[i + Nt][13], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6],gamma[i + Nt][8], gamma[i + Nt][10], gamma[i + Nt][12], gamma[i + Nt][14] });
	}
#endif


}




void bsp_nm_mean_rd_idd_arma(Mat<float>& H_mtx, Col<float> Rx, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma, Mat<uword>det_res_idx, Mat<float>& llr_mtx, size_t Nr, size_t Nt, size_t iterNum, size_t nm, size_t Csym_length, float damp_factor, float sigma2)
{
	size_t i, j, s, iter, k;
	//size_t slen = symbol_length / 2;
	//size_t Csym_length = pow(2, slen);

	//size_t dm = 2; // truncate to 2 symbol in every symbol_factor buffer

	size_t Nt2 = 2 * Nt;
	size_t Nr2 = 2 * Nr;

	float mean_incoming_all = 0;
	float var_incoming_all = 0;

	Col<float> mean_col(Nt2);      // mean of each symbol node
	Col<float> var_col(Nt2);       // varaince of each symbol node
	Col<float> H_mtx_row(Nt2);     //convert the H_mtx_row(j) to Colvec

	Mat<float> mean_incoming(Nt2, Csym_length);  // incoming message's mean of every symbol nodes and their constellations
	Mat<float> var_incoming(Nt2, Csym_length);   // incoming message's variance
	Mat<float> HS_0(Nt2, Csym_length);            // used for the first constellation
	Mat<float> HS(Nt2, Csym_length, fill::zeros);// all constellation

	Mat<float> conss_mat(Nt2, Csym_length);      // reduce time consumption
	conss_mat.each_row() = cons_row;


	/*----------------- new parameters ---------------*/
	Cube<float> pro_miu0(Nt2, nm, Nr2);
	Cube<float> pro_cons(Nt2, nm, Nr2);
	Cube<float> sym_llr(Nt2, nm, Nr2);
	Cube<float> pro_sum(Nt2, 1, Nr2);
	Cube<float> alpha_nm(Nt2, nm, Nr2);
	Cube<float> symllr_max(Nt2, nm, Nr2);
	Mat<float> pro_cons_t(nm, Nt2);
	Mat<uword> pro_idx_mat(Nt2, nm);
	Row<uword> pro_idx_vec(Nt2 * nm);
	Cube<float> pro_temp(size(pro_cons));
	//Cube<float> pro_temp_1(size(pro_cons));

	Cube<float> sGAI(Nt2, 1, Nr2);
	Cube<float> sGAI_Var(Nt2, 1, Nr2);

	//no matter the iteration
	pro_idx_vec = det_res_idx.as_row();
	Mat<float> cons_nm = reshape(cons_row(pro_idx_vec), nm, Nt2);

	cons_nm = trans(cons_nm);


	Mat<uword> tensor = repmat(linspace<Col<uword>>(0, Nt2 - 1, Nt2), 1, nm);
	pro_idx_mat = det_res_idx + Csym_length * tensor;

	pro_idx_mat = pro_idx_mat.as_row();


	// the main iteration
	for (iter = 0; iter < iterNum; iter++)
	{

		if (iter == 0)
			alpha.each_slice() = gamma;
		else
			alpha = gamma - beta.each_slice();

		//alpha.each_slice() = gamma;
		gamma.zeros();


		for (j = 0; j < Nr2; j++)
		{
			Mat<float> alpha_vec = alpha.slice(j).as_row();
			alpha_vec = alpha_vec(pro_idx_mat);
			sym_llr.slice(j) = trans(reshape(alpha_vec, nm, Nt2));
			Col<float> maxllr = max(sym_llr.slice(j), 1);
			symllr_max.slice(j).each_col() = maxllr;

		}

		/*------------------------------------*/

		pro_cons = exp(sym_llr - symllr_max);
		pro_sum = sum(pro_cons, 1);
		int islice = 0;
		pro_sum.each_slice([&pro_miu0, &islice, &nm](Mat<float>& X) {pro_miu0.slice(islice) = repmat(X, 1, nm); islice++; });
		pro_cons = pro_cons / pro_miu0;

		/*------------- add damping factor ------------------*/
		if (iter > 0)
		{
			pro_cons = (1 - damp_factor) * pro_cons + damp_factor * pro_temp;
		}
		pro_temp = pro_cons;

		/*---------------------------------*/

		pro_cons.each_slice([&cons_nm](Mat<float>& X) {X %= cons_nm; });
		sGAI = sum(pro_cons, 1);
		pro_cons.each_slice([&cons_nm](Mat<float>& X) {X %= cons_nm; });
		sGAI_Var = sum(pro_cons, 1);


#if 1
		// 1 // Update beta

		for (j = 0; j < Nr2; j++)
		{
			H_mtx_row = H_mtx.row(j).t();
			mean_col = H_mtx_row % sGAI.slice(j);
			mean_incoming_all = accu(mean_col);
			//square(abs(H_mtx_row))
			var_col = (square(abs(H_mtx_row))) % (sGAI_Var.slice(j) - sGAI.slice(j) % sGAI.slice(j));
			var_incoming_all = accu(var_col);

			// Compute each beta message;
			mean_incoming.each_col() = mean_incoming_all - mean_col; //extrinsic message sum
			var_incoming.each_col() = var_incoming_all - var_col + sigma2;
			HS_0.each_col() = H_mtx_row * cons[0];
			HS.each_col() = H_mtx_row;
			HS = HS % conss_mat;

			fmat llrk = Rx(j) - mean_incoming - HS; // No.k constellation
			llrk = llrk % llrk;
			fmat llr0 = Rx(j) - mean_incoming - HS_0; // No.0 constellation
			llr0 = llr0 % llr0;

			beta.slice(j) = (-llrk + llr0) / 2 / var_incoming;

			/*---------------------------------------*/
			Mat<float> beta_vec = beta.slice(j).as_row();
			beta_vec = beta_vec(pro_idx_mat);
			Col<float> beta_min = min(beta.slice(j), 1);
			Mat<float> beta_set(size(beta.slice(j)));
			beta_set.each_col() = beta_min;
			//beta_set.zeros();
			Mat<float> beta_set_vec = beta_set.as_row();
			beta_set_vec(pro_idx_mat) = beta_vec;
			beta_set = trans(reshape(beta_set_vec, Csym_length, Nt2));
			gamma += beta_set;
			/*---------------------------------------*/

			//beta.slice(j).print("beta is");
			//gamma += beta.slice(j);     //update gamma

		}
#endif

	}

	//gamma.print("gamma is");


#if 0 // Soft output for 16-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		llr[i * symbol_length] = max(gamma[i][2], gamma[i][3]) - max(gamma[i][0], gamma[i][1]);
		llr[i * symbol_length + 1] = max(gamma[i][1], gamma[i][3]) - max(gamma[i][0], gamma[i][2]);
		llr[i * symbol_length + 2] = max(gamma[i + Nt][2], gamma[i + Nt][3]) - max(gamma[i + Nt][0], gamma[i + Nt][1]);
		llr[i * symbol_length + 3] = max(gamma[i + Nt][1], gamma[i + Nt][3]) - max(gamma[i + Nt][0], gamma[i + Nt][2]);
	}
#endif

#if 0 // Soft output for 64-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3] });
		llr[i * symbol_length + 1] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5] });
		llr[i * symbol_length + 2] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 3] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3] });
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6] });
	}
#endif

	if (MODE_TYPE == 8)
	{
#if 1
		//256QAM in armadillo
		// calculate bit-llr using symbol llr
		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(8, 15)) - max(gamma_real.rows(0, 7));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_real.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_real.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(3) = max(gamma_real.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_real.rows(uvec{ 0,2,4,6,8,10,12,14 }));
		// imag part
		llr_mtx.row(4) = max(gamma_imag.rows(8, 15)) - max(gamma_imag.rows(0, 7));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(6) = max(gamma_imag.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(7) = max(gamma_imag.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_imag.rows(uvec{ 0,2,4,6,8,10,12,14 }));

#endif
	}
	else if (MODE_TYPE == 6)
	{
#if 1 // Soft output for 64-QAM modulation using armadillo
		// 64QAM 
		// compute the Bit LLR

		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(4, 7)) - max(gamma_real.rows(0, 3));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 2,3,6,7 })) - max(gamma_real.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 1,3,5,7 })) - max(gamma_real.rows(uvec{ 0,2,4,6 }));
		// imag part
		llr_mtx.row(3) = max(gamma_imag.rows(4, 7)) - max(gamma_imag.rows(0, 3));
		llr_mtx.row(4) = max(gamma_imag.rows(uvec{ 2,3,6,7 })) - max(gamma_imag.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 1,3,5,7 })) - max(gamma_imag.rows(uvec{ 0,2,4,6 }));


#endif
	}
	else if (MODE_TYPE == 4)
	{
#if 1 // Soft output for 16-QAM modulation using armadillo
		fmat gamma_real = gamma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(2, 3)) - max(gamma_real.rows(0, 1));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 1,3 })) - max(gamma_real.rows(uvec{ 0,2 }));
		// imag part
		llr_mtx.row(2) = max(gamma_imag.rows(2, 3)) - max(gamma_imag.rows(0, 1));
		llr_mtx.row(3) = max(gamma_imag.rows(uvec{ 1,3 })) - max(gamma_imag.rows(uvec{ 0,2 }));

#endif
	}
	else
	{
		printf("Invalid modulation type. \n");;
	}


#if 0 // Soft output for 256-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] });
		llr[i * symbol_length + 1] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11] });
		llr[i * symbol_length + 2] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7],gamma[i][10], gamma[i][11],gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5],gamma[i][8], gamma[i][9], gamma[i][12], gamma[i][13] });
		llr[i * symbol_length + 3] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7],gamma[i][9], gamma[i][11],gamma[i][13], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6],gamma[i][8], gamma[i][10], gamma[i][12], gamma[i][14] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11] });
		llr[i * symbol_length + 6] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7],gamma[i + Nt][10], gamma[i + Nt][11],gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5],gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][12], gamma[i + Nt][13] });
		llr[i * symbol_length + 7] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7],gamma[i + Nt][9], gamma[i + Nt][11],gamma[i + Nt][13], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6],gamma[i + Nt][8], gamma[i + Nt][10], gamma[i + Nt][12], gamma[i + Nt][14] });
	}
#endif


}


void exbsp_nm_dm1df1_rd_idd_arma(Mat<float>& H_mtx, Col<float> Rx, float* det_results, size_t symbol_length, size_t Nr, size_t Nt, size_t iterNum, size_t nm, float sigma2, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma_arma, Mat<float>& llr_mtx, float damp_factor)
{
	size_t i, j, s, iter, k;
	size_t slen = symbol_length / 2;
	size_t Csym_length = pow(2, slen);

	size_t dm = 2; // truncate to 2 symbol in every symbol_factor buffer

	size_t Nt2 = 2 * Nt;
	size_t Nr2 = 2 * Nr;

	float mean_incoming_all = 0;
	float var_incoming_all = 0;


	Col<uword> row_index = linspace<Col<uword>>(0, Nt2 - 1, Nt2);
	Row<uword> cub_idx = linspace<Row<uword>>(0, alpha.n_slices - 1, alpha.n_slices) * alpha.n_rows * alpha.n_cols;
	Mat<uword> cub_idx_mat(alpha.n_rows, alpha.n_slices);
	cub_idx_mat.each_row() = cub_idx;
	Col<uword> cub_idx_vec = vectorise(cub_idx_mat);

	Cube<float> Px_1(Nt2, 1, Nr2); // reserve the max LLR symbol probability
	Cube<float> Px_2(Nt2, 1, Nr2); // reserve the second LLR symbol probability

	Cube<float> Px_1_tmp(size(Px_1)); // reserve the max LLR symbol probability
	Cube<float> Px_2_tmp(size(Px_2));
	Col<float> conss_max(Nt2);     // reserve the symbol's constellation of max LLR
	Col<float> conss_sec(Nt2);     // reserve the symbol's constellation of second LLR
	Cube<float> sGAI(Nt2, 1, Nr2);          // mean of the two most likely symbol's constellation

	Col<float> sGAI_col(Nt2);          // mean of the two most likely symbol's constellation
	Col<float> sGAI_Var_col(Nt2);

	Cube<float> sGAI_Var(Nt2, 1, Nr2);      // variance of the two most likely symbol's constellation
	Col<float> mean_col(Nt2);      // mean of each symbol node
	Col<float> var_col(Nt2);       // varaince of each symbol node
	Col<float> H_mtx_row(Nt2);     //convert the H_mtx_row(j) to Colvec

	Mat<float> mean_incoming(Nt2, Csym_length);  // incoming message's mean of every symbol nodes and their constellations
	Mat<float> var_incoming(Nt2, Csym_length);   // incoming message's variance
	Mat<float> HS_0(Nt2, Csym_length);            // used for the first constellation
	Mat<float> HS(Nt2, Csym_length, fill::zeros);// all constellation

	Mat<float> conss_mat(Nt2, Csym_length);      // reduce time consumption
	conss_mat.each_row() = cons_row;

	vector<size_t> idx_ems(Csym_length, 0);
	vector<float> alpha_ems(Csym_length, 0);

	Cube<uword> alpha_sec_idx(Nt2, 1, Nr2);
	Cube<uword> alpha_max_idx(Nt2, 1, Nr2);

	Cube<float> alpha_sec_val(Nt2, 1, Nr2);
	Cube<float> alpha_max_val(Nt2, 1, Nr2);


	/**********************************************/
	Cube<float> sym_llr(Nt2, nm, Nr2);
	Cube<float> symllr_max(Nt2, nm, Nr2);
	Cube<float> pro_miu0(Nt2, nm, Nr2);
	Cube<float> pro_cons(Nt2, nm, Nr2);
	Cube<float> pro_temp(Nt2, nm, Nr2);
	Cube<float> pro_sum(Nt2, 1, Nr2);

	Col<uword> sym_idx(Csym_length);
	//Cube<uword> sym_idx(Nt2, Csym_length, Nr2);
	Cube<float> cons_nm(Nt2, nm, Nr2);

	/**********************************************/


	// the main iteration
	for (iter = 0; iter < iterNum; iter++)
	{
		// Update alpha message and Px
		if (iter == 0)
			alpha.each_slice() = gamma_arma;
		else
			alpha = gamma_arma - beta.each_slice();

		gamma_arma.zeros();


		/*********************************************/
		/*********************************************/
#if 1
		for (j = 0; j < Nr2; j++)
		{
			Mat<float> alp_val_sort = sort(alpha.slice(j), "descend", 1);
			sym_llr.slice(j) = alp_val_sort.cols(0, nm - 1);
			//Mat<float> alp_idx_sort = 
			Col<float> maxllr = max(sym_llr.slice(j), 1);
			symllr_max.slice(j).each_col() = maxllr;

			int irow = 0;
			alpha.slice(j).each_row([&irow, &sym_idx, &j, &cons_nm, &nm](Row<float>& X) {sym_idx = sort_index(X, "descend"); cons_nm.slice(j).row(irow) = cons_col(sym_idx.subvec(0, nm - 1)).t(); irow++; });

		}

		/*------------------------------------*/

		pro_cons = exp(sym_llr - symllr_max);
		pro_sum = sum(pro_cons, 1);
		int islice = 0;
		pro_sum.each_slice([&pro_miu0, &islice, &nm](Mat<float>& X) {pro_miu0.slice(islice) = repmat(X, 1, nm); islice++; });
		pro_cons = pro_cons / pro_miu0;

		/*------------- add damping factor ------------------*/
		if (iter > 0)
		{
			pro_cons = (1 - damp_factor) * pro_cons + damp_factor * pro_temp;
		}
		pro_temp = pro_cons;
		//pro_cons.slice(0).print("pro_cons is");
		/*---------------------------------*/

		//pro_cons.each_slice([&cons_nm](Mat<float>& X) {X %= cons_nm; });
		pro_cons = pro_cons % cons_nm;
		sGAI = sum(pro_cons, 1);
		//pro_cons.each_slice([&cons_nm](Mat<float>& X) {X %= cons_nm; });
		pro_cons = pro_cons % cons_nm;
		sGAI_Var = sum(pro_cons, 1);

		//sGAI.slice(0).print("sGAI is");
		//sGAI_Var.slice(0).print("sGAI Var is");

#endif




		/*********************************************/
		/*********************************************/

#if 0
		Cube<uword> alpha_max_idx = index_max(alpha, 1);  // save the max llr index 
		Cube<float> alpha_max_value = max(alpha, 1);      // save the max llr
		Cube<uword> alpha_ele_idx = Nt2 * alpha_max_idx;
		alpha_ele_idx.each_slice() += row_index;
		Col<uword> alpha_ele_vec = vectorise(alpha_ele_idx);
		alpha_ele_vec += cub_idx_vec;
		alpha.elem(alpha_ele_vec).fill(-1000);
		Cube<uword> alpha_sec_idx = index_max(alpha, 1);   // save the second llr index
		Cube<float> alpha_sec_value = max(alpha, 1);       // save the second llr value

		Cube<float> pTemp = exp(alpha_sec_value - alpha_max_value); // llr represented by most and second likely symbol

		Px_1 = (float)1.0 / ((float)1.0 + pTemp);		  // probability of the most likely symbol
		Px_2 = pTemp / ((float)1.0 + pTemp);				  // probability of the second likely symbol (the two pro add together to 1)

		if (iter > 0)
		{
			Px_1 = (1 - damp_factor) * Px_1 + damp_factor * Px_1_tmp;
			Px_2 = (1 - damp_factor) * Px_2 + damp_factor * Px_2_tmp;
			Px_2 = 1 - Px_1;
		}

		Px_1_tmp = Px_1;
		Px_2_tmp = Px_2;

		Px_1.slice(0).print("px 1 is");
		Px_2.slice(0).print("px 2 is");

#endif

#if 1
		// 1 // Update Px without Sort

		for (j = 0; j < Nr2; j++)
		{
			H_mtx_row = H_mtx.row(j).t();

			mean_col = H_mtx_row % sGAI.slice(j);
			mean_incoming_all = accu(mean_col);
			//square(abs(H_mtx_row))
			var_col = (square(abs(H_mtx_row))) % (sGAI_Var.slice(j) - sGAI.slice(j) % sGAI.slice(j));
			var_incoming_all = accu(var_col);

			// Compute each beta message;
			mean_incoming.each_col() = mean_incoming_all - mean_col; //extrinsic message sum
			var_incoming.each_col() = var_incoming_all - var_col + sigma2;
			HS_0.each_col() = H_mtx_row * cons[0];
			HS.each_col() = H_mtx_row;
			HS = HS % conss_mat;

			fmat llrk = Rx(j) - mean_incoming - HS; // No.k constellation
			llrk = llrk % llrk;
			fmat llr0 = Rx(j) - mean_incoming - HS_0; // No.0 constellation
			llr0 = llr0 % llr0;

			beta.slice(j) = (-llrk + llr0) / 2 / var_incoming;
			gamma_arma += beta.slice(j);     //update gamma
		}
#endif

	}

	if (MODE_TYPE == 8)
	{
#if 1
		//256QAM in armadillo
		// calculate bit-llr using symbol llr
		fmat gamma_real = gamma_arma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma_arma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(8, 15)) - max(gamma_real.rows(0, 7));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_real.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_real.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(3) = max(gamma_real.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_real.rows(uvec{ 0,2,4,6,8,10,12,14 }));
		// imag part
		llr_mtx.row(4) = max(gamma_imag.rows(8, 15)) - max(gamma_imag.rows(0, 7));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(6) = max(gamma_imag.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(7) = max(gamma_imag.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_imag.rows(uvec{ 0,2,4,6,8,10,12,14 }));

#endif
	}
	else if (MODE_TYPE == 6)
	{
#if 1 // Soft output for 64-QAM modulation using armadillo
		// 64QAM 
		// compute the Bit LLR

		fmat gamma_real = gamma_arma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma_arma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(4, 7)) - max(gamma_real.rows(0, 3));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 2,3,6,7 })) - max(gamma_real.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 1,3,5,7 })) - max(gamma_real.rows(uvec{ 0,2,4,6 }));
		// imag part
		llr_mtx.row(3) = max(gamma_imag.rows(4, 7)) - max(gamma_imag.rows(0, 3));
		llr_mtx.row(4) = max(gamma_imag.rows(uvec{ 2,3,6,7 })) - max(gamma_imag.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 1,3,5,7 })) - max(gamma_imag.rows(uvec{ 0,2,4,6 }));


#endif
	}
	else if (MODE_TYPE == 4)
	{
#if 1 // Soft output for 16-QAM modulation using armadillo
		fmat gamma_real = gamma_arma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma_arma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(2, 3)) - max(gamma_real.rows(0, 1));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 1,3 })) - max(gamma_real.rows(uvec{ 0,2 }));
		// imag part
		llr_mtx.row(2) = max(gamma_imag.rows(2, 3)) - max(gamma_imag.rows(0, 1));
		llr_mtx.row(3) = max(gamma_imag.rows(uvec{ 1,3 })) - max(gamma_imag.rows(uvec{ 0,2 }));

#endif
	}
	else
	{
		printf("Invalid modulation type. \n");;
	}


#if 0 // Soft output for 256-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] });
		llr[i * symbol_length + 1] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11] });
		llr[i * symbol_length + 2] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7],gamma[i][10], gamma[i][11],gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5],gamma[i][8], gamma[i][9], gamma[i][12], gamma[i][13] });
		llr[i * symbol_length + 3] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7],gamma[i][9], gamma[i][11],gamma[i][13], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6],gamma[i][8], gamma[i][10], gamma[i][12], gamma[i][14] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11] });
		llr[i * symbol_length + 6] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7],gamma[i + Nt][10], gamma[i + Nt][11],gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5],gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][12], gamma[i + Nt][13] });
		llr[i * symbol_length + 7] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7],gamma[i + Nt][9], gamma[i + Nt][11],gamma[i + Nt][13], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6],gamma[i + Nt][8], gamma[i + Nt][10], gamma[i + Nt][12], gamma[i + Nt][14] });
	}
#endif


}





void bsp_mean_dm1df1_rd_idd_arma(Mat<float>& H_mtx, Col<float> Rx, float* det_results, size_t symbol_length, size_t Nr, size_t Nt, size_t iterNum, size_t nm, float sigma2, Cube<float>& beta, Cube<float>& alpha, Mat<float>& gamma_arma, Mat<float>& llr_mtx,  float damp_factor)
{
	size_t i, j, s, iter, k;
	size_t slen = symbol_length / 2;
	size_t Csym_length = pow(2, slen);

	size_t dm = 2; // truncate to 2 symbol in every symbol_factor buffer

	size_t Nt2 = 2 * Nt;
	size_t Nr2 = 2 * Nr;

	float mean_incoming_all = 0;
	float var_incoming_all = 0;


	Col<uword> row_index = linspace<Col<uword>>(0, Nt2 - 1, Nt2);
	Row<uword> cub_idx = linspace<Row<uword>>(0, alpha.n_slices - 1, alpha.n_slices) * alpha.n_rows * alpha.n_cols;
	Mat<uword> cub_idx_mat(alpha.n_rows, alpha.n_slices);
	cub_idx_mat.each_row() = cub_idx;
	Col<uword> cub_idx_vec = vectorise(cub_idx_mat);

	Cube<float> Px_1(Nt2, 1, Nr2); // reserve the max LLR symbol probability
	Cube<float> Px_2(Nt2, 1, Nr2); // reserve the second LLR symbol probability

	Cube<float> Px_1_tmp(size(Px_1)); // reserve the max LLR symbol probability
	Cube<float> Px_2_tmp(size(Px_2));
	Col<float> conss_max(Nt2);     // reserve the symbol's constellation of max LLR
	Col<float> conss_sec(Nt2);     // reserve the symbol's constellation of second LLR
	Col<float> sGAI(Nt2);          // mean of the two most likely symbol's constellation
	Col<float> sGAI_Var(Nt2);      // variance of the two most likely symbol's constellation
	Col<float> mean_col(Nt2);      // mean of each symbol node
	Col<float> var_col(Nt2);       // varaince of each symbol node
	Col<float> H_mtx_row(Nt2);     //convert the H_mtx_row(j) to Colvec

	Mat<float> mean_incoming(Nt2, Csym_length);  // incoming message's mean of every symbol nodes and their constellations
	Mat<float> var_incoming(Nt2, Csym_length);   // incoming message's variance
	Mat<float> HS_0(Nt2, Csym_length);            // used for the first constellation
	Mat<float> HS(Nt2, Csym_length, fill::zeros);// all constellation

	Mat<float> conss_mat(Nt2, Csym_length);      // reduce time consumption
	conss_mat.each_row() = cons_row;

	vector<size_t> idx_ems(Csym_length, 0);
	vector<float> alpha_ems(Csym_length, 0);

	Cube<uword> alpha_sec_idx(Nt2, 1, Nr2);
	Cube<uword> alpha_max_idx(Nt2, 1, Nr2);

	Cube<float> alpha_sec_val(Nt2, 1, Nr2);
	Cube<float> alpha_max_val(Nt2, 1, Nr2);


	/**********************************************/
	Cube<float> sym_llr(Nt2, nm, Nr2);
	Cube<float> symllr_max(Nt2, nm, Nr2);
	Cube<float> pro_miu0(Nt2, nm, Nr2);
	Cube<float> pro_cons(Nt2, nm, Nr2);
	Cube<float> pro_temp(Nt2, nm, Nr2);
	Cube<float> pro_sum(Nt2, 1, Nr2);

	Col<uword> sym_idx(Csym_length);
	//Cube<uword> sym_idx(Nt2, Csym_length, Nr2);
	Cube<float> cons_nm(Nt2, nm, Nr2);

	/**********************************************/


	// the main iteration
	for (iter = 0; iter < iterNum; iter++)
	{
		// Update alpha message and Px
		if (iter == 0)
			alpha.each_slice() = gamma_arma;
		else
			alpha = gamma_arma - beta.each_slice();

		gamma_arma.zeros();

#if 1
		Cube<uword> alpha_max_idx = index_max(alpha, 1);  // save the max llr index 
		Cube<float> alpha_max_value = max(alpha, 1);      // save the max llr
		Cube<uword> alpha_ele_idx = Nt2 * alpha_max_idx;
		alpha_ele_idx.each_slice() += row_index;
		Col<uword> alpha_ele_vec = vectorise(alpha_ele_idx);
		alpha_ele_vec += cub_idx_vec;
		alpha.elem(alpha_ele_vec).fill(-1000);
		Cube<uword> alpha_sec_idx = index_max(alpha, 1);   // save the second llr index
		Cube<float> alpha_sec_value = max(alpha, 1);       // save the second llr value

		Cube<float> pTemp = exp(alpha_sec_value - alpha_max_value); // llr represented by most and second likely symbol

		Px_1 = (float)1.0 / ((float)1.0 + pTemp);		  // probability of the most likely symbol
		Px_2 = pTemp / ((float)1.0 + pTemp);				  // probability of the second likely symbol (the two pro add together to 1)

		if (iter > 0)
		{
			Px_1 = (1 - damp_factor) * Px_1 + damp_factor * Px_1_tmp;
			Px_2 = (1 - damp_factor) * Px_2 + damp_factor * Px_2_tmp;
			Px_2 = 1 - Px_1;
		}

		Px_1_tmp = Px_1;
		Px_2_tmp = Px_2;

#endif

#if 1
		// 1 // Update Px without Sort

		for (j = 0; j < Nr2; j++)
		{
			
			// first step: compute the total GAI: u = h_i \hat{s}. // mean_all and var_all;
			conss_max = cons_col.elem(alpha_max_idx.slice(j));  // the max llr's constellation	
			conss_sec = cons_col.elem(alpha_sec_idx.slice(j));  // the second llr's constellation
			sGAI = Px_1.slice(j) % conss_max + Px_2.slice(j) % conss_sec;  //  constellation mean
			//sGAI =  conss_max;  //  constellation mean
			sGAI_Var = Px_1.slice(j) % conss_max % conss_max + Px_2.slice(j) % conss_sec % conss_sec; // constellation var
			//sGAI_Var = conss_max % conss_max; // constellation var


			H_mtx_row = H_mtx.row(j).t();

			mean_col = H_mtx_row % sGAI;   // all incoming message to every symbol node    
			mean_incoming_all = accu(mean_col);
			var_col = (H_mtx_row % H_mtx_row) % (sGAI_Var - sGAI % sGAI); // all incoming variance to every symbol node
			var_incoming_all = accu(var_col);

			// Compute each beta message;
			mean_incoming.each_col() = mean_incoming_all - mean_col; //extrinsic message sum
			var_incoming.each_col() = var_incoming_all - var_col + sigma2;
			HS_0.each_col() = H_mtx_row * cons[0];
			HS.each_col() = H_mtx_row;
			HS = HS % conss_mat;

			fmat llrk = Rx(j) - mean_incoming - HS; // No.k constellation
			llrk = llrk % llrk;
			fmat llr0 = Rx(j) - mean_incoming - HS_0; // No.0 constellation
			llr0 = llr0 % llr0;

			beta.slice(j) = (-llrk + llr0) / 2 / var_incoming;
			gamma_arma += beta.slice(j);     //update gamma
		}
#endif

	}

	if (MODE_TYPE == 8)
	{
#if 1
		//256QAM in armadillo
		// calculate bit-llr using symbol llr
		fmat gamma_real = gamma_arma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma_arma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(8, 15)) - max(gamma_real.rows(0, 7));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_real.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_real.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(3) = max(gamma_real.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_real.rows(uvec{ 0,2,4,6,8,10,12,14 }));
		// imag part
		llr_mtx.row(4) = max(gamma_imag.rows(8, 15)) - max(gamma_imag.rows(0, 7));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 4,5,6,7,12,13,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,2,3,8,9,10,11 }));
		llr_mtx.row(6) = max(gamma_imag.rows(uvec{ 2,3,6,7,10,11,14,15 })) - max(gamma_imag.rows(uvec{ 0,1,4,5,8,9,12,13 }));
		llr_mtx.row(7) = max(gamma_imag.rows(uvec{ 1,3,5,7,9,11,13,15 })) - max(gamma_imag.rows(uvec{ 0,2,4,6,8,10,12,14 }));

#endif
		}
	else if (MODE_TYPE == 6)
	{
#if 1 // Soft output for 64-QAM modulation using armadillo
		// 64QAM 
		// compute the Bit LLR

		fmat gamma_real = gamma_arma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma_arma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(4, 7)) - max(gamma_real.rows(0, 3));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 2,3,6,7 })) - max(gamma_real.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(2) = max(gamma_real.rows(uvec{ 1,3,5,7 })) - max(gamma_real.rows(uvec{ 0,2,4,6 }));
		// imag part
		llr_mtx.row(3) = max(gamma_imag.rows(4, 7)) - max(gamma_imag.rows(0, 3));
		llr_mtx.row(4) = max(gamma_imag.rows(uvec{ 2,3,6,7 })) - max(gamma_imag.rows(uvec{ 0,1,4,5 }));
		llr_mtx.row(5) = max(gamma_imag.rows(uvec{ 1,3,5,7 })) - max(gamma_imag.rows(uvec{ 0,2,4,6 }));


#endif
	}
	else if (MODE_TYPE == 4)
	{
#if 1 // Soft output for 16-QAM modulation using armadillo
		fmat gamma_real = gamma_arma.head_rows(Nt).t(); // real part llr  size is slen*nt 16x64A
		fmat gamma_imag = gamma_arma.rows(Nt, Nt2 - 1).t();

		// real part
		llr_mtx.row(0) = max(gamma_real.rows(2, 3)) - max(gamma_real.rows(0, 1));
		llr_mtx.row(1) = max(gamma_real.rows(uvec{ 1,3 })) - max(gamma_real.rows(uvec{ 0,2 }));
		// imag part
		llr_mtx.row(2) = max(gamma_imag.rows(2, 3)) - max(gamma_imag.rows(0, 1));
		llr_mtx.row(3) = max(gamma_imag.rows(uvec{ 1,3 })) - max(gamma_imag.rows(uvec{ 0,2 }));

#endif
	}
	else
	{
		printf("Invalid modulation type. \n");;
	}


#if 0 // Soft output for 256-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] });
		llr[i * symbol_length + 1] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11] });
		llr[i * symbol_length + 2] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7],gamma[i][10], gamma[i][11],gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5],gamma[i][8], gamma[i][9], gamma[i][12], gamma[i][13] });
		llr[i * symbol_length + 3] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7],gamma[i][9], gamma[i][11],gamma[i][13], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6],gamma[i][8], gamma[i][10], gamma[i][12], gamma[i][14] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11] });
		llr[i * symbol_length + 6] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7],gamma[i + Nt][10], gamma[i + Nt][11],gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5],gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][12], gamma[i + Nt][13] });
		llr[i * symbol_length + 7] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7],gamma[i + Nt][9], gamma[i + Nt][11],gamma[i + Nt][13], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6],gamma[i + Nt][8], gamma[i + Nt][10], gamma[i + Nt][12], gamma[i + Nt][14] });
	}
#endif


}



void bsp_mean_dm1df1_rd_idd(float* H_mtx, float* Rx, float* det_results, float* tmp_inv_mtx, float* llr, size_t symbol_length, size_t Nr, size_t Nt, size_t iterNum, float sigma2,
	float delta, float*** Px, float* sMean, float* sVar, float*** beta, float** gamma, float*** alpha, size_t*** sIndex)
{
	size_t i, j, s, iter, k;
	size_t slen = symbol_length / 2;

	size_t Csym_length = pow(2, slen);
	vector<size_t> idx_ems(Csym_length, 0);
	vector<float> alpha_ems(Csym_length, 0);

	size_t dm = 2;

	size_t Nt2 = 2 * Nt;
	size_t Nr2 = 2 * Nr;


#if 0
	// initialization for priori probability: with MMSE detector
	for (i = 0; i < Nt2; i++)
	{
		float s_temp = HUGE_VALF;
		size_t s_index = 0;

		for (s = 0; s < Csym_length; s++)
		{
			if (abs(det_results[i] - cons[s]) < s_temp)
			{
				s_temp = abs(det_results[i] - cons[s]);
				s_index = s;
			}
		}

		for (j = 0; j < Nr2; j++)
		{
			for (k = 0; k < Csym_length; k++)
			{
				Px[i][j][k] = 0;
			}
			sIndex[i][j][0] = s_index;

			// If s_index == 0 不能计算两次Px[i][j][s_index]*cons[s_index]；
			if (s_index == 0)
				sIndex[i][j][1] = 1;

			Px[i][j][0] = 1.0;
		}
	}
#endif


#if 0  // initialize with MMSE; 简化版本

	for (i = 0; i < Nt2; i++)
	{
		for (s = 0; s < Csym_length; s++)
		{
			gamma[i][s] = -abs(cons[s] - det_results[i]);
			for (j = 0; j < Nr2; j++)
			{
				alpha[i][j][s] = gamma[i][s];
			}
		}
	}

#endif // 0


#if 0 // initialize with MMSE; // initial Gamma message
	for (i = 0; i < Nt2; i++)
	{
		float coVar = 2 * abs(tmp_inv_mtx[i * Nt2 + i]);

		// 概率归一化
		float Px_sum = 0;
		for (s = 0; s < Csym_length; s++)
		{
			float fracA = -pow(cons[s] - det_results[i], 2);
			Px_sum += exp(fracA / coVar);
		}

		for (s = 0; s < Csym_length; s++)
		{
			float fracA = -pow(cons[s] - det_results[i], 2);
			Px[i][0][s] = exp(fracA / coVar) / Px_sum;
			gamma[i][s] = log(Px[i][0][s] / Px[i][0][0]);

			for (j = 0; j < Nr2; j++)
			{
				alpha[i][j][s] = gamma[i][s];
			}
		}
	}
#endif


	// the main iteration
	for (iter = 0; iter < iterNum; iter++)
	{
#if 1 // Update alpha message and Px

		for (j = 0; j < Nr2; j++)
		{
			for (i = 0; i < Nt2; i++)
			{
				float max_temp = -HUGE_VALF;
				float mid_temp = -HUGE_VALF;
				for (s = 0; s < Csym_length; s++)
				{
					if (iter == 0)
						alpha[i][j][s] = gamma[i][s];
					else
						alpha[i][j][s] = gamma[i][s] - beta[j][i][s];

					alpha_ems[s] = alpha[i][j][s];
					idx_ems[s] = s;

				}

				sort(idx_ems.begin(), idx_ems.begin() + Csym_length, CompLarge(alpha_ems));

				float pTemp = exp(alpha_ems[idx_ems[1]] - alpha_ems[idx_ems[0]]); 
				//第二大的星座点-最大星座点

				sIndex[i][j][0] = idx_ems[0]; //最大值下标
				sIndex[i][j][1] = idx_ems[1];  //第二大值的下标

				Px[i][j][0] = (float)1.0 / ((float)1.0 + pTemp);
				Px[i][j][1] = pTemp / ((float)1.0 + pTemp);
			}
		}

#endif // 1 // Update Px without Sort


		for (j = 0; j < Nr2; j++)
		{
			// first step: compute the total GAI: u = h_i \hat{s}. // mean_all and var_all;
			float mean_incoming_all = 0;
			float var_incoming_all = 0;


			for (i = 0; i < Nt2; i++)
			{
				float sGAI = 0;
				float sGAI_Var = 0;
				for (s = 0; s < dm; s++)
				{
					size_t indx = sIndex[i][j][s];
					sGAI += Px[i][j][s] * cons[indx];
					sGAI_Var += Px[i][j][s] * cons[indx] * cons[indx];
				}

				sMean[i] = H_mtx[j * Nt2 + i] * sGAI;
				sVar[i] = H_mtx[j * Nt2 + i] * H_mtx[j * Nt2 + i] * (sGAI_Var - sGAI * sGAI);

				mean_incoming_all += sMean[i];
				var_incoming_all += sVar[i];
			}

			// Compute each beta message;
			for (i = 0; i < Nt2; i++)
			{
				float sMean_in = mean_incoming_all - sMean[i];
				float sVar_in = var_incoming_all - sVar[i] + sigma2;
				float HS_0 = H_mtx[j * Nt2 + i] * cons[0];
				for (s = 1; s < Csym_length; s++)
				{
					float HS = H_mtx[j * Nt2 + i] * cons[s];
					beta[j][i][s] = -pow(Rx[j] - sMean_in - HS, 2) / 2 / sVar_in
						+ pow(Rx[j] - sMean_in - HS_0, 2) / 2 / sVar_in;
				}
			}
		}

		for (i = 0; i < Nt2; i++)
		{
			for (s = 0; s < Csym_length; s++)
			{
				gamma[i][s] = 0;
				for (j = 0; j < Nr2; j++)
				{
					gamma[i][s] += beta[j][i][s];
				}
			}
		}

	}


#if 0 // Soft output for 16-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		llr[i * symbol_length] = max(gamma[i][2], gamma[i][3]) - max(gamma[i][0], gamma[i][1]);
		llr[i * symbol_length + 1] = max(gamma[i][1], gamma[i][3]) - max(gamma[i][0], gamma[i][2]);
		llr[i * symbol_length + 2] = max(gamma[i + Nt][2], gamma[i + Nt][3]) - max(gamma[i + Nt][0], gamma[i + Nt][1]);
		llr[i * symbol_length + 3] = max(gamma[i + Nt][1], gamma[i + Nt][3]) - max(gamma[i + Nt][0], gamma[i + Nt][2]);
	}
#endif

#if 0 // Soft output for 64-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3] });
		llr[i * symbol_length + 1] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5] });
		llr[i * symbol_length + 2] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 3] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3] });
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6] });
	}
#endif


#if 1 // Soft output for 256-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] });
		llr[i * symbol_length + 1] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11] });
		llr[i * symbol_length + 2] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7],gamma[i][10], gamma[i][11],gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5],gamma[i][8], gamma[i][9], gamma[i][12], gamma[i][13] });
		llr[i * symbol_length + 3] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7],gamma[i][9], gamma[i][11],gamma[i][13], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6],gamma[i][8], gamma[i][10], gamma[i][12], gamma[i][14] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11] });
		llr[i * symbol_length + 6] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7],gamma[i + Nt][10], gamma[i + Nt][11],gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5],gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][12], gamma[i + Nt][13] });
		llr[i * symbol_length + 7] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7],gamma[i + Nt][9], gamma[i + Nt][11],gamma[i + Nt][13], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6],gamma[i + Nt][8], gamma[i + Nt][10], gamma[i + Nt][12], gamma[i + Nt][14] });
	}
#endif


}



void bsp_dm1df1_rd(float* H_mtx, float* Rx, float* det_results, float* tmp_inv_mtx, float* llr, size_t symbol_length, size_t Nr, size_t Nt, size_t iterNum, float sigma2,
	float delta, float*** beta, float** gamma, float*** alpha, size_t* ems_sym)
{
	size_t i, j, s, iter, k;
	size_t slen = symbol_length / 2;

	size_t Csym_length = pow(2, slen);
	vector<size_t> idx_ems(Csym_length, 0);
	vector<float> alpha_ems(Csym_length, 0);

	size_t Nt2 = 2 * Nt;
	size_t Nr2 = 2 * Nr;


#if 0 // initialize with MMSE; // initial Gamma message
	for (i = 0; i < Nt2; i++)
	{
		float coVar = 2 * abs(tmp_inv_mtx[i * Nt2 + i]);

		// 概率归一化
		float Px_sum = 0;
		for (s = 0; s < Csym_length; s++)
		{
			float fracA = -pow(cons[s] - det_results[i], 2);
			Px_sum += exp(fracA / coVar);
		}

		for (s = 0; s < Csym_length; s++)
		{
			float fracA = -pow(cons[s] - det_results[i], 2);
			Px[i][0][s] = exp(fracA / coVar) / Px_sum;
			gamma[i][s] = log(Px[i][0][s] / Px[i][0][0]);

			for (j = 0; j < Nr2; j++)
			{
				alpha[i][j][s] = gamma[i][s];
			}
		}
	}
#endif

	// the main iteration
	for (iter = 0; iter < iterNum; iter++)
	{

		for (j = 0; j < Nr2; j++)
		{
			for (i = 0; i < Nt2; i++)
			{
				for (s = 0; s < Csym_length; s++)
				{
					alpha[i][j][s] = gamma[i][s] - beta[j][i][s];
				}
			}
		}

		for (j = 0; j < Nr2; j++)
		{
			// step2: update the beta messages
			float R_incoming_all = 0;
			// EMS sort
			for (k = 0; k < Nt2; k++)
			{
				for (s = 0; s < Csym_length; s++)
				{
					alpha_ems[s] = alpha[k][j][s];
					idx_ems[s] = s;
				}
				sort(idx_ems.begin(), idx_ems.begin() + Csym_length, CompLarge(alpha_ems));
				ems_sym[k] = idx_ems[0];
				R_incoming_all += H_mtx[j * Nt2 + k] * cons[ems_sym[k]];
			}

			// Compute each beta message;
			for (i = 0; i < Nt2; i++)
			{
				float R_incoming = R_incoming_all - H_mtx[j * Nt2 + i] * cons[ems_sym[i]];
				float HS_0 = H_mtx[j * Nt2 + i] * cons[0];
				for (s = 1; s < Csym_length; s++)
				{
					float HS = H_mtx[j * Nt2 + i] * cons[s];
					beta[j][i][s] = -pow(Rx[j] - R_incoming - HS, 2) / 2 / sigma2
						+ pow(Rx[j] - R_incoming - HS_0, 2) / 2 / sigma2;
				}
			}
		}

		for (i = 0; i < Nt2; i++)
		{
			for (s = 0; s < Csym_length; s++)
			{
				gamma[i][s] = 0;
				for (j = 0; j < Nr2; j++)
				{
					gamma[i][s] += beta[j][i][s];
				}
			}
		}

	}


#if 0 // Soft output for 16-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		llr[i * symbol_length] = max(gamma[i][2], gamma[i][3]) - max(gamma[i][0], gamma[i][1]);
		llr[i * symbol_length + 1] = max(gamma[i][1], gamma[i][3]) - max(gamma[i][0], gamma[i][2]);
		llr[i * symbol_length + 2] = max(gamma[i + Nt][2], gamma[i + Nt][3]) - max(gamma[i + Nt][0], gamma[i + Nt][1]);
		llr[i * symbol_length + 3] = max(gamma[i + Nt][1], gamma[i + Nt][3]) - max(gamma[i + Nt][0], gamma[i + Nt][2]);
	}
#endif

#if 0 // Soft output for 64-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3] });
		llr[i * symbol_length + 1] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5] });
		llr[i * symbol_length + 2] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 3] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3] });
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6] });
	}
#endif


#if 1 // Soft output for 256-QAM modulation
	// compute the Bit LLR
	for (i = 0; i < Nt; i++)
	{
		// bits with respect to the real part;
		llr[i * symbol_length] = max({ gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7] });
		llr[i * symbol_length + 1] = max({ gamma[i][4], gamma[i][5], gamma[i][6], gamma[i][7], gamma[i][12], gamma[i][13], gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][2], gamma[i][3], gamma[i][8], gamma[i][9], gamma[i][10], gamma[i][11] });
		llr[i * symbol_length + 2] = max({ gamma[i][2], gamma[i][3], gamma[i][6], gamma[i][7],gamma[i][10], gamma[i][11],gamma[i][14], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][1], gamma[i][4], gamma[i][5],gamma[i][8], gamma[i][9], gamma[i][12], gamma[i][13] });
		llr[i * symbol_length + 3] = max({ gamma[i][1], gamma[i][3], gamma[i][5], gamma[i][7],gamma[i][9], gamma[i][11],gamma[i][13], gamma[i][15] }) -
			max({ gamma[i][0], gamma[i][2], gamma[i][4], gamma[i][6],gamma[i][8], gamma[i][10], gamma[i][12], gamma[i][14] });


		// bits with respect to the imag part;
		llr[i * symbol_length + 4] = max({ gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7] });
		llr[i * symbol_length + 5] = max({ gamma[i + Nt][4], gamma[i + Nt][5], gamma[i + Nt][6], gamma[i + Nt][7], gamma[i + Nt][12], gamma[i + Nt][13], gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][10], gamma[i + Nt][11] });
		llr[i * symbol_length + 6] = max({ gamma[i + Nt][2], gamma[i + Nt][3], gamma[i + Nt][6], gamma[i + Nt][7],gamma[i + Nt][10], gamma[i + Nt][11],gamma[i + Nt][14], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][1], gamma[i + Nt][4], gamma[i + Nt][5],gamma[i + Nt][8], gamma[i + Nt][9], gamma[i + Nt][12], gamma[i + Nt][13] });
		llr[i * symbol_length + 7] = max({ gamma[i + Nt][1], gamma[i + Nt][3], gamma[i + Nt][5], gamma[i + Nt][7],gamma[i + Nt][9], gamma[i + Nt][11],gamma[i + Nt][13], gamma[i + Nt][15] }) -
			max({ gamma[i + Nt][0], gamma[i + Nt][2], gamma[i + Nt][4], gamma[i + Nt][6],gamma[i + Nt][8], gamma[i + Nt][10], gamma[i + Nt][12], gamma[i + Nt][14] });
	}
#endif


}




void mmse_detection_float(uint32_t Nt, uint32_t Nr, cf_t* H_mtx, cf_t* Rx, cf_t* tmp_inv_mtx, cf_t* tmp_conv_intf_mtx, cf_t* eq_channel_mtx_col,
	cf_t* mmse_filter_mtx, cf_t* det_results, float sigma2)
{
	uint32_t nof_est_ch = 0;						/* number of the estimated channel matrix */
	uint32_t next_ch = 0;								/* index of the channel matrix of user k */
	uint32_t i, j, n;
	uint32_t n_ch = 0;

	cf_t rho;												/* unbiased parameter of MMSE filter */
	float rho_normal;

	cf_t alpha = { 1, 0 };						/* constant number for cblas_cgemm */
	cf_t beta = { 0, 0 };						/* constant number for cblas_cgemm */

	/* step 1: initialize the identity matrix */
	memset(tmp_conv_intf_mtx, 0, sizeof(cf_t) * Nr * Nr);

	for (i = 0; i < Nt * Nt; i += (Nt + 1)) {
		tmp_conv_intf_mtx[i].real = sigma2;
		tmp_conv_intf_mtx[i].imag = 0;
	}

	/* step 2: compute MMSE filter coefficients computation */

	/* recover the interference plus 
	*/
	cblas_ccopy(Nt * Nt, tmp_conv_intf_mtx, 1,
		tmp_inv_mtx, 1);

	/* G_k'*G_k + IFN */
	cblas_cgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, Nt,
		Nt, Nr, &alpha, &H_mtx[n_ch * next_ch], Nt,
		&H_mtx[n_ch * next_ch], Nt, &alpha, tmp_inv_mtx,
		Nt);

	//inverseMatrix(tmp_inv_mtx, Nr);
	inverseMatrix(tmp_inv_mtx, Nt);

	/* ()^-1 * G_k' */
	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, Nt,
		Nr, Nt, &alpha,
		tmp_inv_mtx, Nt, &H_mtx[n_ch * next_ch],
		Nt, &beta, &mmse_filter_mtx[n_ch * next_ch],
		Nr);

	/* unbiased MMSE filter matrix. */
	for (n = 0; n < Nt; n++) {

		/* get the column of G_k */
		for (j = 0; j < Nr; j++) {
			eq_channel_mtx_col[j] = H_mtx
				[n_ch * next_ch + j * Nt + n];
		}

		cblas_cdotu_sub(Nr,
			&mmse_filter_mtx[n_ch * next_ch + n * Nr],
			1, eq_channel_mtx_col, 1, &rho);

		rho_normal = 1 / rho.real;

		cblas_csscal(Nr, rho_normal,
			&mmse_filter_mtx[n_ch * next_ch + n * Nr], 1);
	}

	// step3: detection using the MMSE filter
	/* MMSE filter */
		//tmp_rx_data^T * filter^T = tmp_det_results (rk*cof*36),
		//the first rk is the lowest d,represents the rk datas on one sc in one ofdm.cof is second d which uses same filter.
		//TODO:attention: rk*128 128*(cof*36)
	uint32_t nof_re_filter = 1;

	cblas_cgemm(CblasRowMajor, CblasTrans, CblasTrans, nof_re_filter,
		Nt, Nr, &alpha, Rx,
		nof_re_filter, &mmse_filter_mtx[n_ch * next_ch],
		Nr, &beta, det_results, Nt);



}


void mmse_symtobit_llr(Col<float>& mmse_res, Mat<float>& llr_mtx, int MODETYPE, int Nt)
{
	// input   mmse_res:	mmse detection results(symbol mean)
	// output  llr_mtx:		bit-llr (MODETYPE x Nt)
	int i;
	if (MODETYPE == 8)
	{
		for (i = 0; i < Nt; i++)
		{
			llr_mtx(0, i) = mmse_res(i);
			llr_mtx(1, i) = abs(llr_mtx(0, i)) - 0.6135715;
			llr_mtx(2, i) = abs(llr_mtx(1, i)) - 0.306786;
			llr_mtx(3, i) = abs(llr_mtx(2, i)) - 0.1533895;

			llr_mtx(4, i) = mmse_res(i + Nt);
			llr_mtx(5, i) = abs(llr_mtx(4, i)) - 0.6135715;
			llr_mtx(6, i) = abs(llr_mtx(5, i)) - 0.306786;
			llr_mtx(7, i) = abs(llr_mtx(6, i)) - 0.1533895;
		}
	}
	else if (MODETYPE == 6)
	{
		for (i = 0; i < Nt; i++)
		{
			llr_mtx(0, i) = mmse_res(i);
			llr_mtx(1, i) = abs(llr_mtx(0, i)) - 0.6172134;
			llr_mtx(2, i) = abs(llr_mtx(1, i)) - 0.3086067;

			llr_mtx(3, i) = mmse_res(i + Nt);
			llr_mtx(4, i) = abs(llr_mtx(3, i)) - 0.6172134;
			llr_mtx(5, i) = abs(llr_mtx(4, i)) - 0.3086067;

		}
	}
	else if(MODETYPE == 4)
	{
		for (i = 0; i < Nt; i++)
		{
			llr_mtx(0, i) = mmse_res(i);
			llr_mtx(1, i) = abs(llr_mtx(0, i)) - 0.632455;

			llr_mtx(2, i) = mmse_res(i + Nt);
			llr_mtx(3, i) = abs(llr_mtx(2, i)) - 0.632455;

		}
	}
	else
	{
		for (i = 0; i < Nt; i++)
		{
			llr_mtx(0, i) = mmse_res(i);
			llr_mtx(1, i) = mmse_res(i + Nt);

		}
	}
}

void mmse_detection_float(size_t Nt2, size_t Nr2, float* H_mtx, float* Rx, float* tmp_inv_mtx, float* tmp_conv_intf_mtx, float* eq_channel_mtx_col,
	float* mmse_filter_mtx, float* det_results, float sigma2)
{
	uint32_t nof_est_ch = 0;						/* number of the estimated channel matrix */
	uint32_t next_ch = 0;								/* index of the channel matrix of user k */
	uint32_t i, j, n;
	uint32_t n_ch = 0;

	float rho;												/* unbiased parameter of MMSE filter */
	float rho_normal;

	float alpha = 1;						/* constant number for cblas_cgemm */
	float beta = 0;						/* constant number for cblas_cgemm */

	/* step 1: initialize the identity matrix */
	memset(tmp_conv_intf_mtx, 0, sizeof(float) * Nr2 * Nr2);

	for (i = 0; i < Nt2 * Nt2; i += (Nt2 + 1)) {
		tmp_conv_intf_mtx[i] = sigma2;
	}

	/* step 2: compute MMSE filter coefficients computation */

	/* recover the interference plus noise */
	cblas_scopy(Nt2 * Nt2, tmp_conv_intf_mtx, 1,
		tmp_inv_mtx, 1);

	/* G_k'*G_k + IFN */
	cblas_sgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, Nt2,
		Nt2, Nr2, alpha, &H_mtx[n_ch * next_ch], Nt2,
		&H_mtx[n_ch * next_ch], Nt2, alpha, tmp_inv_mtx,
		Nt2);

	inverseMatrix(tmp_inv_mtx, Nt2);

	/* ()^-1 * G_k' */
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, Nt2,
		Nr2, Nt2, alpha,
		tmp_inv_mtx, Nt2, &H_mtx[n_ch * next_ch],
		Nt2, beta, &mmse_filter_mtx[n_ch * next_ch],
		Nr2);

	///* unbiased MMSE filter matrix. */
	//for (n = 0; n < Nt; n++) {

	//	/* get the column of G_k */
	//	for (j = 0; j < Nr; j++) {
	//		eq_channel_mtx_col[j] = H_mtx
	//			[n_ch * next_ch + j * Nt + n];
	//	}

	//	cblas_cdotu_sub(Nr,
	//		&mmse_filter_mtx[n_ch * next_ch + n * Nr],
	//		1, eq_channel_mtx_col, 1, &rho);

	//	rho_normal = 1 / rho;

	//	cblas_csscal(Nr, rho_normal,
	//		&mmse_filter_mtx[n_ch * next_ch + n * Nr], 1);
	//}

	// step3: detection using the MMSE filter
	/* MMSE filter */
		//tmp_rx_data^T * filter^T = tmp_det_results (rk*cof*36),
		//the first rk is the lowest d,represents the rk datas on one sc in one ofdm.cof is second d which uses same filter.
		//TODO:attention: rk*128 128*(cof*36)
	uint32_t nof_re_filter = 1;

	cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, nof_re_filter,
		Nt2, Nr2, alpha, Rx,
		nof_re_filter, &mmse_filter_mtx[n_ch * next_ch],
		Nr2, beta, det_results, Nt2);



}

