#pragma once
#include <vector>
#include <fstream>
#include <ctime>
#include <functional>
#include <algorithm>
#include <queue>
#include <iostream>
#include <string.h>
using namespace std;

//float y_array[3744];
float temp_selfcorrect[10000];
bool quantization = false;
void quantization_llr(float* LLR, int N, bool first_quan)
{
	//LLR 为输入的未量化消息；
	//LLR_quantization为输出的已量化消息；
	//int_bit为量化策略中的整数位数；
	//frac_bit_bit为量化策略中的小数位数；
	//N为码长
	int int_bit = 3;
	int frac_bit = 2;
	int num = int_bit + frac_bit;
	int mul_factor = (1 << frac_bit);
	float upp = (float)(1 << num) - 1;
	float lpp = (float)(-(1 << num));
	float low_q = -(1 << int_bit);
	float upp_num = upp / mul_factor;
	if (first_quan)
	{
		for (int i = 0; i < N; i++)
		{
			if (LLR[i] >= 0)
				if ((round(LLR[i] * mul_factor)) <= upp)
					LLR[i] = (round(LLR[i] * mul_factor)) / mul_factor;
				else
					LLR[i] = upp / mul_factor;
			else
				if ((round(LLR[i] * mul_factor)) >= lpp)
					LLR[i] = (round(LLR[i] * mul_factor)) / mul_factor;
				else
					LLR[i] = low_q;
		}
	}
	else
	{
		for (int i = 0; i < N; i++)
		{
			if (LLR[i] > upp_num)
				LLR[i] = upp_num;
			else if (LLR[i] < low_q)
				LLR[i] = low_q;
			else {}
		}
	}
}
float quantization_llr(float LLR)
{
	//LLR 为输入的未量化消息；
	//LLR_quantization为输出的已量化消息；
	//int_bit为量化策略中的整数位数；
	//frac_bit_bit为量化策略中的小数位数；
	//N为码长
	int int_bit = 5;
	int frac_bit = 2;
	int num = int_bit + frac_bit;
	int mul_factor = (1 << frac_bit);
	float upp = (float)(1 << num) - 1;
	float lpp = (float)(-(1 << num));
	float low_q = -(1 << int_bit);
	float upp_num = upp / mul_factor;

	if (LLR > upp_num)
	{
	
		//cout << "LLR!" << LLR << endl;
		return upp_num;
	}
		//return upp_num;
	else if (LLR < low_q)
	{
		
		//cout << "LLR!" << LLR << endl;
		return low_q;
	}
	else return LLR;
}
void quantization_v2c(vector<float>& LLR, int N, bool first_quan)
{
	//LLR 为输入的未量化消息；
	//LLR_quantization为输出的已量化消息；
	//int_bit为量化策略中的整数位数；
	//frac_bit_bit为量化策略中的小数位数；
	//N为码长
	int int_bit = 5;
	int frac_bit = 2;
	int num = int_bit + frac_bit;
	int mul_factor = (1 << frac_bit);
	float upp = (float)(1 << num) - 1;
	float lpp = (float)(-(1 << num));
	float low_q = -(1 << int_bit);
	float upp_num = upp / mul_factor;
	if (first_quan)
	{
		for (int i = 0; i < N; i++)
		{
			if (LLR[i] >= 0)
				if ((round(LLR[i] * mul_factor)) <= upp)
					LLR[i] = (round(LLR[i] * mul_factor)) / mul_factor;
				else
					LLR[i] = upp / mul_factor;
			else
				if ((round(LLR[i] * mul_factor)) >= lpp)
					LLR[i] = (round(LLR[i] * mul_factor)) / mul_factor;
				else
					LLR[i] = low_q;
		}
	}
	else
	{
		for (int i = 0; i < N; i++)
		{
			if (LLR[i] > upp_num)
				LLR[i] = upp_num;
			else if (LLR[i] < low_q)
				LLR[i] = low_q;
			else {}
		}
	}
}
float quantization_v2c(float LLR)
{//4
	//LLR 为输入的未量化消息；
	//LLR_quantization为输出的已量化消息；
	//int_bit为量化策略中的整数位数；
	//frac_bit_bit为量化策略中的小数位数；
	//N为码长
	int int_bit = 4; // 0100
	int frac_bit = 2; // 0010
	int num = int_bit + frac_bit; //0110
	int mul_factor = (1 << frac_bit); // 0100
	float upp = (float)(1 << num) - 1; //1011 (11
	float lpp = (float)(-(1 << num)); // -12
	float low_q = -(1 << int_bit);  // -8
	float upp_num = upp / mul_factor; // 11/4 = 3.25
	// 将LLR限制在upp_num 和 low_q之间
	if (LLR > upp_num)
	{
		
		//cout << "v2c!" << LLR << endl;
		return upp_num;
	}
	//return upp_num;
	else if (LLR < low_q)
	{
		
		//cout << "v2c!" << LLR << endl;
		return low_q;
	}
	else return LLR;

}
float quantization_c2v(float LLR)
{
	//LLR 为输入的未量化消息；
	//LLR_quantization为输出的已量化消息；
	//int_bit为量化策略中的整数位数；
	//frac_bit_bit为量化策略中的小数位数；
	//N为码长
	int int_bit = 3;
	int frac_bit = 2;
	int num = int_bit + frac_bit;
	int mul_factor = (1 << frac_bit); // 4
	float upp = (float)(1 << num) - 1; // 9
	float lpp = (float)(-(1 << num)); // -10
	float low_q = -(1 << int_bit); // -6
	float upp_num = upp / mul_factor; // 9/4=2.25

	if (LLR > upp_num)
	{
		//cout << "c2v!" << LLR << endl;
		return upp_num;
	}
	//return upp_num;
	else if (LLR < low_q)
	{
		//cout << "c2v!" << LLR << endl;
		return low_q;
	}
	else return LLR;

}

float abs_f(float a)
{
	return a < 0 ? -a : a;
}
double mysign(double sig) 
{
	double s;
	if (sig > 0) {
		s = 1;
	}
	else if (sig < 0) {
		s = -1;
	}
	else {
		s = 0;
	}
	return s;
}
float max_f(float a, float b)
{
	return a > b ? a : b;
}
float min_f(float a, float b)
{
	return a < b ? a : b;
}
float lambda_adjusted(float a)
{
	if (abs_f(a) < 1.0)
	{
		return -0.375 * abs_f(a) + 0.6825;
	}
	else if (abs_f(a) < 2.625)
	{
		return -0.1875 * abs_f(a) + 0.5;
	}
	else return 0;
}
float lambda_adjusted_quan(float a)
{
	/*if (abs_f(a) < 1.0)
	{
		return 0.75;

	}
	else if (abs_f(a) < 2.5)
	{
		return 0.5;

	}//A86*/
	//for A784*/
	/*if (abs_f(a) < 0.5)
	{
		return 0.75;

	}
	else if (abs_f(a) < 1.0)
	{
		return 0.5;

	}
	else if (abs_f(a) < 2.5)
	{
		return 0.25;

	}//*/
	if (abs_f(a) < 0.25)
	{
		return 0.75;

	}
	else if (abs_f(a) < 1)
	{
		return 0.5;

	}
	else if (abs_f(a) < 2.25)
	{
		return 0.25;

	}//for A784*/
	else return 0;
}
float adjusted_function(float a, float b)
{
	if (quantization == true)
	{
		return abs_f(min_f(abs_f(a), abs_f(b)) + lambda_adjusted_quan(abs_f(a) + abs_f(b)) - lambda_adjusted_quan(abs_f(abs_f(a) - abs_f(b))));
	}
	return abs_f(min_f(abs_f(a), abs_f(b)) + lambda_adjusted(abs_f(a) + abs_f(b)) - lambda_adjusted(abs_f(abs_f(a) - abs_f(b))));
}
bool abssort(float i, float j) { return (abs_f(i) < abs_f(j)); }

void decodeLogDomainMinSum_converge(float* rx, float* LLR_out, float *y_array, int iter, int M, int N, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, int **H)
{   // rx -input 
	// LLR
	float sig = 1;
	float tanhtemp = 0;
	float x1 = 0;
	float temp=0;

	// initial the message from CN to VN with zero
	for (int k = 0; k < M; k++)
	{
		for (int i = 0; i < CNtoVN[k].size(); i++)
		{
			LLR_CNtoVN[k][i] = 0.0;
		}
	}

	for (int i = 0; i < N; i++) 
	{
		//rx[i] = -2 * rx[i] / N0;
		LLR_VN[i] = rx[i];  //为啥直接用rx？ using detection’s result as prior pro

	}
	for (int it = 0; it < iter; ++it)
	{
		//******************CN_to_VN**********************************
		for (int k = 0; k < M; k++)
		{
			for (int i = 0; i < CNtoVN[k].size(); i++)  // 第i列个non-zero index
			{
				sig = 1;
				tanhtemp = 100000;
				for (int j = 0; j < CNtoVN[k].size(); j++)
				{
					if (j != i)
					{
						x1 = LLR_VN[CNtoVN[k][j]] - LLR_CNtoVN[k][j]; //  LLR_VN 包含所有check-edge，所以要减去
						//if (quantization_flag == 1)
						//{
						//	x1 = quantization_v2c(x1);
						//}

						sig = sig * mysign(x1);  // 记录x1的符号（>0 =1; <0 =-1）
						
						
						tanhtemp = min_f(abs(tanhtemp), abs_f(x1)); // keep the minimum x1
					}
				}

				// extrinct message sum for check node[k]'s output [i] edge
				y_array[i] = sig * max_f(tanhtemp - 0.25, 0);//Offset_MS
				//y_array[i] = 0.85 * sig * tanhtemp;//NMS
			}

			// LLR_CNtoVN reserve extrinct message of check node[k]'s output[i] edge
			// accomplish the check node update
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				LLR_CNtoVN[k][i] = y_array[i];
				//if (quantization_flag == 1)
				//{
				//	LLR_CNtoVN[k][i] = quantization_c2v(LLR_CNtoVN[k][i]);
				//}//*/
			}
		}

		//******************VN_to_CN**********************************
		memset(y_array, 0, sizeof(float)*N); //initial the buffer with value zero

		// every variable node's input edge message add together
		for (int k = 0; k < M; k++)
		{
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				y_array[CNtoVN[k][i]] += LLR_CNtoVN[k][i];
				//if (quantization_flag == 1)
				//{
				//	y_array[CNtoVN[k][i]] = quantization_llr(y_array[CNtoVN[k][i]]);
				//}//*/
			}
		}

		// output message calculation
		for (int i = 0; i < N; i++)
		{
			LLR_VN[i] = y_array[i] + rx[i];
			//if (quantization_flag == 1)
			//{
			//	LLR_VN[i] = quantization_llr(LLR_VN[i]);
			//}//*/

		}
		int summ = 0;
		for (int k = 0; k < M; k++)
		{
			summ = 0;
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{   // 小于零，译码为1
				summ ^= (LLR_VN[CNtoVN[k][i]] < 0); // calculate Hc^T 
			}
			if (summ) // 如果某一位校验位不为零，继续迭代
				break;
		}
		if (!summ) //如果校验结果为0，则结束迭代，输出码字
			break;
	}

	for (int i = 0; i < N; i++)
	{
		LLR_out[i] = LLR_VN[i];
	}

}




void decodeLogDomainMinSum_rowlayered(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, vector<std::vector<float>>& Lqij_temp, vector<std::vector<float>>& Lqij, int** H, int quantization_flag)
{
	int success_flag = 0;
	int flag;
	float sig = 1;
	float tanhtemp = 0;
	float x1 = 0;

	for (int k = 0; k < M; k++)
	{
		for (int i = 0; i < CNtoVN[k].size(); i++)
		{
			LLR_CNtoVN[k][i] = 0.0;
			Lqij_temp[k][i] = 0.0;
		}
	}

	for (int i = 0; i < N; i++)
	{
		//rx[i] = -2 * rx[i] / N0;
		//cout << rx[i] << " ";
		LLR_VN[i] = rx[i];

	}
	//cout << endl;
	for (int it = 0; it < iter; ++it)
	{
		for (int k = 0; k < M; k++)
		{
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				Lqij[k][i] = LLR_VN[CNtoVN[k][i]] - LLR_CNtoVN[k][i];
				if (quantization_flag == 1)
				{
					Lqij[k][i] = quantization_v2c(Lqij[k][i]);
				}
			}

			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				sig = 1;
				tanhtemp = 1000000;
				for (int j = 0; j < CNtoVN[k].size(); j++)
				{
					if (j != i)
					{
						x1 = Lqij[k][j];
						sig = sig * mysign(x1);
						//tanhtemp = abs_f(min_f(abs_f(tanhtemp), abs_f(x1)) + lambda_adjusted(abs_f(tanhtemp) + abs_f(x1)) - lambda_adjusted(abs_f(abs_f(tanhtemp) - abs_f(x1))));
						tanhtemp = min_f(abs(tanhtemp), abs_f(x1));
						//tanhtemp = min_f(abs(tanhtemp), abs_f(x1));
					}
				}
				//LLR_CNtoVN[k][i] = sig * tanhtemp;//min-sum
				LLR_CNtoVN[k][i] = sig* max_f(tanhtemp - 0.25, 0);//OMS
				//cout << LLR_CNtoVN[k][i] << ":";
				if (quantization_flag == 1)
				{
					LLR_CNtoVN[k][i] = quantization_c2v(LLR_CNtoVN[k][i]);
					//cout << LLR_CNtoVN[k][i] << endl;
				}//*/
				LLR_VN[CNtoVN[k][i]] = Lqij[k][i] + LLR_CNtoVN[k][i];
				//cout << LLR_VN[CNtoVN[k][i]] << ":";
				if (quantization_flag == 1)
				{
					LLR_VN[CNtoVN[k][i]] = quantization_llr(LLR_VN[CNtoVN[k][i]]);
					//cout << LLR_VN[CNtoVN[k][i]] << endl;
				}//*/
			}
			

		}
		
	}
	for (int i = 0; i < N; i++)
	{
		LLR_out[i] = LLR_VN[i];
		//cout << LLR_out[i] << " ";
	}

}
void decodeLogDomain_adjMinSum_converge(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<std::vector<float>>& Lqij, vector<std::vector<float>>& Lqij_temp, vector<float>& LLR_VN, int** H, int quantization_flag)
{
	float sig = 1;
	float tanhtemp = 0;
	float x1 = 0;
	float temp = 0;
	float Lqij_min = HUGE_VALF;
	int index_min = 0;

	for (int k = 0; k < M; k++)
	{
		for (int i = 0; i < CNtoVN[k].size(); i++)
		{
			LLR_CNtoVN[k][i] = 0.0;
		}
	}
	for (int i = 0; i < N; i++)
	{
		//rx[i] = -2 * rx[i] / N0;
		LLR_VN[i] = rx[i];

	}
	for (int it = 0; it < iter; ++it)
	{
		//******************VN_to_CN**********************************
		for (int k = 0; k < M; k++)
		{
			Lqij_min = HUGE_VALF;
		    index_min = 0;
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				Lqij[k][i] = LLR_VN[CNtoVN[k][i]] - LLR_CNtoVN[k][i];
				if (quantization_flag == 1)
				{
					quantization_v2c(Lqij[k][i]);
				}
				if (abs_f(Lqij[k][i]) < Lqij_min)
				{
					Lqij_min = abs_f(Lqij[k][i]);
					index_min = i;
				}
			}
			sig = 1;
			tanhtemp = HUGE_VALF;
			for (int j = 0; j < CNtoVN[k].size(); j++)
			{

				if (j != index_min)
				{
					x1 = Lqij[k][j];
					sig = sig * mysign(x1);
					tanhtemp = adjusted_function(tanhtemp, x1);//abs_f(min_f(abs_f(tanhtemp), abs_f(x1)) + lambda_adjusted(abs_f(tanhtemp) + abs_f(x1)) - lambda_adjusted(abs_f(abs_f(tanhtemp) - abs_f(x1))));
					//tanhtemp = min_f(abs(tanhtemp), abs_f(x1));
					if (quantization_flag == 1)
					{
						quantization_c2v(tanhtemp);
					}//*/
				}
			}
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				if (i == index_min)
				{
					LLR_CNtoVN[k][i] = sig * tanhtemp;
				}
				else
				{
					LLR_CNtoVN[k][i] = sig * mysign(Lqij[k][index_min]) * mysign(Lqij[k][i]) * adjusted_function(tanhtemp, Lqij[k][index_min]);
				}
				if (quantization_flag == 1)
				{
					quantization_c2v(LLR_CNtoVN[k][i]);
				}//*/
			}

		}
		//******************CN_to_VN**********************************
		memset(y_array, 0, sizeof(float) * N);
		for (int k = 0; k < M; k++)
		{
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				y_array[CNtoVN[k][i]] += LLR_CNtoVN[k][i];
				if (quantization_flag == 1)
				{
					quantization_llr(y_array[CNtoVN[k][i]]);
				}//*/
			}
		}

		for (int i = 0; i < N; i++)
		{
			LLR_VN[i] = y_array[i] + rx[i];
			if (quantization_flag == 1)
			{
				quantization_llr(LLR_VN[i]);
			}//*/

		}
	}
	for (int i = 0; i < N; i++)
	{
		LLR_out[i] = LLR_VN[i];
	}

}
void decodeLogDomain_adjMinSum_rowlayered(float* rx, float* LLR_out, float *y_array, int iter, int M, int N, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, vector<std::vector<float>>& Lqij, int **H, int quantization_flag)
{
	int early_terminate_flag;
	float sig = 1;
	float tanhtemp = 0;
	float x1 = 0;
	float Lqij_min = HUGE_VALF;
	int index_min = 0;

	for (int k = 0; k < M; k++)
	{
		for (int i = 0; i < CNtoVN[k].size(); i++)
		{
			LLR_CNtoVN[k][i] = 0.0;
		}
	}

	for (int i = 0; i < N; i++)
	{
		//rx[i] = -2 * rx[i] / N0;
		//cout << rx[i] << " ";
		LLR_VN[i] = rx[i];

	}
	//cout << endl;
	for (int it = 0; it < iter; ++it)
	{
		for (int k = 0; k < M; k++)
		{
			Lqij_min = HUGE_VALF;
			index_min = 0;
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				Lqij[k][i] = LLR_VN[CNtoVN[k][i]] - LLR_CNtoVN[k][i];
				if (quantization_flag == 1)
				{
					quantization_v2c(Lqij[k][i]);
				}
				if (abs_f(Lqij[k][i]) < Lqij_min)
				{
					Lqij_min = abs_f(Lqij[k][i]);
					index_min = i;
				}
			}
			sig = 1;
			tanhtemp = HUGE_VALF;
			for (int j = 0; j < CNtoVN[k].size(); j++)
			{

				if (j != index_min)
				{
					x1 = Lqij[k][j];
					sig = sig * mysign(x1);
					tanhtemp = adjusted_function(tanhtemp, x1);//abs_f(min_f(abs_f(tanhtemp), abs_f(x1)) + lambda_adjusted(abs_f(tanhtemp) + abs_f(x1)) - lambda_adjusted(abs_f(abs_f(tanhtemp) - abs_f(x1))));
					//tanhtemp = min_f(abs(tanhtemp), abs_f(x1));
					if (quantization_flag == 1)
					{
						quantization_c2v(tanhtemp);
					}
				}
			}
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				if (i == index_min)
				{
					LLR_CNtoVN[k][i] = sig * tanhtemp;
				}
				else
				{
					LLR_CNtoVN[k][i] = sig * mysign(Lqij[k][index_min]) * mysign(Lqij[k][i]) * adjusted_function(tanhtemp, Lqij[k][index_min]);
				}
				if (quantization_flag == 1)
				{
					quantization_c2v(LLR_CNtoVN[k][i]);
				}
				
				LLR_VN[CNtoVN[k][i]] = Lqij[k][i] + LLR_CNtoVN[k][i];
				if (quantization_flag == 1)
				{
					quantization_llr(LLR_VN[CNtoVN[k][i]]);
				}
			}
			
		}
		early_terminate_flag = 1;
		if (it > 8 && it % 2 == 0)
		{
			int tmp;
			for (int k = 0;k < M;k++)
			{
				tmp = 0;
				for (int j = 0; j < N; j++)
				{
					tmp ^= (LLR_VN[j] >= 0 ? 0 : 1) & H[k][j];
						//cout << uhat[i] << ",";

				}
				if (tmp != 0)
				{
					early_terminate_flag = 0;
					break;
				}
			}
			if (early_terminate_flag == 1)
			{
				break;
			}
		}

	}
	for (int i = 0; i < N; i++)
	{
		LLR_out[i] = LLR_VN[i];
	}

}
void decodeLogDomainMinSum_adaptive_rowlayered(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, int Mb, int Zc, float N0, int adjusted_flag, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, vector<std::vector<float>>& Lqij, vector<std::vector<float>>& Lqij_temp, int** H, int quantization_flag)
{
	float sig = 1;
	float tanhtemp = 0;
	float x1 = 0;
	float temp = 0;
	
	for (int k = 0; k < M; k++)
	{
		for (int i = 0; i < CNtoVN[k].size(); i++)
		{
			LLR_CNtoVN[k][i] = 0.0;
		}
	}
	for (int i = 0; i < N; i++)
	{
		//rx[i] = -2 * rx[i] / N0;
		LLR_VN[i] = rx[i];

	}
	for (int it = 0; it < iter; ++it)
	{
		for (int L = 0;L < Mb;L++)
		{
			//******************VN_to_CN**********************************
			for (int k = L * Zc; k < L * Zc + Zc; k++)
			{
				for (int i = 0; i < CNtoVN[k].size(); i++)
				{
					Lqij[k][i] = LLR_VN[CNtoVN[k][i]] - LLR_CNtoVN[k][i];
					/**************self-corrected*********/
					/*if ((mysign(Lqij[k][i]) != mysign(Lqij_temp[k][i])) && (Lqij_temp[k][i] != 0))
					{
						Lqij[k][i] = 0;
					}
					Lqij_temp[k][i] = Lqij[k][i];//*/

				}
			}
			for (int k = L * Zc; k < L * Zc + Zc; k++)
			{
				for (int i = 0; i < CNtoVN[k].size(); i++)
				{
					sig = 1;
					tanhtemp = HUGE_VALF;
					for (int j = 0; j < CNtoVN[k].size(); j++)
					{
						if (j != i)
						{
							x1 = Lqij[k][j];
							sig = sig * mysign(x1);
							if (adjusted_flag == 1)
							{
								tanhtemp = min_f(abs_f(tanhtemp), abs_f(x1)) + lambda_adjusted(abs_f(tanhtemp) + abs_f(x1)) - lambda_adjusted(abs_f(tanhtemp) - abs_f(x1));
							}
							else tanhtemp = min_f(abs(tanhtemp), abs_f(x1));
						}
					}
					if (adjusted_flag == 1)
					{
						y_array[i] = 0.85 * sig * tanhtemp;
					}
					else
					{
						if (L < 4)
						{
							y_array[i] = 0.85 * sig * max_f(tanhtemp - 0.25, 0);
						}
						//y_array[i] = 0.85 * sig * max_f(tanhtemp - 0.25, 0);//OMS
						else y_array[i] = 0.85 * sig * tanhtemp;
					}
				}
				for (int i = 0; i < CNtoVN[k].size(); i++)
				{
					LLR_CNtoVN[k][i] = y_array[i];
				}
			}
			
			//******************CN_to_VN**********************************
			for (int k = L * Zc; k < L * Zc + Zc; k++)
			{
				for (int i = 0; i < CNtoVN[k].size(); i++)
				{
					LLR_VN[CNtoVN[k][i]] = Lqij[k][i] + LLR_CNtoVN[k][i];
				}
			}
		}
		
	}
	for (int i = 0; i < N; i++)
	{
		LLR_out[i] = LLR_VN[i];
	}

}
void decodeLogDomainSumProduct(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, int** H)
{
	float sig = 1;
	float tanhtemp = 0;
	float x1 = 0;

	for (int k = 0; k < M; k++)
	{
		for (int i = 0; i < CNtoVN[k].size(); i++)
		{
			LLR_CNtoVN[k][i] = 0.0;
		}
	}
	for (int i = 0; i < N; i++)
	{
		//rx[i] = -2 * rx[i] / N0;
		LLR_VN[i] = rx[i];

	}
	for (int it = 0; it < iter; ++it)
	{
		//******************VN_to_CN**********************************
		for (int k = 0; k < M; k++)
		{
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				sig = 1;
				tanhtemp = 1;
				for (int j = 0; j < CNtoVN[k].size(); j++)
				{
					if (j != i)
					{
						x1 = LLR_VN[CNtoVN[k][j]] - LLR_CNtoVN[k][j];
						tanhtemp = tanhtemp * (tanh(x1 / 2));//sum-product 
					}
				}
				y_array[i] = 2 * atanh(tanhtemp);
				if (abs(y_array[i]) > 1000)
					y_array[i] = mysign(y_array[i]) * 1000;
				if (std::isnan(abs(y_array[i]))) y_array[i] = mysign(y_array[i]) * 0.0;
				if ((abs(y_array[i])) < 0.00001) y_array[i] = mysign(y_array[i]) * 0.0;
			}
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				LLR_CNtoVN[k][i] = y_array[i];
			}
		}
		//******************CN_to_VN**********************************
		memset(y_array, 0, sizeof(float) * N);
		for (int k = 0; k < M; k++)
		{
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				y_array[CNtoVN[k][i]] += LLR_CNtoVN[k][i];

			}
		}

		for (int i = 0; i < N; i++)
		{
			LLR_VN[i] = y_array[i] + rx[i];
		}
		int summ = 0;
		for (int k = 0; k < M; k++)
		{
			summ = 0;
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				summ ^= (LLR_VN[CNtoVN[k][i]] < 0);
			}
			if (summ)
				break;
		}
		if (!summ)
			break;
	}
	for (int i = 0; i < N; i++)
	{
		LLR_out[i] = LLR_VN[i];
	}

}
void decodeLogDomainSumProduct_rowlayered(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, vector<std::vector<float>>& Lqij, int** H)
{
	int success_flag = 0;
	int flag;
	float sig = 1;
	float tanhtemp = 0;
	float x1 = 0;
	int early_terminate_flag;

	for (int k = 0; k < M; k++)
	{
		for (int i = 0; i < CNtoVN[k].size(); i++)
		{
			LLR_CNtoVN[k][i] = 0.0;
		}
	}

	for (int i = 0; i < N; i++)
	{
		//rx[i] = -2 * rx[i] / N0;
		//cout << rx[i] << " ";
		LLR_VN[i] = rx[i];

	}
	//cout << endl;
	for (int it = 0; it < iter; ++it)
	{
		for (int k = 0; k < M; k++)
		{
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				Lqij[k][i] = LLR_VN[CNtoVN[k][i]] - LLR_CNtoVN[k][i];
			}
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				sig = 1;
				tanhtemp = 1;
				for (int j = 0; j < CNtoVN[k].size(); j++)
				{
					if (j != i)
					{
						x1 = Lqij[k][j];
						tanhtemp = tanhtemp * (tanh(x1 / 2));//sum-product 
						//tanhtemp = min_f(abs(tanhtemp), abs_f(x1)) + lambda_adjusted(abs(tanhtemp) + abs_f(x1)) - lambda_adjusted(abs(tanhtemp) - abs_f(x1));
						//tanhtemp = min_f(abs(tanhtemp), abs_f(x1));
					}
				}
				y_array[i] = 2 * atanh(tanhtemp);
				if (abs(y_array[i]) > 100)
					y_array[i] = mysign(y_array[i]) * 100;
				if (std::isnan(abs(y_array[i]))) y_array[i] = mysign(y_array[i]) * 0.0;
				if ((abs(y_array[i])) < 0.00001) y_array[i] = mysign(y_array[i]) * 0.0;
				LLR_CNtoVN[k][i] = y_array[i];
				LLR_VN[CNtoVN[k][i]] = Lqij[k][i] + LLR_CNtoVN[k][i];
			}
			

		}
		early_terminate_flag = 1;
		if (it > 8 && it % 2 == 0)
		{
			int tmp;
			for (int k = 0;k < M;k++)
			{
				tmp = 0;
				for (int j = 0; j < N; j++)
				{
					tmp ^= (LLR_VN[j] >= 0 ? 0 : 1) & H[k][j];
					//cout << uhat[i] << ",";

				}
				if (tmp != 0)
				{
					early_terminate_flag = 0;
					break;
				}
			}
			if (early_terminate_flag == 1)
			{
				break;
			}
		}

	}
	for (int i = 0; i < N; i++)
	{
		LLR_out[i] = LLR_VN[i];
	}

}
void decodeLogDomain_ga_MinSum_rowlayered(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, int Zc, float N0, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, vector<std::vector<float>>& Lqij, vector<float>& Lqij_sort, vector<float>& Lqij_index, int** H, int quantization_flag)
{
	int success_flag = 0;
	int early_terminate_flag;
	float sig = 1;
	float tanhtemp = 0;
	float x1 = 0;
	int s = 3;
	int s1 = s + 1;

	for (int k = 0; k < M; k++)
	{
		for (int i = 0; i < CNtoVN[k].size(); i++)
		{
			LLR_CNtoVN[k][i] = 0.0;
		}
	}

	for (int i = 0; i < N; i++)
	{
		//rx[i] = -2 * rx[i] / N0;
		//cout << rx[i] << " ";
		LLR_VN[i] = rx[i];

	}
	//cout << endl;
	for (int it = 0; it < iter; ++it)
	{
		for (int k = 0; k < M; k++)
		{
			//cout << CNtoVN[k].size() << " ";
			//if (k < 4 * Zc)
			if (k < M)
			{
				for (int i = 0; i < CNtoVN[k].size(); i++)
				{
					Lqij[k][i] = LLR_VN[CNtoVN[k][i]] - LLR_CNtoVN[k][i];
					if (quantization_flag == true)
					{
						quantization_v2c(Lqij[k][i]);
					}
					Lqij_sort[i] = Lqij[k][i];
					Lqij_index[i] = i;
				}
				std::sort(Lqij_index.begin(), Lqij_index.begin() + CNtoVN[k].size(), [&](int i1, int i2) { return abs_f(Lqij_sort[i1]) < abs_f(Lqij_sort[i2]); });
				for (int i = 0; i < CNtoVN[k].size(); i++)
				{
					if (i != Lqij_index[0])
					{
						sig = 1;
						tanhtemp = HUGE_VALF;
						for (int j = 0;j < s;j++)
						{
							x1 = Lqij[k][Lqij_index[j]];
							//tanhtemp = abs_f(min_f(abs_f(tanhtemp), abs_f(x1)) + lambda_adjusted(abs_f(tanhtemp) + abs_f(x1)) - lambda_adjusted(abs_f(abs_f(tanhtemp) - abs_f(x1))));
							tanhtemp = adjusted_function(tanhtemp, x1);
							if (quantization_flag == 1)
							{
								quantization_c2v(tanhtemp);
							}
						}
					}
					else
					{
						sig = 1;
						tanhtemp = HUGE_VALF;
						for (int j = 1;j < s1;j++)
						{
							x1 = Lqij[k][Lqij_index[j]];
							//tanhtemp = abs_f(min_f(abs_f(tanhtemp), abs_f(x1)) + lambda_adjusted(abs_f(tanhtemp) + abs_f(x1)) - lambda_adjusted(abs_f(abs_f(tanhtemp) - abs_f(x1))));
							tanhtemp = adjusted_function(tanhtemp, x1);
							if (quantization_flag == 1)
							{
								quantization_c2v(tanhtemp);
							}
						}
					}
					for (int j = 0; j < CNtoVN[k].size(); j++)
					{
						if (j != i)
						{
							x1 = Lqij[k][j];
							sig = sig * mysign(x1);
						}
					}
					//LLR_CNtoVN[k][i] = sig * max_f(tanhtemp - 0.25, 0);//sig * tanhtemp;//min-sum
					LLR_CNtoVN[k][i] = sig * tanhtemp;
					if (quantization_flag == 1)
					{
						quantization_c2v(LLR_CNtoVN[k][i]);
					}
					LLR_VN[CNtoVN[k][i]] = Lqij[k][i] + LLR_CNtoVN[k][i];
					if (quantization_flag == 1)
					{
						quantization_llr(LLR_VN[CNtoVN[k][i]]);
					}
				}
			}
			else
			{
				for (int i = 0; i < CNtoVN[k].size(); i++)
				{
					Lqij[k][i] = LLR_VN[CNtoVN[k][i]] - LLR_CNtoVN[k][i];
					quantization_v2c(Lqij[k][i]);
				}
				for (int i = 0; i < CNtoVN[k].size(); i++)
				{
					sig = 1;
					tanhtemp = HUGE_VALF;
					for (int j = 0; j < CNtoVN[k].size(); j++)
					{
						if (j != i)
						{
							x1 = Lqij[k][j];
							sig = sig * mysign(x1);
							tanhtemp = min_f(abs(tanhtemp), abs_f(x1));
							//tanhtemp = min_f(abs(tanhtemp), abs_f(x1));
						}
					}
					LLR_CNtoVN[k][i] = sig * max_f(tanhtemp - 0.25, 0);//sig * tanhtemp;//min-sum
					if (quantization_flag == 1)
					{
						quantization_c2v(LLR_CNtoVN[k][i]);
					}
			        //LLR_CNtoVN[k][i] = 0.85 * sig * tanhtemp;
					LLR_VN[CNtoVN[k][i]] = Lqij[k][i] + LLR_CNtoVN[k][i];
					if (quantization_flag == 1)
					{
						quantization_llr(LLR_VN[CNtoVN[k][i]]);
					}
				}
			}

		}
		early_terminate_flag = 1;
		if (it > 8 && it % 2 == 0)
		{
			int tmp;
			for (int k = 0;k < M;k++)
			{
				tmp = 0;
				for (int j = 0; j < N; j++)
				{
					tmp ^= (LLR_VN[j] >= 0 ? 0 : 1) & H[k][j];
					//cout << uhat[i] << ",";

				}
				if (tmp != 0)
				{
					early_terminate_flag = 0;
					break;
				}
			}
			if (early_terminate_flag == 1)
			{
				//cout << it << " ";
				break;
			}
		}

	}
	for (int i = 0; i < N; i++)
	{
		LLR_out[i] = LLR_VN[i];
	}

}
void decodeLogDomain_RC_modified_MinSum_rowlayered(float* rx, float* LLR_out, float* y_array, int iter, int M, int N, float N0, int adjusted_flag, vector<std::vector<int>>& CNtoVN, vector<std::vector<float>>& LLR_CNtoVN, vector<float>& LLR_VN, vector<std::vector<float>>& Lqij, vector<float>& Lqij_sort, vector<float>& Lqij_index, int** H)
{
	int early_terminate_flag;
	float sig = 1;
	float tanhtemp = 0;
	float x1 = 0;
	float Lqij_min = HUGE_VALF;
	int index_min = 0;

	for (int k = 0; k < M; k++)
	{
		for (int i = 0; i < CNtoVN[k].size(); i++)
		{
			LLR_CNtoVN[k][i] = 0.0;
		}
	}

	for (int i = 0; i < N; i++)
	{
		//rx[i] = -2 * rx[i] / N0;
		//cout << rx[i] << " ";
		LLR_VN[i] = rx[i];

	}
	//cout << endl;
	for (int it = 0; it < iter; ++it)
	{
		for (int k = 0; k < M; k++)
		{
			Lqij_min = HUGE_VALF;
			index_min = 0;
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				Lqij[k][i] = LLR_VN[CNtoVN[k][i]] - LLR_CNtoVN[k][i];
				Lqij_sort[i] = Lqij[k][i];
				Lqij_index[i] = i;
			}
			std::sort(Lqij_index.begin(), Lqij_index.begin() + CNtoVN[k].size(), [&](int i1, int i2) { return abs_f(Lqij_sort[i1]) < abs_f(Lqij_sort[i2]); });
			sig = 1;
			tanhtemp = HUGE_VALF;
			for (int i = 0; i < CNtoVN[k].size(); i++)
			{
				if (i < 1)
				{
					for (int j = 0; j < CNtoVN[k].size(); j++)
					{
						if (j != i)
						{
							x1 = Lqij[k][Lqij_index[j]];
							sig = sig * mysign(x1);
							tanhtemp = adjusted_function(tanhtemp, x1);//abs_f(min_f(abs_f(tanhtemp), abs_f(x1)) + lambda_adjusted(abs_f(tanhtemp) + abs_f(x1)) - lambda_adjusted(abs_f(abs_f(tanhtemp) - abs_f(x1))));
							//tanhtemp = min_f(abs(tanhtemp), abs_f(x1));
						}
						
					}
					LLR_CNtoVN[k][Lqij_index[i]] = sig * tanhtemp;
				}
				else
				{
					for (int j = 0; j < CNtoVN[k].size(); j++)
					{
						x1 = Lqij[k][Lqij_index[j]];
						sig = sig * mysign(x1);
						tanhtemp = adjusted_function(tanhtemp, x1);//abs_f(min_f(abs_f(tanhtemp), abs_f(x1)) + lambda_adjusted(abs_f(tanhtemp) + abs_f(x1)) - lambda_adjusted(abs_f(abs_f(tanhtemp) - abs_f(x1))));
						//tanhtemp = min_f(abs(tanhtemp), abs_f(x1));
					}
					LLR_CNtoVN[k][Lqij_index[i]] = sig * mysign(Lqij[k][Lqij_index[i]]) * tanhtemp;
				}
				//LLR_CNtoVN[k][i] = 0.85 * tanhtemp;
				LLR_VN[CNtoVN[k][Lqij_index[i]]] = Lqij[k][Lqij_index[i]] + LLR_CNtoVN[k][Lqij_index[i]];
			}

		}
		early_terminate_flag = 1;
		if (it > 8 && it % 2 == 0)
		{
			int tmp;
			for (int k = 0;k < M;k++)
			{
				tmp = 0;
				for (int j = 0; j < N; j++)
				{
					tmp ^= (LLR_VN[j] >= 0 ? 0 : 1) & H[k][j];
					//cout << uhat[i] << ",";

				}
				if (tmp != 0)
				{
					early_terminate_flag = 0;
					break;
				}
			}
			if (early_terminate_flag == 1)
			{
				break;
			}
		}

	}
	for (int i = 0; i < N; i++)
	{
		LLR_out[i] = LLR_VN[i];
	}

}