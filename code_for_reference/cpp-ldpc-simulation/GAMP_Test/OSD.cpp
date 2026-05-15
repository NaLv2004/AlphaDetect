#include <vector>
#include <fstream>
#include <ctime>
#include <functional>
#include <algorithm>
#include <queue>
#include <iostream>
#include "crc.h"
using namespace std;

float y_array[1296];
int temp_swap_row[20000];
void crc_generator_matrix2(int B, int L, int* polynomial, int** G_crc)
{
	// B: DataLength  -- A+L
	// G_crc size is A x B
	for (int i = 0;i < B - L;i++)
	{
		memset(G_crc[i], 0, sizeof(int) * B);  // memset 将G_crc指向的前A个内存用 0来替换；
		for (int k = 0;k < L + 1;k++)
		{
			G_crc[i][i + k] = polynomial[k]; // 0 - 16
		}
	}

}
void swap_row(int* a, int* b, int N)
{
	for (int i = 0; i < N; i++)
	{
		temp_swap_row[i] = a[i];
		a[i] = b[i];
	}
	for (int i = 0; i < N; i++)
	{
		b[i] = temp_swap_row[i];
	}
}
void Gaussian_ele(int CodeLength, int N_info, int** G_aug_swapped)
{

	for (int c = CodeLength - N_info;c < CodeLength;c++)
	{
		if (G_aug_swapped[c - (CodeLength - N_info)][c] == 0)
		{
			for (int r = c - (CodeLength - N_info) + 1;r < N_info;r++)
			{
				if (G_aug_swapped[r][c] == 1)
				{
					swap_row(G_aug_swapped[r], G_aug_swapped[c - (CodeLength - N_info)], CodeLength);
					break;
				}

			}
		}
		for (int r = 0;r < N_info;r++)
		{
			if ((r != c - (CodeLength - N_info)) && G_aug_swapped[r][c] == 1)
			{
				for (int j = 0;j < CodeLength;j++)
				{
					G_aug_swapped[r][j] ^= G_aug_swapped[c - (CodeLength - N_info)][j];
				}
			}
		}
	}

}
void Gaussian_ele1(int CodeLength, int N_info, int** G_aug_swapped, vector<int>& index_indep, vector<int>& index_parity)
{   // CodeLength --B; N_info -- A ;   G_aug_swapped -- size is A x B with CRC in front 
	int independent_flag = 0;
	int cnt_parity = 0;
	int c = 0;
	//cout << N_info << endl;
	for (int cnt = 0;cnt < N_info;cnt++)
	{
		//cout << cnt << endl;
		independent_flag = 0;

		while (independent_flag == 0)
		{

			if (G_aug_swapped[cnt][c] == 1)
			{
				break;
			}
			if (G_aug_swapped[cnt][c] == 0)
			{
				for (int r = cnt;r < N_info;r++)
				{
					if (G_aug_swapped[r][c] == 1)
					{
						swap_row(G_aug_swapped[r], G_aug_swapped[cnt], CodeLength);
						independent_flag = 1;
						break;
					}

				}
				if (independent_flag == 0)
				{
					index_parity[cnt_parity] = c;
					c++; cnt_parity++;
					if (c >= CodeLength)
					{
						cout << "wrong!";
						break;
					}
				}
			}

		}

		index_indep[cnt] = c;
		//cout << cnt << ":" << index_indep[cnt] << endl;
		for (int r = 0;r < N_info;r++)
		{
			if ((r != cnt) && G_aug_swapped[r][c] == 1)
			{
				for (int j = 0;j < CodeLength;j++)
				{
					G_aug_swapped[r][j] ^= G_aug_swapped[cnt][j]; // 按位与
				}
			}
		}
		c++;
	}
	for (int i = cnt_parity;i < CodeLength - N_info;i++)
	{
		index_parity[i] = c;
		c++;
	}
	//for (int i = 0;i < CodeLength - N_info;i++)
	//{
		//cout << i << ":" << index_parity[i] << endl;
	//}//*/

}
void Gaussian_ele_Hardware(int CodeLength, int N_info, int *pivot_flag, int** G_aug_swapped, vector<int>& index_LLR, vector<int>& index_indep, vector<int>& index_parity)
{
	int independent_flag = 0;
	int cnt_parity = 0;
	int c = 0;
	int row_pivot = 0;
	//memset(pivot_flag, 1, sizeof(int) * N_info);
	for (int i = 0;i < N_info;i++)
	{
		pivot_flag[i] = 1;
	}
	//cout << N_info << endl;
	for (int cnt = 0;cnt < N_info;cnt++)
	{
		//cout << cnt << endl;
		independent_flag = 0;
		while (independent_flag == 0)
		{
			for (int r = 0;r < N_info;r++)
			{
				if (pivot_flag[r] == 1 && G_aug_swapped[r][index_LLR[c]] == 1)
				{
					row_pivot = r;
					pivot_flag[r] = 0;
					independent_flag = 1;
					break;
				}

			}
			if (independent_flag == 0)
			{
				index_parity[cnt_parity] = index_LLR[c];
				c++; cnt_parity++;
				if (c >= CodeLength)
				{
					cout << "wrong!";
					break;
				}
			}
			
			
		}
		index_indep[cnt] = index_LLR[c];
		//cout << index_LLR[c] << endl;
		//cout << cnt << ":" << index_indep[cnt] << endl;
		for (int r = 0;r < N_info;r++)
		{
			if ((r != row_pivot) && G_aug_swapped[r][index_LLR[c]] == 1)
			{
				for (int j = 0;j < CodeLength;j++)
				{
					G_aug_swapped[r][j] ^= G_aug_swapped[row_pivot][j];
				}
			}
		}
		c++;
	}

	for (int i = cnt_parity;i < CodeLength - N_info;i++)
	{
		index_parity[i] = index_LLR[c];
		c++;
	}
	//for (int i = 0;i < CodeLength - N_info;i++)
	//{
		//cout << i << ":" << index_parity[i] << endl;
	//}//*/

}
void backTrack(int n, vector<vector<int>>& nums, vector<int>& ins) {
	int i = 1;
	while (n >= i) {
		if (!ins.empty() && i <= ins.back()) {
			i++;
			continue;
		}
		ins.emplace_back(i);
		n -= i;
		backTrack(n, nums, ins);
		ins.pop_back();
		n += i;
		i++;
	}
	if (ins.size() < 1) return;
	if (n == 0)
		nums.emplace_back(ins);
}
void GRAND_set_build(vector<vector<int>>& GRAND_flip_set, vector<int>& current_set, int CodeLength, int AB_LW, bool GRAND_type)
{
	GRAND_flip_set.clear();
	int num_flip = 0;

	if (GRAND_type == 0)
	{
		for (int i = 0; i < CodeLength; i++)
		{
			current_set.clear();
			current_set.push_back(i);
			GRAND_flip_set.push_back(current_set);
			num_flip++;
		}
		if (AB_LW > 1)
		{
			for (int i = 0; i < CodeLength; i++)
			{
				for (int j = 0; j < CodeLength; j++)
				{
					if (j > i)
					{
						current_set.clear();
						current_set.push_back(i);
						current_set.push_back(j);
						GRAND_flip_set.push_back(current_set);
						num_flip++;
					}
				}
			}
			if (AB_LW > 2)
			{
				for (int i = 0; i < CodeLength; i++)
				{
					for (int j = 0; j < CodeLength; j++)
					{
						if (j > i)
						{
							for (int k = 0; k < CodeLength; k++)
							{
								if (k > j)
								{
									current_set.clear();
									current_set.push_back(i);
									current_set.push_back(j);
									current_set.push_back(k);
									GRAND_flip_set.push_back(current_set);
									num_flip++;
								}
							}
						}
					}
				}
			}
		}
	}
	else
	{
		for (int cnt = 1; cnt <= AB_LW; cnt++)
		{
			current_set.clear();
			backTrack(cnt, GRAND_flip_set, current_set);
		}
		cout << GRAND_flip_set.size() << endl;

		//std::sort(index_LLR.begin(), index_LLR.end(), [&](int i1, int i2) { return abs_LLR[i1] < abs_LLR[i2]; });
	}
}
void decodeOrdered_OSD(int CodeLength, int DataLength, int N_info, int CRC_Length, int AB_LW, float* LLR_in, float* LLR_in_abs, int** G_new, int** G_swapped, int** G_swapped2, int* uhat, int* uhat_swapped1, int* uhat_swapped2, unsigned char* uhat_crc_check, int* uhat_best, vector<int>& index_LLR, vector<int>& index_indep, vector<int>& index_parity, vector<vector<int>>& GRAND_flip_set, vector<int>& current_set)
{
	int N = CodeLength;
	int M = CodeLength - DataLength;
	for (int i = 0; i < N; i++)
	{
		LLR_in_abs[i] = abs(LLR_in[i]);
		index_LLR[i] = i;
	}

	std::sort(index_LLR.begin(), index_LLR.end(), [&](int i1, int i2) { return LLR_in_abs[i1] > LLR_in_abs[i2]; });
	for (int i = 0;i < DataLength;i++)
	{
		for (int j = 0;j < N;j++)
		{
			G_swapped[i][j] = G_new[i][index_LLR[j]];
			//cout << index_LLR[j] << " ";
		}
		//cout << endl;
	}
	Gaussian_ele1(N, DataLength, G_swapped, index_indep, index_parity);

	for (int i = 0;i < DataLength;i++)
	{
		for (int j = 0;j < N - M;j++)
		{
			G_swapped2[i][j] = G_swapped[i][index_indep[j]];
		}
		for (int j = 0;j < M;j++)
		{
			G_swapped2[i][j + N - M] = G_swapped[i][index_parity[j]];
		}
	}

	for (int i = 0; i < N - M;i++)
	{
		//if(ldpc_t_struct.H[i][i]!=1)
		//cout << i << " ";
		if (i > 0)
		{
			for (int j = 0;j < i - 1;j++)
			{
				if (G_swapped2[i][j] == 1) cout << "wrong1 at " << "row" << i << "colume" << j << endl;
				//if(ldpc_t_struct.H[i][j]==1)
					//cout << j << " ";
			}
		}
		if (G_swapped2[i][i] != 1) cout << "wrong2 at " << "row" << i << "colume" << i << endl;
		for (int j = i + 1;j < N - M;j++)
		{
			if (G_swapped2[i][j] == 1) cout << "wrong3 at " << "row" << i << "colume" << j << endl;
			//if(ldpc_t_struct.H[i][j]==1)
				//cout << j << " ";
		}
		//cout << endl;
	}//*/
	GRAND_set_build(GRAND_flip_set, current_set, DataLength, AB_LW, 0);
	float PM_min = 0;
	float PM_temp = 0;
	for (int i = 0; i < N; i++)
	{
		uhat[i] = LLR_in[index_LLR[i]] >= 0 ? 0 : 1;
		//cout << uhat[i] << ",";

	}
	for (int i = 0;i < N - M;i++)
	{
		uhat[i] = uhat[index_indep[i]];
	}
	for (int i = N - M;i < N;i++)
	{
		uhat[i] = 0;
		for (int j = 0;j < N - M;j++)
		{
			uhat[i] ^= uhat[j] & G_swapped2[j][i];
		}
	}
	for (int i = 0;i < N - M;i++)
	{
		uhat_swapped1[index_indep[i]] = uhat[i];
	}
	for (int i = N - M;i < N;i++)
	{
		uhat_swapped1[index_parity[i - (N - M)]] = uhat[i];
	}
	for (int i = 0; i < N; i++)
	{
		uhat_swapped2[index_LLR[i]] = uhat_swapped1[i];
		uhat_best[index_LLR[i]] = uhat_swapped1[i];

		//cout << uhat[i] << ",";

	}
	for (int cnt = 0; cnt < N; cnt++)
	{
		int hard_bit = LLR_in[cnt] >= 0 ? 0 : 1;
		PM_min += abs((hard_bit - uhat_swapped2[cnt]) * LLR_in[cnt]);
	}
	if (AB_LW > 0)
	{
		for (int num = 0; num < GRAND_flip_set.size(); num++)
		{
			int crc_pass_flag = 0;
			for (int i = 0; i < N; i++)
			{
				uhat[i] = LLR_in[index_LLR[i]] >= 0 ? 0 : 1;
				//cout << uhat[i] << ",";

			}
			for (int i = 0;i < N - M;i++)
			{
				uhat[i] = uhat[index_indep[i]];
			}
			for (int cnt = 0; cnt < GRAND_flip_set[num].size(); cnt++)
			{
				uhat[GRAND_flip_set[num][cnt]] ^= 1;
			}

			for (int i = N - M;i < N;i++)
			{
				uhat[i] = 0;
				for (int j = 0;j < N - M;j++)
				{
					uhat[i] ^= uhat[j] & G_swapped2[j][i];
				}
			}
			for (int i = 0;i < N - M;i++)
			{
				uhat_swapped1[index_indep[i]] = uhat[i];
			}
			for (int i = N - M;i < N;i++)
			{
				uhat_swapped1[index_parity[i - (N - M)]] = uhat[i];
			}
			for (int i = 0; i < N; i++)
			{
				uhat_swapped2[index_LLR[i]] = uhat_swapped1[i];

				//cout << uhat[i] << ",";

			}
			memset(uhat_crc_check, 0, sizeof(unsigned char)* N_info);
			for (int i = 0;i < N_info - CRC_Length;i++)
			{
				uhat_crc_check[i] = uhat_swapped2[i];
			}
			tx_append_crc(uhat_crc_check, N_info - CRC_Length, CRC_Length, 1);
			for (int i = 0;i < CRC_Length;i++)
			{
				if ((int)uhat_crc_check[i + N_info - CRC_Length] != uhat_swapped2[i + N_info - CRC_Length])
				{
					crc_pass_flag = 0;
					break;
				}
				if (i == CRC_Length - 1)
				{
					crc_pass_flag = 1;
					//cout << "passed!" << endl;
				}
			}
			PM_temp = 0;
			for (int cnt = 0; cnt < N; cnt++)
			{
				int hard_bit = LLR_in[cnt] >= 0 ? 0 : 1;
				PM_temp += abs((hard_bit - uhat_swapped2[cnt]) * LLR_in[cnt]);
			}
			if (PM_temp < PM_min && crc_pass_flag == 1)
			{
				PM_min = PM_temp;
				for (int i = 0; i < N; i++)
				{
					uhat_best[i] = uhat_swapped2[i];
					//cout << "!";


				}
			}
		}
		

	}
}
void decodeOrdered_OSD_CRCaided(int CodeLength, int DataLength, int N_info, int CRC_Length, int AB_LW, float* LLR_in, float* LLR_in_abs, int** G_new, int** G_swapped, int** G_swapped2, int** H, int* uhat, int* uhat_swapped1, int* uhat_swapped2, unsigned char* uhat_crc_check, int* uhat_best, vector<int>& index_LLR, vector<int>& index_indep, vector<int>& index_parity, vector<vector<int>>& GRAND_flip_set, vector<int>& current_set)
{
	int N = CodeLength;
	int M = CodeLength - DataLength;
	int early_terminate_flag;
	for (int i = 0; i < N; i++)
	{
		LLR_in_abs[i] = abs(LLR_in[i]);
		index_LLR[i] = i;
	}

	std::sort(index_LLR.begin(), index_LLR.end(), [&](int i1, int i2) { return LLR_in_abs[i1] > LLR_in_abs[i2]; });
	
	for (int i = 0;i < N_info;i++)
	{
		for (int j = 0;j < N;j++)
		{
			G_swapped[i][j] = G_new[i][index_LLR[j]];
			//cout << index_LLR[j] << " ";
		}
		//cout << endl;
	}
	Gaussian_ele1(N, N_info, G_swapped, index_indep, index_parity);

	for (int i = 0;i < N_info;i++)
	{
		for (int j = 0;j < N_info;j++)
		{
			G_swapped2[i][j] = G_swapped[i][index_indep[j]];
		}
		for (int j = 0;j < N - N_info;j++)
		{
			G_swapped2[i][j + N_info] = G_swapped[i][index_parity[j]];
		}
	}
	GRAND_set_build(GRAND_flip_set, current_set, N_info, AB_LW, 0);
	float PM_min = 0;
	float PM_temp = 0;
	for (int i = 0; i < N; i++)
	{
		uhat[i] = LLR_in[index_LLR[i]] >= 0 ? 0 : 1;
		//cout << uhat[i] << ",";

	}
	for (int i = 0;i < N_info;i++)
	{
		uhat[i] = uhat[index_indep[i]];
	}
	for (int i = N_info;i < N;i++)
	{
		uhat[i] = 0;
		for (int j = 0;j < N_info;j++)
		{
			uhat[i] ^= uhat[j] & G_swapped2[j][i];
		}
	}
	for (int i = 0;i < N_info;i++)
	{
		uhat_swapped1[index_indep[i]] = uhat[i];
	}
	for (int i = N_info;i < N;i++)
	{
		uhat_swapped1[index_parity[i - N_info]] = uhat[i];
	}
	for (int i = 0; i < N; i++)
	{
		uhat_swapped2[index_LLR[i]] = uhat_swapped1[i];
		uhat_best[index_LLR[i]] = uhat_swapped1[i];

		//cout << uhat[i] << ",";

	}
	for (int cnt = 0; cnt < N; cnt++)
	{
		int hard_bit = LLR_in[cnt] >= 0 ? 0 : 1;
		PM_min += abs((hard_bit - uhat_swapped2[cnt]) * LLR_in[cnt]);
		
	}
	//cout << PM_min << endl;
	if (AB_LW > 0)
	{
		for (int num = 0; num < GRAND_flip_set.size(); num++)
		{
			for (int i = 0; i < N; i++)
			{
				uhat[i] = LLR_in[index_LLR[i]] >= 0 ? 0 : 1;
				//cout << uhat[i] << ",";

			}
			for (int i = 0;i < N_info;i++)
			{
				uhat[i] = uhat[index_indep[i]];
			}
			for (int cnt = 0; cnt < GRAND_flip_set[num].size(); cnt++)
			{
				uhat[GRAND_flip_set[num][cnt]] ^= 1;
			}

			for (int i = N_info;i < N;i++)
			{
				uhat[i] = 0;
				for (int j = 0;j < N_info;j++)
				{
					uhat[i] ^= uhat[j] & G_swapped2[j][i];
				}
			}
			for (int i = 0;i < N_info;i++)
			{
				uhat_swapped1[index_indep[i]] = uhat[i];
			}
			for (int i = N_info;i < N;i++)
			{
				uhat_swapped1[index_parity[i - N_info]] = uhat[i];
			}
			for (int i = 0; i < N; i++)
			{
				uhat_swapped2[index_LLR[i]] = uhat_swapped1[i];

				//cout << uhat[i] << ",";

			}
			PM_temp = 0;
			for (int cnt = 0; cnt < N; cnt++)
			{
				int hard_bit = LLR_in[cnt] >= 0 ? 0 : 1;
				PM_temp += abs((hard_bit - uhat_swapped2[cnt]) * LLR_in[cnt]);
			}
			if (PM_temp < PM_min)
			{
				PM_min = PM_temp;
				for (int i = 0; i < N; i++)
				{
					uhat_best[i] = uhat_swapped2[i];
					//cout << "!";


				}
			}

		}
		
	}
}
void decodeOrdered_OSD_CRCaided_Hardware(int CodeLength, int DataLength, int N_info, int CRC_Length, int AB_LW, float* LLR_in, float* LLR_in_abs, int* pivot_flag, int** G_new, int** G_swapped, int** G_swapped2, int** H, int* uhat, int* uhat_encode, int* uhat_swapped2, unsigned char* uhat_crc_check, int* uhat_best, vector<int>& index_LLR, vector<int>& index_indep, vector<int>& index_parity, vector<vector<int>>& GRAND_flip_set, vector<int>& current_set)
{
	int N = CodeLength;
	int M = CodeLength - DataLength;
	int early_terminate_flag;
	for (int i = 0; i < N; i++)
	{
		LLR_in_abs[i] = abs(LLR_in[i]);
		index_LLR[i] = i;
	}

	std::sort(index_LLR.begin(), index_LLR.end(), [&](int i1, int i2) { return LLR_in_abs[i1] > LLR_in_abs[i2]; });

	
	Gaussian_ele_Hardware(N, N_info, pivot_flag, G_new, index_LLR, index_indep, index_parity);

	/*for (int i = 0;i < N_info;i++)
	{
		for (int j = 0;j < N_info;j++)
		{
			G_swapped2[i][j] = G_swapped[i][index_indep[j]];
		}
		for (int j = 0;j < N - N_info;j++)
		{
			G_swapped2[i][j + N_info] = G_swapped[i][index_parity[j]];
		}
	}*/
	GRAND_set_build(GRAND_flip_set, current_set, N_info, AB_LW, 0);
	float PM_min = 0;
	float PM_temp = 0;
	for (int i = 0; i < N; i++)
	{
		uhat[i] = LLR_in[i] >= 0 ? 0 : 1;
		//cout << uhat[i] << ",";

	}
	for (int i = 0;i < N_info;i++)
	{
		uhat[i] = uhat[index_indep[i]];
	}
	for (int i = 0;i < N;i++)
	{
		for (int j = 0;j < N_info;j++)
		{
			uhat_encode[i] ^= uhat[j] & G_new[j][i];
		}
	}
	/*for (int i = 0;i < N_info;i++)
	{
		uhat_swapped1[index_indep[i]] = uhat[i];
	}
	for (int i = N_info;i < N;i++)
	{
		uhat_swapped1[index_parity[i - N_info]] = uhat[i];
	}*/
	for (int i = 0; i < N; i++)
	{
		//uhat_swapped2[index_LLR[i]] = uhat_swapped1[i];
		//uhat_best[index_LLR[i]] = uhat_swapped1[i];
		uhat_best[i] = uhat_encode[i];

		//cout << uhat[i] << ",";

	}
	for (int cnt = 0; cnt < N; cnt++)
	{
		int hard_bit = LLR_in[cnt] >= 0 ? 0 : 1;
		PM_min += abs((hard_bit - uhat_swapped2[cnt]) * LLR_in[cnt]);

	}
	//cout << PM_min << endl;
	/*if (AB_LW > 0)
	{
		for (int num = 0; num < GRAND_flip_set.size(); num++)
		{
			for (int i = 0; i < N; i++)
			{
				uhat[i] = LLR_in[index_LLR[i]] >= 0 ? 0 : 1;
				//cout << uhat[i] << ",";

			}
			for (int i = 0;i < N_info;i++)
			{
				uhat[i] = uhat[index_indep[i]];
			}
			for (int cnt = 0; cnt < GRAND_flip_set[num].size(); cnt++)
			{
				uhat[GRAND_flip_set[num][cnt]] ^= 1;
			}

			for (int i = N_info;i < N;i++)
			{
				uhat[i] = 0;
				for (int j = 0;j < N_info;j++)
				{
					uhat[i] ^= uhat[j] & G_swapped2[j][i];
				}
			}
			for (int i = 0;i < N_info;i++)
			{
				uhat_swapped1[index_indep[i]] = uhat[i];
			}
			for (int i = N_info;i < N;i++)
			{
				uhat_swapped1[index_parity[i - N_info]] = uhat[i];
			}
			for (int i = 0; i < N; i++)
			{
				uhat_swapped2[index_LLR[i]] = uhat_swapped1[i];

				//cout << uhat[i] << ",";

			}
			PM_temp = 0;
			for (int cnt = 0; cnt < N; cnt++)
			{
				int hard_bit = LLR_in[cnt] >= 0 ? 0 : 1;
				PM_temp += abs((hard_bit - uhat_swapped2[cnt]) * LLR_in[cnt]);
			}
			if (PM_temp < PM_min)
			{
				PM_min = PM_temp;
				for (int i = 0; i < N; i++)
				{
					uhat_best[i] = uhat_swapped2[i];
					//cout << "!";


				}
			}

		}

	}*/
}
void decode_OSD_modified_for_hardware(float* LLR_in, int* Bit_out, int M, int N, int** H, vector<int>& index_indep, vector<int>& index_parity, int* pivot_flag, float* LLR_in_abs, vector<int>& index_LLR, int** Htemp, int** H_swapped2, int* LRIP, int* LRIPtemp, int* syndrome, int* syndrometemp, int* Bit_temp)
{
	int i, j, wgt, wgttemp, index_flip;
	float temp;

	//step1: Sort
	for (i = 0; i < N; i++) {
		LLR_in_abs[i] = abs(LLR_in[i]);
		index_LLR[i] = i;
	}
	std::sort(index_LLR.begin(), index_LLR.end(), [&](int i1, int i2) { return LLR_in_abs[i1] < LLR_in_abs[i2]; });
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			//Htemp[i][j] = H[i][index_LLR[j]];
			Htemp[i][j] = H[i][j];
		}
	}
	//step2: Gaussian elimate
	//Gaussian_ele1(N, M, Htemp, index_indep, index_parity);
	Gaussian_ele_Hardware(N, M, pivot_flag, Htemp, index_LLR, index_indep, index_parity);
	/*for (i = 0; i < M; i++) {
		for (j = 0; j < M; j++) {
			H_swapped2[i][j] = Htemp[i][index_indep[j]];
		}
		for (j = 0; j < N - M; j++) {
			H_swapped2[i][j + M] = Htemp[i][index_parity[j]];
		}
	}*/

	/*for (i = 0; i < M; i++)
	{
		//if(ldpc_t_struct.H[i][i]!=1)
		//cout << i << " ";
		if (i > 0)
		{
			for (j = 0; j < i - 1; j++)
			{
				if (H_swapped2[i][j] == 1) cout << "wrong1 at " << "row" << i << "colume" << j << endl;
				//if(ldpc_t_struct.H[i][j]==1)
					//cout << j << " ";
			}
		}
		if (H_swapped2[i][i] != 1) cout << "wrong2 at " << "row" << i << "colume" << i << endl;
		for (j = i + 1; j < M; j++)
		{
			if (H_swapped2[i][j] == 1) cout << "wrong3 at " << "row" << i << "colume" << j << endl;
			//if(ldpc_t_struct.H[i][j]==1)
				//cout << j << " ";
		}
		//cout << endl;
	}*/

	//step3: LRIPs
	for (i = 0; i < N; i++) {
		LRIP[i] = LLR_in[i] >= 0 ? 0 : 1;
	}
	//cout << endl;
	/*for (i = 0; i < M; i++) {
		LRIP[i] = LRIPtemp[index_indep[i]];
		//cout << LRIP[i] << " ";
	}
	for (i = M; i < N; i++) {
		LRIP[i] = LRIPtemp[index_parity[i - M]];
		//cout << LRIP[i] << " ";
	}*/
	//cout << endl;
	//step4: Syndrome calculate
	//cout << endl;
	int cnt1 = 0;
	for (i = 0; i < M; i++) {
		syndrome[i] = 0;
		for (j = 0; j < N; j++) {
			syndrome[i] ^= Htemp[i][j] & LRIP[j];
		}
		cnt1 += syndrome[i];
		//cout << syndrome[i] << " ";
	}
	//cout << cnt1 << endl;
	//cout << endl;
	//step4.5: OSD1 flip
	index_flip = 0;
	wgt = M;
	for (i = 0; i < N; i++) {
		wgttemp = 0;
		for (j = 0; j < M; j++) {
			syndrometemp[j] = syndrome[j] ^ Htemp[j][i];
			wgttemp += syndrometemp[j];
		}
		if (wgttemp < wgt) {
			wgt = wgttemp;
			index_flip = i;
		}
	}
	//cout << wgt << endl;
	LRIP[index_flip] ^= 1;
	for (j = 0; j < M; j++) {
		syndrometemp[j] = syndrome[j] ^ Htemp[j][index_flip];
	}
	for (j = 0; j < M; j++) {
		syndrome[j] = syndrometemp[j];
	}
	
	//step5: OSD0 flip
	for (i = 0; i < M; i++) {
		if (syndrome[i] == 1) {
			for (j = 0; j < M; j++) {//这里是M，不是N
				if (Htemp[i][index_indep[j]] == 1) {
					LRIP[index_indep[j]] ^= 1;
				}
			}
		}
	}
	//step6: Get result
	/*for (i = 0; i < M; i++) {
		Bit_temp[index_indep[i]] = LRIP[i];
	}
	for (i = M; i < N; i++) {
		Bit_temp[index_parity[i - M]] = LRIP[i];
	}*/
	for (i = 0; i < N; i++)
	{
		Bit_out[i] = LRIP[i];
	}
}