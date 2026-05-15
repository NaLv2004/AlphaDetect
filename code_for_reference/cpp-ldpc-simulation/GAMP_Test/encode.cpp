#include <cstdio>
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <stdint.h>
#include "encode.h"
using namespace std;
class Matrix {
public:
	int r, c;
	int** m;
	Matrix(int row, int col) {  //constructor initializa row and col ; value is zero
		r = row;
		c = col;
		m = new int*[row];
		for (int i = 0; i < row; i++)
		{
			m[i] = new int[col];
		}
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				m[i][j] = 0;
	}
	Matrix() {
		r = c = 0;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				m[i][j] = 0;
	}
	~Matrix() {
		for (int i = 0; i < r; ++i)
			delete[]m[i];
		delete[]m;
	}
	void my_malloc() {
		m = new int*[r];
		for (int i = 0; i < r; i++)
		{
			m[i] = new int[c];
		}
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				m[i][j] = 0;
	}
	void clear() {
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				m[i][j] = 0;
	}
};
int Zsets[8][8] = {
	{ 2,4,8,16,32,64,128,256 },
	{ 3,6,12,24,48,96,192,384 },
	{ 5,10,20,40,80,160,320,0 },
	{ 7,14,28,56,112,224,0,0 },
	{ 9,18,36,72,144,288,0,0 },
	{ 11,22,44,88,176,352,0,0 },
	{ 13,26,52,104,208,0,0,0 },
	{ 15,30,60,120,240,0,0,0 }
};
/*存在深拷贝问题 暂时未解决
Matrix blockMultiply(Matrix A,Matrix B) {
Matrix out;
out.r = B.r;
out.c = A.r;
out.my_malloc();
int i, j, k;
for (i = 0; i < A.r; i++) {
for (j = 0; j < A.c; j++) {
if (A.m[i][j] != -1) {
vector<int> tmp;
for (k = A.m[i][j]; k < B.r; k++) {
tmp.push_back(B.m[k][j]);
}
for (k = 0; k < A.m[i][j]-1; k++) {
tmp.push_back(B.m[k][j]);
}
for (k = 0; k < out.r; k++) {
out.m[k][i] += tmp[k];
}
}
}
}
}
*/
Matrix V;		// original base graph
Matrix P;		// P=V % Zc lifted base graph 
Matrix P1;		// message matrix --mb x Kb
Matrix infoVec; // size is Zc x Kb =K
Matrix d;		// size is Zc x mb
Matrix d0;      // size is Zc x 4
Matrix P3;      // size is (mb-4) x 4
Matrix tmpm;    // size is Zc x 4
Matrix p;       // size is Zc x (mb-4)
int** outCball; // size is N x 1 including puncture part
void encode_pre(int K, int Zc, int bgn) {
	if (bgn == 1) {
		V.r = 46;
		V.c = 68;
	}
	else {
		V.r = 42;
		V.c = 52;
	}
	V.my_malloc();

	P.r = V.r; P.c = V.c;
	P.my_malloc();

	P1.r = P.r; P1.c = P.c - P.r;
	P1.my_malloc();

	infoVec.r = Zc; infoVec.c = K / Zc;
	infoVec.my_malloc();

	d.r = infoVec.r; d.c = P1.r;
	d.my_malloc();

	d0.r = d.r; d0.c = 4;
	d0.my_malloc();

	P3.r = P.r - 4; P3.c = 4;
	P3.my_malloc();

	tmpm.r = d0.r; tmpm.c = 4;
	tmpm.my_malloc();

	p.r = tmpm.r; p.c = P3.r;
	p.my_malloc();
	int nsys;		// message column in base graph
	int ncwnodes;   // transmited column in base graph
	if (bgn == 1) {
		nsys = 22;
		ncwnodes = 66;
	}
	else {
		nsys = 10;
		ncwnodes = 50;
	}
	outCball = new int*[(ncwnodes + 2) * Zc];
	for (int i = 0; i < (ncwnodes + 2) * Zc; i++) {
		outCball[i] = new int[1]();
	}
}



void encode(int** infobits, int K, int** out, int Zc, int bgn, int setIdx) {
	// infobits -Kx 1
	int i, j, k;
	int Nplus2Zc;
	V.clear(); P.clear(); P1.clear(); infoVec.clear(); d.clear(); d0.clear(); P3.clear(); tmpm.clear(); p.clear();
	switch (bgn) { // select the original base graph V
	case 1: {
		Nplus2Zc = Zc * (66 + 2);
		switch (setIdx) {
		case 1: {
			fstream fileV;
			fileV.open("BG1S1.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		case 2: {
			fstream fileV;
			fileV.open("BG1S2.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		case 3: {
			fstream fileV;
			fileV.open("BG1S3.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		case 4: {
			fstream fileV;
			fileV.open("BG1S4.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		case 5: {
			fstream fileV;
			fileV.open("BG1S5.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		case 6: {
			fstream fileV;
			fileV.open("BG1S6.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		case 7: {
			fstream fileV;
			fileV.open("BG1S7.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		default: {
			fstream fileV;
			fileV.open("BG1S8.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		}
		break;
	}
	default: {
		Nplus2Zc = Zc * (40 + 2);
		switch (setIdx) {
		case 1: {
			fstream fileV;
			fileV.open("BG2S1.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		case 2: {
			fstream fileV;
			fileV.open("BG2S2.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		case 3: {
			fstream fileV;
			fileV.open("BG2S3.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		case 4: {
			fstream fileV;
			fileV.open("BG2S4.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		case 5: {
			fstream fileV;
			fileV.open("BG2S5.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		case 6: {
			fstream fileV;
			fileV.open("BG2S6.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		case 7: {
			fstream fileV;
			fileV.open("BG2S7.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		default: {
			fstream fileV;
			fileV.open("BG2S8.txt", ios::in);
			for (int i = 0; i < V.r; i++)
				for (int j = 0; j < V.c; j++)
					fileV >> V.m[i][j];
			fileV.close();
			break;
		}
		}
	}
	}

	for (i = 0; i < V.r; i++) {
		for (j = 0; j < V.c; j++) {
			P.m[i][j] = V.m[i][j] % Zc;
		}
	}

	//P1  message part of base graph
	for (i = 0; i < P1.r; i++) {
		for (j = 0; j < P1.c; j++) {
			P1.m[i][j] = P.m[i][j];
		}
	}
	

	int Htype[2][8] = {
		{ 3,3,3,3,3,3,2,3 },
		{ 4,4,4,1,4,4,4,1 }
	};

	// Coloum wise - reserve message bit 
	for (int r = 0; r < batchsize; r++) {
		for (i = 0; i < infoVec.c; i++) { // Kb 
			for (j = 0; j < infoVec.r; j++) { // row Zc
				infoVec.m[j][i] = infobits[infoVec.r * i + j][r];
			}
		}
		
		//将信息和校验矩阵的信息位进行分块相乘，每一行H相乘的结果得到 Kb 个长度为Zc的向量，将所有元素相加（Kb-1次加法），得到一个长度为Zc的向量，H有mb行，得到mb 个这样的向量，存在d中；
		for (i = 0; i < P1.r; i++) {
			for (j = 0; j < P1.c; j++) {
				if (P1.m[i][j] != -1) {
					vector<int> tmp; // length is Zc
					for (k = P1.m[i][j]; k < infoVec.r; k++) {
						tmp.push_back(infoVec.m[k][j]);
					}
					for (k = 0; k < P1.m[i][j]; k++) {
						tmp.push_back(infoVec.m[k][j]);
					}
					//cout << tmp.size();
					for (k = 0; k < d.r; k++) {
						d.m[k][i] += tmp[k];
					}
				}
			}
		}
		 
		// 取d的前四行用于译码；
		for (i = 0; i < d0.r; i++) {
			for (j = 0; j < d0.c; j++) {
				d0.m[i][j] = d.m[i][j];
			}
		}
		 
		vector<int> m1, m2, m3, m4;  //length is Zc
		// bgn==1;[3,3,3,3,3,3,2,3]
		// different choise of base graph section B 
		// calculate the section B parity bits
		switch (Htype[bgn - 1][setIdx - 1])
		{
		case 1: {
			for (i = 0; i < d0.r; i++) {
				int tmp = 0;
				for (j = 0; j < d0.c; j++) {
					tmp += d0.m[i][j];
				}
				m1.push_back(tmp);  // zc x 1
			}
			vector<int> m1tmp;
			for (i = 0; i < d0.r - 1; i++) {
				m1tmp.push_back(m1[i + 1]);
			}
			m1tmp.push_back(m1[0]);

			for (i = 0; i < d0.r; i++) {
				m2.push_back(d0.m[i][0] + m1tmp[i]);
			}
			for (i = 0; i < d0.r; i++) {
				m3.push_back(d0.m[i][1] + m2[i]);
			}
			for (i = 0; i < d0.r; i++) {
				m4.push_back(d0.m[i][2] + m1[i] + m3[i]);
			}
			break;
		}
		case 2: {
			vector<int> m1tmp;
			for (i = 0; i < d0.r; i++) {
				int tmp = 0;
				for (j = 0; j < d0.c; j++) {
					tmp += d0.m[i][j];
				}
				m1tmp.push_back(tmp);
			}
			int shift = 105 % Zc;
			if (shift > 0) {
				for (i = 0; i < shift; i++) {
					m1.push_back(m1tmp[d0.r - shift + i]);
				}
				for (i = 0; i < d0.r - shift; i++) {
					m1.push_back(m1tmp[i]);
				}
			}
			for (i = 0; i < d0.r; i++) {
				m2.push_back(d0.m[i][0] + m1[i]);
			}
			for (i = 0; i < d0.r; i++) {
				m4.push_back(d0.m[i][3] + m1[i]);
			}
			for (i = 0; i < d0.r; i++) {
				m3.push_back(d0.m[i][2] + m4[i]);
			}
			break;
		}
		case 3: {
			for (i = 0; i < d0.r; i++) {
				int tmp = 0;
				for (j = 0; j < d0.c; j++) {
					tmp += d0.m[i][j];
				}
				m1.push_back(tmp);
			}
			vector<int> m1tmp;
			for (i = 0; i < d0.r - 1; i++) {
				m1tmp.push_back(m1[i + 1]);
			}
			m1tmp.push_back(m1[0]);
			for (i = 0; i < d0.r; i++) {
				m2.push_back(d0.m[i][0] + m1tmp[i]);
			}
			for (i = 0; i < d0.r; i++) {
				m3.push_back(d0.m[i][1] + m1[i] + m2[i]);
			}
			for (i = 0; i < d0.r; i++) {
				m4.push_back(d0.m[i][2] + m3[i]);
			}
			break;
		}
		default: { // case 0
			for (i = 0; i < d0.r; i++) {
				int tmp = 0;
				for (j = 0; j < d0.c; j++) {
					tmp += d0.m[i][j];
				}
				m1.push_back(tmp);
			}
			int back = m1[m1.size() - 1];
			for (i = m1.size() - 1; i > 0; i--) {
				m1[i] = m1[i - 1];
			}
			m1[0] = back;
			for (i = 0; i < d0.r; i++) {
				m2.push_back(d0.m[i][0] + m1[i]);
			}
			for (i = 0; i < d0.r; i++) {
				m3.push_back(d0.m[i][1] + m2[i]);
			}
			for (i = 0; i < d0.r; i++) {
				m4.push_back(d0.m[i][3] + m1[i]);
			}
			/*for (i = 0; i < m1.size(); i++) {
			cout << m1[i] << " " << m2[i] << " " << m3[i] << " " << m4[i] << endl;
			}*/
			break;
		}

		}
		//P3 存储的是 base graph 中的 D section
		for (i = 0; i < P3.r; i++) {
			for (j = 0; j < P3.c; j++) {
				P3.m[i][j] = P.m[i + 4][P1.c + j];
			}
		}
		/*for (int i = 0; i < P3.r; i++) {
		for (int j = 0; j < P3.c; j++) {
		cout << P3.m[i][j] << " ";
		}
		cout << endl;
		}*/
		for (i = 0; i < tmpm.r; i++) {
			tmpm.m[i][0] = m1[i];
			tmpm.m[i][1] = m2[i];
			tmpm.m[i][2] = m3[i];
			tmpm.m[i][3] = m4[i];
		}
		//p=blockMultiply(P3, [m1,m2,m3,m4]);

		for (i = 0; i < P3.r; i++) {
			for (j = 0; j < P3.c; j++) {
				if (P3.m[i][j] != -1) {
					vector<int> tmp;
					for (k = P3.m[i][j]; k < tmpm.r; k++) {
						tmp.push_back(tmpm.m[k][j]);
					}
					for (k = 0; k < P3.m[i][j]; k++) {
						tmp.push_back(tmpm.m[k][j]);
					}
					for (k = 0; k < p.r; k++) {
						p.m[k][i] += tmp[k];
					}
				}
			}
		}
		for (i = 0; i < p.r; i++) {
			for (j = 0; j < p.c; j++) {
				p.m[i][j] += d.m[i][j + 4];
			}
		}
		/*for (int i = 0; i < p.r; i++) {
		for (int j = 0; j < p.c; j++) {
		cout << p.m[i][j] << " ";
		}
		cout << endl;
		}*/
		for (i = 0; i < K; i++) {
			out[i][r] = infobits[i][r];
		}
		for (i = 0; i < m1.size(); i++) {
			out[i + K][r] = m1[i] % 2;
		}
		for (i = 0; i < m2.size(); i++) {
			out[i + K + m1.size()][r] = m2[i] % 2;
		}
		for (i = 0; i < m3.size(); i++) {
			out[i + K + m1.size() + m2.size()][r] = m3[i] % 2;
		}
		for (i = 0; i < m4.size(); i++) {
			out[i + K + m1.size() + m2.size() + m3.size()][r] = m4[i] % 2;
		}
		for (i = 0; i < p.c; i++) {
			for (j = 0; j < p.r; j++) {
				/*cout << i * p.r + j + K + m1.size() + m2.size() + m3.size() + m4.size() << endl;*/
				out[i * p.r + j + K + m1.size() + m2.size() + m3.size() + m4.size()][r] = p.m[j][i] % 2;
			}
		}
	}
	/*for (i = 0; i < Nplus2Zc; i++) {
	cout << out[i][2] << endl;
	}*/

}
void nrLdpcEncode(int** in_data, int K, int bgn, int** out_data) {
	// in_data --a Code Block with length is K
	// out_data -- N1 x 1
	int i, j, k, r;
	int nsys, ncwnodes;
	if (bgn == 1) {
		nsys = 22;
		ncwnodes = 66;
	}
	else {
		nsys = 10;
		ncwnodes = 50;
	}
	// find min Zc
	int Zc = 1000000, setIdx;
	for (i = 0; i < 8; i++) {
		for (j = 0; j < 8; j++) {
			if (Zsets[i][j] * nsys >= K && Zsets[i][j] < Zc) {
				Zc = Zsets[i][j];
				setIdx = i + 1;
			}
		}
	}
	int N = Zc * ncwnodes;
	vector<int> loc;
	for (i = 0; i < K; i++) {
		if (in_data[i][0] == -1) {
			loc.push_back(i);  //将filler bit index 存入 loc中
			in_data[i][0] = 0; // 将filler bit value 设为0；
		}
	}
	// 在编码时，将fillerbit设置为0，并将其位置存储下来，编码之后又将其填充为-1，不进行传输；
	encode(in_data, K, outCball, Zc, bgn, setIdx);
	for (i = 0; i < loc.size(); i++) {
		outCball[loc[i]][0] = -1;
	}
	/*for (int i = 0; i < K; i++) {
	cout << outCball[i][0] << " ";
	cout << endl;
	}*/
	for (i = 0; i < N; i++) {
		for (j = 0; j < batchsize; j++) {
			out_data[i][j] = outCball[2 * Zc + i][j];  // 除去前面punctured 但是还有filler bit
		}
	}
}