/**
@file main.cpp
@author Daniel Bahr
@version 0.2.0.6
@copyright
@parblock
CRC++
Copyright (c) 2016, Daniel Bahr
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the name of CRC++ nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
@endparblock
*/

#include "crc.h"

#include <iomanip>  // Includes ::std::hex, ::std::setw
#include <iostream> // Includes ::std::cerr, ::std::endl
#include <string>   // Includes ::std::string
#include <fstream>


using namespace std;
#ifdef CRCPP_USE_NAMESPACE
using ::CRCPP::CRC;
#endif
#define BIT_1   0x01
#define BIT_2   0x02
#define BIT_3   0x04
#define BIT_4   0x08
#define BIT_5   0x10
#define BIT_6   0x20
#define BIT_7   0x40
#define BIT_8   0x80

const CRC::Table<crcpp_uint8, 1> crcTable_1(CRC::CRC_1());
const CRC::Table<crcpp_uint8, 2> crcTable_2(CRC::CRC_2());
const CRC::Table<crcpp_uint8, 3> crcTable_3(CRC::CRC_3());
const CRC::Table<crcpp_uint8, 4> crcTable_4(CRC::CRC_4_ITU());
const CRC::Table<crcpp_uint8, 5> crcTable_5(CRC::CRC_5_ITU());
const CRC::Table<crcpp_uint8, 6> crcTable_6(CRC::CRC_6_5G());
const CRC::Table<crcpp_uint8, 7> crcTable_7(CRC::CRC_7());
const CRC::Table<crcpp_uint8, 8> crcTable_8(CRC::CRC_8_WCDMA());
const CRC::Table<crcpp_uint16, 9> crcTable_9(CRC::CRC_9());
const CRC::Table<crcpp_uint16, 10> crcTable_10(CRC::CRC_10_CDMA2000());
const CRC::Table<crcpp_uint16, 11> crcTable_11(CRC::CRC_11_5G());
const CRC::Table<crcpp_uint16, 12> crcTable_12(CRC::CRC_12_CDMA2000());
const CRC::Table<crcpp_uint16, 13> crcTable_13(CRC::CRC_13_BBC());
const CRC::Table<crcpp_uint16, 14> crcTable_14(CRC::CRC_14());
const CRC::Table<crcpp_uint16, 15> crcTable_15(CRC::CRC_15());
const CRC::Table<crcpp_uint16, 16> crcTable_16(CRC::CRC_16_KERMIT());
const CRC::Table<crcpp_uint32, 17> crcTable_17(CRC::CRC_17_CAN());
const CRC::Table<crcpp_uint32, 18> crcTable_18(CRC::CRC_18());
const CRC::Table<crcpp_uint32, 19> crcTable_19(CRC::CRC_19());
const CRC::Table<crcpp_uint32, 20> crcTable_20(CRC::CRC_20());
const CRC::Table<crcpp_uint32, 21> crcTable_21(CRC::CRC_21_CAN());
const CRC::Table<crcpp_uint32, 22> crcTable_22(CRC::CRC_22());
const CRC::Table<crcpp_uint32, 23> crcTable_23(CRC::CRC_23());
const CRC::Table<crcpp_uint32, 24> crcTable_24(CRC::CRC_24());
const CRC::Table<crcpp_uint32, 32> crcTable_32(CRC::CRC_32_BZIP2());

ofstream tout("crc_check.txt");

void bit2byte(unsigned char* input, unsigned char* output, int Data_length, int Remain_num)
{
	int output_length = Data_length / 8;
	unsigned char current_byte;
	/* create the output vector */
	for (int i = 0; i < output_length; i++) {
		current_byte = input[i * 8];
		current_byte = current_byte + (input[i * 8 + 1] << 1);
		current_byte = current_byte + (input[i * 8 + 2] << 2);
		current_byte = current_byte + (input[i * 8 + 3] << 3);
		current_byte = current_byte + (input[i * 8 + 4] << 4);
		current_byte = current_byte + (input[i * 8 + 5] << 5);
		current_byte = current_byte + (input[i * 8 + 6] << 6);
		current_byte = current_byte + (input[i * 8 + 7] << 7);
		output[i] = current_byte;
	}
	if (Remain_num > 0) {
		output[output_length] = 0;
		for (int i = 0; i < Remain_num; i++)
		{
			output[output_length] += input[output_length * 8 + i] << i;
		}
	}
}

//byte2bit 
void byte2bit(unsigned char* input, unsigned char* output, int DataLength)
{
	int output_length = DataLength * 8;
	int output_index;

	/* create the output vector */
	for (int i = 0; i < DataLength; i++) {
		output_index = 8 * i;
		output[output_index] = input[i] & BIT_1;
		output[output_index + 1] = (input[i] & BIT_2) >> 1;
		output[output_index + 2] = (input[i] & BIT_3) >> 2;
		output[output_index + 3] = (input[i] & BIT_4) >> 3;
		output[output_index + 4] = (input[i] & BIT_5) >> 4;
		output[output_index + 5] = (input[i] & BIT_6) >> 5;
		output[output_index + 6] = (input[i] & BIT_7) >> 6;
		output[output_index + 7] = (input[i] & BIT_8) >> 7;
	}
}
void tx_append_crc(unsigned char* data, int input_data_length, int crc_length, int type)
{   // input_data_length is A; crc_length is L 
	//bit2byte
	int len = (input_data_length + 7) / 8;
	unsigned char* data_bytes = new unsigned char[len];
	bit2byte(data, data_bytes, input_data_length, input_data_length % 8);

	//crc_byte
	switch (crc_length)
	{
	case(1):
	{
		uint8_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_1());
		unsigned char* crc_bytes = new unsigned char[1];
		crc_bytes[0] = crc;
		unsigned char* crc_append = new unsigned char[1];
		byte2bit(crc_bytes, crc_append, 1);

		//Append crc
		for (int j = 0; j < 1; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(2):
	{
		uint8_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_2());
		unsigned char* crc_bytes = new unsigned char[1];
		crc_bytes[0] = crc;
		unsigned char* crc_append = new unsigned char[2];
		byte2bit(crc_bytes, crc_append, 1);

		//Append crc
		for (int j = 0; j < 2; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(3):
	{
		uint8_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_3());
		unsigned char* crc_bytes = new unsigned char[1];
		crc_bytes[0] = crc;
		unsigned char* crc_append = new unsigned char[3];
		byte2bit(crc_bytes, crc_append, 1);

		//Append crc
		for (int j = 0; j < 3; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(4):
	{
		uint8_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_4_ITU());
		unsigned char* crc_bytes = new unsigned char[1];
		crc_bytes[0] = crc;
		unsigned char* crc_append = new unsigned char[4];
		byte2bit(crc_bytes, crc_append, 1);

		//Append crc
		for (int j = 0; j < 4; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(5):
	{
		uint8_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_5_ITU());
		unsigned char* crc_bytes = new unsigned char[1];
		crc_bytes[0] = crc;
		unsigned char* crc_append = new unsigned char[5];
		byte2bit(crc_bytes, crc_append, 1);

		//Append crc
		for (int j = 0; j < 5; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}

		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(6):
	{
		uint8_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_6_5G());
		unsigned char* crc_bytes = new unsigned char[1];
		crc_bytes[0] = crc;
		unsigned char* crc_append = new unsigned char[6];
		byte2bit(crc_bytes, crc_append, 1);

		//Append crc
		for (int j = 0; j < 6; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(7):
	{
		uint8_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_7());
		unsigned char* crc_bytes = new unsigned char[1];
		crc_bytes[0] = crc;
		unsigned char* crc_append = new unsigned char[7];
		byte2bit(crc_bytes, crc_append, 1);

		//Append crc
		for (int j = 0; j < 7; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(8):
	{
		uint8_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_8_WCDMA());
		unsigned char* crc_bytes = new unsigned char[1];
		crc_bytes[0] = crc;
		unsigned char* crc_append = new unsigned char[8];
		byte2bit(crc_bytes, crc_append, 1);

		//Append crc
		for (int j = 0; j < 8; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(9):
	{
		uint16_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_9());
		unsigned char* crc_bytes = new unsigned char[2];
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[9];
		byte2bit(crc_bytes, crc_append, 2);

		//Append crc
		for (int j = 0; j < 9; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(10):
	{
		uint16_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_10_CDMA2000());
		unsigned char* crc_bytes = new unsigned char[2];
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[10];
		byte2bit(crc_bytes, crc_append, 2);

		//Append crc
		for (int j = 0; j < 10; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(11):
	{
		uint16_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_11_5G());
		unsigned char* crc_bytes = new unsigned char[2];
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[11];
		byte2bit(crc_bytes, crc_append, 2);

		//Append crc
		for (int j = 0; j < 11; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(12):
	{
		uint16_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_12_CDMA2000());
		unsigned char* crc_bytes = new unsigned char[2];
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[12];
		byte2bit(crc_bytes, crc_append, 2);

		//Append crc
		for (int j = 0; j < 12; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(13):
	{
		uint16_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_13_BBC());
		unsigned char* crc_bytes = new unsigned char[2];
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[13];
		byte2bit(crc_bytes, crc_append, 2);

		//Append crc
		for (int j = 0; j < 13; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(14):
	{
		uint16_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_14());
		unsigned char* crc_bytes = new unsigned char[2];
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[14];
		byte2bit(crc_bytes, crc_append, 2);

		//Append crc
		for (int j = 0; j < 14; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(15):
	{
		uint16_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_15());
		unsigned char* crc_bytes = new unsigned char[2];
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[15];
		byte2bit(crc_bytes, crc_append, 2);

		//Append crc
		for (int j = 0; j < 15; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(16):
	{
		//uint16_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_16_ARC());
		uint16_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_16_KERMIT());
		unsigned char* crc_bytes = new unsigned char[2];
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[16];
		byte2bit(crc_bytes, crc_append, 2);

		//Append crc
		for (int j = 0; j < 16; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(17):
	{
		uint32_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_17_CAN());
		unsigned char* crc_bytes = new unsigned char[3];
		crc_bytes[2] = (unsigned char)(crc >> 16);
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[17];
		byte2bit(crc_bytes, crc_append, 3);

		//Append crc
		for (int j = 0; j < 17; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(18):
	{
		uint32_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_18());
		unsigned char* crc_bytes = new unsigned char[3];
		crc_bytes[2] = (unsigned char)(crc >> 16);
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[18];
		byte2bit(crc_bytes, crc_append, 3);

		//Append crc
		for (int j = 0; j < 18; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(19):
	{
		uint32_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_19());
		unsigned char* crc_bytes = new unsigned char[3];
		crc_bytes[2] = (unsigned char)(crc >> 16);
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[19];
		byte2bit(crc_bytes, crc_append, 3);

		//Append crc
		for (int j = 0; j < 19; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(20):
	{
		uint32_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_20());
		unsigned char* crc_bytes = new unsigned char[3];
		crc_bytes[2] = (unsigned char)(crc >> 16);
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[20];
		byte2bit(crc_bytes, crc_append, 3);

		//Append crc
		for (int j = 0; j < 20; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(21):
	{
		uint32_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_21_CAN());
		unsigned char* crc_bytes = new unsigned char[3];
		crc_bytes[2] = (unsigned char)(crc >> 16);
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[21];
		byte2bit(crc_bytes, crc_append, 3);

		//Append crc
		for (int j = 0; j < 21; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(22):
	{
		uint32_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_22());
		unsigned char* crc_bytes = new unsigned char[3];
		crc_bytes[2] = (unsigned char)(crc >> 16);
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[22];
		byte2bit(crc_bytes, crc_append, 3);

		//Append crc
		for (int j = 0; j < 22; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(23):
	{
		uint32_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_23());
		unsigned char* crc_bytes = new unsigned char[3];
		crc_bytes[2] = (unsigned char)(crc >> 16);
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[23];
		byte2bit(crc_bytes, crc_append, 3);

		//Append crc
		for (int j = 0; j < 23; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(24):
	{
		//uint32_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_24_5G());
		uint32_t crc;
		if (type == 1)
		{
			crc = CRC::Calculate(data_bytes, len, CRC::CRC_24());
		}
		else crc = CRC::Calculate(data_bytes, len, CRC::CRC_24B());
		//uint32_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_24());
		unsigned char* crc_bytes = new unsigned char[3];
		crc_bytes[2] = (unsigned char)(crc >> 16);
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[24];
		byte2bit(crc_bytes, crc_append, 3);

		//Append crc
		for (int j = 0; j < 24; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	case(32):
	{
		uint32_t crc = CRC::Calculate(data_bytes, len, CRC::CRC_32_BZIP2());
		unsigned char* crc_bytes = new unsigned char[4];
		crc_bytes[3] = (unsigned char)(crc >> 24);
		crc_bytes[2] = (unsigned char)(crc >> 16);
		crc_bytes[1] = (unsigned char)(crc >> 8);
		crc_bytes[0] = (unsigned char)crc;
		unsigned char* crc_append = new unsigned char[32];
		byte2bit(crc_bytes, crc_append, 4);

		//Append crc
		for (int j = 0; j < 32; j++)
		{
			data[j + input_data_length] = crc_append[j];
		}
		delete(crc_bytes);
		delete(crc_append);
		delete(data_bytes);
		break;
	}
	default: {cout << "crc = " << crc_length << " Not defined." << endl; break; }
	}
}
bool rx_check_crc(unsigned char* data, int input_data_length, int crc_length)
{
	int len = (input_data_length - crc_length + 7) / 8;
	unsigned char* data_bytes = new unsigned char[len];

	bit2byte(data, data_bytes, input_data_length - crc_length, (input_data_length - crc_length) % 8);
	bool crc_correct;
	switch (crc_length)
	{
	case(1):
	{
		unsigned char* tx_crc_bytes = new unsigned char[1];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 1, 1);
		uint8_t crc = CRC::Calculate(data_bytes, len, crcTable_1);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc);
		delete(tx_crc_bytes);
		break;
	}
	case(2):
	{
		unsigned char* tx_crc_bytes = new unsigned char[1];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 2, 2);
		uint8_t crc = CRC::Calculate(data_bytes, len, crcTable_2);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc);
		delete(tx_crc_bytes);
		break;
	}
	case(3):
	{
		unsigned char* tx_crc_bytes = new unsigned char[1];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 3, 3);
		uint8_t crc = CRC::Calculate(data_bytes, len, crcTable_3);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc);
		delete(tx_crc_bytes);
		break;
	}
	case(4):
	{
		unsigned char* tx_crc_bytes = new unsigned char[1];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 4, 4);
		uint8_t crc = CRC::Calculate(data_bytes, len, crcTable_4);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc);
		delete(tx_crc_bytes);
		break;
	}
	case(5):
	{
		unsigned char* tx_crc_bytes = new unsigned char[1];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 5, 5);
		uint8_t crc = CRC::Calculate(data_bytes, len, crcTable_5);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc);
		delete(tx_crc_bytes);
		break;
	}
	case(6):
	{
		unsigned char* tx_crc_bytes = new unsigned char[1];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 6, 6);
		uint8_t crc = CRC::Calculate(data_bytes, len, crcTable_6);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc);
		delete(tx_crc_bytes);
		break;
	}
	case(7):
	{
		unsigned char* tx_crc_bytes = new unsigned char[1];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 7, 7);
		uint8_t crc = CRC::Calculate(data_bytes, len, crcTable_7);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc);
		delete(tx_crc_bytes);
		break;
	}
	case(8):
	{
		unsigned char* tx_crc_bytes = new unsigned char[1];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 8, 0);
		uint8_t crc = CRC::Calculate(data_bytes, len, crcTable_8);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc);
		delete(tx_crc_bytes);
		break;
	}
	case(9):
	{
		unsigned char* tx_crc_bytes = new unsigned char[2];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 9, 1);
		uint16_t crc = CRC::Calculate(data_bytes, len, crcTable_9);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8));
		delete(tx_crc_bytes);
		break;
	}
	case(10):
	{
		unsigned char* tx_crc_bytes = new unsigned char[2];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 10, 2);
		uint16_t crc = CRC::Calculate(data_bytes, len, crcTable_10);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8));
		delete(tx_crc_bytes);
		break;
	}
	case(11):
	{
		unsigned char* tx_crc_bytes = new unsigned char[2];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 11, 3);
		uint16_t crc = CRC::Calculate(data_bytes, len, crcTable_11);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8));
		delete(tx_crc_bytes);
		break;
	}
	case(12):
	{
		unsigned char* tx_crc_bytes = new unsigned char[2];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 12, 4);
		uint16_t crc = CRC::Calculate(data_bytes, len, crcTable_12);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8));
		delete(tx_crc_bytes);
		break;
	}
	case(13):
	{
		unsigned char* tx_crc_bytes = new unsigned char[2];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 13, 5);
		uint16_t crc = CRC::Calculate(data_bytes, len, crcTable_13);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8));
		delete(tx_crc_bytes);
		break;
	}
	case(14):
	{
		unsigned char* tx_crc_bytes = new unsigned char[2];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 14, 6);
		uint16_t crc = CRC::Calculate(data_bytes, len, crcTable_14);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8));
		delete(tx_crc_bytes);
		break;
	}
	case(15):
	{
		unsigned char* tx_crc_bytes = new unsigned char[2];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 15, 7);
		uint16_t crc = CRC::Calculate(data_bytes, len, crcTable_15);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8));
		delete(tx_crc_bytes);
		break;
	}
	case(16):
	{
		unsigned char* tx_crc_bytes = new unsigned char[2];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 16, 0);
		uint16_t crc = CRC::Calculate(data_bytes, len, crcTable_16);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8));
		delete(tx_crc_bytes);
		break;
	}
	case(17):
	{
		unsigned char* tx_crc_bytes = new unsigned char[3];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 17, 1);
		uint32_t crc = CRC::Calculate(data_bytes, len, crcTable_17);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8)) && (tx_crc_bytes[2] == (unsigned char)(crc >> 16));
		delete(tx_crc_bytes);
		break;
	}
	case(18):
	{
		unsigned char* tx_crc_bytes = new unsigned char[3];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 18, 2);
		uint32_t crc = CRC::Calculate(data_bytes, len, crcTable_18);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8)) && (tx_crc_bytes[2] == (unsigned char)(crc >> 16));
		delete(tx_crc_bytes);
		break;
	}
	case(19):
	{
		unsigned char* tx_crc_bytes = new unsigned char[3];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 19, 3);
		uint32_t crc = CRC::Calculate(data_bytes, len, crcTable_19);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8)) && (tx_crc_bytes[2] == (unsigned char)(crc >> 16));
		delete(tx_crc_bytes);
		break;
	}
	case(20):
	{
		unsigned char* tx_crc_bytes = new unsigned char[3];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 20, 4);
		uint32_t crc = CRC::Calculate(data_bytes, len, crcTable_20);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8)) && (tx_crc_bytes[2] == (unsigned char)(crc >> 16));
		delete(tx_crc_bytes);
		break;
	}
	case(21):
	{
		unsigned char* tx_crc_bytes = new unsigned char[3];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 21, 5);
		uint32_t crc = CRC::Calculate(data_bytes, len, crcTable_21);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8)) && (tx_crc_bytes[2] == (unsigned char)(crc >> 16));
		delete(tx_crc_bytes);
		break;
	}
	case(22):
	{
		unsigned char* tx_crc_bytes = new unsigned char[3];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 22, 6);
		uint32_t crc = CRC::Calculate(data_bytes, len, crcTable_22);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8)) && (tx_crc_bytes[2] == (unsigned char)(crc >> 16));
		delete(tx_crc_bytes);
		break;
	}
	case(23):
	{
		unsigned char* tx_crc_bytes = new unsigned char[3];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 23, 7);
		uint32_t crc = CRC::Calculate(data_bytes, len, crcTable_23);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8)) && (tx_crc_bytes[2] == (unsigned char)(crc >> 16));
		delete(tx_crc_bytes);
		break;
	}
	case(24):
	{
		unsigned char* tx_crc_bytes = new unsigned char[3];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 24, 0);
		uint32_t crc = CRC::Calculate(data_bytes, len, crcTable_24);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8)) && (tx_crc_bytes[2] == (unsigned char)(crc >> 16));
		delete[] tx_crc_bytes;
		break;
	}
	case(32):
	{
		unsigned char* tx_crc_bytes = new unsigned char[4];

		bit2byte(data + input_data_length - crc_length, tx_crc_bytes, 32, 0);
		uint32_t crc = CRC::Calculate(data_bytes, len, crcTable_32);

		//check crc
		crc_correct = (tx_crc_bytes[0] == (unsigned char)crc) && (tx_crc_bytes[1] == (unsigned char)(crc >> 8)) && (tx_crc_bytes[2] == (unsigned char)(crc >> 16)) && (tx_crc_bytes[3] == (unsigned char)(crc >> 24));
		delete(tx_crc_bytes);
		break;
	}
	default: {cout << " Not defined." << endl; break; }
	}
	return crc_correct;
}