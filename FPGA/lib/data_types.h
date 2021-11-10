#include <ap_int.h>
#include <hls_stream.h>
#include <data_types.h>

#ifndef __DATA_TYPES_H__
#define __DATA_TYPES_H__


typedef ap_uint<512> uint512_dt;
typedef ap_uint<256> uint256_dt;
typedef ap_uint<64> uint64_dt;
typedef ap_uint<32> uint32_dt;


template <class T>
T register_it(T in){
	#pragma HLS inline off
	T x = in;
	return x;
}

typedef union  {
   unsigned long long i;
   double f;
} data_conv64;

typedef union  {
   unsigned int i;
   float f;
} data_conv32;


template<bool FPPrec =1, class Tout=double>
Tout uint2FP_ript(uint64_dt in){
	if(FPPrec == 1){
		data_conv64 tmp;
		tmp.i = in;
		return tmp.f;
	} else {
		data_conv32 tmp;
		tmp.i = in;
		return tmp.f;
	}
};


uint64_dt FP2uint_ript(double in){
	data_conv64 tmp;
	tmp.f = in;
	return tmp.i;
};

uint32_dt FP2uint_ript(float in){
	data_conv32 tmp;
	tmp.f = in;
	return tmp.i;
};



struct data_G{
	unsigned short sizex;
	unsigned short sizey;
	unsigned short xdim0;
	unsigned short end_index;
	unsigned short end_row;
	unsigned int gridsize;
    unsigned int total_itr_512;
    unsigned int total_itr_256;
	unsigned short outer_loop_limit;
	unsigned short endrow_plus2;
	unsigned short endrow_plus1;
	unsigned short endrow_minus1;
	unsigned short endindex_minus1;
};

#endif