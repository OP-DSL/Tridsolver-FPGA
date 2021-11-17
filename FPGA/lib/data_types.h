#include <ap_int.h>
#include <hls_stream.h>

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
	unsigned short sizez;
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

struct data_G_3d{
	unsigned short sizex;
	unsigned short sizey;
	unsigned short sizez;
	unsigned short xdim;
	unsigned short xblocks;
	unsigned short grid_sizey;
	unsigned short grid_sizez;
	unsigned short limit_z;
	unsigned short offset_x;
	unsigned short tile_x;
	unsigned short offset_y;
	unsigned short tile_y;
	unsigned int plane_size;
	unsigned int gridsize_pr;
	unsigned int gridsize_da;
	unsigned int plane_diff;
	unsigned int line_diff;
	unsigned short outer_loop_limit;
	unsigned int total_itr;
	bool last_half;
	unsigned short batches;
};


// Trip count
const int max_size_y = 256;
const int min_size_y = 32;
const int avg_size_y = 256;

const int batch_s = 100;
const int max_block_x = (256/8 + 1);
const int min_block_x = (32/8 + 1);
const int avg_block_x = (256/8 + 1);

const int max_grid = max_block_x * max_size_y * max_size_y * batch_s;
const int min_grid = min_block_x * min_size_y * min_size_y * batch_s;
const int avg_grid = avg_block_x * avg_size_y * avg_size_y * batch_s;

const int max_grid_2 = (max_block_x * max_size_y * max_size_y)/2 * batch_s;
const int min_grid_2 = (min_block_x * min_size_y * min_size_y)/2 * batch_s;
const int avg_grid_2 = (avg_block_x * avg_size_y * avg_size_y)/2 * batch_s;

#endif
