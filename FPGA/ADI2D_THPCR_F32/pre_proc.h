
#ifndef __PRE_PROC_H__
#define __PRE_PROC_H__

#define N_MAX 4096
#define DIM_MAX 1024
#define RN_MAX 512
#define RN_MAX_scl 1024
#define MAX_Sys 32
#define N_BLK 32
#define D_SIZE 32
#define VEC_FACTOR 8
#define MAX_DEPTH_16 DIM_MAX/16
#define SHIFT_BITS 3
const int n_blk = N_BLK;

const int max_size_y = N_MAX;
const int min_size_y = 32;
const int avg_size_y = N_MAX;

const int max_block_x = N_MAX/8 + 1;
const int min_block_x = 32/8 + 1;
const int avg_block_x = N_MAX/8 + 1;

const int max_grid = max_block_x * max_size_y * max_size_y;
const int min_grid = min_block_x * min_size_y * min_size_y;
const int avg_grid = avg_block_x * avg_size_y * avg_size_y;

const int max_grid_2 = (max_block_x * max_size_y * max_size_y)/2;
const int min_grid_2 = (min_block_x * min_size_y * min_size_y)/2;
const int avg_grid_2 = (avg_block_x * avg_size_y * avg_size_y)/2;


const int vec_factor  = VEC_FACTOR;
const int max_depth_16 = MAX_DEPTH_16;
const int max_depth_8 = MAX_DEPTH_16*2;

typedef ap_uint<512> uint512_dt;
typedef ap_uint<256> uint256_dt;
typedef ap_uint<32> uint32_dt;
typedef ap_uint<64> uint64_dt;
typedef short uint12_dt;


template <class T>
T register_it(T in){
	#pragma HLS inline off
	T x = in;
	return x;
}

typedef union  {
   int i;
   float f;
} data_conv;





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

static void read_u(const uint512_dt* u, hls::stream<uint256_dt> &u_stm,
		const uint512_dt* acc, hls::stream<uint256_dt> &acc_stm,
		ap_uint<12> M, ap_uint<12> N, ap_uint<12> B);

static void write_d(uint512_dt* d, hls::stream<uint256_dt> &d_stm,
		uint512_dt* acc, hls::stream<uint256_dt> &acc_stm,
		ap_uint<12> M, ap_uint<12> N, ap_uint<12> B);

static void process_grid( hls::stream<uint256_dt> &rd_buffer, hls::stream<uint256_dt> &wr_buffer,
		hls::stream<uint256_dt> &acc_in,  hls::stream<uint256_dt> &acc_out, struct data_G data_g, bool dnt_acc_updt, bool skip_process);

static int pre_process(const uint512_dt* u,  uint512_dt* d, const uint512_dt* acc_1, uint512_dt* acc_2,
				   int M, int N, int B, bool dnt_acc_updt);

#endif
