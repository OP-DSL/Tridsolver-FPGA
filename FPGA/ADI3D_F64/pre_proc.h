
#ifndef __PRE_PROC_H__
#define __PRE_PROC_H__

#define N_MAX 128
#define N_BLK 64
#define D_SIZE 64
#define VEC_FACTOR 4
#define DEPTH_P N_MAX*N_MAX/4
#define DEPTH_L 2*N_MAX/4
const int n_blk = N_BLK;

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


typedef ap_uint<512> uint512_dt;
typedef ap_uint<256> uint256_dt;
typedef ap_uint<32> uint32_dt;
typedef ap_uint<64> uint64_dt;


template <class T>
T register_it(T in){
	#pragma HLS inline off
	T x = in;
	return x;
}


typedef union  {
   unsigned long long i;
   double f;
} data_conv;




struct data_G{
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

static void read_u(const uint512_dt* u, hls::stream<uint512_dt> &u_stm,
		const uint512_dt* acc, hls::stream<uint512_dt> &acc_stm,
		ap_uint<12> M, ap_uint<12> N, ap_uint<12> L, ap_uint<12> B);

static void write_d(uint512_dt* d, hls::stream<uint512_dt> &d_stm,
		uint512_dt* acc, hls::stream<uint512_dt> &acc_stm,
		ap_uint<12> M, ap_uint<12> N, ap_uint<12> L, ap_uint<12> B);

static void process_tile( hls::stream<uint256_dt> &rd_buffer,  hls::stream<uint256_dt> &d,
		hls::stream<uint256_dt> &acc_in,  hls::stream<uint256_dt> &acc_out, struct data_G data_g, bool dnt_update);

//static int pre_process(const uint512_dt* u, uint512_dt* d,
//		const uint512_dt* acc_1, uint512_dt* acc_2,
//		ap_uint<12> M, ap_uint<12> N, ap_uint<12> L, ap_uint<12> B, bool dnt_update);

#endif
