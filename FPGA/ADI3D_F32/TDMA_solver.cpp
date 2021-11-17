#include <stdio.h>
#include <ap_int.h>
#include <hls_stream.h>
#include "BThomas.hpp"
#include "data_types.h"
#include "stencils.hpp"
#include "DPath.hpp"

static void TDMA_pre_XY(const uint512_dt* d, uint512_dt* u,
		const uint512_dt* acc_1, uint512_dt* acc_2,
		ap_uint<12> M, ap_uint<12> N, ap_uint<12> L, ap_uint<12> B, bool dnt_update){

	static hls::stream<uint256_dt> d_stm[8];
	static hls::stream<uint256_dt> u_stm[8];

    static hls::stream<uint256_dt> streamArray[4];
    static hls::stream<uint256_dt> accStream[4];

    #pragma HLS STREAM variable = streamArray depth = 2
	#pragma HLS STREAM variable = accStream depth = 2

	#pragma HLS STREAM variable = d_stm depth = 2
	#pragma HLS STREAM variable = u_stm depth = 2



    struct data_G_3d data_g;
    data_g.sizex = M-2;
    data_g.sizey = N-2;
    data_g.sizez = L-2;


	data_g.xblocks = (M >> 3);
	data_g.grid_sizey = N;
	data_g.grid_sizez = L;
	data_g.limit_z = L+1;

	unsigned short tiley_1 = (N - 1);
	unsigned int plane_size = data_g.xblocks * N;

	data_g.plane_diff = data_g.xblocks * tiley_1;
	data_g.line_diff = data_g.xblocks - 1;
	data_g.gridsize_pr = plane_size * register_it<unsigned int>(data_g.grid_sizez * B+1);
	data_g.gridsize_da = plane_size * L * B;


	unsigned int t_x = (((N*L*B+255) >> 8));
	unsigned int t_y = (((L*M*B+255) >> 8));
	unsigned int t_z = (((M*N*B+255) >> 8));





	unsigned int Read_limit_X = ((M >> 3)*((N*L*B+7)>>3) << 3);
	unsigned int Read_limit_Y = (((M>>3)*N*L*B+7)>>3) << 3;


	hls::stream<uint256_dt> d_fw_stm[4];
	hls::stream<uint256_dt> c2_fw_stm[4];
	hls::stream<uint256_dt> d2_fw_stm[4];

	#pragma HLS STREAM variable = d_fw_stm depth = 2
	#pragma HLS STREAM variable = c2_fw_stm depth = 2
	#pragma HLS STREAM variable = d2_fw_stm depth = 2



	#pragma HLS dataflow
	read_dat<0>(d, streamArray[0], M, N, L* B);
	read_dat<0>(acc_1, accStream[0], M, N, L* B);
//	read_u(d, streamArray[0], acc_1, accStream[0],  M, N, L, B);
	printf("finished reading TDMA_pre_XY\n");
	stencil_3d<0, float, 128>(streamArray[0], d_stm[0], accStream[0], accStream[1], data_g, dnt_update);


	// Xdim computation
	interleaved_row_block8<0,128>(d_stm[0], d_stm[1], M, N, L*B, 1);
	stream_8x8transpose<0,float>(d_stm[1], d_stm[2], M, N, L*B, 1);
	thomas_interleave<0, float, 128>(d_stm[2], d_fw_stm[0], M, t_x, Read_limit_X);
	thomas_forward<0, float, 128>(d_fw_stm[0], c2_fw_stm[0], d2_fw_stm[0], M, t_x);
	thomas_backward<0, float, 128>(c2_fw_stm[0], d2_fw_stm[0], u_stm[0], M, t_x, Read_limit_X);
	stream_8x8transpose<0,float>(u_stm[0], u_stm[1], M, N, L*B, 1);
	undo_interleaved_row_block8<0,128>(u_stm[1], u_stm[2], M, N, L*B, 1);


	// Ydim computation
	row2col<0, 128>(u_stm[2], u_stm[3], M, N, L*B);
	thomas_interleave<0,float, 128>(u_stm[3], d_fw_stm[1], N, t_y, Read_limit_Y);
	thomas_forward<0, float, 128>(d_fw_stm[1], c2_fw_stm[1], d2_fw_stm[1], N, t_y);
	thomas_backward<0, float, 128>(c2_fw_stm[1], d2_fw_stm[1], u_stm[4], N, t_y, Read_limit_Y);
	col2row<0, 128>(u_stm[4], u_stm[5], M, N, L*B);


	write_dat<0>(u, u_stm[5], M, N, L*B);
	write_dat<0>(acc_2, accStream[1], M, N, L*B);


}

static void TDMA_Z(const uint512_dt* d, uint512_dt* u,
		ap_uint<12> M, ap_uint<12> N, ap_uint<12> L, ap_uint<12> B){

	static hls::stream<uint256_dt> d_stm[4];
	static hls::stream<uint256_dt> u_stm[4];

	static hls::stream<uint32_dt> d_Array[8];
	static hls::stream<uint32_dt> u_Array[8];

	#pragma HLS STREAM variable = d_stm depth = 2
	#pragma HLS STREAM variable = u_stm depth = 2

	#pragma HLS STREAM variable = d_Array depth = 2
	#pragma HLS STREAM variable = u_Array depth = 2

	hls::stream<uint256_dt> d_fw_stm[4];
	hls::stream<uint256_dt> c2_fw_stm[4];
	hls::stream<uint256_dt> d2_fw_stm[4];

	#pragma HLS STREAM variable = d_fw_stm depth = 2
	#pragma HLS STREAM variable = c2_fw_stm depth = 2
	#pragma HLS STREAM variable = d2_fw_stm depth = 2

	unsigned char dirXYZ = 2;

	unsigned int t_z = (((M*N*B+255) >> 8));
	unsigned int Read_limit_Z = (((M>>3)*N*L*B+7)>>3) << 3;

	#pragma HLS dataflow
	read_plane<0>(d, d_stm[0], M, N, L, B, dirXYZ);
//	printf("finished reading Z\n");

	row2col<0, 128>(d_stm[0], d_stm[1], M, L, N*B);
	thomas_interleave<0, float, 128>(d_stm[1], d_fw_stm[0], L, t_z, Read_limit_Z);
	thomas_forward<0, float, 128>(d_fw_stm[0], c2_fw_stm[0], d2_fw_stm[0], L, t_z);
	thomas_backward<0, float, 128>(c2_fw_stm[0], d2_fw_stm[0], u_stm[0], L, t_z, Read_limit_Z);
	col2row<0,128>(u_stm[0], u_stm[3], M, L, N*B);

	write_plane<0>(u, u_stm[3], M, N, L, B, dirXYZ);


}



extern "C" {
void TDMA_batch(
	uint512_dt* d_1,
	uint512_dt* u_1,
	uint512_dt* acc1_1,
	uint512_dt* acc2_1,

	uint512_dt* d_2,
	uint512_dt* u_2,
	uint512_dt* acc1_2,
	uint512_dt* acc2_2,

	int M,
	int N,
	int L,
	int B,
	int iters){

	#pragma HLS INTERFACE depth=4096 m_axi port = d_1 offset = slave bundle = gmem0 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=32 num_write_outstanding=32 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = u_1 offset = slave bundle = gmem0 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=32 num_write_outstanding=32 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = acc1_1 offset = slave bundle = gmem1 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=32 num_write_outstanding=32 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = acc2_1 offset = slave bundle = gmem1 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=32 num_write_outstanding=32 latency=64

	#pragma HLS INTERFACE depth=4096 m_axi port = d_2 offset = slave bundle = gmem2 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=32 num_write_outstanding=32 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = u_2 offset = slave bundle = gmem2 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=32 num_write_outstanding=32 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = acc1_2 offset = slave bundle = gmem3 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=32 num_write_outstanding=32 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = acc2_2 offset = slave bundle = gmem3 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=32 num_write_outstanding=32 latency=64

	#pragma HLS INTERFACE s_axilite port = d_1 bundle = control
	#pragma HLS INTERFACE s_axilite port = u_1 bundle = control
	#pragma HLS INTERFACE s_axilite port = acc1_1 bundle = control
	#pragma HLS INTERFACE s_axilite port = acc2_1 bundle = control

	#pragma HLS INTERFACE s_axilite port = d_2 bundle = control
	#pragma HLS INTERFACE s_axilite port = u_2 bundle = control
	#pragma HLS INTERFACE s_axilite port = acc1_2 bundle = control
	#pragma HLS INTERFACE s_axilite port = acc2_2 bundle = control

	#pragma HLS INTERFACE s_axilite port = N bundle = control
	#pragma HLS INTERFACE s_axilite port = M bundle = control
	#pragma HLS INTERFACE s_axilite port = L bundle = control
	#pragma HLS INTERFACE s_axilite port = iters bundle = control
	#pragma HLS INTERFACE s_axilite port = B bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control


	ap_uint<10> iters_10 = iters;
	for(ap_uint<10> itr = 0; itr < iters_10; itr++){
		#pragma HLS loop_tripcount min=1 max=100 avg=10
		#pragma HLS dependence variable=d_1 intra RAW true
		#pragma HLS dependence variable=u_1 intra RAW true
		#pragma HLS dependence variable=d_2 intra RAW true
		#pragma HLS dependence variable=u_2 intra RAW true

		// first stage
		bool dnt_update = (itr == 0);
		TDMA_pre_XY(d_1, u_1, acc1_1, acc2_1, M, N, L, B, dnt_update);
		TDMA_Z(u_2, d_2, M, N, L, B);

		// second stage
		TDMA_pre_XY(d_2, u_2, acc2_2, acc1_2, M, N, L, B, 0);
		TDMA_Z(u_1, d_1, M, N, L, B);

		// first stage
		TDMA_pre_XY(d_1, u_1, acc2_1, acc1_1, M, N, L, B, 0);
		TDMA_Z(u_2, d_2, M, N, L, B);

		// second stage
		TDMA_pre_XY(d_2, u_2, acc1_2, acc2_2, M, N, L, B, 0);
		TDMA_Z(u_1, d_1, M, N, L, B);
	}

}
}
