#include <stdio.h>
#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
#include "BThomas.hpp"
#include "data_types.h"
#include "stencils.hpp"
#include "DPath.hpp"





static void TDMA1( const uint512_dt* d, uint512_dt* u,
		const uint512_dt* acc1, uint512_dt* acc2,
		int M, int N, int B, unsigned char dirXYZ, bool dnt_acc_updt,
		uint512_dt* buffer1, uint512_dt* buffer2,
		int count_w, int count_b, int count_r){

	static hls::stream<uint256_dt> d_stm_0[5];
	static hls::stream<uint256_dt> u_stm_0[5];

	static hls::stream<uint256_dt> d_stm_1[5];
	static hls::stream<uint256_dt> u_stm_1[5];

	static hls::stream<uint256_dt> d_stm_2[5];
	static hls::stream<uint256_dt> u_stm_2[5];

	static hls::stream<uint256_dt> acc_stm[8];
	static hls::stream<uint512_dt> acc_big[8];

	#pragma HLS STREAM variable = d_stm_0 depth = 2
	#pragma HLS STREAM variable = u_stm_0 depth = 2

	#pragma HLS STREAM variable = d_stm_1 depth = 2
	#pragma HLS STREAM variable = u_stm_1 depth = 2

	#pragma HLS STREAM variable = d_stm_2 depth = 2
	#pragma HLS STREAM variable = u_stm_2 depth = 2

	#pragma HLS STREAM variable = acc_stm depth = 2
	#pragma HLS STREAM variable = acc_big depth = 512

	hls::stream<uint256_dt> d_fw_stm[4];
	hls::stream<uint256_dt> c2_fw_stm[4];
	hls::stream<uint256_dt> d2_fw_stm[4];
	#pragma HLS STREAM variable = d_fw_stm depth = 2
	#pragma HLS STREAM variable = c2_fw_stm depth = 2
	#pragma HLS STREAM variable = d2_fw_stm depth = 2

    struct data_G data_g;
    data_g.sizex = M-2;
    data_g.sizey = N-2;
    data_g.xdim0 = M;
	data_g.end_index = (M >> 2);
	data_g.end_row = N;
	data_g.outer_loop_limit = N+1; // n + D/2
	data_g.gridsize = (data_g.end_row* B + 1) * data_g.end_index;
	data_g.endindex_minus1 = data_g.end_index -1;
	data_g.endrow_plus1 = data_g.end_row + 1;
	data_g.endrow_plus2 = data_g.end_row + 2;
	data_g.endrow_minus1 = data_g.end_row - 1;
	data_g.total_itr_256 = data_g.end_row * data_g.end_index * B;
	data_g.total_itr_512 = (data_g.end_row * data_g.end_index * B + 1) >> 1;

	ap_uint<12> TileX = (M >> 2);
	ap_uint<12> TileY = N;
	ap_uint<18> NTiles = B;

	int total_512 = (data_g.total_itr_256 >> 1);



	ap_uint<9> XBlocks = (M >> 2);
	int ReadLimitX = ((B*N+3)>>2)*(XBlocks << 2);
	int ReadLimitY = (((XBlocks*B+3)>>2))*(N << 2);
	int total_gX = ((B*N+255)>>8);
	int total_gY = ((M*B+255)>>8);




	#pragma HLS dataflow
	read_dat<1>(d, d_stm_0[0], M, N, B);
	read_dat<1>(acc1, acc_stm[0], M, N, B);

	printf("Read succesful\n");

	// iteration one
	stencil_2d<1,double, 128>(d_stm_0[0], d_stm_0[1], acc_stm[0], acc_stm[1], data_g, dnt_acc_updt);

	FIFO_256_512(acc_stm[1], acc_big[0], total_512);
	HBM_fifo(buffer1, buffer2, acc_big[0], acc_big[1], count_w, count_b, count_r);
	FIFO_512_256(acc_big[1], acc_stm[2], total_512);

	interleaved_row_block8<1, 128>(d_stm_0[1], d_stm_0[2], M, N, B, 1);
	stream_8x8transpose<1,double>(d_stm_0[2], d_stm_0[3], M, N, B, 1);
	thomas_interleave<1,double,128>(d_stm_0[3], d_fw_stm[0], M, total_gX, ReadLimitX);
	thomas_forward<1,double,128>(d_fw_stm[0], c2_fw_stm[0], d2_fw_stm[0], M, total_gX);
	thomas_backward<1,double,128>(c2_fw_stm[0], d2_fw_stm[0], u_stm_0[0], M, total_gX, ReadLimitX);
	stream_8x8transpose<1,double>(u_stm_0[0], u_stm_0[1], M, N, B, 1);
	undo_interleaved_row_block8<1, 128>(u_stm_0[1], u_stm_0[2], M, N, B, 1);

	row2col<1,128>(u_stm_0[2], d_stm_0[4], M, N, B);
	thomas_interleave<1,double, 128>(d_stm_0[4], d_fw_stm[1], M, total_gY, ReadLimitY);
	thomas_forward<1,double, 128>(d_fw_stm[1], c2_fw_stm[1], d2_fw_stm[1], M, total_gY);
	thomas_backward<1,double, 128>(c2_fw_stm[1], d2_fw_stm[1], u_stm_0[3], M, total_gY, ReadLimitY);
	col2row<1,128>(u_stm_0[3], u_stm_0[4], M, N, B);

	// iteration two

	stencil_2d<1,double, 128>(u_stm_0[4], d_stm_1[1], acc_stm[2], acc_stm[3], data_g, 0);


	interleaved_row_block8<1, 128>(d_stm_1[1], d_stm_1[2], M, N, B, 1);
	stream_8x8transpose<1,double>(d_stm_1[2], d_stm_1[3], M, N, B, 1);
	thomas_interleave<1,double, 128>(d_stm_1[3], d_fw_stm[2], M, total_gX, ReadLimitX);
	thomas_forward<1,double, 128>(d_fw_stm[2], c2_fw_stm[2], d2_fw_stm[2], M, total_gX);
	thomas_backward<1,double, 128>(c2_fw_stm[2], d2_fw_stm[2], u_stm_1[0], M, total_gX, ReadLimitX);
	stream_8x8transpose<1,double>(u_stm_1[0], u_stm_1[1], M, N, B, 1);
	undo_interleaved_row_block8<1, 128>(u_stm_1[1], u_stm_1[2], M, N, B, 1);

	row2col<1,128>(u_stm_1[2], d_stm_1[4], M, N, B);
	thomas_interleave<1,double, 128>(d_stm_1[4], d_fw_stm[3], M, total_gY, ReadLimitY);
	thomas_forward<1,double, 128>(d_fw_stm[3], c2_fw_stm[3], d2_fw_stm[3], M, total_gY);
	thomas_backward<1,double, 128>(c2_fw_stm[3], d2_fw_stm[3], u_stm_1[3], M, total_gY, ReadLimitY);
	col2row<1,128>(u_stm_1[3], u_stm_1[4], M, N, B);



	write_dat<1>(acc2, acc_stm[3], M, N, B);
	write_dat<1>(u, u_stm_1[4], M, N, B);



}


//[advanced]
//param=compiler.userPostSysLinkOverlayTcl=/ssd_1/kkvasan/vits_ws/adi_2d_unroll/adi_2duroll/src/postSysLink.tcl

extern "C" {
void TDMA_batch(
	uint512_dt* d,
	uint512_dt* u,
	uint512_dt* acc_1,
	uint512_dt* acc_2,
	uint512_dt* buffer1,
	uint512_dt* buffer2,
	int M,
	int N,
	int B,
	int iters,
	int count_w,
	int count_b,
	int count_r){

	#pragma HLS INTERFACE depth=4096 m_axi port = d offset = slave bundle = gmem3 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=4 num_write_outstanding=4 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = u offset = slave bundle = gmem3 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=4 num_write_outstanding=4 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = acc_1 offset = slave bundle = gmem4 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=4 num_write_outstanding=4 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = acc_2 offset = slave bundle = gmem4 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=4 num_write_outstanding=4 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = buffer1 offset = slave bundle = gmem5 max_read_burst_length=4 max_write_burst_length=4 num_read_outstanding=64 num_write_outstanding=64 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = buffer2 offset = slave bundle = gmem6 max_read_burst_length=4 max_write_burst_length=4 num_read_outstanding=64 num_write_outstanding=64 latency=64

	#pragma HLS INTERFACE s_axilite port = d bundle = control
	#pragma HLS INTERFACE s_axilite port = u bundle = control
	#pragma HLS INTERFACE s_axilite port = acc_1 bundle = control
	#pragma HLS INTERFACE s_axilite port = acc_2 bundle = control
	#pragma HLS INTERFACE s_axilite port = buffer1 bundle = control
	#pragma HLS INTERFACE s_axilite port = buffer2 bundle = control

	#pragma HLS INTERFACE s_axilite port = N bundle = control
	#pragma HLS INTERFACE s_axilite port = M bundle = control
	#pragma HLS INTERFACE s_axilite port = B bundle = control
	#pragma HLS INTERFACE s_axilite port = iters bundle = control
	#pragma HLS INTERFACE s_axilite port = count_w bundle = control
	#pragma HLS INTERFACE s_axilite port = count_b bundle = control
	#pragma HLS INTERFACE s_axilite port = count_r bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control


	ap_uint<8> iters_8 = iters;
	for(ap_uint<8> itr = 0; itr < iters_8; itr++){
		#pragma HLS loop_tripcount min=1 max=100 avg=10
		#pragma HLS dependence variable=d intra RAW true
		#pragma HLS dependence variable=u intra RAW true
		bool dnt_acc_updt = (itr == 0) ? 1 : 0;
		TDMA1( u, d, acc_1, acc_2,  M, N, B, 0, dnt_acc_updt, buffer1, buffer2, count_w, count_b, count_r);
		TDMA1( d, u, acc_2, acc_1,  M, N, B, 0, 0, buffer1, buffer2, count_w, count_b, count_r);

	}

}
}
