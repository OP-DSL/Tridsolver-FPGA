#include <stdio.h>
#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
#include "data_types.h"
#include "stencils.hpp"
#include "DPath.hpp"
#include "TiledThomas.hpp"
#include "trsv.hpp"



static void TDMA(const uint512_dt* d, uint512_dt* u,
		const uint512_dt* acc1, uint512_dt* acc2,
		ap_uint<12> Xdim, ap_uint<12> Ydim, ap_uint<12> Batch, ap_uint<12> Batch_acc, unsigned char dirXYZ, bool dnt_acc_updt){


	static hls::stream<uint512_dt> data_512_stm[4];
	static hls::stream<uint256_dt> data_256_stm0[4];
	static hls::stream<uint256_dt> data_256_stm1[4];
	static hls::stream<uint256_dt> d_stm[4];
	static hls::stream<uint256_dt> u_stm[4];
	static hls::stream<uint256_dt> accStream[4];

	#pragma HLS STREAM variable = data_512_stm depth = 2
	#pragma HLS STREAM variable = data_256_stm0 depth = 2
	#pragma HLS STREAM variable = data_256_stm1 depth = 2
	#pragma HLS STREAM variable = d_stm depth = 2
	#pragma HLS STREAM variable = u_stm depth = 2
	#pragma HLS STREAM variable = accStream depth = 2




	static hls::stream<uint256_dt> STAGE_0[8];
	static hls::stream<uint256_dt> STAGE_1[8];
	static hls::stream<uint256_dt> STAGE_2A[8];
	static hls::stream<uint256_dt> STAGE_2B[8];
	static hls::stream<uint256_dt> STAGE_3[8];
	static hls::stream<uint256_dt> STAGE_4[8];
	static hls::stream<uint256_dt> STAGE_5[8];
	static hls::stream<uint256_dt> STAGE_6[8];
	static hls::stream<uint256_dt> STAGE_7[8];

	static hls::stream<float> STAGE_2B_scl[8];
	static hls::stream<float> STAGE_4_scl[8];
	static hls::stream<float> STAGE_6_scl[8];

	#pragma HLS STREAM variable = STAGE_0 depth = 2
	#pragma HLS STREAM variable = STAGE_1 depth = 2
	#pragma HLS STREAM variable = STAGE_2A depth = 1022
	#pragma HLS STREAM variable = STAGE_2B depth = 2
	#pragma HLS STREAM variable = STAGE_3 depth = 2
	#pragma HLS STREAM variable = STAGE_4 depth = 2
	#pragma HLS STREAM variable = STAGE_5 depth = 2
	#pragma HLS STREAM variable = STAGE_6 depth = 510
	#pragma HLS STREAM variable = STAGE_7 depth = 2

	#pragma HLS STREAM variable = STAGE_2B_scl depth = 2
	#pragma HLS STREAM variable = STAGE_4_scl depth = 2
	#pragma HLS STREAM variable = STAGE_6_scl depth = 2

	const int VEC_FACTOR = 8;
	const int D_SIZE = 256/VEC_FACTOR;


	hls::stream<float> pcr_a[VEC_FACTOR];
	hls::stream<float> pcr_b[VEC_FACTOR];
	hls::stream<float> pcr_c[VEC_FACTOR];
	hls::stream<float> pcr_d[VEC_FACTOR];

	hls::stream<float> stm_out[VEC_FACTOR];
	#pragma HLS STREAM variable = stm_out depth = 2

	#pragma HLS STREAM variable = pcr_a depth = 2
	#pragma HLS STREAM variable = pcr_b depth = 2
	#pragma HLS STREAM variable = pcr_c depth = 2
	#pragma HLS STREAM variable = pcr_d depth = 2


    struct data_G data_g;
    data_g.sizex = Xdim-2;
    data_g.sizey = Ydim-2;
    data_g.xdim0 = Xdim;
	data_g.end_index = (Xdim >> 3); // number of blocks with V number of elements to be processed in a single row
	data_g.end_row = Ydim; // includes the boundary
	data_g.outer_loop_limit = Ydim+1; // n + D/2
	data_g.gridsize = (data_g.end_row* Batch + 1) * data_g.end_index;
	data_g.endindex_minus1 = data_g.end_index -1;
	data_g.endrow_plus1 = data_g.end_row + 1;
	data_g.endrow_plus2 = data_g.end_row + 2;
	data_g.endrow_minus1 = data_g.end_row - 1;
	data_g.total_itr_256 = data_g.end_row * data_g.end_index * Batch;
	data_g.total_itr_512 = (data_g.end_row * data_g.end_index * Batch + 1) >> 1;


	unsigned short TileX, TileY;
	unsigned int ReadLimit;
	unsigned short XBlocks = (Xdim >> 3);
	unsigned short offset;
	const ap_uint<4> N_CU = 8;
	bool reorder = (dirXYZ == 0);

	switch(dirXYZ){
		case 0: {ReadLimit = (((Batch*Ydim+7)>>3)*XBlocks) << 3; break;}
		case 1: {ReadLimit = ((((XBlocks*Batch+7)>>3))*Ydim) << 3; break;}
		default: {ReadLimit = (((Batch*Ydim+7)>>3)*XBlocks) << 3; break;}
	}

	int B = (dirXYZ == 0 ? ((Batch*Ydim+127)>>7) << 4 : ((Xdim*Batch+127)>>7)<<4);
	unsigned short N = (dirXYZ == 0 ? Xdim : Ydim);
//	printf("TDMA_comp:Read Limit is %d\n", ReadLimit);



	// New Thomas-Thomas Solver
	ap_uint<4> logn;

	unsigned char Ti, Sys;
	unsigned char chk = (N >> 8);
	switch(chk){
		case 0: {Sys = 4; Ti = 8; logn = 4; break;}
		case 1: {Sys = 4; Ti = 8; logn = 4; break;}
		case 2: {Sys = 2; Ti = 16; logn = 5; break;}
		case 3: {Sys = 2; Ti = 16; logn = 5; break;}
		default: {Sys = 1; Ti = 32; logn = 6; break;}
	}


	const short Tiles = 32;
	unsigned char M = N/Ti;
	ap_uint<24> B_size = B /Sys;
	int Rn = (Ti << 1);
	const ap_uint<24> R_size = ((ap_uint<24>)B_size<< 6);
	const ap_uint<24> R_size_half = ((ap_uint<24>)B_size<< 5);
	const ap_uint<8> R_systems = (Sys << 3);



	bool skip_pre = dirXYZ == 1;
	unsigned int total_512_data = (data_g.end_row * data_g.end_index * Batch) >> 1;
	unsigned int total_512_data_acc = (dirXYZ == 1) ? 0 : total_512_data;

	#pragma HLS dataflow
	read_Tile512<0>(d, data_512_stm[0], Xdim, Ydim, Batch, dirXYZ);
	FIFO_512_256(data_512_stm[0], d_stm[0], total_512_data);
	read_dat<0>(acc1, accStream[0], Xdim, Ydim, Batch);
	stencil_2d<0, float, 1024>(d_stm[0], d_stm[1], accStream[0], accStream[1], data_g, dnt_acc_updt, skip_pre);
	write_dat<0>(acc2, accStream[1], Xdim, Ydim, Batch);
	interleaved_row_block8<0, 1024>(d_stm[1], d_stm[2], Xdim, Ydim, Batch, dirXYZ==0);
	stream_8x8transpose<0, float>(d_stm[2], d_stm[3], Xdim, Ydim, Batch, dirXYZ==0);



	//interleave the systems
	TT_Interleave<0, float, 4096>(d_stm[3], STAGE_0[0], STAGE_0[1], STAGE_0[2],  STAGE_0[3],
			  B_size, Tiles, M, Sys, Ti, ReadLimit);


	// FW sweep
	TT_ForwardSweep<0, float, 4096>(STAGE_0[0], STAGE_0[1], STAGE_0[2],  STAGE_0[3],
			 STAGE_1[0], STAGE_1[1], STAGE_1[2],
			 B_size, Tiles, M, Sys, Ti);


	// BW Sweep
	TT_BackwardSweep<0, float, 4096>(STAGE_1[0], STAGE_1[1], STAGE_1[2],
			STAGE_2A[0], STAGE_2A[1], STAGE_2A[2],
			STAGE_2B[3], STAGE_2B[4], STAGE_2B[5], STAGE_2B[6],
			B_size, Tiles, M, Sys, Ti);


	for(int i = 0; i < Rn*Sys*B_size; i++){
			#pragma HLS PIPELINE II=1
			#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
			uint256_dt tmp_a = STAGE_2B[3].read();
			uint256_dt tmp_b = STAGE_2B[4].read();
			uint256_dt tmp_c = STAGE_2B[5].read();
			uint256_dt tmp_d = STAGE_2B[6].read();

			for(int v = 0; v < VEC_FACTOR; v++){
				pcr_a[v] << uint2FP_ript<0, float>(tmp_a.range(D_SIZE*(v+1)-1,D_SIZE*v));
				pcr_b[v] << uint2FP_ript<0, float>(tmp_b.range(D_SIZE*(v+1)-1,D_SIZE*v));
				pcr_c[v] << uint2FP_ript<0, float>(tmp_c.range(D_SIZE*(v+1)-1,D_SIZE*v));
				pcr_d[v] << uint2FP_ript<0, float>(tmp_d.range(D_SIZE*(v+1)-1,D_SIZE*v));
			}
	}

	for(int v = 0; v < VEC_FACTOR; v++){
		#pragma HLS unroll factor=8
		xf::fintech::trsvCore <float, 512> (pcr_a[v], pcr_b[v], pcr_c[v], pcr_d[v], stm_out[v],  Rn, Sys, logn, B_size);
	}


	for(int itr =0; itr < Sys*Rn*B_size; itr++){
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
		#pragma HLS PIPELINE II=1
		uint256_dt tmp_d;
		for(int v = 0; v < VEC_FACTOR; v++){
			tmp_d.range(D_SIZE*(v+1)-1,D_SIZE*v) = FP2uint_ript(stm_out[v].read());
		}
		if((itr & 1) == 0){
			STAGE_6[0] << tmp_d;
		} else {
			STAGE_6[1] << tmp_d;
		}
	}

	// back substitution
	TT_BackSubstitution<0, float, 4096>(STAGE_2A[0], STAGE_2A[1], STAGE_2A[2],
			STAGE_6[0], STAGE_6[1], u_stm[0],
			B_size, Tiles, M, ReadLimit);
	stream_8x8transpose<0, float>(u_stm[0], u_stm[1], Xdim, Ydim, Batch, dirXYZ==0);
	undo_interleaved_row_block8<0, 1024>(u_stm[1], u_stm[2], Xdim, Ydim, Batch, dirXYZ==0);
	FIFO_256_512(u_stm[2], data_512_stm[3], total_512_data);
	write_Tile512<0>(u, data_512_stm[3], Xdim, Ydim, Batch, dirXYZ);

	printf("finished one iteration\n");


}










extern "C" {
void TDMA_batch(
	uint512_dt* d,
	uint512_dt* u,
	uint512_dt* acc_1,
	uint512_dt* acc_2,
	int M,
	int N,
	int B,
	int iters){

	#pragma HLS INTERFACE depth=4096 m_axi port = d offset = slave bundle = gmem3 max_read_burst_length=8 max_write_burst_length=8 num_read_outstanding=64 num_write_outstanding=64 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = u offset = slave bundle = gmem3 max_read_burst_length=8 max_write_burst_length=8 num_read_outstanding=64 num_write_outstanding=64 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = acc_1 offset = slave bundle = gmem4 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = acc_2 offset = slave bundle = gmem4 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8 latency=64

	#pragma HLS INTERFACE s_axilite port = d bundle = control
	#pragma HLS INTERFACE s_axilite port = u bundle = control
	#pragma HLS INTERFACE s_axilite port = acc_1 bundle = control
	#pragma HLS INTERFACE s_axilite port = acc_2 bundle = control

	#pragma HLS INTERFACE s_axilite port = N bundle = control
	#pragma HLS INTERFACE s_axilite port = M bundle = control
	#pragma HLS INTERFACE s_axilite port = B bundle = control
	#pragma HLS INTERFACE s_axilite port = iters bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control



	for(ap_uint<12> itr = 0; itr < iters; itr++){
		#pragma HLS loop_tripcount min=1 max=100 avg=10
		#pragma HLS dependence variable=d intra RAW true
		#pragma HLS dependence variable=u intra RAW true
		bool dnt_acc_updt = (itr == 0) ? 1 : 0;

		TDMA(u, d, acc_1, acc_2, M, N, B, B,  0, dnt_acc_updt);
		TDMA(d, u, acc_2, acc_1, M, N, B, 0,  1, 0);

		TDMA(u, d, acc_2, acc_1, M, N, B, B, 0, 0);
		TDMA(d, u, acc_1, acc_2, M, N, B, 0, 1, 0);
	}

}
}
