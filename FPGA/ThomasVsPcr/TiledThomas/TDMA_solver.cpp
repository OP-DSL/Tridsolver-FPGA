#include <stdio.h>
#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
#include "pre_proc.cpp"
#include "trsv.hpp"



static float uint2float_ript(uint32_dt in){
	data_conv tmp;
	tmp.i = in;
	return tmp.f;
}

static uint32_dt float2uint_ript(float in){
	data_conv tmp;
	tmp.f = in;
	return tmp.i;
}


// TDMA Modules

static void read_coeff(
		const uint512_dt*a, hls::stream<uint256_dt> &a_stm,
		const uint512_dt*b, hls::stream<uint256_dt> &b_stm,
		const uint512_dt*c, hls::stream<uint256_dt> &c_stm,
		const uint512_dt*d, hls::stream<uint256_dt> &d_stm,
		ap_uint<12> M, ap_uint<12> N, ap_uint<14> B){

	ap_uint<8> XBlocks = (M >> 4);
	int total_itr = XBlocks * N * B;
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS loop_tripcount min=102400 max=204800 avg=204800
		#pragma HLS PIPELINE II=2
		uint512_dt tmp_a =  a[itr];
		a_stm << tmp_a.range(255,0);
		a_stm << tmp_a.range(511,256);

		uint512_dt tmp_b =  b[itr];
		b_stm << tmp_b.range(255,0);
		b_stm << tmp_b.range(511,256);

		uint512_dt tmp_c =  c[itr];
		c_stm << tmp_c.range(255,0);
		c_stm << tmp_c.range(511,256);

		uint512_dt tmp_d =  d[itr];
		d_stm << tmp_d.range(255,0);
		d_stm << tmp_d.range(511,256);
	}
//	printf("read_coeff: i didn't get stuck\n");

}


static void FIFO_256_512(hls::stream<uint256_dt> &stm_in, hls::stream<uint512_dt> &stm_out, int total_512){
	for(int i = 0; i < total_512; i++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=102400 max=204800 avg=204800
		uint512_dt tmp_w;
		tmp_w.range(255,0) = stm_in.read();
		tmp_w.range(511,256) = stm_in.read();
		stm_out << tmp_w;
	}
//	printf("FIFO_256_512: i didn't get stuck\n");
}

static void FIFO_512_256(hls::stream<uint512_dt> &stm_in, hls::stream<uint256_dt> &stm_out, int total_512){
	for(int i = 0; i < total_512; i++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=102400 max=204800 avg=204800
		uint512_dt tmp_w = stm_in.read();
		stm_out << tmp_w.range(255,0);
		stm_out << tmp_w.range(511,256);
	}
//	printf("FIFO_512_256: i didn't get stuck\n");
}



static void interleaved_row_block8(hls::stream<uint256_dt> &stm_in, hls::stream<uint256_dt> &stm_out,
		ap_uint<12> M, ap_uint<12> N, ap_uint<14> B, unsigned char dirXYZ){

	ap_uint<12> TileX, TileY;
	ap_uint<20> NTiles;
	ap_uint<8> XBlocks = (M >> 3);
	unsigned int offset;
	const int N_CU = 8;
	switch(dirXYZ){
		case 0: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+7)>>3; break;}
		case 1: {TileX=8; TileY=N; NTiles = (((XBlocks*B+7)>>3)); break;}
		default: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+7)>>3; break;}
	}

	uint256_dt tmp_M[DIM_MAX*N_CU*2];
	#pragma HLS RESOURCE variable=tmp_M core=XPM_MEMORY latency=2

	ap_uint<18> NTilesp1 = register_it<int>(NTiles+1);
	ap_uint<18> id = 0;
	ap_uint<12> jd =0, kd = 0;
	int total_itr = NTilesp1*TileX*TileY;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<18> i = id;
		ap_uint<12> j = jd, k = kd;

		if(k == TileX -1){
			kd = 0;
		} else {
			kd++;
		}

		if(k == TileX -1 && j == TileY -1){
			jd = 0;
			id++;
		} else if(k == TileX -1){
			jd++;
		}
		unsigned int offsetR = ((i & 1) == 0) ?  DIM_MAX*N_CU : 0;
		unsigned int offsetW = ((i & 1) == 0) ?  0 : DIM_MAX*N_CU;

		bool cmpW = dirXYZ == 1 || (i*TileY + j < B*N);
		int indW = k*TileY+j + offsetW;
		uint256_dt tmpW = 0;
		if(cmpW){
			tmpW = stm_in.read();
		}
		tmp_M[indW] = tmpW;

		int indR = j*TileX + k + offsetR;
		uint256_dt tmpR = tmp_M[indR];
		if(i > 0){
			stm_out << tmpR;
		}
	}
//	printf("interleaved_row_block8: i didn't get stuck\n");
}


static void undo_interleaved_row_block8(hls::stream<uint256_dt> &stm_in, hls::stream<uint256_dt> &stm_out,
		ap_uint<12> M, ap_uint<12> N, ap_uint<14> B, unsigned char dirXYZ){


	ap_uint<12> TileX, TileY;
	ap_uint<20> NTiles;
	ap_uint<8> XBlocks = (M >> 3);
	unsigned int offset;
	const int N_CU = 8;
	switch(dirXYZ){
		case 0: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+7)>>3; break;}
		case 1: {TileX=8; TileY=N; NTiles = (((XBlocks*B+7)>>3)); break;}
		default: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+7)>>3; break;}
	}
	uint256_dt tmp_M[DIM_MAX*N_CU*2];
	#pragma HLS RESOURCE variable=tmp_M core=XPM_MEMORY latency=2
	ap_uint<18> NTilesp1 = register_it<int>(NTiles+1);

	ap_uint<18> id = 0;
	ap_uint<12> jd =0, kd = 0;
	int total_itr = NTilesp1*TileX*TileY;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<18> i = id;
		ap_uint<12> j = jd, k = kd;

		if(k == TileX -1){
			kd = 0;
		} else {
			kd++;
		}

		if(k == TileX -1 && j == TileY -1){
			jd = 0;
			id++;
		} else if(k == TileX -1){
			jd++;
		}
		unsigned int offsetR = ((i & 1) == 0) ?  DIM_MAX*N_CU : 0;
		unsigned int offsetW = ((i & 1) == 0) ?  0 : DIM_MAX*N_CU;

		bool cmpW = dirXYZ == 1 || (i*TileY + j < B*N);
		int indW = j*TileX + k + offsetW;
		uint256_dt tmpW = 0;
		if(cmpW){
			tmpW = stm_in.read();
		}
		tmp_M[indW] = tmpW;

		int indR = k*TileY+j + offsetR;
		uint256_dt tmpR = tmp_M[indR];
		if(i > 0){
			stm_out << tmpR;
		}
	}
//	printf("undo_interleaved_row_block8: i didn't get stuck\n");
}




static void stream_8x8transpose(hls::stream<uint256_dt> &in, hls::stream<uint256_dt> &out0,
		ap_uint<12> M, ap_uint<12> N, ap_uint<14> B, unsigned char dirXYZ){

	ap_uint<12> TileX, TileY;
	ap_uint<32> NTiles;
	ap_uint<9> XBlocks = (M >> 3);
	const int N_CU = 8;
	bool reorder = (dirXYZ == 0);

	switch(dirXYZ){
		case 0: {TileX=XBlocks; TileY=N_CU; NTiles = ((B*N+7)>>3)*XBlocks; break;}
		case 1: {TileX=8; TileY=N; NTiles = (((XBlocks*B+7)>>3))*N; break;}
		default: {TileX=XBlocks; TileY=N_CU; NTiles = ((B*N+7)>>3)*XBlocks; break;}
	}


	loop_read: for(int itr=0; itr < NTiles; itr++){
		#pragma HLS loop_tripcount min=25600 max=204800 avg=204800
		#pragma HLS PIPELINE II=8
		uint256_dt tmp[8], outR;
		for(int i = 0; i < 8; i++){
			tmp[i] = in.read();

		}

		if(reorder){
			for(int i = 0; i < 8; i++){
				outR.range(D_SIZE*1-1, D_SIZE*0) = tmp[0].range(D_SIZE*(i+1)-1,D_SIZE*i);
				outR.range(D_SIZE*2-1, D_SIZE*1) = tmp[1].range(D_SIZE*(i+1)-1,D_SIZE*i);
				outR.range(D_SIZE*3-1, D_SIZE*2) = tmp[2].range(D_SIZE*(i+1)-1,D_SIZE*i);
				outR.range(D_SIZE*4-1, D_SIZE*3) = tmp[3].range(D_SIZE*(i+1)-1,D_SIZE*i);
				outR.range(D_SIZE*5-1, D_SIZE*4) = tmp[4].range(D_SIZE*(i+1)-1,D_SIZE*i);
				outR.range(D_SIZE*6-1, D_SIZE*5) = tmp[5].range(D_SIZE*(i+1)-1,D_SIZE*i);
				outR.range(D_SIZE*7-1, D_SIZE*6) = tmp[6].range(D_SIZE*(i+1)-1,D_SIZE*i);
				outR.range(D_SIZE*8-1, D_SIZE*7) = tmp[7].range(D_SIZE*(i+1)-1,D_SIZE*i);

				out0 << outR;
			}
		}else {
			for(int i = 0; i < 8; i++){
					out0 << tmp[i];
			}
		}

	}
//	printf("stream_8x8transpose: i didn't get stuck\n");
}



static void PCR_solver(hls::stream<uint256_dt> &a_fw_stm, hls::stream<uint256_dt> &b_fw_stm,
		hls::stream<uint256_dt> &c_fw_stm, hls::stream<uint256_dt> &d_fw_stm,
		hls::stream<uint256_dt> &out_stm,
		ap_uint<12> d0, ap_uint<20> B){


	const int N=1024;
	const int vec_factor= 8;

	loop_fw: for(ap_uint<20> itr= 0; itr < B; itr++){
		float inlow[vec_factor][N], indiag[vec_factor][N], inup[vec_factor][N], inrhs[vec_factor][N];
		#pragma HLS array_partition variable=inlow block  factor=vec_factor dim=1
		#pragma HLS array_partition variable=indiag block  factor=vec_factor dim=1
		#pragma HLS array_partition variable=inup block  factor=vec_factor dim=1
		#pragma HLS array_partition variable=inrhs block  factor=vec_factor dim=1

		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
//		#pragma HLS dataflow
		loop_read:for(ap_uint<12> i = 0; i < N; i++){
			#pragma HLS pipeline ii=1
			uint256_dt vec_a = 0, vec_b = 0, vec_c = 0, vec_d = 0;
//			printf("itr:%d i=%d, d0:%d\n", (int)itr, (int)i, (int)d0);
//			if(itr*d0+i < ReadLimit){
				vec_a = a_fw_stm.read();
				vec_b = b_fw_stm.read();
				vec_c = c_fw_stm.read();
				vec_d = d_fw_stm.read();
//			}

			for(int v= 0; v < 8; v++){

				inlow[v][i] = uint2float_ript(vec_a.range(D_SIZE*(v+1)-1,D_SIZE*v));
				indiag[v][i] = uint2float_ript(vec_b.range(D_SIZE*(v+1)-1,D_SIZE*v));
				inup[v][i] = uint2float_ript(vec_c.range(D_SIZE*(v+1)-1,D_SIZE*v));
				inrhs[v][i] = uint2float_ript(vec_d.range(D_SIZE*(v+1)-1,D_SIZE*v));
			}
		}

		loop_solve:for(int v = 0; v < 8; v++){
			#pragma HLS unroll factor=8
			xf::fintech::trsvCore<float, N, 10, 1>(inlow[v], indiag[v], inup[v], inrhs[v]);
		}

		uint256_dt vec_w;
		loop_write:for(ap_uint<12> i = 0; i < N; i++){
			#pragma HLS pipeline ii=1
			for(int v= 0; v < 8; v++){
				vec_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(inrhs[v][i]/indiag[v][i]);

			}
//			if(itr*d0+i < ReadLimit){
				out_stm << vec_w;
//			}
		}
	}
}




static void write_u(uint512_dt* u, hls::stream<uint256_dt> &u_stm,
		ap_uint<12> M, ap_uint<12> N, ap_uint<14> B){

	ap_uint<8> XBlocks = (M >> 4);

	int toltal_itr = XBlocks * N * B;
	for(int itr= 0; itr < toltal_itr; itr++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=102400 max=204800 avg=204800
		uint512_dt tmp;
		tmp.range(255,0) = u_stm.read();
		tmp.range(511,256) = u_stm.read();
		u[itr] = tmp;;
	}
//	printf("write_u: i didn't get stuck\n");

}

static void TDMA1(const uint512_dt* a,  const uint512_dt* b,
		const uint512_dt* c,  const uint512_dt* d,
		uint512_dt* u, int M, int N, int B){

	static hls::stream<uint256_dt> a_stm_0[5];
	static hls::stream<uint256_dt> b_stm_0[5];
	static hls::stream<uint256_dt> c_stm_0[5];
	static hls::stream<uint256_dt> d_stm_0[5];
	static hls::stream<uint256_dt> u_stm_0[5];


	#pragma HLS STREAM variable = a_stm_0 depth = 2
	#pragma HLS STREAM variable = b_stm_0 depth = 2
	#pragma HLS STREAM variable = c_stm_0 depth = 2
	#pragma HLS STREAM variable = d_stm_0 depth = 2
	#pragma HLS STREAM variable = u_stm_0 depth = 2


    struct data_G data_g;
    data_g.sizex = M-2;
    data_g.sizey = N-2;
    data_g.xdim0 = M;
	data_g.end_index = (M >> 3);
	data_g.end_row = N;
	data_g.outer_loop_limit = N+1; // n + D/2
	data_g.gridsize = (data_g.end_row* B + 1) * data_g.end_index;
	data_g.endindex_minus1 = data_g.end_index -1;
	data_g.endrow_plus1 = data_g.end_row + 1;
	data_g.endrow_plus2 = data_g.end_row + 2;
	data_g.endrow_minus1 = data_g.end_row - 1;
	data_g.total_itr_256 = data_g.end_row * data_g.end_index * B;
	data_g.total_itr_512 = (data_g.end_row * data_g.end_index * B + 1) >> 1;

	ap_uint<12> TileX = (M >> 3);
	ap_uint<12> TileY = N;
	ap_uint<18> NTiles = B;


	// thomas solver parameters
	ap_uint<12> TileX_TC, TileY_TC;
	unsigned int ReadLimit_X, ReadLimit_Y;
	ap_uint<9> XBlocks = (M >> 3);

	ReadLimit_X = ((B*N+7)>>3)*(XBlocks << 3);
	ReadLimit_Y = XBlocks*B*N ;

//	int B_X = (B*N+255)>>8;
//	int B_Y = ((M*B+255)>>8);

	int N_sys = (B*N+7)>>3;


	hls::stream<uint256_dt> abcd_fw_stm[8];
	hls::stream<uint256_dt> c2_fw_stm[8];
	hls::stream<uint256_dt> d2_fw_stm[8];

	#pragma HLS STREAM variable = abcd_fw_stm depth = 2
	#pragma HLS STREAM variable = c2_fw_stm depth = 2
	#pragma HLS STREAM variable = d2_fw_stm depth = 2

	int total_512 = (data_g.total_itr_256 >> 1);

	#pragma HLS dataflow
	read_coeff(a, a_stm_0[0], b, b_stm_0[0], c, c_stm_0[0], d, d_stm_0[0], M, N, B);
	printf("Finished reading \n");

	interleaved_row_block8(a_stm_0[0], a_stm_0[2], M, N, B, 0);
	interleaved_row_block8(b_stm_0[0], b_stm_0[2], M, N, B, 0);
	interleaved_row_block8(c_stm_0[0], c_stm_0[2], M, N, B, 0);
	interleaved_row_block8(d_stm_0[0], d_stm_0[2], M, N, B, 0);

	stream_8x8transpose(a_stm_0[2], a_stm_0[3], M, N, B, 0);
	stream_8x8transpose(b_stm_0[2], b_stm_0[3], M, N, B, 0);
	stream_8x8transpose(c_stm_0[2], c_stm_0[3], M, N, B, 0);
	stream_8x8transpose(d_stm_0[2], d_stm_0[3], M, N, B, 0);

//	thomas_interleave(a_stm_0[3], abcd_fw_stm[0], M, B_X, ReadLimit_X);
//	thomas_interleave(b_stm_0[3], abcd_fw_stm[1], M, B_X, ReadLimit_X);
//	thomas_interleave(c_stm_0[3], abcd_fw_stm[2], M, B_X, ReadLimit_X);
//	thomas_interleave(d_stm_0[3], abcd_fw_stm[3], M, B_X, ReadLimit_X);
	printf("PCR solver beginning \n");
	PCR_solver(a_stm_0[3], b_stm_0[3], c_stm_0[3], d_stm_0[3], u_stm_0[0], M, N_sys);
	printf("PCR solver end \n");
//	thomas_backward(c2_fw_stm[0], d2_fw_stm[0], u_stm_0[0], M, B_X, ReadLimit_X);
	stream_8x8transpose(u_stm_0[0], u_stm_0[1], M, N, B, 0);
	undo_interleaved_row_block8(u_stm_0[1], u_stm_0[2], M, N, B, 0);

	write_u(u, u_stm_0[2], M, N, B);



}



//sp=TDMA_batch_2.d:HBM[4:5]
//sp=TDMA_batch_2.u:HBM[4:5]
//sp=TDMA_batch_2.acc_1:HBM[6:7]
//sp=TDMA_batch_2.acc_2:HBM[6:7]
//
//sp=TDMA_batch_3.d:HBM[8:9]
//sp=TDMA_batch_3.u:HBM[8:9]
//sp=TDMA_batch_3.acc_1:HBM[10:11]
//sp=TDMA_batch_3.acc_2:HBM[10:11]
//
//sp=TDMA_batch_4.d:HBM[12:13]
//sp=TDMA_batch_4.u:HBM[12:13]
//sp=TDMA_batch_4.acc_1:HBM[14:15]
//sp=TDMA_batch_4.acc_2:HBM[14:15]
//
//sp=TDMA_batch_5.d:HBM[16:17]
//sp=TDMA_batch_5.u:HBM[16:17]
//sp=TDMA_batch_5.acc_1:HBM[18:19]
//sp=TDMA_batch_5.acc_2:HBM[18:19]
//
//sp=TDMA_batch_6.d:HBM[20:21]
//sp=TDMA_batch_6.u:HBM[20:21]
//sp=TDMA_batch_6.acc_1:HBM[22:23]
//sp=TDMA_batch_6.acc_2:HBM[22:23]




//[advanced]
//param=compiler.userPostSysLinkOverlayTcl=/ssd_1/kkvasan/vits_ws/adi_2d_unroll/adi_2duroll/src/postSysLink.tcl

extern "C" {
void TDMA_batch(
	uint512_dt* a,
	uint512_dt* b,
	uint512_dt* c,
	uint512_dt* d,
	uint512_dt* u,
	int M,
	int N,
	int B,
	int iters){

	#pragma HLS INTERFACE depth=4096 m_axi port = a offset = slave bundle = gmem0 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=4 num_write_outstanding=4 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = b offset = slave bundle = gmem1 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=4 num_write_outstanding=4 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = c offset = slave bundle = gmem2 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=4 num_write_outstanding=4 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = d offset = slave bundle = gmem3 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=4 num_write_outstanding=4 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = u offset = slave bundle = gmem4 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=4 num_write_outstanding=4 latency=64

	#pragma HLS INTERFACE s_axilite port = a bundle = control
	#pragma HLS INTERFACE s_axilite port = b bundle = control
	#pragma HLS INTERFACE s_axilite port = c bundle = control
	#pragma HLS INTERFACE s_axilite port = d bundle = control
	#pragma HLS INTERFACE s_axilite port = u bundle = control

	#pragma HLS INTERFACE s_axilite port = N bundle = control
	#pragma HLS INTERFACE s_axilite port = M bundle = control
	#pragma HLS INTERFACE s_axilite port = B bundle = control
	#pragma HLS INTERFACE s_axilite port = iters bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control



	for(ap_uint<20> itr = 0; itr < iters; itr++){
		#pragma HLS loop_tripcount min=1 max=100 avg=10
		#pragma HLS dependence variable=d intra RAW true
		#pragma HLS dependence variable=u intra RAW true
		bool dnt_acc_updt = (itr == 0) ? 1 : 0;
		TDMA1(a, b, c, d, u,  M, N, B);
	}

}
}
