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
		const uint512_dt*d, hls::stream<uint256_dt> &d_stm0,  hls::stream<uint256_dt> &d_stm1,
		ap_uint<12> M, ap_uint<12> N, ap_uint<8> B, unsigned char dirXYZ){
	ap_uint<12> TileX;
	ap_uint<22> TileY;
	ap_uint<8> NTiles;
	ap_uint<8> XBlocks = (M >> 4);
	unsigned int offset;
	switch(dirXYZ){
		case 0: {TileX=XBlocks; TileY=B*N; NTiles = 1; break;}
		case 1: {TileX=4; TileY=B*N; NTiles = (((XBlocks)>>2)); break;}
		default: {TileX=XBlocks; TileY=N; NTiles = B; break;}
	}

	ap_uint<8> id = 0;
	ap_uint<22> jd =0;
	int total_itr = NTiles*TileY;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
		ap_uint<8> i = id;
		ap_uint<22> j = jd;

		if(j == TileY -1){
			jd = 0;
			id++;
		} else {
			jd++;
		}

//		int offsetX = i * TileX + (j*((XBlocks*B+3)>>2)<<2);
		int offset = j * XBlocks + i* TileX; //(j*((XBlocks*B+3)>>2)<<2);
		for(ap_uint<12>  k = 0; k < TileX; k++){
			#pragma HLS loop_tripcount min=100 max=255 avg=255
			#pragma HLS PIPELINE II=1
			uint512_dt tmp_d =  d[offset+k];
//			d_stm << tmp_d;
			d_stm0 << tmp_d.range(255,0);
			d_stm1 << tmp_d.range(511,256);
		}
	}


//	printf("read_coeff: i didn't get stuck\n");

}


static void read_acc(
		const uint512_dt*d, hls::stream<uint256_dt> &d_stm,
		ap_uint<12> M, ap_uint<12> N, ap_uint<8> B, unsigned char dirXYZ){
	ap_uint<12> TileX;
	ap_uint<22> TileY;
	ap_uint<8> NTiles;
	ap_uint<8> XBlocks = (M >> 4);
	unsigned int offset;
	switch(dirXYZ){
		case 0: {TileX=XBlocks; TileY=B*N; NTiles = 1; break;}
		case 1: {TileX=4; TileY=B*N; NTiles = (((XBlocks)>>2)); break;}
		default: {TileX=XBlocks; TileY=N; NTiles = B; break;}
	}

	ap_uint<8> id = 0;
	ap_uint<22> jd =0;
	int total_itr = XBlocks * N* B;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
		uint512_dt tmp_d =  d[itr];
//		d_stm << tmp_d;
		d_stm << tmp_d.range(255,0);
		d_stm << tmp_d.range(511,256);

	}


//	printf("read_coeff: i didn't get stuck\n");

}


static void stream_convert_512_256(hls::stream<uint256_dt> &in0, hls::stream<uint256_dt> &in1, hls::stream<uint256_dt> &out, unsigned int total_itr){
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
//		uint512_dt tmp = in.read();
//		uint256_dt var_l = tmp.range(255,0);
//		uint256_dt var_h = tmp.range(511,256);;
		out << in0.read();
		out << in1.read();

	}
}

// data width conversion to support 512 bit width memory write interface
static void stream_convert_256_512(hls::stream<uint256_dt> &in, hls::stream<uint256_dt> &out0,  hls::stream<uint256_dt> &out1, unsigned int total_itr){
	for (int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=2
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		uint512_dt tmp;
		out0 << in.read();
		out1 << in.read();
//		out << tmp;
	}
}
// TDMA Modules
static void interleaved_row_block8(hls::stream<uint256_dt> &in, hls::stream<uint256_dt> &out,
				unsigned short M, unsigned short N, unsigned short B, unsigned char dirXYZ){

	ap_uint<12> TileX, TileY;
	ap_uint<22> NTiles;
	ap_uint<9> XBlocks = (M >> 3);
	unsigned int offset;
	const int N_CU = 8;
	switch(dirXYZ){
		case 0: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+7)>>3; break;}
		case 1: {TileX=8; TileY=N; NTiles = (((XBlocks*B+7)>>3)); break;}
		default: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+7)>>3; break;}
	}

	uint256_dt tmp_M[DIM_MAX*N_CU*2];
	#pragma HLS RESOURCE variable=tmp_M core=XPM_MEMORY uram
	ap_uint<22> NTilesp1 = register_it<int>(NTiles+1);
	ap_uint<22> id = 0;
	ap_uint<12> jd =0, kd = 0;
	int total_itr = NTilesp1*TileX*TileY;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<22> i = id;
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
		if(cmpW && i < NTiles){
			tmpW = in.read();
		}
		tmp_M[indW] = tmpW;

		int indR = j*TileX + k + offsetR;
		uint256_dt tmpR = tmp_M[indR];
		if(i > 0){
			out << tmpR;
		}

	}
//	printf("interleaved_row_block8: i didn't get stuck\n");
}


static void undo_interleaved_row_block8(hls::stream<uint256_dt> &in, hls::stream<uint256_dt> &out0,
		unsigned short M, unsigned short N, unsigned short B, unsigned char dirXYZ){


	ap_uint<12> TileX, TileY;
	ap_uint<20> NTiles;
	ap_uint<9> XBlocks = (M >> 3);
	unsigned short offset;
	const int N_CU = 8;
	switch(dirXYZ){
		case 0: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+7)>>3; break;}
		case 1: {TileX=8; TileY=N; NTiles = (((XBlocks*B+7)>>3)); break;}
		default: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+7)>>3; break;}
	}

//	printf("undo_interleaved_row_block8: ReadLimit:%d (i*TileY + j):%d\n", B*N, (NTiles*TileY));

	uint256_dt tmp_M[DIM_MAX*N_CU*2];
	#pragma HLS RESOURCE variable=tmp_M core=XPM_MEMORY uram
	ap_uint<22> NTilesp1 = register_it<int>(NTiles+1);
	ap_uint<22> id = 0;
	ap_uint<12> jd =0, kd = 0;
	int total_itr = NTilesp1*TileX*TileY;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<22> i = id;
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
		if(cmpW && i < NTiles){
			tmpW = in.read();
		}
		tmp_M[indW] = tmpW;

		int indR = k*TileY+j + offsetR;
		uint256_dt tmpR = tmp_M[indR];
		if(i > 0){
			out0 << tmpR;
		}

	}
//	printf("undo_interleaved_row_block8: i didn't get stuck\n");
}

static void stream_8x8transpose(hls::stream<uint256_dt> &in, hls::stream<uint256_dt> &out0,
		unsigned short M, unsigned short N, unsigned short B, unsigned char dirXYZ){

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
		#pragma HLS loop_tripcount min=102400 max=204800 avg=204800
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


static void TT_STAGE0( hls::stream<uint256_dt> &d_stm_in,
		hls::stream<uint256_dt> &a_stm_out, hls::stream<uint256_dt> &b_stm_out, hls::stream<uint256_dt> &c_stm_out, hls::stream<uint256_dt> &d_stm_out,
//		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned int ReadLimit
		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned short Sys, ap_uint<8> Ti, unsigned int ReadLimit
		){

	uint256_dt  a_pre[N_MAX], b_pre[N_MAX], c_pre[N_MAX], d_pre[N_MAX];
	#pragma HLS RESOURCE variable=a_pre core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=b_pre core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=c_pre core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=d_pre core=XPM_MEMORY uram

	ap_uint<22> batL = B_size+1;
	ap_uint<22> batd = 0;
	ap_uint<6> id =0;
	ap_uint<12> jd = 0;
	int total_itr = batL*Tiles*M;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<22> bat = batd;
		ap_uint<6> i = id;
		ap_uint<12> j = jd;

		if(j == M -1){
			jd = 0;
		} else {
			jd++;
		}

		if(j == M -1 && i == Tiles -1){
			id = 0;
			batd++;
		} else if(j == M -1){
			id++;
		}


		uint256_dt tmp_a, tmp_b, tmp_c, tmp_d;
		int count = bat * Tiles*M + i * M + j;
		if(count < ReadLimit && bat < B_size){
			tmp_d = d_stm_in.read();
		} else {
			tmp_a = 0;
			tmp_b = 0;
			tmp_c = 0;
			tmp_d = 0;
		}

		float tmp_a_f[8], tmp_b_f[8], tmp_c_f[8], tmp_d_f[8];
		uint256_dt tmp_a_w, tmp_b_w, tmp_c_w, tmp_d_w;
		for(int v = 0; v < VEC_FACTOR; v++){
			tmp_a_f[v] = uint2float_ript(tmp_a.range(D_SIZE*(v+1)-1,D_SIZE*v));
			tmp_b_f[v] = uint2float_ript(tmp_b.range(D_SIZE*(v+1)-1,D_SIZE*v));
			tmp_c_f[v] = uint2float_ript(tmp_c.range(D_SIZE*(v+1)-1,D_SIZE*v));
			tmp_d_f[v] = uint2float_ript(tmp_d.range(D_SIZE*(v+1)-1,D_SIZE*v));
			ap_uint<12> d0 = (i % Ti)*M + j;
			ap_uint<12> sys_size = Ti*M;


			tmp_a_f[v] = (d0 == 0 || d0 == sys_size-1) ? 0.0f : -0.5f;
			tmp_c_f[v] = (d0 == 0 || d0 == sys_size-1) ? 0.0f : -0.5f;
			tmp_b_f[v] = (d0 == 0 || d0 == sys_size-1) ? 1.0f :  2.0f;

			tmp_a_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(tmp_a_f[v]);
			tmp_b_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(tmp_b_f[v]);
			tmp_c_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(tmp_c_f[v]);
			tmp_d_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(tmp_d_f[v]);

		}

		int offsetW = ((bat & 1) == 0 ? 0 : 2048);
		int offsetR = ((bat & 1) == 1 ? 0 : 2048);

		int ind_w = j*Tiles+i;
		int ind_R = i*M+j;
		a_pre[ind_w+offsetW] = tmp_a_w;
		b_pre[ind_w+offsetW] = tmp_b_w;
		c_pre[ind_w+offsetW] = tmp_c_w;
		d_pre[ind_w+offsetW] = tmp_d_w;

		uint256_dt tmp_a_R = a_pre[ind_R+offsetR];
		uint256_dt tmp_b_R = b_pre[ind_R+offsetR];
		uint256_dt tmp_c_R = c_pre[ind_R+offsetR];
		uint256_dt tmp_d_R = d_pre[ind_R+offsetR];

		if(bat > 0){
			a_stm_out << tmp_a_R;
			b_stm_out << tmp_b_R;
			c_stm_out << tmp_c_R;
			d_stm_out << tmp_d_R;
		}

	}
//	printf("Stage 0 has been returned\n");
}



static void TT_STAGE1(hls::stream<uint256_dt> &a_stm_in, hls::stream<uint256_dt> &b_stm_in,  hls::stream<uint256_dt> &c_stm_in, hls::stream<uint256_dt> &d_stm_in,
		hls::stream<uint256_dt> &a_stm_out, hls::stream<uint256_dt> &c_stm_out, hls::stream<uint256_dt> &d_stm_out,
		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned short Sys, ap_uint<8> Ti
		){

	uint256_dt  a_fw[N_MAX], c_fw[N_MAX], d_fw[N_MAX];
	#pragma HLS RESOURCE variable=a_fw core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=c_fw core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=d_fw core=XPM_MEMORY uram


	uint256_dt window_a2[N_BLK], window_c2[N_BLK], window_d2[N_BLK];

	ap_uint<22> batL = B_size+1;
	ap_uint<22> batd = 0;
	ap_uint<12> id =0;
	ap_uint<8> kd = 0;
	int total_itr = batL*Tiles*M;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
		#pragma HLS dependence variable=window_a2 RAW distance=n_blk true
		#pragma HLS dependence variable=window_c2 RAW distance=n_blk true
		#pragma HLS dependence variable=window_d2 RAW distance=n_blk true

		ap_uint<22> bat = batd;
		ap_uint<12> i = id;
		ap_uint<8> k = kd;

		if(k == Tiles -1){
			kd = 0;
		} else {
			kd++;
		}

		if(k == Tiles -1 && i == M -1){
			id = 0;
			batd++;
		} else if(k == Tiles -1){
			id++;
		}

		int ind =  k*M+i;
		uint256_dt a_vec_r, b_vec_r, c_vec_r, d_vec_r;
		if(bat < B_size){
			a_vec_r = a_stm_in.read();
			b_vec_r = b_stm_in.read();
			c_vec_r = c_stm_in.read();
			d_vec_r = d_stm_in.read();
		}

		uint256_dt a_vec_or = window_a2[k];
		uint256_dt c_vec_or = window_c2[k];
		uint256_dt d_vec_or = window_d2[k];

		uint256_dt vec_a_w, vec_d_w, vec_c_w;

		fw_vec_loop_TM: for(int v =0; v < VEC_FACTOR; v++){
			#pragma HLS unroll
			float a = uint2float_ript(a_vec_r.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float b = uint2float_ript(b_vec_r.range(D_SIZE*(v+1)-1,D_SIZE*v)); //normalised to one
			float c = uint2float_ript(c_vec_r.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float d = uint2float_ript(d_vec_r.range(D_SIZE*(v+1)-1,D_SIZE*v));

			float a_old = uint2float_ript(a_vec_or.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float c_old = uint2float_ript(c_vec_or.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float d_old = uint2float_ript(d_vec_or.range(D_SIZE*(v+1)-1,D_SIZE*v));

			float denom = (i == 0 || i == 1) ? b : (b - a*c_old);
			float r = 1/ denom;
			float d_w1 = (i == 0 || i == 1) ? d : (d - a*d_old);
			float a_w1 = (i == 0 || i == 1) ? a : (-1.0f)*a*a_old;
			float c_w1 = (i == 0 || i == 1) ? c : c;

			float d_w = r*d_w1;
			float a_w = r*a_w1;
			float c_w = r*c_w1;

			vec_a_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(a_w);
			vec_c_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(c_w);
			vec_d_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(d_w);

		}

		window_a2[k] = vec_a_w;
		window_c2[k] = vec_c_w;
		window_d2[k] = vec_d_w;

		int offsetW = ((bat & 1) == 0 ? 0 : 2048);
		int offsetR = ((bat & 1) == 1 ? 0 : 2048);

		int ind_R =  k*M+i;
		int ind_W =  k*M+(M-1-i);

		a_fw[ind_R+offsetW] = vec_a_w;
		c_fw[ind_R+offsetW] = vec_c_w;
		d_fw[ind_R+offsetW] = vec_d_w;

		uint256_dt tmp_a_W = a_fw[ind_W+offsetR];
		uint256_dt tmp_c_W = c_fw[ind_W+offsetR];
		uint256_dt tmp_d_W = d_fw[ind_W+offsetR];

		if(bat > 0){
			a_stm_out << tmp_a_W;
			c_stm_out << tmp_c_W;
			d_stm_out << tmp_d_W;
		}

	}
//	printf("Stage 1 has been returned\n");
}

static void TT_STAGE2(hls::stream<uint256_dt> &a_stm_in, hls::stream<uint256_dt> &c_stm_in, hls::stream<uint256_dt> &d_stm_in,
		hls::stream<uint256_dt> &a_stm_out, hls::stream<uint256_dt> &c_stm_out, hls::stream<uint256_dt> &d_stm_out,
		hls::stream<uint256_dt> &ra_stm_out, hls::stream<uint256_dt> &rb_stm_out, hls::stream<uint256_dt> &rc_stm_out, hls::stream<uint256_dt> &rd_stm_out,
		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned short Sys, ap_uint<8> Ti
		){

	uint256_dt  a_bw[N_MAX], c_bw[N_MAX], d_bw[N_MAX];
	uint256_dt  ra_pre[RN_MAX], rb_pre[RN_MAX], rc_pre[RN_MAX], rd_pre[RN_MAX];
	#pragma HLS RESOURCE variable=a_bw core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=c_bw core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=d_bw core=XPM_MEMORY uram

	uint256_dt window_a_RTM[N_BLK], window_c_RTM[N_BLK], window_d_RTM[N_BLK];

	ap_uint<22> batL = B_size+1;
	ap_uint<22> batd = 0;
	ap_uint<12> idd =0;
	ap_uint<8> kd = 0;
	int total_itr = batL*Tiles*M;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
		#pragma HLS dependence variable=window_a_RTM RAW distance=n_blk true
		#pragma HLS dependence variable=window_c_RTM RAW distance=n_blk true
		#pragma HLS dependence variable=window_d_RTM RAW distance=n_blk true

		ap_uint<22> bat = batd;
		ap_uint<12> id = idd;
		ap_uint<8> k = kd;

		if(k == Tiles -1){
			kd = 0;
		} else {
			kd++;
		}

		if(k == Tiles -1 && id == M -1){
			idd = 0;
			batd++;
		} else if(k == Tiles -1){
			idd++;
		}

		ap_uint<12> i = M - 1 -id;
		int ind =  k*M+i;
		uint256_dt a_vec_r, c_vec_r, d_vec_r;
		if(bat < B_size){
			a_vec_r = a_stm_in.read();
			c_vec_r = c_stm_in.read();
			d_vec_r = d_stm_in.read();
		}

		uint256_dt a_vec_or = window_a_RTM[k];
		uint256_dt c_vec_or = window_c_RTM[k];
		uint256_dt d_vec_or = window_d_RTM[k];

		uint256_dt a_vec_w, b_vec_w, c_vec_w, d_vec_w;
		bw_vec_loop_TM: for(int v = 0; v < VEC_FACTOR; v++){

			float a_r = uint2float_ript(a_vec_r.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float c_r = uint2float_ript(c_vec_r.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float d_r = uint2float_ript(d_vec_r.range(D_SIZE*(v+1)-1,D_SIZE*v));

			float a_or = uint2float_ript(a_vec_or.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float c_or = uint2float_ript(c_vec_or.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float d_or = uint2float_ript(d_vec_or.range(D_SIZE*(v+1)-1,D_SIZE*v));

			float d_w = (i == M-1 || i == M-2) ? d_r : d_r - c_r * d_or;
			float a_w = (i == M-1 || i == M-2 || i == 0) ?a_r : a_r - c_r*a_or;
			float b_w = (i == 0) ? 1.0f - c_r*a_or : 1.0f;
			float c_w = (i == M-1 || i == M-2) ? c_r : -c_r * c_or;

			a_vec_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(a_w);
			b_vec_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(b_w);
			c_vec_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(c_w);
			d_vec_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(d_w);

		}

		int offsetW = ((bat & 1) == 0 ? 0 : 2048);
		int offsetR = ((bat & 1) == 1 ? 0 : 2048);

		int offsetW_rr = ((bat & 1) == 0 ? 0 : 256);
		int offsetR_rr = ((bat & 1) == 1 ? 0 : 256);

		unsigned char ind_r;
		if(i == 0){
			ind_r = (k << 1);
		} else {
			ind_r = (k << 1) + 1;
		}
		unsigned char k_i = ind_r % (Ti<<1);
		unsigned char sys_i = ind_r / (Ti<<1);

		if(i == 0 || i == M-1){
			int ind_rW = k_i*Sys + sys_i;

			ra_pre[ind_r+offsetW_rr] = a_vec_w;
			rb_pre[ind_r+offsetW_rr] = b_vec_w;
			rc_pre[ind_r+offsetW_rr] = c_vec_w;
			rd_pre[ind_r+offsetW_rr] = d_vec_w;
		}

		unsigned char ind_rr = (M-1-i)*Tiles + k;
		uint256_dt tmp_ra_preR = ra_pre[ind_rr+offsetR_rr];
		uint256_dt tmp_rb_preR = rb_pre[ind_rr+offsetR_rr];
		uint256_dt tmp_rc_preR = rc_pre[ind_rr+offsetR_rr];
		uint256_dt tmp_rd_preR = rd_pre[ind_rr+offsetR_rr];

		if((i == M-1 || i == M-2) && bat > 0){
			ra_stm_out << tmp_ra_preR;
			rb_stm_out << tmp_rb_preR;
			rc_stm_out << tmp_rc_preR;
			rd_stm_out << tmp_rd_preR;
		}

		window_a_RTM[k] = a_vec_w;
		window_c_RTM[k] = c_vec_w;
		window_d_RTM[k] = d_vec_w;

		int ind_W = k*M+i;
		a_bw[ind_W+offsetW] = a_vec_w;
		c_bw[ind_W+offsetW] = c_vec_w;
		d_bw[ind_W+offsetW] = d_vec_w;


		int ind_rev = (M-1-i)*Tiles + k;
		uint256_dt tmp_a_preR = a_bw[ind_rev+offsetR];
		uint256_dt tmp_c_preR = c_bw[ind_rev+offsetR];
		uint256_dt tmp_d_preR = d_bw[ind_rev+offsetR];

		if(bat > 0){
			a_stm_out << tmp_a_preR;
			c_stm_out << tmp_c_preR;
			d_stm_out << tmp_d_preR;
		}

	}
//	printf("Stage 2 has been returned\n");
}



static void pcr_solver(hls::stream<uint256_dt> &ra_stm_in, hls::stream<uint256_dt> &rb_stm_in,
		hls::stream<uint256_dt> &rc_stm_in, hls::stream<uint256_dt> &rd_stm_in,
		hls::stream<uint256_dt> &ru_stm_out_t, hls::stream<uint256_dt> &ru_stm_out_b,
		ap_uint<7> n, ap_uint<6> Sys, ap_uint<4> logn, ap_uint<16> bigbatch){

	const int N_max = 512;
	float inlow[VEC_FACTOR][N_max];
	float indiag[VEC_FACTOR][N_max];
	float inup[VEC_FACTOR][N_max];
	float inrhs[VEC_FACTOR][N_max];

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

	// copy loop
//	for(ap_uint<16> bat = 0; bat < bigbatch; bat++){

		#pragma HLS dataflow

		for(int i = 0; i < n*Sys*bigbatch; i++){
			#pragma HLS PIPELINE II=1
			#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
			uint256_dt tmp_a = ra_stm_in.read();
			uint256_dt tmp_b = rb_stm_in.read();
			uint256_dt tmp_c = rc_stm_in.read();
			uint256_dt tmp_d = rd_stm_in.read();

			for(int v = 0; v < VEC_FACTOR; v++){
				pcr_a[v] << uint2float_ript(tmp_a.range(D_SIZE*(v+1)-1,D_SIZE*v));
				pcr_b[v] << uint2float_ript(tmp_b.range(D_SIZE*(v+1)-1,D_SIZE*v));
				pcr_c[v] << uint2float_ript(tmp_c.range(D_SIZE*(v+1)-1,D_SIZE*v));
				pcr_d[v] << uint2float_ript(tmp_d.range(D_SIZE*(v+1)-1,D_SIZE*v));
			}
		}

		for(int v = 0; v < VEC_FACTOR; v++){
			#pragma HLS unroll factor=8
			xf::fintech::trsvCore <float, 512> (pcr_a[v], pcr_b[v], pcr_c[v], pcr_d[v], stm_out[v],  n, Sys, logn, bigbatch);
		}


		for(int itr =0; itr < Sys*n*bigbatch; itr++){
			#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
			#pragma HLS PIPELINE II=1
			uint256_dt tmp_d;
			for(int v = 0; v < VEC_FACTOR; v++){
				tmp_d.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(stm_out[v].read());
			}
			if((itr & 1) == 0){
				ru_stm_out_t << tmp_d;
			} else {
				ru_stm_out_b << tmp_d;
			}
		}


//	}

}


static void TT_STAGE4_scalar(hls::stream<float> &ra_stm_in, hls::stream<float> &rb_stm_in, hls::stream<float> &rc_stm_in, hls::stream<float> &rd_stm_in,
		hls::stream<float> &rc_stm_out, hls::stream<float> &rd_stm_out,
		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned short Sys, ap_uint<8> Ti
		){

	float  rc_fw[N_MAX], rd_fw[N_MAX];
	#pragma HLS RESOURCE variable=rc_fw core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=rd_fw core=XPM_MEMORY uram

	float vec_b_old_FTM[N_BLK], vec_c_old_FTM[N_BLK], vec_d_old_FTM[N_BLK];

	ap_uint<22> batL = B_size+1;
	ap_uint<22> batd = 0;
	ap_uint<6> id =0;
	ap_uint<7> kd = 0;
	const unsigned char TilesN = (Ti << 1);
	int total_itr = batL*MAX_Sys*TilesN;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
		#pragma HLS dependence variable=vec_b_old_FTM RAW distance=n_blk true
		#pragma HLS dependence variable=vec_c_old_FTM RAW distance=n_blk true
		#pragma HLS dependence variable=vec_d_old_FTM RAW distance=n_blk true

		ap_uint<22> bat = batd;
		ap_uint<6> i = id;
		ap_uint<7> k = kd;

		if(i == MAX_Sys -1){
			id = 0;
		} else {
			id++;
		}

		if(i == MAX_Sys -1 && k == TilesN -1){
			kd = 0;
			batd++;
		} else if(i == MAX_Sys -1){
			kd++;
		}

		#pragma HLS dependence variable=vec_b_old_FTM RAW distance=n_blk true
		#pragma HLS dependence variable=vec_c_old_FTM RAW distance=n_blk true
		#pragma HLS dependence variable=vec_d_old_FTM RAW distance=n_blk true

		float a_r, b_r, c_r, d_r;

		if(i < Sys && bat < B_size){
			a_r = ra_stm_in.read();
			b_r = rb_stm_in.read();
			c_r = rc_stm_in.read();
			d_r = rd_stm_in.read();
		}


		float b_or = vec_b_old_FTM[i];
		float c_or = vec_c_old_FTM[i];
		float d_or = vec_d_old_FTM[i];

		float denom = (k == 0) ? b_r : b_r - a_r*c_or;
		float r = 1.0/denom;
		float c_w1 = c_r;
		float d_w1 = (k==0) ? d_r : d_r - a_r*d_or;

		float b_w = 1.0f;
		float c_w = c_w1*r;
		float d_w = d_w1*r;


		vec_b_old_FTM[i] = b_w;
		vec_c_old_FTM[i] = c_w;
		vec_d_old_FTM[i] = d_w;


		int offsetW_rr = ((bat & 1) == 0 ? 0 : N_MAX/2);
		int offsetR_rr = ((bat & 1) == 1 ? 0 : N_MAX/2);

		int pre_add = k + i*TilesN;
		int ind_red =  k + i*TilesN; //(i < Sys) ? pre_add : 255;

		rc_fw[ind_red+offsetW_rr] = c_w; // check this
		rd_fw[ind_red+offsetW_rr] = d_w;


		int ind_red_w = TilesN-1-k + i*TilesN;
		float tmp_cR = rc_fw[ind_red_w+offsetR_rr];
		float tmp_dR = rd_fw[ind_red_w+offsetR_rr];
		if(i < Sys && bat > 0){
			rc_stm_out << tmp_cR;
			rd_stm_out << tmp_dR;
		}

	}

//	printf("Stage 4 has been returned\n");
}



static void TT_STAGE6_scalar(hls::stream<float> &rc_stm_in, hls::stream<float> &rd_stm_in,
		hls::stream<float> &u_top, hls::stream<float> &u_bottom,
		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned short Sys, ap_uint<8> Ti
		){

	float  ru_fw[N_MAX];
	#pragma HLS RESOURCE variable=ru_fw core=XPM_MEMORY uram

	float vec_u_old_RTM[N_BLK];
	ap_uint<8> TilesN = (Ti << 1);

	ap_uint<22> batL = B_size+1;
	ap_uint<22> batd = 0;
	ap_uint<6> id =0;
	ap_uint<8> kdd = 0;
	int total_itr = batL*TilesN*MAX_Sys;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
		#pragma HLS dependence variable=vec_u_old_RTM RAW distance=n_blk true

		ap_uint<22> bat = batd;
		ap_uint<6> i = id;
		ap_uint<8> kd = kdd;

		if(i == MAX_Sys -1){
			id = 0;
		} else {
			id++;
		}

		if(i == MAX_Sys -1 && kd == TilesN -1){
			kdd = 0;
			batd++;
		} else if(i == MAX_Sys -1){
			kdd++;
		}

		#pragma HLS dependence variable=vec_u_old_RTM RAW distance=n_blk true
		// forward solve of the reduced diagonal systems
		float c_r, d_r;
		ap_uint<7> k = TilesN -1 -kd;
		int add_ra = k+i*TilesN;
		ap_uint<12> k_3  =(ap_uint<12>)k<<3;
		ap_uint<12> TilesN_3 = ((ap_uint<12>)TilesN<<3);
		int ind_red = (i&7) + (i>>3)*TilesN_3 + k_3;
		//				printf("i,k,add_ra: %d %d %d %d\n", (int) i, (int) k, (int) ind_red, (int)k_3);

		if(bat < B_size && i < Sys){
			c_r = rc_stm_in.read();
			d_r = rd_stm_in.read();
		}

		float u_or = vec_u_old_RTM[i];
		float f_d = (k == TilesN-1) ? d_r : (d_r-c_r*u_or);
		float u_w = f_d;

		vec_u_old_RTM[i] = u_w;

		int offsetW_rr = ((bat & 1) == 0 ? 0 : N_MAX/2);
		int offsetR_rr = ((bat & 1) == 1 ? 0 : N_MAX/2);

		// declare the memory

		ru_fw[ind_red+offsetW_rr] = u_w;



		//				ap_uint<12> add_rev = kd*MAX_Sys+i;
		ap_uint<12> add_rev = kd*Sys+i;
		ap_uint<12> limit = Sys*TilesN;

		float tmp_uR = ru_fw[add_rev+offsetR_rr];

		if(((add_rev>>3) & 1) == 0 && bat > 0 && i < Sys){
			u_top << tmp_uR;
		} else if(bat > 0 && i < Sys){
			u_bottom << tmp_uR;
		}
	}
//	printf("Stage 6 has been returned\n");
}


static void TT_STAGE7(hls::stream<uint256_dt> &a_stm_in, hls::stream<uint256_dt> &c_stm_in, hls::stream<uint256_dt> &d_stm_in,
		hls::stream<uint256_dt> &u_top_in, hls::stream<uint256_dt> &u_bottom_in, hls::stream<uint256_dt> &u_out,
		ap_uint<22> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned int ReadLimit
		){

	ap_uint<22> batL = B_size;
	ap_uint<22> batd = 0;
	ap_uint<12> id =0;
	ap_uint<8> kd = 0;
	int total_itr = batL*Tiles*M;
	uint256_dt u0 =0, uM =0;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<22> bat = batd;
		ap_uint<8> k = kd;
		ap_uint<12> i = id;

		if(i == M -1){
			id = 0;
		} else {
			id++;
		}

		if(i == M -1 && k == Tiles -1){
			kd = 0;
			batd++;
		} else if(i == M -1){
			kd++;
		}

		int ind = k*M+i;
		uint256_dt uI;

		if(i == 0){
			u0 = u_top_in.read();
			uM = u_bottom_in.read();
		}

		uint256_dt vec_a = a_stm_in.read();
		uint256_dt vec_c = c_stm_in.read();
		uint256_dt vec_d = d_stm_in.read();
		for(int v = 0; v < VEC_FACTOR; v++){
			float a_r = uint2float_ript(vec_a.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float c_r = uint2float_ript(vec_c.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float d_r = uint2float_ript(vec_d.range(D_SIZE*(v+1)-1,D_SIZE*v));

			float u0_r = uint2float_ript(u0.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float uM_r = uint2float_ript(uM.range(D_SIZE*(v+1)-1,D_SIZE*v));

			float uI_w = d_r - a_r * u0_r - c_r*uM_r;

			uI.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(uI_w);
		}
		int count = bat * Tiles*M + k * M + i;
		if(count < ReadLimit){
			u_out << ((i == 0) ? u0 : (i == M-1) ? uM : uI);
		}
	}
//	printf("Stage 7 has been returned\n");
}


static void vec_to_float(hls::stream<uint256_dt> &stm_vec_in, hls::stream<float> &stm_scl_out, int size){

	for(int itr = 0; itr < size; itr++){
		#pragma HLS PIPELINE II=8
		uint256_dt tmp = stm_vec_in.read();
		for(int v = 0; v < VEC_FACTOR; v++){
			float val = uint2float_ript(tmp.range(D_SIZE*(v+1)-1,D_SIZE*v));
			stm_scl_out << val;
		}
	}
}

static void float_to_vec(hls::stream<float> &stm_scl_in, hls::stream<uint256_dt> &stm_vec_out, int size){

	for(int itr = 0; itr < size; itr++){
		#pragma HLS PIPELINE II=8
		uint256_dt tmp;
		for(int v = 0; v < VEC_FACTOR; v++){
			tmp.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(stm_scl_in.read());
		}
		stm_vec_out << tmp;
	}
}


static void write_u(uint512_dt* u, hls::stream<uint256_dt> &u_stm_0, hls::stream<uint256_dt> &u_stm_1,
		ap_uint<12> M, ap_uint<12> N, ap_uint<12> B, unsigned char dirXYZ){


	ap_uint<12> TileX;
	ap_uint<22> TileY;
	ap_uint<8> NTiles;
	ap_uint<8> XBlocks = (M >> 4);
	unsigned int offset;
	switch(dirXYZ){
		case 0: {TileX=XBlocks; TileY=B*N; NTiles = 1; break;}
		case 1: {TileX=4; TileY=B*N; NTiles = (((XBlocks)>>2)); break;}
		default: {TileX=XBlocks; TileY=N; NTiles = B; break;}
	}

	ap_uint<8> id = 0;
	ap_uint<22> jd =0;
	int total_itr = NTiles*TileY;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
		ap_uint<8> i = id;
		ap_uint<22> j = jd;

		if(j == TileY -1){
			jd = 0;
			id++;
		} else {
			jd++;
		}

//		int offsetX = i * TileX + (j*((XBlocks*B+3)>>2)<<2);
		int offset = j * XBlocks + i* TileX; //(j*((XBlocks*B+3)>>2)<<2);
		for(ap_uint<12> k = 0; k < TileX; k++){
			#pragma HLS loop_tripcount min=100 max=255 avg=255
			#pragma HLS PIPELINE II=1
			uint512_dt tmp;
//			tmp = u_stm.read();
			tmp.range(255,0) = u_stm_0.read();
			tmp.range(511,256) = u_stm_1.read();
			u[offset+k] = tmp;;
		}
	}
//	printf("I didn't got stuck at writing\n");

}


static void write_acc(uint512_dt* u, hls::stream<uint256_dt> &u_stm,
		ap_uint<12> M, ap_uint<12> N, ap_uint<12> B, unsigned char dirXYZ){


	ap_uint<12> TileX;
	ap_uint<22> TileY;
	ap_uint<8> NTiles;
	ap_uint<8> XBlocks = (M >> 4);
	int total_itr = XBlocks*N*B;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		#pragma HLS PIPELINE II=2
		uint512_dt tmp;
//		tmp = u_stm.read();
		tmp.range(255,0) = u_stm.read();
		tmp.range(511,256) = u_stm.read();
		u[itr] = tmp;;

	}
//	printf("I didn't got stuck at writing\n");

}


static void TDMA(const uint512_dt* d, uint512_dt* u,
		const uint512_dt* acc1, uint512_dt* acc2,
		ap_uint<12> Xdim, ap_uint<12> Ydim, ap_uint<12> Batch, ap_uint<12> Batch_acc, unsigned char dirXYZ, bool dnt_acc_updt){


	static hls::stream<uint256_dt> data_256_stm0[4];
	static hls::stream<uint256_dt> data_256_stm1[4];
	static hls::stream<uint256_dt> d_stm[4];
	static hls::stream<uint256_dt> u_stm[4];
	static hls::stream<uint256_dt> accStream[4];

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
	read_coeff(d, data_256_stm0[0],  data_256_stm1[0], Xdim, Ydim, Batch, dirXYZ);
	stream_convert_512_256(data_256_stm0[0],  data_256_stm1[0], d_stm[0], total_512_data);
	printf("read d\n");
	read_acc(acc1, accStream[0], Xdim, Ydim, Batch, 0);
	//	stream_convert_512_256(data_512_stm[1], accStream[0], total_512_data);
	printf("read acc1\n");

	process_grid(d_stm[0], d_stm[1], accStream[0], accStream[1], data_g, dnt_acc_updt, skip_pre);
	printf("process_grid done\n");
	//	stream_convert_256_512(accStream[1], data_512_stm[2], total_512_data);
	write_acc(acc2, accStream[1], Xdim, Ydim, Batch, 0);

	interleaved_row_block8(d_stm[1], d_stm[2], Xdim, Ydim, Batch, dirXYZ);


	stream_8x8transpose(d_stm[2], d_stm[3], Xdim, Ydim, Batch, dirXYZ);



	//interleave the systems
	TT_STAGE0(d_stm[3],
			  STAGE_0[0], STAGE_0[1], STAGE_0[2],  STAGE_0[3],
			  B_size, Tiles, M, Sys, Ti, ReadLimit);
		printf("finished stage 0\n");

	// FW sweep
	TT_STAGE1(STAGE_0[0], STAGE_0[1], STAGE_0[2],  STAGE_0[3],
			 STAGE_1[0], STAGE_1[1], STAGE_1[2],
			 B_size, Tiles, M, Sys, Ti);
		printf("finished stage 1\n");


	// BW Sweep
	TT_STAGE2(STAGE_1[0], STAGE_1[1], STAGE_1[2],
			STAGE_2A[0], STAGE_2A[1], STAGE_2A[2],
			STAGE_2B[3], STAGE_2B[4], STAGE_2B[5], STAGE_2B[6],
			B_size, Tiles, M, Sys, Ti);
		printf("finished stage 2\n");



	for(int i = 0; i < Rn*Sys*B_size; i++){
			#pragma HLS PIPELINE II=1
			#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
			uint256_dt tmp_a = STAGE_2B[3].read();
			uint256_dt tmp_b = STAGE_2B[4].read();
			uint256_dt tmp_c = STAGE_2B[5].read();
			uint256_dt tmp_d = STAGE_2B[6].read();

			for(int v = 0; v < VEC_FACTOR; v++){
				pcr_a[v] << uint2float_ript(tmp_a.range(D_SIZE*(v+1)-1,D_SIZE*v));
				pcr_b[v] << uint2float_ript(tmp_b.range(D_SIZE*(v+1)-1,D_SIZE*v));
				pcr_c[v] << uint2float_ript(tmp_c.range(D_SIZE*(v+1)-1,D_SIZE*v));
				pcr_d[v] << uint2float_ript(tmp_d.range(D_SIZE*(v+1)-1,D_SIZE*v));
			}
	}

	for(int v = 0; v < VEC_FACTOR; v++){
		#pragma HLS unroll factor=8
//		internal::trsv_step<float, 512>(pcr_a[v], pcr_b[v], pcr_c[v], pcr_d[v], stm_out[v],  Rn, Sys, (logn+2), B_size);
		xf::fintech::trsvCore <float, 512> (pcr_a[v], pcr_b[v], pcr_c[v], pcr_d[v], stm_out[v],  Rn, Sys, logn, B_size);
	}


	for(int itr =0; itr < Sys*Rn*B_size; itr++){
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
		#pragma HLS PIPELINE II=1
		uint256_dt tmp_d;
		for(int v = 0; v < VEC_FACTOR; v++){
			tmp_d.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(stm_out[v].read());
		}
		if((itr & 1) == 0){
			STAGE_6[0] << tmp_d;
		} else {
			STAGE_6[1] << tmp_d;
		}
	}



		// back substitution
		TT_STAGE7(STAGE_2A[0], STAGE_2A[1], STAGE_2A[2],
				STAGE_6[0], STAGE_6[1], u_stm[0],
				B_size, Tiles, M, ReadLimit);
		printf("finished stage 7\n");


		stream_8x8transpose(u_stm[0], u_stm[1], Xdim, Ydim, Batch, dirXYZ);
		printf("finished stream_8x8transpose\n");
		undo_interleaved_row_block8(u_stm[1], u_stm[2], Xdim, Ydim, Batch, dirXYZ);
		printf("finished undo_interleaved_row_block8\n");

		stream_convert_256_512(u_stm[2], data_256_stm0[3],  data_256_stm1[3], total_512_data);
		write_u(u, data_256_stm0[3],  data_256_stm1[3], Xdim, Ydim, Batch, dirXYZ);
		printf("finished write_u\n");


}










extern "C" {
void TDMA_batch(
//	uint512_dt* a,
//	uint512_dt* b,
//	uint512_dt* c,
	uint512_dt* d,
	uint512_dt* u,
	uint512_dt* acc_1,
	uint512_dt* acc_2,
	int M,
	int N,
	int B,
	int iters){

//	#pragma HLS INTERFACE depth=4096 m_axi port = a offset = slave bundle = gmem0 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=16 num_write_outstanding=16 latency=64
//	#pragma HLS INTERFACE depth=4096 m_axi port = b offset = slave bundle = gmem1 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=16 num_write_outstanding=16 latency=64
//	#pragma HLS INTERFACE depth=4096 m_axi port = c offset = slave bundle = gmem2 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=16 num_write_outstanding=16 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = d offset = slave bundle = gmem3 max_read_burst_length=8 max_write_burst_length=8 num_read_outstanding=64 num_write_outstanding=64 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = u offset = slave bundle = gmem3 max_read_burst_length=8 max_write_burst_length=8 num_read_outstanding=64 num_write_outstanding=64 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = acc_1 offset = slave bundle = gmem4 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8 latency=64
	#pragma HLS INTERFACE depth=4096 m_axi port = acc_2 offset = slave bundle = gmem4 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8 latency=64

//	#pragma HLS INTERFACE s_axilite port = a bundle = control
//	#pragma HLS INTERFACE s_axilite port = b bundle = control
//	#pragma HLS INTERFACE s_axilite port = c bundle = control
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
//		pre_process(u, /*a, b, c,*/ d, acc_1, acc_2, M, N, B, dnt_acc_updt);

		TDMA(u, d, acc_1, acc_2, M, N, B, B,  0, dnt_acc_updt);
		TDMA(d, u, acc_2, acc_1, M, N, B, 0,  1, 0);

		TDMA(u, d, acc_2, acc_1, M, N, B, B, 0, 0);
		TDMA(d, u, acc_1, acc_2, M, N, B, 0, 1, 0);


//		pre_process(d, /*a, b, c,*/ u, acc_2, acc_1,  M, N, B, 0);
//		TDMA(/*a, b, c,*/ u, d, M, N, B, 0);
//		TDMA(/*a, b, c,*/ d, u, M, N, B, 1);
//		pre_process(u, a, b, c, d, acc_2, acc_1,  M, N, B);
	}

}
}
