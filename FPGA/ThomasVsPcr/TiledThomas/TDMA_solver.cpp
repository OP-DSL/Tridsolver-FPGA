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
	ap_uint<9> XBlocks = (M >> 3);
	unsigned int offset;
	const int N_CU = 8;
	switch(dirXYZ){
		case 0: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+7)>>3; break;}
		case 1: {TileX=8; TileY=N; NTiles = (((XBlocks*B+7)>>3)); break;}
		default: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+7)>>3; break;}
	}

	uint256_dt tmp_M[DIM_MAX*2];
	#pragma HLS RESOURCE variable=tmp_M core=XPM_MEMORY uram latency=2

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
		unsigned int offsetR = ((i & 1) == 0) ?  DIM_MAX : 0;
		unsigned int offsetW = ((i & 1) == 0) ?  0 : DIM_MAX;

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
	ap_uint<9> XBlocks = (M >> 3);
	unsigned int offset;
	const int N_CU = 8;
	switch(dirXYZ){
		case 0: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+7)>>3; break;}
		case 1: {TileX=8; TileY=N; NTiles = (((XBlocks*B+7)>>3)); break;}
		default: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+7)>>3; break;}
	}
	uint256_dt tmp_M[DIM_MAX*2];
	#pragma HLS RESOURCE variable=tmp_M core=XPM_MEMORY uram latency=2
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
		unsigned int offsetR = ((i & 1) == 0) ?  DIM_MAX : 0;
		unsigned int offsetW = ((i & 1) == 0) ?  0 : DIM_MAX;

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
		printf("Itr:%d\n", itr);
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





static void TT_STAGE0( hls::stream<uint256_dt> &a_stm_in,  hls::stream<uint256_dt> &b_stm_in,  hls::stream<uint256_dt> &c_stm_in,  hls::stream<uint256_dt> &d_stm_in,
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
			tmp_a = a_stm_in.read();
			tmp_b = b_stm_in.read();
			tmp_c = c_stm_in.read();
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
//			ap_uint<12> d0 = (i % Ti)*M + j;
//			ap_uint<12> sys_size = Ti*M;
//
//
//			tmp_a_f[v] = (d0 == 0 || d0 == sys_size-1) ? 0.0f : -0.5f;
//			tmp_c_f[v] = (d0 == 0 || d0 == sys_size-1) ? 0.0f : -0.5f;
//			tmp_b_f[v] = (d0 == 0 || d0 == sys_size-1) ? 1.0f :  2.0f;

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


	// thomas solver parameters
	ap_uint<12> TileX_TC, TileY_TC;
	unsigned int ReadLimit_X, ReadLimit_Y;
	ap_uint<12> XBlocks = (M >> 3);

	unsigned int ReadLimit = ((B*N+7)>>3)*(XBlocks << 3);
	ReadLimit_Y = XBlocks*B*N ;

//	int B_X = (B*N+255)>>8;
//	int B_Y = ((M*B+255)>>8);

	 //(B*N+7)>>3;

	ap_uint<4> logn;

	unsigned char Ti, Sys;
	unsigned char chk = (M >> 8);
	switch(chk){
		case 0: {Sys = 4; Ti = 8; logn = 4; break;}
		case 1: {Sys = 4; Ti = 8; logn = 4; break;}
		case 2: {Sys = 2; Ti = 16; logn = 5; break;}
		case 3: {Sys = 2; Ti = 16; logn = 5; break;}
		default: {Sys = 1; Ti = 32; logn = 6; break;}
	}

	int N_sys = ((B*N+31)>>5)<<2;

	const short Tiles = 32;
	unsigned char TileM = M/Ti;
	ap_uint<24> B_size = N_sys /Sys;
	int Rn = (Ti << 1);
	const ap_uint<24> R_size = ((ap_uint<24>)B_size<< 6);
	const ap_uint<24> R_size_half = ((ap_uint<24>)B_size<< 5);
	const ap_uint<8> R_systems = (Sys << 3);


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

	printf("Finished 8x8 transpose \n");

	//interleave the systems
	TT_STAGE0(a_stm_0[3], b_stm_0[3], c_stm_0[3], d_stm_0[3],
			  STAGE_0[0], STAGE_0[1], STAGE_0[2],  STAGE_0[3],
			  B_size, Tiles, TileM, Sys, Ti, ReadLimit);
		printf("finished stage 0\n");

	// FW sweep
	TT_STAGE1(STAGE_0[0], STAGE_0[1], STAGE_0[2],  STAGE_0[3],
			 STAGE_1[0], STAGE_1[1], STAGE_1[2],
			 B_size, Tiles, TileM, Sys, Ti);
	printf("finished stage 1\n");


	// BW Sweep
	TT_STAGE2(STAGE_1[0], STAGE_1[1], STAGE_1[2],
			STAGE_2A[0], STAGE_2A[1], STAGE_2A[2],
			STAGE_2B[3], STAGE_2B[4], STAGE_2B[5], STAGE_2B[6],
			B_size, Tiles, TileM, Sys, Ti);
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

	printf("Finished steam conversion\n");

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
			STAGE_6[0], STAGE_6[1], u_stm_0[0],
			B_size, Tiles, TileM, ReadLimit);

	printf("B_size:%d, Tiles:%d, TileM:%d, ReadLimit:%d\n", (int)B_size, (int)Tiles, (int)TileM, ReadLimit);
	printf("finished stage 7\n");



	stream_8x8transpose(u_stm_0[0], u_stm_0[1], M, N, B, 0);
	printf("Finished 8x8 transpose \n");


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
