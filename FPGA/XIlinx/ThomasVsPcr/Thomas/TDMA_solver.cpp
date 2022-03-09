#include <stdio.h>
#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
#include "pre_proc.cpp"



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

static void HBM_fifo(uint512_dt*HBM_buffer1, uint512_dt*HBM_buffer2, hls::stream<uint512_dt> &stm_in, hls::stream<uint512_dt> &stm_out,
		int count_w, int count_b, int count_r){

	int iter = ((count_b + 2* count_w)) >> 1;
	int count_w_N = (count_w >> 1);
	int count_b_N = (count_b >> 1);

	for(int i = 0; i < iter; i++){
		#pragma HLS dependence  variable=HBM_buffer1 WAR distance=1000 true
		#pragma HLS dependence  variable=HBM_buffer2 WAR distance=1000 true
		#pragma HLS loop_tripcount min=60544 max=200000 avg=200000
		#pragma HLS PIPELINE II=2
		uint512_dt tmp_r1, tmp_r2;
		uint512_dt tmp_w1, tmp_w2;

		tmp_r1 =  HBM_buffer1[i];
		tmp_r2 =  HBM_buffer2[i];

		if(i >= count_w_N){
			stm_out << tmp_r1;
			stm_out << tmp_r2;
		}


		if(i < count_w_N + count_b_N){
			tmp_w1 = stm_in.read();
			tmp_w2 = stm_in.read();
		}

		HBM_buffer1[i + count_w_N] = tmp_w1;
		HBM_buffer2[i + count_w_N] = tmp_w2;
	}
//	printf("HBM_fifo: i didn't get stuck\n");
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

static void interleaved_row_col(hls::stream<uint256_dt> &in, hls::stream<uint256_dt> &out,
		ap_uint<12> M, ap_uint<12> N, ap_uint<14> B, unsigned char dirXYZ){

	ap_uint<12> TileX, TileY;
	ap_uint<20> NTiles;
	ap_uint<8> XBlocks = (M >> 3);
	unsigned int offset;
	const int N_CU = 8;

	TileX = XBlocks;
	TileY = N;
	NTiles = B;

	uint256_dt tmp_M[DIM_MAX*DIM_MAX/8*2];
	#pragma HLS RESOURCE variable=tmp_M core=XPM_MEMORY uram latency=2
	uint256_dt tmp;

	ap_uint<18> NTilesp1 = register_it<int>(B+1);
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
		bool cmp = dirXYZ == 1 || (i*TileY + j < B*N);
		int indW = k*TileY+j;
		int indR = j*TileX + k;
		unsigned int offsetR = ((i & 1) == 0) ?  DIM_MAX*DIM_MAX/8 : 0;
		unsigned int offsetW = ((i & 1) == 0) ?  0 : DIM_MAX*DIM_MAX/8;
		if(cmp && i < B){
			tmp = in.read();
		}



		tmp_M[indW+offsetW] = tmp;
		uint256_dt tmp_R = tmp_M[indR+offsetR];
		if(i > 0){
			out << tmp_R;
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


static void interleaved_col_row(hls::stream<uint256_dt> &in, hls::stream<uint256_dt> &out0,
		ap_uint<12> M, ap_uint<12> N, ap_uint<14> B, unsigned char dirXYZ){


	ap_uint<12> TileX, TileY;
	ap_uint<20> NTiles;
	ap_uint<8> XBlocks = (M >> 3);
	unsigned int offset;
	const int N_CU = 8;

	TileX = XBlocks;
	TileY = N;
	NTiles = B;


	uint256_dt tmp_M[DIM_MAX*DIM_MAX/8*2];
	#pragma HLS RESOURCE variable=tmp_M core=XPM_MEMORY uram latency=2
	uint256_dt tmp;

	ap_uint<18> NTilesp1 = register_it<int>(B+1);
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

		bool cmp = dirXYZ == 1 || (i*TileY + j < B*N);

		int indW = j*TileX + k;
		int indR = k*TileY+j;
		unsigned int offsetR = ((i & 1) == 0) ?  DIM_MAX*DIM_MAX/8 : 0;
		unsigned int offsetW = ((i & 1) == 0) ?  0 : DIM_MAX*DIM_MAX/8;

		if(i < B){
			tmp = in.read();
		}

		tmp_M[indW+offsetW] = tmp;
		uint256_dt tmp_R = tmp_M[indR+offsetR];
		if(i > 0  && cmp){
			out0 << tmp_R;
		}
	}
//	printf("undo_interleaved_row_block8: i didn't get stuck\n");
}


static void stream_8x8transpose(hls::stream<uint256_dt> &in, hls::stream<uint256_dt> &out0,
		ap_uint<12> M, ap_uint<12> N, ap_uint<14> B, unsigned char dirXYZ){

	ap_uint<12> TileX, TileY;
	ap_uint<32> NTiles;
	ap_uint<8> XBlocks = (M >> 3);
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


static void thomas_interleave(hls::stream<uint256_dt> &d_stm, hls::stream<uint256_dt> &d_fw_stm,
		ap_uint<12> d0, ap_uint<12> B, int ReadLimit){


	uint256_dt  d2[N_MAX*N_BLK*2];
	#pragma HLS RESOURCE variable=d2 core=XPM_MEMORY uram latency=2
	ap_uint<12> batd1 = 0;
	ap_uint<6> id1 =0;
	ap_uint<12> jd1 = 0;
	ap_uint<12> Bp1 = B+1;

	int total_itr =register_it<int> (Bp1*N_BLK*d0);
	loop_read: for(int itr= 0; itr < total_itr; itr++){
			#pragma HLS PIPELINE II=1
			#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

			ap_uint<12> bat = batd1;
			ap_uint<6> i = id1;
			ap_uint<12> j = jd1;

			if(j == d0 -1){
				jd1 = 0;
			} else {
				jd1++;
			}

			if(j == d0 -1 && i == N_BLK -1){
				id1 = 0;
				batd1++;
			} else if(j == d0 -1){
				id1++;
			}

			unsigned int offsetR = ((bat & 1) == 0) ?  N_MAX*N_BLK : 0;
			unsigned int offsetW = ((bat & 1) == 0) ?  0 : N_MAX*N_BLK;

			ap_uint<20> countr1 = register_it<int>((bat<<5) + i);
			int count = countr1 * d0 + j;
			uint256_dt  tmp_d;
			if(count < ReadLimit){
				tmp_d = d_stm.read();
			} else {
				tmp_d = 0;
			}
			int indW = j*N_BLK + i + offsetW;
			d2[indW] = tmp_d;

			int indR = i*d0+j + offsetR;
			uint256_dt  tmp_R = d2[indR];
			if(bat > 0){
				d_fw_stm << tmp_R;
			}
		}
}


static void thomas_forward(hls::stream<uint256_dt> &a_fw_stm, hls::stream<uint256_dt> &b_fw_stm,
		hls::stream<uint256_dt> &c_fw_stm, hls::stream<uint256_dt> &d_fw_stm,
		hls::stream<uint256_dt> &c2_fw_stm, hls::stream<uint256_dt> &d2_fw_stm,
		ap_uint<12> d0, ap_uint<12> B){


	uint256_dt  c2_fw[N_MAX*N_BLK*2];
	uint256_dt  d2_fw[N_MAX*N_BLK*2];

	#pragma HLS RESOURCE variable=c2_fw core=XPM_MEMORY uram latency=2
	#pragma HLS RESOURCE variable=d2_fw core=XPM_MEMORY uram latency=2


	ap_uint<12> batd2 = 0;
	ap_uint<12> id2 =0;
	ap_uint<6>  kd2 = 0;

	ap_uint<12> Bp1 = B+1;
	int total_itr =register_it<int> (Bp1*N_BLK*d0);

	uint256_dt window_b2[N_BLK], window_c2[N_BLK], window_d2[N_BLK];
	loop_fw: for(int itr= 0; itr < total_itr; itr++){
			#pragma HLS PIPELINE II=1
			#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

			ap_uint<12> bat = batd2;
			ap_uint<12> i = id2;
			ap_uint<6> k = kd2;

			if(k == N_BLK -1){
				kd2 = 0;
			} else {
				kd2++;
			}

			if(k == N_BLK -1 && i == d0 -1){
				id2 = 0;
				batd2++;
			} else if(k == N_BLK -1){
				id2++;
			}

			#pragma HLS dependence variable=window_b2 RAW distance=n_blk true
			#pragma HLS dependence variable=window_d2 RAW distance=n_blk true
			#pragma HLS dependence variable=window_c2 RAW distance=n_blk true


			uint256_dt a2_read = 0;
			uint256_dt b2_read = 0;
			uint256_dt c2_read = 0;
			uint256_dt d2_read = 0;
			if(bat < B){
				a2_read = a_fw_stm.read();
				b2_read = b_fw_stm.read();
				c2_read = c_fw_stm.read();
				d2_read = d_fw_stm.read();
			}

			uint256_dt vec_bb_r = window_b2[k];
			uint256_dt vec_dd_r = window_d2[k];
			uint256_dt vec_cc_r = window_c2[k];

			uint256_dt b2_fw_write, d2_fw_write;
			uint256_dt vec_bb_w, vec_dd_w, vec_cc_w;

			fw_vec_loop: for(int v =0; v < VEC_FACTOR; v++){
				#pragma HLS unroll
				float aa_read = uint2float_ript(a2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
				float bb_read = uint2float_ript(b2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
				float cc_read = uint2float_ript(c2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
				float dd_read = uint2float_ript(d2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));

				float bb_old = uint2float_ript(vec_bb_r.range(D_SIZE*(v+1)-1,D_SIZE*v));
				float dd_old = uint2float_ript(vec_dd_r.range(D_SIZE*(v+1)-1,D_SIZE*v));
				float cc_old = uint2float_ript(vec_cc_r.range(D_SIZE*(v+1)-1,D_SIZE*v));



				float denom = (i == 0) ? bb_read : (bb_read - aa_read*cc_old);
				float r = 1.0/denom;
				float c_w1 = cc_read;
				float d_w1 = (i == 0) ? dd_read : (dd_read - aa_read*dd_old);

				float b_wr = 1.0f;
				float c_wr = c_w1*r;
				float d_wr = d_w1*r;



				b2_fw_write.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(b_wr);
				d2_fw_write.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(d_wr);

				vec_bb_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(b_wr);
				vec_dd_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(d_wr);
				vec_cc_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(c_wr);

			}
			window_b2[k] = vec_bb_w;
			window_d2[k] = vec_dd_w;
			window_c2[k] = vec_cc_w;

			unsigned int offsetR = ((bat & 1) == 0) ?  N_MAX*N_BLK : 0;
			unsigned int offsetW = ((bat & 1) == 0) ?  0 : N_MAX*N_BLK;
			int indW =  k*d0+i+offsetW;
			c2_fw[indW] = vec_cc_w;
	//				b2_fw[ind] = b2_fw_write;
			d2_fw[indW] = d2_fw_write;

			int indR =  k*d0+ (d0-i -1) + offsetR;

			uint256_dt c2_fw_stmR = c2_fw[indR];
			uint256_dt d2_fw_stmR = d2_fw[indR];

			if(bat > 0){
				c2_fw_stm << c2_fw_stmR;
				d2_fw_stm << d2_fw_stmR;
			}


		}
}


static void thomas_backward(hls::stream<uint256_dt> &c2_fw_stm, hls::stream<uint256_dt> &d2_fw_stm, hls::stream<uint256_dt> &u_stm,
		ap_uint<12> d0, ap_uint<12> B, int ReadLimit){

	uint256_dt  u2[N_MAX*N_BLK*2];
	#pragma HLS RESOURCE variable=u2 core=XPM_MEMORY uram = latency=2

	uint256_dt window_u2[N_BLK];

	ap_uint<12> batd3 = 0;
	ap_uint<12> id3 =0;
	ap_uint<6>  kd3 = 0;

	ap_uint<12> Bp1 = B+1;
	int total_itr =register_it<int> (Bp1*N_BLK*d0);

	loop_bw: for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<12> bat = batd3;
		ap_uint<12> id = id3;
		ap_uint<6> k = kd3;

		if(k == N_BLK -1){
			kd3 = 0;
		} else {
			kd3++;
		}

		if(k == N_BLK -1 && id == d0 -1){
			id3 = 0;
			batd3++;
		} else if(k == N_BLK -1){
			id3++;
		}

		#pragma HLS dependence variable=window_u2 RAW distance=n_blk true
		ap_uint<12> i = d0 -1 -id;
		uint256_dt d2_fw_read = 0;
		uint256_dt c2_fw_read = 0;
		if(bat < B){
			d2_fw_read = d2_fw_stm.read();
			c2_fw_read = c2_fw_stm.read();
		}
		uint256_dt u2_write;

		uint256_dt vec_u2_r = window_u2[k];
		uint256_dt vec_u2_w;
		bw_vec_loop: for(int v = 0; v < VEC_FACTOR; v++){
			#pragma HLS unroll
//					float bb_read = uint2float_ript(b2_fw_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float dd_read = uint2float_ript(d2_fw_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float cc_read = uint2float_ript(c2_fw_read.range(D_SIZE*(v+1)-1,D_SIZE*v));

			float u_pre = uint2float_ript(vec_u2_r.range(D_SIZE*(v+1)-1,D_SIZE*v));
			float numer_l = dd_read;
			float numer_o = (dd_read - cc_read * u_pre);
			float numer = (i == d0-1) ? numer_l : numer_o;

			float u_new = numer;
			u2_write.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(u_new);
			vec_u2_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(u_new);
		}

		unsigned int offsetR = ((bat & 1) == 0) ?  N_MAX*N_BLK : 0;
		unsigned int offsetW = ((bat & 1) == 0) ?  0 : N_MAX*N_BLK;

		int indW = k* d0 + i + offsetW;
		int indR = id*N_BLK+k + offsetR;

		u2[indW] = u2_write;
		uint256_dt u_stm_R = u2[indR];
		int count = (bat-1) * N_BLK * d0 + id*N_BLK+k;
		if(bat >0 && count < ReadLimit){
			u_stm << u_stm_R;
		}
		window_u2[k] = vec_u2_w;
	}

}






//static void TDMA_comp(hls::stream<uint256_dt> &d_stm, hls::stream<uint256_dt> &u_stm,
//		ap_uint<12> Xdim, ap_uint<12> Ydim, ap_uint<14> Batch, unsigned char dirXYZ){
//
//
//	ap_uint<12> TileX_TC, TileY_TC;
//	unsigned int ReadLimit_X, ReadLimit_Y;
//	ap_uint<9> XBlocks_TC = (Xdim >> 3);
//
//	ReadLimit_X = ((Batch*Ydim+7)>>3)*(XBlocks_TC << 3);
//	ReadLimit_Y = XBlocks*Batch*Ydim ;
//
//	int B_X = (Batch*Ydim+255)>>8;
//	int B_Y = ((Xdim*Batch+255)>>8);
//
//
//
//	hls::stream<uint256_dt> d_fw_stm;
//	hls::stream<uint256_dt> c2_fw_stm;
//	hls::stream<uint256_dt> d2_fw_stm;
//
//	#pragma HLS STREAM variable = d_fw_stm depth = 2
//	#pragma HLS STREAM variable = c2_fw_stm depth = 2
//	#pragma HLS STREAM variable = d2_fw_stm depth = 2
//
//	#pragma HLS DATAFLOW
//
//	thomas_interleave(d_stm, d_fw_stm, d0, B, ReadLimit);
//	thomas_forward(d_fw_stm, c2_fw_stm, d2_fw_stm, d0, B);
//	thomas_backward(c2_fw_stm, d2_fw_stm, u_stm, d0, B, ReadLimit);
//
//
//}

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

	int B_X = (B*N+255)>>8;
	int B_Y = ((M*B+255)>>8);


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

	thomas_interleave(a_stm_0[3], abcd_fw_stm[0], M, B_X, ReadLimit_X);
	thomas_interleave(b_stm_0[3], abcd_fw_stm[1], M, B_X, ReadLimit_X);
	thomas_interleave(c_stm_0[3], abcd_fw_stm[2], M, B_X, ReadLimit_X);
	thomas_interleave(d_stm_0[3], abcd_fw_stm[3], M, B_X, ReadLimit_X);

	thomas_forward(abcd_fw_stm[0], abcd_fw_stm[1], abcd_fw_stm[2], abcd_fw_stm[3], c2_fw_stm[0], d2_fw_stm[0], M, B_X);
	thomas_backward(c2_fw_stm[0], d2_fw_stm[0], u_stm_0[0], M, B_X, ReadLimit_X);
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



	for(ap_uint<12> itr = 0; itr < iters; itr++){
		#pragma HLS loop_tripcount min=1 max=100 avg=10
		#pragma HLS dependence variable=d intra RAW true
		#pragma HLS dependence variable=u intra RAW true
		bool dnt_acc_updt = (itr == 0) ? 1 : 0;
		TDMA1(a, b, c, d, u,  M, N, B);
	}

}
}
