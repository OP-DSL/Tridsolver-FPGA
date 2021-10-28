#include <stdio.h>
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

static void read_coeff(const uint512_dt*d, hls::stream<uint256_dt> &d_stm,
		ap_uint<9> M, ap_uint<9> N, ap_uint<9> L, ap_uint<10> B, unsigned char dirXYZ){

	// dirXYZ 0-X 1-Y 2-Z

	ap_uint<9> d0,d1,d2;
	ap_uint<24> off_d0, off_d1, off_d2;
	switch(dirXYZ){
		case 0 : {d0 = (M>>4); d1 = N; d2 = L; off_d0 = 1; off_d1 = (M>>4); off_d2 = N*(M>>4); break;}
		case 1 : {d0 = (M>>4); d1 = N; d2 = L; off_d0 = 1; off_d1 = (M>>4); off_d2 = N*(M>>4); break;}
		case 2 : {d0 = (M>>4); d1 = L; d2 = N; off_d0 = 1; off_d1 = N*(M>>4); off_d2 = (M>>4); break;}
		default :{d0 = (M>>4); d1 = N; d2 = L; off_d0 = 1; off_d1 = (M>>4); off_d2 = N*(M>>4); break;}
	}



	ap_uint<10> batd = 0;
	ap_uint<10> id =0;
	ap_uint<10> jd = 0;
	int total_itr = B*d2*d1;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<10> bat = batd;
		ap_uint<10> i = id;
		ap_uint<10> j = jd;


		if(j == d1 -1){
			jd = 0;
		} else {
			jd++;
		}

		if(j == d1 -1 && i == d2 -1){
			id = 0;
			batd++;
		} else if(j == d1 -1){
			id++;
		}
		int ind = bat*(M>>4)*N*L + j*off_d1 + i *off_d2;
		for(ap_uint<9> k = 0; k < d0; k++){
			#pragma HLS loop_tripcount min=2 max=16 avg=16
			#pragma HLS PIPELINE II=2
			uint512_dt tmp_d =  d[ind+k];

			d_stm << tmp_d.range(255,0);
			d_stm << tmp_d.range(511,256);
		}

	}

}


static void interleaved_row_block8(hls::stream<uint256_dt> &stm_in, hls::stream<uint256_dt> &stm_out,
		ap_uint<9> M, ap_uint<9> N, ap_uint<9> L, ap_uint<14> B){

	ap_uint<12> TileX, TileY;
	ap_uint<32> NTiles;
	ap_uint<8> XBlocks = (M >> 3);
	unsigned int offset;
	const int N_CU = 8;
	TileX=XBlocks;
	TileY=N_CU;
	NTiles = (B*N*L+7)>>3;

	uint256_dt tmp_M[N_MAX*N_CU*2];
	#pragma HLS RESOURCE variable=tmp_M core=XPM_MEMORY latency=2

	ap_uint<32> NTilesp1 = register_it<int>(NTiles+1);
	ap_uint<32> id = 0;
	ap_uint<5> jd =0;
	ap_uint<12> kd = 0;
	int total_itr = NTilesp1*TileX*TileY;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<32> i = id;
		ap_uint<5> j = jd;
		ap_uint<12> k = kd;

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
		unsigned int offsetR = ((i & 1) == 0) ?  N_MAX*N_CU : 0;
		unsigned int offsetW = ((i & 1) == 0) ?  0 : N_MAX*N_CU;

		bool cmpW = (i*TileY + j < B*N*L);
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
		ap_uint<9> M, ap_uint<9> N, ap_uint<9> L, ap_uint<14> B){


	ap_uint<12> TileX, TileY;
	ap_uint<32> NTiles;
	ap_uint<8> XBlocks = (M >> 3);
	unsigned int offset;
	const int N_CU = 8;
	TileX=XBlocks;
	TileY=N_CU;
	NTiles = (B*N*L+7)>>3;

	uint256_dt tmp_M[N_MAX*N_CU*2];
	#pragma HLS RESOURCE variable=tmp_M core=XPM_MEMORY latency=2
	ap_uint<32> NTilesp1 = register_it<int>(NTiles+1);

	ap_uint<32> id = 0;
	ap_uint<5> jd =0;
	ap_uint<12> kd = 0;
	int total_itr = NTilesp1*TileX*TileY;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<32> i = id;
		ap_uint<5> j = jd;
		ap_uint<12> k = kd;

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
		unsigned int offsetR = ((i & 1) == 0) ?  N_MAX*N_CU : 0;
		unsigned int offsetW = ((i & 1) == 0) ?  0 : N_MAX*N_CU;

		bool cmpW = i*TileY + j < B*N*L;
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





static void interleaved_row_col(hls::stream<uint256_dt> &in, hls::stream<uint256_dt> &out,
		ap_uint<9> M, ap_uint<9> N, ap_int<9>L, ap_uint<14> B){

	ap_uint<9> TileX, TileY;
	ap_uint<32> NTiles;
	ap_uint<8> XBlocks = (M >> 3);
	unsigned int offset;
	const int N_CU = 8;

	TileX = XBlocks;
	TileY = N;
	NTiles = B*L;

	uint256_dt tmp_M[N_MAX*N_MAX/8*2];
	#pragma HLS RESOURCE variable=tmp_M core=XPM_MEMORY uram latency=2
	uint256_dt tmp;

	ap_uint<32> NTilesp1 = register_it<int>(NTiles+1);
	ap_uint<32> id = 0;
	ap_uint<9> jd =0, kd = 0;
	int total_itr = NTilesp1*TileX*TileY;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<32> i = id;
		ap_uint<9> j = jd, k = kd;

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
		bool cmp = (i*TileY + j < B*N*L);
		int indW = k*TileY+j;
		int indR = j*TileX + k;
		unsigned int offsetR = ((i & 1) == 0) ?  N_MAX*N_MAX/8 : 0;
		unsigned int offsetW = ((i & 1) == 0) ?  0 : N_MAX*N_MAX/8;
		if(cmp && i < NTiles){
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


static void interleaved_col_row(hls::stream<uint256_dt> &in, hls::stream<uint256_dt> &out0,
		ap_uint<9> M, ap_uint<9> N, ap_uint<9> L, ap_uint<14> B){


	ap_uint<9> TileX, TileY;
	ap_uint<32> NTiles;
	ap_uint<8> XBlocks = (M >> 3);
	unsigned int offset;
	const int N_CU = 8;

	TileX = XBlocks;
	TileY = N;
	NTiles = B*L;


	uint256_dt tmp_M[N_MAX*N_MAX/8*2];
	#pragma HLS RESOURCE variable=tmp_M core=XPM_MEMORY uram latency=2
	uint256_dt tmp;

	ap_uint<32> NTilesp1 = register_it<int>(NTiles+1);
	ap_uint<32> id = 0;
	ap_uint<9> jd =0, kd = 0;
	int total_itr = NTilesp1*TileX*TileY;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<32> i = id;
		ap_uint<9> j = jd, k = kd;

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

		bool cmp = i*TileY + j < B*N*L;

		int indW = j*TileX + k;
		int indR = k*TileY+j;
		unsigned int offsetR = ((i & 1) == 0) ?  N_MAX*N_MAX/8 : 0;
		unsigned int offsetW = ((i & 1) == 0) ?  0 : N_MAX*N_MAX/8;

		if(i < NTiles){
			tmp = in.read();
		}

		tmp_M[indW+offsetW] = tmp;
		bool cmpW = (i-1)*TileY + j < B*N*L;
		uint256_dt tmp_R = tmp_M[indR+offsetR];
		if(i > 0  && cmpW){
			out0 << tmp_R;
		}
	}
//	printf("undo_interleaved_row_block8: i didn't get stuck\n");
}


static void stream_8x8transpose(hls::stream<uint256_dt> &in, hls::stream<uint256_dt> &out0,
		ap_uint<12> M, ap_uint<12> N, ap_uint<12> L, ap_uint<12> B){


	unsigned int total_blocks = (unsigned int)((M >> 3)*((N*L*B+7)>>3));
	unsigned int total_read_c =(unsigned int)((M >> 3)*((N*L*B+7)>>3)*8);


	loop_8x8_transpose: for(unsigned int itr=0; itr < total_blocks; itr++){
		#pragma HLS loop_tripcount min=256 max=26214400 avg=26214400
		#pragma HLS PIPELINE II=8
		uint256_dt tmp[8], outR;
		for(int i = 0; i < 8; i++){
			if((itr<<3) + i < total_read_c){
				tmp[i] = in.read();
			}
		}

		for(int i = 0; i < 8; i++){
			outR.range(D_SIZE*1-1, D_SIZE*0) = tmp[0].range(D_SIZE*(i+1)-1,D_SIZE*i);
			outR.range(D_SIZE*2-1, D_SIZE*1) = tmp[1].range(D_SIZE*(i+1)-1,D_SIZE*i);
			outR.range(D_SIZE*3-1, D_SIZE*2) = tmp[2].range(D_SIZE*(i+1)-1,D_SIZE*i);
			outR.range(D_SIZE*4-1, D_SIZE*3) = tmp[3].range(D_SIZE*(i+1)-1,D_SIZE*i);
			outR.range(D_SIZE*5-1, D_SIZE*4) = tmp[4].range(D_SIZE*(i+1)-1,D_SIZE*i);
			outR.range(D_SIZE*6-1, D_SIZE*5) = tmp[5].range(D_SIZE*(i+1)-1,D_SIZE*i);
			outR.range(D_SIZE*7-1, D_SIZE*6) = tmp[6].range(D_SIZE*(i+1)-1,D_SIZE*i);
			outR.range(D_SIZE*8-1, D_SIZE*7) = tmp[7].range(D_SIZE*(i+1)-1,D_SIZE*i);

			if((itr<<3) + i < total_read_c){
				out0 << outR;
			}
		}

	}
}


static void thomas_interleave(hls::stream<uint256_dt> &d_stm, hls::stream<uint256_dt> &d_fw_stm,
		ap_uint<9> d0, ap_uint<24> B, int ReadLimit){


	uint256_dt  d2[N_MAX*N_BLK*2];
	#pragma HLS RESOURCE variable=d2 core=XPM_MEMORY uram latency=2
	ap_uint<24> batd1 = 0;
	ap_uint<6> id1 =0;
	ap_uint<12> jd1 = 0;
	ap_uint<24> Bp1 = B+1;

	int total_itr =register_it<int> (Bp1*N_BLK*d0);
	loop_read: for(int itr= 0; itr < total_itr; itr++){
			#pragma HLS PIPELINE II=1
			#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

			ap_uint<24> bat = batd1;
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

			ap_uint<32> countr1 = (bat*N_BLK) + i;
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


static void thomas_forward(hls::stream<uint256_dt> &d_fw_stm, hls::stream<uint256_dt> &c2_fw_stm, hls::stream<uint256_dt> &d2_fw_stm,
		ap_uint<12> d0, ap_uint<24> B){


	uint256_dt  c2_fw[N_MAX*N_BLK*2];
	uint256_dt  d2_fw[N_MAX*N_BLK*2];

	#pragma HLS RESOURCE variable=c2_fw core=XPM_MEMORY uram latency=2
	#pragma HLS RESOURCE variable=d2_fw core=XPM_MEMORY uram latency=2


	ap_uint<24> batd2 = 0;
	ap_uint<12> id2 =0;
	ap_uint<6>  kd2 = 0;

	ap_uint<24> Bp1 = B+1;
	int total_itr =register_it<int> (Bp1*N_BLK*d0);

	uint256_dt window_b2[N_BLK], window_c2[N_BLK], window_d2[N_BLK];
	loop_fw: for(int itr= 0; itr < total_itr; itr++){
			#pragma HLS PIPELINE II=1
			#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

			ap_uint<24> bat = batd2;
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


			uint256_dt d2_read = 0;
			if(bat < B){
				d2_read = d_fw_stm.read();
			}

			uint256_dt vec_bb_r = window_b2[k];
			uint256_dt vec_dd_r = window_d2[k];
			uint256_dt vec_cc_r = window_c2[k];

			uint256_dt b2_fw_write, d2_fw_write;
			uint256_dt vec_bb_w, vec_dd_w, vec_cc_w;

			fw_vec_loop: for(int v =0; v < VEC_FACTOR; v++){
				#pragma HLS unroll
				float aa_read = (i == 0 || i == d0 -1) ? 0.0f : -0.5f;
				float bb_read = (i == 0 || i == d0 -1) ? 1.0f :  2.0f;
				float cc_read = (i == 0 || i == d0 -1) ? 0.0f : -0.5f;
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
		ap_uint<12> d0, ap_uint<24> B, int ReadLimit){

	uint256_dt  u2[N_MAX*N_BLK*2];
	#pragma HLS RESOURCE variable=u2 core=XPM_MEMORY uram = latency=2

	uint256_dt window_u2[N_BLK];

	ap_uint<24> batd3 = 0;
	ap_uint<12> id3 =0;
	ap_uint<6>  kd3 = 0;

	ap_uint<24> Bp1 = B+1;
	int total_itr =register_it<int> (Bp1*N_BLK*d0);

	loop_bw: for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<24> bat = batd3;
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

//
//static void TDMA_comp(hls::stream<uint256_dt> &d_stm, hls::stream<uint256_dt> &u_stm,
//		ap_uint<12> M, ap_uint<12> N, ap_uint<12> L, ap_uint<12> B, unsigned char dirXYZ){
//
//
//
//
//	unsigned int t_x = (((N*L*B+255) >> 8));
//	unsigned int t_y = (((L*M*B+255) >> 8));
//	unsigned int t_z = (((M*N*B+255) >> 8));
//
//	unsigned int t_yz = (dirXYZ == 1? t_y : t_z);
//	unsigned int total_u_loop = (dirXYZ == 0 ? t_x : t_yz);
//	unsigned char d0 = (dirXYZ == 0 ? M : (dirXYZ == 1? N: L));
//
//
//	unsigned int Read_limit = (dirXYZ == 0 ? (unsigned int)((M >> 3)*((N*L*B+7)>>3) << 3) : (unsigned int)(((M>>3)*N*L*B+7)>>3) << 3);
//
//
//
//	uint256_dt  c2_fw[N_MAX*N_BLK];
//	uint256_dt  d2[N_MAX*N_BLK];
//	uint256_dt  u2[N_MAX*N_BLK];
//
//	uint256_dt d2_fw[N_MAX*N_BLK];
//
//
//	#pragma HLS RESOURCE variable=d2_fw core=XPM_MEMORY uram
//	#pragma HLS RESOURCE variable=u2 core=XPM_MEMORY uram
//	#pragma HLS RESOURCE variable=c2_fw core=XPM_MEMORY uram
//	#pragma HLS RESOURCE variable=d2 core=XPM_MEMORY uram
//
//	for(int bat=0; bat < total_u_loop; bat++){
//		#pragma HLS loop_tripcount min=512 max=25600 avg=25600
//		#pragma HLS DATAFLOW
//		loop_r: for(ap_uint<6> i=0; i < N_BLK; i++){
//			for(ap_uint<12> j = 0; j < d0; j++){
//				#pragma HLS loop_tripcount min=32 max=256 avg=256
//				#pragma HLS PIPELINE II=1
//				int count = (bat * N_BLK + i) * d0 + j;
//				uint256_dt  tmp_d;
//				if(count < Read_limit){
//					tmp_d = d_stm.read();
//				} else {
//					tmp_d = 0;
//				}
//
//				d2[i*d0+j] = tmp_d;
//			}
//		}
//
//		uint256_dt window_b2[N_BLK], window_c2[N_BLK], window_d2[N_BLK];
//		loop_forward: for(ap_uint<12> i=0; i < d0; i++){
//			#pragma HLS loop_tripcount min=32 max=256 avg=256
//			for(ap_uint<6> k = 0; k < N_BLK; k++){
//				#pragma HLS PIPELINE II=1
//				#pragma HLS dependence variable=window_b2 RAW distance=n_blk true
//				#pragma HLS dependence variable=window_d2 RAW distance=n_blk true
//				#pragma HLS dependence variable=window_c2 RAW distance=n_blk true
//
//				int ind =  k*d0+i;
//				uint256_dt d2_read = d2[ind];
//
//				uint256_dt vec_bb_r = window_b2[k];
//				uint256_dt vec_dd_r = window_d2[k];
//				uint256_dt vec_cc_r = window_c2[k];
//
//				uint256_dt b2_fw_write, d2_fw_write;
//				uint256_dt vec_bb_w, vec_dd_w, vec_cc_w;
//
//				fw_vec_loop: for(int v =0; v < VEC_FACTOR; v++){
//					#pragma HLS unroll
//					float aa_read = (i == 0 || i == d0 -1) ? 0.0f : -0.5f; // uint2float_ript(a2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
//					float bb_read = (i == 0 || i == d0 -1) ? 1.0f :  2.0f; //uint2float_ript(b2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
//					float cc_read = (i == 0 || i == d0 -1) ? 0.0f : -0.5f; //uint2float_ript(c2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
//					float dd_read = uint2float_ript(d2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
//
//					float bb_old = uint2float_ript(vec_bb_r.range(D_SIZE*(v+1)-1,D_SIZE*v));
//					float dd_old = uint2float_ript(vec_dd_r.range(D_SIZE*(v+1)-1,D_SIZE*v));
//					float cc_old = uint2float_ript(vec_cc_r.range(D_SIZE*(v+1)-1,D_SIZE*v));
//
//
//
//					float denom = (i == 0) ? bb_read : (bb_read - aa_read*cc_old);
//					float r = 1.0/denom;
//					float c_w1 = cc_read;
//					float d_w1 = (i == 0) ? dd_read : (dd_read - aa_read*dd_old);
//
//					float b_wr = 1.0f;
//					float c_wr = c_w1*r;
//					float d_wr = d_w1*r;
//
//
//
//					b2_fw_write.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(b_wr);
//					d2_fw_write.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(d_wr);
//
//					vec_bb_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(b_wr);
//					vec_dd_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(d_wr);
//					vec_cc_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(c_wr);
//
//				}
//				window_b2[k] = vec_bb_w;
//				window_d2[k] = vec_dd_w;
//				window_c2[k] = vec_cc_w;
//
//				c2_fw[ind] = vec_cc_w;
//				d2_fw[ind] = d2_fw_write;
//			}
//		}
//
//		uint256_dt window_u2[N_BLK];
//		loop_backward: for(ap_uint<12> id = 0; id < d0; id++){
//			#pragma HLS loop_tripcount min=32 max=256 avg=256
//			for(ap_uint<6> k = 0; k < N_BLK; k++){
//				#pragma HLS PIPELINE II=1
//				#pragma HLS dependence variable=window_u2 RAW distance=n_blk true
//				ap_uint<12> i = d0 -1 -id;
//
//				int ind = k* d0 + i;
//				uint256_dt d2_fw_read = d2_fw[ind];
//				uint256_dt c2_fw_read = c2_fw[ind];
//				uint256_dt u2_write;
//
//				uint256_dt vec_u2_r = window_u2[k];
//				uint256_dt vec_u2_w;
//				bw_vec_loop: for(int v = 0; v < VEC_FACTOR; v++){
//					#pragma HLS unroll
//					float dd_read = uint2float_ript(d2_fw_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
//					float cc_read = uint2float_ript(c2_fw_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
//
//					float u_pre = uint2float_ript(vec_u2_r.range(D_SIZE*(v+1)-1,D_SIZE*v));
//					float numer_l = dd_read;
//					float numer_o = (dd_read - cc_read * u_pre);
//					float numer = (i == d0-1) ? numer_l : numer_o;
//
//					float u_new = numer;
//					u2_write.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(u_new);
//					vec_u2_w.range(D_SIZE*(v+1)-1,D_SIZE*v) = float2uint_ript(u_new);
//				}
//
//				u2[ind] = u2_write;
//				window_u2[k] = vec_u2_w;
//			}
//		}
//
//		loop_w: for(int itr=0; itr < d0*N_BLK; itr++){
//			#pragma HLS loop_tripcount min=1024 max=8192 avg=8192
//			#pragma HLS PIPELINE II=1
//			int count = bat * N_BLK * d0 + itr;
//			if(count < Read_limit){
//				u_stm << u2[itr];
//			}
//
//		}
//
//
//	}
//}

static void write_u(uint512_dt* u, hls::stream<uint256_dt> &u_stm,
		ap_uint<9> M, ap_uint<9> N, ap_uint<9> L, ap_uint<10> B, unsigned char dirXYZ){

	ap_uint<9> d0,d1,d2;
	ap_uint<24> off_d0, off_d1, off_d2;
	switch(dirXYZ){
		case 0 : {d0 = (M>>4); d1 = N; d2 = L; off_d0 = 1; off_d1 = (M>>4); off_d2 = N*(M>>4); break;}
		case 1 : {d0 = (M>>4); d1 = N; d2 = L; off_d0 = 1; off_d1 = (M>>4); off_d2 = N*(M>>4); break;}
		case 2 : {d0 = (M>>4); d1 = L; d2 = N; off_d0 = 1; off_d1 = N*(M>>4); off_d2 = (M>>4); break;}
		default :{d0 = (M>>4); d1 = N; d2 = L; off_d0 = 1; off_d1 = (M>>4); off_d2 = N*(M>>4); break;}
	}

	ap_uint<10> batd = 0;
	ap_uint<10> id =0;
	ap_uint<10> jd = 0;
	int total_itr = B*d2*d1;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<10> bat = batd;
		ap_uint<10> i = id;
		ap_uint<10> j = jd;


		if(j == d1 -1){
			jd = 0;
		} else {
			jd++;
		}

		if(j == d1 -1 && i == d2 -1){
			id = 0;
			batd++;
		} else if(j == d1 -1){
			id++;
		}
		int ind = bat*(M>>4)*N*L + j*off_d1 + i *off_d2;
		for(ap_uint<9> k = 0; k < d0; k++){
			#pragma HLS loop_tripcount min=2 max=16 avg=16
			#pragma HLS PIPELINE II=2
			uint512_dt tmp;
			tmp.range(255,0) = u_stm.read();
			tmp.range(511,256) = u_stm.read();
			u[ind+k] = tmp;
		}
	}

}

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



    struct data_G data_g;
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

	read_u(d, streamArray[0], acc_1, accStream[0],  M, N, L, B);
//	printf("finished reading TDMA_pre_XY\n");
	process_tile(streamArray[0], d_stm[0], accStream[0], accStream[1], data_g, dnt_update);


	// Xdim computation
	interleaved_row_block8(d_stm[0], d_stm[1], M, N, L, B);
	stream_8x8transpose(d_stm[1], d_stm[2], M, N, L, B);

	thomas_interleave(d_stm[2], d_fw_stm[0], M, t_x, Read_limit_X);
	thomas_forward(d_fw_stm[0], c2_fw_stm[0], d2_fw_stm[0], M, t_x);
	thomas_backward(c2_fw_stm[0], d2_fw_stm[0], u_stm[0], M, t_x, Read_limit_X);
//	TDMA_comp(d_stm[2], u_stm[0], M, N, L, B, 0);
	stream_8x8transpose(u_stm[0], u_stm[1], M, N, L, B);
	undo_interleaved_row_block8(u_stm[1], u_stm[2], M, N, L, B);


	// Ydim computation
	interleaved_row_col(u_stm[2], u_stm[3], M, N, L, B);

	thomas_interleave(u_stm[3], d_fw_stm[1], N, t_y, Read_limit_Y);
	thomas_forward(d_fw_stm[1], c2_fw_stm[1], d2_fw_stm[1], N, t_y);
	thomas_backward(c2_fw_stm[1], d2_fw_stm[1], u_stm[4], N, t_y, Read_limit_Y);
//	TDMA_comp(u_stm[3], u_stm[4], M, N, L, B, 1);
	interleaved_col_row(u_stm[4], u_stm[5], M, N, L, B);


	write_u(u, u_stm[5], M, N, L, B, 1);
	write_u(acc_2, accStream[1], M, N, L, B, 0);


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
	read_coeff(d, d_stm[0], M, N, L, B, dirXYZ);
//	printf("finished reading Z\n");

	interleaved_row_col(d_stm[0], d_stm[1], M, L, N, B);

	thomas_interleave(d_stm[1], d_fw_stm[0], L, t_z, Read_limit_Z);
	thomas_forward(d_fw_stm[0], c2_fw_stm[0], d2_fw_stm[0], L, t_z);
	thomas_backward(c2_fw_stm[0], d2_fw_stm[0], u_stm[0], L, t_z, Read_limit_Z);
//	TDMA_comp(d_stm[1], u_stm[0], M, N, L, B, dirXYZ);
	interleaved_col_row(u_stm[0], u_stm[3], M, L, N, B);

	write_u(u, u_stm[3], M, N, L, B, dirXYZ);


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
