#include <stdio.h>
#include <ap_int.h>
#include <hls_stream.h>
#include "pre_proc.h"

// Preproc Modules
static void read_u(const uint512_dt* u, hls::stream<uint512_dt> &u_stm,
		const uint512_dt* acc, hls::stream<uint512_dt> &acc_stm,
		ap_uint<12> M, ap_uint<12> N, ap_uint<12> L, ap_uint<12> B){

	int total_itr = (M>>3)* N * L * B;
	for(int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid_2 max=max_grid_2 avg=avg_grid_2
		uint512_dt tmp = u[itr];
		uint512_dt tmp_acc = acc[itr];
		u_stm << tmp;
//		u_stm << tmp.range(511,256);

		acc_stm << tmp_acc;
//		acc_stm << tmp_acc.range(511,256);
	}

}

static void write_d(uint512_dt* d, hls::stream<uint512_dt> &d_stm,
		uint512_dt* acc, hls::stream<uint512_dt> &acc_stm,
		ap_uint<12> M, ap_uint<12> N, ap_uint<12> L, ap_uint<12> B){

	int total_itr = (M>>3)* N * L* B;
	for(int itr = 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=min_grid_2 max=max_grid_2 avg=avg_grid_2
		uint512_dt tmp_d, tmp_acc;

		tmp_d = d_stm.read();
//		tmp_d.range(511,256) = d_stm.read();

		tmp_acc = acc_stm.read();
//		tmp_acc.range(511,256) = acc_stm.read();

		d[itr] = tmp_d;
		acc[itr] = tmp_acc;

	}

}

static void process_tile( hls::stream<uint256_dt> &rd_buffer,  hls::stream<uint256_dt> &d,
		hls::stream<uint256_dt> &acc_in,  hls::stream<uint256_dt> &acc_out, struct data_G data_g, bool dnt_update){
	unsigned short xblocks = data_g.xblocks;
	unsigned short sizex = data_g.sizex;
	unsigned short sizey = data_g.sizey;
	unsigned short sizez = data_g.sizez;
	unsigned short limit_z = data_g.limit_z;
	unsigned short grid_sizey = data_g.grid_sizey;
	unsigned short grid_sizez = data_g.grid_sizez;

	unsigned int line_diff = data_g.line_diff;
	unsigned int plane_diff = data_g.plane_diff;
	unsigned int gridsize = data_g.gridsize_pr;

	unsigned int limit_read = data_g.gridsize_da;

	float s_1_1_2_arr[VEC_FACTOR];
	float s_1_2_1_arr[VEC_FACTOR];
	float s_1_1_1_arr[VEC_FACTOR*3];
	float s_1_0_1_arr[VEC_FACTOR];
	float s_1_1_0_arr[VEC_FACTOR];

	float mem_a[VEC_FACTOR];
	float mem_b[VEC_FACTOR];
	float mem_c[VEC_FACTOR];
	float mem_d[VEC_FACTOR];

	float acc_in_arr[VEC_FACTOR];
	float acc_out_arr[VEC_FACTOR];

	#pragma HLS ARRAY_PARTITION variable=s_1_1_2_arr complete dim=1
	#pragma HLS ARRAY_PARTITION variable=s_1_2_1_arr complete dim=1
	#pragma HLS ARRAY_PARTITION variable=s_1_1_1_arr complete dim=1
	#pragma HLS ARRAY_PARTITION variable=s_1_0_1_arr complete dim=1
	#pragma HLS ARRAY_PARTITION variable=s_1_1_0_arr complete dim=1
	#pragma HLS ARRAY_PARTITION variable=mem_d complete dim=1

	uint256_dt window_1[DEPTH_P];
	uint256_dt window_2[DEPTH_L];
	uint256_dt window_3[DEPTH_L];
	uint256_dt window_4[DEPTH_P];

	#pragma HLS RESOURCE variable=window_1 core=XPM_MEMORY uram latency=2
	#pragma HLS RESOURCE variable=window_2 core=RAM_1P_BRAM latency=2
	#pragma HLS RESOURCE variable=window_3 core=RAM_1P_BRAM latency=2
	#pragma HLS RESOURCE variable=window_4 core=XPM_MEMORY uram latency=2

	uint256_dt s_1_1_2, s_1_2_1, s_1_1_1, s_1_1_1_b, s_1_1_1_f, s_1_0_1, s_1_1_0;
	uint256_dt update_a, update_b, update_c, update_d;
	uint256_dt acc_in_vec, acc_out_vec;


	unsigned short i = 0, j = 0, k = 0;
	unsigned short j_p = 0, j_l = 0;
	for(unsigned int itr = 0; itr < gridsize; itr++) {
		#pragma HLS loop_tripcount min=min_grid max=max_grid avg=avg_grid
		#pragma HLS PIPELINE II=1
		bool cond_x = (k == xblocks);
		bool cond_y = (j == grid_sizey -1);
		bool cond_z = (i == limit_z - 1);

		if(k == xblocks){
			k = 0;
		}

		if(cond_y && cond_x){
			j = 0;
		}else if(cond_x){
			j++;
		}

		if(cond_x && cond_y && cond_z){
			i = 1;
		} else if(cond_y && cond_x){
			i++;
		}



		s_1_1_0 = window_4[j_p];

		s_1_0_1 = window_3[j_l];
		window_4[j_p] = s_1_0_1;

		s_1_1_1_b = s_1_1_1;
		window_3[j_l] = s_1_1_1_b;

		s_1_1_1 = s_1_1_1_f;
		s_1_1_1_f = window_2[j_l]; 	// read

		s_1_2_1 = window_1[j_p];   // read
		window_2[j_l] = s_1_2_1;	//set


		bool cond_tmp1 = (itr < limit_read);
		uint256_dt tmp_rd;
		if(cond_tmp1){
			tmp_rd = rd_buffer.read(); // set
			acc_in_vec = acc_in.read();
		}

		float tmp_rd_arr[VEC_FACTOR];
		vec2arr_tmp_read: for(int v = 0; v < VEC_FACTOR; v++){
			data_conv tmp_rd_u, acc_in_u, s_1_1_2_u;
			tmp_rd_u.i = tmp_rd.range(D_SIZE * (v + 1) - 1, v * D_SIZE);
			acc_in_u.i = acc_in_vec.range(D_SIZE * (v + 1) - 1, v * D_SIZE);
			tmp_rd_arr[v]   			=  tmp_rd_u.f;
			acc_in_arr[v] 				=  acc_in_u.f;

			s_1_1_2_u.f = tmp_rd_arr[v] + acc_in_arr[v];
			s_1_1_2.range(D_SIZE * (v + 1) - 1, v * D_SIZE) = s_1_1_2_u.i;

		}
		window_1[j_p] = s_1_1_2; // set



		j_p++;
		if(j_p == plane_diff){
			j_p = 0;
		}

		j_l++;
		if(j_l == line_diff){
			j_l = 0;
		}




		vec2arr: for(int k = 0; k < VEC_FACTOR; k++){
			data_conv s_1_1_2_u, s_1_2_1_u, s_1_1_1_u, s_1_0_1_u, s_1_1_0_u, s_1_1_1_b_u, s_1_1_1_f_u;
			s_1_1_2_u.i = s_1_1_2.range(D_SIZE * (k + 1) - 1, k * D_SIZE);
			s_1_2_1_u.i = s_1_2_1.range(D_SIZE * (k + 1) - 1, k * D_SIZE);
			s_1_1_1_b_u.i = s_1_1_1_b.range(D_SIZE * (k + 1) - 1, k * D_SIZE);
			s_1_1_1_u.i = s_1_1_1.range(D_SIZE * (k + 1) - 1, k * D_SIZE);
			s_1_1_1_f_u.i = s_1_1_1_f.range(D_SIZE * (k + 1) - 1, k * D_SIZE);
			s_1_0_1_u.i = s_1_0_1.range(D_SIZE * (k + 1) - 1, k * D_SIZE);
			s_1_1_0_u.i = s_1_1_0.range(D_SIZE * (k + 1) - 1, k * D_SIZE);

			s_1_1_2_arr[k]   			=  s_1_1_2_u.f;
			s_1_2_1_arr[k]   			=  s_1_2_1_u.f;
			s_1_1_1_arr[k] 				=  s_1_1_1_b_u.f;
			s_1_1_1_arr[k+VEC_FACTOR] 	=  s_1_1_1_u.f;
			s_1_1_1_arr[k+VEC_FACTOR*2] =  s_1_1_1_f_u.f;
			s_1_0_1_arr[k]   			=  s_1_0_1_u.f;
			s_1_1_0_arr[k]   			=  s_1_1_0_u.f;
		}

		unsigned short y_index = j;
		process: for(short q = 0; q < VEC_FACTOR; q++){
			short index = (k << 2) + q;
			float r1_1_2 =  s_1_1_2_arr[q];
			float r1_2_1 =  s_1_2_1_arr[q];
			float r0_1_1 =  s_1_1_1_arr[q+ VEC_FACTOR-1];
			float r1_1_1 =  s_1_1_1_arr[q+VEC_FACTOR] * 6.0f;
			float r2_1_1 =  s_1_1_1_arr[q+VEC_FACTOR+1];
			float r1_0_1 =  s_1_0_1_arr[q];
			float r1_1_0 =  s_1_1_0_arr[q];

			float f1 = r1_1_2 + r1_2_1;
			float f2 = r0_1_1 - r1_1_1;
			float f3 = r2_1_1 + r1_0_1;

//			#pragma HLS RESOURCE variable=f1 core=FAddSub_nodsp
//			#pragma HLS RESOURCE variable=f2 core=FAddSub_nodsp
//			#pragma HLS RESOURCE variable=f3 core=FAddSub_nodsp

			float r1 = f1 + f2;
			float r2=  f3 + r1_1_0;

			float result  = r1 + r2;
			bool change_cond = register_it <bool>(index <= 0 || index > sizex || (i <= 1) || (i >= limit_z -1) || (y_index <= 0) || (y_index >= grid_sizey -1));
			mem_d[q] = change_cond ? 0.0f : result;

			acc_out_arr[q] = dnt_update ? acc_in_arr[q] : s_1_1_1_arr[q+VEC_FACTOR];
		}

		array2vec: for(int v = 0; v < VEC_FACTOR; v++){
			data_conv tmp_d, acc_out_u;

			tmp_d.f = mem_d[v];
			acc_out_u.f = acc_out_arr[v];
			update_d.range(D_SIZE * (v + 1) - 1, v * D_SIZE) = tmp_d.i;
			acc_out_vec.range(D_SIZE * (v + 1) - 1, v * D_SIZE) = acc_out_u.i;
		}

		bool cond_wr = (i >= 1) && ( i < limit_z);
		if(cond_wr ) {
			d << update_d;
			acc_out << acc_out_vec;
		}

		// move the cell block
		k++;
	}
}

//static int pre_process(const uint512_dt* u, uint512_dt* d,
//		const uint512_dt* acc_1, uint512_dt* acc_2,
//		ap_uint<12> M, ap_uint<12> N, ap_uint<12> L, ap_uint<12> B, bool dnt_update){
//
//
//    static hls::stream<uint256_dt> streamArray[4];
//    static hls::stream<uint256_dt> accStream[4];
//
//    #pragma HLS STREAM variable = streamArray depth = 2
//	#pragma HLS STREAM variable = accStream depth = 2
//
//    struct data_G data_g;
//    data_g.sizex = M-2;
//    data_g.sizey = N-2;
//    data_g.sizez = L-2;
//
//
//	data_g.xblocks = (M >> 3);
//	data_g.grid_sizey = N;
//	data_g.grid_sizez = L;
//	data_g.limit_z = L+1;
//
//	unsigned short tiley_1 = (N - 1);
//	unsigned int plane_size = data_g.xblocks * N;
//
//	data_g.plane_diff = data_g.xblocks * tiley_1;
//	data_g.line_diff = data_g.xblocks - 1;
//	data_g.gridsize_pr = plane_size * register_it<unsigned int>(data_g.grid_sizez * B+1);
//	data_g.gridsize_da = plane_size * L * B;
//
//	#pragma HLS dataflow
//	read_u(u, streamArray[0], acc_1, accStream[0],  M, N, L, B);
//	process_tile(streamArray[0], streamArray[1], accStream[0], accStream[1], data_g, dnt_update);
//	write_d(d, streamArray[1], acc_2,  accStream[1], M, N, L, B);
//
//	return 0;
//
//}
