#include <stdio.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <math.h>
#include "pre_proc.h"

// Preproc Modules
//static void read_u(const uint512_dt* u, hls::stream<uint256_dt> &u_stm,
//		const uint512_dt* acc, hls::stream<uint256_dt> &acc_stm,
//		ap_uint<12> M, ap_uint<12> N, ap_uint<12> B){
//
//	int total_itr = (M>>4)*N*B;
//	for(int itr = 0; itr < total_itr; itr++){
//		#pragma HLS PIPELINE II=2
//		#pragma HLS loop_tripcount min=500000 max=1000000 avg=1000000
//		uint512_dt tmp = u[itr];
//		uint512_dt tmp_acc = acc[itr];
//		u_stm << tmp.range(255,0);
//		u_stm << tmp.range(511,256);
//
//		acc_stm << tmp_acc.range(255,0);
//		acc_stm << tmp_acc.range(511,256);
//	}
//}
//
//static void write_d(uint512_dt* d, hls::stream<uint256_dt> &d_stm,
//		uint512_dt* acc, hls::stream<uint256_dt> &acc_stm,
//		ap_uint<12> M, ap_uint<12> N, ap_uint<12> B){
//
//	int total_itr = (M>>4)*N*B;
//	for(int itr = 0; itr < total_itr; itr++){
//		#pragma HLS PIPELINE II=2
//		#pragma HLS loop_tripcount min=16 max=32 avg=32
//		uint512_dt tmp_a, tmp_b, tmp_c, tmp_d, tmp_acc;
//
//		tmp_d.range(255,0) = d_stm.read();
//		tmp_d.range(511,256) = d_stm.read();
//
//		tmp_acc.range(255,0) = acc_stm.read();
//		tmp_acc.range(511,256) = acc_stm.read();
//
//		d[itr] = tmp_d;
//		acc[itr] = tmp_acc;
//
//	}
//}

static void process_grid( hls::stream<uint256_dt> &rd_buffer, hls::stream<uint256_dt> &wr_buffer,
		hls::stream<uint256_dt> &acc_in,  hls::stream<uint256_dt> &acc_out, struct data_G data_g, bool dnt_acc_updt, bool skip_process){

	short end_index = data_g.end_index;

    // Registers to hold data specified by stencil
	float row_arr3[VEC_FACTOR];
	float row_arr2[VEC_FACTOR + 2];
	float row_arr1[VEC_FACTOR];
	float mem_wr[VEC_FACTOR], acc_out_arr[VEC_FACTOR];
	float a_arr[VEC_FACTOR], b_arr[VEC_FACTOR], c_arr[VEC_FACTOR];
	float acc_in_arr[VEC_FACTOR];


    // partioning array into individual registers
	#pragma HLS ARRAY_PARTITION variable=row_arr3 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=row_arr2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=row_arr1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=mem_wr complete dim=1

	#pragma HLS ARRAY_PARTITION variable=a_arr complete dim=1
	#pragma HLS ARRAY_PARTITION variable=b_arr complete dim=1
	#pragma HLS ARRAY_PARTITION variable=c_arr complete dim=1


    // cyclic buffers to hold larger number of elements
	uint256_dt row1_n[max_depth_8];
	uint256_dt row2_n[max_depth_8];
	uint256_dt row3_n[max_depth_8];

	uint256_dt row1_acc_n[max_depth_8];

//	#pragma HLS RESOURCE variable=row1_n core=XPM_MEMORY uram latency=2
//	#pragma HLS RESOURCE variable=row2_n core=XPM_MEMORY uram latency=2
//	#pragma HLS RESOURCE variable=row3_n core=XPM_MEMORY uram latency=2

	unsigned short sizex = data_g.sizex;
	unsigned short end_row = data_g.end_row;
	unsigned short outer_loop_limit = data_g.outer_loop_limit;
	unsigned int grid_size = data_g.gridsize;
	unsigned short end_index_minus1 = data_g.endindex_minus1;
	unsigned short end_row_plus1 = data_g.endrow_plus1;
	unsigned short end_row_plus2 = data_g.endrow_plus2;
	unsigned short end_row_minus1 = data_g.endrow_minus1;
	unsigned int grid_data_size = data_g.total_itr_256;

    uint256_dt tmp2_f1, tmp2_b1;
    uint256_dt tmp1, tmp2, tmp3;
    uint256_dt rd_in_vec, acc_in_vec, acc_out_vec;
    uint256_dt update_j, a_vec_out, b_vec_out, c_vec_out;

    uint256_dt tmp2_acc, tmp2_acc_f1;


    // flattened loop to reduce the inter loop latency
    unsigned short i = 0, j = 0, j_l = 0;
    unsigned short i_d = 0, j_d = 0;
    for(unsigned int itr = 0; itr < grid_size; itr++) {
        #pragma HLS loop_tripcount min=819200 max=1000000 avg=1000000
        #pragma HLS PIPELINE II=1

    	i = i_d;
    	j = j_d;

    	bool cmp_j = (j == end_index -1 );
    	bool cmp_i = (i == outer_loop_limit -1);

    	if(cmp_j){
    		j_d = 0;
    	} else {
    		j_d++;
    	}

    	if(cmp_j && cmp_i){
    		i_d = 1;
    	} else if(cmp_j){
    		i_d++;
    	}


        tmp1 = row2_n[j_l];

        tmp2_b1 = tmp2;
        row2_n[j_l] = tmp2_b1;

        tmp2 = tmp2_f1;
        tmp2_f1 = row1_n[j_l];

        tmp2_acc = tmp2_acc_f1;
        tmp2_acc_f1 = row1_acc_n[j_l];

        // continuous data-flow for all the grids in the batch
        bool cond_tmp1 = (itr < grid_data_size);
        if(cond_tmp1){
//        	rd_in_vec
        	rd_in_vec= rd_buffer.read();
        	acc_in_vec = acc_in.read();
        }

//        if(!skip_process && cond_tmp1){
//        	acc_in_vec = acc_in.read();
//        }

        vec2arr: for(int v = 0; v < VEC_FACTOR; v++){
        	data_conv tmp3_u, rd_in_u, acc_in_u;
        	acc_in_u.i = acc_in_vec.range(D_SIZE * (v + 1) - 1, v * D_SIZE);
        	rd_in_u.i = rd_in_vec.range(D_SIZE * (v + 1) - 1, v * D_SIZE);
        	tmp3_u.f =  skip_process ? rd_in_u.f : (rd_in_u.f + acc_in_u.f);
        	tmp3.range(D_SIZE * (v + 1) - 1, v * D_SIZE) = tmp3_u.i;
        }


        row1_n[j_l] = tmp3; //rd_in_vec; //tmp3;
        row1_acc_n[j_l] = acc_in_vec;


        // line buffer
        j_l++;
        if(j_l >= end_index - 1){
            j_l = 0;
        }


        for(int k = 0; k < VEC_FACTOR; k++){
            data_conv tmp1_u, tmp2_u, tmp3_u, acc_in_u;
            tmp1_u.i = tmp1.range(D_SIZE * (k + 1) - 1, k * D_SIZE);
            tmp2_u.i = tmp2.range(D_SIZE * (k + 1) - 1, k * D_SIZE);
            tmp3_u.i = tmp3.range(D_SIZE * (k + 1) - 1, k * D_SIZE);
            acc_in_u.i = tmp2_acc.range(D_SIZE * (k + 1) - 1, k * D_SIZE);

            row_arr3[k] =  tmp3_u.f;
            row_arr2[k+1] = tmp2_u.f;
            row_arr1[k] =  tmp1_u.f;

            acc_in_arr[k] = acc_in_u.f;

        }
        data_conv tmp1_o1, tmp2_o2;
        tmp1_o1.i = tmp2_b1.range(D_SIZE * (VEC_FACTOR) - 1, (VEC_FACTOR-1) * D_SIZE);
        tmp2_o2.i = tmp2_f1.range(D_SIZE * (0 + 1) - 1, 0 * D_SIZE);
        row_arr2[0] = tmp1_o1.f;
        row_arr2[VEC_FACTOR + 1] = tmp2_o2.f;


        // stencil computation
        // this loop will be completely unrolled as parent loop is pipelined
        process: for(short q = 0; q < VEC_FACTOR; q++){
            short index = (j << SHIFT_BITS) + q;
            float r1 = ( (row_arr2[q])  + (row_arr2[q+2]) );

            float r2 = ( row_arr1[q]  + row_arr3[q] );

            float f1 = r1 + r2;
            float result  = f1 - row_arr2[q+1]*6.0f;
            bool change_cond = (index <= 0 || index > sizex || (i == 1) || (i == end_row));
            float final_val = change_cond ? 0.0f : result;
            mem_wr[q] = skip_process ? row_arr2[q+1] : final_val;

            a_arr[q] = change_cond ? 0.0f : -0.5f;
            b_arr[q] = change_cond ? 1.0f :  2.0f;
            c_arr[q] = change_cond ? 0.0f : -0.5f;
            acc_out_arr[q] = dnt_acc_updt? acc_in_arr[q] : row_arr2[q+1];
        }

        array2vec: for(int k = 0; k < VEC_FACTOR; k++){
            data_conv tmp, acc_out_u, a_out_u, b_out_u, c_out_u;
            tmp.f = mem_wr[k];
            acc_out_u.f = acc_out_arr[k];
            a_out_u.f = a_arr[k];
            b_out_u.f = b_arr[k];
            c_out_u.f = c_arr[k];
            update_j.range(D_SIZE * (k + 1) - 1, k * D_SIZE) = tmp.i;

            a_vec_out.range(D_SIZE * (k + 1) - 1, k * D_SIZE) = a_out_u.i;
            b_vec_out.range(D_SIZE * (k + 1) - 1, k * D_SIZE) = b_out_u.i;
            c_vec_out.range(D_SIZE * (k + 1) - 1, k * D_SIZE) = c_out_u.i;

            acc_out_vec.range(D_SIZE * (k + 1) - 1, k * D_SIZE) = acc_out_u.i;
        }
        // conditional write to stream interface
        bool cond_wr = (i >= 1);
        if(cond_wr ) {
            wr_buffer << update_j;
            acc_out << acc_out_vec;
        }

//        if(!skip_process && cond_wr){
//            acc_out << acc_out_vec;
//        }

    }
}

//static int pre_process(const uint512_dt* u,  uint512_dt* d, const uint512_dt* acc_1, uint512_dt* acc_2,
//				   int M, int N, int B, bool dnt_acc_updt){
//
//
//    static hls::stream<uint256_dt> streamArray[5];
//    static hls::stream<uint256_dt> accStream[4];
//
//    #pragma HLS STREAM variable = streamArray depth = 2
//	#pragma HLS STREAM variable = accStream depth = 2
//
//
////---------------------------------
//    struct data_G data_g;
//    data_g.sizex = M-2;
//    data_g.sizey = N-2;
//    data_g.xdim0 = M;
//	data_g.end_index = (M >> 3); // number of blocks with V number of elements to be processed in a single row
//	data_g.end_row = N; // includes the boundary
//	data_g.outer_loop_limit = N+1; // n + D/2
//	data_g.gridsize = (data_g.end_row* B + 1) * data_g.end_index;
//	data_g.endindex_minus1 = data_g.end_index -1;
//	data_g.endrow_plus1 = data_g.end_row + 1;
//	data_g.endrow_plus2 = data_g.end_row + 2;
//	data_g.endrow_minus1 = data_g.end_row - 1;
//	data_g.total_itr_256 = data_g.end_row * data_g.end_index * B;
//	data_g.total_itr_512 = (data_g.end_row * data_g.end_index * B + 1) >> 1;
//
//	#pragma HLS dataflow
//	read_u(u, streamArray[0], acc_1, accStream[0],  M, N, B);
//	process_grid(streamArray[0], streamArray[4], accStream[0], accStream[1], data_g, dnt_acc_updt, 0);
//	write_d( d, streamArray[4], acc_2,  accStream[1], M, N, B);
//
//	return 0;
//
//}
