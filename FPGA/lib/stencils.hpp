#include <ap_int.h>
#include <hls_stream.h>
#include <data_types.h>

#ifndef __STENCILS_H__
#define __STENCILS_H__

template <bool FPPREC, class DType, int DMAX>
static void stencil_2d( hls::stream<uint256_dt> &rd_buffer, hls::stream<uint256_dt> &wr_buffer,
        hls::stream<uint256_dt> &acc_in,  hls::stream<uint256_dt> &acc_out, struct data_G data_g, bool dnt_acc_updt){

    #define SHIFT (3-FPPREC)
    #define VFACTOR ((1 << SHIFT))
    #define DSIZE   (256/VFACTOR)

    short end_index = data_g.end_index;

    // Registers to hold data specified by stencil
    DType row_arr3[VFACTOR];
    DType row_arr2[VFACTOR + 2];
    DType row_arr1[VFACTOR];
    DType mem_wr[VFACTOR], acc_out_arr[VFACTOR];
    DType a_arr[VFACTOR], b_arr[VFACTOR], c_arr[VFACTOR];
    DType acc_in_arr[VFACTOR];


    // partioning array into individual registers
    #pragma HLS ARRAY_PARTITION variable=row_arr3 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=row_arr2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=row_arr1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=mem_wr complete dim=1

    #pragma HLS ARRAY_PARTITION variable=a_arr complete dim=1
    #pragma HLS ARRAY_PARTITION variable=b_arr complete dim=1
    #pragma HLS ARRAY_PARTITION variable=c_arr complete dim=1


    // cyclic buffers to hold larger number of elements
    uint256_dt row1_n[DMAX/VFACTOR];
    uint256_dt row2_n[DMAX/VFACTOR];
    uint256_dt row3_n[DMAX/VFACTOR];

    uint256_dt row1_acc_n[DMAX/VFACTOR];


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
        #pragma HLS loop_tripcount min=204800 max=1000000 avg=1000000
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
            rd_in_vec= rd_buffer.read();
            acc_in_vec = acc_in.read();
        }

        vec2arr: for(int v = 0; v < VFACTOR; v++){
            DType tmp = uint2FP_ript<FPPREC, DType>(acc_in_vec.range(DSIZE * (v + 1) - 1, v * DSIZE)) + uint2FP_ript<FPPREC, DType>(rd_in_vec.range(DSIZE * (v + 1) - 1, v * DSIZE));
            tmp3.range(DSIZE * (v + 1) - 1, v * DSIZE) = FP2uint_ript(tmp);

        }


        row1_n[j_l] = tmp3; //rd_in_vec; //tmp3;
        row1_acc_n[j_l] = acc_in_vec;


        // line buffer
        j_l++;
        if(j_l >= end_index - 1){
            j_l = 0;
        }


        for(int k = 0; k < VFACTOR; k++){
            row_arr3[k] =  uint2FP_ript<FPPREC, DType>(tmp3.range(DSIZE * (k + 1) - 1, k * DSIZE));
            row_arr2[k+1] = uint2FP_ript<FPPREC, DType>(tmp2.range(DSIZE * (k + 1) - 1, k * DSIZE));
            row_arr1[k] =  uint2FP_ript<FPPREC, DType>(tmp1.range(DSIZE * (k + 1) - 1, k * DSIZE));
            acc_in_arr[k] = uint2FP_ript<FPPREC, DType>(tmp2_acc.range(DSIZE * (k + 1) - 1, k * DSIZE));

        }

        row_arr2[0] = uint2FP_ript<FPPREC, DType>(tmp2_b1.range(DSIZE * (VFACTOR) - 1, (VFACTOR-1) * DSIZE));
        row_arr2[VFACTOR + 1] = uint2FP_ript<FPPREC, DType>(tmp2_f1.range(DSIZE * (0 + 1) - 1, 0 * DSIZE));


        // stencil computation
        // this loop will be completely unrolled as parent loop is pipelined
        process: for(short q = 0; q < VFACTOR; q++){
            short index = (j << SHIFT) + q;
            DType r1 = ( (row_arr2[q])  + (row_arr2[q+2]) );

            DType r2 = ( row_arr1[q]  + row_arr3[q] );

            DType f1 = r1 + r2;
            DType result  = f1 - row_arr2[q+1]*6.0f;
            bool change_cond = (index <= 0 || index > sizex || (i == 1) || (i == end_row));
            mem_wr[q] = change_cond ? 0.0f : result;

            a_arr[q] = change_cond ? 0.0f : -0.5f;
            b_arr[q] = change_cond ? 1.0f :  2.0f;
            c_arr[q] = change_cond ? 0.0f : -0.5f;
            acc_out_arr[q] = dnt_acc_updt? acc_in_arr[q] : row_arr2[q+1];
        }

        array2vec: for(int k = 0; k < VFACTOR; k++){
            update_j.range(DSIZE * (k + 1) - 1, k * DSIZE) = FP2uint_ript(mem_wr[k]);
            a_vec_out.range(DSIZE * (k + 1) - 1, k * DSIZE) = FP2uint_ript(a_arr[k]);
            b_vec_out.range(DSIZE * (k + 1) - 1, k * DSIZE) = FP2uint_ript(b_arr[k]);
            c_vec_out.range(DSIZE * (k + 1) - 1, k * DSIZE) = FP2uint_ript(c_arr[k]);
            acc_out_vec.range(DSIZE * (k + 1) - 1, k * DSIZE) = FP2uint_ript(acc_out_arr[k]);
        }

        // conditional write to stream interface
        bool cond_wr = (i >= 1);
        if(cond_wr ) {
            wr_buffer << update_j;
            acc_out << acc_out_vec;
        }

    }

    #undef SHIFT
    #undef VFACTOR
    #undef DSIZE
}

#endif
