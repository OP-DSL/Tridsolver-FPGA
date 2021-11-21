#include <ap_int.h>
#include <hls_stream.h>
#include <data_types.h>

#ifndef __STENCILS_H__
#define __STENCILS_H__

template <bool FPPREC, class DType, int DMAX>
static void stencil_2d( hls::stream<uint256_dt> &rd_buffer, hls::stream<uint256_dt> &wr_buffer,
        hls::stream<uint256_dt> &acc_in,  hls::stream<uint256_dt> &acc_out, struct data_G data_g, bool dnt_acc_updt, bool skip_process){

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
            tmp3.range(DSIZE * (v + 1) - 1, v * DSIZE) = skip_process? rd_in_vec.range(DSIZE * (v + 1) - 1, v * DSIZE) :  FP2uint_ript(tmp);

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
            DType final_val = change_cond ? 0.0f : result;
            mem_wr[q] = skip_process ? row_arr2[q+1] : final_val;

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
            DType final_val = change_cond ? 0.0f : result;
            mem_wr[q] = final_val;

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



template <bool FPPREC, class DType, int DMAX>
static void stencil_3d( hls::stream<uint256_dt> &rd_buffer,  hls::stream<uint256_dt> &d,
        hls::stream<uint256_dt> &acc_in,  hls::stream<uint256_dt> &acc_out, struct data_G_3d data_g, bool dnt_update){

    #define SHIFT (3-FPPREC)
    #define VFACTOR ((1 << SHIFT))
    #define DSIZE   (256/VFACTOR)

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

    DType s_1_1_2_arr[VFACTOR];
    DType s_1_2_1_arr[VFACTOR];
    DType s_1_1_1_arr[VFACTOR*3];
    DType s_1_0_1_arr[VFACTOR];
    DType s_1_1_0_arr[VFACTOR];

    DType mem_a[VFACTOR];
    DType mem_b[VFACTOR];
    DType mem_c[VFACTOR];
    DType mem_d[VFACTOR];

    DType acc_in_arr[VFACTOR];
    DType acc_out_arr[VFACTOR];

    #pragma HLS ARRAY_PARTITION variable=s_1_1_2_arr complete dim=1
    #pragma HLS ARRAY_PARTITION variable=s_1_2_1_arr complete dim=1
    #pragma HLS ARRAY_PARTITION variable=s_1_1_1_arr complete dim=1
    #pragma HLS ARRAY_PARTITION variable=s_1_0_1_arr complete dim=1
    #pragma HLS ARRAY_PARTITION variable=s_1_1_0_arr complete dim=1
    #pragma HLS ARRAY_PARTITION variable=mem_d complete dim=1

    uint256_dt window_1[DMAX*DMAX/VFACTOR];
    uint256_dt window_2[DMAX/VFACTOR];
    uint256_dt window_3[DMAX/VFACTOR];
    uint256_dt window_4[DMAX*DMAX/VFACTOR];

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
        s_1_1_1_f = window_2[j_l];  // read

        s_1_2_1 = window_1[j_p];   // read
        window_2[j_l] = s_1_2_1;    //set


        bool cond_tmp1 = (itr < limit_read);
        uint256_dt tmp_rd;
        if(cond_tmp1){
            tmp_rd = rd_buffer.read(); // set
            acc_in_vec = acc_in.read();
        }

        DType tmp_rd_arr[VFACTOR];
        vec2arr_tmp_read: for(int v = 0; v < VFACTOR; v++){
            tmp_rd_arr[v]               =  uint2FP_ript<FPPREC, DType>(tmp_rd.range(DSIZE * (v + 1) - 1, v * DSIZE));
            acc_in_arr[v]               =  uint2FP_ript<FPPREC, DType>(acc_in_vec.range(DSIZE * (v + 1) - 1, v * DSIZE));;
            s_1_1_2.range(DSIZE * (v + 1) - 1, v * DSIZE) = FP2uint_ript(tmp_rd_arr[v] + acc_in_arr[v]);

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




        vec2arr: for(int k = 0; k < VFACTOR; k++){
            s_1_1_2_arr[k]              =  uint2FP_ript<FPPREC, DType>(s_1_1_2.range(DSIZE * (k + 1) - 1, k * DSIZE));
            s_1_2_1_arr[k]              =  uint2FP_ript<FPPREC, DType>(s_1_2_1.range(DSIZE * (k + 1) - 1, k * DSIZE));
            s_1_1_1_arr[k]              =  uint2FP_ript<FPPREC, DType>(s_1_1_1_b.range(DSIZE * (k + 1) - 1, k * DSIZE));
            s_1_1_1_arr[k+VFACTOR]  =  uint2FP_ript<FPPREC, DType>(s_1_1_1.range(DSIZE * (k + 1) - 1, k * DSIZE));
            s_1_1_1_arr[k+VFACTOR*2] =  uint2FP_ript<FPPREC, DType>(s_1_1_1_f.range(DSIZE * (k + 1) - 1, k * DSIZE));
            s_1_0_1_arr[k]              =  uint2FP_ript<FPPREC, DType>(s_1_0_1.range(DSIZE * (k + 1) - 1, k * DSIZE));
            s_1_1_0_arr[k]              =  uint2FP_ript<FPPREC, DType>(s_1_1_0.range(DSIZE * (k + 1) - 1, k * DSIZE));
        }

        unsigned short y_index = j;
        process: for(short q = 0; q < VFACTOR; q++){
            short index = (k << SHIFT) + q;
            DType r1_1_2 =  s_1_1_2_arr[q];
            DType r1_2_1 =  s_1_2_1_arr[q];
            DType r0_1_1 =  s_1_1_1_arr[q+ VFACTOR-1];
            DType r1_1_1 =  s_1_1_1_arr[q+VFACTOR] * 6.0f;
            DType r2_1_1 =  s_1_1_1_arr[q+VFACTOR+1];
            DType r1_0_1 =  s_1_0_1_arr[q];
            DType r1_1_0 =  s_1_1_0_arr[q];

            DType f1 = r1_1_2 + r1_2_1;
            DType f2 = r0_1_1 - r1_1_1;
            DType f3 = r2_1_1 + r1_0_1;


            DType r1 = f1 + f2;
            DType r2=  f3 + r1_1_0;

            DType result  = r1 + r2;
            bool change_cond = register_it <bool>(index <= 0 || index > sizex || (i <= 1) || (i >= limit_z -1) || (y_index <= 0) || (y_index >= grid_sizey -1));
            mem_d[q] = change_cond ? 0.0f : result;

            acc_out_arr[q] = dnt_update ? acc_in_arr[q] : s_1_1_1_arr[q+VFACTOR];
        }

        array2vec: for(int v = 0; v < VFACTOR; v++){
            update_d.range(DSIZE * (v + 1) - 1, v * DSIZE) = FP2uint_ript(mem_d[v]);
            acc_out_vec.range(DSIZE * (v + 1) - 1, v * DSIZE) = FP2uint_ript(acc_out_arr[v]);
        }

        bool cond_wr = (i >= 1) && ( i < limit_z);
        if(cond_wr ) {
            d << update_d;
            acc_out << acc_out_vec;
        }

        // move the cell block
        k++;
    }

    #undef SHIFT
    #undef VFACTOR
    #undef DSIZE

}

#endif
