/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 *  @file trsv.hpp
 *  @brief Tridiagonal solver header file
 *
 *  $DateTime: 2019/04/09 12:00:00 $
 */
#include <stdio.h>
#include <hls_stream.h>
#include <ap_int.h>
#include "pre_proc.h"
#ifndef _XSOLVER_CORE_TRSV_
#define _XSOLVER_CORE_TRSV_

namespace xf {
namespace fintech {
namespace internal {
/**
* @brief Executes one step of odd-even elimination. \n
* For each row it calculates new diagonal element and right hand side element.
\n
* \n
*
* Please note the algorithm is very sensitive to zeros in main diagonal. \n
* Any zeros in main diagonal will lead to division by zero and algorithm fail.
*@tparam T data type used in whole function (double by default)
*@tparam N Size of the operating matrix
*@tparam NCU Number of compute units working in parallel
*@param[in] inlow Input vector of lower diagonal
*@param[in] indiag Input vector of main diagonal
*@param[in] inup Input vector of upper diagonal
*@param[in] inrhs Input vector of Right hand side

*@param[out] outlow Output vector of lower diagonal
*@param[out] outdiag Output vector of main diagonal
*@param[out] outup Output vector of upper diagonal
*@param[out] outrhs Output vector of Right hand side
*/
template <class T, unsigned int N>
void trsv_step(hls::stream<float> &pcr_a, hls::stream<float> &pcr_b, hls::stream<float> &pcr_c, hls::stream<float> &pcr_d, hls::stream<float> &stm_out,
//		T inlow[N], T indiag[N], T inup[N], T inrhs[N], T outlow[N], T outdiag[N], T outup[N], T outrhs[N],
		ap_uint<8> n, ap_uint<8> bsize, ap_int<8>  logn, ap_int<20> bigbatch)  {



    T inlow[N];
    T indiag[N];
    T inup[N];
    T inrhs[N];


#pragma HLS RESOURCE variable = inlow core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = indiag core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = inup core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = inrhs core = RAM_2P_BRAM

    const unsigned int N2 = n >> 1;

    T a[3];
    T b[3];
    T c[3];
    T v[3];

#pragma HLS array_partition variable = a complete
#pragma HLS array_partition variable = b complete
#pragma HLS array_partition variable = c complete
#pragma HLS array_partition variable = v complete


//    ap_uint<8> n = n_inp+35;



    ap_int<8> total_inner = bsize*n+ bsize + 40;
    ap_int<40> total_itr = (total_inner) *logn* bigbatch; // bsize*n*logn*bigbatch + bsize*logn*bigbatch + 40*logn*bigbatch;//total_inner*logn;
    ap_int<8> sd = -1;
    ap_int<8> batd =0;
    ap_int<8> id = -1;
    ap_uint<8> countd = 0;
    int countb = 0;
    LoopLines:for(ap_uint<40> itr = 0; itr < total_itr; itr++){
		#pragma HLS loop_tripcount min=520 max=1000 avg=1000
		#pragma HLS PIPELINE II=1


		#pragma HLS dependence variable = inlow RAW distance=40 true
		#pragma HLS dependence variable = indiag RAW distance=40 true
		#pragma HLS dependence variable = inup RAW distance=40 true
		#pragma HLS dependence variable = inrhs RAW distance=40 true

//    	printf("Itr:%d\n", (int)itr);

    	ap_int<8> bat = batd;
    	ap_int<8> i = id;
    	ap_int<8> s = sd;
    	ap_uint<8> count = countd;

    	bool cmp = (count == total_inner-1);
    	bool cmpn = (i == n-1);


    	if(cmpn || cmp){
    		id = -1;
    	} else {
    		id++;
    	}

    	if(cmp){
    		batd = 0;
    	} else if (cmpn){
    		batd++;
    	}

    	if(cmp){
    		countd = 0;
    	} else {
    		countd++;
    	}

    	if(s == logn-2 && cmp){
    		sd = -1;
    		countb++;
    	} else if(cmp) {
    		sd++;
    	}

        unsigned char offsetR = ((s & 0x1) == 0) ? 0 : 128;
        unsigned char offsetW = ((s & 0x1) == 1) ? 0 : 128;

//        printf("countb:%d count:%d s:%d itr:%d\n", countb, (int) count,  (int)s, (int)itr);



		int addr1 = (s == logn-2) ? (int)(offsetR+count) : (int)(bat*n+ i+1+offsetR);
//		printf("Val1:%d val2:%d\n", (int)(offsetR+count), (int)(bat*n+ i+1+offsetR));

		T val_l, val_d, val_u, val_r;

		if(count < bsize*n && s == -1){
			val_l = pcr_a.read();
			val_d = pcr_b.read();
			val_u = pcr_c.read();
			val_r = pcr_d.read();
		} else {
			val_l= inlow[addr1];
			val_d = indiag[addr1];
			val_u = inup[addr1];
			val_r = inrhs[addr1];
		}

		if(s == logn-2 && count < bsize*n){
			stm_out << (val_r/val_d);
		}



		if(i == -1){
			// init read regs
			for (int r = 0; r < 2; r++) {
				#pragma HLS unroll
				a[r] = 0.0;
				b[r] = 1.0;
				c[r] = 0.0;
				v[r] = 0.0;
			};

			a[2] = val_l;
			b[2] = val_d;
			c[2] = val_u;
			v[2] = val_r;
		}

		// update read regs
		a[0] = a[1];
		a[1] = a[2];
		b[0] = b[1];
		b[1] = b[2];
		c[0] = c[1];
		c[1] = c[2];
		v[0] = v[1];
		v[1] = v[2];

//		unsigned int addr1 = bat*n+ i+1;

		if (i+1 < n) {
			a[2] = val_l;
			b[2] = val_d;
			c[2] = val_u;
			v[2] = val_r;
		} else {
			a[2] = 0.0;
			b[2] = 1.0;
			c[2] = 0.0;
			v[2] = 0.0;
		};


		T low[1];
		T diag[1];
		T up[1];
		T rhs[1];
		#pragma HLS array_partition variable = low complete
		#pragma HLS array_partition variable = diag complete
		#pragma HLS array_partition variable = up complete
		#pragma HLS array_partition variable = rhs complete



		T a_1 = a[0];
		T a0 = a[1];
		T a1 = a[2];

		T b_1 = b[0];
		T b0 = b[1];
		T b1 = b[2];

		T c_1 = c[0];
		T c0 = c[1];
		T c1 = c[2];

		T v_1 = v[0];
		T v0 = v[1];
		T v1 = v[2];

		T k1 = a0 / b_1;
		T ak1 = a_1 * k1;
		T ck1 = c_1 * k1;
		T vk1 = v_1 * k1;

		T k2 = c0 / b1;
		T ak2 = a1 * k2;
		T ck2 = c1 * k2;
		T vk2 = v1 * k2;

		low[0] = -ak1;
		diag[0] = b0 - ck1 - ak2;
		up[0] = -ck2;
		rhs[0] = v0 - vk1 - vk2;


		// write

		unsigned int i2 = (i >> 1);
		unsigned int addc = (i % 2 == 0) ? i2 : (i2 + N2);
		int add_w = addc+bat*n+offsetW;

		if(s  == -1 && count < bsize*n){
			inlow[count] = val_l;
			indiag[count] = val_d;
			inup[count] = val_u;
			inrhs[count] = val_r;
		}else if(i >= 0 && s >= 0){
			inlow[add_w] = low[0];
			indiag[add_w] = diag[0];
			inup[add_w] = up[0];
			inrhs[add_w] = rhs[0];
		}

    }


};

} // namespace internal
/**
  @brief Tridiagonal linear solver
  It solves tridiagonal linear system of equations by eliminating upper and
  lower diagonals
  To get result (U) divide each element of \a inrhs by coresponding element of
  main diagonal \a indiag
  @tparam T data type
  @tparam N matrix size
  @tparam logN log2(N)(TOREMOVE)
  @tparam NCU number of compute units
  @param[in] inlow lower diagonal
  @param[in] indiag diagonal
  @param[in] inup upper diagonal
  @param[in] inrhs right-hand side
 */
template <class T, unsigned int N>
void trsvCore(
		hls::stream<float> &pcr_a, hls::stream<float> &pcr_b, hls::stream<float> &pcr_c, hls::stream<float> &pcr_d,
		hls::stream<float> &stm_out,
		ap_uint<8> n, ap_uint<8> batch, ap_uint<8> logn, ap_uint<20> bigbatch) {
    // TODO: remove logN
    // TODO: N is not power of 2

    const int N2 = N >> 1;
    const int N4 = N >> 2;
//




// for(ap_uint<20> itr = 0; itr < bigbatch; itr++){
//		#pragma HLS loop_tripcount min=25600 max=2000000 avg=2000000


	LoopTop:


		internal::trsv_step<T, N>(pcr_a, pcr_b, pcr_c, pcr_d, stm_out, n, batch, (logn+2), bigbatch);
//		printf("Finished PCR compute\n");


// }

};

} // namespace solver
} // namespace xf

#endif
