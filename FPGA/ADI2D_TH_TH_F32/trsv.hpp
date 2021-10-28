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
void trsv_step(T inlow[N], T indiag[N], T inup[N], T inrhs[N], T outlow[N], T outdiag[N], T outup[N], T outrhs[N], int n, int bsize) {
#pragma HLS dependence variable = inlow inter false
#pragma HLS dependence variable = indiag inter false
#pragma HLS dependence variable = inup inter false
#pragma HLS dependence variable = inrhs inter false

#pragma HLS dependence variable = outlow inter false
#pragma HLS dependence variable = outdiag inter false
#pragma HLS dependence variable = outup inter false
#pragma HLS dependence variable = outrhs inter false

    const unsigned int N2 = n >> 1;

    T a[3];
    T b[3];
    T c[3];
    T v[3];

#pragma HLS array_partition variable = a complete
#pragma HLS array_partition variable = b complete
#pragma HLS array_partition variable = c complete
#pragma HLS array_partition variable = v complete




    LoopLines:
	for(int bat = 0; bat < bsize; bat++){
		for (unsigned int i = 0; i < n; i++) {
	#pragma HLS pipeline
			if(i == 0){
				// init read regs
				for (int r = 0; r < 2; r++) {
					#pragma HLS unroll
					a[r] = 0.0;
					b[r] = 1.0;
					c[r] = 0.0;
					v[r] = 0.0;
				};

				a[2] = inlow[0+bat*n];
				b[2] = indiag[0+bat*n];
				c[2] = inup[0+bat*n];
				v[2] = inrhs[0+bat*n];
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

			unsigned int addr = i+1;
			unsigned int addr1 = bat*n+ i+1;

			if (addr < n) {
				a[2] = inlow[addr1];
				b[2] = indiag[addr1];
				c[2] = inup[addr1];
				v[2] = inrhs[addr1];
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
			outlow[addc+bat*n] = low[0];
			outdiag[addc+bat*n] = diag[0];
			outup[addc+bat*n] = up[0];
			outrhs[addc+bat*n] = rhs[0];

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
void trsvCore(T inlow[N], T indiag[N], T inup[N], T inrhs[N], int n, int batch, int logn) {
    // TODO: remove logN
    // TODO: N is not power of 2

    const int N2 = N >> 1;
    const int N4 = N >> 2;

    T outlow[N];
    T outdiag[N];
    T outup[N];
    T outrhs[N];

#pragma HLS RESOURCE variable = outlow core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = outdiag core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = outup core = RAM_2P_BRAM
#pragma HLS RESOURCE variable = outrhs core = RAM_2P_BRAM


LoopTop:
    for (int s = 0; s < (logn >> 1); s++) {
        internal::trsv_step<T, N>(inlow, indiag, inup, inrhs, outlow, outdiag, outup, outrhs, n, batch);

        internal::trsv_step<T, N>(outlow, outdiag, outup, outrhs, inlow, indiag, inup, inrhs, n, batch);
    };

    if (logn % 2 == 1) {
        internal::trsv_step<T, N>(inlow, indiag, inup, inrhs, outlow, outdiag, outup, outrhs, n, batch);
    LoopWrite:
        for (int i = 0; i < n*batch; i++) {
#pragma HLS pipeline
            inlow[i] = outlow[i];
            indiag[i] = outdiag[i];
            inup[i] = outup[i];
            inrhs[i] = outrhs[i];
        };
    };
};

} // namespace solver
} // namespace xf

#endif
