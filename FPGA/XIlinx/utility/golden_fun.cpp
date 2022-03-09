#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <math.h>
#include "golden_fun.h"



// Golden Thomas solver implementation




template <class DType>
 void golden<DType>::thomas_golden(DType* __restrict a, DType* __restrict b, DType* __restrict c,
		DType* __restrict d, DType* __restrict u, int N, int stride){

		for(int i = 1; i < N; i++){
			int ind = stride*i;
			DType w = a[ind] / b[ind-stride];
			b[ind] = b[ind] - w * c[ind-stride];
			d[ind] = d[ind] - w * d[ind-stride];
		}

		int ind = (N-1)*stride;
		u[ind] = d[ind] / b[ind];

		for(int i =N-2; i >=0; i--){
			int ind = i*stride;
			u[ind] = (d[ind] - c[ind] * u[ind+stride]) / b[ind];

		}

}


template <class DType>
 double golden<DType>::square_error(DType* golden, DType* FPGA, int nx, int ny, int nz){
      double sum = 0;
      int count = 0;
	  for(int k =0; k < nz; k++){
		for(int j = 0; j < ny; j++){
			for(int i=0; i<nx; i++) {
			  int Xdim_B = ((nx *nz + 63)/64)* 64;
			  int ind = j*nx + k*nx*ny + i;
			  DType val1 = golden[ind];
			  DType val2 = FPGA[ind];
			  sum +=  val1*val1 - val2*val2;
			  if((fabs(val1 -val2)/(fabs(val1) + fabs(val2))) > 0.001 && (fabs(val1) + fabs(val2)) > 0.000001 || isnan(val1) || isnan(val2)){
				  printf("i:%d j:%d k:%d golden:%f FPGA:%f\n", i, j, k, val1, val2);
				  count++;
			  }
			}
		}
	  }

    printf("Error count is %d\n", count);
    return sum;
}

// instantiating all variants

template class golden<float>;
template class golden<double>;





