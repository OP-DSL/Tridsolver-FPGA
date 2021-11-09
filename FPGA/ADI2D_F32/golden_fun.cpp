#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <math.h>
#include "golden_fun.h"

//
// linux timing routine
//
#include <sys/time.h>

/*
 * Print essential infromation on the use of the program
 */
void print_help() {
  printf("Please specify the ADI configuration, e.g.: \n$ ./adi_* -nx NX -ny NY -nz NZ -iter ITER -prof PROF\n");
  exit(0);
}

//__attribute__((target(mic)))
void timing_start(int prof, double *timer) {
  if(prof==1) elapsed_time(timer);
}

//__attribute__((target(mic)))
void timing_end(int prof, double *timer, double *elapsed_accumulate, char *str) {
  double elapsed;
  if(prof==1) {
    elapsed = elapsed_time(timer);
    *elapsed_accumulate += elapsed;
    printf("\n elapsed %s (sec): %1.10f (s) \n", str,elapsed);
  }
}






// Golden Thomas solver implementation 
void thomas_golden(float* __restrict a, float* __restrict b, float* __restrict c,
		float* __restrict d, float* __restrict u, int N, int stride){

		for(int i = 1; i < N; i++){
			int ind = stride*i;
			float w = a[ind] / b[ind-stride];
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



double square_error(float* golden, float* FPGA, int nx, int ny, int nz){
      double sum = 0;
      int count = 0;
	  for(int k =0; k < nz; k++){
		for(int j = 0; j < ny; j++){
			for(int i=0; i<nx; i++) {
			  int Xdim_B = ((nx *nz + 63)/64)* 64;
			  int ind = j*nx + k*nx*ny + i;
			  float val1 = golden[ind];
			  float val2 = FPGA[ind];
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
