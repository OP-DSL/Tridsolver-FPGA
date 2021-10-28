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



void thomas_golden(double* __restrict a, double* __restrict b, double* __restrict c,
		double* __restrict d, double* __restrict u, int N, int stride){

//		for(int j = 0; j < batch; j++){
		for(int i = 1; i < N; i++){
			int ind = stride*i; //j*N+i;
//			printf("golden dd_read: %f\n", d[ind]);
			float w = a[ind] / b[ind-stride];
			b[ind] = b[ind] - w * c[ind-stride];
			d[ind] = d[ind] - w * d[ind-stride];
		}
//		}

//		for(int j = 0; j < batch; j++){
		int ind = (N-1)*stride; //(N-1)+j*N;
		u[ind] = d[ind] / b[ind];
//		}

//		for(int j = 0; j < batch; j++){
		for(int i =N-2; i >=0; i--){
			int ind = i*stride; //j*N+i;
			u[ind] = (d[ind] - c[ind] * u[ind+stride]) / b[ind];
//			printf("golden dd_read: %f\n", u[ind]);
		}
//		}

}





//
// tridiagonal solver
//
void trid_cpu(FP* __restrict a, FP* __restrict b, FP* __restrict c, FP* __restrict d, FP* __restrict u, int N, int stride) {
  int   i, ind = 0;
  FP aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
  //
  // forward pass
  //
  bb    = 1.0f/b[0];
  cc    = bb*c[0];
  dd    = bb*d[0];
  c2[0] = cc;
  d2[0] = dd;

  //u[0] = dd;//a[0];
  //*((int*)&u[ind]) = (int)a[0];

  for(i=1; i<N; i++) {
    ind   = ind + stride;
    aa    = a[ind];
    bb    = b[ind] - aa*cc;
    dd    = d[ind] - aa*dd;
    bb    = 1.0f/bb;
    cc    = bb*c[ind];
    dd    = bb*dd;
    c2[i] = cc;
    d2[i] = dd;

    //u[ind] = dd;//a[ind];
    //*((int*)&u[ind]) = (int)a[ind];

  }
  //
  // reverse pass
  //
//  d[ind] = dd;
  u[ind] = dd;//ind;//N-1;//dd;
  for(i=N-2; i>=0; i--) {
    ind    = ind - stride;
    dd     = d2[i] - c2[i]*dd;
//    d[ind] = dd;

    u[ind] = dd;//ind;//i;//dd;//d2[i];//dd;
  }
}

void adi_cpu(FP lambda, FP* __restrict u, FP* __restrict du, FP* __restrict ax, FP* __restrict bx, FP* __restrict cx, FP* __restrict ay, FP* __restrict by, FP* __restrict cy, FP* __restrict az, FP* __restrict bz, FP* __restrict cz, int nx, int ny, int nz, double *elapsed_preproc, double *elapsed_trid_x, double *elapsed_trid_y, double *elapsed_trid_z, int prof) {
  int   i, j, k, ind;
  FP a, b, c, d;
  double elapsed, timer = 0.0;
  //
  // calculate r.h.s. and set tri-diagonal coefficients
  //
  timing_start(prof,&timer);
  for(k=0; k<nz; k++) {
    for(j=0; j<ny; j++) {
      for(i=0; i<nx; i++) {   // i loop innermost for sequential memory access
        ind = k*nx*ny + j*nx + i;
        if(i==0 || i==nx-1 || j==0 || j==ny-1 || k==0 || k==nz-1) {
          d = 0.0f; // Dirichlet b.c.'s
          a = 0.0f;
          b = 1.0f;
          c = 0.0f;
        }
        else {
          d = lambda*(  u[ind-1    ] + u[ind+1]
                      + u[ind-nx   ] + u[ind+nx]
                      + u[ind-nx*ny] + u[ind+nx*ny]
                      - 6.0f*u[ind]);
          a = -0.5f * lambda;
          b =  1.0f + lambda;
          c = -0.5f * lambda;
        }
        du[ind] = d;
        //*((int*)&ax[ind]) = ind;//a;
        ax[ind] = a;
        //ax[ind] = ind;//a;
        bx[ind] = b;
        cx[ind] = c;
        ay[ind] = a;
        by[ind] = b;
        cy[ind] = c;
        az[ind] = a;
        bz[ind] = b;
        cz[ind] = c;
      }
    }
  }
  timing_end(prof,&timer,elapsed_preproc,"preproc");

  //
  // perform tri-diagonal solves in x-direction
  //
  timing_start(prof,&timer);
  for(k=0; k<nz; k++) {
    for(j=0; j<ny; j++) {
      ind = k*nx*ny + j*nx;
      trid_cpu(&ax[ind], &bx[ind], &cx[ind], &du[ind], &u[ind], nx, 1);
    }
  }
  timing_end(prof,&timer,elapsed_trid_x,"trid_x");

  //
  // perform tri-diagonal solves in y-direction
  //
  timing_start(prof,&timer);
  for(k=0; k<nz; k++) {
    for(i=0; i<nx; i++) {
      ind = k*nx*ny + i;
      trid_cpu(&ay[ind], &by[ind], &cy[ind], &du[ind], &u[ind], ny, nx);
    }
  }
  timing_end(prof,&timer,elapsed_trid_y,"trid_y");

  //
  // perform tri-diagonal solves in z-direction
  //
  timing_start(prof,&timer);
  for(j=0; j<ny; j++) {
    for(i=0; i<nx; i++) {
      ind = j*nx + i;
      trid_cpu(&az[ind], &bz[ind], &cz[ind], &du[ind], &u[ind], nz, nx*ny);
      //#pragma ivdep
      //for(k=0; k<NZ; k++) {
      //  u[ind] += du[ind];
      //  //u[ind] = du[ind];
      //  ind    += NX*NY;
      //}
    }
  }

  for(k=0; k<nz; k++) {
    for(j=0; j<ny; j++) {
      for(i=0; i<nx; i++) {
        ind = k*nx*ny + j*nx + i;
        u[ind] += du[ind];
        //u[ind] = du[ind];
      }
    }
  }
  timing_end(prof,&timer,elapsed_trid_z,"trid_z");
}


double square_error(double* golden, double* FPGA, int size){
      double sum = 0;
      int count = 0;
      for(int i = 0; i < size; i = i + 1){
    	  double val1 = golden[i];
    	  double val2 = FPGA[i];
//		  if(i < 100000){
//			  printf("i:%d golden:%f FPGA:%f\n", i, val1, val2);
//		  }
		  sum +=  val1*val1 - val2*val2;
		  if((fabs(val1 -val2)/(fabs(val1) + fabs(val2))) > 0.001 && (fabs(val1) + fabs(val2)) > 0.000001 || isnan(val1) || isnan(val2)){
			  printf("i:%d golden:%f FPGA:%f\n", i, val1, val2);
			  count++;
		  }
      }

    printf("Error count is %d\n", count);
    return sum;
}
