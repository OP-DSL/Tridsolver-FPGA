#include <iostream>
#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <math.h>






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

void pcr_golden(float* __restrict a, float* __restrict b, float* __restrict c,
		float* __restrict d, float* __restrict u, int N, int stride){

	float* a_0  = (float*) malloc (sizeof(float)*N*2);
	float* c_0  = (float*) malloc (sizeof(float)*N*2);
	float* d_0  = (float*) malloc (sizeof(float)*N*2);

	for(int i = 0; i < N; i++){
		a_0[i] = a[i]/b[i];
		c_0[i] = c[i]/b[i];
		d_0[i] = d[i]/b[i];
	}

	int P = ceil(log2(N));

	for(int p = 1; p <= P; p++){
		int s = pow(2,p-1);
		int off_r = (p % 2 == 1) ? 0 : N;
		int off_w = (p % 2 == 1) ? N : 0;
		for(int i = 0; i < N; i++){
			float r = 1/(1 -a_0[off_r+i] * c_0[off_r+i-s] - c_0[off_r + i] * a_0[off_r+i+s]);
			a_0[off_w+i] = -r * a_0[off_r+i]*a_0[off_r+i-s];
			c_0[off_w+i] = -r * c_0[off_r+i]*c_0[off_r+i+s];
			d_0[off_w+i] = r*(d_0[off_r+i] - a_0[off_r+i]*d_0[off_r+i-s] - c_0[off_r+i]*d_0[off_r+i+s]);

		}
	}

	for(int i = 0; i < N; i++){
		int off_r = (P % 2 == 1) ? 0 : N;
		int off_w = (P % 2 == 1) ? N : 0;
		u[i] = d_0[off_w+i];
	}

	free(a_0);
	free(c_0);
	free(d_0);
}

void thomas_pcr(float* __restrict a, float* __restrict b, float* __restrict c,
		float* __restrict d, float* __restrict u, int N, int stride){

      const int tiles = 32;
      int M = N/tiles;
      if(N%tiles){ std::cout << "Error: N is not a multiple of tiles" << std::endl;}

    for(int i = 0; i < N; i++){
      d[i] /= b[i];
      a[i] /= b[i];
      c[i] /= b[i];
      b[i] /= b[i];
      
    }

    // first phase, thomas solver
    for(int tile = 0; tile < tiles; tile++){
      int ind = tile*M;
      // d[ind] = d[ind]/b[ind];
      // a[ind] = a[ind]/b[ind];
      // c[ind] = c[ind]/b[ind];

    // std::cout << "Before thomas sweep" << std::endl;
    // std::cout << "a_first: " << a[ind] << " a_last:" << a[ind+M-1]  << std::endl;

      for(int i = 2; i < M; i++){
        float r = 1/(b[ind+i] - a[ind+i]*c[ind+i-1]);
        d[ind+i] = r*(d[ind+i] - a[ind+i]*d[ind+i-1]);
        a[ind+i] = -r*a[ind+i]*a[ind+i-1];
        c[ind+i] = r*c[ind+i];

        if(tile == 0){ std::cout << "current a is: " << a[ind+i] << std::endl;}
      }

      for(int i = M-3; i >= 1; i--){
        d[ind+i] = d[ind+i] - c[ind+i]*d[ind+i+1];
        a[ind+i] = a[ind+i] - c[ind+i] * a[ind+i +1];
        c[ind+i] = -c[ind+i]*c[ind+i+1];
      }

      b[ind] = b[ind] - c[ind] * a[ind +1];
      d[ind] = d[ind] - c[ind]*d[ind+1];
      c[ind] = -c[ind]*c[ind+1];

      d[ind] /= b[ind];
      c[ind] /= b[ind];
      a[ind] /= b[ind];
      b[ind] /= b[ind];

      // a[ind] = 0;
    }

    int N_r = tiles*2;
    // middle phase, modified PCR algorithm
    float* a_0  = (float*) malloc (sizeof(float)*N_r*2);
    float* b_0  = (float*) malloc (sizeof(float)*N_r*2);
    float* c_0  = (float*) malloc (sizeof(float)*N_r*2);
    float* d_0  = (float*) malloc (sizeof(float)*N_r*2);
    float* u_0  = (float*) malloc (sizeof(float)*N_r*2);
    
  for(int i = 0; i < tiles; i++){
		a_0[2*i] = a[i*M];
    b_0[2*i] = b[i*M];
		c_0[2*i] = c[i*M];
		d_0[2*i] = d[i*M];

    a_0[2*i+1] = a[i*M+M-1];
    b_0[2*i+1] = b[i*M+M-1];
		c_0[2*i+1] = c[i*M+M-1];
		d_0[2*i+1] = d[i*M+M-1];

    // std::cout << "After thomas sweep" << std::endl;
    // std::cout << "a_first: " << a_0[2*i] << " a_last:" << a_0[2*i+1]  << std::endl;
	}

  // thomas_golden(a_0, b_0, c_0, d_0, u_0, N_r, 1);
  pcr_golden(a_0, b_0, c_0, d_0, u_0, N_r, 1);

  // for(int i = 0; i < N_r; i +=1){
  //   int off_r = 0;
	// 	int off_w = N;

  //   float r = 1/(1 - a_0[off_r+i]*c_0[off_r+i-1] - c_0[off_r+i]*a_0[off_r+i+1]);
  //   d_0[off_w+i] = (i % 2 == 1) ? r*(d_0[off_r+i] - a_0[off_r+i]*d_0[off_r+i-1] - c_0[off_r+i]*d_0[off_r+i+1]) : d_0[off_r+i];
  //   a_0[off_w+i] = (i % 2 == 1) ? -r*a_0[off_r+i]*a_0[off_r+i-1] : a_0[off_r+i];
  //   c_0[off_w+i] = (i % 2 == 1) ? -r*c_0[off_r+i]*c_0[off_r+i+1] : c_0[off_r+i];
  // }

  // int P = ceil(log2(N_r));
  // for(int i = 0; i < N_r; i += 1){
  //   for(int p = 1; p <= P; p++){
  //     int off_r = (p % 2 == 1) ? 0 : N;
  //     int off_w = (p % 2 == 1) ? N : 0;
  //     int s = pow(2,p);
  //     float r = 1/(1 - a_0[off_r+i]*c_0[off_r+i-s] - c_0[off_r+i]*a_0[off_r+i+s]);
  //     d_0[off_w+i] = (i % 2 == 1) ?  r*(d_0[off_r+i] - a_0[off_r+i]*d_0[off_r+i-s] - c_0[off_r+i]*d_0[off_r+i+s]) : d_0[off_r+i];
  //     a_0[off_w+i] = (i % 2 == 1) ? -r*a_0[off_r+i]*a_0[off_r+i-s] : a_0[off_r+i];
  //     c_0[off_w+i] = (i % 2 == 1) ? -r*c_0[off_r+i]*c_0[off_r+i+s] : c_0[off_r+i];
  //   }
  // }

  // for(int i = 0; i < N_r; i++){
  //     int off_r = ((P+1) % 2 == 1) ? 0 : N;
  //     int off_w = ((P+1) % 2 == 1) ? N : 0;
  //     a_0[off_w+i] = a_0[off_r+i];
  //     c_0[off_w+i] = c_0[off_r+i];
  //     d_0[off_w+i] = d_0[off_r+i] - a_0[off_r+i]*d_0[off_r+i-1] - c_0[off_r+i]*d_0[off_r+i+1];
  // }

  // last Phase
  for(int tile = 0; tile < tiles; tile++){

    float u0 = u_0[2*tile]; 
    float uM = u_0[2*tile+1];

    int ind = tile*M;
    u[ind] = u0;
    u[ind+M-1] = uM;
    for(int i = 1; i < M-1; i++){
      float r = 1/(b[ind+i] - a[ind+i]*c[ind+i-1]);
      u[ind+i] = d[ind+i] - a[ind+i]*u0 - c[ind+i]*uM;
    }

  }
  free(a_0);
	free(c_0);
	free(d_0);

}




int initialise_coeff(float* a, float* b, float* c, float* d, int N){
    for(int i = 0; i < N; i++){
    if(i == 0 || i == N-1){
      a[i] = -0.0f;
      b[i] = 1.0f;
      c[i] = 0.0f;
      d[i] = 0.0f;
    } else {
      a[i] = -0.5;
      b[i] = 2.0f;
      c[i] = -0.5f;
      d[i] =  i*0.5f + 2;      
    } 
  }

  return 0;
}


int main(int argc, char** argv){

  int N = 128;
  if(argc == 2){
    N = std::stoi(argv[1]);
  }


  float* a  = (float*) malloc (sizeof(float)*N);
  float* b  = (float*) malloc (sizeof(float)*N);
	float* c  = (float*) malloc (sizeof(float)*N);
	float* d  = (float*) malloc (sizeof(float)*N);
  float* u1  = (float*) malloc (sizeof(float)*N);
  float* u2  = (float*) malloc (sizeof(float)*N);

  initialise_coeff(a,b,c,d,N);
  thomas_golden(a, b, c, d, u1, N, 1);

  initialise_coeff(a,b,c,d,N);
  thomas_pcr(a, b, c, d, u2, N, 1);

  for(int i = 0; i < N; i++){
    float val1 = fabs(u1[i]-u2[i]);
    float val2 = fabs(u1[i])+ fabs(u2[i]);
    if(val1/val2 > 0.000001 || isnan(val1) || isnan(val2)){
      std::cout << "An error has occured at i: " << i  << " val1:" << u1[i] << " val2:" << u2[i] << std::endl;
    }
  }

  return 0;

}


