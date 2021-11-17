
/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the scalar-tridiagonal solver distribution.
 *
 * Copyright (c) 2015, Endre László and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Endre László may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Endre László ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Endre László BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <omp.h>
#include <chrono>
#include "xcl2.hpp"
#include "golden_fun.h"
//
// linux timing routine
//



#include <sys/time.h>


// hardware bank
#define MAX_HBM_BANKCOUNT 32
#define BANK_NAME(n) n | XCL_MEM_TOPOLOGY
const int bank[MAX_HBM_BANKCOUNT] = {
    BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),
    BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),
    BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),
    BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
    BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),
    BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29),
    BANK_NAME(30), BANK_NAME(31)};



int main(int argc, char* argv[]) {
  double timer, timer2, elapsed, elapsed_total, elapsed_preproc, elapsed_trid_x, elapsed_trid_y, elapsed_trid_z;

  // 'h_' prefix - CPU (host) memory space

  int i, j, k, ind, it;
  int nx, ny, nz, iter, batch, opt;
//  FP  *__restrict__ h_u, *__restrict__ h_du,
//      *__restrict__ h_ax, *__restrict__ h_bx, *__restrict__ h_cx,
//      *__restrict__ h_ay, *__restrict__ h_by, *__restrict__ h_cy,
//      *__restrict__ h_az, *__restrict__ h_bz, *__restrict__ h_cz,
//      *__restrict__ tmp,
  float  lambda=1.0f; // lam = dt/dx^2

  // Set defaults options
  nx   = 128;
  ny   = 128;
  nz   = 128;
  batch = 1;
  iter = 1;
  opt  = 0;



  const char* pch;
  for ( int n = 1; n < argc; n++ ) {
    pch = strstr(argv[n], "-nx=");
    if(pch != NULL) {
      nx = atoi ( argv[n] + 4 ); continue;
    }
    pch = strstr(argv[n], "-ny=");
    if(pch != NULL) {
      ny = atoi ( argv[n] + 4 ); continue;
    }
    pch = strstr(argv[n], "-nz=");
	if(pch != NULL) {
	  nz = atoi ( argv[n] + 4 ); continue;
	}
    pch = strstr(argv[n], "-batch=");
	if(pch != NULL) {
	  batch = atoi ( argv[n] + 7 ); continue;
	}
    pch = strstr(argv[n], "-iters=");
    if(pch != NULL) {
      iter = atoi ( argv[n] + 7 ); continue;
    }
  }
  // Process arguments
  int opt_index = 0;

  //padding such nx is multiple of 16
  nx = (nx % 16 == 0) ? nx : (nx/16 + 1)* 16;  //

  printf("\nGrid dimensions: %d x %d x %d x %d\n", nx, ny, nz, batch);

  if( nx>N_MAX /*|| ny>N_MAX || nz>N_MAX*/ ) {
    printf("Dimension can not exceed N_MAX=%d due to hard-coded local array sizes\n", N_MAX);
    return -1;
  }
  // allocate memory for arrays

  unsigned int total_size_bytes = sizeof(float)*nx*(((ny*nz*batch+7) >> 3) << 3)+1024*1024*8;

  float* h_u  = (float *)aligned_alloc(4096, total_size_bytes);
  float* h_du = (float *)aligned_alloc(4096, total_size_bytes);

  float* h_ax = (float *)aligned_alloc(4096, total_size_bytes);
  float* h_bx = (float *)aligned_alloc(4096, total_size_bytes);
  float* h_cx = (float *)aligned_alloc(4096, total_size_bytes);

  float* h_ay = (float *)aligned_alloc(4096, total_size_bytes);
  float* h_by = (float *)aligned_alloc(4096, total_size_bytes);
  float* h_cy = (float *)aligned_alloc(4096, total_size_bytes);

  float* h_az = (float *)aligned_alloc(4096, total_size_bytes);
  float* h_bz = (float *)aligned_alloc(4096, total_size_bytes);
  float* h_cz = (float *)aligned_alloc(4096, total_size_bytes);
  float* h_acc = (float *)aligned_alloc(4096, total_size_bytes);


  int num_cus = 1;
  float** d1_u = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d1_du =(float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d1_acc1 = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d1_acc2 = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);

  float** d2_u = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d2_du =(float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d2_acc1 = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d2_acc2 = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);

  for(int i = 0; i < num_cus; i++){
	  d1_u[i]  = (float *)aligned_alloc(4096, total_size_bytes);
	  d1_du[i] = (float *)aligned_alloc(4096, total_size_bytes);
	  d1_acc1[i] = (float *)aligned_alloc(4096, total_size_bytes);
	  d1_acc2[i] = (float *)aligned_alloc(4096, total_size_bytes);

	  d2_u[i]  = (float *)aligned_alloc(4096, total_size_bytes);
	  d2_du[i] = (float *)aligned_alloc(4096, total_size_bytes);
	  d2_acc1[i] = (float *)aligned_alloc(4096, total_size_bytes);
	  d2_acc2[i] = (float *)aligned_alloc(4096, total_size_bytes);
  }


  // Initialize

  for(int bat = 0; bat < batch; bat++){
	  for(int k =0; k < nz; k++){
		for(int j = 0; j < ny; j++){
			for(int i=0; i<nx; i++) {
				ind = bat * nx*ny*nz +k*nx*ny+j*nx+i;
				if(i ==0 || i == nx-1 || j==0 || j==ny-1 || k == 0 || k == nz-1) {
				  h_du[ind] = 1.0f;

				} else {
				  h_du[ind] = 0.0f;
				}
				h_acc[ind] = 0;
			}
		}
	  }
  }

  	for(int i = 0; i < num_cus; i++){
		memcpy(d1_du[i], h_du, total_size_bytes);
		memcpy(d1_u[i], h_u, total_size_bytes);
		memcpy(d1_acc1[i], h_acc, total_size_bytes);
		memcpy(d2_du[i], h_du, total_size_bytes);
		memcpy(d2_u[i], h_u, total_size_bytes);
		memcpy(d2_acc1[i], h_acc, total_size_bytes);
  	}


    // golden computation
  	// Pre proc
  	golden<float> Gold;
  	omp_set_num_threads(1);
    for(int itr = 0; itr < 2*iter; itr++){
  	  #pragma omp parallel for
  	  for(int bat = 0; bat < batch; bat++){
  		  for(int k =0; k < nz; k++){
  			for(int j = 0; j < ny; j++){
  				for(int i=0; i<nx; i++) {
  					ind = bat * nx*ny*nz +k*nx*ny+j*nx+i;
  					float a, b, c;
  					if(i ==0 || i == nx-1 || j==0 || j==ny-1 || k == 0 || k == nz-1) {
  					  h_u[ind] = 0.0f;
  					  a = 0.0f;
  					  b = 1.0f;
  					  c = 0.0f;

  					} else {
  					  h_u[ind] = (h_du[ind-1] + h_du[ind+1] + h_du[ind-nx] + h_du[ind+nx] + h_du[ind-nx*ny] + h_du[ind+nx*ny] - h_du[ind]*6.0f);
  					  a = -0.5f * lambda;
  					  b =  1.0f + lambda;
  					  c = -0.5f * lambda;
  					}

  					h_ax[ind] = a;
  					h_bx[ind] = b;
  					h_cx[ind] = c;

  					h_ay[ind] = a;
  					h_by[ind] = b;
  					h_cy[ind] = c;

  					h_az[ind] = a;
  					h_bz[ind] = b;
  					h_cz[ind] = c;
  				}
  			}
  		  }
  	  }

  		//  solving on x-dimesnion
  	   #pragma omp parallel for
  	   for(int bat = 0; bat < batch; bat++){
  		  for(int i = 0; i < nz; i++){
  			  for(int j = 0; j < ny; j++){
  				  int ind = bat* nx*ny*nz + i*nx*ny+j*nx;
  				  Gold.thomas_golden(&h_ax[ind], &h_bx[ind], &h_cx[ind], &h_u[ind], &h_du[ind], nx, 1);
  			  }
  		  }
  	   }

  //		  // solving on y-direction
  	   #pragma omp parallel for
  	   for(int bat = 0; bat < batch; bat++){
  		  for(int i =0; i < nx; i++){
  			  for(int j = 0; j < nz; j++){
  				  int ind =  bat* nx*ny*nz + i+j*nx*ny;
  				  Gold.thomas_golden(&h_ay[ind], &h_by[ind], &h_cy[ind], &h_du[ind], &h_u[ind], ny, nx);
  			  }
  		  }
  	   }
  	   if(itr == 0){
  		  	for(int i = 0; i < num_cus; i++){
  				memcpy(d2_u[i], h_u, total_size_bytes);
  		  	}
  	   }
  //
  //		  // solving on z direction
  	   #pragma omp parallel for
  	   for(int bat = 0; bat < batch; bat++){
  			for(int i =0; i < ny; i++){
  			  for(int j = 0; j < nx; j++){
  				  int ind = bat* nx*ny*nz + i*nx+j;
  				  Gold.thomas_golden(&h_az[ind], &h_bz[ind], &h_cz[ind], &h_u[ind], &h_du[ind], nz, nx*ny);
  			  }
  			}
  	   }

		#pragma omp parallel for
  	   for(int bat = 0; bat < batch; bat++){
  			for(int k =0; k < nz; k++){
  				for(int j = 0; j < ny; j++){
  					for(int i=0; i<nx; i++) {
  						int ind = bat* nx*ny*nz + k*nx*ny+j*nx+i;
  						h_acc[ind] += h_du[ind];
  						if(itr != 2*iter-1){
  							h_du[ind] = h_acc[ind];
  						}
  					}
  				}
  			}
  	   }
    }




  // FPGA initialisation

  auto binaryFile = argv[1];
  std::cout << "binary file is: " << binaryFile << std::endl;

//    cl::Event event;

  auto devices = xcl::get_xil_devices();
  auto device = devices[0];
  cl_int err;
  OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
  OCL_CHECK(
      err,
      cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
  OCL_CHECK(err,
            std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));


  //Create Program and Kernel
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  devices.resize(1);
  auto start_p = std::chrono::high_resolution_clock::now();
  OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

  std::vector<cl::Kernel> krnls(num_cus);
  for(int i = 0; i < num_cus; i++){
	  std::string cu_id = std::to_string(i+1);
	  std::string krnl_name = "TDMA_batch";
	  std::string krnl_name_full = krnl_name + ":{" + "TDMA_batch_" + cu_id + "}";
	  OCL_CHECK(err, krnls[i] = cl::Kernel(program, krnl_name_full.c_str(), &err));
  }

  auto end_p = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dur_p = end_p -start_p;
  printf("time to program FPGA is %f\n", dur_p.count());



  //Allocate Buffer in Global Memory

  std::vector<cl_mem_ext_ptr_t>  buffer1_u_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer1_du_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer1_acc1_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer1_acc2_ext(num_cus);

  std::vector<cl_mem_ext_ptr_t>  buffer2_u_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer2_du_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer2_acc1_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer2_acc2_ext(num_cus);

  int u_index1[9] =  {0, 4, 8, 12, 16, 20};
  int du_index1[9] = {1, 5, 9, 13, 17, 21};
  int acc_index1[9] ={4, 5, 10, 11, 16, 17, 22, 23, 26};

  int u_index2[9] =  {2, 6, 10, 14, 18, 22};
  int du_index2[9] = {3, 7, 11, 15, 19, 23};
  int acc_index2[9] ={4, 5, 10, 11, 16, 17, 22, 23, 26};

  for(int i = 0; i < num_cus; i++){
	  buffer1_u_ext[i].obj = d1_u[i];
	  buffer1_u_ext[i].param = 0;
	  buffer1_u_ext[i].flags = bank[u_index1[i]];

	  buffer1_du_ext[i].obj = d1_du[i];
	  buffer1_du_ext[i].param = 0;
	  buffer1_du_ext[i].flags = bank[du_index1[i]];

	  buffer1_acc1_ext[i].obj = d1_acc1[i];
	  buffer1_acc1_ext[i].param = 0;
	  buffer1_acc1_ext[i].flags = bank[du_index1[i]];

	  buffer1_acc2_ext[i].obj = d1_acc2[i];
	  buffer1_acc2_ext[i].param = 0;
	  buffer1_acc2_ext[i].flags = bank[u_index1[i]];

	  // second
	  buffer2_u_ext[i].obj = d2_u[i];
	  buffer2_u_ext[i].param = 0;
	  buffer2_u_ext[i].flags = bank[u_index2[i]];

	  buffer2_du_ext[i].obj = d2_du[i];
	  buffer2_du_ext[i].param = 0;
	  buffer2_du_ext[i].flags = bank[du_index2[i]];

	  buffer2_acc1_ext[i].obj = d2_acc1[i];
	  buffer2_acc1_ext[i].param = 0;
	  buffer2_acc1_ext[i].flags = bank[du_index2[i]];

	  buffer2_acc2_ext[i].obj = d2_acc2[i];
	  buffer2_acc2_ext[i].param = 0;
	  buffer2_acc2_ext[i].flags = bank[u_index2[i]];
  }

  std::vector<cl::Buffer> buffer1_u(num_cus);
  std::vector<cl::Buffer> buffer1_du(num_cus);
  std::vector<cl::Buffer> buffer1_acc1(num_cus);
  std::vector<cl::Buffer> buffer1_acc2(num_cus);

  std::vector<cl::Buffer> buffer2_u(num_cus);
  std::vector<cl::Buffer> buffer2_du(num_cus);
  std::vector<cl::Buffer> buffer2_acc1(num_cus);
  std::vector<cl::Buffer> buffer2_acc2(num_cus);

  for(int i = 0; i < num_cus; i++){
	  OCL_CHECK(err, buffer1_u[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer1_u_ext[i], &err));
	  OCL_CHECK(err, buffer1_du[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX| CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer1_du_ext[i], &err));
	  OCL_CHECK(err, buffer1_acc1[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer1_acc1_ext[i], &err));
	  OCL_CHECK(err, buffer1_acc2[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer1_acc2_ext[i], &err));

	  OCL_CHECK(err, buffer2_u[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer2_u_ext[i], &err));
	  OCL_CHECK(err, buffer2_du[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX| CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer2_du_ext[i], &err));
	  OCL_CHECK(err, buffer2_acc1[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer2_acc1_ext[i], &err));
	  OCL_CHECK(err, buffer2_acc2[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer2_acc2_ext[i], &err));
  }

  //Set the Kernel Arguments
  for(int i = 0; i < num_cus; i++){
	  int narg = 0;
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, buffer1_du[i]));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, buffer1_u[i]));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, buffer1_acc1[i]));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, buffer1_acc2[i]));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, buffer2_du[i]));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, buffer2_u[i]));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, buffer2_acc1[i]));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, buffer2_acc2[i]));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, nx));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, ny));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, nz));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, batch));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, iter));
	  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer1_u[i], buffer1_du[i], buffer1_acc1[i], buffer2_u[i], buffer2_du[i], buffer2_acc1[i]}, 0 /* 0 means from host*/));
  }


  //Copy input data to device global memory
//  for(int i = 0; i < num_cus; i++){

//  }
  q.finish();

	//Launch the Kernel
  std::vector<cl::Event> event(num_cus);
  for(int i = 0; i < num_cus; i++){
	  OCL_CHECK(err, err = q.enqueueTask(krnls[i], NULL, &event[i]));
  }

  for(int i = 0; i < num_cus; i++){
	  OCL_CHECK(err, err=event[i].wait());
  }

    uint64_t max = 0;
	for(int i = 0; i < num_cus; i++){
		uint64_t endns = OCL_CHECK(err, event[i].getProfilingInfo<CL_PROFILING_COMMAND_END>(&err));
		uint64_t startns = OCL_CHECK(err, event[i].getProfilingInfo<CL_PROFILING_COMMAND_START>(&err));
		uint64_t nsduration = endns - startns;
		if(max < nsduration){
			max = nsduration;
		}
	}

	double k_time = max/(1000000000.0);

	q.finish();
	for(int i = 0; i < num_cus; i++){
		OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer1_u[i], buffer1_du[i], buffer1_acc1[i], buffer1_acc2[i], buffer2_u[i], buffer2_du[i], buffer2_acc1[i], buffer2_acc2[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
	}
    q.finish();



  // Compute sequentially
  elapsed_time(&timer);







  for(int i = 0; i < num_cus; i++){
	  Gold.square_error(h_du, d1_du[i], nx,ny,nz*batch);
	  Gold.square_error(h_du, d2_du[i], nx,ny,nz*batch);
  }
//  square_error(h_acc, d_acc1, nx*ny*nz);
//  square_error(h_u,  d_u, nx*ny*nz);
//  square_error(h_ax, d_ax, nx*ny*nz);
//  square_error(h_bx, d_bx, nx*ny*nz);
//  square_error(h_cx, d_cx, nx*ny*nz);

  printf("\nComputing ADI on FPGA: %f (s) \n", k_time);
  float bandwidth = iter* 4*2*sizeof(float)* nx* ny*nz*batch/(k_time * 1000000000);
  printf("Bandwidth is %f GB/s\n", bandwidth);

  free(h_u);
  free(h_du);

  free(h_ax);
  free(h_bx);
  free(h_cx);

  free(h_ay);
  free(h_by);
  free(h_cy);

  free(h_az);
  free(h_bz);
  free(h_cz);

  for(int i = 0; i < num_cus; i++){
	  free(d1_u[i]);
	  free(d1_du[i]);
	  free(d1_acc1[i]);
	  free(d1_acc2[i]);

	  free(d2_u[i]);
	  free(d2_du[i]);
	  free(d2_acc1[i]);
	  free(d2_acc2[i]);
  }

  free(d1_u);
  free(d1_du);
  free(d1_acc1);
  free(d1_acc2);

  free(d2_u);
  free(d2_du);
  free(d2_acc1);
  free(d2_acc2);

  printf("Done.\n");



  exit(0);
}


