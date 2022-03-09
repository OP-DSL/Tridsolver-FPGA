
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
#include <chrono>
#include "xcl2.hpp"
#include "golden_fun.h"
#include "omp.h"
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
  int nx, ny, nz, iter, opt;
//  FP  *__restrict__ h_u, *__restrict__ h_du,
//      *__restrict__ h_ax, *__restrict__ h_bx, *__restrict__ h_cx,
//      *__restrict__ h_ay, *__restrict__ h_by, *__restrict__ h_cy,
//      *__restrict__ h_az, *__restrict__ h_bz, *__restrict__ h_cz,
//      *__restrict__ tmp,
  float  lambda=1.0f; // lam = dt/dx^2

  // Set defaults options
  nx   = 256;
  ny   = 256;
  nz   = 256;
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
    pch = strstr(argv[n], "-iters=");
    if(pch != NULL) {
      iter = atoi ( argv[n] + 7 ); continue;
    }
  }
  // Process arguments
  int opt_index = 0;

  //padding such nx is multiple of 16

  nx = (nx % 16 == 0) ? nx : (nx/16 + 1)* 16;  //

  printf("\nGrid dimensions: %d x %d x %d\n", nx, ny, nz);

//  if( nx>N_MAX /*|| ny>N_MAX || nz>N_MAX*/ ) {
//    printf("Dimension can not exceed N_MAX=%d due to hard-coded local array sizes\n", N_MAX);
//    return -1;
//  }
  // allocate memory for arrays
//  int Xdim_B = ((nx *nz + 63)/64)* 64;

	int w_count, b_count, r_count;
	int ReadLimit_x = ((nx*ny+7)>>3)*(nx);
	int ReadLimit_y = (nx/8)*nz*ny ;
	int min_buffer_size = ((16*nx/8) + std::min(ReadLimit_x, 32*nx)*3 + 2*(nx/8*ny) + std::min(ReadLimit_y, 32*ny)*3)/2;
	int fifo_tr_size = (nx/8*ny*nz)/2;

	if(min_buffer_size >= fifo_tr_size){
		w_count = fifo_tr_size;
		b_count = 0;
		r_count = fifo_tr_size;
	} else {
		w_count = min_buffer_size;
		b_count = fifo_tr_size - min_buffer_size;
		r_count = min_buffer_size;
	}

	printf("w_count:%d, b_count:%d r_count:%d\n", w_count, b_count, r_count);
	if(w_count <= 200 && b_count > 0){
		printf("Hardware might hang, exiting\n");
		exit(0);
	}

  unsigned int total_size_bytes = sizeof(float)*nx*ny*nz + w_count * 64;

  float* h_u  = (float *)aligned_alloc(4096, total_size_bytes);
  float* h_utmp  = (float *)aligned_alloc(4096, total_size_bytes);
  float* h_d = (float *)aligned_alloc(4096, total_size_bytes);


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
  float** d_u = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d_a = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d_buffer1 = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d_buffer2 = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d_buffer3 = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d_buffer4 = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d_b = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d_c = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d_d =(float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d_acc1 = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);
  float** d_acc2 = (float**) aligned_alloc(4096, sizeof(float*) * num_cus);

  for(int i = 0; i < num_cus; i++){
	  d_u[i]  = (float *)aligned_alloc(4096, total_size_bytes);
	  d_a[i] = (float *)aligned_alloc(4096, total_size_bytes);
	  d_b[i] = (float *)aligned_alloc(4096, total_size_bytes);
	  d_c[i] = (float *)aligned_alloc(4096, total_size_bytes);
	  d_d[i] = (float *)aligned_alloc(4096, total_size_bytes);
	  d_buffer1[i] = (float *)aligned_alloc(4096, total_size_bytes);
	  d_buffer2[i] = (float *)aligned_alloc(4096, total_size_bytes);
	  d_buffer3[i] = (float *)aligned_alloc(4096, total_size_bytes);
	  d_buffer4[i] = (float *)aligned_alloc(4096, total_size_bytes);
	  d_acc1[i] = (float *)aligned_alloc(4096, total_size_bytes);
	  d_acc2[i] = (float *)aligned_alloc(4096, total_size_bytes);
  }


	  // Initialize
	  for(int k =0; k < nz; k++){
		for(int j = 0; j < ny; j++){
			for(int i=0; i<nx; i++) {
				ind = k*ny*nx + j*nx +i;
				if(i ==0 || i == nx-1 || j==0 || j==ny-1) {
				  h_u[ind] = 1.0f;

				} else {
				  h_u[ind] = (i+j+k)/100;
				}
				h_acc[ind] = 0.0f;
			}
		}
	  }


	  for(int k =0; k < nz; k++){
		for(int j = 0; j < ny; j++){
			for(int i=0; i < nx; i++) {
				ind = k*ny*nx + j*nx + i;
				float a, b, c;
				if(i ==0 || i == nx-1 || j==0 || j==ny-1) {
				  h_d[ind] = 0.0f;
				  a = 0.0f;
				  b = 1.0f;
				  c = 0.0f;

				} else {
				  h_d[ind] = (h_u[ind-1] + h_u[ind+1] + h_u[ind-nx] + h_u[ind+nx] - h_u[ind]*6.0f);
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


  	for(int i = 0; i < num_cus; i++){
  		memcpy(d_a[i], h_ax, total_size_bytes);
  		memcpy(d_b[i], h_bx, total_size_bytes);
		memcpy(d_c[i], h_cx, total_size_bytes);
		memcpy(d_d[i], h_d, total_size_bytes);
		memcpy(d_acc1[i], h_acc, total_size_bytes);
  	}






  // FPGA initialisation

  auto binaryFile = argv[1];
  std::cout << "binary file is: " << binaryFile << std::endl;

//    cl::Event event;

  auto devices = xcl::get_xil_devices();
  auto device = devices[1];
  cl_int err;
  std::vector<cl::Platform> platform;
  OCL_CHECK(err, err = cl::Platform::get(&platform));
  cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform[0])(), 0};


  OCL_CHECK(err, cl::Context context(device, props, NULL, NULL, &err));
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
  std::vector <cl::Device> dev(1, device);
  OCL_CHECK(err, cl::Program program(context, dev, bins, NULL, &err));

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

  std::vector<cl_mem_ext_ptr_t>  buffer_da_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer_db_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer_dc_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer_dd_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer_du_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer_dbuffer1_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer_dbuffer2_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer_dbuffer3_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer_dbuffer4_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer_acc1_ext(num_cus);
  std::vector<cl_mem_ext_ptr_t>  buffer_acc2_ext(num_cus);

  int da_index[12] = {0, 5, 12, 15, 22,14,18,20,24};
  int db_index[12] = {1, 8, 13, 18, 23,14,18,20,24};
  int dc_index[12] = {2, 9, 14, 19, 24,14,18,20,24};


  int dd_index[4]         = {3, 8,  16,  24};
  int du_index[4]         = {4, 9,  17,  25};
  int acc1_index[4]       = {2, 10, 18,  26};
  int acc2_index[4]       = {3, 11, 19,  27};
  int dbuffer1_index[4]   = {4, 12, 20,  28};
  int dbuffer2_index[4]   = {5, 13, 21,  29};
  int dbuffer3_index[4]   = {6, 14, 22,  30};
  int dbuffer4_index[4]   = {7, 15, 23,  31};

  for(int i = 0; i < num_cus; i++){
	  buffer_da_ext[i].obj = d_a[i];
	  buffer_da_ext[i].param = 0;
	  buffer_da_ext[i].flags = bank[da_index[i]];

	  buffer_db_ext[i].obj = d_b[i];
	  buffer_db_ext[i].param = 0;
	  buffer_db_ext[i].flags = bank[db_index[i]];

	  buffer_dc_ext[i].obj = d_c[i];
	  buffer_dc_ext[i].param = 0;
	  buffer_dc_ext[i].flags = bank[dc_index[i]];

	  buffer_dd_ext[i].obj = d_d[i];
	  buffer_dd_ext[i].param = 0;
	  buffer_dd_ext[i].flags = bank[dd_index[i]];

	  buffer_du_ext[i].obj = d_u[i];
	  buffer_du_ext[i].param = 0;
	  buffer_du_ext[i].flags = bank[du_index[i]];

	  buffer_dbuffer1_ext[i].obj = d_buffer1[i];
	  buffer_dbuffer1_ext[i].param = 0;
	  buffer_dbuffer1_ext[i].flags = bank[dbuffer1_index[i]];

	  buffer_dbuffer2_ext[i].obj = d_buffer2[i];
	  buffer_dbuffer2_ext[i].param = 0;
	  buffer_dbuffer2_ext[i].flags = bank[dbuffer2_index[i]];

	  buffer_dbuffer3_ext[i].obj = d_buffer3[i];
	  buffer_dbuffer3_ext[i].param = 0;
	  buffer_dbuffer3_ext[i].flags = bank[dbuffer3_index[i]];

	  buffer_dbuffer4_ext[i].obj = d_buffer4[i];
	  buffer_dbuffer4_ext[i].param = 0;
	  buffer_dbuffer4_ext[i].flags = bank[dbuffer4_index[i]];

	  buffer_acc1_ext[i].obj = d_acc1[i];
	  buffer_acc1_ext[i].param = 0;
	  buffer_acc1_ext[i].flags = bank[acc1_index[i]];

	  buffer_acc2_ext[i].obj = d_acc2[i];
	  buffer_acc2_ext[i].param = 0;
	  buffer_acc2_ext[i].flags = bank[acc2_index[i]];
  }

  std::vector<cl::Buffer> buffer_da(num_cus);
  std::vector<cl::Buffer> buffer_db(num_cus);
  std::vector<cl::Buffer> buffer_dc(num_cus);
  std::vector<cl::Buffer> buffer_dd(num_cus);
  std::vector<cl::Buffer> buffer_du(num_cus);
  std::vector<cl::Buffer> buffer_dbuffer1(num_cus);
  std::vector<cl::Buffer> buffer_dbuffer2(num_cus);
  std::vector<cl::Buffer> buffer_dbuffer3(num_cus);
  std::vector<cl::Buffer> buffer_dbuffer4(num_cus);
  std::vector<cl::Buffer> buffer_acc1(num_cus);
  std::vector<cl::Buffer> buffer_acc2(num_cus);

  for(int i = 0; i < num_cus; i++){
	  OCL_CHECK(err, buffer_da[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer_da_ext[i], &err));
	  OCL_CHECK(err, buffer_db[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer_db_ext[i], &err));
	  OCL_CHECK(err, buffer_dc[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer_dc_ext[i], &err));
	  OCL_CHECK(err, buffer_dd[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer_dd_ext[i], &err));

	  OCL_CHECK(err, buffer_du[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX| CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer_du_ext[i], &err));
//	  OCL_CHECK(err, buffer_dbuffer1[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX| CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer_dbuffer1_ext[i], &err));
//	  OCL_CHECK(err, buffer_dbuffer2[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX| CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer_dbuffer2_ext[i], &err));
//	  OCL_CHECK(err, buffer_dbuffer3[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX| CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer_dbuffer3_ext[i], &err));
//	  OCL_CHECK(err, buffer_dbuffer4[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX| CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer_dbuffer4_ext[i], &err));
//
//	  OCL_CHECK(err, buffer_acc1[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer_acc1_ext[i], &err));
//	  OCL_CHECK(err, buffer_acc2[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, total_size_bytes, &buffer_acc2_ext[i], &err));
  }

  //Set the Kernel Arguments
  for(int i = 0; i < num_cus; i++){
	  int narg = 0;
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, buffer_da[i]));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, buffer_db[i]));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, buffer_dc[i]));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, buffer_dd[i]));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, buffer_du[i]));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, nx));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, ny));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, nz));
	  OCL_CHECK(err, err = (krnls[i]).setArg(narg++, iter));
	  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_da[i], buffer_db[i], buffer_dc[i], buffer_dd[i], buffer_du[i]}, 0 /* 0 means from host*/));
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
		OCL_CHECK(err, err = q.enqueueMigrateMemObjects({/*buffer_da[i], buffer_db[i], buffer_dc[i],*/ buffer_dd[i], buffer_du[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
	}
    q.finish();



  // Compute sequentially
  elapsed_time(&timer);


  // Pre proc
  omp_set_num_threads(16);
  auto start = std::chrono::steady_clock::now();
  for(int itr = 0; itr < 0*iter; itr++){

	  	  #pragma omp parallel for private(i,j,k)
		  for(int k =0; k < nz; k++){
			for(int j = 0; j < ny; j++){
				for(int i=0; i < nx; i++) {
					ind = k*ny*nx + j*nx + i;
					float a, b, c;
					if(i ==0 || i == nx-1 || j==0 || j==ny-1) {
					  h_d[ind] = 0.0f;
					  a = 0.0f;
					  b = 1.0f;
					  c = 0.0f;

					} else {
					  h_d[ind] = (h_u[ind-1] + h_u[ind+1] + h_u[ind-nx] + h_u[ind+nx] - h_u[ind]*6.0f);
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

//		  solving on x-dimesnion
//		  #pragma omp parallel for
		  for(int i = 0; i < nz; i++){
			  for(int j = 0; j < ny; j++){
				  int ind = i*ny*nx+j*nx;
				  thomas_golden(&h_ax[ind], &h_bx[ind], &h_cx[ind], &h_d[ind], &h_utmp[ind], nx, 1);
			  }
		  }

  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "elapsed time on CPU: " << elapsed_seconds.count() << "s\n";



  for(int i = 0; i <  0*num_cus; i++){
//	  square_error(h_ax, d_a[i], nx, ny, nz);
//	  square_error(h_bx, d_b[i], nx, ny, nz);
//	  square_error(h_cx, d_c[i], nx, ny, nz);
	  square_error(h_utmp, d_u[i], nx, ny, nz);
//	  square_error(h_acc, d_acc1[i], nx, ny, nz);
//	  square_error(h_d, d_d[i], nx, ny, nz);
  }
//  square_error(h_acc, d_acc1, nx*ny*nz);
//  square_error(h_u,  d_u, nx*ny*nz);
//  square_error(h_ax, d_ax, nx*ny*nz);
//  square_error(h_bx, d_bx, nx*ny*nz);
//  square_error(h_cx, d_cx, nx*ny*nz);

  printf("\nComputing ADI on FPGA: %f (s) \n", k_time);
  float bandwidth = iter*5*sizeof(float)* nx* ny*nz/(k_time * 1000000000);
  printf("Bandwidth is %f GB/s\n", bandwidth);

  free(h_d);
  free(h_u);
  free(h_utmp);

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
	  free(d_a[i]);
	  free(d_b[i]);
	  free(d_c[i]);
	  free(d_d[i]);
	  free(d_u[i]);
	  free(d_buffer1[i]);
	  free(d_buffer2[i]);
	  free(d_buffer3[i]);
	  free(d_buffer4[i]);
	  free(d_acc1[i]);
	  free(d_acc2[i]);
  }

  free(d_u);
  free(d_a);
  free(d_b);
  free(d_c);
  free(d_d);
  free(d_buffer1);
  free(d_buffer2);
  free(d_buffer3);
  free(d_buffer4);

  free(d_acc1);
  free(d_acc2);

  printf("Done.\n");



  exit(0);
}


