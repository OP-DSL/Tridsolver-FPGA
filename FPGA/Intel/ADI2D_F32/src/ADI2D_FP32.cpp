//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of DPC++. This sample runs on both CPU and GPU (or FPGA). When run, it
// computes on both the CPU and offload device, then compares results. If the
// code executes on both CPU and offload device, the device name and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// DPC++ material used in the code sample:
// •  A one dimensional array of data.
// •  A device queue, buffer, accessor, and kernel.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>
#endif

#include "data_types.h"
#include "stencils.hpp"
#include "BThomas.hpp"
#include "DPath.hpp"

// #include "BThomas.hpp"


#define UFACTOR 2

template <class DType>
 void thomas_golden(DType* __restrict a, DType* __restrict b, DType* __restrict c,
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



template <size_t idx>  struct stencil_read_write_id;
template<int VFACTOR, int idx1, int idx2>
event stencil_read_write(queue &q, buffer<struct dPath16, 1> &in_buf, buffer<struct dPath16, 1> &out_buf,
                 int total_itr , ac_int<12,true> n_iter, int delay){

      event e1 = q.submit([&](handler &h) {
      accessor in(in_buf, h);
      accessor out(out_buf, h);
      // int total_itr = ((nx*ny)*(nz))/VFACTOR;

      h.single_task<class stencil_read_write_id<idx1>>([=] () [[intel::kernel_args_restrict]]{

      [[intel::disable_loop_pipelining]]
      for(ac_int<12,true> itr = 0; itr < n_iter; itr++){

        accessor ptrR = ((itr & 1) == 0) ? in : out;
        accessor ptrW = ((itr & 1) == 1) ? in : out;

        [[intel::ivdep]]
        [[intel::initiation_interval(1)]]
        for(int i = 0; i < total_itr+delay; i++){
          struct dPath16 vecR = ptrR[i+delay];
          if(i < total_itr){
            pipeM::PipeAt<idx1>::write(vecR);
          }
          struct dPath16 vecW;
          if(i >= delay){
            vecW = pipeM::PipeAt<idx2>::read();
          }
          ptrW[i] = vecW;
        }
      }
        
      });
      });

      return e1;
}

template <size_t idx>  struct PipeConvert_512_256_id;
template<int VFACTOR, int idx1, int idx2>
void PipeConvert_512_256(queue &q, int total_itr, ac_int<12,true> n_iter){
      event e1 = q.submit([&](handler &h) {

      ac_int<40,true> count = total_itr*n_iter;
      h.single_task<class PipeConvert_512_256_id<idx2>>([=] () [[intel::kernel_args_restrict]]{
        struct dPath16 data16;
        [[intel::initiation_interval(1)]]
        for(ac_int<40,true> i = 0; i < count; i++){
          struct dPath data;
          if((i&1) == 0){
            data16 = pipeM::PipeAt<idx1>::read();
          }

          #pragma unroll VFACTOR
          for(int v = 0; v < VFACTOR; v++){
            if((i&1) == 0){
              data.data[v] = data16.data[v];
            } else {
              data.data[v] = data16.data[v+VFACTOR];
            }
          }
          pipeS::PipeAt<idx2>::write(data);
        }
        
      });
    });
}

template <size_t idx>  struct PipeS2B_id;
template<int idx1, int idx2>
void PipeS2B(queue &q, int total_itr, ac_int<12,true> n_iter){
      event e1 = q.submit([&](handler &h) {

      ac_int<40,true> count = total_itr*n_iter;
      h.single_task<class PipeS2B_id<idx2>>([=] () [[intel::kernel_args_restrict]]{
        struct dPath16 data16;
        [[intel::initiation_interval(1)]]
        for(ac_int<40,true> i = 0; i < count; i++){
          struct dPath data;
          data = pipeS::PipeAt<idx1>::read();
          pipeB::PipeAt<idx2>::write(data);
        }
        
      });
    });
}

template <size_t idx>  struct PipeB2S_id;
template<int idx1, int idx2>
void PipeB2S(queue &q, int total_itr, ac_int<12,true> n_iter){
      event e1 = q.submit([&](handler &h) {

      ac_int<40,true> count = total_itr*n_iter;
      h.single_task<class PipeB2S_id<idx2>>([=] () [[intel::kernel_args_restrict]]{
        struct dPath16 data16;
        [[intel::initiation_interval(1)]]
        for(ac_int<40,true> i = 0; i < count; i++){
          struct dPath data;
          data = pipeB::PipeAt<idx1>::read();
          pipeS::PipeAt<idx2>::write(data);
        }
        
      });
    });
}




template <size_t idx>  struct PipeConvert_256_512_id;
template <int VFACTOR, int idx1, int idx2>
void PipeConvert_256_512(queue &q, int total_itr,  ac_int<12,true> n_iter){
    ac_int<40,true>  count = total_itr*n_iter;
    event e3 = q.submit([&](handler &h) {
    h.single_task<class PipeConvert_256_512_id<idx1>>([=] () [[intel::kernel_args_restrict]]{
      struct dPath16 data16;
      [[intel::initiation_interval(1)]]
      for(ac_int<40,true>  i = 0; i < count; i++){
        struct dPath data;
        data = pipeS::PipeAt<idx1>::read();
        #pragma unroll VFACTOR
        for(int v = 0; v < VFACTOR; v++){
          if((i & 1) == 0){
            data16.data[v] = data.data[v];
          } else {
            data16.data[v+VFACTOR] = data.data[v];
          }
        }
        if((i & 1) == 1){
          pipeM::PipeAt<idx2>::write(data16);
        }
      }

      
      
    });
    });
}


template<int idx, int idx1>
static void trid_solver(queue &q, struct data_G data_g, int nx, int ny, int nz, int n_iter, int total_itr_8, int B_X, int ReadLimit_X, int B_X_r, int count_limit_x, 
              int B_Y, int ReadLimit_Y, int B_Y_r, int count_limit_y

    ){

      stencil_2d<0, float, D_MAX, 0+idx, 1+idx, 2000+idx1, 2001+idx1>(q, data_g, n_iter, 0);
      PipeS2B<2001+idx1,idx1>(q, total_itr_8, n_iter);
      PipeB2S<idx1,2002+idx1>(q, total_itr_8, n_iter);

      interleaved_row_block8<0, D_MAX, 1+idx, 4+idx>(q, nx, ny, nz, n_iter, true);
      stream_8x8transpose<0, float, 4+idx, 5+idx>(q, nx, ny, nz, n_iter, true);
      thomas_interleave<0, float, D_MAX, 5+idx, 6+idx, 8+idx>(q, nx, B_X, ReadLimit_X, n_iter);
      thomas_generate_r<0, float, D_MAX, 7+idx, 8+idx>(q, nx, B_X_r, count_limit_x, n_iter);
      thomas_forward<0, float, D_MAX, 6+idx, 9+idx>(q, nx, B_X, n_iter);
      thomas_backward<0, float, D_MAX, 9+idx, 11+idx>(q, nx, B_X, ReadLimit_X, n_iter);
      stream_8x8transpose<0, float, 11+idx, 12+idx>(q, nx, ny, nz, n_iter, true);
      undo_interleaved_row_block8<0, D_MAX, 12+idx, 13+idx>(q, nx, ny, nz, n_iter, true);

      row2col<0, D_MAX, 13+idx, 14+idx>(q, nx, ny, nz, n_iter);
      thomas_interleave<0, float, D_MAX, 14+idx, 15+idx, 17+idx>(q, ny, B_Y, ReadLimit_Y, n_iter);
      thomas_generate_r<0, float, D_MAX, 16+idx, 17+idx>(q, ny, B_Y_r, count_limit_y, n_iter);
      thomas_forward<0, float, D_MAX, 15+idx, 18+idx>(q, ny, B_Y, n_iter);
      thomas_backward<0, float, D_MAX, 18+idx, 20+idx>(q, ny, B_Y, ReadLimit_Y, n_iter);
      col2row<0, D_MAX, 20+idx, 21+idx>(q, nx, ny, nz, n_iter);
  }

template <int N> struct loop {
  static void instantiate(queue &q, struct data_G data_g, int nx, int ny, int nz, int n_iter, int total_itr_8, int B_X, int ReadLimit_X, int B_X_r, int count_limit_x, 
              int B_Y, int ReadLimit_Y, int B_Y_r, int count_limit_y){

      loop<N-1>::instantiate(q, data_g, nx, ny, nz, n_iter, total_itr_8, B_X, ReadLimit_X, B_X_r, count_limit_x, 
              B_Y, ReadLimit_Y, B_Y_r, count_limit_y);
      trid_solver<N*21, 2*N>(q, data_g, nx, ny, nz, n_iter, total_itr_8, B_X, ReadLimit_X, B_X_r, count_limit_x, 
              B_Y, ReadLimit_Y, B_Y_r, count_limit_y);


  }
};

template <> 
struct loop<0> {
  static void instantiate(queue &q, struct data_G data_g, int nx, int ny, int nz, int n_iter, int total_itr_8, int B_X, int ReadLimit_X, int B_X_r, int count_limit_x, 
              int B_Y, int ReadLimit_Y, int B_Y_r, int count_limit_y){

      trid_solver<0, 0>(q, data_g, nx, ny, nz, n_iter, total_itr_8, B_X, ReadLimit_X, B_X_r, count_limit_x, 
              B_Y, ReadLimit_Y, B_Y_r, count_limit_y);


  }
};



//************************************
// Vector add in DPC++ on device: returns sum in 4th parameter "sum_parallel".
//************************************
void stencil_comp(queue &q, IntVector &input, IntVector &output, IntVector &acc_1, IntVector &acc_2, 
                  int n_iter, int nx, int ny, int nz, int delay1, int delay2) {
  // Create the range object for the vectors managed by the buffer.
  range<1> num_items{input.size()};
  int vec_size = input.size();

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer in_buf(input);
  buffer out_buf(output);
  buffer acc1_buf(acc_1);
  buffer acc2_buf(acc_2);

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  double kernel_time = 0;
  std::cout << "starting writing to the pipe\n" << std::endl;
  dpc_common::TimeInterval exe_time;


    int total_itr_16 = ((nx*ny)*(nz))/(16);
    int total_itr_8 = ((nx*ny)*(nz))/(8);
    int total_itrS = ((nx/8)*(ny*nz+1));


      struct data_G data_g;
      data_g.sizex = nx-2;
      data_g.sizey = ny-2;
      data_g.xdim0 = nx;
      data_g.end_index = (nx >> 3);
      data_g.end_row = ny;
      data_g.outer_loop_limit = ny+1; // n + D/2
      data_g.gridsize = (data_g.end_row* nz + 1) * data_g.end_index;
      data_g.endindex_minus1 = data_g.end_index -1;
      data_g.endrow_plus1 = data_g.end_row + 1;
      data_g.endrow_plus2 = data_g.end_row + 2;
      data_g.endrow_minus1 = data_g.end_row - 1;
      data_g.total_itr_256 = data_g.end_row * data_g.end_index * nz;

   


      // thomas solver parameters
      int XBlocks = (nx >> 3);

      int ReadLimit_X = ((nz*ny+7)>>3)*(XBlocks << 3);
      int ReadLimit_Y = XBlocks*nz*ny ;


      int block_g =9*8;
      int B_X = (nz*ny+block_g-1)/block_g;
      int B_Y = ((nx*nz+block_g-1)/block_g);

      int block_g_r =45*8;
      int B_X_r = (nz*ny+block_g_r-1)/block_g_r;
      int B_Y_r = ((nx*nz+block_g_r-1)/block_g_r);

      int count_limit_x = (B_X*block_g*nx)/8;
      int count_limit_y = (B_Y*block_g*ny)/8;

    // reading from memory
      event e = stencil_read_write<16,0, 1>(q, in_buf, out_buf, total_itr_16, n_iter, delay1);
      stencil_read_write<16, 2, 3>(q, acc1_buf, acc2_buf, total_itr_16, n_iter, delay2);

      PipeConvert_512_256<8, 0, 0>(q, total_itr_8, n_iter);
      PipeConvert_512_256<8, 2, 2000>(q, total_itr_8, n_iter);
      


      #define SHIFT1 17
      // Iter one

      loop<UFACTOR-1>::instantiate(q, data_g, nx, ny, nz, n_iter, total_itr_8, B_X, ReadLimit_X, B_X_r, count_limit_x, 
              B_Y, ReadLimit_Y, B_Y_r, count_limit_y);


      // stencil_2d<0, float, D_MAX, 0, 1, 500, 501>(q, data_g, n_iter, 0);
      // // PipeS2B<501,0>(q, total_itr_8, n_iter);
      // // PipeB2S<0,502>(q, total_itr_8, n_iter);

      // interleaved_row_block8<0, D_MAX, 1, 2>(q, nx, ny, nz, n_iter, true);
      // stream_8x8transpose<0, float, 2, 3>(q, nx, ny, nz, n_iter, true);
      // thomas_interleave<0, float, D_MAX, 3, 4, 300>(q, nx, B_X, ReadLimit_X, n_iter);
      // thomas_generate_r<0, float, D_MAX, 5, 300>(q, nx, B_X_r, count_limit_x, n_iter);
      // thomas_forward<0, float, D_MAX, 4, 6>(q, nx, B_X, n_iter);
      // thomas_backward<0, float, D_MAX, 6, 8>(q, nx, B_X, ReadLimit_X, n_iter);
      // stream_8x8transpose<0, float, 8, 9>(q, nx, ny, nz, n_iter, true);
      // undo_interleaved_row_block8<0, D_MAX, 9, 10>(q, nx, ny, nz, n_iter, true);

      // row2col<0, D_MAX, 10, 11>(q, nx, ny, nz, n_iter);
      // thomas_interleave<0, float, D_MAX, 11, 12, 301>(q, ny, B_Y, ReadLimit_Y, n_iter);
      // thomas_generate_r<0, float, D_MAX, 13, 301>(q, ny, B_Y_r, count_limit_y, n_iter);
      // thomas_forward<0, float, D_MAX, 12, 14>(q, ny, B_Y, n_iter);
      // thomas_backward<0, float, D_MAX, 14, 16>(q, ny, B_Y, ReadLimit_Y, n_iter);
      // col2row<0, D_MAX, 16, 17>(q, nx, ny, nz, n_iter);

      // Iter Two
      // stencil_2d<0, float, D_MAX, 0+SHIFT1, 1+SHIFT1, 501, 503>(q, data_g, n_iter, 0);
      // // PipeS2B<503,1>(q, total_itr_8, n_iter);
      // // PipeB2S<1,504>(q, total_itr_8, n_iter);

      // interleaved_row_block8<0, D_MAX, 1+SHIFT1, 2+SHIFT1>(q, nx, ny, nz, n_iter, true);
      // stream_8x8transpose<0, float, 2+SHIFT1, 3+SHIFT1>(q, nx, ny, nz, n_iter, true);
      // thomas_interleave<0, float, D_MAX, 3+SHIFT1, 4+SHIFT1, 300+SHIFT1>(q, nx, B_X, ReadLimit_X, n_iter);
      // thomas_generate_r<0, float, D_MAX, 5+SHIFT1, 300+SHIFT1>(q, nx, B_X_r, count_limit_x, n_iter);
      // thomas_forward<0, float, D_MAX, 4+SHIFT1, 6+SHIFT1>(q, nx, B_X, n_iter);
      // thomas_backward<0, float, D_MAX, 6+SHIFT1, 8+SHIFT1>(q, nx, B_X, ReadLimit_X, n_iter);
      // stream_8x8transpose<0, float, 8+SHIFT1, 9+SHIFT1>(q, nx, ny, nz, n_iter, true);
      // undo_interleaved_row_block8<0, D_MAX, 9+SHIFT1, 10+SHIFT1>(q, nx, ny, nz, n_iter, true);

      // row2col<0, D_MAX, 10+SHIFT1, 11+SHIFT1>(q, nx, ny, nz, n_iter);
      // thomas_interleave<0, float, D_MAX, 11+SHIFT1, 12+SHIFT1, 301+SHIFT1>(q, ny, B_Y, ReadLimit_Y, n_iter);
      // thomas_generate_r<0, float, D_MAX, 13+SHIFT1, 301+SHIFT1>(q, ny, B_Y_r, count_limit_y, n_iter);
      // thomas_forward<0, float, D_MAX, 12+SHIFT1, 14+SHIFT1>(q, ny, B_Y, n_iter);
      // thomas_backward<0, float, D_MAX, 14+SHIFT1, 16+SHIFT1>(q, ny, B_Y, ReadLimit_Y, n_iter);
      // col2row<0, D_MAX, 11+SHIFT1, 17+SHIFT1>(q, nx, ny, nz, n_iter);



      



      PipeConvert_256_512<8, 21*UFACTOR, 1>(q, total_itr_8, n_iter);
      PipeConvert_256_512<8, 2002+(UFACTOR-1)*2, 3>(q, total_itr_8, n_iter);

      #undef SHIFT1
      //write back to memory
      // stencil_write<16>(q, out_buf, total_itr_16, kernel_time);
      q.wait();

      double start0 = e.get_profiling_info<info::event_profiling::command_start>();
      double end0 = e.get_profiling_info<info::event_profiling::command_end>(); 
      kernel_time += (end0-start0)*1e-9;

      
    //   // reading from memory
    //   stencil_read<16>(q, out_buf, total_itr_16);
    //   PipeConvert_512_256<8>(q, total_itr_8);
    //   loop<UFACTOR, UFACTOR>::instantiate(q, nx, ny, nz, total_itrS);
    //   PipeConvert_256_512<UFACTOR, 8>(q, total_itr_8);
    //   //write back to memory
    //   stencil_write<16>(q, in_buf, total_itr_16, kernel_time);
    //   q.wait();
    // }

    std::cout << "fimished reading from the pipe\n" << std::endl;

    double exe_elapsed = exe_time.Elapsed();
    double bandwidth = 2.0*v_factor*vec_size*sizeof(int)*n_iter/(kernel_time*1000000000);
    std::cout << "Elapsed time: " << kernel_time << std::endl;
    std::cout << "Bandwidth(GB/s): " << bandwidth << std::endl;
}

//************************************
// Initialize the vector from 0 to vector_size - 1
//************************************
template<int VFACTOR>
void InitializeVector(IntVector &a, int delay) {
  for (size_t i = 0; i < a.size(); i++){
    for(int v = 0; v < VFACTOR; v++){
        a[i].data[v] = ((i-delay)* VFACTOR+v)*0.01 +0.5f;
    }
  }
}

void InitializeVectorS(IntVectorS &a) {
  for (size_t i = 0; i < a.size(); i++){
      a[i] = i*0.01  +0.5f;
  }
}

template<int VFACTOR>
void InitializeVector_Acc(IntVector &a, int delay) {
  for (size_t i = 0; i < a.size(); i++){
    for(int v = 0; v < VFACTOR; v++){
        a[i].data[v] = 0;
    }
  }
}

void InitializeVectorS_acc(IntVectorS &a) {
  for (size_t i = 0; i < a.size(); i++){
      a[i] = 0;
  }
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {

  int n_iter = 1;
  int nx = 128, ny = 128, nz=2;
  // Change vector_size if it was passed as argument
  if (argc > 1) n_iter = std::stoi(argv[1]);
  if (argc > 2) nx = std::stoi(argv[2]);
  if (argc > 3) ny = std::stoi(argv[3]);
  if (argc > 4) nz = std::stoi(argv[4]);

  nx = (nx % 8 == 0 ? nx : (nx/8+1)*8);
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  INTEL::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  INTEL::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif

  // Create vector objects with "vector_size" to store the input and output data.

  // int delay = (nx/v_factor)*UFACTOR+15000;

  int delay1 = ((56*nx + nx/4*ny)/2 + 18*30/2)*UFACTOR + 811;
  int delay2 = ((56*nx + nx/4*ny)/2 + 18*30/2)*(UFACTOR-1) + 811 ; //((nx/8)/2 + 3*30/2)*2 + 811 + 100 + 10000;

  IntVector in_vec, out_parallel, acc_1, acc_2;
  IntVectorS ax_h, bx_h, cx_h;
  IntVectorS ay_h, by_h, cy_h;

  IntVectorS in_vec_h, out_sequential, acc_h;
  in_vec.resize(nx/v_factor*ny*nz+delay1);
  in_vec_h.resize(nx*ny*nz);

  ax_h.resize(nx*ny*nz);
  bx_h.resize(nx*ny*nz);
  cx_h.resize(nx*ny*nz);

  ay_h.resize(nx*ny*nz);
  by_h.resize(nx*ny*nz);
  cy_h.resize(nx*ny*nz);

  acc_h.resize(nx*ny*nz);

  out_sequential.resize(nx*ny*nz);
  out_parallel.resize(nx/v_factor*ny*nz+delay1*2);

  acc_1.resize(nx/v_factor*ny*nz+delay2*2);
  acc_2.resize(nx/v_factor*ny*nz+delay2*2);

  InitializeVector_Acc<v_factor>(acc_1, delay2);
  InitializeVectorS_acc(acc_h);

  // Initialize input vectors with values from 0 to vector_size - 1
  InitializeVector<v_factor>(in_vec, delay1);
  InitializeVectorS(in_vec_h);
  try {
    queue q(d_selector,  dpc_common::exception_handler, property::queue::enable_profiling{});

    // queue q2(d_selector,  dpc_common::exception_handler);


    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << in_vec.size() << "\n";

    // Vector addition in DPC++
    
    stencil_comp(q, in_vec, out_parallel, acc_1, acc_2, 2*n_iter, nx, ny, nz, delay1, delay2);

  } catch (exception const &e) {
    std::cout << "An exception is caught for vector add.\n";
    std::terminate();
  }



  // Compute the sum of two vectors in sequential for validation.

  for(int itr= 0; itr < 2*n_iter; itr++){
    for(int k = 0; k < nz; k++){
      for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
          int ind = k*nx*ny + j*nx + i;
           
          if(i > 0 && i < nx -1 && j > 0 && j < ny -1){
            out_sequential.at(ind) = in_vec_h.at(ind-1) - in_vec_h.at(ind)*6.0f + in_vec_h.at(ind+1) + in_vec_h.at(ind-nx) + in_vec_h.at(ind+nx);
            ax_h.at(ind) = -0.5f;
            bx_h.at(ind) = 2.0f;
            cx_h.at(ind) = -0.5f;

            ay_h.at(ind) = -0.5f;
            by_h.at(ind) = 2.0f;
            cy_h.at(ind) = -0.5f;

          } else {
            out_sequential.at(ind) = 0.0f; 
            ax_h.at(ind) = 0.0f;
            bx_h.at(ind) = 1.0f;
            cx_h.at(ind) = 0.0f;

            ay_h.at(ind) = 0.0f;
            by_h.at(ind) = 1.0f;
            cy_h.at(ind) = 0.0f;
          }
        }
      }
    }

    for(int i = 0; i < ny*nz; i++){
      int ind = i*nx;
      thomas_golden<float>(ax_h.data()+ind, bx_h.data()+ind, cx_h.data()+ind, out_sequential.data()+ind, in_vec_h.data()+ind, nx, 1);
    }


    for(int i =0; i < nx; i++){
      for(int j = 0; j < nz; j++){
        int ind = i+j*nx*ny;
        thomas_golden<float>(ay_h.data()+ind, by_h.data()+ind, cy_h.data()+ind, in_vec_h.data()+ind, out_sequential.data()+ind, ny, nx);
      }
    }
    


    for(int k = 0; k < nz; k++){
      for(int j = 0; j < ny; j++){
        for(int i = 0; i < nx; i++){
          int ind = k*nx*ny + j*nx + i;
          acc_h.at(ind) += out_sequential.at(ind);
          in_vec_h.at(ind) = acc_h.at(ind); //out_sequential.at(ind);
        }
      }
    }



  }
   std::cout << "No error until here\n";

  // for (size_t i = 0; i < out_sequential.size(); i++)
  //   out_sequential.at(i) = in_vec.at(i) + 50;

  // Verify that the two vectors are equal. 
  for(int k = 0; k < nz; k++){ 
    for(int j = 0; j < ny; j++){
      for(int i = 0; i < nx/v_factor; i++){
        for(int v = 0; v < v_factor; v++){
          int ind = k*nx*ny + j*nx + i*v_factor;
          float chk = fabs((out_sequential.at(ind+v) - in_vec.at(ind/v_factor+delay1).data[v])/(out_sequential.at(ind+v)));
          if(chk > 0.00001 && fabs(out_sequential.at(ind+v)) > 0.00001 || isnan(out_sequential.at(ind+v)) || isnan(in_vec.at(ind/v_factor+delay1).data[v])){
            // std::cout << out_parallel.at(ind/v_factor+delay).data[v] << " ";
            std::cout << "j,i, k, ind: " << j  << " " << i << " " << k << " " << ind << " " << out_sequential.at(ind+v) << " " << in_vec.at(ind/v_factor+delay1).data[v] <<  std::endl;
            // return -1;
          }
        }
        // std::cout << std::endl;
      }
    }
  }

  // for (size_t i = 0; i < out_sequential.size(); i++) {
  //   if (in_vec_h.at(i) != in_vec.at(i)) {
  //     std::cout << "Vector add failed on device.\n";
  //     return -1;
  //   }
  // }

  // int indices[]{0, 1, 2, (static_cast<int>(in_vec.size()) - 1)};
  // constexpr size_t indices_size = sizeof(indices) / sizeof(int);

  // // Print out the result of vector add.
  // for (int i = 0; i < indices_size; i++) {
  //   int j = indices[i];
  //   if (i == indices_size - 1) std::cout << "...\n";
  //   std::cout << "[" << j << "]: " << in_vec[j] << " + 50 = "
  //             << out_parallel[j] << "\n";
  // }
  in_vec_h.clear();
  in_vec.clear();
  out_sequential.clear();
  out_parallel.clear();

  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}
