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

using namespace sycl;

// Vector type and data size for this example.
size_t vector_size = 10000;
typedef std::vector<float> IntVector; 
const int unroll_factor = 2;

struct dPath {
  [[intel::fpga_register]] float data[4];
};

struct dPath16 {
  [[intel::fpga_register]] float data[16];
};

using rd_pipe = INTEL::pipe<class pVec16_r, dPath16, 8>;
using wr_pipe = INTEL::pipe<class pVec16_w, dPath16, 8>;

#define UFACTOR 2

struct pipeS{
  pipeS() = delete;
  template <size_t idx>  struct struct_id;

  template <size_t idx>
  struct Pipes{
    using pipeA = INTEL::pipe<struct_id<idx>, dPath, 8>;
  };

  template <size_t idx>
  using PipeAt = typename Pipes<idx>::pipeA;
};


using PipeBlock = pipeS;

// using rd_pipe1 = INTEL::pipe<class rd_pipe1, dPath, 8>;
// using wr_pipe1 = INTEL::pipe<class wr_pipe1, dPath, 8>;


// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};



template<int VFACTOR>
void read_dat(queue &q, const float* in, int size){
      event e1 = q.submit([&](handler &h) {

      int total_itr = (size)/(VFACTOR*2);
      const int VFACTOR2 = VFACTOR*2;
      h.single_task<class producer>([=] () [[intel::kernel_args_restrict]]{

        [[intel::initiation_interval(1)]]
        for(int i = 0; i < total_itr; i++){
          struct dPath16 vec;
          #pragma unroll VFACTOR2
          for(int v = 0; v < VFACTOR2; v++){
            vec.data[v] = in[i*VFACTOR2+v];
          }
          rd_pipe::write(vec);
        }
        
      });
    });
}

template<int VFACTOR>
void PipeConvert_512_128(queue &q, int size){
      event e1 = q.submit([&](handler &h) {

      int total_itr = size/(VFACTOR);
      h.single_task<class PipeConvert_512_256>([=] () [[intel::kernel_args_restrict]]{
        struct dPath16 data16;
        [[intel::initiation_interval(1)]]
        for(int i = 0; i < total_itr; i++){
          struct dPath data;
          if((i&3) == 0){
            data16 = rd_pipe::read();
          }

          #pragma unroll VFACTOR
          for(int v = 0; v < VFACTOR; v++){
            switch((i&3)){
              case 0: {data.data[v] = data16.data[v]; break;}
              case 1: {data.data[v] = data16.data[v+VFACTOR]; break;}
              case 2: {data.data[v] = data16.data[v+VFACTOR*2]; break;}
              case 3: {data.data[v] = data16.data[v+VFACTOR*3]; break;}
            }
          }
          pipeS::PipeAt<0>::write(data);
        }
        
      });
    });
}

template <size_t idx>  struct struct_idX;
template<int idx, int IdX, int DMAX, int VFACTOR> 
void stencil_compute(queue &q,  ac_int<12,true> nx, ac_int<12,true> ny, ac_int<12,true> nz, ac_int<12,true> batch){
    event e2 = q.submit([&](handler &h) {

    std::string instance_name="compute"+std::to_string(idx);
    h.single_task<class struct_idX<IdX>>([=] () [[intel::kernel_args_restrict]]{
    int total_itr = ((nx/VFACTOR)*ny*(batch*nz+1));

    const int max_dpethl = DMAX/VFACTOR;
    const int max_dpethP = DMAX*DMAX/VFACTOR;

    struct dPath s_1_1_2, s_1_2_1, s_1_1_1, s_1_1_1_b, s_1_1_1_f, s_1_0_1, s_1_1_0;

    [[intel::fpga_memory("BLOCK_RAM")]] struct dPath window_1[max_dpethP];
    struct dPath window_2[max_dpethl];
    struct dPath window_3[max_dpethl];
    [[intel::fpga_memory("BLOCK_RAM")]] struct dPath window_4[max_dpethP];



    struct dPath vec_wr;
    [[intel::fpga_register]] float mid_row[VFACTOR+2];
    ac_int<12,true>  j_ld = 0, j_pd = 0;


    ac_int<12,true> id = 0, jd = 0, kd = 0, batd = 0;;
    unsigned int mesh_size = (nx*ny)/VFACTOR;
    ac_int<12,true> rEnd = (nx/VFACTOR)-1;

    [[intel::initiation_interval(1)]]
    for(int itr = 0; itr < total_itr; itr++){
      ac_int<12,true> i = id; // itr % rEnd; //id;
      ac_int<12,true> j = jd; //itr / rEnd ;///jd;
      ac_int<12,true> k = kd;
      ac_int<12,true> bat = batd;

      ac_int<12,true> j_l = j_ld;
      ac_int<12,true> j_p = j_pd;

      if(i == rEnd){
        id = 0;
      } else {
        id++;
      }

      if(i == rEnd && j == ny-1){
        jd = 0;
      } else if(i == rEnd){
        jd++;
      }


      if(i == rEnd && j == ny-1 && k == nz){
        kd = 1;
      }else if(i == rEnd && j == ny-1){
        kd++;
      }


      s_1_1_0 = window_4[j_p];

      s_1_0_1 = window_3[j_l];
      window_4[j_p] = s_1_0_1;

      s_1_1_1_b = s_1_1_1;
      window_3[j_l] = s_1_1_1_b;

      s_1_1_1 = s_1_1_1_f;
      s_1_1_1_f = window_2[j_l];  // read

      s_1_2_1 = window_1[j_p];   // read
      window_2[j_l] = s_1_2_1;  //set

      if(itr < (nx/VFACTOR)*ny*nz*batch){
        s_1_1_2 = pipeS::PipeAt<idx>::read();
      }

      window_1[j_p] = s_1_1_2;

    
      if(j_l >= nx/VFACTOR -2){
        j_ld = 0;
      } else {
        j_ld++;
      }

      if(j_p >= (nx/VFACTOR)*(ny-1) - 1){
        j_pd = 0;
      } else {
        j_pd++;
      }

      #pragma unroll VFACTOR
      for(int v = 0; v < VFACTOR; v++){
        mid_row[v+1] = s_1_1_1.data[v]; 
      }

      mid_row[0] = s_1_1_1_b.data[VFACTOR-1];
      mid_row[VFACTOR+1] = s_1_1_1_f.data[0];

      #pragma unroll VFACTOR
      for(short q = 0; q < VFACTOR; q++){
        short index = (i * VFACTOR) + q;
        float r1_1_2 =  s_1_1_2.data[q] * (0.02f);
        float r1_2_1 =  s_1_2_1.data[q] * (0.04f);
        float r0_1_1 =  mid_row[q] * (0.05f);
        float r1_1_1 =  mid_row[q+1] * (0.79f);
        float r2_1_1 =  mid_row[q+2] * (0.06f);
        float r1_0_1 =  s_1_0_1.data[q] * (0.03f);
        float r1_1_0 =  s_1_1_0.data[q] * (0.01f);

        float f1 = r1_1_2 + r1_2_1;
        float f2 = r0_1_1 + r1_1_1;
        float f3 = r2_1_1 + r1_0_1;


        float r1 = f1 + f2;
        float r2=  f3 + r1_1_0;

        float result  = r1 + r2;
        bool change_cond = (index <= 0 || index >= nx-1 || (k <= 1) || (k >= nz) || (j <= 0) || (j >= ny -1));
        vec_wr.data[q] = change_cond ? mid_row[q+1] : result;
      }

      bool cond_wr = (k >= 1) && ( k < nz+1);

      // if(itr < (nx>>3)*ny*nz){
      if(itr >= (nx/VFACTOR)*ny){
        pipeS::PipeAt<idx+1>::write(vec_wr);
      }
    }
    
  });
  });
}



template <int idx, int VFACTOR>
void PipeConvert_128_512(queue &q, int size){
    event e3 = q.submit([&](handler &h) {
    h.single_task<class pipeConvert_256_512>([=] () [[intel::kernel_args_restrict]]{
      int total_itr = size/(VFACTOR);
      struct dPath16 data16;
      [[intel::initiation_interval(1)]]
      for(int i = 0; i < total_itr; i++){
        struct dPath data;
        data = pipeS::PipeAt<idx>::read();
        #pragma unroll VFACTOR
        for(int v = 0; v < VFACTOR; v++){
          switch((i & 3)){
            case 0: {data16.data[v] = data.data[v]; break;}
            case 1: {data16.data[v+VFACTOR] = data.data[v]; break;}
            case 2: {data16.data[v+VFACTOR*2] = data.data[v]; break;}
            case 3: {data16.data[v+VFACTOR*3] = data.data[v]; break;}
          }
        }
        if((i & 3) == 3){
          wr_pipe::write(data16);
        }
      }
    });
    });
}


template <int VFACTOR>
void write_dat(queue &q, float* out, int size, double &kernel_time){
    event e3 = q.submit([&](handler &h) {
    // accessor out(out_buf, h, write_only);
    h.single_task<class stencil_write>([=] () [[intel::kernel_args_restrict]]{
      int total_itr = (size)/(VFACTOR*2);
      const int VFACTOR2 = VFACTOR*2;
      [[intel::initiation_interval(1)]]
      for(int i = 0; i < total_itr; i++){
        struct dPath16 vec;
        vec = wr_pipe::read();
        #pragma unroll VFACTOR2
        for(int v = 0; v < VFACTOR2; v++){
          out[i*VFACTOR2+v] = vec.data[v];
        }
      }
      
    });
    });

    double start0 = e3.get_profiling_info<info::event_profiling::command_start>();
    double end0 = e3.get_profiling_info<info::event_profiling::command_end>(); 
    kernel_time += (end0-start0)*1e-9;
}


template <int N, int n> struct loop {
  static void instantiate(queue &q, int nx, int ny, int nz, int batch){
    loop<N-1, n-1>::instantiate(q, nx, ny, nz, batch);
    stencil_compute<N-1, n-1, 128, 4>(q, nx, ny, nz, batch);
  }
};

template<> 
struct loop<1, 1>{
  static void instantiate(queue &q, int nx, int ny, int nz, int batch){
    stencil_compute<0, 0, 128, 4>(q, nx, ny, nz, batch);
  }
};

// loop<90> l;


//************************************
// Vector add in DPC++ on device: returns sum in 4th parameter "sum_parallel".
//************************************
void stencil_comp(queue &q, float* input, float* output, int n_iter, int nx, int ny, int nz, int batch) {
  // Create the range object for the vectors managed by the buffer.
  
  int vec_size = nx*ny*nz*batch;

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  // buffer in_buf(input);
  // buffer out_buf(output);

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  double kernel_time = 0;
  std::cout << "starting writing to the pipe\n" << std::endl;
  dpc_common::TimeInterval exe_time;

    int size = nx*ny*nz*batch;
    for(int itr = 0; itr < n_iter; itr++){

      // reading from memory
      read_dat<8>(q, input, size);
      PipeConvert_512_128<4>(q, size);
      loop<UFACTOR, UFACTOR>::instantiate(q, nx, ny, nz, batch);
      PipeConvert_128_512<UFACTOR, 4>(q, size);
      //write back to memory
      write_dat<8>(q, output, size, kernel_time);
      q.wait();

      
      // reading from memory
      read_dat<8>(q, output, size);
      PipeConvert_512_128<4>(q, size);
      loop<UFACTOR, UFACTOR>::instantiate(q, nx, ny, nz, batch);
      PipeConvert_128_512<UFACTOR, 4>(q, size);
      //write back to memory
      write_dat<8>(q, input, size, kernel_time);
      

      q.wait();

    }

    std::cout << "fimished reading from the pipe\n" << std::endl;

    double exe_elapsed = exe_time.Elapsed();
    double bandwidth = 2.0*vec_size*sizeof(int)*n_iter*2.0/(kernel_time*1000000000);
    std::cout << "Elapsed time: " << kernel_time << std::endl;
    std::cout << "Bandwidth(GB/s): " << bandwidth << std::endl;
}

//************************************
// Initialize the vector from 0 to vector_size - 1
//************************************


void InitializeCoeffs(float* a, float* b, float*c , float* d, int nx, int ny){
  for(int i = 0; i < ny; i++){
    for(int j = 0; j < nx; j++){
      int index = i*nx+j;
      if(j == nx-1 || j == 0){
        a[index] = 0.0f;
        b[index] = 1.0f;
        c[index] = 0.0f;
        d[index] = index*0.1+5;
      } else {
        a[index] = -0.5f;
        b[index] = 2.0f;
        c[index] = -0.5f;
        d[index] = index*0.1+5;
      }
    }
  }
}

//************************************
// Thomas Golden Computation 
//************************************

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

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {

  int n_iter = 1;
  int nx = 32, ny = 32, nz=4, batch = 4;
  // Change vector_size if it was passed as argument
  if (argc > 1) n_iter = std::stoi(argv[1]);
  if (argc > 2) nx = std::stoi(argv[2]);
  if (argc > 3) ny = std::stoi(argv[3]);
  if (argc > 4) nz = std::stoi(argv[4]);
  if (argc > 5) batch = std::stoi(argv[5]);

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



  
  

  try {
    queue q(d_selector,  dpc_common::exception_handler, property::queue::enable_profiling{});

    float* a_h = malloc_host<float>(nx*ny, q);
    float* b_h = malloc_host<float>(nx*ny, q);
    float* c_h = malloc_host<float>(nx*ny, q);
    float* d_h = malloc_host<float>(nx*ny, q);
    float* u_h = malloc_host<float>(nx*ny, q);


    float* a_d = malloc_shared<float>(nx*ny, q);
    float* b_d = malloc_shared<float>(nx*ny, q);
    float* c_d = malloc_shared<float>(nx*ny, q);
    float* d_d = malloc_shared<float>(nx*ny, q);
    float* u_d = malloc_shared<float>(nx*ny, q);


    InitializeCoeffs(a_h, b_h, c_h, d_h, nx, ny);
    InitializeCoeffs(a_d, b_d, c_d, d_d, nx, ny);
    // queue q2(d_selector,  dpc_common::exception_handler);


    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";


    // Vector addition in DPC++
    
    // stencil_comp(q, in_vec, out_parallel, n_iter, nx, ny, nz, batch);

    // Compute the sum of two vectors in sequential for validation.
    for(int i = 0; i < ny; i++){
      thomas_golden<float>(&a_h[i*nx], &b_h[i*nx], &c_h[i*nx], &d_h[i*nx], &u_h[i*nx], nx, 1);
    }




    // Verify that the two vectors are equal. 
    for(int i = 0; i < ny; i++){
      for(int j = 0; j < nx; j++){
        int ind = i*nx + j;
        float chk = fabs((u_h[ind] - u_d[ind])/(u_h[ind]));
        if(chk > 0.00001 && fabs(u_h[ind]) > 0.00001){
          std::cout << "i,j: " << i  << " " << j << " " << u_h[ind] << " " << u_d[ind] <<  std::endl;
          // return -1;
        }
      }
    }
 

      free(a_h, q);
      free(b_h, q);
      free(c_h, q);
      free(d_h, q);
      free(u_h, q);

      free(a_d, q);
      free(b_d, q);
      free(c_d, q);
      free(d_d, q);
      free(u_d, q);

  } catch (exception const &e) {
    std::cout << "An exception is caught for vector add.\n";
    std::terminate();
  }



  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}
